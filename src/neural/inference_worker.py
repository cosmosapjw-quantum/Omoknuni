"""
GPU Inference Worker Implementation
==================================

Threaded GPU inference worker for batched neural network evaluation.
Optimized for RTX 3060 Ti with queue-based communication and dynamic batching.

The worker runs in a dedicated thread, consuming inference requests from a shared
input queue and distributing results to thread-specific output queues.
"""

import torch
import numpy as np
import time
import threading
from typing import List, Dict, Optional, Tuple, Any
from queue import Queue, Empty, Full
from collections import deque
import logging
import psutil
import os
from dataclasses import dataclass

# GPU monitoring (optional import)
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    pynvml = None

# Import contract interfaces
import sys
sys.path.append('specs/001-goal-create-spec')
from contracts.inference_api import (
    InferenceWorker,
    InferenceRequest,
    InferenceResult
)

# Import neural network model
from src.neural.model import AlphaZeroNet, create_model_for_game


class GPUInferenceWorker(InferenceWorker):
    """GPU-based inference worker with batched processing.

    Runs neural network inference on GPU with dynamic batching for optimal
    throughput and GPU utilization.

    Args:
        model_path: Path to trained PyTorch model
        device: Device for inference ('cuda:0' or 'cpu')
        batch_size: Maximum batch size for GPU inference
        timeout_ms: Batch timeout in milliseconds
        use_mixed_precision: Enable fp16 inference
    """

    def __init__(self,
                 model_path: str,
                 device: str = 'cuda:0',
                 batch_size: int = 64,
                 timeout_ms: float = 3.0,
                 use_mixed_precision: bool = True):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms / 1000.0  # Convert to seconds
        self.use_mixed_precision = use_mixed_precision

        # Dynamic micro-batching parameters
        self.min_batch_size = max(1, min(32, batch_size))  # Target ≥32 for efficiency
        self.max_timeout_ms = min(3.0, timeout_ms) / 1000.0  # Target ≤3ms for responsiveness
        self.target_gpu_utilization = 0.80  # Target >80% GPU utilization

        # Adaptive batching state
        self._performance_history = deque(maxlen=100)  # Track recent performance
        self._current_optimal_batch = self.min_batch_size
        self._gpu_handle = None

        # Thread control
        self._stop_event = threading.Event()
        self._worker_thread = None
        self._is_running = False

        # Model and computation
        self.model = None
        self.input_shape = None

        # Performance tracking
        self._metrics = {
            'total_requests': 0,
            'total_batches': 0,
            'total_inference_time': 0.0,
            'batch_sizes': deque(maxlen=1000),  # Recent batch sizes
            'inference_times': deque(maxlen=1000),  # Recent inference times
        }
        self._metrics_lock = threading.Lock()

        # Setup logging
        self.logger = logging.getLogger(f'InferenceWorker[{device}]')

        # Initialize GPU monitoring
        self._init_gpu_monitoring()

        # Initialize model
        self._load_model()

    def _load_model(self) -> None:
        """Load and initialize the neural network model."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found: {self.model_path}")

            # Load model state dict
            state_dict = torch.load(self.model_path, map_location=self.device)

            # Extract model configuration from state dict or use defaults
            # For now, create a default Gomoku model - in practice this would
            # be extracted from model metadata
            self.model = create_model_for_game('gomoku')

            # Move to device and set eval mode
            self.model = self.model.to(self.device)
            self.model.eval()

            # Initialize lazy layers with dummy input
            with torch.no_grad():
                dummy_input = torch.randn(1, 7, 15, 15, device=self.device)
                _ = self.model(dummy_input)

            # Now load the actual weights
            self.model.load_state_dict(state_dict)

            # Enable mixed precision if requested and using CUDA
            if self.use_mixed_precision and self.device.startswith('cuda'):
                from src.neural.model import enable_mixed_precision
                self.model = enable_mixed_precision(self.model)

            self.logger.info(f"Model loaded successfully on {self.device}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _init_gpu_monitoring(self) -> None:
        """Initialize GPU monitoring for utilization tracking."""
        if not NVML_AVAILABLE or not self.device.startswith('cuda'):
            self.logger.info("GPU monitoring not available (no pynvml or CPU device)")
            return

        try:
            pynvml.nvmlInit()
            device_id = int(self.device.split(':')[1]) if ':' in self.device else 0
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            self.logger.info(f"GPU monitoring initialized for device {device_id}")
        except Exception as e:
            self.logger.warning(f"Could not initialize GPU monitoring: {e}")
            self._gpu_handle = None

    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage.

        Returns:
            GPU utilization as percentage (0.0-1.0), or 0.0 if unavailable
        """
        if not self._gpu_handle:
            return 0.0

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
            return util.gpu / 100.0
        except Exception:
            return 0.0

    def warmup(self, input_shape: Tuple[int, int, int]) -> None:
        """Warmup GPU with dummy inference calls.

        Critical for consistent latency measurements. Must be called
        before starting inference loop.

        Args:
            input_shape: (channels, height, width) for input tensors
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        self.input_shape = input_shape
        self.logger.info(f"Warming up GPU with input shape {input_shape}")

        # Warm up with different batch sizes
        warmup_batches = [1, 8, 16, 32, min(64, self.batch_size)]

        with torch.no_grad():
            for batch_size in warmup_batches:
                dummy_input = torch.randn(batch_size, *input_shape, device=self.device)

                # Warmup runs
                for _ in range(3):
                    if self.use_mixed_precision and self.device.startswith('cuda'):
                        with torch.amp.autocast('cuda'):
                            _ = self.model(dummy_input)
                    else:
                        _ = self.model(dummy_input)

                # Synchronize GPU
                if self.device.startswith('cuda'):
                    torch.cuda.synchronize()

        self.logger.info("GPU warmup completed")

    def start_worker(self, input_queue: Queue, output_queues: List[Queue]) -> None:
        """Start the inference worker thread.

        Args:
            input_queue: Queue of inference requests
            output_queues: List of result queues, one per search thread
        """
        if self._is_running:
            raise RuntimeError("Worker already running")

        self.logger.info("Starting inference worker thread")

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self.inference_loop,
            args=(input_queue, output_queues),
            name="InferenceWorker",
            daemon=True
        )
        self._worker_thread.start()
        self._is_running = True

    def stop_worker(self, timeout: float = 5.0) -> None:
        """Stop the inference worker thread.

        Args:
            timeout: Maximum time to wait for thread to stop
        """
        if not self._is_running:
            return

        self.logger.info("Stopping inference worker thread")

        self._stop_event.set()

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

            if self._worker_thread.is_alive():
                self.logger.warning("Worker thread did not stop within timeout")
            else:
                self.logger.info("Worker thread stopped successfully")

        self._is_running = False
        self._worker_thread = None

    def inference_loop(self,
                      input_queue: Queue,
                      output_queues: List[Queue]) -> None:
        """Main inference loop for batched processing.

        Runs in dedicated thread, consuming requests from input_queue
        and distributing results to thread-specific output_queues.

        Args:
            input_queue: Queue of inference requests
            output_queues: List of result queues, one per search thread
        """
        self.logger.info("Inference loop started")

        try:
            while not self._stop_event.is_set():
                # Collect batch of requests
                batch_requests = self._collect_batch(input_queue)

                if not batch_requests:
                    continue

                # Process batch
                start_time = time.time()
                batch_results = self._process_batch(batch_requests)
                inference_time = time.time() - start_time

                # Distribute results to output queues
                self._distribute_results(batch_results, output_queues)

                # Update metrics
                self._update_metrics(len(batch_requests), inference_time)

        except Exception as e:
            self.logger.error(f"Error in inference loop: {e}")
            raise
        finally:
            self.logger.info("Inference loop ended")

    def _collect_batch(self, input_queue: Queue) -> List[InferenceRequest]:
        """Collect a batch of requests with dynamic micro-batching.

        Uses sophisticated count-based (≥32) OR timeout-based (≤3ms) batching
        to optimize for target >80% GPU utilization.

        Args:
            input_queue: Queue of inference requests

        Returns:
            List of inference requests (may be empty)
        """
        batch = []
        start_time = time.time()

        # Determine optimal batch size based on recent performance
        target_batch_size = self._get_optimal_batch_size()

        # Try to get first request with micro-timeout
        try:
            first_request = input_queue.get(timeout=self.max_timeout_ms)
            batch.append(first_request)
        except Empty:
            return batch

        # Phase 1: Quick collection to target batch size
        # Collect aggressively for first few requests
        quick_timeout = min(0.001, self.max_timeout_ms / 4)  # 1ms or quarter of max

        while len(batch) < target_batch_size:
            elapsed = time.time() - start_time

            # If we're approaching max timeout, be more aggressive
            if elapsed > self.max_timeout_ms * 0.8:
                break

            try:
                request = input_queue.get(timeout=quick_timeout)
                batch.append(request)
            except Empty:
                break

        # Phase 2: Smart timeout-based collection
        # If we haven't reached min efficient batch size, wait a bit longer
        if len(batch) < self.min_batch_size:
            remaining_timeout = max(0, self.max_timeout_ms - (time.time() - start_time))

            while len(batch) < self.min_batch_size and remaining_timeout > 0:
                try:
                    request = input_queue.get(timeout=remaining_timeout)
                    batch.append(request)

                    # Update remaining timeout
                    elapsed = time.time() - start_time
                    remaining_timeout = max(0, self.max_timeout_ms - elapsed)

                except Empty:
                    break

        # Phase 3: Opportunistic collection
        # If we have time left and haven't hit max batch size, collect more
        remaining_time = self.max_timeout_ms - (time.time() - start_time)
        if remaining_time > 0 and len(batch) < self.batch_size:
            # Use very short timeout for opportunistic collection
            opportunistic_timeout = min(0.0005, remaining_time)  # 0.5ms max

            while len(batch) < self.batch_size and remaining_time > 0:
                try:
                    request = input_queue.get(timeout=opportunistic_timeout)
                    batch.append(request)
                    remaining_time = self.max_timeout_ms - (time.time() - start_time)
                except Empty:
                    break

        return batch

    def _get_optimal_batch_size(self) -> int:
        """Determine optimal batch size based on recent performance.

        Returns:
            Optimal batch size for current conditions
        """
        # If no performance history, start with minimum efficient size
        if not self._performance_history:
            return self.min_batch_size

        # Get recent GPU utilization if available
        gpu_util = self._get_gpu_utilization()

        # Analyze recent performance
        recent_perf = list(self._performance_history)[-10:]  # Last 10 batches
        if not recent_perf:
            return self._current_optimal_batch

        avg_throughput = sum(p['throughput'] for p in recent_perf) / len(recent_perf)
        avg_batch_size = sum(p['batch_size'] for p in recent_perf) / len(recent_perf)

        # Adaptive logic based on GPU utilization and throughput
        if gpu_util > 0:  # GPU monitoring available
            if gpu_util < self.target_gpu_utilization * 0.9:  # Below 72%
                # Increase batch size to improve GPU utilization
                self._current_optimal_batch = min(
                    self.batch_size,
                    int(self._current_optimal_batch * 1.1)
                )
            elif gpu_util > self.target_gpu_utilization * 1.1:  # Above 88%
                # Decrease batch size to avoid overload
                self._current_optimal_batch = max(
                    self.min_batch_size,
                    int(self._current_optimal_batch * 0.9)
                )
        else:
            # No GPU monitoring - use throughput-based adaptation
            if len(recent_perf) >= 5:
                # Check if throughput is improving with larger batches
                recent_5 = recent_perf[-5:]
                throughput_trend = (recent_5[-1]['throughput'] - recent_5[0]['throughput']) / 5

                if throughput_trend > 0 and avg_batch_size < self.batch_size * 0.8:
                    # Throughput improving, try larger batches
                    self._current_optimal_batch = min(
                        self.batch_size,
                        int(self._current_optimal_batch * 1.05)
                    )
                elif throughput_trend < 0 and avg_batch_size > self.min_batch_size * 1.2:
                    # Throughput declining, try smaller batches
                    self._current_optimal_batch = max(
                        self.min_batch_size,
                        int(self._current_optimal_batch * 0.95)
                    )

        return self._current_optimal_batch

    def _process_batch(self, requests: List[InferenceRequest]) -> List[InferenceResult]:
        """Process a batch of inference requests.

        Args:
            requests: List of inference requests

        Returns:
            List of inference results
        """
        if not requests:
            return []

        # Extract features from requests
        positions = [req.features for req in requests]

        # Run batch inference
        start_time = time.time()
        policies, values = self.batch_inference(positions)
        processing_time_ms = (time.time() - start_time) * 1000

        # Create results
        results = []
        for i, request in enumerate(requests):
            result = InferenceResult(
                node_id=request.leaf_node_id,
                policy=policies[i],
                value=values[i].item(),
                path=request.path,
                processing_time_ms=processing_time_ms / len(requests)  # Per-sample time
            )
            results.append(result)

        return results

    def _distribute_results(self,
                          results: List[InferenceResult],
                          output_queues: List[Queue]) -> None:
        """Distribute results to appropriate output queues.

        Args:
            results: List of inference results
            output_queues: List of result queues, one per thread
        """
        for result in results:
            # For now, distribute results based on node_id hash
            # In practice, this would use thread_id from the request
            queue_idx = result.node_id % len(output_queues)

            try:
                output_queues[queue_idx].put(result, timeout=1.0)
            except Full:
                self.logger.warning(f"Output queue {queue_idx} full, dropping result")

    def batch_inference(self,
                       positions: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Process batch of positions through neural network.

        Args:
            positions: List of feature tensors, each (C, H, W)

        Returns:
            tuple: (policies, values)
                policies: Policy probabilities (batch_size, num_actions)
                values: Position values (batch_size,)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Convert to torch tensor
        batch_tensor = torch.tensor(
            np.stack(positions),
            dtype=torch.float32,
            device=self.device
        )

        # Run inference
        with torch.no_grad():
            if self.use_mixed_precision and self.device.startswith('cuda'):
                with torch.amp.autocast('cuda'):
                    policy_logits, values = self.model(batch_tensor)
            else:
                policy_logits, values = self.model(batch_tensor)

            # Convert to probabilities
            policies = torch.softmax(policy_logits, dim=1)

        # Convert back to numpy
        policies_np = policies.cpu().numpy()
        values_np = values.cpu().numpy().squeeze(-1)

        return policies_np, values_np

    def _update_metrics(self, batch_size: int, inference_time: float) -> None:
        """Update performance metrics with GPU utilization tracking.

        Args:
            batch_size: Size of processed batch
            inference_time: Time taken for inference (seconds)
        """
        positions_per_second = batch_size / inference_time if inference_time > 0 else 0
        gpu_util = self._get_gpu_utilization()

        with self._metrics_lock:
            self._metrics['total_requests'] += batch_size
            self._metrics['total_batches'] += 1
            self._metrics['total_inference_time'] += inference_time
            self._metrics['batch_sizes'].append(batch_size)
            self._metrics['inference_times'].append(inference_time)

            # Add GPU utilization metrics
            if 'gpu_utilization_samples' not in self._metrics:
                self._metrics['gpu_utilization_samples'] = deque(maxlen=100)

            if gpu_util > 0:
                self._metrics['gpu_utilization_samples'].append(gpu_util)

        # Store performance data for adaptive batching
        perf_data = {
            'batch_size': batch_size,
            'inference_time': inference_time,
            'throughput': positions_per_second,
            'gpu_utilization': gpu_util,
            'timestamp': time.time()
        }
        self._performance_history.append(perf_data)

        # Enhanced logging with GPU utilization
        gpu_info = f", GPU: {gpu_util*100:.1f}%" if gpu_util > 0 else ""
        self.logger.debug(
            f"Batch processed: {batch_size} positions in {inference_time:.3f}s "
            f"({positions_per_second:.1f} pos/s{gpu_info})"
        )

        # Log performance summary periodically
        if self._metrics['total_batches'] % 50 == 0:
            with self._metrics_lock:
                avg_gpu = 0
                if 'gpu_utilization_samples' in self._metrics and self._metrics['gpu_utilization_samples']:
                    avg_gpu = sum(self._metrics['gpu_utilization_samples']) / len(self._metrics['gpu_utilization_samples']) * 100

            gpu_status = "" if avg_gpu == 0 else f", avg GPU util: {avg_gpu:.1f}%"
            avg_throughput = self._metrics['total_requests'] / self._metrics['total_inference_time'] if self._metrics['total_inference_time'] > 0 else 0
            self.logger.info(
                f"Performance summary: {avg_throughput:.1f} pos/s, "
                f"avg batch: {self._metrics['total_requests'] / self._metrics['total_batches']:.1f}{gpu_status}"
            )

    def get_metrics(self) -> Dict[str, float]:
        """Get enhanced inference performance metrics including micro-batching data.

        Returns:
            dict: Comprehensive metrics including GPU utilization, adaptive batching info,
                 and performance targets
        """
        with self._metrics_lock:
            # Calculate averages
            recent_batch_sizes = list(self._metrics['batch_sizes'])
            recent_inference_times = list(self._metrics['inference_times'])

            avg_batch_size = np.mean(recent_batch_sizes) if recent_batch_sizes else 0.0
            avg_inference_time = np.mean(recent_inference_times) if recent_inference_times else 0.0

            # Calculate inference rate
            if avg_inference_time > 0:
                inference_rate = avg_batch_size / avg_inference_time
            else:
                inference_rate = 0.0

            # GPU utilization metrics
            current_gpu_util = self._get_gpu_utilization()
            avg_gpu_util = 0.0
            if 'gpu_utilization_samples' in self._metrics and self._metrics['gpu_utilization_samples']:
                avg_gpu_util = sum(self._metrics['gpu_utilization_samples']) / len(self._metrics['gpu_utilization_samples'])

            memory_usage_gb = self._get_memory_usage()

            metrics = {
                # Core performance metrics
                'gpu_utilization': current_gpu_util,
                'avg_gpu_utilization': avg_gpu_util,
                'average_batch_size': avg_batch_size,
                'inference_rate': inference_rate,
                'memory_usage_gb': memory_usage_gb,
                'total_requests': self._metrics['total_requests'],
                'total_batches': self._metrics['total_batches'],
                'total_inference_time': self._metrics['total_inference_time'],

                # Micro-batching configuration
                'current_optimal_batch': self._current_optimal_batch,
                'min_batch_size': self.min_batch_size,
                'max_timeout_ms': self.max_timeout_ms * 1000,
                'target_gpu_utilization': self.target_gpu_utilization,

                # Performance targets status
                'meets_batch_target': avg_batch_size >= self.min_batch_size,
                'meets_gpu_target': avg_gpu_util >= self.target_gpu_utilization,
                'timeout_compliance': avg_inference_time <= self.max_timeout_ms
            }

            return metrics

    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage (enhanced version).

        Returns:
            GPU utilization as percentage (0.0-1.0), or 0.0 if unavailable
        """
        # Use nvidia-ml-py if available and initialized
        if self._gpu_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                return util.gpu / 100.0
            except Exception:
                pass

        # Fallback to memory-based estimation
        try:
            if self.device.startswith('cuda') and torch.cuda.is_available():
                device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
                memory_used = torch.cuda.memory_allocated(device_idx)
                memory_total = torch.cuda.get_device_properties(device_idx).total_memory
                return (memory_used / memory_total)  # Return as 0.0-1.0 not percentage
            else:
                return 0.0
        except Exception:
            return 0.0

    def _get_memory_usage(self) -> float:
        """Get current VRAM usage in GB."""
        try:
            if self.device.startswith('cuda') and torch.cuda.is_available():
                device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
                memory_used = torch.cuda.memory_allocated(device_idx)
                return memory_used / (1024**3)  # Convert to GB
            else:
                # For CPU, return system memory usage of current process
                process = psutil.Process(os.getpid())
                return process.memory_info().rss / (1024**3)
        except Exception:
            return 0.0

    def is_running(self) -> bool:
        """Check if worker thread is running."""
        return self._is_running

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensure clean shutdown."""
        self.stop_worker()


def create_inference_worker(model_path: str,
                           device: str = 'cuda:0',
                           **kwargs) -> InferenceWorker:
    """Factory function to create inference worker.

    Args:
        model_path: Path to trained PyTorch model
        device: Inference device
        **kwargs: Additional worker configuration

    Returns:
        InferenceWorker: Configured inference worker instance
    """
    return GPUInferenceWorker(model_path, device=device, **kwargs)


class MockInferenceWorker(InferenceWorker):
    """Mock inference worker for testing without GPU requirements.

    Provides the same interface as GPUInferenceWorker but uses random
    outputs for testing purposes.
    """

    def __init__(self,
                 model_path: str,
                 device: str = 'cpu',
                 batch_size: int = 64,
                 timeout_ms: float = 3.0,
                 use_mixed_precision: bool = False):
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms / 1000.0
        self.use_mixed_precision = use_mixed_precision

        # Thread control
        self._stop_event = threading.Event()
        self._worker_thread = None
        self._is_running = False

        # Mock metrics
        self._request_count = 0
        self._batch_count = 0

        # Setup logging
        self.logger = logging.getLogger('MockInferenceWorker')

    def warmup(self, input_shape: Tuple[int, int, int]) -> None:
        """Mock warmup - just log."""
        self.logger.info(f"Mock warmup with shape {input_shape}")

    def inference_loop(self,
                      input_queue: Queue,
                      output_queues: List[Queue]) -> None:
        """Mock inference loop."""
        self.logger.info("Mock inference loop started")

        try:
            while not self._stop_event.is_set():
                try:
                    request = input_queue.get(timeout=self.timeout_ms)

                    # Mock processing time
                    time.sleep(0.001)  # 1ms mock processing

                    # Create mock result
                    mock_policy = np.random.dirichlet(np.ones(225))  # Gomoku 15x15
                    mock_value = np.random.uniform(-1, 1)

                    result = InferenceResult(
                        node_id=request.leaf_node_id,
                        policy=mock_policy,
                        value=mock_value,
                        path=request.path,
                        processing_time_ms=1.0
                    )

                    # Distribute to random output queue
                    if output_queues:
                        queue_idx = request.leaf_node_id % len(output_queues)
                        output_queues[queue_idx].put(result, timeout=1.0)

                    self._request_count += 1

                except Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Mock inference error: {e}")

        finally:
            self.logger.info("Mock inference loop ended")

    def batch_inference(self,
                       positions: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Mock batch inference."""
        batch_size = len(positions)

        # Mock policies and values
        policies = np.random.dirichlet(np.ones(225), size=batch_size)
        values = np.random.uniform(-1, 1, size=batch_size)

        return policies, values

    def get_metrics(self) -> Dict[str, float]:
        """Mock metrics."""
        return {
            'gpu_utilization': 0.0,
            'average_batch_size': 1.0,
            'inference_rate': self._request_count / max(1, time.time()),
            'memory_usage_gb': 0.1,
            'total_requests': self._request_count,
            'total_batches': self._batch_count,
            'total_inference_time': self._request_count * 0.001
        }

    def start_worker(self, input_queue: Queue, output_queues: List[Queue]) -> None:
        """Start mock worker."""
        if self._is_running:
            raise RuntimeError("Worker already running")

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self.inference_loop,
            args=(input_queue, output_queues),
            daemon=True
        )
        self._worker_thread.start()
        self._is_running = True

    def stop_worker(self, timeout: float = 5.0) -> None:
        """Stop mock worker."""
        if not self._is_running:
            return

        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=timeout)
        self._is_running = False

    def is_running(self) -> bool:
        """Check if mock worker is running."""
        return self._is_running