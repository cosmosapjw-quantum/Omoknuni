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
        """Collect a batch of requests from input queue.

        Args:
            input_queue: Queue of inference requests

        Returns:
            List of inference requests (may be empty)
        """
        batch = []
        start_time = time.time()

        # Try to get first request with timeout
        try:
            first_request = input_queue.get(timeout=self.timeout_ms)
            batch.append(first_request)
        except Empty:
            return batch

        # Collect additional requests up to batch_size or timeout
        while len(batch) < self.batch_size:
            elapsed = time.time() - start_time
            remaining_timeout = max(0, self.timeout_ms - elapsed)

            if remaining_timeout <= 0:
                break

            try:
                request = input_queue.get(timeout=remaining_timeout)
                batch.append(request)
            except Empty:
                break

        return batch

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
        """Update performance metrics.

        Args:
            batch_size: Size of processed batch
            inference_time: Time taken for inference (seconds)
        """
        with self._metrics_lock:
            self._metrics['total_requests'] += batch_size
            self._metrics['total_batches'] += 1
            self._metrics['total_inference_time'] += inference_time
            self._metrics['batch_sizes'].append(batch_size)
            self._metrics['inference_times'].append(inference_time)

    def get_metrics(self) -> Dict[str, float]:
        """Get inference performance metrics.

        Returns:
            dict: Metrics including:
                - 'gpu_utilization': Current GPU usage percentage
                - 'average_batch_size': Mean batch size over recent window
                - 'inference_rate': Positions processed per second
                - 'memory_usage_gb': Current VRAM usage in GB
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

            # Get GPU utilization and memory usage
            gpu_utilization = self._get_gpu_utilization()
            memory_usage_gb = self._get_memory_usage()

            return {
                'gpu_utilization': gpu_utilization,
                'average_batch_size': avg_batch_size,
                'inference_rate': inference_rate,
                'memory_usage_gb': memory_usage_gb,
                'total_requests': self._metrics['total_requests'],
                'total_batches': self._metrics['total_batches'],
                'total_inference_time': self._metrics['total_inference_time']
            }

    def _get_gpu_utilization(self) -> float:
        """Get current GPU utilization percentage."""
        try:
            if self.device.startswith('cuda') and torch.cuda.is_available():
                # Simple memory-based estimation since we don't have nvidia-ml-py
                device_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
                memory_used = torch.cuda.memory_allocated(device_idx)
                memory_total = torch.cuda.get_device_properties(device_idx).total_memory
                return (memory_used / memory_total) * 100.0
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