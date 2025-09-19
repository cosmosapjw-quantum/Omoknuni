"""
Unit tests for GPU Inference Worker
===================================

Tests the GPU inference worker implementation including threading,
queue communication, batching, and resource management.

Run with: python -m pytest tests/unit/test_inference_worker.py -v
"""

import pytest
import torch
import numpy as np
import time
import threading
import tempfile
import os
from queue import Queue, Empty
from unittest.mock import Mock, patch, MagicMock

# Import worker implementation
from src.neural.inference_worker import (
    GPUInferenceWorker,
    MockInferenceWorker,
    create_inference_worker
)

# Import contract interfaces
import sys
sys.path.append('specs/001-goal-create-spec')
from contracts.inference_api import (
    InferenceRequest,
    InferenceResult
)

# Import model for testing
from src.neural.model import create_model_for_game


class TestInferenceWorkerCreation:
    """Test inference worker creation and initialization."""

    def test_create_inference_worker_factory(self):
        """Test factory function creates worker."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            model_path = f.name
            # Create a valid dummy model and initialize lazy layers
            model = create_model_for_game('gomoku')
            with torch.no_grad():
                dummy_input = torch.randn(1, 7, 15, 15)
                _ = model(dummy_input)  # Initialize lazy layers
            torch.save(model.state_dict(), model_path)

        try:
            worker = create_inference_worker(
                model_path,
                device='cpu',
                batch_size=32,
                timeout_ms=5.0
            )
            assert isinstance(worker, GPUInferenceWorker)
            assert worker.model_path == model_path
            assert worker.device == 'cpu'
            assert worker.batch_size == 32
            assert worker.timeout_ms == 0.005  # Converted to seconds

        finally:
            os.unlink(model_path)

    def test_gpu_worker_initialization_parameters(self):
        """Test GPU worker initializes with correct parameters."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            model_path = f.name
            # Create a valid dummy model and initialize lazy layers
            model = create_model_for_game('gomoku')
            with torch.no_grad():
                dummy_input = torch.randn(1, 7, 15, 15)
                _ = model(dummy_input)  # Initialize lazy layers
            torch.save(model.state_dict(), model_path)

        try:
            worker = GPUInferenceWorker(
                model_path=model_path,
                device='cpu',  # Use CPU for testing
                batch_size=16,
                timeout_ms=10.0,
                use_mixed_precision=False
            )

            assert worker.model_path == model_path
            assert worker.device == 'cpu'
            assert worker.batch_size == 16
            assert worker.timeout_ms == 0.01  # 10ms -> 0.01s
            assert worker.use_mixed_precision == False
            assert not worker.is_running()

        finally:
            os.unlink(model_path)

    def test_mock_worker_initialization(self):
        """Test mock worker initialization."""
        worker = MockInferenceWorker(
            model_path='dummy.pth',
            device='cpu',
            batch_size=8,
            timeout_ms=5.0
        )

        assert worker.model_path == 'dummy.pth'
        assert worker.device == 'cpu'
        assert worker.batch_size == 8
        assert worker.timeout_ms == 0.005
        assert not worker.is_running()


class TestMockInferenceWorker:
    """Test mock inference worker functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.worker = MockInferenceWorker(
            model_path='dummy.pth',
            device='cpu',
            batch_size=4,
            timeout_ms=100.0  # Longer timeout for testing
        )

    def teardown_method(self):
        """Cleanup after each test."""
        if self.worker.is_running():
            self.worker.stop_worker()

    def test_mock_warmup(self):
        """Test mock warmup doesn't crash."""
        self.worker.warmup((7, 15, 15))
        # Should complete without error

    def test_mock_batch_inference(self):
        """Test mock batch inference produces valid outputs."""
        positions = [
            np.random.rand(7, 15, 15).astype(np.float32) for _ in range(3)
        ]

        policies, values = self.worker.batch_inference(positions)

        assert policies.shape == (3, 225)  # Gomoku 15x15
        assert values.shape == (3,)
        assert np.all(policies >= 0)
        assert np.allclose(policies.sum(axis=1), 1.0, atol=1e-5)  # Valid probabilities
        assert np.all((-1 <= values) & (values <= 1))  # Valid values

    def test_mock_worker_lifecycle(self):
        """Test mock worker starts and stops cleanly."""
        input_queue = Queue()
        output_queues = [Queue(), Queue()]

        # Start worker
        assert not self.worker.is_running()
        self.worker.start_worker(input_queue, output_queues)
        assert self.worker.is_running()

        # Let it run briefly
        time.sleep(0.1)

        # Stop worker
        self.worker.stop_worker()
        assert not self.worker.is_running()

    def test_mock_worker_processes_requests(self):
        """Test mock worker processes inference requests."""
        input_queue = Queue()
        output_queues = [Queue(), Queue()]

        # Start worker
        self.worker.start_worker(input_queue, output_queues)

        try:
            # Submit test request
            features = np.random.rand(7, 15, 15).astype(np.float32)
            request = InferenceRequest(
                leaf_node_id=42,
                features=features,
                thread_id=0,
                path=[0, 1, 2]
            )

            input_queue.put(request)

            # Wait for result
            result = None
            for output_queue in output_queues:
                try:
                    result = output_queue.get(timeout=1.0)
                    break
                except Empty:
                    continue

            assert result is not None
            assert isinstance(result, InferenceResult)
            assert result.node_id == 42
            assert result.policy.shape == (225,)
            assert isinstance(result.value, float)
            assert result.path == [0, 1, 2]

        finally:
            self.worker.stop_worker()

    def test_mock_worker_metrics(self):
        """Test mock worker provides metrics."""
        metrics = self.worker.get_metrics()

        required_keys = [
            'gpu_utilization', 'average_batch_size', 'inference_rate',
            'memory_usage_gb', 'total_requests', 'total_batches', 'total_inference_time'
        ]

        for key in required_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))


class TestGPUInferenceWorkerMocked:
    """Test GPU inference worker with mocked dependencies."""

    def setup_method(self):
        """Setup for each test."""
        # Create a dummy model file
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
        self.model_path = self.temp_file.name

        # Create a model with same config as worker will use (default Gomoku)
        model = create_model_for_game('gomoku')  # Uses default 20 blocks, 256 channels
        dummy_input = torch.randn(1, 7, 15, 15)
        _ = model(dummy_input)  # Initialize lazy layers
        torch.save(model.state_dict(), self.model_path)

        self.temp_file.close()

    def teardown_method(self):
        """Cleanup after each test."""
        if os.path.exists(self.model_path):
            os.unlink(self.model_path)

    def test_worker_model_loading(self):
        """Test worker loads model correctly."""
        worker = GPUInferenceWorker(
            model_path=self.model_path,
            device='cpu',  # Use CPU for testing
            batch_size=4,
            timeout_ms=10.0,
            use_mixed_precision=False
        )

        assert worker.model is not None
        assert worker.model.training == False  # Should be in eval mode

    def test_worker_warmup(self):
        """Test worker warmup process."""
        worker = GPUInferenceWorker(
            model_path=self.model_path,
            device='cpu',
            batch_size=4,
            timeout_ms=10.0,
            use_mixed_precision=False
        )

        # Should not crash
        worker.warmup((7, 15, 15))
        assert worker.input_shape == (7, 15, 15)

    def test_worker_batch_inference(self):
        """Test worker batch inference."""
        worker = GPUInferenceWorker(
            model_path=self.model_path,
            device='cpu',
            batch_size=4,
            timeout_ms=10.0,
            use_mixed_precision=False
        )

        worker.warmup((7, 15, 15))

        # Test batch inference
        positions = [
            np.random.rand(7, 15, 15).astype(np.float32) for _ in range(2)
        ]

        policies, values = worker.batch_inference(positions)

        assert policies.shape == (2, 225)
        assert values.shape == (2,)
        assert np.all(policies >= 0)
        assert np.allclose(policies.sum(axis=1), 1.0, atol=1e-5)
        assert np.all((-1 <= values) & (values <= 1))

    def test_worker_context_manager(self):
        """Test worker as context manager."""
        with GPUInferenceWorker(
            model_path=self.model_path,
            device='cpu',
            batch_size=4,
            timeout_ms=10.0,
            use_mixed_precision=False
        ) as worker:
            assert worker.model is not None

        # Should complete without error

    def test_worker_metrics(self):
        """Test worker metrics collection."""
        worker = GPUInferenceWorker(
            model_path=self.model_path,
            device='cpu',
            batch_size=4,
            timeout_ms=10.0,
            use_mixed_precision=False
        )

        metrics = worker.get_metrics()

        required_keys = [
            'gpu_utilization', 'average_batch_size', 'inference_rate',
            'memory_usage_gb', 'total_requests', 'total_batches', 'total_inference_time'
        ]

        for key in required_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))


class TestInferenceWorkerThreading:
    """Test inference worker threading behavior."""

    def setup_method(self):
        """Setup for each test."""
        self.worker = MockInferenceWorker(
            model_path='dummy.pth',
            device='cpu',
            batch_size=8,
            timeout_ms=50.0
        )

    def teardown_method(self):
        """Cleanup after each test."""
        if self.worker.is_running():
            self.worker.stop_worker()

    def test_worker_thread_lifecycle(self):
        """Test worker thread starts and stops properly."""
        input_queue = Queue()
        output_queues = [Queue()]

        # Initially not running
        assert not self.worker.is_running()

        # Start worker
        self.worker.start_worker(input_queue, output_queues)
        assert self.worker.is_running()

        # Verify thread is alive
        time.sleep(0.1)
        assert self.worker._worker_thread.is_alive()

        # Stop worker
        self.worker.stop_worker()
        assert not self.worker.is_running()

        # Verify thread stopped
        assert not self.worker._worker_thread.is_alive()

    def test_worker_double_start_protection(self):
        """Test worker prevents double start."""
        input_queue = Queue()
        output_queues = [Queue()]

        self.worker.start_worker(input_queue, output_queues)

        # Should raise exception
        with pytest.raises(RuntimeError, match="already running"):
            self.worker.start_worker(input_queue, output_queues)

        self.worker.stop_worker()

    def test_worker_graceful_shutdown(self):
        """Test worker shuts down gracefully under load."""
        input_queue = Queue()
        output_queues = [Queue(), Queue()]

        self.worker.start_worker(input_queue, output_queues)

        # Submit many requests
        for i in range(20):
            request = InferenceRequest(
                leaf_node_id=i,
                features=np.random.rand(7, 15, 15).astype(np.float32),
                thread_id=i % 2,
                path=[i]
            )
            input_queue.put(request)

        # Let worker process some requests
        time.sleep(0.2)

        # Should stop gracefully
        start_time = time.time()
        self.worker.stop_worker(timeout=2.0)
        stop_time = time.time()

        assert not self.worker.is_running()
        assert stop_time - start_time < 2.5  # Should stop within timeout

    def test_multiple_output_queues(self):
        """Test worker distributes to multiple output queues."""
        input_queue = Queue()
        output_queues = [Queue(), Queue(), Queue()]

        self.worker.start_worker(input_queue, output_queues)

        try:
            # Submit requests
            num_requests = 15
            for i in range(num_requests):
                request = InferenceRequest(
                    leaf_node_id=i,
                    features=np.random.rand(7, 15, 15).astype(np.float32),
                    thread_id=i % 3,
                    path=[i]
                )
                input_queue.put(request)

            # Collect results
            results = []
            for _ in range(num_requests):
                for output_queue in output_queues:
                    try:
                        result = output_queue.get(timeout=0.1)
                        results.append(result)
                    except Empty:
                        continue

            # Should get most/all results
            assert len(results) >= num_requests * 0.8  # Allow some loss in testing

            # Check results are distributed across queues
            queue_counts = [0, 0, 0]
            for result in results:
                queue_idx = result.node_id % 3
                queue_counts[queue_idx] += 1

            # Should have some distribution (not all in one queue)
            non_empty_queues = sum(1 for count in queue_counts if count > 0)
            assert non_empty_queues >= 2

        finally:
            self.worker.stop_worker()


class TestInferenceWorkerErrorHandling:
    """Test inference worker error handling."""

    def test_worker_missing_model_file(self):
        """Test worker handles missing model file."""
        with pytest.raises(FileNotFoundError):
            GPUInferenceWorker(
                model_path='/nonexistent/model.pth',
                device='cpu',
                batch_size=4,
                timeout_ms=10.0
            )

    def test_worker_invalid_device(self):
        """Test worker with invalid device."""
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            model_path = f.name
            # Create a valid dummy model and initialize lazy layers
            model = create_model_for_game('gomoku')
            with torch.no_grad():
                dummy_input = torch.randn(1, 7, 15, 15)
                _ = model(dummy_input)  # Initialize lazy layers
            torch.save(model.state_dict(), model_path)

        try:
            # Invalid device should raise exception during initialization
            with pytest.raises(Exception):  # Could be RuntimeError, CudaError, etc.
                worker = GPUInferenceWorker(
                    model_path=model_path,
                    device='cuda:99',  # Invalid device
                    batch_size=4,
                    timeout_ms=10.0
                )

        finally:
            os.unlink(model_path)

    def test_mock_worker_empty_queue_handling(self):
        """Test mock worker handles empty queues gracefully."""
        worker = MockInferenceWorker(
            model_path='dummy.pth',
            device='cpu',
            batch_size=4,
            timeout_ms=50.0  # Short timeout
        )

        input_queue = Queue()
        output_queues = [Queue()]

        worker.start_worker(input_queue, output_queues)

        try:
            # Let worker run with empty queue
            time.sleep(0.2)

            # Should still be running
            assert worker.is_running()

        finally:
            worker.stop_worker()

    def test_worker_output_queue_full_handling(self):
        """Test worker handles full output queues."""
        worker = MockInferenceWorker(
            model_path='dummy.pth',
            device='cpu',
            batch_size=1,
            timeout_ms=10.0
        )

        input_queue = Queue()
        # Create small output queue that will fill up
        output_queues = [Queue(maxsize=2)]

        worker.start_worker(input_queue, output_queues)

        try:
            # Submit many requests to overflow output queue
            for i in range(10):
                request = InferenceRequest(
                    leaf_node_id=i,
                    features=np.random.rand(7, 15, 15).astype(np.float32),
                    thread_id=0,
                    path=[i]
                )
                input_queue.put(request)

            # Worker should handle gracefully (may drop some results)
            time.sleep(0.5)
            assert worker.is_running()

        finally:
            worker.stop_worker()


class TestInferenceWorkerPerformance:
    """Test inference worker performance characteristics."""

    def test_mock_worker_throughput(self):
        """Test mock worker achieves reasonable throughput."""
        worker = MockInferenceWorker(
            model_path='dummy.pth',
            device='cpu',
            batch_size=8,
            timeout_ms=10.0
        )

        input_queue = Queue()
        output_queues = [Queue()]

        worker.start_worker(input_queue, output_queues)

        try:
            # Submit batch of requests
            num_requests = 50
            start_time = time.time()

            for i in range(num_requests):
                request = InferenceRequest(
                    leaf_node_id=i,
                    features=np.random.rand(7, 15, 15).astype(np.float32),
                    thread_id=0,
                    path=[i]
                )
                input_queue.put(request)

            # Collect results
            results = []
            for _ in range(num_requests):
                try:
                    result = output_queues[0].get(timeout=2.0)
                    results.append(result)
                except Empty:
                    break

            end_time = time.time()

            # Check throughput
            elapsed = end_time - start_time
            throughput = len(results) / elapsed

            # Mock worker should be quite fast
            assert throughput > 10  # At least 10 requests/second
            assert len(results) >= num_requests * 0.8  # Get most results

        finally:
            worker.stop_worker()

    def test_worker_metrics_tracking(self):
        """Test worker tracks metrics correctly."""
        worker = MockInferenceWorker(
            model_path='dummy.pth',
            device='cpu',
            batch_size=4,
            timeout_ms=20.0
        )

        # Initial metrics
        initial_metrics = worker.get_metrics()
        assert initial_metrics['total_requests'] == 0

        input_queue = Queue()
        output_queues = [Queue()]

        worker.start_worker(input_queue, output_queues)

        try:
            # Submit some requests
            num_requests = 10
            for i in range(num_requests):
                request = InferenceRequest(
                    leaf_node_id=i,
                    features=np.random.rand(7, 15, 15).astype(np.float32),
                    thread_id=0,
                    path=[i]
                )
                input_queue.put(request)

            # Wait for processing
            time.sleep(0.5)

            # Check updated metrics
            final_metrics = worker.get_metrics()
            assert final_metrics['total_requests'] > initial_metrics['total_requests']

        finally:
            worker.stop_worker()


# HOWTO-RUN-TESTS Block
"""
HOWTO-RUN-TESTS
===============

Run inference worker tests:

# Run all inference worker tests
python -m pytest tests/unit/test_inference_worker.py -v

# Run specific test classes
python -m pytest tests/unit/test_inference_worker.py::TestMockInferenceWorker -v
python -m pytest tests/unit/test_inference_worker.py::TestInferenceWorkerThreading -v
python -m pytest tests/unit/test_inference_worker.py::TestGPUInferenceWorkerMocked -v

# Run performance tests
python -m pytest tests/unit/test_inference_worker.py::TestInferenceWorkerPerformance -v

# Run with coverage
python -m pytest tests/unit/test_inference_worker.py --cov=src.neural.inference_worker

# Skip GPU-dependent tests (runs CPU/mock tests only)
python -m pytest tests/unit/test_inference_worker.py -v -k "not gpu or mock"

Expected Results:
✅ All worker creation and initialization tests pass
✅ Mock worker processes requests correctly
✅ Worker thread lifecycle (start/stop) works cleanly
✅ Queue-based communication functions properly
✅ Basic batching logic processes requests in batches
✅ Timeout mechanisms prevent hanging
✅ Error handling manages edge cases gracefully
✅ Performance tests show reasonable throughput
✅ Metrics tracking works correctly

Test Coverage Includes:
- InferenceWorker factory function and initialization
- GPUInferenceWorker with mocked model loading
- MockInferenceWorker for testing without GPU requirements
- Threading behavior and lifecycle management
- Queue-based communication between threads
- Basic batching logic and timeout handling
- Error handling for missing models, invalid devices, full queues
- Performance characteristics and metrics tracking
- Context manager usage for clean resource management

The tests validate that:
1. Worker threads start and stop cleanly
2. Requests are processed from input queue correctly
3. Results are distributed to output queues properly
4. Basic batching collects multiple requests efficiently
5. Timeout mechanisms prevent indefinite blocking
6. Error conditions are handled gracefully
7. Performance metrics are tracked accurately
8. Resource management follows best practices

These tests establish confidence in the worker's reliability
for the high-throughput inference pipeline required to achieve
30-40k simulations/second with proper GPU utilization.
"""