"""
Unit tests for Search Coordinator
=================================

Tests the asynchronous search coordinator for multi-threaded MCTS with
inference request queueing and performance monitoring.
"""

import pytest
import time
import threading
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from concurrent.futures import Future
import queue

from src.core.search_coordinator import (
    SearchCoordinator,
    SearchRequest,
    SearchResult,
    InferenceRequest,
    CoordinatorMetrics,
    create_search_coordinator
)


class MockInferenceWorker:
    """Mock inference worker for testing."""

    def __init__(self):
        self.running = False
        self.requests_processed = 0

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def process_batch(self, batch):
        self.requests_processed += len(batch)
        return [(np.random.rand(225), np.random.uniform(-1, 1)) for _ in batch]


class MockGameState:
    """Mock game state for testing."""

    def __init__(self):
        self.move_count = 0
        self.terminal = False

    def get_tensor_representation(self):
        return np.random.rand(36, 15, 15)

    def is_terminal(self):
        return self.terminal

    def get_legal_moves(self):
        return np.ones(225, dtype=bool)


class TestSearchCoordinator:
    """Test search coordinator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_inference_worker = MockInferenceWorker()
        self.coordinator = SearchCoordinator(
            inference_worker=self.mock_inference_worker,
            max_threads=4,
            max_queue_size=100,
            monitoring_interval=0.1
        )

    def teardown_method(self):
        """Clean up after tests."""
        if self.coordinator.running:
            self.coordinator.stop()

    def test_coordinator_initialization(self):
        """Test coordinator initializes correctly."""
        assert self.coordinator.max_threads == 4
        assert self.coordinator.max_queue_size == 100
        assert self.coordinator.monitoring_interval == 0.1
        assert not self.coordinator.running
        assert self.coordinator.metrics.active_searches == 0
        assert self.coordinator.metrics.completed_searches == 0

    def test_start_and_stop(self):
        """Test coordinator starts and stops correctly."""
        # Test start
        self.coordinator.start()
        assert self.coordinator.running
        assert self.mock_inference_worker.running
        assert self.coordinator.inference_coordinator_thread.is_alive()
        assert self.coordinator.metrics_monitor_thread.is_alive()

        # Test stop
        self.coordinator.stop()
        assert not self.coordinator.running
        assert not self.coordinator.inference_coordinator_thread.is_alive()
        assert not self.coordinator.metrics_monitor_thread.is_alive()

    def test_submit_search_not_running(self):
        """Test submitting search when coordinator is not running."""
        request = SearchRequest(
            request_id="test_001",
            game_state=MockGameState(),
            simulations=100
        )

        with pytest.raises(RuntimeError, match="not running"):
            self.coordinator.submit_search(request)

    def test_submit_search_success(self):
        """Test successful search submission."""
        self.coordinator.start()

        request = SearchRequest(
            request_id="test_001",
            game_state=MockGameState(),
            simulations=100
        )

        future = self.coordinator.submit_search(request)
        assert isinstance(future, Future)
        assert request.request_id in self.coordinator.active_searches

        # Wait for completion
        result = future.result(timeout=5.0)
        assert isinstance(result, SearchResult)
        assert result.request_id == "test_001"
        assert result.best_move >= 0
        assert len(result.policy) == 225
        assert -1 <= result.value <= 1

    def test_multiple_concurrent_searches(self):
        """Test multiple concurrent search requests."""
        self.coordinator.start()

        requests = []
        futures = []

        for i in range(5):
            request = SearchRequest(
                request_id=f"test_{i:03d}",
                game_state=MockGameState(),
                simulations=50
            )
            requests.append(request)
            futures.append(self.coordinator.submit_search(request))

        # Wait for all to complete
        results = []
        for future in futures:
            result = future.result(timeout=10.0)
            results.append(result)

        assert len(results) == 5
        assert all(isinstance(r, SearchResult) for r in results)
        assert len(set(r.request_id for r in results)) == 5  # All unique

    def test_inference_request_queueing(self):
        """Test inference request queueing functionality."""
        self.coordinator.start()

        game_state = MockGameState()
        thread_id = threading.get_ident()

        future = self.coordinator.request_inference(game_state, thread_id)
        assert isinstance(future, Future)

        # Request should be in queue
        assert self.coordinator.inference_request_queue.qsize() > 0

        # Let it process
        time.sleep(0.1)

    def test_inference_queue_full(self):
        """Test behavior when inference queue is full."""
        # Create coordinator with very small queue
        small_coordinator = SearchCoordinator(
            inference_worker=self.mock_inference_worker,
            max_queue_size=1
        )

        # Fill the queue
        future1 = small_coordinator.request_inference(MockGameState(), 1)
        assert isinstance(future1, Future)

        # This should raise queue.Full
        future2 = small_coordinator.request_inference(MockGameState(), 2)

        # The future should contain the exception
        with pytest.raises(queue.Full):
            future2.result(timeout=1.0)

    def test_metrics_collection(self):
        """Test metrics collection and monitoring."""
        self.coordinator.start()

        # Submit some searches
        for i in range(3):
            request = SearchRequest(
                request_id=f"test_{i}",
                game_state=MockGameState(),
                simulations=50
            )
            self.coordinator.submit_search(request)

        # Let searches start
        time.sleep(0.05)

        metrics = self.coordinator.get_metrics()
        assert metrics.active_searches > 0
        assert metrics.thread_utilization > 0

        # Wait for completion
        time.sleep(1.0)

        final_metrics = self.coordinator.get_metrics()
        assert final_metrics.completed_searches >= 3

    def test_metrics_history_tracking(self):
        """Test metrics history tracking."""
        self.coordinator.start()

        # Let metrics monitor run for a bit
        time.sleep(0.3)

        assert len(self.coordinator.metrics_history) > 0
        assert all(isinstance(m, CoordinatorMetrics) for m in self.coordinator.metrics_history)

    def test_search_callback(self):
        """Test search result callback functionality."""
        self.coordinator.start()

        callback_results = []

        def test_callback(result):
            callback_results.append(result)

        request = SearchRequest(
            request_id="test_callback",
            game_state=MockGameState(),
            simulations=50,
            result_callback=test_callback
        )

        future = self.coordinator.submit_search(request)
        result = future.result(timeout=5.0)

        # Callback should have been called
        assert len(callback_results) == 1
        assert callback_results[0].request_id == "test_callback"

    def test_search_with_temperature_and_noise(self):
        """Test search request with temperature and noise parameters."""
        self.coordinator.start()

        request = SearchRequest(
            request_id="test_temp_noise",
            game_state=MockGameState(),
            simulations=100,
            temperature=0.8,
            add_noise=True
        )

        future = self.coordinator.submit_search(request)
        result = future.result(timeout=5.0)

        assert result.request_id == "test_temp_noise"
        # Temperature and noise would affect the actual search in real implementation

    def test_search_timing_metrics(self):
        """Test search timing tracking."""
        self.coordinator.start()

        request = SearchRequest(
            request_id="test_timing",
            game_state=MockGameState(),
            simulations=100
        )

        start_time = time.time()
        future = self.coordinator.submit_search(request)
        result = future.result(timeout=5.0)
        end_time = time.time()

        assert result.processing_time_ms > 0
        assert result.processing_time_ms < (end_time - start_time) * 1000 + 100  # Some tolerance

    def test_shutdown_graceful(self):
        """Test graceful shutdown cancels pending searches."""
        self.coordinator.start()

        # Submit searches that will take some time
        futures = []
        for i in range(5):
            request = SearchRequest(
                request_id=f"test_shutdown_{i}",
                game_state=MockGameState(),
                simulations=1000  # Large number to ensure they're running
            )
            futures.append(self.coordinator.submit_search(request))

        # Let searches start
        time.sleep(0.1)

        # Stop coordinator
        self.coordinator.stop()

        # Some futures may be cancelled
        cancelled_count = sum(1 for f in futures if f.cancelled())
        completed_count = sum(1 for f in futures if f.done() and not f.cancelled())

        # Should have some combination of cancelled and completed
        assert cancelled_count + completed_count == len(futures)

    @patch('src.core.search_coordinator.MetricsCollector')
    def test_telemetry_integration(self, mock_metrics_collector):
        """Test integration with telemetry system."""
        mock_telemetry = Mock()
        mock_metrics_collector.return_value = mock_telemetry

        coordinator = SearchCoordinator(
            inference_worker=self.mock_inference_worker,
            max_threads=2
        )
        coordinator.start()

        # Let metrics monitor run
        time.sleep(0.2)

        coordinator.stop()

        # Should have recorded metrics to telemetry
        assert mock_telemetry.record_gauge.call_count > 0

    def test_thread_safety(self):
        """Test thread safety of coordinator operations."""
        self.coordinator.start()

        results = []
        errors = []

        def submit_searches():
            try:
                for i in range(10):
                    request = SearchRequest(
                        request_id=f"thread_test_{threading.get_ident()}_{i}",
                        game_state=MockGameState(),
                        simulations=50
                    )
                    future = self.coordinator.submit_search(request)
                    result = future.result(timeout=5.0)
                    results.append(result)
            except Exception as e:
                errors.append(e)

        # Run multiple threads submitting searches
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=submit_searches)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0
        assert len(results) == 30  # 3 threads * 10 searches each
        assert len(set(r.request_id for r in results)) == 30  # All unique


class TestSearchCoordinatorFactory:
    """Test search coordinator factory function."""

    def test_create_search_coordinator_default_config(self):
        """Test factory with default configuration."""
        inference_worker = MockInferenceWorker()
        config = {}

        coordinator = create_search_coordinator(inference_worker, config)

        assert coordinator.max_threads == 8  # Default
        assert coordinator.max_queue_size == 1000  # Default
        assert coordinator.monitoring_interval == 1.0  # Default

    def test_create_search_coordinator_custom_config(self):
        """Test factory with custom configuration."""
        inference_worker = MockInferenceWorker()
        config = {
            'max_threads': 12,
            'max_queue_size': 2000,
            'monitoring_interval': 0.5
        }

        coordinator = create_search_coordinator(inference_worker, config)

        assert coordinator.max_threads == 12
        assert coordinator.max_queue_size == 2000
        assert coordinator.monitoring_interval == 0.5


class TestDataStructures:
    """Test data structure functionality."""

    def test_search_request_creation(self):
        """Test SearchRequest creation and attributes."""
        game_state = MockGameState()

        request = SearchRequest(
            request_id="test_req",
            game_state=game_state,
            simulations=800,
            time_limit_ms=5000.0,
            temperature=0.7,
            add_noise=True
        )

        assert request.request_id == "test_req"
        assert request.game_state is game_state
        assert request.simulations == 800
        assert request.time_limit_ms == 5000.0
        assert request.temperature == 0.7
        assert request.add_noise is True

    def test_search_result_creation(self):
        """Test SearchResult creation and attributes."""
        policy = np.random.rand(225)
        search_info = {'simulations': 800, 'depth': 15}

        result = SearchResult(
            request_id="test_result",
            best_move=42,
            policy=policy,
            value=0.5,
            search_info=search_info,
            processing_time_ms=150.5
        )

        assert result.request_id == "test_result"
        assert result.best_move == 42
        assert np.array_equal(result.policy, policy)
        assert result.value == 0.5
        assert result.search_info == search_info
        assert result.processing_time_ms == 150.5

    def test_inference_request_creation(self):
        """Test InferenceRequest creation and attributes."""
        game_state = MockGameState()
        future = Future()

        request = InferenceRequest(
            request_id="inf_test",
            game_state=game_state,
            thread_id=12345,
            result_future=future
        )

        assert request.request_id == "inf_test"
        assert request.game_state is game_state
        assert request.thread_id == 12345
        assert request.result_future is future
        assert isinstance(request.timestamp, float)

    def test_coordinator_metrics_defaults(self):
        """Test CoordinatorMetrics default values."""
        metrics = CoordinatorMetrics()

        assert metrics.active_searches == 0
        assert metrics.completed_searches == 0
        assert metrics.total_simulations == 0
        assert metrics.average_search_time_ms == 0.0
        assert metrics.thread_utilization == 0.0
        assert metrics.inference_queue_depth == 0
        assert metrics.searches_per_second == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])