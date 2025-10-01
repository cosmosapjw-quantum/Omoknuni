"""
Asynchronous Search Coordinator
===============================

Coordinates multiple MCTS search threads with asynchronous neural network inference.
Manages thread pools, request queuing, and performance monitoring for optimal throughput.

Key responsibilities:
- Thread pool management for parallel MCTS search
- Inference request queueing and result distribution
- Performance monitoring and metrics collection
- Thread synchronization and communication
"""

import threading
import time
import queue
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from threading import Event, Lock, RLock
import uuid

# Import telemetry for performance monitoring
from src.telemetry.metrics import MetricsCollector

# Import inference interfaces
from src.neural.inference_worker import GPUInferenceWorker
from src.neural.cpu_inference import CPUInferenceWorker

# Import error handling framework
from src.utils.errors import (
    ThreadCoordinationError, CriticalInferenceError, InferenceError,
    ThreadHealthMonitor, with_error_handling, error_reporter
)


@dataclass
class SearchRequest:
    """Request for MCTS search on a position."""

    request_id: str
    game_state: Any  # GameState instance
    simulations: int
    time_limit_ms: Optional[float] = None
    temperature: float = 1.0
    add_noise: bool = False
    result_callback: Optional[Callable] = None


@dataclass
class SearchResult:
    """Result of MCTS search."""

    request_id: str
    best_move: int
    policy: np.ndarray  # Visit count distribution (normalized)
    value: float        # Position evaluation
    search_info: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0


@dataclass
class InferenceRequest:
    """Request for neural network inference."""

    request_id: str
    game_state: Any
    thread_id: int
    timestamp: float = field(default_factory=time.time)
    result_future: Optional[Future] = None


@dataclass
class CoordinatorMetrics:
    """Performance metrics for the search coordinator."""

    active_searches: int = 0
    completed_searches: int = 0
    total_simulations: int = 0
    average_search_time_ms: float = 0.0
    thread_utilization: float = 0.0
    inference_queue_depth: int = 0
    searches_per_second: float = 0.0


class SearchCoordinator:
    """Asynchronous search coordinator for multi-threaded MCTS."""

    def __init__(self,
                 inference_worker: GPUInferenceWorker,
                 max_threads: int = 8,
                 max_queue_size: int = 1000,
                 monitoring_interval: float = 1.0):
        """Initialize search coordinator.

        Args:
            inference_worker: GPU inference worker for neural network evaluation
            max_threads: Maximum number of search threads
            max_queue_size: Maximum inference request queue size
            monitoring_interval: Performance monitoring update interval (seconds)
        """
        self.inference_worker = inference_worker
        self.max_threads = max_threads
        self.max_queue_size = max_queue_size
        self.monitoring_interval = monitoring_interval

        # Thread management
        self.thread_pool = ThreadPoolExecutor(max_workers=max_threads, thread_name_prefix="search")
        self.active_searches: Dict[str, Future] = {}
        self.search_lock = RLock()

        # Inference coordination
        self.inference_request_queue = queue.Queue(maxsize=max_queue_size)
        self.pending_inference_requests: Dict[str, InferenceRequest] = {}
        self.inference_lock = threading.Lock()

        # Performance monitoring
        self.metrics = CoordinatorMetrics()
        self.metrics_history = deque(maxlen=100)
        self.metrics_lock = threading.Lock()
        self.last_metrics_update = time.time()

        # Control flags
        self.running = False
        self.shutdown_event = Event()

        # Background threads
        self.inference_coordinator_thread: Optional[threading.Thread] = None
        self.metrics_monitor_thread: Optional[threading.Thread] = None

        # Initialize telemetry
        self.telemetry = MetricsCollector()

        # Search timing tracking
        self.search_start_times: Dict[str, float] = {}
        self.completed_search_times = deque(maxlen=1000)

        # Logger
        self.logger = logging.getLogger(__name__)

        # Error handling and thread health monitoring
        self.thread_health = ThreadHealthMonitor(
            max_consecutive_failures=5,  # More aggressive than default
            failure_backoff=0.5,
            max_backoff=5.0
        )
        self.critical_error_count = 0
        self.max_critical_errors = 3

    def start(self) -> None:
        """Start the search coordinator and background threads."""
        if self.running:
            self.logger.warning("Search coordinator already running")
            return

        self.logger.info(f"Starting search coordinator with {self.max_threads} threads")
        self.running = True
        self.shutdown_event.clear()

        # Start inference worker if not already running
        if not hasattr(self.inference_worker, 'running') or not self.inference_worker.running:
            self.inference_worker.start()

        # Start background coordination threads
        self.inference_coordinator_thread = threading.Thread(
            target=self._inference_coordinator_loop,
            name="inference_coordinator",
            daemon=True
        )
        self.inference_coordinator_thread.start()

        self.metrics_monitor_thread = threading.Thread(
            target=self._metrics_monitor_loop,
            name="metrics_monitor",
            daemon=True
        )
        self.metrics_monitor_thread.start()

        self.logger.info("Search coordinator started successfully")

    def stop(self) -> None:
        """Stop the search coordinator and shutdown threads."""
        if not self.running:
            return

        self.logger.info("Stopping search coordinator")
        self.running = False
        self.shutdown_event.set()

        # Cancel all pending searches
        with self.search_lock:
            for request_id, future in list(self.active_searches.items()):
                future.cancel()
            self.active_searches.clear()

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)

        # Wait for background threads to finish
        if self.inference_coordinator_thread and self.inference_coordinator_thread.is_alive():
            self.inference_coordinator_thread.join(timeout=5.0)
        if self.metrics_monitor_thread and self.metrics_monitor_thread.is_alive():
            self.metrics_monitor_thread.join(timeout=5.0)

        self.logger.info("Search coordinator stopped")

    def submit_search(self, request: SearchRequest) -> Future[SearchResult]:
        """Submit a search request for asynchronous execution.

        Args:
            request: Search request to execute

        Returns:
            Future that will contain the search result

        Raises:
            RuntimeError: If coordinator is not running
            ValueError: If request is invalid
        """
        if not self.running:
            raise RuntimeError("Search coordinator is not running")

        if not request.request_id:
            request.request_id = str(uuid.uuid4())

        self.logger.debug(f"Submitting search request {request.request_id}")

        # Record search start time
        with self.metrics_lock:
            self.search_start_times[request.request_id] = time.time()
            self.metrics.active_searches += 1

        # Submit to thread pool
        future = self.thread_pool.submit(self._execute_search, request)

        with self.search_lock:
            self.active_searches[request.request_id] = future

        # Add completion callback to clean up
        future.add_done_callback(lambda f: self._search_completed(request.request_id, f))

        return future

    def request_inference(self, game_state: Any, thread_id: int) -> Future[Tuple[np.ndarray, float]]:
        """Request neural network inference for a game state.

        Args:
            game_state: Game state to evaluate
            thread_id: ID of requesting thread

        Returns:
            Future containing (policy, value) tuple

        Raises:
            queue.Full: If inference queue is full
        """
        request_id = str(uuid.uuid4())
        future = Future()

        inference_req = InferenceRequest(
            request_id=request_id,
            game_state=game_state,
            thread_id=thread_id,
            result_future=future
        )

        try:
            self.inference_request_queue.put_nowait(inference_req)

            with self.inference_lock:
                self.pending_inference_requests[request_id] = inference_req

        except queue.Full:
            future.set_exception(queue.Full("Inference request queue is full"))

        return future

    def get_metrics(self) -> CoordinatorMetrics:
        """Get current performance metrics.

        Returns:
            Current coordinator metrics
        """
        with self.metrics_lock:
            return CoordinatorMetrics(
                active_searches=self.metrics.active_searches,
                completed_searches=self.metrics.completed_searches,
                total_simulations=self.metrics.total_simulations,
                average_search_time_ms=self.metrics.average_search_time_ms,
                thread_utilization=self.metrics.thread_utilization,
                inference_queue_depth=self.inference_request_queue.qsize(),
                searches_per_second=self.metrics.searches_per_second
            )

    def _execute_search(self, request: SearchRequest) -> SearchResult:
        """Execute a single search request using real MCTS implementation."""
        thread_id = threading.get_ident()
        start_time = time.time()

        self.logger.debug(f"Executing search {request.request_id} on thread {thread_id}")

        try:
            # Import MCTS engine locally to avoid circular imports
            from .mcts import AlphaZeroMCTS

            # Create MCTS engine with inference function
            def inference_fn(game_state):
                return self.request_inference(game_state, thread_id)

            mcts = AlphaZeroMCTS(inference_fn, num_threads=self.max_threads)

            # Run MCTS search
            visit_counts = mcts.search(
                root_state=request.game_state,
                simulations=request.simulations,
                add_noise=request.add_noise
            )

            # Extract policy and best move
            policy = mcts.get_policy(request.game_state, request.temperature)
            best_move = int(np.argmax(policy))
            value = mcts.get_value(request.game_state)

            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            result = SearchResult(
                request_id=request.request_id,
                best_move=best_move,
                policy=policy,
                value=value,
                processing_time_ms=processing_time,
                search_info={
                    'simulations_completed': request.simulations,
                    'thread_id': thread_id,
                    'tree_size': mcts.tree_size,
                    'visit_counts': visit_counts,
                    'legal_moves': len(request.game_state.get_legal_moves()),
                    'action_space_size': request.game_state.action_space_size
                }
            )

            # Call result callback if provided
            if request.result_callback:
                try:
                    request.result_callback(result)
                except Exception as e:
                    self.logger.error(f"Error in result callback: {e}")

            return result

        except Exception as e:
            self.logger.error(f"Error executing search {request.request_id}: {e}")
            raise
        finally:
            # Update metrics
            with self.metrics_lock:
                self.metrics.total_simulations += request.simulations

    def _search_completed(self, request_id: str, future: Future) -> None:
        """Callback when search completes."""
        with self.search_lock:
            self.active_searches.pop(request_id, None)

        with self.metrics_lock:
            self.metrics.active_searches = max(0, self.metrics.active_searches - 1)
            self.metrics.completed_searches += 1

            # Update timing metrics
            start_time = self.search_start_times.pop(request_id, None)
            if start_time:
                search_time = (time.time() - start_time) * 1000  # Convert to ms
                self.completed_search_times.append(search_time)

                if self.completed_search_times:
                    self.metrics.average_search_time_ms = sum(self.completed_search_times) / len(self.completed_search_times)

    def _inference_coordinator_loop(self) -> None:
        """Background thread that coordinates inference requests with the GPU worker."""
        self.logger.info("Inference coordinator thread started")

        while not self.shutdown_event.is_set():
            try:
                # Get inference request with timeout
                try:
                    request = self.inference_request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Process the inference request
                self._process_inference_request(request)

                # Record successful operation
                self.thread_health.record_success("inference_coordinator")

            except CriticalInferenceError as e:
                self.critical_error_count += 1
                error_reporter.report_error(e, {"thread": "inference_coordinator", "critical_count": self.critical_error_count})

                if self.critical_error_count >= self.max_critical_errors:
                    self.logger.critical(
                        f"Too many critical errors ({self.critical_error_count}), triggering emergency shutdown"
                    )
                    self._trigger_emergency_shutdown()
                    break

                # Apply backoff for critical errors
                if not self.thread_health.record_failure("inference_coordinator", e):
                    self.logger.critical("Inference coordinator thread terminating due to repeated failures")
                    break

            except (InferenceError, ThreadCoordinationError) as e:
                error_reporter.report_error(e, {"thread": "inference_coordinator"})

                if not self.thread_health.record_failure("inference_coordinator", e):
                    self.logger.critical("Inference coordinator thread terminating due to repeated failures")
                    break

            except Exception as e:
                # Wrap unexpected exceptions
                wrapped_error = ThreadCoordinationError(
                    f"Unexpected error in inference coordinator: {e}",
                    thread_name="inference_coordinator"
                )
                error_reporter.report_error(wrapped_error, {"thread": "inference_coordinator", "original_error": str(e)})

                if not self.thread_health.record_failure("inference_coordinator", wrapped_error):
                    self.logger.critical("Inference coordinator thread terminating due to repeated failures")
                    break

        self.logger.info("Inference coordinator thread stopped")

    def _process_inference_request(self, request: InferenceRequest) -> None:
        """Process a single inference request."""
        try:
            # Extract features from game state
            # In real implementation, this would call game_state.get_tensor_representation()
            # For simulation, create mock features
            features = np.random.rand(36, 15, 15)  # Mock Gomoku features

            # Submit to GPU inference worker
            # Note: This is a simplified interface - actual implementation would
            # need to batch requests and handle the GPU worker's specific API

            # For now, simulate inference result
            time.sleep(0.001)  # Simulate GPU inference time

            # Mock inference result
            policy = np.random.dirichlet([1.0] * 225)
            value = np.random.uniform(-1.0, 1.0)

            # Set result in future
            if request.result_future and not request.result_future.done():
                request.result_future.set_result((policy, value))

        except Exception as e:
            if request.result_future and not request.result_future.done():
                request.result_future.set_exception(e)
        finally:
            # Clean up
            with self.inference_lock:
                self.pending_inference_requests.pop(request.request_id, None)

    def _metrics_monitor_loop(self) -> None:
        """Background thread for performance monitoring."""
        self.logger.info("Metrics monitor thread started")

        while not self.shutdown_event.is_set():
            try:
                self._update_metrics()
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in metrics monitor: {e}")

        self.logger.info("Metrics monitor thread stopped")

    def _update_metrics(self) -> None:
        """Update performance metrics."""
        current_time = time.time()

        with self.metrics_lock:
            # Calculate thread utilization
            active_threads = len(self.active_searches)
            self.metrics.thread_utilization = (active_threads / self.max_threads) * 100

            # Calculate searches per second
            time_delta = current_time - self.last_metrics_update
            if time_delta > 0:
                completed_since_last = self.metrics.completed_searches
                if hasattr(self, '_last_completed_searches'):
                    completed_since_last -= self._last_completed_searches
                self.metrics.searches_per_second = completed_since_last / time_delta
                self._last_completed_searches = self.metrics.completed_searches

            # Update queue depth
            self.metrics.inference_queue_depth = self.inference_request_queue.qsize()

            # Store metrics history
            metrics_snapshot = CoordinatorMetrics(
                active_searches=self.metrics.active_searches,
                completed_searches=self.metrics.completed_searches,
                total_simulations=self.metrics.total_simulations,
                average_search_time_ms=self.metrics.average_search_time_ms,
                thread_utilization=self.metrics.thread_utilization,
                inference_queue_depth=self.metrics.inference_queue_depth,
                searches_per_second=self.metrics.searches_per_second
            )
            self.metrics_history.append(metrics_snapshot)

            self.last_metrics_update = current_time

        # Report to telemetry system (Note: MetricsCollector uses specific recording methods)
        # For now, we'll integrate with the existing metrics system through specialized methods
        # In a full implementation, we would extend MetricsCollector to support custom gauges

    def _trigger_emergency_shutdown(self) -> None:
        """Trigger emergency shutdown due to critical errors."""
        self.logger.critical("Emergency shutdown triggered")
        self.shutdown_event.set()

        # Try to notify any waiting requests
        try:
            while not self.inference_request_queue.empty():
                request = self.inference_request_queue.get_nowait()
                if request.result_future and not request.result_future.done():
                    request.result_future.set_exception(
                        CriticalInferenceError("System emergency shutdown")
                    )
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Error during emergency shutdown cleanup: {e}")

    @with_error_handling(reraise=False)
    def stop(self) -> None:
        """Stop the search coordinator and all background threads."""
        self.logger.info("Stopping search coordinator...")

        # Signal shutdown
        self.shutdown_event.set()
        self.running = False

        # Stop thread pool
        if hasattr(self, 'thread_pool'):
            try:
                self.thread_pool.shutdown(wait=True)
            except Exception as e:
                self.logger.error(f"Error stopping thread pool: {e}")

        # Stop inference worker
        if hasattr(self, 'inference_worker') and hasattr(self.inference_worker, 'stop'):
            try:
                self.inference_worker.stop()
            except Exception as e:
                self.logger.error(f"Error stopping inference worker: {e}")

        # Wait for coordinator threads to finish
        coordinator_thread = getattr(self, 'inference_coordinator_thread', None)
        if coordinator_thread:
            try:
                coordinator_thread.join(timeout=5.0)
                if coordinator_thread.is_alive():
                    self.logger.warning("Inference coordinator thread did not shut down gracefully")
            except Exception as e:
                self.logger.error(f"Error joining inference coordinator thread: {e}")

        metrics_thread = getattr(self, 'metrics_monitor_thread', None)
        if metrics_thread:
            try:
                metrics_thread.join(timeout=5.0)
                if metrics_thread.is_alive():
                    self.logger.warning("Metrics monitor thread did not shut down gracefully")
            except Exception as e:
                self.logger.error(f"Error joining metrics monitor thread: {e}")

        # Report final error summary
        error_summary = error_reporter.get_error_summary()
        if error_summary['total_errors'] > 0:
            self.logger.info(f"Final error summary: {error_summary['total_errors']} total errors")

        self.logger.info("Search coordinator stopped")


def create_search_coordinator(inference_worker: GPUInferenceWorker,
                             config: Dict[str, Any]) -> SearchCoordinator:
    """Factory function to create search coordinator with configuration.

    Args:
        inference_worker: GPU inference worker instance
        config: Configuration dictionary

    Returns:
        Configured SearchCoordinator instance
    """
    return SearchCoordinator(
        inference_worker=inference_worker,
        max_threads=config.get('max_threads', 8),
        max_queue_size=config.get('max_queue_size', 1000),
        monitoring_interval=config.get('monitoring_interval', 1.0)
    )
