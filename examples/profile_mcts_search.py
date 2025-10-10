"""
Profile MCTS Search - Real-world profiling of MCTS engine
==========================================================

Demonstrates profiling a real MCTS search with neural network inference.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.profiling import ProfilingSession, ProfilerConfig
from src.core.mcts import AlphaZeroMCTS
from src.neural.model import create_model_for_game

try:
    from alphazero_py import GomokuState
except ImportError:
    print("Warning: alphazero_py not available, using mock state")
    GomokuState = None


class MockInferenceWorker:
    """Mock inference worker for demo purposes."""

    def __init__(self):
        import numpy as np
        self.np = np

    def batch_inference(self, positions):
        """Simulate batch inference."""
        time.sleep(0.002)  # 2ms latency
        batch_size = len(positions)
        policies = self.np.random.randn(batch_size, 225).astype(self.np.float32)
        policies = self.np.exp(policies) / self.np.exp(policies).sum(axis=1, keepdims=True)
        values = self.np.random.randn(batch_size, 1).astype(self.np.float32)
        return policies, values


def create_mock_inference_fn(profiler_session):
    """Create inference function with profiling."""
    import numpy as np
    from concurrent.futures import Future

    worker = MockInferenceWorker()

    def inference_fn(game_state):
        """Inference function that tracks profiling metrics."""
        future = Future()

        # Track inference request
        request_id = f"req_{time.time()}"

        if profiler_session.inference:
            with profiler_session.inference.track_request(request_id, thread_id=1, batch_size=1):
                # Simulate queue wait
                with profiler_session.inference.track_stage(request_id, "queue_wait"):
                    time.sleep(0.0005)  # 0.5ms queue wait

                # Extract features
                features = game_state.get_enhanced_tensor_representation()

                # Run inference
                with profiler_session.inference.track_stage(request_id, "inference"):
                    policies, values = worker.batch_inference([features])

                policy = policies[0]
                value = values[0][0]

                future.set_result((policy, value))
        else:
            # Fallback without profiling
            features = game_state.get_enhanced_tensor_representation()
            policies, values = worker.batch_inference([features])
            future.set_result((policies[0], values[0][0]))

        return future

    return inference_fn


def profile_mcts_search(num_simulations: int = 800, num_threads: int = 4):
    """Profile a complete MCTS search."""
    print(f"Profiling MCTS search: {num_simulations} simulations, {num_threads} threads")

    # Configure profiling
    config = ProfilerConfig(
        enable_gil_profiling=True,
        enable_inference_profiling=True,
        enable_thread_profiling=True,
        enable_memory_profiling=True,
        enable_cpp_instrumentation=True,
        auto_save_reports=True,
        report_directory="profiling_reports/mcts_search"
    )

    with ProfilingSession(config) as session:
        # Create game state
        if GomokuState:
            root_state = GomokuState()
        else:
            print("Using mock state (alphazero_py not available)")
            from tests.mocks.mock_game_state import MockGameState
            root_state = MockGameState(action_space_size=225, board_size=15)

        # Create inference function with profiling
        inference_fn = create_mock_inference_fn(session)

        # Track GIL for MCTS initialization
        if session.gil:
            with session.gil.section("mcts_init"):
                mcts = AlphaZeroMCTS(
                    inference_fn=inference_fn,
                    c_puct=1.25,
                    num_threads=num_threads,
                    use_async_inference=True,
                    async_batch_size=32,
                    async_timeout_ms=1.0,
                    enable_instrumentation=True
                )

        # Run search with profiling
        print("Running search...")
        search_start = time.perf_counter()

        if session.gil:
            with session.gil.section("mcts_search"):
                visit_counts = mcts.search(root_state, num_simulations, add_noise=True)
        else:
            visit_counts = mcts.search(root_state, num_simulations, add_noise=True)

        search_time = time.perf_counter() - search_start

        print(f"Search completed in {search_time:.3f}s")
        print(f"Throughput: {num_simulations/search_time:.1f} sims/sec")

        # Get policy
        if session.gil:
            with session.gil.section("get_policy"):
                policy = mcts.get_policy(root_state, temperature=1.0)

        # Force garbage collection to see impact
        if session.memory:
            print("Forcing GC...")
            gc_stats = session.memory.force_gc()
            print(f"  Collected {gc_stats['collected_objects']} objects in {gc_stats['duration_ms']:.2f}ms")

    # Analyze results
    print("\n" + "=" * 60)
    print("Profiling Results")
    print("=" * 60)

    metrics = session.get_all_metrics()

    # GIL Analysis
    gil_summary = metrics.get('gil_metrics', {}).get('summary', {})
    print("\n--- GIL Analysis ---")
    print(f"GIL Efficiency: {gil_summary.get('gil_efficiency', 0.0):.1f}%")
    print(f"Average Wait Time: {gil_summary.get('avg_wait_time_per_thread', 0.0)*1000:.2f}ms")
    print(f"Contention Events: {gil_summary.get('total_contention_events', 0)}")

    # Inference Analysis
    inference_summary = metrics.get('inference_metrics', {}).get('summary', {})
    print("\n--- Inference Analysis ---")
    print(f"Average Latency: {inference_summary.get('avg_latency_us', 0.0)/1000:.2f}ms")
    print(f"P99 Latency: {inference_summary.get('p99_latency_us', 0.0)/1000:.2f}ms")
    print(f"Average Batch Size: {inference_summary.get('avg_batch_size', 0.0):.1f}")
    print(f"Requests/sec: {inference_summary.get('requests_per_second', 0.0):.1f}")

    # Stage breakdown
    print("\nInference Stage Breakdown:")
    stage_breakdown = metrics.get('inference_metrics', {}).get('stage_breakdown', {})
    for stage, stats in sorted(stage_breakdown.items(), key=lambda x: x[1]['percentage'], reverse=True):
        print(f"  {stage.replace('_', ' ').title():25s}: {stats['percentage']:5.1f}% ({stats['avg_us']:8.2f}μs avg)")

    # Thread Analysis
    thread_summary = metrics.get('thread_metrics', {}).get('summary', {})
    print("\n--- Thread Analysis ---")
    print(f"Thread Utilization: {metrics.get('thread_metrics', {}).get('pool_summary', {}).get('avg_thread_utilization', 0.0):.1f}%")
    print(f"Average Future Latency: {thread_summary.get('avg_future_latency_us', 0.0)/1000:.2f}ms")
    print(f"Success Rate: {thread_summary.get('success_rate', 0.0):.1f}%")

    # Memory Analysis
    memory_summary = metrics.get('memory_metrics', {}).get('summary', {})
    print("\n--- Memory Analysis ---")
    print(f"Peak Memory: {memory_summary.get('peak_memory_mb', 0.0):.1f}MB")
    print(f"Memory Growth: {memory_summary.get('memory_growth_mb', 0.0):.1f}MB")
    print(f"GC Events: {memory_summary.get('total_gc_events', 0)}")
    print(f"GC Rate: {memory_summary.get('gc_events_per_second', 0.0):.2f}/sec")

    # C++ Instrumentation
    cpp_metrics = metrics.get('cpp_instrumentation', {})
    if cpp_metrics:
        print("\n--- C++ Instrumentation (Top 5) ---")
        sorted_metrics = sorted(
            cpp_metrics.items(),
            key=lambda x: x[1].get('total_elapsed_ns', 0),
            reverse=True
        )
        for metric_name, data in sorted_metrics[:5]:
            call_count = data.get('call_count', 0)
            total_ms = data.get('total_elapsed_ns', 0) / 1e6
            avg_us = (data.get('total_elapsed_ns', 0) / call_count / 1000) if call_count > 0 else 0
            print(f"  {metric_name:25s}: {call_count:8,} calls, {total_ms:8.2f}ms total, {avg_us:6.2f}μs avg")

    print("\n" + "=" * 60)
    print(f"Detailed reports saved to: profiling_reports/mcts_search/")
    print("=" * 60)

    return metrics


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Profile MCTS search")
    parser.add_argument("--simulations", type=int, default=800, help="Number of simulations")
    parser.add_argument("--threads", type=int, default=4, help="Number of threads")
    args = parser.parse_args()

    try:
        profile_mcts_search(
            num_simulations=args.simulations,
            num_threads=args.threads
        )
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
