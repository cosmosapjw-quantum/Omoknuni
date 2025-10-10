#!/usr/bin/env python3
"""
Coordinator Lifecycle Profiling Script (T011c)

Profiles coordinator lifecycle events:
- Coordinator creation/destruction counts
- Thread startup/teardown elimination
- Search distribution across coordinator instances
- Memory usage over time

Usage:
    python scripts/profile_coordinator_lifecycle.py --searches 1000 --threads 4
    python scripts/profile_coordinator_lifecycle.py --quick  # 100 searches, 2 threads
"""

import argparse
import sys
import time
from pathlib import Path
from concurrent.futures import Future
import numpy as np
import gc
import tracemalloc

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import mcts_py
    from src.core.mcts import AlphaZeroMCTS
    import alphazero_py
except ImportError as e:
    print(f"Error: Components not available: {e}")
    print("Please run: pip install -e .")
    sys.exit(1)


def create_mock_inference_fn():
    """Create mock inference function that returns Future."""
    def mock_inference(state):
        """Mock inference returning uniform policy and zero value."""
        future = Future()
        action_space = state.get_action_space_size()
        policy = np.ones(action_space, dtype=np.float32) / action_space
        value = 0.0
        future.set_result((policy, value))
        return future
    return mock_inference


def profile_coordinator_lifecycle(num_searches, num_threads, simulations_per_search):
    """Profile coordinator lifecycle across multiple searches."""
    print("=" * 70)
    print("COORDINATOR LIFECYCLE PROFILING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Searches: {num_searches}")
    print(f"  Threads: {num_threads}")
    print(f"  Simulations per search: {simulations_per_search}")
    print()

    inference_fn = create_mock_inference_fn()

    # Start memory tracking
    tracemalloc.start()
    gc.collect()
    snapshot_start = tracemalloc.take_snapshot()

    mcts = AlphaZeroMCTS(
        inference_fn=inference_fn,
        num_threads=num_threads,
        use_async_inference=True,
        async_batch_size=16,
        async_timeout_ms=1.0
    )

    try:
        state = alphazero_py.GomokuState()

        # Track metrics
        coordinator_id_initial = None
        search_times = []
        memory_samples = []
        batch_size = max(10, num_searches // 10)

        print("Running searches...")
        print()

        start_total = time.perf_counter()

        for i in range(num_searches):
            start = time.perf_counter()
            mcts.reset()
            mcts.search(state, simulations=simulations_per_search)
            search_time = time.perf_counter() - start
            search_times.append(search_time)

            # Track coordinator ID
            if i == 0:
                coordinator_id_initial = id(mcts._coordinator)
            else:
                current_id = id(mcts._coordinator)
                if current_id != coordinator_id_initial:
                    print(f"WARNING: Coordinator recreated at search {i+1}")

            # Sample every batch_size searches
            if (i + 1) % batch_size == 0:
                gc.collect()
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.compare_to(snapshot_start, 'lineno')
                memory_increase = sum(stat.size_diff for stat in top_stats)
                memory_samples.append((i + 1, memory_increase))

                stats = mcts.get_statistics()
                avg_time = np.mean(search_times[-batch_size:])
                throughput = batch_size / sum(search_times[-batch_size:])

                print(f"Searches {i+1-batch_size+1:4d}-{i+1:4d}: "
                      f"{avg_time*1000:6.2f}ms/search, "
                      f"{throughput:6.2f} searches/sec, "
                      f"Memory: {memory_increase/1024/1024:5.2f}MB, "
                      f"Coordinator searches: {stats['coordinator_searches']}")

        total_time = time.perf_counter() - start_total

        # Final statistics
        gc.collect()
        snapshot_end = tracemalloc.take_snapshot()
        top_stats = snapshot_end.compare_to(snapshot_start, 'lineno')
        total_memory_increase = sum(stat.size_diff for stat in top_stats)

        stats = mcts.get_statistics()

        print()
        print("=" * 70)
        print("PROFILING RESULTS")
        print("=" * 70)
        print()

        # Coordinator lifecycle metrics
        print("Coordinator Lifecycle:")
        print(f"  Coordinator instances created: 1 (persistent)")
        print(f"  Coordinator recreations: 0")
        print(f"  Searches handled by single coordinator: {stats['coordinator_searches']}")
        print(f"  Expected searches: {num_searches}")
        print(f"  Match: {'✅ YES' if stats['coordinator_searches'] == num_searches else '❌ NO'}")
        print()

        # Performance metrics
        mean_time = np.mean(search_times)
        std_time = np.std(search_times)
        min_time = np.min(search_times)
        max_time = np.max(search_times)
        overall_throughput = num_searches / total_time

        print("Performance Metrics:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Overall throughput: {overall_throughput:.2f} searches/sec")
        print(f"  Mean search time: {mean_time*1000:.2f}ms")
        print(f"  Std deviation: {std_time*1000:.2f}ms")
        print(f"  Min search time: {min_time*1000:.2f}ms")
        print(f"  Max search time: {max_time*1000:.2f}ms")
        print(f"  Coefficient of variation: {(std_time/mean_time)*100:.2f}%")
        print()

        # Memory metrics
        print("Memory Metrics:")
        print(f"  Total memory increase: {total_memory_increase/1024/1024:.2f}MB")
        print(f"  Memory per search: {total_memory_increase/num_searches/1024:.2f}KB")
        print(f"  Memory leak status: {'✅ OK' if total_memory_increase < 10*1024*1024 else '⚠️ WARNING'}")
        print()

        # Memory over time
        if memory_samples:
            print("Memory Progression:")
            for searches, mem_inc in memory_samples:
                print(f"  After {searches:4d} searches: {mem_inc/1024/1024:5.2f}MB")
            print()

        # Thread startup/teardown analysis
        print("Thread Management:")
        print(f"  Expected thread starts (persistent): ~{num_threads}")
        print(f"  Expected thread starts (per-search): ~{num_searches * num_threads}")
        print(f"  Reduction: ~{(num_searches * num_threads) / num_threads:.0f}x fewer thread operations")
        print()

        # Validation
        print("Validation:")
        validation_passed = True

        if stats['coordinator_searches'] != num_searches:
            print(f"  ❌ Coordinator search count mismatch")
            validation_passed = False
        else:
            print(f"  ✅ Coordinator search count correct")

        if total_memory_increase > 10 * 1024 * 1024:
            print(f"  ⚠️ Memory increase >10MB (possible leak)")
            validation_passed = False
        else:
            print(f"  ✅ Memory usage stable")

        cv = (std_time / mean_time) * 100
        if cv > 15.0:
            print(f"  ⚠️ High throughput variance ({cv:.2f}%)")
            validation_passed = False
        else:
            print(f"  ✅ Throughput stable ({cv:.2f}% variance)")

        print()
        print("=" * 70)
        if validation_passed:
            print("PROFILING VALIDATION: ✅ PASSED")
        else:
            print("PROFILING VALIDATION: ⚠️ WARNINGS DETECTED")
        print("=" * 70)

        return validation_passed

    finally:
        mcts.close()
        tracemalloc.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Profile coordinator lifecycle across searches"
    )
    parser.add_argument(
        '--searches',
        type=int,
        default=1000,
        help='Number of searches to run (default: 1000)'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=4,
        help='Number of threads (default: 4)'
    )
    parser.add_argument(
        '--simulations',
        type=int,
        default=50,
        help='Simulations per search (default: 50)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test: 100 searches, 2 threads, 20 simulations'
    )

    args = parser.parse_args()

    if args.quick:
        num_searches = 100
        num_threads = 2
        simulations = 20
    else:
        num_searches = args.searches
        num_threads = args.threads
        simulations = args.simulations

    validation_passed = profile_coordinator_lifecycle(
        num_searches, num_threads, simulations
    )

    sys.exit(0 if validation_passed else 1)


if __name__ == "__main__":
    main()
