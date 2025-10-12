#!/usr/bin/env python3
"""Profile DLPack tensor creation overhead."""

import time
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def profile_tensor_creation(batch_size: int, iterations: int, detailed: bool = False):
    """Profile batch tensor creation."""

    try:
        import mcts_py
        from src.games.gomoku import GomokuState
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure MCTS extension is built: python -m pip install -e .")
        return None

    # Create states
    states = [GomokuState() for _ in range(batch_size)]

    timings = []
    for i in range(iterations):
        start = time.perf_counter()

        try:
            dlpack_capsule = mcts_py.create_batch_tensor_from_states(states)
        except Exception as e:
            print(f"❌ Error creating tensor: {e}")
            return None

        elapsed = (time.perf_counter() - start) * 1000  # ms
        timings.append(elapsed)

        if detailed and i % 100 == 0:
            print(f"  Iteration {i+1}/{iterations}: {elapsed:.2f}ms")

    mean_time = np.mean(timings)
    std_time = np.std(timings)

    print(f"\nBatch Tensor Creation Profile:")
    print(f"  Configuration: batch={batch_size}, iterations={iterations}")
    print(f"  Mean: {mean_time:.2f} ms")
    print(f"  Stddev: {std_time:.2f} ms ({std_time/mean_time*100:.1f}%)")
    print(f"  Min: {np.min(timings):.2f} ms")
    print(f"  Max: {np.max(timings):.2f} ms")
    print(f"  p50: {np.percentile(timings, 50):.2f} ms")
    print(f"  p95: {np.percentile(timings, 95):.2f} ms")
    print(f"  p99: {np.percentile(timings, 99):.2f} ms")
    print(f"\n  Target: <1.0 ms per batch")

    if mean_time > 1.0:
        overhead = mean_time - 1.0
        print(f"\n  ❌ FAIL: {mean_time:.2f}ms > 1.0ms target")
        print(f"  Overhead: {overhead:.2f}ms needs investigation")
        print(f"  Potential causes:")
        print(f"    - GIL acquisition overhead")
        print(f"    - Feature extraction not parallelized (OpenMP)")
        print(f"    - Pinned buffer allocation (should be cached)")
        print(f"    - DLPack capsule creation overhead")
        print(f"\n  Recommended profiling:")
        print(f"    perf record -g python {__file__} --batch-size {batch_size} --iterations 1000")
        print(f"    perf script | python scripts/flamegraph.pl > tensor_creation.svg")
    else:
        print(f"\n  ✅ PASS: {mean_time:.2f}ms < 1.0ms target")

    # Estimate impact on throughput
    batches_per_sec = 1000 / mean_time
    overhead_per_sec = (mean_time - 1.0) * batches_per_sec if mean_time > 1.0 else 0
    print(f"\n  Impact Analysis:")
    print(f"    Batches/sec (approx): {batches_per_sec:.0f}")
    if overhead_per_sec > 0:
        print(f"    Wasted time/sec: {overhead_per_sec:.0f}ms")
        print(f"    Potential speedup if fixed: {mean_time / 1.0:.2f}×")

    return mean_time

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Profile DLPack tensor creation overhead")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size to test (default: 64)")
    parser.add_argument("--iterations", type=int, default=1000,
                       help="Number of iterations (default: 1000)")
    parser.add_argument("--detailed", action="store_true",
                       help="Print detailed timing for each iteration")
    args = parser.parse_args()

    print("=" * 80)
    print("DLPack Tensor Creation Profiler")
    print("=" * 80)

    result = profile_tensor_creation(args.batch_size, args.iterations, args.detailed)

    if result is None:
        sys.exit(1)
    elif result > 1.0:
        sys.exit(1)  # Fail if above target
    else:
        sys.exit(0)  # Pass
