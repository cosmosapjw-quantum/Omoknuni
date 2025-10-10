#!/usr/bin/env python3
"""
Profile MCTS overhead to identify bottlenecks.

Uses cProfile to find hot spots in the MCTS execution.
"""

import sys
import os
import time
import cProfile
import pstats
import io
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.mcts import AlphaZeroMCTS
from src.core.dlpack_inference_bridge import DLPackInferenceBridge
from src.neural.model import create_model_for_game
import alphazero_py

def run_mcts_search():
    """Run a single MCTS search for profiling."""
    # Create model
    device = torch.device('cuda')
    model = create_model_for_game('gomoku').to(device)
    model.eval()

    # Create inference bridge
    inference_bridge = DLPackInferenceBridge(
        model=model,
        device=device,
        use_mixed_precision=True
    )

    # Create MCTS with baseline configuration
    mcts = AlphaZeroMCTS(
        inference_fn=inference_bridge,
        num_threads=4,
        use_async_inference=True,
        async_batch_size=64,
        async_timeout_ms=1.0
    )

    # Create initial state
    state = alphazero_py.GomokuState()

    # Warmup
    print("Warming up...")
    mcts.search(state, simulations=100)
    mcts.reset()

    # Profile the actual search
    print("Profiling 1000 simulations...")
    start = time.perf_counter()
    mcts.search(state, simulations=1000)
    elapsed = time.perf_counter() - start

    throughput = 1000 / elapsed
    print(f"\nThroughput: {throughput:.0f} sims/sec ({elapsed:.3f}s)")

    # Get statistics
    stats = mcts.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total searches: {stats.get('total_searches', 0)}")
    print(f"  Total backups: {stats.get('total_backups', 0)}")

    mcts.close()

    return throughput

def main():
    """Profile MCTS execution."""
    print("="*70)
    print("MCTS Overhead Profiling")
    print("="*70)

    # Run with profiler
    profiler = cProfile.Profile()
    profiler.enable()

    throughput = run_mcts_search()

    profiler.disable()

    # Print profiling results
    print("\n" + "="*70)
    print("PROFILING RESULTS")
    print("="*70)

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions

    print(s.getvalue())

    # Also print by total time
    print("\n" + "="*70)
    print("TOP FUNCTIONS BY TOTAL TIME")
    print("="*70)

    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    ps.print_stats(20)  # Top 20 functions

    print(s.getvalue())

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Throughput: {throughput:.0f} sims/sec")
    print(f"Target: 3,831 sims/sec")
    print(f"Gap: {3831/throughput:.2f}× slower")

if __name__ == '__main__':
    main()
