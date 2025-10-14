#!/usr/bin/env python3
"""
MCTS Phase Profiling Script (Phase 1B)
=======================================

Profiles MCTS execution using existing C++ instrumentation to identify
bottlenecks in selection, expansion, and backup phases.

Usage:
    python scripts/profile_mcts_phases.py --simulations 800 --threads 8

Output:
    - Time breakdown by phase (selection, expansion, backup)
    - Thread efficiency metrics
    - Contention analysis (virtual loss, expansion conflicts)
    - Chrome Trace export for visualization
"""

import sys
import time
import argparse
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import mcts_py
import alphazero_py
from src.core.mcts import AlphaZeroMCTS
from src.neural.inference_worker import GPUInferenceWorker


def profile_mcts_phases(game_name='gomoku', simulations=800, num_threads=8):
    """Profile MCTS execution with detailed phase breakdown."""
    print("=" * 70)
    print("MCTS PHASE PROFILING (Phase 1B)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Game:        {game_name}")
    print(f"  Simulations: {simulations}")
    print(f"  Threads:     {num_threads}")
    print()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create initial state
    if game_name == 'gomoku':
        initial_state = alphazero_py.GomokuState()
    elif game_name == 'chess':
        initial_state = alphazero_py.ChessState()
    elif game_name == 'go':
        initial_state = alphazero_py.GoState()
    else:
        raise ValueError(f"Unknown game: {game_name}")

    # Create GPU inference worker
    print("\nCreating GPUInferenceWorker...")
    worker = GPUInferenceWorker(
        model_path=None,
        device=device.type,
        batch_size=64,
        timeout_ms=3.0,
        use_mixed_precision=False
    )

    # Enable instrumentation
    print("Enabling C++ instrumentation...")
    mcts_py.set_instrumentation_enabled(True)
    mcts_py.reset_instrumentation_metrics()

    # Create MCTS engine
    print("Creating MCTS engine...")
    mcts = AlphaZeroMCTS(
        inference_fn=worker,
        num_threads=num_threads,
        use_async_inference=True,
        async_batch_size=16,
        async_timeout_ms=10.0,
        enable_instrumentation=True
    )

    try:
        # Warmup run (small, not counted)
        print("\nWarming up...")
        mcts.search(initial_state, simulations=100)
        mcts.reset()
        mcts_py.reset_instrumentation_metrics()

        # Profiling run
        print(f"\nProfiling search with {simulations} simulations...")
        start_time = time.perf_counter()
        mcts.search(initial_state, simulations=simulations)
        elapsed_time = time.perf_counter() - start_time

        # Get statistics
        mcts_stats = mcts.get_statistics()
        worker_metrics = worker.get_metrics()
        instrumentation = mcts_py.get_instrumentation_snapshot()

        # Calculate metrics
        total_simulations = mcts_stats['simulations_completed']
        throughput = total_simulations / elapsed_time

        # Analyze phase timing
        selection_data = instrumentation.get('Selection', {})
        expansion_data = instrumentation.get('Expansion', {})
        backup_data = instrumentation.get('Backup', {})
        vl_apply_data = instrumentation.get('VirtualLossApply', {})
        vl_remove_data = instrumentation.get('VirtualLossRemove', {})

        # Calculate percentages
        total_tracked_ns = (
            selection_data.get('total_ns', 0) +
            expansion_data.get('total_ns', 0) +
            backup_data.get('total_ns', 0)
        )

        selection_pct = (selection_data.get('total_ns', 0) / total_tracked_ns * 100) if total_tracked_ns > 0 else 0
        expansion_pct = (expansion_data.get('total_ns', 0) / total_tracked_ns * 100) if total_tracked_ns > 0 else 0
        backup_pct = (backup_data.get('total_ns', 0) / total_tracked_ns * 100) if total_tracked_ns > 0 else 0

        # Contention metrics
        expansion_conflicts = instrumentation.get('ExpansionConflict', {}).get('calls', 0)
        busy_edges_masked = instrumentation.get('BusyEdgeMasked', {}).get('calls', 0)
        selection_retries = instrumentation.get('SelectionRetry', {}).get('calls', 0)

        # Calculate thread efficiency
        # Thread efficiency = actual_throughput / (ideal_throughput * num_threads)
        # Assuming single-thread throughput ~1,200 sims/sec (from benchmarks)
        single_thread_throughput = 1200
        ideal_parallel_throughput = single_thread_throughput * num_threads
        thread_efficiency = (throughput / ideal_parallel_throughput) * 100 if ideal_parallel_throughput > 0 else 0

        # Results
        print("\n" + "=" * 70)
        print("PROFILING RESULTS")
        print("=" * 70)

        print("\nPerformance Metrics:")
        print(f"  Total time:           {elapsed_time:.3f}s")
        print(f"  Throughput:           {throughput:.1f} sims/sec")
        print(f"  Total simulations:    {total_simulations}")
        print(f"  GPU utilization:      {worker_metrics['avg_gpu_utilization']*100:.1f}%")
        print(f"  Avg batch size:       {worker_metrics['average_batch_size']:.1f}")
        print(f"  Thread efficiency:    {thread_efficiency:.1f}%")

        print("\nPhase Timing Breakdown:")
        print(f"  Selection:            {selection_data.get('avg_ns', 0)/1000:.1f} μs/call  "
              f"({selection_pct:.1f}% of total)")
        print(f"    Calls:              {selection_data.get('calls', 0)}")
        print(f"    Total time:         {selection_data.get('total_ns', 0)/1e9:.3f}s")

        print(f"\n  Expansion:            {expansion_data.get('avg_ns', 0)/1000:.1f} μs/call  "
              f"({expansion_pct:.1f}% of total)")
        print(f"    Calls:              {expansion_data.get('calls', 0)}")
        print(f"    Total time:         {expansion_data.get('total_ns', 0)/1e9:.3f}s")

        print(f"\n  Backup:               {backup_data.get('avg_ns', 0)/1000:.1f} μs/call  "
              f"({backup_pct:.1f}% of total)")
        print(f"    Calls:              {backup_data.get('calls', 0)}")
        print(f"    Total time:         {backup_data.get('total_ns', 0)/1e9:.3f}s")

        print("\nVirtual Loss Tracking:")
        print(f"  Apply:                {vl_apply_data.get('avg_ns', 0)/1000:.1f} μs/call")
        print(f"    Calls:              {vl_apply_data.get('calls', 0)}")
        print(f"  Remove:               {vl_remove_data.get('avg_ns', 0)/1000:.1f} μs/call")
        print(f"    Calls:              {vl_remove_data.get('calls', 0)}")

        print("\nContention Analysis:")
        print(f"  Expansion conflicts:  {expansion_conflicts}")
        print(f"    Rate:               {expansion_conflicts/total_simulations*100:.1f}% of simulations")
        print(f"  Busy edges masked:    {busy_edges_masked}")
        print(f"    Rate:               {busy_edges_masked/total_simulations*100:.1f}% of simulations")
        print(f"  Selection retries:    {selection_retries}")
        print(f"    Rate:               {selection_retries/total_simulations*100:.1f}% of simulations")

        # Identify bottleneck
        print("\n" + "=" * 70)
        print("BOTTLENECK IDENTIFICATION")
        print("=" * 70)

        # Phase bottleneck
        if selection_pct > 50:
            bottleneck_phase = "Selection"
        elif expansion_pct > 50:
            bottleneck_phase = "Expansion"
        elif backup_pct > 50:
            bottleneck_phase = "Backup"
        else:
            bottleneck_phase = "Balanced (no single dominant phase)"

        print(f"\nPhase Bottleneck: {bottleneck_phase}")

        # Contention bottleneck
        contention_rate = (expansion_conflicts + busy_edges_masked + selection_retries) / total_simulations * 100
        if contention_rate > 20:
            print(f"🔴 HIGH CONTENTION DETECTED ({contention_rate:.1f}% conflict rate)")
            print("\nThread coordination overhead is significant!")
            print("Recommendations:")
            print("  1. Reduce virtual loss magnitude")
            print("  2. Increase tree depth to spread threads")
            print("  3. Consider reducing thread count")
        elif contention_rate > 10:
            print(f"⚠️  MODERATE CONTENTION ({contention_rate:.1f}% conflict rate)")
            print("\nSome thread coordination overhead, but not severe.")
        else:
            print(f"✅ LOW CONTENTION ({contention_rate:.1f}% conflict rate)")
            print("\nThread coordination is working well.")

        # Thread efficiency bottleneck
        if thread_efficiency < 30:
            print(f"\n🔴 VERY LOW THREAD EFFICIENCY ({thread_efficiency:.1f}%)")
            print("\nThreads are spending most time idle or blocked!")
            print("Likely causes:")
            print(f"  1. {bottleneck_phase} phase is too slow")
            print("  2. Waiting for GPU inference results")
            print("  3. Insufficient parallelism in tree structure")
        elif thread_efficiency < 50:
            print(f"\n⚠️  LOW THREAD EFFICIENCY ({thread_efficiency:.1f}%)")
            print("\nThreads have significant idle time.")
        else:
            print(f"\n✅ REASONABLE THREAD EFFICIENCY ({thread_efficiency:.1f}%)")

        # GPU utilization bottleneck
        gpu_util = worker_metrics['avg_gpu_utilization'] * 100
        if gpu_util < 40:
            print(f"\n🔴 LOW GPU UTILIZATION ({gpu_util:.1f}%)")
            print("\nGPU is idle most of the time - threads not generating enough work!")
        elif gpu_util < 65:
            print(f"\n⚠️  MODERATE GPU UTILIZATION ({gpu_util:.1f}%)")
        else:
            print(f"\n✅ GOOD GPU UTILIZATION ({gpu_util:.1f}%)")

        # Comprehensive analysis
        print("\n" + "=" * 70)
        print("COMPREHENSIVE ANALYSIS")
        print("=" * 70)

        print(f"\nPrimary Bottleneck: ", end="")
        if selection_pct > 50:
            print("SELECTION PHASE (tree traversal overhead)")
            print("Recommendations:")
            print("  - Profile PUCT computation")
            print("  - Check state cloning overhead (implement state pooling)")
            print("  - Verify AVX2 vectorization is active")
        elif expansion_pct > 50:
            print("EXPANSION PHASE (inference wait time)")
            print("Recommendations:")
            print("  - Check GPU inference speed")
            print("  - Reduce batch timeout (faster submission)")
            print("  - Increase batch size (better GPU utilization)")
        elif backup_pct > 50:
            print("BACKUP PHASE (value propagation overhead)")
            print("Recommendations:")
            print("  - Check atomic operation contention")
            print("  - Profile value update performance")
            print("  - Verify path traversal efficiency")
        else:
            print("NO SINGLE DOMINANT PHASE")
            print(f"Balanced load: Selection {selection_pct:.1f}%, Expansion {expansion_pct:.1f}%, Backup {backup_pct:.1f}%")
            print("\nThis suggests overall system overhead rather than phase-specific bottleneck.")
            print("Likely causes:")
            print(f"  1. Thread coordination overhead ({contention_rate:.1f}% conflicts)")
            print(f"  2. Low thread efficiency ({thread_efficiency:.1f}%)")
            print(f"  3. GPU underutilization ({gpu_util:.1f}%)")

        print("\n" + "=" * 70)

        # Export detailed results
        results = {
            'performance': {
                'throughput': throughput,
                'thread_efficiency': thread_efficiency,
                'gpu_utilization': gpu_util,
                'elapsed_time': elapsed_time,
            },
            'phases': {
                'selection': {
                    'avg_us': selection_data.get('avg_ns', 0) / 1000,
                    'total_s': selection_data.get('total_ns', 0) / 1e9,
                    'calls': selection_data.get('calls', 0),
                    'percentage': selection_pct,
                },
                'expansion': {
                    'avg_us': expansion_data.get('avg_ns', 0) / 1000,
                    'total_s': expansion_data.get('total_ns', 0) / 1e9,
                    'calls': expansion_data.get('calls', 0),
                    'percentage': expansion_pct,
                },
                'backup': {
                    'avg_us': backup_data.get('avg_ns', 0) / 1000,
                    'total_s': backup_data.get('total_ns', 0) / 1e9,
                    'calls': backup_data.get('calls', 0),
                    'percentage': backup_pct,
                },
            },
            'contention': {
                'expansion_conflicts': expansion_conflicts,
                'busy_edges_masked': busy_edges_masked,
                'selection_retries': selection_retries,
                'total_rate': contention_rate,
            },
            'bottleneck': bottleneck_phase,
            'full_instrumentation': instrumentation,
        }

        return results

    finally:
        mcts.close()
        worker.stop_worker()


def main():
    parser = argparse.ArgumentParser(
        description='Profile MCTS phases for bottleneck identification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--game', type=str, default='gomoku',
                       choices=['gomoku', 'chess', 'go'],
                       help='Game type (default: gomoku)')
    parser.add_argument('--simulations', type=int, default=800,
                       help='Number of simulations (default: 800)')
    parser.add_argument('--threads', type=int, default=8,
                       help='Number of threads (default: 8)')

    args = parser.parse_args()

    # Run profiling
    results = profile_mcts_phases(
        game_name=args.game,
        simulations=args.simulations,
        num_threads=args.threads
    )

    # Save results
    output_file = Path(__file__).parent.parent / "mcts_phase_profiling_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == '__main__':
    main()
