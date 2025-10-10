"""
Profiling Demo - Example usage of the profiling framework
==========================================================

Demonstrates how to use the profiling framework to analyze MCTS performance.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.profiling import ProfilingSession, ProfilerConfig


def simulate_mcts_workload(num_searches: int = 100):
    """Simulate MCTS workload for profiling."""
    print(f"Running {num_searches} simulated MCTS searches...")

    for i in range(num_searches):
        # Simulate search with some computation
        time.sleep(0.01)  # Simulate 10ms per search

        # Simulate some numpy operations (inference)
        batch = np.random.randn(32, 36, 15, 15).astype(np.float32)
        policy = np.random.randn(32, 225).astype(np.float32)
        value = np.random.randn(32, 1).astype(np.float32)

        # Simulate some GC pressure
        temp = [list(range(100)) for _ in range(10)]

        if i % 10 == 0:
            print(f"  Progress: {i}/{num_searches}")

    print("Workload complete!")


def demo_basic_profiling():
    """Basic profiling example with context manager."""
    print("=" * 60)
    print("Demo 1: Basic Profiling with Context Manager")
    print("=" * 60)

    config = ProfilerConfig(
        enable_gil_profiling=True,
        enable_inference_profiling=True,
        enable_thread_profiling=True,
        enable_memory_profiling=True,
        auto_save_reports=True,
        report_directory="profiling_reports/demo1"
    )

    with ProfilingSession(config) as session:
        simulate_mcts_workload(num_searches=50)

    # Metrics are automatically collected and saved
    print("\nProfiling complete! Reports saved to profiling_reports/demo1/")


def demo_manual_profiling():
    """Manual profiling with explicit start/stop."""
    print("\n" + "=" * 60)
    print("Demo 2: Manual Profiling with Explicit Control")
    print("=" * 60)

    config = ProfilerConfig(
        enable_gil_profiling=True,
        enable_memory_profiling=True,
        memory_snapshot_interval=0.5,  # Faster snapshots
        auto_save_reports=False  # Manual save
    )

    session = ProfilingSession(config)
    session.start()

    try:
        simulate_mcts_workload(num_searches=30)
    finally:
        session.stop()

    # Get metrics
    metrics = session.get_all_metrics()
    print("\n--- Metrics Summary ---")
    print(f"GIL Efficiency: {metrics.get('gil_metrics', {}).get('summary', {}).get('gil_efficiency', 0.0):.1f}%")
    print(f"Memory Growth: {metrics.get('memory_metrics', {}).get('summary', {}).get('memory_growth_mb', 0.0):.1f} MB")

    # Save reports manually
    reports = session.save_reports("profiling_reports/demo2")
    print(f"\nReports saved:")
    for report_type, path in reports.items():
        if path:
            print(f"  {report_type}: {path}")


def demo_targeted_profiling():
    """Targeted profiling of specific code sections."""
    print("\n" + "=" * 60)
    print("Demo 3: Targeted Section Profiling")
    print("=" * 60)

    config = ProfilerConfig(
        enable_inference_profiling=True,
        enable_memory_profiling=True,
        auto_save_reports=False
    )

    with ProfilingSession(config) as session:
        # Profile specific sections
        for i in range(10):
            # Simulate batch collection
            if session.inference:
                with session.inference.track_request(f"req_{i}", thread_id=1, batch_size=32):
                    with session.inference.track_stage(f"req_{i}", "queue_wait"):
                        time.sleep(0.001)  # 1ms wait

                    with session.inference.track_stage(f"req_{i}", "inference"):
                        # Simulate inference
                        batch = np.random.randn(32, 36, 15, 15).astype(np.float32)
                        time.sleep(0.005)  # 5ms inference

            # Track memory for a section
            if session.memory:
                with session.memory.track_section("allocation_test"):
                    # Allocate some memory
                    data = [np.random.randn(1000, 1000) for _ in range(5)]
                    time.sleep(0.01)

    # Analyze results
    metrics = session.get_all_metrics()
    inference_metrics = metrics.get('inference_metrics', {}).get('summary', {})
    print(f"\nInference Metrics:")
    print(f"  Avg Latency: {inference_metrics.get('avg_latency_us', 0.0)/1000:.2f} ms")
    print(f"  P99 Latency: {inference_metrics.get('p99_latency_us', 0.0)/1000:.2f} ms")

    memory_metrics = metrics.get('memory_metrics', {}).get('section_analysis', {})
    print(f"\nMemory Section Analysis:")
    for section_name, stats in memory_metrics.items():
        print(f"  {section_name}:")
        print(f"    Invocations: {stats['num_invocations']}")
        print(f"    Avg Delta: {stats['avg_memory_delta_mb']:.2f} MB")


def demo_gil_analysis():
    """Detailed GIL analysis."""
    print("\n" + "=" * 60)
    print("Demo 4: Detailed GIL Analysis")
    print("=" * 60)

    config = ProfilerConfig(
        enable_gil_profiling=True,
        gil_sample_rate=0.0001,  # Very fast sampling (0.1ms)
        gil_track_hotspots=True,
        auto_save_reports=False
    )

    with ProfilingSession(config) as session:
        gil = session.gil

        # Simulate GIL-heavy workload
        for i in range(20):
            # Python code (with GIL)
            if gil:
                gil.mark_gil_release(f"loop_iteration_{i}")

            # Simulate nogil work (C++ computation)
            time.sleep(0.002)

            if gil:
                gil.mark_gil_acquire(f"loop_iteration_{i}")

            # More Python work
            result = sum(range(10000))

    # Analyze GIL metrics
    metrics = session.get_all_metrics()
    gil_metrics = metrics.get('gil_metrics', {})

    print("\n--- GIL Analysis ---")
    summary = gil_metrics.get('summary', {})
    print(f"GIL Utilization: {summary.get('gil_utilization', 0.0):.1f}%")
    print(f"GIL Efficiency: {summary.get('gil_efficiency', 0.0):.1f}%")
    print(f"Contention Events: {summary.get('total_contention_events', 0)}")

    print("\nTop GIL Wait Hotspots:")
    for hotspot in gil_metrics.get('top_wait_hotspots', [])[:5]:
        print(f"  {hotspot['location']}: {hotspot['total_wait_time_ms']:.2f} ms")


def main():
    """Run all profiling demos."""
    print("MCTS Profiling Framework Demo")
    print("=" * 60)

    try:
        demo_basic_profiling()
        demo_manual_profiling()
        demo_targeted_profiling()
        demo_gil_analysis()

        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)
        print("\nCheck the profiling_reports/ directory for detailed reports.")

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
