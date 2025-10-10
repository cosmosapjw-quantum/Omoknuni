#!/usr/bin/env python3
"""
Comprehensive MCTS Performance Profiler
========================================

A powerful profiling tool that performs deep analysis of the MCTS system,
including C++ backend, Python coordination, and GPU inference.

This tool replaces scattered legacy performance scripts with a unified,
comprehensive profiling system.
"""

import argparse
import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from datetime import datetime
from collections import defaultdict
import cProfile
import pstats
from io import StringIO
import tracemalloc
import gc

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Core imports
from src.core.mcts import AlphaZeroMCTS
from src.games.game_state import IGameState
from src.neural.inference_worker import GPUInferenceWorker
from src.telemetry.gpu_profiler import GPUProfiler
from src.profiling.gil_profiler import GILProfiler

# C++ profiling integration
try:
    import mcts_py
    CPP_PROFILING_AVAILABLE = hasattr(mcts_py, 'set_instrumentation_enabled')
except ImportError:
    CPP_PROFILING_AVAILABLE = False
    print("Warning: C++ profiling not available")

# GPU monitoring
try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: NVIDIA GPU monitoring not available")


class ComprehensiveMCTSProfiler:
    """
    Comprehensive profiler for MCTS system performance analysis.

    Performs deep analysis of:
    - C++ MCTS operations (selection, expansion, backup)
    - Python coordination overhead
    - GPU inference performance
    - Thread synchronization and contention
    - Memory usage patterns
    - Cache efficiency
    - Pipeline bottlenecks
    """

    def __init__(self,
                 game_type: str = "gomoku",
                 threads: int = 4,
                 simulations: int = 800,
                 batch_size: int = 32,
                 timeout_ms: float = 2.0,
                 output_dir: str = "profiling_reports"):
        """
        Initialize comprehensive profiler.

        Args:
            game_type: Game to profile (gomoku/chess/go)
            threads: Number of MCTS threads
            simulations: Simulations per search
            batch_size: GPU batch size
            timeout_ms: Batch collection timeout
            output_dir: Directory for profiling reports
        """
        self.game_type = game_type
        self.threads = threads
        self.simulations = simulations
        self.batch_size = batch_size
        self.timeout_ms = timeout_ms
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Create timestamp for this profiling session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)

        # Profiling components
        self.gil_profiler = GILProfiler(sample_rate=0.001)  # 1ms sampling
        self.gpu_profiler = None
        self.cpp_metrics = {}
        self.python_profile = None
        self.memory_snapshots = []

        # Results storage
        self.results = {
            'session_id': self.session_id,
            'configuration': {
                'game_type': game_type,
                'threads': threads,
                'simulations': simulations,
                'batch_size': batch_size,
                'timeout_ms': timeout_ms
            },
            'metrics': {}
        }

        print(f"Comprehensive MCTS Profiler initialized")
        print(f"Session ID: {self.session_id}")
        print(f"Output directory: {self.session_dir}")

    def setup_game_and_mcts(self) -> Tuple[IGameState, AlphaZeroMCTS]:
        """Setup game state and MCTS engine."""
        print("\n=== Setting up game and MCTS engine ===")

        # Create initial game state
        from src.games.game_state import create_game_state
        root_state = create_game_state(self.game_type)

        # Create GPU inference worker
        print(f"Creating GPU inference worker (batch_size={self.batch_size})...")
        gpu_worker = GPUInferenceWorker(
            model_path=None,  # Will use default weights
            device='cuda:0' if NVML_AVAILABLE else 'cpu',
            batch_size=self.batch_size,
            timeout_ms=self.timeout_ms,
            use_mixed_precision=True
        )

        # Initialize GPU profiler if available
        if NVML_AVAILABLE:
            self.gpu_profiler = GPUProfiler(device='cuda:0')

        # Warmup GPU
        input_shape = root_state.get_enhanced_tensor_representation().shape
        gpu_worker.warmup(input_shape)

        # Create MCTS engine
        print(f"Creating MCTS engine (threads={self.threads})...")
        mcts = AlphaZeroMCTS(
            inference_fn=gpu_worker,
            num_threads=self.threads,
            use_async_inference=True,
            async_batch_size=self.batch_size,
            async_timeout_ms=self.timeout_ms,
            enable_instrumentation=CPP_PROFILING_AVAILABLE
        )

        return root_state, mcts

    def profile_cpp_backend(self, mcts: AlphaZeroMCTS, root_state: IGameState):
        """Profile C++ MCTS backend operations."""
        print("\n=== Profiling C++ Backend ===")

        if CPP_PROFILING_AVAILABLE:
            # Enable C++ instrumentation
            mcts_py.set_instrumentation_enabled(True)
            mcts_py.reset_instrumentation_metrics()

        # Run searches
        print(f"Running {self.simulations} simulations...")
        start_time = time.perf_counter()

        mcts.search(root_state, self.simulations)

        elapsed = time.perf_counter() - start_time

        # Collect C++ metrics
        if CPP_PROFILING_AVAILABLE:
            cpp_snapshot = mcts_py.get_instrumentation_snapshot()
            self.cpp_metrics = self._parse_cpp_metrics(cpp_snapshot)

            print(f"C++ profiling complete:")
            print(f"  Total time: {elapsed:.3f}s")
            print(f"  Simulations/sec: {self.simulations/elapsed:.1f}")

            # Show top operations
            if 'timers' in self.cpp_metrics:
                print("\n  Top C++ operations:")
                sorted_ops = sorted(
                    self.cpp_metrics['timers'].items(),
                    key=lambda x: x[1]['total_ns'],
                    reverse=True
                )[:10]

                for op, stats in sorted_ops:
                    avg_ns = stats['total_ns'] / max(1, stats['count'])
                    pct = 100.0 * stats['total_ns'] / (elapsed * 1e9)
                    print(f"    {op}: {stats['count']} calls, "
                          f"{avg_ns/1000:.1f}μs avg, {pct:.1f}% total")

    def profile_python_coordination(self, mcts: AlphaZeroMCTS, root_state: IGameState):
        """Profile Python coordination layer."""
        print("\n=== Profiling Python Coordination ===")

        # Start GIL profiling
        self.gil_profiler.start()

        # Start cProfile
        profiler = cProfile.Profile()
        profiler.enable()

        # Run search
        print(f"Running search with Python profiling...")
        start_time = time.perf_counter()

        mcts.search(root_state, self.simulations // 4)  # Shorter run for Python profiling

        elapsed = time.perf_counter() - start_time

        # Stop profiling
        profiler.disable()
        self.gil_profiler.stop()

        # Analyze Python profile
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        self.python_profile = s.getvalue()

        # Get GIL metrics
        gil_metrics = self.gil_profiler.get_metrics()

        print(f"Python profiling complete:")
        print(f"  Total time: {elapsed:.3f}s")

        if 'summary' in gil_metrics:
            summary = gil_metrics['summary']
            print(f"  GIL efficiency: {summary.get('gil_efficiency', 0):.1f}%")
            print(f"  Threads with GIL contention: {summary.get('num_threads', 0)}")

        # Store results
        self.results['metrics']['python'] = {
            'elapsed_time': elapsed,
            'gil_metrics': gil_metrics,
            'top_functions': self._extract_top_functions(ps)
        }

    def profile_gpu_inference(self, gpu_worker: GPUInferenceWorker):
        """Profile GPU inference performance."""
        print("\n=== Profiling GPU Inference ===")

        if not self.gpu_profiler:
            print("  GPU profiling not available")
            return

        with self.gpu_profiler:
            # Create batch of positions
            print(f"  Creating batch of {self.batch_size} positions...")
            if self.game_type == "gomoku":
                shape = (36, 15, 15)
            elif self.game_type == "chess":
                shape = (30, 8, 8)
            else:  # go
                shape = (25, 19, 19)

            positions = [np.random.randn(*shape).astype(np.float32)
                        for _ in range(self.batch_size)]

            # Profile inference
            print(f"  Running {100} inference batches...")
            inference_times = []

            for i in range(100):
                start = time.perf_counter()

                with self.gpu_profiler.profile_batch(len(positions)):
                    policies, values = gpu_worker.batch_inference(positions)

                inference_times.append(time.perf_counter() - start)

                if i % 20 == 0:
                    print(f"    Batch {i}/100 complete")

        # Get GPU metrics
        from dataclasses import asdict
        gpu_report = self.gpu_profiler.generate_report()
        gpu_metrics = asdict(gpu_report) if gpu_report else {}

        print(f"\nGPU profiling complete:")
        print(f"  Avg inference time: {np.mean(inference_times)*1000:.2f}ms")
        print(f"  Throughput: {self.batch_size/np.mean(inference_times):.1f} samples/sec")

        if gpu_metrics:
            print(f"  GPU utilization: {gpu_metrics.get('avg_gpu_utilization', 0):.1f}%")
            print(f"  Memory used: {gpu_metrics.get('avg_memory_used_mb', 0):.2f}MB")

        self.results['metrics']['gpu'] = {
            'inference_times_ms': [t*1000 for t in inference_times],
            'throughput_samples_per_sec': self.batch_size/np.mean(inference_times),
            'gpu_metrics': gpu_metrics
        }

    def profile_memory_usage(self, mcts: AlphaZeroMCTS, root_state: IGameState):
        """Profile memory usage patterns."""
        print("\n=== Profiling Memory Usage ===")

        # Start memory tracking
        tracemalloc.start()
        gc.collect()

        initial_snapshot = tracemalloc.take_snapshot()

        # Run searches
        print(f"Running searches while tracking memory...")
        memory_samples = []

        for i in range(10):
            mcts.search(root_state, self.simulations // 10)

            # Take memory snapshot
            current_snapshot = tracemalloc.take_snapshot()
            stats = current_snapshot.compare_to(initial_snapshot, 'lineno')

            total_memory = sum(stat.size for stat in stats)
            memory_samples.append(total_memory / 1024**2)  # Convert to MB

            print(f"  Search {i+1}/10: {memory_samples[-1]:.1f}MB allocated")

        # Get top memory consumers
        top_stats = current_snapshot.statistics('lineno')[:10]

        tracemalloc.stop()

        print(f"\nMemory profiling complete:")
        print(f"  Peak memory: {max(memory_samples):.1f}MB")
        print(f"  Avg memory: {np.mean(memory_samples):.1f}MB")

        print("\n  Top memory allocations:")
        for stat in top_stats[:5]:
            print(f"    {stat.traceback}: {stat.size/1024**2:.1f}MB")

        self.results['metrics']['memory'] = {
            'samples_mb': memory_samples,
            'peak_mb': max(memory_samples),
            'avg_mb': np.mean(memory_samples)
        }

    def profile_thread_contention(self, mcts: AlphaZeroMCTS, root_state: IGameState):
        """Profile thread contention and synchronization."""
        print("\n=== Profiling Thread Contention ===")

        # Run searches with different thread counts
        thread_counts = [1, 2, 4, 8, min(12, os.cpu_count() or 12)]
        results = []

        # Get inference worker from original MCTS
        gpu_worker = mcts.inference_fn

        for num_threads in thread_counts:
            if num_threads > (os.cpu_count() or 12):
                continue

            print(f"  Testing with {num_threads} threads...")

            # Create new MCTS instance with specific thread count
            test_mcts = AlphaZeroMCTS(
                inference_fn=gpu_worker,
                num_threads=num_threads,
                use_async_inference=mcts.use_async_inference,
                async_batch_size=mcts.async_batch_size,
                async_timeout_ms=mcts.async_timeout_ms,
                enable_instrumentation=False  # Disable for performance testing
            )

            # Measure throughput
            start = time.perf_counter()
            test_mcts.search(root_state, self.simulations)
            elapsed = time.perf_counter() - start

            throughput = self.simulations / elapsed
            results.append({
                'threads': num_threads,
                'elapsed': elapsed,
                'throughput': throughput,
                'efficiency': throughput / (throughput if num_threads == 1 else results[0]['throughput'])
            })

            print(f"    Throughput: {throughput:.1f} sims/sec")
            print(f"    Parallel efficiency: {results[-1]['efficiency']*100:.1f}%")

            # Cleanup
            test_mcts.close()

        self.results['metrics']['thread_scaling'] = results

    def analyze_bottlenecks(self):
        """Analyze and identify performance bottlenecks."""
        print("\n=== Bottleneck Analysis ===")

        bottlenecks = []

        # Analyze C++ bottlenecks
        if self.cpp_metrics and 'timers' in self.cpp_metrics:
            total_cpp_time = sum(m['total_ns'] for m in self.cpp_metrics['timers'].values())

            for op, stats in self.cpp_metrics['timers'].items():
                pct = 100.0 * stats['total_ns'] / total_cpp_time
                if pct > 10:  # Operations taking >10% of time
                    bottlenecks.append({
                        'category': 'C++',
                        'operation': op,
                        'percentage': pct,
                        'severity': 'high' if pct > 30 else 'medium'
                    })

        # Analyze Python/GIL bottlenecks
        if 'python' in self.results['metrics']:
            gil_metrics = self.results['metrics']['python'].get('gil_metrics', {})
            if 'summary' in gil_metrics:
                gil_efficiency = gil_metrics['summary'].get('gil_efficiency', 100)
                if gil_efficiency < 70:
                    bottlenecks.append({
                        'category': 'Python',
                        'operation': 'GIL contention',
                        'percentage': 100 - gil_efficiency,
                        'severity': 'high' if gil_efficiency < 50 else 'medium'
                    })

        # Analyze GPU bottlenecks
        if 'gpu' in self.results['metrics']:
            gpu_metrics = self.results['metrics']['gpu'].get('gpu_metrics', {})
            gpu_util = gpu_metrics.get('avg_gpu_utilization', 0)
            if gpu_util < 60:
                bottlenecks.append({
                    'category': 'GPU',
                    'operation': 'Low GPU utilization',
                    'percentage': 100 - gpu_util,
                    'severity': 'high' if gpu_util < 30 else 'medium'
                    })

        # Sort by severity and percentage
        bottlenecks.sort(key=lambda x: (x['severity'] == 'high', x['percentage']), reverse=True)

        print("\nIdentified bottlenecks:")
        for b in bottlenecks[:10]:
            severity_marker = "🔴" if b['severity'] == 'high' else "🟡"
            print(f"  {severity_marker} {b['category']}: {b['operation']} ({b['percentage']:.1f}%)")

        self.results['bottlenecks'] = bottlenecks

    def generate_report(self):
        """Generate comprehensive profiling report."""
        print("\n=== Generating Report ===")

        # JSON report
        json_path = self.session_dir / "profile_report.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"  JSON report: {json_path}")

        # Text report
        text_path = self.session_dir / "profile_report.txt"
        with open(text_path, 'w') as f:
            f.write(self._generate_text_report())
        print(f"  Text report: {text_path}")

        # Python profile
        if self.python_profile:
            python_path = self.session_dir / "python_profile.txt"
            with open(python_path, 'w') as f:
                f.write(self.python_profile)
            print(f"  Python profile: {python_path}")

        # Markdown summary
        md_path = self.session_dir / "summary.md"
        with open(md_path, 'w') as f:
            f.write(self._generate_markdown_summary())
        print(f"  Markdown summary: {md_path}")

        print(f"\n✅ All reports saved to: {self.session_dir}")

    def _parse_cpp_metrics(self, snapshot: Dict) -> Dict:
        """Parse C++ instrumentation snapshot."""
        metrics = {
            'timers': {},
            'counters': {}
        }

        for key, value in snapshot.items():
            if isinstance(value, dict) and 'call_count' in value:
                metrics['timers'][key] = {
                    'count': value['call_count'],
                    'total_ns': value.get('total_elapsed_ns', 0)
                }
            else:
                metrics['counters'][key] = value

        return metrics

    def _extract_top_functions(self, stats) -> List[Dict]:
        """Extract top functions from pstats."""
        top_functions = []

        # This is a simplified extraction - would need more robust parsing
        # in production
        return top_functions

    def _generate_text_report(self) -> str:
        """Generate detailed text report."""
        lines = []
        lines.append("=" * 80)
        lines.append(f"COMPREHENSIVE MCTS PROFILING REPORT")
        lines.append(f"Session: {self.session_id}")
        lines.append("=" * 80)
        lines.append("")

        # Configuration
        lines.append("Configuration:")
        for key, value in self.results['configuration'].items():
            lines.append(f"  {key}: {value}")
        lines.append("")

        # Bottlenecks
        if 'bottlenecks' in self.results:
            lines.append("Top Bottlenecks:")
            for b in self.results['bottlenecks'][:5]:
                lines.append(f"  - {b['category']}: {b['operation']} ({b['percentage']:.1f}%)")
            lines.append("")

        # Performance metrics
        lines.append("Performance Metrics:")

        if self.cpp_metrics:
            total_time = sum(m['total_ns'] for m in self.cpp_metrics.get('timers', {}).values())
            sims_per_sec = self.simulations / (total_time / 1e9) if total_time > 0 else 0
            lines.append(f"  C++ throughput: {sims_per_sec:.1f} sims/sec")

        if 'gpu' in self.results['metrics']:
            gpu_throughput = self.results['metrics']['gpu'].get('throughput_samples_per_sec', 0)
            lines.append(f"  GPU throughput: {gpu_throughput:.1f} samples/sec")

        if 'memory' in self.results['metrics']:
            peak_memory = self.results['metrics']['memory'].get('peak_mb', 0)
            lines.append(f"  Peak memory: {peak_memory:.1f}MB")

        return "\n".join(lines)

    def _generate_markdown_summary(self) -> str:
        """Generate markdown summary report."""
        lines = []
        lines.append(f"# MCTS Profiling Report - {self.session_id}")
        lines.append("")
        lines.append("## Executive Summary")
        lines.append("")

        # Key findings
        lines.append("### Key Findings")
        lines.append("")

        if 'bottlenecks' in self.results and self.results['bottlenecks']:
            lines.append("**Critical Bottlenecks:**")
            for b in self.results['bottlenecks'][:3]:
                severity = "🔴" if b['severity'] == 'high' else "🟡"
                lines.append(f"- {severity} **{b['category']}**: {b['operation']} ({b['percentage']:.1f}% impact)")
            lines.append("")

        # Performance metrics table
        lines.append("### Performance Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")

        if self.cpp_metrics:
            total_time = sum(m['total_ns'] for m in self.cpp_metrics.get('timers', {}).values())
            sims_per_sec = self.simulations / (total_time / 1e9) if total_time > 0 else 0
            lines.append(f"| C++ Throughput | {sims_per_sec:.1f} sims/sec |")

        if 'gpu' in self.results['metrics']:
            gpu_metrics = self.results['metrics']['gpu'].get('gpu_metrics', {})
            lines.append(f"| GPU Utilization | {gpu_metrics.get('avg_gpu_utilization', 0):.1f}% |")
            lines.append(f"| GPU Throughput | {self.results['metrics']['gpu'].get('throughput_samples_per_sec', 0):.1f} samples/sec |")

        if 'thread_scaling' in self.results['metrics']:
            best = max(self.results['metrics']['thread_scaling'], key=lambda x: x['throughput'])
            lines.append(f"| Best Thread Count | {best['threads']} threads |")
            lines.append(f"| Peak Throughput | {best['throughput']:.1f} sims/sec |")

        lines.append("")

        # Recommendations
        lines.append("### Recommendations")
        lines.append("")

        # Generate recommendations based on bottlenecks
        recommendations = self._generate_recommendations()
        for i, rec in enumerate(recommendations[:5], 1):
            lines.append(f"{i}. {rec}")

        return "\n".join(lines)

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on profiling."""
        recommendations = []

        if 'bottlenecks' in self.results:
            for b in self.results['bottlenecks']:
                if b['category'] == 'C++' and 'selection' in b['operation'].lower():
                    recommendations.append("Optimize MCTS selection with better vectorization or caching")
                elif b['category'] == 'Python' and 'GIL' in b['operation']:
                    recommendations.append("Reduce GIL contention by moving more operations to C++")
                elif b['category'] == 'GPU' and b['severity'] == 'high':
                    recommendations.append("Increase batch size or reduce inference timeout to improve GPU utilization")

        # Thread scaling recommendations
        if 'thread_scaling' in self.results['metrics']:
            scaling = self.results['metrics']['thread_scaling']
            if scaling[-1]['efficiency'] < 0.5:
                recommendations.append("Reduce thread count - severe contention detected")

        # Memory recommendations
        if 'memory' in self.results['metrics']:
            peak_mb = self.results['metrics']['memory'].get('peak_mb', 0)
            if peak_mb > 1000:
                recommendations.append("Optimize memory usage - consider smaller tree size or better node packing")

        return recommendations

    def run_full_profiling(self):
        """Run complete profiling suite."""
        print("\n" + "="*80)
        print("COMPREHENSIVE MCTS PROFILING")
        print("="*80)

        try:
            # Setup
            root_state, mcts = self.setup_game_and_mcts()

            # GPU inference worker
            gpu_worker = mcts.inference_fn

            # Run all profiling stages
            self.profile_cpp_backend(mcts, root_state)
            self.profile_python_coordination(mcts, root_state)

            if isinstance(gpu_worker, GPUInferenceWorker):
                self.profile_gpu_inference(gpu_worker)

            self.profile_memory_usage(mcts, root_state)
            self.profile_thread_contention(mcts, root_state)

            # Analysis
            self.analyze_bottlenecks()

            # Report generation
            self.generate_report()

            # Cleanup
            mcts.close()

            print("\n✅ Profiling complete!")
            print(f"📊 Reports saved to: {self.session_dir}")

        except Exception as e:
            print(f"\n❌ Profiling failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive MCTS Performance Profiler"
    )

    parser.add_argument(
        '--game',
        choices=['gomoku', 'chess', 'go'],
        default='gomoku',
        help='Game type to profile'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=4,
        help='Number of MCTS threads'
    )
    parser.add_argument(
        '--simulations',
        type=int,
        default=800,
        help='Simulations per search'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='GPU batch size'
    )
    parser.add_argument(
        '--timeout-ms',
        type=float,
        default=2.0,
        help='Batch collection timeout (ms)'
    )
    parser.add_argument(
        '--output-dir',
        default='profiling_reports',
        help='Output directory for reports'
    )

    args = parser.parse_args()

    # Create and run profiler
    profiler = ComprehensiveMCTSProfiler(
        game_type=args.game,
        threads=args.threads,
        simulations=args.simulations,
        batch_size=args.batch_size,
        timeout_ms=args.timeout_ms,
        output_dir=args.output_dir
    )

    profiler.run_full_profiling()


if __name__ == "__main__":
    main()