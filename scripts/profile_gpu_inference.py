#!/usr/bin/env python3
"""
GPU Inference Profiling Script
==============================

Comprehensive GPU profiling for neural network inference pipeline.
Profiles the complete inference stack with detailed metrics.

Usage:
    # Basic profiling (100 batches)
    python scripts/profile_gpu_inference.py --game gomoku --batch-size 64

    # Extended profiling with TensorBoard export
    python scripts/profile_gpu_inference.py --game gomoku --batch-size 64 \\
        --num-batches 500 --tensorboard --mixed-precision

    # Compare batch sizes
    python scripts/profile_gpu_inference.py --game gomoku --batch-sizes 16,32,64,128 \\
        --num-batches 200 --compare

    # Profile with real inference worker
    python scripts/profile_gpu_inference.py --game gomoku --use-inference-worker \\
        --queue-depth 256 --num-batches 1000

Output:
    - Console summary with key metrics
    - JSON report: runs/gpu_profiling/<session_id>_report.json
    - TensorBoard traces: runs/gpu_profiling/tensorboard/<session_id>/
    - Plots: runs/gpu_profiling/<session_id>_plots.png
"""

import argparse
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neural.model import create_model_for_game, AlphaZeroNet
from src.telemetry.gpu_profiler import GPUProfiler, profile_inference_pipeline
from src.neural.inference_worker import GPUInferenceWorker


def create_dummy_dataloader(game: str, batch_size: int, num_batches: int):
    """Create dummy dataloader for profiling."""
    if game == 'gomoku':
        input_shape = (36, 15, 15)
    elif game == 'chess':
        input_shape = (30, 8, 8)
    elif game == 'go':
        input_shape = (25, 19, 19)
    else:
        raise ValueError(f"Unknown game: {game}")

    class DummyDataset:
        def __init__(self, num_batches, batch_size, input_shape):
            self.num_batches = num_batches
            self.batch_size = batch_size
            self.input_shape = input_shape

        def __iter__(self):
            for _ in range(self.num_batches):
                batch = torch.randn(self.batch_size, *self.input_shape)
                yield batch

        def __len__(self):
            return self.num_batches

    return DummyDataset(num_batches, batch_size, input_shape)


def profile_single_config(
    game: str,
    batch_size: int,
    num_batches: int,
    device: str,
    use_mixed_precision: bool,
    tensorboard: bool,
    log_dir: str
):
    """Profile a single configuration."""
    print(f"\n{'='*80}")
    print(f"Profiling Configuration")
    print(f"{'='*80}")
    print(f"Game:              {game}")
    print(f"Batch size:        {batch_size}")
    print(f"Num batches:       {num_batches}")
    print(f"Mixed precision:   {use_mixed_precision}")
    print(f"Device:            {device}")
    print(f"{'='*80}\n")

    # Create model
    model = create_model_for_game(game)
    model = model.to(device)
    model.eval()

    if use_mixed_precision:
        from src.neural.model import enable_mixed_precision
        model = enable_mixed_precision(model)

    # Create dataloader
    dataloader = create_dummy_dataloader(game, batch_size, num_batches)

    # Create profiler
    profiler = GPUProfiler(
        device=device,
        log_dir=log_dir,
        enable_nvml=True,
        enable_torch_profiler=tensorboard,
        tensorboard_export=tensorboard
    )

    # Warmup
    print("Warming up GPU...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 10:
                break
            inputs = batch.to(device)
            if use_mixed_precision:
                with torch.cuda.amp.autocast():
                    _ = model(inputs)
            else:
                _ = model(inputs)

    # Profile
    print(f"Profiling {num_batches} batches...")
    profiler.start_profiling(
        trace_memory=True,
        profile_tensorboard=tensorboard,
        active_steps=min(50, num_batches)
    )

    dataloader = create_dummy_dataloader(game, batch_size, num_batches)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs = batch

            with profiler.profile_transfer('h2d'):
                inputs_gpu = inputs.to(device, non_blocking=True)

            with profiler.profile_batch(batch_size=batch_size):
                if use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        policy, value = model(inputs_gpu)
                else:
                    policy, value = model(inputs_gpu)

                with profiler.profile_transfer('d2h'):
                    policy_cpu = policy.cpu()
                    value_cpu = value.cpu()

    profiler.stop_profiling()

    # Generate report
    report = profiler.generate_report()
    profiler.print_summary()

    # Export
    report_path = profiler.export_report()
    print(f"\nReport exported to: {report_path}")

    # Plot results
    plot_profiling_results(report, profiler, log_dir)

    return report


def plot_profiling_results(report, profiler, log_dir):
    """Generate visualization plots."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'GPU Profiling Results - {report.session_id}', fontsize=16)

    # 1. GPU Utilization over time
    ax = axes[0, 0]
    if profiler._cuda_metrics:
        timestamps = [(m.timestamp - report.start_time) for m in profiler._cuda_metrics]
        gpu_utils = [m.gpu_utilization for m in profiler._cuda_metrics]
        ax.plot(timestamps, gpu_utils, 'b-', alpha=0.7)
        ax.axhline(y=80, color='g', linestyle='--', label='Target (80%)')
        ax.axhline(y=report.avg_gpu_utilization, color='r', linestyle='--',
                   label=f'Average ({report.avg_gpu_utilization:.1f}%)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('GPU Utilization (%)')
        ax.set_title('GPU Utilization Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 2. Memory usage over time
    ax = axes[0, 1]
    if profiler._cuda_metrics:
        timestamps = [(m.timestamp - report.start_time) for m in profiler._cuda_metrics]
        mem_used = [m.memory_used_mb for m in profiler._cuda_metrics]
        mem_total = profiler._cuda_metrics[0].memory_total_mb
        ax.plot(timestamps, mem_used, 'b-', alpha=0.7)
        ax.axhline(y=mem_total, color='r', linestyle='--', label=f'Total ({mem_total:.0f} MB)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Memory Used (MB)')
        ax.set_title('GPU Memory Usage Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 3. Batch timing breakdown
    ax = axes[0, 2]
    if report.batch_metrics:
        batch_ids = [b.batch_id for b in report.batch_metrics[:100]]  # First 100
        h2d_times = [b.h2d_transfer_ms for b in report.batch_metrics[:100]]
        inference_times = [b.inference_ms for b in report.batch_metrics[:100]]
        d2h_times = [b.d2h_transfer_ms for b in report.batch_metrics[:100]]

        ax.bar(batch_ids, h2d_times, label='H2D Transfer', alpha=0.7)
        ax.bar(batch_ids, inference_times, bottom=h2d_times, label='Inference', alpha=0.7)
        ax.bar(batch_ids, d2h_times,
               bottom=[h+i for h, i in zip(h2d_times, inference_times)],
               label='D2H Transfer', alpha=0.7)

        ax.set_xlabel('Batch ID')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Batch Timing Breakdown (First 100)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # 4. Throughput over time
    ax = axes[1, 0]
    if report.batch_metrics:
        batch_ids = [b.batch_id for b in report.batch_metrics]
        throughputs = [b.samples_per_second for b in report.batch_metrics]
        ax.plot(batch_ids, throughputs, 'b-', alpha=0.7)
        ax.axhline(y=report.avg_throughput, color='r', linestyle='--',
                   label=f'Average ({report.avg_throughput:.1f} samples/s)')
        ax.set_xlabel('Batch ID')
        ax.set_ylabel('Throughput (samples/sec)')
        ax.set_title('Inference Throughput Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 5. Power consumption
    ax = axes[1, 1]
    if profiler._cuda_metrics:
        timestamps = [(m.timestamp - report.start_time) for m in profiler._cuda_metrics]
        power = [m.power_draw_watts for m in profiler._cuda_metrics]
        power_limit = profiler._cuda_metrics[0].power_limit_watts
        ax.plot(timestamps, power, 'b-', alpha=0.7)
        ax.axhline(y=power_limit, color='r', linestyle='--',
                   label=f'Limit ({power_limit:.0f}W)')
        ax.axhline(y=report.avg_power_watts, color='g', linestyle='--',
                   label=f'Average ({report.avg_power_watts:.1f}W)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power Draw (W)')
        ax.set_title('GPU Power Consumption')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 6. SM Clock frequency
    ax = axes[1, 2]
    if profiler._cuda_metrics:
        timestamps = [(m.timestamp - report.start_time) for m in profiler._cuda_metrics]
        clocks = [m.sm_clock_mhz for m in profiler._cuda_metrics]
        ax.plot(timestamps, clocks, 'b-', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('SM Clock (MHz)')
        ax.set_title('GPU Clock Frequency')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(log_dir) / f"{report.session_id}_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plots saved to: {plot_path}")
    plt.close()


def compare_batch_sizes(
    game: str,
    batch_sizes: list,
    num_batches: int,
    device: str,
    use_mixed_precision: bool,
    log_dir: str
):
    """Compare performance across different batch sizes."""
    print(f"\n{'='*80}")
    print(f"Batch Size Comparison")
    print(f"{'='*80}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"{'='*80}\n")

    results = []

    for batch_size in batch_sizes:
        try:
            report = profile_single_config(
                game=game,
                batch_size=batch_size,
                num_batches=num_batches,
                device=device,
                use_mixed_precision=use_mixed_precision,
                tensorboard=False,  # Disable for comparison runs
                log_dir=log_dir
            )
            results.append((batch_size, report))
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"\nBatch size {batch_size} failed with OOM, skipping...")
                torch.cuda.empty_cache()
            else:
                raise

    # Plot comparison
    if results:
        plot_batch_size_comparison(results, log_dir)

    return results


def plot_batch_size_comparison(results, log_dir):
    """Plot batch size comparison."""
    import matplotlib
    matplotlib.use('Agg')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Batch Size Comparison', fontsize=16)

    batch_sizes = [bs for bs, _ in results]
    throughputs = [r.avg_throughput for _, r in results]
    gpu_utils = [r.avg_gpu_utilization for _, r in results]
    mem_used = [r.avg_memory_used_mb for _, r in results]
    inference_times = [r.avg_inference_ms for _, r in results]

    # Throughput vs batch size
    ax = axes[0, 0]
    ax.plot(batch_sizes, throughputs, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Throughput (samples/sec)')
    ax.set_title('Throughput vs Batch Size')
    ax.grid(True, alpha=0.3)

    # GPU utilization vs batch size
    ax = axes[0, 1]
    ax.plot(batch_sizes, gpu_utils, 'go-', linewidth=2, markersize=8)
    ax.axhline(y=80, color='r', linestyle='--', label='Target (80%)')
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('GPU Utilization (%)')
    ax.set_title('GPU Utilization vs Batch Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Memory usage vs batch size
    ax = axes[1, 0]
    ax.plot(batch_sizes, mem_used, 'ro-', linewidth=2, markersize=8)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Memory Used (MB)')
    ax.set_title('Memory Usage vs Batch Size')
    ax.grid(True, alpha=0.3)

    # Inference time vs batch size
    ax = axes[1, 1]
    ax.plot(batch_sizes, inference_times, 'mo-', linewidth=2, markersize=8)
    ax.set_xlabel('Batch Size')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Inference Time vs Batch Size')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(log_dir) / "batch_size_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plots saved to: {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Profile GPU inference performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--game', type=str, default='gomoku',
                       choices=['gomoku', 'chess', 'go'],
                       help='Game type')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for inference')
    parser.add_argument('--batch-sizes', type=str,
                       help='Comma-separated batch sizes for comparison (e.g., 16,32,64)')
    parser.add_argument('--num-batches', type=int, default=100,
                       help='Number of batches to profile')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='CUDA device')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Enable FP16 mixed precision')
    parser.add_argument('--tensorboard', action='store_true',
                       help='Export TensorBoard traces')
    parser.add_argument('--compare', action='store_true',
                       help='Run batch size comparison')
    parser.add_argument('--log-dir', type=str, default='runs/gpu_profiling',
                       help='Output directory')
    parser.add_argument('--use-inference-worker', action='store_true',
                       help='Profile using InferenceWorker instead of direct model')
    parser.add_argument('--queue-depth', type=int, default=256,
                       help='Queue depth for InferenceWorker profiling')

    args = parser.parse_args()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    print(f"\nGPU Profiling Configuration:")
    print(f"  PyTorch:     {torch.__version__}")
    print(f"  CUDA:        {torch.version.cuda}")
    print(f"  Device:      {torch.cuda.get_device_name(args.device)}")
    print(f"  Compute cap: {'.'.join(map(str, torch.cuda.get_device_capability(args.device)))}")

    # Run profiling
    if args.compare or args.batch_sizes:
        if args.batch_sizes:
            batch_sizes = [int(bs) for bs in args.batch_sizes.split(',')]
        else:
            batch_sizes = [16, 32, 64, 128]

        compare_batch_sizes(
            game=args.game,
            batch_sizes=batch_sizes,
            num_batches=args.num_batches,
            device=args.device,
            use_mixed_precision=args.mixed_precision,
            log_dir=args.log_dir
        )
    else:
        profile_single_config(
            game=args.game,
            batch_size=args.batch_size,
            num_batches=args.num_batches,
            device=args.device,
            use_mixed_precision=args.mixed_precision,
            tensorboard=args.tensorboard,
            log_dir=args.log_dir
        )

    print("\nProfiling complete!")


if __name__ == '__main__':
    main()
