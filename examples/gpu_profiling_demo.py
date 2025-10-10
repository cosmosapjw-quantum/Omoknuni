#!/usr/bin/env python3
"""
GPU Profiling Demo
==================

Interactive demonstration of GPU profiling capabilities.

This example shows:
1. Basic profiling workflow
2. Transfer time analysis
3. Batch size optimization
4. Mixed precision comparison
5. Report generation and visualization
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.neural.model import create_model_for_game
from src.telemetry.gpu_profiler import GPUProfiler


def demo_basic_profiling():
    """Demonstrate basic GPU profiling."""
    print("\n" + "="*80)
    print("Demo 1: Basic GPU Profiling")
    print("="*80 + "\n")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU profiling demo")
        return

    # Create model
    print("Creating Gomoku model...")
    model = create_model_for_game('gomoku').cuda().eval()
    print(f"Model parameters: {model.get_num_parameters():,}")

    # Create profiler
    profiler = GPUProfiler(
        device='cuda:0',
        log_dir='runs/demo_profiling',
        enable_nvml=True
    )

    print("\nStarting profiling...")
    profiler.start_profiling()

    # Run inference batches
    print("Running 20 inference batches...")
    batch_size = 32
    num_batches = 20

    with torch.no_grad():
        for i in range(num_batches):
            # Create random input (Gomoku: 36 planes, 15x15 board)
            input_tensor = torch.randn(batch_size, 36, 15, 15).cuda()

            with profiler.profile_batch(batch_size=batch_size):
                policy, value = model(input_tensor)

            if (i + 1) % 5 == 0:
                print(f"  Completed {i+1}/{num_batches} batches")

    print("\nStopping profiler...")
    profiler.stop_profiling()

    # Generate and print report
    print("\nGenerating report...")
    profiler.print_summary()

    # Export report
    report_path = profiler.export_report(filename='demo_basic_report.json')
    print(f"\nReport saved to: {report_path}")

    return profiler.generate_report()


def demo_transfer_profiling():
    """Demonstrate H2D/D2H transfer profiling."""
    print("\n" + "="*80)
    print("Demo 2: Transfer Time Analysis")
    print("="*80 + "\n")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping demo")
        return

    model = create_model_for_game('gomoku').cuda().eval()

    profiler = GPUProfiler(
        device='cuda:0',
        log_dir='runs/demo_profiling'
    )

    print("Profiling with explicit H2D/D2H timing...")
    profiler.start_profiling()

    batch_size = 64
    num_batches = 10

    with torch.no_grad():
        for i in range(num_batches):
            # Create input on CPU
            input_cpu = torch.randn(batch_size, 36, 15, 15)

            with profiler.profile_batch(batch_size=batch_size):
                # Profile H2D transfer
                with profiler.profile_transfer('h2d'):
                    input_gpu = input_cpu.to('cuda:0', non_blocking=True)

                # Inference (already on GPU)
                policy_gpu, value_gpu = model(input_gpu)

                # Profile D2H transfer
                with profiler.profile_transfer('d2h'):
                    policy_cpu = policy_gpu.cpu()
                    value_cpu = value_gpu.cpu()

    profiler.stop_profiling()

    # Analyze transfer times
    report = profiler.generate_report()

    print("\nTransfer Time Analysis:")
    print(f"  Avg H2D transfer: {report.avg_h2d_transfer_ms:.3f} ms")
    print(f"  Avg Inference:    {report.avg_inference_ms:.3f} ms")
    print(f"  Avg D2H transfer: {report.avg_d2h_transfer_ms:.3f} ms")

    h2d_pct = 100 * report.avg_h2d_transfer_ms / (report.avg_h2d_transfer_ms + report.avg_inference_ms + report.avg_d2h_transfer_ms)
    inf_pct = 100 * report.avg_inference_ms / (report.avg_h2d_transfer_ms + report.avg_inference_ms + report.avg_d2h_transfer_ms)
    d2h_pct = 100 * report.avg_d2h_transfer_ms / (report.avg_h2d_transfer_ms + report.avg_inference_ms + report.avg_d2h_transfer_ms)

    print(f"\n  H2D Transfer: {h2d_pct:.1f}%")
    print(f"  Inference:    {inf_pct:.1f}%")
    print(f"  D2H Transfer: {d2h_pct:.1f}%")

    return report


def demo_batch_size_comparison():
    """Demonstrate batch size optimization."""
    print("\n" + "="*80)
    print("Demo 3: Batch Size Optimization")
    print("="*80 + "\n")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping demo")
        return

    model = create_model_for_game('gomoku').cuda().eval()

    batch_sizes = [16, 32, 64]
    results = []

    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")

        profiler = GPUProfiler(
            device='cuda:0',
            log_dir='runs/demo_profiling'
        )

        profiler.start_profiling()

        with torch.no_grad():
            for i in range(10):
                input_tensor = torch.randn(batch_size, 36, 15, 15).cuda()
                with profiler.profile_batch(batch_size=batch_size):
                    _ = model(input_tensor)

        profiler.stop_profiling()

        report = profiler.generate_report()
        results.append((batch_size, report))

        print(f"  Throughput: {report.avg_throughput:.1f} samples/sec")
        print(f"  GPU Util:   {report.avg_gpu_utilization:.1f}%")
        print(f"  Memory:     {report.avg_memory_used_mb:.0f} MB")

    # Summary
    print("\n" + "-"*80)
    print("Batch Size Comparison Summary:")
    print("-"*80)
    print(f"{'Batch Size':<12} {'Throughput':<15} {'GPU Util':<12} {'Memory':<10}")
    print("-"*80)

    for batch_size, report in results:
        print(f"{batch_size:<12} {report.avg_throughput:<15.1f} {report.avg_gpu_utilization:<12.1f} {report.avg_memory_used_mb:<10.0f}")

    # Find optimal
    optimal_idx = np.argmax([r.avg_throughput for _, r in results])
    optimal_batch_size = results[optimal_idx][0]
    print(f"\nOptimal batch size: {optimal_batch_size}")

    return results


def demo_mixed_precision():
    """Demonstrate FP32 vs FP16 profiling."""
    print("\n" + "="*80)
    print("Demo 4: Mixed Precision Comparison")
    print("="*80 + "\n")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping demo")
        return

    model = create_model_for_game('gomoku').cuda().eval()

    batch_size = 64
    num_batches = 20

    # Profile FP32
    print("Profiling FP32 inference...")
    profiler_fp32 = GPUProfiler(device='cuda:0', log_dir='runs/demo_profiling')
    profiler_fp32.start_profiling()

    with torch.no_grad():
        for i in range(num_batches):
            input_tensor = torch.randn(batch_size, 36, 15, 15).cuda()
            with profiler_fp32.profile_batch(batch_size=batch_size):
                _ = model(input_tensor)

    profiler_fp32.stop_profiling()
    report_fp32 = profiler_fp32.generate_report()

    # Profile FP16
    print("\nProfiling FP16 inference (mixed precision)...")
    profiler_fp16 = GPUProfiler(device='cuda:0', log_dir='runs/demo_profiling')
    profiler_fp16.start_profiling()

    with torch.no_grad():
        for i in range(num_batches):
            input_tensor = torch.randn(batch_size, 36, 15, 15).cuda()
            with profiler_fp16.profile_batch(batch_size=batch_size):
                with torch.cuda.amp.autocast():
                    _ = model(input_tensor)

    profiler_fp16.stop_profiling()
    report_fp16 = profiler_fp16.generate_report()

    # Compare
    print("\n" + "-"*80)
    print("FP32 vs FP16 Comparison:")
    print("-"*80)
    print(f"{'Metric':<25} {'FP32':<15} {'FP16':<15} {'Speedup':<10}")
    print("-"*80)

    speedup_throughput = report_fp16.avg_throughput / report_fp32.avg_throughput
    speedup_latency = report_fp32.avg_inference_ms / report_fp16.avg_inference_ms
    memory_reduction = (report_fp32.avg_memory_used_mb - report_fp16.avg_memory_used_mb) / report_fp32.avg_memory_used_mb

    print(f"{'Throughput (samples/s)':<25} {report_fp32.avg_throughput:<15.1f} {report_fp16.avg_throughput:<15.1f} {speedup_throughput:<10.2f}x")
    print(f"{'Inference time (ms)':<25} {report_fp32.avg_inference_ms:<15.3f} {report_fp16.avg_inference_ms:<15.3f} {speedup_latency:<10.2f}x")
    print(f"{'GPU utilization (%)':<25} {report_fp32.avg_gpu_utilization:<15.1f} {report_fp16.avg_gpu_utilization:<15.1f} {'-':<10}")
    print(f"{'Memory (MB)':<25} {report_fp32.avg_memory_used_mb:<15.0f} {report_fp16.avg_memory_used_mb:<15.0f} {memory_reduction*100:<10.1f}%")
    print(f"{'Power (W)':<25} {report_fp32.avg_power_watts:<15.1f} {report_fp16.avg_power_watts:<15.1f} {'-':<10}")

    return report_fp32, report_fp16


def demo_realtime_monitoring():
    """Demonstrate real-time metrics monitoring."""
    print("\n" + "="*80)
    print("Demo 5: Real-Time Metrics Monitoring")
    print("="*80 + "\n")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping demo")
        return

    model = create_model_for_game('gomoku').cuda().eval()

    profiler = GPUProfiler(
        device='cuda:0',
        log_dir='runs/demo_profiling',
        sampling_interval_ms=100
    )

    print("Starting real-time monitoring (will print metrics every 10 batches)...")
    profiler.start_profiling()

    batch_size = 32
    num_batches = 50

    with torch.no_grad():
        for i in range(num_batches):
            input_tensor = torch.randn(batch_size, 36, 15, 15).cuda()

            with profiler.profile_batch(batch_size=batch_size):
                _ = model(input_tensor)

            # Print real-time metrics every 10 batches
            if (i + 1) % 10 == 0:
                metrics = profiler.get_realtime_metrics()

                print(f"\nBatch {i+1}/{num_batches}:")
                if 'cuda' in metrics:
                    cuda = metrics['cuda']
                    print(f"  GPU Util:  {cuda['gpu_utilization']:.1f}%")
                    print(f"  Memory:    {cuda['memory_used_mb']:.0f} MB")
                    print(f"  Power:     {cuda['power_draw_watts']:.1f} W")
                    print(f"  Temp:      {cuda['temperature_c']}°C")

                if 'recent_batches' in metrics:
                    recent = metrics['recent_batches']
                    print(f"  Avg Time:  {recent['avg_time_ms']:.3f} ms")
                    print(f"  Throughput: {recent['avg_throughput']:.1f} samples/s")

    profiler.stop_profiling()
    print("\nMonitoring complete!")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("GPU Profiling System Demo")
    print("="*80)

    if not torch.cuda.is_available():
        print("\nERROR: CUDA not available. GPU profiling requires CUDA.")
        return

    # Print system info
    print(f"\nSystem Information:")
    print(f"  PyTorch version:   {torch.__version__}")
    print(f"  CUDA version:      {torch.version.cuda}")
    print(f"  Device:            {torch.cuda.get_device_name(0)}")
    print(f"  Compute capability: {'.'.join(map(str, torch.cuda.get_device_capability(0)))}")
    print(f"  Total memory:      {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")

    # Run demos
    try:
        demo_basic_profiling()
        demo_transfer_profiling()
        demo_batch_size_comparison()
        demo_mixed_precision()
        demo_realtime_monitoring()

        print("\n" + "="*80)
        print("All demos completed successfully!")
        print("="*80)
        print(f"\nResults saved to: runs/demo_profiling/")
        print("To view TensorBoard traces, run:")
        print("  tensorboard --logdir runs/demo_profiling/tensorboard")

    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
