# GPU Profiling System Guide

Comprehensive GPU profiling for neural network inference optimization on RTX 3060 Ti.

## Overview

The GPU profiling system provides multi-level performance analysis:

1. **Hardware Metrics** - Real-time GPU utilization, memory, power, clocks via NVML
2. **Kernel Profiling** - Detailed CUDA kernel timing and analysis via torch.profiler
3. **Transfer Profiling** - H2D/D2H PCIe transfer timing with CUDA events
4. **Memory Analysis** - Allocation patterns, fragmentation, peak usage
5. **Batch Analysis** - Per-batch timing breakdown and throughput
6. **TensorBoard Export** - Interactive visualization of profiling data

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        GPUProfiler                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────┐│
│  │  NVML Monitoring │  │ torch.profiler   │  │CUDA Events││
│  │  (Hardware)      │  │ (Kernels)        │  │(Timing)   ││
│  └────────┬─────────┘  └────────┬─────────┘  └─────┬─────┘│
│           │                     │                   │      │
│           ▼                     ▼                   ▼      │
│  ┌─────────────────────────────────────────────────────┐  │
│  │            Metrics Collection Thread                │  │
│  │  • GPU utilization (100ms intervals)                │  │
│  │  • Memory bandwidth                                 │  │
│  │  • Power consumption                                │  │
│  │  • Temperature/clock frequency                      │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │            Batch Profiling Context                  │  │
│  │  • CUDA event timing (microsecond precision)        │  │
│  │  • H2D/D2H transfer breakdown                       │  │
│  │  • Inference time measurement                       │  │
│  │  • Queue depth tracking                             │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │            Report Generation                        │  │
│  │  • JSON export                                      │  │
│  │  • TensorBoard traces                               │  │
│  │  • Matplotlib visualizations                        │  │
│  │  • Performance analysis                             │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Profiling

```python
from src.telemetry.gpu_profiler import GPUProfiler
from src.neural.model import create_model_for_game
import torch

# Create model
model = create_model_for_game('gomoku').cuda().eval()

# Create profiler
profiler = GPUProfiler(
    device='cuda:0',
    log_dir='runs/gpu_profiling',
    enable_nvml=True,
    tensorboard_export=True
)

# Start profiling
profiler.start_profiling()

# Run inference workload
with torch.no_grad():
    for batch in dataloader:
        inputs = batch.to('cuda:0')

        with profiler.profile_batch(batch_size=len(inputs)):
            outputs = model(inputs)

# Stop and generate report
profiler.stop_profiling()
report = profiler.generate_report()
profiler.print_summary()
profiler.export_report()
```

### Using Context Manager

```python
with GPUProfiler(device='cuda:0') as profiler:
    for batch in dataloader:
        with profiler.profile_batch(batch_size=len(batch)):
            outputs = model(batch.cuda())
```

### Profiling Transfer Times

```python
profiler.start_profiling()

for batch in dataloader:
    # Profile H2D transfer
    with profiler.profile_transfer('h2d'):
        inputs_gpu = batch.to('cuda:0', non_blocking=True)

    # Profile inference
    with profiler.profile_batch(batch_size=len(batch)):
        with torch.no_grad():
            outputs = model(inputs_gpu)

    # Profile D2H transfer
    with profiler.profile_transfer('d2h'):
        outputs_cpu = outputs.cpu()

profiler.stop_profiling()
```

## Command-Line Tools

### Basic Profiling

```bash
# Profile Gomoku inference (default: 64 batch size, 100 batches)
python scripts/profile_gpu_inference.py --game gomoku

# Profile with mixed precision
python scripts/profile_gpu_inference.py --game gomoku --batch-size 64 --mixed-precision

# Extended profiling with TensorBoard export
python scripts/profile_gpu_inference.py --game gomoku --num-batches 500 --tensorboard
```

### Batch Size Comparison

```bash
# Compare multiple batch sizes
python scripts/profile_gpu_inference.py --game gomoku --batch-sizes 16,32,64,128 --compare

# Quick comparison
python scripts/profile_gpu_inference.py --game gomoku --compare
```

### Advanced Options

```bash
# Full profiling session
python scripts/profile_gpu_inference.py \
    --game gomoku \
    --batch-size 64 \
    --num-batches 1000 \
    --mixed-precision \
    --tensorboard \
    --log-dir runs/profiling_$(date +%Y%m%d_%H%M%S)
```

## Metrics Reference

### CUDAMetrics (Hardware)

Collected via NVML at ~100ms intervals:

- **gpu_utilization** - GPU core utilization (0-100%)
- **memory_utilization** - Memory controller utilization (0-100%)
- **memory_used_mb** - VRAM in use
- **sm_clock_mhz** - SM clock frequency
- **memory_clock_mhz** - Memory clock frequency
- **power_draw_watts** - Current power consumption
- **temperature_c** - GPU temperature
- **fan_speed_percent** - Fan speed (0-100%)
- **pcie_throughput_mbps** - PCIe bandwidth usage

### InferenceBatchMetrics (Per-Batch)

Captured for each inference batch:

- **batch_size** - Number of samples
- **total_time_ms** - End-to-end batch time
- **h2d_transfer_ms** - Host-to-Device transfer time
- **inference_ms** - GPU inference time
- **d2h_transfer_ms** - Device-to-Host transfer time
- **samples_per_second** - Throughput for this batch
- **gpu_utilization** - GPU state during batch
- **memory_used_mb** - VRAM used
- **power_draw_watts** - Power during batch

### ProfilingSession (Aggregate)

Summary statistics for entire session:

- **total_batches** - Total batches processed
- **total_samples** - Total samples processed
- **avg_batch_size** - Average batch size
- **avg_throughput** - Overall samples/second
- **avg_gpu_utilization** - Mean GPU utilization
- **p50/p95_gpu_utilization** - Percentile statistics
- **memory_efficiency** - Used / Total ratio
- **total_energy_joules** - Total energy consumed

## Performance Targets (RTX 3060 Ti)

### Target Metrics

| Metric | Target | Baseline | Optimized |
|--------|--------|----------|-----------|
| GPU Utilization | 80-92% | ~60% | 85-92% |
| Batch Size | 32-64 | 32 | 64 |
| Inference Latency (FP32) | <10ms | 12ms | 8ms |
| Inference Latency (FP16) | <5ms | 8ms | 4ms |
| Memory Bandwidth | >70% | 50% | 75% |
| Tensor Core Util (FP16) | >50% | 0% | 60% |

### Interpreting Results

**GPU Utilization:**
- **<60%** - Severe CPU bottleneck or small batch sizes
- **60-80%** - Good, but room for optimization
- **80-92%** - Excellent, target range
- **>95%** - May indicate batch sizes too large

**Memory Bandwidth:**
- **<50%** - Compute-bound workload (good for inference)
- **50-70%** - Balanced
- **>70%** - Memory-bound (check memory access patterns)

**Inference Time Breakdown:**
- **H2D Transfer >20%** - Use pinned memory, non-blocking transfers
- **Inference >60%** - Normal for compute-heavy models
- **D2H Transfer >20%** - Reduce output size or batch results

## Optimization Workflow

### 1. Baseline Profiling

```bash
# Establish baseline
python scripts/profile_gpu_inference.py --game gomoku --batch-size 64 --num-batches 200
```

### 2. Identify Bottlenecks

Check the report for:
- GPU utilization < 80%
- High H2D/D2H transfer times
- Memory fragmentation
- Power throttling

### 3. Apply Optimizations

**For Low GPU Utilization:**
```bash
# Test larger batch sizes
python scripts/profile_gpu_inference.py --game gomoku --batch-sizes 32,64,96,128 --compare
```

**For Transfer Bottlenecks:**
```python
# Use pinned memory and non-blocking transfers
profiler = GPUProfiler(device='cuda:0')

for batch in dataloader:
    # Pinned memory allocated once
    inputs_pinned = torch.empty(batch.shape, pin_memory=True)
    inputs_pinned.copy_(batch)

    with profiler.profile_transfer('h2d'):
        inputs_gpu = inputs_pinned.to('cuda:0', non_blocking=True)
```

**For Inference Time:**
```bash
# Enable mixed precision
python scripts/profile_gpu_inference.py --game gomoku --mixed-precision
```

### 4. Validate Improvements

```bash
# Re-run with optimizations
python scripts/profile_gpu_inference.py --game gomoku --batch-size 64 --mixed-precision --num-batches 200

# Compare reports
```

## TensorBoard Visualization

### Export and View

```python
profiler = GPUProfiler(
    device='cuda:0',
    log_dir='runs/gpu_profiling',
    tensorboard_export=True
)

profiler.start_profiling(profile_tensorboard=True)
# ... run workload ...
profiler.stop_profiling()
```

```bash
# Launch TensorBoard
tensorboard --logdir runs/gpu_profiling/tensorboard

# Open browser to http://localhost:6006
```

### Available Views

1. **Operator View** - Time spent in each operation
2. **Kernel View** - CUDA kernel execution times
3. **Trace View** - Timeline visualization
4. **Memory View** - Memory allocation over time

## Integration with Existing Systems

### With InferenceWorker

```python
from src.neural.inference_worker import GPUInferenceWorker
from src.telemetry.gpu_profiler import GPUProfiler

worker = GPUInferenceWorker(
    model_path='models/gomoku_latest.pth',
    device='cuda:0',
    batch_size=64
)

profiler = GPUProfiler(device='cuda:0')

# Inject profiler into worker's batch_inference method
original_batch_inference = worker.batch_inference

def profiled_batch_inference(positions):
    with profiler.profile_batch(batch_size=len(positions)):
        return original_batch_inference(positions)

worker.batch_inference = profiled_batch_inference

# Run worker normally
profiler.start_profiling()
# ... worker.start_worker() ...
# ... run MCTS search ...
profiler.stop_profiling()
```

### With Telemetry System

```python
from src.telemetry.metrics import get_metrics_collector
from src.telemetry.gpu_profiler import GPUProfiler

metrics_collector = get_metrics_collector()
profiler = GPUProfiler(device='cuda:0')

# Both systems can run concurrently
metrics_collector.start_collection()
profiler.start_profiling()

# ... run workload ...

profiler.stop_profiling()
metrics_collector.stop_collection()

# Combine reports
prometheus_metrics = metrics_collector.get_prometheus_metrics()
profiler_report = profiler.generate_report()
```

## Troubleshooting

### NVML Not Available

```
WARNING: pynvml not available, hardware monitoring disabled
```

**Solution:**
```bash
pip install nvidia-ml-py
```

### CUDA Out of Memory During Profiling

Torch profiler adds memory overhead.

**Solution:**
```python
# Disable torch profiler or reduce batch size
profiler = GPUProfiler(
    device='cuda:0',
    enable_torch_profiler=False  # Reduce overhead
)
```

### TensorBoard Not Generating Traces

**Solution:**
```python
profiler.start_profiling(
    profile_tensorboard=True,
    active_steps=50  # Ensure active_steps > 0
)

# Must call profiler.step() or use profile_batch context
```

### Profiling Adds Too Much Overhead

**Solution:**
```python
# Minimal overhead configuration
profiler = GPUProfiler(
    device='cuda:0',
    enable_torch_profiler=False,  # Disable kernel profiling
    memory_profiling=False,       # Disable memory snapshots
    sampling_interval_ms=500      # Reduce sampling frequency
)
```

## Best Practices

### 1. Warmup Before Profiling

```python
# Run several warmup iterations
model.eval()
with torch.no_grad():
    for _ in range(10):
        dummy_input = torch.randn(64, 36, 15, 15).cuda()
        _ = model(dummy_input)

# Clear metrics
torch.cuda.reset_peak_memory_stats()

# Now start profiling
profiler.start_profiling()
```

### 2. Profile Representative Workloads

```python
# Use realistic batch sizes and input distributions
# Don't profile with synthetic/constant tensors if possible

# Good: Use actual game states
states = load_game_states('data/positions.npz')

# Avoid: Random tensors (may not trigger same code paths)
dummy_data = torch.randn(...)
```

### 3. Multiple Profiling Runs

```python
# Run multiple times to account for variance
results = []
for run in range(3):
    with GPUProfiler(device='cuda:0') as profiler:
        # ... run workload ...
        report = profiler.generate_report()
        results.append(report)

# Analyze variance
avg_throughput = np.mean([r.avg_throughput for r in results])
std_throughput = np.std([r.avg_throughput for r in results])
```

### 4. Compare Before/After

```python
# Baseline
with GPUProfiler(log_dir='runs/baseline') as profiler:
    baseline_report = run_inference()

# With optimization
with GPUProfiler(log_dir='runs/optimized') as profiler:
    optimized_report = run_inference()

# Compare
speedup = optimized_report.avg_throughput / baseline_report.avg_throughput
print(f"Speedup: {speedup:.2f}x")
```

## API Reference

See inline documentation in `src/telemetry/gpu_profiler.py` for complete API reference.

### Key Classes

- **GPUProfiler** - Main profiler class
- **CUDAMetrics** - Hardware metrics snapshot
- **InferenceBatchMetrics** - Per-batch metrics
- **MemoryMetrics** - Memory usage snapshot
- **ProfilingSession** - Complete session report

### Key Methods

- `start_profiling()` - Begin profiling session
- `stop_profiling()` - End profiling session
- `profile_batch(batch_size)` - Context manager for batch profiling
- `profile_transfer(direction)` - Context manager for transfer profiling
- `get_realtime_metrics()` - Get current metrics
- `generate_report()` - Generate session report
- `export_report()` - Export to JSON
- `print_summary()` - Print human-readable summary

## Examples

See:
- `scripts/profile_gpu_inference.py` - Comprehensive profiling script
- `tests/unit/test_gpu_profiler.py` - Unit tests with usage examples
- `examples/profiling_demo.py` - Interactive profiling demo

## Performance Impact

### Overhead Comparison

| Configuration | Overhead | Use Case |
|--------------|----------|----------|
| Minimal (no torch.profiler) | <1% | Production profiling |
| NVML only | ~1% | Continuous monitoring |
| Full (torch.profiler + memory) | 5-10% | Deep analysis |
| TensorBoard export | 10-15% | One-time profiling |

### Recommendations

- **Production**: NVML only, no torch.profiler
- **Development**: Full profiling with TensorBoard
- **CI/CD**: Minimal configuration for regression detection
- **Research**: Full configuration with multiple runs

## References

- [NVIDIA Management Library (NVML) Documentation](https://developer.nvidia.com/nvidia-management-library-nvml)
- [PyTorch Profiler Tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [TensorBoard Profiler Plugin](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras)

## Support

For issues or questions:
1. Check this guide and API documentation
2. Review test cases in `tests/unit/test_gpu_profiler.py`
3. Open an issue with profiling report and system info
