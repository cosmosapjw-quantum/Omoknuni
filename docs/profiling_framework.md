# Python Profiling Framework for MCTS Coordination

## Overview

Comprehensive profiling system for analyzing Python coordination overhead in the MCTS engine. Designed to identify bottlenecks in:

- **GIL contention**: Time wasted waiting for the Global Interpreter Lock
- **Inference pipeline**: Batching efficiency, queue wait times, GPU transfer overhead
- **Thread coordination**: ThreadPoolExecutor overhead, Future creation/collection costs
- **Memory management**: Allocation patterns, garbage collection pauses, memory leaks

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ProfilingSession                         │
│  (Unified coordinator for all profilers)                    │
└──────────────┬──────────────────────────────────────────────┘
               │
       ┌───────┴───────┬──────────┬──────────┬──────────┐
       │               │          │          │          │
┌──────▼─────┐  ┌──────▼────┐ ┌──▼───────┐ ┌▼─────────┐ ┌▼──────────┐
│    GIL     │  │ Inference │ │ Thread   │ │ Memory   │ │    C++    │
│  Profiler  │  │ Profiler  │ │ Profiler │ │ Profiler │ │ Instr.    │
└────────────┘  └───────────┘ └──────────┘ └──────────┘ └───────────┘
      │               │             │            │             │
      │               │             │            │             │
   GIL wait      Queue depth   Future        Memory        C++ timing
   hotspots      Batch eff.    latency       growth        counters
   Contention    DLPack use    Thread util   GC impact     Selection
                                                           Expansion
```

## Components

### 1. GIL Profiler (`gil_profiler.py`)

Tracks Python GIL acquisition, release, and contention.

**Key Metrics:**
- Time with GIL vs without GIL per thread
- GIL wait times and hotspots
- Contention events between threads
- GIL efficiency (% time in nogil code)

**Usage:**
```python
from src.profiling import GILProfiler

profiler = GILProfiler()
profiler.start()

# Mark GIL release/acquire manually
profiler.mark_gil_release("entering_cpp")
# ... C++ code ...
profiler.mark_gil_acquire("returning_from_cpp")

# Or use sections
with profiler.section("mcts_search"):
    # ... code ...

profiler.stop()
metrics = profiler.get_metrics()
```

**Output:**
```json
{
  "summary": {
    "gil_utilization": 45.2,
    "gil_efficiency": 68.3,
    "avg_wait_time_per_thread": 0.0023,
    "total_contention_events": 127
  },
  "top_wait_hotspots": [
    {"location": "mcts.py:238", "total_wait_time_ms": 15.3},
    {"location": "inference.py:445", "total_wait_time_ms": 8.7}
  ]
}
```

### 2. Inference Pipeline Profiler (`inference_profiler.py`)

Profiles neural network inference pipeline end-to-end.

**Key Metrics:**
- Request latency breakdown (queue → batch → GPU → results)
- Batch collection efficiency
- DLPack zero-copy effectiveness
- Queue depth and wait times

**Usage:**
```python
from src.profiling import InferencePipelineProfiler

profiler = InferencePipelineProfiler()
profiler.start()

# Track complete request
with profiler.track_request("req_123", thread_id=1, batch_size=32):
    # Track individual stages
    with profiler.track_stage("req_123", "queue_wait"):
        # ... waiting in queue ...

    with profiler.track_stage("req_123", "inference"):
        # ... GPU inference ...

# Record batch metrics
profiler.record_batch(
    batch_id="batch_456",
    batch_size=64,
    collection_time_us=500,
    processing_time_us=2000,
    distribution_time_us=100,
    target_batch_size=64
)

profiler.stop()
metrics = profiler.get_metrics()
```

**Output:**
```json
{
  "summary": {
    "avg_latency_us": 2350.5,
    "p99_latency_us": 4521.2,
    "avg_batch_size": 58.3,
    "dlpack_usage_rate": 0.98,
    "batch_efficiency": 0.91
  },
  "stage_breakdown": {
    "queue_wait": {"avg_us": 523.2, "percentage": 22.3},
    "inference": {"avg_us": 1450.1, "percentage": 61.7},
    "result_distribution": {"avg_us": 125.3, "percentage": 5.3}
  }
}
```

### 3. Thread Coordinator Profiler (`thread_profiler.py`)

Profiles thread coordination overhead (ThreadPoolExecutor, Futures).

**Key Metrics:**
- Future creation and result collection overhead
- Thread pool utilization
- Task submission and execution latency
- Thread lifecycle costs

**Usage:**
```python
from src.profiling import ThreadCoordinatorProfiler

profiler = ThreadCoordinatorProfiler()
profiler.start()

# Track future lifecycle
with profiler.track_future("task_123", thread_id=1):
    with profiler.track_future_stage("task_123", "submission"):
        future = executor.submit(work_fn)

    with profiler.track_future_stage("task_123", "execution"):
        result = future.result()

# Record pool state
profiler.record_pool_state(
    pool_size=8,
    active_threads=6,
    queued_tasks=12,
    max_queue_size=100
)

profiler.stop()
metrics = profiler.get_metrics()
```

**Output:**
```json
{
  "summary": {
    "thread_utilization": 75.3,
    "avg_future_latency_us": 1523.4,
    "success_rate": 99.2,
    "futures_per_second": 342.1
  },
  "pool_summary": {
    "avg_thread_utilization": 75.3,
    "avg_queue_utilization": 12.4
  }
}
```

### 4. Memory Profiler (`memory_profiler.py`)

Profiles memory allocation and garbage collection.

**Key Metrics:**
- Memory growth over time
- Garbage collection frequency and duration
- Allocation hotspots (via tracemalloc)
- Memory leak detection

**Usage:**
```python
from src.profiling import MemoryProfiler

profiler = MemoryProfiler(
    snapshot_interval=1.0,
    enable_tracemalloc=True,
    track_gc_events=True
)
profiler.start()

# Track specific sections
with profiler.track_section("inference_batch"):
    # ... allocate memory ...

# Force GC and measure
gc_stats = profiler.force_gc()

profiler.stop()
metrics = profiler.get_metrics()
```

**Output:**
```json
{
  "summary": {
    "peak_memory_mb": 1245.3,
    "memory_growth_mb": 23.5,
    "gc_events_per_second": 0.15,
    "total_objects_collected": 15234
  },
  "section_analysis": {
    "inference_batch": {
      "num_invocations": 150,
      "avg_memory_delta_mb": 2.3,
      "max_memory_delta_mb": 8.7
    }
  },
  "leak_candidates": []
}
```

### 5. Profiling Session (`profiling_session.py`)

Unified session manager that coordinates all profilers.

**Usage:**
```python
from src.profiling import ProfilingSession, ProfilerConfig

# Configure profiling
config = ProfilerConfig(
    enable_gil_profiling=True,
    enable_inference_profiling=True,
    enable_thread_profiling=True,
    enable_memory_profiling=True,
    enable_cpp_instrumentation=True,
    auto_save_reports=True,
    report_directory="profiling_reports"
)

# Use as context manager
with ProfilingSession(config) as session:
    # ... run workload ...

    # Access individual profilers
    if session.gil:
        with session.gil.section("search"):
            # ... code ...

# Get all metrics
metrics = session.get_all_metrics()

# Reports automatically saved (JSON + HTML + Flamegraph)
```

## Integration with C++ Instrumentation

The profiling framework integrates with C++ instrumentation via `mcts_py`:

```python
# C++ instrumentation is automatically enabled
with ProfilingSession(config) as session:
    # ... MCTS operations ...

metrics = session.get_all_metrics()
cpp_metrics = metrics['cpp_instrumentation']

# Example output:
# {
#   "selection": {"call_count": 15234, "total_elapsed_ns": 523000000},
#   "expansion": {"call_count": 1523, "total_elapsed_ns": 234000000},
#   "queue_submit": {"call_count": 234, "total_elapsed_ns": 12000000}
# }
```

## Report Generation

Three report formats are generated:

### 1. JSON Report
Machine-readable structured data with all metrics.

### 2. HTML Report
Interactive dashboard with:
- Summary cards with color-coded alerts
- Plotly charts for time-series data
- Tables for top hotspots
- Stage breakdown visualizations

### 3. Flamegraph (SVG)
Call stack visualization (requires py-spy or similar).

## Performance Impact

The profiling framework is designed for minimal overhead:

| Profiler | Overhead (typical) | Notes |
|----------|-------------------|-------|
| GIL Profiler | <2% | Sampling-based (1ms intervals) |
| Inference Profiler | <1% | Event-based, no hot loops |
| Thread Profiler | <1% | Event-based |
| Memory Profiler | 5-10% | When tracemalloc enabled |
| C++ Instrumentation | <1% | Lock-free atomics |

**Recommendation:** Enable all profilers during development/debugging. For production profiling, disable tracemalloc and reduce GIL sampling rate.

## Common Use Cases

### 1. Diagnose Low Throughput

```python
config = ProfilerConfig(
    enable_gil_profiling=True,
    enable_inference_profiling=True,
    enable_cpp_instrumentation=True
)

with ProfilingSession(config) as session:
    mcts.search(root_state, 800)

metrics = session.get_all_metrics()

# Check GIL efficiency
if metrics['gil_metrics']['summary']['gil_efficiency'] < 70:
    print("WARNING: Spending too much time with GIL!")
    print("Hotspots:", metrics['gil_metrics']['top_wait_hotspots'])

# Check inference batching
if metrics['inference_metrics']['summary']['avg_batch_size'] < 32:
    print("WARNING: Poor batch efficiency!")
```

### 2. Optimize Batch Collection

```python
config = ProfilerConfig(enable_inference_profiling=True)

with ProfilingSession(config) as session:
    # ... run searches ...

# Analyze stage breakdown
stages = metrics['inference_metrics']['stage_breakdown']
for stage, stats in sorted(stages.items(), key=lambda x: x[1]['percentage'], reverse=True):
    print(f"{stage}: {stats['percentage']:.1f}%")
```

### 3. Detect Memory Leaks

```python
config = ProfilerConfig(
    enable_memory_profiling=True,
    memory_enable_tracemalloc=True,
    memory_snapshot_interval=0.5
)

with ProfilingSession(config) as session:
    # Run long workload
    for i in range(1000):
        mcts.search(root_state, 100)

leak_candidates = metrics['memory_metrics']['leak_candidates']
if leak_candidates:
    print("WARNING: Potential memory leaks detected!")
    for leak in leak_candidates:
        print(f"  Growth: {leak['growth_mb']:.1f} MB over {leak['window_duration_sec']:.1f}s")
```

### 4. Analyze Thread Contention

```python
config = ProfilerConfig(
    enable_thread_profiling=True,
    thread_track_lifecycle=True
)

with ProfilingSession(config) as session:
    # ... parallel searches ...

if metrics['thread_metrics']['pool_summary']['avg_thread_utilization'] < 50:
    print("WARNING: Poor thread utilization!")
    print(f"Thread churn rate: {metrics['thread_metrics']['summary']['thread_churn_rate']:.2f}/sec")
```

## Best Practices

1. **Start Simple**: Begin with `ProfilingSession` context manager, then drill down into specific profilers

2. **Profile Incrementally**: Enable one profiler at a time to isolate overhead and identify bottlenecks

3. **Use Sections**: Wrap critical code sections for targeted profiling:
   ```python
   with session.gil.section("critical_path"):
       # ... code ...
   ```

4. **Compare Baselines**: Save baseline metrics and compare after optimizations:
   ```python
   baseline = session.get_all_metrics()
   # ... optimize ...
   optimized = session.get_all_metrics()
   ```

5. **Disable in Production**: Set `PROFILING_ENABLED=false` in production config

6. **Review Reports**: HTML reports provide interactive visualizations - use them!

## Troubleshooting

### High GIL Contention
- **Symptom**: `gil_efficiency < 70%`, high `avg_wait_time_per_thread`
- **Solution**: Move more code to C++/Cython with `nogil`, reduce GIL-holding sections

### Poor Batch Efficiency
- **Symptom**: `avg_batch_size < 32`, `batch_efficiency < 0.8`
- **Solution**: Increase `async_timeout_ms`, adjust batch collection strategy

### High Memory Growth
- **Symptom**: `memory_growth_mb > 500`, leak candidates detected
- **Solution**: Check for circular references, review object lifecycles, force GC

### Low Thread Utilization
- **Symptom**: `thread_utilization < 60%`
- **Solution**: Reduce thread count, check for serialization bottlenecks

## API Reference

See individual module docstrings for complete API documentation:
- `src/profiling/gil_profiler.py`
- `src/profiling/inference_profiler.py`
- `src/profiling/thread_profiler.py`
- `src/profiling/memory_profiler.py`
- `src/profiling/profiling_session.py`

## Examples

Complete examples are available in:
- `examples/profiling_demo.py` - Basic profiling demonstrations
- `examples/profile_mcts_search.py` - Real MCTS search profiling

Run examples:
```bash
python examples/profiling_demo.py
python examples/profile_mcts_search.py --simulations 800 --threads 4
```

## Future Enhancements

Potential improvements for future iterations:

1. **Real-time Dashboard**: Live web dashboard for monitoring production systems
2. **Flamegraph Integration**: Direct integration with py-spy/austin for call stack profiling
3. **Distributed Profiling**: Profile multi-process/multi-machine MCTS deployments
4. **ML-based Analysis**: Automatic anomaly detection and optimization suggestions
5. **Integration with Prometheus**: Export metrics to Prometheus for long-term monitoring

## License

See project LICENSE file.
