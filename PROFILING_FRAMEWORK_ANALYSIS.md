# Profiling Framework Analysis

**Date**: 2025-10-17
**Purpose**: Comprehensive analysis of current benchmark/profiling framework before upgrade
**Status**: Phase 1 - Architecture Understanding

---

## Executive Summary

The current profiling framework is a sophisticated dual-layer (C++/Python) system with:
- **295+ metrics** across C++ side (timers, counters, gauges, hardware counters)
- **4 Python profilers** (GIL, Inference, Thread, Memory)
- **Multiple output formats** (JSON, Chrome Trace, Markdown)
- **Integration layer** (UnifiedProfiler coordinating both sides)
- **Campaign infrastructure** (run_profiling_suite.sh orchestrating benchmarks)

**Complexity Assessment**: HIGH
- Multiple layers of abstraction
- Thread-local storage patterns
- Cross-language integration (C++ ↔ Python)
- Multiple output generators

**Risk Assessment**: MEDIUM-HIGH
- Breaking changes could disable profiling entirely
- Tests depend on profiling output format
- Performance-critical paths use PROFILE_SCOPE macros

---

## Architecture Overview

### Layer 1: C++ Core Profiling (`cpp_extensions/mcts/profiling/`)

#### 1.1 Metric Definitions (`enhanced_metrics.hpp`)
```
ProfileMetric enum (295 metrics):
├── Selection Phase (0-19): PUCT, traversal, AVX2, cache hits/misses
├── Expansion Phase (20-39): State cloning, legal moves, NN requests, DLPack
├── Backup Phase (40-54): Path traversal, atomics, CAS retries
├── Virtual Loss (55-64): Apply/remove, contention, spin waits
├── Queue Operations (65-84): Submit, collect, batching, condition variables
├── Memory (85-104): Allocation, arenas, cache lines, false sharing
├── Synchronization (105-119): Mutex, atomic CAS, spinlocks, futex
├── Hardware Counters (120-144): CPU cycles, IPC, cache misses, branch misses
├── Thread Operations (145-164): Idle, active, coordinator, parallel regions
├── Pipeline (165-184): Full pipeline, stages, batching
├── GPU (185-204): Utilization, inference, memory transfers
└── Network (205-224): Tensor serialization, distributed training

MetricType:
- Timer: Duration measurement
- Counter: Event counting
- Gauge: Current value
- Histogram: Distribution
- HardwareCounter: CPU performance counters

MetricCategory:
- Selection, Expansion, Backup, VirtualLoss
- Queue, Memory, Synchronization
- Hardware, Thread, Pipeline
- GPU, Network
```

#### 1.2 Thread-Local Metrics (`thread_metrics.hpp`)
```cpp
ThreadMetrics class:
- Per-thread metric storage (avoids lock contention)
- Nested scope tracking (enter_scope/exit_scope)
- Aggregation for report generation

ThreadMetricsStorage:
- Singleton managing all thread metrics
- get_thread_metrics(): Thread-local access
- get_all_metrics(): Aggregation across threads
```

#### 1.3 Enhanced Profiler (`enhanced_profiler.hpp/cpp`)
```cpp
EnhancedProfiler (singleton):
├── Session Management
│   ├── start_session(name): Begin profiling
│   ├── stop_session(): End profiling
│   └── reset_metrics(): Clear all data
├── Recording
│   ├── PROFILE_SCOPE(metric): RAII timing
│   ├── PROFILE_COUNTER(metric, value): Event counting
│   └── PROFILE_GAUGE(metric, value): Value tracking
├── Export
│   ├── export_json(filename): Raw metric data
│   ├── export_chrome_trace(filename): Timeline visualization
│   ├── export_markdown(filename): Human-readable report
│   └── print_summary(): Console output
└── Configuration
    ├── set_enabled(bool): Enable/disable
    └── set_level(ProfileLevel): Basic/Full/Hardware
```

**RAII Pattern**: `ScopedProfiler` ensures automatic start/stop
```cpp
{
    PROFILE_SCOPE(SelectionTotal);  // Start timer on entry
    // ... selection logic ...
}  // Stop timer on exit (even with exceptions)
```

**Conditional Compilation**:
```cpp
#if PROFILE_LEVEL_VALUE > 0
    #define PROFILE_SCOPE(metric) /* real implementation */
#else
    #define PROFILE_SCOPE(metric) ((void)0)  // no-op
#endif
```

#### 1.4 Statistical Analyzer (`statistical_analyzer.hpp`)
```cpp
Features:
- Percentile calculation (p50, p90, p95, p99)
- Mean, median, stddev
- Min, max tracking
- Histogram generation
- Anomaly detection
```

---

### Layer 2: Python Profiling (`src/profiling/`)

#### 2.1 GIL Profiler (`gil_profiler.py`)
```python
GILProfiler:
├── Purpose: Track GIL contention and Python thread activity
├── Metrics:
│   ├── GIL hold time per thread
│   ├── GIL wait time per thread
│   ├── GIL switch frequency
│   └── GIL hotspots (where GIL is held longest)
└── Sampling: 1ms intervals (configurable)
```

#### 2.2 Inference Profiler (`inference_profiler.py`)
```python
InferencePipelineProfiler:
├── Purpose: Track neural network inference performance
├── Metrics:
│   ├── Batch formation time
│   ├── GPU inference time
│   ├── Result distribution time
│   ├── Batch size distribution
│   └── GPU utilization
└── Integration: Hooks into BatchInferenceCoordinator
```

#### 2.3 Thread Profiler (`thread_profiler.py`)
```python
ThreadCoordinatorProfiler:
├── Purpose: Track thread lifecycle and coordination
├── Metrics:
│   ├── Thread creation/destruction
│   ├── Thread idle time
│   ├── Thread active time
│   ├── Coordinator wait time
│   └── Thread pool efficiency
└── Integration: Monitors background threads
```

#### 2.4 Memory Profiler (`memory_profiler.py`)
```python
MemoryProfiler:
├── Purpose: Track memory usage and GC behavior
├── Metrics:
│   ├── RSS (Resident Set Size)
│   ├── VMS (Virtual Memory Size)
│   ├── Heap allocation
│   ├── GC collections (gen0, gen1, gen2)
│   └── Memory leaks (via tracemalloc)
└── Sampling: 1-second intervals
```

#### 2.5 Profiling Session (`profiling_session.py`)
```python
ProfilingSession:
├── Purpose: Unified interface coordinating all profilers
├── Lifecycle:
│   ├── __init__(config): Setup all profilers
│   ├── start(): Enable C++ + Python profiling
│   ├── stop(): Disable profiling, collect metrics
│   └── __enter__/__exit__: Context manager support
├── Integration:
│   ├── C++ EnhancedProfiler: Via mcts_py bindings
│   ├── Python profilers: Direct Python objects
│   └── Unified reports: Merged JSON/Markdown
└── Usage:
    with ProfilingSession() as session:
        # workload runs here
    metrics = session.get_all_metrics()
    session.save_reports()
```

---

### Layer 3: Campaign Infrastructure (`scripts/`)

#### 3.1 Profiling Suite Runner (`run_profiling_suite.sh`)
```bash
Sequential execution:
├── Step 0: Profiling Setup Validation
│   └── validate_profiling_setup.py
│       ├── Check PROFILE_LEVEL_VALUE=3
│       └── Verify <10% unaccounted time
├── Step 1: Wall-Clock Validation
│   └── wall_clock_validation.py
│       └── Ground-truth baseline (no profiling overhead)
├── Step 2: Profiling Campaign
│   └── profiling_campaign.py
│       ├── Parameter sweep (sims, threads, batch sizes)
│       ├── UnifiedProfiler per trial
│       └── JSON + Chrome Trace + Markdown per trial
├── Step 3: Results Analysis
│   └── analyze_profiling_results.py
│       ├── Statistical analysis across trials
│       ├── Bottleneck identification
│       └── Optimal configuration detection
└── Step 4: Completeness Check
    └── Verify <10% unaccounted time across all trials

Modes:
- --quick: 1-2 minutes (100-200 sims, 1-4 threads)
- --full: 15 minutes (100-1600 sims, 1-8 threads, 3 reps)
- --production: 40 minutes (2k-16k sims, 1-12 threads, 5 reps)
```

#### 3.2 Profiling Campaign (`profiling_campaign.py`)
```python
Features:
├── Parameter sweep configuration
├── Trial execution with retry logic
├── Per-trial profiling (C++ + Python)
├── Result aggregation
├── Campaign summary generation
└── CSV export for analysis

Output structure:
campaign_dir/
├── campaign_summary.json
├── results.csv
└── trial_001/
    ├── cpp_profiling.json
    ├── cpp_trace.json
    ├── cpp_report.md
    └── python_profiling.json
```

#### 3.3 Analysis Tools (`analyze_profiling_results.py`)
```python
Features:
├── Load campaign results (JSON)
├── Statistical analysis:
│   ├── Mean, median, stddev
│   ├── Min, max
│   ├── Percentiles (p50, p90, p95, p99)
│   └── Coefficient of variation
├── Bottleneck identification:
│   ├── Top operations by time
│   ├── Slowest metrics
│   └── Scaling efficiency analysis
├── Optimal configuration detection:
│   ├── Best throughput
│   └── Most consistent config
└── Output:
    ├── Console summary
    ├── Detailed markdown report
    └── Recommendations
```

---

## Integration Points

### C++ ↔ Python Bridge (`cpp_extensions/mcts/python_bindings.cpp`)
```cpp
py::class_<EnhancedProfiler>(m, "EnhancedProfiler")
    .def("start_session", &EnhancedProfiler::start_session)
    .def("stop_session", &EnhancedProfiler::stop_session)
    .def("reset_metrics", &EnhancedProfiler::reset_metrics)
    .def("export_json", &EnhancedProfiler::export_json)
    .def("export_chrome_trace", &EnhancedProfiler::export_chrome_trace)
    .def("export_markdown", &EnhancedProfiler::export_markdown)
    .def("print_summary", &EnhancedProfiler::print_summary)
    .def("set_enabled", &EnhancedProfiler::set_enabled)
    .def("is_enabled", &EnhancedProfiler::is_enabled);
```

### Usage Pattern
```python
import mcts_py
from src.profiling import ProfilingSession

# Python starts C++ profiling
profiler = mcts_py.EnhancedProfiler.instance()
profiler.set_enabled(True)
profiler.start_session("my_benchmark")

# Python profiling runs concurrently
with ProfilingSession() as py_session:
    # Run workload (both C++ and Python are profiling)
    runner.run_continuous(...)

# Stop C++ profiling
profiler.stop_session()
profiler.export_json("cpp_metrics.json")

# Get Python metrics
py_metrics = py_session.get_all_metrics()
```

---

## Current Pain Points

### 1. Complexity
- **295 metrics**: Hard to understand which are important
- **Multiple layers**: C++ → Python → Scripts (3 layers of abstraction)
- **Output fragmentation**: JSON, Chrome Trace, Markdown, CSV (4 formats)

### 2. Usability Issues
- **Manual coordination**: User must manually start C++ + Python profiling
- **Trial setup**: Complex campaign configuration (sims, threads, batch sizes)
- **Report interpretation**: Too much data, hard to find actionable insights

### 3. Performance Overhead
- **295 timers**: Even with PROFILE_LEVEL_VALUE checks, still overhead
- **Thread-local storage**: TLS lookups on every PROFILE_SCOPE
- **Chrome Trace**: Large JSON files (>100MB for long runs)

### 4. Maintenance Burden
- **Metric synchronization**: C++ enum must match Python constants
- **Output format changes**: Break downstream analysis tools
- **Test dependencies**: Tests parse profiling output (brittle)

### 5. Missing Features
- **Real-time monitoring**: No live dashboard
- **Flamegraph export**: Chrome Trace is verbose, want flamegraphs
- **Regression detection**: No automatic detection of performance regressions
- **Baseline comparison**: Hard to compare before/after optimizations

---

## Opportunities for Upgrade

### High Priority
1. **Simplify metric set**: Reduce from 295 to ~50 core metrics
2. **Auto-coordination**: Unified profiling start/stop (C++ + Python together)
3. **Better output**: Add flamegraph export, improve markdown reports
4. **Regression detection**: Automatic comparison with baseline

### Medium Priority
5. **Real-time monitoring**: Optional live dashboard (Prometheus/Grafana?)
6. **Hardware counter validation**: Verify PERF_EVENT_PARANOID settings
7. **Profile-guided optimization**: Use profiling data to suggest fixes
8. **Distributed profiling**: Support for multi-GPU/multi-node

### Low Priority
9. **Memory leak detection**: Enhanced tracemalloc integration
10. **GPU profiling**: Better CUDA/ROCm profiling integration

---

## Upgrade Strategy (Draft)

### Phase 1: Simplification (Weeks 1-2)
- Reduce metric count (295 → 50 core metrics)
- Consolidate output formats (keep JSON + Chrome Trace, improve Markdown)
- Simplify campaign configuration (presets: quick/full/production)

### Phase 2: Integration (Week 3)
- Unified profiling session (auto-coordinate C++ + Python)
- Single start/stop interface
- Merged reports (C++ + Python in one JSON)

### Phase 3: Analysis (Week 4)
- Baseline comparison
- Regression detection
- Flamegraph export
- Better markdown reports (executive summary, top bottlenecks, recommendations)

### Phase 4: Advanced (Week 5+)
- Real-time monitoring (optional)
- Hardware counter validation
- Profile-guided optimization suggestions

---

## Critical Questions Before Proceeding

1. **Metric reduction**: Which of the 295 metrics are actually used?
   - Need to analyze existing campaign results
   - Identify top 50 metrics by importance

2. **Backwards compatibility**: Do we need to maintain old output formats?
   - Existing tests may parse JSON
   - Existing analysis tools may depend on format

3. **Performance impact**: What's the current profiling overhead?
   - Measure with/without profiling
   - Ensure overhead < 5% in production

4. **User needs**: What do users actually need from profiling?
   - Ask: What decisions are made based on profiling data?
   - Focus on actionable insights, not raw data

---

## Next Steps

1. **Analyze metric usage**: Parse existing campaign results to see which metrics matter
2. **Survey current users**: What do they use profiling for?
3. **Prototype simplifications**: Create proof-of-concept for unified session
4. **Validate backwards compatibility**: Ensure tests still pass

---

## Files to Review in Detail

### Critical (Must understand before changes):
- `cpp_extensions/mcts/profiling/enhanced_profiler.{hpp,cpp}`
- `cpp_extensions/mcts/profiling/thread_metrics.hpp`
- `src/profiling/profiling_session.py`
- `scripts/run_profiling_suite.sh`
- `scripts/profiling_campaign.py`

### Important (Should understand):
- `cpp_extensions/mcts/profiling/enhanced_metrics.hpp` (all 295 metrics)
- `src/profiling/*.py` (all Python profilers)
- `scripts/analyze_profiling_results.py`

### Nice to have:
- `tests/unit/test_*profiling*.py` (test dependencies)
- `docs/profiling/` (documentation)

---

## Checkpoint Status

✅ **Phase 1 Complete**: Architecture Understanding
- Mapped C++ profiling layer (EnhancedProfiler, 295 metrics)
- Mapped Python profiling layer (4 profilers)
- Mapped campaign infrastructure (run_profiling_suite.sh)
- Identified integration points (C++ ↔ Python)
- Documented pain points and opportunities

⏸️ **Phase 2 Pending**: Metric Usage Analysis
- Parse existing campaign results
- Identify which metrics are actually used
- Determine core metric set (~50 metrics)

⏸️ **Phase 3 Pending**: Upgrade Plan
- Design unified profiling session
- Design simplified output formats
- Design baseline comparison
- Design regression detection

---

## Recommendation

**PAUSE AND CLARIFY** before proceeding:

The framework is complex and production-critical. Before making changes:
1. What specific problems are we trying to solve?
2. What are the success criteria for the upgrade?
3. What's the acceptable risk level for breakage?
4. Do we have rollback plan if things break?

**Suggest**:
- Start with **non-breaking additions** (e.g., add flamegraph export)
- Then **opt-in improvements** (e.g., unified session as alternative to manual)
- Finally **breaking changes** (e.g., metric reduction) with migration guide

This minimizes risk while delivering value incrementally.
