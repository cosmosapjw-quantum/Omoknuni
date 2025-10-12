# T016/T017 Benchmark Results and Analysis

**Date**: 2025-10-13
**Session**: Comprehensive performance benchmarking and baseline investigation
**Status**: ⚠️ CRITICAL PERFORMANCE REGRESSION IDENTIFIED

## Executive Summary

After implementing all Phase 1 and Phase 2 optimizations (T001-T014, including T006c condition variables and T008f FP16 mixed precision), performance is **significantly below expectations**:

- **Current Best**: 2,235 sims/sec (4 threads)
- **Baseline (Spec 003)**: 3,831 sims/sec (8 threads)
- **Regression**: -42% from baseline
- **Target**: 25,000 sims/sec
- **Achievement**: 8.9% of target

## Benchmark Results (T016)

### Configuration

All optimizations confirmed ENABLED:
- ✅ T006c_condition_variables
- ✅ T008f_fp16_mixed_precision
- ✅ T007_dlpack_zero_copy
- ✅ T009_thread_local_arenas

### Thread Scaling Analysis

| Threads | Throughput | GPU Util | CPU Util | Parallel Efficiency |
|---------|-----------|----------|----------|---------------------|
| 1       | 1,364 sims/sec | 48% | 2.5% | 100% (baseline) |
| 2       | 1,241 sims/sec | 50% | 2.5% | **45.5%** ⚠️ |
| 4       | **2,235 sims/sec** | 56% | 2.9% | **41.0%** ⚠️ |
| 8       | 1,450 sims/sec | 64% | 2.1% | **13.3%** 🔴 |
| 12      | 1,025 sims/sec | 58% | 3.3% | **6.3%** 🔴 |

**Critical Finding**: Parallel efficiency collapses dramatically with additional threads:
- Adding 2nd thread: -45% efficiency (should be ~90%)
- 4 threads: Only 41% of linear scaling
- 8 threads: Only 13% of linear scaling
- 12 threads: **6.3%** efficiency (catastrophic)

### Profiler Analysis (comprehensive_mcts_profiler.py)

**Single-thread baseline**:
- 1 thread: 1,148 sims/sec (clean baseline)
- C++ throughput: 1,904 sims/sec

**Multi-thread degradation**:
- 2 threads: 478 sims/sec (42% efficient, **HUGE** drop)
- 4 threads: 488 sims/sec (42% efficient)
- 8 threads: 468 sims/sec (41% efficient)
- 12 threads: 373 sims/sec (32% efficient)

**Profiler verdict**: "Python: GIL contention (100.0% impact)"

## Baseline Investigation (T017)

### Spec 003 Configuration (Achieved 3,831 sims/sec)

```yaml
game: gomoku
model: 23.8M parameters (ResNet-20, 256 channels)
precision: FP32
search:
  batch_size: 32
  timeout_ms: 1.0
  threads: 8
  simulations: 1600
hardware:
  cpu: AMD Ryzen 9 5900X
  gpu: NVIDIA RTX 3060 Ti
```

### Current Configuration

```yaml
game: gomoku
model: 10.1M parameters (ResNet-20, 256 channels, REDUCED)
precision: FP16 (via autocast)
search:
  batch_size: 32  # same
  timeout_ms: 1.0 # same (default 2.0 in code)
  threads: 4-8    # varying
  simulations: 1600
hardware: same
```

**Key Differences**:
1. Model size reduced from 23.8M → 10.1M parameters
2. FP16 enabled (should be 1.5-2× faster)
3. Condition variables replacing polling (should be 1.3-1.5× faster)
4. DLPack zero-copy (should be 2-3× faster for tensor prep)

**Expected**: ~18-36k sims/sec (6-12× baseline)
**Actual**: 2,235 sims/sec (0.58× baseline)

## Root Cause Analysis

### Primary Bottleneck: Severe Thread Contention

The data conclusively shows that **adding threads makes performance WORSE**:

1. **Thread Coordination Overhead**
   - 2 threads: Only 45% efficiency vs single-thread baseline
   - This indicates massive contention on shared resources
   - GIL profiler shows "100% GIL contention" but this seems false (C++ should release GIL)

2. **Possible Causes**:
   a. **Async queue blocking** - Even with condition variables, threads may be blocking excessively
   b. **Python ThreadPoolExecutor overhead** - Threads spawned per-search create overhead
   c. **Virtual loss coordination** - Atomic operations causing cache line bouncing
   d. **Tree contention** - Multiple threads fighting for same nodes (should be prevented by busy-edge masking)
   e. **Inference coordinator** - T011 persistent coordinator may have issues
   f. **Configuration drift** - Some setting changed between baseline and now

3. **GPU Underutilization**
   - Best GPU util: 64% (at 8 threads, worst throughput!)
   - Target: 80-92%
   - GPU is starved because CPU threads aren't feeding it fast enough

### Secondary Issues

1. **Batch Size**: Averaging 48 (good, close to target 48-64)
2. **Memory**: 100MB (excellent, well under 1GB target)
3. **Model Size**: Reduced to 10.1M but performance worse (suggests CPU bottleneck, not GPU)

## FP16 Validation

FP16 IS enabled (confirmed in benchmark output):
```
/home/cosmosapjw/omoknuni/src/core/dlpack_inference_bridge.py:393: FutureWarning:
  `torch.cuda.amp.autocast(args...)` is deprecated.
  Please use `torch.amp.autocast('cuda', args...)` instead.
  with torch.cuda.amp.autocast():
```

**Action Required**: Update to new API format, but FP16 is confirmed working.

## Critical Findings

### ✅ What's Working

1. **Optimizations are enabled**: All flags confirmed active
2. **Code paths correct**: DLPackInferenceBridge detected and used
3. **Batch sizes optimal**: ~48 average (target 32-64)
4. **Memory efficient**: <100MB (target <1GB)

### 🔴 What's Broken

1. **Thread scaling**: Completely broken, efficiency < 50% with 2+ threads
2. **Overall throughput**: 42% slower than baseline despite all optimizations
3. **GPU utilization**: Only 64% vs 80-92% target
4. **Performance delta**: -20.5k sims/sec from target (8.9% of goal)

## Recommendations

### Immediate Actions (Priority Order)

1. **Profile thread contention in C++** (CRITICAL)
   - Use perf/vtune to identify actual contention points
   - Check atomic operations, mutex usage, cache line bouncing
   - Instrument AsyncInferenceQueue with detailed timing

2. **Single-thread optimization first**
   - Current: 1,364 sims/sec (single thread)
   - Target: At least 5,000-8,000 sims/sec single-thread
   - Then fix parallelization

3. **Compare against Spec 003 commit**
   - Checkout e933bc5 (Spec 003 complete)
   - Run identical benchmark
   - Git bisect to find regression commit

4. **Reduce thread coordinator overhead**
   - ThreadPoolExecutor may be creating threads per-search
   - Consider persistent worker threads (T011 addressed this but may be buggy)

5. **Profile GPU timeline**
   - Use nsys/nvprof to see actual GPU kernel launches
   - Confirm FP16 kernels are running (not falling back to FP32)
   - Check H2D/D2H transfer patterns

### Configuration Experiments

Test single-thread performance with:
```bash
# Baseline: Current implementation
python scripts/benchmark_throughput.py --threads 1 --simulations 10000

# Disable condition variables (revert T006c)
# Disable FP16 (revert T008f)
# Test each optimization in isolation

# Compare against Spec 003
git checkout e933bc5
python scripts/benchmark_throughput.py --threads 8 --simulations 10000
```

### Code Investigation

Priority files to audit:
1. `cpp_extensions/mcts/continuous_simulation_runner.cpp` - Thread loop
2. `cpp_extensions/mcts/async_inference_queue.cpp` - Condition variable implementation
3. `cpp_extensions/mcts/batch_inference_coordinator.cpp` - Coordinator lifecycle
4. `src/core/mcts.py` - Python ThreadPoolExecutor usage (line 317-322)

## Next Steps

1. ✅ T016 complete: Benchmarks run, results documented
2. ✅ T017 complete: Baseline configuration identified
3. 🔴 **NEW: T017b - Root cause thread contention** (estimate: 2-3 days)
4. 🔴 **NEW: T017c - Compare against Spec 003 commit** (estimate: 1 day)
5. T018-T020: Defer until regression fixed

## Conclusion

Despite implementing all planned optimizations (T001-T014), performance has **regressed** by 42% from baseline. The optimizations themselves appear to be enabled and working, but a **critical thread contention issue** is preventing scaling beyond 1-2 threads.

**Immediate priority**: Identify and fix the thread contention bottleneck. Until this is resolved, parameter tuning (T018-T020) will have minimal impact.

**Path forward**: Focus on single-thread optimization first (target: 5-8k sims/sec), then fix parallelization to achieve >90% efficiency with 4-8 threads. Only then can we realistically target 25k+ sims/sec.

---

**Artifacts**:
- Benchmark results: `benchmark_results.json`
- Profiler session: `profiling_reports/session_20251013_070415/`
- This report: `docs/performance/T016_T017_benchmark_results.md`
