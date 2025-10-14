# Benchmark Results - Post-OpenMP Fix

**Date**: 2025-10-14
**Session**: After T002-T006 OpenMP implementation
**Configuration**: OMP_NUM_THREADS=12, 8 MCTS threads, Gomoku 15×15

---

## Summary

**Status**: ⚠️ **CRITICAL ISSUE IDENTIFIED**

OpenMP is active and working (0.06ms feature extraction confirmed), but overall throughput is still limited due to **BATCHING FAILURE**. The inference queue is not properly batching requests, resulting in batch size = 1.0 instead of target 32-64.

---

## Throughput Results

| Threads | Throughput (sims/sec) | vs Single-Thread | Efficiency |
|---------|----------------------|------------------|------------|
| 1 | 1,398 | 1.00× | 100% |
| 2 | 2,249 | 1.61× | 80.5% |
| 4 | 2,899 | 2.07× | 51.8% |
| **8** | **2,923** | **2.09×** | **26.1%** |

**Thread Scaling Analysis**:
- 1→8 threads: Only 2.09× speedup (expected ~6-7× for 8 threads)
- **Thread efficiency @ 8 threads: 26.1%** (target 75%)
- **PROBLEM**: Plateau at 4 threads, no additional benefit from 8 threads

---

## Critical Issues Identified

### 🔴 Issue 1: Batch Size = 1.0 (TARGET: 32-64)

**Observation**:
```
Average batch size: 1.0
Min batch size: 1
Max batch size: 1
Total inference calls: 801 (for 800 simulations)
```

**Impact**:
- GPU is processing 1 state at a time instead of 32-64
- GPU utilization likely <10% (severely underutilized)
- Throughput capped by serial inference calls

**Root Cause**: Inference queue is not accumulating requests before submitting to GPU

**Expected Behavior**:
- Requests should accumulate to batch_size (32-64)
- OR submit after timeout (0.5-1.0ms)
- Actual: Submitting immediately (batch size 1)

### 🟡 Issue 2: Thread Scaling Plateau

**Observation**:
- 4 threads: 2,899 sims/sec
- 8 threads: 2,923 sims/sec (+0.8% improvement)

**Analysis**:
- Adding more threads provides minimal benefit
- Likely cause: Threads are blocked waiting for inference (serial bottleneck)
- With batch size = 1, GPU becomes serial bottleneck

### ✅ Issue 3: OpenMP Feature Extraction (RESOLVED)

**Validation**:
- Feature extraction: 0.06ms per batch-64 ✅
- OpenMP threads: 12 active ✅
- This is NOT the bottleneck anymore

---

## Comparison to Baselines

| Metric | Baseline (Spec 003) | Regression | Current | Target | Status |
|--------|---------------------|-----------|---------|--------|--------|
| Throughput (8T) | 3,831 sims/sec | 2,147 sims/sec | **2,923 sims/sec** | ≥8,000 | ❌ |
| Batch Size | ~50-60 | Unknown | **1.0** | 32-64 | ❌ |
| Thread Efficiency | ~70-80% | Unknown | **26.1%** | ≥75% | ❌ |
| Feature Extraction | ~1ms | 7.5ms | **0.06ms** | <1ms | ✅ |

**Progress**:
- vs Regression (2,147): +36% improvement (2,147 → 2,923)
- vs Baseline (3,831): -24% deficit (2,923 / 3,831 = 76%)
- vs Target (8,000): -63% deficit (2,923 / 8,000 = 37%)

---

## Root Cause Analysis

### Why is batch size = 1?

**Hypothesis 1**: Async inference queue not implemented
- Check: Does `AsyncInferenceQueue` exist in codebase?
- Status: Needs investigation

**Hypothesis 2**: Timeout = 0 (immediate submission)
- Check: What is current batch timeout configuration?
- Status: Needs investigation

**Hypothesis 3**: Only 1 thread submitting at a time
- If only 1 thread submits to queue while others wait, batch accumulation is impossible
- Status: Likely culprit

**Hypothesis 4**: Queue → GPU path is synchronous
- Each submission waits for GPU result before next submission
- Would explain batch size = 1 (no accumulation possible)
- Status: Very likely

---

## Inference System Architecture (Current vs Target)

### Current (Broken Batching):
```
Thread 1 → Submit(state) → Wait for result → GPU processes 1 state
Thread 2 → Submit(state) → Wait for result → GPU processes 1 state
Thread 3 → Submit(state) → Wait for result → GPU processes 1 state
...
```
**Result**: Serial processing, batch size = 1

### Target (Proper Batching):
```
Thread 1 → Submit(state) ┐
Thread 2 → Submit(state) ├→ Queue accumulates → GPU processes 32-64 states
Thread 3 → Submit(state) ┘      ↓
All threads ← Results distributed
```
**Result**: Parallel accumulation, batch size = 32-64

---

## Diagnostic Commands

### Check if AsyncInferenceQueue exists:
```bash
find cpp_extensions -name "*async*" -o -name "*queue*" -o -name "*batch*"
grep -r "AsyncInferenceQueue" cpp_extensions/
```

### Check batch configuration:
```bash
grep -r "batch_size\|batch_timeout" src/
```

### Check if coordination is implemented:
```bash
grep -r "BatchInferenceCoordinator\|condition_variable" cpp_extensions/
```

---

## Next Actions (CRITICAL PATH UPDATED)

### ❌ Phase 1 CPU Optimizations (T007-T013) - DEFER

**Reason**: State pooling, condition variables, and thread-local arenas won't fix the batching problem. We need to fix the inference architecture first.

### ✅ Immediate Action: Fix Batching System

**Priority**: CRITICAL - This is the primary throughput bottleneck

**Tasks Required**:
1. **Investigate current inference architecture**
   - Check if AsyncInferenceQueue is implemented
   - Check if BatchInferenceCoordinator exists
   - Identify why batch size = 1

2. **Implement missing components** (if needed)
   - AsyncInferenceQueue with accumulation
   - Batch timeout mechanism (0.5-1.0ms)
   - Thread coordination for request accumulation

3. **Validate batching works**
   - Target: Avg batch size ≥32
   - Expected throughput improvement: 5-10× (from eliminating serial bottleneck)

### 📋 Revised Critical Path

```
Current Status:
✅ T002-T006: OpenMP implementation (COMPLETE)
⏸️  T007-T013: CPU optimizations (DEFERRED)

New Critical Path:
🔴 BLOCKER: Fix inference batching (batch size 1 → 32-64)
   └─> Expected: 2,923 → 6,000-10,000 sims/sec

After batching fix:
✅ T014: Re-run benchmark to validate
   └─> If ≥8k: DONE
   └─> If <8k: Resume Phase 1 (T007-T013)
```

---

## Acceptance Criteria Status

From spec.md Section 4.1:

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Throughput (8 threads) | ≥8,000 sims/sec | 2,923 sims/sec | ❌ FAIL |
| GPU Utilization | 80-95% | Unknown (likely <10%) | ❌ FAIL |
| Avg Batch Size | 32-64 | 1.0 | ❌ FAIL |
| Thread Efficiency | ≥75% @ 8T | 26.1% @ 8T | ❌ FAIL |
| Feature Extraction | <1ms | 0.06ms | ✅ PASS |

**Overall**: ❌ **FAIL** - Batching system is broken

---

## Decision Required

### Per spec.md Section 4.1 and /speckit.implement instructions:

> "If a KPI fails, stop and open a 'Needs Decision' note inside tasks.md"

**KPI FAILURE**:
- Throughput: 2,923 / 8,000 = 36.5% of target ❌
- Batch size: 1.0 / 32 = 3.1% of minimum ❌
- Thread efficiency: 26.1% / 75% = 34.8% of target ❌

**STOP HERE** - Opening "Needs Decision" note in tasks.md

**Question for Decision**:
1. Should we investigate the batching architecture now?
2. Should we implement AsyncInferenceQueue + BatchCoordinator (from plan.md Phase 2)?
3. Or should we continue with Phase 1 CPU optimizations and revisit batching later?

**Recommendation**: Fix batching FIRST - it's a 5-10× multiplier, much larger than any CPU optimization.

---

## Technical Debt Identified

1. **Inference batching not working** - CRITICAL
2. **Thread coordination inefficient** - MAJOR
3. **GPU severely underutilized** - MAJOR
4. **No batch accumulation mechanism** - CRITICAL

---

## Files to Investigate

Priority order:
1. `cpp_extensions/mcts/async_inference_queue.cpp` - Does it exist?
2. `cpp_extensions/mcts/batch_inference_coordinator.cpp` - Is it implemented?
3. `src/neural/inference_worker.py` - How does GPU inference get called?
4. `cpp_extensions/mcts/continuous_simulation_runner.cpp` - How are requests submitted?

---

## End of Benchmark Report

**Status**: ⚠️ CRITICAL BATCHING ISSUE - DECISION REQUIRED
**Next Step**: Investigate batching architecture OR proceed per user decision
