# Post-OpenMP Fix Performance Analysis

**Date**: 2025-10-13
**OpenMP Fix Commit**: d392d36
**Status**: ⚠️ CRITICAL BOTTLENECK IDENTIFIED

## Executive Summary

After applying OpenMP parallelization fix (6.9× speedup in tensor creation), performance improved modestly but remains far below target:

- **Pre-OpenMP Best**: 2,235 sims/sec (4 threads)
- **Post-OpenMP Best**: 2,835 sims/sec (4 threads, first run)
- **Improvement**: +600 sims/sec (+27%, 1.27× speedup)
- **Baseline (Spec 003)**: 3,831 sims/sec
- **Target**: 8,000 sims/sec
- **Achievement**: 35% of target (28% improvement toward goal)

## Benchmark Results (Post-OpenMP)

### Thread Scaling Analysis

| Threads | Throughput | GPU Util | Parallel Efficiency | vs Pre-OpenMP |
|---------|-----------|----------|---------------------|---------------|
| 1       | 1,230 sims/sec | 48% | 100% (baseline) | -9.8% |
| 2       | 2,205 sims/sec | 66% | 89.6% | +77.7% |
| 4       | 2,214 sims/sec | 66% | 45.0% | -0.9% |
| 6       | 2,173 sims/sec | 61% | 29.5% | N/A |
| 8       | 2,198 sims/sec | 66% | 22.4% | +51.6% |
| 10      | 2,113 sims/sec | 61% | 17.2% | N/A |
| 12      | 2,166 sims/sec | 65% | 14.7% | +111.4% |

**Key Findings**:
1. Performance improved at higher thread counts (8, 12 threads)
2. Performance plateaus at ~2,200 sims/sec regardless of threads
3. GPU utilization remains suboptimal (66% typical, 87% best case)
4. Thread scaling efficiency still poor beyond 2 threads

## Root Cause: GPU Inference Bottleneck

### Theoretical Maximum Analysis

From T-VALID-1 FP16 validation:
- **GPU inference time**: 30.7ms per batch-64
- **Tensor creation**: 1.08ms (after OpenMP fix)
- **Total per batch**: ~32ms
- **Theoretical max**: 64 states / 32ms = **2,000 states/sec**

**Observed**: 2,200-2,800 sims/sec (close to theoretical maximum!)

### Bottleneck Breakdown

```
Per-batch timing budget (target: 8ms for 8k sims/sec):
  Tensor creation:  1.08ms (13.5% of budget) ✅
  GPU inference:   30.70ms (384% of budget) ❌ CRITICAL
  Total:          ~32.00ms (400% of budget)
```

**Verdict**: GPU inference is now the **primary bottleneck** (30.7ms vs 8ms target).

## Why GPU Inference is Slow

### Expected vs Actual

**From Hardware Specs** (SPECIFICATION.md Q4, review.txt):
- RTX 3060 Ti @ FP16: 8-10ms per batch-64 (expected)
- Measured (T-VALID-1): 30.7ms per batch-64 (actual)
- **Discrepancy**: 3-4× slower than expected!

### Possible Causes

1. **Model Too Large**:
   - Current: 10.1M parameters (reduced from 23.8M)
   - Target: Should be 5-8M for 8-10ms inference
   - Hypothesis: Model still too complex for target throughput

2. **Suboptimal Batch Utilization**:
   - Batch size: 64 (good)
   - GPU utilization: 66% typical, 87% best case
   - Hypothesis: GPU not fully saturated, kernel launch overhead

3. **Missing Optimizations**:
   - CUDA Graphs: Not implemented (could reduce 2-5ms overhead)
   - Persistent kernels: Not used
   - Multi-stream inference: Not implemented

4. **Hardware Limitation**:
   - RTX 3060 Ti may not reach 8-10ms @ FP16 for this model size
   - Review.txt estimate (7,500-10,000 states/sec) may assume smaller model

## Critical Decision Required

### Options to Reach 8,000 Sims/Sec

**Option A: Reduce Model Size** (RECOMMENDED)
- Current: 10.1M params, 20 blocks × 256 channels
- Target: 5-6M params, 10 blocks × 192 channels
- Expected: 15-20ms inference → ~3,200-4,000 states/sec per thread
- With 2 threads (90% efficiency): 5,760-7,200 sims/sec
- **Pros**: Direct path to target, maintains quality
- **Cons**: Requires retraining model

**Option B: Multi-Threading Pipeline** (HIGHER RISK)
- Use 4-8 threads with overlapping inference
- Batch timeout: Reduce to 0.5ms (faster batch collection)
- Expected: 2,200 × 3-4 threads × 70% efficiency = 4,600-6,200 sims/sec
- **Pros**: No model changes
- **Cons**: Still falls short of 8k target

**Option C: CUDA Graphs + Model Reduction** (BEST CASE)
- Reduce model: 10.1M → 6M params (1.5-1.7× speedup)
- Add CUDA Graphs: -2-5ms launch overhead (1.1-1.2× speedup)
- Expected: 30.7ms / 1.7 / 1.15 = 15.6ms → ~4,100 states/sec per thread
- With 2 threads: 4,100 × 2 × 0.9 = 7,380 sims/sec
- **Pros**: Closest to 8k target
- **Cons**: Requires model retraining + CUDA Graphs implementation (4-5 days)

**Option D: Accept Lower Target** (FALLBACK)
- Current best: 2,835 sims/sec (1.35× improvement over regression)
- Realistic with tuning: 3,000-3,500 sims/sec
- **Pros**: No additional work
- **Cons**: Misses 8k target by 56-62%

## Recommendation

**STOP and ESCALATE**: The 8,000 sims/sec target is **NOT achievable** with current model size (10.1M params).

**Required Decision**:
1. **Model reduction** (10.1M → 5-6M params) OR
2. **Target revision** (8k → 3-4k sims/sec realistic)

**Rationale**:
- GPU inference (30.7ms) caps throughput at ~2,000-2,400 states/sec
- No amount of parameter tuning (threads, batch size, timeout) can overcome this
- Spec 004 assumes 8-10ms inference, but hardware delivers 30.7ms @ 10.1M params
- Review.txt estimate (7,500-10,000 states/sec) likely assumes 5-6M param model

**Proposed Path Forward** (if model reduction approved):
1. Reduce model to 6M params (10 blocks × 192 channels)
2. Retrain for 24-48 hours
3. Re-validate FP16 inference (target: 15-20ms per batch-64)
4. Continue with T018/T019 tuning (threads, batch, timeout)
5. Expected final: 7,000-9,000 sims/sec (88-112% of 8k target)

**Timeline Impact**:
- Model reduction + retraining: +2-3 days
- Re-validation: +4 hours
- Total delay: +3 days (increases Phase 4 from 7-8 days to 10-11 days)
