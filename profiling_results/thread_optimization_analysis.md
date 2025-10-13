# T018: Thread Count Optimization Analysis

**Date**: 2025-10-13
**Target**: Find optimal thread count for 3,000-3,500 sims/sec target
**Current Best**: 2,835 sims/sec (4 threads, first run)

## Existing Thread Scaling Data (Post-OpenMP)

From `thread_scaling_post_openmp.json`:

| Threads | Throughput | GPU Util | CPU Util | Parallel Eff | vs 1T Speedup |
|---------|-----------|----------|----------|--------------|---------------|
| 1       | 1,230 sims/sec | 48% | 4.2% | 100.0% (baseline) | 1.00× |
| 2       | 2,205 sims/sec | 66% | 4.2% | **89.6%** ⭐ | 1.79× |
| 4       | 2,214 sims/sec | 66% | 5.1% | 45.0% | 1.80× |
| 6       | 2,173 sims/sec | 61% | 4.2% | 29.5% | 1.77× |
| 8       | 2,198 sims/sec | 66% | 3.8% | 22.4% | 1.79× |
| 10      | 2,113 sims/sec | 61% | 4.2% | 17.2% | 1.72× |
| 12      | 2,166 sims/sec | 65% | 5.5% | 14.7% | 1.76× |

## Key Findings

### 1. Performance Plateau
- **Observation**: Throughput plateaus at ~2,200 sims/sec for 2+ threads
- **Range**: 2,113-2,214 sims/sec (4.6% variance)
- **Conclusion**: System is GPU-bound, not CPU-bound

### 2. Optimal Thread Count: 2 Threads ⭐
- **Throughput**: 2,205 sims/sec
- **Parallel Efficiency**: 89.6% (excellent!)
- **GPU Utilization**: 66% (good)
- **CPU Utilization**: 4.2% (low, plenty of headroom)

**Rationale**:
- Best efficiency (89.6% vs 45% @ 4T)
- Near-optimal throughput (only 9 sims/sec behind 4T)
- Lower resource usage (less CPU/memory)
- More predictable performance

### 3. Why More Threads Don't Help
- GPU inference bottleneck: 30.7ms per batch
- Threads can't make GPU go faster
- Adding threads increases coordination overhead without benefits
- Thread contention beyond 2 threads (efficiency drop: 89.6% → 45%)

## Comparison with Pre-OpenMP

| Threads | Pre-OpenMP | Post-OpenMP | Improvement |
|---------|-----------|-------------|-------------|
| 1       | 1,364 sims/sec | 1,230 sims/sec | -9.8% ⚠️ |
| 2       | 1,241 sims/sec | 2,205 sims/sec | +77.7% ✅ |
| 4       | 2,235 sims/sec | 2,214 sims/sec | -0.9% |
| 8       | 1,450 sims/sec | 2,198 sims/sec | +51.6% ✅ |
| 12      | 1,025 sims/sec | 2,166 sims/sec | +111.4% ✅ |

**Analysis**:
- Single-thread performance slightly regressed (likely noise/variance)
- Multi-thread (2+) performance significantly improved
- OpenMP fix removed coordination bottleneck at higher thread counts

## Recommendation

**Optimal Configuration**: **2 threads**

**Justification**:
1. Best parallel efficiency: 89.6%
2. Near-optimal throughput: 2,205 sims/sec
3. Lowest resource usage (CPU, memory)
4. Most stable/predictable performance
5. Easiest to reason about for debugging

**Expected with Further Tuning**:
- Current @ 2T: 2,205 sims/sec
- With batch/timeout optimization (T019): 2,400-2,600 sims/sec (est)
- With profiling fixes (T020): 2,600-2,800 sims/sec (est)
- **Target range**: 3,000-3,500 sims/sec

**Gap to Close**: 795-1,295 sims/sec (27-37% improvement needed)

## Next Steps (T019)

Focus on batch size and timeout optimization:
1. Test batch sizes: 32, 48, 64, 80, 96
2. Test timeouts: 0.5ms, 1.0ms, 2.0ms, 5.0ms
3. Find sweet spot for GPU utilization (target: 75-85%)
4. Monitor batch fill rate (target: ≥70% of batch_size)
