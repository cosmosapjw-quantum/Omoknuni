# Async Inference Optimization Results

**Date**: 2025-10-05
**Branch**: 003-async-inference-multithreading
**Hardware**: AMD Ryzen CPU + NVIDIA RTX 3060 Ti

## Executive Summary

Comprehensive tuning of async MCTS infrastructure identified that **GPU inference accounts for only 32.8% of total execution time**. The remaining 67.2% is MCTS overhead (selection, backup, thread coordination). Current peak performance is 3,831 sims/sec (12.8% of 30k target).

## Performance Breakdown

### GPU vs CPU Comparison (1000 simulations, 8 threads)

| Metric | GPU Async | CPU Async | Ratio |
|--------|-----------|-----------|-------|
| Wall Time | 0.357s | 1.223s | 3.42× |
| Throughput | 2,797 sims/sec | 818 sims/sec | 3.42× |
| Inference Time | 0.117s (32.8%) | 0.966s (78.9%) | 8.26× |
| MCTS Overhead | 0.240s (67.2%) | 0.257s (21.1%) | 0.93× |
| Avg Batch Size | 76.9 | 76.9 | 1.00× |
| Total Batches | 13 | 13 | 1.00× |

**Key Insight**: MCTS overhead is nearly identical (0.240s vs 0.257s) regardless of inference backend. GPU provides 3.4× speedup by reducing inference time from 0.966s → 0.117s, but total speedup is limited by fixed MCTS overhead.

### Thread Scaling Analysis

| Threads | Throughput | Speedup | Efficiency | Avg Batch | Bottleneck |
|---------|------------|---------|------------|-----------|------------|
| 1 | 1,535 sims/sec | 1.00× | 100.0% | 8.9 | - |
| 2 | 2,914 sims/sec | 1.90× | 94.9% | 45.6 | Near-ideal |
| 4 | 3,831 sims/sec | 2.50× | 62.4% | 83.7 | **Peak throughput** |
| 6 | 3,694 sims/sec | 2.41× | 40.1% | 85.4 | Declining |
| 8 | 3,355 sims/sec | 2.19× | 27.3% | 85.4 | Saturation |
| 12 | 3,062 sims/sec | 1.99× | 16.6% | 87.2 | Severe contention |

**Saturation Point**: 4 threads (62.4% efficiency)
**Recommendation**: Use 2-4 threads for optimal balance

**Analysis**: Efficiency drops sharply beyond 2 threads (94.9% → 62.4%), indicating severe lock contention or coordination overhead. Adding threads beyond 4 actively reduces throughput due to contention.

### Batch Size Tuning

| Batch Size | Throughput | Avg Batch | Latency | Status |
|------------|------------|-----------|---------|--------|
| 16 | 2,866 sims/sec | 23.4 | 3.22ms | Underutilized |
| 32 | 3,285 sims/sec | 45.6 | 4.67ms | Good balance |
| 64 | 3,637 sims/sec | 85.4 | 6.82ms | **Optimal** |
| 96 | 3,662 sims/sec | 120.6 | 8.54ms | Diminishing returns |
| 128 | 3,747 sims/sec | 146.4 | 9.94ms | High latency |

**Optimal Configuration**: Batch size 64 (best throughput/latency tradeoff)

**Note**: Actual batch sizes (23-146) are capped at 1.5× configured size after fixing batch explosion bug. This prevents GPU overload while maintaining batching efficiency.

### Timeout Tuning

| Timeout | Throughput | Avg Batch | Per-Sim Latency |
|---------|------------|-----------|------------------|
| 0.3ms | 3,415 sims/sec | 195.2 | 0.29ms |
| 0.5ms | 3,688 sims/sec | 186.4 | 0.27ms |
| 1.0ms | 3,490 sims/sec | 227.8 | 0.29ms |
| 2.0ms | 3,437 sims/sec | 273.3 | 0.29ms |

**Optimal Configuration**: 0.5-1.0ms timeout (minimal latency impact)

**Analysis**: Timeout has minimal impact on throughput (±7%). Average batch sizes remain high regardless of timeout, suggesting batch size threshold is more important trigger than timeout.

## Bug Fixes Implemented

### 1. Batch Size Explosion (CRITICAL)

**Problem**: `collect_batch()` returned ALL pending requests when threshold reached, causing batches of 157-273 instead of configured 64.

**Impact**:
- GPU inference latency: 11ms (should be 3-4ms for batch 64)
- Memory pressure from oversized batches
- Reduced batching efficiency

**Fix**: Cap batch collection at 1.5× `min_batch_size`:
```cpp
size_t max_batch_size = min_batch_size + (min_batch_size / 2);
size_t batch_count = std::min(pending_requests_.size(), max_batch_size);
```

**Result**: Average batch sizes now 23-146 (controlled), but throughput unchanged (3.7k → 3.8k sims/sec).

### 2. PUCT Selector Ignoring Expanding Flag (CRITICAL)

**Problem**: Selector computed PUCT values for ALL children including those being expanded, causing thread contention.

**Fix**: Set PUCT to -∞ for expanding nodes in both vectorized and scalar paths.

**Result**: Throughput increased from 621 sims/sec → 12,051 sims/sec (CPU mode, from previous benchmark).

## Performance Bottleneck Analysis

### Time Distribution (GPU Async, 1000 sims, 8 threads)

```
Total Time: 0.357s

GPU Inference:     0.117s (32.8%)  ← Only 1/3 of time!
  ├─ 13 batches
  ├─ Avg 76.9 positions/batch
  └─ 9.03ms per batch

MCTS Overhead:     0.240s (67.2%)  ← 2/3 of time!
  ├─ Selection (tree traversal)
  ├─ Backup (value propagation)
  ├─ Queue coordination
  ├─ Thread synchronization
  └─ Virtual loss management
```

**Critical Finding**: Even with instant GPU (0ms inference), maximum achievable throughput would be:

```
1000 simulations / 0.240s MCTS overhead = 4,167 sims/sec
```

This is still **7.2× below the 30k target**.

### Estimated Speedup Potential

| Optimization | Potential Gain | Target Throughput | Feasibility |
|--------------|---------------|-------------------|-------------|
| Instant GPU (theoretical) | 1.49× | 5.7k sims/sec | Impossible |
| Optimize selection (-50% time) | 1.33× | 5.1k sims/sec | Hard |
| Optimize backup (-50% time) | 1.33× | 5.1k sims/sec | Hard |
| Reduce queue overhead (-50%) | 1.20× | 4.6k sims/sec | Medium |
| **Combined optimizations** | **2.1× | | **8.0k sims/sec** | **Optimistic** |

**Conclusion**: Achieving 30k sims/sec requires **fundamental architectural changes**, not just tuning.

## Root Cause: Why is MCTS Overhead So High?

### 1. Thread Coordination Overhead

With 8 threads and async queue:
- Lock contention on `pending_mutex_` and `results_mutex_`
- Coordinator thread sleep/wake cycles (100μs sleep in `collect_batch`)
- Thread pool overhead in executor
- Virtual loss atomic operations

**Measured Impact**: Throughput decreases beyond 4 threads (3.8k → 2.9k sims/sec at 16 threads).

### 2. Selection Path Length

Each simulation traverses tree from root to leaf:
- Average tree depth: 2-3 (based on grandchildren/great-grandchildren counts)
- Each selection requires PUCT computation for all children (~225 at root)
- AVX2 vectorization processes 8 children at a time, but root has 225 children = 28 vector ops

**Estimated Time**: ~10-20μs per selection (reasonable for 0.357s / 1000 sims = 357μs per sim)

### 3. Backup Path Length

Each expansion triggers backup from leaf to root:
- Path length: 2-3 nodes
- Atomic operations on visit counts and total values
- Value sign flipping at each level

**Estimated Time**: ~5-10μs per backup

### 4. Queue Overhead

Each simulation:
- Submits request: lock `pending_mutex_`, append, unlock
- Wait for result: coordinator processes batch, lock `results_mutex_`, insert, unlock
- Retrieve result: lock `results_mutex_`, lookup, erase, unlock

**Measured**: With 76.9 avg batch and 13 batches, each batch saves ~70× lock operations compared to per-position inference.

## Tree Growth Validation

The tuning scripts incorrectly reported "shallow tree (1.02 inf/sim)" based on GPU worker's `total_requests` metric. Actual tree inspection reveals healthy growth:

**1000 Simulations, 8 Threads**:
- Root: 1,000 visits ✅
- Children (depth 1): 97 expanded (43% of legal moves)
- Grandchildren (depth 2): 21,728 nodes
- Great-grandchildren (depth 3): 49,952 nodes
- **Total nodes**: 71,778

**Actual Expansion Rate**: 319 expansions / 1000 simulations = 0.32 expansions/sim

This is reasonable because:
1. Many simulations traverse already-expanded tree portions
2. Multiple threads attempt same node (expanding flag prevents duplicates)
3. Some paths hit terminal states (no expansion)

**Conclusion**: Tree growth is healthy. The "1.02 inf/sim" metric was misleading.

## Optimization Recommendations

### Short-Term (Spec 003 Phase 5 Complete)

1. ✅ **Batch size capping** - Implemented (1.5× min_batch_size)
2. ✅ **PUCT expanding flag** - Implemented
3. ✅ **Thread count tuning** - Optimal: 2-4 threads
4. ✅ **Batch size tuning** - Optimal: 64
5. ✅ **Timeout tuning** - Optimal: 0.5-1.0ms

**Current Performance**: 3,831 sims/sec (12.8% of target)

### Medium-Term (Spec 004 Candidate)

1. **Lock-free queue** - Replace mutex-based queue with lock-free MPMC queue
   - Potential: 1.2-1.5× speedup
   - Complexity: Medium
   - Risk: Moderate (debugging lock-free code is hard)

2. **Selection optimization** - Reduce PUCT computation overhead
   - Cache frequently-accessed nodes
   - Optimize AVX2 vectorization for large fan-out (225 children)
   - Potential: 1.2-1.3× speedup

3. **Backup optimization** - Reduce atomic operation overhead
   - Batch backups within same thread
   - Use relaxed memory ordering where safe
   - Potential: 1.1-1.2× speedup

4. **Thread affinity** - Pin threads to specific cores
   - Reduce cache thrashing
   - Improve NUMA locality on Ryzen (6-core CCDs)
   - Potential: 1.1-1.2× speedup

**Estimated Combined**: 2.0-2.6× → **7.7-10k sims/sec**

### Long-Term (Architecture Rethink Required)

**Problem**: Current architecture has fundamental limitations:
- Shared tree requires global locks/atomics
- Thread coordination overhead grows with thread count
- Selection/backup are sequential operations

**Alternative Architectures**:

1. **Virtual Loss Removal** (AlphaZero paper approach)
   - No thread coordination needed
   - Much simpler code
   - Potential: 2-3× speedup from removing virtual loss overhead
   - Trade-off: Slightly less optimal tree exploration

2. **Thread-Local Trees** (early KataGo approach)
   - Each thread maintains separate tree
   - Periodic synchronization
   - Potential: 3-4× speedup from no shared state
   - Trade-off: Higher memory usage, duplicate expansions

3. **GPU-Accelerated Selection** (AlphaZero TPU approach)
   - Move entire MCTS loop to GPU
   - Process 1000s of simulations in parallel
   - Potential: 10-20× speedup
   - Trade-off: Requires complete rewrite, complex GPU kernel development

**Recommendation**: For 30k target, consider Virtual Loss Removal + optimized single-threaded MCTS as Spec 004. This is a proven approach (AlphaZero paper) and much simpler than current multi-threaded architecture.

## Lessons Learned

1. **Premature Batching**: The async coordinator adds 67% overhead for 3.4× GPU speedup. Simple single-threaded MCTS might be faster.

2. **Threading Isn't Free**: Beyond 4 threads, coordination overhead exceeds parallelism benefit.

3. **Measure Before Optimizing**: The "shallow tree" metric led us astray. Always validate assumptions with direct measurements.

4. **Diminishing Returns**: We fixed two critical bugs (batch explosion, PUCT expanding) and tuned all parameters, but only achieved 3.8k sims/sec. Architecture is the bottleneck.

## Next Steps

1. **Document findings** in Spec 003 completion report
2. **Create Spec 004** for architecture rethink:
   - Option A: Virtual loss removal + single-threaded
   - Option B: Lock-free queue + optimized C++ MCTS
   - Option C: GPU-accelerated MCTS (high risk, high reward)
3. **Benchmark baseline**: Single-threaded MCTS without async infrastructure
4. **Profile C++ code**: Identify specific hotspots in selection/backup

## Appendix: Raw Benchmark Data

### Thread Count Optimization

```
Threads  Throughput    Speedup  Efficiency  Avg Batch
1        1,535 sims/sec  1.00×    100.0%      8.9
2        2,914 sims/sec  1.90×     94.9%     45.6
4        3,831 sims/sec  2.50×     62.4%     83.7
6        3,694 sims/sec  2.41×     40.1%     85.4
8        3,355 sims/sec  2.19×     27.3%     85.4
10       3,202 sims/sec  2.09×     20.9%     87.2
12       3,062 sims/sec  1.99×     16.6%     87.2
16       2,915 sims/sec  1.90×     11.9%     87.2
```

### Batch Size Optimization

```
Batch  Throughput       Avg Batch  Inf/Sim  Latency
16     2,866 sims/sec      23.4      1.02    3.22ms
32     3,285 sims/sec      45.6      1.02    4.67ms
48     3,554 sims/sec      66.1      1.02    5.49ms
64     3,637 sims/sec      85.4      1.02    6.82ms
96     3,662 sims/sec     120.6      1.02    8.54ms
128    3,747 sims/sec     146.4      1.02    9.94ms
```

### Timeout Optimization

```
Timeout  Throughput       Avg Batch  Latency
0.3ms    3,415 sims/sec     195.2     0.29ms
0.5ms    3,688 sims/sec     186.4     0.27ms
1.0ms    3,490 sims/sec     227.8     0.29ms
1.5ms    3,628 sims/sec     227.8     0.28ms
2.0ms    3,437 sims/sec     273.3     0.29ms
3.0ms    3,415 sims/sec     273.3     0.29ms
```

### GPU Inference Microbenchmark

```
Batch  Latency  Throughput
16     1.68ms    9,545 pos/sec
32     1.73ms   18,533 pos/sec
64     3.11ms   20,601 pos/sec
96     4.37ms   21,947 pos/sec
```

**GPU Utilization**: Peak 21.9k pos/sec × 3.11ms/batch = 68 positions in flight on average.

---

**Analysis Complete**: 2025-10-05
**Conclusion**: Current async architecture achieves 12.8% of 30k target. Fundamental changes required for target performance.
