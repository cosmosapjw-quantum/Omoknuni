# Timeout Experiment Failure
**Date**: 2025-10-14
**Branch**: `004-mcts-throughput-recovery`
**Experiment**: Reduce coordinator timeout from 10ms → 1ms

---

## Hypothesis (WRONG)

**Reasoning**: GPU profiling showed 30% utilization. Coordinator timeout of 10ms seemed excessive compared to 75ms batch inference time. Reducing timeout to 1ms should allow faster batch submission, increasing GPU utilization.

**Expected Outcome**: 1.5-2× throughput improvement from higher GPU utilization

---

## Actual Result: CATASTROPHIC FAILURE

### Performance Impact

| Threads | Throughput (10ms) | Throughput (1ms) | Regression |
|---------|-------------------|------------------|------------|
| 1       | 1,117 sims/sec    | 1,117 sims/sec   | 0% (no change) |
| 2       | 1,943 sims/sec    | 260 sims/sec     | **-86.6%** |
| 4       | 1,744 sims/sec    | 204 sims/sec     | **-88.3%** |
| 8       | ~1,800 sims/sec   | 192 sims/sec     | **-89.3%** |

**Average degradation**: ~88% performance loss for multi-threaded workloads

---

## Root Cause Analysis

### Why Did This Fail So Badly?

**1. Batch Accumulation Time**

With 8 threads submitting requests, the coordinator needs time to collect a reasonable batch:
- **10ms timeout**: Enough time for 8 threads to generate ~16-23 requests
- **1ms timeout**: Only time for ~2-3 requests before timeout fires

**Result**: Batch size drops from 22.9 → ~2-3, causing:
- More frequent GPU calls (higher overhead)
- Smaller batches (less efficient GPU utilization)
- More coordinator cycles (more contention)

---

**2. GPU Inference Is Slower Than Tree Search**

**GPU inference**: 75ms per batch-64 = 1.17ms per state (with overhead)

**Tree search**: Fast enough that threads generate requests quickly

**With 1ms timeout**:
- Coordinator fires timeout every 1ms
- But GPU takes 75ms to process batch
- This creates a FIFO queue backlog
- Threads wait longer for results
- Throughput collapses

---

**3. Coordinator Overhead**

Each coordinator cycle has overhead:
- Collect batch (mutex lock)
- Call Python (GIL)
- Submit results (mutex lock)

**10ms timeout**: ~100 cycles/sec, overhead negligible

**1ms timeout**: ~1000 cycles/sec, overhead dominates

---

## Key Insight: Timeout Must Balance Two Factors

### Factor 1: Batch Accumulation Time

- **Too short** (<1ms): Not enough time to collect min_batch_size requests
- **Too long** (>50ms): Excessive latency, threads idle waiting

**Optimal**: 5-15ms for 8 threads with min_batch=16

---

### Factor 2: GPU Inference Time

- **If timeout << inference time**: Batches queue up, threads block
- **If timeout >> inference time**: GPU idles between batches

**Optimal**: timeout ≈ (inference_time / num_threads) for continuous flow

---

## Why 10ms Works Well

**Batch-23 inference time**: ~54ms (interpolated from profiling)

**With 8 threads**:
- Threads generate requests in parallel
- 10ms allows accumulation of 16-23 requests
- GPU finishes batch (~54ms) while coordinator prepares next batch
- Pipeline stays full

**With 1ms**:
- Only 2-3 requests per batch
- 50× more coordinator cycles per second
- GPU gets 50× more batches but each is 10× smaller
- Net result: 10× slower overall

---

## Corrected Understanding

**GPU utilization (30%) is NOT because timeout is too high.**

**Real bottleneck**: Threads are not generating enough NN evaluation requests. Either:
1. Cache hit rate is high (reusing existing evaluations)
2. Thread coordination overhead (virtual loss, tree contention)
3. Tree search is slower than expected

---

## Lesson Learned

**Do NOT blindly optimize parameters based on single metrics.**

GPU utilization (30%) looked like a problem, but reducing timeout made it WORSE. The 70% idle time is NOT because the coordinator is slow - it's because the threads aren't feeding enough work to the pipeline.

---

## Next Steps (Corrected)

### ✅ Revert to 10ms timeout
**Status**: COMPLETE

### 🔴 Profile thread idle time
**Goal**: Measure where threads spend time during MCTS search

**Questions**:
1. What % of time are threads in tree search vs waiting for NN?
2. What % of NN requests are cache hits (parent already evaluated)?
3. What's the actual coordinator cycle time?
4. Are threads blocked by virtual loss contention?

**Tool**: C++ instrumentation with high-resolution timers

**Expected Outcome**: Identify actual bottleneck (tree search, cache, coordinator, contention)

---

### ⏸️ Defer Phase 1 T007-T009 (state pooling)

**Rationale**: Still only 10-15% gain, not on critical path for 3× improvement needed

---

## Conclusion

**Experiment FAILED** - reducing timeout from 10ms → 1ms caused 88% throughput loss.

**Key Finding**: 10ms timeout is NOT the bottleneck. Low GPU utilization (30%) is caused by threads not generating enough NN evaluation requests, NOT by slow batch submission.

**Recommended Action**: Instrument thread idle time to identify actual bottleneck (cache hits, tree search, or coordination overhead).

---

**End of Report**
