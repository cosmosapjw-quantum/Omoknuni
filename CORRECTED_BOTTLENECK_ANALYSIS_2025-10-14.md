# Corrected Bottleneck Analysis
**Date**: 2025-10-14
**Status**: All previous analysis was WRONG - starting fresh with review.txt findings

---

## Critical Realization

**I was solving the WRONG problem!**

My previous analysis focused on:
- ❌ Thread efficiency (15.4%) - MISLEADING METRIC
- ❌ GPU utilization (30-60%) - NOT THE PRIMARY BOTTLENECK
- ❌ Cache hit rate (0%) - NOT THE ISSUE
- ❌ Thread count optimization - WRONG APPROACH
- ❌ Overlapped execution - ALREADY EXISTS
- ❌ Feature extraction (7.5ms) - VERIFIED NOT THE BOTTLENECK

---

## The ACTUAL Bottlenecks (from review.txt)

### Current Performance Reality

**Measured**: ~2,147 sims/sec (regression from 3,831 baseline)
**Target**: ≥8,000 sims/sec (hardware-grounded, realistic)
**Gap**: 3.7× improvement needed

---

### Profiling Truth (lines 14-19 of review.txt)

**Time distribution**:
- Neural network inference (GPU): **32.8%**
- MCTS coordination (CPU): **67.2%** ← PRIMARY BOTTLENECK

**This means**: CPU-side MCTS logic is the dominant bottleneck, NOT the GPU!

---

## Bottleneck #1: MCTS Coordination Overhead (67.2% of time)

**What this includes** (from review.txt lines 14-19):
- Selection (tree traversal, PUCT computation)
- Expansion (node allocation, policy masking)
- Backup (value propagation, atomic updates)
- Thread synchronization

**Impact**: Massive CPU overhead per simulation

**Evidence**:
- Only 32.8% time in GPU inference
- 67.2% time in CPU coordination
- CPU is the limiting factor, NOT GPU

---

## Bottleneck #2: Excessive State Cloning (lines 37-54)

**Current behavior**: Each simulation clones state **2-3 times**

**Where cloning happens**:
1. `run_simulation()` clones root state at start
2. `continuous_simulation_runner.cpp` clones for queue submission:
   ```cpp
   queue_state = current_state->clone()
   ```
3. `AsyncInferenceQueue.submit_request()` clones again internally:
   ```cpp
   request.state = state->clone()
   ```

**Impact**:
- Wasteful CPU cycles (Gomoku: 225 cells × 36 planes per clone)
- Memory allocation pressure
- Python GC overhead (if Python objects involved)
- **"State cloning waste (2–3× per simulation)" explicitly listed as key issue**

**Current**: NOT zero-copy despite using DLPack for tensors!

---

## Bottleneck #3: Thread Contention & Locking (lines 71-110)

**Symptoms**: Threads idle ~60% of the time (lines 79, 102-103)

**Root causes**:

### A) Global Allocation Mutex (lines 71-78)

**Location**: `tree.cpp` - `allocate_nodes()` uses `allocation_mutex_`

**Problem**:
- Multi-node allocation (children expansion) takes global lock
- Many threads expanding concurrently → serialize on lock
- Threads stall waiting for lock

**Evidence**: "severity increases with thread count – more threads trying to expand leads to more waiting"

### B) Atomic Contention on Hot Nodes (lines 82-90)

**Problem**:
- Root node and high-level nodes hit by ALL threads
- Atomic updates (visit count, value) on same cache lines
- False sharing and contention

**Mitigation already attempted**: Batching updates to cut atomic ops in half (Phase 3/4)

### C) Thread Coordination Waiting (lines 95-110)

**Scenarios**:
- Thread finds leaf already being expanded → backs off, sleeps 50-100µs
- Queue saturated (4096 in-flight cap) → thread sleeps
- All simulations submitted, waiting for results → thread sleeps

**Result**: "~1.489s of 2.5s search was thread idle time (~60%)" (line 102)

### D) Poor Multi-Thread Scaling (lines 65-69)

**Measured**:
- 4 threads: 3,831 sims/sec (baseline)
- 8 threads: 3,355 sims/sec (~12.5% efficiency gain from doubling)
- 12 threads: 3,062 sims/sec (~16% efficiency - WORSE)

**Conclusion**: Beyond 4 threads, efficiency plummets due to contention

---

## Bottleneck #4: Busy-Wait on Results (lines 125-136)

**Current behavior**:
- `ContinuousSimulationRunner` calls `process_completed_results()` in loop
- If no results ready (returns 0) → sleep 50-100µs
- Repeat until results available
- **This is spin-waiting!**

**Problem**:
- Wastes CPU cycles in useless loops
- Contributes to high coordination overhead (67.2%)
- BatchInferenceCoordinator pushes results but doesn't wake threads

**Solution**: Use condition variable for push-based notification (lines 131-135)

---

## Feature Extraction: NOT THE PRIMARY BOTTLENECK

**My testing** (just now):
- Default: 6.21ms for batch-64
- OMP_NUM_THREADS=1: 8.64ms
- OMP_NUM_THREADS=12: 1.57ms (optimal)

**BUT**: Testing MCTS throughput:
- OMP_NUM_THREADS=1: 1,543 sims/sec
- OMP_NUM_THREADS=12: 1,529 sims/sec
- **NO SIGNIFICANT DIFFERENCE!**

**Conclusion**: Feature extraction is NOT the bottleneck for overall MCTS throughput. The 7.5ms cited in review.txt is accurate for the operation itself, but it's not the limiting factor in practice because:
1. Batching amortizes this cost
2. MCTS coordination (67.2%) dominates
3. State cloning overhead is larger impact

---

## What I Got Wrong

### Wrong #1: Thread Efficiency Metric

**My calculation**: 1,800 / (1,200 × 8) = 15.4% efficiency

**Why it's misleading**:
- Assumes linear scaling is possible
- MCTS has fundamental sequential dependencies
- Tree structure limits parallelism
- 15.4% is a SYMPTOM, not the ROOT CAUSE

**Real issue**: 60% idle time due to contention/locking (review.txt line 102)

### Wrong #2: GPU Utilization Focus

**My thinking**: 30% GPU util → need more throughput

**Reality**:
- GPU is only 32.8% of total time
- Even if GPU was 100% utilized, only 1.3× improvement
- CPU coordination (67.2%) is the primary bottleneck

**Real issue**: Fix CPU overhead first, THEN GPU matters

### Wrong #3: Overlapped Execution

**My thinking**: Threads wait synchronously for GPU results

**Reality**:
- Architecture IS already overlapped (submit + process in same loop)
- Waiting is due to CONTENTION, not architecture
- The sleep at lines 171-173 is CORRECT (avoids busy-wait)

**Real issue**: Reduce contention so threads don't need to wait

### Wrong #4: Increase Thread Count

**My attempt**: Test 12-16 threads

**Result**: WORSE performance (contention increases)

**Reality** (review.txt lines 113-123):
- Optimal: 2-4 threads
- Beyond that: efficiency plummets due to synchronization overhead
- More threads = more contention on locks and atomics

---

## The CORRECT Fix Priority

### Priority #1: Eliminate State Cloning (Biggest Win)

**Implementation** (review.txt lines 164-208):

**A) Thread-Local State Pools**:
```cpp
// Each thread maintains reusable state buffer
class ThreadLocalStatePool {
    std::vector<std::unique_ptr<IGameState>> pool_;
    IGameState* acquire() { /* reuse or allocate */ }
    void release(IGameState* state) { /* return to pool */ }
};
```

**B) Pass State by Reference to Queue**:
- Change `submit_request()` to NOT clone internally
- Transfer ownership via `std::move`
- Queue holds state until inference complete

**C) Pre-compute Legal Moves**:
- Call `state.getLegalMoves()` before queuing
- Store in `InferenceRequest` structure
- Expansion uses stored moves (no state needed)

**Expected gain**: 1.3-1.5× throughput (eliminate 2-3× cloning overhead)

---

### Priority #2: Reduce Thread Contention

**A) Fix Allocation Lock** (review.txt lines 225-236):
- Batch allocate children outside lock
- Thread-local arenas hand out blocks of 256 nodes
- Lock-free allocation for contiguous ranges

**B) Optimize Atomic Updates** (review.txt lines 237-243):
- Use relaxed memory order where safe
- Ensure 64-byte alignment (already done)
- Atomic CAS loops are already efficient (compare_exchange_weak)

**C) Reduce Waiting/Sleeping** (review.txt lines 212-224):
- Replace spin-loop with condition variable
- Coordinator signals when results ready
- Threads block efficiently (not polling every 50µs)

**Expected gain**: 1.5-2.0× throughput (reduce 60% idle time)

---

### Priority #3: Tune Thread/CPU Affinity

**Current issue** (review.txt lines 244-250):
- Thread affinity uses `hash(thread::id) % 24`
- Suboptimal distribution across physical cores
- May use SMT siblings before saturating cores

**Fix**:
- Pin first 12 threads to cores 0-11 explicitly
- One thread per physical core
- Avoid SMT until physical cores saturated

**Expected gain**: 1.15× throughput (reduce cross-CCX traffic)

---

### Priority #4: Streamline Python ↔ C++ Interface

**Current overhead** (review.txt lines 59-62, 258-307):
- Python/GIL overhead: ~67% of runtime (actually coordination, but interface contributes)
- Batch list preparation and result conversion
- Per-state Python processing in batch callback

**Optimizations** (review.txt lines 260-280):
- Ensure DLPack fast path active (zero-copy)
- Return NumPy arrays (not Python lists)
- Avoid per-state Python loops
- `std::move` policy vectors (not copy)

**Expected gain**: 1.1-1.2× throughput (reduce Python overhead)

---

## Combined Expected Improvement

**Sequential application**:
1. State cloning elimination: 2,147 → 3,221 sims/sec (1.5×)
2. Thread contention fixes: 3,221 → 5,797 sims/sec (1.8×)
3. Thread affinity tuning: 5,797 → 6,667 sims/sec (1.15×)
4. Python interface streamlining: 6,667 → 7,334 sims/sec (1.1×)

**Conservative estimate**: ~7,300 sims/sec (91% of 8,000 target)

**With all fixes + tuning**: 8,000-10,000 sims/sec achievable

---

## What About the 8,000 Target?

**review.txt verdict** (lines 340-348):
- Hardware realistic throughput: **~8k sims/sec sustained**
- With lighter model: **~10k sims/sec possible**
- Repository revised target from 25k to **8-10k after factoring in RTX 3060 Ti limits**

**This IS achievable** with the correct fixes above!

---

## Action Plan

### Phase 1: State Cloning Elimination (1-2 days)

**Task 1**: Implement thread-local state pools
**Task 2**: Refactor queue to accept moved states (not clone)
**Task 3**: Pre-compute legal moves in request
**Task 4**: Validate with benchmarks

**Expected**: 3,000-3,500 sims/sec

---

### Phase 2: Thread Contention Reduction (2-3 days)

**Task 1**: Replace allocation mutex with lock-free batching
**Task 2**: Add condition variable for result notification
**Task 3**: Optimize atomic operations where possible

**Expected**: 5,500-6,500 sims/sec

---

### Phase 3: CPU Affinity & Interface (1 day)

**Task 1**: Fix thread affinity pinning
**Task 2**: Optimize Python ↔ C++ batch interface
**Task 3**: Final tuning and validation

**Expected**: 7,500-8,500 sims/sec

---

### Phase 4: Optional Model Optimization

**If still below target after Phase 3**:
- Implement lightweight ResNet from review.txt (lines 621-1396)
- Expected: 1.5-2× additional speedup
- Total: 12,000-17,000 sims/sec possible

---

## Conclusion

**My previous work was chasing symptoms, not root causes.**

**The REAL bottlenecks** per review.txt:
1. ✅ MCTS coordination overhead (67.2%) - state cloning + contention
2. ✅ State cloning (2-3× per simulation) - wasteful
3. ✅ Thread contention (60% idle time) - locking issues
4. ✅ Python interface overhead - room for improvement

**Feature extraction (7.5ms)**: Verified NOT the bottleneck in practice

**Next step**: Implement state pooling (Priority #1) to eliminate 2-3× cloning overhead

---

**End of Corrected Analysis**
