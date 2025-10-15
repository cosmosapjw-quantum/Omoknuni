# Overlapped Execution Analysis
**Date**: 2025-10-14
**Finding**: Current architecture is already overlapped, but limited by fundamental constraints

---

## The Misconception

**I initially thought**: Threads are idle because they wait synchronously for GPU results

**Reality**: The architecture is **already overlapped** (submit + process in same loop), but limited by:
1. **Tree structure constraints**
2. **MCTS algorithm constraints**
3. **Sequential dependency requirements**

---

## Understanding The Real Constraints

### MCTS Algorithm Requires Sequential Dependencies

**Each simulation has dependencies**:
1. Select to leaf → **Requires updated N, W, Q values from previous sims**
2. Expand node → **Requires inference result**
3. Backup value → **Updates N, W, Q for next sim**

**Why this limits parallelism**:
- Simulation i+1 needs results from simulation i (same path)
- Can't build infinite pipeline - tree structure limits independence
- Virtual loss prevents same-node expansion, but not same-path traversal

---

## Why Thread Efficiency Is Low (15%)

### Single-Thread Analysis

**Single thread: 1,171 sims/sec**

**Time breakdown per simulation**:
- Tree search (select + backup): ~0.5ms
- GPU wait (inference): ~0.35ms (1000ms / 22 batch size / ~130 batches/sec)
- Total: ~0.85ms per sim

**Wait percentage**: 0.35 / 0.85 = **41% wait time even for single thread!**

---

### Multi-Thread Analysis

**8 threads: 1,796 sims/sec**

**Efficiency**: 1,796 / (1,171 × 8) = **19.1%**

**Why so low**?

**Theory 1: Tree contention** (RULED OUT)
- 0% expansion conflicts
- 0% busy edges
- 0% selection retries
- Tree coordination is excellent!

**Theory 2: GPU bottleneck** (PARTIALLY TRUE)
- Single thread already 41% GPU wait
- With 8 threads, more coordination needed
- But GPU util is only 61%, so not fully saturated

**Theory 3: Insufficient tree depth** (LIKELY)
- Early in search, tree is shallow
- Limited independent paths for threads
- Threads compete for same nodes despite virtual loss

---

## The Real Bottleneck: Tree Structure

### Hypothesis

**Early search (first 100-200 sims)**:
- Tree is shallow (depth ~5-10)
- Few independent paths (branching factor ~10-20)
- 8 threads compete for ~50-200 positions
- High collision rate (despite virtual loss)

**Later search (200-800 sims)**:
- Tree is deeper (depth ~10-20)
- More independent paths (branching factor compounds)
- 8 threads have ~1000+ positions to explore
- Lower collision rate

**Expected behavior**:
- Early: Low thread efficiency (~10-15%)
- Mid: Moderate efficiency (~30-40%)
- Late: Higher efficiency (~50-60%)

**Observed**: Average 15-19% efficiency suggests we spend most time in early/mid phase

---

## Why "Overlapped Execution" Won't Help Much

**Current architecture** (continuous_simulation_runner.cpp):
```cpp
while (completed < num_simulations) {
    // Submit phase (if quota not reached)
    if (submitted < num_simulations) {
        submit_request(...);  // Non-blocking
    }

    // Process phase (always try)
    process_completed_results(...);  // Non-blocking

    // Sleep only if no work available
    if (no_results && (all_submitted || waiting)) {
        sleep(10-20μs);
    }
}
```

**This is ALREADY overlapped!** Threads:
1. Submit multiple requests (up to 4096 in-flight)
2. Process results as they arrive
3. Only sleep when truly blocked

**What more "overlapping" would require**:
- Multiple **independent** simulations per thread
- But MCTS simulations are NOT independent!
- Each sim updates tree, affecting next sim

---

## The Actual Solution: Not What I Expected

### Option A: Increase Thread Count (SIMPLE, EFFECTIVE)

**Rationale**:
- More threads → more parallel tree exploration
- Better tree coverage → less collision
- Higher GPU saturation → better batch utilization

**Test**: Try 16-24 threads instead of 8

**Expected gain**: 1.5-2× throughput

**Why it works**:
- Doesn't change architecture
- Exploits existing tree parallelism
- Simple parameter change

---

### Option B: Increase Simulations Per Search (WORKLOAD CHANGE)

**Rationale**:
- Longer searches → deeper trees
- Deeper trees → more parallelism
- Better thread efficiency in later phases

**Test**: Run with 1600-3200 simulations instead of 800

**Expected**: Higher average thread efficiency

**Why it works**:
- Tree depth grows with simulations
- More opportunities for parallel exploration
- Amortizes early-phase inefficiency

---

### Option C: Hybrid Batch Processing (COMPLEX)

**Idea**: Process multiple root positions in parallel

**Implementation**:
- Instead of 1 root × 800 sims
- Do 8 roots × 100 sims each (embarrassingly parallel)
- Each thread owns a separate root

**Expected gain**: 5-8× throughput (near-linear scaling)

**Trade-off**:
- Different MCTS semantics (8 shallow trees vs 1 deep tree)
- May affect play strength
- Only works for batch inference scenarios (training)

---

## Recommendation: Test Option A First

**Next step**: Increase thread count from 8 → 16

**Hypothesis**:
- 16 threads: ~2,800-3,200 sims/sec (1.6-1.8× improvement)
- 24 threads: ~3,500-4,000 sims/sec (2.0-2.2× improvement)
- 32 threads: ~4,000-4,500 sims/sec (diminishing returns)

**Why I expect this to work**:
- More threads → better tree coverage
- Less thread collision (more independent paths)
- Better GPU batch accumulation
- No architecture changes needed

**Time**: 5 minutes to test

---

## Conclusion

**My initial analysis was partially wrong**:
- ❌ Architecture is NOT lacking overlapped execution (it already has it)
- ❌ Implementing "overlapped execution" won't help (already done)
- ✅ Thread efficiency is low due to tree structure constraints
- ✅ More threads = more parallelism = higher throughput

**Recommended immediate action**: Test with 16-24 threads

**If that works**: We can hit 3,000-4,000 sims/sec with simple parameter change

**If that doesn't work**: Need to profile actual tree depth/collision rate

---

**End of Analysis**
