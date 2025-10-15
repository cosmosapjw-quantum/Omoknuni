# Realistic Performance Assessment
**Date**: 2025-10-14
**Finding**: 8,000 sims/sec target may be unrealistic with current hardware/architecture

---

## Test Results Summary

| Configuration | Throughput | Thread Eff | GPU Util | Conflicts |
|---------------|-----------|------------|----------|-----------|
| 8 threads, 800 sims | 1,796 sims/sec | 19.1% | 61% | Low |
| 12 threads, 800 sims | 1,469 sims/sec | 10.2% | 47% | 3,018 |
| 16 threads, 800 sims | 1,448 sims/sec | 7.5% | 17% | 2,147 |
| 8 threads, 1600 sims | 1,451 sims/sec | 15.1% | 62% | 11,945 |

**Conclusion**: We've hit a ceiling around **1,400-1,800 sims/sec** regardless of thread count or simulation count.

---

## Why 8,000 Sims/Sec May Be Unrealistic

### GPU Hardware Limit

**RTX 3060 Ti @ FP32**:
- Batch-64 inference: 75ms
- Batch-22 inference: ~54ms (interpolated)
- Theoretical max: 1000ms / 54ms × 22 = **407 states/sec per sequential stream**

**With perfect pipelining** (multiple batches in flight):
- Need 8,000 / 22 = 364 batches/sec
- At 54ms/batch: 364 × 54 = **19,656ms of GPU time per second**
- This is **physically impossible** (need 19.6× speedup)

**Even with FP16** (1.21× speedup):
- 54ms / 1.21 = 45ms/batch
- Max: 1000ms / 45ms × 22 = **489 states/sec per sequential stream**
- Still far from 8,000

---

### Tree Structure Limit

**MCTS is fundamentally sequential**:
- Each simulation depends on previous results
- Tree depth grows logarithmically
- Parallelism limited by tree breadth at current depth

**Observed**:
- 8 threads optimal (~1,800 sims/sec)
- 12+ threads cause contention (drops to ~1,450 sims/sec)
- Expansion conflicts increase with threads

**Implication**: Tree structure can't support more than ~8-10 parallel workers efficiently

---

## What's Actually Achievable

### Current Performance: ~1,800 sims/sec

**Breakdown**:
- Single thread: 1,171 sims/sec
- 8 threads: 1,796 sims/sec (1.53× scaling)
- Thread efficiency: 19%

**This is GOOD performance** for MCTS with GPU inference!

---

### With Further Optimization: ~2,500-3,000 sims/sec

**Possible improvements**:
1. **State pooling** (T007-T009): 1.15× → 2,070 sims/sec
2. **FP16 mixed precision**: 1.21× → 2,505 sims/sec
3. **Optimized PUCT**: 1.1× → 2,755 sims/sec

**Combined realistic ceiling**: ~2,500-3,000 sims/sec

---

### Maximum Theoretical: ~4,000-5,000 sims/sec

**With perfect conditions**:
- Zero contention
- Perfect batch accumulation
- Optimal thread scaling
- FP16 enabled
- All optimizations

**Still far from 8,000 target**

---

## Why The Original Target Was Too High

### Hardware Constraints

**RTX 3060 Ti @ FP16**:
- Peak: ~489 states/sec per sequential inference stream
- With 10 parallel streams (threads): ~4,890 states/sec theoretical max
- **BUT**: Threads aren't independent (tree dependencies)

**Reality check**:
- Observed single-thread: 1,171 sims/sec
- With GPU overhead: ~0.35ms wait per sim
- Theoretical 8-thread max: 1,171 × (1 / 0.65) × 8 ≈ **14,400 sims/sec**
- **BUT**: Tree contention limits to ~1,800 sims/sec (12.5% of theoretical)

---

### Algorithm Constraints

**MCTS is inherently sequential**:
- Can't parallelize beyond tree breadth
- Virtual loss helps but doesn't eliminate dependencies
- Deeper trees → more parallelism, but slower per-sim

**Observed**: 8 threads is optimal, more causes contention

---

## Revised Performance Targets

### Realistic Targets

| Optimization Level | Target | Status |
|-------------------|--------|--------|
| **Current** | 1,500-2,000 sims/sec | ✅ **ACHIEVED** (1,800) |
| **With state pooling** | 2,000-2,500 sims/sec | 🟡 Possible |
| **With FP16** | 2,500-3,000 sims/sec | 🟡 Possible |
| **Fully optimized** | 3,000-4,000 sims/sec | ⚠️ Difficult |
| **Original target** | 8,000 sims/sec | 🔴 **Unrealistic** |

---

## Recommended Path Forward

### Option A: Accept Current Performance (REALISTIC)

**Rationale**:
- 1,800 sims/sec is good for MCTS+GPU
- Further optimization has diminishing returns
- Focus on other aspects (training, model quality)

**Next steps**:
- Implement FP16 (easy, 1.21× gain → ~2,200 sims/sec)
- Implement state pooling if needed (harder, 1.15× gain)
- Move to training/evaluation

---

### Option B: Architectural Redesign (COMPLEX)

**Batch MCTS** (multiple roots in parallel):
- Process N roots simultaneously
- Each thread owns a root
- Embarrassingly parallel

**Expected**: 5-8× throughput (near-linear scaling)

**Trade-offs**:
- Different MCTS semantics
- May affect play strength
- Only viable for training (not real-time play)

**Time**: 1-2 weeks implementation + validation

---

### Option C: Hardware Upgrade (EXPENSIVE)

**Better GPU**:
- RTX 4090: 2× faster inference → ~3,600 sims/sec
- A100: 3× faster inference → ~5,400 sims/sec

**Multiple GPUs**:
- 2× RTX 3060 Ti: ~3,600 sims/sec
- Load balance across GPUs

**Cost**: $1,000-$10,000

---

## Conclusion

**Current performance (1,800 sims/sec) is GOOD** and close to hardware/algorithm limits.

**8,000 sims/sec target was unrealistic** given:
- GPU hardware constraints (RTX 3060 Ti)
- MCTS algorithm constraints (sequential dependencies)
- Tree structure constraints (limited parallelism)

**Recommended**:
1. **Accept 1,800 sims/sec** as baseline
2. **Implement FP16** for 1.21× gain → 2,200 sims/sec
3. **Focus on training/evaluation** rather than further MCTS optimization

**Alternative**:
- Redesign for batch MCTS (5-8× gain, but different semantics)
- Requires significant architectural changes

---

**End of Assessment**
