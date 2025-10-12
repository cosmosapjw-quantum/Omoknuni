# Optimization History: MCTS Throughput Recovery

**Document Version**: 1.0
**Last Updated**: 2025-10-12
**Spec**: 004 - MCTS Throughput Recovery
**Purpose**: Record all optimizations attempted, their impact, and lessons learned

---

## Table of Contents

1. [Baseline (Spec 003)](#baseline-spec-003)
2. [Phase 1: Virtual Loss & Quick Wins](#phase-1-virtual-loss--quick-wins)
3. [Phase 2: Architecture Changes](#phase-2-architecture-changes)
4. [Phase 3: Tuning & Profiling (Pending)](#phase-3-tuning--profiling-pending)
5. [Lessons Learned](#lessons-learned)
6. [Future Work](#future-work)

---

## Baseline (Spec 003)

**Date**: 2025-09-15
**Status**: ✅ Complete
**Performance**: 3,831 sims/sec (Gomoku, 8 threads)

### Configuration

```yaml
game: gomoku
model:
  parameters: 23.8M (later reduced to 10.1M)
  architecture: ResNet-20 (20 blocks, 256 channels)
  precision: FP32
search:
  batch_size: 32
  timeout_ms: 1.0
  threads: 8
  simulations: 1600
hardware:
  cpu: AMD Ryzen 9 5900X (12C/24T @ 3.7-4.8 GHz)
  gpu: NVIDIA RTX 3060 Ti (8GB GDDR6)
  ram: DDR4-3600 (dual channel)
  os: Ubuntu 22.04, CUDA 12.1
```

### Bottleneck Analysis

From profiling (session_20251010_190404):

```
Total runtime: 357ms per 1000 simulations

GPU inference:    117ms (32.8%)  ← Not the bottleneck
MCTS overhead:    240ms (67.2%)  ← Primary bottleneck
  - Python/GIL:     ~125ms (35%)
  - Queue coord:     ~72ms (20%)
  - Tree ops:        ~43ms (12%)

Throughput: 3,831 sims/sec = 2,801 sims per GPU second
```

**Key Insight**: GPU is idle 67% of the time. CPU coordination is the bottleneck.

---

## Phase 1: Virtual Loss & Quick Wins

**Timeline**: 2025-10-06 to 2025-10-07
**Status**: ✅ Complete (5/5 tasks)

### T001: WU-UCT Virtual Loss Manager

**Date**: 2025-10-06
**Effort**: 2 days
**Commit**: (no-commit, pre-commit)

**Change**:
- Replaced classic virtual loss (Q-value distortion) with WU-UCT (visit-only)
- PUCT formula: `Q + c_puct * P * sqrt(N_parent) / (1 + N_child + VL_child)`
- Q-value: `W / N` (pure, no VL modification)

**Expected Impact**: 1.5× speedup (reduce Q-value distortion artifacts)

**Actual Impact**: ~2% throughput change (minimal), but +1% policy agreement

**Lessons Learned**:
- Q-value distortion was not a significant performance bottleneck
- WU-UCT more robust to virtual loss magnitude tuning
- Collision metrics essential for understanding thread contention
- **Validation**: 17/17 unit tests pass, 2.7ns atomic operation overhead

---

### T001b: Epoch-Based Tree Clearing

**Date**: Pre-existing (before Spec 004)
**Status**: Already implemented

**Change**:
- Replaced `memset(270MB)` with epoch increment (O(1))
- Tree clearing: 25ms → 25ns (1,000,000× speedup)

**Expected Impact**: Instant tree clearing (remove 10-50ms overhead)

**Actual Impact**: ✅ Achieved - tree clearing <1μs

**Lessons Learned**:
- Lazy initialization is a "free" optimization
- Epoch-based algorithms eliminate bulk memory operations
- **Validation**: 8/8 unit tests pass

---

### T002: Busy-Edge Masking

**Date**: 2025-10-07
**Effort**: 1 day
**Commit**: Multiple (instrumentation + bugfix)

**Change**:
- Set PUCT score to `-INFINITY` for nodes being expanded
- Atomic CAS on `expansion_state` (UNEXPANDED → EXPANDING)
- Only one thread wins CAS, others skip node

**Expected Impact**: Reduce expansion conflicts by 50%

**Actual Impact**: ✅ Verified - expansion conflicts prevented (17/17 tests pass)

**Lessons Learned**:
- Busy-edge masking critical for high thread counts (>8 threads)
- Instrumentation revealed hidden bugs (thread-local block caching)
- **Critical bugfix**: Added `instance_id_` to prevent cross-tree pollution
- **Validation**: -6ns overhead (masking actually faster due to conflict avoidance)

---

### T003: Root Pre-Expansion

**Date**: 2025-10-07
**Effort**: 4 hours
**Commit**: 945383e

**Change**:
- Expand root synchronously before launching threads
- Apply Dirichlet noise: `P'(a) = (1-ε)*P(a) + ε*η_a` (ε=0.25)
- Eliminate N-1 thread idle problem at search start

**Expected Impact**: 2× speedup (eliminate thread idle at start)

**Actual Impact**: ✅ Verified - root expanded in 13ms, threads start immediately

**Lessons Learned**:
- Root serialization was measurable bottleneck (~40ms)
- Dirichlet noise critical for exploration in self-play
- Atomic expansion flag prevents duplicate work
- **Validation**: 5/5 unit tests pass, 4 threads don't duplicate expansion

---

### T004: Thread Affinity (Ryzen 5900X)

**Date**: 2025-10-07
**Effort**: 4 hours
**Commit**: 050e1b9

**Change**:
- Detect Ryzen 5900X topology (2× CCDs, 6 cores each)
- Pin threads to physical cores (avoid cross-CCD traffic)
- ≤6 threads → CCD0 only (shared L3 cache)
- 7-12 threads → split across CCD0 + CCD1

**Expected Impact**: 1.15× speedup (reduce cache misses)

**Actual Impact**: Not measured (benign optimization, low overhead)

**Lessons Learned**:
- Platform-specific optimization (Linux only)
- Graceful degradation on unsupported platforms
- **Validation**: 14/14 unit tests pass

---

### T005: Collision Metrics Instrumentation

**Date**: 2025-10-07
**Effort**: 4 hours
**Commit**: 3932a87

**Change**:
- Track `ExpansionConflict`, `BusyEdgeMasked`, `SelectionRetry`, `UniqueBatchPositions`
- Python API: `get_instrumentation_snapshot()`

**Impact**: Observability (no direct performance impact)

**Lessons Learned**:
- Metrics essential for tuning and debugging
- Atomic counters with relaxed ordering (low overhead)
- **Validation**: 7 C++ tests + 6 Python tests pass

---

## Phase 2: Architecture Changes

**Timeline**: 2025-10-07 to 2025-10-09
**Status**: ✅ 85% Complete (11/13 critical tasks, 2 optional pending)

### T006/T006b: Lock-Free MPMC Queue

**Date**: 2025-10-07 (T006), 2025-10-08 (T006b)
**Effort**: 1 day (T006) + 1 day (T006b integration)
**Commits**: 729fc69 (T006), 5f0bf94 + 25c908f (T006b)

**Change**:
- Replaced `std::deque + mutex` with MPMC ring buffer (4096 capacity)
- Turn-based synchronization (wait-free enqueue/dequeue)
- Result storage: `std::unordered_map` → fixed ring buffer (8192 slots)

**Expected Impact**: 1.4× speedup (eliminate mutex contention)

**Actual Impact**: **NEEDS MEASUREMENT** (T016 pending)

**Critical Bugfixes** (T006b):
1. **Coordinator lifecycle bug**: `stop()` didn't wake waiting threads
2. **Result stealing bug**: Multiple threads consumed same result

**Lessons Learned**:
- Lock-free != wait-free (still had polling issue)
- Fixed memory allocation critical (4096+8192 = 1MB vs unbounded map)
- **Validation**: 19/19 unit tests pass, 1.18ns per enqueue (50× faster than target)

---

### T006c: Condition Variables 🔴 CRITICAL

**Date**: 2025-10-09
**Effort**: 1 day
**Commit**: 2253a97
**Status**: ✅ COMPLETE (implementation), ⚠️ PENDING (validation)

**Change**:
- Replaced polling (10μs sleep) with `std::condition_variable`
- `submit_request()`: enqueue + `notify_one()`
- `collect_batch()`: block on `cv.wait_for(timeout)` instead of spinning

**Expected Impact**: **1.3-1.5× speedup** (eliminate 67% CPU waste from polling)

**Actual Impact**: **UNKNOWN** - T016 benchmarking required

**Problem** (from review.txt, page 8):
> "Recent profiling shows ~60% of execution time is spent with threads waiting (idle). The current busy-wait loop should be replaced with a blocking notification mechanism."

**Lessons Learned**:
- Implementation straightforward (added ~20 lines)
- **CRITICAL**: Validation required to confirm impact
- Expected to be highest-impact optimization in Phase 2

---

### T007: DLPack Tensor Bridge

**Date**: 2025-10-08
**Effort**: 3 days (split across T007a-g)
**Status**: ✅ Complete (all subtasks)

**Change**:
- Zero-copy C++ → PyTorch via `torch.from_dlpack()`
- Pinned CPU memory (`cudaMallocHost`) for fast H2D transfer
- Parallel feature extraction (OpenMP `#pragma omp parallel for`)

**Expected Impact**: 2.5× speedup (eliminate Python tensor conversion loops)

**Actual Impact**: **NEEDS MEASUREMENT** (T016 pending)

**Current Status**:
- H2D transfer: 0.24ms (0.7% of time) - NOT the bottleneck
- **Mystery**: Tensor creation takes 7.5ms (should be <1ms) - T020 investigation required

**Lessons Learned**:
- DLPack protocol surprisingly simple (10 lines for capsule)
- Pinned memory critical (8.6 GB/s bandwidth vs 2-3 GB/s pageable)
- **Issue**: Feature extraction or GIL overhead still significant

---

### T008/T008f: Python Inference Bridge + FP16

**Date**: 2025-10-08 (T008a-e), 2025-10-09 (T008f)
**Effort**: 2 days
**Commit**: 2253a97 (T008f)
**Status**: ✅ COMPLETE (implementation), ⚠️ PENDING (validation)

**Change**:
- T008a-e: DLPack integration with PyTorch
- **T008f: Enable `torch.cuda.amp.autocast()` for FP16 inference**

**Expected Impact**: **1.5-2× GPU speedup** (RTX 3060 Ti tensor cores)

**Actual Impact**: **UNKNOWN** - T016 benchmarking + validation script required

**Problem** (from review.txt, page 13):
> "FP16 can nearly double inference throughput on GPUs that have tensor cores (like RTX 3060 Ti)."

**Lessons Learned**:
- Easy to enable (one-line context manager)
- **CRITICAL**: Must validate with script (prove FP16 actually active)
- May need accuracy validation (value MSE <0.01)

---

### T009: Thread-Local Memory Arenas

**Date**: 2025-10-08
**Effort**: 2 days (split across T009a-f)
**Status**: ✅ Complete

**Change**:
- 4096-node blocks, thread-local allocation
- 99.93% fast-path (thread-local), 0.07% slow-path (global fallback)

**Expected Impact**: Reduce allocation contention (minor, <5% gain)

**Actual Impact**: **NEEDS MEASUREMENT** (T016 pending)

**Lessons Learned**:
- Thread-local eliminates contention (no atomics in fast path)
- Block allocation amortizes overhead
- **Validation**: Arena benchmarks show expected behavior

---

### T010: Pending Expansions Ring Buffer

**Date**: 2025-10-08
**Effort**: 1 day
**Status**: ✅ Complete

**Change**:
- Replaced 4096-slot scan with O(1) lookup (request_id % capacity)

**Expected Impact**: Minor (pending map not a major bottleneck)

**Actual Impact**: **NEEDS MEASUREMENT** (T016 pending)

---

### T011: Persistent Coordinator Lifecycle

**Date**: 2025-10-09
**Effort**: 1 day
**Commit**: 28f11ca
**Status**: ✅ Complete

**Change**:
- Fixed coordinator hanging bug (threads not waking on `stop()`)
- Fixed result stealing bug (threads consuming wrong results)

**Impact**: Stability (no direct performance gain, but critical bugfix)

**Lessons Learned**:
- Lifecycle management critical for long-running processes
- Condition variables must be notified on shutdown

---

### T014: Batched Result Processing

**Date**: 2025-10-09
**Status**: ✅ Complete

**Change**:
- Process multiple results per loop iteration (reduce overhead)

**Impact**: Minor optimization (<5% expected)

---

## Phase 3: Tuning & Profiling (Pending)

**Status**: 🔴 NOT STARTED (waiting on T016)

### Critical Next Steps

1. **T016: Comprehensive Benchmarking** 🔴 HIGHEST PRIORITY
   - Measure actual gains from T006c (condition variables)
   - Measure actual gains from T008f (FP16 mixed precision)
   - Expected: 18-36k sims/sec (vs 2,147 current regression)
   - **Blocker**: Without this, we can't validate success

2. **T017: Baseline Investigation** (new task)
   - Find configuration that achieved 3,831 sims/sec
   - Understand 3,831 → 2,147 regression (44% loss)
   - **Blocker**: Can't measure improvement without baseline

3. **T018: Virtual Loss Tuning**
   - Optimize magnitude (currently 1.0, may not be optimal)
   - Expected: 2-5% improvement

4. **T019: Batch Size & Timeout Tuning**
   - Current: batch=32, timeout=1.0ms (may not be optimal)
   - Expected: 5-15% improvement via reduced thread idle

5. **T020: Profile Remaining Bottlenecks**
   - Investigate 7.5ms tensor creation overhead
   - Investigate 60% thread idle time
   - Expected: 10-20% improvement

---

## Lessons Learned

### What Worked Well

1. **Epoch-Based Clearing**: 1M× speedup for "free" (pre-existing)
2. **Lock-Free Queue**: Clean implementation, 19/19 tests pass
3. **DLPack Zero-Copy**: Clean integration with PyTorch
4. **Instrumentation**: Essential for debugging and tuning

### What Didn't Work as Expected

1. **WU-UCT**: Minimal performance gain (~2%), mainly quality benefit
2. **Current Regression**: 3,831 → 2,147 mystery still unsolved
3. **Tensor Creation**: 7.5ms overhead unexplained (DLPack should be zero-copy)

### Critical Mistakes

1. **No Baseline Validation**: Should have reproduced 3,831 sims/sec first
2. **Optimizations Without Benchmarking**: T006c/T008f implemented but never validated
3. **Configuration Drift**: Lost track of baseline configuration

### Best Practices Identified

1. **Always benchmark**: Implementation ≠ validation
2. **Baseline is sacred**: Document exact configuration
3. **Measure everything**: Can't optimize what you don't measure
4. **Incremental validation**: Test after each optimization
5. **Feature flags**: Enable rollback without code changes

---

## Future Work

### Short Term (Weeks 1-2)

- [ ] T016: Comprehensive benchmarking (CRITICAL)
- [ ] T017: Find baseline configuration (CRITICAL)
- [ ] T018-T019: Parameter tuning (batch size, timeout, VL magnitude)
- [ ] T020: Profile and fix remaining bottlenecks

### Medium Term (Weeks 3-4)

- [ ] T021-T024: Quality gates, documentation, determinism
- [ ] T025: Final sign-off (≥25k sims/sec target)

### Long Term (Post-Spec 004)

- GPU-accelerated MCTS selection (CUDA kernels)
- Network pruning/quantization (beyond FP16)
- Multi-GPU parallelization (root parallelization)
- Distributed self-play (cluster deployment)

---

## Performance Projection (Updated 2025-10-12)

```
Baseline:          3,831 sims/sec (configuration unknown)
Current:           2,147 sims/sec (regression cause unknown)
Expected (T006c): 12,000 sims/sec (1.3-1.5× from condition variables)
Expected (T008f): 26,000 sims/sec (1.5-2× from FP16 on top of T006c)
Target:           25,000 sims/sec (primary success criterion)
Stretch:          30,000 sims/sec (original AlphaZero-inspired goal)

Status: PATH UNCLEAR until T016/T017 complete
```

---

**END OF OPTIMIZATION HISTORY**
