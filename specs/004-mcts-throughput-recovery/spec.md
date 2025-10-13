# Specification 004: MCTS Throughput Recovery

**Status**: IN PROGRESS
**Last Updated**: 2025-10-13 (Targets Revised)
**Current Phase**: Phase 3 (Final Optimizations) → Phase 4 (Integration & Tuning)
**Version**: 1.2 (Revised Targets)

## ⚠️ **CRITICAL UPDATE (2025-10-13)**: Revised Realistic Targets

**Analysis Findings** (from review.txt + SPECIFICATION.md clarification):
- **Original Target**: 25,000-30,000 sims/sec (unrealistic on Ryzen 5900X + RTX 3060 Ti)
- **GPU Hardware Limit**: 8,000-10,000 states/sec @ FP16 mixed precision (RTX 3060 Ti)
- **Coordination Overhead Budget**: 20-25% (currently 67%, target reduction critical)
- **REVISED REALISTIC TARGET**: **≥8,000 sims/sec** (2.1× baseline, 3.7× current)
- **Stretch Goal**: ≥10,000 sims/sec (2.6× baseline, requires perfect tuning)
- **Aspirational**: ≥15-25k sims/sec (requires model pruning, multi-GPU, changes out of scope)

**Rationale**: Per SPECIFICATION.md Section 12.1 Q1 resolution, GPU @ FP16 physically cannot exceed 10k states/sec. All targets updated to be hardware-grounded and achievable within constitutional constraints.

## Related Documents
- **[CONSTITUTION.md](CONSTITUTION.md)**: Project authority, constraints, enforcement (**v1.1 revised targets**)
- **[SPECIFICATION.md](SPECIFICATION.md)**: Detailed functional requirements, **resolved ambiguities** (Section 12)
- **[TECHNICAL_PLAN.md](TECHNICAL_PLAN.md)**: **Current implementation plan** (v2.0, hardware-grounded)
- **[plan-LEGACY-v1.0.md](plan-LEGACY-v1.0.md)**: Legacy plan (superseded by TECHNICAL_PLAN.md)
- **[TASKS.md](TASKS.md)**: Current Phase 4 task breakdown with validation status
- **[tasks-phases-1-3-ARCHIVE.md](tasks-phases-1-3-ARCHIVE.md)**: Historical Phase 1-3 tasks (archived)

## Current Progress (Updated 2025-10-09 after review.pdf analysis)

### ✅ Completed Tasks (Phase 1 + Phase 2 Major Components)

**Phase 1: Virtual Loss & Quick Wins** - ✅ **COMPLETE**
- T001: WU-UCT virtual loss implementation ✅
- T001b: Epoch-based tree clearing ✅ (was already complete, instant <1μs clear)
- T002: Busy-edge masking ✅
- T003: Root pre-expansion ✅
- T004: Thread affinity for Ryzen 5900X ✅
- T005: Collision metrics instrumentation ✅

**Phase 2: Architecture Changes** - ✅ **100% COMPLETE** (2025-10-09)
- T006: Lock-free MPMC queue implementation ✅
- T006b: Lock-free AsyncInferenceQueue integration ✅
- **T006c: Replace polling with condition variables** ✅ **COMPLETE** (commit 2253a97)
- T007: DLPack Tensor Bridge ✅ (T007a-g complete)
  - T007a: DLPack spec research and API design ✅
  - T007b: Pinned memory buffer allocation ✅
  - T007c: DLPack tensor capsule structure ✅
  - T007d: Batch tensor creation ✅
  - T007e: Direct feature extraction (all games) ✅
  - T007f: Python bindings ✅
  - T007g: Validation and benchmarking ✅
- T008: Python Inference Bridge ✅ (T008a-b,e,f complete, T008c-d skipped)
  - T008a: DLPackInferenceBridge class interface design ✅
  - T008b: torch.from_dlpack() conversion ✅
  - T008c: Pre-allocate GPU buffers ⏭️ (skipped - not needed)
  - T008d: Non-blocking GPU transfers ⏭️ (skipped - already async)
  - T008e: Integration testing and validation ✅
  - **T008f: Enable FP16 mixed precision** ✅ **COMPLETE** (commit 2253a97)
- T009: Per-Thread Memory Arenas ✅ (T009a-f complete)
  - T009a: ThreadLocalArena architecture design ✅
  - T009b: Arena data structure implementation ✅
  - T009c: Lock-free allocation ⏭️ (skipped - thread-local eliminates need)
  - T009d: Free list management ✅
  - T009e: MCTS tree integration (pragmatic: 4096-block allocation) ✅
  - T009f: Validation and benchmarking ✅
- T010: Replace pending expansions map with ring buffer ✅

### ✅ Critical Optimizations COMPLETE (from review.pdf)

**1. T006c: Condition Variables** ✅ **COMPLETE** (review.pdf pages 8-9)
- **Problem**: Current polling wastes 67% of CPU time
- **Solution**: Use `std::condition_variable` for efficient blocking
- **Impact**: **1.3-1.5× throughput improvement**
- **Status**: COMPLETE (2025-10-09, commit 2253a97)

**2. T008f: FP16 Mixed Precision** ✅ **COMPLETE** (review.pdf pages 8, 13)
- **Problem**: Not validated that FP16 is enabled
- **Solution**: Validate `torch.cuda.amp.autocast()` and benchmark
- **Impact**: **1.5-2× GPU inference speedup** (RTX 3060 Ti has tensor cores)
- **Status**: COMPLETE (2025-10-09, commit 2253a97)

### 🟡 Phase 3: Final Optimizations (20% Complete)
- ✅ T011: Persistent coordinator lifecycle (T011a-c complete, commit 28f11ca)
- T012: Relaxed memory ordering
- T013: Selection prefetching
- ✅ T014: Batched result processing (complete)
- T015: Hot/cold child separation

### ⏸️ Phase 4: Integration & Tuning (Not Started)
- T016: Performance benchmark suite 🔴 **HIGH PRIORITY**
- T017: A/B testing framework
- T018: Tune virtual loss magnitude
- T019: Optimize batch size and timeout
- T020: Profile and fix remaining bottlenecks
- T021-T025: Validation and documentation

### Key Achievements
- **Zero-Copy DLPack Pipeline**: Complete end-to-end (C++ → PyTorch)
- **Lock-Free Infrastructure**: Eliminated mutexes from hot paths (but still polling)
- **Thread-Local Arenas**: 99.93% fast-path allocation, 4096-node blocks
- **Enhanced Feature Extraction**: Zero-copy for Gomoku, optimized for Chess/Go
- **Critical Bug Fixes**: Coordinator lifecycle, result-stealing, thread-local caching
- **Memory Efficiency**: 1MB queue, 270MB tree (10M nodes), 1MB DLPack buffers

### Performance Status

**CRITICAL STATUS UPDATE (2025-10-12)**:
- **Current Regression**: 2,147 sims/sec (56% of baseline, cause UNKNOWN)
- **Baseline**: 3,831 sims/sec (configuration unknown, needs investigation)
- **Phase 1+2 Complete**: All critical optimizations implemented (T001-T010, T006c, T008f)
- **Phase 3 Partial**: T011 (coordinator lifecycle) + T014 (batched results) complete
- **Expected (with T006c+T008f)**: 18-36k sims/sec (theoretical, UNVALIDATED)
- **Actual Performance**: **UNKNOWN** - T016 benchmarking CRITICAL NEXT STEP
- **Blockers**: T017 (find baseline config) + T016 (measure actual gains) required
- **Target**: ≥25,000 sims/sec (path unclear until T016/T017 complete)

### Validation Status (2025-10-13)

**T-VALID-1: FP16 Mixed Precision Validation** ✅ **PASS**
- Speedup: 1.72× (52.83ms → 30.69ms @ batch-64)
- Policy Probability MSE: 0.000007 (target: <0.01)
- Value MSE: 0.000000 (target: <0.01)
- **Conclusion**: FP16 delivers significant speedup without quality degradation

**T-VALID-2: Tensor Creation Profiling** ❌ **FAIL**
- Mean: 7.50 ± 0.20 ms (target: <1.0ms)
- Root Cause: Feature extraction loop NOT parallelized with OpenMP
- Location: `cpp_extensions/mcts/dlpack_bridge.cpp:431-434`
- **Required Fix**: Add `#pragma omp parallel for` to feature extraction loop
- Expected Improvement: 7.5ms → <1.0ms with 12-thread parallelization

See [validation_report_2025-10-13.md](../../docs/performance/validation_report_2025-10-13.md) for detailed results.

### Critical Path Forward
1. ✅ **~~T006c~~** - Condition variables (COMPLETE)
2. ✅ **~~T008f~~** - FP16 mixed precision (COMPLETE, validated 1.72× speedup)
3. ✅ **~~T011~~** - Persistent coordinator (COMPLETE)
4. 🔴 **OpenMP Fix** - Parallelize feature extraction (CRITICAL, blocks performance)
5. 🔴 **T016** - Comprehensive benchmarking to measure actual gains (2 days) **NEXT PRIORITY**
6. **T018-T019** - Optimize batch size, timeout, virtual loss (2 days)
7. **T020** - Profile and fix remaining bottlenecks (2 days)
8. **T025** - Final performance validation and sign-off (1 day)

---

## Problem Statement

The current MCTS implementation achieves only **3,831 simulations/second** (baseline) on AMD Ryzen 5900X + RTX 3060 Ti hardware, with recent regression to **2,147 sims/sec** (cause under investigation). Performance analysis reveals that GPU inference accounts for only 32.8% of runtime while MCTS overhead (selection, backup, coordination) consumes 67.2%.

**Revised Target (2025-10-13)**: After hardware analysis, the RTX 3060 Ti @ FP16 can deliver 8,000-10,000 states/sec maximum. This specification defines optimizations to achieve **≥8,000 simulations/second** (realistic hardware-grounded target) while maintaining search quality. The original 25k-30k target would require model pruning or multi-GPU setup (out of scope).

## Success Criteria

### Performance Requirements
- **Primary**: Achieve ≥8,000 simulations/second on target hardware (revised 2025-10-13)
- **Minimum Viable**: ≥6,000 sims/sec (1.6× baseline, 2.8× current regression)
- **Stretch Goal**: ≥10,000 sims/sec (2.6× baseline, perfect tuning required)
- **GPU Utilization**: Maintain ≥80% GPU utilization during search (85% stretch)
- **Thread Efficiency**: Achieve ≥75% multi-thread efficiency at 8 threads
- **Batch Size**: Average batch size ≥48 positions (75% of max 64)
- **Memory Usage**: Tree memory <1GB for 10M nodes

### Quality Requirements
- **Search Quality**: Win rate vs baseline ≥99.5% (no strength regression)
- **Policy Agreement**: Top move agreement with baseline ≥95%
- **Value Accuracy**: Value MSE vs baseline ≤0.01
- **Collision Rate**: Path collision rate ≤5% (threads selecting same node)

### Compatibility Requirements
- **API Stability**: Maintain backward compatibility with existing interfaces
- **Python Version**: Support Python 3.12
- **PyTorch Version**: Support PyTorch 2.0+
- **Platform**: Linux (Ubuntu 22.04) with CUDA 12.1

## Root Cause Analysis

### Critical Bottlenecks Identified

1. **Python Overhead (60-70% of runtime)**
   - State-to-tensor conversion in Python loops
   - GIL acquisition for every batch inference
   - ThreadPoolExecutor coordination overhead
   - Policy list conversion (numpy → Python)

2. **Async Queue Inefficiency (67% of MCTS overhead)**
   - Busy-wait polling with microsecond sleeps
   - std::unordered_map for pending expansions
   - Per-result mutex-protected operations
   - No proper condition variable usage

3. **Threading Bottlenecks**
   - Root expansion serialization (N-1 threads idle)
   - Virtual loss causing Q-value distortion
   - Cross-CCD cache line bouncing (Ryzen specific)
   - No thread affinity configuration

4. **Memory Management Issues**
   - Tree clearing overhead (memset 270MB every search)
   - Global allocation mutex contention
   - Atomic contention on hot cache lines
   - Inefficient memory layout for SIMD
   - Thread oversubscription in self-play

## Solution Architecture

### Core Design: Optimized Shared Tree with WU-UCT

Maintain single shared tree architecture with enhanced virtual loss:
- **WU-UCT Style**: Visit-only virtual loss (no Q-value distortion)
- **Busy-Edge Masking**: Prevent selection of nodes being expanded
- **Root Pre-Expansion**: Expand root before launching threads
- **Lock-Free Coordination**: MPMC ring buffer for queue operations

### Key Optimizations

#### 1. Virtual Loss Enhancement (1.5× speedup)
- Implement WU-UCT visit-only accounting
- Add busy-edge masking during selection
- Tune virtual loss magnitude (1.0 default)

#### 2. Python Overhead Elimination (2.5× speedup)
- Zero-copy DLPack tensor bridge
- Persistent Python inference thread (holds GIL)
- Direct numpy array returns (no list conversion)

#### 3. Lock-Free Queue (1.4× speedup)
- MPMC ring buffer implementation
- Condition variable wait/notify pattern
- Batched result processing

#### 4. Thread Optimization (1.3× speedup)
- Thread affinity for Ryzen CCDs
- Per-thread memory arenas
- Root pre-expansion strategy

#### 5. Memory Optimization (1.2× speedup)
- Epoch-based lazy tree clearing (avoid 270MB memset)
- Relaxed atomic memory ordering
- Cache-line aligned data structures
- SIMD-friendly memory layout
- Shared thread pool for self-play

## Implementation Phases

### Phase 1: Virtual Loss & Quick Wins (Week 1)
- WU-UCT implementation
- Busy-edge masking
- Root pre-expansion
- Thread affinity

**Expected: 2.1k → 4k sims/sec (2× improvement, revised 2025-10-13)**

### Phase 2: Architecture Changes (Week 2)
- Lock-free queue implementation
- Zero-copy tensor bridge (with OpenMP fix)
- Per-thread memory arenas
- FP16 mixed precision (validated: 1.72× GPU speedup)

**Expected: 4k → 7k sims/sec (1.75× improvement, revised 2025-10-13)**

### Phase 3: Final Optimizations (Week 3)
- Persistent Python thread
- Relaxed memory ordering
- Performance tuning
- Batch size/timeout optimization

**Expected: 7k → 8-10k sims/sec (1.2-1.4× improvement, target range, revised 2025-10-13)**

## Validation Strategy

### Performance Testing
```bash
# Throughput benchmark
python scripts/test_mcts.py --game gomoku --simulations 10000 --threads 8

# GPU utilization monitoring
nvidia-smi dmon -s u -i 0

# Thread efficiency analysis
perf stat -e task-clock,context-switches,cpu-migrations python scripts/test_mcts.py
```

### Quality Testing
```bash
# A/B test against baseline
python scripts/compare_search_quality.py --baseline v003 --candidate v004

# Policy agreement test
python scripts/test_policy_agreement.py --threshold 0.95

# Value accuracy test
python scripts/test_value_mse.py --threshold 0.01
```

### Collision Metrics
```cpp
// Instrumentation to add
struct CollisionMetrics {
    std::atomic<uint64_t> selection_retries{0};
    std::atomic<uint64_t> duplicate_paths{0};
    std::atomic<uint64_t> unique_batch_positions{0};
    std::atomic<uint64_t> expansion_conflicts{0};
};
```

## Risk Analysis

### High Risks
1. **WU-UCT Changes Break Search Quality**
   - Mitigation: Incremental testing, A/B comparison
   - Fallback: Keep classic virtual loss option

2. **Lock-Free Queue Introduces Bugs**
   - Mitigation: Use proven library, extensive testing
   - Fallback: Optimized mutex-based queue

3. **DLPack Incompatibility**
   - Mitigation: Version testing, numpy fallback
   - Fallback: Optimized copy path

### Medium Risks
4. **Thread Affinity Portability**
   - Mitigation: Platform detection, optional feature
   - Impact: Limited to Ryzen users

5. **Memory Ordering Bugs**
   - Mitigation: TSan testing, conservative defaults
   - Fallback: Sequential consistency

## Acceptance Criteria

### Implementation Completeness
- [✅] Phase 1: Virtual loss & quick wins (T001-T005 complete)
- [🔄] Phase 2: Architecture changes (T006/T006b/T010 complete, T007/T008/T009 split and pending)
- [ ] Phase 3: Final optimizations (not started)

### Minimum Viable Performance
- [ ] ≥20,000 simulations/second (67% of target)
- [ ] ≥80% GPU utilization
- [ ] ≤10% path collision rate
- [✅] No search quality regression (validated through Phase 1)

### Target Performance
- [ ] ≥25,000 simulations/second (83% of target)
- [ ] ≥85% GPU utilization
- [✅] ≤5% path collision rate (instrumentation complete)
- [ ] <1% performance variance

### Stretch Goals
- [ ] ≥30,000 simulations/second (100% of target)
- [ ] ≥90% GPU utilization
- [ ] ≤2% path collision rate
- [ ] Support for 16+ threads

### Technical Quality
- [✅] All async integration tests passing (11/11)
- [✅] Thread safety verified (TSan clean)
- [✅] Lock-free queue correctness validated (19/19 unit tests)
- [✅] Memory usage within targets (<1GB for tree, 1MB for queue)
- [ ] DLPack zero-copy verified
- [ ] Arena allocation 10× faster than malloc
- [ ] End-to-end performance benchmarked

## Dependencies

### External Libraries
- `pybind11` ≥2.10.0 (DLPack support)
- `boost::lockfree` (optional, for queue)
- `pytorch` ≥2.0 (DLPack compatible)

### Internal Components
- Tree structure (SoA layout maintained)
- Game interface (unchanged)
- Neural network API (extended for DLPack)

## Timeline

- **Week 1**: Virtual loss optimizations + quick wins
- **Week 2**: Lock-free queue + zero-copy tensors
- **Week 3**: Final optimizations + tuning
- **Week 4**: Validation + documentation

**Total Duration**: 4 weeks
**Expected Outcome**: 26,000+ sims/sec (87% of target)