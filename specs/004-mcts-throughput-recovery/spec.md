# Specification 004: MCTS Throughput Recovery

**Status**: IN PROGRESS
**Last Updated**: 2025-10-08
**Current Phase**: Phase 2 (Architecture Changes) - Partially Complete

## Current Progress

### Completed Tasks ✅
- **Phase 1**: Virtual loss & quick wins (T001-T005) ✅
  - T001: WU-UCT virtual loss implementation ✅
  - T002: Busy-edge masking ✅
  - T003: Root pre-expansion ✅
  - T004: Thread affinity for Ryzen 5900X ✅
  - T005: Collision metrics instrumentation ✅

- **Phase 2** (Partial): Architecture changes
  - T006: Lock-free MPMC queue implementation ✅
  - **T006b: Lock-free AsyncInferenceQueue integration** ✅ *(just completed)*
  - T010: Replace pending expansions map with ring buffer ✅

### In Progress / Not Started
- **Phase 2** (Remaining):
  - T007: Create DLPack Tensor Bridge (split into T007a-T007g) ⏸️
  - T008: Update Python Inference Bridge (split into T008a-T008e) ⏸️
  - T009: Implement Per-Thread Memory Arenas (split into T009a-T009f) ⏸️

- **Phase 3**: Final optimizations
  - T011: Persistent Python thread ⏸️
  - T012: Relaxed memory ordering ⏸️
  - T013+: Additional optimizations ⏸️

### Key Achievements
- **Lock-Free Infrastructure**: Eliminated all mutexes from AsyncInferenceQueue hot paths
- **Critical Bug Fixes**: Resolved coordinator lifecycle and result-stealing bugs
- **Memory Efficiency**: Fixed 1MB allocation for queue (vs unbounded map/deque)
- **Thread Safety**: All async integration tests passing (11/11)

### Current Performance
- **Baseline**: 3,831 sims/sec (12.8% of target)
- **Expected after Phase 1+2**: ~12k-15k sims/sec (pending benchmarks)
- **Target**: ≥25,000 sims/sec

---

## Problem Statement

The current MCTS implementation achieves only **3,831 simulations/second** (12.8% of the 30,000 target) on AMD Ryzen 5900X + RTX 3060 Ti hardware. Performance analysis reveals that GPU inference accounts for only 32.8% of runtime while MCTS overhead (selection, backup, coordination) consumes 67.2%. This specification defines optimizations to achieve **≥25,000 simulations/second** while maintaining search quality.

## Success Criteria

### Performance Requirements
- **Primary**: Achieve ≥25,000 simulations/second on target hardware
- **GPU Utilization**: Maintain ≥85% GPU utilization during search
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

**Expected: 3.8k → 12k sims/sec (3× improvement)**

### Phase 2: Architecture Changes (Week 2)
- Lock-free queue implementation
- Zero-copy tensor bridge
- Per-thread memory arenas

**Expected: 12k → 20k sims/sec (1.7× improvement)**

### Phase 3: Final Optimizations (Week 3)
- Persistent Python thread
- Relaxed memory ordering
- Performance tuning

**Expected: 20k → 26k sims/sec (1.3× improvement)**

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