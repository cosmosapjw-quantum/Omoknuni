# Specification 004: MCTS Throughput Recovery

## Executive Summary

This specification addresses the critical performance bottleneck in our MCTS implementation, which currently achieves only **3,831 simulations/second (12.8% of the 30,000 target)**. Through comprehensive analysis and optimization, we define a path to achieve **≥25,000 simulations/second** while maintaining search quality.

**Key Finding**: GPU inference accounts for only 32.8% of runtime while MCTS overhead consumes 67.2%, indicating the bottleneck is CPU-side coordination, not neural network evaluation.

## Document Structure

### Core Specifications

1. **[spec.md](spec.md)** - Requirements and Success Criteria
   - Performance targets: ≥25,000 sims/sec, ≥85% GPU utilization
   - Quality requirements: No regression in search strength
   - Phased implementation approach over 4 weeks

2. **[plan.md](plan.md)** - Technical Implementation Plan
   - WU-UCT virtual loss (preserves Q-value accuracy)
   - Lock-free MPMC queue architecture
   - Zero-copy DLPack tensor bridge
   - Thread affinity optimization for Ryzen 5900X

3. **[tasks.md](tasks.md)** - Detailed Task Breakdown
   - 25 concrete implementation tasks
   - Dependencies and risk analysis
   - Week-by-week milestones

### Technical Documentation

4. **[research.md](research.md)** - Technical Analysis
   - Root cause analysis of bottlenecks
   - State-of-the-art comparison (KataGo, Leela Zero)
   - Virtual loss research (WU-UCT vs classic)
   - Experimental results and benchmarks

5. **[data-model.md](data-model.md)** - Data Structure Specifications
   - Lock-free queue implementation
   - DLPack tensor bridge design
   - Thread-local memory arenas
   - Cache-optimized layouts

### API Contracts

6. **[contracts/](contracts/)** - API Specifications
   - [virtual_loss_api.yaml](contracts/virtual_loss_api.yaml) - WU-UCT manager
   - [queue_api.yaml](contracts/queue_api.yaml) - Lock-free queue
   - [dlpack_api.yaml](contracts/dlpack_api.yaml) - Zero-copy tensors
   - [thread_api.yaml](contracts/thread_api.yaml) - Thread management
   - [inference_api.yaml](contracts/inference_api.yaml) - Complete pipeline

### Implementation Guide

7. **[quickstart.md](quickstart.md)** - Build and Validation
   - Step-by-step build instructions
   - Configuration examples
   - Validation procedures
   - Troubleshooting guide

## Optimization Strategy

### Phase 1: Quick Wins (Week 1)
- WU-UCT virtual loss implementation
- **Epoch-based tree clearing** (avoid 270MB memset overhead)
- Busy-edge masking for collision avoidance
- Root pre-expansion to eliminate startup bottleneck
- Thread affinity for Ryzen CCDs

**Expected: 3.8k → 12k sims/sec (3× improvement)**

### Phase 2: Architecture Changes (Week 2)
- Lock-free MPMC queue replacing mutex-based queue
- DLPack zero-copy tensor bridge
- Per-thread memory arenas

**Expected: 12k → 20k sims/sec (5.2× total)**

### Phase 3: Final Optimizations (Week 3)
- Persistent Python thread (holds GIL)
- Relaxed memory ordering
- Batch size and timeout tuning

**Expected: 20k → 26k sims/sec (6.8× total)**

### Phase 4: Validation & Tuning (Week 4)
- Extended soak testing
- A/B quality comparison
- Performance documentation

## Key Innovations

### 1. WU-UCT Virtual Loss
Traditional virtual loss distorts Q-values during selection, causing suboptimal choices with many threads. WU-UCT only adjusts the exploration term:

```cpp
// Classic VL (distorts Q)
q_distorted = (total_value - virtual_loss) / (visits + in_flight)

// WU-UCT (preserves Q)
q_true = total_value / visits
exploration_adjustment = in_flight * vl_magnitude
```

### 2. Lock-Free Queue
Replaces mutex-protected `std::unordered_map` with MPMC ring buffer:
- 10× reduction in queue overhead
- Wait-free single producer/consumer
- Predictable latency

### 3. Zero-Copy Pipeline
Eliminates Python data conversion overhead:
```
C++ State → DLPack Tensor → PyTorch GPU → Results → C++
```
Total copies: 0 (only GPU transfers)

### 4. Thread Affinity (Ryzen 5900X)
- MCTS threads → CCD0 (shared L3 cache)
- Inference thread → CCD1 (isolated)
- 15-20% cache miss reduction

## Performance Targets

| Metric | Current | Target | Expected |
|--------|---------|--------|----------|
| Throughput | 3,831 sims/sec | 25,000 | 26,000 |
| GPU Utilization | 32.8% | 85% | 85% |
| Collision Rate | 18.2% | ≤5% | 4.7% |
| Thread Efficiency | 42% @ 4 threads | 75% @ 8 | 78% |
| Batch Size | Variable | 48 avg | 48 |
| Memory Usage | <1GB | <1GB | ~500MB |

## Quality Guarantees

- **No algorithmic changes** to MCTS logic
- **Policy agreement** ≥95% with baseline
- **Value MSE** ≤0.01
- **Win rate** vs baseline ≥99.5%

## Risk Mitigation

### High Risk Components
1. **Lock-free queue**: Fallback to optimized mutex version
2. **DLPack bridge**: Fallback to numpy conversion
3. **WU-UCT convergence**: Tunable magnitude parameter

### Testing Strategy
- Thread sanitizer validation
- 24-hour soak tests
- A/B quality comparison
- Gradual rollout (10% → 100%)

## Implementation Status

- [✅] Specification complete
- [✅] Technical research complete
- [✅] API contracts defined
- [✅] Data models specified
- [✅] Implementation plan ready
- [✅] **Phase 1 implementation COMPLETE** (T001-T005)
  - WU-UCT virtual loss ✅
  - Busy-edge masking ✅
  - Root pre-expansion ✅
  - Thread affinity ✅
  - Collision metrics ✅
- [🔄] **Phase 2 implementation IN PROGRESS** (T006-T010)
  - Lock-free MPMC queue ✅ (T006)
  - **Lock-free AsyncInferenceQueue integration** ✅ (T006b - Just Completed!)
  - Pending expansions map replaced ✅ (T010)
  - DLPack tensor bridge ⏸️ (T007 → split into T007a-g)
  - Python inference bridge ⏸️ (T008 → split into T008a-e)
  - Per-thread memory arenas ⏸️ (T009 → split into T009a-f)
- [ ] Phase 3 implementation (not started)
- [ ] Validation complete

## Next Steps

1. **Complete Phase 2 remaining tasks** (T007-T009 subtasks)
   - T007a-g: DLPack tensor bridge (7 subtasks, 4-6 hours each)
   - T008a-e: Python inference bridge (5 subtasks, 2-3 hours each)
   - T009a-f: Per-thread memory arenas (6 subtasks, 3-5 hours each)
2. **Benchmark Phase 1+2 combined performance**
3. **Begin Phase 3** tasks (T011-T013+)
4. **Full system validation** with quality and performance tests

## Success Criteria

The project will be considered successful when:
- ✅ Achieves ≥25,000 simulations/second sustained
- ✅ Maintains ≥85% GPU utilization
- ✅ Reduces collision rate to ≤5%
- ✅ Shows no regression in search quality
- ✅ Passes 24-hour stability test

## References

- [Performance Analysis (Review)](../../review.pdf)
- [MCTS Implementation Guide](../../mcts_guide.md)
- [Project Guidelines](../../CLAUDE.md)
- Original target: 30,000-40,000 sims/sec on Ryzen 5900X + RTX 3060 Ti

## Contact

For questions or clarifications about this specification, refer to the technical documentation or review the implementation plan.

---

**Document Version**: 1.1.0
**Last Updated**: 2025-10-08
**Status**: IN PROGRESS - Phase 2 Partially Complete (T006b just completed)