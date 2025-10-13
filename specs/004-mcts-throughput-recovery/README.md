# Specification 004: MCTS Throughput Recovery

## Executive Summary

This specification addresses the critical performance bottleneck in our MCTS implementation. **Baseline performance: 3,831 sims/sec. Current performance: 2,147 sims/sec (56% REGRESSION, cause under investigation)**. Through comprehensive analysis and optimization, we define a path to achieve **≥8,000 simulations/second (REVISED, hardware-grounded)** while maintaining search quality.

**⚠️ CRITICAL UPDATE (2025-10-13)**: **Targets revised to 6-10k sims/sec** based on review.txt hardware analysis. Original 25-30k target unrealistic on Ryzen 5900X + RTX 3060 Ti (GPU caps at 8-10k states/sec @ FP16). See CONSTITUTION.md v1.1 and SPECIFICATION.md Section 12.1 Q1 for detailed rationale.

**Key Finding**: GPU inference accounts for only 32.8% of runtime while MCTS overhead consumes 67.2%, indicating the bottleneck is CPU-side coordination, not neural network evaluation.

**Validation Status (2025-10-13)**: Phase 1+2 optimizations VALIDATED with **MIXED RESULTS**:
- ✅ **T-VALID-1 (FP16)**: PASS - 1.72× speedup confirmed, numerical stability excellent
- ❌ **T-VALID-2 (Tensor)**: FAIL - 7.5ms overhead (missing OpenMP parallelization)
- 🔴 **CRITICAL BLOCKER**: Feature extraction not parallelized, caps throughput at ~1,675 states/sec
- 📋 **Next Action**: Fix OpenMP in [dlpack_bridge.cpp:431-434](../../cpp_extensions/mcts/dlpack_bridge.cpp#L431-L434)
- 📄 **Report**: [validation_report_2025-10-13.md](../../docs/performance/validation_report_2025-10-13.md)

## Document Structure

### Project Authority & Requirements

1. **[CONSTITUTION.md](CONSTITUTION.md)** - **PROJECT AUTHORITY** (v1.1 revised)
   - Mission and scope (**≥8,000 sims/sec target**, revised from 25k)
   - Architectural constraints (Python PyTorch NN only, NO libtorch/TensorRT)
   - Technical principles (throughput > quality with bounds)
   - Performance requirements (CPU efficiency, GPU utilization, latency budgets)
   - Decision authority and change control
   - **NOTE**: This document supersedes all prior architectural decisions

2. **[SPECIFICATION.md](SPECIFICATION.md)** - **DETAILED FUNCTIONAL REQUIREMENTS**
   - Precise definitions (simulation, throughput measurement methodology)
   - User stories (self-play training, interactive play, profiling)
   - 17 functional requirements (MCTS algorithm, async inference, memory management)
   - Non-functional requirements (performance by game, scalability, correctness)
   - KPIs and benchmarks (8 KPIs with exact measurement commands)
   - **Section 12: Resolved Ambiguities** (10 blocking questions with protocols)

### Core Specifications

3. **[spec.md](spec.md)** - Implementation Status and Progress Tracking (v1.2)
   - **Performance targets: ≥8,000 sims/sec** (revised), ≥80% GPU utilization
   - Quality requirements: No regression in search strength
   - Phased implementation approach over 4 weeks
   - Current progress tracking (Phase 1 ✅, Phase 2 ✅, Phase 3 🟡, Phase 4 🔴)

4. **[TECHNICAL_PLAN.md](TECHNICAL_PLAN.md)** - **CURRENT IMPLEMENTATION PLAN** (v2.0)
   - Architecture overview with hardware mapping (Ryzen 5900X CCDs)
   - Data contracts & interfaces (DLPack, SoA, thread-local pools)
   - Performance tactics from review.txt (idle reduction, atomic minimization)
   - Instrumentation & benchmarking (timing hooks, fixed-seed suites)
   - Rollout & risk management (feature flags, test gates)
   - **Acceptance criteria with exact commands and expected outputs**
   - **Migration checklist with 7-8 day timeline**

5. **[plan.md](plan.md)** - Legacy Technical Plan (superseded by TECHNICAL_PLAN.md)
   - Retained for historical reference
   - Original 25k target and architecture design
   - See TECHNICAL_PLAN.md for current implementation

6. **[tasks.md](tasks.md)** - Detailed Task Breakdown
   - 25 concrete implementation tasks
   - Dependencies and risk analysis
   - Week-by-week milestones

### Technical Documentation

7. **[research.md](research.md)** - Technical Analysis
   - Root cause analysis of bottlenecks
   - State-of-the-art comparison (KataGo, Leela Zero)
   - Virtual loss research (WU-UCT vs classic)
   - Experimental results and benchmarks

8. **[data-model.md](data-model.md)** - Data Structure Specifications
   - Lock-free queue implementation
   - DLPack tensor bridge design
   - Thread-local memory arenas
   - Cache-optimized layouts

### API Contracts

9. **[contracts/](contracts/)** - API Specifications
   - [virtual_loss_api.yaml](contracts/virtual_loss_api.yaml) - WU-UCT manager
   - [queue_api.yaml](contracts/queue_api.yaml) - Lock-free queue
   - [dlpack_api.yaml](contracts/dlpack_api.yaml) - Zero-copy tensors
   - [thread_api.yaml](contracts/thread_api.yaml) - Thread management
   - [inference_api.yaml](contracts/inference_api.yaml) - Complete pipeline

### Implementation Guide

10. **[quickstart.md](quickstart.md)** - Build and Validation
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

## Performance Targets (REVISED 2025-10-13)

| Metric | Baseline | Current | **Target (Revised)** | **Rationale** |
|--------|----------|---------|---------------------|---------------|
| Throughput | 3,831 sims/sec | 2,147 sims/sec | **≥8,000 sims/sec** | GPU @ FP16 caps at 8-10k states/sec |
| GPU Utilization | 32.8% | 11.2% | **≥80%** | Realistic batch fill @ 64 |
| Collision Rate | 18.2% | UNKNOWN | ≤5% | WU-UCT + busy-edge |
| Thread Efficiency | 42% @ 4 threads | 62% @ 4 threads | **≥70% @ 8 threads** | Coordination overhead budget |
| Batch Size | Variable | UNKNOWN | ≥40 avg (62.5%) | Adjusted from 75% expectation |
| Memory Usage | <1GB | 270MB | <1GB | Achieved (27B/node SoA) |

**Stretch Goal**: ≥10,000 sims/sec (2.6× baseline, requires perfect tuning)
**Aspirational** (out of scope): ≥15-25k (model pruning, multi-GPU), ≥30k (TensorRT, fundamental redesign)

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

## Implementation Status (Updated 2025-10-12)

- [✅] Specification complete
- [✅] Technical research complete
- [✅] API contracts defined
- [✅] Data models specified
- [✅] Implementation plan ready
- [✅] **Phase 1 implementation COMPLETE** (T001-T005)
  - WU-UCT virtual loss ✅
  - Epoch-based tree clearing ✅ (1M× speedup)
  - Busy-edge masking ✅
  - Root pre-expansion ✅
  - Thread affinity ✅
  - Collision metrics ✅
- [✅] **Phase 2 implementation COMPLETE** (T006-T010) **UNVALIDATED**
  - Lock-free MPMC queue ✅ (T006)
  - AsyncInferenceQueue integration ✅ (T006b)
  - Condition variables ✅ (T006c, commit 2253a97) **UNVALIDATED**
  - DLPack tensor bridge ✅ (T007a-g complete)
  - Python inference bridge ✅ (T008a-b,e,f ✅; T008c-d skipped)
  - FP16 mixed precision ✅ (T008f, commit 2253a97) **UNVALIDATED**
  - Per-thread memory arenas ✅ (T009a-f complete)
  - Pending expansions map replaced ✅ (T010)
- [🟡] **Phase 3 implementation 80% COMPLETE** (T011-T015)
  - Persistent coordinator lifecycle ✅ (T011)
  - Batched result processing ✅ (T014)
  - T012/T013/T015 pending
- [🟡] **Phase 4 VALIDATION COMPLETE** (2025-10-13)
  - ✅ **T-VALID-1**: FP16 validated - 1.72× speedup, numerical stability excellent
  - ❌ **T-VALID-2**: Tensor creation FAIL - 7.5ms (OpenMP missing)
  - 🔴 **BLOCKER FOUND**: Feature extraction not parallelized in dlpack_bridge.cpp
  - Validation scripts executed (profile_tensor_creation.py, validate_fp16_inference.py)
  - Comprehensive report: [validation_report_2025-10-13.md](../../docs/performance/validation_report_2025-10-13.md)

## Critical Path Forward (UPDATED 2025-10-13)

**✅ IMMEDIATE VALIDATIONS COMPLETE** (2 hours - 2025-10-13):

1. ✅ **T-VALID-1: FP16 Mixed Precision** - **PASS**
   ```bash
   python scripts/validate_fp16_inference.py --batch-size 64 --iterations 100
   ```
   - **Result**: 1.72× speedup (FP32: 52.8ms → FP16: 30.7ms per batch-64)
   - **Numerical Stability**: Policy Prob MSE 0.000007, Value MSE 0.000000
   - **Conclusion**: T008f validated successfully, FP16 tensor cores active

2. ❌ **T-VALID-2: Tensor Creation** - **FAIL (CRITICAL BLOCKER)**
   ```bash
   python scripts/profile_tensor_creation.py --batch-size 64 --iterations 1000
   ```
   - **Result**: 7.50 ± 0.20 ms (FAIL: >1.0ms target)
   - **Root Cause**: Feature extraction loop NOT parallelized with OpenMP
   - **Location**: [dlpack_bridge.cpp:431-434](../../cpp_extensions/mcts/dlpack_bridge.cpp#L431-L434)
   - **Impact**: Caps throughput at ~1,675 states/sec, explains 2,147 regression

---

**🔴 CRITICAL FIX REQUIRED** (2 hours, BLOCKING):

**Fix OpenMP Parallelization in Tensor Creation**
```cpp
// cpp_extensions/mcts/dlpack_bridge.cpp:431
#pragma omp parallel for schedule(static) if(batch_size > 8)
for (int i = 0; i < batch_size; ++i) {
    float* state_buffer = data + (i * state_size);
    states[i]->extract_features_to_buffer(state_buffer);
}
```
- **Expected**: 7.5ms → <1.0ms (12-thread Ryzen 5900X)
- **Rebuild**: `pip install -e . --force-reinstall --no-deps`
- **Re-validate**: Should PASS with <1.0ms

---

**Performance Regression Investigation (After Critical Fix):**

**Current Status**: Phase 1+2 optimizations implemented but performance REGRESSED from 3,831 to 2,147 sims/sec (44% loss). Must validate FP16 and DLPack FIRST to rule out critical bugs.

**Next Steps:**

3. **T017: Baseline Configuration Investigation** (2 days, time-boxed)
   - **Problem**: 3,831 baseline configuration unknown, can't reproduce
   - **Protocol**: Day 1: Git archaeology (configs, logs, profiling). Day 2: Grid search if not found.
   - **Fallback**: Establish new baseline from grid search, declare 3,831 "historical"
   - **Status**: Protocol defined in SPECIFICATION.md Section 12.1 Q2

4. **T016: Run Comprehensive Benchmarks** (2 days)
   - **Prerequisite**: Complete T017 first (need baseline for comparison)
   - **Action**: Full benchmark suite (games × threads × batch sizes × timeouts)
   - **Goal**: Measure actual performance vs baseline
   - **Expected**: 6-8k sims/sec (if FP16 + DLPack working, revised from 18-36k)

5. **T018-T019: Tune Parameters** (2 days)
   - **Prerequisite**: Complete T016 first
   - **Protocol**: Grid search [4-12 threads] × [16-64 batch] × [0.5-5.0ms timeout]
   - **Goal**: Find optimal config for ≥8k sims/sec target
   - **Status**: Protocol defined in SPECIFICATION.md Section 12.2 Q6/Q7

## Success Criteria (REVISED 2025-10-13)

The project will be considered successful when:
- ✅ Achieves **≥8,000 simulations/second** sustained (was 25k, revised to hardware-grounded target)
- ✅ Maintains **≥80% GPU utilization** (was 85%, adjusted for realistic batch fill)
- ✅ Reduces collision rate to ≤5%
- ✅ Shows no regression in search quality (≥99.5% win rate vs baseline)
- ✅ Passes 24-hour stability test

**Stretch Goals**:
- ≥10,000 sims/sec (2.6× baseline, requires perfect tuning)
- ≥85% GPU utilization (near-perfect batch scheduling)

## References

- [Performance Analysis (Review)](../../review.pdf)
- [MCTS Implementation Guide](../../mcts_guide.md)
- [Project Guidelines](../../CLAUDE.md)
- Original target: 30,000-40,000 sims/sec on Ryzen 5900X + RTX 3060 Ti

## Contact

For questions or clarifications about this specification, refer to the technical documentation or review the implementation plan.

## Document Hierarchy

**Primary Authority**: [CONSTITUTION.md](CONSTITUTION.md) → [SPECIFICATION.md](SPECIFICATION.md) → **[TECHNICAL_PLAN.md](TECHNICAL_PLAN.md)** → [tasks.md](tasks.md)

**Change Control**: All changes to scope, requirements, or architecture must update CONSTITUTION.md and SPECIFICATION.md first, then regenerate TECHNICAL_PLAN.md and tasks.md via `/speckit.plan` and `/speckit.tasks`.

**Note**: [plan.md](plan.md) is legacy (superseded by TECHNICAL_PLAN.md v2.0 with revised 8k target).

---

**Document Version**: 1.5.0 (Revised Targets)
**Last Updated**: 2025-10-13
**Status**: Phase 1+2 Complete but UNVALIDATED - **IMMEDIATE**: Validate FP16 + Profile tensor creation (2 hours) BEFORE T016/T017

**Key Changes in v1.5**:
- Revised realistic targets: 8k (target), 10k (stretch), 25-30k (aspirational, out of scope)
- Added immediate validation actions (FP16, tensor creation) before benchmarking
- Updated all performance expectations based on GPU hardware limits (RTX 3060 Ti @ FP16)
- Clarified baseline investigation protocol (T017: 2-day time-boxed)
- Defined parameter tuning grid search protocols (T018/T019)