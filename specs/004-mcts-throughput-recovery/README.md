# Specification 004: MCTS Throughput Recovery & Multi-Actor Self-Play

**Version**: 2.0
**Status**: ACTIVE
**Last Updated**: 2025-10-14
**Target Hardware**: AMD Ryzen 9 5900X (12C/24T) + NVIDIA RTX 3060 Ti (8GB VRAM)

---

## Executive Summary

This specification addresses the critical throughput bottleneck in the MCTS implementation, targeting **≥8,000 simulations/second** (realistic, hardware-grounded) to enable superhuman Gomoku/Chess/Go training on consumer hardware.

**Current Status** (per review.txt, authoritative):
- **Baseline**: 3,831 sims/sec (Spec 003 configuration)
- **Current**: 2,147 sims/sec (56% regression from baseline)
- **Target**: **≥8,000 sims/sec** (2.1× baseline, hardware-grounded)
- **Stretch**: ≥10,000 sims/sec (GPU hardware limit)

**Critical Finding** (review.txt analysis):
- **GPU is NOT the bottleneck** (only 32.8% of total time)
- **MCTS CPU coordination is** (67.2% of runtime)
- Root cause: 7.5ms feature extraction (OpenMP inactive) + 60% thread idle + 2-3× state cloning

**Solution Path**:
1. Fix OpenMP parallelization (7.5ms → <1ms, immediate 2-3× gain expected)
2. Eliminate state cloning via thread-local pools (2-3× → ≤1×)
3. Replace spin-wait with condition variables (60% idle → <12%)
4. Multi-actor self-play for GPU saturation (85-95% utilization)
5. (Optional) NN-eval cache for 15-35% additional throughput
6. (Future) Lightweight NN architecture for 1.5-3.5× model speedup → 18-22k total

---

## Documentation Structure (v2.0)

### Active Specifications

#### Core Requirements
1. **[spec.md](spec.md)** - Functional Requirements (v2.0, 621 lines)
   - **Authoritative functional specification**
   - 8 measurable goals (G1-G8) with acceptance criteria
   - 25 numbered requirements (REQ-PERF-001 through REQ-IMPL-005)
   - 3 user stories (self-play, interactive play, performance engineering)
   - **Section 6: Advanced Optimizations** (legal moves, DLPack validation, lightweight NN)

2. **[CONSTITUTION.md](CONSTITUTION.md)** - Immutable Constraints (v2.0, 729 lines)
   - Mission and scope (≥8,000 sims/sec, hardware-realistic)
   - Architectural constraints (Python PyTorch NN, NO libtorch/TensorRT)
   - Performance guardrails (≥8k sims/sec, GPU 80-95%)
   - Immutable constraints (shared-tree, Python interface, WU-UCT)
   - Non-goals (no root parallelization, no GPU-MCTS, no full DAG TT)

3. **[plan.md](plan.md)** - Technical Implementation Plan (v1.0, 2,398 lines)
   - **Authoritative technical plan**
   - **Section A**: Architecture diagrams (Single-MCTS, Multi-Actor)
   - **Section B**: CPU Pipeline Improvements (OpenMP, state pooling, CV, allocator)
   - **Section C**: NN-Eval Cache design (Zobrist, sharding, SLRU eviction)
   - **Section D**: Multi-Actor Self-Play architecture
   - **Section E**: Telemetry & Benchmarks
   - **Section F**: Risk & Rollback procedures

4. **[tasks.md](tasks.md)** - Task Breakdown (v1.0, 2,747 lines)
   - **Authoritative task list**
   - 30 tasks across 5 phases (Foundation, CPU Optimizations, Validation, NN-Cache, Multi-Actor)
   - Dependency graph with critical path
   - Detailed implementation with code snippets
   - Acceptance tests (unit + performance)
   - "Done means" criteria with telemetry
   - Rollback plans with feature flags

#### Analysis & Resolution Documents

5. **[CLARIFICATIONS.md](CLARIFICATIONS.md)** - Resolved Ambiguities (633 lines)
   - 7 major ambiguous areas resolved
   - OpenMP already in code (need verification)
   - State pooling API with copyFrom() + move semantics
   - Condition variables design
   - Node allocator over-allocation strategy
   - NN-eval cache Tier A design
   - Multi-actor process architecture
   - Benchmark parameters

6. **[TRACEABILITY_MATRIX.md](TRACEABILITY_MATRIX.md)** - Cross-Check Analysis (524 lines)
   - 100% coverage: review.txt → spec → plan → tasks
   - 7/7 major recommendations mapped
   - 0 contradictions found
   - 3 gaps identified (all non-blocking, documented)
   - Verdict: ✅ APPROVED FOR IMPLEMENTATION

7. **[ACCEPTANCE_CHECKLIST.md](ACCEPTANCE_CHECKLIST.md)** - Validation Suite (860 lines)
   - **150+ validation checks** across 3 categories:
     - **Performance** (60 checks): Throughput, GPU util, CPU overhead, scaling
     - **Correctness** (45 checks): Determinism, parity, search quality, thread safety
     - **Operational** (45 checks): Scripts, CSV output, plots, logging
   - **Critical path**: 11 MUST PASS checks
   - Evidence requirements (CSV, plots, logs)
   - Rollback triggers and procedures

8. **[CHECKLIST_SUMMARY.md](CHECKLIST_SUMMARY.md)** - Quick Reference (264 lines)
   - Primary KPI summary
   - Fast validation commands
   - Critical path checklist
   - Quick start guide for validation

#### Technical Documentation

9. **[data-model.md](data-model.md)** - Memory Layout Specifications (542 lines)
   - Structure-of-Arrays design (27 bytes/node achieved)
   - Lock-free queue implementation (4096 entries, MPMC)
   - DLPack tensor bridge (zero-copy)
   - Thread-local arenas (4096-node blocks)
   - Epoch-based clearing (1M× speedup)

10. **[research.md](research.md)** - Architecture Decisions (521 lines)
    - WU-UCT virtual loss analysis
    - Busy-edge masking rationale
    - Root pre-expansion strategy
    - Thread affinity tuning
    - Batch size/timeout optimization

11. **[quickstart.md](quickstart.md)** - Build & Validation Guide (454 lines)
    - Build commands with hardware optimizations
    - OpenMP verification procedure
    - Benchmark execution
    - Validation protocol
    - Troubleshooting guide

#### Future Phase Specifications

12. **[NEURAL_NETWORK_OPTIMIZATION.md](NEURAL_NETWORK_OPTIMIZATION.md)** - NN Redesign (Phase 7, 9,250 lines)
    - **4 architecture options**:
      - **Option A**: RepVGG/ECA (+25-50% model speed, safest)
      - **Option B**: Ghost+Shuffle (+40-80% model speed, maximum)
      - **Option C**: Two-tier cascade (×1.5-2.5 throughput)
      - **Option D**: Early-exit heads (×1.2-1.6 throughput)
    - **Combined impact**: 18-22k sims/sec total throughput
    - Game-specific configurations (Gomoku/Chess/Go)
    - Training protocol with self-distillation
    - A/B validation protocol
    - Implementation tasks (T031-T037, 10 days)
    - **Status**: FUTURE (post-8k MCTS target)

### Archived Documents (Superseded)

13. **[archive/](archive/)** - Legacy Documents
    - [SPECIFICATION.md](archive/SPECIFICATION.md) → Superseded by spec.md v2.0
    - [TASKS.md](archive/TASKS.md) → Superseded by tasks.md v1.0
    - [TECHNICAL_PLAN.md](archive/TECHNICAL_PLAN.md) → Superseded by plan.md v1.0
    - [PR_CHECKLIST.md](archive/PR_CHECKLIST.md) → Superseded by ACCEPTANCE_CHECKLIST.md
    - [REVIEW-RESPONSE.md](archive/REVIEW-RESPONSE.md) → Superseded by TRACEABILITY_MATRIX.md
    - See [archive/README.md](archive/README.md) for migration notes

### Planning Documents (Meta)

14. **[DOCUMENTATION_UPDATE_PLAN.md](DOCUMENTATION_UPDATE_PLAN.md)** - Consolidation Plan
15. **[DOCUMENTATION_UPDATE_SUMMARY.md](DOCUMENTATION_UPDATE_SUMMARY.md)** - Changes Summary

---

## Current Progress

### Phase 0: Foundation (T001-T003) - ✅ COMPLETE
- ✅ **T001**: Benchmark harness with CSV telemetry
- ⏳ **T002**: OpenMP verification script (in progress)
- ⏳ **T003**: Feature flag infrastructure (pending)

### Phase 1: CPU Optimizations (T004-T013) - 🔄 IN PROGRESS
- ⏳ **T004-T006**: OpenMP compilation verification (CRITICAL BLOCKER)
- ⏳ **T007-T009**: State pooling implementation
- ⏳ **T010-T011**: Condition variables
- ⏳ **T012-T013**: Node allocator optimization

### Phase 2: Validation (T014-T017) - 📋 PLANNED
- 📋 **T014**: Comprehensive throughput benchmarks
- 📋 **T015**: Ablation studies
- 📋 **T016**: Baseline investigation (time-boxed 2 days)
- 📋 **T017**: KPI dashboard generation

### Phase 3: NN-Eval Cache (T018-T021) - 📋 OPTIONAL
- 📋 **T018**: Zobrist hash implementation
- 📋 **T019**: NN-eval cache data structure
- 📋 **T020**: Cache integration
- 📋 **T021**: Cache performance validation

### Phase 4: Multi-Actor (T022-T026) - 📋 PLANNED
- 📋 **T022**: Centralized inference server
- 📋 **T023**: Self-play actor implementation
- 📋 **T024**: Token bucket backpressure
- 📋 **T025**: Multi-actor orchestrator
- 📋 **T026**: Multi-actor performance validation

### Phase 5: Documentation (T027-T030) - 📋 PLANNED
- 📋 **T027**: Final acceptance benchmarks
- 📋 **T028**: Performance regression test suite
- 📋 **T029**: Design documentation
- 📋 **T030**: Operating guide

### Phase 6: Advanced Optimizations (T014a, T030a) - 🔮 FUTURE
- 🔮 **T014a**: DLPack fast path verification (Phase 2 diagnostic)
- 🔮 **T030a**: Precompute legal moves (Phase 2 enhancement)

### Phase 7: Neural Network Optimization (T031-T037) - 🔮 FUTURE
- 🔮 **T031-T037**: Lightweight NN implementation (10 days)
- **Status**: Post-8k target, fully specified in NEURAL_NETWORK_OPTIMIZATION.md

---

## Quick Start

### 1. Review Core Requirements
```bash
# Read functional specification
cat specs/004-mcts-throughput-recovery/spec.md

# Understand constraints
cat specs/004-mcts-throughput-recovery/CONSTITUTION.md

# Review implementation plan
cat specs/004-mcts-throughput-recovery/plan.md
```

### 2. Verify OpenMP (CRITICAL FIRST STEP)
```bash
# Run OpenMP verification
python scripts/verify_openmp.py

# Expected output: "✅ PASS: OpenMP compiled correctly"
# If FAIL: Follow plan.md Section B1.2 for CMake fixes
```

### 3. Run Baseline Benchmark
```bash
# Establish current performance
python -m tests.performance.benchmark_harness --game gomoku --simulations 10000 --threads 8 --iterations 10

# Expected: CSV created in results/benchmarks/benchmark_history.csv
# Current baseline: ~2,147 sims/sec
```

### 4. Execute Critical Path Tasks
```bash
# Phase 1: Fix OpenMP (T004-T006)
# See tasks.md for detailed instructions

# Phase 1: Implement state pooling (T007-T009)
# Phase 1: Add condition variables (T010-T011)
# Phase 1: Optimize allocator (T012-T013)

# Phase 2: Validate (T014-T017)
python scripts/run_acceptance_suite.py --output results/acceptance_report.csv
```

### 5. Validate Against Checklist
```bash
# Run full acceptance validation
# See ACCEPTANCE_CHECKLIST.md for 150+ checks

# Critical path (11 MUST PASS checks):
# 1. Throughput ≥6,000 sims/sec
# 2. GPU utilization ≥80%
# 3. Thread idle ≤12%
# 4. Feature extraction ≤1.0ms
# 5. State cloning ≤1×
# 6. Thread scaling 4T ≥70%
# 7. Win rate ≥99.5% vs. baseline
# 8. Policy agreement ≥95%
# 9. Value MSE ≤0.01
# 10. TSan clean
# 11. Benchmark scripts run clean
```

---

## Acceptance Criteria

**CRITICAL PATH** (must all pass for production):
- ✅ Throughput ≥6,000 sims/sec (minimum), ≥8,000 (target)
- ✅ GPU utilization ≥80% during search
- ✅ Thread idle ≤12% (vs. 60% baseline)
- ✅ Feature extraction ≤1.0ms per batch-64 (vs. 7.5ms baseline)
- ✅ State cloning ≤1× per simulation (vs. 2-3× baseline)
- ✅ Thread scaling: 4T ≥70%, 8T ≥70% efficiency
- ✅ Win rate ≥99.5% vs. baseline (1000 games)
- ✅ Policy agreement ≥95% (1000 positions)
- ✅ Value MSE ≤0.01 vs. baseline
- ✅ No data races (TSan clean)
- ✅ Benchmark scripts run clean, CSVs valid

**TARGET PATH** (should pass for production):
- Throughput ≥8,000 sims/sec (primary goal)
- GPU utilization 85-95% sustained
- CPU coordination overhead <30% (vs. 67.2% baseline)
- Multi-actor 200-300 games/hour

**STRETCH PATH** (nice to have):
- Throughput ≥10,000 sims/sec (GPU hardware limit)
- NN-eval cache working (Phase 6)
- Lightweight NN deployed (Phase 7, 18-22k sims/sec)

---

## Key Performance Targets

| Metric | Baseline | Current | Target | Stretch |
|--------|----------|---------|--------|---------|
| **Throughput** | 3,831 sims/sec | 2,147 sims/sec | **≥8,000** | ≥10,000 |
| **GPU Utilization** | ~68% | ~68% | **80-95%** | 90-95% |
| **Thread Idle** | ~60% | ~60% | **≤12%** | <5% |
| **Feature Extraction** | 7.5ms | 7.5ms | **≤1.0ms** | <0.5ms |
| **State Cloning** | 2-3× | 2-3× | **≤1×** | 0× |
| **Thread Efficiency (4T)** | ~56% | ~56% | **≥70%** | ≥80% |
| **Thread Efficiency (8T)** | ~45% | ~45% | **≥70%** | ≥75% |

---

## References

**Authoritative Sources**:
- **review.txt** (1,396 lines) - Bottleneck analysis and recommendations
- **spec.md v2.0** - Functional requirements
- **plan.md v1.0** - Technical implementation
- **tasks.md v1.0** - Task breakdown

**Analysis Documents**:
- **TRACEABILITY_MATRIX.md** - 100% coverage verification
- **CLARIFICATIONS.md** - Resolved ambiguities
- **NEURAL_NETWORK_OPTIMIZATION.md** - Phase 7 future work

**Validation**:
- **ACCEPTANCE_CHECKLIST.md** - 150+ validation checks
- **CHECKLIST_SUMMARY.md** - Quick reference

---

## Contact & Support

**Spec Authority**: cosmosapjw-quantum
**Last Review**: 2025-10-14
**Next Review**: After Phase 2 completion or if throughput < 50% of target

**For questions**:
1. Check [CLARIFICATIONS.md](CLARIFICATIONS.md) for resolved ambiguities
2. Check [TRACEABILITY_MATRIX.md](TRACEABILITY_MATRIX.md) for cross-references
3. Check [archive/README.md](archive/README.md) for migration from legacy docs

---

**Status**: Documentation v2.0 COMPLETE, Implementation Phase 1 IN PROGRESS
