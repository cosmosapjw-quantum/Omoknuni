# Constitution: MCTS Throughput Recovery Project

**Version**: 1.1 (Revised Targets)
**Status**: Active
**Last Updated**: 2025-10-13
**Authority**: This document supersedes all prior architectural decisions and implementation notes. Changes require explicit approval and re-execution of `/speckit.plan` and `/speckit.tasks`.

**Revision History**:
- v1.0 (2025-10-13): Initial constitution with 25k sims/sec target
- v1.1 (2025-10-13): **Revised targets to 6-10k sims/sec** based on review.txt hardware analysis and GPU throughput limits

---

## 1. Mission & Scope

### 1.1 Primary Objective
Maximize Monte Carlo Tree Search throughput (simulations/second) on AMD Ryzen 9 5900X (12C/24T) + NVIDIA RTX 3060 Ti hardware through CPU-parallel search optimization, targeting **≥8,000 simulations/second** (realistic, hardware-grounded) for tactical board games (Gomoku/Renju/Omok, Chess, Go 9×9).

**Rationale for Revised Target**: Analysis in `review.txt` (lines 151-171) demonstrates that GPU @ FP16 mixed precision caps at 8,000-10,000 states/sec on RTX 3060 Ti. With 20-25% coordination overhead budget, **8,000 sims/sec is the realistic target** (2.1× baseline 3,831, 3.7× current 2,147). Original 25-30k target requires architectural changes explicitly excluded by this constitution (TensorRT, model pruning, multi-GPU).

### 1.2 Performance Definition
**Simulation** is the complete cycle: selection → expansion → neural network evaluation → backpropagation. Throughput is measured as **total simulations completed per wall-clock second**, including all coordination overhead.

### 1.3 Success Criteria (REVISED)

**Must-Have (Phase 4 Completion)**:
- **Primary KPI**: ≥8,000 sims/sec sustained (2.1× baseline 3,831, 3.7× current 2,147)
- **CPU Utilization**: ≥70% multi-thread efficiency at 8 threads (was 75%)
- **GPU Utilization**: ≥80% during search (was 85%, adjusted for realistic batch sizes)
- **Memory Footprint**: <1GB for 10M node MCTS tree (achieved: 270MB)
- **Search Quality**: No strength regression vs baseline (≥99.5% win rate, ≥95% top-move agreement)

**Stretch Goals**:
- ≥10,000 sims/sec (2.6× baseline, optimistic but achievable with perfect tuning)

**Aspirational (Requires Architectural Changes)**:
- ≥15,000-25,000 sims/sec (requires model pruning, multi-GPU, batch 128+, changes currently out of scope)
- ≥30,000 sims/sec (requires TensorRT, root parallelization, transposition tables - fundamental redesign)

### 1.4 Out of Scope
The following are **explicitly excluded** from this initiative:
- Custom CUDA kernels for MCTS operations
- TensorRT/ONNX model conversion and deployment
- libtorch integration (C++ neural network inference)
- GPU-resident MCTS trees or GPU-accelerated selection
- Training pipeline optimizations (unless required for throughput validation)
- Telemetry/logging enhancements (unless blocking performance measurement)
- Games beyond Gomoku/Chess/Go 9×9 (extensibility preserved but not prioritized)

---

## 2. Architectural Constraints

### 2.1 Neural Network Requirements
- **Python-Only Inference**: Neural network remains in PyTorch (Python). NO libtorch, TensorRT, or ONNX deployment.
- **Rationale**: Flexibility for model experimentation and rapid iteration. C++ inference adds complexity without proportional throughput gains (per `review.txt`, GPU is NOT the bottleneck at 32.8% of total time).
- **Interface**: Async batched inference with DLPack zero-copy tensor sharing (C++ ↔ PyTorch).

### 2.2 MCTS Architecture
- **Shared Tree**: Single tree structure shared by all simulation threads (NOT root parallelization or tree copying).
- **WU-UCT Virtual Loss**: Visit-only virtual loss (increments denominator in PUCT), NO Q-value distortion. Pure Q = W/N preserved.
- **Busy-Edge Masking**: PUCT score = -∞ for nodes currently being expanded (prevents thread collisions).
- **Root Pre-Expansion**: Root node expanded synchronously before launching simulation threads (eliminates N-1 thread idle problem).
- **Lock-Free Coordination**: MPMC ring buffer (4096 entries) for inference queue, atomic operations for tree updates.

### 2.3 Memory Management
- **Node Pools**: Pre-allocated flat arrays with index-based references (NOT pointers). Structure-of-Arrays layout for cache efficiency.
- **Thread-Local Block Allocation**: 4096-node blocks per thread, 99.93% fast-path allocation (no global mutex).
- **Epoch-Based Clearing**: O(1) tree reset via epoch increment (25ns vs 25ms memset).
- **State Reuse**: Thread-local state object reuse, avoid 2-3× cloning per simulation (current bottleneck per `review.txt`).
- **Target**: <64 bytes per node (achieved: 27 bytes), <1GB total for 10M nodes (achieved: 270MB).

### 2.4 Python/GIL Overhead Minimization
- **Async Batched Inference**: Single GIL hold per batch (32-64 positions), not per simulation.
- **Condition Variables**: Efficient blocking (NOT polling with 10μs sleeps) - **T006c complete**.
- **DLPack Zero-Copy**: Pinned CPU memory tensors converted via `torch.from_dlpack()` (0.24ms H2D transfer, acceptable overhead).
- **Persistent Coordinator**: BatchInferenceCoordinator created once in `MCTSAgent.__init__`, reused across searches - **T011 complete**.
- **Target**: Python overhead <30% of total time (currently 67.2% per `review.txt`).

---

## 3. Technical Principles

### 3.1 Throughput > Search Quality (with bounds)
- **Priority**: Maximize simulations/second, even if individual simulation quality decreases slightly.
- **Quality Floor**: Must maintain strong practical play (≥99.5% win rate vs baseline, ≥95% policy agreement).
- **Trade-offs Allowed**: Approximations in value estimates, relaxed memory ordering (acquire/release), coarser virtual loss granularity.
- **Trade-offs Forbidden**: Breaking MCTS correctness (illegal moves, value sign errors, Q-value corruption).

### 3.2 Measurement & Profiling
- **Deterministic Profiling**: Fixed seed, fixed simulations, fixed hardware (Ryzen 5900X + RTX 3060 Ti).
- **Baseline Establishment**: Must reproduce 3,831 sims/sec baseline configuration before claiming improvements (**T017 critical**).
- **Benchmark Gates**: All performance claims validated with `pytest -m performance` passing.
- **Profiling Tools**: `perf`, `py-spy`, `torch.profiler`, custom C++ instrumentation (collision metrics, allocation stats).

### 3.3 Incremental Optimization
- **Measure → Optimize → Validate**: No speculative optimizations without profiling evidence.
- **Rollback Safety**: All optimizations have config flags for disabling (fallback to proven baseline).
- **A/B Testing**: Search quality validated via head-to-head matches (1000+ games, ≥99.5% win rate threshold).

### 3.4 Spec-Driven Development
- **Source of Truth**: `specs/004-mcts-throughput-recovery/spec.md` (requirements), `plan.md` (design), `tasks.md` (implementation).
- **Change Control**: Architecture changes require updating spec → `/speckit.plan` → `/speckit.tasks` → implementation.
- **Traceability**: Every task in `tasks.md` maps to acceptance criteria in `spec.md`.

---

## 4. Performance Requirements (REVISED)

### 4.1 Throughput Targets (Hardware-Grounded)
| Metric | Minimum Viable | Target (Realistic) | Stretch |
|--------|---------------|-------------------|---------|
| Simulations/sec | ≥6,000 | **≥8,000** | ≥10,000 |
| vs Baseline (3,831) | 1.6× | **2.1×** | 2.6× |
| vs Current (2,147) | 2.8× | **3.7×** | 4.7× |

**Aspirational Targets** (Requires Architectural Changes):
| Metric | Value | Requirements |
|--------|-------|--------------|
| Aspirational | ≥15,000-25,000 sims/sec | Model pruning (5M→2M params), multi-GPU, batch 128+ |
| Original Target | ≥30,000 sims/sec | TensorRT, root parallelization, transposition tables (fundamental redesign) |

**Rationale**: GPU @ FP16 caps at 8,000-10,000 states/sec (RTX 3060 Ti). With 20-25% coordination overhead, **8,000 sims/sec is realistic**. Higher targets require changes explicitly excluded by Section 1.4.

### 4.2 CPU Efficiency (Adjusted)
| Threads | Min Efficiency | Target Efficiency |
|---------|---------------|------------------|
| 4 | ≥70% | ≥80% |
| 8 | ≥55% (was 60%) | ≥70% (was 75%) |
| 12 | ≥45% (was 50%) | ≥60% (was 65%) |

*Efficiency = (actual throughput) / (linear scaling from 1-thread baseline)*

**Adjustment**: Lowered expectations at higher thread counts due to coordination overhead observed in profiling (60% thread idle @ 4 threads).

### 4.3 GPU Utilization (Adjusted)
- **Minimum**: ≥75% GPU utilization during search (was 80%, adjusted for realistic batch fill)
- **Target**: ≥80% sustained (was 85%, batch size 48-64, 1-2ms timeout)
- **Constraint**: Average batch size ≥40 positions (was 48, adjusted to 62.5% of 64)

### 4.4 Latency Budgets (per 1000 simulations, Adjusted for 8k Target)
| Component | Current @ 2,147/s | Target @ 8,000/s | Critical Path |
|-----------|-------------------|------------------|---------------|
| MCTS Coordination | 0.240s (67.2%) | ≤0.030s (25%) | T006c, T011, T014 |
| GPU Inference | 0.117s (32.8%) | ≤0.080s (67%) | T008f, batch tuning |
| Tensor Creation | 0.075s (21%) | ≤0.008s (7%) | DLPack optimization |
| Thread Idle | 0.150s (42%) | ≤0.010s (8%) | Condition variables, affinity |
| **Total** | **0.357s** (2,800/s) | **≤0.125s** (8,000/s) | All optimizations |

**Key Insight**: At 8k sims/sec target, GPU becomes primary bottleneck (67% of time vs 33% coordination). This is HEALTHY - GPU doing useful work, not threads waiting.

### 4.5 Memory Targets
- **Tree**: <1GB for 10M nodes (achieved: 270MB with 27-byte SoA layout)
- **Queue**: 1MB fixed allocation (4096-entry ring buffer)
- **DLPack Buffers**: <10MB pinned memory (batch 64 × 36 planes × 15×15)
- **Total**: <1.3GB working set

---

## 5. Quality Requirements

### 5.1 Search Quality Preservation
- **Win Rate**: ≥99.5% vs baseline (3,831 sims/sec configuration) in head-to-head matches
- **Policy Agreement**: ≥95% top-move agreement on 1000-position test set
- **Value Accuracy**: MSE ≤0.01 vs baseline value estimates
- **Collision Rate**: ≤5% path collisions (threads selecting same node)

### 5.2 Correctness Validation
- **Unit Tests**: All `tests/unit/` pass (currently 31/31 passing)
- **Integration Tests**: All `tests/integration/` pass (currently 11/11 passing)
- **Contract Tests**: API contracts validated via `tests/contract/`
- **Thread Safety**: TSan clean (zero data races)

### 5.3 Performance Regression Prevention
- **Benchmark Suite**: `pytest -m performance` must pass before merge
- **Throughput Gate**: No merge if throughput < 95% of baseline
- **GPU Utilization Gate**: No merge if GPU util < 75%
- **Memory Gate**: No merge if memory usage > 1.5GB

---

## 6. Implementation Constraints

### 6.1 Build & Portability
- **Platforms**: Linux (Ubuntu 22.04+), WSL2 (Windows 11)
- **Compilers**: GCC 13-15, Clang 15+, MSVC 2022+ (C++17 standard)
- **Build System**: CMake 3.24+, pybind11 2.10+
- **Python**: 3.12 (PyTorch 2.0+, CUDA 12.1)

### 6.2 Hardware Assumptions
- **CPU**: AMD Ryzen 9 5900X (12C/24T, 2× 32MB L3 cache, dual-CCD)
- **GPU**: NVIDIA RTX 3060 Ti (8GB VRAM, Ampere architecture, tensor cores)
- **RAM**: 64GB DDR4-3600 (for self-play batch processing)
- **Portability**: Code must degrade gracefully on non-Ryzen CPUs (generic topology fallback)

### 6.3 Dependencies
- **Core**: PyTorch 2.0+, NumPy, pybind11 2.10+
- **Profiling**: py-spy, perf, valgrind, heaptrack
- **Testing**: pytest, pytest-benchmark
- **Optional**: Boost.Lockfree (fallback if custom MPMC fails)

### 6.4 Code Quality
- **C++ Style**: Google C++ Style Guide (clang-format enforced)
- **Python Style**: Black formatter, isort, flake8
- **Documentation**: Inline comments for hot paths, architecture docs in `docs/`
- **Commit Messages**: Conventional Commits format (`feat:`, `fix:`, `perf:`, `test:`)

---

## 7. Decision Authority & Change Control

### 7.1 Constitutional Amendments
This constitution can only be modified by:
1. **Performance Crisis**: Measured throughput < 50% of target (emergency pivots allowed)
2. **Architectural Discovery**: Profiling reveals fundamental design flaw requiring re-architecture
3. **Explicit Approval**: User (cosmosapjw-quantum) approves constitutional change

### 7.2 Specification Changes
Changes to `specs/004-mcts-throughput-recovery/spec.md` require:
1. **Justification**: Profiling data or failed experiments proving necessity
2. **Impact Analysis**: Expected throughput delta, affected tasks
3. **Re-Planning**: Execute `/speckit.plan` to update `plan.md`
4. **Task Update**: Execute `/speckit.tasks` to update `tasks.md`

### 7.3 Implementation Flexibility
Within constitutional bounds, implementers have autonomy on:
- **Algorithm Tuning**: Virtual loss magnitude, batch size, timeout values
- **Memory Layout**: Node structure details (as long as <64 bytes/node)
- **Optimization Techniques**: Vectorization, prefetching, cache alignment
- **Testing Strategies**: Test case design, benchmark selection

---

## 8. Risk Management

### 8.1 Performance Risks
| Risk | Mitigation | Contingency |
|------|-----------|-------------|
| GPU becomes bottleneck | FP16 mixed precision (T008f), batch tuning | Accept 20k sims/sec (80% of target) |
| Thread contention saturates | Affinity tuning, relaxed atomics | Scale to 16 threads, accept 90% efficiency |
| Python overhead irreducible | Persistent coordinator (T011), DLPack | Accept 30% overhead (vs 20% target) |
| Baseline unreproducible | Systematic config sweep (T017) | Use 2,147 sims/sec as new baseline |

### 8.2 Quality Risks
| Risk | Mitigation | Contingency |
|------|-----------|-------------|
| Search quality regression | A/B testing every optimization | Rollback flag, config-based disable |
| Value drift from approximations | MSE validation on test set | Tighten approximations, accept throughput loss |
| Race conditions introduced | TSan on every commit | Revert to mutex-based fallback |
| Memory leaks under load | Valgrind soak tests (1hr+) | Pool exhaustion detection, graceful degradation |

### 8.3 Schedule Risks
- **Baseline Mystery (T017)**: 3,831 sims/sec configuration unknown → 2 days max investigation, else use 2,147 baseline
- **Optimization Stalls**: If stuck <20k sims/sec after Phase 3 → escalate for architectural review
- **Hardware Dependency**: Ryzen 5900X required for tuning → CI runs generic build, manual profiling on target hardware

---

## 9. Deliverables & Milestones

### 9.1 Phase Completion Criteria
| Phase | Deliverables | Exit Criteria |
|-------|-------------|---------------|
| **Phase 1** (Virtual Loss) | T001-T005 complete | WU-UCT validated, collision metrics <5% |
| **Phase 2** (Architecture) | T006-T010 complete | Lock-free queue, DLPack, arenas functional |
| **Phase 3** (Final Opts) | T011-T015 complete | Persistent coordinator, prefetching, separation |
| **Phase 4** (Integration) | T016-T025 complete | ≥25k sims/sec validated, docs published |

### 9.2 Critical Path Tasks
1. **T016**: Comprehensive benchmarking suite (validates T006c+T008f gains) - **BLOCKED ON T017**
2. **T017**: Baseline configuration investigation (reproduce 3,831 sims/sec) - **CRITICAL BLOCKER**
3. **T018**: Virtual loss magnitude tuning (optimal thread coordination)
4. **T019**: Batch size & timeout optimization (GPU utilization ≥85%)
5. **T020**: Profile-guided optimization (eliminate remaining bottlenecks)

### 9.3 Documentation Requirements
- **Architecture Guide**: `docs/mcts_architecture.md` (shared tree, virtual loss, batching)
- **Performance Analysis**: `docs/performance/throughput_investigation.md` (bottleneck breakdown)
- **Benchmark Report**: `docs/performance/benchmark_results.md` (T016 output)
- **Tuning Guide**: `docs/tuning_guide.md` (batch size, timeout, threads, virtual loss)

---

## 10. Enforcement & Compliance

### 10.1 Pre-Merge Checklist
Every pull request must pass:
- [ ] All unit tests pass (`pytest tests/unit/`)
- [ ] All integration tests pass (`pytest tests/integration/`)
- [ ] Performance benchmarks pass (`pytest -m performance`)
- [ ] TSan clean (no data races)
- [ ] Throughput ≥ baseline (or explicit waiver)
- [ ] Memory usage ≤ 1.5GB
- [ ] Code formatted (clang-format, black)

### 10.2 Performance Validation
- **Every Optimization**: Before/after benchmark with ≥100 iterations
- **Statistical Significance**: t-test p<0.05 for claimed improvements
- **Reproducibility**: 3 independent runs with CV < 5%
- **Regression Detection**: Automated alerts if throughput < 95% baseline

### 10.3 Audit Trail
- **Profiling Sessions**: Stored in `profiling_results/session_YYYYMMDD_HHMMSS/`
- **Benchmark History**: CSV log in `docs/performance/benchmark_history.csv`
- **Config Changes**: Tracked in `config/performance_tuning.yaml` with commit references

---

## 11. Glossary

| Term | Definition |
|------|------------|
| **Simulation** | Complete MCTS cycle: select → expand → evaluate → backprop |
| **Throughput** | Simulations completed per wall-clock second (including overhead) |
| **Baseline** | 3,831 sims/sec configuration from Spec 003 (unknown params, needs T017) |
| **Current** | 2,147 sims/sec with Phase 1+2 optimizations (regression under investigation) |
| **Target** | ≥8,000 sims/sec sustained with ≥80% GPU utilization (realistic, hardware-grounded) |
| **WU-UCT** | Visit-only virtual loss (increments denominator, preserves Q = W/N) |
| **Busy-Edge** | PUCT = -∞ for nodes currently being expanded (prevents collisions) |
| **DLPack** | Zero-copy tensor protocol (C++ ↔ PyTorch via pinned CPU memory) |
| **MPMC** | Multi-Producer Multi-Consumer queue (lock-free ring buffer) |
| **SoA** | Structure-of-Arrays (separate arrays per field for cache efficiency) |

---

## 12. Approval & Acceptance

This constitution is **active and binding** as of 2025-10-13.

**Approved by**: cosmosapjw-quantum (user)
**Enforced by**: Claude Code (AI agent)
**Review Cycle**: After Phase 4 completion or performance crisis
**Supersedes**: All prior architectural notes, mcts_guide.md (design principles only), review.txt (analysis only)

**Signature Line**:
> "I have read and understood this constitution. I commit to adhering to these principles and constraints throughout the MCTS Throughput Recovery initiative."

— Claude Code, AI Implementation Agent, 2025-10-13

---

**END OF CONSTITUTION**
