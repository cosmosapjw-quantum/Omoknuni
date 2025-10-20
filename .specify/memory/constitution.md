<!--
SYNC IMPACT REPORT
==================
Version Change: INITIAL → 1.0.0
Ratification: 2025-10-20

New Constitution Created:
- 6 core engineering principles added
- Performance targets and acceptance criteria defined
- Operating guardrails established
- Documentation cross-references integrated
- Governance and compliance procedures defined

Template Alignment:
✅ plan-template.md: Constitution Check section will reference these 6 principles
✅ spec-template.md: Requirements must align with zero-copy and threading principles
✅ tasks-template.md: Tasks must include profiling validation checkpoints
⚠ Command files: Manual review recommended to ensure profiling workflow integration

Follow-up Actions:
- Review .claude/commands/speckit.plan.md to ensure Constitution Check references all 6 principles
- Review .claude/commands/speckit.implement.md to ensure profiling gates are enforced
- Update any runtime guidance docs to reference this constitution

Last Updated: 2025-10-20
-->

# MCTS Optimization Project Constitution

## Core Principles

### I. Zero-Copy First (NON-NEGOTIABLE)

**Rule**: All game-state transformations in hot paths MUST operate in-place or use make/unmake patterns. State cloning is PROHIBITED.

**Requirements**:
- Feature extraction MUST extract directly from game state without cloning
- Move application MUST use `make_move()` / `unmake_move()` pattern
- Queue submission MUST pass feature buffer references or file descriptors, not copied states
- State pool (`state_pool.cpp/hpp`) MUST remain unused (proven 56% regression via `copyFrom()` overhead)

**Validation**:
- Code review MUST flag ANY `clone()`, `copy()`, or `new State()` calls in simulation paths
- Profiling MUST show zero time attributed to state copying (<1% allowed for edge cases)
- Memory allocations in hot paths MUST be zero (measured via allocation profiler)

**Rationale**: State cloning is the PRIMARY bottleneck (86.6% of execution time, 418μs per clone). Eliminating this is critical path to 8,000+ sims/sec target.

---

### II. Coordinator Efficiency (NON-NEGOTIABLE)

**Rule**: Coordinators MUST NOT serialize the system. Favor low-latency batching, multi-coordinator designs, and condition variables over polling.

**Requirements**:
- Batch formation MUST use condition variables (NOT polling loops)
- Coordinator MUST NOT reallocate feature buffers in hot loops (pre-allocate and reuse)
- Multi-coordinator architecture (Phase 3A) MUST be preferred over multi-process (Phase 3B) unless profiling proves GIL as bottleneck
- Escalation to multi-process requires written justification based on profiling data

**Validation**:
- Profiling MUST show coordinator blocking <10% of iteration time (currently 99.6%)
- Feature buffer allocations MUST be zero per iteration (pre-allocated pools only)
- If implementing multi-process: document profiling evidence that GIL is >50% bottleneck

**Rationale**: Coordinator blocking is secondary bottleneck (99.6% of time). Efficient batching and multi-coordinator can achieve 12-20k sims/sec without multi-process complexity.

---

### III. Python-C++ Boundary Discipline

**Rule**: Minimize Python-C++ crossings. Reuse tensors. Use pinned memory. Use non-blocking GPU transfers.

**Requirements**:
- Simulation loops MUST execute entirely in C++ with zero GIL re-entry except for inference callbacks
- Tensor allocation MUST use pinned memory pools (4KB-4MB size classes)
- GPU transfers MUST use `non_blocking=True` with CUDA streams
- DLPack tensors MUST be reused across iterations (zero allocation per batch)
- **Python-side overhead** (tensor build + H2D transfer) MUST complete in **≤2.0ms per batch** (measured at **p95** over 100 trials, batch_size=64, pinned memory enabled, non-blocking transfer) — currently 37ms baseline, target after Phase 2 optimizations
- **GPU inference kernel time** is **model-dependent**; gate against **baseline + 20%** (e.g., if baseline ResNet-20 FP16 measures 0.8ms p95, threshold = 0.96ms p95) — prevents regression from model changes, not a fixed target

**Validation**:
- `py-spy` profiling MUST show <5% time in Python during search
- GPU profiling MUST show 80%+ utilization (currently ~68%)
- **Python-side overhead** MUST be measured at p95 and ≤2.0ms per batch (separate from GPU kernel time)
- **GPU kernel time** MUST be measured at p95 and gated against baseline + 20% headroom

**Rationale**: Python-C++ boundary crossings waste time on GIL acquisition, memory copies (6× per batch currently), and synchronization. Proper discipline enables 58-75× performance gains.

---

### IV. Threading Saturation

**Rule**: Saturate all 12 cores on Ryzen 5900X. Avoid lock contention. Prefer thread-local storage. OpenMP MUST be linked and verified.

**Requirements**:
- MCTS simulation MUST use 8-12 worker threads (validated via benchmarks)
- Feature extraction MUST use OpenMP parallelization (verify `libomp.so` linkage)
- Lock contention MUST be <1% of execution time (use thread-local arenas)
- Thread-local allocation MUST maintain 99.93%+ fast-path rate (currently achieved)
- OpenMP success rate MUST be >95% (currently 0% - CRITICAL FIX)

**Validation**:
- `ldd build/lib.linux-x86_64-3.12/mcts_py*.so | grep gomp` MUST show OpenMP linkage
- Thread efficiency MUST be ≥70% at 8 threads (currently 45% at 4 threads)
- Profiling MUST confirm feature extraction time <2ms (currently 7.5ms without OpenMP)

**Rationale**: Broken OpenMP is a critical defect. Fixing it alone provides ~4× speedup (7.5ms → 1.5ms feature extraction). Thread saturation is essential for target throughput.

---

### V. Legacy Code Discipline

**Rule**: Treat `mcts_guide.md` and old `simulation_runner.cpp/hpp` as legacy references ONLY. Focus on `continuous_simulation_runner.cpp/hpp` and current queue/coordinator pipeline.

**Requirements**:
- New features MUST be implemented in `continuous_simulation_runner.*` (NOT old files)
- Code reviews MUST reject changes to deprecated files unless explicitly removing them
- Documentation MUST clearly mark deprecated files with `[LEGACY - DO NOT USE]` headers
- Old interfaces MUST NOT be extended or maintained

**Validation**:
- Git commits MUST NOT touch legacy files except for deletion
- Code coverage tools MUST exclude legacy files from metrics

**Rationale**: Maintaining multiple implementations creates confusion and tech debt. The continuous simulation runner is the current architecture.

---

### VI. Evidence-Based Gates (NON-NEGOTIABLE)

**Rule**: Every optimization phase MUST meet its sims/sec target with automated profiling validation. Rollbacks are mandatory if targets are missed.

**Requirements**:
- Phase 1 MUST achieve 1,500-3,000 sims/sec before Phase 2 starts
- Phase 2 MUST achieve 7,000-9,000 sims/sec (TARGET) or trigger rollback
- Phase 3A MUST achieve 12,000-20,000 sims/sec (STRETCH) or document why
- Each phase MUST include deterministic benchmarks (100+ trials, <5% variance)
- Profiling data MUST be committed to repository (reproducibility requirement)

**Validation**:
- Automated benchmark script MUST run after each phase completion
- Profiling campaign MUST be executed with appropriate trial count:
  - **Baseline/Investigation**: 560 trials (full parameter space: 4 sim counts × 7 thread counts × 4 batch sizes × 5 reps) for exploratory analysis
  - **Phase Validation**: 100 trials minimum (single optimal config with 100 repetitions for statistical significance at 95% CI)
  - **All campaigns**: 100% execution time capture required (no "unknown" time categories >1% of total)
- Results MUST be analyzed via `scripts/profiling/analyze_campaign.py`
- Rollback procedure MUST be documented in implementation plan

**Rationale**: Performance work without measurement is speculation. Gates prevent accumulation of untested changes and enable rapid diagnosis of regressions.

---

## Performance Targets & Acceptance

### Baseline (Measured 2025-10-18)
- **Simulations/sec**: 120.4 (mean, 560 trials)
- **GPU Utilization**: ~68% (batch size 64)
- **Coordinator Blocking**: 99.6% of iteration time
- **State Cloning Time**: 418μs per clone (86.6% contribution)
- **OpenMP Success Rate**: 0% (critical defect)

### Phase 1 Targets (Expected 10-25× gain)
- **Simulations/sec**: 1,500-3,000
- **State Cloning**: ELIMINATED (0μs contribution)
- **Coordinator Blocking**: <50% (condition variables added)
- **Feature Extraction**: In-place, zero allocations

**Acceptance**: Automated profiling shows state cloning <1% of time AND throughput ≥1,500 sims/sec

### Phase 2 Targets (Expected 58-75× gain) ✅ **TARGET**
- **Simulations/sec**: 7,000-9,000 (PRIMARY GOAL)
- **GPU Utilization**: 80%+
- **Tensor Creation**: <2ms per batch (currently 37ms)
- **OpenMP Success**: >95% (currently 0%)
- **Thread Efficiency**: ≥70% @ 8 threads

**Acceptance**: Automated profiling shows throughput ≥7,000 sims/sec AND GPU utilization ≥80%

### Phase 3A Targets (Expected 100-166× gain) 🎯 **STRETCH**
- **Simulations/sec**: 12,000-20,000
- **Coordinator Architecture**: **Default K=3** parallel coordinators on RTX 3060 Ti (GA104); final K **auto-tuned** at startup from {1,2,3,4} via micro-benchmark (3-5s); one CUDA stream per coordinator; shared lock-free MPMC request queue
- **GPU Streams**: Multi-stream inference (K streams, dynamically scheduled by GPU; streams enable kernel overlap and copy/compute concurrency, not SM partitioning)*
- **Thread Scaling**: Linear-ish scaling (K × 0.8 to K × 0.95 efficiency accounts for GIL/queue contention)

*Note: CUDA streams are work queues, not resource allocations. The GPU scheduler dynamically assigns SMs to kernels based on occupancy and availability. Optimal stream count is system-dependent and validated empirically.

**Acceptance**: Automated profiling shows throughput ≥12,000 sims/sec (OPTIONAL - only if needed)

### Phase 3B Targets (Expected 166-291× gain) 🚀 **ADVANCED**
- **Simulations/sec**: 20,000-35,000
- **Architecture**: Multi-process (GIL bypass)
- **IPC**: Shared memory tensor transfer
- **Complexity**: 6+ weeks implementation

**Acceptance**: ONLY implement if Phase 3A insufficient AND target >25,000 sims/sec

---

## Operating Guardrails

### Code Review Requirements

**MUST CHECK** on every pull request:
1. **State Cloning**: Search for `clone()`, `copy()`, `new State()` in simulation paths → REJECT if found
2. **Coordinator Allocation**: Verify feature buffers pre-allocated (not per-iteration) → REJECT if allocating
3. **OpenMP Linkage**: Verify `libomp.so` in `ldd` output if C++ changes → REJECT if missing
4. **Profiling Data**: Verify benchmark results included for performance claims → REJECT if missing
5. **Legacy Files**: Verify no changes to deprecated files → REJECT unless deleting
6. **Thread Safety**: Verify atomic operations or thread-local storage for shared data → REJECT if unsafe

### Technology Constraints

**MUST NOT**:
- Use C++ LibTorch (Python PyTorch only - keeps model iteration fast/flexible)
- Propose distributed/multi-host solutions (single-machine optimization only)
- Modify game rules/logic unless unavoidable (focus on feature extraction + queue)

**MUST USE**:
- Hardware: Ryzen 5900X (12c/24t), 64GB RAM, RTX 3060 Ti
- Search Variant: PUCT (NOT WU-UCT, RAVE, or progressive widening - legacy/deprecated)
- Current Architecture: `continuous_simulation_runner.cpp/hpp` (NOT old `simulation_runner.*`)

### Documentation Cross-References (Canonical)

**Implementation teams MUST consult**:
- [OPTIMIZATION_DOCUMENTATION_INDEX.md](OPTIMIZATION_DOCUMENTATION_INDEX.md) - Navigation, quick start, reading order
- [EXTERNAL_REVIEWS_COMPARISON_ANALYSIS.md](EXTERNAL_REVIEWS_COMPARISON_ANALYSIS.md) - Agreement on root causes, best approaches
- [MCTS_OPTIMIZATION_MASTER_PLAN.md](MCTS_OPTIMIZATION_MASTER_PLAN.md) - Phase 1-3 detailed plan, code examples, validation
- [MCTS_OPTIMIZATION_MASTER_PLAN_ENHANCEMENTS.md](MCTS_OPTIMIZATION_MASTER_PLAN_ENHANCEMENTS.md) - Condition variables, virtual-loss restart, Phase 3B multi-process
- [ARCHITECTURE_TRADEOFFS.md](ARCHITECTURE_TRADEOFFS.md) - Decision framework: multi-coordinator vs multi-process

**References are canonical** - discrepancies between code and docs MUST be escalated.

---

## Development Workflow

### Standard Flow (MANDATORY)

1. **Research** - Understand existing patterns and profiling data
2. **Plan** - Propose approach with expected sims/sec gain
3. **Implement** - Build with deterministic benchmarks
4. **Validate** - Run profiling campaign (100+ trials), verify targets met
5. **Rollback** - If targets missed, revert and document findings

**Example**:
```bash
# Research
cat MCTS_OPTIMIZATION_MASTER_PLAN.md | grep "Phase 1"

# Plan
echo "Implementing in-place feature extraction. Expected: 1,500-3,000 sims/sec"

# Implement
# ... code changes ...

# Validate
python scripts/profiling/run_campaign.py --trials 100 --phase 1
python scripts/profiling/analyze_campaign.py output/

# Decision
if [ sims_per_sec -ge 1500 ]; then
  git commit -m "feat: Phase 1 complete - achieved 2,147 sims/sec"
else
  git reset --hard HEAD^
  echo "ROLLBACK: Only achieved 800 sims/sec. Investigating..."
fi
```

### Profiling Requirements

**Every phase MUST**:
- Run 100+ trial profiling campaign
- Capture 100% of execution time (no "unknown" categories >1%)
- Analyze dominant metrics (>50% time contribution)
- Compare to baseline and phase targets
- Commit results to `docs/performance/phase_X_results.md`

**Profiling Tools**:
```bash
# C++ instrumentation (already in place)
python scripts/profiling/run_campaign.py --trials 560

# Python profiling (if needed)
py-spy record --native -o profile.svg -- python scripts/test_mcts.py

# GPU profiling (if needed)
nsys profile --stats=true python scripts/test_mcts.py
```

---

## Governance

### Constitution Enforcement

**This constitution supersedes all other practices, guides, and historical decisions.**

**Violations MUST**:
1. Be identified during code review
2. Be rejected with reference to specific principle number (e.g., "Violates Principle I: Zero-Copy First")
3. Require written justification if exception is necessary (documented in Complexity Tracking section of plan.md)
4. Trigger immediate rollback if discovered post-merge

**Compliance Review**:
- Every PR MUST pass constitution checklist (6 principles + guardrails)
- Quarterly review of profiling data to verify targets maintained
- Annual constitution review to update targets based on hardware evolution

### Amendment Procedure

**Constitution changes require**:
1. Proposal with rationale (written document)
2. Profiling data supporting need for change
3. Approval from project maintainer (`cosmosapjw-quantum`)
4. Migration plan for existing code
5. Version increment following semantic versioning:
   - **MAJOR**: Backward-incompatible principle removals/redefinitions
   - **MINOR**: New principle/section added
   - **PATCH**: Clarifications, wording, typo fixes

**Example Amendment**:
```
Proposal: Add Principle VII - Memory Budget Enforcement
Rationale: Tree memory exceeded 1GB target in Phase 3A testing
Data: Profiling shows 1.2GB usage at 15M nodes
Migration: Add memory usage telemetry to all simulation loops
Version: 1.1.0 (MINOR - new principle)
```

### Versioning Policy

**Current Version**: 1.0.1
**Ratified**: 2025-10-20
**Last Amended**: 2025-10-20

**Version History**:
- **1.0.1** (2025-10-20): PATCH amendments:
  - Coordinator count: Changed from fixed "4 coordinators" to **"default K=3 on RTX 3060 Ti, auto-tuned at startup from {1,2,3,4}"** based on user clarification Q5 and empirical validation requirement (Principle VI)
  - Trial count requirements clarified: 560 trials for baseline/exploration, 100 trials for phase validation
  - Python callback budget split into controllable (Python-side ≤2.0ms p95) vs model-dependent (GPU kernel baseline+20%)
  - Added footnote: CUDA streams are work queues, not SM partitions; scheduler dynamically allocates resources
- **1.0.0** (2025-10-20): Initial ratification with 6 NON-NEGOTIABLE principles

---

**Version**: 1.0.0 | **Ratified**: 2025-10-20 | **Last Amended**: 2025-10-20
