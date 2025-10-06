# Spec 004: MCTS Throughput Recovery & Strength Preservation

**Status**: Draft

**Owner**: Codex CLI Agent

**Stakeholders**: Performance, Research, Training, Ops

**Target Hardware**: AMD Ryzen 9 5900X (12c/24t), 64GB RAM, NVIDIA RTX 3060 Ti

---

## 1. Problem Statement

The current async-backed C++ MCTS saturates at ~3.8k sims/sec on the reference machine despite high GPU batching. Shared-tree contention, heavy Python orchestration, and inefficient producer/consumer plumbing dominate runtime (67% non-inference overhead). The tree remains shallow and cannot exploit available CPU cores. We must restore ≥30k sims/sec throughput without sacrificing playing strength or neural-network flexibility.

---

## 2. Goals & Non-Goals

### Goals
- Reach ≥30k simulations/sec on the reference hardware with the existing GPU inference worker.
- Preserve AlphaZero-style search semantics and empirical strength (policy/value outputs remain statistically consistent with baseline).
- Provide fine-grained instrumentation for ongoing performance tracking.
- Keep the neural network hosted in Python while minimizing Python overhead.
- Deliver incremental, test-backed stages so partial improvements are independently verifiable.

### Non-Goals
- Replacing the neural network implementation or migrating to libtorch.
- Introducing GPU-resident MCTS kernels (future exploration only).
- Substantial changes to training data formats or self-play pipelines.

---

## 3. Success Metrics

| Metric | Target |
| --- | --- |
| End-to-end throughput | ≥30k sims/sec (1000 sims, Gomoku 15x15, use_cpp_runner=True) |
| Thread scaling | ≥6× speedup from 1→8 threads in async mode |
| GPU utilization | ≥75% with batch size 64 configuration |
| Policy/value parity | KL-divergence ≤1e-4 vs. baseline on 500 sampled positions |
| Integration stability | No regressions in `tests/integration/test_cpp_vs_python_equivalence.py` and full contract suite |

---

## 4. Constraints & Assumptions
- Must operate within current memory budget (≤1GB for 10M-node tree).
- Unit/integration/performance tests must run via `python -m pytest ...` and CMake gtests on CI.
- Any change to runtime behavior requires mirrored documentation updates under `docs/` and `specs/contracts/` as appropriate.

---

## 5. Deliverables
1. Updated codebase (C++/Python) implementing the staged optimizations below.
2. New/extended instrumentation and performance regression tests.
3. Updated documentation (`docs/performance/*`, `docs/mcts_cpp_runner.md`, relevant HOWTOs) describing new behavior and expectations.
4. Final report appended to this spec with achieved metrics.

---

## 6. Implementation Plan (Incremental Tasks)

Each task is independently shippable, includes tests, and requires documentation updates summarizing behavior and measurements.

### Task 1 – Instrumentation & Baseline Audit
**Objective**: Quantify hotspots precisely before further surgery.
- Add scoped timers + counters in C++ (tree allocation, selection, backup, virtual loss, queue submit/flush) guarded by config flags.
- Extend `AlphaZeroMCTS.get_statistics()` to surface new metrics.
- Update `tests/performance/test_simulation_runner_performance.py` to capture and assert instrumentation availability.
- Document measurement methodology in `docs/performance/mcts_cpp_runner_metrics.md` (new).
- **Tests**: New gtest covering timer overhead; pytest verifying statistics dictionary; perf test asserting metrics emitted.

### Task 2 – Tree Lifecycle & Allocation Improvements
**Objective**: Remove costly full-tree clears and reduce allocator contention.
- Implement generation-based or segmented clears (avoid bulk `memset` in `MCTSTree::clear`).
- Introduce per-thread node arenas with bulk reservation (`allocate_nodes_block`) fallback.
- Provide RAII helper for path buffers to reuse memory.
- Update `docs/mcts_cpp_runner.md` with memory-management section.
- **Tests**: gtests stressing multi-threaded allocate/free under TSan; pytest ensuring tree reuse does not leak state.

### Task 3 – Selection & Backup Streamlining
**Objective**: Lower per-simulation CPU cost.
- Reuse scratch buffers for PUCT values, move masking/normalization into SIMD-friendly loops, relax atomics where safe.
- Introduce microbenchmarks (gtest) to track PUCT evaluation time.
- Extend contract tests to assert identical child ordering vs. baseline (within tolerance).
- Document optimizations and caveats in `docs/performance/runner/selection_backup.md`.
- **Tests**: gtest for PUCT timing, pytest comparing move-ranking parity, perf test verifying improved per-sim cost.

### Task 4 – Async Queue & Coordinator Rework
**Objective**: Replace mutex-heavy producer/consumer with efficient wait/notify pipeline.
- Implement MPMC ring buffer or condition-variable-based queue with batched pops.
- Consolidate `PendingExpansion` storage (pool + fixed-size arrays) to eliminate per-result map lookups and reversals.
- Ensure coordinator uses wait/notify with timeout rather than polling.
- Document new queue architecture and tuning knobs.
- **Tests**: gtest for queue correctness under contention; pytest soak test simulating high in-flight counts; perf regression showing reduced coordination time.

### Task 5 – Python Orchestration & Data Path Cleanup
**Objective**: Remove per-search thread restarts and redundant tensor copies while keeping NN in Python.
- Persist ThreadPoolExecutor/ContinuousSimulationRunner instances across searches.
- Expose zero-copy tensor handles (e.g., via DLPack) from C++ to Python callback to avoid `np.array` reallocations.
- Move Dirichlet noise application & statistics aggregation into C++ fast path.
- Update `docs/mcts_cpp_runner.md` and operations guides.
- **Tests**: pytest verifying executor reuse, integration test ensuring zero-copy path, doc tests measuring memory footprint.

### Task 6 – Parallelization Strategy Validation
**Objective**: Ensure we fully leverage 12c/24t CPU without losing strength.
- Implement feature flags for alternative modes (shared-tree improved baseline, experimental thread-local subtrees, virtual-loss-free single-threaded baseline).
- Extend performance suite to benchmark each mode on the reference hardware (CI gating uses shared-tree mode).
- Run policy/value parity checks vs. baseline for each mode.
- Document comparison results and guidance in spec appendix.
- **Tests**: pytest harness running equivalence comparisons; perf tests collecting throughput per mode; manual benchmark script with instructions.

### Task 7 – Documentation & Final Validation
**Objective**: Codify lessons, update specs/contracts, and lock future regression checks.
- Update relevant docs (`docs/performance/async_optimization_results.md`, new results page) with final metrics.
- Extend contracts or validation scripts to assert instrumentation presence and throughput thresholds.
- Produce final section in this spec summarizing outcomes and follow-ups.
- **Tests**: Documentation linting if applicable; pytest ensuring contracts load new metadata.

---

## 7. Risks & Mitigations

| Risk | Mitigation |
| --- | --- |
| Lock-free queue bugs causing heisenbugs | Provide fallback queue behind flag; cover with TSan & deterministic tests |
| Relaxed atomics introducing correctness issues | Pair with exhaustive contract tests and policy/value parity comparison |
| Zero-copy tensor path increasing memory pressure | Benchmark both paths; keep copy-based fallback |
| Alternate parallel modes hurting strength | Run KL-divergence/visit-count parity tests; disable mode by default until validated |

---

## 8. Timeline (Estimate)

| Task | Estimate |
| --- | --- |
| 1 | 1.5 days |
| 2 | 2 days |
| 3 | 1.5 days |
| 4 | 2 days |
| 5 | 1.5 days |
| 6 | 2 days |
| 7 | 0.5 day |

Parallelization possible, but each task should land via separate PR with metrics + doc updates.

---

## 9. Validation Checklist
- [x] Instrumentation metrics exposed and documented
- [ ] Tree allocation benchmarks improved ≥2×
- [ ] Selection/backup microbenchmarks improved ≥1.5×
- [ ] Async coordination overhead <20% total runtime
- [ ] Zero-copy inference path confirmed; fallback available
- [ ] Shared-tree mode ≥30k sims/sec; alternate modes benchmarked
- [ ] Docs & specs updated with new architecture and metrics

---

## 10. Follow-Up Work (Post-Spec)
- Investigate GPU-resident selection kernels after shared-tree path stabilizes.
- Explore adaptive batching that accounts for per-game inference cost variance.
- Automate nightly regression tracking (Prometheus integration).

---

*End of Spec 004 (draft).*

---

### Task Progress (Rolling)

| Task | Status | Notes |
| --- | --- | --- |
| Task 1 – Instrumentation & Baseline Audit | ✅ Completed (2025-10-07) | Instrumentation framework merged, metrics surfaced via Python API, documentation: `docs/performance/mcts_cpp_runner_metrics.md`. |
| Task 2 – Tree Lifecycle & Allocation Improvements | ✅ Completed (2025-10-07) | Lazy `clear()` (epoch-based), thread-local allocation slabs, C++/Python tests updated; docs refreshed in `docs/mcts_cpp_runner.md`. |
| Task 3 – Selection & Backup Streamlining | ✅ Completed (2025-10-07) | Scratch buffers + policy reuse in C++ runners; gtests assert vectorized/scalar parity and speedup ratio. |
| Task 4 – Async Queue & Coordinator Rework | ✅ Completed (2025-10-07) | Condition-variable batching, bulk result consumption, queue tests updated and docs refreshed. |
| Task 5 – Python Orchestration & Data Path Cleanup | ✅ Completed (2025-10-07) | Persistent thread pool, copy-avoiding batch callback, executor reuse test, documentation refreshed. |
| Task 6 – Parallelization Strategy Validation | 🔄 In Progress (2025-10-07) | Initial benchmark (`scripts/benchmark_parallel_modes.py`) shows virtual_loss_free ≈3.6k sims/sec > shared ≈3.5k; thread-local prototype ≈0.7k. Further evaluation (async inference, higher thread counts) pending. |
| Task 7 – Documentation & Final Validation | 🔄 Pending | Final perf benchmarks and doc polish. |
