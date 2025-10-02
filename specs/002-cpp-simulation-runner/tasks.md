# Tasks: C++ MCTS Simulation Runner
**Spec ID**: 002-cpp-simulation-runner
**Source**: spec.md & plan.md & PYTHON_FIXES_REQUIRED.md (2025-10-02 update)

_Format: `Summary | File:Lines | Changes | Acceptance | Est`_

---

## Phase 0 — Python Training Fixes (CRITICAL - Blocks Execution)

- [x] **T001** Policy loss function fix
  - **File**: `src/training/trainer.py:601`
  - **Change**: Replace `F.cross_entropy(policy_pred, policy_target)` → `F.kl_div(F.log_softmax(policy_pred, dim=1), policy_target, reduction='batchmean')`
  - **Reason**: Fix `RuntimeError: expected scalar type Long but got Float`
  - **Acceptance**: ✅ Training runs first batch without exception (validated with synthetic test)
  - **Est**: 15min
  - **Completed**: 2025-10-02 by implement-next (33d9fce1)

- [x] **T002** TrainingConfig fields
  - **File**: `src/training/training_loop.py:47-94`
  - **Change**: Add to `TrainingConfig` dataclass:
    ```python
    mcts_threads: int = 8
    batch_size_min: int = 32
    batch_size_max: int = 64
    inference_timeout_ms: float = 3.0
    ```
  - **Reason**: Fix `AttributeError` when accessing missing fields (lines 199-206)
  - **Acceptance**: ✅ `TrainingConfig` instantiates without errors (validated with default and custom values)
  - **Est**: 15min
  - **Completed**: 2025-10-02 by implement-next (33d9fce1)

- [x] **T003** Config factory function
  - **File**: `src/training/training_loop.py:789-840`
  - **Change**: Filter dict to only include valid TrainingConfig fields before instantiation:
    ```python
    from dataclasses import fields
    valid_fields = {f.name for f in fields(TrainingConfig)}
    filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
    config = TrainingConfig(**filtered_config)
    ```
  - **Reason**: Fix `TypeError` from unknown kwargs to `TrainingConfig`
  - **Acceptance**: ✅ `create_training_loop()` works with all config files (default, development, production, gomoku_48h_training)
  - **Est**: 30min
  - **Completed**: 2025-10-02 by implement-next (33d9fce1)

- [x] **T004** Signal handler guard
  - **File**: `src/training/training_loop.py:162-164`
  - **Change**: Guard signal registration:
    ```python
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    ```
  - **Reason**: Fix `ValueError: signal only works in main thread`
  - **Acceptance**: ✅ Training loop works from worker threads (test passes: `test_signal_handler_from_worker_thread`)
  - **Est**: 10min
  - **Completed**: 2025-10-02 by implement-next (33d9fce1)

- [x] **T005** Training pipeline smoke test
  - **File**: `tests/integration/test_training_pipeline.py`
  - **Run**: `python -m pytest tests/integration/test_training_pipeline.py::TestTrainingPipelineIntegration::test_training_initialization -v`
  - **Acceptance**: ✅ Training loop initializes without crashes, all Phase 0 fixes validated (test passes)
  - **Est**: 10min
  - **Completed**: 2025-10-02 by implement-next (33d9fce1)

## Phase 1 — Build & Move Storage

- [x] **T006** Build wiring
  - **Files**: `cpp_extensions/mcts/CMakeLists.txt`, `pyproject.toml`, `simulation_runner.cpp`
  - **Changes**:
    - Add `simulation_runner.cpp` to `add_library(mcts_core ...)` in CMakeLists
    - Add sanitizer options: `ENABLE_ASAN`, `ENABLE_TSAN`, `ENABLE_UBSAN` with proper flags as list
    - Update `pyproject.toml` scikit-build config with sanitizer documentation
    - Comment out game interface include in `simulation_runner.cpp` (Phase 2 dependency)
  - **Acceptance**: ✅ `pip install -e . --force-reinstall --config-settings build-dir=build` succeeds, ASan build works with `-DENABLE_ASAN=ON`
  - **Est**: 30min
  - **Completed**: 2025-10-02 by implement-next (4749fe51)

- [x] **T007** Contract tests (passing)
  - **File**: `tests/contract/test_simulation_runner_api.py` (NEW)
  - **Content**: Import `mcts_py.SimulationRunner`, instantiate with MCTS components, validate API surface
  - **Changes**:
    - Added `#include "simulation_runner.hpp"` to `cpp_extensions/mcts/python_bindings.cpp`
    - Created Python binding for SimulationRunner class with constructor
    - Implemented 12 contract tests: class existence, instantiation, kwargs, type validation, docstring, multiple instances, different components, shared tree, custom configs, lifecycle
  - **Acceptance**: ✅ All 12 tests pass, SimulationRunner API exposed to Python correctly
  - **Est**: 1h
  - **Completed**: 2025-10-02 by implement-next (077799e)

- [x] **T008** Tree move storage
  - **Files**: `cpp_extensions/mcts/tree.hpp`, `tree.cpp`, `python_bindings.cpp`
  - **Changes**:
    - Add `alignas(64) uint16_t* moves_` to `MCTSTree` class
    - Implement `uint16_t get_move(NodeIndex idx)` and `void set_move(NodeIndex idx, uint16_t move)`
    - Add allocation in constructor, deallocation in destructor
    - Update `clear()` to reset moves array
    - Update `get_memory_usage()` to include moves array
    - Add pybind: `.def("get_move", ...)` `.def("set_move", ...)`
    - Expose `deallocate_node` and `deallocate_nodes` to Python
  - **Test files**: `tests/unit/test_tree_move_storage.cpp` (C++ standalone), `tests/contract/test_move_storage_api.py` (Python)
  - **Acceptance**: ✅ C++ tests pass (8/8), Python tests pass (10/10), memory 19.07MB for 10M nodes (vs 1000MB)
  - **Est**: 2h
  - **Completed**: 2025-10-02 by implement-next (c4bd022)

## Phase 2 — C++ Runner Core

- [x] **T009** Select leaf
  - **File**: `cpp_extensions/mcts/simulation_runner.cpp:select_leaf()`
  - **Changes**:
    - Use `PUCTSelector::select_child` with reusable `std::vector<NodeIndex> path_`
    - Lookup legal moves via `tree_->get_move(child_idx)`
    - Apply virtual loss during traversal
    - Include game state interface (`igamestate.h`)
    - Add public test wrapper `select_leaf_public()` in header
  - **Test file**: `tests/unit/test_simulation_select_leaf.cpp` (C++ standalone with deterministic TestGameState fixture)
  - **Acceptance**: ✅ Path buffer populated, legal move selection verified, virtual loss applied, 4/4 tests passing
  - **Est**: 2h
  - **Completed**: 2025-10-02 by implement-next (79f96b5)

- [x] **T010** Expand node
  - **File**: `cpp_extensions/mcts/simulation_runner.cpp:expand_node()`
  - **Changes**:
    - Implemented expand_node() with terminal detection and inference callback invocation
    - Applied legal move masking and policy renormalization
    - Allocated children via `tree_.allocate_nodes(num_moves)` with fallback for full tree
    - Recorded moves using `tree_.set_move(child_idx, move_idx)`
    - Implemented `get_terminal_value()` for perspective-based value conversion
  - **Test file**: `tests/integration/test_expansion_with_callback.py` (6 tests: basic expansion, policy masking, terminal, callback, move indices, restricted moves)
  - **Acceptance**: ✅ All 6 tests pass, child priors + move indices correct, callback usage verified
  - **Est**: 3h
  - **Completed**: 2025-10-02 by implement-next

- [x] **T011** Backup value
  - **File**: `cpp_extensions/mcts/simulation_runner.cpp:backup_value()`
  - **Changes**:
    - Implemented backup_value() delegating to `BackupManager::backup_value_along_path(path, value, &virtual_loss_)`
    - BackupManager handles sign flipping at each tree level automatically
    - Virtual loss removal integrated via passing VirtualLossManager pointer
    - Added `backup_value_public()` test wrapper in simulation_runner.hpp
  - **Test file**: `tests/unit/test_simulation_backup.cpp` (6 C++ standalone tests)
  - **Acceptance**: ✅ All 6 tests pass - single node backup, two-level sign flip, three-level sign flip, virtual loss removal, multiple backups, terminal value
  - **Est**: 1.5h
  - **Completed**: 2025-10-02 by implement-next

- [x] **T012** Connect pipeline
  - **File**: `cpp_extensions/mcts/simulation_runner.cpp:run_simulation()`
  - **Changes**:
    - Implemented run_simulation() connecting select_leaf() → expand_node() → backup_value()
    - Clones game state to preserve root during traversal
    - Virtual loss managed automatically (applied in select_leaf, removed in backup_value)
    - Returns bool success flag (true on success, false if clone fails)
    - Uses path_buffer_ member for reuse across simulations
  - **Tests**:
    - Contract tests: `tests/contract/test_simulation_runner_api.py` (12 tests, all passing)
    - Integration tests: `tests/integration/test_simulation_pipeline.py` (6 tests, all passing)
  - **Acceptance**: ✅ Contract tests pass (12/12), integration tests pass (6/6), full pipeline validated
  - **Est**: 2h
  - **Completed**: 2025-10-02 by implement-next

## Phase 3 — Python Integration

- [ ] **T013** PyInferenceCallback bridge
  - **File**: `cpp_extensions/mcts/inference_callback.cpp` (NEW), `python_bindings.cpp`
  - **Changes**:
    - Implement `PyInferenceCallback::request_inference(IGameState&)` → `(policy, value)`
    - Accept Python callable, block on `Future.result()` without GIL
    - Add pybind: `.def("__call__", ..., py::call_guard<py::gil_scoped_release>())`
  - **Test file**: `tests/contract/test_inference_callback.py` (GIL release, timeout handling)
  - **Acceptance**: GIL released during C++ work, timeouts handled gracefully
  - **Est**: 1h

- [ ] **T014** AlphaZeroMCTS refactor
  - **File**: `src/core/mcts.py:152-238`
  - **Changes**:
    - Add `use_cpp_runner: bool = True` parameter
    - Implement `_search_cpp(root_state, simulations)` dispatching to `SimulationRunner::run_simulation`
    - **DELETE**: `ThreadPoolExecutor` creation (lines 198-238) in C++ mode
    - **DELETE**: `_move_mapping` dict (lines 136,169,518,565-566), replace with `tree.get_move()`
    - Add warning log if `use_cpp_runner=False`: `"WARNING: Python simulation loop active. Performance degraded 122-163×."`
  - **Test file**: `tests/integration/test_cpp_vs_python_equivalence.py` (±1e-6 on deterministic fixture)
  - **Acceptance**: Policies/values identical between modes, C++ mode ≥30k sims/sec
  - **Est**: 2h

- [ ] **T015** SearchCoordinator fix
  - **File**: `src/core/search_coordinator.py`
  - **Changes**:
    - **DELETE**: Duplicate `stop()` at line 549
    - **CONSOLIDATE**: shutdown logic (cancel futures, drain pool, stop worker) in first `stop()` (line 185)
    - **REPLACE**: Dummy inference (lines 434-477) with `GPUInferenceWorker` call
    - **SHARE**: Bounded thread pool (no nested executors)
  - **Test file**: `tests/integration/test_coordinator_shutdown.py` (start/stop repeatedly, check threads)
  - **Acceptance**: Clean shutdown, no thread leaks, GPU worker connected
  - **Est**: 3h

- [ ] **T016** Inference bridge
  - **File**: `src/core/cpp_inference_bridge.py` (NEW)
  - **Changes**:
    - Implement `CppInferenceBridge` class wrapping `GPUInferenceWorker`
    - `__call__(cpp_state)` → package features → submit to queue → return `Future`
    - Handle CPU fallback routing, timeouts, `InferenceError` propagation
  - **Test file**: `tests/unit/test_inference_bridge.py` (GPU success, CPU fallback, timeout)
  - **Acceptance**: Batching works, CPU fallback triggers correctly, timeouts surface
  - **Est**: 2h

## Phase 4 — Testing & Performance

- [ ] **T017** Performance tests
  - **File**: `tests/performance/test_simulation_runner_performance.py` (NEW)
  - **Content**:
    - Drive lightweight Gomoku fixture
    - Assert ≥30k sims/sec on 8 CPU threads
    - Test thread scaling: 1→8 threads ≥75% efficiency
    - Assert GPU batch size 32-64, utilization 80-92%
  - **Acceptance**: CI enforces thresholds, fails on regression
  - **Est**: 2h

- [ ] **T018** Integration tests
  - **Files**: `tests/integration/test_inference_integration.py`, `test_training_pipeline.py`
  - **Changes**: Enable C++ runner, compare outputs vs legacy mode on seeded scenarios
  - **Test file**: `tests/integration/test_gil_release.py` (<10% Python time profiling)
  - **Acceptance**: Deterministic fixtures show ±1e-6 equivalence, GIL contention <10%
  - **Est**: 3h

- [ ] **T019** C++ unit tests expansion
  - **Files**: `tests/unit/test_tree_move_storage.cpp`, `test_move_storage_concurrent.cpp` (NEW)
  - **Changes**: Cover edge cases, virtual loss guard stability, selection determinism under threads
  - **Run**: Under ThreadSanitizer
  - **Acceptance**: gtests pass, TSan clean
  - **Est**: 2h

- [ ] **T020** Soak & sanitizer tests
  - **File**: `tests/soak/test_long_run.py`
  - **Changes**: Enable C++ runner, run ≥1 hour, assert <10MB leak
  - **CI**: Enable ASan/TSan pipelines for runner path
  - **Command**: `python scripts/build_with_sanitizers.py --all && python -m pytest tests/soak/`
  - **Acceptance**: No leaks, sanitizers clean
  - **Est**: 3h

## Phase 5 — Documentation & Evidence

- [ ] **T021** Docs refresh
  - **Files**: `docs/mcts_guide.md`, `docs/performance/*`, `CLAUDE.md`
  - **Changes**: Document C++ runner flow, integration steps, performance figures, troubleshooting
  - **Acceptance**: Docs describe new architecture accurately
  - **Est**: 2h

- [ ] **T022** AGENTS + Spec sync
  - **Files**: `AGENTS.md`, `specs/002-cpp-simulation-runner/*`
  - **Changes**: Update workflow guidance, verify spec/plan/tasks reflect shipped code, mark PYTHON_FIXES_REQUIRED.md complete
  - **Acceptance**: Repository guidelines current, spec synchronized
  - **Est**: 1h

- [ ] **T023** Evidence bundle
  - **Actions**:
    - Capture profiling charts: throughput, GIL time, GPU utilization
    - Generate Python vs C++ comparison graphs
    - Store in `docs/performance/runner/`
    - Attach to implementation PR
  - **Acceptance**: Artifacts stored, PR includes validation summary
  - **Est**: 1h

---

## Tracking
- **Total Tasks**: 23 (Phase 0: 5, Phase 1: 3, Phase 2: 4, Phase 3: 4, Phase 4: 4, Phase 5: 3)
- **Completed**: 12 / 23 (52.2%) (Phase 0: ✅ 5/5, Phase 1: ✅ 3/3, Phase 2: ✅ 4/4)
- **Next Up**: T013 (PyInferenceCallback Bridge) - Start Phase 3
- **Critical Path**: T001-T005 (Phase 0) → T006-T008 (Phase 1) → T009-T012 (Phase 2) → T013-T016 (Phase 3) → T017-T020 (Phase 4) → T021-T023 (Phase 5)
- **Estimated Total**: 5 days (0.5 + 1 + 1.5 + 1 + 1 + 0.5 buffer)
- **Phase 1 Complete**: Build wiring, contract tests, move storage
- **Phase 2 Complete**: Select leaf, expansion, backup, and pipeline connection all implemented and tested
- Update this checklist after each task completion to stay aligned with Spec-Driven Development.
