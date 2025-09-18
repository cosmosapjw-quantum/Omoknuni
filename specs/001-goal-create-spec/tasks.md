# Tasks: High-Performance AlphaZero Engine

**Input**: Design documents from `/specs/001-goal-create-spec/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Format: `Summary | Steps | Acceptance | Owner | Est`

---

## PHASE 0 — Repo & Telemetry

- [x] **T001** Setup project structure and build system | Create src/{core,games,neural,training,telemetry,utils}/ directories, cpp_extensions/{mcts,games,utils}/ directories, tests/{contract,integration,unit,performance}/ directories, pyproject.toml with scikit-build-core, requirements.txt with PyTorch 2.x, pybind11, Cython dependencies | ✅ Project structure matches plan.md, all dependencies install cleanly, build system configured with -O3 -march=znver3 -fopenmp flags | Dev | 3h
  *Completed: 2025-09-16, Author: Claude Code*

- [x] **T002** [P] Initialize CI/CD pipeline | Create .github/workflows/ci.yml with pytest, flake8, mypy checks, GPU testing on self-hosted runner, performance regression detection | ✅ All checks pass on sample code, GPU tests run successfully, build artifacts cached | Dev | 2h
  *Completed: 2025-09-17, Author: Claude Code*

- [x] **T003** [P] Implement basic telemetry framework | Create src/telemetry/metrics.py with Prometheus-compatible metrics collection, GPU utilization monitoring, memory usage tracking, structured logging setup | ✅ Metrics collection functional, can track simulations/sec, GPU util, memory usage | Dev | 3h
  *Completed: 2025-09-17, Author: Claude Code*

- [x] **T004** [P] GPU warmup and device detection | Create src/neural/device_manager.py with CUDA availability check, GPU warming with dummy inference, optimal batch size detection, RTX 3060 Ti specific optimizations | ✅ GPU detected, warmed up in <5s, optimal batch size determined automatically | Dev | 2h
  *Completed: 2025-09-17, Author: Claude Code*

---

## PHASE 1 — Tree Core

- [x] **T005** [P] Contract test for MCTS API | Create tests/contract/test_mcts_api.py testing all functions in contracts/mcts_api.py, verify signatures match exactly, tests must FAIL initially | ✅ All API contract tests fail with NotImplementedError, test coverage 100% | Dev | 2h
  *Completed: 2025-09-17, Author: Claude Code*

- [x] **T006** Implement SoA memory layout | Create cpp_extensions/mcts/tree.hpp with aligned float32 arrays for N,W,P,VL, int32 arrays for parent/child indices, uint8 flags array, 64-byte alignment for SIMD | ✅ Memory layout uses <64 bytes per node, arrays aligned to 64-byte boundaries, supports 50M nodes | Dev | 4h
  *Completed: 2025-09-17, Author: Claude Code*

- [x] **T007** Node pool pre-allocation | Extend tree.hpp with pre-allocated node pools, index-based node references, memory management with reuse, bounds checking | ✅ Tree memory <1GB for 10M nodes (270MB achieved), no malloc/free in hot paths, O(1) node allocation (330M allocations/sec) | Dev | 3h
  *Completed: 2025-09-17, Author: Claude Code, Commit: a7a1d7e*

- [x] **T008** Vectorized PUCT selection | Create cpp_extensions/mcts/selection.cpp with vectorized UCB calculation, single-pass child selection, SIMD optimizations for AVX2 | ✅ Selection vectorized, 3.6-5.2x faster than naive implementation (target achieved with realistic child counts), works with variable child counts | Dev | 4h
  *Completed: 2025-09-17, Author: Claude Code*

- [x] **T009** Virtual loss mechanism | Implement cpp_extensions/mcts/virtual_loss.cpp with atomic virtual loss application/removal, thread-safe path traversal, configurable VL magnitude | ✅ Virtual loss prevents duplicate selection, atomic operations working, configurable +1.0 default | Dev | 3h
  *Completed: 2025-09-18, Author: Claude Code*

- [x] **T010** Value backup with sign flipping | Create cpp_extensions/mcts/backup.cpp with atomic visit count/value updates, proper value sign alternation per ply, path traversal from leaf to root | ✅ Backup correctly flips signs each level, atomic updates working, visit counts accurate | Dev | 3h
  *Completed: 2025-09-18, Author: Claude Code*

- [ ] **T011** Single-threaded MCTS integration test | Create tests/integration/test_mcts_single_thread.py testing complete MCTS cycle: select→expand→evaluate→backup, verify tree integrity | Single-threaded search completes, tree structure valid, performance >10k nodes/sec | Dev | 2h

---

## PHASE 2 — Inference Worker

- [ ] **T012** [P] Contract test for inference API | Create tests/contract/test_inference_api.py testing all functions in contracts/inference_api.py, verify GPU/CPU compatibility, batch processing | All inference API contract tests fail with NotImplementedError, covers all use cases | Dev | 2h

- [ ] **T013** ResNet architecture implementation | Create src/neural/model.py with ResidualBlock+SE attention, 20 blocks with 256 channels, policy and value heads, mixed precision support | Model forward pass works, parameters ~10M, fits in 8GB VRAM with batch size 64 | Dev | 4h

- [ ] **T014** GPU inference worker thread | Create src/neural/inference_worker.py with dedicated thread, queue-based communication, dynamic batching logic, timeout mechanism | Worker thread starts/stops cleanly, processes requests from queue, respects batch size limits | Dev | 4h

- [ ] **T015** Dynamic micro-batching | Extend inference_worker.py with count-based (≥32) OR timeout-based (≤3ms) batching, batch formation and dispatch | Batching achieves target parameters: ≥32 positions OR ≤3ms timeout, GPU utilization >80% | Dev | 3h

- [ ] **T016** Mixed precision inference | Integrate torch.cuda.amp.autocast in inference_worker.py, fp16 computation with fp32 fallback, gradient scaling for training | Inference uses fp16, 2x memory efficiency, no accuracy degradation, automatic fallback | Dev | 2h

- [ ] **T017** Pinned memory optimization | Add pinned CUDA memory buffers, pre-allocated input/output tensors, efficient H2D/D2H transfers | Memory transfers optimized, buffers reused, no allocation in inference loop | Dev | 2h

- [ ] **T018** CPU fallback mechanism | Create src/neural/cpu_inference.py with CPU-only inference path, automatic fallback on CUDA OOM, performance monitoring | CPU inference works, automatic fallback on GPU failure, degrades gracefully | Dev | 3h

- [ ] **T019** Inference integration test | Create tests/integration/test_inference_integration.py testing full inference pipeline with multiple threads, batch formation, result distribution | Inference pipeline handles concurrent requests, results correctly distributed to threads | Dev | 2h

---

## PHASE 3 — Game Adapters

- [ ] **T020** [P] Gomoku game implementation | Create cpp_extensions/games/gomoku.cpp with 15x15 board, 5-in-a-row detection, legal move generation, feature extraction (7 planes) | Gomoku rules correct, legal moves accurate, feature extraction matches spec, terminal detection works | Dev | 3h

- [ ] **T021** [P] Chess game implementation | Create cpp_extensions/games/chess.cpp with full chess rules, castling, en passant, Chess960 support, feature extraction (12 planes) | Chess rules complete including special moves, Chess960 positions generated, legal move validation | Dev | 4h

- [ ] **T022** [P] Go game implementation | Create cpp_extensions/games/go.cpp with variable board sizes 9x9-19x19, capture detection, ko rule, feature extraction (17 planes) | Go rules implemented, captures work correctly, ko prevention, supports multiple board sizes | Dev | 4h

- [ ] **T023** [P] Game adapter interface | Create cpp_extensions/games/interface.cpp with unified GameState interface, game type detection, polymorphic dispatch | All games implement common interface, game switching works without code changes | Dev | 2h

- [ ] **T024** [P] Python bindings for games | Create cpp_extensions/games/python_bindings.cpp with pybind11 bindings for all games, numpy array compatibility | Python can instantiate any game, call methods, get numpy arrays for features | Dev | 3h

- [ ] **T025** [P] Game rule unit tests | Create tests/unit/test_game_rules.py with comprehensive rule verification for all games, edge cases, performance tests | All game rules verified correct, edge cases handled, no illegal moves possible | Dev | 3h

---

## PHASE 4 — Self-Play & Replay

- [ ] **T026** [P] Contract test for training API | Create tests/contract/test_training_api.py testing all functions in contracts/training_api.py, self-play generation, experience buffer operations | All training API contract tests fail with NotImplementedError, comprehensive coverage | Dev | 2h

- [ ] **T027** Asynchronous search coordinator | Create src/core/search_coordinator.py with thread pool management, inference request queueing, result distribution, performance monitoring | Coordinator manages multiple search threads, handles inference requests asynchronously | Dev | 4h

- [ ] **T028** Self-play game generator | Create src/training/self_play.py with temperature scheduling, Dirichlet noise injection, game outcome determination, position augmentation | Self-play generates complete games, applies noise for exploration, saves training positions | Dev | 4h

- [ ] **T029** Memory-mapped experience buffer | Create src/training/experience_buffer.py with mmap storage, Parquet format, LRU RAM cache, efficient random sampling | Buffer stores 1M+ experiences, memory-mapped for large datasets, fast random access | Dev | 3h

- [ ] **T030** Experience replay sampling | Extend experience_buffer.py with uniform random sampling, game balance, batch construction for training | Sampling produces balanced batches, respects game type distribution, efficient iteration | Dev | 2h

- [ ] **T031** Self-play generation test | Create tests/integration/test_self_play_generation.py testing complete game generation, experience extraction, buffer storage | End-to-end self-play works, generates training data, stores in buffer correctly | Dev | 3h

---

## PHASE 5 — Training

- [ ] **T032** Model trainer implementation | Create src/training/trainer.py with AdamW optimizer, cosine learning rate scheduling, mixed precision training, gradient clipping | Trainer processes batches, applies updates, learning rate schedules correctly | Dev | 4h

- [ ] **T033** Training loop orchestration | Create src/training/training_loop.py with self-play → experience → training cycle, checkpoint management, validation tracking | Training loop runs continuously, manages checkpoints, tracks progress metrics | Dev | 3h

- [ ] **T034** Model evaluation system | Create src/training/evaluator.py with head-to-head model comparison, ELO rating calculation, performance measurement | Evaluation compares models accurately, calculates meaningful strength ratings | Dev | 3h

- [ ] **T035** Training stability monitoring | Extend trainer.py with NaN detection, gradient norm monitoring, loss convergence tracking, early stopping | Training stable, detects divergence, automatic recovery, comprehensive monitoring | Dev | 2h

- [ ] **T036** Checkpoint management | Create src/training/checkpoint_manager.py with automatic saving, best model selection, model versioning, cleanup policies | Checkpoints saved automatically, best model tracking, old checkpoints cleaned up | Dev | 2h

- [ ] **T037** Training pipeline integration test | Create tests/integration/test_training_pipeline.py testing full pipeline from self-play through model updates | Complete training iteration works, model improves measurably, checkpoints saved | Dev | 3h

---

## PHASE 6 — Performance Tuning

- [ ] **T038** [P] Thread count optimization | Create scripts/tune_threads.py with parameter sweep for thread counts 1-16, performance measurement, contention detection | Optimal thread count determined (target 8-10), contention <10%, peak performance achieved | Dev | 3h

- [ ] **T039** [P] Virtual loss magnitude tuning | Create scripts/tune_virtual_loss.py with VL values 0.5-3.0, thread efficiency measurement, exploration balance | Optimal VL found (target ~1.0), thread efficiency >85%, exploration preserved | Dev | 2h

- [ ] **T040** [P] Batch size optimization | Create scripts/tune_batch_size.py with GPU memory profiling, throughput measurement, latency analysis | Optimal batch sizes per game determined, GPU utilization >80%, memory <85% VRAM | Dev | 3h

- [ ] **T041** [P] Inference timeout tuning | Create scripts/tune_timeout.py with timeout values 1-10ms, throughput vs latency analysis | Optimal timeout found (target 3ms), balances throughput and responsiveness | Dev | 2h

- [ ] **T042** 1-hour soak test | Create tests/soak/test_memory_stability.py with continuous operation, memory leak detection, performance degradation monitoring | 1-hour test passes, memory growth <10MB/hour, performance stable, no crashes | Dev | 4h

- [ ] **T043** Performance regression suite | Create tests/performance/test_benchmarks.py with automated performance testing, regression detection, reporting | Benchmarks detect performance regressions, report clear metrics, run in CI | Dev | 3h

---

## PHASE 7 — Packaging & Docs

- [ ] **T044** [P] Docker containerization | Create Dockerfile with CUDA 12.x base, optimized build, production configuration, health checks | Docker image builds successfully, runs in container, all functionality works | Dev | 3h

- [ ] **T045** [P] Configuration management | Create config/default.yaml with all tunable parameters, environment overrides, validation | Configuration system works, parameters tunable, defaults sensible, validation prevents errors | Dev | 2h

- [ ] **T046** [P] Operations runbook | Create docs/operations.md with deployment procedures, monitoring setup, troubleshooting guide, maintenance tasks | Operations documented, deployment repeatable, troubleshooting comprehensive | Dev | 3h

- [ ] **T047** [P] API documentation | Create docs/api.md with complete API reference, usage examples, parameter descriptions | API fully documented, examples work, parameters clearly explained | Dev | 2h

- [ ] **T048** [P] Training guide | Create docs/training_guide.md with hyperparameter recommendations, game-specific settings, troubleshooting | Training guide enables users to achieve target performance, troubleshooting comprehensive | Dev | 3h

---

## PHASE 8 — Hardening

- [ ] **T049** [P] Sanitizer builds | Update pyproject.toml with AddressSanitizer and ThreadSanitizer builds, CI integration | Sanitizer builds pass, detect no memory errors or race conditions | Dev | 2h

- [ ] **T050** [P] OOM recovery mechanisms | Add CUDA OOM detection in inference_worker.py, automatic batch size reduction, graceful degradation | OOM handled gracefully, system continues operation, batch size adjusts automatically | Dev | 3h

- [ ] **T051** [P] Memory leak detection | Create scripts/check_memory_leaks.py with valgrind integration, Python memory profiling, automated testing | Memory leak detection integrated, no leaks in core components, automated monitoring | Dev | 3h

- [ ] **T052** [P] Error handling hardening | Add comprehensive error handling throughout codebase, graceful degradation, informative error messages | Error handling comprehensive, system stable under adverse conditions, errors informative | Dev | 4h

- [ ] **T053** Final integration test | Create tests/integration/test_full_system.py testing complete training run from initialization to superhuman performance | System achieves all performance targets, training converges, no regressions | Dev | 4h

---

## Dependencies

### Critical Path Dependencies:
- **T005 (MCTS contract test) → T006-T011** (MCTS implementation)
- **T012 (Inference contract test) → T013-T019** (Inference implementation)
- **T026 (Training contract test) → T027-T037** (Training implementation)
- **T006-T010 → T011** (Tree components → MCTS integration)
- **T013-T018 → T019** (Inference components → Integration)
- **T020-T025 → T027** (Games → Search coordinator)
- **T027-T031 → T037** (Self-play components → Training integration)

### Parallel Execution Groups:
```bash
# Phase 0 - Setup (can run in parallel)
Task: "Setup CI/CD pipeline"
Task: "Implement basic telemetry framework"
Task: "GPU warmup and device detection"

# Phase 1 - After T005, these can run in parallel:
Task: "Implement SoA memory layout"
Task: "Vectorized PUCT selection"
Task: "Virtual loss mechanism"

# Phase 3 - Game implementations (fully parallel):
Task: "Gomoku game implementation"
Task: "Chess game implementation"
Task: "Go game implementation"
Task: "Game adapter interface"
Task: "Python bindings for games"
Task: "Game rule unit tests"
```

## Definition of Done

**Performance Targets Met:**
- [ ] 30,000+ simulations/second with neural network inference
- [ ] 80-92% GPU utilization sustained during search
- [ ] <1GB memory footprint for 10M node trees
- [ ] 200+ self-play games/hour generation rate
- [ ] No memory leaks detectable over 24-hour continuous run
- [ ] Thread-safe operation with 12 parallel workers
- [ ] Superhuman Gomoku play achieved within 48 hours training

**Quality Gates Passed:**
- [ ] All unit tests passing (100% pass rate)
- [ ] All integration tests passing
- [ ] All contract tests passing
- [ ] Soak test (1-hour) passes with memory stability
- [ ] Performance benchmarks meet or exceed targets
- [ ] Thread safety verified with sanitizers
- [ ] Documentation complete and comprehensive

**Review & Acceptance Checklist (from spec.md):**
- [ ] Unit tests verify backup value sign flipping at each tree level
- [ ] Unit tests confirm illegal move masking prevents invalid selections
- [ ] Unit tests validate terminal state detection across all supported games
- [ ] Performance benchmarks achieve 30k+ simulations/second targets
- [ ] GPU utilization metrics consistently show 80-92% usage during search
- [ ] Memory profiling confirms <1GB tree footprint and no leaks over 1-hour operation
- [ ] Deterministic test mode produces identical results with fixed random seed
- [ ] Documentation coverage includes all hyperparameters and their tuning rationales

**Open Questions Resolved:**
- [ ] Feature plane counts determined for each game (Gomoku: 7, Chess: 12, Go: 17)
- [ ] CPUCT exploration schedules optimized per game type
- [ ] Dirichlet noise alpha values tuned (Gomoku: 0.3, Chess: 0.2, Go: 0.03)
- [ ] Transposition table sizing optimized relative to available memory
- [ ] Virtual loss magnitude finalized through empirical testing
- [ ] Batch timeout values balanced for latency vs GPU utilization

---

**Total Estimated Effort:** 149 hours across 53 tasks
**Critical Path:** ~35 hours (setup → contracts → core implementations → integration)
**Parallel Opportunities:** 28 tasks marked [P] can execute concurrently

This task breakdown is immediately executable with each task containing specific acceptance criteria and file paths for implementation.