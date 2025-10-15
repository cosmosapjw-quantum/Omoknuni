# Specification 004: MCTS Throughput Recovery & Multi-Actor Self-Play Data Generation

**Version**: 2.0 (Multi-Actor Self-Play Edition)
**Status**: ACTIVE
**Last Updated**: 2025-10-14
**Target Hardware**: AMD Ryzen 9 5900X (12C/24T) + NVIDIA RTX 3060 Ti (8GB VRAM)
**Games**: Gomoku 15×15, Chess 8×8, Go 9×9
**Authority**: Supersedes SPECIFICATION.md v1.1; implements CONSTITUTION.md v2.0

---

## 1. Executive Summary

### 1.1 Purpose

This specification defines requirements to recover and maximize MCTS throughput on CPU-parallel architectures with Python-based neural network inference, targeting **≥8,000 simulations/second** sustained (stretch ≥10,000) on Ryzen 5900X + RTX 3060 Ti hardware, plus enable concurrent multi-actor self-play for data generation at **200-300 games/hour**.

**Current Status** (per ./review.txt, authoritative):
- **Baseline**: 3,831 sims/sec (Spec 003 configuration, exact params TBD)
- **Current**: 2,147 sims/sec (56% regression from baseline, cause: state cloning + coordination overhead)
- **Target**: ≥8,000 sims/sec (2.1× baseline, 3.7× current, hardware-grounded)

### 1.2 Primary Bottlenecks (from ./review.txt + Validation 2025-10-14)

**Critical Finding**: GPU is NOT the bottleneck; CPU MCTS coordination is.

**Time Distribution** (review.txt lines 14-19):
- Neural network inference (GPU): **32.8%**
- MCTS coordination (CPU): **67.2%** ← PRIMARY BOTTLENECK

**Conclusion**: Even if GPU was 100% efficient, only 1.3× improvement possible. CPU coordination is the real limiter.

1. **State Cloning Waste** (HIGHEST PRIORITY - review.txt lines 37-54):
   - Current: 2-3× clones per simulation
   - Locations:
     * `continuous_simulation_runner.cpp:78` - Clone root state
     * `continuous_simulation_runner.cpp:115` - Clone for queue
     * `async_inference_queue.cpp:37` - Clone on EVERY retry (most wasteful!)
   - Impact: Wasteful CPU cycles, memory allocation pressure, Python GC overhead
   - Fix: Thread-local state pools + move semantics + `copyFrom()`
   - Expected Gain: 1.3-1.5× throughput

2. **Thread Contention & Locking** (review.txt lines 71-110):
   - Thread idle: 60% of execution (~1.489s out of 2.5s search, line 102)
   - Global allocation mutex serializes expansions (lines 71-78)
   - Spin-waiting on results wastes CPU (lines 125-136)
   - Poor scaling: 4T→8T only 12.5% efficiency gain (lines 65-69)
   - Fix: Condition variables + thread-local arenas + relaxed atomics
   - Expected Gain: 1.5-2.0× throughput

3. **Thread/CPU Affinity** (review.txt lines 244-250):
   - Current: `hash(thread::id) % 24` suboptimal
   - Issue: May use SMT siblings before saturating physical cores
   - Fix: Pin to physical cores (0-11) first, explicit CCD0/CCD1 pinning
   - Expected Gain: 1.15× throughput

4. **Python ↔ C++ Interface Overhead** (review.txt lines 258-307):
   - Batch callback overhead, list→array conversions
   - DLPack fast path validation needed
   - Fix: Ensure zero-copy active, return NumPy arrays, use move semantics
   - Expected Gain: 1.1-1.2× throughput

5. **Feature Extraction** (NOT PRIMARY BOTTLENECK - Validated 2025-10-14):
   - OpenMP present in code: `dlpack_bridge.cpp:431-434`
   - Testing shows: 8.64ms @ 1 thread → 1.57ms @ 12 threads (5.5× speedup)
   - **BUT**: MCTS throughput SAME (1,543 vs 1,529 sims/sec) regardless of OMP threads
   - Conclusion: Feature extraction NOT limiting factor (batching amortizes cost)
   - Status: OpenMP working correctly, NOT a priority fix

6. **GPU Inference** (32.8% of runtime, SECONDARY):
   - RTX 3060 Ti @ FP16: 30.7ms per batch-64 → theoretical max 2,014 states/sec
   - Current utilization: 11.2-68% (variable, depends on batching)
   - Target: 80-95% utilization with multi-actor batching
   - FP16 mixed precision: Already implemented (T008f ✅, 1.72× speedup validated)

### 1.3 Solution Architecture

**Single MCTS Optimizations** (Corrected Priority Order):
1. Eliminate redundant state cloning (2-3× → ≤1×) - HIGHEST PRIORITY
2. Fix thread contention (condition variables + arenas + relaxed atomics)
3. Tune thread affinity (Ryzen dual-CCD topology, physical cores first)
4. Streamline Python ↔ C++ interface (DLPack fast path, NumPy arrays, move semantics)
5. FP16 mixed precision (ALREADY COMPLETE - validated 1.72× speedup, T008f ✅)

**Multi-Actor Self-Play** (Phase 5):
- Many single-thread MCTS games → one centralized inference queue
- Target: 8-12 concurrent games feeding single GPU batcher
- Expected: Avg batch size ≥51/64 (0.8× max), GPU util 80-95%

**NN-Eval Cache** (Tier A, Phase 6 optional):
- Cache `(hash, π, V)` tuples keyed by Zobrist hash
- Tree statistics (N, W, Q) remain per-node (NOT shared across parents)
- Expected gains: Chess 20-50%, Gomoku 10-30%, Go 15-35% GPU call reduction

---

## 2. Goals & Success Criteria

### 2.1 Measurable KPIs (Acceptance Criteria)

**G1: Absolute Throughput** (PRIMARY):
| Configuration | Minimum | Target | Stretch | Measurement |
|---------------|---------|--------|---------|-------------|
| Single MCTS (13×13 Gomoku) | 6,000 sims/sec | **≥8,000** | ≥10,000 | `benchmark_throughput.py --threads 8` |
| Multi-actor (8 games) | N/A | **≥8,000** | ≥10,000 | Same tool, multi-process mode |
| Chess 8×8 | 6,500 sims/sec | ≥8,500 | ≥10,500 | Same |
| Go 9×9 | 7,000 sims/sec | ≥9,000 | ≥11,000 | Same |

**Evidence**: Review.txt analysis shows GPU @ FP16 caps at 8-10k states/sec on RTX 3060 Ti; this is the hardware-grounded maximum.

**G2: GPU Utilization**:
- Single MCTS: **≥80%** during search (batch=64, timeout≤2ms)
- Multi-actor: **≥85%** sustained (8-12 games, optimal actor count)
- Measurement: `nvidia-smi dmon -s u -i 0` during benchmark

**G3: CPU Coordination Overhead**:
- Target: **<30%** of total time (currently 67.2% per review.txt)
- Breakdown: Thread idle ≤12%, feature prep ≤5%, Python/GIL ≤5%, sync ≤8%
- Measurement: C++ instrumentation in `ContinuousSimulationRunner`

**G4: Feature Preparation Speed**:
- Current: 7.5ms per batch-64 (CRITICAL bottleneck)
- Target: **≤1.0ms** per batch-64 with OpenMP parallelization
- Measurement: `profile_tensor_creation.py` (T-VALID-2 protocol)

**G5: State Cloning Efficiency**:
- Current: 2-3× clones per simulation (review.txt finding)
- Target: **≤1× clone** per simulation (ideally zero extra copies)
- Implementation: Thread-local state pooling, move semantics

**G6: Thread Scaling Efficiency**:
- 4 threads: **≥70%** efficiency (minimum), ≥80% (target)
- 8 threads: **≥70%** efficiency (target, relaxed from 75% due to mutex contention)
- Measurement: `(actual_throughput) / (single_thread × num_threads)`

**G7: Multi-Actor Self-Play** (Phase 5):
- Data generation: **200-300 games/hour** @ 800 sims/move (Gomoku 15×15)
- Actor count: **8-12 concurrent games** feeding single batcher
- Avg batch size: **≥51/64** (0.8× consistency)
- GPU util: **85-95%** during multi-actor runs

**G8: Search Quality Preservation**:
- Win rate: **≥99.5%** vs baseline (1000+ games, 95% confidence)
- Policy agreement: **≥95%** top-move agreement (1000-position test set)
- Value MSE: **≤0.01** vs baseline estimates
- Collision rate: **≤5%** path collisions (threads selecting same node)

### 2.2 Non-Goals (Explicitly Out of Scope)

Per CONSTITUTION.md Section 1.4:
- ❌ **NO libtorch** (C++ PyTorch inference)
- ❌ **NO TensorRT/ONNX** model conversion
- ❌ **NO root parallelization** (separate trees per thread)
- ❌ **NO GPU-MCTS** (GPU-resident trees)
- ❌ **NO full DAG TT** (shared statistics across parents, deferred to Phase 7)
- ❌ **NO training pipeline** optimizations (unless blocking throughput)

**Rationale**: GPU is NOT the bottleneck (32.8% of time per review.txt). Maintaining Python PyTorch provides flexibility for model experimentation without sacrificing achievable performance.

---

## 3. User Stories

### US1: Self-Play Training Operator

**As a** reinforcement learning researcher
**I want to** generate 200-300 self-play games per hour at 800 sims/move
**So that** I can train superhuman Gomoku models within 48 hours

**Acceptance Criteria**:
- 8,000 sims/sec × 800 sims/move = 100ms per move
- 100-move game = 10 seconds per game
- 200 games/hour = 1 game per 18 seconds (includes setup overhead)
- GPU utilization ≥85%, batch size consistently ≥51/64

**Validation**: Run `scripts/selfplay.py --games 20 --simulations 800` and measure throughput over 20-game batch.

### US2: Interactive Play & Analysis

**As a** competitive player
**I want to** receive move recommendations within 3 seconds
**So that** I can use the engine for real-time game analysis

**Acceptance Criteria**:
- 1600 simulations ≤ 3 seconds (533 sims/sec minimum, easily met at 8k)
- Policy distribution and value estimate displayed
- Top 5 moves with visit counts and Q-values
- Consistent latency (CV < 10%)

**Validation**: Interactive play mode with fixed 1600-sim budget, measure p95 latency.

### US3: Performance Engineer

**As a** performance engineer
**I want to** measure throughput with deterministic, reproducible configurations
**So that** I can validate optimization effectiveness with statistical rigor

**Acceptance Criteria**:
- Fixed seed, fixed game state, fixed simulation count
- N≥10 independent runs with CV < 5%
- Detailed breakdown: selection, expansion, inference, backup time
- Automated regression alerts if throughput < 95% baseline

**Validation**: Benchmark suite passes (`pytest -m performance`), historical CSV log updated.

---

## 4. Functional Requirements

### FR1: CPU-Side MCTS Optimization

**FR1.1: Parallel Feature Extraction** (CRITICAL FIX):
- **Current**: Sequential extraction (7.5ms per batch-64)
- **Required**: Add `#pragma omp parallel for` to `dlpack_bridge.cpp:431-434`
- **Expected**: 7.5ms → <1.0ms with 12-thread parallelization
- **Validation**: T-VALID-2 profiling shows ≤1.0ms mean, CV < 10%

**FR1.2: State Cloning Elimination**:
- Implement thread-local `IGameState` pools (reuse across simulations)
- Add `copy_from(other)` method for efficient state reset (no heap allocations)
- Transfer ownership via `std::move` to queue (not clone) for in-flight expansions
- **Validation**: Memory profiler shows constant allocation, no growth over 1000+ searches

**FR1.3: Thread Affinity (Ryzen 5900X)**:
- Pin threads to physical cores: CCD0 (cores 0-5), CCD1 (cores 6-11)
- Avoid SMT siblings (cores 12-23) unless >12 threads required
- **Validation**: `lscpu --extended` confirms pinning, perf reports reduced cache misses

**FR1.4: Condition Variable Coordination** (T006c COMPLETE):
- Replace spin-waits with `std::condition_variable` blocking
- Notify threads when batch results ready (NOT polling with 10μs sleeps)
- **Validation**: Thread CPU usage drops to near-zero when idle (no busy-wait)

**FR1.5: WU-UCT Virtual Loss** (T001 COMPLETE):
- Visit-only virtual loss: `PUCT = Q + c*P*sqrt(N_parent)/(1+N+VL)`, where Q = W/N pure
- Default magnitude: 1.0 (tunable 0.5-3.0)
- **Validation**: Unit tests verify Q unchanged with in-flight simulations, collision rate ≤5%

### FR2: Multi-Actor Self-Play Architecture

**FR2.1: Concurrent Game Processes**:
- Run **G=8-12** concurrent self-play games (separate processes or threads)
- Each actor: 1-2 MCTS threads per game (NOT 8-12 threads per game)
- Shared global inference queue (all actors push to same MPMC ring buffer)
- **Rationale**: Review.txt explicitly recommends "many single-thread actors → one inference server" to maximize GPU batching

**FR2.2: Centralized Batch Coordinator**:
- Single `BatchInferenceCoordinator` instance (receives requests from all actors)
- Batch collection: min_size=64, timeout=1-2ms (tunable)
- Result demultiplexing: return results to originating actor via game_id
- **Validation**: Average batch size ≥51/64 (0.8× consistency), GPU util 85-95%

**FR2.3: Backpressure & Fairness**:
- Per-actor visit budget (e.g., 800 simulations/move)
- Token-bucket rate limiting (limit in-flight requests per actor)
- Adaptive actor scaling: increase G until GPU util ~90% or batch size plateaus
- **Validation**: No actor starvation (min 10% of total throughput), GPU util does not degrade with actor count

### FR3: NN-Evaluation Cache (Tier A, Phase 6 Optional)

**FR3.1: Policy/Value Cache Design**:
- Cache stores `(hash, policy_topk, value)` keyed by Zobrist hash (64-bit)
- Tree statistics (N, W, Q) remain per-node (NOT shared across parents)
- On expansion: check cache → if hit, skip GPU; else enqueue
- On inference: store result in cache for future reuse
- **Key Format** (Markov-minimal): board + side + rule-critical flags (NOT full history)

**FR3.2: Memory & Concurrency**:
- Cache size: 1M-10M entries (tunable, target 2-8GB with quantization)
- Quantization: FP16 or int8 for policy, top-K=16-48 moves (not full board)
- Eviction: Per-net SLRU (Segmented LRU) with `net_version` tagging
- Concurrency: Sharded hash table (64 shards) with reader-writer locks or lock-free F14
- **Validation**: Hit rate measured, 20-50% for Chess, 10-30% Gomoku, 15-35% Go (per review.txt estimates)

### FR4: FP16 Mixed Precision (T008f COMPLETE)

**FR4.1: GPU Inference Acceleration**:
- Use `torch.cuda.amp.autocast()` for FP16 tensor core utilization
- **Validated** (T-VALID-1): 1.72× speedup (52.83ms → 30.69ms @ batch-64)
- Numerical stability: Policy MSE 0.000007, Value MSE 0.000000 (both < 0.01 threshold)

---

## 5. Non-Functional Requirements

### NFR1: Performance Targets (Hardware-Grounded)

**Single MCTS (Phase 4)**:
| Metric | Minimum Viable | Target (Realistic) | Stretch | Hardware Limit |
|--------|---------------|-------------------|---------|---------------|
| Simulations/sec | ≥6,000 | **≥8,000** | ≥10,000 | ~10,000 (GPU cap) |
| vs Baseline (3,831) | 1.6× | **2.1×** | 2.6× | 2.6× |
| vs Current (2,147) | 2.8× | **3.7×** | 4.7× | 4.7× |
| GPU Utilization | ≥75% | **≥80%** | ≥85% | ~90% (realistic) |
| Thread Efficiency (8T) | ≥60% | **≥70%** | ≥75% | ~80% (theoretical) |

**Multi-Actor Self-Play (Phase 5)**:
| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| Games/hour (Gomoku) | 150 | **200-300** | 400 |
| Actor count | 6 | **8-12** | 14 |
| GPU utilization | 80% | **85-95%** | 95%+ |
| Avg batch size | 40 | **≥51** (0.8×64) | 58+ |

**Evidence**: Review.txt GPU @ FP16 analysis → RTX 3060 Ti caps at 8-10k states/sec maximum.

### NFR2: Latency Budgets (per 1000 simulations @ 8k target)

| Component | Current | Target | Critical Path |
|-----------|---------|--------|---------------|
| MCTS Coordination | 240ms (67.2%) | **≤40ms** (25%) | State pooling, sync |
| GPU Inference | 117ms (32.8%) | **≤80ms** (67%) | FP16, batch tuning |
| Feature Extraction | 75ms (21%) | **≤10ms** (8%) | **OpenMP fix (CRITICAL)** |
| Thread Idle | 150ms (42%) | **≤15ms** (12%) | Condition variables, affinity |
| **Total** | **357ms** | **≤125ms** | All optimizations |

### NFR3: Memory Footprint

- Tree: <1GB for 10M nodes (achieved: 270MB with 27-byte SoA)
- Queue: 1MB fixed (4096-entry ring buffer)
- DLPack buffers: <10MB pinned memory
- NN-eval cache: 2-8GB (optional Phase 6, tunable)
- **Total**: <1.3GB (single MCTS) or <10GB (with cache + multi-actor)

### NFR4: Correctness & Quality

- **Thread Safety**: TSan clean (zero data races @ 24 threads)
- **Search Quality**: Win rate ≥99.5% vs baseline, policy agreement ≥95%
- **Collision Rate**: ≤5% path collisions
- **Memory Stability**: 24-hour soak test, RSS growth <1MB/hour

### NFR5: Reproducibility

- **Deterministic seeds**: Fixed seed → identical throughput (±2% CV over 3 runs)
- **Benchmark gates**: `pytest -m performance` passes before merge
- **Regression alerts**: Throughput < 95% baseline triggers CI failure

---

## 6. Implementation Plan Summary

### Phase 1: Quick Wins ✅ COMPLETE
- T001-T005: WU-UCT, epoch clearing, busy-edge, affinity, metrics
- **Delivered**: Collision rate <0.5%, thread efficiency foundations

### Phase 2: Architecture ✅ COMPLETE
- T006-T010: Lock-free queue, DLPack, FP16, thread arenas, persistent coordinator
- **Delivered**: Zero-copy pipeline, 1.72× GPU speedup (FP16), condition variables

### Phase 3: Optimizations ✅ PARTIAL (85%)
- T011 ✅ Persistent coordinator, T014 ✅ Batched results
- T012-T013-T015 deferred (relaxed atomics, prefetching, hot/cold separation)

### Phase 4: Validation & Fixes 🔴 REQUIRED
- **CRITICAL FIX**: Add OpenMP to `dlpack_bridge.cpp:431-434` (7.5ms → <1ms)
- **CRITICAL FIX**: Eliminate redundant state cloning (2-3× → ≤1×)
- T016: Comprehensive benchmarking (measure gains from OpenMP fix)
- T017: Baseline investigation (reproduce 3,831 sims/sec config)
- T018: Thread tuning (optimal count: 2-8 based on profiling)
- T019: Batch/timeout tuning (optimal: batch-64 @ 1-2ms timeout)

### Phase 5: Multi-Actor Self-Play (NEW)
- Implement concurrent game processes (8-12 actors)
- Shared inference queue integration
- Adaptive actor scaling based on GPU util
- **Expected**: 200-300 games/hour, 85-95% GPU util

### Phase 6: NN-Eval Cache (OPTIONAL)
- Zobrist hashing for Gomoku/Chess/Go
- Tier A cache (policy/value only, NO shared stats)
- Sharded hash table with SLRU eviction
- **Expected**: 10-50% GPU call reduction depending on game

---

## 7. Acceptance Criteria (Phase 4 Completion)

### Must-Have (Blocking Release):
- [ ] **FR1.1**: OpenMP fix applied, T-VALID-2 passes (≤1.0ms tensor creation)
- [ ] **FR1.2**: State cloning ≤1× per simulation (memory profiler validates)
- [ ] **G1**: Throughput ≥8,000 sims/sec (Gomoku 13×13, 8 threads, batch-64)
- [ ] **G2**: GPU utilization ≥80% during search
- [ ] **G3**: CPU coordination <30% of total time
- [ ] **G8**: Search quality preserved (win rate ≥99.5%, policy agreement ≥95%)

### Should-Have (Quality Goals):
- [ ] **G6**: Thread efficiency ≥70% @ 8 threads
- [ ] **NFR2**: Latency budgets met (≤125ms per 1000 sims)
- [ ] **NFR4**: TSan clean, 24-hour soak test passes

### Nice-to-Have (Stretch Goals):
- [ ] **G1 Stretch**: Throughput ≥10,000 sims/sec
- [ ] **G7**: Multi-actor self-play 200-300 games/hour
- [ ] **FR3**: NN-eval cache 20-50% hit rate (Chess)

---

## 8. Risks & Mitigations

### R1: OpenMP Fix Insufficient (LOW PROBABILITY)
- **Risk**: 7.5ms → 3ms instead of <1ms (false sharing, bandwidth saturation)
- **Mitigation**: Profile with `perf mem`, optimize memory access patterns
- **Contingency**: Accept <1.5ms, tune thread count to compensate

### R2: State Cloning Refactor Breaks Correctness (MEDIUM PROBABILITY)
- **Risk**: Thread-local pooling introduces subtle bugs (use-after-free, race conditions)
- **Mitigation**: Extensive unit tests, TSan validation, incremental rollout
- **Contingency**: Rollback to `clone()`, optimize other paths

### R3: Baseline 3,831 Unreproducible (HIGH PROBABILITY)
- **Risk**: Cannot validate improvement claims, T017 investigation fails
- **Mitigation**: 2-day time-boxed investigation, systematic config sweep
- **Contingency**: Use 2,147 as new baseline, adjust targets to 10× improvement (21,470 sims/sec → revise to 8k realistic)

### R4: Multi-Actor Adds Latency Variance (MEDIUM PROBABILITY)
- **Risk**: Per-actor throughput highly variable, some games starve
- **Mitigation**: Per-actor timeouts, token-bucket backpressure, priority queuing
- **Contingency**: Use 6-8 actors (conservative), accept 75% GPU util

### R5: Thread Contention Saturates (HIGH PROBABILITY, OBSERVED)
- **Risk**: Current evidence shows 60% thread idle, efficiency collapse beyond 4 threads
- **Mitigation**: Affinity tuning, relaxed atomics, per-thread result queues
- **Contingency**: Scale to 16 threads with SMT, accept 50-60% efficiency

---

## 9. Measurement & Telemetry

### KPI Dashboard (Tracked per Benchmark Run):
1. **Absolute throughput** (sims/sec)
2. **GPU utilization** (% during search)
3. **Thread efficiency** (% vs linear scaling)
4. **Average batch size** (positions per GPU call)
5. **Coordination overhead** (% of total time)
6. **Feature extraction time** (ms per batch-64)
7. **Collision rate** (% path collisions)
8. **Memory RSS** (GB during search)

### Profiling Protocol:
- **Fixed Configuration**: seed=42, game=gomoku, board=15×15, sims=10000, threads=8, batch=64, timeout=1.0ms
- **Iterations**: N≥10 runs per benchmark
- **Statistics**: Report mean ± stddev, 95% confidence interval, CV < 5% required
- **Storage**: `profiling_results/session_YYYYMMDD_HHMMSS/` with summary.md + CSV data

---

## 10. Glossary

| Term | Definition |
|------|------------|
| **Simulation** | Complete MCTS cycle: selection → expansion → NN evaluation → backpropagation |
| **Throughput** | Simulations per wall-clock second (including all overhead) |
| **Baseline** | 3,831 sims/sec (Spec 003 configuration, exact params TBD via T017) |
| **Current** | 2,147 sims/sec (56% regression, cause: OpenMP missing + state cloning waste) |
| **Target** | ≥8,000 sims/sec (2.1× baseline, hardware-grounded on RTX 3060 Ti @ FP16) |
| **WU-UCT** | Visit-only virtual loss (increments denominator, preserves Q = W/N) |
| **Busy-Edge** | PUCT = -∞ for nodes being expanded (prevents collisions) |
| **DLPack** | Zero-copy tensor protocol (C++ ↔ PyTorch via pinned memory) |
| **MPMC** | Multi-Producer Multi-Consumer lock-free ring buffer |
| **SoA** | Structure-of-Arrays (separate arrays per field for cache efficiency) |
| **Actor** | Single self-play game (1-2 MCTS threads, local tree, shared inference queue) |
| **NN-Eval Cache** | Transposition table for policy/value reuse (Tier A: eval-only, NO shared stats) |
| **Tier A TT** | Cache stores inferences, tree stores statistics (safe design) |
| **Tier B TT** | Full DAG TT with merged statistics across parents (deferred, requires MCGS) |

---

## 11. Numbered Requirements (Traceability)

### Performance Requirements:
1. **REQ-PERF-001**: Throughput ≥8,000 sims/sec (Gomoku 13×13, 8 threads, batch-64)
2. **REQ-PERF-002**: GPU utilization ≥80% during search
3. **REQ-PERF-003**: CPU coordination overhead <30% of total time
4. **REQ-PERF-004**: Feature extraction ≤1.0ms per batch-64
5. **REQ-PERF-005**: Thread efficiency ≥70% @ 8 threads
6. **REQ-PERF-006**: State cloning ≤1× per simulation
7. **REQ-PERF-007**: Multi-actor self-play 200-300 games/hour

### Architecture Requirements:
8. **REQ-ARCH-001**: Python PyTorch inference ONLY (NO libtorch/TensorRT)
9. **REQ-ARCH-002**: Shared tree architecture (NOT root parallelization)
10. **REQ-ARCH-003**: WU-UCT virtual loss (visit-only, pure Q = W/N)
11. **REQ-ARCH-004**: Lock-free MPMC queue (4096 entries, condition variables)
12. **REQ-ARCH-005**: DLPack zero-copy tensors (pinned CPU memory)
13. **REQ-ARCH-006**: Thread-local state pooling (reuse across simulations)
14. **REQ-ARCH-007**: Multi-actor shared inference queue (8-12 games → one batcher)

### Quality Requirements:
15. **REQ-QUAL-001**: Search quality ≥99.5% win rate vs baseline
16. **REQ-QUAL-002**: Policy agreement ≥95% (1000-position test set)
17. **REQ-QUAL-003**: Value MSE ≤0.01 vs baseline
18. **REQ-QUAL-004**: Collision rate ≤5% path collisions
19. **REQ-QUAL-005**: TSan clean (zero data races @ 24 threads)
20. **REQ-QUAL-006**: Memory stability (24-hour soak, RSS growth <1MB/hour)

### Implementation Requirements:
21. **REQ-IMPL-001**: OpenMP parallelization in `dlpack_bridge.cpp:431-434`
22. **REQ-IMPL-002**: Thread affinity for Ryzen 5900X (CCD0/CCD1 pinning)
23. **REQ-IMPL-003**: Condition variables replace spin-waits (T006c)
24. **REQ-IMPL-004**: FP16 mixed precision (T008f, validated 1.72× speedup)
25. **REQ-IMPL-005**: NN-eval cache Tier A (optional Phase 6, Zobrist hash keyed)

---

## 12. Approval & Authority

**This specification is ACTIVE and BINDING as of 2025-10-14.**

**Authority Chain**:
1. CONSTITUTION.md v2.0 (non-negotiable rules)
2. **This spec.md** (functional requirements)
3. plan.md (technical design, TBD via `/speckit.plan`)
4. tasks.md (implementation breakdown, TBD via `/speckit.tasks`)

**Stakeholders**:
- **Product Owner**: cosmosapjw-quantum (user)
- **Implementation Lead**: Claude Code (AI agent)
- **Evidence Base**: ./review.txt (authoritative source-of-truth)

**Change Control**:
All spec changes require:
1. Profiling evidence or failed experiments justifying change
2. Impact analysis (expected throughput delta, affected requirements)
3. Re-execution of `/speckit.plan` and `/speckit.tasks`

**Review Cycle**: After Phase 4/5/6 completion or if throughput < 50% of target

---

## 6. Advanced Optimizations (Optional Phases - Post-8k Target)

### 6.1 Precompute Legal Moves (Phase 2 Enhancement)

**FR6.1: Legal Move Precomputation**:

**Problem** (review.txt lines 189-201): `expand_node_with_result()` calls `state.getLegalMoves()` during expansion, accessing state object unnecessarily.

**Requirements**:
- Store `std::vector<Move> legal_moves` in `InferenceRequest` structure
- Store `int current_player` in `InferenceRequest` structure
- Populate in `ContinuousSimulationRunner` before queue submit (before state ownership transfer)
- Use stored moves in `expand_node_with_result()` (skip `getLegalMoves()` call)

**Expected Impact**:
- Expansion time reduction: 10-20% (micro-benchmark validation)
- Total throughput gain: +5-10% (measured via T016 after implementation)

**Validation**:
- Parity test: Expansion results identical with/without precomputation
- Performance test: `expand_node_with_result()` time reduced ≥10%
- Instrumentation: `getLegalMoves()` NOT called during expansion (log verification)

**Rollback**: Feature flag `PRECOMPUTE_LEGAL_MOVES` (default: false)

**Dependencies**: Requires T009 (state pooling) complete for clean ownership semantics

**Status**: DEFERRED to Phase 2 (not blocking 8k target per TRACEABILITY_MATRIX.md GAP 2)

---

### 6.2 DLPack Fast Path Validation (Phase 2 Diagnostic)

**FR6.2: DLPack Zero-Copy Verification**:

**Problem** (review.txt lines 260-280): Uncertain if `DLPackInferenceBridge` is active or falling back to NumPy conversion in batch callback.

**Requirements**:
- Instrument `PyBatchInferenceCallback` to log conversion path
- Verify `isinstance(bridge, DLPackInferenceBridge) == True` at runtime
- Measure batch callback time (target <0.5ms per batch-64)
- Telemetry field: `conversion_path` ∈ {"dlpack_fast", "numpy_fallback"}

**Acceptance**:
- ✅ DLPack fast path confirmed in 100% of batches (no fallback)
- ✅ Batch callback overhead <1% of total time
- ✅ Telemetry field `conversion_path == "dlpack_fast"`

**Expected Impact**:
- If fallback detected: Fix provides 5-15% throughput gain
- If fast path confirmed: No action needed (diagnostic only)

**Rollback**: N/A (diagnostic only, no functional changes)

**Dependencies**: OpenMP fix (T004-T006) addresses bulk of Python/GIL overhead; this is validation only

**Status**: DEFERRED to Phase 2 (low priority per TRACEABILITY_MATRIX.md GAP 1)

---

### 6.3 Lightweight Neural Network (Phase 7 Architecture)

**FR6.3: Neural Network Architecture Optimization**:

**Problem** (review.txt lines 621-1396): GPU inference is only 32.8% of total time at baseline. Doubling NN speed directly raises throughput, enabling 10k+ sims/sec.

**Architecture Options** (ranked by safety):

**Option A: RepVGG/ECA ResNet** (Safest, +25-50% model speed):
- Train with multi-branch residuals (3×3 + 1×1 + identity)
- Fuse to single 3×3 conv at inference (BN folding)
- Replace SE with ECA (Efficient Channel Attention, near-zero overhead)
- Expected: 1.25-1.5× model speedup, strength ≈ baseline or improved

**Option B: Ghost Bottleneck + ShuffleV2** (Maximum speed, +40-80%):
- Entry/exit: RepECA blocks (clean features)
- Middle: Ghost bottlenecks (FLOP reduction) or ShuffleV2 (bandwidth reduction)
- Add auxiliary tactical heads (threat detection) for strength preservation
- Expected: 1.4-1.8× model speedup, small strength dent mitigated by aux tasks

**Option C: Two-Tier Evaluator Cascade** (×1.5-2.5 throughput):
- Micro-net (C=24-32, B=2-3) runs first
- Gate: If entropy < τ or threat detected → accept; else escalate to main net
- 30-60% of positions skip main net (trivial moves)
- Expected: ×1.5-2.5 end-to-end throughput, superhuman preserved by conservative gate

**Option D: Early-Exit Heads** (×1.2-1.6 throughput, stackable):
- Auxiliary policy/value heads at block 3, block 6
- Exit if |value| > threshold or entropy < threshold
- Position-dependent speedup
- Expected: ×1.2-1.6 average throughput

**Combined Impact** (Option B + C):
- Model speedup: 1.5-1.8×
- Cascade multiplier: ×1.5-2.5
- **Total throughput: 18-22k sims/sec** (vs. 8k MCTS-only target)

**Target Performance** (3060 Ti, FP16, optimal batching):
| MCTS Throughput | NN Architecture | Total Throughput |
|-----------------|-----------------|------------------|
| 8k sims/sec | Baseline (SE-ResNet) | 8k sims/sec |
| 8k sims/sec | RepVGG/ECA | 10-12k sims/sec |
| 8k sims/sec | Ghost+Shuffle | 12-14k sims/sec |
| 8k sims/sec | Ghost+Shuffle+Cascade | **18-22k sims/sec** |

**Validation**:
- ELO ≥ baseline (1000-game match, 95% confidence)
- Policy agreement ≥95% (1000-position test set)
- If ELO within -10 but sims/sec ≥1.7×: ACCEPTABLE (training data throughput prioritized)

**Game-Specific Configs**:
- Gomoku Freestyle: Ghost mid-trunk, early-exit [4,8], entropy≤0.75
- Renju/Omok: Ghost+Shuffle hybrid, early-exit [6], entropy≤0.65, aux threat heads
- Chess 8×8: Ghost, early-exit [6], entropy≤0.90 (conservative)
- Go 9×9: Ghost, early-exit [4,8], entropy≤0.85

**Reference**: See `NEURAL_NETWORK_OPTIMIZATION.md` for complete architecture specs, training protocol, and ablation study design

**Dependencies**: Requires 8k sims/sec MCTS target achieved (validates GPU is bottleneck before NN optimization)

**Status**: FUTURE (Phase 7, post-8k target)

---

**END OF SPECIFICATION v2.0**

**Next Steps**:
1. Execute `/speckit.plan` to generate TECHNICAL_PLAN.md
2. Execute `/speckit.tasks` to generate TASKS.md breakdown
3. Implement critical fixes (OpenMP, state cloning)
4. Validate via T016 benchmarking suite
