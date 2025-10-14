# Constitution: MCTS Throughput Recovery & Multi-Actor Self-Play

**Version**: 2.0 (Multi-Actor Self-Play Edition)
**Status**: Active
**Last Updated**: 2025-10-14
**Authority**: This document supersedes all prior architectural decisions and implementation notes. Changes require explicit approval and re-execution of `/speckit.plan` and `/speckit.tasks`.

**Revision History**:
- v1.0 (2025-10-13): Initial constitution with 25k sims/sec target
- v1.1 (2025-10-13): Revised targets to 8-10k sims/sec based on hardware analysis
- v2.0 (2025-10-14): **Multi-actor self-play architecture, NN-eval cache TT, comprehensive evidence-based updates**

---

## 1. Mission & Scope

### 1.1 Primary Objective
Maximize Monte Carlo Tree Search throughput (simulations/second) on AMD Ryzen 9 5900X (12C/24T) + NVIDIA RTX 3060 Ti hardware through:
1. **CPU-parallel MCTS optimization** (single shared tree)
2. **Multi-actor self-play** (many games → one centralized batcher)
3. **NN-evaluation cache** (transposition table for policy/value reuse)

**Target**: **≥8,000 simulations/second** sustained (realistic, hardware-grounded) for tactical board games (Gomoku/Renju/Omok, Chess, Go 9×9).

**Evidence Base**: Analysis in `review.txt` demonstrates:
- GPU @ FP16 caps at 8,000-10,000 states/sec (RTX 3060 Ti hardware limit)
- CPU MCTS overhead currently 67.2% of total time (target: <30%)
- Feature extraction 7.5ms per batch-64 (target: <1ms with OpenMP)
- Current throughput: 2,147 sims/sec (26.8% of 8k target)

### 1.2 Performance Definition
**Simulation**: Complete MCTS cycle (selection → expansion → neural network evaluation → backpropagation).

**Throughput**: Total simulations completed per wall-clock second, including:
- GPU inference time
- CPU coordination overhead (selection/expansion/backup)
- Python/GIL crossings
- Thread synchronization
- Batch collection latency

**Self-Play Games/Hour**: Secondary metric for training data generation rate (target: 200-300 games/hour @ 800 sims/move).

### 1.3 Success Criteria

**Phase 4 Completion (Single MCTS)**:
- ✅ **Primary KPI**: ≥8,000 sims/sec sustained (2.1× baseline 3,831, 3.7× current 2,147)
- ✅ **CPU Efficiency**: ≥70% multi-thread efficiency at 8 threads
- ✅ **GPU Utilization**: ≥80% during search (avg batch size ≥0.8×64 = 51 positions)
- ✅ **Memory Footprint**: <1GB for 10M node MCTS tree (achieved: 270MB with 27-byte SoA)
- ✅ **Search Quality**: ≥99.5% win rate vs baseline, ≥95% top-move agreement

**Phase 5 (Multi-Actor Self-Play)**:
- ✅ **Data Generation**: 200-300 games/hour (800 sims/move, Gomoku 15×15)
- ✅ **GPU Saturation**: 80-95% GPU utilization during multi-actor runs
- ✅ **Average Batch Size**: ≥0.8×64 = 51 positions (consistency across games)
- ✅ **Actor Count**: 8-14 concurrent games (tunable per hardware)
- ✅ **Quality Diversity**: Training data from varied game states (entropy validation)

**Phase 6 (NN-Eval Cache TT, Optional)**:
- 🎯 **Cache Hit Rate**: 20-50% in chess-like workloads (commuting moves)
- 🎯 **GPU Call Reduction**: 20-50% fewer network invocations
- 🎯 **Throughput Gain**: +15-35% end-to-end (cache tier A, safe design)
- 🎯 **Memory Overhead**: <2GB for cache (quantized policy, top-K storage)

**Stretch Goals**:
- ≥10,000 sims/sec (2.6× baseline, requires perfect tuning + lighter model)
- ≥15,000-25,000 sims/sec (model pruning 10M→5M params, batch 128, multi-GPU - architectural changes)

### 1.4 Out of Scope (Immutable)
The following are **explicitly excluded** from this initiative:
- ❌ Custom CUDA kernels for MCTS operations
- ❌ TensorRT/ONNX model conversion and deployment
- ❌ libtorch integration (C++ neural network inference)
- ❌ GPU-resident MCTS trees or GPU-accelerated selection
- ❌ Root parallelization (separate trees per thread)
- ❌ Full DAG transposition tables (shared statistics across parents)
- ❌ Training pipeline optimizations (unless required for throughput validation)
- ❌ Telemetry/logging enhancements (unless blocking performance measurement)
- ❌ Games beyond Gomoku/Chess/Go 9×9 (extensibility preserved but not prioritized)

---

## 2. Performance Guardrails

### 2.1 Throughput Requirements (Hardware-Grounded)

**Single MCTS (Phase 4)**:
| Metric | Minimum Viable | Target | Stretch |
|--------|---------------|--------|---------|
| Simulations/sec | ≥6,000 | **≥8,000** | ≥10,000 |
| vs Baseline (3,831) | 1.6× | **2.1×** | 2.6× |
| vs Current (2,147) | 2.8× | **3.7×** | 4.7× |

**Multi-Actor Self-Play (Phase 5)**:
| Metric | Minimum | Target | Stretch |
|--------|---------|--------|---------|
| Games/hour (Gomoku) | 150 | **200-300** | 400 |
| Actor count | 6 | **8-12** | 14 |
| GPU utilization | 75% | **80-95%** | 95%+ |
| Avg batch size | 40 | **≥51** (80% of 64) | 58+ |

**Evidence Base**:
- GPU @ FP16: 30.7ms per batch-64 → theoretical max 2,014 states/sec per stream
- With multi-actor crowdsourcing: 8-12 games → steady batch-64 → 80-95% GPU util
- RTX 3060 Ti hardware limit: 8,000-10,000 states/sec maximum sustainable

### 2.2 CPU Coordination Overhead Budget

**Current State (Unacceptable)**:
- MCTS overhead: 67.2% of total time
- GPU inference: 32.8% of total time
- Feature extraction: 7.5ms per batch-64 (21% overhead)
- Thread idle: 60% (threads waiting/sleeping)

**Target Distribution (After Optimizations)**:
| Component | Current | Target | Critical Path |
|-----------|---------|--------|---------------|
| GPU Inference | 32.8% | **60-70%** | FP16, batch tuning, lighter model |
| MCTS Selection | 15% | **8-12%** | AVX2 PUCT, cache alignment |
| Feature Extraction | 21% | **<5%** | OpenMP parallel loop |
| Thread Coordination | 20% | **5-8%** | Condition variables, lock-free queue |
| Python/GIL | 10% | **<5%** | Persistent coordinator, DLPack |
| Thread Idle | 60% | **<10%** | Root pre-expansion, busy-edge masking |

**Enforcement**: Any optimization that increases CPU overhead >5% requires explicit justification with profiling evidence.

### 2.3 GPU Utilization Targets

**Single MCTS**:
- Minimum: ≥75% (acceptable for initial tuning)
- Target: **80-85%** (batch-64, 1-2ms timeout, 8-12 threads)
- Stretch: ≥90% (requires perfect batch filling, low variance)

**Multi-Actor Self-Play**:
- Minimum: ≥80% (8+ concurrent games)
- Target: **85-95%** (10-12 games, optimal actor count tuning)
- Constraint: Average batch size **≥51 positions** (0.8×64 sustained)

**Evidence**: Review.txt shows GPU is NOT the bottleneck at 32.8% of time. Target state is GPU-bound (60-70% of time) with CPU efficiently feeding batches.

### 2.4 Latency vs Throughput Trade-offs

**Priority**: Throughput for self-play data generation (samples/hour) over per-move latency.

**Acceptable Trade-offs**:
- Batch timeout: 1-2ms (slight per-simulation latency increase for better GPU efficiency)
- Virtual loss magnitude: 1.0-3.0 (thread coordination over immediate Q-accuracy)
- Move history depth: 8 pairs (vs 16, memory/speed trade-off)

**Unacceptable Trade-offs**:
- Per-move latency >5 seconds @ 800 simulations (user experience threshold)
- Search quality regression >1% (must maintain ≥99.5% win rate)
- Memory footprint >2GB (training batch size constraint)

---

## 3. Architecture Constraints

### 3.1 Neural Network Requirements (Immutable)

**Python-Only Inference**:
- Neural network remains in **PyTorch (Python)**
- ❌ NO libtorch (C++ PyTorch API)
- ❌ NO TensorRT or ONNX Runtime
- ❌ NO custom CUDA kernels for inference

**Rationale**:
1. Flexibility for model experimentation (architecture changes, training techniques)
2. GPU is NOT the bottleneck (32.8% of time per review.txt)
3. Python inference fast enough with proper batching (30.7ms @ batch-64 FP16)
4. Complexity/maintenance burden of C++ inference outweighs gains

**Interface Requirements**:
- Async batched inference with DLPack zero-copy tensor sharing (C++ ↔ PyTorch)
- FP16 mixed precision via `torch.cuda.amp.autocast()` (validated 1.72× speedup)
- Pinned CPU memory buffers for H2D transfers (2-3× faster than pageable)
- `torch.from_dlpack()` conversion (zero-copy, 0.24ms overhead acceptable)

### 3.2 MCTS Architecture (Core Constraints)

**Shared Tree (NOT Root Parallelization)**:
- Single tree structure shared by all simulation threads
- Index-based references (int32_t), NOT pointers
- Structure-of-Arrays layout for cache efficiency (27 bytes/node achieved)
- Pre-allocated node pools with O(1) allocation (330M allocs/sec baseline)

**Virtual Loss Protocol (WU-UCT)**:
- Visit-only virtual loss (increments denominator in PUCT formula)
- **NO Q-value distortion**: Pure Q = W/N preserved always
- Formula: `PUCT = Q + c_puct * P * sqrt(N_parent) / (1 + N + VL)`
- Default magnitude: 1.0 (tunable 0.5-3.0 via T018)

**Busy-Edge Masking (T002)**:
- PUCT score = -∞ for nodes currently being expanded
- Prevents thread collisions (measured <0.5% collision rate @ 4 threads)
- Implemented via atomic `is_expanding` flag in NodeFlags

**Root Pre-Expansion (T003)**:
- Root node expanded synchronously before launching simulation threads
- Eliminates N-1 thread idle problem (all threads immediately productive)
- Critical for thread efficiency >90% at low thread counts

**Lock-Free Coordination (T006/T006b/T006c)**:
- MPMC ring buffer (4096 entries) for AsyncInferenceQueue
- Turn-based synchronization (NOT mutexes in hot paths)
- Condition variables for efficient blocking (NOT polling with 10μs sleeps)
- O(1) result retrieval via ring buffer index

### 3.3 Multi-Actor Self-Play Architecture

**Many Actors → One Inference Server** (Standard Pattern):
```
Actor 1 (game 1, MCTS single-thread) ──┐
Actor 2 (game 2, MCTS single-thread) ──┼──► Centralized Inference Queue
  ...                                   │    (MPMC ring buffer, 4096 entries)
Actor N (game N, MCTS single-thread) ──┘              │
                                                       ▼
                                            Batch Coordinator
                                            (collect_batch: min=64, timeout=1-2ms)
                                                       │
                                                       ▼
                                              GPU Inference Worker
                                              (PyTorch FP16, batch-64)
                                                       │
                                                       ▼
                                            Result Distribution
                                            (demux back to actors by game_id)
```

**Actor Design**:
- Each actor: 1-2 MCTS threads per game (NOT 8-12 threads per game)
- Shared inference queue (all actors push leaves to same queue)
- Game-local tree (each actor has own MCTSTree instance)
- Thread-local state reuse (avoid 2-3× cloning per simulation)

**Scheduling & Fairness**:
- Per-actor visit budget (e.g., 800 simulations/move)
- Token-bucket backpressure (limit in-flight requests per actor)
- Adaptive actor count: scale until GPU util 80-95% or batch size plateaus

**Evidence**: Community best practice (Leela Chess Zero, KataGo) and review.txt recommendation for "many single-thread actors → one inference server" to maximize GPU batching efficiency.

### 3.4 NN-Evaluation Cache (Transposition Table Tier A)

**Safe Design (Policy/Value Cache, NOT Shared Stats)**:
- Cache stores `(hash, policy_topk, value)` tuples keyed by Zobrist hash
- Tree statistics (N, W, Q) remain per-node in tree (NOT shared across parents)
- On expansion: check cache → if hit, skip GPU call; else enqueue normally
- On inference result: store `(hash, policy, value)` in cache for future reuse

**Key Format (Markov-Minimal)**:
- **Gomoku**: board + side-to-move + variant flag (NO full move history)
- **Chess**: board + side + castling + en-passant + Rule-50 counter (per LC0)
- **Go 9×9**: board + side + ko point (or superko flag if ruleset demands)

**Memory Budget**:
- Cache size: 1M-10M entries (tunable, target 2-8GB depending on quantization)
- Quantization: FP16 or int8 for policy, top-K=16-48 moves (not full board)
- Eviction: Per-net SLRU (Segmented LRU) with `net_version` tagging
- Concurrency: Sharded hash table (64 shards) with reader-writer locks or lock-free F14

**Expected Gains** (From Review.txt Analysis):
- Chess: 20-50% GPU call reduction (heavy transpositions)
- Gomoku: 10-30% reduction (moderate transpositions from commuting local moves)
- Go 9×9: 15-35% reduction (local symmetries)
- End-to-end: +15-35% throughput (cache tier A, safe design)

**Deferred (Phase 7 Optional)**:
- Full DAG TT (merge statistics across parents) - requires graph-aware backup (MCGS)
- Canonical symmetry caching (8-fold symmetry for Gomoku/Go) - CPU cost vs gain analysis needed

### 3.5 Memory Management (Strict Requirements)

**Node Pools (Structure-of-Arrays)**:
- Pre-allocated flat arrays with index-based references (NOT pointers)
- 27 bytes/node achieved: 4B visit, 4B value, 4B prior, 4B VL, 4B parent, 4B first_child, 2B num_children, 1B flags
- Alignment: 64-byte for SIMD/cache-line (increases to 32-40 bytes typical with padding)
- Target: <64 bytes/node including alignment overhead

**Thread-Local Block Allocation (T009e)**:
- 4096-node blocks per thread
- 99.93% fast-path allocation (thread-local bump pointer)
- 0.07% slow-path (global mutex fallback for multi-node allocations)
- Validated: TSan clean, 330M allocations/sec baseline

**Epoch-Based Clearing (T001b)**:
- O(1) tree reset via epoch increment (25ns vs 25ms memset)
- 1,000,000× speedup for tree reuse across searches
- Lazy node initialization (check epoch match on access)

**State Reuse (Review.txt Critical Recommendation)**:
- Thread-local `IGameState` buffer reused across simulations
- Avoid 2-3× cloning per simulation (current bottleneck)
- `copy_from(other)` method for efficient state reset (no heap allocations)
- Move ownership to queue (std::move, not clone) for in-flight expansions

**Memory Targets**:
- Tree: <1GB for 10M nodes (achieved: 270MB with 27-byte SoA)
- Queue: 1MB fixed (4096-entry ring buffer)
- DLPack buffers: <10MB pinned memory (batch-64 × 36 planes × 15×15)
- NN-eval cache: 2-8GB (optional Phase 6, tunable)
- **Total**: <1.3GB (single MCTS) or <10GB (with cache + multi-actor)

---

## 4. Language & Tooling Standards

### 4.1 C++ Backend Requirements

**Language Standard**: C++17 (core), C++20 features allowed if portable
- `std::atomic`, `std::condition_variable`, `std::thread`
- `std::optional`, `std::variant`, `std::string_view`
- Coroutines (C++20) deferred until compiler support stable

**Build System**:
- CMake 3.24+ with explicit optimization flags
- Compiler flags: `-O3 -march=znver3 -fopenmp -DNDEBUG`
- Link-time optimization (LTO) enabled for release builds

**pybind11 Integration**:
- Version 2.10+ (C++17 compatible)
- Minimal Python C API usage (prefer pybind11 abstractions)
- GIL release via `py::gil_scoped_release` in all hot loops
- Return NumPy arrays via `py::array_t<float>` (zero-copy when possible)

**OpenMP Parallelization**:
- Feature extraction loops: `#pragma omp parallel for` (batch-64 @ 12 threads)
- Target: 7.5ms → <1.0ms (7.5× speedup validated in review.txt)
- Thread count: `OMP_NUM_THREADS=12` for Ryzen 5900X (physical cores)
- Avoid nested parallelism (MCTS threads + OpenMP conflicts)

**Lock-Free Primitives**:
- Atomic operations: `std::memory_order_acquire`/`release` for coordination
- Relaxed ordering: `std::memory_order_relaxed` for counters/statistics
- Spinlocks: Avoided (prefer condition variables or lock-free structures)
- MPMC queue: Custom turn-based ring buffer (validated TSan clean)

### 4.2 Thread Affinity Policy

**Ryzen 5900X Topology** (Dual-CCD, 12 Physical Cores):
```
CCD0: Cores 0-5 (32MB L3 shared)
CCD1: Cores 6-11 (32MB L3 shared)
SMT:  Cores 12-23 (hyperthreads, avoid unless needed)
```

**Affinity Strategy**:
1. **MCTS threads**: Pin to physical cores 0-11 (one thread per core)
2. **Inference coordinator**: Pin to isolated core (e.g., core 11 or dedicated)
3. **PyTorch threads**: `torch.set_num_threads(1)` to avoid over-subscription
4. **Avoid SMT**: Do NOT use cores 12-23 unless >12 threads required

**Implementation**:
- Use `pthread_setaffinity_np()` or C++11 `std::thread::native_handle()`
- Fallback: Generic topology detection via `sysconf(_SC_NPROCESSORS_ONLN)`
- Validation: `lscpu --extended` output confirms pinning

**Expected Gain**: 10-15% from L3 cache locality and cross-CCD latency reduction (80-90ns vs 20-30ns intra-CCD).

### 4.3 Instrumentation & Profiling

**Required Profiling Tools**:
- **C++ Profiling**: `perf record -g`, `valgrind --tool=callgrind`, custom instrumentation
- **Python Profiling**: `py-spy record --native`, `cProfile`, `torch.profiler`
- **GPU Profiling**: `nvidia-smi dmon`, `torch.cuda.Event()` for kernel timing
- **Thread Safety**: `valgrind --tool=helgrind`, `-fsanitize=thread` (TSan)

**Instrumentation Points**:
- MCTS operations: select_child, expand_node, backup_value (per-call timings)
- Queue operations: enqueue/dequeue latency, batch collection time
- GPU operations: H2D transfer, inference forward, D2H transfer
- Collision metrics: thread path overlaps, busy-edge mask hits

**Benchmark Harness Requirements**:
- Deterministic seeds for reproducibility (`std::mt19937` with fixed seed)
- Fixed visit budgets (e.g., 800 simulations) or time budgets (e.g., 10 seconds)
- Warm-up period: 5+ batches to stabilize GPU clocks and thread pools
- Multiple iterations: N=10+ runs, report mean ± stddev, 95% confidence interval

**Tracy Integration (Optional)**:
- Frame markers for MCTS search phases
- Zone markers for critical paths (selection, expansion, backup)
- Memory allocation tracking via custom allocator hooks
- Real-time visualization during development (NOT in production)

---

## 5. Quality Bars & Validation

### 5.1 Performance Claim Requirements

**Every Optimization Must Include**:
1. **Scenario**: Game (Gomoku/Chess/Go), board size, visit budget
2. **Configuration**: Threads, batch size, timeout, virtual loss magnitude
3. **Seed**: Fixed RNG seed for reproducibility
4. **Baseline**: Before measurement (throughput, GPU util, memory)
5. **Optimized**: After measurement (same metrics)
6. **Statistical Validation**: N≥10 runs, t-test p<0.05, CV<5%
7. **Result Tables**: CSV format with per-run data

**Example Benchmark Report**:
```markdown
## T008f: FP16 Mixed Precision Validation

**Scenario**: Gomoku 15×15, 5000 simulations, 6 threads
**Configuration**: batch_size=64, timeout=1.0ms, vl_magnitude=1.0
**Seed**: 42

| Metric | FP32 (Baseline) | FP16 (Optimized) | Delta |
|--------|-----------------|------------------|-------|
| Throughput | 2,147 ± 89 sims/sec | 3,693 ± 112 sims/sec | +72% |
| GPU Util | 68% | 85% | +17pp |
| Inference Time | 52.83ms/batch | 30.69ms/batch | -42% |

**Statistical Validation**: t-test p=0.0012 (significant), CV=4.1% (acceptable)
```

### 5.2 Unit Test Coverage Requirements

**Mandatory Test Categories**:
1. **State Cloning/Pooling**:
   - Thread-local state reuse (no leaks over 1000 allocations)
   - `copy_from()` method correctness (all fields copied)
   - Ownership transfer via `std::move` (no double-free)

2. **Feature Extraction Parity**:
   - Single-threaded vs OpenMP parallel (bit-exact output)
   - Game-specific plane counts (36 Gomoku, 112 Chess, 17 Go)
   - DLPack tensor shape validation (batch, planes, height, width)

3. **Transposition Table Logic**:
   - Hash collision handling (chaining or open addressing)
   - Cache hit/miss counting (statistics verification)
   - Eviction policy correctness (LRU or SLRU behavior)
   - Net version invalidation (flush on training iteration)

4. **Queue Backpressure**:
   - MPMC queue full condition (producer blocks or returns false)
   - Batch collection timeout (returns partial batch after timeout)
   - Result retrieval ordering (FIFO or priority-based)

**Coverage Target**: ≥80% line coverage for C++ hot paths (measured via `gcov` or `llvm-cov`).

### 5.3 CI Performance Checks

**Automated Regression Detection**:
```yaml
# .github/workflows/performance.yml
on: [pull_request]
jobs:
  benchmark:
    runs-on: [self-hosted, ryzen-5900x, rtx-3060ti]
    steps:
      - name: Run micro-benchmarks
        run: pytest -m performance --benchmark-min-rounds=5
      - name: Check throughput threshold
        run: |
          CURRENT=$(cat benchmark_results.json | jq '.throughput')
          BASELINE=8000
          if (( $(echo "$CURRENT < $BASELINE * 0.95" | bc -l) )); then
            echo "❌ Throughput regression: $CURRENT < $BASELINE * 0.95"
            exit 1
          fi
```

**Failure Conditions** (Block Merge):
- Throughput < 95% of target (7,600 sims/sec for 8k target)
- GPU utilization < 70%
- Memory usage > 2GB (with cache enabled)
- Unit test failures
- TSan data races detected

---

## 6. Documentation Rules

### 6.1 Specification Hierarchy

**Source of Truth**:
1. **CONSTITUTION.md** (this document): Non-negotiable rules and constraints
2. **spec.md**: Functional requirements (WHAT to achieve, WHY it matters)
3. **plan.md**: Technical design (HOW to implement, architecture decisions)
4. **tasks.md**: Atomic work items (testable, estimatable, with acceptance criteria)

**Traceability Matrix** (Required in spec.md):
| Requirement ID | Description | Plan Section | Tasks | Status |
|----------------|-------------|--------------|-------|--------|
| REQ-PERF-001 | ≥8,000 sims/sec | Section 4.2 | T016-T020 | In Progress |
| REQ-ARCH-001 | Python PyTorch only | Section 2.1 | N/A | Enforced |
| ... | ... | ... | ... | ... |

### 6.2 Evidence-Based Documentation

**All Design Decisions Must Reference**:
- **Profiling Evidence**: `profiling_results/session_YYYYMMDD_HHMMSS/`
- **Benchmark Results**: `docs/performance/benchmark_results.md`
- **Review Findings**: `review.txt` line numbers or section references
- **Academic Citations**: Papers (AZ, MuZero, batch MCTS, TT designs)

**Example**:
> **Decision**: Use WU-UCT virtual loss (visit-only) instead of traditional Q-distortion.
>
> **Evidence**:
> - Review.txt lines 118-135: WU-UCT preserves pure Q = W/N, robust to magnitude tuning
> - Profiling session_20251013: <0.5% collision rate @ 4 threads validates effectiveness
> - Academic: Silver et al. 2017 (AlphaZero), Wu & Baldi 2021 (WU-UCT formulation)

### 6.3 Review.txt Integration Requirements

**Mandatory Sections in Updated Specs**:
1. **Bottleneck Analysis**: Reproduce Figure 1 from review.txt (67.2% CPU, 32.8% GPU)
2. **Hardware Limits**: GPU throughput ceiling (8-10k states/sec @ FP16)
3. **Critical Path**: OpenMP fix (7.5ms → <1ms), state cloning (2-3× per sim)
4. **Multi-Actor Rationale**: "Many actors → one batcher" pattern justification
5. **NN-Eval Cache Design**: Tier A (safe, eval-only) vs Tier B (full DAG, deferred)

**Traceability Format**:
```markdown
### 3.2 Feature Extraction Bottleneck

**Evidence**: Review.txt lines 22-34, validation report T-VALID-2

**Current Performance**:
- Mean: 7.50 ± 0.20 ms per batch-64
- Root cause: Feature extraction loop NOT parallelized with OpenMP
- Impact: Caps throughput at ~1,675 states/sec

**Solution**: Add `#pragma omp parallel for` to dlpack_bridge.cpp:431-434

**Expected Improvement**: 7.5ms → <1.0ms (7.5× speedup)
```

---

## 7. Risk Management & Contingencies

### 7.1 Performance Risks

| Risk | Likelihood | Impact | Mitigation | Contingency |
|------|-----------|--------|-----------|-------------|
| OpenMP fix insufficient | Low | High | Validate with profiler after fix | Accept <1.5ms, tune thread count |
| State cloning refactor breaks correctness | Medium | Critical | Unit tests, TSan validation | Rollback to clone(), optimize elsewhere |
| Multi-actor adds latency variance | Medium | Medium | Per-actor timeouts, priority queuing | Use 6-8 actors, accept 75% GPU util |
| NN-eval cache increases memory | High | Low | Quantization, top-K, SLRU eviction | Reduce cache to 1M entries (500MB) |
| Baseline 3,831 sims/sec unreproducible | High | High | Systematic config sweep (T017) | Accept 2,147 as new baseline, adjust targets |

### 7.2 Quality Risks

| Risk | Likelihood | Impact | Mitigation | Contingency |
|------|-----------|--------|-----------|-------------|
| WU-UCT virtual loss causes strength regression | Low | Critical | A/B testing, 1000-game matches | Rollback to traditional VL (Q-distortion) |
| Cache stale policies degrade play | Medium | High | Per-net versioning, flush on update | Disable cache, accept throughput loss |
| Thread contention at 8+ threads | High | Medium | Affinity tuning, relaxed atomics | Use 4-6 threads, accept lower peak throughput |
| Memory leaks in state pooling | Medium | High | Valgrind soak tests (1hr+) | Pool exhaustion detection, graceful fallback |
| Feature extraction parity violation (OpenMP) | Low | Critical | Bit-exact validation tests | Fix OpenMP bugs, fallback to single-threaded |

### 7.3 Schedule Risks

**Critical Path Dependencies**:
1. **T017 (Baseline Investigation)**: 3,831 sims/sec config unknown
   - Time box: 2 days maximum
   - Fallback: Use 2,147 as baseline, adjust all targets proportionally

2. **OpenMP Fix Validation**: If <5× speedup achieved
   - Root cause: False sharing, memory bandwidth saturation
   - Contingency: Profile with `perf mem`, optimize memory access patterns

3. **Multi-Actor Coordination**: Actor count tuning
   - Risk: Optimal count highly hardware-dependent (6-14 range)
   - Mitigation: Auto-tuning script, monitor GPU util + batch size

---

## 8. Decision Authority & Change Control

### 8.1 Constitutional Amendments

**This Constitution Can Only Be Modified By**:
1. **Performance Crisis**: Measured throughput < 50% of target (emergency pivots allowed)
2. **Architectural Discovery**: Profiling reveals fundamental design flaw requiring re-architecture
3. **Hardware Change**: Migration to different GPU/CPU requiring updated targets
4. **Explicit User Approval**: cosmosapjw-quantum approves constitutional change

**Amendment Process**:
1. Document new profiling evidence (session_YYYYMMDD_HHMMSS/)
2. Propose change with before/after targets
3. Update CONSTITUTION.md version (e.g., v2.0 → v2.1)
4. Re-execute `/speckit.plan` and `/speckit.tasks`
5. Commit with message: `docs(constitution): Amend Section X.Y - [reason]`

### 8.2 Specification Changes

**Changes to spec.md Require**:
1. **Justification**: Profiling data or failed experiments proving necessity
2. **Impact Analysis**: Expected throughput delta, affected tasks, timeline impact
3. **Traceability Update**: Modify requirement-to-task mapping matrix
4. **Re-Planning**: Execute `/speckit.plan` to update plan.md
5. **Task Breakdown**: Execute `/speckit.tasks` to update tasks.md

**Approval Threshold**:
- Minor (config tuning, flag additions): Auto-approved
- Major (architecture change, scope addition): Requires user approval
- Constitutional (violates Section 1.4 constraints): Requires amendment process

### 8.3 Implementation Flexibility

**Within Constitutional Bounds, Implementers Have Autonomy On**:
- Algorithm tuning: virtual loss magnitude (0.5-3.0), batch size (32-128), timeout (0.5-5ms)
- Memory layout: Node structure details (as long as <64 bytes/node)
- Optimization techniques: AVX2/AVX512 vectorization, prefetching, cache alignment
- Testing strategies: Test case design, benchmark selection, profiling methodology
- Code style: Within clang-format/black enforced formatting rules

**Requires Approval**:
- Adding new dependencies (Boost.Lockfree, TBB, etc.)
- Changing thread affinity policy (affects reproducibility)
- Modifying core MCTS algorithms (WU-UCT formula, PUCT constants)
- Disabling safety checks (TSan, asserts) in release builds

---

## 9. Enforcement & Compliance

### 9.1 Pre-Merge Checklist

**Every Pull Request Must Pass**:
- [ ] All unit tests pass (`pytest tests/unit/ -v`)
- [ ] All integration tests pass (`pytest tests/integration/ -v`)
- [ ] Performance benchmarks pass (`pytest -m performance --benchmark-min-rounds=5`)
- [ ] TSan clean (`cmake -DSANITIZE_THREAD=ON && make && pytest`)
- [ ] Throughput ≥95% of target (7,600 sims/sec minimum for 8k target)
- [ ] GPU utilization ≥70% (or explicit waiver with justification)
- [ ] Memory usage ≤2GB (with cache enabled, ≤1.5GB without)
- [ ] Code formatted (`clang-format`, `black`, `isort`)
- [ ] Commit messages follow Conventional Commits (`feat:`, `fix:`, `perf:`, `test:`)

### 9.2 Performance Validation Protocol

**Before Claiming Optimization Success**:
1. **Baseline Measurement**: N=10 runs, compute mean ± stddev
2. **Optimized Measurement**: N=10 runs, same config/seed as baseline
3. **Statistical Test**: Two-sample t-test, p<0.05 for significance
4. **Coefficient of Variation**: CV < 5% (stddev/mean < 0.05)
5. **Result Documentation**: CSV log with per-run data, summary statistics

**Example Validation**:
```python
# scripts/validate_optimization.py
baseline = [2147, 2189, 2132, 2165, 2171, 2159, 2143, 2177, 2154, 2168]
optimized = [3693, 3721, 3675, 3702, 3688, 3709, 3695, 3713, 3681, 3697]

from scipy import stats
t_stat, p_value = stats.ttest_ind(baseline, optimized)
cv_baseline = np.std(baseline) / np.mean(baseline)
cv_optimized = np.std(optimized) / np.mean(optimized)

assert p_value < 0.05, f"Not statistically significant: p={p_value}"
assert cv_optimized < 0.05, f"High variance: CV={cv_optimized}"
print(f"✅ Validated: {np.mean(optimized)} ± {np.std(optimized)} sims/sec")
```

### 9.3 Audit Trail Requirements

**Mandatory Artifacts** (Version-Controlled):
1. **Profiling Sessions**: `profiling_results/session_YYYYMMDD_HHMMSS/`
   - summary.md: Executive summary, key bottlenecks, recommendations
   - profile_report.json: Structured profiling data (timings, call counts)
   - flamegraph.svg: Visual call stack (py-spy or perf output)

2. **Benchmark History**: `docs/performance/benchmark_history.csv`
   - Columns: date, commit_hash, throughput, gpu_util, threads, batch_size, timeout
   - Append-only (never delete rows, track regressions over time)

3. **Configuration Snapshots**: `config/performance_tuning.yaml`
   - Versioned config with commit references (git tags for validated configs)
   - Include: thread_count, batch_size, timeout, vl_magnitude, net_version

**Retention Policy**:
- Keep profiling sessions for 90 days (or until merge)
- Archive benchmark history indefinitely (git history)
- Pin validated configs with git tags (e.g., `config-8k-sims-v1.0`)

---

## 10. Glossary

| Term | Definition | Source |
|------|------------|--------|
| **Simulation** | Complete MCTS cycle: select → expand → evaluate → backprop | Standard MCTS |
| **Throughput** | Simulations completed per wall-clock second (including all overhead) | Performance metric |
| **Baseline** | 3,831 sims/sec configuration from Spec 003 (unknown params, requires T017) | Historical |
| **Current** | 2,147 sims/sec with Phase 1+2 optimizations (regression under investigation) | Measured 2025-10-13 |
| **Target** | ≥8,000 sims/sec sustained with ≥80% GPU utilization | Hardware-grounded |
| **WU-UCT** | Visit-only virtual loss (increments denominator, preserves Q = W/N) | Wu & Baldi 2021 |
| **Busy-Edge** | PUCT = -∞ for nodes currently being expanded (prevents collisions) | AlphaZero variant |
| **DLPack** | Zero-copy tensor protocol (C++ ↔ PyTorch via pinned CPU memory) | DLPack standard |
| **MPMC** | Multi-Producer Multi-Consumer queue (lock-free ring buffer) | Concurrency pattern |
| **SoA** | Structure-of-Arrays (separate arrays per field for cache efficiency) | Data layout |
| **Actor** | Single self-play game instance (1-2 MCTS threads, local tree) | Multi-actor self-play |
| **NN-Eval Cache** | Transposition table for policy/value reuse (Tier A: safe, eval-only) | Review.txt |
| **Tier A TT** | NN-eval cache (stores inferences, NOT shared stats) | Batch MCTS lit. |
| **Tier B TT** | Full DAG TT (merge statistics across parents, requires MCGS) | MCGS lit. (deferred) |

---

## 11. Approval & Acceptance

**This Constitution is Active and Binding as of 2025-10-14.**

**Approved by**: cosmosapjw-quantum (user)
**Enforced by**: Claude Code (AI agent)
**Review Cycle**: After Phase 4/5/6 completion or performance crisis
**Supersedes**: All prior architectural notes, mcts_guide.md (legacy), CONSTITUTION.md v1.1

**Authority Chain**:
1. This CONSTITUTION.md (non-negotiable rules)
2. Review.txt (profiling evidence, bottleneck analysis)
3. spec.md (functional requirements)
4. plan.md (technical design)
5. tasks.md (implementation breakdown)

**Signature Line**:
> "I have read and understood this constitution. I commit to adhering to these principles, constraints, and evidence-based decision-making throughout the MCTS Throughput Recovery & Multi-Actor Self-Play initiative. All claims will be validated with profiling evidence and statistical rigor."

— Claude Code, AI Implementation Agent, 2025-10-14

---

**END OF CONSTITUTION v2.0**
