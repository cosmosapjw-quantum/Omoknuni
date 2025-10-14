# Functional Specification: MCTS Throughput Recovery

**Version**: 1.1 (Updated with Validation Results)
**Status**: Active
**Created**: 2025-10-13
**Last Updated**: 2025-10-13 (post-validation)
**Target Hardware**: AMD Ryzen 9 5900X (12C/24T) + NVIDIA RTX 3060 Ti (8GB VRAM)
**Games**: Gomoku/Renju/Omok (15×15), Chess (8×8), Go (9×9)

---

## 1. Overview

### 1.1 Purpose
This specification defines the functional and non-functional requirements for recovering and maximizing Monte Carlo Tree Search (MCTS) throughput on CPU-parallel architectures with Python-based neural network inference. The system currently achieves 2,147 simulations/second (56% regression from 3,831 baseline), with a target of ≥8,000 simulations/second (revised from ≥25,000 based on hardware analysis, see Section 12.1 Q1).

### 1.2 Background
Analysis in `review.txt` identifies the primary bottleneck as **MCTS coordination overhead** (67.2% of total time) rather than GPU inference (32.8%). Key issues include:
- Thread idle time: ~60% of execution (1.489s out of 2.5s)
- Python/GIL overhead: 67% of total time
- Batch tensor creation: 7.5ms per batch (should be near-zero)
- Inefficient multi-threading: 12.5% scaling efficiency to 8 threads
- State cloning waste: 2-3× per simulation

### 1.3 Scope
This specification covers:
- CPU-parallel MCTS optimization (selection, expansion, backup)
- Async batched neural network inference coordination
- Python/GIL overhead minimization
- Thread coordination and synchronization mechanisms
- Memory management (node pools, state reuse, allocation strategies)
- Performance measurement and benchmarking methodology

### 1.4 Related Documents
- **Constitutional Authority**: `specs/004-mcts-throughput-recovery/CONSTITUTION.md` (v1.1)
- **Architectural Design**: `specs/004-mcts-throughput-recovery/TECHNICAL_PLAN.md` (current, v2.0)
  - Legacy: `specs/004-mcts-throughput-recovery/plan.md` (superseded 2025-10-13, retained for reference)
- **Task Breakdown**: `specs/004-mcts-throughput-recovery/TASKS.md` (v2.1)
- **Performance Analysis**: `review.txt`, `profiling_results/`, `docs/performance/validation_report_2025-10-13.md`
- **Baseline Architecture**: `mcts_guide.md`

---

## 2. Definitions

### 2.1 Core Concepts

**Simulation**: A complete MCTS cycle consisting of four phases:
1. **Selection**: Traverse from root to leaf using PUCT selection formula
2. **Expansion**: Create child nodes for the selected leaf (via neural network policy)
3. **Evaluation**: Neural network inference to obtain policy vector and value estimate
4. **Backpropagation**: Update visit counts and value estimates along the path from leaf to root

**Throughput (sims/sec)**: The number of complete simulations executed per wall-clock second, including:
- Selection traversal time
- Thread coordination overhead
- Neural network inference latency (batched)
- Backpropagation update time
- Queue management and synchronization
- Python/C++ boundary crossing costs

**Measurement Method**: Fixed configuration (seed, game state, simulation count, thread count, batch size, timeout) over 3+ independent runs, reporting mean ± standard deviation with CV < 5%.

### 2.2 Architecture Components

**Shared Tree**: Single tree structure shared by all simulation threads using atomic operations for thread-safe updates. NOT root parallelization (separate trees per thread) or tree copying.

**WU-UCT Virtual Loss**: Visit-only virtual loss mechanism that increments the denominator in the PUCT exploration term without distorting Q-values. Formula: `PUCT = Q + c * P * sqrt(N_parent) / (1 + N + VL)`, where Q = W/N remains pure.

**Busy-Edge Masking**: Setting PUCT score to `-∞` for nodes currently being expanded by another thread, preventing duplicate expansion attempts.

**Async Batched Inference**: Neural network inference requests accumulated in a lock-free queue until batch size threshold (32-64) or timeout (0.5-2.0ms) is reached, processed in a single GPU call, then results distributed to requesting threads.

**DLPack Zero-Copy**: Tensor protocol for sharing memory between C++ and PyTorch without copying data. In practice, uses pinned CPU memory (`kDLCUDAHost`) with 0.24ms H2D transfer, which is acceptable overhead (0.7% of total time per `review.txt`).

### 2.3 Performance Terminology

**Baseline**: 3,831 simulations/second configuration from Spec 003 (parameters unknown, requires T017 investigation).

**Current**: 2,147 simulations/second with Phase 1+2 optimizations applied (T001-T010, T006c, T008f complete).

**Validation Status** (2025-10-13):
- ✅ T-VALID-1 (FP16): PASS - 1.72× speedup confirmed (30.7ms vs 52.8ms FP32)
- ❌ T-VALID-2 (Tensor): FAIL - 7.5ms overhead (missing OpenMP parallelization)
- 🔴 Critical blocker: Feature extraction not parallelized, caps throughput at ~1,675 states/sec
- 📄 Report: `docs/performance/validation_report_2025-10-13.md`

**Target**: ≥8,000 simulations/second sustained throughput with ≥80% GPU utilization (revised from ≥25,000, see Section 12.1 Q1).

**Thread Efficiency**: `(actual_throughput) / (single_thread_throughput × num_threads)` expressed as percentage.

**GPU Utilization**: Percentage of time GPU execution units are actively computing, measured via `nvidia-smi` or profiling tools.

---

## 3. Goals

### 3.1 Primary Goals

**G1: Achieve Target Throughput** (Revised 2025-10-13, see Section 12.1 Q1)
- **Minimum Viable**: ≥6,000 sims/sec (1.6× baseline, 2.8× current)
- **Target (Realistic)**: ≥8,000 sims/sec (2.1× baseline, 3.7× current)
- **Stretch (Optimistic)**: ≥10,000 sims/sec (2.6× baseline, 4.7× current)
- **Aspirational** (out of scope, requires architectural changes): ≥15,000-30,000 sims/sec

**G2: Maximize CPU Utilization**
- 4 threads: ≥70% efficiency (minimum), ≥80% efficiency (target)
- 8 threads: ≥60% efficiency (minimum), ≥75% efficiency (target)
- 12 threads: ≥50% efficiency (minimum), ≥65% efficiency (target)

**G3: Maximize GPU Utilization**
- Target: ≥80% GPU utilization during search (batch size 64, timeout <2ms)
- Stretch: ≥85% sustained (optimistic, may require batch size 128)
- Constraint: Average batch size ≥48 positions (75% of max 64)

**G4: Reduce Coordination Overhead**
- MCTS coordination: From 67.2% → ≤30% of total time
- Thread idle time: From 42% → ≤12% of execution time
- Batch tensor creation: From 7.5ms → ≤1.0ms per batch
- Python/GIL overhead: From 67% → ≤30% of total time

**G5: Maintain Search Quality**
- Win rate: ≥99.5% vs baseline (3,831 sims/sec configuration)
- Policy agreement: ≥95% top-move agreement on 1000-position test set
- Value accuracy: MSE ≤0.01 vs baseline value estimates
- Collision rate: ≤5% path collisions (threads selecting same node)

### 3.2 Secondary Goals

**G6: Memory Efficiency**
- Tree memory: <1GB for 10M nodes (achieved: 270MB)
- Queue allocation: 1MB fixed (4096-entry ring buffer)
- DLPack buffers: <10MB pinned memory
- Total working set: <1.3GB

**G7: Reproducibility & Measurability**
- Deterministic profiling with fixed seeds
- Automated benchmark suite (`pytest -m performance`)
- Performance regression detection (alert if <95% baseline)
- Comprehensive instrumentation (collision metrics, allocation stats)

---

## 4. Non-Goals & Out-of-Scope

### 4.1 Explicitly Excluded

**NG1: Neural Network Implementation Changes**
- NO libtorch (C++ PyTorch inference)
- NO TensorRT model conversion or deployment
- NO ONNX model export and runtime
- NO custom CUDA kernels for neural network operations
- **Rationale**: Python PyTorch provides flexibility for model experimentation. Per `review.txt`, GPU is NOT the bottleneck (32.8% of time), so C++ inference adds complexity without proportional gains.

**NG2: GPU-Accelerated MCTS**
- NO GPU-resident MCTS trees
- NO GPU-based selection or backpropagation
- NO CUDA kernels for MCTS operations
- **Rationale**: CPU-parallel shared-tree MCTS is the established architecture. GPU MCTS requires fundamental redesign with uncertain performance benefits.

**NG3: Training Pipeline Optimization**
- NO self-play generation improvements (unless blocking throughput validation)
- NO experience replay buffer optimizations
- NO training speed enhancements
- NO model architecture changes
- **Rationale**: Training is out of scope; focus is pure MCTS search throughput.

**NG4: Telemetry & Logging Enhancements**
- NO advanced logging systems (unless required for performance measurement)
- NO real-time monitoring dashboards
- NO distributed tracing infrastructure
- **Rationale**: Telemetry overhead can degrade performance; instrumentation limited to benchmarking needs.

**NG5: Game Expansion**
- NO support for games beyond Gomoku/Chess/Go 9×9
- NO large board sizes (Go 19×19 deferred)
- NO game-specific optimizations beyond existing feature extraction
- **Rationale**: Extensibility preserved but not prioritized; focus on reference implementations.

### 4.2 Deferred Items
- Go 19×19 support (larger state space, requires validation)
- Multi-GPU inference (single GPU sufficient for target throughput)
- NUMA optimization beyond Ryzen dual-CCD (hardware-specific)

---

## 5. User Stories

### 5.1 Self-Play Training (Primary Use Case)

**As a** reinforcement learning researcher
**I want to** generate 200-300 self-play games per hour
**So that** I can train strong models within 48-72 hours for Gomoku

**Acceptance Criteria**:
- 800 simulations/move at 25,000 sims/sec = 32ms per move
- 100-move average game length = 3.2 seconds per game
- 200 games/hour = 1 game per 18 seconds (includes game setup overhead)
- GPU utilization ≥85% during search
- Search quality: ≥99.5% win rate vs baseline

**Current Gap**: At 2,147 sims/sec, 800 simulations take 373ms/move → 37 seconds/game → 97 games/hour (48% of target)

### 5.2 Interactive Play & Analysis

**As a** competitive player
**I want to** receive move recommendations within 3 seconds
**So that** I can use the engine for real-time game analysis

**Acceptance Criteria**:
- 1600 simulations in ≤3 seconds (533 sims/sec minimum)
- Policy distribution and value estimate displayed
- Top 5 moves with visit counts and Q-values
- Consistent latency (CV < 10%)

**Current Status**: 1600 simulations at 2,147 sims/sec = 0.75 seconds (MEETS requirement, but headroom for larger searches)

### 5.3 Performance Profiling & Optimization

**As a** performance engineer
**I want to** measure throughput with deterministic configurations
**So that** I can validate optimization effectiveness

**Acceptance Criteria**:
- Fixed seed, fixed game state, fixed simulation count
- 3+ independent runs with CV < 5%
- Detailed breakdown: selection time, expansion time, inference time, backup time
- Instrumentation: collision metrics, batch sizes, GPU utilization
- Reproducible on target hardware (Ryzen 5900X + RTX 3060 Ti)

**Current Status**: Profiling infrastructure exists (`profiling_results/`), but baseline configuration unknown (T017 blocker)

### 5.4 Configuration Tuning

**As a** system administrator
**I want to** tune batch size, timeout, and thread count for my hardware
**So that** I can maximize throughput on different CPU/GPU combinations

**Acceptance Criteria**:
- Tuning scripts: `scripts/tune_batch_size.py`, `scripts/tune_timeout.py`, `scripts/tune_threads.py`
- Automated sweep over parameter ranges
- Output: optimal configuration with 95% confidence interval
- Documentation: `docs/tuning_guide.md` with hardware-specific recommendations

**Current Status**: Tuning scripts exist, but optimal configurations not validated (T018-T019 pending)

---

## 6. Functional Requirements

### 6.1 MCTS Search Algorithm

**FR1: Shared Tree Architecture**
- **Description**: Single tree structure shared by all simulation threads
- **Implementation**: Atomic operations for visit counts, total values, virtual losses
- **Rationale**: Memory efficiency (270MB vs 2.16GB for 8 separate trees), information sharing across threads
- **Validation**: TSan clean (zero data races), collision rate ≤5%

**FR2: WU-UCT Virtual Loss**
- **Description**: Visit-only virtual loss that preserves pure Q = W/N
- **Formula**: `PUCT = (W/N) + c * P * sqrt(N_parent) / (1 + N + VL)`
- **Implementation**: Separate `in_flight_counts_` array with atomic increments
- **Rationale**: Prevents Q-value distortion while discouraging re-selection
- **Validation**: Unit tests verify Q unchanged with in-flight simulations

**FR3: Busy-Edge Masking**
- **Description**: Set PUCT = -∞ for nodes currently being expanded
- **Implementation**: Check `is_expanding()` flag in selection, skip if true
- **Rationale**: Prevents duplicate expansion attempts, reduces wasted work
- **Validation**: Thread safety tests ensure only 1 winner in 20-thread contention

**FR4: Root Pre-Expansion**
- **Description**: Expand root node synchronously before launching simulation threads
- **Implementation**: `ensure_root_expanded()` method with atomic expansion flag
- **Rationale**: Eliminates N-1 thread idle problem (threads waiting for root expansion)
- **Validation**: Only 1 inference request for root, subsequent calls are no-ops

**FR5: Dirichlet Noise**
- **Description**: Add exploration noise to root node priors during self-play
- **Formula**: `P'(a) = (1 - ε) * P(a) + ε * η_a`, where η ~ Dirichlet(α)
- **Parameters**: ε = 0.25, α = 0.3 (Gomoku), α = 0.03 (Chess/Go)
- **Rationale**: Ensures exploration diversity during self-play training
- **Validation**: Prior distribution preservation verified in unit tests

### 6.2 Async Batched Inference

**FR6: Lock-Free Inference Queue**
- **Description**: MPMC ring buffer (4096 entries) for inference requests
- **Implementation**: Turn-based synchronization with atomic head/tail pointers
- **Performance**: <2ns per enqueue operation, wait-free for producers
- **Validation**: Concurrent tests (SPSC, MPSC, SPMC, MPMC) with 10k items

**FR7: Condition Variable Coordination**
- **Description**: Efficient blocking instead of polling with 10μs sleeps
- **Implementation**: `std::condition_variable` notified on request submission
- **Rationale**: Reclaims 67% CPU wasted on polling per `review.txt`
- **Validation**: No busy-wait loops, thread CPU usage drops to near-zero when idle

**FR8: Batch Collection Strategy**
- **Description**: Collect batch when count ≥ min_batch_size OR timeout expires
- **Parameters**: min_batch_size = 32-64, timeout = 0.5-2.0ms (tunable)
- **Rationale**: Balance GPU efficiency (larger batches) vs thread responsiveness (shorter waits)
- **Validation**: Average batch size ≥48, GPU utilization ≥85%

**FR9: DLPack Zero-Copy Tensors**
- **Description**: Share memory between C++ and PyTorch via DLPack protocol
- **Implementation**: Pinned CPU memory (`kDLCUDAHost`), `torch.from_dlpack()` conversion
- **Performance**: 0.24ms H2D transfer per batch (0.7% of total time, acceptable)
- **Validation**: No unnecessary copies, memory bandwidth ≥8.6 GB/s

**FR10: FP16 Mixed Precision**
- **Description**: GPU inference using FP16 tensor cores on RTX 3060 Ti
- **Implementation**: `torch.cuda.amp.autocast()` context manager
- **Expected Impact**: 1.5-2× GPU inference speedup per `review.txt`
- **Validation**: Numerical stability (value MSE ≤0.01 vs FP32), actual speedup measured

### 6.3 Memory Management

**FR11: Thread-Local Block Allocation**
- **Description**: 4096-node blocks allocated per thread, no global mutex in fast path
- **Implementation**: Thread-local storage with atomic fallback for exhaustion
- **Performance**: 99.93% fast-path allocation, 0.07% slow-path (global mutex)
- **Validation**: Allocation speed benchmark (0.0077 μs/node), thread safety tests

**FR12: Epoch-Based Tree Clearing**
- **Description**: O(1) tree reset via epoch counter increment (no memset)
- **Implementation**: Compare node epoch with global epoch on access
- **Performance**: 25ns tree clear (vs 25ms memset for 10M nodes)
- **Validation**: Lazy initialization correctness, memory profile shows no memset

**FR13: State Object Reuse**
- **Description**: Thread-local game state objects reused across simulations
- **Implementation**: Pool of pre-allocated states, `reset()` instead of `clone()`
- **Rationale**: Avoids 2-3× cloning per simulation (current waste per `review.txt`)
- **Validation**: Memory profiler shows constant allocation, no growth

**FR14: Structure-of-Arrays Layout**
- **Description**: Node data stored in separate arrays per field (N, W, P, VL)
- **Rationale**: Cache efficiency, SIMD-friendly, 64-byte alignment
- **Memory**: 27 bytes per node (visit_count: 4B, total_value: 4B, prior: 4B, virtual_loss: 4B, parent: 4B, first_child: 4B, metadata: 3B)
- **Validation**: Memory footprint 270MB for 10M nodes (<1GB target)

### 6.4 Thread Coordination

**FR15: Thread Affinity (Ryzen 5900X)**
- **Description**: Pin threads to physical cores for cache locality
- **Topology**: CCD0 (cores 0-5), CCD1 (cores 6-11), avoid SMT siblings
- **Strategy**: ≤6 threads → CCD0 only; 7-12 threads → distribute across CCDs
- **Rationale**: Minimize cross-CCD traffic, maximize L3 cache hits
- **Validation**: Topology detection functional, affinity setting verified

**FR16: Batched Backpropagation**
- **Description**: Accumulate value updates and apply atomically in batch
- **Implementation**: Thread-local accumulation buffer, single atomic add per node
- **Rationale**: Reduces atomic contention (was identified in profiling)
- **Validation**: Correctness tests (values sum correctly), contention reduced

**FR17: Persistent Coordinator Lifecycle**
- **Description**: BatchInferenceCoordinator created once in `MCTSAgent.__init__`
- **Implementation**: Reused across all searches, stopped only in `close()` or `__del__`
- **Rationale**: Eliminates per-search thread startup/teardown (67% of overhead per `review.txt`)
- **Validation**: 1000+ searches with single coordinator, no restarts

---

## 7. Non-Functional Requirements

### 7.1 Performance

**NFR1: Throughput Targets by Game** (Revised 2025-10-13)

| Game | Board Size | Input Planes | Minimum (sims/sec) | Target (sims/sec) | Stretch (sims/sec) | Rationale |
|------|-----------|--------------|-------------------|------------------|-------------------|-----------|
| Gomoku/Renju | 15×15 | 36 | 5,500 | 7,500 | 9,500 | Most expensive (36 planes) |
| Chess | 8×8 | 30 | 6,000 | 8,000 | 10,000 | Baseline game |
| Go | 9×9 | 25 | 6,500 | 8,500 | 10,500 | Cheapest (25 planes, 9×9 board) |

**Note**: Targets adjusted proportionally from revised 8k baseline. Gomoku penalized ~6% for higher plane count.

**NFR2: Latency Budgets (per 1000 simulations)**

| Component | Current | Target | Max Acceptable |
|-----------|---------|--------|---------------|
| MCTS Coordination | 240ms (67.2%) | 80ms (30%) | 120ms (40%) |
| GPU Inference | 117ms (32.8%) | 60ms (25%) | 100ms (35%) |
| Tensor Creation | 75ms (21%) | 10ms (4%) | 20ms (8%) |
| Thread Idle | 150ms (42%) | 30ms (12%) | 50ms (18%) |
| **Total** | **357ms** | **120ms** | **180ms** |

**NFR3: Scalability**

| Threads | Min Efficiency | Target Efficiency | Max Throughput Gain |
|---------|---------------|------------------|-------------------|
| 1 | 100% (baseline) | 100% | 1.0× |
| 2 | ≥85% | ≥90% | 1.8× |
| 4 | ≥70% | ≥80% | 3.2× |
| 8 | ≥60% | ≥75% | 6.0× |
| 12 | ≥50% | ≥65% | 7.8× |

**NFR4: GPU Utilization by Batch Size**

| Batch Size | Min GPU Util | Target GPU Util | Avg Batch Fill |
|-----------|-------------|----------------|---------------|
| 32 | ≥75% | ≥80% | ≥24 (75%) |
| 64 | ≥80% | ≥85% | ≥48 (75%) |
| 128 | ≥85% | ≥90% | ≥96 (75%) |

### 7.2 Correctness

**NFR5: Search Quality Preservation**
- Win rate: ≥99.5% vs baseline (1000+ games, 95% confidence)
- Policy agreement: ≥95% top-move agreement (1000-position test set)
- Value accuracy: MSE ≤0.01 vs baseline value estimates
- Illegal moves: 0 illegal move selections (hard requirement)

**NFR6: Thread Safety**
- TSan clean: Zero data races under maximum thread count (24 threads)
- Helgrind clean: No race conditions detected
- Correctness tests: All unit/integration tests pass
- Stress tests: 1-hour soak test with no crashes or corruption

**NFR7: Determinism (Optional)**
- Fixed seed → identical search tree (when single-threaded)
- Multi-threaded: visit count distribution reproducible (±1% variance)
- Profiling runs: CV < 5% across 3+ independent runs

### 7.3 Portability

**NFR8: Platform Support**
- **Primary**: Linux (Ubuntu 22.04+), Ryzen 5900X, RTX 3060 Ti
- **Secondary**: WSL2 (Windows 11), generic x86-64 CPUs, NVIDIA GPUs
- **Degradation**: Thread affinity disabled on non-Ryzen CPUs
- **Build**: CMake 3.24+, GCC 13-15/Clang 15+, Python 3.12

**NFR9: Hardware Assumptions**
- CPU: 12+ cores, 64GB RAM, 2× L3 cache (dual-CCD or NUMA)
- GPU: NVIDIA GPU with tensor cores (Volta/Turing/Ampere), 8GB+ VRAM
- Storage: SSD for profiling data and checkpoints

**NFR10: Dependency Constraints**
- PyTorch 2.0+ (DLPack support, AMP API)
- pybind11 2.10+ (DLPack capsule support)
- NumPy 1.20+ (array protocol compatibility)
- CUDA 12.1+ (tensor core support, FP16 performance)

### 7.4 Maintainability

**NFR11: Code Quality**
- C++ style: Google C++ Style Guide, clang-format enforced
- Python style: Black formatter, isort, flake8
- Documentation: Inline comments for hot paths (selection, backup)
- Test coverage: ≥80% for core MCTS logic

**NFR12: Profiling & Instrumentation**
- Collision metrics: path collisions, expansion conflicts, busy-edge blocks
- Allocation stats: fast-path %, slow-path %, free-list reuse %
- Timing breakdown: selection time, expansion time, inference time, backup time
- GPU metrics: utilization, batch sizes, kernel durations

**NFR13: Configuration**
- YAML config files: `config/performance_tuning.yaml`
- Runtime flags: `--threads`, `--batch-size`, `--timeout`
- Environment variables: `MCTS_PROFILING=1`, `MCTS_INSTRUMENTATION=1`
- Tuning scripts: automated parameter sweeps with optimal config output

---

## 8. KPIs & Benchmarks

### 8.1 Primary KPIs

**KPI1: Absolute Throughput**
- **Metric**: Simulations per second (sims/sec)
- **Measurement**: `python scripts/benchmark_throughput.py --game gomoku --simulations 10000 --threads 8 --iterations 3`
- **Target**: ≥8,000 sims/sec (mean over 3 runs, CV < 5%) [revised from 25k, see Section 12.1 Q1]
- **Baseline**: 3,831 sims/sec (Spec 003 configuration, needs T017 reproduction)
- **Current**: 2,147 sims/sec (Phase 1+2 complete, regression under investigation)
- **Validation Status** (2025-10-13): Critical blocker found (tensor creation 7.5ms), fix required before target achievable
- **Validation**: Must exceed target before Phase 4 completion

**KPI2: Thread Efficiency**
- **Metric**: `(actual_throughput) / (single_thread_baseline × num_threads)`
- **Measurement**: Sweep threads 1,2,4,8,12 with fixed simulations (10,000)
- **Target**: ≥75% at 8 threads, ≥65% at 12 threads
- **Validation**: Regression alert if efficiency < 90% of prior run

**KPI3: GPU Utilization**
- **Metric**: Average GPU compute percentage during search
- **Measurement**: `nvidia-smi dmon -s u -i 0` during benchmark
- **Target**: ≥80% sustained (batch 64, timeout 1.0ms) [revised from 85%]
- **Stretch**: ≥85% (optimistic, may require batch 128)
- **Current**: 11.2% (per `profiling_results/`, severe underutilization)
- **Validation**: Must exceed 80% before declaring optimization complete

**KPI4: Coordination Overhead**
- **Metric**: MCTS coordination time as % of total time
- **Measurement**: C++ instrumentation in `ContinuousSimulationRunner`
- **Target**: ≤30% (currently 67.2% per `review.txt`)
- **Breakdown**: Thread idle ≤12%, queue ops ≤10%, backup ≤8%
- **Validation**: Profiling breakdown must show target distribution

### 8.2 Secondary KPIs

**KPI5: Average Batch Size**
- **Metric**: Mean positions per GPU inference batch
- **Target**: ≥48 (75% of max 64)
- **Measurement**: Instrumentation counter in `BatchInferenceCoordinator`
- **Validation**: Histogram shows majority of batches in [48, 64] range

**KPI6: Collision Rate**
- **Metric**: Path collisions / total simulations
- **Target**: ≤5%
- **Measurement**: `ExpansionConflict` + `BusyEdgeMasked` counters
- **Validation**: A/B test with/without optimizations, collision rate improves

**KPI7: Memory Footprint**
- **Metric**: Resident Set Size (RSS) during search
- **Target**: <1.3GB (tree 270MB + queue 1MB + buffers 10MB + Python overhead)
- **Measurement**: `memory_profiler` or `/proc/self/status` parsing
- **Validation**: No growth over 1000+ searches (leak detection)

**KPI8: Search Quality**
- **Metric**: Win rate vs baseline in head-to-head matches
- **Target**: ≥99.5% (1000 games, 95% confidence interval)
- **Measurement**: `python scripts/compare_search_quality.py --baseline v003 --candidate v004`
- **Validation**: Policy agreement ≥95%, value MSE ≤0.01

### 8.3 Benchmark Suites

**Benchmark 1: Throughput Scaling (threads)**
```bash
pytest tests/performance/test_thread_scaling.py -v
# Sweeps threads [1, 2, 4, 8, 12] with fixed simulations
# Output: sims/sec, efficiency %, CPU utilization %
# Artifacts: CSV data, scaling curve plot
```

**Benchmark 2: Batch Size Optimization**
```bash
pytest tests/performance/test_batch_optimization.py -v
# Sweeps batch sizes [16, 32, 64, 128] with fixed threads (8)
# Output: sims/sec, GPU util %, avg batch size
# Artifacts: Optimal batch size with 95% CI
```

**Benchmark 3: Timeout Tuning**
```bash
pytest tests/performance/test_timeout_tuning.py -v
# Sweeps timeouts [0.5, 1.0, 2.0, 5.0] ms with fixed batch (64)
# Output: sims/sec, GPU util %, thread idle %
# Artifacts: Optimal timeout with trade-off analysis
```

**Benchmark 4: Regression Detection**
```bash
pytest tests/performance/test_regression.py -v
# Compares current vs baseline throughput (T017 config)
# Output: PASS/FAIL (≥95% baseline), delta %, statistical significance
# Artifacts: Historical trend CSV, alert on regression
```

**Benchmark 5: Search Quality Validation**
```bash
pytest tests/performance/test_search_quality.py -v
# Head-to-head matches (1000 games) vs baseline
# Output: win rate %, policy agreement %, value MSE
# Artifacts: Game records (SGF/PGN), statistical analysis
```

### 8.4 Profiling Sessions

**Session 1: Baseline Reproduction (T017)**
```bash
python scripts/profile_baseline_config.py \
  --sweep-threads 1,2,4,8,12 \
  --sweep-batch 16,32,64 \
  --sweep-timeout 0.5,1.0,2.0 \
  --simulations 10000 \
  --iterations 3
# Goal: Find configuration that achieved 3,831 sims/sec
# Output: profiling_results/session_YYYYMMDD_HHMMSS/baseline_config.json
```

**Session 2: Optimization Validation (T016)**
```bash
python scripts/profile_optimizations.py \
  --compare T006c,T008f,T011 \
  --baseline 3831_config \
  --threads 8 \
  --batch 64 \
  --timeout 1.0 \
  --simulations 10000 \
  --iterations 5
# Goal: Measure actual gains from T006c+T008f (expected 18-36k sims/sec)
# Output: profiling_results/optimization_impact/T006c_T008f_results.json
```

**Session 3: Bottleneck Analysis (T020)**
```bash
python scripts/profile_bottlenecks.py \
  --tool py-spy,perf,torch-profiler \
  --threads 8 \
  --batch 64 \
  --simulations 10000
# Goal: Identify remaining hotspots (if still <25k sims/sec)
# Output: profiling_results/bottleneck_analysis/flame_graph.svg, perf_report.txt
```

---

## 9. User Acceptance Criteria

### 9.1 Self-Play Training Acceptance

**UAC1: Training Time (Gomoku)**
- **Criterion**: Achieve superhuman Gomoku play within 48 hours of training
- **Measurement**: Generate 50,000 self-play games at 200 games/hour
- **Dependencies**: 25,000 sims/sec sustained, 800 sims/move, 100-move avg game length
- **Validation**: Model defeats baseline agent (1600 sims/move) in ≥95% of 100 games

**UAC2: GPU Efficiency**
- **Criterion**: GPU utilization ≥85% during self-play batch generation
- **Measurement**: `nvidia-smi dmon` during `python scripts/selfplay.py`
- **Validation**: No GPU idle periods >100ms during search phases

**UAC3: System Stability**
- **Criterion**: 24-hour continuous self-play with no crashes or memory leaks
- **Measurement**: Valgrind leak check, RSS monitoring every 1 hour
- **Validation**: Memory growth <1MB/hour, zero segfaults

### 9.2 Interactive Play Acceptance

**UAC4: Response Time**
- **Criterion**: 1600 simulations complete in ≤3 seconds (interactive play)
- **Measurement**: `python scripts/interactive_play.py --time-limit 3.0`
- **Validation**: 95th percentile latency ≤3.5 seconds (with 10% buffer)

**UAC5: Move Quality**
- **Criterion**: Recommended moves match strong engine (Katago for Go, Stockfish for Chess)
- **Measurement**: Top-move agreement on 100-position tactical puzzles
- **Validation**: Agreement ≥85% (accounts for style differences)

### 9.3 Performance Tuning Acceptance

**UAC6: Configuration Discovery**
- **Criterion**: Automated tuning scripts find optimal config for user hardware
- **Measurement**: `python scripts/tune_all.py --hardware-profile`
- **Output**: `config/optimized_HOSTNAME.yaml` with batch size, timeout, threads
- **Validation**: Optimized config achieves ≥95% of theoretical max throughput

**UAC7: Reproducibility**
- **Criterion**: Published benchmark results reproducible on equivalent hardware
- **Measurement**: External validation on Ryzen 5900X + RTX 3060 Ti
- **Tolerance**: ±5% variance from published numbers
- **Validation**: 3+ independent runs by different users achieve target

---

## 10. Risks & Mitigations

### 10.1 Performance Risks

**Risk 1: GPU Becomes Bottleneck**
- **Probability**: Low (GPU currently 32.8% of time, room for growth)
- **Impact**: Medium (limits max throughput to ~10k sims/sec if GPU saturates)
- **Mitigation**: FP16 mixed precision (T008f, 1.5-2× speedup), batch size tuning (T019)
- **Contingency**: Accept 20,000 sims/sec (80% of target), document GPU as bottleneck
- **Monitoring**: Track GPU utilization per benchmark run, alert if >95%

**Risk 2: Thread Contention Saturates**
- **Probability**: Medium (current 60% thread idle suggests coordination issues)
- **Impact**: High (prevents scaling beyond 4-8 threads efficiently)
- **Mitigation**: Condition variables (T006c), affinity tuning (T004), relaxed atomics (T012)
- **Contingency**: Scale to 16 threads with SMT, accept 90% efficiency (vs 75% target)
- **Monitoring**: Thread efficiency metrics per benchmark, flame graphs for contention

**Risk 3: Python Overhead Irreducible**
- **Probability**: Medium (67% overhead currently, target 30%)
- **Impact**: High (limits max throughput even with perfect MCTS)
- **Mitigation**: Persistent coordinator (T011), DLPack zero-copy (T007), batched inference
- **Contingency**: Accept 30-35% Python overhead (vs 20% ideal), document as limit
- **Monitoring**: GIL hold time per `py-spy`, Python/C++ boundary crossing counts

**Risk 4: Baseline Unreproducible**
- **Probability**: Medium (3,831 sims/sec config unknown, T017 investigation ongoing)
- **Impact**: Medium (invalidates improvement claims, requires new baseline)
- **Mitigation**: Systematic config sweep (threads, batch, timeout, model size)
- **Contingency**: Use 2,147 sims/sec as new baseline, target 10× improvement (21,470 sims/sec)
- **Monitoring**: Document all config parameters in benchmark artifacts

### 10.2 Quality Risks

**Risk 5: Search Quality Regression**
- **Probability**: Low (optimizations designed to preserve correctness)
- **Impact**: Critical (unacceptable if win rate < 99.5% vs baseline)
- **Mitigation**: A/B testing every optimization (T017), MSE validation, policy agreement
- **Contingency**: Rollback flag per optimization (config-based disable), revert commits
- **Monitoring**: Automated quality tests in CI, alert on regression

**Risk 6: Value Drift from Approximations**
- **Probability**: Low (WU-UCT preserves Q = W/N, relaxed atomics use acquire/release)
- **Impact**: Medium (slight value inaccuracy acceptable if <1% MSE)
- **Mitigation**: MSE validation on 1000-position test set, tighten approximations if needed
- **Contingency**: Accept throughput loss (5-10%) to ensure value accuracy
- **Monitoring**: Value MSE tracked in quality benchmarks

**Risk 7: Race Conditions Introduced**
- **Probability**: Low (TSan on every commit, lock-free algorithms proven)
- **Impact**: Critical (crashes, corrupted trees, incorrect results)
- **Mitigation**: TSan/Helgrind in CI, stress tests (1-hour soak), code review
- **Contingency**: Revert to mutex-based fallback (queue, result storage)
- **Monitoring**: TSan warnings in CI logs, crash reports

**Risk 8: Memory Leaks Under Load**
- **Probability**: Low (epoch-based clearing eliminates leaks, pools reused)
- **Impact**: Medium (requires restarts during long self-play sessions)
- **Mitigation**: Valgrind leak checks, RSS monitoring, pool exhaustion detection
- **Contingency**: Graceful degradation (pause self-play, clear tree, resume)
- **Monitoring**: Memory growth rate tracked, alert if >10MB/hour

### 10.3 Schedule Risks

**Risk 9: Optimization Stalls**
- **Probability**: Medium (unknown bottlenecks may remain after Phase 3)
- **Impact**: High (blocks Phase 4 completion if <20k sims/sec)
- **Mitigation**: Profile-guided optimization (T020), systematic bottleneck elimination
- **Contingency**: Escalate for architectural review, consider alternative approaches
- **Monitoring**: Throughput progress tracked per task completion

**Risk 10: Hardware Dependency**
- **Probability**: Low (code portable, degradation on non-Ryzen hardware)
- **Impact**: Medium (requires Ryzen 5900X for optimal performance)
- **Mitigation**: Generic topology fallback, CI runs on standard hardware
- **Contingency**: Document hardware-specific optimizations, provide tuning guide
- **Monitoring**: Test suite runs on diverse hardware (GitHub Actions, local machines)

---

## 11. Dependencies

### 11.1 Internal Dependencies

**D1: Baseline Configuration (T017)**
- **Blocker**: All optimization validation depends on reproducing 3,831 sims/sec
- **Status**: Critical path, highest priority
- **Resolution**: 2-day investigation budget, systematic config sweep

**D2: Benchmarking Infrastructure (T016)**
- **Blocker**: Can't measure T006c+T008f gains without comprehensive benchmark suite
- **Status**: Blocked on T017 (need baseline for comparison)
- **Resolution**: Parallel development, use 2,147 sims/sec as interim baseline

**D3: Lock-Free Queue (T006/T006b)**
- **Dependency**: Condition variables (T006c) require lock-free queue foundation
- **Status**: ✅ Complete (19/19 tests passing, integrated)
- **Resolution**: No action needed

**D4: DLPack Bridge (T007)**
- **Dependency**: FP16 mixed precision (T008f) requires zero-copy tensor sharing
- **Status**: ✅ Complete (T007a-g implemented, validated)
- **Resolution**: No action needed

### 11.2 External Dependencies

**D5: PyTorch 2.0+**
- **Requirement**: DLPack support (`torch.from_dlpack()`), AMP API (`torch.cuda.amp.autocast()`)
- **Status**: ✅ Available (PyTorch 2.0 released 2023-03)
- **Risk**: None (stable API)

**D6: pybind11 2.10+**
- **Requirement**: DLPack capsule support (`py::capsule`)
- **Status**: ✅ Available (pybind11 2.10 released 2022-11)
- **Risk**: None (stable API)

**D7: CUDA 12.1+**
- **Requirement**: Tensor core support for FP16, cuDNN optimizations
- **Status**: ✅ Available (CUDA 12.1 released 2023-03)
- **Risk**: Low (driver compatibility on older GPUs)

**D8: Ryzen 5900X Hardware**
- **Requirement**: Dual-CCD topology for thread affinity validation
- **Status**: ✅ Available (target hardware in use)
- **Risk**: Low (generic fallback for other CPUs)

### 11.3 Task Dependencies

**Critical Path**:
1. T017 (baseline) → T016 (benchmarking) → T018/T019 (tuning) → T020 (profiling) → T025 (validation)

**Parallel Tracks**:
- T012 (relaxed atomics) | T013 (prefetching) | T015 (hot/cold separation) → T020 (validation)

**Completed Prerequisites**:
- ✅ T001-T005 (Phase 1: Virtual Loss)
- ✅ T006-T010, T006c, T008f (Phase 2: Architecture)
- ✅ T011 (Phase 3: Persistent Coordinator)

---

## 12. Open Questions → RESOLVED AMBIGUITIES

**NOTE**: All critical blocking ambiguities have been resolved through analysis of review.txt, profiling data, and hardware constraints. The following documents final decisions.

---

### 12.1 Performance Questions

**Q1: What are the REALISTIC throughput targets?** ✅ **RESOLVED**
- **Ambiguity**: SPECIFICATION targeted ≥25,000 sims/sec, but review.txt states this is **unrealistic** on Ryzen 5900X + RTX 3060 Ti
- **Evidence**:
  - GPU cap @ FP32: 3,885 states/sec (8.24ms/batch-32)
  - GPU cap @ FP16: ~7,500-10,000 states/sec (review.txt line 34)
  - Review.txt conclusion: "realistic goal is on the order of 5,000–10,000 sims/sec"
- **RESOLUTION**:
  ```
  Revised Targets (Hardware-Grounded):
    Minimum Viable:  ≥6,000 sims/sec  (1.6× baseline, 2.8× current)
    Target (Realistic): ≥8,000 sims/sec  (2.1× baseline, 3.7× current)
    Stretch (Optimistic): ≥10,000 sims/sec (2.6× baseline, 4.7× current)
    Aspirational (Requires Changes): ≥15-25k sims/sec (model pruning, multi-GPU)

  Rationale:
    - GPU @ FP16 + 80% util: 8,000-10,000 states/sec achievable
    - 20% coordination overhead budget: 6,400-8,000 sims/sec realistic
    - 25k target retained as "future work" requiring architectural changes
  ```
- **Impact**: CRITICAL - Updates all milestones, acceptance criteria, and success metrics

**Q2: What is the exact baseline configuration?** ✅ **RESOLUTION PROTOCOL**
- **Ambiguity**: 3,831 sims/sec baseline is completely unknown (thread count, batch size, timeout, model size)
- **Blocking**: Cannot validate T006c+T008f gains, cannot claim "6.5× baseline" improvements
- **RESOLUTION (T017 Protocol)**:
  ```
  Time-Boxed Investigation (2 Days Max):
    Day 1 - Archaeological Dig:
      - Search git history for commits around 3,831 measurement (Spec 003 era)
      - Check config/performance_tuning.yaml history
      - Search profiling_results/ for matching throughput
      - Review logs, benchmark outputs, commit messages

    Day 2 - Systematic Reproduction:
      IF config found: Reproduce exactly, validate ±5%
      IF NOT found: Run grid search:
        threads: [1, 2, 4, 8, 12]
        batch_size: [16, 32, 64]
        timeout: [0.5, 1.0, 2.0, 5.0] ms
        Document new baseline as best config from grid

    Fallback (End Day 2):
      - Declare 3,831 "historical, unreproducible"
      - Use best grid config as NEW baseline
      - Update all specs to reference new baseline
  ```
- **Impact**: CRITICAL - Blocks T016 benchmarking and all improvement claims

**Q3: Is DLPack truly "zero-copy"?** ✅ **RESOLVED (MISLEADING TERMINOLOGY)**
- **Ambiguity**: SPECIFICATION claims "zero-copy" but review.txt proves it's **not true**:
  - Uses `kDLCUDAHost` (pinned CPU), NOT `kDLCUDA` (GPU device memory)
  - Incurs 0.24ms H2D transfer per batch = 0.7% overhead
- **RESOLUTION**:
  ```
  Accurate Terminology:
    - Replace "zero-copy" with "fast pinned-memory transfer"
    - Acknowledge 0.24ms H2D (acceptable, not worth complexity for 0.7% gain)
    - Update FR9 description: "Pinned CPU memory with 8.6 GB/s H2D bandwidth"

  Decision: KEEP current implementation (pinned CPU memory)
  ```
- **Impact**: MEDIUM - Corrects misleading claims, sets accurate expectations

**Q4: Has FP16 mixed precision been validated?** ✅ **RESOLVED (2025-10-13)**

**Ambiguity**: T008f was marked "COMPLETE" but never empirically validated (pre-2025-10-13)
  - Expected 1.5-2× GPU speedup (review.txt line 34)
  - No benchmark results documented

**RESOLUTION** (2025-10-13): T-VALID-1 executed and **PASSED**
```
Validation Results:
  FP32 Inference: 52.83 ± 0.39 ms/batch-64 (1,211 states/sec)
  FP16 Inference: 30.69 ± 0.46 ms/batch-64 (2,085 states/sec)
  Speedup: 1.72× (PASS: ≥1.5× required)

  Numerical Stability:
    Policy Probability MSE: 0.000007 (PASS: <0.01 required)
    Value MSE: 0.000000 (PASS: <0.01 required)

  Conclusion: FP16 tensor cores active and working correctly
```

**Report**: `docs/performance/validation_report_2025-10-13.md` Section T-VALID-1
**Impact**: RESOLVED - T008f validated successfully, 8k target remains achievable with FP16 speedup

**Q5: Is 7.5ms tensor creation overhead still present?** ❌ **CRITICAL ISSUE CONFIRMED (2025-10-13)**

**Ambiguity**: Review.txt (line 17) identified 7.5ms batch tensor creation as "enormous overhead" (pre-2025-10-13)
  - If true: 7.5ms + 8ms GPU = 15.5ms/batch → max 64 sims/sec (CATASTROPHIC)
  - Expected with DLPack: <1.0ms (review.txt line 72)

**RESOLUTION** (2025-10-13): T-VALID-2 executed and **FAILED**
```
Profiling Results:
  Configuration: batch=64, iterations=1000
  Mean: 7.50 ± 0.20 ms (FAIL: >1.0ms target)
  Min: 7.34 ms, Max: 10.50 ms
  p50: 7.47 ms, p95: 7.68 ms, p99: 8.16 ms

Root Cause Analysis:
  Location: cpp_extensions/mcts/dlpack_bridge.cpp:431-434
  Issue: Feature extraction loop NOT parallelized with OpenMP
  Evidence: Sequential extraction: 64 states × ~0.12ms/state = 7.5ms (matches measurement)
  Expected: With 12-thread OpenMP: 7.5ms / 12 = 0.625ms (within target)

Impact Analysis:
  Batches/sec: 133
  Wasted time/sec: 867ms (6.5ms overhead × 133)
  Potential speedup if fixed: 7.50×
  Throughput cap: ~1,675 states/sec (explains 2,147 regression)

Required Fix:
  // cpp_extensions/mcts/dlpack_bridge.cpp:431
  #pragma omp parallel for schedule(static) if(batch_size > 8)
  for (int i = 0; i < batch_size; ++i) {
      float* state_buffer = data + (i * state_size);
      states[i]->extract_features_to_buffer(state_buffer);
  }

Expected Result: 7.5ms → <1.0ms with 12-thread parallelization
```

**Report**: `docs/performance/validation_report_2025-10-13.md` Section T-VALID-2
**Impact**: **CRITICAL BLOCKER** - This issue must be fixed before proceeding to T016/T017. Caps throughput at ~1,675 states/sec and explains regression from 3,831 to 2,147 sims/sec.

### 12.2 Architecture Questions

**Q6: What is the optimal thread count?** ✅ **TUNING PROTOCOL (T018)**
- **Ambiguity**: Current best is 4 threads @ 2,147 sims/sec, but should we use 8-12?
  - Review.txt suggests 12 threads to keep GPU fed
  - But warns against 24 hyperthreads (diminishing returns)
- **RESOLUTION (T018 Tuning Strategy)**:
  ```
  Thread Range: 4-12 physical cores (avoid SMT initially)

  Affinity Strategy:
    - 4 threads: CCD0 only (cores 0-3), max L3 locality
    - 8 threads: CCD0 + CCD1 split (cores 0-3, 6-9)
    - 12 threads: All physical (cores 0-11), avoid 12-23

  Test Matrix:
    threads: [4, 6, 8, 10, 12]
    batch_size: [32, 48, 64]
    timeout_ms: [1.0, 2.0]

  Hypothesis:
    - Current 4-thread saturation suggests coordination bottleneck
    - Need 8-12 threads to generate 64+ pending requests for batch 64
    - Expected optimal: 8-10 threads @ batch 64 (balance GPU vs contention)
  ```
- **Impact**: HIGH - Critical for GPU utilization, blocks T019

**Q7: What is the batch size vs timeout trade-off?** ✅ **TUNING PROTOCOL (T019)**
- **Ambiguity**: SPEC says "32-64, 0.5-2.0ms" but review.txt suggests smaller batches may help:
  - "try batch size 16... trade some GPU utilization for more responsiveness"
  - Current 60% thread idle suggests batch collection too slow
- **RESOLUTION (T019 Grid Search)**:
  ```
  Hypothesis: Thread idle caused by slow batch collection

  Test Matrix:
    batch_size: [16, 32, 48, 64]
    timeout_ms: [0.5, 1.0, 2.0, 5.0]
    threads: [optimal from T018]

  Metrics:
    - Primary: sims/sec (throughput)
    - Secondary: GPU util % (target 80-85%)
    - Tertiary: thread idle % (target <20%)
    - Quaternary: avg batch fill (target ≥75% of max)

  Decision Criteria:
    - IF GPU util < 70%: Increase batch size
    - IF thread idle > 40%: Decrease timeout or batch size
    - IF sims/sec plateaus: Current config optimal

  Expected Outcome:
    - Optimal likely: batch 48-64 @ timeout 1.0-2.0ms
    - Fallback: batch 32 @ timeout 0.5ms if contention high
  ```
- **Impact**: HIGH - Critical for balance between GPU and thread efficiency

**Q8: Should we implement transposition tables?** ✅ **OUT OF SCOPE**
- **Ambiguity**: Review.txt and mcts_guide.md mention transposition tables for 40k sims/sec
- **RESOLUTION**:
  ```
  Decision: OUT OF SCOPE for Spec 004 (Phase 1-4)

  Rationale:
    - Complex to implement (state hashing, thread-safe cache, eviction)
    - Benefit unclear for Gomoku (low transposition rate vs Chess/Go)
    - Review.txt: needed for 40k (beyond hardware capability)

  Future Work (Spec 005):
    - Implement after achieving 8-10k baseline
    - Estimate 1.2-1.5× improvement (reuse NN evaluations)
    - Requires: thread-safe LRU cache, zobrist hashing

  Document in SPECIFICATION.md Section 4:
    - Add to "Out of Scope": "Transposition tables (deferred to Spec 005)"
  ```
- **Impact**: LOW - Clarifies scope, avoids scope creep

**Q9: How much does state cloning cost?** ✅ **PROFILING REQUIRED**
- **Ambiguity**: Review.txt identifies 2-3× state cloning per simulation as waste, but no measurements
  - Gomoku 15×15 = 225 cells × 4 bytes = 900 bytes per clone
  - At 10k sims/sec: 2-3 clones × 100ns × 10k = 2-3ms (0.2-0.3%?)
- **RESOLUTION (Micro-Benchmark)**:
  ```
  Profiling Protocol:
    Command: python scripts/profile_state_cloning.py \
               --game gomoku --simulations 10000 --iterations 100

    Measurements:
      - Clones per simulation (expect 2-3)
      - Time per clone (expect 50-100ns memcpy)
      - Total overhead as % of search time

    Decision Tree:
      - IF cloning > 5% of time: Implement FR13 (thread-local state pools)
      - IF cloning < 2% of time: Defer FR13, focus on coordination

    Expected Result: LOW PRIORITY (< 1% overhead)
  ```
- **Impact**: LOW - Quantifies cost, informs FR13 prioritization

**Q10: Is 4096-slot result buffer scan wasteful?** ✅ **CLARIFIED (T014)**
- **Ambiguity**: Review.txt mentions scanning 4096-slot buffer as wasted CPU, but unclear if T014 addresses it
- **RESOLUTION**:
  ```
  Current Implementation (T006b):
    - ResultSlot array with O(1) lookup: try_get_result(request_id % 8192)
    - Each thread fetches only its own results (no full scan)
    - consume_ready_results() exists but deprecated (compatibility only)

  Verification (T014):
    - Confirm ContinuousSimulationRunner uses try_get_result(), NOT consume_ready_results()
    - Deprecate/remove consume_ready_results() if unused
    - Document: "O(1) result retrieval per thread, no buffer scan"

  Conclusion: ALREADY FIXED in T006b, just needs verification
  ```
- **Impact**: LOW - Micro-optimization already addressed

### 12.3 Quality Questions (Defer to Post-Optimization)

**Q11: How much value drift is acceptable?** ✅ **DEFER TO T020**
- **Context**: Relaxed memory ordering (T012) may introduce slight value inconsistencies
- **RESOLUTION**:
  ```
  Decision: DEFER until T012 implemented

  Validation Protocol (when T012 ready):
    - A/B test: Relaxed atomics vs sequential consistency
    - Measure: Win rate delta, value MSE, policy agreement
    - Accept: MSE ≤0.01, win rate ≥99.5%
    - Reject: MSE >0.05 or win rate <99%

  Rationale: T012 is Phase 3, focus on Phase 4 validation first
  ```
- **Impact**: LOW - Deferred to post-T012 validation

**Q12: Should we validate search tree correctness?** ✅ **OPTIONAL DEBUG MODE**
- **Context**: Complex atomic operations (WU-UCT, batched backup) could introduce subtle bugs
- **RESOLUTION**:
  ```
  Decision: Implement optional validation mode

  Debug Mode (MCTS_VALIDATE=1):
    - Expensive checks: visit count sums, value consistency, tree invariants
    - Tree traversal validation: no orphaned nodes, correct parent/child links
    - Atomic operation validation: no underflow, no negative values
    - Performance: 10-100× slower, ONLY for debugging

  Production Mode (default):
    - No validation overhead
    - Rely on unit tests + integration tests for correctness

  Implementation: Low priority (T020 or later)
  ```
- **Impact**: LOW - Nice-to-have for debugging, not blocking

**Q13: What is acceptable policy drift?** ✅ **KEEP 95% TARGET**
- **Context**: 95% top-move agreement threshold seems strict
- **RESOLUTION**:
  ```
  Decision: Keep 95% as target, 90% as minimum acceptable

  Rationale:
    - AlphaZero agents with different seeds disagree 10-20%
    - But we're comparing SAME algorithm with optimizations
    - 95% ensures optimizations don't change search behavior
    - 90% minimum allows minor differences (rounding, ordering)

  Measurement Protocol:
    - 1000-position test set (Gomoku tactical positions)
    - Compare: Baseline (3,831 config) vs Optimized (8k+ target)
    - Metric: Top-move agreement % + win rate %
    - Accept: Agreement ≥95% AND win rate ≥99.5%
  ```
- **Impact**: LOW - Quality floor maintained

---

### 12.4 Summary of Resolutions

| Question | Status | Priority | Action Required |
|----------|--------|----------|----------------|
| Q1: Realistic targets | ✅ RESOLVED | **CRITICAL** | Update targets to 6-10k sims/sec |
| Q2: Baseline config | ✅ PROTOCOL | **CRITICAL** | Execute T017 (2-day investigation) |
| Q3: DLPack zero-copy | ✅ RESOLVED | MEDIUM | Correct terminology, document 0.24ms H2D |
| Q4: FP16 validation | ⚠️ **IMMEDIATE** | **CRITICAL** | Run validate_fp16_inference.py (1 hour) |
| Q5: Tensor creation overhead | ⚠️ **IMMEDIATE** | **CRITICAL** | Run profile_tensor_creation.py (1 hour) |
| Q6: Optimal thread count | ✅ PROTOCOL | HIGH | Execute T018 (grid search) |
| Q7: Batch/timeout trade-off | ✅ PROTOCOL | HIGH | Execute T019 (grid search) |
| Q8: Transposition tables | ✅ OUT OF SCOPE | LOW | Defer to Spec 005 |
| Q9: State cloning cost | ✅ PROTOCOL | LOW | Micro-benchmark (optional) |
| Q10: Result buffer scan | ✅ CLARIFIED | LOW | Verify T006b implementation |
| Q11: Value drift | ✅ DEFER | LOW | Post-T012 validation |
| Q12: Tree validation | ✅ OPTIONAL | LOW | Debug mode (nice-to-have) |
| Q13: Policy drift threshold | ✅ KEEP 95% | LOW | Use in T020 validation |

**IMMEDIATE ACTIONS** (Before continuing planning):
1. ⚠️ **Q4: Validate FP16** (1 hour) - Proves 1.5-2× GPU speedup or reveals bug
2. ⚠️ **Q5: Profile tensor creation** (1 hour) - Confirms <1ms or reveals critical bug

**CRITICAL PATH** (For planning):
1. ✅ **Q1: Update targets to 6-10k** (realistic, hardware-grounded)
2. ✅ **Q2: Execute T017** (find baseline or establish new one)
3. ✅ **Q6/Q7: Execute T018/T019** (optimize threads, batch, timeout)

---

## 13. Success Metrics Summary (REVISED - Hardware-Grounded)

**NOTE**: Targets updated per Q1 resolution based on review.txt analysis and RTX 3060 Ti hardware limits.

### 13.1 Must-Have (Phase 4 Completion)

| Metric | Baseline | Current | Target (Realistic) | Validation |
|--------|----------|---------|-------------------|-----------|
| **Throughput** | 3,831 sims/sec | 2,147 sims/sec | **≥8,000 sims/sec** | Benchmark suite pass |
| **GPU Utilization** | ~32% | 11.2% | **≥80%** | nvidia-smi during search |
| **Thread Efficiency (8T)** | Unknown | Unknown (4T @ 62%) | **≥70%** | Scaling benchmark |
| **Search Quality** | Baseline | Baseline | **≥99.5% win rate** | Head-to-head matches |

**Rationale**: GPU @ FP16 caps at 8,000-10,000 states/sec. With 20% coordination overhead, 8,000 sims/sec is realistic target (2.1× baseline, 3.7× current).

### 13.2 Should-Have (Quality Goals)

| Metric | Target | Validation |
|--------|--------|-----------|
| **Coordination Overhead** | ≤25% of time (was 67%) | Profiling breakdown |
| **Average Batch Size** | ≥48 (75% of 64) | Instrumentation counter |
| **Collision Rate** | ≤5% | Collision metrics |
| **Memory Footprint** | <1.3GB | RSS monitoring |
| **Thread Idle Time** | ≤20% (was 60%) | Thread profiling |

### 13.3 Nice-to-Have (Stretch Goals)

| Metric | Target | Validation |
|--------|--------|-----------|
| **Throughput (Stretch)** | **≥10,000 sims/sec** | Benchmark suite |
| **GPU Utilization (Stretch)** | ≥85% | nvidia-smi |
| **Thread Efficiency (12T)** | ≥65% | Scaling benchmark |

### 13.4 Aspirational (Requires Architectural Changes)

| Metric | Target | Changes Required |
|--------|--------|-----------------|
| **Throughput (Aspirational)** | ≥15,000-25,000 sims/sec | Model pruning (5M→2M params), batch 128+, multi-GPU |
| **Throughput (Original Target)** | ≥30,000 sims/sec | Fundamental redesign: root parallelization, transposition tables, TensorRT |

**Note**: 25k-30k sims/sec requires changes explicitly excluded by CONSTITUTION.md (TensorRT, model simplification, multi-GPU). Current scope targets **8-10k sims/sec** as realistic on Ryzen 5900X + RTX 3060 Ti with Python PyTorch.

---

## 14. Approval & Sign-Off

This specification is **approved and active** as of 2025-10-13.

**Stakeholders**:
- **Product Owner**: cosmosapjw-quantum (user)
- **Implementation Lead**: Claude Code (AI agent)
- **Quality Assurance**: Automated benchmark suite (`pytest -m performance`)

**Review Cycle**: After Phase 4 completion or if throughput < 50% of target

**Change Control**: All specification changes require:
1. Justification (profiling data or failed experiments)
2. Impact analysis (expected throughput delta)
3. Re-execution of `/speckit.plan` and `/speckit.tasks`

**Acceptance Criteria**: Phase 4 completion requires:
- [ ] All Must-Have metrics achieved
- [ ] All benchmark suites pass
- [ ] No performance regressions vs baseline
- [ ] Documentation complete (architecture guide, tuning guide, benchmark report)

---

**END OF SPECIFICATION**
