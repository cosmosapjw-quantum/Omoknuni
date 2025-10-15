# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## GOLDEN RULE: Always follow Spec-Driven Development (SDD)

**Always maintain synchronization between code and specifications:**
- Code changes MUST reflect updates in active spec directories (currently `/specs/004-mcts-throughput-recovery/`)
- Completed specs: `/specs/001-goal-create-spec/` (foundation), `/specs/002-cpp-simulation-runner/` (7× speedup), `/specs/003-async-inference-batching/` (full optimization)
- API contracts in spec `contracts/` directories define interfaces
- Data models in spec `data-model.md` files define structures
- This CLAUDE.md must stay current with actual implementations

## Project Overview

This is a high-performance AlphaZero-style reinforcement learning engine targeting board games (Gomoku, Chess, Go) on consumer hardware. The system targets 8,000 simulations/second (realistic, hardware-grounded) through a hybrid CPU/GPU architecture:

- **CPU**: Shared-tree MCTS with 2-4 threads using Structure-of-Arrays memory layout (optimal efficiency)
- **GPU**: Asynchronous micro-batched neural network inference (batch size 64, 0.5-1.0ms timeout)
- **Target Hardware**: AMD Ryzen 9 5900X (12C/24T, dual-CCD) + NVIDIA RTX 3060 Ti (8GB VRAM, Ampere)
- **Current Performance**: 2,147 sims/sec (REGRESSION from 3,831 baseline, cause: state cloning waste + coordination overhead)
- **Performance Goals**: 80% GPU utilization, <1GB tree memory, 200-300 games/hour self-play
- **Bottleneck** (CORRECTED 2025-10-14): CPU coordination (67.2% of time) vs GPU inference (32.8%) - NOT GPU!
- **Spec 004 Status**: Profiling complete ✅, Ready for Priority #1 fix (state pooling) 🔴
- **Validation Results (2025-10-14 - COMPREHENSIVE PROFILING COMPLETE)**:
  - ✅ **FP16**: Working correctly (1.72× speedup, T008f complete)
  - ✅ **OpenMP**: Working correctly (8.64ms→1.57ms @ 12 threads) BUT NOT THE BOTTLENECK
  - ✅ **MCTS Throughput**: SAME regardless of OMP threads (1,543 vs 1,529 sims/sec)
  - ❌ **WRONG ANALYSIS**: Feature extraction NOT the primary bottleneck (batching amortizes cost)
  - ✅ **CORRECT ANALYSIS**: State cloning (2-3× per sim) + thread contention (60% idle) are real issues
- **Real Bottlenecks** (per review.txt + validation):
  1. **State cloning waste**: 2-3× clones per simulation (review.txt lines 37-54) - HIGHEST PRIORITY
  2. **Thread contention**: 60% idle time, global mutex, spin-waits (review.txt lines 71-136)
  3. **Thread affinity**: Suboptimal CCD pinning (review.txt lines 244-250)
  4. **Python interface**: Batch callback overhead (review.txt lines 258-307)
- **Expected Performance After Fixes**: 7,300-8,500 sims/sec (91-106% of 8k target)
- **Full Analysis**: [COMPREHENSIVE_CORRECTED_PLAN_2025-10-14.md](COMPREHENSIVE_CORRECTED_PLAN_2025-10-14.md)

## Architecture Overview

The codebase follows a hybrid approach with Python orchestration and C++/pybind11 for performance-critical operations:

### Core Components
- **C++ Simulation Runner** (`cpp_extensions/mcts/simulation_runner.cpp`): Zero-GIL MCTS simulation pipeline (select → expand → backup) achieving 1,744+ sims/sec (7× Python baseline), targeting 8k sims/sec with full optimization
- **MCTS Engine** (`cpp_extensions/mcts/`): C++17 implementation with WU-UCT virtual loss, busy-edge masking, AVX2-vectorized PUCT selection (3.6-5.2x speedup), epoch-based tree clearing (1M× speedup)
- **Async Inference Queue** (`cpp_extensions/mcts/async_inference_queue.cpp`): Lock-free MPMC ring buffer (4096 entries) with condition variables (T006c complete), eliminating polling waste
- **DLPack Tensor Bridge** (`cpp_extensions/dlpack/`): Zero-copy tensor sharing (C++ ↔ PyTorch), pinned memory pools (4KB-4MB size classes), PyCapsule wrappers for torch.from_dlpack()
- **Thread-Local Arenas** (`cpp_extensions/mcts/thread_local_arena.cpp`): 4096-node block allocation, 99.93% fast-path (thread-local), 0.07% slow-path (mutex fallback)
- **Game Adapters** (`cpp_extensions/games/`): Uniform interface for Gomoku/Chess/Go with in-place move application, zero-copy feature extraction to DLPack tensors
- **Python Bindings** (`cpp_extensions/mcts/python_bindings.cpp`): pybind11 module exposing SimulationRunner, PyInferenceCallback bridge, DLPack tensors, move storage API
- **Neural Network** (`src/neural/`): ResNet with Squeeze-Excitation blocks (20 blocks, 256 channels), FP16 mixed precision implemented and validated (T008f ✅ T-VALID-1 PASS, 1.72× speedup)
- **Training Pipeline** (`src/training/`): Experience replay buffer using memory-mapped files, AdamW optimizer with cosine scheduling

### Memory Design
- Structure-of-Arrays layout: 27 bytes per MCTS node (target achieved: <64 bytes)
- **Thread-Local Arenas** (Spec 004 T009a-f): 4096-node blocks, 99.93% fast-path allocation, lock-free in common case
- Pre-allocated node pools with O(1) allocation (330M allocations/second baseline, enhanced with arenas)
- Free list for efficient node reuse without malloc/free in hot paths
- **Epoch-Based Clearing** (Spec 004 T001b): O(1) tree clear via epoch increment (25ns) vs memset (25ms), 1M× speedup
- Index-based references with 64-byte alignment for SIMD operations on Ryzen CCDs
- Memory efficiency: 10M nodes = 270MB tree + 1MB queue + 1MB DLPack buffers = 272MB total (well under 1GB target)

### Game-Specific Tensor Representations

**Enhanced Feature Extraction** (see `specs/001-goal-create-spec/data-model.md` for complete specifications):

- **Gomoku (15×15)**: 36 planes with tactical analysis
  - Stone positions, 8-pair move history, player indicator, rule variations
  - Enhanced: threat detection (immediate five, four, open three)
  - Enhanced: run-length analysis in 4 directions

- **Chess (8×8)**: 30 planes with complete game state
  - 12 piece types (6 × 2 colors), castling rights, en passant
  - 8-pair move history per player (destination squares)
  - Chess960 support with proper castling encoding

- **Go (19×19)**: 25 planes with proper history separation
  - Stone positions, ko position, 8-pair move history per player
  - Enhanced: capture patterns (group liberties 1/2/3/4+)
  - Legal move indicator, player turn

**Note**: These enhanced representations (36/30/25 planes) provide superior positional understanding compared to basic AlphaZero features (7/12/17 planes).

## Development Commands

### Build System
```bash
# Setup development environment
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Build C++ extensions with hardware optimizations
export CFLAGS="-O3 -march=znver3 -fopenmp"
export CXXFLAGS="-O3 -march=znver3 -fopenmp"
python -m pip install -e . --config-settings build-dir=build

# Rebuild after C++ changes
pip install -e . --force-reinstall --no-deps
```

### Docker Deployment (Recommended)
```bash
# Build and run development environment
./scripts/docker/build.sh -t development
./scripts/docker/run.sh dev

# Production deployment
./scripts/docker/build.sh -t runtime
docker-compose up -d runtime

# Training environment
docker-compose up -d training

# Run benchmarks in Docker
docker-compose run --rm benchmark
```

### Testing
```bash
# Contract tests (must fail initially, then pass after implementation)
python -m pytest tests/contract/ -v

# Unit tests with thread safety verification
python -m pytest tests/unit/ -v

# Integration tests (end-to-end pipelines)
python -m pytest tests/integration/ -v

# Performance benchmarks and regression detection
python -m pytest tests/performance/ -v

# Run specific benchmark categories
python -m pytest -m "performance" -v
python -m pytest -m "benchmark" -v

# Run benchmarks directly with detailed output
python tests/performance/test_benchmarks.py

# Memory leak detection (1-hour soak test)
python -m pytest tests/soak/ -v

# C++ simulation runner tests
python -m pytest tests/contract/test_simulation_runner_api.py -v        # API contracts
python -m pytest tests/integration/test_cpp_vs_python_equivalence.py -v # Equivalence
python -m pytest tests/integration/test_gil_release.py -v -s             # GIL release
python -m pytest tests/performance/test_simulation_runner_performance.py -v # Throughput

# Test Python bindings specifically
python -m pytest tests/unit/test_python_bindings.py -v

# Run Python bindings demo
python examples/python_bindings_demo.py
```

### Performance Validation
```bash
# Basic MCTS performance test
python scripts/test_mcts.py --game gomoku --simulations 1000 --threads 8

# GPU inference benchmarking
python scripts/test_inference.py --model models/gomoku_random.pth --batch-sizes 32,64

# Memory stability test
python scripts/soak_test.py --duration 3600 --game gomoku

# Validate neural network inference optimizations
python scripts/validate_micro_batching.py
python scripts/validate_mixed_precision.py
python scripts/validate_pinned_memory.py
python scripts/validate_cpu_fallback.py

# Thread count optimization
python scripts/tune_threads.py --game gomoku --quick-test
python scripts/tune_threads.py --game gomoku --simulations 800 --iterations 50

# Virtual loss magnitude optimization
python scripts/tune_virtual_loss.py --game gomoku --quick-test
python scripts/tune_virtual_loss.py --game gomoku --simulations 800 --iterations 50 --threads 8

# Batch size optimization
python scripts/tune_batch_size.py --game gomoku --quick-test
python scripts/tune_batch_size.py --game gomoku --iterations 100 --max-vram 85

# Inference timeout optimization
python scripts/tune_timeout.py --game gomoku --quick-test
python scripts/tune_timeout.py --game gomoku --min-timeout 1 --max-timeout 10 --iterations 100
```

## Project Structure

```
src/
├── core/                # MCTS search coordination and threading
├── games/               # Python bindings for game implementations
├── neural/              # Neural network models and GPU inference
├── training/            # Self-play generation and model training
├── telemetry/           # Performance monitoring and metrics
└── utils/               # Shared utilities and comprehensive configuration system

config/                  # YAML configuration files (default/dev/prod)
docs/                    # Documentation (operations runbook, API reference)

cpp_extensions/          # Performance-critical C++ code
├── mcts/                # Core MCTS tree operations
├── games/               # Game rule implementations
└── utils/               # Memory management and vectorization

tests/
├── contract/            # API contract validation (TDD)
├── integration/         # End-to-end system tests
├── unit/                # Component unit tests
└── performance/         # Benchmarking and regression tests

specs/                   # Specification-driven development
├── 001-goal-create-spec/      # ✅ Foundation (complete)
├── 002-cpp-simulation-runner/ # ✅ 7× speedup (complete)
├── 003-async-inference-batching/ # ✅ Full optimization (complete)
└── 004-mcts-throughput-recovery/  # 🔄 85% complete (active)
    ├── spec.md          # Functional requirements and progress tracking
    ├── plan.md          # Technical implementation plan with critical optimizations
    ├── tasks.md         # Detailed task breakdown (Phase 1 ✅, Phase 2 🟡 85%)
    ├── data-model.md    # Memory layout and data structures
    ├── research.md      # Architecture decisions and rationale
    ├── quickstart.md    # Build and validation procedures
    ├── README.md        # Spec 004 overview and status
    └── contracts/       # API interface definitions
```

## Key Performance Constraints

- **GIL Release**: All hot loops in C++/Cython with `nogil` blocks
- **Memory Efficiency**: Structure-of-Arrays layout, 32-64 bytes per node
- **Thread Safety**: Atomic operations for visit counts, virtual loss coordination
- **GPU Optimization**: Dynamic batching, mixed precision, pinned memory buffers
- **Hardware Targeting**: AVX2 vectorization, dual-CCD thread affinity on Ryzen 5900X

## Specification-Driven Development

This project uses the `.specify/` framework for feature development:

- `/specify` - Create new feature specifications from natural language
- `/plan` - Transform specifications into implementation plans
- `/tasks` - Generate detailed task breakdowns for execution

Implementation follows Test-Driven Development with contract tests that must fail initially before implementation.

## Critical Performance Targets

- 8,000 simulations/second including neural network inference (realistic, hardware-grounded)
- 80% GPU utilization sustained during search operations (revised 2025-10-13)
- <1GB memory footprint for 10M node MCTS trees (✅ achieved: 270MB for 10M nodes)
- Thread-safe operation with 8-12 parallel workers
- No memory leaks over 24-hour continuous runs
- Superhuman Gomoku performance within 48 hours of training

## Development Philosophy

- **Simplicity**: Write simple, straightforward code
- **Readability**: Make code easy to understand
- **Performance**: Consider performance without sacrificing readability
- **Maintainability**: Write code that's easy to update
- **Testability**: Ensure code is testable
- **Reusability**: Create reusable components and functions
- **Less Code = Less Debt**: Minimize code footprint

## Coding Best Practices

- **Early Returns**: Use to avoid nested conditions
- **Descriptive Names**: Use clear variable/function names (prefix handlers with "handle")
- **Constants Over Functions**: Use constants where possible
- **DRY Code**: Don't repeat yourself
- **Functional Style**: Prefer functional, immutable approaches when not verbose
- **Minimal Changes**: Only modify code related to the task at hand
- **Function Ordering**: Define composing functions before their components
- **TODO Comments**: Mark issues in existing code with "TODO:" prefix
- **Simplicity**: Prioritize simplicity and readability over clever solutions
- **Build Iteratively** Start with minimal functionality and verify it works before adding complexity
- **Run Tests**: Test your code frequently with realistic inputs and validate outputs
- **Build Test Environments**: Create testing environments for components that are difficult to validate directly
- **Functional Code**: Use functional and stateless approaches where they improve clarity
- **Clean logic**: Keep core logic clean and push implementation details to the edges
- **File Organization**: Balance file organization with simplicity - use an appropriate number of files for the project scale

## Pull Requests & Git Commit

- Create a detailed message of what changed. Focus on the high level description of
  the problem it tries to solve, and how it is solved. Don't go into the specifics of the
  code unless it adds clarity.

- Always add `cosmosapjw-quantum` as reviewer.

- NEVER ever mention a `co-authored-by` or similar aspects. In particular, never
  mention the tool used to create the commit message or PR.

## Core Workflow: Research → Plan → Implement → Validate

**Start every feature with:** "Let me research the codebase and create a plan before implementing."

1. **Research** - Understand existing patterns and architecture
2. **Plan** - Propose approach and verify with you
3. **Implement** - Build with tests and error handling
4. **Validate** - ALWAYS run formatters, linters, and tests after implementation

## Code Organization

**Keep functions small and focused:**
- If you need comments to explain sections, split into functions
- Group related functionality into clear packages
- Prefer many small files over few large ones

## Architecture Principles

**This is always a feature branch:**
- Delete old code completely - no deprecation needed
- No versioned names (processV2, handleNew, ClientOld)
- No migration code unless explicitly requested
- No "removed code" comments - just delete it

**Prefer explicit over implicit:**
- Clear function names over clever abstractions
- Obvious data flow over hidden magic
- Direct dependencies over service locators

## Maximize Efficiency

**Parallel operations:** Run multiple searches, reads, and greps in single messages
**Multiple agents:** Split complex tasks - one for tests, one for implementation
**Batch similar work:** Group related file edits together

## Problem Solving

**When stuck:** Stop. The simple solution is usually correct.

**When uncertain:** "Let me ultrathink about this architecture."

**When choosing:** "I see approach A (simple) vs B (flexible). Which do you prefer?"

Your redirects prevent over-engineering. When uncertain about implementation, stop and ask for guidance.

## Implementation Guidelines

### Python Code (Orchestration Only)
- **Use Python only for coordination**: configuration, data loading, logging, high-level flow control
- **Never put Python in performance-critical loops**: Hot loops must be in C++/Cython
- **Pre-allocate everything**: Create numpy arrays/buffers once, reuse them
- **Async patterns for I/O**: Use queues and threading for GPU communication

### C++ Code (Performance Critical)
- **Structure of Arrays (SoA)**: Split objects into separate arrays per field for cache efficiency
- **64-byte alignment**: Align hot data structures to cache lines using `alignas(64)`
- **Index-based references**: Use `int32_t` indices instead of pointers between nodes
- **Pre-allocate pools**: Reuse memory, avoid `malloc` in hot paths
- **Compile flags**: Always use `-O3 -march=znver3 -fopenmp` for Ryzen 5900X

### Cython Code (Hot Loops)
- **Always use nogil blocks**: Any function called frequently must release GIL
- **Type everything**: Every variable needs `cdef` declaration
- **Disable bounds checking**: Use `@cython.boundscheck(False)`
- **Stack allocation**: Use `float[362] scores` not heap allocation

## MCTS Implementation Specifics

### Architecture (from specs)
- **One shared tree**: All threads work on same tree with atomics (NOT separate trees)
- **WU-UCT Virtual Loss** (Spec 004 T001): Visit-only virtual loss, no Q-value distortion, pure Q = W/N
- **Busy-Edge Masking** (Spec 004 T002): PUCT = -∞ for expanding nodes, prevents thread contention
- **Root Pre-Expansion** (Spec 004 T003): Root pre-expanded with N=1, eliminates N-1 thread idle problem
- **Virtual loss = 1.0**: Default magnitude tunable via scripts/tune_virtual_loss.py
- **Value sign flipping**: Value MUST flip sign at each tree level during backup
- **Dynamic batching**: Batch by count (≥32) OR timeout (≤3ms), whichever comes first
- **Lock-Free Queue** (Spec 004 T006/T006b/T006c): MPMC ring buffer (4096 entries), turn-based synchronization, ✅ **condition variables implemented and validated**
- **Zero-Copy Pipeline** (Spec 004 T007a-g): DLPack tensors, pinned memory, torch.from_dlpack(), complete C++ → PyTorch zero-copy path

### C++ Simulation Runner (Spec 002)
- **Zero GIL Re-entry**: Full simulation (select → expand → backup) in C++ without GIL
- **Move Storage in Tree**: `uint16_t* moves_` array (20MB for 10M nodes vs 1000MB Python dict)
- **Thread-Safe Allocation**: Mutex-protected node pools with atomic counters (TSan clean)
- **PyInferenceCallback Bridge**: Re-acquire GIL only for neural network inference
- **Current Performance**: 1,744 sims/sec (7× Python baseline, 8k target with full optimization)
- **Integration**: `src/core/mcts.py` uses `mcts_py.SimulationRunner` as primary execution path

See `docs/mcts_cpp_runner.md` for detailed architecture and `docs/performance/cpp_runner_results.md` for validation results.

### Performance Targets (from specs, updated 2025-10-13)
| Metric | Target (Revised) | Achieved | Status |
|--------|------------------|----------|--------|
| Simulations/sec | 8,000 (realistic) | 2,147 (26.8% of target) | 🔴 OpenMP fix required |
| GPU utilization | 80% | ~68% (batch 64) | ⚠️ Tensor creation bottleneck |
| FP16 mixed precision | 1.5-2× speedup | ✅ 1.72× (T-VALID-1) | ✅ Complete |
| Tensor creation | <1.0ms | ❌ 7.5ms (T-VALID-2) | 🔴 OpenMP missing |
| Average batch size | 32-64 | 45-85 (optimal: 64) | ✅ Complete |
| Batch timeout | ≤3ms | 0.5-1.0ms optimal | ✅ Complete |
| Thread efficiency | ≥70% @ 8 threads | 45% @ 4 threads | ⚠️ Coordination overhead |
| Tree memory | <1GB | ✅ 270MB (10M nodes) | ✅ Complete |
| Move storage | <50MB | ✅ 20MB (10M nodes) | ✅ Complete |
| Node footprint | <64 bytes | ✅ 27 bytes | ✅ Complete |
| Thread safety | TSan clean | ✅ 6 races fixed | ✅ Complete |

**Critical Finding (Validation 2025-10-13)**: Feature extraction loop at dlpack_bridge.cpp:431-434 NOT parallelized with OpenMP. This 7.5ms overhead caps throughput at ~1,675 states/sec, explaining regression from 3,831 to 2,147 sims/sec. GPU hardware limit (RTX 3060 Ti @ FP16) caps realistic target at 8,000-10,000 sims/sec.

**Spec 004 Progress (2025-10-13)**:
- **Phase 1 Complete** (✅): WU-UCT virtual loss, epoch clearing, busy-edge masking, root pre-expansion, thread affinity, collision metrics
- **Phase 2 Complete + Validated** (✅): Lock-free queue (T006/T006b/T006c ✅), DLPack bridge (T007a-g ✅), FP16 mixed precision (T008f ✅ T-VALID-1 PASS), Python integration (T008a-b,e ✅), thread arenas (T009a-f ✅), pending map replacement (T010 ✅)
- **Phase 3 Partial** (🟡): Persistent coordinator (T011 ✅), batched results (T014 ✅), remaining optimizations deferred
- **Current Status** (🔴): Validation complete, OpenMP parallelization missing (critical blocker)
- **Critical Next Steps**: Fix OpenMP → re-validate T-VALID-2 → T017 baseline investigation + T016 benchmarking
- **Realistic Path**: Fix OpenMP → 2,000-3,000 sims/sec (est) → T017/T016 analysis → parameter tuning → 6,000-10,000 sims/sec (target range)

See `specs/004-mcts-throughput-recovery/spec.md` for detailed status and `review.pdf` for comprehensive analysis.

## Common Pitfalls

1. **GIL in hot loops**: Use Cython with `nogil` or C++/pybind11
2. **Allocating in search**: ✅ Pre-allocated node pools with free list (330M allocs/sec)
3. **Missing illegal move masking**: Always mask then renormalize policy
4. **Wrong value signs**: Value must flip with each tree level
5. **No GPU warmup**: First inference is 10x slower without warmup
6. **Synchronous GPU calls**: Use async queuing pattern instead

## Troubleshooting

- **Low GPU util** → increase batch size, enable mixed precision, add warmup
- **Thread contention** → reduce to 8-12 threads, check atomic operations
- **Memory leaks** → verify pool reuse, run with sanitizers in debug
- **Value drift** → check sign flipping in backup, calibrate value head

## Glossary

**SoA (Structure of Arrays)**: Each field (N, W, P, VL) stored contiguously for cache locality
**Virtual Loss (VL)**: Temporary penalty during selection to prevent thread collisions
**PUCT**: Upper confidence bound formula: `Q + c_puct * P * sqrt(N_parent) / (1 + N_child)`
**Dynamic Batching**: Batch by count OR timeout, whichever comes first
**Mixed Precision**: fp16 computation with fp32 fallback for numerical stability
- always never mind how much the given this task takes and ensure all the functionality is validated and ensure the test passes with no omission or skip. always think very hard and deeply. memorize this also in claude memory: do NOT consider time limit specified in documentation. you job is to FULLY achieve the goal with NO simplification.