# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## GOLDEN RULE: Always follow Spec-Driven Development (SDD)

**Always maintain synchronization between code and specifications:**
- Code changes MUST reflect updates in `/specs/001-goal-create-spec/`
- API contracts in `/specs/001-goal-create-spec/contracts/` define the interface
- Data models in `/specs/001-goal-create-spec/data-model.md` define structures
- This CLAUDE.md must stay current with actual implementations

## Project Overview

This is a high-performance AlphaZero-style reinforcement learning engine targeting board games (Gomoku, Chess, Go) on consumer hardware. The system achieves 30-40k simulations/second through a hybrid CPU/GPU architecture:

- **CPU**: Shared-tree MCTS with 8-12 threads using Structure-of-Arrays memory layout
- **GPU**: Asynchronous micro-batched neural network inference (32-64 positions, ≤3ms timeout)
- **Target Hardware**: AMD Ryzen 5900X + NVIDIA RTX 3060 Ti (8GB VRAM)
- **Performance Goals**: 80-92% GPU utilization, <1GB tree memory, 200-300 games/hour self-play

## Architecture Overview

The codebase follows a hybrid approach with Python orchestration and C++/pybind11 for performance-critical operations:

### Core Components
- **MCTS Engine** (`cpp_extensions/mcts/`): C++17 implementation with atomic operations, virtual loss coordination, vectorized PUCT selection
- **Game Adapters** (`cpp_extensions/games/`): Uniform interface for Gomoku/Chess/Go with in-place move application and feature extraction
- **Neural Network** (`src/neural/`): ResNet with Squeeze-Excitation blocks (20 blocks, 256 channels), mixed precision fp16
- **Training Pipeline** (`src/training/`): Experience replay buffer using memory-mapped files, AdamW optimizer with cosine scheduling

### Memory Design
- Structure-of-Arrays layout: 27 bytes per MCTS node (target achieved: <64 bytes)
- Pre-allocated node pools with O(1) allocation (330M allocations/second)
- Free list for efficient node reuse without malloc/free in hot paths
- Index-based references with 64-byte alignment for SIMD operations on Ryzen CCDs
- Memory efficiency: 10M nodes = 270MB total (well under 1GB target)

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

# Memory leak detection (1-hour soak test)
python -m pytest tests/soak/ -v
```

### Performance Validation
```bash
# Basic MCTS performance test
python scripts/test_mcts.py --game gomoku --simulations 1000 --threads 8

# GPU inference benchmarking
python scripts/test_inference.py --model models/gomoku_random.pth --batch-sizes 32,64

# Memory stability test
python scripts/soak_test.py --duration 3600 --game gomoku
```

## Project Structure

```
src/
├── core/                # MCTS search coordination and threading
├── games/               # Python bindings for game implementations
├── neural/              # Neural network models and GPU inference
├── training/            # Self-play generation and model training
├── telemetry/           # Performance monitoring and metrics
└── utils/               # Shared utilities and configuration

cpp_extensions/          # Performance-critical C++ code
├── mcts/                # Core MCTS tree operations
├── games/               # Game rule implementations
└── utils/               # Memory management and vectorization

tests/
├── contract/            # API contract validation (TDD)
├── integration/         # End-to-end system tests
├── unit/                # Component unit tests
└── performance/         # Benchmarking and regression tests

specs/001-goal-create-spec/  # Implementation documentation
├── spec.md              # Functional requirements
├── plan.md              # Technical implementation plan
├── tasks.md             # Detailed task breakdown
├── data-model.md        # Memory layout and data structures
├── research.md          # Architecture decisions and rationale
├── quickstart.md        # Build and validation procedures
└── contracts/           # API interface definitions
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

- 30,000+ simulations/second including neural network inference
- 80-92% GPU utilization sustained during search operations
- <1GB memory footprint for 10M node MCTS trees
- Thread-safe operation with 12 parallel workers
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
- **Virtual loss = 1.0**: Default value to prevent thread collisions
- **Value sign flipping**: Value MUST flip sign at each tree level during backup
- **Dynamic batching**: Batch by count (≥32) OR timeout (≤3ms), whichever comes first

### Performance Targets (from specs)
| Metric | Target | Notes |
|--------|--------|-------|
| Simulations/sec | 30-40k | Including neural network inference |
| GPU utilization | 80-92% | Realistic target, not 95%+ |
| Average batch size | 32-64 | For RTX 3060 Ti optimization |
| Tree memory | <1GB | ✅ 270MB achieved for 10M nodes |
| Games/hour | 200-300 | Self-play generation rate |

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