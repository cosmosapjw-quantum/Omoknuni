# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## GOLDEN RULE: Always follow Spec-Driven-Development(SDD) principle.

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
- Structure-of-Arrays layout: 32-64 bytes per MCTS node across separate aligned arrays
- Pre-allocated node pools with index-based references (no pointers)
- 64-byte alignment for SIMD operations on Ryzen CCDs

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
- **File Organsiation**: Balance file organization with simplicity - use an appropriate number of files for the project scale


## Pull Requests

- Create a detailed message of what changed. Focus on the high level description of
  the problem it tries to solve, and how it is solved. Don't go into the specifics of the
  code unless it adds clarity.

- Always add `ArthurClune` as reviewer.

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


# AlphaZero‑Style Engine — Practical Playbook

*A concise, battle‑tested guide for Python/C++/Cython development and for building an AlphaZero‑style board‑game AI on a Ryzen 9 5900X + RTX 3060 Ti stack.*

---

## Python Playbook (orchestrators & glue)

**Role:** Use Python for *coordination only*—threads, queues, I/O, and GPU inference. Push hot loops into C++/Cython with the GIL released.

### Design Rules

* Keep Python out of tight loops; everything that may run >10⁶ times belongs in native code.
* Side‑effects explicit; pass immutable data (or indices/handles) across layers.
* Async batch the neural net: **count OR timeout** policy (e.g., batch≥32 *or* ≤2 ms).

### Minimal Project Skeleton

```text
your_pkg/
  pyproject.toml        # ruff + mypy + pytest; hatchling or setuptools
  your_pkg/__init__.py
  your_pkg/config.py    # dataclasses for run config
  your_pkg/logging.py   # structured logging
  your_pkg/queues.py    # request/result queues
  your_pkg/gpu_worker.py
  your_pkg/orchestrator.py
tests/
  test_gpu_worker.py
  test_orchestrator.py
```

### Production Defaults (Python 3.12)

* Linters/tests: `pip install ruff mypy pytest`
* Logging: INFO in dev, WARN in prod; JSON logs for long jobs.
* Torch inference: `torch.set_num_threads(1)`, `torch.no_grad()`, AMP autocast; enable TF32 on Ampere.
* Micro‑batching: warm up 10 passes to stabilize kernels and autotuners.

### Reusable Patterns

```python
# queues.py
import queue
NUM_THREADS = 12
INFER_Q_MAX = 1024
in_q  = queue.Queue(maxsize=INFER_Q_MAX)
out_q = [queue.Queue() for _ in range(NUM_THREADS)]
```

```python
# gpu_worker.py (sketch)
import time, queue, torch, torch.nn.functional as F
class GPUWorker:
    def __init__(self, model, device="cuda:0", max_bs=64, timeout_ms=2):
        self.m = model.eval().to(device)
        self.dev = device
        self.max_bs = max_bs
        self.dt = timeout_ms/1000
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    def loop(self, in_q, out_qs):
        while True:
            reqs, feats = [], []
            deadline = time.perf_counter() + self.dt
            while len(reqs) < self.max_bs and (t := deadline - time.perf_counter()) > 0:
                try:
                    r = in_q.get(timeout=t)
                    reqs.append(r); feats.append(r.features)
                except queue.Empty:
                    break
            if not reqs:
                continue
            x = torch.stack(feats).to(self.dev, non_blocking=True).half()
            with torch.cuda.amp.autocast(), torch.no_grad():
                p_logits, v = self.m(x)
                p = F.softmax(p_logits, 1)
                v = torch.tanh(v).squeeze(-1)
            for r, pi, val in zip(reqs, p, v):
                out_qs[r.thread_id].put((r.leaf_node_id,
                                         pi.detach().cpu().numpy(),
                                         float(val),
                                         r.path))
```

### Guardrails

* Never block CPU search threads on GPU; they enqueue and post‑process only.
* Python owns *coordination*, C++/Cython owns *search lifecycle*.
* Unit tests: illegal‑move masking, value sign (player perspective), virtual‑loss, terminal handling.

---

## ++ Playbook (engines & hot paths)

**Role:** Implement the MCTS engine, state transitions, and memory‑tight node pools.

### Design Rules

* **SoA layout**, 64‑B alignment. Index nodes; avoid pointer chasing between nodes.
* **No allocations** in hot paths—pre‑allocate pools and reuse.
* Shared‑tree parallel with **atomics + virtual loss** (start at 1.0).
* Build: Debug = ASan/UBSan/O0–O1; Release = `-O3 -march=znver3 -fopenmp -DNDEBUG`.

### Minimal Structures

```cpp
// mcts_tree.hpp
#include <cstdint>
#include <atomic>
struct MCTSTree {
  alignas(64) float* N;   // visit counts
  alignas(64) float* W;   // total values
  alignas(64) float* P;   // priors
  alignas(64) float* VL;  // virtual losses
  int32_t* parent;
  int32_t* first_child;
  uint16_t* n_children;
  uint8_t* flags; // bits: expanded|terminal|player
  size_t num_nodes=1, max_nodes;
  // preallocate buffers in ctor; memset to 0
};
```

### PyBind11 Gateway (release the GIL)

```cpp
// bindings.cpp
#include <pybind11/pybind11.h>
namespace py=pybind11;
py::array_t<float> search(GameState& s, int sims, float cpuct, int threads){
  py::gil_scoped_release release;
  MCTSTree T(s, cpuct);
  T.search_parallel(sims, threads);
  auto visits = T.root_visits();
  return py::array_t<float>(visits.size(), visits.data());
}
PYBIND11_MODULE(mcts_cpp, m){ m.def("search", &search); }
```

### CMake Flags

```cmake
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=znver3 -fopenmp -fvisibility=hidden")
# Debug: -O0 -g -fsanitize=address,undefined
```

### Guardrails

* After parallel runs: validate invariants (sum of child visits == parent visits).
* Aim for 8–12 CPU threads on the 5900X for best contention/throughput balance.
* Keep node footprint \~32–64 B; use perf to monitor L1/L2 miss rates.

---

## Cython Playbook (lightweight hot‑loops)

**Role:** When PyBind11 feels heavy but Python is too slow—tight loops with `nogil`.

### Design Rules

* Put hot loops in `cdef` functions with `nogil`.
* Use typed memoryviews or raw pointers; disable boundscheck/wraparound.
* Use OpenMP or compiler atomics for VL/N/W updates.

### File Set

```text
mcts_core.pyx
mcts_core.pxd    # cdef structs/signatures shared across modules
pyproject.toml   # build via setuptools/cythonize with OpenMP
```

### Selection/Backup Template

```cython
# mcts_core.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
from libc.math cimport sqrt
cdef inline int select_child(float* N, float* W, float* P, float* VL,
                             int first, int n_children, float parent_N, float cpuct) nogil:
    cdef int best=-1, i
    cdef float best_s=-1e30, s, n, q, u, psq = sqrt(parent_N if parent_N>1 else 1.0)
    for i in range(n_children):
        n = N[first+i]
        q = (W[first+i]/n) if n>0 else 0.0
        q -= VL[first+i]/(1.0+n)
        u = cpuct * P[first+i] * psq / (1.0 + n)
        s = q + u
        if s>best_s: best_s=s; best=first+i
    return best
```

### Build Snippet

```toml
# pyproject.toml (snippet)
[build-system]
requires = ["setuptools", "wheel", "cython"]
build-backend = "setuptools.build_meta"
[tool.setuptools.extension."mcts_core"]
sources = ["mcts_core.pyx"]
extra_compile_args = ["-O3","-fopenmp"]
extra_link_args    = ["-fopenmp"]
```

### Guardrails

* Ensure `nogil` covers the entire hot region (profile!).
* Sanity‑test atomics: GCC/Clang built‑ins or wrap via small C++ helpers.
* CI builds both with and without OpenMP to maintain portability.

---

## AlphaZero Build Blueprint

### Architecture (practical & scalable)

```
CPU threads (8–12)  -->  Shared‑tree MCTS (C++/Cython, SoA, VL)
          |                   |
          | enqueue           | gather leaf features
          v                   v
                 GPU Worker (Py, AMP, micro‑batch 32–64, 1–3 ms timeout)
                          |  pi, v
                          v
                     Backup to tree (flip v each ply)
```

### Throughput Targets (RTX 3060 Ti + R9 5900X)

* 30–40k sims/s with NN in the loop, depending on game features.
* 80–92% GPU util with AMP + dynamic batching.
* Batch size \~32–64; memory footprint <1 GB for \~10–20 M nodes.

### Minimal Bring‑Up Checklist

1. **Game API (C++)**: `is_terminal()`, `get_terminal_value()`, `get_legal_moves(mask)`, `apply_move_inplace(a)`, `extract_features(float*)`.
2. **MCTS**:

   * Selection: vectorized PUCT within node; mask illegals; **flip value on backup**.
   * Expansion: allocate from pool; set priors; mark expanded.
   * Backup: `N+=1; W+=v; v=-v` up the path; **remove virtual loss** on exit.
3. **GPU Worker**:

   * AMP + `torch.no_grad()`, `cudnn.benchmark=True`, TF32 enabled.
   * **Batch by count OR time**; warm‑up 10 iterations.
4. **Self‑Play**:

   * Root Dirichlet noise (α tuned per game), temperature schedule (τ=1.0 early plies).
   * Symmetry augmentation (rot/reflection for grid games).
5. **Training**:

   * Loss = `KL(policy)` + `λ·MSE(value)`; grad clip; cosine LR with warm restarts.
   * Shuffle across games; track value calibration and policy entropy.
6. **Production Hygiene**:

   * Per‑move wall‑time caps; deterministic seeds for eval; frequent checkpoints.
   * Metrics (sims/s, GPU util, batch mean/var); leak checks & invariants.

### Pitfalls that Kill Strength

* GIL held in hot loops; Python objects stored inside nodes.
* Allocating during expansion; synchronous GPU calls from search threads.
* Wrong value sign on backup; not masking illegals; forgetting `torch.no_grad()`.
* Oversized virtual loss (start at **1.0**); no GPU warm‑up; starving the batcher.

### Safe Hyperparameter Defaults

* `cpuct ≈ 1.25`
* `virtual_loss = 1.0`
* CPU threads = **8–12** (R9 5900X)
* GPU batch target **32–64**, timeout **1–3 ms**
* Mixed precision **on**, TF32 **on**

---

## efinition of Ready (DoR)

A module is *ready* when all are true:

* **Measurable**: has a micro‑benchmark; prints sims/s or it/s.
* **Deterministic**: fixed seed reproduces identical outcomes.
* **Memory‑safe**: pool reuse proven; no growth after 10k iterations.
* **Thread‑safe**: 10× stress test; tree invariants hold post‑run.
* **Interfaced**: C‑level POD boundaries; Python sees arrays/indices only.
* **Profiled**: top‑3 hotspots listed; no allocations in hotspots.

---

## Quick Recipes & Commands

### Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel
pip install torch torchvision torchaudio  # choose CUDA build matching drivers
pip install ruff mypy pytest
```

### Build C++/PyBind11

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -G Ninja ..
ninja
pytest -q
```

### Build Cython Extension

```bash
pip install cython
python -m pip install -e .  # uses pyproject.toml with cythonize
```

### Quick Benchmark Harness (Python)

```python
import time
from your_pkg import orchestrator

start = time.perf_counter()
res = orchestrator.run_once(num_sims=50000)
elapsed = time.perf_counter() - start
print({"sims": 50000, "sec": elapsed, "sims_per_sec": 50000/elapsed})
```

---

## Troubleshooting Checklist

* **Low GPU util** → increase micro‑batch cap; shorten timeout slightly; enable AMP & TF32; warm‑up runs.
* **Thread contention** → reduce threads to 8–12; ensure O(1) contention updates (atomics); widen SoA cache lines.
* **Value drift** → check perspective flips on backup; calibrate value head (isotonic/temperature).
* **Exploding memory** → verify pool reuse; guard against duplicate expansions; run leak sanitizer in Debug.
* **Starved batches** → deploy *count OR time* policy; push more parallel rollouts.

---

## Defaults & Glossary

**Hardware assumptions:** Ryzen 9 5900X (12C/24T), RTX 3060 Ti (8 GB), 64 GB RAM.

**Torch defaults:** `torch.no_grad()`, AMP autocast, `cudnn.benchmark=True`, TF32 on Ampere, `torch.set_num_threads(1)`.

**SoA (Structure of Arrays):** Each field (N, W, P, VL, …) stored contiguously to maximize cache locality and SIMD potential.

**Virtual Loss (VL):** Temporary penalty applied during selection to discourage other threads from choosing the same path; removed on exit.

**PUCT:** Upper confidence bound with prior; typical form `Q + c_puct * P * sqrt(N_parent) / (1 + N_child)`.

**Determinism:** Fix seeds at PyTorch, NumPy, C++ RNG, and game RNG; ensure no nondeterministic kernels in critical eval paths.