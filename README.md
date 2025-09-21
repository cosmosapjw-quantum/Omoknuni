# Omoknuni - High-Performance AlphaZero Engine

A production-ready AlphaZero-style reinforcement learning engine for board games (Gomoku, Chess, Go) optimized for consumer hardware.

## Project Status

🚧 **Under Development** - Currently implementing core architecture

### Completed
- [x] **T001**: Project structure and build system setup
- [x] **T002**: CI/CD pipeline with GitHub Actions
- [x] **T003**: Basic telemetry framework with Prometheus metrics
- [x] **T004**: GPU warmup and device detection system
- [x] **T005**: MCTS API contract tests for Test-Driven Development
- [x] **T006**: Structure-of-Arrays memory layout (27 bytes/node, <1GB for 50M nodes)
- [x] **T007**: Node pool pre-allocation for O(1) MCTS tree operations (330M allocations/sec)
- [x] **T008**: Vectorized PUCT selection with AVX2 SIMD optimizations (3.6-5.2x speedup)
- [x] **T009**: Thread-safe virtual loss mechanism for MCTS coordination
- [x] **T010**: Value backup mechanism with proper sign flipping per tree level
- [x] **T011**: Single-threaded MCTS integration test for complete search cycle
- [x] **T012**: Contract test for neural network inference API
- [x] **T013**: ResNet architecture with Squeeze-Excitation attention (20 blocks, 256 channels)
- [x] **T014**: GPU inference worker with queue-based threading and dynamic batching
- [x] **T015**: Dynamic micro-batching with count-based (≥32) OR timeout-based (≤3ms) optimization
- [x] **T016**: Mixed precision inference with fp16 computation and automatic fallback mechanisms
- [x] **T017**: Pinned memory optimization for efficient GPU data transfers (H2D/D2H optimized)
- [x] **T018**: CPU fallback mechanism for robust inference reliability
- [x] **T019**: Inference integration test for full pipeline validation
- [x] **T023**: Game adapter interface with unified polymorphic dispatch across all games

### Current Architecture

- **Hybrid CPU/GPU**: Shared-tree MCTS on CPU with asynchronous GPU neural network inference
- **Performance Target**: 30-40k simulations/second with 80-92% GPU utilization
- **Target Hardware**: AMD Ryzen 5900X + NVIDIA RTX 3060 Ti (8GB VRAM)
- **Memory Efficient**: Structure-of-Arrays layout, <1GB tree memory for 10M nodes (27 bytes/node achieved)
- **Advanced Batching**: Dynamic micro-batching with count-based (≥32) OR timeout-based (≤3ms) optimization
- **GPU Monitoring**: Real-time utilization tracking with nvidia-ml-py and adaptive batch sizing
- **Vectorized Operations**: AVX2-optimized PUCT selection with 3.6-5.2x performance improvement
- **Thread Safety**: Virtual loss coordination and atomic operations for 8-12 parallel workers
- **Memory Optimization**: Pinned CUDA memory buffers for faster H2D/D2H transfers with automatic fallback
- **Robust Inference**: CPU fallback mechanism with automatic GPU failure detection and seamless switching
- **Unified Game Interface**: Polymorphic dispatch enabling MCTS to work with Chess, Go, and Gomoku seamlessly
- **Game Type Detection**: Automatic detection from notation (FEN for Chess, SGF for Go, coordinate for Gomoku)
- **Standard Format Support**: Export/import in established formats (PGN, SGF, custom notation)
- **Telemetry**: Prometheus-compatible metrics with comprehensive performance monitoring

## Quick Start

### Prerequisites
- Python 3.12+
- CMake 3.18+
- CUDA 12.x (for GPU acceleration)
- C++17 compatible compiler with OpenMP support

### Setup Development Environment

```bash
# Clone and setup
git clone <repository>
cd omoknuni

# Create virtual environment
python3.12 -m venv venv --prompt omoknuni
source venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Build C++ extensions (when available)
export CFLAGS="-O3 -march=znver3 -fopenmp"
export CXXFLAGS="-O3 -march=znver3 -fopenmp"
python -m pip install -e . --config-settings build-dir=build
```

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest tests/unit/          # Unit tests
python -m pytest tests/contract/      # API contract tests
python -m pytest tests/integration/   # Integration tests
python -m pytest tests/performance/   # Performance benchmarks

# Run with coverage
python -m pytest --cov=src --cov-report=html

# Test micro-batching performance
python scripts/validate_micro_batching.py

# Test mixed precision inference
python scripts/validate_mixed_precision.py

# Test pinned memory optimization
python scripts/validate_pinned_memory.py

# Test CPU fallback mechanism
python scripts/validate_cpu_fallback.py

# Test game adapter interface
python -m pytest tests/unit/test_game_adapter_interface.py -v

# Test inference integration pipeline
python -m pytest tests/integration/test_inference_integration.py -v
```

### Project Structure

```
├── src/                    # Python orchestration layer
│   ├── core/              # MCTS search coordination
│   ├── games/             # Game implementations
│   ├── neural/            # Neural networks, GPU inference, micro-batching & CPU fallback
│   ├── training/          # Self-play & training pipeline
│   ├── telemetry/         # Performance monitoring
│   └── utils/             # Shared utilities
├── cpp_extensions/        # Performance-critical C++ code
│   ├── mcts/             # Core MCTS tree operations
│   ├── games/            # Game rule implementations & unified interface
│   └── utils/            # Memory management & vectorization
└── tests/                # Comprehensive test suite
    ├── contract/         # API contract validation
    ├── integration/      # End-to-end system tests
    ├── unit/            # Component unit tests
    └── performance/     # Benchmarking & regression tests
```

## Specification

This project follows [Spec-Driven Development](specs/001-goal-create-spec/). See:
- [Feature Specification](specs/001-goal-create-spec/spec.md) - Requirements and objectives
- [Implementation Plan](specs/001-goal-create-spec/plan.md) - Technical architecture
- [Task Breakdown](specs/001-goal-create-spec/tasks.md) - Detailed implementation tasks

## Performance Targets

- **Simulations/sec**: 30,000+ including neural network inference
- **GPU utilization**: 80-92% sustained during search (adaptive micro-batching implemented)
- **Batch efficiency**: ≥32 positions OR ≤3ms timeout (T015 specification met)
- **Tree memory**: <1GB for typical search configurations (27 bytes/node achieved)
- **Node allocation**: O(1) operations with 330M allocations/second
- **PUCT selection**: 3.6-5.2x speedup with AVX2 vectorization
- **Memory transfers**: Optimized H2D/D2H transfers using pinned CUDA memory buffers
- **Reliability**: Automatic CPU fallback with seamless inference continuation on GPU failures
- **Training speed**: 200-300 self-play games per hour
- **Games supported**: Gomoku, Chess (including Chess960), Go (9x9 to 19x19)

## Development Philosophy

- **Simplicity**: Write straightforward, readable code
- **Performance**: CPU/GPU optimized without sacrificing maintainability
- **Testability**: Comprehensive test coverage with contract-driven development
- **Spec-Driven**: All changes must reflect specification updates

## License

[License details to be added]