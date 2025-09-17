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

### Current Architecture

- **Hybrid CPU/GPU**: Shared-tree MCTS on CPU with asynchronous GPU neural network inference
- **Performance Target**: 30-40k simulations/second with 80-92% GPU utilization
- **Target Hardware**: AMD Ryzen 5900X + NVIDIA RTX 3060 Ti (8GB VRAM)
- **Memory Efficient**: Structure-of-Arrays layout, <1GB tree memory for 10M nodes
- **Telemetry**: Prometheus-compatible metrics with GPU monitoring and structured logging
- **Device Management**: RTX 3060 Ti optimizations with automatic batch size estimation

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
```

### Project Structure

```
├── src/                    # Python orchestration layer
│   ├── core/              # MCTS search coordination
│   ├── games/             # Game implementations
│   ├── neural/            # Neural networks & GPU inference
│   ├── training/          # Self-play & training pipeline
│   ├── telemetry/         # Performance monitoring
│   └── utils/             # Shared utilities
├── cpp_extensions/        # Performance-critical C++ code
│   ├── mcts/             # Core MCTS tree operations
│   ├── games/            # Game rule implementations
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
- **GPU utilization**: 80-92% sustained during search
- **Tree memory**: <1GB for typical search configurations
- **Training speed**: 200-300 self-play games per hour
- **Games supported**: Gomoku, Chess (including Chess960), Go (9x9 to 19x19)

## Development Philosophy

- **Simplicity**: Write straightforward, readable code
- **Performance**: CPU/GPU optimized without sacrificing maintainability
- **Testability**: Comprehensive test coverage with contract-driven development
- **Spec-Driven**: All changes must reflect specification updates

## License

[License details to be added]