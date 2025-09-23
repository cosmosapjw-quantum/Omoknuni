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
- [x] **T020**: Gomoku game implementation with enhanced 36-plane tensor representation
- [x] **T021**: Chess game implementation with enhanced 30-plane tensor representation
- [x] **T022**: Go game implementation with enhanced 25-plane tensor representation
- [x] **T023**: Game adapter interface with unified polymorphic dispatch across all games
- [x] **T024**: Python bindings for games with pybind11 and numpy array compatibility
- [x] **T025**: Game rule unit tests with comprehensive verification for all games
- [x] **T026**: Contract test for training API with comprehensive coverage
- [x] **T027**: Asynchronous search coordinator with thread pool management
- [x] **T028**: Self-play game generator with comprehensive testing and validation

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
- **Python Bindings**: pybind11 integration with numpy array compatibility for neural network feature extraction
- **High-Performance Interop**: 250k+ tensor extractions/second, C-contiguous memory layout for zero-copy operations
- **Enhanced Tensor Representations**: Advanced feature planes for superior positional understanding
  - **Gomoku**: 36 planes with threat detection, run-length analysis, and rule variations
  - **Chess**: 30 planes with castling rights, en passant, and 8-pair move history
  - **Go**: 25 planes with proper move history separation and capture patterns
- **Self-Play Training**: Complete game generation with temperature scheduling and Dirichlet noise
  - **Temperature Control**: Configurable exploration→exploitation transitions
  - **Game Variations**: Support for Renju/Omok, Chess960, Chinese/Japanese/Korean Go rules
  - **Bias Detection**: Statistical analysis to ensure fair move distributions
  - **Policy Health**: Entropy monitoring and MCTS convergence validation
- **Experience Replay**: Memory-mapped buffer with Parquet storage and intelligent LRU caching
  - **High Performance**: 14.8K examples/sec addition, 643 samples/sec balanced sampling
  - **Memory Efficient**: Configurable buffer size (default 1M examples) with automatic cleanup
  - **Persistent Storage**: Parquet columnar format with thread-safe concurrent access
  - **Smart Caching**: LRU cache for frequently accessed examples (default 512MB)
  - **Balanced Sampling**: Exact game type ratios with temporal uniformity to prevent bias
  - **Training Iterator**: 136K samples/sec continuous batch generation with shuffle buffering
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

# Test experience buffer with memory-mapped storage
python scripts/validate_experience_buffer.py

# Test advanced experience sampling with balanced distribution
python scripts/validate_experience_sampling.py

# Test game adapter interface
python -m pytest tests/unit/test_game_adapter_interface.py -v

# Test inference integration pipeline
python -m pytest tests/integration/test_inference_integration.py -v

# Test Python bindings for games
python -m pytest tests/unit/test_python_bindings.py -v

# Run Python bindings demonstration
python examples/python_bindings_demo.py

# Test comprehensive game rules across all games
python -m pytest tests/unit/test_game_rules.py -v

# Test self-play game generation
python -m pytest tests/unit/test_self_play.py -v

# Test comprehensive self-play analysis
python -m pytest tests/integration/test_self_play_comprehensive.py -v

# Test terminal detection and game variations
python -m pytest tests/integration/test_terminal_detection_variations.py -v

# Run comprehensive self-play testing (all games and variations)
python scripts/test_self_play_comprehensive.py --quick-test --games 5 --output results/

# Run full comprehensive test with visualizations
python scripts/test_self_play_comprehensive.py --games 20 --output results/full_test
```

### Self-Play Training & Testing

The engine includes comprehensive self-play testing to ensure training data quality:

```bash
# Quick self-play validation across all games
python scripts/test_self_play_comprehensive.py --quick-test

# Full comprehensive analysis with move bias detection
python scripts/test_self_play_comprehensive.py --games 50 --output results/comprehensive

# Analyze existing results
python scripts/test_self_play_comprehensive.py --analyze-only results/analysis_results.json
```

**Self-Play Features:**
- **Move Bias Analysis**: Statistical tests to detect spatial bias, corner/edge preferences
- **Policy Entropy Monitoring**: Tracks exploration→exploitation balance throughout games
- **Game Variation Testing**: Validates Renju/Omok, Chess960, and Go rule variations
- **Terminal Detection**: Comprehensive validation of win/draw/timeout conditions
- **Health Metrics**: MCTS convergence quality, temperature scheduling effectiveness
- **Visualization**: Automated generation of bias analysis and entropy pattern plots
```

### Project Structure

```
├── src/                    # Python orchestration layer
│   ├── core/              # MCTS search coordination
│   ├── games/             # Game implementations
│   ├── neural/            # Neural networks, GPU inference, micro-batching & CPU fallback
│   ├── training/          # Self-play & training pipeline (T028 comprehensive testing)
│   ├── telemetry/         # Performance monitoring
│   └── utils/             # Shared utilities
├── cpp_extensions/        # Performance-critical C++ code
│   ├── mcts/             # Core MCTS tree operations
│   ├── games/            # Game rule implementations, unified interface & Python bindings
│   └── utils/            # Memory management & vectorization
├── tests/                # Comprehensive test suite
│   ├── contract/         # API contract validation
│   ├── integration/      # End-to-end system tests (including comprehensive self-play)
│   ├── unit/            # Component unit tests (including Python bindings & self-play)
│   └── performance/     # Benchmarking & regression tests
├── scripts/              # Testing and validation scripts
│   └── test_self_play_comprehensive.py  # Full self-play analysis with visualizations
└── examples/             # Usage demonstrations and tutorials
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
- **Self-Play Quality**: Comprehensive testing validates training data integrity
  - Move bias detection with statistical significance testing
  - Policy entropy monitoring (exploration→exploitation balance)
  - Terminal detection accuracy across all game variations
  - Temperature scheduling effectiveness validation
- **Tensor representations**: Enhanced feature planes for stronger tactical play
  - Gomoku: 36 planes (threat detection, run-length analysis, rule variations)
  - Chess: 30 planes (castling, en passant, 8-pair move history)
  - Go: 25 planes (proper move history separation, capture patterns)

## Development Philosophy

- **Simplicity**: Write straightforward, readable code
- **Performance**: CPU/GPU optimized without sacrificing maintainability
- **Testability**: Comprehensive test coverage with contract-driven development
- **Spec-Driven**: All changes must reflect specification updates

## License

[License details to be added]