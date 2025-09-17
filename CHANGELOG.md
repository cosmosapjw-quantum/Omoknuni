# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Initial project structure with Python and C++ extension support
- Build system configuration with scikit-build-core, CMake, and optimization flags for Ryzen 5900X
- Directory structure for MCTS engine, game implementations, neural networks, training pipeline, and telemetry
- Dependencies configuration for PyTorch 2.x, pybind11, Cython, and development tools
- Migrated existing C++ game logic files to proper cpp_extensions structure
- Complete CMake configuration for games and utils modules
- .gitignore file to exclude venv and build artifacts
- **T002**: Complete CI/CD pipeline with GitHub Actions
- Multi-stage testing pipeline (lint, unit tests, integration tests, GPU tests)
- Performance regression detection with benchmark comparison
- Build artifact caching for faster CI runs
- Sample test suites for validation
- **T003**: Basic telemetry framework implementation
- Prometheus-compatible metrics collection with comprehensive performance tracking
- GPU utilization monitoring using nvidia-ml-py with automatic fallback
- Memory usage tracking for both system RAM and GPU VRAM
- Structured logging framework with JSON output and contextual information
- Performance metrics for simulations/second, inference batching, and resource utilization

### Changed
- Moved existing game logic from `/games` to `/cpp_extensions/games/`
- Moved core game utilities to `/cpp_extensions/utils/`
- Updated CMakeLists.txt files to build with actual C++ source files

### Fixed

### Removed
- Original `/games` directory (files moved to cpp_extensions)

---

## [0.1.0] - 2025-09-16

### Added
- **T001**: Project structure and build system setup
  - Created modular directory structure: `src/{core,games,neural,training,telemetry,utils}/`
  - Added C++ extensions structure: `cpp_extensions/{mcts,games,utils}/`
  - Set up comprehensive test structure: `tests/{contract,integration,unit,performance}/`
  - Configured build system with performance optimizations (`-O3 -march=znver3 -fopenmp`)
  - Created pyproject.toml with scikit-build-core for seamless Python/C++ integration
  - Added requirements.txt with all necessary dependencies for AI/ML workloads