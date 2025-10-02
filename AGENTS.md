# Repository Guidelines

## Project Structure & Modules
- `src/`: runtime packages for MCTS (`core`), inference (`neural`), training, telemetry, utilities, and Python game shims backed by the `alphazero_py` bridge.
- `cpp_extensions/`: SoA MCTS kernels, chess/go/gomoku engines, and utilities built via the root `CMakeLists.txt`, emitting artifacts into `build/cpp_extensions` for pybind; keep `specs/002-cpp-simulation-runner` aligned.
- `tests/`: pytest suites (`unit`, `integration`, `contract`, `performance`, `soak`) plus gtest targets defined in `tests/unit/CMakeLists.txt`; fixtures and helpers live in `tests/conftest.py`.
- `config`, `specs/contracts`, `docs`, `scripts`, `results`, and `evaluation_results` hold configs, contracts, playbooks, automation, and experiment outputs—update them with code changes.

## Build & Tooling
```bash
python3.12 -m venv venv --prompt omoknuni
source venv/bin/activate
pip install -r requirements.txt
python -m pip install -e .[dev] --config-settings build-dir=build
```
- Run `python -m pytest` (filter with paths/markers) plus `cmake --build build --target run_unit_tests`; add `--cov=src --cov-report=html` when coverage matters.
- For native work, rebuild with `python scripts/build_with_sanitizers.py --sanitizer asan` (or `--all`) and run `scripts/validate_*` / `scripts/tune_*.py` before touching inference, batching, or self-play.
- Use `docker-compose run --rm dev python -m pytest` or `benchmark` to mirror CI, and reference the HOWTO guides for API, operations, OOM recovery, and sanitizers.

## Coding Standards & Contracts
- Python: 4-space indents, `black` 88-column formatting, `isort` (`profile=black`), `flake8`, strict `mypy`; keep public docstrings and reuse the structured errors in `src/utils/errors.py`.
- Naming stays `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE` for constants; maintain type hints.
- C++ keeps brace-on-next-line formatting, Structure-of-Arrays layouts, AVX2 flags, and descriptive namespaces; avoid new macros unless justified.
- Update `specs/contracts/*` with implementation changes—the runtime imports these contracts for validation.

## Testing Expectations
- Respect pytest markers from `pytest.ini` and keep the C++ gtests green (`cmake --build build --target run_unit_tests`).
- Guard performance budgets with `tests/performance/test_benchmarks.py` and `tests/performance/test_simulation_runner_performance.py`; refresh `results/benchmarks` only with recorded deltas.
- Exercise system suites (`tests/integration/test_training_pipeline.py`, `test_self_play_*`, `test_full_system*`) under `use_cpp_runner=True`, falling back to the legacy loop for debugging; use Docker when needed.
- Run endurance checks (`tests/soak/`, `python scripts/check_memory_leaks.py --all`) and confirm sanitizer builds after touching `cpp_extensions/`.

## Ops & PR Practice
- Central configs (`config/default.yaml`, `development.yaml`, `production.yaml`) feed `src/utils/config.py`; document `ALPHAZERO_*` overrides or migrations in each PR.
- Keep telemetry consistent: `src/telemetry/logger.py` and `metrics.py` feed Prometheus dashboards—mirror new fields in `docs/operations.md` and alerting rules.
- Docs in `docs/operations.md`, `docs/CI_PIPELINE.md`, and `docs/performance/*` set expectations; refresh them whenever behavior or thresholds shift.
- Use task-tagged commits, describe problem/solution/validation, attach metrics when performance can move, and call out skipped work.
