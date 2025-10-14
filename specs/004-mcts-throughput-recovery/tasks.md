# Task Breakdown: MCTS Throughput Recovery & Multi-Actor Self-Play

**Version**: 1.0
**Date**: 2025-10-14
**Status**: READY
**Authority**: Implements plan.md v1.0 under spec.md v2.0

---

## Task Dependency Graph

```
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 0: Foundation (Telemetry & Benchmarking)                      │
├─────────────────────────────────────────────────────────────────────┤
│ T001 → T002 → T003                                                  │
└─────────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 1: CPU Pipeline Optimizations (Critical Path)                 │
├─────────────────────────────────────────────────────────────────────┤
│ T004 → T005 → T006 (OpenMP)                                         │
│ T007 → T008 → T009 (State Pooling)                                  │
│ T010 → T011 (Condition Variables)                                   │
│ T012 → T013 (Node Allocator)                                        │
└─────────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Validation & Baseline Investigation                        │
├─────────────────────────────────────────────────────────────────────┤
│ T014 → T015 (Comprehensive benchmarks)                              │
│ T016 (Baseline investigation)                                       │
│ T017 (Ablation studies)                                             │
└─────────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 3: NN-Eval Cache (Optional)                                   │
├─────────────────────────────────────────────────────────────────────┤
│ T018 → T019 → T020 → T021                                           │
└─────────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 4: Multi-Actor Self-Play                                      │
├─────────────────────────────────────────────────────────────────────┤
│ T022 → T023 → T024 → T025 → T026                                    │
└─────────────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 5: Final Validation & Documentation                           │
├─────────────────────────────────────────────────────────────────────┤
│ T027 → T028 → T029 → T030                                           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## PHASE 0: Foundation (Telemetry & Benchmarking)

### T001: Benchmark Harness Infrastructure

**Goal**: Create unified benchmark harness with CSV telemetry for reproducible performance measurement.

**Scope**: 1.0 day

**Dependencies**: None (Foundation task)

**Files to Create**:
1. `tests/performance/benchmark_harness.py`
2. `tests/performance/telemetry.py`
3. `tests/performance/fixtures.py`
4. `results/benchmarks/benchmark_history.csv`

**Files to Modify**:
- `tests/performance/conftest.py` (add fixtures)

**Implementation Details**:

```python
# tests/performance/benchmark_harness.py
class BenchmarkHarness:
    """Unified benchmark harness with telemetry."""

    def __init__(self, output_dir: str = "results/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_benchmark(self,
                      config: BenchmarkConfig,
                      iterations: int = 10) -> BenchmarkResult:
        """
        Run benchmark with specified configuration.

        Args:
            config: Benchmark configuration (game, threads, batch, etc.)
            iterations: Number of runs for statistical validation

        Returns:
            BenchmarkResult with mean, std, CV, telemetry
        """
        measurements = []

        for i in range(iterations):
            # Set seed for reproducibility
            np.random.seed(config.seed + i)

            # Run single benchmark
            telemetry = self._run_single_benchmark(config)
            measurements.append(telemetry)

        # Compute statistics
        result = self._compute_statistics(measurements)

        # Save to CSV
        self._save_results(config, result)

        return result

    def _run_single_benchmark(self, config: BenchmarkConfig) -> Telemetry:
        """Run single benchmark iteration with full telemetry."""
        # GPU warmup
        self._warmup_gpu(num_warmup=5)

        # Start telemetry collection
        telemetry = Telemetry()
        telemetry.start()

        # Run MCTS search
        mcts = self._create_mcts(config)
        start = time.perf_counter()
        mcts.search(root_state, config.num_simulations)
        elapsed = time.perf_counter() - start

        # Stop telemetry
        telemetry.stop()

        # Collect metrics
        telemetry.throughput = config.num_simulations / elapsed
        telemetry.gpu_util = self._measure_gpu_utilization()
        telemetry.avg_batch_size = mcts.get_avg_batch_size()
        # ... more metrics ...

        return telemetry
```

```python
# tests/performance/telemetry.py
@dataclass
class Telemetry:
    """Performance telemetry for single benchmark run."""

    # PRIMARY KPIs
    throughput: float = 0.0  # simulations/sec
    gpu_util_percent: float = 0.0  # 0-100
    cpu_util_percent: float = 0.0  # 0-100
    avg_batch_size: float = 0.0  # positions per batch

    # CPU BREAKDOWN
    feature_extraction_ms: float = 0.0  # per batch-64
    selection_time_ms: float = 0.0  # total
    expansion_time_ms: float = 0.0  # total
    backup_time_ms: float = 0.0  # total

    # THREAD METRICS
    thread_efficiency: float = 0.0  # vs linear scaling
    thread_idle_percent: float = 0.0  # 0-100

    # MEMORY
    memory_rss_mb: float = 0.0  # Resident set size
    memory_peak_mb: float = 0.0  # Peak usage

    # CACHE (if enabled)
    cache_hit_rate: float = 0.0  # 0-1
    cache_size_entries: int = 0

    # METADATA
    timestamp: str = ""
    git_commit: str = ""
    config: Dict = field(default_factory=dict)
```

**Acceptance Tests**:

1. **Unit Test**: `tests/unit/test_benchmark_harness.py`
   ```python
   def test_harness_creates_session_directory():
       harness = BenchmarkHarness()
       assert harness.output_dir.exists()

   def test_csv_output_format():
       harness = BenchmarkHarness()
       result = harness.run_benchmark(sample_config, iterations=3)
       csv_path = harness.output_dir / "benchmark_history.csv"
       assert csv_path.exists()
       df = pd.read_csv(csv_path)
       assert 'throughput' in df.columns
       assert 'gpu_util_percent' in df.columns
   ```

2. **Integration Test**: Run 3 iterations, verify CV < 10%
   ```bash
   pytest tests/performance/test_benchmark_harness.py -v
   ```

**Done Means**:
- ✅ CSV file created with correct schema
- ✅ Telemetry captures all KPIs from spec.md
- ✅ Reproducible results (same seed → same output ±2%)
- ✅ 3 iterations complete in <5 minutes

**Rollback Plan**:
- N/A (foundation task, no runtime changes)

---

### T002: OpenMP Verification Script

**Goal**: Create diagnostic script to verify OpenMP compilation and runtime configuration.

**Scope**: 0.5 day

**Dependencies**: T001

**Files to Create**:
1. `scripts/verify_openmp.py`
2. `cpp_extensions/mcts/openmp_diagnostics.cpp`

**Files to Modify**:
- `cpp_extensions/mcts/python_bindings.cpp` (add diagnostic functions)

**Implementation Details**:

```cpp
// cpp_extensions/mcts/openmp_diagnostics.cpp
#include <omp.h>
#include <iostream>
#include <vector>

namespace mcts {

struct OpenMPDiagnostics {
    bool compiled_with_openmp;
    int max_threads;
    int num_procs;
    bool nested_enabled;
    std::string schedule_type;
    std::vector<std::string> warnings;
};

OpenMPDiagnostics check_openmp_status() {
    OpenMPDiagnostics diag;

    #ifdef _OPENMP
        diag.compiled_with_openmp = true;
        diag.max_threads = omp_get_max_threads();
        diag.num_procs = omp_get_num_procs();
        diag.nested_enabled = omp_get_nested();
    #else
        diag.compiled_with_openmp = false;
        diag.warnings.push_back("CRITICAL: OpenMP not compiled!");
        return diag;
    #endif

    // Check environment configuration
    const char* num_threads_env = std::getenv("OMP_NUM_THREADS");
    if (num_threads_env == nullptr) {
        diag.warnings.push_back(
            "WARNING: OMP_NUM_THREADS not set (recommend 12 for Ryzen 5900X)"
        );
    }

    if (diag.nested_enabled) {
        diag.warnings.push_back(
            "WARNING: Nested parallelism enabled (conflicts with MCTS threads)"
        );
    }

    if (diag.max_threads < 12) {
        diag.warnings.push_back(
            "WARNING: max_threads < 12 (suboptimal for feature extraction)"
        );
    }

    return diag;
}

} // namespace mcts
```

```python
# scripts/verify_openmp.py
"""Verify OpenMP compilation and runtime configuration."""

import sys
import os
import subprocess
import mcts_py

def check_openmp_symbols():
    """Check if OpenMP symbols present in compiled library."""
    result = subprocess.run(
        ["nm", mcts_py.__file__],
        capture_output=True,
        text=True
    )
    openmp_symbols = [
        'GOMP_parallel',
        'omp_get_num_threads',
        'omp_get_max_threads',
    ]
    found = []
    for symbol in openmp_symbols:
        if symbol in result.stdout:
            found.append(symbol)

    return found, openmp_symbols


def check_runtime_config():
    """Check runtime environment configuration."""
    config = {
        'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'),
        'OMP_PROC_BIND': os.environ.get('OMP_PROC_BIND'),
        'OMP_PLACES': os.environ.get('OMP_PLACES'),
        'OMP_NESTED': os.environ.get('OMP_NESTED'),
    }
    return config


def main():
    print("="*60)
    print("OpenMP Verification Report")
    print("="*60)

    # Check compilation
    print("\n1. Compilation Check:")
    found, expected = check_openmp_symbols()
    print(f"   OpenMP symbols found: {len(found)}/{len(expected)}")
    if len(found) == len(expected):
        print("   ✅ PASS: OpenMP compiled correctly")
    else:
        print("   ❌ FAIL: OpenMP not compiled")
        print(f"   Missing: {set(expected) - set(found)}")
        sys.exit(1)

    # Check runtime config
    print("\n2. Runtime Configuration:")
    config = check_runtime_config()
    for key, value in config.items():
        status = "✅" if value else "❌"
        print(f"   {status} {key}: {value or 'NOT SET'}")

    # Check C++ diagnostics
    print("\n3. C++ Diagnostics:")
    diag = mcts_py.check_openmp_status()
    print(f"   Max threads: {diag['max_threads']}")
    print(f"   Num processors: {diag['num_procs']}")

    if diag['warnings']:
        print("\n4. Warnings:")
        for warning in diag['warnings']:
            print(f"   ⚠️  {warning}")

    # Recommendations
    print("\n5. Recommendations:")
    if config['OMP_NUM_THREADS'] != '12':
        print("   → Set OMP_NUM_THREADS=12")
    if config['OMP_PROC_BIND'] != 'close':
        print("   → Set OMP_PROC_BIND=close")
    if config['OMP_PLACES'] != 'cores':
        print("   → Set OMP_PLACES=cores")
    if config['OMP_NESTED'] != 'false':
        print("   → Set OMP_NESTED=false")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
```

**Acceptance Tests**:

1. **Script runs successfully**:
   ```bash
   python scripts/verify_openmp.py
   ```

2. **Detects missing OpenMP**:
   - Build without `-fopenmp` → script reports FAIL

3. **Detects misconfiguration**:
   - Unset `OMP_NUM_THREADS` → script warns

**Done Means**:
- ✅ Script reports OpenMP status (compiled: yes/no)
- ✅ Script checks environment variables
- ✅ Script provides actionable recommendations
- ✅ Exit code 0 if OK, 1 if critical issues

**Rollback Plan**:
- N/A (diagnostic tool, no runtime changes)

---

### T003: Feature Flag Infrastructure

**Goal**: Implement runtime feature flags for safe rollback of all optimizations.

**Scope**: 0.5 day

**Dependencies**: None

**Files to Create**:
1. `cpp_extensions/mcts/feature_flags.hpp`
2. `cpp_extensions/mcts/feature_flags.cpp`
3. `tests/unit/test_feature_flags.py`

**Files to Modify**:
- `cpp_extensions/mcts/python_bindings.cpp` (expose to Python)

**Implementation Details**:

```cpp
// cpp_extensions/mcts/feature_flags.hpp
#pragma once

#include <cstdlib>
#include <string>
#include <atomic>

namespace mcts {

/**
 * @brief Runtime feature flags for safe rollback
 *
 * All optimizations can be disabled via environment variables.
 * Thread-safe, read-only after initialization.
 */
class FeatureFlags {
public:
    /**
     * @brief Initialize feature flags from environment
     *
     * Call once at module load time.
     */
    static void initialize();

    /**
     * @brief Check if OpenMP parallelization is enabled
     *
     * Env: MCTS_OPENMP_ENABLED (default: true)
     */
    static bool is_openmp_enabled() {
        return openmp_enabled_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Check if state pooling is enabled
     *
     * Env: MCTS_STATE_POOLING_ENABLED (default: true)
     */
    static bool is_state_pooling_enabled() {
        return state_pooling_enabled_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Check if condition variables are enabled
     *
     * Env: MCTS_CV_ENABLED (default: true)
     */
    static bool are_condition_variables_enabled() {
        return cv_enabled_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Check if NN-eval cache is enabled
     *
     * Env: MCTS_CACHE_ENABLED (default: false)
     */
    static bool is_nn_cache_enabled() {
        return nn_cache_enabled_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Check if multi-actor mode is enabled
     *
     * Env: MCTS_MULTI_ACTOR_ENABLED (default: false)
     */
    static bool is_multi_actor_enabled() {
        return multi_actor_enabled_.load(std::memory_order_relaxed);
    }

    /**
     * @brief Get all flags as dict (for logging)
     */
    static std::map<std::string, bool> get_all_flags();

private:
    static std::atomic<bool> openmp_enabled_;
    static std::atomic<bool> state_pooling_enabled_;
    static std::atomic<bool> cv_enabled_;
    static std::atomic<bool> nn_cache_enabled_;
    static std::atomic<bool> multi_actor_enabled_;

    static bool get_bool_env(const char* name, bool default_value);
};

} // namespace mcts
```

**Acceptance Tests**:

```python
# tests/unit/test_feature_flags.py
import os
import mcts_py

def test_default_flags():
    """Verify default flag values."""
    flags = mcts_py.get_feature_flags()
    assert flags['openmp_enabled'] == True
    assert flags['state_pooling_enabled'] == True
    assert flags['cv_enabled'] == True
    assert flags['nn_cache_enabled'] == False  # Opt-in
    assert flags['multi_actor_enabled'] == False  # Opt-in

def test_environment_override():
    """Verify environment variables override defaults."""
    os.environ['MCTS_OPENMP_ENABLED'] = '0'
    mcts_py.reinitialize_feature_flags()

    flags = mcts_py.get_feature_flags()
    assert flags['openmp_enabled'] == False

    # Cleanup
    del os.environ['MCTS_OPENMP_ENABLED']
```

**Done Means**:
- ✅ All feature flags readable from Python
- ✅ Environment variables override defaults
- ✅ Thread-safe (atomic loads)
- ✅ Unit tests pass

**Rollback Plan**:
- Disable all flags via environment:
  ```bash
  export MCTS_OPENMP_ENABLED=0
  export MCTS_STATE_POOLING_ENABLED=0
  export MCTS_CV_ENABLED=0
  ```

---

## PHASE 1: CPU Pipeline Optimizations

### T004: OpenMP Compilation Verification & Fix

**Goal**: Ensure OpenMP is compiled into C++ extensions with correct flags.

**Scope**: 0.5 day

**Dependencies**: T002, T003

**Files to Modify**:
1. `CMakeLists.txt` (verify/fix OpenMP configuration)
2. `setup.py` (verify compiler flags)
3. `cpp_extensions/mcts/dlpack_bridge.cpp` (verify pragma present)

**Implementation Details**:

```cmake
# CMakeLists.txt
# Verify OpenMP configuration
find_package(OpenMP REQUIRED)

if(NOT OpenMP_CXX_FOUND)
    message(FATAL_ERROR "OpenMP not found! Install with: apt-get install libomp-dev")
endif()

# Add OpenMP flags to MCTS library
target_link_libraries(mcts_py PUBLIC OpenMP::OpenMP_CXX)
target_compile_options(mcts_py PRIVATE ${OpenMP_CXX_FLAGS})

# Explicitly add -fopenmp for safety
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(mcts_py PRIVATE -fopenmp)
endif()

# Print confirmation
message(STATUS "OpenMP CXX Flags: ${OpenMP_CXX_FLAGS}")
message(STATUS "OpenMP CXX Libraries: ${OpenMP_CXX_LIBRARIES}")
```

**Verification Steps**:

1. **Clean rebuild**:
   ```bash
   rm -rf build/
   pip install -e . --force-reinstall --no-deps --config-settings build-dir=build
   ```

2. **Check CMake output**:
   ```bash
   grep -i "openmp" build/CMakeCache.txt
   # Should show: OpenMP_CXX_FLAGS=-fopenmp
   ```

3. **Check compiled symbols**:
   ```bash
   python scripts/verify_openmp.py
   # Should report: ✅ PASS: OpenMP compiled correctly
   ```

**Acceptance Tests**:

1. **Build Test**:
   ```bash
   # Fresh build from scratch
   rm -rf build/
   pip install -e . --force-reinstall --no-deps
   python -c "import mcts_py; print(mcts_py.get_omp_max_threads())"
   # Should print non-zero value (e.g., 12)
   ```

2. **Symbol Test**:
   ```bash
   nm build/lib.*/mcts_py*.so | grep -i omp | head -5
   # Should show GOMP symbols
   ```

**Done Means**:
- ✅ `verify_openmp.py` reports compilation success
- ✅ `omp_get_max_threads()` returns >0
- ✅ CMake logs show OpenMP flags
- ✅ Library contains GOMP symbols

**Rollback Plan**:
- Set `MCTS_OPENMP_ENABLED=0` to disable at runtime (feature flag)
- Code already has `#pragma omp parallel for if(...)` guarding

---

### T005: OpenMP Runtime Configuration

**Goal**: Configure optimal OpenMP environment for Ryzen 5900X dual-CCD topology.

**Scope**: 0.5 day

**Dependencies**: T004

**Files to Create**:
1. `scripts/configure_openmp.sh`
2. `docs/openmp_tuning.md`

**Files to Modify**:
- All benchmark scripts (add environment setup)

**Implementation Details**:

```bash
# scripts/configure_openmp.sh
#!/bin/bash
# Configure OpenMP environment for Ryzen 5900X

# CRITICAL: Set thread count to physical cores
export OMP_NUM_THREADS=12

# Bind threads to nearby cores (minimize cross-CCD latency)
export OMP_PROC_BIND=close

# Use physical cores (not hyperthreads)
export OMP_PLACES=cores

# Disable nested parallelism (conflicts with MCTS threads)
export OMP_NESTED=false

# Wait policy: ACTIVE (spin-wait for low latency)
export OMP_WAIT_POLICY=ACTIVE

# Dynamic adjustment: OFF (predictable performance)
export OMP_DYNAMIC=false

# Verify configuration
echo "OpenMP Configuration:"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  OMP_PROC_BIND: $OMP_PROC_BIND"
echo "  OMP_PLACES: $OMP_PLACES"
echo "  OMP_NESTED: $OMP_NESTED"

# Confirm with Python
python3 -c "import mcts_py; print(f'Max threads: {mcts_py.get_omp_max_threads()}')"
```

**Modify Benchmark Scripts**:

```python
# Add to ALL benchmark scripts (benchmark_throughput.py, etc.)
def setup_openmp_environment():
    """Configure OpenMP before any C++ imports."""
    import os
    os.environ['OMP_NUM_THREADS'] = '12'
    os.environ['OMP_PROC_BIND'] = 'close'
    os.environ['OMP_PLACES'] = 'cores'
    os.environ['OMP_NESTED'] = 'false'
    os.environ['OMP_WAIT_POLICY'] = 'ACTIVE'
    os.environ['OMP_DYNAMIC'] = 'false'

# Call BEFORE importing mcts_py
if __name__ == '__main__':
    setup_openmp_environment()
    import mcts_py  # Now OpenMP configured correctly
    main()
```

**Acceptance Tests**:

1. **Environment Test**:
   ```bash
   source scripts/configure_openmp.sh
   python scripts/verify_openmp.py
   # Should report all ✅ checks
   ```

2. **Thread Count Test**:
   ```python
   import os
   os.environ['OMP_NUM_THREADS'] = '12'
   import mcts_py
   assert mcts_py.get_omp_max_threads() == 12
   ```

**Done Means**:
- ✅ `configure_openmp.sh` script created
- ✅ All benchmark scripts call `setup_openmp_environment()`
- ✅ `verify_openmp.py` reports correct configuration
- ✅ Documentation updated with tuning guide

**Rollback Plan**:
- Unset environment variables (returns to system defaults)

---

### T006: OpenMP Performance Validation

**Goal**: Validate OpenMP parallelization achieves target <1ms per batch-64.

**Scope**: 1.0 day

**Dependencies**: T004, T005

**Files to Create**:
1. `tests/performance/test_openmp_feature_extraction.py`
2. `scripts/profile_feature_extraction.py`

**Implementation Details**:

```python
# tests/performance/test_openmp_feature_extraction.py
"""Validate OpenMP parallelization performance."""

import pytest
import numpy as np
import time
from src.games.gomoku_state import GomokuState
from cpp_extensions.dlpack import DLPackTensorBridge

@pytest.mark.performance
def test_feature_extraction_speed():
    """Validate <1ms per batch-64 (TARGET from spec)."""
    bridge = DLPackTensorBridge()
    states = [GomokuState() for _ in range(64)]

    # Warmup (5 iterations to stabilize)
    for _ in range(5):
        bridge.create_batch_tensor(states)

    # Measure (10 iterations for statistics)
    times = []
    for _ in range(10):
        start = time.perf_counter()
        tensor = bridge.create_batch_tensor(states)
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    mean_time = np.mean(times)
    std_time = np.std(times)
    cv = std_time / mean_time

    print(f"\nFeature Extraction Performance:")
    print(f"  Mean: {mean_time:.2f} ms")
    print(f"  Std:  {std_time:.2f} ms")
    print(f"  CV:   {cv:.2%}")
    print(f"  Min:  {np.min(times):.2f} ms")
    print(f"  Max:  {np.max(times):.2f} ms")

    # ACCEPTANCE CRITERIA (from spec.md)
    assert mean_time < 1.0, \
        f"TOO SLOW: {mean_time:.2f}ms (target <1.0ms)"
    assert cv < 0.10, \
        f"HIGH VARIANCE: CV={cv:.2%} (target <10%)"

@pytest.mark.performance
def test_feature_extraction_parity():
    """Verify OpenMP parallel matches single-thread (bit-exact)."""
    import os

    bridge = DLPackTensorBridge()
    states = [GomokuState() for _ in range(64)]

    # Extract with OpenMP (12 threads)
    os.environ['OMP_NUM_THREADS'] = '12'
    tensor_parallel = bridge.create_batch_tensor(states)

    # Extract single-threaded
    os.environ['OMP_NUM_THREADS'] = '1'
    tensor_serial = bridge.create_batch_tensor(states)

    # Restore
    os.environ['OMP_NUM_THREADS'] = '12'

    # Bit-exact comparison
    np.testing.assert_array_equal(
        tensor_parallel,
        tensor_serial,
        err_msg="OpenMP parallel output differs from serial"
    )

@pytest.mark.performance
def test_openmp_speedup():
    """Measure actual speedup from OpenMP."""
    import os
    bridge = DLPackTensorBridge()
    states = [GomokuState() for _ in range(64)]

    def benchmark_with_threads(num_threads: int) -> float:
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        # Warmup
        for _ in range(5):
            bridge.create_batch_tensor(states)
        # Measure
        times = []
        for _ in range(10):
            start = time.perf_counter()
            bridge.create_batch_tensor(states)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        return np.mean(times)

    time_1_thread = benchmark_with_threads(1)
    time_12_threads = benchmark_with_threads(12)

    speedup = time_1_thread / time_12_threads

    print(f"\nOpenMP Speedup:")
    print(f"  1 thread:  {time_1_thread*1000:.2f} ms")
    print(f"  12 threads: {time_12_threads*1000:.2f} ms")
    print(f"  Speedup:    {speedup:.2f}×")

    # Expect at least 5× speedup (realistic with overhead)
    assert speedup >= 5.0, \
        f"POOR SPEEDUP: {speedup:.2f}× (expected ≥5×)"

    # Restore
    os.environ['OMP_NUM_THREADS'] = '12'
```

**Done Means**:
- ✅ Feature extraction <1ms per batch-64 (mean)
- ✅ CV < 10% (low variance)
- ✅ Bit-exact parity with serial version
- ✅ Speedup ≥5× over single-thread
- ✅ Test passes in CI

**Rollback Plan**:
- Feature flag: `MCTS_OPENMP_ENABLED=0`
- Performance test will fail, but code still works (slower)

---

### T007: Thread-Local State Pool Implementation

**Goal**: Implement thread-local state pooling to eliminate 2-3× clones per simulation.

**Scope**: 1.5 days

**Dependencies**: T003

**Files to Create**:
1. `cpp_extensions/mcts/state_pool.hpp`
2. `cpp_extensions/mcts/state_pool.cpp`
3. `tests/unit/test_state_pool.cpp`

**Files to Modify**:
- `cpp_extensions/mcts/python_bindings.cpp` (expose to Python for testing)

**Implementation Details**:

```cpp
// cpp_extensions/mcts/state_pool.hpp
#pragma once

#include "../utils/igamestate.h"
#include <vector>
#include <memory>
#include <cstddef>

namespace mcts {

/**
 * @brief Thread-local pool of reusable IGameState objects
 *
 * Eliminates heap allocations during simulation by reusing
 * pre-allocated state objects via copyFrom().
 *
 * Thread Safety: One pool per thread, no synchronization needed.
 */
class ThreadLocalStatePool {
public:
    /**
     * @brief Initialize pool with template state
     *
     * @param template_state Prototype for cloning
     * @param initial_capacity Pre-allocate this many states
     */
    explicit ThreadLocalStatePool(const IGameState& template_state,
                                    size_t initial_capacity = 16);

    ~ThreadLocalStatePool() = default;

    // Non-copyable
    ThreadLocalStatePool(const ThreadLocalStatePool&) = delete;
    ThreadLocalStatePool& operator=(const ThreadLocalStatePool&) = delete;

    /**
     * @brief Get working state for simulation (reused every sim)
     *
     * Caller should use copyFrom() to reset from root state.
     *
     * @return Reference to reusable working state
     */
    IGameState& get_working_state();

    /**
     * @brief Acquire state for pending expansion (ownership transfer)
     *
     * Returns state from pool if available, otherwise allocates new.
     *
     * @return unique_ptr to state (caller owns)
     */
    std::unique_ptr<IGameState> acquire_pending_state();

    /**
     * @brief Return state to pool (recycle after expansion)
     *
     * @param state State to return (ownership transferred to pool)
     */
    void release_pending_state(std::unique_ptr<IGameState> state);

    /**
     * @brief Get pool statistics
     */
    struct Stats {
        size_t working_state_reuses;   // copyFrom() calls
        size_t pending_acquisitions;   // acquire() calls
        size_t pool_hits;               // Reused from pool
        size_t pool_misses;             // Required allocation
        size_t current_pool_size;       // States in pool now
    };
    Stats get_stats() const;

    /**
     * @brief Reset statistics
     */
    void reset_stats();

private:
    std::unique_ptr<IGameState> working_state_;
    std::vector<std::unique_ptr<IGameState>> pending_pool_;
    Stats stats_;
};

} // namespace mcts
```

**Acceptance Tests**:

```cpp
// tests/unit/test_state_pool.cpp
#include "state_pool.hpp"
#include "gomoku_state.hpp"
#include <gtest/gtest.h>

TEST(StatePool, WorkingStateReuse) {
    GomokuState template_state(15);
    ThreadLocalStatePool pool(template_state, 4);

    // Get working state
    IGameState& state = pool.get_working_state();
    state.makeMove(0);

    // Reset from template
    state.copyFrom(template_state);

    // Verify reset
    EXPECT_EQ(state.getMoveHistory().size(), 0);

    // Statistics
    auto stats = pool.get_stats();
    EXPECT_EQ(stats.working_state_reuses, 1);
}

TEST(StatePool, PendingStateAcquireRelease) {
    GomokuState template_state(15);
    ThreadLocalStatePool pool(template_state, 4);

    // Acquire 10 states
    std::vector<std::unique_ptr<IGameState>> states;
    for (int i = 0; i < 10; ++i) {
        states.push_back(pool.acquire_pending_state());
    }

    // Statistics
    auto stats = pool.get_stats();
    EXPECT_EQ(stats.pending_acquisitions, 10);
    EXPECT_EQ(stats.pool_hits, 4);  // Initial capacity
    EXPECT_EQ(stats.pool_misses, 6);  // Allocated new

    // Release back to pool
    for (auto& state : states) {
        pool.release_pending_state(std::move(state));
    }

    // Pool should have 10 states now
    stats = pool.get_stats();
    EXPECT_EQ(stats.current_pool_size, 10);
}

TEST(StatePool, ZeroAllocationSteadyState) {
    GomokuState template_state(15);
    ThreadLocalStatePool pool(template_state, 16);

    // Pre-fill pool
    std::vector<std::unique_ptr<IGameState>> states;
    for (int i = 0; i < 16; ++i) {
        states.push_back(pool.acquire_pending_state());
    }
    for (auto& state : states) {
        pool.release_pending_state(std::move(state));
    }

    // Reset stats
    pool.reset_stats();

    // Now acquire/release 100 times (steady state)
    for (int i = 0; i < 100; ++i) {
        auto state = pool.acquire_pending_state();
        pool.release_pending_state(std::move(state));
    }

    // Should be 100% pool hits (no allocations)
    auto stats = pool.get_stats();
    EXPECT_EQ(stats.pool_misses, 0);
    EXPECT_EQ(stats.pool_hits, 100);
}
```

**Done Means**:
- ✅ Pool compiles and links
- ✅ All unit tests pass
- ✅ Statistics tracking works
- ✅ Zero allocations in steady state (test validates)

**Rollback Plan**:
- Feature flag: `MCTS_STATE_POOLING_ENABLED=0`
- Falls back to `clone()` everywhere (slower but safe)

---

### T008: Integrate State Pool into ContinuousSimulationRunner

**Goal**: Modify ContinuousSimulationRunner to use state pool instead of cloning.

**Scope**: 1.0 day

**Dependencies**: T007

**Files to Modify**:
1. `cpp_extensions/mcts/continuous_simulation_runner.hpp`
2. `cpp_extensions/mcts/continuous_simulation_runner.cpp`

**Implementation Details**:

```cpp
// cpp_extensions/mcts/continuous_simulation_runner.hpp
#include "state_pool.hpp"
#include "feature_flags.hpp"

class ContinuousSimulationRunner : public SimulationRunner {
private:
    // NEW: Thread-local state pool
    std::unique_ptr<ThreadLocalStatePool> state_pool_;

public:
    ContinuousSimulationRunner(MCTSTree& tree,
                                PUCTSelector& selector,
                                BackupManager& backup,
                                VirtualLossManager& virtual_loss,
                                const IGameState& template_state)
        : SimulationRunner(tree, selector, backup, virtual_loss) {

        // Initialize state pool if feature enabled
        if (FeatureFlags::is_state_pooling_enabled()) {
            state_pool_ = std::make_unique<ThreadLocalStatePool>(
                template_state,
                16  // Initial capacity
            );
        }
    }

    // ... rest of class ...
};
```

**Modify simulation loop** (`continuous_simulation_runner.cpp:70-120`):

```cpp
int ContinuousSimulationRunner::run_continuous(
    IGameState& root_state,
    NodeIndex root_index,
    AsyncInferenceQueue& queue,
    int num_simulations) {

    // ... setup code ...

    while (completed < num_simulations) {
        bool waiting_for_leaf = false;

        // Phase 1: Select to leaf and submit inference
        if (submitted < num_simulations) {
            // NEW: Reuse working state OR clone (fallback)
            std::unique_ptr<IGameState> current_state;

            if (state_pool_) {
                // ✅ OPTIMIZED: Reuse via copyFrom
                IGameState& working = state_pool_->get_working_state();
                working.copyFrom(root_state);
                current_state.reset(&working);  // Non-owning reference
            } else {
                // FALLBACK: Traditional clone
                current_state = root_state.clone();
            }

            // Clear and reuse path buffer
            path_buffer_.clear();

            // Select to leaf
            NodeIndex leaf = select_leaf(root_index, *current_state, path_buffer_);

            // Check terminal
            if (current_state->isTerminal()) {
                float value = get_terminal_value(*current_state);
                std::reverse(path_buffer_.begin(), path_buffer_.end());
                backup_value(path_buffer_, value);
                completed++;
                submitted++;
                continue;
            }

            // Check expansion state
            bool submission_ready = true;
            if (!tree_.atomic_try_mark_expanding(leaf)) {
                // Another thread is expanding
                waiting_for_leaf = true;
                submission_ready = false;
            }

            if (submission_ready) {
                // NEW: Acquire state from pool OR clone
                std::unique_ptr<IGameState> queue_state;

                if (state_pool_) {
                    // ✅ OPTIMIZED: Acquire from pool, copy position
                    queue_state = state_pool_->acquire_pending_state();
                    queue_state->copyFrom(*current_state);
                } else {
                    // FALLBACK: Clone
                    queue_state = current_state->clone();
                }

                // Submit to queue (move ownership)
                uint64_t request_id = queue.submit_request(
                    std::move(queue_state),
                    leaf,
                    path_buffer_
                );

                // Store in pending buffer
                store_pending_expansion(request_id, leaf, path_buffer_);
                submitted++;
            }
        }

        // Phase 2: Process results
        // ... (unchanged) ...
    }

    // ... cleanup ...

    return completed;
}
```

**Done Means**:
- ✅ Code compiles without errors
- ✅ Backward compatible (works with flag disabled)
- ✅ Pool statistics show reuse (not 100% allocations)

**Rollback Plan**:
- Feature flag: `MCTS_STATE_POOLING_ENABLED=0`
- Code path uses traditional `clone()`

---

### T009: Remove Clone in AsyncInferenceQueue

**Goal**: Remove redundant clone in AsyncInferenceQueue::submit_request().

**Scope**: 0.5 day

**Dependencies**: T008

**Files to Modify**:
1. `cpp_extensions/mcts/async_inference_queue.cpp`

**Implementation Details**:

**Current code** (~line 130):
```cpp
uint64_t AsyncInferenceQueue::submit_request(
    std::unique_ptr<IGameState> state,
    NodeIndex node_index,
    std::vector<NodeIndex> path) {

    uint64_t request_id = next_request_id_.fetch_add(1);

    InferenceRequest request;
    request.request_id = request_id;
    request.state = state->clone();  // 🔴 EXTRA CLONE!
    request.node_index = node_index;
    request.path = std::move(path);

    // Enqueue
    if (!pending_requests_.try_enqueue(std::move(request))) {
        throw std::runtime_error("Queue full");
    }

    pending_count_.fetch_add(1);
    request_ready_.notify_one();

    return request_id;
}
```

**Optimized code**:
```cpp
uint64_t AsyncInferenceQueue::submit_request(
    std::unique_ptr<IGameState> state,
    NodeIndex node_index,
    std::vector<NodeIndex> path) {

    uint64_t request_id = next_request_id_.fetch_add(1);

    InferenceRequest request;
    request.request_id = request_id;
    request.state = std::move(state);  // ✅ MOVE (no clone)
    request.node_index = node_index;
    request.path = std::move(path);

    // Enqueue
    if (!pending_requests_.try_enqueue(std::move(request))) {
        throw std::runtime_error("Queue full");
    }

    pending_count_.fetch_add(1);
    request_ready_.notify_one();

    return request_id;
}
```

**Acceptance Tests**:

```cpp
// tests/unit/test_async_inference_queue.cpp
TEST(AsyncInferenceQueue, NoRedundantClone) {
    AsyncInferenceQueue queue;
    GomokuState state(15);

    // Track allocations
    size_t allocs_before = get_allocation_count();

    auto state_ptr = state.clone();
    uint64_t req_id = queue.submit_request(
        std::move(state_ptr),
        0,
        {}
    );

    size_t allocs_after = get_allocation_count();

    // Should be 1 allocation (initial clone), not 2
    EXPECT_EQ(allocs_after - allocs_before, 1);
}
```

**Done Means**:
- ✅ Code compiles
- ✅ No clone inside submit_request()
- ✅ Unit test validates single allocation
- ✅ Integration test shows throughput improvement

**Rollback Plan**:
- Revert commit (one-line change)
- No feature flag needed (safe optimization)

---

### T010: Add Results Ready Condition Variable

**Goal**: Add condition variable for results ready, eliminating spin-wait polling.

**Scope**: 1.0 day

**Dependencies**: T003

**Files to Modify**:
1. `cpp_extensions/mcts/async_inference_queue.hpp`
2. `cpp_extensions/mcts/async_inference_queue.cpp`

**Implementation Details**:

```cpp
// async_inference_queue.hpp
class AsyncInferenceQueue {
private:
    // Existing: Request ready CV
    std::mutex cv_mutex_;
    std::condition_variable request_ready_;

    // NEW: Results ready CV
    std::condition_variable results_ready_;

    std::atomic<bool> shutting_down_{false};

    // ... rest ...
};
```

```cpp
// async_inference_queue.cpp

void AsyncInferenceQueue::submit_results(
    const std::vector<InferenceResult>& results) {

    // Store results in results_buffer_
    for (const auto& result : results) {
        // ... store result ...
    }

    results_count_.fetch_add(results.size(), std::memory_order_release);

    // NEW: Wake threads waiting for results
    {
        std::lock_guard<std::mutex> lock(cv_mutex_);
        results_ready_.notify_all();
    }
}

// NEW: Blocking wait for results
bool AsyncInferenceQueue::wait_for_results(
    std::chrono::milliseconds timeout) {

    if (FeatureFlags::are_condition_variables_enabled()) {
        // ✅ OPTIMIZED: Block on CV
        std::unique_lock<std::mutex> lock(cv_mutex_);
        return results_ready_.wait_for(lock, timeout, [this]() {
            return results_count_.load() > 0 ||
                   shutting_down_.load();
        });
    } else {
        // FALLBACK: Spin-wait
        auto deadline = std::chrono::steady_clock::now() + timeout;
        while (std::chrono::steady_clock::now() < deadline) {
            if (results_count_.load() > 0) {
                return true;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        return false;
    }
}
```

**Acceptance Tests**:

```cpp
// tests/unit/test_condition_variables.cpp
TEST(ConditionVariables, BlockingWaitWakesOnResults) {
    AsyncInferenceQueue queue;

    // Submit results in separate thread
    std::thread producer([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        InferenceResult result{};
        queue.submit_results({result});
    });

    // Wait for results (should block then wake)
    auto start = std::chrono::steady_clock::now();
    bool success = queue.wait_for_results(std::chrono::seconds(1));
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_TRUE(success);
    EXPECT_GE(elapsed, std::chrono::milliseconds(90));
    EXPECT_LT(elapsed, std::chrono::milliseconds(200));

    producer.join();
}

TEST(ConditionVariables, TimeoutWorks) {
    AsyncInferenceQueue queue;

    // No results submitted
    auto start = std::chrono::steady_clock::now();
    bool success = queue.wait_for_results(std::chrono::milliseconds(100));
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_FALSE(success);  // Timeout
    EXPECT_GE(elapsed, std::chrono::milliseconds(90));
}
```

**Done Means**:
- ✅ CV added to AsyncInferenceQueue
- ✅ submit_results() notifies waiters
- ✅ wait_for_results() blocks efficiently
- ✅ Unit tests pass
- ✅ Feature flag controls behavior

**Rollback Plan**:
- Feature flag: `MCTS_CV_ENABLED=0`
- Falls back to spin-wait (current behavior)

---

### T011: Integrate CV into ContinuousSimulationRunner

**Goal**: Replace spin-wait in ContinuousSimulationRunner with blocking CV wait.

**Scope**: 1.0 day

**Dependencies**: T010

**Files to Modify**:
1. `cpp_extensions/mcts/continuous_simulation_runner.cpp`

**Implementation Details**:

**Current code** (~line 140-160):
```cpp
// Phase 2: Process results (SPIN-WAIT)
while (completed < num_simulations) {
    // ... select and submit ...

    // Process results
    int processed = process_completed_results();

    // If no results and have pending, spin-wait
    if (processed == 0 && pending_count_.load() > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
}
```

**Optimized code**:
```cpp
// Phase 2: Process results (BLOCKING WAIT)
while (completed < num_simulations) {
    // Phase 2a: Try to submit more (non-blocking)
    while (submitted < num_simulations && !should_wait_for_results()) {
        // ... select and submit ...
        submitted++;
    }

    // Phase 2b: Wait for results (blocking if pending)
    if (pending_count_.load() > 0) {
        // ✅ OPTIMIZED: Block on CV
        queue.wait_for_results(std::chrono::seconds(5));
    }

    // Phase 2c: Process results (batch)
    int processed = process_completed_results();
    completed += processed;
}

bool should_wait_for_results() const {
    // Wait if:
    // 1. Have pending requests, AND
    // 2. Queue is full OR reached in-flight limit
    return pending_count_.load() > 0 &&
           (queue.pending_count() >= queue_capacity_ * 0.9 ||
            pending_count_.load() >= max_in_flight_);
}
```

**Acceptance Tests**:

```python
# tests/integration/test_condition_variables_integration.py
import pytest
import time
import psutil
import os

@pytest.mark.integration
def test_thread_cpu_usage_with_cv():
    """Verify threads idle efficiently with CV (not spinning)."""
    import mcts_py

    # Enable CV
    os.environ['MCTS_CV_ENABLED'] = '1'
    mcts_py.reinitialize_feature_flags()

    # Start benchmark in background
    process = psutil.Process()
    cpu_before = process.cpu_percent(interval=1.0)

    # Run MCTS search (threads will wait for results)
    # ... run search ...

    cpu_during = process.cpu_percent(interval=1.0)

    # CPU should be low when threads idle (not 100% spinning)
    # Expect <10% CPU when waiting
    assert cpu_during < 50, \
        f"High CPU during wait: {cpu_during}% (expected <50%)"

@pytest.mark.integration
def test_throughput_unchanged_with_cv():
    """Verify CV doesn't hurt throughput."""
    # Run with CV enabled
    os.environ['MCTS_CV_ENABLED'] = '1'
    throughput_with_cv = benchmark_throughput()

    # Run with CV disabled (spin-wait)
    os.environ['MCTS_CV_ENABLED'] = '0'
    throughput_without_cv = benchmark_throughput()

    # Throughput should be similar (±5%)
    ratio = throughput_with_cv / throughput_without_cv
    assert 0.95 <= ratio <= 1.05, \
        f"CV changed throughput: {ratio:.2f}× (expected ~1.0×)"
```

**Done Means**:
- ✅ Spin-wait replaced with CV blocking
- ✅ Code compiles and runs
- ✅ Thread CPU usage low when idle (<10%)
- ✅ Throughput unchanged (±5%)
- ✅ Feature flag controls behavior

**Rollback Plan**:
- Feature flag: `MCTS_CV_ENABLED=0`
- Reverts to spin-wait

---

### T012: Node Allocator Over-Allocation

**Goal**: Optimize node allocator to support contiguous allocations from thread-local blocks.

**Scope**: 1.5 days

**Dependencies**: T003

**Files to Modify**:
1. `cpp_extensions/mcts/tree.cpp`
2. `cpp_extensions/mcts/tree.hpp`

**Implementation Details**:

```cpp
// tree.cpp

std::vector<NodeIndex> MCTSTree::allocate_nodes(int count) {
    if (count <= 0 || count > 256) {
        throw std::invalid_argument("Invalid node count");
    }

    // Single-node fast path (unchanged)
    if (count == 1) {
        return {allocate_single_node()};
    }

    // NEW: Multi-node allocation from thread-local block
    auto& block = get_thread_local_block();

    if (block.tree == this &&
        block.epoch == current_epoch_.load() &&
        block.remaining >= static_cast<uint32_t>(count)) {

        // ✅ FAST PATH: Allocate contiguous from thread-local
        NodeIndex start_idx = block.next;
        block.next += count;
        block.remaining -= count;
        block.allocations_from_block += count;

        std::vector<NodeIndex> indices(count);
        for (int i = 0; i < count; ++i) {
            indices[i] = start_idx + i;
        }

        return indices;
    }

    // Slow path: Refill thread-local block
    if (refill_thread_local_block()) {
        // Retry allocation (recursive)
        return allocate_nodes(count);
    }

    // Fallback: Direct global allocation (rare)
    return allocate_from_global_mutex(count);
}

bool MCTSTree::refill_thread_local_block() {
    std::lock_guard<std::mutex> lock(allocation_mutex_);

    auto& block = get_thread_local_block();

    // Check space available
    size_t available = max_nodes_ - next_free_index_.load();
    if (available == 0) {
        return false;  // Tree full
    }

    // Allocate block (cap at kThreadBlockSize)
    uint32_t block_size = std::min(
        kThreadBlockSize,
        static_cast<uint32_t>(available)
    );

    NodeIndex start_idx = next_free_index_.fetch_add(
        block_size,
        std::memory_order_relaxed
    );

    // Update thread-local block
    block.tree = this;
    block.tree_id = instance_id_;
    block.next = start_idx;
    block.remaining = block_size;
    block.epoch = current_epoch_.load();
    block.allocations_from_global++;

    return true;
}
```

**Acceptance Tests**:

```cpp
// tests/unit/test_node_allocator.cpp
TEST(NodeAllocator, ContiguousAllocationFromThreadLocal) {
    MCTSTree tree(10'000);

    // Allocate multiple children (e.g., 50 nodes)
    auto indices = tree.allocate_nodes(50);

    // Verify contiguous
    EXPECT_EQ(indices.size(), 50);
    for (size_t i = 1; i < indices.size(); ++i) {
        EXPECT_EQ(indices[i], indices[i-1] + 1);
    }
}

TEST(NodeAllocator, MultiThreadedContiguousAllocation) {
    MCTSTree tree(1'000'000);

    std::vector<std::thread> threads;
    std::atomic<int> total_allocated{0};

    for (int t = 0; t < 8; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 100; ++i) {
                auto indices = tree.allocate_nodes(10);
                EXPECT_EQ(indices.size(), 10);
                total_allocated.fetch_add(10);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    EXPECT_EQ(total_allocated.load(), 8 * 100 * 10);
}

TEST(NodeAllocator, LockContentionReduction) {
    MCTSTree tree(1'000'000);

    // Measure lock contention
    auto start = std::chrono::steady_clock::now();

    std::vector<std::thread> threads;
    for (int t = 0; t < 12; ++t) {
        threads.emplace_back([&]() {
            for (int i = 0; i < 1000; ++i) {
                tree.allocate_nodes(10);
            }
        });
    }

    for (auto& t : threads) {
        t.join();
    }

    auto elapsed = std::chrono::steady_clock::now() - start;
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    // With thread-local blocks, should complete quickly
    // Expect <100ms for 12k allocations
    EXPECT_LT(ms, 100);
}
```

**Done Means**:
- ✅ Contiguous allocation from thread-local blocks
- ✅ Refill takes global mutex (rare)
- ✅ Unit tests pass (contiguity, thread safety)
- ✅ Lock contention reduced (measured)

**Rollback Plan**:
- Revert commit (code is backward compatible)
- No feature flag needed (pure optimization)

---

### T013: Node Allocator Performance Validation

**Goal**: Validate node allocator reduces lock contention and improves thread scaling.

**Scope**: 0.5 day

**Dependencies**: T012

**Files to Create**:
1. `tests/performance/test_node_allocator_performance.py`

**Implementation Details**:

```python
# tests/performance/test_node_allocator_performance.py
import pytest
import time
import mcts_py

@pytest.mark.performance
def test_node_allocation_throughput():
    """Measure node allocation throughput."""
    tree = mcts_py.MCTSTree(1_000_000)

    # Warmup
    for _ in range(100):
        tree.allocate_nodes(10)

    # Measure
    start = time.perf_counter()
    num_allocations = 10_000
    for _ in range(num_allocations):
        tree.allocate_nodes(10)
    elapsed = time.perf_counter() - start

    allocs_per_sec = num_allocations / elapsed

    print(f"\nNode allocation throughput: {allocs_per_sec:.0f} allocs/sec")

    # Target: >100k allocs/sec (fast path)
    assert allocs_per_sec > 100_000

@pytest.mark.performance
def test_lock_contention_with_threads():
    """Measure lock contention at different thread counts."""
    results = {}

    for num_threads in [1, 2, 4, 8, 12]:
        tree = mcts_py.MCTSTree(1_000_000)

        start = time.perf_counter()

        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=lambda: [
                tree.allocate_nodes(10) for _ in range(1000)
            ])
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        elapsed = time.perf_counter() - start
        results[num_threads] = elapsed

    print("\nLock contention scaling:")
    for threads, time_sec in results.items():
        print(f"  {threads} threads: {time_sec:.3f}s")

    # Check scaling efficiency
    # Expect near-linear up to 8 threads
    time_1 = results[1]
    time_8 = results[8]
    efficiency = time_1 / (time_8 / 8)

    print(f"  8-thread efficiency: {efficiency:.1%}")

    # Target: ≥70% efficiency at 8 threads
    assert efficiency >= 0.70
```

**Done Means**:
- ✅ Allocation throughput >100k/sec
- ✅ Thread scaling efficiency ≥70% at 8 threads
- ✅ Test passes in CI

**Rollback Plan**:
- Revert T012 commit if performance regresses

---

## PHASE 2: Validation & Baseline Investigation

### T014: Comprehensive Throughput Benchmark Suite

**Goal**: Create comprehensive benchmark suite validating all Phase 1 optimizations.

**Scope**: 1.5 days

**Dependencies**: T001, T006, T009, T011, T013

**Files to Create**:
1. `tests/performance/test_phase4_validation.py`
2. `scripts/run_comprehensive_benchmarks.py`

**Implementation Details**:

(See plan.md Section E1 for full implementation)

**Key Tests**:
1. Throughput scaling (threads × batch size)
2. OpenMP feature extraction (<1ms validation)
3. State pooling zero clones
4. Condition variable efficiency
5. Node allocator scaling

**Done Means**:
- ✅ All tests pass
- ✅ Throughput ≥8,000 sims/sec (PRIMARY KPI)
- ✅ GPU util ≥80%
- ✅ Feature extraction <1ms
- ✅ Thread efficiency ≥70% @ 8 threads

**Rollback Plan**:
- Individual feature flags to isolate issues

---

### T014a: DLPack Fast Path Verification (Diagnostic)

**Goal**: Verify DLPack zero-copy tensor conversion is using fast path (not numpy fallback).

**Scope**: 0.25 day

**Dependencies**: T014

**Files to Create**:
1. `scripts/validate_dlpack_fast_path.py`

**Files to Modify**:
1. `cpp_extensions/mcts/python_bindings.cpp` - Add timing instrumentation
2. `cpp_extensions/mcts/feature_flags.hpp` - Add DLPACK_LOGGING_ENABLED flag
3. `src/neural/inference_worker.py` - Add conversion path logging

**Implementation Details**:

(See plan.md Section B5 for full implementation)

**Validation Script** (`scripts/validate_dlpack_fast_path.py`):
```python
"""Validate DLPack fast path usage."""
import os
import logging
from src.core.mcts import AlphaZeroMCTS
from src.games.gomoku_state import GomokuState

def validate_dlpack_fast_path():
    # Enable logging
    os.environ['MCTS_DLPACK_LOGGING_ENABLED'] = '1'

    # Run simulations
    state = GomokuState(board_size=15)
    mcts = AlphaZeroMCTS(state, num_threads=8, batch_size=64)
    mcts.search(state, num_simulations=1000)

    # Check telemetry
    stats = mcts.get_stats()
    avg_capsule_time = stats['dlpack_capsule_time_ms'] / stats['batch_count']
    avg_callback_time = stats['python_callback_time_ms'] / stats['batch_count']

    # Validation
    assert avg_capsule_time < 0.1, f"Capsule time {avg_capsule_time:.3f}ms > 0.1ms (copy detected?)"
    assert avg_callback_time < 0.5, f"Callback time {avg_callback_time:.3f}ms > 0.5ms (fallback?)"

    print("✅ VALIDATION PASSED: DLPack fast path confirmed")
    return True
```

**Instrumentation** (C++ side):
```cpp
// In python_bindings.cpp
py::object PyBatchInferenceCallback::__call__(
    const std::vector<std::unique_ptr<IGameState>>& states) {

    auto start = std::chrono::high_resolution_clock::now();

    // Create DLPack capsule
    py::object capsule = create_dlpack_capsule(states);

    auto capsule_end = std::chrono::high_resolution_clock::now();
    double capsule_ms = std::chrono::duration<double, std::milli>(
        capsule_end - start
    ).count();

    // Call Python inference
    py::gil_scoped_acquire gil;
    py::object result = python_callback_(capsule);

    auto callback_end = std::chrono::high_resolution_clock::now();
    double callback_ms = std::chrono::duration<double, std::milli>(
        callback_end - capsule_end
    ).count();

    // Log if enabled
    if (FeatureFlags::is_dlpack_logging_enabled()) {
        logger.info("DLPack: capsule={:.3f}ms, callback={:.3f}ms", capsule_ms, callback_ms);
    }

    // Accumulate telemetry
    stats_.dlpack_capsule_time_ms += capsule_ms;
    stats_.python_callback_time_ms += callback_ms;
    stats_.batch_count++;

    return result;
}
```

**Python-Side Logging** (`inference_worker.py`):
```python
def batch_inference(self, capsule):
    # Try fast path
    try:
        tensor = torch.from_dlpack(capsule)
        conversion_path = "dlpack_fast"
    except Exception as e:
        logger.warning(f"DLPack conversion failed: {e}, using numpy fallback")
        tensor = torch.from_numpy(np.array(capsule))
        conversion_path = "numpy_fallback"

    if os.environ.get('MCTS_DLPACK_LOGGING_ENABLED') == '1':
        logger.info(f"Conversion path: {conversion_path}")

    # Run inference
    return self.model(tensor)
```

**Expected Results**:
```
DLPack Performance:
  Total batches: 15
  Avg capsule time: 0.023ms per batch   ← Very small (zero-copy)
  Avg callback time: 0.112ms per batch  ← Small overhead

Validation Criteria:
  ✅ PASS: Capsule creation <0.1ms (zero-copy confirmed)
  ✅ PASS: Callback overhead <0.5ms (fast path confirmed)

Logs show: "Conversion path: dlpack_fast"

✅ VALIDATION PASSED: DLPack fast path confirmed
```

**Done Means**:
- ✅ Validation script runs without errors
- ✅ Capsule creation time <0.1ms per batch
- ✅ Callback overhead <0.5ms per batch
- ✅ Logs confirm "dlpack_fast" (not "numpy_fallback")
- ✅ Telemetry fields captured in stats

**Rollback Plan**:
- N/A (diagnostic only, no code changes)
- If fallback detected: Investigate DLPack integration (see plan.md B5.6)

**Impact**:
- DIAGNOSTIC priority (Phase 2 validation)
- If fast path confirmed: No action needed
- If fallback detected: 4-16% potential throughput gain from fixing

---

### T015: Ablation Study Suite

**Goal**: Create ablation study framework to isolate impact of each optimization.

**Scope**: 1.0 day

**Dependencies**: T014

**Files to Create**:
1. `scripts/run_ablations.py`
2. `scripts/analyze_ablations.py`

**Implementation Details**:

(See plan.md Section E2 for full implementation)

**Ablation Configurations**:
- Baseline (all OFF)
- OpenMP only
- State pooling only
- CV only
- All ON
- All ON + cache

**Done Means**:
- ✅ Ablation script runs all configurations
- ✅ Results saved to JSON
- ✅ Summary table generated
- ✅ Impact of each optimization measured

**Rollback Plan**:
- N/A (analysis tool)

---

### T016: Baseline Investigation (T017)

**Goal**: Systematic investigation to reproduce 3,831 sims/sec baseline configuration.

**Scope**: 2.0 days (time-boxed)

**Dependencies**: T014

**Files to Create**:
1. `scripts/baseline_investigation.py`
2. `results/baseline_investigation/config_sweep.csv`

**Implementation Details**:

```python
# scripts/baseline_investigation.py
"""Systematic grid search for baseline 3,831 sims/sec config."""

import itertools
import pandas as pd

# Configuration space
THREADS = [2, 4, 6, 8]
BATCH_SIZES = [32, 48, 64, 96]
TIMEOUTS = [0.5, 1.0, 1.5, 2.0]
VL_MAGNITUDES = [0.5, 1.0, 1.5, 2.0, 3.0]

# Total: 4×4×4×5 = 320 configurations

def run_baseline_search():
    results = []

    for threads, batch, timeout, vl in itertools.product(
        THREADS, BATCH_SIZES, TIMEOUTS, VL_MAGNITUDES
    ):
        config = {
            'threads': threads,
            'batch_size': batch,
            'timeout_ms': timeout,
            'vl_magnitude': vl,
        }

        # Run benchmark
        throughput = run_single_benchmark(config)

        result = {**config, 'throughput': throughput}
        results.append(result)

        print(f"Config {len(results)}/320: {throughput:.0f} sims/sec")

        # Early exit if found
        if throughput >= 3800:
            print(f"✅ FOUND BASELINE CONFIG: {config}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv('results/baseline_investigation/config_sweep.csv')

    # Find best config
    best = df.loc[df['throughput'].idxmax()]
    print(f"\nBest config: {best.to_dict()}")
    print(f"Throughput: {best['throughput']:.0f} sims/sec")
```

**Done Means**:
- ✅ Grid search completes (320 configs tested)
- ✅ Results saved to CSV
- ✅ Best configuration identified
- ✅ Report generated with findings

**Rollback Plan**:
- N/A (investigation, not code change)

**Time-Box**: 2 days maximum. If baseline not reproduced, document findings and proceed with current 2,147 sims/sec as new baseline.

---

### T017: KPI Dashboard Generation

**Goal**: Create automated KPI dashboard from benchmark history.

**Scope**: 0.5 day

**Dependencies**: T014, T015

**Files to Create**:
1. `scripts/generate_kpi_dashboard.py`
2. `results/dashboards/kpi_dashboard.html`

**Implementation Details**:

(See plan.md Section E3 for implementation)

**Dashboard Plots**:
1. Throughput over time
2. GPU utilization
3. Thread scaling
4. Feature extraction time
5. Batch size distribution
6. Cache hit rate (if enabled)

**Done Means**:
- ✅ Script generates HTML dashboard
- ✅ All KPIs plotted
- ✅ Dashboard viewable in browser
- ✅ Updates automatically from CSV

**Rollback Plan**:
- N/A (visualization tool)

---

## PHASE 3: NN-Eval Cache (Optional)

### T018: Zobrist Hash Implementation

**Goal**: Implement Zobrist hashing for Gomoku, Chess, Go.

**Scope**: 1.0 day

**Dependencies**: T003

**Files to Create**:
1. `cpp_extensions/cache/zobrist_hash.hpp`
2. `cpp_extensions/cache/zobrist_hash.cpp`
3. `tests/unit/test_zobrist_hash.cpp`

**Implementation Details**:

(See plan.md Section C2 for full implementation)

**Done Means**:
- ✅ Zobrist hasher compiles
- ✅ Hashes are deterministic (same state → same hash)
- ✅ Hashes are unique (collision rate <0.01%)
- ✅ Unit tests pass

**Rollback Plan**:
- Feature flag: `MCTS_CACHE_ENABLED=0` (not used yet)

---

### T019: NN-Eval Cache Data Structure

**Goal**: Implement sharded NN-eval cache with SLRU eviction.

**Scope**: 1.5 days

**Dependencies**: T018

**Files to Create**:
1. `cpp_extensions/cache/nn_eval_cache.hpp`
2. `cpp_extensions/cache/nn_eval_cache.cpp`
3. `tests/unit/test_nn_eval_cache.cpp`

**Implementation Details**:

(See plan.md Section C3 for full implementation)

**Done Means**:
- ✅ Cache compiles and links
- ✅ Lookup/insert work correctly
- ✅ SLRU eviction works
- ✅ Thread-safe (64 shards)
- ✅ Unit tests pass

**Rollback Plan**:
- Feature flag: `MCTS_CACHE_ENABLED=0`

---

### T020: Cache Integration into Batch Coordinator

**Goal**: Integrate NN-eval cache into BatchInferenceCoordinator.

**Scope**: 1.0 day

**Dependencies**: T019

**Files to Modify**:
1. `cpp_extensions/mcts/batch_inference_coordinator.cpp`
2. `cpp_extensions/mcts/batch_inference_coordinator.hpp`

**Implementation Details**:

(See plan.md Section C4 for implementation)

**Done Means**:
- ✅ Cache lookup before GPU
- ✅ Cache insert after GPU
- ✅ Hit rate telemetry collected
- ✅ Integration test shows hits/misses

**Rollback Plan**:
- Feature flag: `MCTS_CACHE_ENABLED=0`

---

### T021: Cache Performance Validation

**Goal**: Measure cache hit rate and throughput improvement.

**Scope**: 1.0 day

**Dependencies**: T020

**Files to Create**:
1. `tests/performance/test_nn_eval_cache_performance.py`

**Implementation Details**:

```python
@pytest.mark.performance
def test_cache_hit_rate():
    """Measure cache hit rate in typical workload."""
    import os
    os.environ['MCTS_CACHE_ENABLED'] = '1'

    # Run 1000 simulations (multiple games)
    # ...

    cache_stats = mcts.get_cache_stats()
    hit_rate = cache_stats['hits'] / cache_stats['lookups']

    print(f"\nCache hit rate: {hit_rate:.1%}")

    # Target: ≥10% for Gomoku (conservative)
    assert hit_rate >= 0.10

@pytest.mark.performance
def test_cache_throughput_improvement():
    """Measure throughput gain from cache."""
    # Without cache
    os.environ['MCTS_CACHE_ENABLED'] = '0'
    throughput_no_cache = benchmark_throughput()

    # With cache
    os.environ['MCTS_CACHE_ENABLED'] = '1'
    throughput_with_cache = benchmark_throughput()

    improvement = (throughput_with_cache / throughput_no_cache - 1) * 100

    print(f"\nCache throughput gain: +{improvement:.1f}%")

    # Target: +15% minimum (from spec)
    assert improvement >= 15.0
```

**Done Means**:
- ✅ Hit rate ≥10% (Gomoku)
- ✅ Throughput improvement ≥15%
- ✅ Test passes

**Rollback Plan**:
- Feature flag: `MCTS_CACHE_ENABLED=0`

---

## PHASE 4: Multi-Actor Self-Play

### T022: Centralized Inference Server

**Goal**: Implement centralized GPU inference server for multi-actor self-play.

**Scope**: 1.5 days

**Dependencies**: T003

**Files to Create**:
1. `src/self_play/inference_server.py`
2. `tests/unit/test_inference_server.py`

**Implementation Details**:

(See plan.md Section D1 for full implementation)

**Done Means**:
- ✅ Server starts and accepts requests
- ✅ Batching works (collects batch-64)
- ✅ Fairness policy works (round-robin)
- ✅ Results demultiplexed correctly
- ✅ Unit tests pass

**Rollback Plan**:
- Feature flag: `MCTS_MULTI_ACTOR_ENABLED=0`
- Don't start multi-actor mode

---

### T023: Self-Play Actor Implementation

**Goal**: Implement self-play actor process.

**Scope**: 1.0 day

**Dependencies**: T022

**Files to Create**:
1. `src/self_play/actor.py`
2. `tests/unit/test_actor.py`

**Implementation Details**:

(See plan.md Section D2 for implementation)

**Done Means**:
- ✅ Actor plays complete game
- ✅ Submits inference requests to server
- ✅ Receives results correctly
- ✅ Generates training data
- ✅ Unit tests pass

**Rollback Plan**:
- Feature flag: `MCTS_MULTI_ACTOR_ENABLED=0`

---

### T024: Token Bucket Backpressure

**Goal**: Implement token bucket backpressure for actors.

**Scope**: 0.5 day

**Dependencies**: T023

**Files to Modify**:
1. `src/self_play/actor.py` (add TokenBucketBackpressure)

**Implementation Details**:

(See plan.md Section D2 for TokenBucketBackpressure class)

**Done Means**:
- ✅ Token bucket limits in-flight requests
- ✅ Actors block when tokens depleted
- ✅ Refill rate configurable
- ✅ Unit test validates blocking

**Rollback Plan**:
- Disable backpressure (set capacity=∞)

---

### T025: Multi-Actor Orchestrator

**Goal**: Create orchestrator script to launch server + actors.

**Scope**: 1.0 day

**Dependencies**: T023, T024

**Files to Create**:
1. `scripts/run_multi_actor_selfplay.py`
2. `tests/integration/test_multi_actor_selfplay.py`

**Implementation Details**:

(See plan.md Section D3 for implementation)

**Done Means**:
- ✅ Script launches server + actors
- ✅ All processes communicate correctly
- ✅ Graceful shutdown works
- ✅ Integration test runs 8 actors

**Rollback Plan**:
- Feature flag: `MCTS_MULTI_ACTOR_ENABLED=0`
- Use single-actor self-play

---

### T026: Multi-Actor Performance Validation

**Goal**: Validate multi-actor achieves 200-300 games/hour @ 80-95% GPU util.

**Scope**: 1.0 day

**Dependencies**: T025

**Files to Create**:
1. `tests/performance/test_multi_actor_performance.py`

**Implementation Details**:

```python
@pytest.mark.performance
@pytest.mark.slow
def test_multi_actor_throughput():
    """Validate 200-300 games/hour target."""
    num_actors = 8
    games_per_actor = 5
    simulations_per_move = 800

    start = time.time()

    # Run multi-actor self-play
    run_multi_actor_selfplay(
        num_actors=num_actors,
        games_per_actor=games_per_actor,
        simulations_per_move=simulations_per_move,
    )

    elapsed_hours = (time.time() - start) / 3600
    total_games = num_actors * games_per_actor
    games_per_hour = total_games / elapsed_hours

    print(f"\nMulti-actor performance:")
    print(f"  Games: {total_games}")
    print(f"  Time: {elapsed_hours:.2f} hours")
    print(f"  Games/hour: {games_per_hour:.0f}")

    # Target: ≥200 games/hour
    assert games_per_hour >= 200

@pytest.mark.performance
def test_multi_actor_gpu_utilization():
    """Validate GPU utilization 80-95%."""
    # Start multi-actor self-play
    process = start_multi_actor_background()

    time.sleep(30)  # Warm-up

    # Measure GPU util for 60 seconds
    gpu_utils = []
    for _ in range(60):
        util = measure_gpu_utilization()
        gpu_utils.append(util)
        time.sleep(1)

    avg_gpu_util = np.mean(gpu_utils)

    print(f"\nMulti-actor GPU utilization: {avg_gpu_util:.1f}%")

    # Target: 80-95%
    assert 80 <= avg_gpu_util <= 95

    process.terminate()
```

**Done Means**:
- ✅ Games/hour ≥200
- ✅ GPU utilization 80-95%
- ✅ Avg batch size ≥51/64
- ✅ Tests pass

**Rollback Plan**:
- Reduce actor count if issues
- Disable multi-actor mode

---

## PHASE 5: Final Validation & Documentation

### T027: Final Acceptance Benchmarks

**Goal**: Run comprehensive acceptance benchmarks validating all spec.md requirements.

**Scope**: 1.0 day

**Dependencies**: T014, T021, T026

**Files to Create**:
1. `tests/acceptance/test_spec_004_acceptance.py`
2. `results/acceptance/acceptance_report.md`

**Implementation Details**:

```python
# tests/acceptance/test_spec_004_acceptance.py
"""Spec 004 Acceptance Tests - Validate ALL requirements."""

import pytest

@pytest.mark.acceptance
class TestPhase4Acceptance:
    """Phase 4 (Single MCTS) acceptance criteria."""

    def test_req_perf_001_throughput(self):
        """REQ-PERF-001: Throughput ≥8,000 sims/sec."""
        throughput = benchmark_throughput(
            game='gomoku',
            board_size=15,
            simulations=10000,
            threads=8,
            batch_size=64,
        )
        assert throughput >= 8000, \
            f"FAIL: {throughput:.0f} sims/sec (target ≥8000)"

    def test_req_perf_002_gpu_util(self):
        """REQ-PERF-002: GPU utilization ≥80%."""
        gpu_util = measure_gpu_utilization_during_search()
        assert gpu_util >= 80, \
            f"FAIL: {gpu_util:.1f}% (target ≥80%)"

    def test_req_perf_004_feature_extraction(self):
        """REQ-PERF-004: Feature extraction ≤1.0ms."""
        feature_time = measure_feature_extraction_time()
        assert feature_time <= 1.0, \
            f"FAIL: {feature_time:.2f}ms (target ≤1.0ms)"

    # ... all 25 requirements ...

@pytest.mark.acceptance
class TestPhase5Acceptance:
    """Phase 5 (Multi-Actor) acceptance criteria."""

    def test_req_perf_007_games_per_hour(self):
        """REQ-PERF-007: 200-300 games/hour."""
        games_per_hour = measure_multi_actor_throughput()
        assert 200 <= games_per_hour <= 300

    # ... rest of phase 5 tests ...
```

**Done Means**:
- ✅ All 25 requirements tested
- ✅ Acceptance report generated
- ✅ All tests PASS
- ✅ Results committed to repo

**Rollback Plan**:
- If acceptance fails, identify failing requirement
- Use feature flags to isolate issue
- Document deviation in acceptance report

---

### T028: Performance Regression Test Suite

**Goal**: Create CI-integrated regression test suite.

**Scope**: 0.5 day

**Dependencies**: T027

**Files to Create**:
1. `.github/workflows/performance_regression.yml`
2. `tests/regression/test_performance_regression.py`

**Implementation Details**:

```yaml
# .github/workflows/performance_regression.yml
name: Performance Regression Tests

on:
  pull_request:
    branches: [main]

jobs:
  regression:
    runs-on: [self-hosted, ryzen-5900x, rtx-3060ti]

    steps:
      - uses: actions/checkout@v3

      - name: Setup OpenMP
        run: source scripts/configure_openmp.sh

      - name: Build extensions
        run: pip install -e .

      - name: Run regression tests
        run: |
          pytest tests/regression/ -v --benchmark-only

      - name: Check thresholds
        run: |
          python scripts/check_regression_thresholds.py
          # Fails if throughput < 95% of target (7,600 sims/sec)
```

**Done Means**:
- ✅ CI workflow created
- ✅ Regression tests run on PRs
- ✅ Automatic failure if throughput < 95% target
- ✅ Documented in README

**Rollback Plan**:
- Disable CI check if too strict

---

### T029: Design Documentation

**Goal**: Document architecture, design decisions, and troubleshooting.

**Scope**: 1.0 day

**Dependencies**: T027

**Files to Create**:
1. `docs/design/phase4_architecture.md`
2. `docs/design/state_pooling_design.md`
3. `docs/design/nn_eval_cache_design.md`
4. `docs/operations/troubleshooting.md`

**Implementation Details**:

```markdown
# docs/design/phase4_architecture.md

## Architecture Overview

[Mermaid diagrams from plan.md]

## Component Descriptions

### State Pooling
- Thread-local pools eliminate allocations
- copyFrom() for in-place reset
- std::move() for ownership transfer
- Expected: 0× clones per simulation (vs 2-3× baseline)

### Condition Variables
- results_ready_ CV wakes threads
- Eliminates spin-wait (60% idle → <1%)
- Timeout handling for safety

### Node Allocator
- Thread-local blocks (4096 nodes)
- Contiguous allocation support
- Lock frequency: 99.9% → 0.1%

## Performance Impact

[Tables with before/after metrics]
```

```markdown
# docs/operations/troubleshooting.md

## Common Issues

### OpenMP Not Working
**Symptoms**: Feature extraction >5ms per batch-64
**Diagnosis**: `python scripts/verify_openmp.py`
**Fix**:
1. Verify compilation: `nm mcts_py*.so | grep GOMP`
2. Set OMP_NUM_THREADS=12
3. Rebuild: `pip install -e . --force-reinstall`

### Low GPU Utilization
**Symptoms**: GPU util <50%
**Diagnosis**: Check avg batch size
**Fix**:
1. Increase batch size (64 → 96)
2. Reduce timeout (2ms → 1ms)
3. Enable multi-actor mode

### Thread Contention
**Symptoms**: Throughput drops with >4 threads
**Diagnosis**: Check thread affinity
**Fix**:
1. Verify OMP_PROC_BIND=close
2. Check core pinning
3. Reduce thread count

## Feature Flags

[Table of all flags with descriptions]
```

**Done Means**:
- ✅ Architecture documented
- ✅ Design decisions explained
- ✅ Troubleshooting guide created
- ✅ Feature flags documented

**Rollback Plan**:
- N/A (documentation)

---

### T030: Operating Guide & Runbook

**Goal**: Create operational runbook for running optimized MCTS.

**Scope**: 0.5 day

**Dependencies**: T029

**Files to Create**:
1. `docs/operations/runbook.md`
2. `docs/operations/tuning_guide.md`

**Implementation Details**:

```markdown
# docs/operations/runbook.md

## Quick Start

### 1. Environment Setup
```bash
source scripts/configure_openmp.sh
pip install -e .
```

### 2. Verify OpenMP
```bash
python scripts/verify_openmp.py
# Should see all ✅ checks
```

### 3. Run Benchmark
```bash
python scripts/benchmark_throughput.py \
  --game gomoku \
  --threads 8 \
  --batch-size 64 \
  --simulations 10000
```

**Expected Output**:
- Throughput: ≥8,000 sims/sec
- GPU util: 80-95%
- Feature extraction: <1ms

### 4. Run Self-Play
```bash
python scripts/run_multi_actor_selfplay.py \
  --model-path models/gomoku.pth \
  --num-actors 8 \
  --games-per-actor 25
```

## Monitoring

### Key Metrics
1. Throughput (sims/sec) - PRIMARY KPI
2. GPU utilization (%) - Target 80-95%
3. Avg batch size - Target ≥51/64
4. Feature extraction time (ms) - Target <1ms

### Telemetry Collection
```bash
python scripts/generate_kpi_dashboard.py
# Opens results/dashboards/kpi_dashboard.html
```

## Troubleshooting

[Link to troubleshooting.md]
```

```markdown
# docs/operations/tuning_guide.md

## Tuning Parameters

### Thread Count
- **Default**: 8 threads
- **Range**: 1-12 threads
- **Tuning**: Run `scripts/tune_threads.py`
- **Optimal**: Usually 4-8 threads (hardware-dependent)

### Batch Size
- **Default**: 64
- **Range**: 32-128
- **Trade-off**: Larger = better GPU util, higher latency
- **Tuning**: Run `scripts/tune_batch_size.py`

### Timeout
- **Default**: 1.0ms
- **Range**: 0.5-2.0ms
- **Trade-off**: Lower = faster response, smaller batches
- **Tuning**: Run `scripts/tune_timeout.py`

### Virtual Loss Magnitude
- **Default**: 1.0
- **Range**: 0.5-3.0
- **Trade-off**: Higher = less collision, more Q distortion
- **Tuning**: Run `scripts/tune_virtual_loss.py`

## Performance Optimization Checklist

- [ ] OpenMP compiled and configured (OMP_NUM_THREADS=12)
- [ ] Thread affinity set (OMP_PROC_BIND=close)
- [ ] GPU warmup enabled (5+ batches before timing)
- [ ] Batch size optimized (64 for most cases)
- [ ] Timeout tuned (1-2ms typical)
- [ ] Feature extraction <1ms (verify with test)
- [ ] State pooling enabled (MCTS_STATE_POOLING_ENABLED=1)
- [ ] Condition variables enabled (MCTS_CV_ENABLED=1)
```

**Done Means**:
- ✅ Runbook covers common operations
- ✅ Tuning guide explains all parameters
- ✅ Checklist provided for optimization
- ✅ Examples work correctly

**Rollback Plan**:
- N/A (documentation)

---

### T030a: Precompute Legal Moves (Optional Enhancement)

**Goal**: Eliminate redundant legal move generation by precomputing in simulation runner and passing to expansion.

**Scope**: 1.0 day

**Dependencies**: T009 (State Pooling), T014 (Validation)

**Files to Modify**:
1. `cpp_extensions/mcts/inference_queue.hpp` - Extend InferenceRequest structure
2. `cpp_extensions/mcts/continuous_simulation_runner.cpp` - Precompute legal moves
3. `cpp_extensions/mcts/tree.cpp` - Use precomputed moves in expand_node_with_result
4. `cpp_extensions/mcts/batch_inference_coordinator.cpp` - Pass moves through pipeline
5. `cpp_extensions/mcts/feature_flags.hpp` - Add PRECOMPUTE_LEGAL_MOVES flag

**Files to Create**:
1. `tests/unit/test_precomputed_legal_moves.py` - Validation tests

**Implementation Details**:

(See plan.md Section B2.4 for full implementation)

**Key Changes**:

1. **Extend InferenceRequest**:
```cpp
struct InferenceRequest {
    uint64_t request_id;
    std::unique_ptr<IGameState> state;
    NodeIndex node_index;
    std::vector<NodeIndex> path;

    // NEW: Precomputed legal moves
    std::vector<int> legal_moves;
    int current_player;
};
```

2. **Populate in Simulation Runner**:
```cpp
// Extract legal moves BEFORE copying state
std::vector<int> legal_moves = working_state.getLegalMoves();
int current_player = working_state.getCurrentPlayer();

// Submit with legal moves
queue.submit_request(
    std::move(queue_state), leaf, path_buffer_,
    legal_moves, current_player  // NEW
);
```

3. **Use in Expansion**:
```cpp
void MCTSTree::expand_node_with_result(
    NodeIndex parent_idx,
    const InferenceResult& result,
    const std::vector<int>& legal_moves) {  // NEW: Pass directly

    // No re-computation needed!
    for (int i = 0; i < legal_moves.size(); ++i) {
        int move = legal_moves[i];
        float prior = result.policy[move];
        // ... initialize child ...
    }
}
```

**Validation Tests**:
```python
def test_legal_moves_precomputation():
    """Verify legal moves computed only once per expansion."""
    state = GomokuState()
    mcts = AlphaZeroMCTS(state)
    mcts.search(state, num_simulations=100)

    stats = mcts.get_stats()
    ratio = stats['legal_moves_computed'] / stats['total_expansions']

    # Should be ~1× (not 2×)
    assert ratio < 1.1, f"Legal moves computed {ratio:.2f}× per expansion"

def test_expansion_parity_with_precomputation():
    """Verify precomputed moves produce identical results."""
    state = GomokuState()

    # With precomputation
    os.environ['MCTS_PRECOMPUTE_LEGAL_MOVES'] = '1'
    mcts1 = AlphaZeroMCTS(state)
    mcts1.search(state, num_simulations=100)
    policy1 = mcts1.get_policy(state)

    # Without precomputation
    os.environ['MCTS_PRECOMPUTE_LEGAL_MOVES'] = '0'
    mcts2 = AlphaZeroMCTS(state)
    mcts2.search(state, num_simulations=100)
    policy2 = mcts2.get_policy(state)

    # Should be identical
    np.testing.assert_allclose(policy1, policy2, rtol=0.01)
```

**Done Means**:
- ✅ InferenceRequest extended with legal_moves field
- ✅ Simulation runner precomputes moves before queuing
- ✅ expand_node_with_result uses precomputed moves
- ✅ Parity tests pass (identical search results)
- ✅ Telemetry shows ~1× legal move computation per expansion (not 2×)
- ✅ Throughput improvement measured (expected 10-20%)

**Rollback Plan**:
- Feature flag: `MCTS_PRECOMPUTE_LEGAL_MOVES=0`
- Rollback commit if parity tests fail

**Expected Performance Impact**:
- Gomoku: 10-15% expansion speedup
- Chess: 15-20% expansion speedup (complex move generation)
- Go: 10-15% expansion speedup
- Total throughput: +10-20% (game-dependent)

**Status**: OPTIONAL (Phase 2 enhancement, medium priority)

---

## PHASE 6: Neural Network Optimization (Future)

**Status**: FUTURE phase (post-8k MCTS target)
**Timeline**: 10 days (single-threaded)
**Expected Gain**: 18-22k sims/sec total throughput (2.5-3× from 8k baseline)

**Reference**: See `NEURAL_NETWORK_OPTIMIZATION.md` for complete specification (9,250 lines)

---

### T031: RepECA Block Implementation

**Goal**: Implement RepVGG + ECA attention blocks with train/deploy duality.

**Scope**: 1.5 days

**Dependencies**: T014 (8k baseline achieved)

**Files to Create**:
1. `src/neural/modules/repeca_block.py`
2. `src/neural/modules/structural_reparameterization.py`
3. `tests/unit/test_repeca_block.py`

**Implementation Details**:

(See NEURAL_NETWORK_OPTIMIZATION.md Section 3.1 for full implementation)

**Key Features**:
- Multi-branch training (Conv3×3 + Conv1×1 + identity)
- Single-branch deployment (fused Conv3×3)
- ECA channel attention (local cross-channel interaction)
- Batch normalization folding

**Expected Speedup**: +25-50% model inference speed

**Done Means**:
- ✅ RepECA block passes unit tests
- ✅ Train/deploy fusion verified (bit-exact output)
- ✅ Model inference 1.25-1.5× faster
- ✅ Accuracy preserved (policy agreement ≥95%)

**Rollback Plan**:
- Keep baseline SE-ResNet model
- Feature flag: `USE_REPECA_BLOCKS=0`

---

### T032: Ghost Bottleneck Implementation

**Goal**: Implement Ghost convolutions for cheap feature generation.

**Scope**: 1.0 day

**Dependencies**: T031

**Files to Create**:
1. `src/neural/modules/ghost_bottleneck.py`
2. `tests/unit/test_ghost_bottleneck.py`

**Implementation Details**:

(See NEURAL_NETWORK_OPTIMIZATION.md Section 3.2)

**Key Features**:
- Intrinsic features via Conv1×1 (50% channels)
- Ghost features via DWConv3×3 (50% channels)
- Shuffle operation for feature mixing

**Expected Speedup**: +40-80% model inference speed

**Done Means**:
- ✅ Ghost bottleneck passes unit tests
- ✅ Model inference 1.4-1.8× faster
- ✅ Accuracy preserved (win rate ≥99%)

**Rollback Plan**:
- Feature flag: `USE_GHOST_BOTTLENECK=0`

---

### T033: Early-Exit Heads Implementation

**Goal**: Implement confidence-based early exits for simple positions.

**Scope**: 1.5 days

**Dependencies**: T032

**Files to Create**:
1. `src/neural/modules/early_exit_head.py`
2. `src/neural/training/early_exit_loss.py`
3. `tests/unit/test_early_exit.py`

**Implementation Details**:

(See NEURAL_NETWORK_OPTIMIZATION.md Section 3.4)

**Key Features**:
- Shallow heads at blocks 8, 12, 16 (main at 20)
- Entropy + threat detection confidence
- Dynamic exit based on position complexity

**Expected Speedup**: +20-60% effective throughput (position-dependent)

**Done Means**:
- ✅ Early-exit heads implemented
- ✅ Training converges with multi-head loss
- ✅ Average inference time reduced 1.2-1.6×
- ✅ Search quality preserved (policy agreement ≥93%)

**Rollback Plan**:
- Use only final head (skip early exits)

---

### T034: Two-Tier Cascade Implementation

**Goal**: Implement micro-net first pass with escalation to main net.

**Scope**: 2.0 days

**Dependencies**: T033

**Files to Create**:
1. `src/neural/models/micro_net.py` (8 blocks, 128 channels)
2. `src/neural/inference/cascade_inference.py`
3. `tests/unit/test_cascade_inference.py`

**Implementation Details**:

(See NEURAL_NETWORK_OPTIMIZATION.md Section 3.3)

**Key Features**:
- Micro-net: 8 blocks, 128 channels (~3× faster)
- Escalation triggers: entropy > τ, threat detected, near-terminal
- Acceptance rate: 30-60% (position-dependent)

**Expected Speedup**: +50-150% effective throughput

**Done Means**:
- ✅ Micro-net trained to convergence
- ✅ Cascade inference working
- ✅ Total throughput 1.5-2.5× improvement
- ✅ Search quality preserved (win rate ≥98%)

**Rollback Plan**:
- Use only main net (disable micro-net)

---

### T035: Auxiliary Threat Detection Heads

**Goal**: Add threat detection heads to preserve tactical strength.

**Scope**: 1.0 day

**Dependencies**: T034

**Files to Create**:
1. `src/neural/modules/threat_detection_head.py`
2. `src/neural/training/threat_loss.py`
3. `tests/unit/test_threat_detection.py`

**Implementation Details**:

(See NEURAL_NETWORK_OPTIMIZATION.md Section 4)

**Key Features**:
- Threat head outputs: 4-in-a-row, 3-in-a-row (per direction)
- Multi-task loss: policy + value + threats
- Used for confidence calibration in cascade

**Expected Impact**: Preserve strength despite architecture changes

**Done Means**:
- ✅ Threat heads implemented
- ✅ Training converges with multi-task loss
- ✅ Threat detection accuracy ≥90%
- ✅ Search quality matches baseline (win rate ≥99.5%)

**Rollback Plan**:
- N/A (auxiliary, doesn't affect core functionality)

---

### T036: Training & Validation Protocol

**Goal**: Train optimized models and validate via A/B testing.

**Scope**: 2.0 days (mostly compute time)

**Dependencies**: T035

**Files to Create**:
1. `src/training/train_optimized_model.py`
2. `scripts/run_ab_validation.py`
3. `scripts/measure_model_latency.py`

**Implementation Details**:

(See NEURAL_NETWORK_OPTIMIZATION.md Section 6)

**Training Protocol**:
1. **Self-Distillation**: Train lightweight model with baseline as teacher
2. **Multi-Task Loss**: Policy + value + threats + early-exit heads
3. **Validation**: 1000-game matches against baseline

**A/B Validation**:
```bash
# Measure model latency
python scripts/measure_model_latency.py \
  --baseline models/baseline.pth \
  --optimized models/optimized.pth \
  --batch-size 64

# Expected: 1.5-3.5× faster inference

# Validate strength
python scripts/run_ab_validation.py \
  --baseline models/baseline.pth \
  --optimized models/optimized.pth \
  --games 1000

# Expected: Win rate ≥98% (within ±2%)
```

**Done Means**:
- ✅ Optimized model trained to convergence
- ✅ Model latency 1.5-3.5× faster than baseline
- ✅ A/B validation win rate ≥98%
- ✅ Policy agreement ≥93%
- ✅ Total MCTS throughput ≥18k sims/sec

**Rollback Plan**:
- Keep baseline model if validation fails

---

### T037: NN Optimization Throughput Benchmarking

**Goal**: Measure end-to-end throughput improvement from NN optimizations.

**Scope**: 1.0 day

**Dependencies**: T036

**Files to Create**:
1. `tests/performance/test_nn_optimization_throughput.py`
2. `scripts/benchmark_nn_optimization.py`

**Implementation Details**:

**Benchmark Suite**:
```python
def test_nn_optimization_throughput():
    """Measure throughput with optimized NN."""
    configs = [
        {'model': 'baseline', 'expected': 8000},
        {'model': 'repeca', 'expected': 10000},
        {'model': 'ghost', 'expected': 12000},
        {'model': 'cascade', 'expected': 18000},
    ]

    for config in configs:
        throughput = run_benchmark(config)
        assert throughput >= config['expected'] * 0.95, \
            f"{config['model']}: {throughput} < {config['expected']}"
```

**Expected Results**:
| Configuration | Expected Throughput | Achieved | Status |
|---------------|---------------------|----------|--------|
| Baseline | 8,000 sims/sec | TBD | ⏳ |
| RepECA | 10,000 sims/sec | TBD | ⏳ |
| Ghost+Shuffle | 12,000 sims/sec | TBD | ⏳ |
| Two-Tier Cascade | **18-22k sims/sec** | TBD | ⏳ |

**Done Means**:
- ✅ All configurations benchmarked
- ✅ RepECA achieves ≥10k sims/sec
- ✅ Cascade achieves ≥18k sims/sec
- ✅ Results documented in final report

**Rollback Plan**:
- Revert to baseline if any config fails acceptance

---

## Summary: Task Statistics

**Total Tasks**: 39 (30 core + 9 enhancement/future)
**Total Estimated Time**: 40.75 days (single-threaded)

**By Phase**:
- Phase 0 (Foundation): 2.0 days (3 tasks)
- Phase 1 (CPU Optimizations): 11.5 days (10 tasks)
- Phase 2 (Validation): 4.75 days (5 tasks) - *includes T014a (diagnostic)*
- Phase 3 (NN-Cache): 4.5 days (4 tasks)
- Phase 4 (Multi-Actor): 5.5 days (5 tasks)
- Phase 5 (Documentation): 3.5 days (5 tasks) - *includes T030a (optional)*
- Phase 6 (NN Optimization): 10.0 days (7 tasks) - *FUTURE phase*

**Critical Path** (Phase 0 → Phase 1 → Phase 2):
- T001 → T002 → T003 → T004 → T005 → T006 (OpenMP)
- T007 → T008 → T009 (State Pooling)
- T010 → T011 (Condition Variables)
- T012 → T013 (Node Allocator)
- T014 → T015 → T016 (Validation)
- **Total Critical Path**: ~18 days

**Parallelizable** (after critical path):
- Phase 3 (NN-Cache) - Optional, can run in parallel with Phase 4
- Phase 4 (Multi-Actor) - Independent after Phase 2
- Phase 5 (Docs) - Can start early, update iteratively

**Core Phases (0-5)**: 3-4 weeks with 1 engineer (accounting for parallelization)
**Phase 6 (NN Optimization)**: FUTURE work (post-8k target), 2 weeks additional

**New Tasks Added** (v1.1 update):
- **T014a**: DLPack Fast Path Verification (0.25 days) - Phase 2 diagnostic
- **T030a**: Precompute Legal Moves (1.0 day) - Phase 5 optional enhancement
- **T031-T037**: Lightweight NN Optimization (10.0 days) - Phase 6 future work

**Status Update**:
- Core tasks (T001-T030): READY for implementation
- Enhancement tasks (T014a, T030a): OPTIONAL but specified
- Phase 6 (T031-T037): FUTURE (fully specified in NEURAL_NETWORK_OPTIMIZATION.md)

---

**END OF TASK BREAKDOWN v1.1**

**Ready for**: `/speckit.implement` or manual implementation following task order.
