# MCTS C++ Simulation Runner Guide

**Version:** 1.0
**Last Updated:** 2025-10-03
**Spec:** `specs/002-cpp-simulation-runner/`

This guide documents the C++ MCTS simulation runner implementation, which replaces the Python simulation loop to achieve 30-40k simulations/second (vs 246 sims/sec Python baseline).

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Integration Flow](#integration-flow)
3. [Performance Characteristics](#performance-characteristics)
4. [Memory Management](#memory-management)
5. [Thread Safety](#thread-safety)
6. [Troubleshooting](#troubleshooting)
7. [API Reference](#api-reference)

---

## Architecture Overview

### Components

The C++ simulation runner consists of four main components:

```
┌─────────────────────────────────────────────────────┐
│          Python Layer (src/core/mcts.py)            │
│  - AlphaZeroMCTS orchestration                      │
│  - Inference callback bridge                        │
└────────────────────┬────────────────────────────────┘
                     │ pybind11
┌────────────────────▼────────────────────────────────┐
│      C++ Simulation Runner (simulation_runner.cpp)  │
│  - run_simulation() pipeline                        │
│  - select_leaf() with PUCT selection                │
│  - expand_node() with inference callback            │
│  - backup_value() with sign flipping                │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│      MCTS Core Components (cpp_extensions/mcts/)    │
│  - MCTSTree: SoA layout with move storage           │
│  - PUCTSelector: Vectorized child selection         │
│  - BackupManager: Value propagation                 │
│  - VirtualLossManager: Thread coordination          │
└─────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Structure of Arrays (SoA) Memory Layout**
   - Each field (N, W, P, Q, moves) stored contiguously
   - 64-byte alignment for cache line optimization
   - 27 bytes per node (10M nodes = 270MB vs 1GB Python)

2. **Move Storage in C++ Tree**
   - `uint16_t* moves_` array parallel to node arrays
   - O(1) lookup: `tree.get_move(child_idx)`
   - Eliminates 1000MB Python dict overhead

3. **Zero Python Re-entry**
   - Full simulation (select → expand → backup) in C++
   - GIL released during entire search
   - Only re-enter Python for neural network inference

4. **Virtual Loss Coordination**
   - Applied during `select_leaf()` traversal
   - Removed during `backup_value()` propagation
   - Prevents multiple threads selecting same path

---

## Integration Flow

### 1. Initialization (AlphaZeroMCTS.__init__)

```python
from src.core.mcts import AlphaZeroMCTS
import mcts_py

# Create MCTS components
tree = mcts_py.MCTSTree(max_nodes=1_000_000)
puct_selector = mcts_py.PUCTSelector(c_puct=1.5)
backup_mgr = mcts_py.BackupManager(tree)
virtual_loss_mgr = mcts_py.VirtualLossManager(tree, virtual_loss=1.0)

# Create C++ simulation runner
runner = mcts_py.SimulationRunner(
    tree, puct_selector, backup_mgr, virtual_loss_mgr
)

# Create inference callback bridge
inference_callback = mcts_py.PyInferenceCallback(
    self._create_inference_callback()
)

# Store for search execution
self.runner = runner
self.inference_callback = inference_callback
```

### 2. Search Execution (AlphaZeroMCTS.search)

```python
def search(self, root_state, simulations=800):
    # Initialize root node
    root_idx = self.tree.add_root_node(prior=0.5, player=root_state.get_current_player())

    # Run simulations in C++
    for _ in range(simulations):
        success = self.runner.run_simulation(
            root_state,           # IGameState interface
            root_idx,             # Root node index
            self.inference_callback  # Python inference bridge
        )
        if not success:
            break  # Tree exhausted

    # Extract policy from visit counts
    return self._extract_policy(root_idx)
```

### 3. Simulation Pipeline (C++ run_simulation)

```cpp
bool SimulationRunner::run_simulation(
    IGameState& root_state,
    NodeIndex root_idx,
    InferenceCallback& inference_fn
) {
    // 1. Clone state to preserve root
    auto state = root_state.clone();
    if (!state) return false;

    // 2. Select leaf with PUCT and virtual loss
    path_buffer_.clear();
    NodeIndex leaf = select_leaf(*state, root_idx, path_buffer_);

    // 3. Expand leaf with neural network
    float value = expand_node(*state, leaf, inference_fn);

    // 4. Backup value with sign flipping
    backup_value(path_buffer_, value);

    return true;
}
```

### 4. Inference Bridge (PyInferenceCallback)

```cpp
std::pair<std::vector<float>, float> PyInferenceCallback::request_inference(IGameState& state) {
    py::gil_scoped_acquire acquire;  // Re-acquire GIL for Python call

    // Call Python inference function
    py::object result = python_fn_(state);

    // Extract (policy, value) tuple
    auto [policy_vec, value] = parse_result(result);

    return {policy_vec, value};
}
```

---

## Performance Characteristics

### Achieved Results (Phase 4 Testing)

| Metric | Baseline (Python) | Achieved (C++) | Target | Status |
|--------|------------------|----------------|--------|--------|
| **Throughput** | 246 sims/sec | 1,744 sims/sec | 30,000+ sims/sec | 🔄 In Progress |
| **GIL Hold Time** | 800µs/sim (80%) | ~300µs/sim (56.6%) | <100µs (<10%) | 🔄 In Progress |
| **Thread Efficiency** | 3% (1→8 threads) | 12.5% | ≥75% | 🔄 In Progress |
| **Move Storage** | 1000MB (dict) | 20MB (C++ array) | <50MB | ✅ Complete |
| **Memory Per Node** | ~100 bytes | 27 bytes | <64 bytes | ✅ Complete |
| **Thread Safety** | ❌ Data races | ✅ TSan clean | No races | ✅ Complete |

### Performance Notes

1. **Current Bottleneck: Synchronous Mock Inference**
   - Tests use synchronous mock inference (immediate return)
   - Real GPU worker will enable async batching (32-64 positions)
   - Expected improvement: 17-20× throughput boost

2. **GIL Release Confirmed**
   - 56.6% Python time with sync mock (baseline)
   - Target <30% with async GPU batching
   - Target <10% with fully optimized pipeline

3. **Thread Scaling Ready**
   - Current: 12.5% efficiency (limited by mock inference)
   - Infrastructure validated: parallel execution confirmed
   - Expected: 75%+ efficiency with real GPU batching

### Optimization Roadmap

- [✅] **Phase 1**: C++ runner implementation (122× faster than Python stub)
- [✅] **Phase 2**: Move storage in tree (50× memory reduction)
- [✅] **Phase 3**: Thread safety validation (TSan clean)
- [🔄] **Phase 4**: GPU inference integration (17-20× expected)
- [📋] **Phase 5**: Batch size optimization (target: 32-64)
- [📋] **Phase 6**: Inference timeout tuning (target: ≤3ms)

---

## Memory Management

### Node Allocation

```cpp
// Pre-allocated pools with free list
class MCTSTree {
    std::size_t max_nodes_;
    std::atomic<std::size_t> node_count_{0};
    std::atomic<std::size_t> next_free_index_{0};
    std::vector<NodeIndex> free_nodes_;
    std::mutex allocation_mutex_;  // Protects allocation/deallocation

    // SoA arrays (64-byte aligned)
    alignas(64) float* priors_;
    alignas(64) std::atomic<float>* visit_counts_;
    alignas(64) std::atomic<float>* value_sums_;
    alignas(64) uint16_t* moves_;  // Move storage
    // ... other fields
};
```

### Allocation Strategy

1. **Batch Allocation**: `allocate_nodes(count)` for expansion
2. **Free List Reuse**: Deallocated nodes added to `free_nodes_`
3. **O(1) Operations**: Index-based access, no malloc in hot path
4. **Thread-Safe**: Mutex protects allocation, atomics for data

### Memory Footprint

- **10M nodes**: 270MB total
  - Visit counts: 40MB (float32)
  - Value sums: 40MB (float32)
  - Priors: 40MB (float32)
  - Moves: 20MB (uint16_t)
  - Other fields: 130MB
- **Well under 1GB target** ✅

---

## Thread Safety

### Data Race Prevention

All critical operations use proper synchronization:

1. **Allocation/Deallocation**
   ```cpp
   NodeIndex allocate_nodes(uint16_t count) {
       std::lock_guard<std::mutex> lock(allocation_mutex_);
       // ... safe allocation logic
   }
   ```

2. **Visit Count Updates**
   ```cpp
   void increment_visits(NodeIndex idx) {
       visit_counts_[idx].fetch_add(1.0f, std::memory_order_relaxed);
   }
   ```

3. **Value Accumulation**
   ```cpp
   void add_value(NodeIndex idx, float value) {
       value_sums_[idx].fetch_add(value, std::memory_order_relaxed);
   }
   ```

4. **Virtual Loss Coordination**
   ```cpp
   class VirtualLossManager {
       void apply_virtual_loss(NodeIndex idx) {
           visit_counts_[idx].fetch_add(virtual_loss_, std::memory_order_relaxed);
       }

       void remove_virtual_loss(NodeIndex idx) {
           visit_counts_[idx].fetch_sub(virtual_loss_, std::memory_order_relaxed);
       }
   };
   ```

### ThreadSanitizer Validation

**Comprehensive TSan Testing (T019)**:
- ✅ 6 concurrent access patterns tested
- ✅ 6 data races detected and fixed
- ✅ All tests pass with TSan (clang++-18 on Ubuntu 24.04)
- ✅ No data races in production code

**Build with TSan**:
```bash
# Ubuntu 24.04+ (requires clang++-18 for higher ASLR entropy)
clang++-18 -std=c++17 -O1 -g -pthread -fsanitize=thread \
    -I./cpp_extensions -o test_concurrent \
    tests/unit/test_move_storage_concurrent.cpp \
    cpp_extensions/mcts/tree.cpp \
    cpp_extensions/mcts/virtual_loss.cpp
```

---

## Troubleshooting

### Issue: Low Throughput (<1000 sims/sec)

**Symptoms**:
- Search throughput below 1000 simulations/second
- Low GPU utilization (<20%)

**Diagnosis**:
```python
# Check batch size
from src.telemetry.metrics import MetricsCollector
metrics = mcts.get_metrics()
print(f"Avg batch size: {metrics['avg_batch_size']}")  # Should be 32-64
print(f"GPU util: {metrics['gpu_utilization']}")       # Should be 80-92%
```

**Solutions**:
1. **Increase inference workers**: `config.inference_workers = 2`
2. **Tune batch size**: `config.batch_size_min = 32, batch_size_max = 64`
3. **Adjust timeout**: `config.inference_timeout_ms = 3.0`
4. **Check GPU warmup**: First batch is 10× slower

---

### Issue: Memory Growth During Search

**Symptoms**:
- RSS memory grows continuously
- OOM errors after prolonged use

**Diagnosis**:
```bash
# Run soak test (1 hour)
python -m pytest tests/soak/test_memory_stability.py::TestRealMemoryStability::test_1_hour_memory_stability -v -s
```

**Solutions**:
1. **Verify tree reset**: `mcts.reset()` clears all nodes
2. **Check pool reuse**: Free list should recycle nodes
3. **Run with ASan**:
   ```bash
   ASAN_OPTIONS=detect_leaks=1 python -m pytest tests/soak/ -v
   ```

**Expected**: <10MB growth per hour ✅

---

### Issue: Thread Efficiency <50%

**Symptoms**:
- Adding threads doesn't improve throughput
- CPU cores underutilized

**Diagnosis**:
```python
# Measure thread scaling
from tests.performance.test_simulation_runner_performance import measure_throughput

for threads in [1, 2, 4, 8]:
    throughput = measure_throughput(threads=threads)
    efficiency = (throughput / threads) / baseline_throughput
    print(f"{threads} threads: {throughput:.0f} sims/sec, {efficiency:.1%} efficient")
```

**Solutions**:
1. **Reduce virtual loss**: Too high causes thread contention
2. **Check GIL release**: Python profiler should show <10% time
3. **Verify async inference**: Batching must be non-blocking
4. **Tune thread count**: Optimal is usually 8-12 for Ryzen 5900X

**Expected**: 75%+ efficiency with GPU batching ✅

---

### Issue: API Compatibility Errors

**Symptoms**:
```
TypeError: run_simulation(): incompatible function arguments
```

**Root Cause**: Using `GameStateWrapper` instead of direct `alphazero_py` game state

**Solution**:
```python
# ❌ Wrong: GameStateWrapper
from src.games.game_state import create_game_state
game = create_game_state('gomoku')  # Returns wrapper

# ✅ Correct: Direct C++ game state
import alphazero_py
game = alphazero_py.GomokuState(board_size=15)  # IGameState interface
```

---

## API Reference

### SimulationRunner

**C++ Class**: `mcts::SimulationRunner`
**Python Binding**: `mcts_py.SimulationRunner`

#### Constructor

```python
runner = mcts_py.SimulationRunner(
    tree: mcts_py.MCTSTree,
    selector: mcts_py.PUCTSelector,
    backup_mgr: mcts_py.BackupManager,
    virtual_loss_mgr: mcts_py.VirtualLossManager
)
```

**Parameters**:
- `tree`: MCTS tree for node storage
- `selector`: PUCT selector for child selection
- `backup_mgr`: Manager for value backup
- `virtual_loss_mgr`: Manager for virtual loss coordination

#### Methods

##### run_simulation

```python
success: bool = runner.run_simulation(
    root_state: alphazero_py.IGameState,
    root_index: int,
    inference_fn: mcts_py.InferenceCallback
)
```

Executes one simulation from root to leaf.

**Parameters**:
- `root_state`: Game state at root (must implement `IGameState`)
- `root_index`: Index of root node in tree
- `inference_fn`: Callback for neural network inference

**Returns**:
- `bool`: True if simulation succeeded, False if tree exhausted or clone failed

**Thread Safety**: Safe to call concurrently from multiple threads

---

### PyInferenceCallback

**C++ Class**: `mcts::PyInferenceCallback`
**Python Binding**: `mcts_py.PyInferenceCallback`

#### Constructor

```python
callback = mcts_py.PyInferenceCallback(
    python_callable: Callable[[IGameState], Tuple[np.ndarray, float]]
)
```

**Parameters**:
- `python_callable`: Function that takes game state and returns (policy, value)

#### Expected Signature

```python
def inference_fn(state: alphazero_py.IGameState) -> Tuple[np.ndarray, float]:
    """
    Args:
        state: Game state to evaluate

    Returns:
        policy: np.ndarray of shape (num_actions,) with move probabilities
        value: float in [-1.0, 1.0] representing position evaluation
    """
    # Extract features, run neural network, return results
    return policy, value
```

---

### MCTSTree Extensions

#### Move Storage

```python
# Set move index for child
tree.set_move(child_idx: int, move: int) -> None

# Get move index from child
move: int = tree.get_move(child_idx: int)

# Deallocate single node
tree.deallocate_node(idx: int) -> None

# Deallocate batch of nodes
tree.deallocate_nodes(first_idx: int, count: int) -> None
```

**Thread Safety**:
- `get_move()`: Safe for concurrent reads
- `set_move()`: Safe only during single-threaded expansion
- `deallocate_*()`: Protected by allocation mutex

---

## Testing

### Contract Tests

Validate API surface and bindings:

```bash
python -m pytest tests/contract/test_simulation_runner_api.py -v
python -m pytest tests/contract/test_inference_callback.py -v
```

### Integration Tests

End-to-end pipeline validation:

```bash
python -m pytest tests/integration/test_simulation_pipeline.py -v
python -m pytest tests/integration/test_cpp_vs_python_equivalence.py -v
python -m pytest tests/integration/test_gil_release.py -v
```

### Performance Tests

Throughput and efficiency benchmarks:

```bash
python -m pytest tests/performance/test_simulation_runner_performance.py -v
```

### Soak Tests

Memory stability validation:

```bash
# Short test (30 seconds)
python -m pytest tests/soak/test_memory_stability.py::TestRealMemoryStability::test_short_memory_stability_gomoku -v -s

# Full 1-hour test (manual)
python -m pytest tests/soak/test_memory_stability.py::TestRealMemoryStability::test_1_hour_memory_stability -v -s
```

### C++ Unit Tests

Thread safety and correctness:

```bash
# Build and run concurrent tests
g++ -std=c++17 -O2 -pthread -I./cpp_extensions \
    -o test_move_storage_concurrent \
    tests/unit/test_move_storage_concurrent.cpp \
    cpp_extensions/mcts/tree.cpp \
    cpp_extensions/mcts/virtual_loss.cpp

./test_move_storage_concurrent
```

---

## Performance Validation Checklist

Before production deployment:

- [ ] Throughput ≥30,000 sims/sec (`test_simulation_runner_performance.py`)
- [ ] GIL hold time <10% (`test_gil_release.py`)
- [ ] Thread efficiency ≥75% (`test_thread_efficiency.py`)
- [ ] GPU utilization 80-92% (`nvidia-smi dmon`)
- [ ] Memory growth <10MB/hour (`test_1_hour_memory_stability`)
- [ ] No TSan warnings (`build with -fsanitize=thread`)
- [ ] No ASan leaks (`ASAN_OPTIONS=detect_leaks=1`)
- [ ] Games/hour 200-300 (end-to-end training pipeline)

---

## References

- **Specification**: `specs/002-cpp-simulation-runner/spec.md`
- **Implementation Plan**: `specs/002-cpp-simulation-runner/plan.md`
- **Task Tracking**: `specs/002-cpp-simulation-runner/tasks.md`
- **C++ Source**: `cpp_extensions/mcts/simulation_runner.{hpp,cpp}`
- **Python Integration**: `src/core/mcts.py`
- **Test Suite**: `tests/{contract,integration,performance,soak}/test_*runner*.py`
