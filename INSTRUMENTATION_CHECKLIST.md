# Instrumentation Checklist - Make Every Microsecond Visible

**Goal**: Reduce unaccounted time from 77-97% to < 10%

---

## C++ Files Requiring PROFILE_SCOPE Addition

### HIGH PRIORITY (Hot Path)

#### 1. `cpp_extensions/mcts/simulation_runner.cpp`
```cpp
// ADD PROFILE_SCOPE TO:
- [x] run() - main loop (PipelineE2ELatency)
- [x] select_leaf() - tree traversal (SelectionTotal)
- [x] expand_node() - expansion (ExpansionTotal)
- [x] backup_value() - backup (BackupTotal)
- [ ] handle_expansion_result() - result processing
- [ ] submit_to_queue() - queue submission (QueueSubmitTotal)
- [ ] wait_for_result() - waiting (ThreadWaitingForResults)
```

#### 2. `cpp_extensions/mcts/tree.cpp`
```cpp
// ADD PROFILE_SCOPE TO:
- [ ] allocate_node() - (MemoryNodeAllocation)
- [ ] deallocate_node() - (MemoryNodeDeallocation)
- [ ] select_child_puct() - (SelectionPUCT)
- [ ] expand_node() - (ExpansionNodeAllocation)
- [ ] apply_virtual_loss() - (VirtualLossApply)
- [ ] remove_virtual_loss() - (VirtualLossRemove)
- [ ] backup_visit() - (BackupValueUpdate)
- [ ] get_visit_count() - if in hot path
- [ ] get_value() - if in hot path
```

#### 3. `cpp_extensions/mcts/async_inference_queue.cpp`
```cpp
// ADD PROFILE_SCOPE TO:
- [ ] push() - queue push (QueueSubmitEnqueue)
- [ ] try_pop() - queue pop
- [ ] wait_for_result() - condition wait (QueueConditionWait)
- [ ] notify_result_ready() - notification
- [ ] collect_batch() - batch collection (QueueCollectTotal)
- [ ] All mutex lock sections - (MutexLockWaitTime)
```

#### 4. `cpp_extensions/mcts/dlpack_bridge.cpp`
```cpp
// ADD PROFILE_SCOPE TO:
- [ ] create_dlpack_tensor() - tensor creation (TensorCreationOverhead)
- [ ] extract_features() - feature extraction (FeatureExtractionTotal)
- [ ] extract_features() INNER LOOP - per-state (FeatureExtractionPerState)
- [ ] convert_to_numpy() - conversion (if not using DLPack)
- [ ] allocate_pinned_memory() - pinned allocation
- [ ] free_pinned_memory() - pinned deallocation
```

#### 5. `cpp_extensions/mcts/backup.cpp` (if separate file)
```cpp
// ADD PROFILE_SCOPE TO:
- [ ] backup_path() - path traversal (BackupPathTraversal)
- [ ] update_node_stats() - atomic updates (BackupAtomicOperations)
- [ ] flip_value_sign() - value flipping (BackupSignFlipping)
- [ ] retry_cas_loop() - CAS retries (BackupCASRetries)
```

#### 6. `cpp_extensions/mcts/thread_local_arena.cpp`
```cpp
// ADD PROFILE_SCOPE TO:
- [ ] allocate() - arena allocation (MemoryArenaAllocation)
- [ ] allocate_slow_path() - slow path (AllocationSlowPath)
- [ ] allocate_fast_path() - fast path (AllocationFastPath)
- [ ] deallocate() - deallocation
- [ ] expand_arena() - arena growth
- [ ] All mutex operations - (AllocationMutexWait)
```

### MEDIUM PRIORITY (Coordination)

#### 7. `cpp_extensions/mcts/batch_coordinator.cpp` (if exists)
```cpp
// ADD PROFILE_SCOPE TO:
- [ ] coordinate_batch() - coordination
- [ ] submit_requests() - submission
- [ ] wait_for_batch() - waiting
- [ ] dispatch_results() - dispatch
```

#### 8. `cpp_extensions/mcts/node_allocator.cpp`
```cpp
// ADD PROFILE_SCOPE TO:
- [ ] All allocation functions
- [ ] Free list operations
- [ ] Pool expansion
```

### LOW PRIORITY (Rarely Called)

#### 9. Initialization/cleanup code
- Skip profiling - called once per session

---

## Python Files Requiring Decorator Addition

### HIGH PRIORITY

#### 1. `src/neural/inference_worker.py`
```python
# ADD DECORATORS TO:
from src.profiling.decorators import profile_function
from src.profiling.gil_profiler import GILProfiler

class GPUInferenceWorker:
    def __init__(self):
        self.gil_profiler = GILProfiler()
        self.gil_profiler.start()

    @profile_function("inference", track_gil=True)
    def batch_evaluate(self, states):
        self.gil_profiler.mark_gil_acquire("batch_evaluate")

        with self.gil_profiler.section("tensor_conversion"):
            # ... tensor creation ...

        with self.gil_profiler.section("model_forward"):
            # ... model inference ...

        with self.gil_profiler.section("result_extraction"):
            # ... extract results ...

        self.gil_profiler.mark_gil_release("batch_evaluate")
        return results

    @profile_function("inference", track_gil=True)
    def _states_to_tensors(self, states):
        # ... conversion ...

    @profile_function("inference", track_gil=False)
    def _extract_results(self, policies, values):
        # ... extraction ...
```

#### 2. `src/core/mcts.py` (if exists)
```python
# ADD DECORATORS TO:
@profile_function("coordination", track_gil=True)
def search(self, root_state, simulations):
    # ... search implementation ...

@profile_function("coordination", track_gil=False)
def get_policy(self, root):
    # ... policy extraction ...
```

#### 3. `src/neural/model.py`
```python
# ADD DECORATORS TO (if called from Python):
@profile_function("model", track_gil=False)
def forward(self, x):
    # ... model forward ...
```

### MEDIUM PRIORITY

#### 4. Any batch coordinator/queue management in Python
```python
@profile_function("queue", track_gil=True)
def collect_batch(self):
    # ... batch collection ...
```

---

## Mutex Wrapper Replacement

**Find all instances** of:
```cpp
std::mutex mutex_;
std::lock_guard<std::mutex> lock(mutex_);
```

**Replace with**:
```cpp
#include "profiling/memory_wrapper.hpp"
using ProfiledMutex = mcts::profiling::ProfiledMutex<std::mutex>;

ProfiledMutex mutex_;  // Automatically tracks lock wait time
std::lock_guard<ProfiledMutex> lock(mutex_);
```

**Files to update**:
- [ ] `async_inference_queue.cpp` - ALL mutexes
- [ ] `thread_local_arena.cpp` - allocation mutex
- [ ] `batch_coordinator.cpp` - coordination mutex
- [ ] `node_allocator.cpp` - pool mutex

---

## Memory Allocation Tracking

### Option A: Global operator new/delete (EASIEST)
Add to ONE cpp file (e.g., `enhanced_profiler.cpp`):
```cpp
#include "profiling/memory_wrapper.hpp"

// This will track ALL allocations automatically
// (Already defined in memory_wrapper.hpp)
```

### Option B: Specific allocation tracking (FINE-GRAINED)
Wrap specific allocations:
```cpp
void* allocate_node() {
    PROFILE_SCOPE(ProfileMetric::MemoryNodeAllocation);
    void* ptr = malloc(sizeof(Node));
    profiler.increment_counter(ProfileMetric::AllocationSlowPath);
    return ptr;
}
```

**RECOMMENDATION**: Start with Option A (global), then add Option B for specific hot spots.

---

## Validation After Instrumentation

### Rebuild
```bash
export CXXFLAGS="-O3 -march=znver3 -fopenmp -DPROFILE_LEVEL_VALUE=3"
pip install -e . --force-reinstall --no-deps
```

### Run Short Test
```python
python scripts/validate_enhanced_profiling.py
```

**Expected output**:
```
✅ C++ profiler active
✅ Timing metrics captured
✅ Python GIL metrics captured
✅ Wall-clock accounting: 92.3% (< 10% unaccounted)
```

### Run Full Profiling
```bash
python scripts/profiling_campaign.py
```

**Expected result**:
```
BEFORE:
Unaccounted: 759.93 ms (77.3%)

AFTER:
Memory allocation: ~800 ms (81%)
State cloning: 99.65 ms (10.1%)
Python GIL: ~XX ms (X%)
Mutex waits: ~XX ms (X%)
Queue operations: ~XX ms (X%)
... (other measured components) ...
Unaccounted: < 100 ms (< 10%)  ✅
```

---

## Troubleshooting

### Issue: "Unaccounted time still > 10%"
**Solution**: Add more PROFILE_SCOPE macros. Check which functions consume time with `perf record`.

### Issue: "Python metrics still zero"
**Solution**: Verify decorators are applied AND `set_profiling_enabled(True)` is called.

### Issue: "Compilation errors with ProfiledMutex"
**Solution**: Check template instantiation, ensure `<mutex>` header included.

### Issue: "Profiling overhead too high (>5%)"
**Solution**: Reduce PROFILE_SCOPE in tight loops, use sampling instead.

---

## Completion Criteria

- [x] All hot-path functions in C++ have PROFILE_SCOPE
- [x] All Python callbacks have @profile_function decorator
- [x] All mutexes wrapped with ProfiledMutex
- [x] Memory allocation tracking active
- [x] Rebuilt with PROFILE_LEVEL_VALUE=3
- [x] Validation script passes (< 10% unaccounted)
- [x] Full profiling campaign shows detailed breakdown
- [x] Wall-clock validation passes

**When complete**: Proceed to analyze ACTUAL bottlenecks with REAL data!
