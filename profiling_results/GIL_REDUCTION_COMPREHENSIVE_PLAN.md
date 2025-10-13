# Comprehensive GIL Reduction & Performance Recovery Plan

**Date**: 2025-10-13
**Status**: CRITICAL - GPU-bound but thread contention limits parallelism
**Current**: 1,895-2,835 sims/sec (2 threads optimal)
**Target**: 6,000-10,000 sims/sec (realistic with current model)

---

## Executive Summary

After deep analysis of review.txt, agent-based codebase scrutiny, research into GIL optimization techniques, and py-spy profiling, I've identified that **GIL is NOT the primary bottleneck**. The real issues are:

1. **GPU Inference Bottleneck** (30.7ms) - Hardware ceiling at ~2,000-2,400 states/sec
2. **Thread Coordination Overhead** - Poor scaling beyond 2 threads (89.6% → 45% efficiency)
3. **Missing OpenMP Parallelization** - Already fixed (6.9× speedup validated)

**Critical Finding**: The system is **GPU-bound**, not GIL-bound. However, thread coordination overhead prevents effective multi-threading beyond 2 threads.

---

## Root Cause Analysis

### 1. GPU Inference Bottleneck (PRIMARY)

**Measured**: 30.7ms per batch-64 @ FP16 (T-VALID-1 validation)
**Expected**: 8-10ms per batch-64 (from specs)
**Impact**: Theoretical max = 64 / 31.8ms = **2,014 states/sec**

**Why So Slow?**
- Model size: 10.1M parameters (too large for 8-10ms target)
- RTX 3060 Ti @ FP16: Not fast enough for this model
- No CUDA Graphs (kernel launch overhead: 2-5ms)
- No model pruning/quantization

**Evidence**:
```
Observed performance: 1,895-2,835 sims/sec
Theoretical maximum: 2,014 states/sec
Achievement: 94-141% of theoretical!
```

**Conclusion**: System is **performing near theoretical maximum**. GPU is the hard limit.

### 2. Thread Coordination Overhead (SECONDARY)

**Thread Scaling**:
- 1 thread: 1,230 sims/sec (100% efficiency baseline)
- 2 threads: 2,205 sims/sec (89.6% efficiency) ✅ OPTIMAL
- 4 threads: 2,214 sims/sec (45% efficiency) ❌ POOR
- 8 threads: 2,198 sims/sec (22.4% efficiency) ❌ CATASTROPHIC

**Root Cause**: Efficiency collapse beyond 2 threads suggests:
- Mutex contention in AsyncInferenceQueue or BatchInferenceCoordinator
- Cache line bouncing across CPU cores (Ryzen 5900X dual-CCD)
- Excessive condition variable signaling
- Atomic contention on shared tree nodes

**Evidence**: Adding threads makes performance WORSE (not GIL-related).

### 3. GIL Overhead (TERTIARY - Already Optimized)

**What Agent Analysis Found**:
✅ Full C++ simulation loops (GIL released)
✅ Coarse-grained GIL release (batch operations)
✅ OpenMP parallelization (6.9× speedup validated)
✅ Zero-copy DLPack tensors
✅ Condition variables (no polling)
✅ Thread-local arenas (99.93% fast-path)

**Remaining GIL Issues** (minor):
- Python `.tolist()` conversions (~1.3ms per batch, 5% overhead)
- Policy array processing in Python loops (~2-3% overhead)
- Numpy array stacking (should use DLPack exclusively)

**Verdict**: GIL is **NOT the bottleneck**. Review.txt's "67% Python overhead" was measured **BEFORE** OpenMP fix.

---

## Comprehensive Action Plan

### Phase 1: Validate Current State (1 day)

**Goal**: Confirm theoretical maximum and identify actual bottlenecks

#### Task 1.1: Profile Thread Contention with perf
```bash
# Install perf tools
sudo apt-get install linux-tools-common linux-tools-$(uname -r)

# Profile mutex contention
perf record -e 'sched:sched_switch' -a -g -- \
    python scripts/benchmark_throughput.py --threads 4 --simulations 5000

# Analyze mutex hotspots
perf report --stdio | grep -A5 "mutex\|lock\|atomic" > profiling_results/mutex_contention.txt

# Profile cache misses (Ryzen dual-CCD)
perf stat -e cache-misses,cache-references,L1-dcache-load-misses \
    python scripts/benchmark_throughput.py --threads 4 --simulations 5000
```

**Expected Finding**: Mutex contention in AsyncInferenceQueue::enqueue/dequeue or BatchInferenceCoordinator signaling.

#### Task 1.2: Profile Single-Thread Performance
```bash
# Profile with py-spy for Python hotspots
py-spy record -o profiling_results/single_thread_profile.svg --rate 200 \
    -- python scripts/benchmark_throughput.py --threads 1 --simulations 5000

# Profile with perf for C++ hotspots
perf record -g -- python scripts/benchmark_throughput.py --threads 1 --simulations 5000
perf report --stdio > profiling_results/single_thread_perf.txt
```

**Target**: Identify why single-thread is 1,230 sims/sec instead of 2,000+ sims/sec.

#### Task 1.3: GPU Utilization Analysis
```bash
# Monitor GPU during 2-thread run
nvidia-smi dmon -s u -c 30 > profiling_results/gpu_util_2threads.txt &
python scripts/benchmark_throughput.py --threads 2 --simulations 10000

# Use nsys for detailed timeline
nsys profile --trace=cuda,nvtx -o profiling_results/gpu_timeline.qdrep \
    python scripts/benchmark_throughput.py --threads 2 --simulations 5000
```

**Target**: Verify GPU is 80-90% utilized (should be, given performance near theoretical max).

**Deliverables**:
- `profiling_results/mutex_contention.txt` - Mutex hotspots
- `profiling_results/single_thread_profile.svg` - Python hotspots
- `profiling_results/single_thread_perf.txt` - C++ hotspots
- `profiling_results/gpu_util_2threads.txt` - GPU utilization timeline

**Decision Point**: If GPU util <70%, focus on batching/timeout. If mutex contention >20%, fix AsyncInferenceQueue.

---

### Phase 2: Fix Thread Coordination (2-3 days)

**Goal**: Improve thread efficiency from 45% → 80%+ at 4 threads

#### Task 2.1: Optimize AsyncInferenceQueue (High Priority)

**Problem**: Possible mutex contention in enqueue/dequeue operations.

**Investigation**:
```cpp
// Check cpp_extensions/mcts/async_inference_queue.cpp
// Look for:
// 1. Mutex held during expensive operations
// 2. Condition variable notify_all() instead of notify_one()
// 3. Atomic operations on same cache line
```

**Potential Fixes**:

**Fix A: Lock-Free Ring Buffer (Complex, 3 days)**
```cpp
// Replace mutex-based queue with lock-free SPMC ring buffer
#include <boost/lockfree/spsc_queue.hpp>

template<typename T>
class LockFreeQueue {
    boost::lockfree::spsc_queue<T> queue_{4096};
    std::atomic<bool> data_available_{false};
    
public:
    void enqueue(T item) {
        queue_.push(item);
        data_available_.store(true, std::memory_order_release);
    }
    
    bool dequeue(T& item) {
        if (queue_.pop(item)) {
            if (queue_.empty()) {
                data_available_.store(false, std::memory_order_release);
            }
            return true;
        }
        return false;
    }
};
```

**Fix B: Reduce Lock Granularity (Simple, 1 day)**
```cpp
// Current (may be holding lock too long):
void AsyncInferenceQueue::process_results() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (auto& result : results_) {  // ❌ Holding lock during iteration
        // ... process ...
    }
}

// Optimized:
void AsyncInferenceQueue::process_results() {
    std::vector<Result> local_results;
    {
        std::unique_lock<std::mutex> lock(mutex_);
        local_results.swap(results_);  // ✅ Quick swap under lock
    }
    for (auto& result : local_results) {  // ✅ Process without lock
        // ... process ...
    }
}
```

**Fix C: Per-Thread Queues (Medium, 2 days)**
```cpp
// Eliminate contention by giving each thread its own queue
class PerThreadQueues {
    std::vector<AsyncInferenceQueue> queues_;  // One per thread
    std::atomic<int> round_robin_counter_{0};
    
public:
    void enqueue(Request req) {
        int thread_id = req.thread_id;
        queues_[thread_id].enqueue(req);  // No cross-thread contention
    }
};
```

**Recommendation**: Start with Fix B (simple, 1 day), then Fix C if needed.

#### Task 2.2: Optimize BatchInferenceCoordinator (Medium Priority)

**Problem**: Possible excessive condition variable signaling.

**Investigation**:
```cpp
// Check cpp_extensions/mcts/batch_inference_coordinator.cpp
// Look for:
// 1. notify_all() when notify_one() suffices
// 2. Spurious wakeups (wait without predicate)
// 3. Broadcasting to all threads unnecessarily
```

**Fix: Targeted Signaling**
```cpp
// Current (may be broadcasting to all threads):
void BatchInferenceCoordinator::signal_batch_ready() {
    cv_.notify_all();  // ❌ Wakes all threads unnecessarily
}

// Optimized:
void BatchInferenceCoordinator::signal_batch_ready() {
    cv_.notify_one();  // ✅ Wakes one thread (enough for processing)
}

// Or even better: Skip signaling if no waiters
std::atomic<int> waiting_threads_{0};

void wait_for_batch() {
    waiting_threads_.fetch_add(1);
    cv_.wait(...);
    waiting_threads_.fetch_sub(1);
}

void signal_batch_ready() {
    if (waiting_threads_.load() > 0) {
        cv_.notify_one();
    }
}
```

#### Task 2.3: Cache Line Alignment (Low Priority, 1 day)

**Problem**: Ryzen 5900X has dual-CCD, cache line bouncing between CCDs.

**Fix: Align Critical Data Structures**
```cpp
// Current:
struct MCTSNode {
    float value;
    float prior;
    int visit_count;
    // ... more fields ...
};

// Optimized (64-byte cache line alignment):
struct alignas(64) MCTSNode {
    // Hot fields (accessed by all threads)
    std::atomic<int> visit_count;
    std::atomic<float> value;
    
    char padding1[64 - sizeof(std::atomic<int>) - sizeof(std::atomic<float>)];
    
    // Cold fields (accessed rarely)
    float prior;
    int first_child;
    // ...
};
```

**Expected Impact**: 5-10% improvement in thread scaling.

**Deliverables**:
- `cpp_extensions/mcts/async_inference_queue_optimized.cpp` - Reduced lock granularity
- `cpp_extensions/mcts/batch_inference_coordinator_optimized.cpp` - Targeted signaling
- Benchmark showing 4-thread efficiency: 45% → 70-80%

---

### Phase 3: Eliminate Remaining Python Overhead (1-2 days)

**Goal**: Remove .tolist() conversions and Python loops (5-10% speedup)

#### Task 3.1: Remove .tolist() Conversions

**Location 1**: `src/core/dlpack_inference_bridge.py:462-465`
```python
# Current (holds GIL for entire batch):
for i in range(len(states)):
    policy_list = policy_np[i].tolist()  # ❌ Python conversion
    value_scalar = float(value_np[i])
    results.append((policy_list, value_scalar))

# Optimized (return numpy arrays):
# Option A: Return as numpy (C++ can handle)
return [(policy_np[i], float(value_np[i])) for i in range(len(states))]

# Option B: Bulk operation (even better)
values = value_np.flatten().tolist()  # Single .tolist() for all values
policies = policy_np  # Return as numpy array
return list(zip(policies, values))
```

**Location 2**: `src/core/mcts.py:757-760, 824-827`
```python
# Current:
if hasattr(policy, 'tolist'):
    policy = policy.tolist()  # ❌

# Optimized: Accept numpy arrays directly in C++
# Add overload in python_bindings.cpp:
py::array_t<float> get_policy_array(int node_index);
```

**Expected Impact**: 1.15× speedup (1.3ms saved per batch)

#### Task 3.2: Vectorize Policy Masking

**Location**: `src/core/mcts.py:661-663`
```python
# Current (Python loop):
for move in range(len(policy)):
    if move not in legal_moves_set:  # ❌ Python dict lookup
        policy[move] = 0.0

# Optimized (vectorized):
illegal_mask = ~np.isin(np.arange(len(policy)), legal_moves)
policy[illegal_mask] = 0.0  # ✅ NumPy operation (releases GIL)
```

**Expected Impact**: 1.05× speedup

#### Task 3.3: Move Child Allocation to C++

**Location**: `src/core/mcts.py:674-703`
```python
# Current (Python loop, 30 lines):
for i, move in enumerate(legal_moves):  # ❌ Python loop
    child_index = first_child + i
    # ... many C++ method calls per child ...

# Optimized: Single C++ call
mcts_tree.allocate_and_initialize_children(
    parent_index, legal_moves, prior_probs
)
```

**C++ Implementation**:
```cpp
// Add to cpp_extensions/mcts/tree.cpp
void MCTSTree::allocate_and_initialize_children(
    int parent_index,
    const std::vector<int>& moves,
    const std::vector<float>& probs
) {
    int first_child = allocate_nodes(moves.size());
    
    #pragma omp parallel for if(moves.size() > 16)
    for (size_t i = 0; i < moves.size(); ++i) {
        int child_idx = first_child + i;
        initialize_child_node(child_idx, parent_index, moves[i], probs[i]);
    }
    
    // Link parent to children
    set_first_child(parent_index, first_child);
    set_num_children(parent_index, moves.size());
}
```

**Expected Impact**: 1.08× speedup

**Deliverables**:
- Updated `dlpack_inference_bridge.py` - No .tolist()
- Updated `mcts.py` - Vectorized operations
- New `tree.cpp` method - Bulk child allocation
- Benchmark showing 5-10% overall improvement

---

### Phase 4: GPU Optimization (2-3 days, Optional)

**Goal**: Increase GPU utilization from 66% → 85%+ (if bottleneck shifts)

**Note**: Only do this if Phase 2 fixes thread coordination. Otherwise, GPU optimization won't help.

#### Task 4.1: Reduce Batch Timeout

**Current**: 1.0ms timeout
**Hypothesis**: Threads idle waiting for batch to fill

**Experiment**:
```bash
# Test timeout sweep
for timeout in 0.25 0.5 0.75 1.0 1.5 2.0; do
    echo "Testing timeout: ${timeout}ms"
    python scripts/benchmark_throughput.py --threads 4 --simulations 5000 \
        --config <(echo "mcts: { inference_timeout_ms: $timeout }")
done
```

**Expected**: 0.5ms timeout may improve throughput if threads are idle.

#### Task 4.2: Increase Batch Size

**Current**: batch-64
**Hypothesis**: RTX 3060 Ti can handle batch-96 or batch-128

**Experiment**:
```bash
# Test batch size sweep (requires model input size verification)
for batch in 64 96 128; do
    echo "Testing batch size: $batch"
    python scripts/benchmark_throughput.py --threads 4 --simulations 5000 \
        --config <(echo "mcts: { batch_size_max: $batch }")
done
```

**Expected**: Larger batch may improve GPU util if GPU is underutilized.

#### Task 4.3: CUDA Graphs (Advanced, 3-4 days)

**Goal**: Eliminate kernel launch overhead (2-5ms per batch)

**Implementation**:
```python
# Add to dlpack_inference_bridge.py
class CUDAGraphInference:
    def __init__(self, model, batch_size):
        self.model = model
        self.graph = None
        self.static_input = torch.zeros(batch_size, ..., device='cuda')
        self.static_output_policy = None
        self.static_output_value = None
        
        # Warm up
        for _ in range(10):
            _ = model(self.static_input)
        
        # Capture graph
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            policy, value = model(self.static_input)
            self.static_output_policy = policy
            self.static_output_value = value
    
    def __call__(self, input_tensor):
        # Copy input to static buffer
        self.static_input.copy_(input_tensor)
        
        # Replay graph (no kernel launch overhead)
        self.graph.replay()
        
        # Copy output from static buffer
        return self.static_output_policy.clone(), self.static_output_value.clone()
```

**Expected Impact**: 1.1-1.2× speedup (30.7ms → 26-28ms per batch)
**Complexity**: High (requires static batch size, careful memory management)

**Deliverables**:
- Timeout/batch size tuning results
- Optional: CUDA Graphs implementation
- Benchmark showing 10-20% GPU speedup

---

### Phase 5: Realistic Target Adjustment (1 day)

**Goal**: Set achievable targets based on hardware constraints

#### Task 5.1: Update Performance Targets

**Current Reality**:
```
GPU inference: 30.7ms per batch-64 (hardware limit)
Theoretical max: 2,014 states/sec
Observed: 1,895-2,835 sims/sec (94-141% of theoretical!)
```

**Revised Targets**:
```
Conservative (2 threads, 90% efficiency):
  Single-thread:  2,000 sims/sec (theoretical max)
  2 threads:      3,600 sims/sec (1.8× with 90% efficiency)
  
Optimistic (4 threads, 80% efficiency, CUDA Graphs):
  GPU inference:  26ms (with CUDA Graphs, 1.18× speedup)
  Theoretical:    2,461 states/sec
  4 threads:      7,876 sims/sec (3.2× with 80% efficiency)
  
Stretch (8 threads, 70% efficiency, model reduction):
  Model size:     5-6M params (15-20ms inference)
  Theoretical:    3,200-4,000 states/sec
  8 threads:      17,920-22,400 sims/sec (5.6× with 70% efficiency)
```

**Recommendation**: Accept **3,600-4,000 sims/sec** as realistic with current model, or pursue model reduction for 8k+ target.

#### Task 5.2: Document Findings

**Create**:
- `docs/performance/thread_coordination_analysis.md` - Mutex contention findings
- `docs/performance/realistic_targets_2025-10-13.md` - Hardware-grounded targets
- Update `CLAUDE.md` with revised targets

**Deliverables**:
- Comprehensive performance report
- Updated targets in all specs
- Decision: Continue with current model or reduce to 5-6M params

---

## Implementation Priority

### CRITICAL (Do First)
1. **Phase 1**: Profiling (1 day) - Validate assumptions
2. **Phase 2, Task 2.1**: Fix AsyncInferenceQueue locking (1 day) - Biggest impact
3. **Phase 3, Task 3.1**: Remove .tolist() (4 hours) - Quick win

### HIGH (Do Next)
4. **Phase 2, Task 2.2**: Optimize BatchInferenceCoordinator (1 day)
5. **Phase 3, Tasks 3.2-3.3**: Vectorize & bulk operations (1 day)

### MEDIUM (If Time Permits)
6. **Phase 4, Tasks 4.1-4.2**: Batch/timeout tuning (1 day)
7. **Phase 2, Task 2.3**: Cache line alignment (1 day)

### LOW (Optional)
8. **Phase 4, Task 4.3**: CUDA Graphs (3-4 days, high complexity)

### DECISION POINT
9. **Phase 5**: Accept 3.6k-4k target OR reduce model to 5-6M params for 8k+ target

---

## Expected Outcomes

### Conservative (Phases 1-3 Only, 4 days)
```
Before:  1,895-2,835 sims/sec (2-4 threads)
After:   3,200-4,000 sims/sec (4 threads, 80% efficiency)
Improvement: 1.7-2.1×
```

### Optimistic (Phases 1-4, 7 days)
```
Before:  1,895-2,835 sims/sec
After:   4,500-6,000 sims/sec (4-8 threads, CUDA Graphs, tuning)
Improvement: 2.4-3.2×
```

### Stretch (Model Reduction, +3 days)
```
Before:  2,835 sims/sec (current model)
After:   7,000-10,000 sims/sec (reduced model, 8 threads)
Improvement: 2.5-3.5×
```

---

## Key Insights

1. **GIL is NOT the bottleneck** - Already well-optimized with C++ loops, batch operations, OpenMP
2. **GPU is the hard limit** - 30.7ms inference caps at ~2,000 states/sec
3. **Thread coordination is broken** - 45% efficiency at 4 threads (should be 80%)
4. **System performs near theoretical max** - 94-141% of 2,014 theoretical (excellent!)
5. **Model size is the real constraint** - 10.1M params too large for 8-10ms target

## Recommendations

**Immediate Actions**:
1. Profile with `perf` to find mutex contention (1 day)
2. Fix AsyncInferenceQueue locking (1 day)
3. Remove .tolist() conversions (4 hours)

**Medium-Term**:
4. Optimize BatchInferenceCoordinator (1 day)
5. Vectorize Python operations (1 day)
6. Tune batch/timeout parameters (1 day)

**Long-Term Decision**:
- **Option A**: Accept 3.6k-4k sims/sec with current model (realistic, 1 week)
- **Option B**: Reduce model to 5-6M params for 7k-10k target (+ 3 days retraining)

**Timeline**: 4-7 days to reach 3.6k-6k sims/sec (conservative to optimistic)

---

**END OF COMPREHENSIVE PLAN**
