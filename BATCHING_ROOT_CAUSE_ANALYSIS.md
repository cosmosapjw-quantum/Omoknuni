# Batching Root Cause Analysis
**Date**: 2025-10-14
**Issue**: Batch size = 1.0 instead of target 32-64
**Impact**: Throughput capped at 2,923 sims/sec instead of 6,000-10,000 target

---

## Root Cause Identified

### The Problem

In [src/core/mcts.py:295-296](src/core/mcts.py#L295-L296):

```python
self._coordinator.start(self.async_queue, self._batch_callback,
                       self.async_batch_size, self.async_timeout_ms)
```

**Current Parameters**:
- `batch_size` = 32 (from `self.async_batch_size`)
- `timeout_ms` = 2.0 (from `self.async_timeout_ms`)

### The Bug

In [cpp_extensions/mcts/async_inference_queue.cpp:65-77](cpp_extensions/mcts/async_inference_queue.cpp#L65-L77):

```cpp
std::vector<InferenceRequest> AsyncInferenceQueue::collect_batch(size_t min_batch_size,
                                                                   double timeout_ms) {
    std::vector<InferenceRequest> batch;

    const auto max_batch_size = (min_batch_size > 0)
        ? (min_batch_size + (min_batch_size / 2))  // ← 32 + 16 = 48
        : 4096;  // Max queue capacity

    const auto timeout_duration = duration<double, std::milli>(timeout_ms);
    const auto deadline = steady_clock::now() + timeout_duration;

    // T006c: Wait for min_batch_size with timeout using condition variable
    if (min_batch_size > 0 && timeout_ms > 0.0) {
        while (batch.size() < min_batch_size && !shutting_down_.load(...)) {
            // ← This loop waits until batch.size() >= 32
```

**The Issue**: The `collect_batch()` function expects `min_batch_size` to be the **minimum** threshold before submitting, and it will **wait** until:
1. Batch reaches `min_batch_size` (32), OR
2. Timeout expires (2.0ms)

**But the coordinator is being called from a single thread** in [batch_inference_coordinator.cpp:108](cpp_extensions/mcts/batch_inference_coordinator.cpp#L108):

```cpp
void BatchInferenceCoordinator::coordinator_loop() {
    while (running_.load(std::memory_order_acquire)) {
        // This blocks waiting for min_batch_size=32 OR timeout=2.0ms
        std::vector<InferenceRequest> batch = queue_->collect_batch(batch_size_, timeout_ms_);
```

### Why Batch Size = 1.0

Looking at the benchmark results:
- **Inference calls**: 801 for 800 simulations
- **Batch size**: min=1, max=1, avg=1.0

**Root Cause Chain**:

1. **Coordinator thread calls `collect_batch(32, 2.0)`**
2. **Simulation threads submit requests individually** (no coordination)
3. **Condition variable wait strategy**:
   - Waits up to 2.0ms for batch.size() >= 32
   - If only 1-2 requests arrive in 2.0ms window → submits batch-1 or batch-2
   - **Timeout is TOO SHORT** for accumulation

4. **Why only 1 request per batch?**
   - With 8 simulation threads, each thread:
     - Submits request → waits for result → continues
   - Threads are **serialized waiting for GPU** instead of accumulating
   - Coordinator sees: 1 request → timeout after 2.0ms → submit batch-1
   - Result → GPU processes batch-1 → thread gets result → repeat

### The Real Architecture Problem

**Current (Broken)**:
```
Thread 1: submit() → wait_for_result(req_1) → BLOCKING
Thread 2: [idle, waiting for Thread 1]
Thread 3: [idle, waiting for Thread 1]
...

Coordinator: collect_batch(min=32) → timeout after 2.0ms → batch=[req_1] → GPU
            ↑ Only 1 request in queue because threads are serialized!
```

**Expected (Working)**:
```
Thread 1: submit(req_1) → continue selecting next leaf
Thread 2: submit(req_2) → continue selecting next leaf
Thread 3: submit(req_3) → continue selecting next leaf
...
[All threads continue submitting without blocking]

Coordinator: collect_batch(min=32, timeout=2.0ms)
            ↓ Accumulates 32+ requests in queue
            batch=[req_1..req_32+] → GPU (SINGLE BATCH)
            ↓ Results ready
            All threads: check results → backup → continue
```

### The Missing Piece

Looking at [continuous_simulation_runner.cpp:186-220](cpp_extensions/mcts/continuous_simulation_runner.cpp#L186-L220):

```cpp
int ContinuousSimulationRunner::process_completed_results(AsyncInferenceQueue& queue) {
    // ...
    for (size_t i = 0; i < PENDING_BUFFER_CAPACITY; ++i) {
        PendingSlot& slot = pending_buffer_[i];

        // Try to get result for this specific request
        auto result_opt = queue.try_get_result(slot.request_id);
        if (!result_opt.has_value()) {
            continue;  // Result not ready yet ← Threads should CONTINUE submitting
        }
        // ...
    }
}
```

**The design LOOKS correct** - threads should continue submitting while waiting for results.

But looking at [continuous_simulation_runner.cpp:70-174](cpp_extensions/mcts/continuous_simulation_runner.cpp#L70-L174):

```cpp
while (completed < num_simulations) {
    bool waiting_for_leaf = false;

    // Phase 1: Select to leaf and submit inference (NON-BLOCKING)
    if (submitted < num_simulations) {
        // ... submission logic ...
        submitted++;
    }

    // Phase 2: Process completed results (NON-BLOCKING)
    int processed = process_completed_results(queue);
    completed += processed;

    // ⚠️ PROBLEM: Yield if no results available
    if (processed == 0) {
        bool all_submitted = submitted >= num_simulations;
        if (all_submitted || waiting_for_leaf) {
            auto sleep_duration = waiting_for_leaf ? std::chrono::microseconds(50)
                                                   : std::chrono::microseconds(100);
            std::this_thread::sleep_for(sleep_duration);  // ← THREAD BLOCKS HERE!
        }
    }
}
```

### The Actual Root Cause

**Threads ARE continuing to submit**, but the issue is:

1. **In-flight limit** at [continuous_simulation_runner.cpp:122-135](cpp_extensions/mcts/continuous_simulation_runner.cpp#L122-L135):
   ```cpp
   constexpr std::size_t kMaxInFlight = 4096;
   std::size_t backoff_loops = 0;
   size_t pending = pending_count_.load(std::memory_order_relaxed);
   while (queue.pending_count() >= kMaxInFlight || pending >= kMaxInFlight) {
       waiting_for_leaf = true;
       int flushed = process_completed_results(queue);
       if (flushed == 0) {
           std::this_thread::sleep_for(std::chrono::microseconds(100));  // ← BLOCKS
       }
   ```

2. **Batch collection happens in coordinator thread**, which calls `collect_batch()` **once** and then waits for GPU to complete.

3. **Coordinator is SYNCHRONOUS** at [batch_inference_coordinator.cpp:104-156](cpp_extensions/mcts/batch_inference_coordinator.cpp#L104-L156):
   ```cpp
   void BatchInferenceCoordinator::coordinator_loop() {
       while (running_.load(...)) {
           // Step 1: Collect batch (blocks up to timeout_ms)
           std::vector<InferenceRequest> batch = queue_->collect_batch(batch_size_, timeout_ms_);

           // Step 2: Call Python GPU inference (BLOCKS with GIL)
           inference_results = callback_->batch_inference(states);

           // Step 3: Submit results
           queue_->submit_results(results);

           // ↑ Only THEN does it loop back to collect next batch
       }
   }
   ```

### The Smoking Gun

**Coordinator is SERIAL**:
1. Collect batch (blocks 2.0ms or until min_batch_size)
2. Process GPU (blocks ~0.8ms for batch-64 @ FP16)
3. Submit results
4. **GOTO 1** (collect next batch)

**With 2.0ms timeout and 0.8ms GPU time:**
- Total cycle time: 2.0ms + 0.8ms = 2.8ms per batch
- If batch size = 1, throughput = 1 / 2.8ms = **357 batches/sec** = **357 sims/sec**
- But we're seeing 2,923 sims/sec, so batches are completing faster

**Wait, that doesn't match...**

Let me re-analyze. With 2,923 sims/sec and batch=1:
- 2,923 inferences/sec = 2,923 batches/sec
- Time per batch = 1000ms / 2,923 = **0.34ms per batch**

**So timeout is NOT triggering** (2.0ms). Something else is causing immediate submission.

### Aha! The Real Bug

Looking back at [async_inference_queue.cpp:80-109](cpp_extensions/mcts/async_inference_queue.cpp#L80-L109):

```cpp
if (min_batch_size > 0 && timeout_ms > 0.0) {
    while (batch.size() < min_batch_size && !shutting_down_.load(...)) {
        InferenceRequest request;
        // Lock-free dequeue attempt
        if (pending_requests_.try_dequeue(request)) {
            batch.push_back(std::move(request));
            pending_count_.fetch_sub(1, std::memory_order_relaxed);
        } else {
            // ← Queue is empty, wait on condition variable
            // ...
            if (now >= deadline) {
                break;  // Timeout expired
            }
        }
    }
}

// ⚠️ Opportunistically grab more up to max_batch_size
while (batch.size() < max_batch_size) {
    InferenceRequest request;
    if (!pending_requests_.try_dequeue(request)) {
        break;  // Queue empty ← IMMEDIATE RETURN!
    }
    batch.push_back(std::move(request));
    //...
}

return batch;  // ← Returns whatever was collected (even if batch.size() < min_batch_size after timeout)
```

**The issue**: After the timeout wait loop exits, the function **still tries to collect more** opportunistically, but if the queue is empty, it returns immediately with whatever it has.

**So when does batch size = 1 happen?**

1. Coordinator calls `collect_batch(32, 2.0)`
2. Queue has 1 request
3. Loop dequeues 1 request → batch.size() = 1
4. Loop tries to dequeue again → queue empty
5. Queue empty, so wait on condition variable
6. **BUT**: With serial GPU processing, by the time coordinator comes back to collect the next batch:
   - Only 1-2 NEW requests have been submitted (threads waiting for previous batch results)
   - Timeout triggers → returns batch-1 or batch-2

### The ACTUAL Root Cause

**The coordinator is SERIAL and threads are WAITING FOR RESULTS before submitting more**:

```
Timeline:
T=0ms:   Thread 1 submits req_1
T=0.1ms: Coordinator collects [req_1] (queue empty, no other requests yet)
T=0.2ms: GPU processes batch-1
T=0.3ms: Results ready
T=0.3ms: Thread 1 gets result, continues, submits req_2
T=0.4ms: Coordinator collects [req_2] (queue empty again)
... SERIAL PROCESSING
```

### Why Threads Aren't Submitting Concurrently

Looking at the run_continuous loop again - **threads DO submit multiple requests** before checking results:

```cpp
while (completed < num_simulations) {
    // Phase 1: Submit (if quota not reached)
    if (submitted < num_simulations) {
        // ... submit logic ...
        submitted++;  // ← Thread DOES continue submitting
    }

    // Phase 2: Check results (non-blocking)
    int processed = process_completed_results(queue);
    completed += processed;

    // Only sleep if no results AND (all submitted OR waiting for leaf)
    if (processed == 0) {
        if (all_submitted || waiting_for_leaf) {
            sleep(...);
        }
    }
}
```

### The REAL Issue: Configuration Parameters

Looking at [src/core/mcts.py:163-164](src/core/mcts.py#L163-L164):

```python
self.async_batch_size = async_batch_size  # Default: 32
self.async_timeout_ms = async_timeout_ms  # Default: 2.0
```

And the coordinator start call at [mcts.py:295-296](src/core/mcts.py#L295-L296):

```python
self._coordinator.start(self.async_queue, self._batch_callback,
                       self.async_batch_size, self.async_timeout_ms)
                       # ↑ batch_size=32    ↑ timeout=2.0ms
```

**BUT**: The Python side is passing these to `GPUInferenceWorker`:

Looking at [src/neural/inference_worker.py:739-813](src/neural/inference_worker.py#L739-L813):

```python
def _collect_batch(self, input_queue: Queue) -> List[InferenceRequest]:
    """Collect a batch of requests with dynamic micro-batching."""
    batch = []
    start_time = time.time()

    # Determine optimal batch size
    target_batch_size = self._get_optimal_batch_size()

    # Try to get first request with micro-timeout
    first_request = input_queue.get(timeout=self.max_timeout_ms)  # ← PYTHON QUEUE, NOT C++ QUEUE!
    batch.append(first_request)

    // ...
```

**WAIT!** The GPUInferenceWorker is using Python Queue, not the C++ AsyncInferenceQueue!

### The Architecture Mismatch

**There are TWO batching systems:**

1. **C++ AsyncInferenceQueue** with `collect_batch(min_batch_size, timeout_ms)`
   - Used by: `BatchInferenceCoordinator`
   - Parameters: batch_size=32, timeout=2.0ms
   - Status: **NOT USED** in the actual inference path!

2. **Python GPUInferenceWorker._collect_batch()**
   - Used by: inference_loop in separate thread
   - Parameters: batch_size=64, timeout=3.0ms (defaults)
   - Status: **NOT CONNECTED** to C++ queue!

### The Missing Connection

The `BatchInferenceCoordinator` calls:
```cpp
inference_results = callback_->batch_inference(states);  // ← Calls Python
```

This goes to [mcts.py:798-823](src/core/mcts.py#L798-L823):

```python
def fast_batch_callback(game_states: List[IGameState]) -> List[Tuple[List[float], float]]:
    """Direct GPU batch inference - single call for entire batch."""
    # ...
    # ✅ SINGLE batched GPU call
    policies, values = self.inference_fn.batch_inference(positions)
    # ...
```

Which calls `GPUInferenceWorker.batch_inference()` at [inference_worker.py:928-984](src/neural/inference_worker.py#L928-L984):

```python
def batch_inference(self, positions: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Process batch of positions through neural network."""
    # ... DIRECT GPU INFERENCE, NO QUEUE ...
    batch_tensor = self._create_batch_tensor_optimized(positions)
    with torch.no_grad():
        policy_logits, values = self._run_inference_with_precision(batch_tensor)
```

**AHA!** The `batch_inference()` method is **SYNCHRONOUS** - it does NOT use the queue at all!

### The Complete Picture

```
C++ Simulation Threads
  ↓ submit_request()
AsyncInferenceQueue (C++)
  ↓ collect_batch(32, 2.0ms) ← Accumulates requests
BatchInferenceCoordinator (C++)
  ↓ batch_inference(states) ← Calls Python with FULL batch
Python fast_batch_callback
  ↓ self.inference_fn.batch_inference(positions) ← Direct GPU call
GPUInferenceWorker.batch_inference()
  ↓ SYNCHRONOUS GPU inference
GPU (RTX 3060 Ti)
```

**So the C++ queue IS accumulating properly**, but why batch size = 1?

### Final Analysis: Check Default Parameters

Let me check what parameters the benchmark is actually passing...

Looking at test benchmark at [tests/performance/test_simulation_runner_performance.py](tests/performance/test_simulation_runner_performance.py), I need to check:

1. What `async_batch_size` is set to
2. What `async_timeout_ms` is set to
3. Whether the coordinator is even being created properly

**HYPOTHESIS**: The parameters are set incorrectly, or there's a default override somewhere.

Let me check the test configuration.

---

## Fix Strategy

**Option 1**: Increase timeout to allow accumulation (10-50ms instead of 2.0ms)
**Option 2**: Reduce min_batch_size to match realistic accumulation (8-16 instead of 32)
**Option 3**: Fix thread coordination to ensure concurrent submission (eliminate serialization)
**Option 4**: Check if coordinator parameters are being passed correctly

**NEXT STEP**: Read the benchmark test to see what parameters it's using.
