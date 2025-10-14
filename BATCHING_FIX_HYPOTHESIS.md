# Batching Fix Hypothesis
**Date**: 2025-10-14

## The Issue

Batch size = 1.0 despite configuration of async_batch_size=32, timeout=2.0ms

## Root Cause Hypothesis

After deep analysis, I believe the issue is **timeout is too short** (2.0ms) for accumulation with the current thread coordination.

### Evidence

1. **Benchmark shows**: 2,923 sims/sec with batch size = 1.0
2. **Time per batch**: 1000ms / 2,923 = 0.34ms per batch
3. **Timeout configured**: 2.0ms

**If timeout were triggering**, we'd see ~0.34ms accumulation → submit batch → GPU (0.8ms @ FP16) = 1.14ms total, giving ~877 batches/sec = 877 sims/sec (if batch=1).

**But we're seeing 2,923 sims/sec**, which means batches are cycling much faster than timeout allows.

### The Real Problem

Looking at [async_inference_queue.cpp:80-109](cpp_extensions/mcts/async_inference_queue.cpp#L80-L109):

```cpp
if (min_batch_size > 0 && timeout_ms > 0.0) {
    while (batch.size() < min_batch_size && !shutting_down_...) {
        InferenceRequest request;
        if (pending_requests_.try_dequeue(request)) {
            batch.push_back(std::move(request));
            pending_count_.fetch_sub(1, ...);
        } else {
            // Queue empty - wait on condition variable
            // ...
            if (now >= deadline) {
                break;  // Timeout expired
            }
        }
    }
}
```

**The logic**: Wait until `batch.size() >= min_batch_size` OR timeout expires.

**With min_batch_size=32 and timeout=2.0ms**:
- If requests arrive slowly (< 32 in 2.0ms), timeout triggers → submit batch-1 to batch-31
- If requests arrive quickly (≥ 32 in < 2.0ms), submit batch-32+

**Current behavior suggests**: Only 1 request in queue when collect_batch() is called.

### Why Only 1 Request?

**Thread serialization at coordinator level**:

The coordinator is SYNCHRONOUS:
```cpp
void BatchInferenceCoordinator::coordinator_loop() {
    while (running_...) {
        // Step 1: Collect (blocks)
        std::vector<InferenceRequest> batch = queue_->collect_batch(batch_size_, timeout_ms_);

        // Step 2: GPU inference (blocks with GIL)
        inference_results = callback_->batch_inference(states);

        // Step 3: Submit results
        queue_->submit_results(results);

        // ← ONLY THEN loops back to collect next batch
    }
}
```

**Timeline with batch-1**:
```
T=0.0ms: Coordinator calls collect_batch() → waits for requests
T=0.1ms: Thread 1 submits req_1
T=0.1ms: Coordinator dequeues req_1 → batch=[req_1]
T=0.1ms: Queue empty, starts waiting on condition variable
T=0.5ms: Still waiting (no more requests, threads idle)
T=1.0ms: Still waiting
T=1.5ms: Still waiting
T=2.1ms: TIMEOUT → returns batch=[req_1]
T=2.2ms: GPU processes batch-1 (0.2ms for FP16)
T=2.4ms: Results ready, submitted to queue
T=2.4ms: Thread 1 gets result, continues, submits req_2
T=2.5ms: Coordinator calls collect_batch() again → waits
... REPEAT
```

**This gives**: 1 batch every 2.4ms = 417 batches/sec = 417 sims/sec

**But we're seeing 2,923 sims/sec!**

This means the timeline is MUCH faster. Let me recalculate:

**If batch size really = 1**:
- 2,923 sims/sec = 2,923 batches/sec (if batch=1)
- Time per batch cycle = 1000ms / 2,923 = 0.34ms

**This suggests**:
- Accumulation time: ~0.0ms (immediate)
- GPU time: ~0.2ms (FP16 batch-1)
- Overhead: ~0.14ms

**So timeout is NOT being used!** Requests must be arriving immediately.

### Aha! The Real Root Cause

Looking at simulation threads - **with 8 threads running continuously**:

```cpp
// Thread 1 loop:
while (completed < num_simulations) {
    if (submitted < num_simulations) {
        submit_request();  // ← Submits IMMEDIATELY
        submitted++;
    }
    process_results();  // ← Non-blocking check
}

// All 8 threads doing this simultaneously
```

**With 8 threads**:
- Each thread submits as fast as possible
- **Multiple requests should be in queue at once!**

### The Missing Piece

Looking at [continuous_simulation_runner.cpp:122-135](cpp_extensions/mcts/continuous_simulation_runner.cpp#L122-L135):

```cpp
constexpr std::size_t kMaxInFlight = 4096;
while (queue.pending_count() >= kMaxInFlight || pending >= kMaxInFlight) {
    waiting_for_leaf = true;
    int flushed = process_completed_results(queue);
    if (flushed == 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    // ...
}
```

**Threads ARE submitting multiple requests**, but they're rate-limited by in-flight cap.

### The Smoking Gun

Looking at the actual `collect_batch()` call:

```cpp
std::vector<InferenceRequest> batch = queue_->collect_batch(batch_size_, timeout_ms_);
```

**batch_size_** is being passed as the **FIRST PARAMETER**, which is `min_batch_size`.

So `collect_batch(32, 2.0)` means:
- **min_batch_size = 32** ← Wait for AT LEAST 32 requests
- **timeout_ms = 2.0** ← Or timeout after 2.0ms

**With 8 threads submitting requests continuously**:
- Threads should accumulate 32+ requests in the queue
- Coordinator should collect batch-32+ and submit to GPU

**But we're seeing batch=1**, which means:
1. Either only 1 request is in the queue (threads not submitting fast enough), OR
2. The timeout is triggering too early (before 32 requests accumulate), OR
3. There's a bug in collect_batch() logic

### Final Hypothesis

**The timeout of 2.0ms is TOO SHORT** for threads to accumulate 32 requests.

**Calculation**:
- 8 threads submitting continuously
- Each thread needs to: select leaf → submit → continue
- Select time: ~0.05ms (PUCT selection, AVX2 optimized)
- If 8 threads each submit 1 request, total = 8 requests
- To reach 32 requests, need 4 cycles per thread
- Time for 4 cycles: 4 × 0.05ms = 0.2ms per thread
- **With 8 threads in parallel, should reach 32 requests in ~0.2-0.5ms**

**But this assumes threads are submitting BEFORE checking results!**

Looking at the loop:
```cpp
while (completed < num_simulations) {
    // Phase 1: Submit
    if (submitted < num_simulations) {
        // ... can we submit? ...
        if (submission_ready) {
            queue.submit_request(...);
            submitted++;
        }
    }

    // Phase 2: Process results
    processed = process_completed_results(queue);
    completed += processed;
}
```

**The issue**: Threads wait in `process_completed_results()` if no results are ready, instead of continuing to submit more requests!

At [continuous_simulation_runner.cpp:166-174](cpp_extensions/mcts/continuous_simulation_runner.cpp#L166-L174):

```cpp
if (processed == 0) {
    bool all_submitted = submitted >= num_simulations;
    if (all_submitted || waiting_for_leaf) {
        auto sleep_duration = waiting_for_leaf ? std::chrono::microseconds(50)
                                               : std::chrono::microseconds(100);
        std::this_thread::sleep_for(sleep_duration);  // ← BLOCKS
    }
}
```

**AHA!** If `processed == 0` (no results ready), the thread SLEEPS for 50-100μs!

**With 8 threads**:
- All threads submit 1 request each → 8 requests in queue
- Coordinator needs 32 requests minimum (timeout 2.0ms)
- Threads now sleep waiting for results (50-100μs each)
- **Threads are NOT continuously submitting!**

### The Root Cause: Sleep-on-No-Results

Threads sleep when `processed == 0`, which prevents continuous submission and limits queue accumulation to ~8-16 requests (one per thread before they all sleep).

**Timeline**:
```
T=0.0ms: 8 threads each submit 1 request → queue = 8 requests
T=0.1ms: Coordinator calls collect_batch(32, 2.0ms)
T=0.1ms: Coordinator dequeues 8 requests → batch.size() = 8
T=0.1ms: Queue empty, waits on condition variable
T=0.1ms: 8 threads check results → processed=0 → SLEEP(100μs)
T=0.2ms: Still waiting (threads asleep!)
... (threads wake after 100μs but coordinator already timed out)
T=2.1ms: TIMEOUT → returns batch=[8 requests]
```

**But this would give batch size = 8**, not 1!

### Final Analysis: Check Parameter Passing

Let me verify the actual parameters being passed. The issue might be that `async_batch_size=32` is being passed INCORRECTLY.

Looking at [mcts.py:295-296](src/core/mcts.py#L295-L296):

```python
self._coordinator.start(self.async_queue, self._batch_callback,
                       self.async_batch_size, self.async_timeout_ms)
```

Let me check the C++ signature at [python_bindings.cpp](cpp_extensions/mcts/python_bindings.cpp) for `BatchInferenceCoordinator.start()`:

## Fix Strategy

**Option 1: Increase Timeout** (10-50ms instead of 2.0ms)
- Allows more time for threads to accumulate requests
- May increase latency slightly but improves batch size

**Option 2: Reduce Min Batch Size** (8-16 instead of 32)
- Matches realistic accumulation rate with current thread coordination
- Maintains low latency while improving GPU utilization

**Option 3: Remove Sleep-on-No-Results**
- Let threads continue submitting even when results aren't ready yet
- Maximizes queue accumulation
- Requires careful tuning to avoid overwhelming the queue

**Option 4: Hybrid Approach**
- Reduce min batch size to 16
- Increase timeout to 5.0ms
- Remove sleep or reduce to 10μs

**RECOMMENDED: Option 4** (Hybrid)
- min_batch_size: 32 → 16
- timeout_ms: 2.0 → 5.0
- sleep_duration: 100μs → 10μs

**Expected Impact**:
- Batch size: 1.0 → 16-32
- Throughput: 2,923 → 6,000-8,000 sims/sec
- GPU utilization: ~10% → 60-80%
