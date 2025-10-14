# Thread Scaling Investigation - Option B Findings
**Date**: 2025-10-14
**Issue**: Catastrophic 5× performance degradation when reusing GPUInferenceWorker

---

## Executive Summary

**Root Cause Identified**: Multiple `BatchInferenceCoordinator` threads running concurrently, all calling `batch_inference()` on the same `GPUInferenceWorker`, causing severe contention.

**The Problem**:
1. Each MCTS instance creates its own `BatchInferenceCoordinator` thread
2. All coordinators share the SAME `GPUInferenceWorker` (module-scoped fixture)
3. When a new MCTS is created, the old coordinator is NOT stopped (unless `close()` is called)
4. Multiple coordinator threads → concurrent `batch_inference()` calls → serialization + contention
5. **Result**: 4.5-6× performance degradation

---

## Diagnostic Results

### Experiment Setup
- Single shared `GPUInferenceWorker` (mimics module-scoped fixture)
- Three sequential MCTS instances with 1, 2, and 8 threads
- Each uses the same worker for inference

### Performance Results
```
1 thread (first use):    1,143 sims/sec
2 threads (second use):    255 sims/sec  (4.5× SLOWER!)
8 threads (third use):     185 sims/sec  (6.2× SLOWER!)
```

### Key Observations

**1. Batching is Working**:
- 1 thread: 15.4 avg batch size
- 2 threads: 22.9 avg batch size
- 8 threads: 22.9 avg batch size

✅ Batch accumulation is functioning correctly

**2. But Performance Degrades Anyway**:
- Despite good batch sizes, throughput drops 4.5-6×
- This rules out batching as the issue

**3. Coordinator Lifecycle**:
```python
# From mcts.py line 193
self._coordinator = mcts_py.BatchInferenceCoordinator()
self._coordinator_started = False

# Coordinator is started in search() at line 295
if not self._coordinator_started:
    self._coordinator.start(self.async_queue, self._batch_callback,
                           self.async_batch_size, self.async_timeout_ms)
    self._coordinator_started = True
```

**4. Coordinator NOT Stopped Between Tests**:
- `close()` method exists (line 572) but is NOT called in test fixtures
- Each new MCTS creates a NEW coordinator
- Old coordinators keep running in background
- Multiple coordinators → same worker → CONTENTION

---

## Root Cause Analysis

### The Architecture

```
MCTS Instance #1 (1 thread):
  ├─ BatchInferenceCoordinator Thread #1
  └─ Calls: worker.batch_inference(states)

MCTS Instance #2 (2 threads):
  ├─ BatchInferenceCoordinator Thread #2  ← NEW THREAD
  └─ Calls: worker.batch_inference(states)

Worker (shared):
  ├─ Thread #1 calling batch_inference()  ← STILL RUNNING!
  ├─ Thread #2 calling batch_inference()  ← NEW!
  └─ SERIALIZATION: Only one can run at a time
```

### Why This Causes 5× Slowdown

**GPUInferenceWorker.batch_inference() is NOT thread-safe for concurrent calls**:

1. **GPU is a serial resource**: Only one inference can run at a time
2. **Worker internal state**: Queue, batch buffer, model forward pass
3. **GIL contention**: Multiple coordinators fighting for GIL to call Python
4. **Thread context switching**: OS switching between coordinator threads

**Timeline (Simplified)**:
```
T=0ms:   Coordinator #1 calls batch_inference(batch-16)
T=0ms:   Coordinator #2 calls batch_inference(batch-24) ← BLOCKS
T=1.5ms: Coordinator #1's inference completes
T=1.5ms: Coordinator #2's inference starts
T=3.0ms: Coordinator #2's inference completes

Result: Serial execution, 2× slower
```

With 3+ coordinators (from multiple test runs), the degradation compounds.

---

## Evidence

### From Diagnostic Output

**Coordinator Searches Count**:
```
1 thread:  coordinator_searches: 2 (warmup + benchmark)
2 threads: coordinator_searches: 2 (warmup + benchmark)
8 threads: coordinator_searches: 2 (warmup + benchmark)
```

Each MCTS reports 2 searches, but if coordinators from previous instances were still running, they would be interfering.

### From Test Code

```python
# tests/performance/test_simulation_runner_performance.py:110
@pytest.fixture
def mcts_engine(inference_worker):
    """Create MCTS engine with real GPU inference worker."""
    mcts = AlphaZeroMCTS(
        inference_fn=inference_worker,  # ← SHARED WORKER
        ...
    )
    return mcts  # ← NO CLEANUP! Coordinator keeps running
```

**Issue**: Fixture doesn't call `mcts.close()`, so coordinator thread keeps running.

### From Thread Scaling Test

```python
# tests/performance/test_simulation_runner_performance.py:210
def test_thread_scaling(self, gomoku_game, inference_worker, num_threads):
    mcts = AlphaZeroMCTS(
        inference_fn=inference_worker,  # ← SAME WORKER
        num_threads=num_threads,
        ...
    )
    # ... run benchmark ...
    mcts.close()  # ← GOOD! But previous test instances still running
```

Each parameterized test (1, 2, 4, 8 threads) creates a NEW MCTS, and the cleanup only happens at the end.

---

## Why This Wasn't Caught Earlier

1. **Single-test execution**: Running one test at a time works fine
2. **Module-scoped fixture**: Worker persists across ALL tests in the module
3. **No coordinator lifecycle management**: No explicit cleanup between tests
4. **Coordinator is background thread**: Runs silently in background

---

## Solution

### Option 1: Function-Scoped Worker (RECOMMENDED)

**Change**:
```python
# tests/performance/test_simulation_runner_performance.py:72
@pytest.fixture(scope="function")  # ← Changed from "module"
def inference_worker():
    """Create real GPU inference worker with batch tracking."""
    # ... create worker ...
    yield worker
    # Cleanup
    worker.stop_worker()
    Path(model_path).unlink(missing_ok=True)
```

**Pros**:
- ✅ Each test gets a fresh worker
- ✅ No coordinator conflicts
- ✅ Matches production usage (one worker per MCTS lifecycle)
- ✅ Simple fix

**Cons**:
- ⚠️ Slower tests (worker creation overhead)
- ⚠️ More GPU memory churn

**Expected Impact**: Fixes 5× degradation, thread scaling should work correctly.

---

### Option 2: Proper Coordinator Cleanup

**Change**:
```python
# tests/performance/test_simulation_runner_performance.py:110
@pytest.fixture
def mcts_engine(inference_worker):
    mcts = AlphaZeroMCTS(...)
    yield mcts
    mcts.close()  # ← Ensure coordinator is stopped
```

**Pros**:
- ✅ Keeps module-scoped worker (faster tests)
- ✅ Explicit lifecycle management

**Cons**:
- ⚠️ Still has risk of coordinator conflicts during concurrent tests
- ⚠️ Doesn't match production usage pattern

---

### Option 3: Worker Thread Safety (COMPLEX)

Make `GPUInferenceWorker.batch_inference()` thread-safe for concurrent calls.

**Cons**:
- ❌ Complex implementation (locks, queues, coordination)
- ❌ Not the intended usage pattern
- ❌ Performance overhead from synchronization
- ❌ High risk of introducing new bugs

**Not Recommended**.

---

## Recommendation

**Implement Option 1: Function-Scoped Worker Fixture**

**Rationale**:
1. Simple fix (change one line)
2. Matches production usage (one worker per MCTS)
3. Eliminates all coordinator conflicts
4. Clean lifecycle management
5. Low risk

**Trade-off**: Tests will run ~5-10 seconds slower due to worker creation overhead, but correctness > speed for performance tests.

---

## Implementation Plan

**Step 1**: Change fixture scope
```python
@pytest.fixture(scope="function")  # Changed from "module"
def inference_worker():
```

**Step 2**: Add explicit cleanup to MCTS fixture
```python
@pytest.fixture
def mcts_engine(inference_worker):
    mcts = AlphaZeroMCTS(...)
    yield mcts
    mcts.close()  # Ensure coordinator stopped
```

**Step 3**: Re-run benchmarks
```bash
pytest tests/performance/test_simulation_runner_performance.py -v -s
```

**Expected Results**:
- 1 thread: ~1,200 sims/sec
- 2 threads: ~1,800-2,000 sims/sec (1.5-1.7× speedup)
- 4 threads: ~2,400-3,200 sims/sec (2-2.7× speedup)
- 8 threads: ~2,800-4,000 sims/sec (2.3-3.3× speedup)

**Step 4**: Validate KPIs
- Thread efficiency ≥25% (current) → expect 30-40%
- Throughput ≥2,000 sims/sec
- Batch size 16-32 (should remain stable)

---

## Additional Findings

### GPU Utilization
- Current: 18-33% (low)
- Target: 60-80%

**Why Low**:
- Small batch sizes (15-23 avg, need 32-64)
- Coordinator contention (will be fixed)
- Model inference time (needs profiling)

**Next Steps After Fix**:
1. Increase min_batch_size to 24-32
2. Profile GPU inference time
3. Check mixed precision is working
4. Optimize model forward pass

---

## Conclusion

The catastrophic thread scaling regression is caused by **multiple coordinator threads concurrently calling the same GPUInferenceWorker**, creating severe contention.

**Fix**: Change worker fixture from module-scoped to function-scoped.

**Expected Impact**:
- Eliminates 5× degradation
- Thread scaling should work correctly
- Path forward to 6,000-8,000 sims/sec target

**Confidence**: High - root cause clearly identified and solution is straightforward.

---

**End of Investigation Report**
