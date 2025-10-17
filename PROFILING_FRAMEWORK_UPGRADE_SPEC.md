# Profiling Framework Upgrade Specification

**Document Version**: 1.0
**Date**: 2025-10-17
**Purpose**: Eliminate 81-98% "unknown" time in ContinuousSimulationRunner profiling
**Scope**: Comprehensive instrumentation of all MCTS execution paths

## Executive Summary

**Problem**: Current profiling shows 81-98% of execution time as "unknown" in ContinuousRunner, making bottleneck identification impossible.

**Root Cause**: Profiling framework only instruments LEAF functions (make_move, PUCT, unmake_move) but NOT CONTAINER functions (main loops, coordination logic, async infrastructure).

**Solution**: Add comprehensive instrumentation to:
1. BatchInferenceCoordinator (ZERO current instrumentation)
2. ContinuousRunner main loop phases (MINIMAL current instrumentation)
3. Python callback overhead (NOT TRACKED)
4. State management operations (PARTIALLY tracked)

**Expected Outcome**: Reduce "unknown" time from 81-98% to <5%, enabling accurate bottleneck identification.

---

## Analysis of Current Instrumentation

### 1. continuous_simulation_runner.cpp

#### WELL INSTRUMENTED (✅):
- `select_leaf_with_make_unmake()` - Tree traversal, PUCT, make/unmake
- `unwind_path()` - State restoration

#### MISSING INSTRUMENTATION (🔴):

**run_continuous() [Lines 46-250]:**
- Main loop overhead (while loop at line 111)
- Phase 1: Select to leaf [Lines 114-218]
  - State cloning for queue (line 165) - **CRITICAL: This is likely the bottleneck**
  - Queue submission (lines 193-200)
  - Pending buffer management (lines 202-214)
  - Backoff loops when queue full (lines 177-190)
- Phase 2: Process results (line 221) - partially instrumented
- Sleeping/yielding overhead (lines 224-233) - **Could be significant**
- Thread-local state initialization (line 108)

**ensure_root_expanded() [Lines 466-539]:**
- Root expansion synchronous wait (lines 510-521)
- State cloning (line 493)
- Dirichlet noise generation (line 530)

**expand_node_with_result() [Lines 370-464]:**
- Policy masking/normalization (lines 397-423)
- Child allocation (line 427)
- Child initialization loops (lines 434-447)
- Atomic flag setting (line 456)

**process_completed_results() [Lines 252-368]:**
- Phase 1: Collecting ready results (lines 262-286)
- Phase 2: Batch node expansions (lines 293-306)
- Phase 3: Backup preparation (lines 308-336)
- Phase 4: Atomic updates (lines 338-346)
- Phase 5: Flag clearing (lines 349-352)
- Phase 6: State pool returns (lines 354-365)

### 2. batch_inference_coordinator.cpp

#### ZERO INSTRUMENTATION (🔴🔴🔴):

**coordinator_loop() [Lines 90-200]:**
- Main worker thread loop - **ENTIRE COORDINATOR THREAD UNINSTRUMENTED**
- Phase 1: collect_batch() call (line 95) - blocking operation
- Phase 2: Feature extraction (lines 110-160)
  - OpenMP parallelization (lines 132-146)
  - Serial extraction fallback (lines 148-157)
  - Feature buffer allocation/resize
- Phase 3: Python callback (lines 162-175) - **GPU inference call**
- Phase 4: Result submission (line 187)
- Fallback path (lines 191-198)

**Critical Issue**: The BatchInferenceCoordinator runs in a SEPARATE THREAD and has ZERO profiling. This could be where all the "unknown" time is going!

### 3. async_inference_queue.cpp

#### WELL INSTRUMENTED (✅):
- `submit_request()` - Full instrumentation with ScopedMetric + PROFILE_SCOPE
- `collect_batch()` - Full instrumentation including wait time
- `try_get_result()` - Full instrumentation

#### COULD BE ENHANCED (⚠️):
- Lock contention on cv_mutex_ (line 125)
- Memory allocation overhead in vector operations

### 4. Python Callback (Python-side)

#### NOT TRACKED (🔴):
- GIL acquisition time
- PyTorch tensor operations
- GPU inference latency
- Result marshaling back to C++

---

## Proposed Instrumentation Strategy

### Phase 1: Critical Path Instrumentation (Highest Priority)

**Goal**: Capture the most likely bottlenecks first

#### 1.1 BatchInferenceCoordinator (CRITICAL - Separate thread!)

Add new ProfileMetrics (in `enhanced_metrics.hpp`):
```cpp
// Coordinator thread metrics
CoordinatorLoopTotal,           // Total time in coordinator_loop()
CoordinatorCollectBatch,        // Time blocked in queue.collect_batch()
CoordinatorFeatureExtraction,   // Time extracting features (total)
CoordinatorFeatureExtractionOMP, // Time in OpenMP parallel region
CoordinatorFeatureAllocation,   // Time allocating feature vectors
CoordinatorPythonCallback,      // Time in callback->batch_inference_features()
CoordinatorGILWait,             // Time waiting for GIL (if measurable)
CoordinatorResultSubmit,        // Time submitting results to queue
CoordinatorIdleTime,            // Time when no requests available

// Feature extraction details
FeatureExtractionPerState,      // Per-state extraction time (avg)
OMPThreadCount,                 // Number of threads used by OpenMP
OMPOverhead,                    // OpenMP parallelization overhead
```

Instrumentation locations:
- Line 90: `coordinator_loop()` - Wrap entire loop body
- Line 95: `collect_batch()` - Measure blocking time
- Lines 110-160: Feature extraction - Separate serial vs parallel paths
- Line 169: Python callback - Measure GPU inference
- Line 187: Result submission

#### 1.2 ContinuousRunner Main Loop

Add new ProfileMetrics:
```cpp
// Main loop metrics
RunContinuousLoop,              // Total loop iteration time
RunContinuousPhase1,            // Phase 1: Select to leaf
RunContinuousPhase2,            // Phase 2: Process results
RunContinuousQueueClone,        // State cloning for queue submission (CRITICAL)
RunContinuousQueueSubmit,       // Queue submission time
RunContinuousPendingBuffer,     // Pending buffer management
RunContinuousBackoffLoop,       // Time in backoff/wait loops
RunContinuousSleepYield,        // Time sleeping/yielding
RunContinuousThreadLocalInit,   // Thread-local state initialization

// Root expansion
RootExpansionTotal,             // Total root expansion time
RootExpansionWait,              // Synchronous wait for inference
RootExpansionDirichlet,         // Dirichlet noise generation

// Node expansion
NodeExpansionPolicyMask,        // Policy masking/normalization
NodeExpansionChildAlloc,        // Child node allocation
NodeExpansionChildInit,         // Child initialization loop
NodeExpansionAtomicFlag,        // Atomic flag operations

// Batch result processing
BatchResultsCollect,            // Phase 1: Collect ready results
BatchResultsExpand,             // Phase 2: Expand nodes
BatchResultsBackupPrep,         // Phase 3: Prepare backup data
BatchResultsAtomicUpdate,       // Phase 4: Atomic tree updates
BatchResultsClearFlags,         // Phase 5: Clear expanding flags
BatchResultsReturnStates,       // Phase 6: Return to state pool
```

### Phase 2: Detailed Instrumentation

#### 2.1 State Management Operations

Already tracked but need separate metrics:
- State cloning (currently lumped into StateCloneTotal)
- State pool operations (acquire/release)
- copyFrom() operations (if any remain)

#### 2.2 Synchronization Primitives

Add metrics for:
- Mutex lock contention
- Atomic CAS failures
- Condition variable wait times
- Spin-lock overhead

#### 2.3 Memory Operations

Track:
- Vector allocations/resizes
- Arena allocator operations
- Free list operations

### Phase 3: Python-Side Instrumentation

Enhance Python profiling in `unified_profiler.py` to track:
- GIL acquisition duration
- Torch tensor creation
- GPU kernel launch overhead
- Result marshaling

---

## Implementation Plan

### Step 1: Add New ProfileMetric Enums

**File**: `cpp_extensions/mcts/profiling/enhanced_metrics.hpp`

Add ~30 new metrics for coordinator, main loop, and detailed operations.

### Step 2: Instrument BatchInferenceCoordinator

**File**: `cpp_extensions/mcts/batch_inference_coordinator.cpp`

Priority: **CRITICAL** - This is a separate thread with ZERO current instrumentation!

```cpp
void BatchInferenceCoordinator::coordinator_loop() {
    while (running_.load(std::memory_order_acquire)) {
        PROFILE_SCOPE(ProfileMetric::CoordinatorLoopTotal);

        // Phase 1: Collect batch
        std::vector<InferenceRequest> batch;
        {
            PROFILE_SCOPE(ProfileMetric::CoordinatorCollectBatch);
            batch = queue_->collect_batch(batch_size_, timeout_ms_);
        }

        if (batch.empty()) {
            PROFILE_COUNTER(ProfileMetric::CoordinatorIdleTime, 1);
            continue;
        }

        // Phase 2: Extract features
        {
            PROFILE_SCOPE(ProfileMetric::CoordinatorFeatureExtraction);
            // ... feature extraction with separate OMP instrumentation
        }

        // Phase 3: Python callback
        {
            PROFILE_SCOPE(ProfileMetric::CoordinatorPythonCallback);
            inference_results = callback_->batch_inference_features(...);
        }

        // Phase 4: Submit results
        {
            PROFILE_SCOPE(ProfileMetric::CoordinatorResultSubmit);
            queue_->submit_results(results);
        }
    }
}
```

### Step 3: Instrument ContinuousRunner Main Loop

**File**: `cpp_extensions/mcts/continuous_simulation_runner.cpp`

Add instrumentation to `run_continuous()`:

```cpp
int ContinuousSimulationRunner::run_continuous(...) {
    while (completed < num_simulations) {
        PROFILE_SCOPE(ProfileMetric::RunContinuousLoop);

        // Phase 1: Select to leaf
        if (submitted < num_simulations) {
            PROFILE_SCOPE(ProfileMetric::RunContinuousPhase1);

            // State cloning for queue
            {
                PROFILE_SCOPE(ProfileMetric::RunContinuousQueueClone);
                queue_state = tls.state->clone();
            }

            // Queue submission
            {
                PROFILE_SCOPE(ProfileMetric::RunContinuousQueueSubmit);
                uint64_t request_id = queue.submit_request(...);
            }

            // Pending buffer management
            {
                PROFILE_SCOPE(ProfileMetric::RunContinuousPendingBuffer);
                // ... buffer operations
            }
        }

        // Phase 2: Process results
        {
            PROFILE_SCOPE(ProfileMetric::RunContinuousPhase2);
            int processed = process_completed_results(queue);
        }

        // Sleep/yield if needed
        if (processed == 0) {
            PROFILE_SCOPE(ProfileMetric::RunContinuousSleepYield);
            std::this_thread::sleep_for(...);
        }
    }
}
```

### Step 4: Instrument process_completed_results() Phases

Add instrumentation to each of the 6 phases:

```cpp
int ContinuousSimulationRunner::process_completed_results(...) {
    // Phase 1: Collect
    {
        PROFILE_SCOPE(ProfileMetric::BatchResultsCollect);
        for (size_t i = 0; i < PENDING_BUFFER_CAPACITY; ++i) {
            // ... collect ready results
        }
    }

    // Phase 2: Expand
    {
        PROFILE_SCOPE(ProfileMetric::BatchResultsExpand);
        for (auto& ready : ready_results) {
            // ... expand nodes
        }
    }

    // Phase 3: Backup prep
    {
        PROFILE_SCOPE(ProfileMetric::BatchResultsBackupPrep);
        // ... prepare batched updates
    }

    // Phase 4: Atomic updates
    {
        PROFILE_SCOPE(ProfileMetric::BatchResultsAtomicUpdate);
        for (const auto& [node_index, update] : node_updates) {
            // ... atomic operations
        }
    }

    // Phase 5: Clear flags
    {
        PROFILE_SCOPE(ProfileMetric::BatchResultsClearFlags);
        for (auto& ready : ready_results) {
            // ... clear flags
        }
    }

    // Phase 6: Return states
    {
        PROFILE_SCOPE(ProfileMetric::BatchResultsReturnStates);
        for (auto& ready : ready_results) {
            // ... return to pool
        }
    }
}
```

### Step 5: Instrument expand_node_with_result()

Break down into phases:

```cpp
bool ContinuousSimulationRunner::expand_node_with_result(...) {
    // Policy masking
    {
        PROFILE_SCOPE(ProfileMetric::NodeExpansionPolicyMask);
        // ... mask and normalize policy
    }

    // Child allocation
    {
        PROFILE_SCOPE(ProfileMetric::NodeExpansionChildAlloc);
        NodeIndex first_child = tree_.allocate_nodes(num_children);
    }

    // Child initialization
    {
        PROFILE_SCOPE(ProfileMetric::NodeExpansionChildInit);
        for (uint16_t i = 0; i < num_children; ++i) {
            // ... initialize children
        }
    }

    // Atomic flag
    {
        PROFILE_SCOPE(ProfileMetric::NodeExpansionAtomicFlag);
        if (!tree_.atomic_try_set_expanded(leaf)) {
            return false;
        }
    }
}
```

### Step 6: Update Profiling Analysis

**File**: `scripts/unified_profiler.py`

Add analysis for new metrics:
- Coordinator thread breakdown
- Main loop phase breakdown
- State cloning hotspots
- Sleep/idle time analysis

### Step 7: Rebuild and Test

```bash
# Clear Python cache
rm -rf scripts/__pycache__

# Rebuild C++ with new instrumentation
pip install -e . --force-reinstall --no-deps

# Run profiling
python scripts/unified_profiler.py --simulations 100 --threads 1 --batch-size 64 \
    --runner-type continuous --output /tmp/profiling_test
```

---

## Expected Results

### Before (Current State):
```
Top Operations by Total Time:
  unknown                        |  2401 calls |  10.82ms |  81-98%
  state_clone_total              |  2401 calls |   0.24ms |   0.0%
  selection_make_move            |  2401 calls |   0.14ms |   0.1%
```

### After (With Full Instrumentation):
```
Top Operations by Total Time:
  coordinator_loop_total         |   100 calls |   8.50ms |  35.0%  ← Coordinator thread
  coordinator_python_callback    |    15 calls |   5.20ms |  21.4%  ← GPU inference
  run_continuous_queue_clone     |  2401 calls |   3.10ms |  12.8%  ← State cloning
  coordinator_feature_extraction |    15 calls |   2.30ms |   9.5%  ← Feature extract
  run_continuous_sleep_yield     |  1500 calls |   1.80ms |   7.4%  ← Idle time
  batch_results_atomic_update    |    15 calls |   0.90ms |   3.7%  ← Tree updates
  selection_traversal            |  2401 calls |   0.70ms |   2.9%  ← MCTS traversal
  unknown                        |     ? calls |   0.50ms |   2.1%  ← Reduced to <5%!
```

**Key Insights Enabled:**
1. If coordinator_python_callback is 21% → GPU is the bottleneck
2. If run_continuous_queue_clone is 13% → State cloning is the bottleneck
3. If run_continuous_sleep_yield is 7% → Too much idle time, need better coordination
4. If coordinator_feature_extraction is 10% → OpenMP not working or inefficient

---

## Validation Criteria

**Success Metrics:**
1. ✅ "Unknown" time reduced from 81-98% to <5%
2. ✅ Top 5 operations account for >80% of total time
3. ✅ Clear bottleneck identification possible
4. ✅ Profiling overhead remains <10% of wall-clock time

**Failure Modes:**
1. 🔴 "Unknown" time still >20% → Need more instrumentation
2. 🔴 Profiling overhead >20% → Too much instrumentation, need to reduce
3. 🔴 Results inconsistent across runs → Instrumentation affecting execution

---

## Risk Mitigation

**Risk 1**: Excessive profiling overhead slows down execution
- **Mitigation**: Use PROFILE_SCOPE (lightweight) instead of ScopedMetric where possible
- **Mitigation**: Only instrument outer loops, not inner tight loops

**Risk 2**: Thread-local profiling not captured correctly
- **Mitigation**: Ensure EnhancedProfiler is thread-safe
- **Mitigation**: Test with multiple threads (1, 4, 8)

**Risk 3**: Python-side profiling interferes with GIL
- **Mitigation**: Keep Python profiling minimal
- **Mitigation**: Use separate profiling session for Python vs C++

---

## Next Steps

1. ✅ Create this specification document
2. ⏳ Review and approve specification
3. ⏳ Implement Phase 1 (critical path instrumentation)
4. ⏳ Test and validate with small workload (10 simulations)
5. ⏳ Implement Phase 2 (detailed instrumentation)
6. ⏳ Run full profiling campaign (100-2000 simulations)
7. ⏳ Analyze results and identify true bottleneck
8. ⏳ Implement optimization based on findings

---

## Appendix: Key Files to Modify

### C++ Source Files:
1. `cpp_extensions/mcts/profiling/enhanced_metrics.hpp` - Add ~30 new metrics
2. `cpp_extensions/mcts/batch_inference_coordinator.cpp` - Add full instrumentation
3. `cpp_extensions/mcts/continuous_simulation_runner.cpp` - Add main loop instrumentation
4. `cpp_extensions/mcts/async_inference_queue.cpp` - Minor enhancements

### Python Files:
5. `scripts/unified_profiler.py` - Add analysis for new metrics
6. `scripts/profiling_campaign.py` - Ensure continuous runner is used

### Build:
7. `pip install -e . --force-reinstall --no-deps` - Rebuild after changes

---

## Conclusion

The current profiling framework is well-designed but **incomplete**. It instruments leaf functions excellently but misses the container functions where the actual overhead resides. The 81-98% "unknown" time is almost certainly in:

1. **BatchInferenceCoordinator thread** (separate thread, ZERO instrumentation)
2. **ContinuousRunner main loop** (loop overhead, state cloning, coordination)
3. **Python callback** (GIL, GPU inference)

With comprehensive instrumentation following this spec, we will definitively identify the bottleneck and can then implement the correct optimization (whether it's state pooling, OpenMP, async queue tuning, or something else entirely).

**Estimated Implementation Time**: 2-3 hours
**Estimated Validation Time**: 1 hour
**Total**: 3-4 hours for complete profiling framework upgrade
