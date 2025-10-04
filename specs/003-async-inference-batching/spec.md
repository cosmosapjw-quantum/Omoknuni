# Spec: Async Inference Batching for 30k+ Simulations/Second

**Status:** Draft
**Author:** System Analysis
**Created:** 2025-10-04
**Target:** Achieve 30,000+ simulations/second with GPU batching

## Executive Summary

Current MCTS implementation achieves only **900-1,000 sims/sec** (3% of target) due to excessive Python/C++ boundary crossings. Each simulation requires a synchronous Python callback for neural network inference, causing 4.2× overhead compared to pure C++ execution.

This specification defines an async inference batching system that:
- Runs simulations continuously in C++ without returning to Python
- Queues inference requests in C++ data structures
- Batches requests and calls Python ONCE per batch (64-128 positions)
- Processes results and continues search without GIL reacquisition
- **Target: 30,000+ simulations/second with 8-12 threads**

## Problem Statement

### Current Performance

| Component | Time per Simulation | Throughput (1 thread) |
|-----------|--------------------|-----------------------|
| C++ tree operations (selection + backup) | 0.26ms | 3,846 sims/sec |
| Python callback overhead | +1.10ms | 714 sims/sec |
| **Total (current)** | **1.36ms** | **735 sims/sec** |

**With 512 threads:** 905 sims/sec (33× below 30k target)

### Root Cause

```cpp
// Current SimulationRunner::expand_node() - BLOCKS on every simulation
auto [policy, value] = inference_fn.request_inference(state);  // ⚠️ Acquires GIL, waits
```

**Issues:**
1. **Synchronous blocking**: Thread waits for Python to return
2. **Excessive GIL crossings**: N simulations = N GIL acquire/release cycles
3. **No parallelism during inference**: All threads block waiting for GPU
4. **Python overhead**: 1.1ms per callback (4.2× C++ execution time)

### Profiling Evidence

```
Profiler results (500 simulations, 1 thread):
  _collect_batch:        1.556s  (waiting for requests)
  batch_inference:       1.039s  (GPU inference)
  C++ tree operations:   0.259s  (actual MCTS work)

Bottleneck: Threads finish simulations in bursts, GPU worker idles between batches
```

## Requirements

### Functional Requirements

**FR1: Async Inference Queue (C++)**
- C++ threads submit inference requests to shared queue
- Queue stores: game state pointer, node index, path
- Non-blocking submission (no thread stalls)

**FR2: Batched Python Callback**
- Collect N requests (N=32-128) before Python call
- Single GIL acquisition per batch
- Return N (policy, value) pairs

**FR3: Continuous Simulation Loop**
- Threads run simulations without blocking on inference
- Check for completed inferences asynchronously
- Expand nodes with results and continue searching

**FR4: Thread Safety**
- Multiple C++ threads accessing shared queue
- Atomic operations for queue management
- Virtual loss coordination unchanged

### Performance Requirements

**PR1: Target Throughput**
- **30,000+ simulations/second** with 8-12 threads
- 60-80% GPU utilization
- Average batch size: 32-64 positions

**PR2: Latency**
- Inference queue: <0.1ms to submit request
- Batch collection: 0.5-2.0ms timeout
- Result processing: <0.5ms per result

**PR3: Memory**
- Inference queue: <10MB for 10k pending requests
- No memory leaks during continuous operation

**PR4: Scalability**
- Linear speedup with thread count (up to 12 threads)
- 75-85% parallel efficiency
- Graceful degradation beyond optimal thread count

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────┐
│  C++ MCTS Threads (8-12 threads)                        │
│                                                          │
│  Thread 1:  select → queue_inference → continue         │
│  Thread 2:  select → queue_inference → continue         │
│  Thread 3:  select → queue_inference → continue         │
│  ...                                                     │
│  Thread N:  select → queue_inference → continue         │
│                                                          │
│  (All threads running CONTINUOUSLY, never blocking)     │
│                                                          │
│  Periodic:  process_completed_inferences()              │
│             ↓                                            │
│         Expand + Backup                                 │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ↓ Shared Queue (C++)
┌─────────────────────────────────────────────────────────┐
│  AsyncInferenceQueue (C++)                              │
│  ┌────────────────────────────────────────┐             │
│  │ Pending: [req1, req2, ..., reqN]      │             │
│  └────────────────────────────────────────┘             │
│                                                          │
│  When batch ready (N≥32 OR timeout):                    │
│    1. Acquire GIL (ONCE)                                │
│    2. Call Python: batch_inference(states)              │
│    3. Release GIL                                       │
│    4. Distribute results to threads                     │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ↓ Python (GIL acquired ONCE per batch)
┌─────────────────────────────────────────────────────────┐
│  GPUInferenceWorker                                     │
│                                                          │
│  Receives: [state1, state2, ..., stateN]                │
│  Returns: [(policy1, value1), ..., (policyN, valueN)]   │
│                                                          │
│  GPU execution: 2-4ms for batch of 64                   │
└─────────────────────────────────────────────────────────┘
```

### Component Breakdown

**1. AsyncInferenceQueue (C++)**
- Thread-safe queue for pending inference requests
- Lock-free or mutex-protected (benchmark both)
- Timeout-based batch collection
- Result distribution back to threads

**2. ContinuousSimulationRunner (C++)**
- Extends SimulationRunner
- Runs simulations in infinite loop until quota reached
- Non-blocking inference submission
- Periodic result processing

**3. BatchInferenceCallback (C++/Python bridge)**
- Takes vector of game states
- Returns vector of (policy, value) tuples
- Single GIL acquisition per batch

**4. Integration with AlphaZeroMCTS (Python)**
- Creates AsyncInferenceQueue
- Spawns C++ worker threads
- Provides batch inference callback
- Monitors completion

## Performance Projections

### Theoretical Maximum

```
C++ tree operations: 0.26ms per simulation
With 12 threads:     12 × (1/0.00026) = 46,154 sims/sec (no inference delay)
```

### With Async Batching

```
Assumptions:
- Batch size: 48 positions
- GPU inference: 2.5ms per batch
- Threads: 12
- Batch frequency: Every ~10 simulations per thread

Timeline per batch cycle:
  1. Threads run 10 sims each (120 total): 10 × 0.26ms = 2.6ms
  2. GPU inference (batched):              2.5ms
  3. Result processing:                    0.5ms

Total: 5.6ms for 120 simulations = 21,428 sims/sec per cycle

With overlap (threads continue during GPU):
  - Threads idle during: 2.5ms GPU + 0.5ms processing = 3.0ms
  - Threads working:     ~80% of time

Effective throughput: 21,428 × 0.8 = ~17,000 sims/sec
```

### Optimized (Target)

```
With better overlap and tuning:
- Larger batches (64)
- Shorter timeout (0.5ms)
- More threads (16)
- Optimized result distribution

Target: 30,000-35,000 sims/sec
```

## Success Criteria

### Must Have

✅ **SC1:** Achieve ≥30,000 sims/sec with 8-12 threads on target hardware
✅ **SC2:** GPU utilization ≥60% during search
✅ **SC3:** Average batch size ≥32 positions
✅ **SC4:** Zero memory leaks in 1-hour continuous operation
✅ **SC5:** Thread-safe under TSan/Helgrind validation

### Should Have

⚠️ **SC6:** Achieve ≥35,000 sims/sec with tuning
⚠️ **SC7:** GPU utilization ≥80%
⚠️ **SC8:** Linear speedup up to 12 threads (efficiency ≥75%)

### Could Have

💡 **SC9:** Support multiple concurrent searches
💡 **SC10:** Adaptive batch sizing based on load
💡 **SC11:** Per-thread result queues for better locality

## Non-Goals

❌ **NG1:** Distributed multi-GPU support (future spec)
❌ **NG2:** Dynamic neural network reloading
❌ **NG3:** Alternative search algorithms (keep focused on MCTS)
❌ **NG4:** Training pipeline integration (separate concern)

## References

### Current Implementation Files

- `cpp_extensions/mcts/simulation_runner.cpp` - Current synchronous runner
- `cpp_extensions/mcts/simulation_runner.hpp` - Runner interface
- `src/core/simple_gpu_mcts.py` - Python batching wrapper
- `src/neural/inference_worker.py` - GPU batching logic

### Performance Measurements

- `MCTS_EVALUATION_REPORT.md` - C++ runner validation (1,744 sims/sec baseline)
- `IMPLEMENTATION_SUMMARY.md` - Analysis of over-engineering issues
- `tests/profile_simple_batch_mcts.py` - Profiling results
- `scripts/profile_mcts_bottleneck.py` - Bottleneck analysis

### Related Specifications

- `specs/002-cpp-simulation-runner/` - C++ simulation runner (phase 1)
- `mcts_guide.md` - Original architecture document (30k target specified)

## Open Questions

**Q1:** Lock-free queue vs mutex-protected?
- **Decision needed:** Benchmark both approaches
- **Criteria:** Throughput, latency variance, complexity

**Q2:** Per-thread result queues vs global result map?
- **Option A:** Each thread has own result queue (better locality)
- **Option B:** Global map with node_id → result (simpler)

**Q3:** How to handle inference failures gracefully?
- **Current:** Fallback to uniform policy
- **Proposed:** Retry queue? Mark node as unexpandable?

**Q4:** Optimal batch size and timeout?
- **Range:** Batch 16-128, timeout 0.3-3.0ms
- **Method:** Grid search with benchmark suite

## Risks and Mitigation

### High Risk

**R1: Thread synchronization bugs**
- *Impact:* Crashes, deadlocks, race conditions
- *Mitigation:* Extensive testing with TSan, unit tests, formal verification of queue

**R2: GIL management errors**
- *Impact:* Deadlocks, crashes
- *Mitigation:* Use pybind11 RAII guards, test GIL release/acquire thoroughly

**R3: Performance regression**
- *Impact:* Slower than current implementation
- *Mitigation:* Benchmark at each stage, keep fallback to synchronous mode

### Medium Risk

**R4: Memory leaks in queue**
- *Impact:* OOM in long-running searches
- *Mitigation:* Valgrind, ASan testing, smart pointers

**R5: Load imbalance across threads**
- *Impact:* Sublinear speedup
- *Mitigation:* Dynamic work stealing, monitor thread utilization

### Low Risk

**R6: Incompatibility with existing code**
- *Impact:* Need to refactor callers
- *Mitigation:* Maintain backward compatibility, feature flags

## Appendix: Profiling Data

### Current Performance Breakdown

```
Component                          Time (ms)    % of Total
──────────────────────────────────────────────────────────
C++ selection (tree traversal)     0.13        9.6%
C++ expansion (node allocation)    0.08        5.9%
C++ backup (value propagation)     0.05        3.7%
Python callback overhead           0.60       44.1%
GPU inference wait                 0.50       36.8%
──────────────────────────────────────────────────────────
Total per simulation               1.36       100%
Throughput (single thread)         735 sims/sec
```

### With 512 Threads (Current Max)

```
Metric                    Value       Notes
─────────────────────────────────────────────────────────
Throughput                905 s/s     Plateaus at 256+ threads
Avg batch size            62.5        Good batching
GPU utilization           42.4%       Underutilized
Threads blocking          ~80%        Waiting on inference
GIL crossings/sec         905         One per simulation
```

### Target with Async Batching

```
Metric                    Current     Target      Improvement
────────────────────────────────────────────────────────────
Throughput                905 s/s     30,000 s/s  33×
GPU utilization           42.4%       70-85%      1.7-2.0×
GIL crossings/sec         905         ~500        50% reduction
Avg batch size            62.5        48-64       Similar
Thread utilization        20%         75-85%      3.8-4.3×
```
