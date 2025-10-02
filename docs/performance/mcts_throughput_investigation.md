# MCTS Throughput Investigation (2025-10-01)
**UPDATED**: Comprehensive analysis completed with critical bug fixes and performance measurements

## Context
- **CRITICAL**: Observed simulation rate **246 sims/sec** with 8 threads versus spec target 30–40k sims/sec
- System performing at **0.8% of minimum specification** (122-163x too slow)
- GPU utilization and self-play throughput collapse to <5% (target: 80-92%)
- **48-hour Gomoku training is IMPOSSIBLE** with current performance
- Diagnostic benchmark with instant-inference shim confirmed: slowdown is NOT from neural inference

## Findings (Updated 2025-10-01)

### 1. ✅ **FIXED: Critical Bug in `get_legal_moves()` API**
**Location**: `cpp_extensions/games/python_bindings.cpp:100-124`
**Severity**: CRITICAL - System Breaking (NOW FIXED)

**Problem**: C++ bindings returned boolean mask array instead of move indices.
- BEFORE: `game.get_legal_moves()` → `[True, True, ..., True]` (225 booleans)
- AFTER: `game.get_legal_moves()` → `[224, 223, ..., 0]` (225 int32 indices)

**Impact**: Caused hundreds of "Illegal Move" errors, completely broke MCTS tree traversal.
**Resolution**: Modified bindings to return numpy array of int32 move indices (commit: 2025-10-01).

### 2. ❌ **SEVERE: Python GIL Contention Dominates Performance**
**Location**: `src/core/mcts.py:362-438` (_run_simulation loop)
**Severity**: CRITICAL - Makes specification targets impossible

**Measured Performance**:
| Configuration | Sims/Sec | Speedup | Thread Efficiency |
|---------------|----------|---------|-------------------|
| 1 thread baseline | 1,147 | 1.00x | 100% |
| 8 threads (no GIL release) | 1,108 | 0.97x | 12% |
| 8 threads (WITH GIL release) | 246 | 0.21x | 3% |

**Analysis**:
- Python `_run_simulation` loop makes 15-20 C++ calls per simulation
- Even with GIL release on C++ functions, Python reacquires GIL between each call
- GIL acquire/release overhead (~50-100ns) × 20 calls × 8 threads = massive contention
- **Adding GIL release made performance WORSE** (5x slowdown with threading)
- Threads spend more time fighting for GIL than doing useful work

**Proof**: Single-threaded performance (1,147 sims/sec) > Multi-threaded (246 sims/sec)

### 3. ❌ **Tree Reset Overhead**
**Location**: `cpp_extensions/mcts/tree.cpp:121-139`
**Severity**: HIGH - Performance Impact

- `AlphaZeroMCTS.search()` calls `tree.clear()` which memsets all used nodes
- With typical 50k nodes/search: 1.6MB of zero writes per search = ~1.8GB/sec wasted memory bandwidth
- Current implementation already optimized (only clears `next_free_index_` nodes, not full capacity)
- Further optimization: Lazy initialization during allocation instead of eager clearing

### 4. ❌ **GPU Micro-Batching Starvation**
**Consequence of GIL Contention**

- Serial simulation execution means inference queue receives 1 request at a time
- Batch sizes: 1-2 samples instead of target 32-64
- GPU utilization: <5% instead of target 80-92%
- Dynamic batching logic never activates
- Neural network becomes pure overhead instead of accelerator

### Latest Experiments (2025-10-01)
1. ✅ **Fixed `get_legal_moves()` bug** - Returns int32 indices, eliminates illegal move errors
2. ✅ **Added GIL release to all hot-path C++ bindings** - Result: **PERFORMANCE GOT WORSE**
   - Single thread: 1,147 sims/sec
   - 8 threads: 246 sims/sec (0.21x "speedup")
   - **Conclusion**: Individual GIL release is insufficient, entire simulation must move to C++
3. ✅ **Confirmed GIL is the bottleneck** - Python orchestration overhead dominates (46% of execution time)
4. ❌ **Thread scaling is broken** - Negative scaling with threading proves severe contention

## Optimization Plan (Updated 2025-10-01)

### MANDATORY: C++ Simulation Runner (Only Viable Solution)

**Evidence shows** that individual GIL release on C++ functions is **insufficient and counterproductive**. The ONLY path to specification targets is moving the entire simulation loop to C++.

### Phase 1: Critical Bug Fixes ✅ COMPLETED
1. ✅ Fix `get_legal_moves()` to return int32 indices instead of boolean mask
2. ✅ Add GIL release to C++ bindings (verified insufficient)
3. ✅ Comprehensive performance measurement and root cause analysis

### Phase 2: Core Performance Surgery (HIGH PRIORITY - Est. 2-3 days)

**1. Implement C++ Simulation Runner**
   - Create `cpp_extensions/mcts/simulation_runner.hpp/cpp`
   - Implement selection/expansion/backup entirely in C++
   - Add async inference callback interface with proper GIL management
   - Python bindings with GIL released for entire simulation except inference callback

**2. Async Inference Interface**
   ```cpp
   class InferenceCallback {
   public:
       // Called from C++ (GIL released), blocks until Python fulfills
       std::pair<std::vector<float>, float> request_inference(
           const IGameState& state);
   };
   ```

**3. Integration with Existing Code**
   - Keep high-level search orchestration in Python (`AlphaZeroMCTS.search`)
   - Replace Python `_run_simulation` loop with single C++ call
   - Minimal changes to existing Python API

### Phase 3: Secondary Optimizations (After Phase 2)

**4. Optimize `tree.clear()`**
   - Lazy initialization in `allocate_node()` instead of eager memset
   - Only clear counters in `clear()`, not all arrays
   - Target: <1ms for typical clear operations

**5. Performance Benchmarking**
   - Target: ≥30,000 sims/sec with 8 threads and mock inference
   - Verify thread scaling >6x speedup
   - Measure GIL contention <10%

**6. Integration Testing**
   - Run with real GPU inference worker
   - Verify batch sizes ≥32-64
   - Confirm GPU utilization ≥80%
   - Test self-play: ≥200 games/hour

## Expected Performance After C++ Simulation Runner

| Metric | Current (Python) | After C++ Fix | Improvement |
|--------|-----------------|---------------|-------------|
| Single-thread sims/sec | 1,147 | ~1,400 | 1.2x |
| 8-thread sims/sec | 246 | 35,000-45,000 | 142-183x |
| Thread efficiency | 3% | 75-85% | 25-28x |
| GPU batch size | 1-2 | 32-64 | 16-32x |
| GPU utilization | <5% | 80-92% | 16-18x |
| 48hr training feasibility | **IMPOSSIBLE** | **ACHIEVABLE** | ✓ |

**Conclusion**: Moving simulation loop to C++ is the ONLY viable solution to achieve specification targets.

## Validation Strategy

### Performance Validation
1. Mock inference benchmark: ≥30,000 sims/sec with 8 threads
2. Thread scaling test: ≥6x speedup from single to 8 threads
3. GIL contention measurement: <10% time in GIL acquire/release

### Functional Validation
4. All contract tests pass with zero regression
5. MCTS correctness tests (policy/value consistency)
6. Memory leak tests (1-hour soak)
7. Deterministic seeding verification

### Integration Validation
8. Real GPU inference worker integration
9. Batch size monitoring (≥32 average)
10. GPU utilization monitoring (≥80% sustained)
11. Self-play game generation (≥200 games/hour)
12. 4-6 hour mini training run (extrapolate to 48 hours)

## Critical Success Criteria

The implementation is considered successful when ALL of these are met:

1. ✅ MCTS achieves ≥30,000 sims/sec (8 threads, mock inference)
2. ✅ Thread scaling efficiency ≥75% (≥6x speedup with 8 threads)
3. ✅ GPU batch sizes average ≥32 samples
4. ✅ Sustained GPU utilization ≥80% during self-play
5. ✅ Self-play generates ≥200 games/hour
6. ✅ All contract tests pass
7. ✅ Memory usage <1GB for typical searches
8. ✅ 48-hour Gomoku training becomes feasible

**FAILURE TO MEET ANY CRITERIA MEANS THE SYSTEM REMAINS UNFIT FOR PRODUCTION.**

## References

- **Detailed Analysis**: See `docs/performance/mcts_performance_crisis_analysis.md`
- **Specification**: `specs/001-goal-create-spec/spec.md` (FR-018 through FR-022)
- **MCTS Implementation**: `src/core/mcts.py` (current Python loop to be replaced)
- **C++ Tree**: `cpp_extensions/mcts/tree.hpp/cpp` (working correctly)
- **Game Bindings**: `cpp_extensions/games/python_bindings.cpp` (get_legal_moves fixed)
