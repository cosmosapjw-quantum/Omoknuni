# Pull Request Checklist: MCTS Throughput Optimization

**Specification**: Spec 004 - MCTS Throughput Recovery
**Authority**: CONSTITUTION.md v1.1, SPECIFICATION.md v1.1, TECHNICAL_PLAN.md v2.0
**Target**: ≥8,000 sims/sec (2.1× baseline, 3.7× current)
**Last Updated**: 2025-10-13

---

## 🎯 Constitutional Compliance (MANDATORY)

### ✅ Neural Network Architecture Constraints

**REQUIREMENT**: Neural network MUST remain in Python PyTorch (CONSTITUTION.md §2.1)

- [ ] **Confirm NO libtorch**: Code does not link against libtorch C++ library
  - Check: No `#include <torch/torch.h>` or `#include <torch/script.h>` in C++ files
  - Check: No `find_package(Torch)` or `torch::jit::load()` in CMakeLists.txt/setup.py
  - Command: `grep -r "torch/torch.h\|torch/script.h\|torch::jit" cpp_extensions/`
  - Expected: No matches

- [ ] **Confirm NO TensorRT**: Code does not use TensorRT for inference acceleration
  - Check: No `#include <NvInfer.h>` or TensorRT API calls
  - Check: No `.trt`, `.engine`, or `.plan` model files in repository
  - Command: `grep -r "NvInfer\|tensorrt\|TRT" cpp_extensions/ src/`
  - Expected: No matches

- [ ] **Confirm NO ONNX**: Code does not convert models to ONNX format for deployment
  - Check: No `torch.onnx.export()` calls in production code (test/research OK)
  - Check: No ONNX Runtime (`#include <onnxruntime_cxx_api.h>`) in C++ code
  - Check: No `.onnx` model files in production paths (models/, checkpoints/)
  - Command: `grep -r "torch.onnx.export\|onnxruntime" src/ cpp_extensions/`
  - Expected: No matches (or only in tests/research with clear comments)

- [ ] **Confirm Python PyTorch Inference**: Neural network calls remain in Python
  - Check: `model.forward()` or `model()` calls are in Python files (src/neural/, src/core/)
  - Check: C++ code only creates DLPack tensors, does NOT perform inference
  - Evidence: Point to specific Python files with inference calls (e.g., `src/neural/inference_bridge.py`)

**Rationale Box** (MANDATORY if any violations):
```
If any of the above checks fail, explain why and provide constitutional justification:

Example:
- ONNX export found in src/research/model_analysis.py
- Justification: Research-only code for model introspection, not production inference
- Gatekeeper: Confirmed with @cosmosapjw-quantum that research ONNX is acceptable
```

---

## 📊 Performance Reproducibility (CRITICAL)

### ✅ Throughput Measurement Evidence

**REQUIREMENT**: All throughput claims MUST be reproducible (CONSTITUTION.md §3.2)

- [ ] **Baseline Configuration Documented**: 3,831 sims/sec config identified or new baseline established
  - If reproducing baseline: Provide exact command, config, and seed
  - If establishing new baseline: Provide justification (T017 investigation summary)
  - Evidence: Link to T017 investigation report or configuration file

- [ ] **Throughput Command Reproducible**: Exact benchmark command provided
  ```bash
  # MANDATORY: Fill in actual command used
  python scripts/benchmark_throughput.py \
    --game gomoku \
    --simulations 10000 \
    --threads 8 \
    --batch-size 64 \
    --timeout 1.0 \
    --seed 42 \
    --iterations 5
  ```
  - [ ] Command copy-pasteable from PR description
  - [ ] Hardware specified: AMD Ryzen 9 5900X + NVIDIA RTX 3060 Ti
  - [ ] Python version: 3.12
  - [ ] PyTorch version: (fill in)
  - [ ] CUDA version: (fill in)

- [ ] **Results Attached**: Benchmark artifacts committed to repository
  - [ ] CSV file: `profiling_results/pr_<PR_NUMBER>_throughput.csv` (raw data)
  - [ ] JSON file: `profiling_results/pr_<PR_NUMBER>_summary.json` (metadata + summary)
  - [ ] Plot/Chart: `docs/performance/pr_<PR_NUMBER>_scaling.png` (optional but recommended)
  - [ ] Results Schema: Follows TECHNICAL_PLAN.md §D.3 benchmark result schema

- [ ] **Statistical Validity**: Results meet reproducibility criteria
  - [ ] Iterations: ≥3 (minimum), ≥5 (recommended)
  - [ ] Coefficient of Variation: CV < 5% (must pass)
  - [ ] Confidence Interval: 95% CI reported for mean throughput
  - Example:
    ```
    Mean: 8,234 sims/sec
    Std Dev: 127.3 sims/sec
    CV: 1.54% (PASS: <5%)
    95% CI: [8,089, 8,412] sims/sec
    ```

- [ ] **Comparison to Baseline**: Improvement quantified
  - [ ] Current vs Baseline: 2,147 → _____ sims/sec (+____ %, ___× improvement)
  - [ ] vs Target (8,000): _____ / 8,000 = _____ % of target
  - [ ] Status: ⬜ Below target ⬜ At target ⬜ Exceeds target

### ✅ KPI Dashboard (Required for Phase 4 PRs)

**REQUIREMENT**: Track all 8 KPIs from SPECIFICATION.md §8

| KPI | Target | Measured | Status | Evidence |
|-----|--------|----------|--------|----------|
| **KPI1: Throughput** | ≥8,000 sims/sec | _____ | ⬜ PASS / ⬜ FAIL | profiling_results/pr_XXX_throughput.csv |
| **KPI2: Thread Efficiency** | ≥70% @ 8 threads | _____% | ⬜ PASS / ⬜ FAIL | profiling_results/pr_XXX_thread_scaling.csv |
| **KPI3: GPU Utilization** | ≥80% sustained | _____% | ⬜ PASS / ⬜ FAIL | profiling_results/pr_XXX_gpu_util.log |
| **KPI4: Coordination Overhead** | ≤30% of total | _____% | ⬜ PASS / ⬜ FAIL | profiling_results/pr_XXX_breakdown.json |
| **KPI5: Avg Batch Size** | ≥48 (75% of 64) | _____ | ⬜ PASS / ⬜ FAIL | profiling_results/pr_XXX_batch_histogram.csv |
| **KPI6: Collision Rate** | ≤5% | _____% | ⬜ PASS / ⬜ FAIL | Instrumentation counters in summary.json |
| **KPI7: Memory Footprint** | <1.3GB RSS | _____ GB | ⬜ PASS / ⬜ FAIL | `/proc/self/status` snapshot |
| **KPI8: Search Quality** | ≥99.5% win rate | _____% | ⬜ PASS / ⬜ FAIL | tests/quality/test_vs_baseline.py results |

**Notes**:
- KPI PASS/FAIL based on SPECIFICATION.md §8.1-8.2 acceptance thresholds
- All FAIL statuses require justification and mitigation plan
- Phase 4 completion gate: KPI1, KPI3, KPI4, KPI8 MUST pass

---

## 🔧 Parameter Tuning Evidence (If Applicable)

**REQUIREMENT**: Configuration changes must be justified by tuning sweeps (T018/T019)

### ✅ Thread Count Tuning (T018)

If this PR changes `simulation_threads` parameter:

- [ ] **Sweep Performed**: Thread counts [4, 8, 12] tested
  - Command: `pytest tests/performance/test_thread_scaling.py -v`
  - Results: `profiling_results/pr_XXX_thread_sweep.csv` committed
  - Plot: `docs/performance/pr_XXX_thread_scaling.png` (efficiency curve)

- [ ] **Optimal Thread Count Justified**:
  - Chosen value: _____ threads
  - Rationale: _____
  - Trade-off analysis: (e.g., "8 threads = 71% efficiency vs 12 threads = 58% efficiency")

- [ ] **Affinity Strategy Documented** (CONSTITUTION.md §2.2):
  - CCD0 cores: _____
  - CCD1 cores: _____
  - Rationale: (e.g., "CCD0 for MCTS, CCD1 for coordinator to minimize cross-CCD traffic")

### ✅ Batch Size Tuning (T019)

If this PR changes `batch_size` parameter:

- [ ] **Sweep Performed**: Batch sizes [32, 64, 128] tested
  - Command: `pytest tests/performance/test_batch_optimization.py -v`
  - Results: `profiling_results/pr_XXX_batch_sweep.csv` committed
  - Plot: `docs/performance/pr_XXX_batch_gpu_util.png` (throughput vs GPU util)

- [ ] **Optimal Batch Size Justified**:
  - Chosen value: _____
  - GPU Utilization: _____% (must be ≥80% per KPI3)
  - Avg Fill Rate: _____% (e.g., "58.7 / 64 = 91.7% full batches")
  - VRAM Headroom: _____ GB free (must be >1GB for safety)

- [ ] **VRAM Constraint Verified** (8GB RTX 3060 Ti):
  - Model VRAM: _____ GB (FP16 weights + activations)
  - Batch VRAM: _____ GB (batch_size × input tensor size)
  - Total: _____ GB (must be <7GB to leave 1GB safety margin)
  - Command: `nvidia-smi` during benchmark

### ✅ Timeout Tuning (T019)

If this PR changes `batch_timeout_ms` parameter:

- [ ] **Sweep Performed**: Timeouts [0.5, 1.0, 2.0, 5.0] ms tested
  - Command: `pytest tests/performance/test_timeout_tuning.py -v`
  - Results: `profiling_results/pr_XXX_timeout_sweep.csv` committed

- [ ] **Optimal Timeout Justified**:
  - Chosen value: _____ ms
  - GPU Utilization: _____% (higher timeout = worse GPU util due to idle)
  - Thread Idle Time: _____% (lower timeout = more thread idle waiting for batch)
  - Trade-off: (e.g., "1.0ms balances 82% GPU util vs 18% thread idle")

---

## ⚛️ Thread Safety & Contention Audit (CRITICAL)

**REQUIREMENT**: Atomic operations and lock contention must be profiled (CONSTITUTION.md §2.2)

### ✅ Atomic Operations Audit

If this PR modifies MCTS tree operations (visit counts, value updates, virtual loss):

- [ ] **Atomic Operation Inventory**: List all atomic operations added/modified
  - Operations: (e.g., `visit_count_.fetch_add(1, std::memory_order_relaxed)`)
  - Memory Ordering: _____ (must justify relaxed vs acquire/release vs seq_cst)
  - Frequency: _____ ops/sec (estimate or profile)

- [ ] **Contention Profiling**: Measure atomic contention hotspots
  - Tool: `perf c2c` or `cpp_extensions/mcts/instrumentation.cpp` counters
  - Results: `profiling_results/pr_XXX_contention.txt` committed
  - Hotspots: List top 3 contended atomics (if any)
  - Example:
    ```
    Top Atomic Contention:
    1. Node::visit_count_ (63% cache misses, 2.3M conflicts/sec)
    2. VirtualLossManager::vl_array_ (12% cache misses, 410K conflicts/sec)
    3. AsyncInferenceQueue::head_ (5% cache misses, 180K conflicts/sec)
    ```

- [ ] **Mitigation Applied** (if contention >10%):
  - Strategy: (e.g., "Increased virtual loss magnitude to reduce path overlap")
  - Validation: Collision rate ≤5% (KPI6)

### ✅ Lock-Free Queue Verification (T006/T006b)

If this PR modifies `AsyncInferenceQueue`:

- [ ] **Lock-Free Guarantee**: Confirm no mutex/locks in hot path
  - Check: `AsyncInferenceQueue::push()` and `pop()` are wait-free or lock-free
  - Evidence: Code review notes or formal verification

- [ ] **ABA Problem Mitigation**: If using CAS, explain ABA handling
  - Strategy: (e.g., "Tagged pointers with 64-bit generation counter")

- [ ] **Condition Variable Usage** (T006c): Blocking mechanism profiled
  - Check: NO polling loops (`while (!ready) { std::this_thread::sleep_for(10us); }`)
  - Check: `std::condition_variable` or `pthread_cond_wait()` used instead
  - Validation: `strace` or `perf` shows zero CPU usage when threads blocked
  - Evidence: `profiling_results/pr_XXX_strace_idle.log` (no busy-wait syscalls)

### ✅ TSan Clean Build

**REQUIREMENT**: ThreadSanitizer must pass with zero data races

- [ ] **TSan Build Succeeded**:
  ```bash
  export CXXFLAGS="-fsanitize=thread -g -O1"
  pip install -e . --force-reinstall --no-deps
  ```
  - Build log: Clean (no warnings)

- [ ] **TSan Tests Passed**:
  ```bash
  TSAN_OPTIONS="halt_on_error=1" pytest tests/integration/test_gil_release.py -v
  TSAN_OPTIONS="halt_on_error=1" pytest tests/unit/test_thread_safety.py -v
  ```
  - Results: Zero data races detected
  - If races found: MUST be fixed before merge (blocking)

---

## 📈 Profiling Artifacts Committed (MANDATORY)

**REQUIREMENT**: All profiling data must be version-controlled (CONSTITUTION.md §3.2)

### ✅ CSV/JSON Profiling Data

- [ ] **Throughput Data**: `profiling_results/pr_<PR_NUMBER>_throughput.csv`
  - Schema: timestamp, game, threads, batch_size, timeout_ms, mean_sims_per_sec, std_dev, cv
  - Rows: ≥3 (one per iteration)

- [ ] **Thread Scaling Data**: `profiling_results/pr_<PR_NUMBER>_thread_scaling.csv` (if applicable)
  - Schema: threads, mean_throughput, efficiency_pct, cpu_util_pct
  - Rows: At least [4, 8, 12] threads

- [ ] **Batch Optimization Data**: `profiling_results/pr_<PR_NUMBER>_batch_sweep.csv` (if applicable)
  - Schema: batch_size, mean_throughput, gpu_util_pct, avg_batch_fill
  - Rows: At least [32, 64, 128] batch sizes

- [ ] **Summary JSON**: `profiling_results/pr_<PR_NUMBER>_summary.json`
  - Must follow TECHNICAL_PLAN.md §D.3 schema
  - Includes: hardware, configuration, results, metrics, timing_breakdown
  - Validates: `python scripts/validate_benchmark_json.py profiling_results/pr_XXX_summary.json`

### ✅ Profiling Logs (Optional but Recommended)

- [ ] **GPU Utilization Log**: `profiling_results/pr_<PR_NUMBER>_gpu_util.log`
  - Command: `nvidia-smi dmon -s u -i 0 -c 100 > pr_XXX_gpu_util.log` (during benchmark)
  - Extract: Avg GPU % from log for KPI3

- [ ] **Perf Profile**: `profiling_results/pr_<PR_NUMBER>_perf.txt`
  - Command: `perf record -g python scripts/benchmark_throughput.py ... && perf report > pr_XXX_perf.txt`
  - Useful for: Identifying CPU hotspots, validating optimization impact

- [ ] **Python Profiler**: `profiling_results/pr_<PR_NUMBER>_pyspy.svg` (flamegraph)
  - Command: `py-spy record -o pr_XXX_pyspy.svg -- python scripts/benchmark_throughput.py ...`
  - Useful for: Confirming Python overhead <30% (CONSTITUTION.md §2.4)

### ✅ Memory Profiling (If Memory Changes)

- [ ] **Valgrind/ASan Leak Check**: Zero memory leaks
  ```bash
  export ASAN_OPTIONS="detect_leaks=1:halt_on_error=1"
  pytest tests/soak/test_memory_stability.py -v
  ```
  - Results: No leaks detected over 1-hour soak test

- [ ] **Memory Growth Test**: Stable RSS over 1000 searches
  - Command: `python scripts/soak_test.py --duration 3600 --game gomoku`
  - Results: RSS growth <10% over 1 hour (confirms no slow leaks)

---

## ✅ Quality Gates (Search Correctness)

**REQUIREMENT**: Optimizations MUST NOT regress search quality (CONSTITUTION.md §3.1)

### ✅ Policy Agreement

- [ ] **Policy Agreement Test**:
  ```bash
  pytest tests/quality/test_policy_agreement.py -v
  # Compares policy distribution from optimized vs baseline MCTS
  ```
  - Target: ≥95% top-move agreement on 1000-position test set
  - Result: _____% agreement (PASS/FAIL)
  - Evidence: `tests/quality/policy_agreement_report.txt` committed

### ✅ Win Rate Validation

- [ ] **A/B Testing Match**:
  ```bash
  python scripts/compare_search_quality.py \
    --baseline v003 \
    --candidate v004_pr_XXX \
    --games 1000 \
    --simulations 800
  ```
  - Target: ≥99.5% win rate vs baseline (allows 0.5% regression)
  - Result: _____% win rate (_____ wins, _____ losses, _____ draws)
  - Confidence Interval: 95% CI [_____, _____]
  - Status: ⬜ PASS / ⬜ FAIL

### ✅ Value Accuracy

- [ ] **Value MSE Test**:
  ```bash
  pytest tests/quality/test_value_accuracy.py -v
  # Compares value estimates from optimized vs baseline MCTS
  ```
  - Target: MSE ≤ 0.01 (1% error tolerance)
  - Result: MSE = _____ (PASS/FAIL)

**Quality Gate Failure Protocol**:
- If ANY quality gate fails: PR is BLOCKED until root cause identified
- Mitigation options:
  1. Revert optimization
  2. Tune parameters (virtual loss magnitude, exploration constant)
  3. Document acceptable trade-off with approval from @cosmosapjw-quantum

---

## 📝 Documentation Updates

### ✅ Specification Updates (If Architecture Changed)

- [ ] **CONSTITUTION.md**: Updated if constraints changed (requires explicit approval)
- [ ] **SPECIFICATION.md**: Updated if requirements changed (especially Section 12 Q&A)
- [ ] **TECHNICAL_PLAN.md**: Updated if implementation approach changed
- [ ] **TASKS.md**: Task status updated (mark completed, add new tasks if needed)

### ✅ Code Documentation

- [ ] **Inline Comments**: Complex algorithms explained (especially lock-free code, atomics)
- [ ] **API Docstrings**: Public functions documented (pybind11 bindings, Python APIs)
- [ ] **Performance Notes**: Hot paths annotated with expected throughput/latency

### ✅ Changelog Entry

- [ ] **CHANGELOG.md**: Entry added with format:
  ```markdown
  ## [Unreleased] - PR #XXX - <Title>

  **Throughput**: 2,147 → _____ sims/sec (+_____%, _____× improvement)
  **KPIs**: [list PASS/FAIL status of 8 KPIs]

  ### Changed
  - [List technical changes]

  ### Performance
  - [Quantified improvements with benchmark commands]

  ### Testing
  - [New tests added, profiling performed]
  ```

---

## 🧪 Test Coverage

### ✅ Performance Tests

- [ ] **Benchmark Tests Pass**:
  ```bash
  pytest tests/performance/ -v -m benchmark
  ```
  - All existing benchmarks pass with new changes
  - New benchmarks added for new optimizations

### ✅ Regression Tests

- [ ] **Performance Regression Detector**:
  ```bash
  pytest tests/performance/test_throughput_regression.py -v
  # Compares current PR vs baseline (3,831 sims/sec or latest main)
  ```
  - Threshold: No regression >10% without justification
  - Result: _____ sims/sec current vs _____ baseline (_____ % change)

### ✅ Integration Tests

- [ ] **End-to-End Tests Pass**:
  ```bash
  pytest tests/integration/ -v
  # Tests full MCTS search with optimizations enabled
  ```
  - Confirms optimized path produces correct results
  - Tests Python ↔ C++ boundary with DLPack

---

## 🚦 PR Readiness Checklist

**Final Gates** (ALL must be checked before requesting review):

- [ ] ✅ **Constitutional Compliance**: NO libtorch/TensorRT/ONNX, NN stays in Python
- [ ] ✅ **Reproducibility**: Benchmark command + artifacts committed
- [ ] ✅ **KPI Dashboard**: 8 KPIs measured, all critical KPIs PASS
- [ ] ✅ **Thread Safety**: TSan clean, contention profiled
- [ ] ✅ **Profiling Artifacts**: CSV/JSON committed to `profiling_results/pr_XXX_*`
- [ ] ✅ **Quality Gates**: Policy agreement ≥95%, win rate ≥99.5%, value MSE ≤0.01
- [ ] ✅ **Documentation**: Spec updated, inline comments added, changelog entry
- [ ] ✅ **Tests Pass**: Performance tests, regression tests, integration tests all green

**Approval Authority**:
- **Performance PRs** (Phase 4): Requires @cosmosapjw-quantum review + KPI dashboard PASS
- **Architecture PRs** (CONSTITUTION changes): Requires explicit written approval + spec update

---

## 📋 PR Description Template

**Copy this template into PR description**:

```markdown
## Summary
[1-2 sentence summary of optimization]

## Performance Impact
**Throughput**: 2,147 → _____ sims/sec (+_____%, _____× improvement)
**Progress to Target**: _____ / 8,000 sims/sec (_____ % of target)

## KPI Dashboard
| KPI | Target | Result | Status |
|-----|--------|--------|--------|
| KPI1: Throughput | ≥8,000 | _____ | ⬜ PASS / ⬜ FAIL |
| KPI2: Thread Efficiency | ≥70% @ 8T | _____% | ⬜ PASS / ⬜ FAIL |
| KPI3: GPU Util | ≥80% | _____% | ⬜ PASS / ⬜ FAIL |
| KPI4: Coord Overhead | ≤30% | _____% | ⬜ PASS / ⬜ FAIL |
| KPI8: Win Rate | ≥99.5% | _____% | ⬜ PASS / ⬜ FAIL |

## Reproducibility
```bash
# Copy-paste benchmark command
python scripts/benchmark_throughput.py \
  --game gomoku \
  --simulations 10000 \
  --threads 8 \
  --batch-size 64 \
  --timeout 1.0 \
  --seed 42 \
  --iterations 5
```

**Artifacts**:
- Throughput: `profiling_results/pr_XXX_throughput.csv`
- Summary: `profiling_results/pr_XXX_summary.json`
- GPU Util: `profiling_results/pr_XXX_gpu_util.log`

## Constitutional Compliance
- ✅ NO libtorch (verified: grep shows no torch/torch.h)
- ✅ NO TensorRT (verified: grep shows no NvInfer.h)
- ✅ NO ONNX (verified: no production onnx.export calls)
- ✅ Neural network inference remains in Python (src/neural/inference_bridge.py)

## Testing
- ✅ TSan clean (zero data races)
- ✅ Performance tests pass
- ✅ Quality gates pass (policy agreement ___%, win rate ___%)

## Reviewer Notes
[Any special considerations, trade-offs, or areas needing scrutiny]

## Checklist
- [ ] All items in PR_CHECKLIST.md completed
- [ ] Profiling artifacts committed
- [ ] Documentation updated
- [ ] Ready for @cosmosapjw-quantum review
```

---

**END OF CHECKLIST**

**Questions?** Contact @cosmosapjw-quantum or reference:
- CONSTITUTION.md §2-4 (constraints and requirements)
- SPECIFICATION.md §8 (KPIs and benchmarks)
- TECHNICAL_PLAN.md §D (instrumentation and benchmarking)
