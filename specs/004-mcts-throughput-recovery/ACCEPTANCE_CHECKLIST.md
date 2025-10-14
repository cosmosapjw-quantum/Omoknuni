# Acceptance Checklist: MCTS Throughput Recovery & Multi-Actor Self-Play

**Version**: 1.0
**Date**: 2025-10-14
**Authority**: Derived from spec.md v2.0 + plan.md v1.0
**Status**: READY FOR VALIDATION

---

## How to Use This Checklist

**Execution Order**:
1. **Foundation** (T001-T003) → Run operational checks first
2. **CPU Optimizations** (T004-T013) → Run correctness after each optimization
3. **Validation** (T014-T017) → Run full performance acceptance
4. **Advanced** (T018-T026) → Run extended validation for optional phases

**Success Criteria**:
- **MUST PASS**: All items marked `[CRITICAL]`
- **SHOULD PASS**: All items marked `[TARGET]`
- **MAY PASS**: All items marked `[STRETCH]`

**Reporting**:
- Record results in `results/benchmarks/acceptance_report_YYYYMMDD.csv`
- Failed checks trigger rollback via feature flags (see plan.md Section F)

---

## Section 1: Performance Acceptance Checklist

### 1.1 Throughput Requirements (spec.md G1)

**Single MCTS Mode** (Gomoku 15×15, 10k simulations, optimal threads):

- [ ] **[CRITICAL]** `≥6,000 sims/sec` (minimum acceptable)
  - Command: `python -m tests.performance.benchmark_harness --game gomoku --simulations 10000 --threads 8 --iterations 10`
  - Validation: `mean_throughput >= 6000`, `CV < 10%`
  - Current baseline: 2,147 sims/sec
  - Evidence: CSV row with timestamp, git commit

- [ ] **[TARGET]** `≥8,000 sims/sec` (primary goal)
  - Same command as above
  - Validation: `mean_throughput >= 8000`, `CV < 5%`
  - Expected after: OpenMP fix + state pooling + CV sync
  - Evidence: CSV + screenshot of summary

- [ ] **[STRETCH]** `≥10,000 sims/sec` (hardware limit)
  - Same command as above
  - Validation: `mean_throughput >= 10000`
  - Expected after: All optimizations + NN-cache + lightweight NN
  - Evidence: CSV + GPU utilization plot

**Multi-Game Validation** (8 threads, 10k sims):

- [ ] **[TARGET]** Chess 8×8: `≥8,500 sims/sec`
  - Command: `python -m tests.performance.benchmark_harness --game chess --simulations 10000 --threads 8`
  - Validation: `mean_throughput >= 8500`

- [ ] **[TARGET]** Go 9×9: `≥9,000 sims/sec`
  - Command: `python -m tests.performance.benchmark_harness --game go --simulations 10000 --threads 8`
  - Validation: `mean_throughput >= 9000`

**Multi-Actor Mode** (8 concurrent games, 800 sims/move):

- [ ] **[TARGET]** `≥8,000 sims/sec sustained` across all actors
  - Command: `python scripts/selfplay.py --games 8 --simulations 800 --duration 300`
  - Validation: `total_sims / total_time >= 8000`
  - Evidence: Multi-actor telemetry CSV

- [ ] **[TARGET]** `200-300 games/hour` data generation rate
  - Command: `python scripts/selfplay.py --games 12 --duration 3600`
  - Validation: `completed_games >= 200`, `completed_games <= 350` (allow overhead)
  - Evidence: Self-play logs with game completion timestamps

---

### 1.2 GPU Utilization (spec.md G2)

**Single MCTS Mode**:

- [ ] **[CRITICAL]** `≥80% GPU utilization` during search
  - Command: `nvidia-smi dmon -s u -i 0 -c 60` (run during benchmark)
  - Validation: `mean(gpu_util) >= 80%` over 60-second window
  - Current baseline: ~68% (batch=64, timeout=1ms)
  - Evidence: nvidia-smi log + telemetry GPU field

- [ ] **[TARGET]** `85-95% GPU utilization` sustained
  - Same monitoring command
  - Validation: `80 <= mean(gpu_util) <= 95` (avoid thrashing at 100%)
  - Expected after: Multi-actor batching

**Multi-Actor Mode**:

- [ ] **[TARGET]** `≥85% GPU utilization` with optimal actor count
  - Command: `nvidia-smi dmon -s u -i 0` during multi-actor run
  - Validation: `mean(gpu_util) >= 85%` sustained over 5-minute window
  - Evidence: GPU utilization time series plot

- [ ] **[STRETCH]** `90-95% GPU utilization` without saturation
  - Same monitoring
  - Validation: `90 <= mean(gpu_util) <= 95`, `p99_batch_latency <= 50ms`

---

### 1.3 CPU Coordination Overhead (spec.md G3)

**Target Breakdown** (from review.txt analysis):
- Current: 67.2% MCTS overhead (2.5s search, 1.489s idle)
- Target: <30% total coordination overhead

**Component Metrics**:

- [ ] **[CRITICAL]** Thread idle time `≤12%` (vs. 60% current)
  - Measurement: C++ instrumentation in `ContinuousSimulationRunner::get_stats()`
  - Validation: `telemetry.thread_idle_percent <= 12.0`
  - Evidence: Telemetry CSV field `thread_idle_percent`

- [ ] **[TARGET]** Feature preparation `≤5%` of total time
  - Measurement: `dlpack_bridge.cpp` timing instrumentation
  - Validation: `(feature_extraction_ms * batches_submitted) / (total_time_sec * 1000) <= 0.05`
  - Evidence: Telemetry CSV field `feature_extraction_ms`

- [ ] **[TARGET]** Python/GIL overhead `≤5%` of total time
  - Measurement: Python-side batch callback timing
  - Validation: `(python_batch_callback_ms * batches) / (total_time_sec * 1000) <= 0.05`
  - Evidence: New telemetry field (add to T014a if needed)

- [ ] **[TARGET]** Synchronization overhead `≤8%` of total time
  - Measurement: Mutex + CV wait time instrumentation
  - Validation: `(mutex_wait_ms + cv_wait_ms) / (total_time_sec * 1000) <= 0.08`
  - Evidence: Telemetry CSV fields

- [ ] **[TARGET]** Total MCTS overhead `<30%` of total time
  - Calculation: `100 - gpu_overhead_percent < 30`
  - Validation: `mcts_overhead_percent < 30.0`
  - Evidence: Telemetry CSV field `mcts_overhead_percent`

---

### 1.4 Feature Preparation Speed (spec.md G4)

**Critical Fix Validation**:

- [ ] **[CRITICAL]** `≤1.0ms per batch-64` mean time
  - Command: `python scripts/profile_tensor_creation.py --batch-size 64 --iterations 100`
  - Validation: `mean_time_ms <= 1.0`, `CV < 10%`
  - Current baseline: 7.5ms (T-VALID-2 failed)
  - Evidence: Profiling report CSV

- [ ] **[CRITICAL]** OpenMP parallelization active
  - Command: `python scripts/verify_openmp.py`
  - Validation: `compiled_with_openmp == True`, `max_threads >= 12`, no warnings
  - Evidence: OpenMP diagnostic report

- [ ] **[TARGET]** `<0.5ms per batch-64` mean time (optimal)
  - Same profiling command
  - Validation: `mean_time_ms < 0.5`
  - Expected after: OpenMP + false sharing fixes

- [ ] **[STRETCH]** `≤10× variance` in batch times (consistency)
  - Same profiling command
  - Validation: `max_time_ms / min_time_ms <= 10`
  - Evidence: Profiling histogram

---

### 1.5 State Cloning Efficiency (spec.md G5)

**Cloning Reduction**:

- [ ] **[CRITICAL]** `≤1× clone per simulation` (vs. 2-3× current)
  - Measurement: Instrument clone() calls in IGameState
  - Validation: `clone_count / num_simulations <= 1.0`
  - Evidence: Memory profiler output

- [ ] **[TARGET]** `0× extra clones` (only initial state)
  - Measurement: Same instrumentation
  - Validation: `clone_count / num_simulations == 0` (only copyFrom() used)
  - Expected after: Thread-local state pooling (T007-T009)
  - Evidence: Memory allocation trace

- [ ] **[TARGET]** Constant memory allocation over 1000+ searches
  - Command: `python -m memory_profiler scripts/soak_test.py --duration 600`
  - Validation: `memory_growth_rate < 1MB/min` (no leaks)
  - Evidence: Memory profiler graph

---

### 1.6 Thread Scaling Efficiency (spec.md G6)

**Scaling Validation** (Gomoku 15×15, 10k sims):

- [ ] **[CRITICAL]** 4 threads: `≥70% efficiency` minimum
  - Command: `python -m tests.performance.benchmark_harness --threads 4`
  - Calculation: `efficiency = (throughput_4T / throughput_1T) / 4`
  - Validation: `efficiency >= 0.70`
  - Evidence: Thread scaling CSV

- [ ] **[TARGET]** 4 threads: `≥80% efficiency` target
  - Same command
  - Validation: `efficiency >= 0.80`

- [ ] **[TARGET]** 8 threads: `≥70% efficiency`
  - Command: `python -m tests.performance.benchmark_harness --threads 8`
  - Calculation: `efficiency = (throughput_8T / throughput_1T) / 8`
  - Validation: `efficiency >= 0.70`
  - Current baseline: ~45% (3,831 / 8 vs. single-thread estimate)

- [ ] **[STRETCH]** 8 threads: `≥75% efficiency`
  - Same command
  - Validation: `efficiency >= 0.75`

- [ ] **[STRETCH]** 12 threads: `≥60% efficiency`
  - Command: `python -m tests.performance.benchmark_harness --threads 12`
  - Validation: `efficiency >= 0.60` (diminishing returns expected)

---

### 1.7 Batching Metrics (spec.md implicit)

**Batch Size Consistency**:

- [ ] **[TARGET]** Single MCTS: `avg_batch_size ≥51` (0.8× max=64)
  - Measurement: Telemetry field `avg_batch_size`
  - Validation: `mean(avg_batch_size) >= 51` over 10 runs
  - Evidence: Benchmark CSV

- [ ] **[TARGET]** Multi-Actor: `avg_batch_size ≥51` sustained
  - Measurement: Same field in multi-actor mode
  - Validation: `mean(avg_batch_size) >= 51` over 8-12 actors
  - Evidence: Multi-actor telemetry

- [ ] **[STRETCH]** Single MCTS: `avg_batch_size ≥58` (0.9× max)
  - Same measurement
  - Validation: `mean(avg_batch_size) >= 58`

**Batch Timeout Validation**:

- [ ] **[TARGET]** Batch timeout `≤2ms` mean (vs. ≤3ms spec)
  - Measurement: Telemetry field `batch_timeout_ms`
  - Validation: `mean(batch_timeout_ms) <= 2.0`
  - Evidence: Telemetry CSV

- [ ] **[STRETCH]** Batch timeout `≤1ms` mean (optimal)
  - Same measurement
  - Validation: `mean(batch_timeout_ms) <= 1.0`
  - Expected after: Multi-actor high request rate

---

### 1.8 Latency Metrics (spec.md US2)

**Interactive Play Latency** (1600 simulations):

- [ ] **[TARGET]** `p95 latency ≤3 seconds` for 1600 sims
  - Command: `python scripts/interactive_latency_test.py --simulations 1600 --runs 100`
  - Validation: `p95(latency) <= 3.0`, `CV < 10%`
  - Evidence: Latency histogram

- [ ] **[STRETCH]** `p95 latency ≤2 seconds` for 1600 sims
  - Same command
  - Validation: `p95(latency) <= 2.0`

**Queue Latency** (coordination overhead):

- [ ] **[TARGET]** Queue wait time `<10ms` p95 per simulation
  - Measurement: Telemetry field `queue_wait_ms / num_simulations`
  - Validation: `p95(queue_wait_per_sim) < 10.0`
  - Evidence: Queue latency distribution

---

## Section 2: Correctness Checklist

### 2.1 Determinism & Reproducibility (spec.md US3)

**Seed Reproducibility**:

- [ ] **[CRITICAL]** Same seed → same output (±2% tolerance)
  - Command: `python -m pytest tests/unit/test_benchmark_harness.py::TestBenchmarkHarness::test_reproducibility_same_seed -v`
  - Validation: All assertions pass
  - Evidence: Test output log

- [ ] **[CRITICAL]** Coefficient of variation `<5%` for N≥10 runs
  - Command: `python -m tests.performance.benchmark_harness --iterations 10 --seed 42`
  - Validation: `CV(throughput) < 0.05`
  - Evidence: Telemetry statistics CSV

- [ ] **[TARGET]** Identical tree structure with same seed
  - Command: `python tests/integration/test_determinism.py --seed 42 --runs 2`
  - Validation: `tree_hash_run1 == tree_hash_run2`, `visit_counts_match == True`
  - Evidence: Determinism test report

---

### 2.2 Parity Tests (Optimizations vs. Baseline)

**CPU Optimization Parity**:

- [ ] **[CRITICAL]** OpenMP feature extraction: bit-identical output
  - Command: `python tests/integration/test_openmp_parity.py --batch-size 64 --iterations 100`
  - Validation: `max_diff < 1e-6` (FP32 tolerance)
  - Evidence: Parity test report

- [ ] **[CRITICAL]** State pooling: identical search results
  - Command: `python tests/integration/test_state_pooling_parity.py --simulations 1000`
  - Validation: `policy_agreement >= 99.9%`, `value_mse < 1e-4`
  - Evidence: Parity test CSV

- [ ] **[CRITICAL]** Condition variables: no behavioral change
  - Command: `python tests/integration/test_cv_parity.py --simulations 5000`
  - Validation: `visit_count_agreement >= 99.9%`, `top_move_agreement == 100%`
  - Evidence: CV parity report

- [ ] **[TARGET]** Node allocator optimization: identical tree
  - Command: `python tests/integration/test_allocator_parity.py --simulations 10000`
  - Validation: `tree_structure_match == True`, `node_count_match == True`
  - Evidence: Allocator parity report

**Advanced Feature Parity**:

- [ ] **[TARGET]** FP16 mixed precision: `value_mse < 0.01` vs. FP32
  - Command: `python tests/integration/test_fp16_parity.py --positions 1000`
  - Validation: `value_mse < 0.01`, `policy_agreement >= 95%`
  - Evidence: FP16 validation report (T-VALID-1 complete)

- [ ] **[TARGET]** NN-eval cache: policy agreement `≥99.5%`
  - Command: `python tests/integration/test_nn_cache_parity.py --simulations 10000`
  - Validation: `top_move_agreement >= 99.5%`, `cache_hit_rate > 20%`
  - Evidence: Cache parity report

---

### 2.3 Search Quality Preservation (spec.md G8)

**Win Rate vs. Baseline**:

- [ ] **[CRITICAL]** `≥99.5% win rate` vs. baseline (1000+ games)
  - Command: `python scripts/evaluate_strength.py --opponent baseline --games 1000 --simulations 800`
  - Validation: `win_rate >= 0.995`, `confidence_95 == True`
  - Evidence: ELO rating report

- [ ] **[TARGET]** `≥99% win rate` vs. baseline at higher visits
  - Same command with `--simulations 1600`
  - Validation: `win_rate >= 0.99`

**Policy Agreement**:

- [ ] **[CRITICAL]** `≥95% top-move agreement` (1000-position test set)
  - Command: `python scripts/policy_agreement.py --positions tests/fixtures/positions_1000.json`
  - Validation: `top1_agreement >= 0.95`, `top3_agreement >= 0.98`
  - Evidence: Policy agreement CSV

- [ ] **[TARGET]** `≥98% top-move agreement`
  - Same command
  - Validation: `top1_agreement >= 0.98`

**Value Accuracy**:

- [ ] **[CRITICAL]** `value_mse ≤0.01` vs. baseline estimates
  - Command: `python scripts/value_mse.py --positions 1000`
  - Validation: `mse <= 0.01`, `mean_abs_error <= 0.05`
  - Evidence: Value accuracy report

---

### 2.4 Thread Safety & Collision Metrics (spec.md G8)

**Virtual Loss Collision Rate**:

- [ ] **[CRITICAL]** `≤5% path collisions` at 4 threads
  - Measurement: Telemetry field `virtual_loss_collisions / num_simulations`
  - Validation: `collision_rate <= 0.05`
  - Evidence: Telemetry CSV

- [ ] **[TARGET]** `≤8% path collisions` at 8 threads
  - Same measurement
  - Validation: `collision_rate <= 0.08`
  - Current baseline: <0.5% at 4 threads (review.txt line 89)

**Thread Safety (TSan Clean)**:

- [ ] **[CRITICAL]** No data races detected by ThreadSanitizer
  - Command: `TSAN_OPTIONS="history_size=7" python -m pytest tests/integration/test_thread_safety.py -v`
  - Validation: `exit_code == 0`, no "WARNING: ThreadSanitizer" in output
  - Evidence: TSan log (clean)

- [ ] **[CRITICAL]** No deadlocks in 10-minute soak test
  - Command: `timeout 600 python scripts/soak_test.py --threads 12`
  - Validation: `exit_code == 0` (no timeout), `completed == True`
  - Evidence: Soak test completion log

---

### 2.5 NN-Eval Cache Invariants (plan.md Section C)

**Cache Correctness** (if Phase 6 enabled):

- [ ] **[CRITICAL]** Cache hits return identical (π, V) as GPU
  - Command: `python tests/integration/test_nn_cache_correctness.py --samples 10000`
  - Validation: `policy_l2_error < 1e-5`, `value_abs_error < 1e-4`
  - Evidence: Cache correctness report

- [ ] **[CRITICAL]** No stale entries after net_version update
  - Command: `python tests/integration/test_cache_invalidation.py`
  - Validation: `stale_entries_after_update == 0`
  - Evidence: Cache invalidation test log

- [ ] **[TARGET]** Cache hit rate matches expected range
  - Gomoku: 10-30%, Chess: 20-50%, Go: 15-35% (per review.txt)
  - Measurement: Telemetry field `cache_hit_rate`
  - Validation: Game-specific bounds
  - Evidence: Cache hit rate histogram

- [ ] **[TARGET]** No cache corruption under concurrent access
  - Command: `python tests/integration/test_cache_concurrency.py --threads 16 --duration 300`
  - Validation: `hash_collisions == 0`, `corruption_detected == False`
  - Evidence: Concurrency stress test report

**Zobrist Hashing**:

- [ ] **[CRITICAL]** Unique hashes for distinct positions
  - Command: `python tests/unit/test_zobrist_uniqueness.py --positions 100000`
  - Validation: `collision_rate < 1e-6` (birthday paradox expected)
  - Evidence: Hash distribution analysis

- [ ] **[CRITICAL]** Identical hashes for transposed positions (symmetry check)
  - Command: `python tests/unit/test_zobrist_symmetry.py --rotations 8 --reflections 4`
  - Validation: `canonical_hash_consistent == True` (if symmetry reduction enabled)
  - Evidence: Symmetry test report

---

### 2.6 Multi-Actor Correctness (plan.md Section D)

**Process Isolation**:

- [ ] **[CRITICAL]** No GIL contention across actor processes
  - Measurement: Per-actor CPU utilization
  - Validation: `sum(actor_cpu_percent) <= 100 * num_actors` (no superlinear scaling)
  - Evidence: Process CPU usage log

- [ ] **[CRITICAL]** Correct result demultiplexing (no cross-actor pollution)
  - Command: `python tests/integration/test_multi_actor_demux.py --actors 12 --simulations 5000`
  - Validation: `misrouted_results == 0`, `all_results_received == True`
  - Evidence: Demux correctness report

**Fairness Policy**:

- [ ] **[TARGET]** Token bucket backpressure prevents starvation
  - Measurement: Per-actor request counts over 5 minutes
  - Validation: `max(requests_per_actor) / min(requests_per_actor) < 2.0`
  - Evidence: Fairness histogram

- [ ] **[TARGET]** No actor exceeds in-flight cap (256 per CLARIFICATIONS.md)
  - Measurement: Max in-flight requests per actor
  - Validation: `max(in_flight_per_actor) <= 256`
  - Evidence: Backpressure telemetry

---

## Section 3: Operational Checklist

### 3.1 Benchmark Scripts Execution

**Foundation Benchmarks** (T001):

- [ ] **[CRITICAL]** Benchmark harness runs without errors
  - Command: `python -m tests.performance.benchmark_harness --game gomoku --iterations 3`
  - Validation: `exit_code == 0`, CSV created, no exceptions
  - Evidence: Benchmark output log

- [ ] **[CRITICAL]** Telemetry captures all required KPIs
  - Validation: CSV contains all fields from `telemetry.py` (60+ columns)
  - Evidence: CSV header inspection

- [ ] **[TARGET]** Quick benchmark completes in `<5 minutes` (3 iterations)
  - Command: `time python -m tests.performance.benchmark_harness --iterations 3`
  - Validation: `elapsed_time < 300` seconds
  - Evidence: Execution timestamp

**OpenMP Verification** (T002):

- [ ] **[CRITICAL]** OpenMP verification script runs successfully
  - Command: `python scripts/verify_openmp.py`
  - Validation: `exit_code == 0`, "✅ PASS: OpenMP compiled correctly"
  - Evidence: OpenMP verification report

- [ ] **[CRITICAL]** Runtime environment check passes
  - Same command
  - Validation: `OMP_NUM_THREADS == 12` (or explicitly set)
  - Evidence: Environment variable report

**Comprehensive Benchmarks** (T014):

- [ ] **[CRITICAL]** Thread scaling benchmark runs clean
  - Command: `python scripts/benchmark_thread_scaling.py --threads 1,2,4,6,8,10,12`
  - Validation: `exit_code == 0`, CSV with 7 rows
  - Evidence: Thread scaling CSV

- [ ] **[TARGET]** Batch size tuning benchmark runs clean
  - Command: `python scripts/benchmark_batch_size.py --sizes 16,32,48,64,96,128`
  - Validation: `exit_code == 0`, CSV with 6 rows
  - Evidence: Batch size CSV

- [ ] **[TARGET]** Timeout tuning benchmark runs clean
  - Command: `python scripts/benchmark_timeout.py --timeouts 0.5,1.0,2.0,3.0,5.0`
  - Validation: `exit_code == 0`, CSV with 5 rows
  - Evidence: Timeout CSV

**Ablation Studies** (T015):

- [ ] **[CRITICAL]** Ablation benchmark runs successfully
  - Command: `python scripts/ablation_study.py --configs baseline,openmp,state_pooling,condition_vars,all_cpu,all_optimizations`
  - Validation: `exit_code == 0`, CSV with 6 rows (one per config)
  - Evidence: Ablation CSV

- [ ] **[TARGET]** Each ablation config produces valid telemetry
  - Validation: All 6 configs have `throughput > 0`, `gpu_util_percent > 0`
  - Evidence: Ablation telemetry inspection

**Baseline Investigation** (T016):

- [ ] **[TARGET]** Baseline investigation completes (time-boxed 2 days)
  - Command: `python scripts/investigate_baseline.py --max-duration 172800`
  - Validation: `exit_code == 0`, findings documented in report
  - Evidence: Baseline investigation report.md

---

### 3.2 CSV Output Validation

**CSV Schema Compliance**:

- [ ] **[CRITICAL]** `benchmark_history.csv` exists and is parseable
  - Validation: `pd.read_csv("results/benchmarks/benchmark_history.csv")` succeeds
  - Evidence: DataFrame shape `(N, 60+)` where N = total runs

- [ ] **[CRITICAL]** All required telemetry fields present
  - Validation: CSV headers include all fields from `telemetry.py` dataclass
  - Required fields (minimum):
    - `throughput`, `gpu_util_percent`, `cpu_util_percent`, `avg_batch_size`
    - `feature_extraction_ms`, `thread_idle_percent`, `memory_peak_mb`
    - `timestamp`, `git_commit`, `config_*` (flattened config)
  - Evidence: CSV header list

- [ ] **[TARGET]** CSV accumulates across sessions (append mode)
  - Test: Run benchmark twice, check row count increases
  - Validation: `rows_after == rows_before + new_runs`
  - Evidence: CSV row count log

- [ ] **[TARGET]** No data corruption in CSV (valid JSON in config fields)
  - Validation: All `config_*` fields are valid primitives (no JSON strings)
  - Evidence: CSV data type check

**Git Metadata in CSV**:

- [ ] **[CRITICAL]** Git commit hash recorded in every row
  - Validation: `git_commit` field != "unknown", length == 8
  - Evidence: CSV sample showing commit hashes

- [ ] **[TARGET]** Git branch recorded in every row
  - Validation: `git_branch` field != "unknown"
  - Current branch: `004-mcts-throughput-recovery`
  - Evidence: CSV sample showing branch names

---

### 3.3 Plots & Visualization

**Automated Plot Generation** (T017):

- [ ] **[CRITICAL]** KPI dashboard renders without errors
  - Command: `python scripts/generate_kpi_dashboard.py --input results/benchmarks/benchmark_history.csv --output results/plots/`
  - Validation: `exit_code == 0`, PNG files created
  - Evidence: Dashboard PNG files

- [ ] **[TARGET]** Throughput time series plot exists
  - File: `results/plots/throughput_timeseries.png`
  - Validation: File exists, shows data points, trendline visible
  - Evidence: Plot screenshot

- [ ] **[TARGET]** Thread scaling plot exists
  - File: `results/plots/thread_scaling.png`
  - Validation: Shows throughput vs. threads (1-12), efficiency line
  - Evidence: Plot screenshot

- [ ] **[TARGET]** GPU utilization histogram exists
  - File: `results/plots/gpu_utilization_histogram.png`
  - Validation: Shows distribution, target line at 80%
  - Evidence: Plot screenshot

- [ ] **[TARGET]** Ablation study comparison plot exists
  - File: `results/plots/ablation_comparison.png`
  - Validation: Bar chart with 6 configurations, baseline normalized
  - Evidence: Plot screenshot

- [ ] **[STRETCH]** Interactive dashboard (optional, if Streamlit/Dash implemented)
  - Command: `streamlit run scripts/dashboard.py`
  - Validation: Dashboard loads, shows live metrics
  - Evidence: Dashboard screenshot

**Regression Plots**:

- [ ] **[TARGET]** Regression test time series exists
  - File: `results/plots/regression_history.png`
  - Validation: Shows throughput over git commits, alert threshold line
  - Evidence: Plot screenshot

- [ ] **[TARGET]** Performance regression alerts trigger correctly
  - Test: Inject 10% throughput drop, verify alert
  - Validation: Alert email/log generated
  - Evidence: Alert log

---

### 3.4 Logging & Diagnostics

**Logging Configuration**:

- [ ] **[CRITICAL]** Benchmarks log to `results/benchmarks/logs/` directory
  - Validation: Log files created with timestamp names
  - Evidence: `ls results/benchmarks/logs/`

- [ ] **[TARGET]** Log level configurable via environment variable
  - Test: `LOG_LEVEL=DEBUG python -m tests.performance.benchmark_harness`
  - Validation: Debug messages appear in log
  - Evidence: Log file excerpt

- [ ] **[TARGET]** Errors logged with full stack traces
  - Test: Trigger intentional error (invalid config)
  - Validation: Stack trace in log file
  - Evidence: Error log excerpt

**Diagnostic Tools**:

- [ ] **[CRITICAL]** OpenMP diagnostic tool works
  - Command: `python scripts/verify_openmp.py`
  - Validation: Reports compilation status, thread count, environment vars
  - Evidence: Diagnostic report

- [ ] **[TARGET]** Memory profiler integration works
  - Command: `python -m memory_profiler scripts/soak_test.py --duration 60`
  - Validation: Memory usage graph generated
  - Evidence: Memory profiler output

- [ ] **[TARGET]** GPU profiler integration works
  - Command: `nvidia-smi dmon -s u -i 0 -c 10 | tee results/gpu_profile.log`
  - Validation: GPU utilization logged
  - Evidence: GPU profile log

---

### 3.5 CI/CD Integration (Future)

**Regression Test Suite** (T028):

- [ ] **[TARGET]** Pytest performance markers work
  - Command: `pytest -m performance -v`
  - Validation: All performance tests discovered and run
  - Evidence: Pytest summary

- [ ] **[TARGET]** Performance regression test fails on <95% baseline
  - Test: Inject performance degradation
  - Command: `pytest tests/performance/test_regression.py`
  - Validation: Test fails with clear error message
  - Evidence: Test failure log

- [ ] **[STRETCH]** CI pipeline runs performance tests on merge
  - GitHub Actions: `.github/workflows/performance.yml`
  - Validation: Workflow runs on PR, posts results as comment
  - Evidence: GitHub Actions log

**Benchmark Artifacts**:

- [ ] **[TARGET]** CI uploads benchmark CSVs as artifacts
  - Validation: Artifacts available for download
  - Evidence: GitHub Actions artifacts section

- [ ] **[TARGET]** CI generates performance comparison report
  - Validation: Report shows before/after throughput
  - Evidence: CI report in PR comment

---

## Section 4: Advanced Feature Validation (Optional Phases)

### 4.1 NN-Eval Cache Validation (Phase 6 - T018-T021)

**If NN-cache enabled**:

- [ ] **[TARGET]** Cache hit rate meets expectations
  - Gomoku: 10-30%, Chess: 20-50%, Go: 15-35%
  - Validation: Telemetry field `cache_hit_rate` in expected range
  - Evidence: Cache hit rate CSV

- [ ] **[TARGET]** Cache memory usage within budget
  - Budget: 2M entries × 224 bytes = 448 MB (per CLARIFICATIONS.md)
  - Validation: `cache_memory_mb <= 500` (allow overhead)
  - Evidence: Memory profiler

- [ ] **[TARGET]** SLRU eviction working correctly
  - Test: Fill cache, verify LRU eviction
  - Validation: `evictions > 0`, no OOM errors
  - Evidence: Cache eviction telemetry

- [ ] **[TARGET]** Sharded cache reduces contention
  - Validation: `cache_lock_contention_ms < 1% * total_time_ms`
  - Evidence: Lock profiler output

---

### 4.2 Multi-Actor Self-Play Validation (Phase 5 - T022-T026)

**If multi-actor enabled**:

- [ ] **[TARGET]** 8-12 actors achieve target throughput
  - Command: `python scripts/selfplay.py --games 12 --simulations 800 --duration 600`
  - Validation: `total_sims / 600 >= 8000`
  - Evidence: Multi-actor telemetry

- [ ] **[TARGET]** Fairness policy prevents starvation
  - Validation: `max(actor_throughput) / min(actor_throughput) < 2.0`
  - Evidence: Per-actor throughput histogram

- [ ] **[TARGET]** Token bucket backpressure active
  - Validation: `backpressure_events > 0` (at least some throttling)
  - Evidence: Backpressure event log

- [ ] **[TARGET]** Data generation rate meets target
  - Validation: `games_per_hour >= 200`, `games_per_hour <= 300`
  - Evidence: Self-play completion log

---

## Section 5: Final Acceptance Criteria

**CRITICAL PATH** (must all pass):

- [ ] **[CRITICAL]** Throughput ≥6,000 sims/sec (Gomoku, single MCTS)
- [ ] **[CRITICAL]** GPU utilization ≥80% during search
- [ ] **[CRITICAL]** Thread idle ≤12% (vs. 60% baseline)
- [ ] **[CRITICAL]** Feature extraction ≤1.0ms per batch-64
- [ ] **[CRITICAL]** State cloning ≤1× per simulation
- [ ] **[CRITICAL]** Thread scaling 4T efficiency ≥70%
- [ ] **[CRITICAL]** Win rate ≥99.5% vs. baseline (1000 games)
- [ ] **[CRITICAL]** Policy agreement ≥95% (1000 positions)
- [ ] **[CRITICAL]** Value MSE ≤0.01 vs. baseline
- [ ] **[CRITICAL]** No data races (TSan clean)
- [ ] **[CRITICAL]** Benchmark scripts run clean, CSVs valid

**TARGET PATH** (should pass for production):

- [ ] **[TARGET]** Throughput ≥8,000 sims/sec (primary goal)
- [ ] **[TARGET]** GPU utilization 85-95% sustained
- [ ] **[TARGET]** CPU coordination overhead <30%
- [ ] **[TARGET]** Thread scaling 8T efficiency ≥70%
- [ ] **[TARGET]** Multi-actor 200-300 games/hour
- [ ] **[TARGET]** All plots render correctly

**STRETCH PATH** (nice to have):

- [ ] **[STRETCH]** Throughput ≥10,000 sims/sec (hardware limit)
- [ ] **[STRETCH]** Thread scaling 8T efficiency ≥75%
- [ ] **[STRETCH]** NN-cache working (if Phase 6 enabled)

---

## Section 6: Rollback Triggers

**CRITICAL FAILURES** (immediate rollback):

- [ ] Throughput regression >20% from previous commit
- [ ] TSan detects data races
- [ ] Win rate <95% vs. baseline
- [ ] Memory leak >10MB/min growth
- [ ] Deadlock in soak test

**Rollback Procedure** (plan.md Section F):

1. Set feature flag to `False` in config
2. Re-run acceptance benchmarks
3. Verify rollback restores baseline performance
4. File bug report with telemetry CSV
5. Revert commit if flag rollback insufficient

**Feature Flags** (from T003):
- `OPENMP_ENABLED`: Feature extraction parallelization
- `STATE_POOLING_ENABLED`: Thread-local state reuse
- `CONDITION_VARS_ENABLED`: CV synchronization
- `NODE_ALLOCATOR_OPTIMIZED`: Over-allocation in arenas
- `NN_CACHE_ENABLED`: Transposition table cache

---

## Appendix A: Command Reference

**Quick Validation Commands**:

```bash
# Full acceptance validation (run after all optimizations)
python scripts/run_acceptance_suite.py --output results/acceptance_report.csv

# Performance-only validation
python scripts/run_performance_suite.py --critical-only

# Correctness-only validation
python scripts/run_correctness_suite.py --parity-tests

# Operational validation
python scripts/run_ops_suite.py --csv-check --plot-check
```

**Manual Checklist Execution**:

```bash
# Section 1: Performance
pytest tests/performance/ -v -m acceptance

# Section 2: Correctness
pytest tests/integration/ -v -m parity

# Section 3: Operational
pytest tests/operational/ -v -m csv_output
```

---

## Appendix B: Evidence Artifacts

**Required Evidence for Acceptance** (commit to repo):

1. `results/acceptance_report_YYYYMMDD.csv` - Full telemetry
2. `results/benchmarks/benchmark_history.csv` - Historical data
3. `results/plots/` - All KPI plots (PNG)
4. `results/logs/acceptance_YYYYMMDD.log` - Execution log
5. `results/validation/` - Parity test reports
6. `results/profiling/` - Profiler outputs (OpenMP, memory, GPU)
7. `docs/acceptance_summary.md` - Human-readable summary

**Automated Report Generation**:

```bash
# Generate full acceptance report
python scripts/generate_acceptance_report.py \
  --csv results/acceptance_report.csv \
  --plots results/plots/ \
  --output docs/acceptance_summary.md
```

---

**END OF CHECKLIST v1.0**

**Next Steps**:
1. Execute T001-T003 (Foundation) → Validate Operational checklist (Section 3)
2. Execute T004-T013 (CPU Optimizations) → Validate Correctness after each (Section 2)
3. Execute T014-T017 (Validation) → Validate Performance acceptance (Section 1)
4. Generate acceptance report → Commit evidence artifacts
5. Decision: Proceed to Phase 5 (Multi-Actor) or Phase 6 (NN-Cache) based on results
