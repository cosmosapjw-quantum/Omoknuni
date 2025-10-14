# Acceptance Checklist Summary

**Document**: [ACCEPTANCE_CHECKLIST.md](./ACCEPTANCE_CHECKLIST.md)
**Version**: 1.0
**Date**: 2025-10-14
**Total Checks**: 150+ items across 3 categories

---

## Quick Reference

### Section 1: Performance Acceptance (60 checks)

**Primary KPIs** (CRITICAL path):
- ✅ Throughput ≥6,000 sims/sec (minimum) → ≥8,000 (target) → ≥10,000 (stretch)
- ✅ GPU utilization ≥80% (minimum) → 85-95% (target)
- ✅ Thread idle ≤12% (vs. 60% current)
- ✅ Feature extraction ≤1.0ms per batch-64 (vs. 7.5ms current)
- ✅ State cloning ≤1× per simulation (vs. 2-3× current)
- ✅ Thread scaling 4T≥70%, 8T≥70% efficiency

**Commands**:
```bash
# Primary throughput test
python -m tests.performance.benchmark_harness --game gomoku --simulations 10000 --threads 8 --iterations 10

# GPU monitoring
nvidia-smi dmon -s u -i 0 -c 60

# Feature extraction profiling
python scripts/profile_tensor_creation.py --batch-size 64 --iterations 100

# Thread scaling
python scripts/benchmark_thread_scaling.py --threads 1,2,4,6,8,10,12
```

---

### Section 2: Correctness Validation (45 checks)

**Determinism** (CRITICAL):
- ✅ Same seed → same output (±2% tolerance)
- ✅ CV < 5% for N≥10 runs

**Parity Tests** (CRITICAL):
- ✅ OpenMP: bit-identical output vs. sequential
- ✅ State pooling: identical search results
- ✅ Condition variables: no behavioral change
- ✅ FP16 mixed precision: value MSE < 0.01

**Search Quality** (CRITICAL):
- ✅ Win rate ≥99.5% vs. baseline (1000 games)
- ✅ Policy agreement ≥95% (1000 positions)
- ✅ Value MSE ≤0.01

**Thread Safety** (CRITICAL):
- ✅ TSan clean (no data races)
- ✅ No deadlocks in 10-minute soak test
- ✅ Collision rate ≤5% at 4 threads

**Commands**:
```bash
# Determinism
python -m pytest tests/unit/test_benchmark_harness.py::TestBenchmarkHarness::test_reproducibility_same_seed -v

# Parity tests
python tests/integration/test_openmp_parity.py --batch-size 64
python tests/integration/test_state_pooling_parity.py --simulations 1000
python tests/integration/test_cv_parity.py --simulations 5000

# Search quality
python scripts/evaluate_strength.py --opponent baseline --games 1000 --simulations 800
python scripts/policy_agreement.py --positions tests/fixtures/positions_1000.json

# Thread safety
TSAN_OPTIONS="history_size=7" python -m pytest tests/integration/test_thread_safety.py -v
timeout 600 python scripts/soak_test.py --threads 12
```

---

### Section 3: Operational Readiness (45 checks)

**Benchmark Scripts** (CRITICAL):
- ✅ Benchmark harness runs without errors
- ✅ OpenMP verification script passes
- ✅ Thread scaling benchmark runs clean
- ✅ Ablation study completes successfully

**CSV Output** (CRITICAL):
- ✅ `benchmark_history.csv` exists and is parseable
- ✅ All 60+ telemetry fields present
- ✅ Git commit/branch recorded in every row
- ✅ CSV accumulates across sessions (append mode)

**Plots & Visualization** (TARGET):
- ✅ KPI dashboard renders without errors
- ✅ Throughput time series plot
- ✅ Thread scaling plot
- ✅ GPU utilization histogram
- ✅ Ablation comparison plot

**Commands**:
```bash
# Benchmark harness
python -m tests.performance.benchmark_harness --game gomoku --iterations 3

# OpenMP verification
python scripts/verify_openmp.py

# Ablation study
python scripts/ablation_study.py --configs baseline,openmp,state_pooling,condition_vars,all_cpu,all_optimizations

# Plot generation
python scripts/generate_kpi_dashboard.py --input results/benchmarks/benchmark_history.csv --output results/plots/

# CSV validation
python -c "import pandas as pd; df = pd.read_csv('results/benchmarks/benchmark_history.csv'); print(df.shape)"
```

---

## Critical Path (MUST ALL PASS)

**11 Critical Checks** (blocking acceptance):

1. ✅ Throughput ≥6,000 sims/sec (Gomoku, single MCTS)
2. ✅ GPU utilization ≥80% during search
3. ✅ Thread idle ≤12% (vs. 60% baseline)
4. ✅ Feature extraction ≤1.0ms per batch-64
5. ✅ State cloning ≤1× per simulation
6. ✅ Thread scaling 4T efficiency ≥70%
7. ✅ Win rate ≥99.5% vs. baseline (1000 games)
8. ✅ Policy agreement ≥95% (1000 positions)
9. ✅ Value MSE ≤0.01 vs. baseline
10. ✅ No data races (TSan clean)
11. ✅ Benchmark scripts run clean, CSVs valid

**Fast Validation Command**:
```bash
# Run critical-only checks (automated suite)
python scripts/run_acceptance_suite.py --critical-only --output results/acceptance_report.csv
```

---

## Target Path (Production Quality)

**Additional 15 Target Checks** (should pass for production):

- Throughput ≥8,000 sims/sec (primary goal)
- GPU utilization 85-95% sustained
- CPU coordination overhead <30%
- Thread scaling 8T efficiency ≥70%
- Multi-actor 200-300 games/hour
- All plots render correctly
- NN-cache working (if Phase 6 enabled)

---

## Evidence Artifacts

**Required for Acceptance** (commit to repo):

1. `results/acceptance_report_YYYYMMDD.csv` - Full telemetry
2. `results/benchmarks/benchmark_history.csv` - Historical data
3. `results/plots/*.png` - All KPI plots
4. `results/logs/acceptance_YYYYMMDD.log` - Execution log
5. `results/validation/*_report.md` - Parity test reports
6. `docs/acceptance_summary.md` - Human-readable summary

**Automated Report**:
```bash
python scripts/generate_acceptance_report.py \
  --csv results/acceptance_report.csv \
  --plots results/plots/ \
  --output docs/acceptance_summary.md
```

---

## Rollback Triggers

**CRITICAL FAILURES** (immediate rollback):

- Throughput regression >20% from previous commit
- TSan detects data races
- Win rate <95% vs. baseline
- Memory leak >10MB/min growth
- Deadlock in soak test

**Rollback via Feature Flags** (T003):
```python
# config/performance.yml
features:
  openmp_enabled: false  # Rollback OpenMP
  state_pooling_enabled: false  # Rollback state pooling
  condition_vars_enabled: false  # Rollback CV sync
  node_allocator_optimized: false  # Rollback allocator
  nn_cache_enabled: false  # Rollback NN-cache
```

---

## Execution Timeline

**Phase 0-1** (T001-T013):
1. Run **Operational checks** (Section 3) → Verify infrastructure
2. Run **Correctness checks** (Section 2) → After each optimization
3. Run **Performance checks** (Section 1) → After all CPU optimizations

**Phase 2** (T014-T017):
1. Run **full Performance suite** → Validate all KPIs
2. Generate **plots and reports** → Evidence artifacts
3. **Decision point**: Accept or rollback

**Phase 5-6** (Optional):
- Multi-actor validation (if enabled)
- NN-cache validation (if enabled)

---

## Quick Start

**Day 1** (After T001-T003 complete):
```bash
# 1. Verify operational readiness
python -m tests.performance.benchmark_harness --iterations 3
python scripts/verify_openmp.py

# 2. Establish baseline
python scripts/benchmark_thread_scaling.py --threads 1,2,4,8
```

**After Each Optimization** (T004-T013):
```bash
# 1. Run parity test
python tests/integration/test_<optimization>_parity.py

# 2. Run quick throughput check
python -m tests.performance.benchmark_harness --iterations 3

# 3. Check for regressions
python scripts/check_regression.py --baseline results/benchmarks/baseline.csv
```

**Final Acceptance** (After T014-T017):
```bash
# 1. Full acceptance suite
python scripts/run_acceptance_suite.py --output results/acceptance_report.csv

# 2. Generate report
python scripts/generate_acceptance_report.py \
  --csv results/acceptance_report.csv \
  --output docs/acceptance_summary.md

# 3. Commit evidence
git add results/ docs/acceptance_summary.md
git commit -m "perf: Acceptance validation complete - 8,000 sims/sec achieved"
```

---

**For detailed checklists, see**: [ACCEPTANCE_CHECKLIST.md](./ACCEPTANCE_CHECKLIST.md)
