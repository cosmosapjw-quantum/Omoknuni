# Comprehensive Profiling Modes

## Overview

The profiling suite now provides **three distinct modes** for different use cases, ranging from quick smoke tests to production-grade analysis with statistical rigor.

## Mode Comparison

| Mode | Duration | Trials | Use Case |
|------|----------|--------|----------|
| **--quick** | ~1 min | 4 | Smoke testing, development iteration |
| **--full** | ~15 min | 135 | Thorough exploration, optimization work |
| **--production** | ~40 min | 560 | Release validation, regression detection |

---

## 1. Quick Mode (~1 minute)

**Purpose**: Fast smoke testing during development

**Command**:
```bash
./scripts/run_profiling_suite.sh --quick
```

**Configuration**:
- Simulations: 100, 200
- Threads: 1, 4
- Batch sizes: 64
- Repetitions: 1
- **Total trials**: 2 × 2 × 1 × 1 = **4 trials**
- Wall-clock validation: 100 sims × 3 runs

**When to use**:
- Quick sanity checks after code changes
- Development iteration
- CI/CD smoke tests
- Verifying profiling infrastructure works

**What it reveals**:
- Rough performance ballpark (~2,000 sims/sec)
- Major bottlenecks (state cloning visible)
- System is functional

**What it misses**:
- Statistical significance (no repetitions)
- Thread scaling behavior (only 2 thread counts)
- Sustained load patterns (max 200 sims)
- Batch size optimization

---

## 2. Full Mode (~15 minutes) [DEFAULT]

**Purpose**: Thorough bottleneck exploration for optimization work

**Command**:
```bash
./scripts/run_profiling_suite.sh --full
# or
./scripts/run_profiling_suite.sh  # (full is default)
```

**Configuration**:
- Simulations: 2,000, 4,000, 8,000
- Threads: 1, 2, 4, 6, 8
- Batch sizes: 32, 64, 128
- Repetitions: 3
- **Total trials**: 3 × 5 × 3 × 3 = **135 trials**
- Wall-clock validation: 4,000 sims × 10 runs

**When to use**:
- Performance optimization sessions
- Bottleneck identification
- Thread count tuning
- Batch size optimization
- Before/after optimization comparisons

**What it reveals**:
- Performance at target scale (8,000 sims)
- Thread scaling efficiency (1→2→4→6→8)
- Batch size sweet spot (32/64/128)
- Statistical variance (CV via 3 reps)
- Bottleneck priorities with confidence

**Timing breakdown**:
- Wall-clock: ~1 minute (40 sims × 10 runs × 0.25s)
- Campaign: ~13 minutes (135 trials × 6s avg)
- Analysis: ~1 minute

---

## 3. Production Mode (~40 minutes)

**Purpose**: Production-grade analysis with full statistical rigor

**Command**:
```bash
./scripts/run_profiling_suite.sh --production
```

**Configuration**:
- Simulations: 2,000, 4,000, 8,000, 16,000
- Threads: 1, 2, 4, 6, 8, 10, 12
- Batch sizes: 16, 32, 64, 128
- Repetitions: 5
- **Total trials**: 4 × 7 × 4 × 5 = **560 trials**
- Wall-clock validation: 8,000 sims × 20 runs

**When to use**:
- Release candidate validation
- Performance regression detection
- Production deployment verification
- Benchmark suite for papers/reports
- A/B testing of major changes

**What it reveals**:
- Full thread scaling curve (1→12 cores)
- Batch size impact across range (16→128)
- Performance beyond target (16k sims)
- High-confidence statistics (5 reps, CV < 5%)
- CCD affinity effects (Ryzen 5900X specific)

**Timing breakdown**:
- Wall-clock: ~5 minutes (8k sims × 20 runs × 1.5s)
- Campaign: ~32 minutes (560 trials × 3.5s avg)
- Analysis: ~3 minutes

---

## Output Structure

All modes produce identical output structure:

```
profiling_suite_TIMESTAMP/
├── suite.log                                 # Full execution log
├── campaign/
│   ├── campaign_summary.json                 # Aggregate results
│   ├── results.csv                           # Tabular data for analysis
│   └── trial_NNN/
│       ├── cpp_profiling.json                # 295 C++ metrics + counters
│       ├── cpp_trace.json                    # Chrome timeline
│       ├── cpp_report.md                     # Human-readable report
│       ├── python_profiling.json             # Python-side metrics
│       └── result.json                       # Trial metadata
```

---

## Metrics Collected (All Modes)

### C++ Profiling (295 metrics)
- **Timing**: 8 phase timings (selection, expansion, backup, etc.)
- **Counters**: 6 operations (state_clone_count, mutex_contention, CAS, etc.)
- **Gauges**: 4 peak values (tree_node_count, memory, etc.)
- **Bottlenecks**: Automated detection with severity scores

### Python Profiling
- GIL acquisition/release times
- Inference callback overhead
- Thread coordination metrics

### Analysis
- Thread scaling efficiency
- Batch size optimization
- Statistical variance (with repetitions)
- Bottleneck prioritization
- Target comparison (vs 8,000 sims/sec)

---

## Custom Configurations

Override any parameter:

```bash
# Custom simulation counts
./scripts/run_profiling_suite.sh --full \
  --simulations "1000,2000,4000" \
  --threads "2,4,8" \
  --batch-sizes "64"

# Direct usage (bypass suite wrapper)
python scripts/profiling_campaign.py \
  --simulations 8000 \
  --threads 1,4,8,12 \
  --batch-sizes 32,64,128 \
  --repetitions 5 \
  --output my_results/
```

---

## Interpreting Results

### Quick Mode Results
```
🏆 Best: 1,850 sims/sec @ 200 sims, 4 threads, batch 64
🎯 Target: 8,000 sims/sec → 23% achieved
🔴 Bottleneck: state_clone_total (86% of time)
```
**Action**: Use --full for detailed optimization

### Full Mode Results
```
🏆 Best: 1,980 sims/sec @ 8k sims, 8 threads, batch 64
📊 Thread scaling: 1.8x speedup (1→8 threads) [ideal: 8x]
📦 Batch sweet spot: 64 (1,950 sims/sec vs 1,850 @ 32)
📈 Consistency: CV=4.2% (good, <5%)
🔴 Primary bottleneck: state_cloning (83.2 ± 2.1% of time)
```
**Action**: Implement state pooling (Priority #1)

### Production Mode Results
```
🏆 Best: 2,015 ± 48 sims/sec @ 8k sims, 10 threads, batch 64
📊 Thread scaling: Plateaus at 8-10 threads (CCD contention)
📦 Batch optimization: 64-128 both optimal (1,980-2,000 sims/sec)
📈 High confidence: CV=2.4% across 5 runs
🔴 Bottlenecks (ranked by impact):
   1. state_cloning: 81.3% ± 1.8% → Expected 4x speedup with pooling
   2. thread_idle: 12.1% ± 0.9% → CCD affinity tuning needed
   3. mutex_contention: 3.2% ± 0.4% → Fine-grained locking
```
**Action**: Validated optimization roadmap ready

---

## Technical Details

### Why these simulation counts?

| Sims | Runtime @ 2k/sec | Purpose |
|------|------------------|---------|
| 100-200 | 0.05-0.1s | Quick smoke test |
| 2,000 | 1.0s | Minimum for stable measurements |
| 4,000 | 2.0s | Mid-range baseline |
| 8,000 | 4.0s | Target throughput test |
| 16,000 | 8.0s | Beyond-target stress test |

### Why these thread counts?

- **1**: Baseline (no parallelism)
- **2**: Minimum parallelism
- **4**: Typical laptop/server
- **6**: Ryzen 5900X single CCD
- **8**: Common server config
- **10**: Near dual-CCD boundary
- **12**: Full Ryzen 5900X (2× CCD)

### Why these repetitions?

| Reps | CV | Confidence | Use Case |
|------|-----|------------|----------|
| 1 | N/A | None | Exploration only |
| 3 | ~5% | Medium | Optimization work |
| 5 | ~2-3% | High | Production validation |

---

## Performance Expectations

### Quick Mode
- 4 trials × 0.1s = 0.4s campaign
- Total: ~1 minute

### Full Mode
- 135 trials × ~6s avg = 810s = 13.5 min campaign
- Wall-clock: 40 runs × 15s = 600s = 10 min
- Analysis: ~1 min
- **Total: ~25 minutes** (conservative estimate)

### Production Mode
- 560 trials × ~3.5s avg = 1,960s = 32 min campaign
- Wall-clock: 160 runs × 2s = 320s = 5 min
- Analysis: ~3 min
- **Total: ~40 minutes**

---

## Next Steps

1. **Run quick mode** to verify setup:
   ```bash
   ./scripts/run_profiling_suite.sh --quick
   ```

2. **Run full mode** for optimization work:
   ```bash
   ./scripts/run_profiling_suite.sh --full
   ```

3. **Analyze results**:
   ```bash
   cat profiling_suite_*/campaign/campaign_summary.json | jq
   ```

4. **View Chrome timeline**:
   - Open `chrome://tracing` in Chrome
   - Load `profiling_suite_*/campaign/trial_001/cpp_trace.json`

5. **Implement fixes** based on bottleneck priorities

6. **Validate with production mode** before release:
   ```bash
   ./scripts/run_profiling_suite.sh --production
   ```

---

## Troubleshooting

### Profiling runs too fast (< 5 seconds)
- Check you're using --full or --production, not --quick
- Verify repetitions are being used (look for "rep X/Y")
- Check trial count in output header

### No metrics collected
- Verify C++ extensions were rebuilt with `PROFILE_LEVEL_VALUE=3`
- Check `cpp_profiling.json` has non-empty `counters` section
- Run `pip install -e . --force-reinstall --no-deps`

### High variance (CV > 10%)
- System under load (close other programs)
- Thermal throttling (check CPU temps)
- Increase repetitions (--production uses 5)
- Run longer simulations (8k+ more stable)

---

## FAQ

**Q: Why is --full the default?**
A: It provides the best balance of depth vs time for serious optimization work.

**Q: When should I use --quick?**
A: Only for smoke tests. It's too brief for meaningful bottleneck analysis.

**Q: How often should I run --production?**
A: Before major releases, for regression detection, or when publishing benchmarks.

**Q: Can I run just one component?**
A: Yes! Run `python scripts/profiling_campaign.py` or `python scripts/wall_clock_validation.py` directly.

**Q: How do I compare two profiling runs?**
A: Use `scripts/analyze_profiling_results.py` with multiple campaign summaries.

---

**Status**: ✅ All modes implemented and validated (2025-10-15)
