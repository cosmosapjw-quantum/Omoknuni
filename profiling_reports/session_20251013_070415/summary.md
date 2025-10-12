# MCTS Profiling Report - 20251013_070415

## Executive Summary

### Key Findings

**Critical Bottlenecks:**
- 🔴 **Python**: GIL contention (100.0% impact)

### Performance Metrics

| Metric | Value |
|--------|-------|
| C++ Throughput | 0.0 sims/sec |
| Best Thread Count | 1 threads |
| Peak Throughput | 1148.3 sims/sec |

### Recommendations

1. Reduce GIL contention by moving more operations to C++
2. Reduce thread count - severe contention detected