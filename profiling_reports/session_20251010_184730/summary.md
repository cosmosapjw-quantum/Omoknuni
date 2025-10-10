# MCTS Profiling Report - 20251010_184730

## Executive Summary

### Key Findings

**Critical Bottlenecks:**
- 🔴 **Python**: GIL contention (100.0% impact)

### Performance Metrics

| Metric | Value |
|--------|-------|
| C++ Throughput | 0.0 sims/sec |
| Best Thread Count | 1 threads |
| Peak Throughput | 897.1 sims/sec |

### Recommendations

1. Reduce GIL contention by moving more operations to C++
2. Reduce thread count - severe contention detected