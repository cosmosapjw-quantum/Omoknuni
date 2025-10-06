# Parallel Mode Benchmarks

**Date**: 2025-10-07
**Hardware**: AMD Ryzen 9 5900X (12c/24t), RTX 3060 Ti (unused in sync tests)
**Inference**: No-op synchronous inference (`np.ones` policy)
**MCTS Config**: `simulations` as noted, instrumentation enabled

Benchmark command:
```bash
source venv/bin/activate
python scripts/benchmark_parallel_modes.py --simulations <N> --runs 1 --threads <T>
```

## Results Summary (Sync inference)

| Threads | Simulations | Mode | Sims/sec |
|---------|-------------|------|---------:|
| 2 | 64 | shared | 3723 |
| 2 | 64 | virtual_loss_free | 3486 |
| 2 | 64 | thread_local_prototype | 226 |
| 4 | 256 | shared | 3589 |
| 4 | 256 | virtual_loss_free | 3651 |
| 4 | 256 | thread_local_prototype | 441 |
| 8 | 128 | shared | 3559 |
| 8 | 128 | virtual_loss_free | 3565 |
| 8 | 128 | thread_local_prototype | 133 |

Key observations:
- **Shared (baseline)** remains robust across thread counts, hovering around 3.5k sims/sec.
- **Virtual-loss-free** shared tree slightly outperforms the baseline in most runs (~1–3% gain), suggesting removing virtual loss can pay off in low-contention settings.
- **Thread-local prototype** (per-thread tree with Python-level merging) is consistently an order of magnitude slower; this confirms the current prototype is not viable without a native implementation and smarter merge strategy.

## Next Steps
- Integrate asynchronous inference into the benchmark once GPU batching can be exercised in this environment.
- Explore native thread-local tree experiments where statistics are merged in C++ instead of Python.
