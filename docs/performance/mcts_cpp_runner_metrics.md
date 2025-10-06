# MCTS C++ Runner Instrumentation Metrics

**Last Updated**: 2025-10-07

This guide describes the instrumentation framework introduced in Spec 004 to
track performance hotspots across the C++ simulation runner and associated
subsystems.

## Overview

Instrumentation is disabled by default and can be toggled at runtime via the
Python bindings. When enabled, scoped timers and counters measure the following
operations:

| Metric Key | Description |
| --- | --- |
| `tree_clear` | Time spent resetting the shared tree between searches |
| `tree_allocate_node` | Single-node allocations from the tree arena |
| `tree_allocate_nodes` | Bulk node allocations |
| `selection` | Time spent in PUCT selection (`SimulationRunner::select_leaf`) |
| `expansion` | Time spent expanding nodes and running inference callbacks |
| `backup` | Time spent propagating values along the path |
| `virtual_loss_apply` | Count of virtual loss applications |
| `virtual_loss_remove` | Count of virtual loss removals |
| `queue_submit` | Async queue submission latency |
| `queue_collect` | Batch collection latency (producer side) |
| `queue_process_results` | Time spent flushing completed inference results |
| `queue_try_get_result` | Per-result retrieval cost |

Each metric records the number of calls and the cumulative elapsed nanoseconds.
Average cost per call can be derived by dividing `total_ns` by `calls`.

## Enabling Instrumentation

```python
from src.core.mcts import AlphaZeroMCTS
import alphazero_py

engine = AlphaZeroMCTS(
    inference_fn=my_inference,
    num_threads=8,
    enable_instrumentation=True,
)

engine.search(alphazero_py.GomokuState(board_size=15), simulations=128)
stats = engine.get_statistics()
print(stats["instrumentation"])
```

Instrumentation can also be toggled after construction:

```python
engine.set_instrumentation_enabled(True)
engine.reset_instrumentation_metrics()
```

## Resetting Counters

Use `reset_instrumentation_metrics()` to clear all counters and timers before a
new measurement run. Counters are also reset automatically when instrumentation
is enabled through the constructor.

## Accessing Raw Metrics from C++

The pybind interface exposes utility functions:

```python
import mcts_py

mcts_py.set_instrumentation_enabled(True)
# ... run workloads ...
snapshot = mcts_py.get_instrumentation_snapshot()
mcts_py.reset_instrumentation_metrics()
```

`get_instrumentation_snapshot()` returns a dictionary keyed by metric name with
fields:

```python
{
    "selection": {
        "calls": 512,
        "total_ns": 1234567890,
        "avg_ns": 2410.13
    },
    ...
}
```

## Validation

- Unit coverage: `tests/unit/test_instrumentation_metrics.cpp`
- Runtime validation: `tests/performance/test_simulation_runner_performance.py::test_instrumentation_metrics_available`

These tests ensure metrics emit data and can be consumed from the Python API.

---

For any new performance-focused changes, enable instrumentation during local
profiling sessions and attach the resulting metrics to the corresponding spec
updates.
