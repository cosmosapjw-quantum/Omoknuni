# GIL Optimization Quick Reference

**One-page cheat sheet for Python/C++ performance optimization**

---

## 🔥 Top 10 Techniques (Ordered by Impact)

| # | Technique | When to Use | Speedup | Complexity | Status |
|---|-----------|-------------|---------|------------|--------|
| 1 | **Full C++ Loops** | Hot loops (>1k calls/sec) | 7-10× | High | ✅ Done |
| 2 | **Coarse GIL Release** | Batch operations | 1.5-3× | Low | ✅ Done |
| 3 | **Batch Operations** | Neural network inference | 32-64× | Medium | ✅ Done |
| 4 | **Thread-Local Storage** | Allocations (>1M/sec) | 2-5× | Medium | ✅ Done |
| 5 | **OpenMP Parallel** | Independent loop iterations | 6-10× | Low | ✅ Done |
| 6 | **Condition Variables** | Producer-consumer queues | 1.3-1.5× | Medium | ✅ Done |
| 7 | **Zero-Copy DLPack** | Large tensor transfers | 2-10× | High | ✅ Done |
| 8 | **NumPy Vectorization** | Array operations in Python | 10-100× | Low | ✅ Partial |
| 9 | **Persistent Workers** | Long-running servers | 1.2-1.5× | Medium | ✅ Done |
| 10 | **Multiprocessing** | Self-play (independent) | 8-12× | Low | ❌ TODO |

---

## 🚀 Quick Wins (Copy-Paste Ready)

### 1. Release GIL for C++ Function
```cpp
PYBIND11_MODULE(mcts_py, m) {
    m.def("expensive_function", &expensive_function,
          py::call_guard<py::gil_scoped_release>());  // Add this line
}
```

### 2. Release GIL in Function Body
```cpp
void process_batch(const std::vector<State*>& states) {
    // Release GIL for entire batch
    py::gil_scoped_release release;

    for (State* state : states) {
        do_work(state);  // All work happens without GIL
    }
    // GIL auto-reacquired here
}
```

### 3. OpenMP Parallel Loop
```cpp
#pragma omp parallel for schedule(static) if(batch_size > 8)
for (int i = 0; i < batch_size; ++i) {
    process(items[i]);  // Parallel execution
}
```

### 4. Condition Variable Instead of Polling
```cpp
// ❌ BAD: Polling
while (running_) {
    if (has_work()) { do_work(); }
    else { sleep(10us); }  // Wastes 67% CPU
}

// ✅ GOOD: Condition variable
std::unique_lock<std::mutex> lock(mutex_);
cv_.wait(lock, [this] { return has_work() || !running_; });
if (has_work()) { do_work(); }
```

### 5. NumPy Vectorization
```python
# ❌ BAD: Python loop
for i in range(len(policy)):
    if not legal_mask[i]:
        policy[i] = 0.0
return policy / policy.sum()

# ✅ GOOD: Vectorized
masked = policy * legal_mask  # GIL released
return masked / masked.sum()  # GIL released
```

---

## 🐛 Common Pitfalls

### ❌ Pitfall 1: Fine-Grained GIL Release
```cpp
// BAD: 10µs GIL + 1µs work = 90% overhead
for (int i = 0; i < 10000; ++i) {
    py::gil_scoped_release r;
    quick_op(i);
}

// GOOD: 10µs GIL + 10ms work = 0.1% overhead
{
    py::gil_scoped_release r;
    for (int i = 0; i < 10000; ++i) {
        quick_op(i);
    }
}
```

### ❌ Pitfall 2: Python Objects Without GIL
```cpp
// CRASH: Accessing Python list without GIL
void process(py::list states) {
    py::gil_scoped_release r;
    for (auto s : states) {  // CRASH!
        do_work(s);
    }
}

// FIX: Convert to C++ first
void process(py::list states) {
    std::vector<State*> cpp_states;
    for (auto s : states) {
        cpp_states.push_back(s.cast<State*>());
    }

    py::gil_scoped_release r;
    for (State* s : cpp_states) {
        do_work(s);  // Safe
    }
}
```

### ❌ Pitfall 3: NumPy Object Arrays
```python
# NO GIL RELEASE: dtype=object
policies = np.array([p1, p2, p3])  # dtype=object
result = policies.sum()  # Slow

# GIL RELEASED: dtype=float32
policies = np.array([p1, p2, p3], dtype=np.float32)
result = policies.sum()  # Fast
```

---

## 🔍 Debugging Commands

### Python GIL Contention
```bash
# Measure GIL contention (target: <20%)
pip install gil_load
python -m gil_load script.py

# Visual profiling (GIL-aware)
pip install py-spy
py-spy record -o profile.svg -- python script.py
```

### C++ Mutex Contention
```bash
# Record mutex contention
perf record -e 'sched:sched_switch' -a -g -- python script.py
perf report --stdio | grep mutex

# Cache line bouncing
perf c2c record -- python script.py
perf c2c report --stdio
```

### GPU Profiling
```bash
# GPU utilization (target: 80-92%)
nvidia-smi dmon -s u

# Timeline profiling
nsys profile --trace=cuda,nvtx -o profile.qdrep python script.py
nsys-ui profile.qdrep
```

---

## 📊 Performance Checklist

Before claiming "GIL is the bottleneck":

- [ ] GIL actually released in hot loops (`py::gil_scoped_release`)
- [ ] No Python objects in GIL-free code
- [ ] NumPy arrays are numeric dtypes (not `object`)
- [ ] Batch sizes optimal (32-64 for this GPU)
- [ ] Thread count optimal (4-8 for 12-core CPU)
- [ ] GPU utilization high (80-92%)
- [ ] No hidden tensor copies (verify with `data_ptr()`)
- [ ] Persistent workers (not thread pool per-search)
- [ ] Profiled with `gil_load` or `py-spy` (<20% GIL contention)

---

## 🎯 Current Status (This Codebase)

### ✅ Implemented (8/10)
1. Full C++ loops ✅
2. Coarse GIL release ✅
3. Batch operations ✅
4. Thread-local storage ✅
5. OpenMP parallel ✅
6. Condition variables ✅
7. Zero-copy DLPack ✅
8. Persistent workers ✅

### ❌ TODO (2/10)
9. NumPy vectorization (partial, limited applicability)
10. Multiprocessing for self-play (not implemented)

### 🔴 Critical Issues
- **Single-thread**: 1,364 sims/sec (target: 5,000-8,000)
- **2-thread efficiency**: 45% (target: 90%)
- **4-thread efficiency**: 41% (target: 80%)
- **GPU utilization**: 56% (target: 80-92%)

**Root Cause:** Thread contention (likely C++ mutex, not GIL)

**Next Steps:**
1. Profile with `perf` to identify mutex contention
2. Fix single-thread performance
3. Fix 2-thread efficiency
4. Scale to 4-8 threads

---

## 📚 Further Reading

- **Full Guide**: `/home/cosmosapjw/omoknuni/docs/GIL_OPTIMIZATION_GUIDE.md` (10 techniques, code examples)
- **Research Summary**: `/home/cosmosapjw/omoknuni/docs/GIL_RESEARCH_SUMMARY.md` (key findings)
- **pybind11 Docs**: https://pybind11.readthedocs.io/en/stable/advanced/misc.html
- **NumPy Thread Safety**: https://numpy.org/doc/stable/reference/thread_safety.html
- **PyTorch Internals**: https://blog.ezyang.com/2019/05/pytorch-internals/

---

**Last Updated**: 2025-10-13
**Next Review**: After thread contention fix
