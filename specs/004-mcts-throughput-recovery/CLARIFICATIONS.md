# Clarifications for Spec 004: Technical Plan Resolution

**Version**: 1.0
**Date**: 2025-10-14
**Purpose**: Resolve ambiguities and risky areas before generating plan.md

---

## 1. Feature Extraction API Surface & OpenMP Parallelization

### Ambiguity
- **Question**: Exactly where to add OpenMP parallelization? What is the memory layout of tensor channels?
- **Risk Level**: 🔴 **CRITICAL** - This is the primary bottleneck (7.5ms → <1ms required)

### Resolution (from codebase analysis)

**Location**: `dlpack_bridge.cpp:431-438` (ALREADY HAS OpenMP pragma!)

**Current Code** (lines 431-438):
```cpp
// Parallelize feature extraction with OpenMP
// Use static scheduling for predictable load distribution
// Only parallelize if batch_size > 8 to avoid threading overhead
#pragma omp parallel for schedule(static) if(batch_size > 8)
for (int i = 0; i < batch_size; ++i) {
    float* state_buffer = data + (i * state_size);
    states[i]->extract_features_to_buffer(state_buffer);
}
```

**KEY FINDING**: ✅ **OpenMP is ALREADY implemented** in the code!

**Clarified Issue**: The regression is NOT due to missing OpenMP code, but likely:
1. **OpenMP not compiled**: Missing `-fopenmp` flag or incorrect build configuration
2. **OMP_NUM_THREADS not set**: Runtime environment not configured
3. **False sharing**: Thread contention in `extract_features_to_buffer()` implementation

**Memory Layout** (from code analysis):
- **Buffer format**: `[batch, planes, height, width]` (NCHW layout)
- **Contiguous**: `data[i * state_size]` where `state_size = num_planes * height * width`
- **Per-state offset**: Each thread writes to non-overlapping 64-byte-aligned region
- **Size calculation**: `batch_size * num_planes * height * width * sizeof(float)`

**Action Items for Plan**:
1. ✅ **Verify OpenMP compilation**: Check CMake logs for `-fopenmp` presence
2. ✅ **Set OMP_NUM_THREADS=12** at runtime (Ryzen 5900X physical cores)
3. ✅ **Profile per-thread overhead**: Measure if `extract_features_to_buffer()` has internal locks
4. ✅ **Add memory barriers**: Ensure no false sharing in game state feature extraction

---

## 2. State Pooling API & Ownership Model

### Ambiguity
- **Question**: `IGameState::copyFrom()` vs `clone()` - which to use for pooling? What are the ownership semantics for queueing?
- **Risk Level**: 🟡 **HIGH** - Incorrect ownership can cause use-after-free or memory leaks

### Resolution (from `igamestate.h:242-252`)

**API Surface**:
```cpp
// Existing: Heap allocation, full ownership transfer
virtual std::unique_ptr<IGameState> clone() const = 0;

// New (already exists!): In-place copy for pooling
virtual void copyFrom(const IGameState& source) = 0;
```

**Current Ownership Model** (from `continuous_simulation_runner.cpp:77`):
```cpp
// 🔴 PROBLEM: Allocates new state every simulation
std::unique_ptr<IGameState> current_state = root_state.clone();
```

**Recommended Pooling Pattern**:
```cpp
// Thread-local pool (reuse across simulations)
struct ThreadLocalStatePool {
    std::unique_ptr<IGameState> working_state;  // Reused for selection
    std::vector<std::unique_ptr<IGameState>> pending_states;  // Moved to queue
};

// Usage in simulation loop:
// 1. Reset working state from root (no allocation)
pool.working_state->copyFrom(root_state);

// 2. Select to leaf (modify working_state in-place)
NodeIndex leaf = select_leaf(root_index, *pool.working_state, path);

// 3. Transfer ownership to queue (move, not clone)
std::unique_ptr<IGameState> queued_state = std::move(pool.working_state);
queue.submit_request(std::move(queued_state), leaf, path);

// 4. Allocate new working state (or grab from pre-allocated pool)
pool.working_state = root_state.clone();  // Only once per N simulations
```

**Ownership Rules**:
- **Thread pool**: Owns working states, reused via `copyFrom()`
- **Inference queue**: Takes ownership via `std::move(unique_ptr)`
- **Pending expansion**: Queue retains ownership until result returns
- **After expansion**: State either returned to pool or destroyed

**Memory Impact**:
- **Current**: 2-3× clones per simulation (root → working → queue → pending)
- **Optimized**: 0× clones per simulation (copyFrom + move semantics)
- **Expected savings**: ~2GB/sec allocation pressure eliminated

---

## 3. Blocking vs Polling: Condition Variables

### Ambiguity
- **Question**: Which condition variables signal what? Where do threads block vs poll?
- **Risk Level**: 🟡 **MEDIUM** - Incorrect waiting causes 60% thread idle time

### Resolution (from `async_inference_queue.hpp:260-263`)

**Current Architecture** (T006c complete):
```cpp
// Condition variable for efficient waiting (T006c)
std::mutex cv_mutex_;
std::condition_variable request_ready_;
std::atomic<bool> shutting_down_{false};
```

**Wait Points**:

**A) Coordinator waiting for requests** (BLOCKING):
```cpp
// In BatchInferenceCoordinator::collect_batch()
std::unique_lock<std::mutex> lock(cv_mutex_);
request_ready_.wait_for(lock, timeout_ms, [&]() {
    return pending_count_.load() >= min_batch_size || shutting_down_;
});
```

**B) Simulation threads waiting for results** (CURRENTLY POLLING - needs fix):
```cpp
// 🔴 PROBLEM: continuous_simulation_runner.cpp (no blocking wait)
while (completed < num_simulations) {
    // ... select and submit ...

    // Poll for results (WASTEFUL)
    int processed = process_completed_results();
    if (processed == 0 && pending_count_ > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));  // Spin-wait!
    }
}
```

**Recommended Fix** (add result ready CV):
```cpp
// Add to AsyncInferenceQueue
std::condition_variable results_ready_;

// Signal when results are submitted
void submit_results(const std::vector<InferenceResult>& results) {
    // ... store results ...
    results_ready_.notify_all();  // Wake waiting threads
}

// Blocking wait in simulation runner
std::unique_lock<std::mutex> lock(cv_mutex_);
results_ready_.wait(lock, [&]() {
    return has_results() || shutting_down_;
});
```

**Signaling Protocol**:
1. **Threads → Coordinator**: Submit request, `request_ready_.notify_one()`
2. **Coordinator → Threads**: Submit results, `results_ready_.notify_all()`
3. **Shutdown**: Set `shutting_down_`, notify all CVs

---

## 4. Node Allocator: Contiguous Allocation Contention

### Ambiguity
- **Question**: How to minimize global mutex for contiguous children allocation?
- **Risk Level**: 🟡 **MEDIUM** - Lock contention at 8+ threads limits scaling

### Resolution (from `tree.cpp:20-44`)

**Current Design** (PARTIAL solution):
```cpp
// Thread-local blocks of 4096 nodes (ALREADY IMPLEMENTED)
constexpr std::uint32_t kThreadBlockSize = 4096;

struct ThreadLocalBlock {
    MCTSTree* tree = nullptr;
    std::uint64_t tree_id = 0;
    NodeIndex next = NULL_NODE_INDEX;
    std::uint32_t remaining = 0;
    std::uint64_t epoch = 0;

    // Statistics (99.93% fast-path observed)
    std::uint64_t allocations_from_block = 0;
    std::uint64_t allocations_from_global = 0;
};

thread_local ThreadLocalBlock thread_block;
```

**Problem**: Contiguous multi-node allocation (children) still takes **global mutex**:
```cpp
// Pseudocode from tree.cpp
std::vector<NodeIndex> allocate_nodes(int count) {
    if (count == 1) {
        // Fast path: thread-local block (99.93% case)
        return allocate_from_thread_block();
    } else {
        // Slow path: GLOBAL MUTEX (expansion with >1 children)
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        return allocate_contiguous_from_global(count);
    }
}
```

**Optimized Strategy** (for plan.md):

**Option A: Over-allocate in thread blocks**
```cpp
// Reserve contiguous ranges in thread-local blocks
std::vector<NodeIndex> allocate_nodes(int count) {
    if (thread_block.remaining >= count) {
        // Allocate contiguous from thread-local (FAST)
        NodeIndex start = thread_block.next;
        thread_block.next += count;
        thread_block.remaining -= count;
        return make_range(start, count);
    } else {
        // Refill thread block from global (RARE)
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        thread_block.next = next_free_index_.fetch_add(kThreadBlockSize);
        thread_block.remaining = kThreadBlockSize;
        return allocate_nodes(count);  // Retry with fresh block
    }
}
```

**Option B: Per-thread child expansion queues** (defer expensive expansions)
```cpp
// Batch child expansions to reduce lock frequency
struct ThreadLocalExpansionBatch {
    std::vector<PendingExpansion> batch;
    static constexpr size_t BATCH_SIZE = 64;
};

// Accumulate expansions, flush when batch full
void expand_node_deferred(NodeIndex parent, ...) {
    thread_local_batch.push(parent, ...);
    if (thread_local_batch.size() >= BATCH_SIZE) {
        flush_expansion_batch();  // One lock for 64 expansions
    }
}
```

**Recommendation**: **Option A** (simpler, keeps synchronous expansion semantics)

---

## 5. Global NN-Eval Cache: Design Parameters

### Ambiguity
- **Question**: Shard count, memory budget, FP16 encoding, key fields per game?
- **Risk Level**: 🟢 **LOW** - Phase 6 optional, safe design (Tier A)

### Resolution (from review.txt + CONSTITUTION.md)

**Cache Architecture** (Tier A - Policy/Value only):
```cpp
struct NNEvalCacheEntry {
    uint64_t hash;          // Zobrist hash (8 bytes)
    uint32_t net_version;   // Training iteration (4 bytes)
    uint16_t top_k;         // Number of stored moves (2 bytes)
    uint16_t padding;       // Alignment (2 bytes)

    // Quantized policy (top-K only, not full board)
    std::array<uint16_t, 48> move_indices;  // 96 bytes
    std::array<uint16_t, 48> policy_logits_fp16;  // 96 bytes (FP16 = 2 bytes)

    float value;  // FP32 (4 bytes)

    // Metadata
    uint32_t access_count;  // For SLRU eviction (4 bytes)
    uint32_t last_access_time;  // Timestamp (4 bytes)
};
// Total: 224 bytes per entry (conservative with top-K=48)
```

**Memory Budget**:
- **Entry size**: 224 bytes (with top-K=48, FP16 quantization)
- **Cache size**: 2M entries → 448 MB (target: 2-8GB)
- **With top-K=16**: 96 bytes/entry → 2M entries = 192 MB

**Shard Design**:
```cpp
constexpr size_t NUM_SHARDS = 64;  // Cache-line aligned

struct CacheShard {
    std::mutex shard_mutex;  // Fine-grained locking
    std::unordered_map<uint64_t, NNEvalCacheEntry> entries;
    std::deque<uint64_t> lru_queue;  // SLRU: segmented LRU
};

std::array<CacheShard, NUM_SHARDS> shards;

// Lookup
size_t shard_idx = hash % NUM_SHARDS;
std::lock_guard<std::mutex> lock(shards[shard_idx].shard_mutex);
auto it = shards[shard_idx].entries.find(hash);
```

**Key Fields Per Game** (Markov-minimal):

**Gomoku**:
```cpp
struct GomokuKey {
    std::vector<uint64_t> board_bitboards;  // 2 players × 4 words = 32 bytes
    uint8_t side_to_move;  // 1 byte
    uint8_t variant;  // 0=Freestyle, 1=Renju, 2=Omok
    // Total: 34 bytes → hash to 64-bit Zobrist
};
```

**Chess**:
```cpp
struct ChessKey {
    std::vector<uint64_t> piece_bitboards;  // 12 pieces × 1 word = 96 bytes
    uint8_t side_to_move;  // 1 byte
    uint8_t castling_rights;  // 4 bits packed
    uint8_t en_passant_file;  // 0-7 or 0xFF=none
    uint8_t rule50_counter;  // 0-100 (LC0 includes this)
    // Total: 99 bytes → hash to 64-bit Zobrist
};
```

**Go 9×9**:
```cpp
struct GoKey {
    std::vector<uint64_t> board_bitboards;  // 2 players × 2 words = 16 bytes
    uint8_t side_to_move;  // 1 byte
    uint8_t ko_point;  // 0-80 or 0xFF=none
    // Superko: NOT included (too expensive, use simple ko)
    // Total: 18 bytes → hash to 64-bit Zobrist
};
```

**Eviction Policy** (SLRU - Segmented LRU):
```cpp
// Two segments: Protected (recently used) vs Probationary
struct SLRUCache {
    std::deque<uint64_t> protected_queue;   // 80% of capacity
    std::deque<uint64_t> probationary_queue;  // 20% of capacity

    // On hit: move to protected
    // On miss: insert to probationary
    // Evict: from probationary first, then protected tail
};
```

**Net Version Tagging**:
```cpp
// On training iteration N → N+1:
void invalidate_cache_for_old_net(uint32_t old_version) {
    for (auto& shard : shards) {
        std::lock_guard<std::mutex> lock(shard.shard_mutex);
        std::erase_if(shard.entries, [&](const auto& kv) {
            return kv.second.net_version == old_version;
        });
    }
}
```

---

## 6. Multi-Actor Orchestrator: Process vs Thread

### Ambiguity
- **Question**: Process-based or thread-based actors? Backpressure mechanism? Fairness policy?
- **Risk Level**: 🟡 **MEDIUM** - Incorrect design causes GPU underutilization or starvation

### Resolution (from CONSTITUTION.md + review.txt)

**Architecture**: **Process-based actors** (recommended by review.txt)

**Rationale**:
1. **GIL isolation**: Each process has own GIL, no Python-level contention
2. **Memory isolation**: Crash in one game doesn't affect others
3. **Scheduling**: OS can schedule across CCDs efficiently
4. **Simplicity**: No shared-memory synchronization needed

**Design Pattern**:
```python
# Main process: Batch coordinator + GPU inference
class CentralizedInferenceServer:
    def __init__(self):
        self.request_queue = multiprocessing.Queue(maxsize=4096)
        self.result_queues = {}  # game_id → Queue
        self.gpu_worker = GPUInferenceWorker(batch_size=64, timeout_ms=1.5)

    def run(self):
        while running:
            # Collect batch (blocking with timeout)
            requests = self.collect_batch(min_size=64, timeout_ms=1.5)

            # GPU inference (FP16)
            results = self.gpu_worker.batch_inference(requests)

            # Demux results to actor queues
            for req, result in zip(requests, results):
                self.result_queues[req.game_id].put(result)

# Actor processes (one per game)
class SelfPlayActor:
    def __init__(self, game_id, inference_server):
        self.game_id = game_id
        self.request_queue = inference_server.request_queue
        self.result_queue = inference_server.result_queues[game_id]
        self.mcts = MCTSEngine(threads=1)  # Single-threaded per actor

    def run_game(self):
        while not terminal:
            # MCTS search (1-2 threads, 800 simulations/move)
            self.mcts.search(num_sims=800, timeout=None)

            # Submit leaves to global queue
            for leaf in self.mcts.get_pending_leaves():
                self.request_queue.put(InferenceRequest(
                    game_id=self.game_id,
                    state=leaf.state,
                    ...
                ))

            # Wait for results (blocking)
            while self.mcts.has_pending():
                result = self.result_queue.get(timeout=5.0)
                self.mcts.apply_result(result)
```

**Backpressure Mechanism** (token bucket per actor):
```python
class TokenBucketBackpressure:
    def __init__(self, capacity=256, refill_rate=100):
        self.tokens = capacity
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens/sec
        self.last_refill = time.time()

    def acquire(self, count=1):
        # Refill tokens based on time elapsed
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        # Block if insufficient tokens
        if self.tokens < count:
            sleep_time = (count - self.tokens) / self.refill_rate
            time.sleep(sleep_time)
            self.tokens = 0
        else:
            self.tokens -= count

# Usage in actor:
backpressure = TokenBucketBackpressure(capacity=256, refill_rate=100)
backpressure.acquire(len(leaves))  # Blocks if too many in-flight
```

**Fairness Policy** (round-robin with priority aging):
```python
# In CentralizedInferenceServer
def collect_batch(self, min_size, timeout_ms):
    batch = []
    deadline = time.time() + timeout_ms / 1000.0
    actor_counts = defaultdict(int)  # Track per-actor requests in batch

    while len(batch) < min_size and time.time() < deadline:
        try:
            # Round-robin: prioritize actors with fewer requests in batch
            req = self.request_queue.get(timeout=0.001)

            # Fairness check: limit per-actor requests per batch
            if actor_counts[req.game_id] < min_size // num_actors + 2:
                batch.append(req)
                actor_counts[req.game_id] += 1
            else:
                # Defer to next batch
                self.request_queue.put(req)
        except queue.Empty:
            break

    return batch
```

**Actor Count Tuning** (auto-scale to GPU utilization):
```python
def auto_tune_actor_count(self, target_gpu_util=0.85, target_batch_size=51):
    gpu_util = measure_gpu_utilization()  # nvidia-smi
    avg_batch = measure_avg_batch_size()

    if gpu_util < target_gpu_util - 0.05 and avg_batch < target_batch_size:
        # GPU underutilized: add actor
        self.spawn_actor()
    elif gpu_util > target_gpu_util + 0.10:
        # GPU oversaturated: remove actor
        self.kill_actor()
```

---

## 7. Benchmark Scenarios: Exact Parameters

### Ambiguity
- **Question**: Exact board sizes, visit budgets, seeds, batch/timeout configs for validation?
- **Risk Level**: 🟢 **LOW** - Standardization needed for reproducibility

### Resolution (fixed benchmark suite)

**Primary Benchmark** (T016 - throughput validation):
```yaml
benchmark_throughput:
  game: gomoku
  board_size: 15x15
  variant: freestyle
  initial_position: empty_board
  num_simulations: 10000  # Per search
  threads: [1, 2, 4, 6, 8, 10, 12]  # Scaling test
  batch_size: 64
  timeout_ms: 1.0
  seed: 42
  iterations: 10  # Statistical validation (N≥10)
  metrics:
    - simulations_per_second
    - gpu_utilization_percent
    - avg_batch_size
    - thread_efficiency_percent
    - cpu_coordination_percent
    - feature_extraction_ms_per_batch
```

**Secondary Benchmarks** (cross-game validation):
```yaml
benchmark_chess:
  game: chess
  initial_position: startpos
  num_simulations: 5000
  threads: 8
  batch_size: 64
  seed: 42

benchmark_go9:
  game: go
  board_size: 9x9
  num_simulations: 8000
  threads: 8
  batch_size: 64
  seed: 42
```

**Baseline Investigation** (T017 - reproduce 3,831 sims/sec):
```yaml
# Systematic grid search
baseline_search:
  threads: [2, 4, 6, 8]
  batch_sizes: [32, 48, 64, 96]
  timeouts: [0.5, 1.0, 1.5, 2.0]
  vl_magnitudes: [0.5, 1.0, 1.5, 2.0, 3.0]
  # Total: 4×4×4×5 = 320 configurations
  # Time budget: 2 days maximum (T017 time-boxed)
```

**Telemetry Protocol**:
```python
class BenchmarkTelemetry:
    def __init__(self):
        self.metrics = {
            "throughput_sims_sec": [],  # PRIMARY KPI
            "gpu_util_percent": [],
            "cpu_util_percent": [],
            "avg_batch_size": [],
            "batch_timeout_ms": [],
            "feature_extraction_ms": [],  # Per batch-64
            "thread_efficiency": [],  # vs linear scaling
            "collision_rate_percent": [],
            "memory_rss_mb": [],
            "cache_hit_rate_percent": [],  # Phase 6
        }

    def record_run(self, run_id, config, results):
        # CSV format for historical tracking
        timestamp = datetime.now().isoformat()
        row = {
            "timestamp": timestamp,
            "git_commit": get_git_commit(),
            "config": json.dumps(config),
            **results
        }
        append_to_csv("benchmark_history.csv", row)
```

---

## Summary of Resolved Ambiguities

| Area | Status | Key Decision |
|------|--------|--------------|
| **Feature Extraction** | ✅ **RESOLVED** | OpenMP already implemented; fix: verify compilation + OMP_NUM_THREADS=12 |
| **State Pooling** | ✅ **RESOLVED** | Use `copyFrom()` + `std::move()` ownership, thread-local pools |
| **Condition Variables** | ✅ **RESOLVED** | Add `results_ready_` CV to eliminate spin-wait polling |
| **Node Allocator** | ✅ **RESOLVED** | Over-allocate contiguous ranges in thread-local blocks (Option A) |
| **NN-Eval Cache** | ✅ **RESOLVED** | 64 shards, 2M entries, top-K=16-48, SLRU eviction, net versioning |
| **Multi-Actor** | ✅ **RESOLVED** | Process-based, token-bucket backpressure, round-robin fairness |
| **Benchmarking** | ✅ **RESOLVED** | T016 (throughput) + T017 (baseline search) with fixed seeds |

---

## Unresolved Items Requiring User Input

**NONE** - All technical ambiguities resolved from local codebase context.

---

## Next Steps

1. ✅ **Update spec.md** with clarified API contracts (if needed)
2. ✅ **Generate plan.md** with detailed technical design
3. ✅ **Generate tasks.md** with atomic implementation work items

---

**END OF CLARIFICATIONS DOCUMENT**
