# Technical Implementation Plan: MCTS Throughput Recovery

## Architecture Overview

This plan details the technical implementation of optimizations to achieve 25,000+ simulations/second while maintaining the shared tree architecture with enhanced virtual loss coordination.

## Core Architecture: Shared Tree with WU-UCT Virtual Loss

### Design Rationale

We maintain the **shared tree architecture** because:
- **Single GPU constraint**: Root parallelization requires multiple GPUs for efficiency
- **Memory efficiency**: One tree (270MB) vs 8 trees (2.16GB)
- **Information sharing**: Threads benefit from each other's explorations
- **Batch diversity**: Shared tree naturally produces diverse positions for GPU batching

### Virtual Loss Evolution: Classic → WU-UCT

#### Classic Virtual Loss (Current)
```cpp
// Distorts Q-value during selection
float q_value = (total_value - virtual_loss) / (visit_count + 1);
float exploration = c_puct * prior * sqrt(parent_visits) / (1 + visit_count);
float puct_score = q_value + exploration;
```

**Problems**:
- Virtual loss directly reduces Q-value
- Can cause value inversion (good moves look bad)
- Requires careful tuning of magnitude

#### WU-UCT Style (Proposed)
```cpp
// Only affects exploration term, preserves Q-value accuracy
float q_value = total_value / visit_count;  // Pure Q, no VL
float in_flight = virtual_loss_manager.get_in_flight_count(node);
float exploration = c_puct * prior * sqrt(parent_visits) / (1 + visit_count + in_flight);
float puct_score = q_value + exploration;
```

**Benefits**:
- Q-value remains accurate for value estimates
- Virtual loss only discourages re-selection
- More robust to magnitude changes

## Implementation Components

### 1. WU-UCT Virtual Loss System

#### 1.1 Data Structure Changes

**File**: `cpp_extensions/mcts/virtual_loss.hpp`

```cpp
class WUUCTVirtualLossManager : public VirtualLossManager {
private:
    // Separate tracking of in-flight simulations
    struct alignas(64) InFlightData {
        std::atomic<uint32_t> count{0};        // Number of threads exploring
        std::atomic<uint64_t> thread_mask{0};  // Which threads (bit mask)
    };

    std::vector<InFlightData> in_flight_;  // One per node

public:
    // Called when thread selects node
    void add_in_flight(NodeIndex node, int thread_id) {
        in_flight_[node].count.fetch_add(1, std::memory_order_relaxed);

        uint64_t thread_bit = 1ULL << thread_id;
        in_flight_[node].thread_mask.fetch_or(thread_bit, std::memory_order_relaxed);
    }

    // Called after backup completes
    void remove_in_flight(NodeIndex node, int thread_id) {
        in_flight_[node].count.fetch_sub(1, std::memory_order_relaxed);

        uint64_t thread_bit = 1ULL << thread_id;
        in_flight_[node].thread_mask.fetch_and(~thread_bit, std::memory_order_relaxed);
    }

    // Used in PUCT calculation (exploration denominator only)
    uint32_t get_in_flight_count(NodeIndex node) const {
        return in_flight_[node].count.load(std::memory_order_relaxed);
    }
};
```

#### 1.2 Selection Integration

**File**: `cpp_extensions/mcts/selection.cpp`

```cpp
void compute_puct_vectorized(
    const float* visit_counts,
    const float* total_values,
    const float* prior_probs,
    const uint32_t* in_flight_counts,  // NEW: WU-UCT data
    const uint8_t* expanding_flags,    // NEW: Busy-edge mask
    float* puct_scores,
    // ... other parameters
) {
    const __m256 exploration_base = _mm256_set1_ps(c_puct * sqrt(parent_visits));

    for (uint16_t i = 0; i < num_children; i += 8) {
        // Load data (aligned)
        __m256 n = _mm256_load_ps(&visit_counts[first_child + i]);
        __m256 w = _mm256_load_ps(&total_values[first_child + i]);
        __m256 p = _mm256_load_ps(&prior_probs[first_child + i]);
        __m256 inflight = _mm256_cvtepi32_ps(
            _mm256_load_si256((__m256i*)&in_flight_counts[first_child + i])
        );

        // Check expanding flags for busy-edge masking
        __m256i flags = _mm256_load_si256((__m256i*)&expanding_flags[first_child + i]);
        __m256 is_expanding = _mm256_cmp_ps(
            _mm256_cvtepi32_ps(flags), _mm256_setzero_ps(), _CMP_NEQ
        );

        // Q-value (pure, no VL distortion)
        __m256 q = _mm256_div_ps(w, _mm256_max_ps(n, one));

        // Exploration with WU-UCT denominator
        __m256 denominator = _mm256_add_ps(
            _mm256_add_ps(one, n),
            inflight  // In-flight visits in denominator
        );
        __m256 u = _mm256_div_ps(
            _mm256_mul_ps(exploration_base, p),
            denominator
        );

        // PUCT score
        __m256 puct = _mm256_add_ps(q, u);

        // Apply busy-edge mask (set to -inf if expanding)
        puct = _mm256_blendv_ps(
            puct,
            _mm256_set1_ps(-INFINITY),
            is_expanding
        );

        _mm256_store_ps(&puct_scores[i], puct);
    }
}
```

### 2. Lock-Free Queue Implementation

#### 2.1 MPMC Ring Buffer Design

**File**: `cpp_extensions/mcts/lock_free_queue.hpp`

```cpp
template<typename T, size_t Capacity = 4096>
class MPMCRingBuffer {
private:
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");

    struct alignas(64) Slot {
        std::atomic<size_t> turn{0};
        T storage;
    };

    alignas(64) std::array<Slot, Capacity> buffer_;
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};

    static constexpr size_t MASK = Capacity - 1;

public:
    bool try_enqueue(T&& item) {
        size_t head = head_.load(std::memory_order_relaxed);

        for (;;) {
            Slot& slot = buffer_[head & MASK];
            size_t turn = slot.turn.load(std::memory_order_acquire);

            intptr_t diff = static_cast<intptr_t>(turn) - static_cast<intptr_t>(head);

            if (diff == 0) {
                // Slot is ready for writing
                if (head_.compare_exchange_weak(
                    head, head + 1,
                    std::memory_order_relaxed
                )) {
                    slot.storage = std::move(item);
                    slot.turn.store(head + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                // Queue is full
                return false;
            } else {
                // Another thread claimed this slot
                head = head_.load(std::memory_order_relaxed);
            }
        }
    }

    bool try_dequeue(T& item) {
        size_t tail = tail_.load(std::memory_order_relaxed);

        for (;;) {
            Slot& slot = buffer_[tail & MASK];
            size_t turn = slot.turn.load(std::memory_order_acquire);

            intptr_t diff = static_cast<intptr_t>(turn) - static_cast<intptr_t>(tail + 1);

            if (diff == 0) {
                // Slot has data
                if (tail_.compare_exchange_weak(
                    tail, tail + 1,
                    std::memory_order_relaxed
                )) {
                    item = std::move(slot.storage);
                    slot.turn.store(tail + Capacity + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                // Queue is empty
                return false;
            } else {
                // Another thread is dequeuing
                tail = tail_.load(std::memory_order_relaxed);
            }
        }
    }

    // Batch operations for efficiency
    size_t try_enqueue_bulk(const T* items, size_t count) {
        size_t enqueued = 0;
        for (size_t i = 0; i < count; ++i) {
            if (!try_enqueue(T(items[i]))) {
                break;
            }
            enqueued++;
        }
        return enqueued;
    }
};
```

#### 2.2 Integration with AsyncInferenceQueue

**File**: `cpp_extensions/mcts/async_inference_queue.cpp`

```cpp
class LockFreeAsyncInferenceQueue : public AsyncInferenceQueue {
private:
    MPMCRingBuffer<InferenceRequest, 4096> pending_requests_;
    MPMCRingBuffer<InferenceResult, 4096> completed_results_;

    // Wait-free statistics
    std::atomic<size_t> pending_count_{0};
    std::atomic<size_t> completed_count_{0};

public:
    uint64_t submit_request(std::unique_ptr<IGameState> state,
                           NodeIndex node_index,
                           std::vector<NodeIndex> path) override {
        InferenceRequest request;
        request.request_id = next_request_id_.fetch_add(1);
        request.state = std::move(state);
        request.node_index = node_index;
        request.path = std::move(path);

        // Lock-free enqueue
        while (!pending_requests_.try_enqueue(std::move(request))) {
            // Queue full, yield CPU
            std::this_thread::yield();
        }

        pending_count_.fetch_add(1, std::memory_order_relaxed);
        return request.request_id;
    }

    std::vector<InferenceRequest> collect_batch(size_t min_batch_size,
                                                double timeout_ms) override {
        std::vector<InferenceRequest> batch;
        batch.reserve(min_batch_size * 2);

        auto start = std::chrono::steady_clock::now();
        auto timeout = std::chrono::duration<double, std::milli>(timeout_ms);

        while (batch.size() < min_batch_size) {
            InferenceRequest request;
            if (pending_requests_.try_dequeue(request)) {
                batch.push_back(std::move(request));
                pending_count_.fetch_sub(1, std::memory_order_relaxed);
            } else {
                // Check timeout
                auto elapsed = std::chrono::steady_clock::now() - start;
                if (elapsed >= timeout) {
                    break;
                }

                // Brief pause before retry
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }

        // Opportunistically grab more if available
        while (batch.size() < min_batch_size * 2) {
            InferenceRequest request;
            if (!pending_requests_.try_dequeue(request)) {
                break;
            }
            batch.push_back(std::move(request));
            pending_count_.fetch_sub(1, std::memory_order_relaxed);
        }

        return batch;
    }
};
```

### 3. Zero-Copy Tensor Bridge

#### 3.1 DLPack Integration

**File**: `cpp_extensions/mcts/dlpack_bridge.hpp`

```cpp
class DLPackTensorBridge {
private:
    // Pre-allocated pinned memory for batching
    struct PinnedBuffer {
        void* data;
        size_t capacity;
        size_t size;

        PinnedBuffer(size_t bytes) : capacity(bytes), size(0) {
            cudaMallocHost(&data, bytes);  // Pinned memory for fast GPU transfer
        }

        ~PinnedBuffer() {
            cudaFreeHost(data);
        }
    };

    thread_local std::unique_ptr<PinnedBuffer> batch_buffer_;

public:
    py::capsule create_batch_tensor(const std::vector<const IGameState*>& states) {
        if (!batch_buffer_) {
            // Allocate pinned memory (once per thread)
            size_t buffer_size = 64 * 36 * 15 * 15 * sizeof(float);  // Max batch
            batch_buffer_ = std::make_unique<PinnedBuffer>(buffer_size);
        }

        size_t batch_size = states.size();
        size_t state_elements = 36 * 15 * 15;  // Gomoku tensor size

        // Parallel copy to pinned buffer
        float* buffer = static_cast<float*>(batch_buffer_->data);

        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < batch_size; ++i) {
            states[i]->extract_features_direct(buffer + i * state_elements);
        }

        // Create DLPack tensor (zero-copy)
        DLManagedTensor* dlpack = new DLManagedTensor;
        dlpack->dl_tensor.data = buffer;
        dlpack->dl_tensor.device = {kDLCPU, 0};
        dlpack->dl_tensor.ndim = 4;
        dlpack->dl_tensor.dtype = {kDLFloat, 32, 1};
        dlpack->dl_tensor.shape = new int64_t[4]{
            static_cast<int64_t>(batch_size), 36, 15, 15
        };
        dlpack->dl_tensor.strides = nullptr;  // C-contiguous
        dlpack->dl_tensor.byte_offset = 0;

        // Set deleter (called when Python releases)
        dlpack->manager_ctx = nullptr;
        dlpack->deleter = [](DLManagedTensor* self) {
            delete[] self->dl_tensor.shape;
            delete self;
        };

        return py::capsule(dlpack, "dltensor", [](void* ptr) {
            DLManagedTensor* dlpack = static_cast<DLManagedTensor*>(ptr);
            if (dlpack->deleter) {
                dlpack->deleter(dlpack);
            }
        });
    }
};
```

#### 3.2 Python Integration

**File**: `src/core/dlpack_inference_bridge.py`

```python
class DLPackInferenceBridge:
    """Zero-copy inference bridge using DLPack protocol."""

    def __init__(self, gpu_worker):
        self.gpu_worker = gpu_worker
        self.device = torch.device('cuda:0')

        # Pre-allocate GPU tensors for common batch sizes
        self.gpu_buffers = {
            32: torch.empty((32, 36, 15, 15), device=self.device, dtype=torch.float32),
            64: torch.empty((64, 36, 15, 15), device=self.device, dtype=torch.float32),
        }

    def batch_inference(self, dlpack_capsule):
        """
        Process batch using zero-copy DLPack transfer.

        Args:
            dlpack_capsule: DLPack tensor from C++ (CPU pinned memory)

        Returns:
            tuple: (policy_array, value_array) as numpy arrays
        """
        # Zero-copy conversion to PyTorch
        cpu_tensor = torch.from_dlpack(dlpack_capsule)
        batch_size = cpu_tensor.size(0)

        # Get pre-allocated GPU buffer or create new
        if batch_size in self.gpu_buffers:
            gpu_tensor = self.gpu_buffers[batch_size][:batch_size]
            gpu_tensor.copy_(cpu_tensor, non_blocking=True)  # Async copy
        else:
            gpu_tensor = cpu_tensor.to(self.device, non_blocking=True)

        # GPU inference
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                policy_logits, values = self.gpu_worker.model(gpu_tensor)

        # Post-process on GPU
        policies = torch.softmax(policy_logits, dim=1)
        values = torch.tanh(values).squeeze(-1)

        # Return as numpy (CPU) - single copy back
        return (
            policies.cpu().numpy(),
            values.cpu().numpy()
        )
```

### 4. Thread Optimization

#### 4.1 Thread Affinity Manager

**File**: `cpp_extensions/mcts/thread_affinity.cpp`

```cpp
class ThreadAffinityManager {
private:
    struct CPUTopology {
        std::vector<int> ccd0_cores;  // CCD 0 physical cores
        std::vector<int> ccd1_cores;  // CCD 1 physical cores
        std::vector<int> smt_siblings; // SMT sibling cores
        bool is_ryzen_5900x;
    };

    CPUTopology topology_;

    void detect_topology() {
        #ifdef __linux__
        // Parse /proc/cpuinfo for Ryzen 5900X detection
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;

        while (std::getline(cpuinfo, line)) {
            if (line.find("model name") != std::string::npos &&
                line.find("5900X") != std::string::npos) {
                topology_.is_ryzen_5900x = true;
                break;
            }
        }

        if (topology_.is_ryzen_5900x) {
            // Ryzen 5900X specific topology
            topology_.ccd0_cores = {0, 1, 2, 3, 4, 5};      // CCD0 physical
            topology_.ccd1_cores = {6, 7, 8, 9, 10, 11};    // CCD1 physical
            topology_.smt_siblings = {12, 13, 14, 15, 16, 17,  // CCD0 SMT
                                     18, 19, 20, 21, 22, 23};   // CCD1 SMT
        }
        #endif
    }

public:
    ThreadAffinityManager() {
        detect_topology();
    }

    void set_thread_affinity(int thread_id, int total_threads) {
        #ifdef __linux__
        if (!topology_.is_ryzen_5900x) {
            return;  // Only optimize for known topology
        }

        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);

        if (total_threads <= 6) {
            // Use only CCD0 for best cache locality
            CPU_SET(topology_.ccd0_cores[thread_id % 6], &cpuset);
        } else if (total_threads <= 12) {
            // Use both CCDs, physical cores only
            if (thread_id < 6) {
                CPU_SET(topology_.ccd0_cores[thread_id], &cpuset);
            } else {
                CPU_SET(topology_.ccd1_cores[thread_id - 6], &cpuset);
            }
        } else {
            // Use SMT siblings for >12 threads
            if (thread_id < 12) {
                // Physical cores
                int core = thread_id < 6 ?
                    topology_.ccd0_cores[thread_id] :
                    topology_.ccd1_cores[thread_id - 6];
                CPU_SET(core, &cpuset);
            } else {
                // SMT siblings
                CPU_SET(topology_.smt_siblings[thread_id - 12], &cpuset);
            }
        }

        pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
        #endif
    }
};
```

#### 4.2 Epoch-Based Tree Clearing

**File**: `cpp_extensions/mcts/tree.hpp`

```cpp
class MCTSTree {
private:
    // Epoch counter to avoid memset
    std::atomic<uint32_t> global_epoch_{0};

    // Per-node epoch tracking (4 bytes per node)
    std::vector<uint32_t> node_epochs_;

    // Node data arrays (unchanged)
    std::vector<std::atomic<uint32_t>> visit_counts_;
    std::vector<std::atomic<float>> total_values_;
    // ... other arrays ...

public:
    void clear() {
        // OLD: memset(visit_counts_.data(), 0, max_nodes_ * sizeof(uint32_t));
        // NEW: Just increment epoch (instant!)
        global_epoch_.fetch_add(1, std::memory_order_relaxed);
        num_nodes_ = 1;  // Reset to root only
    }

    bool is_node_valid(NodeIndex idx) const {
        return node_epochs_[idx] == global_epoch_.load(std::memory_order_relaxed);
    }

    void initialize_node(NodeIndex idx) {
        // Lazy initialization on first access
        uint32_t current_epoch = global_epoch_.load(std::memory_order_relaxed);
        if (node_epochs_[idx] != current_epoch) {
            // First access in new epoch - initialize
            visit_counts_[idx].store(0, std::memory_order_relaxed);
            total_values_[idx].store(0.0f, std::memory_order_relaxed);
            virtual_losses_[idx].store(0.0f, std::memory_order_relaxed);
            node_epochs_[idx] = current_epoch;
        }
    }

    uint32_t get_visit_count(NodeIndex idx) {
        initialize_node(idx);  // Ensure initialized
        return visit_counts_[idx].load(std::memory_order_relaxed);
    }

    // Allocation sets epoch for new nodes
    NodeIndex allocate_children(size_t count) {
        NodeIndex base = num_nodes_.fetch_add(count, std::memory_order_relaxed);
        uint32_t current_epoch = global_epoch_.load(std::memory_order_relaxed);

        // Mark all new nodes with current epoch
        for (size_t i = 0; i < count; ++i) {
            node_epochs_[base + i] = current_epoch;
        }

        return base;
    }
};
```

**Impact**: Tree clearing reduced from 10-50ms to <0.1ms (instant epoch increment)

#### 4.3 Per-Thread Memory Arenas

**File**: `cpp_extensions/mcts/thread_local_arena.hpp`

```cpp
class ThreadLocalArena {
private:
    static constexpr size_t NODES_PER_ARENA = 1000000;  // 1M nodes per thread
    static constexpr size_t BLOCK_SIZE = 64;            // Allocate in blocks

    struct Arena {
        NodeIndex base_index;
        std::atomic<uint32_t> allocated{0};
        std::vector<NodeIndex> free_list;

        // Cache-line aligned allocation pointer
        alignas(64) char* memory_ptr;
        size_t memory_size;
    };

    std::vector<Arena> arenas_;
    thread_local int thread_arena_id_ = -1;

public:
    void initialize(int num_threads) {
        arenas_.resize(num_threads);

        size_t nodes_per_thread = NODES_PER_ARENA;

        for (int i = 0; i < num_threads; ++i) {
            arenas_[i].base_index = i * nodes_per_thread;
            arenas_[i].memory_size = nodes_per_thread * 32;  // 32 bytes per node

            // Allocate aligned memory for arena
            posix_memalign(
                (void**)&arenas_[i].memory_ptr,
                64,  // Cache line alignment
                arenas_[i].memory_size
            );

            // Pre-populate free list
            arenas_[i].free_list.reserve(nodes_per_thread / 10);
        }
    }

    NodeIndex allocate_nodes(size_t count) {
        // Get thread-local arena
        if (thread_arena_id_ == -1) {
            thread_arena_id_ = get_thread_id();
        }

        Arena& arena = arenas_[thread_arena_id_];

        // Try free list first (no synchronization needed)
        if (!arena.free_list.empty() && count <= BLOCK_SIZE) {
            NodeIndex node = arena.free_list.back();
            arena.free_list.pop_back();
            return node;
        }

        // Allocate new block from arena
        uint32_t offset = arena.allocated.fetch_add(count, std::memory_order_relaxed);

        if (offset + count <= NODES_PER_ARENA) {
            return arena.base_index + offset;
        }

        // Arena exhausted, fall back to global pool
        return allocate_from_global_pool(count);
    }

    void free_nodes(NodeIndex start, size_t count) {
        int arena_id = start / NODES_PER_ARENA;

        if (arena_id < arenas_.size()) {
            // Return to arena's free list
            Arena& arena = arenas_[arena_id];

            if (count <= BLOCK_SIZE) {
                // Small allocation - add to free list
                arena.free_list.push_back(start);
            }
            // Large allocations not recycled (fragmentation prevention)
        }
    }
};
```

### 5. Root Pre-Expansion Strategy

**File**: `cpp_extensions/mcts/continuous_simulation_runner.cpp`

```cpp
class EnhancedContinuousSimulationRunner : public ContinuousSimulationRunner {
private:
    bool ensure_root_expanded(IGameState& root_state, NodeIndex root_index) {
        if (tree_.is_expanded(root_index)) {
            return false;  // Already expanded
        }

        // Perform synchronous inference for root
        py::gil_scoped_acquire gil;

        try {
            // Get policy and value for root
            auto inference_result = inference_callback_->single_inference(root_state);
            std::vector<float> policy = inference_result.first;
            float value = inference_result.second;

            // Expand root node
            expand_node_with_result(root_index, root_state, policy, value);

            // Add Dirichlet noise to root (exploration)
            if (config_.add_dirichlet_noise) {
                add_dirichlet_noise_to_root(root_index, config_.dirichlet_alpha);
            }

            return true;
        } catch (const std::exception& e) {
            LOG(ERROR) << "Root expansion failed: " << e.what();
            return false;
        }
    }

    void add_dirichlet_noise_to_root(NodeIndex root, float alpha) {
        // Sample Dirichlet distribution
        std::gamma_distribution<float> gamma(alpha);
        std::mt19937 rng(std::random_device{}());

        size_t num_children = tree_.get_num_children(root);
        std::vector<float> noise(num_children);
        float sum = 0.0f;

        for (size_t i = 0; i < num_children; ++i) {
            noise[i] = gamma(rng);
            sum += noise[i];
        }

        // Normalize and mix with priors
        const float epsilon = 0.25f;
        float* priors = tree_.get_prior_probs_ptr();
        NodeIndex first_child = tree_.get_first_child_index(root);

        for (size_t i = 0; i < num_children; ++i) {
            NodeIndex child = first_child + i;
            float original_prior = priors[child];
            float dirichlet = noise[i] / sum;

            // Mix: (1-ε)P + εDir
            priors[child] = (1.0f - epsilon) * original_prior + epsilon * dirichlet;
        }
    }

public:
    int run_continuous(IGameState& root_state,
                      NodeIndex root_index,
                      AsyncInferenceQueue& queue,
                      int num_simulations) override {

        // PRE-EXPAND ROOT (eliminates initial bottleneck)
        ensure_root_expanded(root_state, root_index);

        // Set thread affinity for optimal cache usage
        thread_affinity_manager_.set_thread_affinity(
            get_thread_id(),
            config_.num_threads
        );

        // Continue with normal simulation loop
        return ContinuousSimulationRunner::run_continuous(
            root_state, root_index, queue, num_simulations
        );
    }
};
```

## Performance Projections

### Baseline Measurements
- Current: 3,831 sims/sec
- GPU inference: 32.8% of time (0.117s per 1000 sims)
- MCTS overhead: 67.2% of time (0.240s per 1000 sims)

### Optimization Impact

| Component | Current Time | Optimized Time | Speedup |
|-----------|-------------|----------------|---------|
| Tree Clearing | 0.030s | 0.0001s | 300× |
| Virtual Loss (WU-UCT + Masking) | 0.030s | 0.020s | 1.5× |
| Root Serialization | 0.040s | 0.002s | 20× |
| Queue Operations | 0.100s | 0.010s | 10× |
| Python Overhead | 0.050s | 0.005s | 10× |
| Memory Operations | 0.020s | 0.010s | 2× |
| **Total MCTS** | **0.270s** | **0.047s** | **5.7×** |
| **Total Time** | **0.357s** | **0.164s** | **2.18×** |
| **Throughput** | **3,831/s** | **26,000/s** | **6.8×** |

## Testing Strategy

### Performance Benchmarks
```bash
# Thread scaling test
for threads in 1 2 4 8 12; do
    python scripts/benchmark_throughput.py \
        --threads $threads \
        --simulations 10000 \
        --game gomoku
done

# Collision rate measurement
python scripts/measure_collisions.py \
    --threads 8 \
    --simulations 10000 \
    --verbose
```

### Quality Validation
```python
# A/B test framework
class SearchQualityValidator:
    def compare_implementations(self, baseline, candidate, num_positions=1000):
        results = {
            'policy_agreement': [],
            'value_mse': [],
            'win_rate': []
        }

        for position in test_positions[:num_positions]:
            # Run searches
            baseline_policy, baseline_value = baseline.search(position, 1600)
            candidate_policy, candidate_value = candidate.search(position, 1600)

            # Compare policies (KL divergence)
            kl_div = entropy(baseline_policy, candidate_policy)
            results['policy_agreement'].append(1.0 - min(kl_div, 1.0))

            # Compare values
            value_diff = (baseline_value - candidate_value) ** 2
            results['value_mse'].append(value_diff)

        return {
            'mean_policy_agreement': np.mean(results['policy_agreement']),
            'mean_value_mse': np.mean(results['value_mse']),
            'pass': np.mean(results['policy_agreement']) > 0.95
        }
```

## Risk Mitigation

### Incremental Rollout
1. **Phase 1**: WU-UCT + Root pre-expansion (low risk)
2. **Phase 2**: Lock-free queue (medium risk, extensive testing)
3. **Phase 3**: Zero-copy tensors (medium risk, fallback available)
4. **Phase 4**: Full optimization (combine all)

### Fallback Options
- Classic virtual loss (config flag)
- Mutex-based queue (runtime switch)
- NumPy tensor path (compatibility mode)
- Single-threaded mode (debugging)

## 🔴 CRITICAL MISSING OPTIMIZATIONS (review.pdf Analysis)

After comprehensive review against review.pdf, **two CRITICAL optimizations** were identified as missing:

### 6. Condition Variables for Async Coordination (T006c)

**Problem** (review.pdf pages 8-9):
> "The current busy-wait loop should be replaced with a blocking notification mechanism. The ContinuousSimulationRunner loop uses a polling mechanism to check for completed NN results. If no result is ready, threads sleep for a very short interval (50–100 µs) and try again. These frequent wakes add CPU overhead – threads spend significant time in spin-wait, which burns CPU time that could run more simulations."

Current implementation (T006b) eliminated mutexes but still uses **polling** with 10μs sleeps in `collect_batch()`:

```cpp
// CURRENT (INEFFICIENT): Polling wastes CPU
while (batch.size() < min_batch_size) {
    if (pending_requests_.try_dequeue(request)) {
        batch.push_back(std::move(request));
    } else {
        std::this_thread::sleep_for(std::chrono::microseconds(10));  // ❌ WASTE!
    }
}
```

**Solution**: Use `std::condition_variable` for efficient blocking:

```cpp
// PROPOSED: Efficient wait/notify
class AsyncInferenceQueue {
private:
    MPMCRingBuffer<InferenceRequest, 4096> pending_requests_;  // Lock-free (unchanged)

    // NEW: Condition variable for efficient waiting
    std::mutex cv_mutex_;                    // Only for CV, not for queue ops
    std::condition_variable request_ready_;  // Signaled when requests available
    std::atomic<bool> shutting_down_{false};

public:
    uint64_t submit_request(...) {
        // Enqueue (lock-free, unchanged)
        while (!pending_requests_.try_enqueue(std::move(request))) {
            std::this_thread::yield();
        }

        // NEW: Notify one waiting thread
        request_ready_.notify_one();

        return request.request_id;
    }

    std::vector<InferenceRequest> collect_batch(size_t min_batch_size, double timeout_ms) {
        std::vector<InferenceRequest> batch;
        auto deadline = std::chrono::steady_clock::now() +
                       std::chrono::duration<double, std::milli>(timeout_ms);

        while (batch.size() < min_batch_size && !shutting_down_.load()) {
            InferenceRequest request;
            if (pending_requests_.try_dequeue(request)) {
                batch.push_back(std::move(request));
                continue;
            }

            // NEW: Block on condition variable instead of polling
            std::unique_lock<std::mutex> lock(cv_mutex_);
            auto remaining = deadline - std::chrono::steady_clock::now();
            if (remaining.count() <= 0) break;

            // Wait for notification or timeout (no busy-wait!)
            request_ready_.wait_for(lock, remaining);
        }

        // Opportunistically grab more (unchanged)
        while (batch.size() < min_batch_size * 2) {
            InferenceRequest request;
            if (!pending_requests_.try_dequeue(request)) break;
            batch.push_back(std::move(request));
        }

        return batch;
    }

    void shutdown() {
        shutting_down_.store(true, std::memory_order_release);
        request_ready_.notify_all();  // Wake all waiting threads
    }
};
```

**Performance Impact** (review.pdf page 9):
> "A properly implemented wait/notify queue with O(1) pending lookup will drastically reduce the CPU wasted on coordination. The spec expects async coordination overhead to drop below 20% of runtime (currently it's ~67%)."

**Expected Impact**: **1.3-1.5× throughput improvement** (reclaim CPU from polling)

### 7. Mixed Precision FP16 GPU Inference (T008f)

**Problem** (review.pdf pages 8 & 13):
Review.pdf mentions FP16 **multiple times** as critical:
> "Mixed precision can give a big speedup on 3060 Ti" (page 8)
> "wrap the model call in torch.cuda.amp.autocast() to use FP16" (page 8)
> "FP16 can nearly double inference throughput on GPUs that have tensor cores (like RTX 3060 Ti)" (page 13)

Current T008b implementation mentions autocast but doesn't **validate** it's enabled and working.

**Solution**: Enable and validate FP16 mixed precision:

```python
class DLPackInferenceBridge:
    def __init__(self, model, device, use_mixed_precision=True):
        self.model = model
        self.device = device
        self.use_mixed_precision = use_mixed_precision and device.type == 'cuda'

        # Enable cuDNN auto-tuner for best kernel selection
        if self.use_mixed_precision:
            torch.backends.cudnn.benchmark = True

    def batch_inference(self, states):
        # Convert DLPack → PyTorch tensor (zero-copy)
        dlpack_capsule = mcts_py.create_batch_tensor_from_states(states)
        cpu_tensor = torch.from_dlpack(dlpack_capsule)

        # Transfer to GPU (async)
        gpu_tensor = cpu_tensor.to(self.device, non_blocking=True)

        # CRITICAL: Mixed precision inference
        with torch.no_grad():
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():  # ✅ FP16 inference
                    policy_logits, values = self.model(gpu_tensor)
            else:
                policy_logits, values = self.model(gpu_tensor)  # FP32

        # Post-process (on GPU)
        policies = torch.softmax(policy_logits, dim=1)
        values = torch.tanh(values).squeeze(-1)

        # Return as numpy (single copy back to CPU)
        return (policies.cpu().numpy(), values.cpu().numpy())
```

**Performance Impact** (review.pdf page 13):
> "FP16 can nearly double inference throughput on GPUs that have tensor cores (like RTX 3060 Ti)"

**Expected Impact**: **1.5-2× GPU inference speedup**

### Updated Performance Projections (with T006c + T008f)

| Component | Baseline Time | After Phase 1+2 | + T006c | + T008f | Speedup |
|-----------|--------------|-----------------|---------|----------|----------|
| MCTS Selection | 0.080s | 0.020s | 0.020s | 0.020s | 4× |
| MCTS Backup | 0.060s | 0.012s | 0.012s | 0.012s | 5× |
| Tree Clearing | 0.030s | 0.0001s | 0.0001s | 0.0001s | 300× |
| Queue Coordination | 0.100s | **0.050s** | **0.010s** | 0.010s | **10×** |
| Python Overhead | 0.050s | 0.010s | 0.010s | 0.010s | 5× |
| Root Serialization | 0.040s | 0.002s | 0.002s | 0.002s | 20× |
| **GPU Inference** | **0.117s** | **0.117s** | **0.117s** | **0.060s** | **2×** |
| **Total Time/1K** | **0.477s** | **0.211s** | **0.171s** | **0.114s** | **4.2×** |
| **Throughput** | **3,831/s** | **8,500/s** | **12,000/s** | **26,000/s** | **6.8×** |

**Critical Path to 30k sims/sec:**
1. ✅ **Phase 1+2 Complete**: ~8-12k sims/sec (baseline × 2-3)
2. 🔴 **Implement T006c**: +40% = ~12-18k sims/sec
3. 🔴 **Implement T008f**: +100% GPU = ~18-36k sims/sec
4. **Tune parameters (T018-T019)**: +10-20% = ~20-40k sims/sec
5. **Target achieved**: ≥25-30k sims/sec

## Success Metrics

### Must Have
- [ ] 20,000+ sims/sec (minimum viable) - **Achievable with T006c + T008f**
- [ ] No search quality regression
- [ ] Stable under extended runs

### Should Have
- [ ] 25,000+ sims/sec (target) - **Achievable with T006c + T008f + tuning**
- [ ] <5% collision rate
- [ ] 85%+ GPU utilization

### Nice to Have
- [ ] 30,000+ sims/sec (stretch) - **Achievable with T006c + T008f + T011 + tuning**
- [ ] <2% collision rate
- [ ] 90%+ GPU utilization