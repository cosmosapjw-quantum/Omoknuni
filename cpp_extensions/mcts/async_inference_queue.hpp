/**
 * @file async_inference_queue.hpp
 * @brief Lock-free async inference queue for non-blocking MCTS simulation
 *
 * This module implements a wait-free queue system that decouples MCTS simulation
 * threads from neural network inference. Simulations submit inference requests
 * asynchronously and continue working, while a background coordinator batches
 * requests and calls Python inference once per batch.
 *
 * Performance targets:
 * - Request submission: <0.1ms (wait-free with MPMCRingBuffer)
 * - Batch collection: triggered by count (≥32) OR timeout (≤2ms)
 * - Result retrieval: <0.1ms (lock-free O(1) ring buffer lookup)
 * - Memory: Fixed 8MB allocation (4096 requests + 8192 results)
 *
 * Key design principles:
 * - Wait-free request submission (no locks, no blocking)
 * - Lock-free result retrieval with O(1) ring buffer indexing
 * - Timeout-based batch collection via condition variables (T006c - efficient blocking)
 * - Fixed memory footprint with predictable allocation
 *
 * Architecture (T006b):
 * - Lock-free MPMCRingBuffer for pending requests (capacity 4096)
 * - Ring buffer array for completed results (capacity 8192)
 * - Atomic counters for queue depth monitoring
 * - No mutexes or condition variables in hot paths
 */

#pragma once

#include "tree.hpp"
#include "lock_free_queue.hpp"
#include "../utils/igamestate.h"
#include <vector>
#include <array>
#include <optional>
#include <memory>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <condition_variable>

namespace mcts {

// Forward declaration for game state interface
using IGameState = alphazero::core::IGameState;

/**
 * @brief Request for neural network inference
 *
 * Represents a single position that needs evaluation. Submitted by
 * simulation threads during tree traversal.
 *
 * **Optimization (T018g)**: Features are pre-extracted in C++ before
 * queue submission, eliminating the need for state cloning and heap
 * allocation. This reduces per-simulation overhead from 418μs to ~10μs.
 */
struct InferenceRequest {
    uint64_t request_id;                        // Unique identifier for this request
    std::vector<float> features;                // Pre-extracted features (C × H × W)
    int action_space_size;                      // Action space size (for fallback policy)
    int board_size;                             // Board size (for tensor reshaping)
    int num_feature_planes;                     // Number of feature planes
    NodeIndex node_index;                       // Tree node to expand
    std::vector<NodeIndex> path;                // Path from root to this node

    // Default constructor for container compatibility
    InferenceRequest() : action_space_size(0), board_size(0), num_feature_planes(0) {}

    // Move-only type (owns features vector)
    InferenceRequest(InferenceRequest&&) = default;
    InferenceRequest& operator=(InferenceRequest&&) = default;
    InferenceRequest(const InferenceRequest&) = delete;
    InferenceRequest& operator=(const InferenceRequest&) = delete;
};

/**
 * @brief Result from neural network inference
 *
 * Contains policy and value for a previously submitted request.
 */
struct InferenceResult {
    uint64_t request_id;                        // Matches original request
    std::vector<float> policy;                  // Prior probabilities over actions
    float value;                                // Position evaluation [-1, 1]

    // Copyable and movable
    InferenceResult() = default;
    InferenceResult(const InferenceResult&) = default;
    InferenceResult(InferenceResult&&) = default;
    InferenceResult& operator=(const InferenceResult&) = default;
    InferenceResult& operator=(InferenceResult&&) = default;
};

/**
 * @brief Thread-safe async inference queue
 *
 * Decouples MCTS simulation threads from GPU inference by providing:
 * 1. Non-blocking request submission (threads never wait)
 * 2. Batched request collection (by count or timeout)
 * 3. Result distribution back to threads
 *
 * Thread Safety:
 * - Multiple threads can submit requests concurrently
 * - Single coordinator thread collects batches
 * - Multiple threads can retrieve results concurrently
 * - All operations protected by mutexes
 *
 * Performance Characteristics:
 * - Request submission: O(1), <0.1ms
 * - Batch collection: O(batch_size), <2ms timeout
 * - Result retrieval: O(1), <0.1ms
 * - Memory: ~100 bytes per pending request
 */
class AsyncInferenceQueue {
public:
    /**
     * @brief Construct empty inference queue
     */
    AsyncInferenceQueue();

    /**
     * @brief Destructor (cleanup any pending requests)
     */
    ~AsyncInferenceQueue();

    /**
     * @brief Submit inference request with pre-extracted features (non-blocking)
     *
     * Adds request to pending queue and returns immediately. Thread does NOT
     * wait for inference to complete.
     *
     * **Optimization (T018g)**: Features are pre-extracted in C++ to eliminate
     * the 418μs clone overhead. This results in 3.7× throughput improvement.
     *
     * Thread Safety: Safe to call from multiple threads concurrently
     *
     * @param features Pre-extracted feature tensor (C×H×W flattened)
     * @param action_space_size Action space size (for fallback policy)
     * @param board_size Board size (for tensor reshaping in Python)
     * @param num_feature_planes Number of feature planes (for reshaping)
     * @param node_index Tree node to expand with result
     * @param path Path from root to node (for backup)
     * @return Unique request ID for retrieving result later
     */
    uint64_t submit_request(std::vector<float> features,
                            int action_space_size,
                            int board_size,
                            int num_feature_planes,
                            NodeIndex node_index,
                            std::vector<NodeIndex> path);

    /**
     * @brief Collect batch of pending requests
     *
     * Returns when EITHER:
     * - Number of pending requests >= min_batch_size
     * - Timeout elapsed (timeout_ms milliseconds)
     *
     * Whichever condition is met first triggers the batch return.
     *
     * Thread Safety: Should only be called by single coordinator thread
     *
     * @param min_batch_size Minimum batch size to wait for (e.g., 32)
     * @param timeout_ms Maximum wait time in milliseconds (e.g., 2.0)
     * @return Vector of requests to process (empty if timeout with no requests)
     */
    std::vector<InferenceRequest> collect_batch(size_t min_batch_size, double timeout_ms);

    /**
     * @brief Submit batch of inference results
     *
     * Called by coordinator thread after GPU inference completes.
     * Makes results available for simulation threads to retrieve.
     *
     * Thread Safety: Should only be called by single coordinator thread
     *
     * @param results Vector of results matching previously collected requests
     */
    void submit_results(const std::vector<InferenceResult>& results);

    /**
     * @brief Try to retrieve result for a request (non-blocking)
     *
     * Checks if result is available for given request ID. If found, returns
     * the result and removes it from the map (consumed).
     *
     * Thread Safety: Safe to call from multiple threads concurrently
     *
     * @param request_id Request ID from submit_request()
     * @return Result if available, std::nullopt otherwise
     */
    std::optional<InferenceResult> try_get_result(uint64_t request_id);

    /**
     * @brief Consume all ready results in a single batch.
     *
     * Moves the completed results into a vector and clears the internal map.
     *
     * Thread Safety: Safe to call from multiple threads; typically used by
     * the async coordinator / simulation runners.
     */
    std::vector<InferenceResult> consume_ready_results();

    /**
     * @brief Check if any results are available
     *
     * Quick check before calling try_get_result() to avoid unnecessary polling.
     *
     * Thread Safety: Safe to call from multiple threads concurrently
     *
     * @return true if results map is non-empty
     */
    bool has_results() const;

    /**
     * @brief Get number of pending requests
     *
     * Useful for monitoring queue depth and detecting backpressure.
     *
     * Thread Safety: Safe to call from any thread
     *
     * @return Number of requests waiting for inference
     */
    size_t pending_count() const;

    /**
     * @brief Get number of completed results waiting for retrieval
     *
     * Useful for monitoring if results are being consumed quickly enough.
     *
     * Thread Safety: Safe to call from any thread
     *
     * @return Number of results available for retrieval
     */
    size_t results_count() const;

    /**
     * @brief Get memory usage estimate in bytes
     *
     * Includes pending requests and completed results.
     *
     * @return Estimated memory usage
     */
    size_t get_memory_usage() const;

    /**
     * @brief Wake up threads waiting in collect_batch()
     *
     * This is called during coordinator shutdown to ensure threads waiting
     * on condition variables are woken up so they can check the running flag
     * and exit cleanly.
     *
     * Thread Safety: Safe to call from any thread
     */
    void shutdown();

    /**
     * @brief Snapshot the request IDs with completed inference results.
     *
     * Thread Safety: Safe to call from any thread.
     *
     * @return Vector of request IDs currently ready for retrieval
     */
    [[deprecated("Use try_get_result() instead - no bulk operations needed")]]
    std::vector<uint64_t> get_ready_request_ids() const;

private:
    // Request ID generation
    std::atomic<uint64_t> next_request_id_{0};

    // Lock-free pending requests queue (T006b)
    MPMCRingBuffer<InferenceRequest, 4096> pending_requests_;
    std::atomic<size_t> pending_count_{0};

    // Condition variable for efficient waiting (T006c)
    std::mutex cv_mutex_;
    std::condition_variable request_ready_;
    std::atomic<bool> shutting_down_{false};

    // Lock-free completed results ring buffer (T006b)
    static constexpr size_t RESULTS_BUFFER_CAPACITY = 8192;

    struct alignas(64) ResultSlot {
        std::atomic<bool> occupied{false};
        uint64_t request_id{0};
        InferenceResult data;
    };

    std::array<ResultSlot, RESULTS_BUFFER_CAPACITY> results_buffer_;
    std::atomic<size_t> results_count_{0};
};

} // namespace mcts
