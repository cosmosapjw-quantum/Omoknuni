/**
 * @file batch_inference_coordinator.hpp
 * @brief Background thread coordinator for batched GPU inference
 *
 * This module implements a background thread that continuously collects
 * inference requests from AsyncInferenceQueue, batches them, calls Python
 * for GPU inference (single GIL crossing), and distributes results back.
 *
 * Key features:
 * - Single background thread for batching
 * - Reduces GIL crossings from N (per simulation) to 1 (per batch)
 * - Dual-trigger batching (count OR timeout)
 * - Clean lifecycle management (start/stop/join)
 */

#pragma once

#include "async_inference_queue.hpp"
#include "batch_inference_callback.hpp"
#include <thread>
#include <atomic>
#include <memory>

namespace mcts {

/**
 * @brief Background coordinator for batched GPU inference
 *
 * Spawns a background thread that continuously:
 * 1. Collects batch from AsyncInferenceQueue (wait up to timeout_ms)
 * 2. Calls BatchInferenceCallback for GPU inference (GIL acquired ONCE)
 * 3. Submits results back to queue for distribution
 *
 * This reduces GIL time from >50% to <30% by batching all GPU calls.
 *
 * Example usage:
 *   AsyncInferenceQueue queue;
 *   PyBatchInferenceCallback callback(python_batch_fn);
 *
 *   BatchInferenceCoordinator coordinator;
 *   coordinator.start(queue, callback, 32, 2.0);  // batch_size=32, timeout=2ms
 *
 *   // Simulations run in other threads, submitting to queue
 *   // Coordinator processes batches in background
 *
 *   coordinator.stop();  // Clean shutdown
 */
class BatchInferenceCoordinator {
public:
    /**
     * @brief Construct coordinator (thread not started yet)
     */
    BatchInferenceCoordinator() = default;

    /**
     * @brief Destructor ensures clean shutdown
     */
    ~BatchInferenceCoordinator() {
        stop();
    }

    // Prevent copying (thread ownership issues)
    BatchInferenceCoordinator(const BatchInferenceCoordinator&) = delete;
    BatchInferenceCoordinator& operator=(const BatchInferenceCoordinator&) = delete;

    /**
     * @brief Start background coordinator thread
     *
     * Spawns a worker thread that runs coordinator_loop().
     * Thread continuously collects batches and processes them.
     *
     * @param queue Reference to AsyncInferenceQueue for request/result exchange
     * @param callback Reference to BatchInferenceCallback for GPU inference
     * @param batch_size Minimum batch size before triggering inference (e.g., 32)
     * @param timeout_ms Maximum wait time for batch collection (e.g., 2.0ms)
     */
    void start(AsyncInferenceQueue& queue,
               BatchInferenceCallback& callback,
               size_t batch_size,
               double timeout_ms);

    /**
     * @brief Stop background thread and wait for completion
     *
     * Sets running flag to false, waits for thread to exit via join().
     * Safe to call multiple times (idempotent).
     */
    void stop();

    /**
     * @brief Check if coordinator is running
     */
    bool is_running() const {
        return running_.load(std::memory_order_acquire);
    }

private:
    /**
     * @brief Main coordinator loop (runs in background thread)
     *
     * Loop structure:
     * 1. Collect batch from queue (blocking up to timeout_ms)
     * 2. If batch empty, continue (no requests available)
     * 3. Extract state pointers from batch
     * 4. Call callback.batch_inference() - GIL ACQUIRED ONCE
     * 5. Build InferenceResult vector from callback results
     * 6. Submit results back to queue
     * 7. Repeat until running_ == false
     */
    void coordinator_loop();

    // Background worker thread
    std::thread worker_thread_;

    // Thread lifecycle control
    std::atomic<bool> running_{false};

    // Coordinator parameters (set by start())
    AsyncInferenceQueue* queue_{nullptr};
    BatchInferenceCallback* callback_{nullptr};
    size_t batch_size_{32};
    double timeout_ms_{2.0};
};

} // namespace mcts
