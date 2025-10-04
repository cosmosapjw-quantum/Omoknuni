/**
 * @file batch_inference_coordinator.cpp
 * @brief Implementation of background batching coordinator
 */

#include "batch_inference_coordinator.hpp"
#include <stdexcept>

namespace mcts {

void BatchInferenceCoordinator::start(AsyncInferenceQueue& queue,
                                       BatchInferenceCallback& callback,
                                       size_t batch_size,
                                       double timeout_ms) {
    // Check if already running
    if (running_.load(std::memory_order_acquire)) {
        throw std::runtime_error("BatchInferenceCoordinator already running");
    }

    // Store parameters
    queue_ = &queue;
    callback_ = &callback;
    batch_size_ = batch_size;
    timeout_ms_ = timeout_ms;

    // Set running flag
    running_.store(true, std::memory_order_release);

    // Spawn worker thread
    worker_thread_ = std::thread(&BatchInferenceCoordinator::coordinator_loop, this);
}

void BatchInferenceCoordinator::stop() {
    // Check if running
    if (!running_.load(std::memory_order_acquire)) {
        return;  // Already stopped
    }

    // Signal thread to stop
    running_.store(false, std::memory_order_release);

    // Wait for thread to finish
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
}

void BatchInferenceCoordinator::coordinator_loop() {
    while (running_.load(std::memory_order_acquire)) {
        // Phase 1: Collect batch from queue
        // This blocks up to timeout_ms, returns early if batch_size reached
        std::vector<InferenceRequest> batch = queue_->collect_batch(batch_size_, timeout_ms_);

        // Check if batch is empty (timeout with no requests)
        if (batch.empty()) {
            continue;  // No work to do, loop again
        }

        // Phase 2: Extract state pointers from batch
        std::vector<const IGameState*> states;
        states.reserve(batch.size());
        for (const auto& request : batch) {
            states.push_back(request.state.get());
        }

        // Phase 3: Call Python for GPU inference (GIL ACQUIRED ONCE)
        // This is the only GIL crossing in the entire batch
        std::vector<std::pair<std::vector<float>, float>> inference_results;
        try {
            inference_results = callback_->batch_inference(states);
        } catch (const std::exception& e) {
            // Inference failed - submit zero results to unblock waiting threads
            // TODO: Better error handling (fallback policy?)
            continue;
        }

        // Phase 4: Build InferenceResult vector
        if (inference_results.size() != batch.size()) {
            // Result count mismatch - skip this batch
            continue;
        }

        std::vector<InferenceResult> results;
        results.reserve(batch.size());
        for (size_t i = 0; i < batch.size(); ++i) {
            InferenceResult result;
            result.request_id = batch[i].request_id;
            result.policy = inference_results[i].first;
            result.value = inference_results[i].second;
            results.push_back(std::move(result));
        }

        // Phase 5: Submit results back to queue
        queue_->submit_results(results);
    }
}

} // namespace mcts
