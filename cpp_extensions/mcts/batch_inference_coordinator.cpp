/**
 * @file batch_inference_coordinator.cpp
 * @brief Implementation of background batching coordinator
 */

#include "batch_inference_coordinator.hpp"
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace {

using namespace mcts;

InferenceResult make_fallback_result(const InferenceRequest& request) {
    InferenceResult fallback;
    fallback.request_id = request.request_id;
    fallback.value = 0.0f;

    const IGameState* state = request.state.get();
    int action_space = state ? state->getActionSpaceSize() : 0;
    fallback.policy.assign(action_space > 0 ? action_space : 1, 0.0f);

    if (state && action_space > 0) {
        auto legal_moves = state->getLegalMoves();
        if (!legal_moves.empty()) {
            float prob = 1.0f / static_cast<float>(legal_moves.size());
            for (int move : legal_moves) {
                int index = move;
                if (index < 0 || index >= action_space) {
                    index = action_space - 1;
                }
                fallback.policy[index] = prob;
            }
        } else {
            float prob = 1.0f / static_cast<float>(action_space);
            std::fill(fallback.policy.begin(), fallback.policy.end(), prob);
        }
    } else if (action_space > 0) {
        float prob = 1.0f / static_cast<float>(action_space);
        std::fill(fallback.policy.begin(), fallback.policy.end(), prob);
    } else {
        fallback.policy[0] = 1.0f;
    }

    return fallback;
}

std::vector<InferenceResult> build_fallback_results(const std::vector<InferenceRequest>& batch) {
    std::vector<InferenceResult> results;
    results.reserve(batch.size());
    for (const auto& request : batch) {
        results.push_back(make_fallback_result(request));
    }
    return results;
}

} // namespace

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

    // Wake up coordinator thread if it's waiting in collect_batch()
    if (queue_) {
        queue_->shutdown();
    }

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
        bool had_error = false;
        try {
            inference_results = callback_->batch_inference(states);
        } catch (const std::exception& e) {
            had_error = true;
            std::cerr << "Batch inference failed: " << e.what() << std::endl;
        }

        if (!had_error && inference_results.size() == batch.size()) {
            std::vector<InferenceResult> results;
            results.reserve(batch.size());
            for (size_t i = 0; i < batch.size(); ++i) {
                InferenceResult result;
                result.request_id = batch[i].request_id;
                result.policy = std::move(inference_results[i].first);
                result.value = inference_results[i].second;
                results.push_back(std::move(result));
            }
            queue_->submit_results(results);
            continue;
        }

        // Fallback path: either inference threw or result size mismatched
        if (!had_error && inference_results.size() != batch.size()) {
            std::cerr << "Batch inference returned mismatched result count (" << inference_results.size()
                      << " vs " << batch.size() << "), using uniform fallback\n";
        }

        auto fallback_results = build_fallback_results(batch);
        queue_->submit_results(fallback_results);
    }
}

} // namespace mcts
