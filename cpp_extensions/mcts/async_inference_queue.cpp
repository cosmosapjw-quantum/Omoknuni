/**
 * @file async_inference_queue.cpp
 * @brief Implementation of async inference queue
 */

#include "async_inference_queue.hpp"
#include <chrono>
#include <thread>

namespace mcts {

AsyncInferenceQueue::AsyncInferenceQueue()
    : next_request_id_(0) {
}

AsyncInferenceQueue::~AsyncInferenceQueue() {
    // Cleanup any pending requests and results
    // (RAII handles mutex destruction)
}

uint64_t AsyncInferenceQueue::submit_request(std::unique_ptr<IGameState> state,
                                               NodeIndex node_index,
                                               std::vector<NodeIndex> path) {
    // Generate unique request ID
    uint64_t request_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);

    // Create request
    InferenceRequest request;
    request.request_id = request_id;
    request.state = std::move(state);
    request.node_index = node_index;
    request.path = std::move(path);

    // Add to pending queue (thread-safe)
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        pending_requests_.push_back(std::move(request));
    }

    return request_id;
}

std::vector<InferenceRequest> AsyncInferenceQueue::collect_batch(size_t min_batch_size,
                                                                   double timeout_ms) {
    using namespace std::chrono;
    auto start_time = steady_clock::now();
    auto timeout_duration = duration<double, std::milli>(timeout_ms);

    std::vector<InferenceRequest> batch;

    while (true) {
        {
            std::lock_guard<std::mutex> lock(pending_mutex_);

            // Check if we have enough requests
            if (pending_requests_.size() >= min_batch_size) {
                // Cap batch size to avoid processing too many at once
                // This prevents batch size explosion (observed 157-273 vs configured 64)
                // Target: 1.5× min_batch_size for optimal GPU utilization without overload
                size_t max_batch_size = min_batch_size + (min_batch_size / 2);
                size_t batch_count = std::min(pending_requests_.size(), max_batch_size);

                batch.reserve(batch_count);
                auto it = pending_requests_.begin();
                for (size_t i = 0; i < batch_count && it != pending_requests_.end(); ++i, ++it) {
                    batch.push_back(std::move(*it));
                }
                pending_requests_.erase(pending_requests_.begin(), it);
                return batch;
            }

            // Check if timeout elapsed
            auto elapsed = steady_clock::now() - start_time;
            if (elapsed >= timeout_duration) {
                // Return whatever we have (might be empty), but cap at max_batch_size
                if (!pending_requests_.empty()) {
                    size_t max_batch_size = min_batch_size + (min_batch_size / 2);
                    size_t batch_count = std::min(pending_requests_.size(), max_batch_size);

                    batch.reserve(batch_count);
                    auto it = pending_requests_.begin();
                    for (size_t i = 0; i < batch_count && it != pending_requests_.end(); ++i, ++it) {
                        batch.push_back(std::move(*it));
                    }
                    pending_requests_.erase(pending_requests_.begin(), it);
                }
                return batch;
            }
        }

        // Brief sleep to avoid busy-waiting
        std::this_thread::sleep_for(microseconds(100));
    }
}

void AsyncInferenceQueue::submit_results(const std::vector<InferenceResult>& results) {
    std::lock_guard<std::mutex> lock(results_mutex_);

    for (const auto& result : results) {
        // Insert result into map (keyed by request_id)
        completed_results_[result.request_id] = result;
    }
}

std::optional<InferenceResult> AsyncInferenceQueue::try_get_result(uint64_t request_id) {
    std::lock_guard<std::mutex> lock(results_mutex_);

    auto it = completed_results_.find(request_id);
    if (it == completed_results_.end()) {
        return std::nullopt;
    }

    // Move result out and erase from map (consume)
    InferenceResult result = std::move(it->second);
    completed_results_.erase(it);

    return result;
}

bool AsyncInferenceQueue::has_results() const {
    std::lock_guard<std::mutex> lock(results_mutex_);
    return !completed_results_.empty();
}

size_t AsyncInferenceQueue::pending_count() const {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    return pending_requests_.size();
}

size_t AsyncInferenceQueue::results_count() const {
    std::lock_guard<std::mutex> lock(results_mutex_);
    return completed_results_.size();
}

size_t AsyncInferenceQueue::get_memory_usage() const {
    size_t total = 0;

    // Pending requests
    {
        std::lock_guard<std::mutex> lock(pending_mutex_);
        // Estimate: ~100 bytes per request (state pointer, path vector, metadata)
        total += pending_requests_.size() * 100;
    }

    // Completed results
    {
        std::lock_guard<std::mutex> lock(results_mutex_);
        // Estimate: ~500 bytes per result (policy vector ~225 floats avg * 4 bytes + overhead)
        total += completed_results_.size() * 500;
    }

    return total;
}

std::vector<uint64_t> AsyncInferenceQueue::get_ready_request_ids() const {
    std::lock_guard<std::mutex> lock(results_mutex_);
    std::vector<uint64_t> ids;
    ids.reserve(completed_results_.size());
    for (const auto& entry : completed_results_) {
        ids.push_back(entry.first);
    }
    return ids;
}

} // namespace mcts
