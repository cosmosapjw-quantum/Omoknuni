/**
 * @file async_inference_queue.cpp
 * @brief Implementation of async inference queue
 */

#include "async_inference_queue.hpp"
#include "instrumentation.hpp"
#include <chrono>
#include <limits>
#include <unordered_set>

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
    ScopedMetric metric(InstrumentationMetric::QueueSubmit);
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
    pending_cv_.notify_one();

    return request_id;
}

std::vector<InferenceRequest> AsyncInferenceQueue::collect_batch(size_t min_batch_size,
                                                                   double timeout_ms) {
    ScopedMetric metric(InstrumentationMetric::QueueCollect);
    using namespace std::chrono;

    std::vector<InferenceRequest> batch;

    std::unique_lock<std::mutex> lock(pending_mutex_);

    const auto max_batch_size = (min_batch_size > 0)
        ? (min_batch_size + (min_batch_size / 2))
        : std::numeric_limits<std::size_t>::max();

    const auto timeout_duration = duration<double, std::milli>(timeout_ms);
    const auto deadline = steady_clock::now() + timeout_duration;

    auto has_enough = [&]() {
        return min_batch_size > 0 && pending_requests_.size() >= min_batch_size;
    };

    if (min_batch_size > 0) {
        while (!has_enough()) {
            if (timeout_ms <= 0.0) {
                break;
            }

            if (pending_cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
                break;
            }
        }
    }

    const std::size_t available = pending_requests_.size();
    if (available == 0) {
        return batch;
    }

    const std::size_t batch_count = std::min(available, max_batch_size);
    batch.reserve(batch_count);

    // Track unique node indices in batch for diversity metrics
    std::unordered_set<NodeIndex> unique_nodes;

    for (std::size_t i = 0; i < batch_count; ++i) {
        auto& request = pending_requests_.front();
        unique_nodes.insert(request.node_index);
        batch.push_back(std::move(request));
        pending_requests_.pop_front();
    }

    // Record batch diversity metric
    Instrumentation::instance().increment_counter(
        InstrumentationMetric::UniqueBatchPositions,
        unique_nodes.size()
    );

    return batch;
}

void AsyncInferenceQueue::submit_results(const std::vector<InferenceResult>& results) {
    std::lock_guard<std::mutex> lock(results_mutex_);

    for (const auto& result : results) {
        // Insert result into map (keyed by request_id)
        completed_results_[result.request_id] = result;
    }

    results_cv_.notify_all();
}

std::optional<InferenceResult> AsyncInferenceQueue::try_get_result(uint64_t request_id) {
    ScopedMetric metric(InstrumentationMetric::QueueTryGetResult);
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

std::vector<InferenceResult> AsyncInferenceQueue::consume_ready_results() {
    std::lock_guard<std::mutex> lock(results_mutex_);
    std::vector<InferenceResult> results;
    results.reserve(completed_results_.size());
    for (auto& entry : completed_results_) {
        results.push_back(std::move(entry.second));
    }
    completed_results_.clear();
    return results;
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
