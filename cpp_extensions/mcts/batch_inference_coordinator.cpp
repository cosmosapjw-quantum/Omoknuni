/**
 * @file batch_inference_coordinator.cpp
 * @brief Implementation of background batching coordinator
 *
 * PROFILING UPGRADE 2025-10-17:
 * Added comprehensive instrumentation to eliminate "unknown" time bottleneck
 * This coordinator runs in a SEPARATE THREAD - crucial to instrument!
 */

#include "batch_inference_coordinator.hpp"
#include "profiling/enhanced_profiler.hpp"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace mcts::profiling;

namespace {

using namespace mcts;

InferenceResult make_fallback_result(const InferenceRequest& request) {
    InferenceResult fallback;
    fallback.request_id = request.request_id;
    fallback.value = 0.0f;

    // Use action_space_size from request (no state needed after T018g optimization)
    int action_space = request.action_space_size;
    fallback.policy.assign(action_space > 0 ? action_space : 1, 0.0f);

    // Uniform distribution over all actions (legal moves not available without state)
    if (action_space > 0) {
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
        // === PROFILING UPGRADE: Instrument entire loop iteration ===
        PROFILE_SCOPE(ProfileMetric::CoordinatorLoopIteration);

        // Phase 1: Collect batch from queue
        // This blocks up to timeout_ms, returns early if batch_size reached
        std::vector<InferenceRequest> batch;
        {
            PROFILE_SCOPE(ProfileMetric::CoordinatorCollectBatch);
            batch = queue_->collect_batch(batch_size_, timeout_ms_);
        }

        // Check if batch is empty (timeout with no requests)
        if (batch.empty()) {
            PROFILE_COUNTER(ProfileMetric::CoordinatorCollectBatchEmpty, 1);
            PROFILE_SCOPE(ProfileMetric::CoordinatorIdleTime);
            continue;  // No work to do, loop again
        }

        // Track batch processing
        PROFILE_COUNTER(ProfileMetric::CoordinatorBatchCount, 1);

        // Phase 2: Extract features from batch with OpenMP parallelization (T019)
        // Use OpenMP to parallelize feature extraction across the batch
        // Expected: 5-10× speedup with 8 threads on batch of 64
        std::vector<std::vector<float>> features_batch;
        std::vector<int> board_sizes;
        std::vector<int> num_planes_list;

        {
            PROFILE_SCOPE(ProfileMetric::CoordinatorFeatureExtraction);

            // Allocate feature buffers
            {
                PROFILE_SCOPE(ProfileMetric::CoordinatorFeatureAllocation);
                features_batch.resize(batch.size());
                board_sizes.reserve(batch.size());
                num_planes_list.reserve(batch.size());
            }

            // Collect metadata (serial, fast)
            for (const auto& request : batch) {
                board_sizes.push_back(request.board_size);
                num_planes_list.push_back(request.num_feature_planes);
            }

            // T019: Extract features in parallel using OpenMP
            // Only parallelize if batch size > 8 to avoid threading overhead
            int batch_size_int = static_cast<int>(batch.size());
            bool should_parallelize = (batch_size_int > 8);

            #ifdef _OPENMP
            if (should_parallelize) {
                PROFILE_SCOPE(ProfileMetric::CoordinatorFeatureExtractionOMP);

                int omp_threads = 0;
                #pragma omp parallel
                {
                    #pragma omp single
                    {
                        omp_threads = omp_get_num_threads();
                    }

                    // Distribute work with static scheduling
                    #pragma omp for schedule(static)
                    for (int i = 0; i < batch_size_int; ++i) {
                        int num_planes = batch[i].num_feature_planes;
                        int board_size = batch[i].board_size;
                        size_t features_size = num_planes * board_size * board_size;

                        features_batch[i].resize(features_size);
                        batch[i].state->extract_features_to_buffer(features_batch[i].data());
                    }
                }

                // Track OpenMP thread count
                PROFILE_GAUGE(ProfileMetric::CoordinatorOMPThreadCount, omp_threads);
            } else {
            #endif
                PROFILE_SCOPE(ProfileMetric::CoordinatorFeatureExtractionSerial);
                // Serial extraction (batch_size <= 8 or OpenMP disabled)
                for (int i = 0; i < batch_size_int; ++i) {
                    int num_planes = batch[i].num_feature_planes;
                    int board_size = batch[i].board_size;
                    size_t features_size = num_planes * board_size * board_size;

                    features_batch[i].resize(features_size);
                    batch[i].state->extract_features_to_buffer(features_batch[i].data());
                }
            #ifdef _OPENMP
            }
            #endif
        }

        // Phase 3: Call Python for GPU inference (GIL ACQUIRED ONCE)
        // This is the only GIL crossing in the entire batch
        // Uses pre-extracted features (no state cloning overhead!)
        std::vector<std::pair<std::vector<float>, float>> inference_results;
        bool had_error = false;
        {
            PROFILE_SCOPE(ProfileMetric::CoordinatorPythonCallback);
            try {
                // Call virtual method (PyBatchInferenceCallback overrides this)
                inference_results = callback_->batch_inference_features(
                    features_batch, board_sizes, num_planes_list
                );
            } catch (const std::exception& e) {
                had_error = true;
                std::cerr << "Batch inference failed: " << e.what() << std::endl;
            }
        }

        // Phase 4: Submit results to queue
        if (!had_error && inference_results.size() == batch.size()) {
            PROFILE_SCOPE(ProfileMetric::CoordinatorResultSubmit);

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
        {
            PROFILE_SCOPE(ProfileMetric::CoordinatorFallbackPath);

            if (!had_error && inference_results.size() != batch.size()) {
                std::cerr << "Batch inference returned mismatched result count (" << inference_results.size()
                          << " vs " << batch.size() << "), using uniform fallback\n";
            }

            auto fallback_results = build_fallback_results(batch);

            {
                PROFILE_SCOPE(ProfileMetric::CoordinatorResultSubmit);
                queue_->submit_results(fallback_results);
            }
        }
    }
}

} // namespace mcts
