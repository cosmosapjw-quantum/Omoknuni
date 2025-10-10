/**
 * @file profiling_test.cpp
 * @brief Test executable for MCTS profiling system
 */

#include "enhanced_profiler.hpp"
#include "enhanced_metrics.hpp"
#include "thread_metrics.hpp"
#include "../tree.hpp"
#include "../selection.hpp"
#include "../backup.hpp"
#include "../simulation_runner.hpp"
#include <thread>
#include <chrono>
#include <random>
#include <iostream>
#include <vector>

using namespace mcts::profiling;

// Simulate MCTS operations
void simulate_selection(int iterations) {
    PROFILE_SCOPE(ProfileMetric::SelectionTotal);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i = 0; i < iterations; ++i) {
        PROFILE_SCOPE(ProfileMetric::SelectionPUCT);

        // Simulate PUCT computation
        volatile double result = 0.0;
        for (int j = 0; j < 100; ++j) {
            result += std::sqrt(dis(gen)) * 1.25;
        }

        // Simulate busy edge detection
        if (dis(gen) < 0.1) {
            PROFILE_COUNTER(ProfileMetric::SelectionBusyEdgeSkip, 1);
        }

        // Update depth gauge
        PROFILE_GAUGE(ProfileMetric::SelectionDepth, i % 20);
    }
}

void simulate_expansion(int iterations) {
    PROFILE_SCOPE(ProfileMetric::ExpansionTotal);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(10, 100);

    for (int i = 0; i < iterations; ++i) {
        // Simulate neural network wait
        {
            PROFILE_SCOPE(ProfileMetric::ExpansionNeuralNetWait);
            std::this_thread::sleep_for(std::chrono::microseconds(dis(gen)));
        }

        // Simulate node allocation
        {
            PROFILE_SCOPE(ProfileMetric::ExpansionNodeAllocation);
            volatile int nodes = dis(gen);
            for (int j = 0; j < nodes; ++j) {
                // Simulate allocation
            }
        }

        // Track conflicts
        if (dis(gen) < 20) {
            PROFILE_COUNTER(ProfileMetric::ExpansionConflicts, 1);
        }
    }
}

void simulate_backup(int iterations) {
    PROFILE_SCOPE(ProfileMetric::BackupTotal);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 10);

    for (int i = 0; i < iterations; ++i) {
        // Simulate value update
        {
            PROFILE_SCOPE(ProfileMetric::BackupValueUpdate);
            volatile double value = 0.0;
            for (int j = 0; j < dis(gen); ++j) {
                value += 0.1;
            }
        }

        // Track CAS retries
        int retries = dis(gen) - 1;
        if (retries > 0) {
            PROFILE_COUNTER(ProfileMetric::BackupCASRetries, retries);
        }
    }
}

void worker_thread(int thread_id, int iterations) {
    std::cout << "Thread " << thread_id << " starting..." << std::endl;

    // Mix of operations
    for (int i = 0; i < iterations; ++i) {
        simulate_selection(10);
        simulate_expansion(5);
        simulate_backup(20);

        // Simulate queue operations
        PROFILE_COUNTER(ProfileMetric::QueueSubmitTotal, 1);

        if (i % 10 == 0) {
            PROFILE_GAUGE(ProfileMetric::QueueCollectBatchSize, thread_id * 10 + i);
        }
    }

    std::cout << "Thread " << thread_id << " completed." << std::endl;
}

void test_thread_contention() {
    std::cout << "\n=== Testing Thread Contention ===" << std::endl;

    EnhancedProfiler& profiler = EnhancedProfiler::instance();
    profiler.start_session("thread_contention_test");

    // Test with different thread counts
    std::vector<int> thread_counts = {1, 2, 4, 8};

    for (int num_threads : thread_counts) {
        std::cout << "\nTesting with " << num_threads << " threads..." << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<std::thread> threads;
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(worker_thread, i, 100);
        }

        for (auto& t : threads) {
            t.join();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Completed in " << duration.count() << "ms" << std::endl;
    }

    profiler.stop_session();
    profiler.print_summary();
}

void test_memory_patterns() {
    std::cout << "\n=== Testing Memory Access Patterns ===" << std::endl;

    EnhancedProfiler& profiler = EnhancedProfiler::instance();
    profiler.start_session("memory_patterns_test");

    const size_t ARRAY_SIZE = 1024 * 1024;  // 1M elements
    std::vector<double> data(ARRAY_SIZE, 0.0);

    // Sequential access pattern
    {
        PROFILE_SCOPE(ProfileMetric::MemoryCacheLineLoads);
        for (size_t i = 0; i < ARRAY_SIZE; ++i) {
            data[i] = i * 0.1;
        }
    }

    // Random access pattern (cache unfriendly)
    {
        PROFILE_SCOPE(ProfileMetric::MemoryCacheLineEvictions);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, ARRAY_SIZE - 1);

        for (size_t i = 0; i < ARRAY_SIZE; ++i) {
            data[dis(gen)] = i * 0.1;
        }
    }

    // Strided access (potential false sharing)
    {
        PROFILE_SCOPE(ProfileMetric::MemoryFalseSharing);
        const size_t STRIDE = 64;  // Cache line size
        for (size_t i = 0; i < ARRAY_SIZE; i += STRIDE) {
            data[i] = i * 0.1;
        }
    }

    profiler.stop_session();
    profiler.print_summary();
}

void test_pipeline_stages() {
    std::cout << "\n=== Testing Pipeline Stages ===" << std::endl;

    EnhancedProfiler& profiler = EnhancedProfiler::instance();
    profiler.start_session("pipeline_test");

    const int ITERATIONS = 1000;

    for (int i = 0; i < ITERATIONS; ++i) {
        // Full pipeline
        {
            PROFILE_SCOPE(ProfileMetric::PipelineE2ELatency);

            // Selection stage
            {
                PROFILE_SCOPE(ProfileMetric::PipelineStageSelection);
                simulate_selection(1);
            }

            // Expansion stage
            {
                PROFILE_SCOPE(ProfileMetric::PipelineStageExpansion);
                simulate_expansion(1);
            }

            // Inference stage
            {
                PROFILE_SCOPE(ProfileMetric::PipelineStageInference);
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }

            // Backup stage
            {
                PROFILE_SCOPE(ProfileMetric::PipelineStageBackup);
                simulate_backup(1);
            }
        }

        // Track pipeline metrics
        PROFILE_GAUGE(ProfileMetric::PipelineQueueDepth, i % 100);
        PROFILE_GAUGE(ProfileMetric::PipelineUtilization, 75 + (i % 25));
    }

    // Calculate throughput
    PROFILE_GAUGE(ProfileMetric::PipelineThroughput, ITERATIONS);

    profiler.stop_session();
    profiler.print_summary();
}

int main(int argc, char** argv) {
    std::cout << "MCTS Profiling Test Suite" << std::endl;
    std::cout << "=========================" << std::endl;

    // Enable profiling
    EnhancedProfiler& profiler = EnhancedProfiler::instance();
    profiler.set_enabled(true);
    profiler.set_level(ProfileLevel::Full);

    // Run tests
    test_thread_contention();
    test_memory_patterns();
    test_pipeline_stages();

    // Export reports
    std::cout << "\n=== Exporting Reports ===" << std::endl;
    profiler.export_json("mcts_profile.json");
    profiler.export_chrome_trace("mcts_trace.json");
    profiler.export_markdown("mcts_profile.md");

    std::cout << "\nAll tests completed successfully!" << std::endl;
    std::cout << "Reports saved to:" << std::endl;
    std::cout << "  - mcts_profile.json (detailed metrics)" << std::endl;
    std::cout << "  - mcts_trace.json (Chrome trace viewer)" << std::endl;
    std::cout << "  - mcts_profile.md (summary report)" << std::endl;

    return 0;
}