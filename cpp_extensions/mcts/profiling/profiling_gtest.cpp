/**
 * @file profiling_gtest.cpp
 * @brief Google Test suite for MCTS profiling system
 */

#include <gtest/gtest.h>
#include "enhanced_profiler.hpp"
#include "enhanced_metrics.hpp"
#include "thread_metrics.hpp"
#include <thread>
#include <chrono>
#include <random>
#include <atomic>
#include <vector>

using namespace mcts::profiling;

class ProfilingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset profiler before each test
        EnhancedProfiler::instance().reset_metrics();
        EnhancedProfiler::instance().set_enabled(true);
        EnhancedProfiler::instance().set_level(ProfileLevel::Full);
    }

    void TearDown() override {
        // Disable profiling after each test
        EnhancedProfiler::instance().set_enabled(false);
    }
};

// Test basic profiling functionality
TEST_F(ProfilingTest, BasicScopedProfiling) {
    auto& profiler = EnhancedProfiler::instance();
    profiler.start_session("test_basic");

    // Profile a simple operation
    {
        PROFILE_SCOPE(ProfileMetric::SelectionTotal);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    profiler.stop_session();

    // Generate report and verify
    auto report = profiler.generate_report();
    EXPECT_EQ(report.session_name, "test_basic");
    EXPECT_GT(report.duration_ns, 0);
    EXPECT_GT(report.timing_stats.size(), 0);

    // Check that SelectionTotal was recorded
    EXPECT_TRUE(report.timing_stats.count(ProfileMetric::SelectionTotal) > 0);
    auto& stats = report.timing_stats.at(ProfileMetric::SelectionTotal);
    EXPECT_GT(stats.total, 10000000);  // At least 10ms in nanoseconds
}

// Test counter metrics
TEST_F(ProfilingTest, CounterMetrics) {
    auto& profiler = EnhancedProfiler::instance();
    profiler.start_session("test_counters");

    // Increment counters
    for (int i = 0; i < 100; ++i) {
        PROFILE_COUNTER(ProfileMetric::SelectionBusyEdgeSkip, 1);
        if (i % 10 == 0) {
            PROFILE_COUNTER(ProfileMetric::ExpansionConflicts, 2);
        }
    }

    profiler.stop_session();
    auto report = profiler.generate_report();

    // Verify counter values
    EXPECT_EQ(report.counters[ProfileMetric::SelectionBusyEdgeSkip], 100);
    EXPECT_EQ(report.counters[ProfileMetric::ExpansionConflicts], 20);
}

// Test gauge metrics
TEST_F(ProfilingTest, GaugeMetrics) {
    auto& profiler = EnhancedProfiler::instance();
    profiler.start_session("test_gauges");

    // Update gauges
    PROFILE_GAUGE(ProfileMetric::SelectionDepth, 5);
    PROFILE_GAUGE(ProfileMetric::SelectionDepth, 10);
    PROFILE_GAUGE(ProfileMetric::SelectionDepth, 3);

    PROFILE_GAUGE(ProfileMetric::QueueCollectBatchSize, 32);
    PROFILE_GAUGE(ProfileMetric::QueueCollectBatchSize, 64);

    profiler.stop_session();
    auto report = profiler.generate_report();

    // Gauges should keep the maximum value
    EXPECT_EQ(report.gauges[ProfileMetric::SelectionDepth], 10);
    EXPECT_EQ(report.gauges[ProfileMetric::QueueCollectBatchSize], 64);
}

// Test thread-local metrics
TEST_F(ProfilingTest, ThreadLocalMetrics) {
    auto& profiler = EnhancedProfiler::instance();
    profiler.start_session("test_thread_local");

    std::atomic<int> counter{0};
    std::atomic<int> threads_done{0};
    std::vector<std::thread> threads;

    // Launch multiple threads
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&counter, &threads_done, t]() {
            for (int i = 0; i < 10; ++i) {
                PROFILE_SCOPE(ProfileMetric::SelectionTotal);
                PROFILE_COUNTER(ProfileMetric::SelectionRetries, 1);
                counter++;
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            threads_done++;
        });
    }

    // Wait for all threads to finish their profiling work
    while (threads_done.load() < 4) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    // Now stop session to capture metrics while thread-local storage still valid
    profiler.stop_session();

    // Wait for threads to fully exit
    for (auto& t : threads) {
        t.join();
    }

    auto report = profiler.generate_report();

    // Verify all thread operations were recorded
    EXPECT_EQ(counter.load(), 40);  // 4 threads * 10 iterations
    EXPECT_EQ(report.counters[ProfileMetric::SelectionRetries], 40);
}

// Test nested profiling scopes
TEST_F(ProfilingTest, NestedScopes) {
    auto& profiler = EnhancedProfiler::instance();
    profiler.start_session("test_nested");

    {
        PROFILE_SCOPE(ProfileMetric::SelectionTotal);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));

        {
            PROFILE_SCOPE(ProfileMetric::SelectionPUCT);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }

        {
            PROFILE_SCOPE(ProfileMetric::SelectionTreeTraversal);
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
        }
    }

    profiler.stop_session();
    auto report = profiler.generate_report();

    // Verify nested scopes were recorded
    EXPECT_TRUE(report.timing_stats.count(ProfileMetric::SelectionTotal) > 0);
    EXPECT_TRUE(report.timing_stats.count(ProfileMetric::SelectionPUCT) > 0);
    EXPECT_TRUE(report.timing_stats.count(ProfileMetric::SelectionTreeTraversal) > 0);

    // Parent scope should have longer duration than children
    auto total_time = report.timing_stats[ProfileMetric::SelectionTotal].total;
    auto puct_time = report.timing_stats[ProfileMetric::SelectionPUCT].total;
    auto traversal_time = report.timing_stats[ProfileMetric::SelectionTreeTraversal].total;

    EXPECT_GT(total_time, puct_time);
    EXPECT_GT(total_time, traversal_time);
}

// Test ring buffer overflow handling
TEST_F(ProfilingTest, RingBufferOverflow) {
    auto& metrics = ThreadMetricsStorage::instance().get_thread_metrics();

    // Fill ring buffer beyond capacity
    const size_t iterations = TimingRingBuffer<4096>::kCapacity + 100;

    for (size_t i = 0; i < iterations; ++i) {
        metrics.record_timing(
            ProfileMetric::SelectionTotal,
            i * 1000000,  // start_ns
            1000000       // duration_ns
        );
    }

    // Buffer should handle overflow gracefully
    auto& buffer = metrics.get_timing_buffer();
    EXPECT_GT(buffer.overflows(), 0);
    EXPECT_LE(buffer.size(), TimingRingBuffer<4096>::kCapacity);
}

// Test statistical analysis
TEST_F(ProfilingTest, StatisticalAnalysis) {
    StatisticalAnalyzer analyzer;

    // Create sample data with known distribution
    std::vector<uint64_t> samples;
    for (int i = 1; i <= 100; ++i) {
        samples.push_back(i * 1000);  // 1ms to 100ms
    }

    auto stats = analyzer.compute(samples);

    // Verify statistical calculations
    EXPECT_EQ(stats.count, 100);
    EXPECT_EQ(stats.min, 1000.0);
    EXPECT_EQ(stats.max, 100000.0);
    EXPECT_NEAR(stats.mean, 50500.0, 1.0);  // Average of 1-100 * 1000
    EXPECT_NEAR(stats.p50, 50000.0, 1000.0);  // Median
    EXPECT_GT(stats.stddev, 0);  // Non-zero standard deviation
}

// Test bottleneck detection
TEST_F(ProfilingTest, BottleneckDetection) {
    auto& profiler = EnhancedProfiler::instance();
    profiler.start_session("test_bottleneck");

    // Simulate operations with different durations
    for (int i = 0; i < 100; ++i) {
        {
            PROFILE_SCOPE(ProfileMetric::SelectionTotal);
            std::this_thread::sleep_for(std::chrono::microseconds(1000));  // 1ms - bottleneck
        }

        {
            PROFILE_SCOPE(ProfileMetric::ExpansionTotal);
            std::this_thread::sleep_for(std::chrono::microseconds(100));   // 0.1ms
        }

        {
            PROFILE_SCOPE(ProfileMetric::BackupTotal);
            std::this_thread::sleep_for(std::chrono::microseconds(50));    // 0.05ms
        }
    }

    profiler.stop_session();
    auto report = profiler.generate_report();

    // SelectionTotal should be identified as bottleneck
    ASSERT_FALSE(report.bottlenecks.empty());

    // Find selection bottleneck
    auto it = std::find_if(report.bottlenecks.begin(), report.bottlenecks.end(),
        [](const Bottleneck& b) {
            return b.metric == ProfileMetric::SelectionTotal;
        });

    EXPECT_NE(it, report.bottlenecks.end());
    if (it != report.bottlenecks.end()) {
        EXPECT_GT(it->severity, 50.0);  // Should be high severity
    }
}

// Test export functionality
TEST_F(ProfilingTest, ExportFormats) {
    auto& profiler = EnhancedProfiler::instance();
    profiler.start_session("test_export");

    // Generate some data
    for (int i = 0; i < 10; ++i) {
        PROFILE_SCOPE(ProfileMetric::SelectionTotal);
        PROFILE_COUNTER(ProfileMetric::SelectionRetries, i);
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    profiler.stop_session();
    auto report = profiler.generate_report();

    // Test JSON export
    std::string json = report.to_json();
    EXPECT_FALSE(json.empty());
    EXPECT_NE(json.find("session_name"), std::string::npos);
    EXPECT_NE(json.find("timing_stats"), std::string::npos);

    // Test Chrome trace export
    std::string trace = report.to_chrome_trace();
    EXPECT_FALSE(trace.empty());
    EXPECT_NE(trace.find("ph"), std::string::npos);  // Chrome trace event phase

    // Test Markdown export
    std::string markdown = report.to_markdown();
    EXPECT_FALSE(markdown.empty());
    EXPECT_NE(markdown.find("# Profiling Report"), std::string::npos);
}

// Test parallel efficiency calculation
TEST_F(ProfilingTest, ParallelEfficiency) {
    auto& profiler = EnhancedProfiler::instance();
    profiler.start_session("test_parallel");

    std::vector<std::thread> threads;
    std::atomic<int> active_count{0};
    std::atomic<int> threads_done{0};

    // Simulate parallel work with some contention
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&active_count, &threads_done]() {
            for (int i = 0; i < 100; ++i) {
                PROFILE_SCOPE(ProfileMetric::ThreadActiveTime);
                active_count++;

                // Simulate work
                volatile double result = 0.0;
                for (int j = 0; j < 1000; ++j) {
                    result += std::sqrt(j);
                }

                // Simulate contention
                if (active_count.load() > 2) {
                    PROFILE_SCOPE(ProfileMetric::ThreadWaitTime);
                    std::this_thread::sleep_for(std::chrono::microseconds(10));
                }

                active_count--;
            }
            threads_done++;
        });
    }

    // Wait for all threads to finish their profiling work
    while (threads_done.load() < 4) {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    // Now stop session to capture metrics while thread-local storage still valid
    profiler.stop_session();

    for (auto& t : threads) {
        t.join();
    }

    auto report = profiler.generate_report();

    // Should have recorded both active and wait times
    EXPECT_TRUE(report.timing_stats.count(ProfileMetric::ThreadActiveTime) > 0);
    EXPECT_TRUE(report.timing_stats.count(ProfileMetric::ThreadWaitTime) > 0);

    // Calculate parallel efficiency
    if (report.gauges.count(ProfileMetric::ParallelEfficiency)) {
        auto efficiency = report.gauges[ProfileMetric::ParallelEfficiency];
        EXPECT_GT(efficiency, 0);
        EXPECT_LE(efficiency, 100);
    }
}

// Test profiling overhead
TEST_F(ProfilingTest, ProfilingOverhead) {
    const int iterations = 10000;

    // Measure baseline (no profiling) - do meaningful work
    EnhancedProfiler::instance().set_enabled(false);
    auto start_no_prof = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        // Do substantial work to make profiling overhead relatively small
        volatile double result = 0.0;
        for (int j = 0; j < 1000; ++j) {
            result += std::sqrt(i + j);
        }
        (void)result;
    }

    auto duration_no_prof = std::chrono::high_resolution_clock::now() - start_no_prof;

    // Measure with profiling
    EnhancedProfiler::instance().set_enabled(true);
    auto start_with_prof = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        PROFILE_SCOPE(ProfileMetric::SelectionTotal);
        // Same work as baseline
        volatile double result = 0.0;
        for (int j = 0; j < 1000; ++j) {
            result += std::sqrt(i + j);
        }
        (void)result;
    }

    auto duration_with_prof = std::chrono::high_resolution_clock::now() - start_with_prof;

    // Calculate overhead
    double overhead_ratio = static_cast<double>(duration_with_prof.count()) /
                           duration_no_prof.count();

    // Profiling overhead should be reasonable (<20% for real workloads)
    // Note: For tiny operations, overhead appears large, but for real MCTS operations
    // that do substantial work, overhead is minimal
    EXPECT_LT(overhead_ratio, 1.20) << "Profiling overhead too high: "
                                     << (overhead_ratio - 1.0) * 100 << "%";

    std::cout << "Profiling overhead: "
              << (overhead_ratio - 1.0) * 100 << "%" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}