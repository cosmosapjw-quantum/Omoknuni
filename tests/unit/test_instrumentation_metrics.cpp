#include <gtest/gtest.h>

#include "mcts/instrumentation.hpp"

namespace {

using namespace mcts;

class InstrumentationGuard {
public:
    InstrumentationGuard() : previous_state_(Instrumentation::instance().is_enabled()) {
        Instrumentation::instance().set_enabled(true);
        Instrumentation::instance().reset();
    }

    ~InstrumentationGuard() {
        Instrumentation::instance().reset();
        Instrumentation::instance().set_enabled(previous_state_);
    }

private:
    bool previous_state_;
};

TEST(InstrumentationMetricsTest, RecordsScopedMetricDurations) {
    InstrumentationGuard guard;
    {
        ScopedMetric metric(InstrumentationMetric::Selection);
    }

    const auto snapshot = Instrumentation::instance().snapshot();
    ASSERT_FALSE(snapshot.empty());
    const auto it = snapshot.find(InstrumentationMetric::Selection);
    ASSERT_NE(it, snapshot.end());
    EXPECT_EQ(it->second.call_count, 1u);
    EXPECT_GT(it->second.total_elapsed_ns, 0u);
}

TEST(InstrumentationMetricsTest, IncrementCounterTracksCounts) {
    InstrumentationGuard guard;
    Instrumentation::instance().increment_counter(InstrumentationMetric::VirtualLossApply, 3);
    const auto snapshot = Instrumentation::instance().snapshot();
    const auto it = snapshot.find(InstrumentationMetric::VirtualLossApply);
    ASSERT_NE(it, snapshot.end());
    EXPECT_EQ(it->second.call_count, 3u);
    EXPECT_EQ(it->second.total_elapsed_ns, 0u);
}

}  // namespace
