// cpp_extensions/mcts/state_pool.cpp
// T018b: ThreadLocalStatePool Implementation

#include "state_pool.hpp"
#include "dlpack_bridge.hpp"  // For GameType definition
#include "../games/gomoku/gomoku_state.h"
#include "../games/chess/chess_state.h"
#include "../games/go/go_state.h"
#include <thread>
#include <stdexcept>

namespace mcts {

using alphazero::games::gomoku::GomokuState;
using alphazero::games::chess::ChessState;
using alphazero::games::go::GoState;

ThreadLocalStatePool::ThreadLocalStatePool(GameType game_type, size_t pool_size)
    : pool_size_(pool_size), next_free_(0) {

    if (pool_size == 0) {
        throw std::invalid_argument("ThreadLocalStatePool: pool_size must be > 0");
    }

    pool_.reserve(pool_size);

    // Pre-allocate all states based on game type
    for (size_t i = 0; i < pool_size; ++i) {
        switch (game_type) {
            case GameType::GOMOKU:
                pool_.emplace_back(std::make_unique<GomokuState>());
                break;
            case GameType::CHESS:
                pool_.emplace_back(std::make_unique<ChessState>());
                break;
            case GameType::GO:
                pool_.emplace_back(std::make_unique<GoState>());
                break;
            default:
                throw std::invalid_argument("ThreadLocalStatePool: unsupported game type");
        }
    }
}

IGameState* ThreadLocalStatePool::acquire() {
    // Lock-free atomic increment (ring buffer index)
    // Performance: ~5ns (single fetch_add instruction)
    total_acquires_.fetch_add(1, std::memory_order_relaxed);
    size_t idx = next_free_.fetch_add(1, std::memory_order_relaxed);

    // Wrap around pool (ring buffer)
    size_t pool_idx = idx % pool_size_;

    // Update peak usage tracking
    // Usage = (total_acquires / pool_size) + 1
    // If peak_usage > 1, pool is being wrapped (concurrent usage > pool_size)
    size_t current_usage = (idx / pool_size_) + 1;
    size_t peak = peak_usage_.load(std::memory_order_relaxed);

    // CAS loop to update peak (rare, low contention)
    while (current_usage > peak) {
        if (peak_usage_.compare_exchange_weak(peak, current_usage,
                                              std::memory_order_relaxed,
                                              std::memory_order_relaxed)) {
            break;
        }
    }

    return pool_[pool_idx].get();
}

void ThreadLocalStatePool::release(IGameState* state) {
    // No-op: Ring buffer automatically reuses states
    // Included for API symmetry and future optimization potential
    // Performance: ~2ns (single fetch_add for stats tracking)
    total_releases_.fetch_add(1, std::memory_order_relaxed);

    // Note: 'state' parameter unused - kept for API compatibility
    // In ring buffer design, states are reused implicitly on next acquire()
    (void)state;  // Suppress unused parameter warning
}

ThreadLocalStatePool::Stats ThreadLocalStatePool::get_stats() const {
    return Stats{
        .total_acquires = total_acquires_.load(std::memory_order_relaxed),
        .total_releases = total_releases_.load(std::memory_order_relaxed),
        .peak_usage = peak_usage_.load(std::memory_order_relaxed),
        .pool_size = pool_size_
    };
}

void ThreadLocalStatePool::reset_stats() {
    total_acquires_.store(0, std::memory_order_relaxed);
    total_releases_.store(0, std::memory_order_relaxed);
    peak_usage_.store(0, std::memory_order_relaxed);
}

// Thread-local singleton accessor
ThreadLocalStatePool* get_thread_state_pool(GameType game_type, size_t pool_size) {
    // Thread-local storage: each thread has its own pool (no contention)
    // Lazy initialization: created on first call per thread
    // Lifetime: persists for thread duration
    thread_local std::unique_ptr<ThreadLocalStatePool> pool;

    if (!pool) {
        pool = std::make_unique<ThreadLocalStatePool>(game_type, pool_size);
    }

    return pool.get();
}

} // namespace mcts
