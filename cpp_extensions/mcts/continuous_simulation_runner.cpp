/**
 * @file continuous_simulation_runner.cpp
 * @brief Implementation of continuous simulation runner
 */

#include "continuous_simulation_runner.hpp"
#include "instrumentation.hpp"
#include "thread_affinity.hpp"
#include "../utils/igamestate.h"
#include <algorithm>
#include <thread>
#include <chrono>
#include <random>

namespace mcts {

ContinuousSimulationRunner::ContinuousSimulationRunner(MCTSTree& tree,
                                                         PUCTSelector& selector,
                                                         BackupManager& backup,
                                                         VirtualLossManager& virtual_loss)
    : SimulationRunner(tree, selector, backup, virtual_loss) {
}

int ContinuousSimulationRunner::run_continuous(IGameState& root_state,
                                                 NodeIndex root_index,
                                                 AsyncInferenceQueue& queue,
                                                 int num_simulations) {
    int completed = 0;
    int submitted = 0;

    // Clear pending buffer
    for (auto& slot : pending_buffer_) {
        slot.occupied.store(false, std::memory_order_relaxed);
    }
    pending_count_.store(0, std::memory_order_relaxed);

    // THREAD AFFINITY: Pin thread to optimal CPU core for cache locality
    // Expected impact: 1.15× speedup from reduced cross-CCD traffic
    static thread_local ThreadAffinityManager affinity_mgr;
    static thread_local int thread_id = -1;
    static thread_local bool affinity_set = false;

    if (!affinity_set) {
        // Determine thread ID using std::hash of thread::id
        thread_id = static_cast<int>(
            std::hash<std::thread::id>{}(std::this_thread::get_id()) % 24
        );

        // Set affinity (assumes reasonable thread count for hardware)
        int recommended_threads = affinity_mgr.get_recommended_thread_count();
        affinity_mgr.set_thread_affinity(thread_id, recommended_threads);
        affinity_set = true;
    }

    // PRE-EXPAND ROOT: Eliminates N-1 thread idle problem where threads
    // race to expand root but only one succeeds. By expanding synchronously
    // before threading, all threads can immediately start productive work.
    // Expected impact: 2× speedup (eliminates initial serialization bottleneck)
    ensure_root_expanded(root_state, root_index, queue);

    auto release_virtual_loss = [this](const std::vector<NodeIndex>& path) {
        if (path.size() <= 1) {
            return;
        }
        for (size_t i = 1; i < path.size(); ++i) {
            virtual_loss_.remove_virtual_loss(path[i]);
        }
    };

    // Continuous loop until quota reached
    while (completed < num_simulations) {
        bool waiting_for_leaf = false;

        // Phase 1: Select to leaf and submit inference (NON-BLOCKING)
        if (submitted < num_simulations) {
            // Clone state for this simulation
            std::unique_ptr<IGameState> current_state = root_state.clone();
            if (!current_state) {
                continue;  // Skip on clone failure
            }

            // Clear and reuse path buffer
            path_buffer_.clear();

            // Select to leaf
            NodeIndex leaf = select_leaf(root_index, *current_state, path_buffer_);

            // Check if terminal
            if (current_state->isTerminal()) {
                // Terminal node - backup immediately, no inference needed
                float value = get_terminal_value(*current_state);
                std::reverse(path_buffer_.begin(), path_buffer_.end());
                backup_value(path_buffer_, value);
                completed++;
                submitted++;
                continue;
            }

            // Ensure only one in-flight expansion per node
            bool submission_ready = true;
            if (!tree_.atomic_try_mark_expanding(leaf)) {
                // Track expansion conflicts (busy-edge prevented duplicate expansion)
                Instrumentation::instance().increment_counter(InstrumentationMetric::ExpansionConflict);
                release_virtual_loss(path_buffer_);
                waiting_for_leaf = true;
                submission_ready = false;
            }

            std::unique_ptr<IGameState> queue_state;
            if (submission_ready) {
                // Clone state for queue submission (queue takes ownership)
                queue_state = current_state->clone();
                if (!queue_state) {
                    tree_.clear_expanding_flag(leaf);
                    release_virtual_loss(path_buffer_);
                    waiting_for_leaf = true;
                    submission_ready = false;
                }
            }

            if (submission_ready) {
                constexpr std::size_t kMaxInFlight = 4096;
                std::size_t backoff_loops = 0;
                size_t pending = pending_count_.load(std::memory_order_relaxed);
                while (queue.pending_count() >= kMaxInFlight || pending >= kMaxInFlight) {
                    waiting_for_leaf = true;
                    int flushed = process_completed_results(queue);
                    if (flushed == 0) {
                        std::this_thread::sleep_for(std::chrono::microseconds(100));
                    }
                    if (++backoff_loops > 1024) {
                        break;  // Prevent unbounded waiting
                    }
                    pending = pending_count_.load(std::memory_order_relaxed);
                }

                // Submit request (NON-BLOCKING)
                uint64_t request_id = queue.submit_request(
                    std::move(queue_state),
                    leaf,
                    path_buffer_
                );

                // Track pending expansion using ring buffer (O(1) direct indexing)
                size_t slot_index = request_id % PENDING_BUFFER_CAPACITY;
                PendingSlot& slot = pending_buffer_[slot_index];

                // Store request data
                slot.request_id = request_id;
                slot.data.leaf_node = leaf;
                slot.data.path = path_buffer_;  // Copy path
                slot.data.state = std::move(current_state);  // Keep state for expansion

                // Mark slot as occupied (release to ensure data is visible)
                slot.occupied.store(true, std::memory_order_release);
                pending_count_.fetch_add(1, std::memory_order_relaxed);

                submitted++;
            }
        }

        // Phase 2: Process completed results (NON-BLOCKING)
        int processed = process_completed_results(queue);
        completed += processed;

        // Yield briefly if no results available to avoid busy-waiting
        if (processed == 0) {
            bool all_submitted = submitted >= num_simulations;
            if (all_submitted || waiting_for_leaf) {
                auto sleep_duration = waiting_for_leaf ? std::chrono::microseconds(50)
                                                       : std::chrono::microseconds(100);
                std::this_thread::sleep_for(sleep_duration);
            }
        }
    }

    // Clear pending buffer
    for (auto& slot : pending_buffer_) {
        slot.occupied.store(false, std::memory_order_relaxed);
    }
    pending_count_.store(0, std::memory_order_relaxed);

    return completed;
}

int ContinuousSimulationRunner::process_completed_results(AsyncInferenceQueue& queue) {
    ScopedMetric metric(InstrumentationMetric::QueueProcessResults);
    int processed = 0;

    // Check each pending slot for completed results
    // This avoids the race where consume_ready_results() steals other threads' results
    for (size_t i = 0; i < PENDING_BUFFER_CAPACITY; ++i) {
        PendingSlot& slot = pending_buffer_[i];

        // Check if slot is occupied
        if (!slot.occupied.load(std::memory_order_acquire)) {
            continue;
        }

        // Try to get result for this specific request
        auto result_opt = queue.try_get_result(slot.request_id);
        if (!result_opt.has_value()) {
            continue;  // Result not ready yet
        }

        auto& result = result_opt.value();

        // Extract pending data
        PendingExpansion pending = std::move(slot.data);

        // Mark slot as free
        slot.occupied.store(false, std::memory_order_release);
        pending_count_.fetch_sub(1, std::memory_order_relaxed);

        if (pending.state && expand_node_with_result(
                pending.leaf_node,
                *pending.state,
                result.policy,
                result.value)) {
            std::vector<NodeIndex> path = pending.path;
            std::reverse(path.begin(), path.end());
            backup_value(path, result.value);
        } else {
            std::vector<NodeIndex> path = pending.path;
            std::reverse(path.begin(), path.end());
            backup_value(path, result.value);
        }

        tree_.clear_expanding_flag(pending.leaf_node);
        processed++;
    }

    return processed;
}

bool ContinuousSimulationRunner::expand_node_with_result(
    NodeIndex leaf,
    const IGameState& state,
    const std::vector<float>& policy,
    float value) {

    // ✅ CRITICAL FIX: Check if already expanded (but don't claim yet)
    // We'll claim after allocating children to avoid race where threads see
    // expanded=true but num_children=0
    NodeFlags flags = tree_.get_flags(leaf);
    if (flags.is_expanded()) {
        return false;  // Already expanded by another thread
    }

    // Get legal moves
    std::vector<int> legal_moves = state.getLegalMoves();
    if (legal_moves.empty()) {
        return false;
    }

    // Validate policy size
    int action_space_size = state.getActionSpaceSize();
    if (static_cast<int>(policy.size()) != action_space_size) {
        return false;
    }

    // Mask and normalize policy
    float policy_sum = 0.0f;
    thread_local std::vector<float> masked_policy_buffer;
    masked_policy_buffer.resize(legal_moves.size());
    auto& masked_policy = masked_policy_buffer;

    for (size_t i = 0; i < legal_moves.size(); ++i) {
        int move = legal_moves[i];
        if (move >= 0 && move < action_space_size) {
            masked_policy[i] = policy[move];
            policy_sum += policy[move];
        } else {
            masked_policy[i] = 0.0f;
        }
    }

    // Normalize
    if (policy_sum > 0.0f) {
        const float inv_sum = 1.0f / policy_sum;
        for (float& p : masked_policy) {
            p *= inv_sum;
        }
    } else {
        const float uniform_prob = 1.0f / static_cast<float>(legal_moves.size());
        for (float& p : masked_policy) {
            p = uniform_prob;
        }
    }

    // Allocate children
    uint16_t num_children = static_cast<uint16_t>(legal_moves.size());
    NodeIndex first_child = tree_.allocate_nodes(num_children);

    if (first_child == NULL_NODE_INDEX) {
        return false;  // Tree full
    }

    // Initialize children
    for (uint16_t i = 0; i < num_children; ++i) {
        NodeIndex child_idx = first_child + i;

        tree_.set_prior_prob(child_idx, masked_policy[i]);
        tree_.set_move(child_idx, static_cast<uint16_t>(legal_moves[i]));
        tree_.set_parent_index(child_idx, leaf);
        tree_.set_visit_count(child_idx, 0.0f);
        tree_.set_total_value(child_idx, 0.0f);
        tree_.set_virtual_loss(child_idx, 0.0f);

        NodeFlags child_flags;
        child_flags.set_current_player(state.getCurrentPlayer() == 1 ? 1 : 0);
        tree_.set_flags(child_idx, child_flags);
    }

    // Update parent with children info
    tree_.set_first_child_index(leaf, first_child);
    tree_.set_num_children(leaf, num_children);

    // ✅ CRITICAL: Atomically set expanded flag AFTER children are ready
    // This ensures other threads see a fully initialized node
    // If another thread wins this race, we wasted some work but tree stays consistent
    if (!tree_.atomic_try_set_expanded(leaf)) {
        // Another thread set expanded flag - this is very rare but can happen
        // Our allocated children will be orphaned, but tree remains valid
        // This is acceptable vs. the alternative of exposing partially initialized nodes
        return false;
    }

    return true;
}

bool ContinuousSimulationRunner::ensure_root_expanded(IGameState& root_state,
                                                       NodeIndex root_index,
                                                       AsyncInferenceQueue& queue) {
    // Check if root is already expanded
    NodeFlags flags = tree_.get_flags(root_index);
    if (flags.is_expanded()) {
        return false;  // Already expanded, nothing to do
    }

    // Check if we can mark it for expansion atomically
    if (!tree_.atomic_try_mark_expanding(root_index)) {
        // Another thread is already expanding it, wait for completion
        while (!tree_.get_flags(root_index).is_expanded()) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        return false;
    }

    // We won the race - perform synchronous expansion
    try {
        // Submit inference request and wait for result
        std::unique_ptr<IGameState> state_copy = root_state.clone();
        if (!state_copy) {
            tree_.clear_expanding_flag(root_index);
            return false;
        }

        uint64_t request_id = queue.submit_request(std::move(state_copy), root_index, {root_index});

        // Wait for result (synchronous for root expansion only)
        std::optional<InferenceResult> result;
        const auto start_time = std::chrono::steady_clock::now();
        const auto timeout = std::chrono::seconds(5);  // 5 second timeout

        while (!result.has_value()) {
            result = queue.try_get_result(request_id);
            if (!result.has_value()) {
                // Check timeout
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                if (elapsed > timeout) {
                    tree_.clear_expanding_flag(root_index);
                    return false;  // Timeout
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }

        // Expand root with the result
        bool expanded = expand_node_with_result(root_index, root_state, result->policy, result->value);
        tree_.clear_expanding_flag(root_index);

        if (expanded) {
            // Add Dirichlet noise for exploration (AlphaZero approach)
            // Use alpha=0.3 for Go-like games (can be made configurable later)
            add_dirichlet_noise(root_index, 0.3f);
        }

        return expanded;

    } catch (const std::exception& e) {
        tree_.clear_expanding_flag(root_index);
        return false;
    }
}

void ContinuousSimulationRunner::add_dirichlet_noise(NodeIndex root_index, float alpha) {
    std::uint16_t num_children = tree_.get_num_children(root_index);
    if (num_children == 0) {
        return;  // No children to add noise to
    }

    // Sample from Gamma distribution to create Dirichlet noise
    // Dir(α) can be generated as: η_i = Gamma(α, 1) / Σ Gamma(α, 1)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::gamma_distribution<float> gamma_dist(alpha, 1.0f);

    std::vector<float> noise(num_children);
    float sum = 0.0f;

    for (std::uint16_t i = 0; i < num_children; ++i) {
        noise[i] = gamma_dist(gen);
        sum += noise[i];
    }

    // Normalize Dirichlet samples
    if (sum > 0.0f) {
        for (float& n : noise) {
            n /= sum;
        }
    } else {
        // Fallback to uniform if all zeros (extremely rare)
        float uniform = 1.0f / num_children;
        for (float& n : noise) {
            n = uniform;
        }
    }

    // Mix with priors: P'(a) = (1 - ε) * P(a) + ε * η_a
    const float epsilon = 0.25f;  // AlphaZero uses 0.25
    NodeIndex first_child = tree_.get_first_child_index(root_index);

    for (std::uint16_t i = 0; i < num_children; ++i) {
        NodeIndex child = first_child + i;
        float original_prior = tree_.get_prior_prob(child);
        float mixed_prior = (1.0f - epsilon) * original_prior + epsilon * noise[i];
        tree_.set_prior_prob(child, mixed_prior);
    }
}

} // namespace mcts
