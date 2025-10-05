/**
 * @file continuous_simulation_runner.cpp
 * @brief Implementation of continuous simulation runner
 */

#include "continuous_simulation_runner.hpp"
#include "../utils/igamestate.h"
#include <algorithm>
#include <thread>
#include <chrono>

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
    pending_expansions_.clear();

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
                while (queue.pending_count() >= kMaxInFlight || pending_expansions_.size() >= kMaxInFlight) {
                    waiting_for_leaf = true;
                    int flushed = process_completed_results(queue);
                    if (flushed == 0) {
                        std::this_thread::sleep_for(std::chrono::microseconds(100));
                    }
                    if (++backoff_loops > 1024) {
                        break;  // Prevent unbounded waiting
                    }
                }

                // Submit request (NON-BLOCKING)
                uint64_t request_id = queue.submit_request(
                    std::move(queue_state),
                    leaf,
                    path_buffer_
                );

                // Track pending expansion (keep original state for expansion)
                PendingExpansion pending;
                pending.leaf_node = leaf;
                pending.path = path_buffer_;  // Copy path
                pending.state = std::move(current_state);  // Keep state for expansion
                pending_expansions_[request_id] = std::move(pending);

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

    pending_expansions_.clear();
    return completed;
}

int ContinuousSimulationRunner::process_completed_results(AsyncInferenceQueue& queue) {
    int processed = 0;

    while (queue.has_results()) {
        auto ready_ids = queue.get_ready_request_ids();
        bool handled_any = false;

        for (uint64_t request_id : ready_ids) {
            auto pending_it = pending_expansions_.find(request_id);
            if (pending_it == pending_expansions_.end()) {
                continue;
            }

            auto result_opt = queue.try_get_result(request_id);
            if (!result_opt.has_value()) {
                continue;
            }

            handled_any = true;
            auto pending = std::move(pending_it->second);
            pending_expansions_.erase(pending_it);

            const auto& result = result_opt.value();

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

        if (!handled_any) {
            break;
        }
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
    std::vector<float> masked_policy(legal_moves.size());

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
        for (float& p : masked_policy) {
            p /= policy_sum;
        }
    } else {
        float uniform_prob = 1.0f / legal_moves.size();
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

} // namespace mcts
