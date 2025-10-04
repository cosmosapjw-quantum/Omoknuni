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

    // Continuous loop until quota reached
    while (completed < num_simulations) {
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

            // Check if already expanded
            NodeFlags flags = tree_.get_flags(leaf);
            if (flags.is_expanded()) {
                // Already expanded - shouldn't happen often
                // Backup neutral value and continue
                std::reverse(path_buffer_.begin(), path_buffer_.end());
                backup_value(path_buffer_, 0.0f);
                completed++;
                submitted++;
                continue;
            }

            // Non-terminal unexpanded node - needs inference
            // Clone state for queue submission (queue takes ownership)
            std::unique_ptr<IGameState> queue_state = current_state->clone();
            if (!queue_state) {
                continue;  // Skip on clone failure
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

        // Phase 2: Process completed results (NON-BLOCKING)
        int processed = process_completed_results(queue);
        completed += processed;

        // Yield briefly if no results available to avoid busy-waiting
        if (processed == 0 && submitted >= num_simulations) {
            // All submitted, just waiting for results
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    return completed;
}

int ContinuousSimulationRunner::process_completed_results(AsyncInferenceQueue& queue) {
    int processed = 0;

    // Process all available results in a batch
    while (queue.has_results()) {
        // Find a pending expansion with available result
        bool found_result = false;

        for (auto it = pending_expansions_.begin(); it != pending_expansions_.end(); ) {
            uint64_t request_id = it->first;
            auto& pending = it->second;

            // Try to get result (non-blocking)
            auto result_opt = queue.try_get_result(request_id);
            if (!result_opt.has_value()) {
                ++it;
                continue;
            }

            found_result = true;
            const auto& result = result_opt.value();

            // Expand node with result (we have the state stored)
            if (pending.state && expand_node_with_result(
                    pending.leaf_node,
                    *pending.state,
                    result.policy,
                    result.value)) {
                // Expansion successful, backup value
                std::vector<NodeIndex> path = pending.path;
                std::reverse(path.begin(), path.end());
                backup_value(path, result.value);
            } else {
                // Expansion failed, backup neutral value
                std::vector<NodeIndex> path = pending.path;
                std::reverse(path.begin(), path.end());
                backup_value(path, 0.0f);
            }

            processed++;

            // Remove from pending
            it = pending_expansions_.erase(it);
        }

        if (!found_result) {
            // No more results match our pending expansions
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

    // Update parent
    tree_.set_first_child_index(leaf, first_child);
    tree_.set_num_children(leaf, num_children);

    // Mark as expanded
    NodeFlags flags = tree_.get_flags(leaf);
    flags.set_expanded(true);
    tree_.set_flags(leaf, flags);

    return true;
}

} // namespace mcts
