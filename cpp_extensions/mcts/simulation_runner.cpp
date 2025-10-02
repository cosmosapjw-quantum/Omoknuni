/**
 * @file simulation_runner.cpp
 * @brief Implementation of high-performance MCTS simulation runner
 */

#include "simulation_runner.hpp"
#include "../utils/igamestate.h"
#include <stdexcept>
#include <algorithm>  // for std::reverse
#include <cmath>

namespace mcts {

SimulationRunner::SimulationRunner(MCTSTree& tree,
                                   PUCTSelector& selector,
                                   BackupManager& backup,
                                   VirtualLossManager& virtual_loss)
    : tree_(tree)
    , selector_(selector)
    , backup_(backup)
    , virtual_loss_(virtual_loss) {
    // Pre-allocate path buffer to avoid reallocations
    // Typical MCTS depth: 10-50 nodes, reserve 256 for safety
    path_buffer_.reserve(256);
}

bool SimulationRunner::run_simulation(IGameState& root_state,
                                       NodeIndex root_index,
                                       InferenceCallback& inference_fn) {
    // Clone the root state to preserve original during traversal
    // Each simulation needs its own game state copy for move application
    std::unique_ptr<IGameState> current_state = root_state.clone();

    if (!current_state) {
        // Clone failed - should not happen but handle gracefully
        return false;
    }

    // Phase 1: Selection - Traverse tree to leaf using PUCT
    // Updates path_buffer_ with nodes from root to leaf
    // Applies moves to current_state during traversal
    // Applies virtual loss to prevent thread collisions
    NodeIndex leaf = select_leaf(root_index, *current_state, path_buffer_);

    // Phase 2: Expansion - Evaluate leaf with neural network and add children
    // If terminal: returns game result value, no children added
    // If non-terminal: calls inference, masks policy, allocates children
    // Returns value estimate for this position
    float leaf_value = expand_node(leaf, *current_state, inference_fn);

    // Phase 3: Backup - Propagate value up the path with sign flipping
    // Updates visit counts and Q-values atomically
    // Removes virtual loss applied during selection
    // Sign flips at each level: leaf→parent→grandparent...
    //
    // CRITICAL: BackupManager expects path in leaf-to-root order,
    // but select_leaf builds it in root-to-leaf order, so reverse it
    std::reverse(path_buffer_.begin(), path_buffer_.end());
    backup_value(path_buffer_, leaf_value);

    // Simulation completed successfully
    return true;
}

NodeIndex SimulationRunner::select_leaf(NodeIndex root,
                                        IGameState& current_state,
                                        std::vector<NodeIndex>& path) {
    // Clear path and start from root
    path.clear();
    path.push_back(root);

    NodeIndex current = root;

    // Traverse tree using PUCT until reaching a leaf (unexpanded or terminal)
    while (true) {
        // Check if current node is terminal
        if (current_state.isTerminal()) {
            break;  // Reached terminal node
        }

        // Check if current node is expanded
        NodeFlags flags = tree_.get_flags(current);
        if (!flags.is_expanded()) {
            break;  // Reached unexpanded leaf
        }

        // Node is expanded - select best child using PUCT
        SelectionResult result = selector_.select_child(tree_, current);

        if (!result.valid || result.selected_child == NULL_NODE_INDEX) {
            // No valid child found (shouldn't happen if expanded)
            break;
        }

        // Get the move that led to this child
        uint16_t move_index = tree_.get_move(result.selected_child);

        // Apply virtual loss to the selected child
        virtual_loss_.apply_virtual_loss(result.selected_child);

        // Apply the move to the game state
        current_state.makeMove(static_cast<int>(move_index));

        // Move to the selected child
        current = result.selected_child;
        path.push_back(current);
    }

    return current;
}

float SimulationRunner::expand_node(NodeIndex leaf,
                                    IGameState& state,
                                    InferenceCallback& inference_fn) {
    // Check if the state is terminal
    if (state.isTerminal()) {
        // Mark node as terminal and return the game result value
        NodeFlags flags = tree_.get_flags(leaf);
        flags.set_terminal(true);
        tree_.set_flags(leaf, flags);
        return get_terminal_value(state);
    }

    // Get legal moves for this position
    std::vector<int> legal_moves = state.getLegalMoves();

    if (legal_moves.empty()) {
        // No legal moves but not terminal - shouldn't happen in well-formed games
        // Treat as draw
        return 0.0f;
    }

    // Request neural network inference (this acquires GIL via pybind11)
    auto [policy, value] = inference_fn.request_inference(state);

    // Validate policy size matches action space
    int action_space_size = state.getActionSpaceSize();
    if (static_cast<int>(policy.size()) != action_space_size) {
        throw std::runtime_error("Policy size mismatch: got " +
                                std::to_string(policy.size()) +
                                " expected " + std::to_string(action_space_size));
    }

    // Apply legal move masking and renormalize
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

    // Normalize the masked policy
    if (policy_sum > 0.0f) {
        for (float& p : masked_policy) {
            p /= policy_sum;
        }
    } else {
        // Uniform distribution if all priors were zero
        float uniform_prob = 1.0f / legal_moves.size();
        for (float& p : masked_policy) {
            p = uniform_prob;
        }
    }

    // Allocate child nodes
    uint16_t num_children = static_cast<uint16_t>(legal_moves.size());
    NodeIndex first_child = tree_.allocate_nodes(num_children);

    if (first_child == NULL_NODE_INDEX) {
        // Tree is full - cannot expand
        // Return the value estimate without expansion
        return value;
    }

    // Initialize children
    for (uint16_t i = 0; i < num_children; ++i) {
        NodeIndex child_idx = first_child + i;

        // Set prior probability
        tree_.set_prior_prob(child_idx, masked_policy[i]);

        // Record the move that leads to this child
        tree_.set_move(child_idx, static_cast<uint16_t>(legal_moves[i]));

        // Set parent index
        tree_.set_parent_index(child_idx, leaf);

        // Initialize visit count and value to zero
        tree_.set_visit_count(child_idx, 0.0f);
        tree_.set_total_value(child_idx, 0.0f);
        tree_.set_virtual_loss(child_idx, 0.0f);

        // Initialize flags with current player
        NodeFlags child_flags;
        child_flags.set_current_player(state.getCurrentPlayer() == 1 ? 1 : 0);
        tree_.set_flags(child_idx, child_flags);
    }

    // Update parent node to link to children
    tree_.set_first_child_index(leaf, first_child);
    tree_.set_num_children(leaf, num_children);

    // Mark node as expanded
    NodeFlags flags = tree_.get_flags(leaf);
    flags.set_expanded(true);
    tree_.set_flags(leaf, flags);

    return value;
}

void SimulationRunner::backup_value(const std::vector<NodeIndex>& path,
                                    float leaf_value) {
    // Delegate to BackupManager which handles:
    // - Value sign flipping at each tree level (alternating player perspective)
    // - Atomic visit count and value updates for thread safety
    // - Virtual loss removal along the path

    BackupResult result = backup_.backup_value_along_path(
        path,
        leaf_value,
        &virtual_loss_  // Remove virtual loss during backup
    );

    // Note: BackupManager logs warnings internally if backup fails
    // The result can be checked but we don't throw here to allow graceful degradation
    (void)result;  // Suppress unused variable warning
}

float SimulationRunner::get_terminal_value(const IGameState& state) {
    GameResult result = state.getGameResult();
    int current_player = state.getCurrentPlayer();

    switch (result) {
        case GameResult::WIN_PLAYER1:
            // If current player is player 1, it's a win (+1), otherwise loss (-1)
            return (current_player == 1) ? 1.0f : -1.0f;

        case GameResult::WIN_PLAYER2:
            // If current player is player 2, it's a win (+1), otherwise loss (-1)
            return (current_player == 2) ? 1.0f : -1.0f;

        case GameResult::DRAW:
        case GameResult::NO_RESULT:
            // Draw or no result (e.g., Japanese Go rules edge cases)
            return 0.0f;

        case GameResult::ONGOING:
        default:
            // This shouldn't happen for terminal states
            return 0.0f;
    }
}

} // namespace mcts
