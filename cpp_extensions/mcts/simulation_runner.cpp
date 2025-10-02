/**
 * @file simulation_runner.cpp
 * @brief Implementation of high-performance MCTS simulation runner
 */

#include "simulation_runner.hpp"
#include "../games/interface.h"
#include <stdexcept>
#include <algorithm>
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
    // TODO: Phase 2 - Implement full simulation logic
    // For now, return stub that throws
    throw std::runtime_error("SimulationRunner::run_simulation not implemented yet - Phase 1 stub");
}

NodeIndex SimulationRunner::select_leaf(NodeIndex root,
                                        IGameState& current_state,
                                        std::vector<NodeIndex>& path) {
    // TODO: Phase 2 - Implement selection phase
    // Traverse tree using PUCT until reaching unexpanded or terminal node
    throw std::runtime_error("SimulationRunner::select_leaf not implemented yet - Phase 1 stub");
}

float SimulationRunner::expand_node(NodeIndex leaf,
                                    IGameState& state,
                                    InferenceCallback& inference_fn) {
    // TODO: Phase 2 - Implement expansion phase
    // 1. Check if terminal → return game result value
    // 2. Request neural network inference via callback (acquires GIL)
    // 3. Mask policy to legal moves and normalize
    // 4. Allocate child nodes and initialize
    // 5. Mark node as expanded
    throw std::runtime_error("SimulationRunner::expand_node not implemented yet - Phase 1 stub");
}

void SimulationRunner::backup_value(const std::vector<NodeIndex>& path,
                                    float leaf_value) {
    // TODO: Phase 2 - Implement backup phase
    // Propagate value from leaf to root using BackupManager
    // BackupManager automatically handles:
    // - Value sign flipping at each level
    // - Virtual loss removal
    // - Atomic visit count updates
    throw std::runtime_error("SimulationRunner::backup_value not implemented yet - Phase 1 stub");
}

float SimulationRunner::get_terminal_value(const IGameState& state) {
    // TODO: Phase 2 - Implement terminal value extraction
    // Convert GameResult to value from current player's perspective
    throw std::runtime_error("SimulationRunner::get_terminal_value not implemented yet - Phase 1 stub");
}

} // namespace mcts
