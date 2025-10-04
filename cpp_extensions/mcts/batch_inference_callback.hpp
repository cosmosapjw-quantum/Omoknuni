/**
 * @file batch_inference_callback.hpp
 * @brief Abstract batch inference callback interface (no Python dependencies)
 *
 * This header defines the pure C++ interface for batch inference callbacks.
 * It has no pybind11 dependencies, allowing it to be used in the mcts_core
 * static library.
 */

#pragma once

#include "../utils/igamestate.h"
#include <vector>
#include <utility>

namespace mcts {

/**
 * @brief Abstract batch inference callback interface
 *
 * Allows C++ simulation runner to request batched neural network inference.
 * Batching reduces GIL crossings from N (per simulation) to 1 (per batch).
 *
 * This is a pure C++ interface - concrete implementations may use Python
 * (via PyBatchInferenceCallback) or native C++ inference backends.
 */
class BatchInferenceCallback {
public:
    virtual ~BatchInferenceCallback() = default;

    /**
     * @brief Request neural network inference for a batch of game states
     *
     * @param states Vector of game state pointers to evaluate
     * @return Vector of (policy vector, value scalar) pairs
     *
     * Thread safety: Implementation must be thread-safe if called from
     * multiple threads (e.g., in BatchInferenceCoordinator background thread).
     */
    virtual std::vector<std::pair<std::vector<float>, float>>
    batch_inference(const std::vector<const IGameState*>& states) = 0;
};

} // namespace mcts
