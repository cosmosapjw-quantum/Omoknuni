/**
 * @file inference_callback.hpp
 * @brief Python inference callback bridge for MCTS simulation runner
 *
 * This module provides a bridge between C++ SimulationRunner and Python
 * neural network inference. It wraps a Python callable and handles GIL
 * management automatically via pybind11.
 *
 * Key features:
 * - Wraps Python callable (lambda, function, method, etc.)
 * - Automatic GIL acquisition when calling Python
 * - Error handling for inference failures
 * - Type conversion between Python and C++
 */

#pragma once

#include "simulation_runner.hpp"
#include "../utils/igamestate.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

namespace mcts {

/**
 * @brief Python inference callback implementation
 *
 * This class wraps a Python callable to make it usable as an InferenceCallback
 * in C++ code. It handles GIL management and type conversions automatically.
 *
 * The Python callable should have signature:
 *   def inference_fn(state: IGameState) -> tuple[list[float], float]:
 *       ...
 *       return (policy_vector, value_scalar)
 *
 * Example usage from Python:
 *   def my_inference(state):
 *       # Neural network inference here
 *       policy = [0.1, 0.2, ...]  # Probability distribution
 *       value = 0.5                # Position evaluation
 *       return (policy, value)
 *
 *   callback = mcts_py.PyInferenceCallback(my_inference)
 *   runner.run_simulation(state, root_index, callback)
 */
class PyInferenceCallback : public InferenceCallback {
public:
    /**
     * @brief Construct callback with a Python callable
     *
     * @param python_fn Python callable that takes IGameState and returns (policy, value)
     */
    explicit PyInferenceCallback(py::object python_fn)
        : python_fn_(python_fn) {
        // Verify the callable is valid (has __call__ attribute)
        if (!py::hasattr(python_fn, "__call__")) {
            throw std::invalid_argument(
                "PyInferenceCallback requires a callable Python object"
            );
        }
    }

    /**
     * @brief Request inference from Python callable
     *
     * This method acquires the GIL, calls the Python function, and converts
     * the result back to C++ types. The GIL is automatically managed by
     * pybind11 py::object.
     *
     * @param state Game state to evaluate
     * @return Pair of (policy vector, value scalar)
     * @throws std::runtime_error if Python call fails or returns invalid data
     */
    std::pair<std::vector<float>, float>
    request_inference(const IGameState& state) override {
        try {
            // Call Python function with game state
            // GIL is automatically acquired by pybind11 when calling py::object
            // Cast const pointer to non-const for Python (Python wrapper won't modify it)
            IGameState* state_ptr = const_cast<IGameState*>(&state);
            py::object result = python_fn_(py::cast(state_ptr, py::return_value_policy::reference));

            // Extract tuple (policy, value) from Python result
            if (!py::isinstance<py::tuple>(result) || py::len(result) != 2) {
                throw std::runtime_error(
                    "Inference callback must return (policy_list, value_float) tuple"
                );
            }

            py::tuple result_tuple = result.cast<py::tuple>();

            // Extract policy (list/array of floats)
            py::object policy_obj = result_tuple[0];
            std::vector<float> policy;

            // Handle both list and numpy array
            if (py::isinstance<py::list>(policy_obj)) {
                policy = policy_obj.cast<std::vector<float>>();
            } else if (py::hasattr(policy_obj, "tolist")) {
                // NumPy array - convert to list first
                py::object policy_list = policy_obj.attr("tolist")();
                policy = policy_list.cast<std::vector<float>>();
            } else {
                throw std::runtime_error(
                    "Policy must be a list or numpy array of floats"
                );
            }

            // Extract value (scalar float)
            float value = result_tuple[1].cast<float>();

            return std::make_pair(policy, value);

        } catch (const py::error_already_set& e) {
            // Python exception occurred
            throw std::runtime_error(
                std::string("Python inference callback failed: ") + e.what()
            );
        } catch (const py::cast_error& e) {
            // Type conversion failed
            throw std::runtime_error(
                std::string("Inference callback type conversion failed: ") + e.what()
            );
        }
    }

private:
    py::object python_fn_;  ///< Python callable for inference
};

/**
 * @brief Abstract batch inference callback interface
 *
 * Allows C++ simulation runner to request batched neural network inference from Python.
 * Batching reduces GIL crossings from N (per simulation) to 1 (per batch).
 */
class BatchInferenceCallback {
public:
    virtual ~BatchInferenceCallback() = default;

    /**
     * @brief Request neural network inference for a batch of game states
     *
     * @param states Vector of game state pointers to evaluate
     * @return Vector of (policy vector, value scalar) pairs
     */
    virtual std::vector<std::pair<std::vector<float>, float>>
    batch_inference(const std::vector<const IGameState*>& states) = 0;
};

/**
 * @brief Python batch inference callback implementation
 *
 * Wraps a Python callable for batched inference. Reduces GIL crossings
 * from N (one per simulation) to 1 (one per batch).
 *
 * The Python callable should have signature:
 *   def batch_inference(states: list[IGameState]) -> list[tuple[list[float], float]]:
 *       # Batch all states through neural network
 *       return [(policy1, value1), (policy2, value2), ...]
 */
class PyBatchInferenceCallback : public BatchInferenceCallback {
public:
    explicit PyBatchInferenceCallback(py::object python_fn)
        : python_fn_(python_fn) {
        if (!py::hasattr(python_fn, "__call__")) {
            throw std::invalid_argument(
                "PyBatchInferenceCallback requires a callable Python object"
            );
        }
    }

    std::vector<std::pair<std::vector<float>, float>>
    batch_inference(const std::vector<const IGameState*>& states) override {
        try {
            // Convert C++ state pointers to Python list
            py::list py_states;
            for (const auto* state : states) {
                // Cast away const for Python (won't be modified)
                IGameState* state_ptr = const_cast<IGameState*>(state);
                py_states.append(py::cast(state_ptr, py::return_value_policy::reference));
            }

            // Call Python batch inference (GIL acquired automatically)
            py::object result = python_fn_(py_states);

            // Extract results
            if (!py::isinstance<py::list>(result)) {
                throw std::runtime_error(
                    "Batch inference must return list of (policy, value) tuples"
                );
            }

            py::list result_list = result.cast<py::list>();
            if (py::len(result_list) != states.size()) {
                throw std::runtime_error(
                    "Batch inference returned wrong number of results"
                );
            }

            std::vector<std::pair<std::vector<float>, float>> results;
            results.reserve(states.size());

            for (size_t i = 0; i < py::len(result_list); ++i) {
                py::tuple item = result_list[i].cast<py::tuple>();
                if (py::len(item) != 2) {
                    throw std::runtime_error(
                        "Each result must be (policy, value) tuple"
                    );
                }

                // Extract policy
                py::object policy_obj = item[0];
                std::vector<float> policy;
                if (py::isinstance<py::list>(policy_obj)) {
                    policy = policy_obj.cast<std::vector<float>>();
                } else if (py::hasattr(policy_obj, "tolist")) {
                    policy = policy_obj.attr("tolist")().cast<std::vector<float>>();
                } else {
                    throw std::runtime_error("Policy must be list or numpy array");
                }

                // Extract value
                float value = item[1].cast<float>();

                results.emplace_back(policy, value);
            }

            return results;

        } catch (const py::error_already_set& e) {
            throw std::runtime_error(
                std::string("Python batch inference failed: ") + e.what()
            );
        }
    }

private:
    py::object python_fn_;
};

} // namespace mcts
