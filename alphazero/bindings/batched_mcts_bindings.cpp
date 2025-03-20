#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <mutex>
#include <thread>
#include <future>
#include <atomic>

#include "../core/mcts/mcts.h"
#include "../core/mcts/mcts_node.h"
#include "../core/mcts/transposition_table.h"
#include "../core/mcts/batch_evaluator.h"

namespace py = pybind11;
using namespace alphazero;

// Define the GomokuMCTS class with batch support
class BatchedGomokuMCTS {
public:
    BatchedGomokuMCTS(int num_simulations = 800,
               float c_puct = 1.5f,
               float dirichlet_alpha = 0.3f,
               float dirichlet_noise_weight = 0.25f,
               float virtual_loss_weight = 1.0f,
               bool use_transposition_table = true,
               size_t transposition_table_size = 1000000,
               int num_threads = 1)
        : mcts(num_simulations, c_puct, dirichlet_alpha, dirichlet_noise_weight,
               virtual_loss_weight, use_transposition_table, transposition_table_size, num_threads) {
    }
    
    MCTS mcts;
};

// Process a Python list or numpy array of board states into C++ vectors
std::vector<std::vector<float>> process_board_batch(const py::object& board_batch) {
    std::vector<std::vector<float>> result;
    
    // Check if we're dealing with a list of boards or a single board
    if (py::isinstance<py::list>(board_batch) || py::isinstance<py::array>(board_batch)) {
        // Try to iterate through the batch
        try {
            for (const auto& board : board_batch) {
                // Convert each board to a flat vector of floats
                std::vector<float> board_vec;
                
                // Handle numpy arrays or lists
                if (py::isinstance<py::array>(board)) {
                    py::array_t<float> arr = board.cast<py::array_t<float>>();
                    py::buffer_info buf = arr.request();
                    float* ptr = static_cast<float*>(buf.ptr);
                    board_vec.assign(ptr, ptr + buf.size);
                } else if (py::isinstance<py::list>(board)) {
                    board_vec = board.cast<std::vector<float>>();
                }
                
                result.push_back(board_vec);
            }
        } catch (const py::error_already_set& e) {
            // Handle a single board case (not really a batch, but we'll make it one)
            std::vector<float> board_vec;
            
            if (py::isinstance<py::array>(board_batch)) {
                py::array_t<float> arr = board_batch.cast<py::array_t<float>>();
                py::buffer_info buf = arr.request();
                float* ptr = static_cast<float*>(buf.ptr);
                board_vec.assign(ptr, ptr + buf.size);
            } else if (py::isinstance<py::list>(board_batch)) {
                board_vec = board_batch.cast<std::vector<float>>();
            }
            
            if (!board_vec.empty()) {
                result.push_back(board_vec);
            }
        }
    }
    
    return result;
}

// Process a batch of integer boards into float state tensors
std::vector<std::vector<float>> process_int_board_batch(const py::object& board_batch) {
    std::vector<std::vector<float>> result;
    
    // Check if we're dealing with a list of boards or a single board
    if (py::isinstance<py::list>(board_batch) || py::isinstance<py::array>(board_batch)) {
        // Try to iterate through the batch
        try {
            for (const auto& board : board_batch) {
                // Convert each integer board to a float tensor
                std::vector<int> int_board;
                
                // Handle numpy arrays or lists
                if (py::isinstance<py::array>(board)) {
                    py::array_t<int> arr = board.cast<py::array_t<int>>();
                    py::buffer_info buf = arr.request();
                    int* ptr = static_cast<int*>(buf.ptr);
                    int_board.assign(ptr, ptr + buf.size);
                } else if (py::isinstance<py::list>(board)) {
                    int_board = board.cast<std::vector<int>>();
                }
                
                // Convert to float tensor
                std::vector<float> board_vec(int_board.size());
                for (size_t i = 0; i < int_board.size(); ++i) {
                    if (int_board[i] == 1) {
                        board_vec[i] = 1.0f;  // Player 1
                    } else if (int_board[i] == 2) {
                        board_vec[i] = -1.0f; // Player 2
                    } else {
                        board_vec[i] = 0.0f;  // Empty
                    }
                }
                
                result.push_back(board_vec);
            }
        } catch (const py::error_already_set& e) {
            // Handle a single board case
            std::vector<int> int_board;
            
            if (py::isinstance<py::array>(board_batch)) {
                py::array_t<int> arr = board_batch.cast<py::array_t<int>>();
                py::buffer_info buf = arr.request();
                int* ptr = static_cast<int*>(buf.ptr);
                int_board.assign(ptr, ptr + buf.size);
            } else if (py::isinstance<py::list>(board_batch)) {
                int_board = board_batch.cast<std::vector<int>>();
            }
            
            // Convert to float tensor
            std::vector<float> board_vec(int_board.size());
            for (size_t i = 0; i < int_board.size(); ++i) {
                if (int_board[i] == 1) {
                    board_vec[i] = 1.0f;  // Player 1
                } else if (int_board[i] == 2) {
                    board_vec[i] = -1.0f; // Player 2
                } else {
                    board_vec[i] = 0.0f;  // Empty
                }
            }
            
            if (!board_vec.empty()) {
                result.push_back(board_vec);
            }
        }
    }
    
    return result;
}

// Python batch evaluation function wrapper
// This handles GIL release/acquire properly
std::vector<std::pair<std::vector<float>, float>> py_batch_evaluator(
    const std::vector<std::vector<float>>& batch,
    const py::function& evaluator,
    std::mutex& gil_mutex) {
    
    std::vector<std::pair<std::vector<float>, float>> results;
    
    {
        // Lock to ensure only one thread calls into Python at a time
        std::lock_guard<std::mutex> lock(gil_mutex);
        
        // Acquire the GIL before calling into Python
        py::gil_scoped_acquire acquire;
        
        try {
            // Call the Python batch evaluator function
            py::object batch_results = evaluator(batch);
            
            // Process the results
            for (const auto& result : batch_results) {
                auto policy = result.attr("__getitem__")(0).cast<std::vector<float>>();
                auto value = result.attr("__getitem__")(1).cast<float>();
                results.push_back({policy, value});
            }
        }
        catch (py::error_already_set& e) {
            std::cerr << "Python error in batch evaluator: " << e.what() << std::endl;
            
            // Return default values on error
            for (const auto& state : batch) {
                results.push_back({std::vector<float>(state.size(), 1.0f / state.size()), 0.0f});
            }
        }
    }
    
    return results;
}

PYBIND11_MODULE(batched_cpp_mcts, m) {
    m.doc() = "Batched C++ MCTS implementation for AlphaZero with efficient leaf parallelization";
    
    // Bind the BatchEvaluator class
    py::class_<BatchEvaluator>(m, "BatchEvaluator")
        .def(py::init<BatchEvaluator::EvaluationFunction, size_t, size_t>(),
             py::arg("evaluation_function"),
             py::arg("batch_size") = 16,
             py::arg("max_wait_ms") = 10)
        .def("start", &BatchEvaluator::start)
        .def("stop", &BatchEvaluator::stop)
        .def("enqueue_position", &BatchEvaluator::enqueue_position)
        .def("get_result", &BatchEvaluator::get_result)
        .def("get_stats", &BatchEvaluator::get_stats)
        .def("reset_stats", &BatchEvaluator::reset_stats);
    
    // Bind the MCTS class with batched search
    py::class_<MCTS>(m, "MCTS")
        .def(py::init<int, float, float, float, float, bool, size_t, int>(),
             py::arg("num_simulations") = 800,
             py::arg("c_puct") = 1.5f,
             py::arg("dirichlet_alpha") = 0.3f,
             py::arg("dirichlet_noise_weight") = 0.25f,
             py::arg("virtual_loss_weight") = 1.0f,
             py::arg("use_transposition_table") = true,
             py::arg("transposition_table_size") = 1000000,
             py::arg("num_threads") = 1)
        .def("set_num_simulations", &MCTS::set_num_simulations)
        .def("get_num_simulations", &MCTS::get_num_simulations)
        .def("set_c_puct", &MCTS::set_c_puct)
        .def("get_c_puct", &MCTS::get_c_puct)
        .def("set_dirichlet_noise", &MCTS::set_dirichlet_noise)
        .def("set_virtual_loss_weight", &MCTS::set_virtual_loss_weight)
        .def("set_temperature", &MCTS::set_temperature)
        .def("search_batched", [](MCTS& self, 
                               const std::vector<float>& state_tensor,
                               const std::vector<int>& legal_moves,
                               const py::function& batch_evaluator,
                               size_t batch_size = 16,
                               size_t max_wait_ms = 10,
                               bool progressive_widening = false) {
            // Create a mutex for GIL safety
            std::mutex gil_mutex;
            
            // Create a C++ wrapper for the Python batch evaluator function
            auto safe_batch_evaluator = [&batch_evaluator, &gil_mutex](const std::vector<std::vector<float>>& batch) {
                return py_batch_evaluator(batch, batch_evaluator, gil_mutex);
            };
            
            // Release the GIL before starting C++ threads
            py::gil_scoped_release release;
            
            // Call the batched search method
            return self.search_batched(state_tensor, legal_moves, safe_batch_evaluator, 
                                      batch_size, max_wait_ms, progressive_widening);
        },
        py::arg("state_tensor"),
        py::arg("legal_moves"),
        py::arg("batch_evaluator"),
        py::arg("batch_size") = 16,
        py::arg("max_wait_ms") = 10,
        py::arg("progressive_widening") = false)
        .def("select_move", &MCTS::select_move, py::arg("temperature") = 1.0f)
        .def("get_probabilities", &MCTS::get_probabilities)
        .def("update_with_move", &MCTS::update_with_move)
        .def("reset", &MCTS::reset)
        .def("get_tree_size", &MCTS::get_tree_size);
    
    // Create a game-specific wrapper for the batched MCTS
    py::class_<BatchedGomokuMCTS>(m, "BatchedGomokuMCTS")
        .def(py::init<int, float, float, float, float, bool, size_t, int>(),
             py::arg("num_simulations") = 800,
             py::arg("c_puct") = 1.5f,
             py::arg("dirichlet_alpha") = 0.3f,
             py::arg("dirichlet_noise_weight") = 0.25f,
             py::arg("virtual_loss_weight") = 1.0f,
             py::arg("use_transposition_table") = true,
             py::arg("transposition_table_size") = 1000000,
             py::arg("num_threads") = 1)
        .def("search_batched", [](BatchedGomokuMCTS& self, 
                              py::array_t<int> board, 
                              const std::vector<int>& legal_moves, 
                              const py::function& batch_evaluator,
                              size_t batch_size = 16,
                              size_t max_wait_ms = 10) {
            // Process board to get state tensor
            py::buffer_info buf = board.request();
            
            // Copy data safely
            std::vector<int> board_vec(buf.size);
            int* ptr = static_cast<int*>(buf.ptr);
            for (ssize_t i = 0; i < buf.size; ++i) {
                board_vec[i] = ptr[i];
            }
            
            // Convert to float tensor
            std::vector<float> state_tensor(board_vec.size(), 0.0f);
            for (size_t i = 0; i < board_vec.size(); ++i) {
                if (board_vec[i] == 1) {
                    state_tensor[i] = 1.0f;  // Player 1
                } else if (board_vec[i] == 2) {
                    state_tensor[i] = -1.0f; // Player 2
                } else {
                    state_tensor[i] = 0.0f;  // Empty
                }
            }
            
            // Create a mutex for GIL safety
            std::mutex gil_mutex;
            
            // Create a wrapper for batch evaluation
            auto safe_batch_evaluator = [&batch_evaluator, &gil_mutex](const std::vector<std::vector<float>>& batch) {
                return py_batch_evaluator(batch, batch_evaluator, gil_mutex);
            };
            
            // Release the GIL before starting C++ threads
            py::gil_scoped_release release;
            
            // Call the batched search method
            return self.mcts.search_batched(state_tensor, legal_moves, safe_batch_evaluator,
                                           batch_size, max_wait_ms);
        },
        py::arg("board"),
        py::arg("legal_moves"),
        py::arg("batch_evaluator"),
        py::arg("batch_size") = 16,
        py::arg("max_wait_ms") = 10)
        .def("select_move", [](BatchedGomokuMCTS& self, float temperature) {
            return self.mcts.select_move(temperature);
        }, py::arg("temperature") = 1.0f)
        .def("update_with_move", [](BatchedGomokuMCTS& self, int move) {
            self.mcts.update_with_move(move);
        })
        .def("set_temperature", [](BatchedGomokuMCTS& self, float temperature) {
            self.mcts.set_temperature(temperature);
        })
        .def("set_num_simulations", [](BatchedGomokuMCTS& self, int num_simulations) {
            self.mcts.set_num_simulations(num_simulations);
        })
        .def("get_tree_size", [](BatchedGomokuMCTS& self) {
            return self.mcts.get_tree_size();
        });
}