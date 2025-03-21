#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <iostream>

#include "../core/mcts/mcts.h"
#include "../core/mcts/mcts_node.h"
#include "../core/mcts/transposition_table.h"
#include "../core/mcts/batch_evaluator.h"

namespace py = pybind11;
using namespace alphazero;

// Define the GomokuMCTS class (wrapper around MCTS for Gomoku-specific functionality)
class GomokuMCTS {
public:
    GomokuMCTS(int num_simulations = 800,
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

PYBIND11_MODULE(cpp_mcts, m) {
    m.doc() = "C++ MCTS implementation for AlphaZero";
    
    // Bind the MCTSNode class
    py::class_<MCTSNode>(m, "MCTSNode")
        .def(py::init<float, MCTSNode*, int>(),
             py::arg("prior") = 0.0f,
             py::arg("parent") = nullptr,
             py::arg("move") = -1)
        .def_readonly("visit_count", &MCTSNode::visit_count)
        .def_readonly("prior", &MCTSNode::prior)
        .def_readonly("move", &MCTSNode::move)
        .def("value", &MCTSNode::value)
        .def("is_expanded", &MCTSNode::is_expanded)
        .def("get_visit_counts", &MCTSNode::get_visit_counts);
    
    // Bind the TranspositionTable class
    py::class_<TranspositionTable>(m, "TranspositionTable")
        .def(py::init<size_t>(), py::arg("max_size") = 1000000)
        .def("lookup", &TranspositionTable::lookup, py::return_value_policy::reference)
        .def("store", &TranspositionTable::store)
        .def("clear", &TranspositionTable::clear)
        .def("size", &TranspositionTable::size)
        .def("contains", &TranspositionTable::contains);
    
    // Bind the LRUTranspositionTable class
    py::class_<LRUTranspositionTable, TranspositionTable>(m, "LRUTranspositionTable")
        .def(py::init<size_t>(), py::arg("max_size") = 1000000);
    
    // Bind the MCTS class
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
        .def("set_c_puct", &MCTS::set_c_puct)
        .def("set_dirichlet_noise", &MCTS::set_dirichlet_noise)
        .def("set_virtual_loss_weight", &MCTS::set_virtual_loss_weight)
        .def("set_temperature", &MCTS::set_temperature)
        .def("search", &MCTS::search, 
             py::arg("state_tensor"),
             py::arg("legal_moves"),
             py::arg("evaluator"),
             py::arg("progressive_widening") = false)
        .def("select_move", &MCTS::select_move, py::arg("temperature") = 1.0f)
        .def("get_probabilities", &MCTS::get_probabilities)
        .def("update_with_move", &MCTS::update_with_move)
        .def("reset", &MCTS::reset)
        .def("get_root", &MCTS::get_root, py::return_value_policy::reference)
        .def("get_tree_size", &MCTS::get_tree_size);
    
    // Create a game-specific wrapper for the MCTS
    py::class_<GomokuMCTS>(m, "GomokuMCTS")
        .def(py::init<int, float, float, float, float, bool, size_t, int>(),
             py::arg("num_simulations") = 800,
             py::arg("c_puct") = 1.5f,
             py::arg("dirichlet_alpha") = 0.3f,
             py::arg("dirichlet_noise_weight") = 0.25f,
             py::arg("virtual_loss_weight") = 1.0f,
             py::arg("use_transposition_table") = true,
             py::arg("transposition_table_size") = 1000000,
             py::arg("num_threads") = 1)
        .def("search", [](GomokuMCTS& self, py::array_t<int> board, const std::vector<int>& legal_moves, 
                         const std::function<std::pair<std::vector<float>, float>(const std::vector<int>&)>& evaluator) {
            // Convert numpy array to vector more safely
            py::buffer_info buf = board.request();
            
            // Copy data safely
            std::vector<int> board_vec(buf.size);
            int* ptr = static_cast<int*>(buf.ptr);
            for (ssize_t i = 0; i < buf.size; ++i) {
                board_vec[i] = ptr[i];
            }
            
            // Convert to float tensor
            std::vector<float> state_tensor(board_vec.size(), 0.0f);
            
            // Convert integer board values to float representation (-1, 0, 1)
            for (size_t i = 0; i < board_vec.size(); ++i) {
                if (board_vec[i] == 1) {
                    state_tensor[i] = 1.0f;  // Player 1
                } else if (board_vec[i] == 2) {
                    state_tensor[i] = -1.0f; // Player 2
                } else {
                    state_tensor[i] = 0.0f;  // Empty
                }
            }
            
            // Detect the board size to initialize the Zobrist hash correctly
            int board_size = static_cast<int>(std::sqrt(board_vec.size()));
            
            // Create wrapper around evaluator
            auto debug_evaluator = [&evaluator, &board_vec](const std::vector<float>& ignored) {
                auto result = evaluator(board_vec);
                return result;
            };
            
            // Run search
            auto result = self.mcts.search(state_tensor, legal_moves, debug_evaluator);
            
            return result;
        })
        .def("select_move", [](GomokuMCTS& self, float temperature) {
            return self.mcts.select_move(temperature);
        }, py::arg("temperature") = 1.0f)
        .def("update_with_move", [](GomokuMCTS& self, int move) {
            self.mcts.update_with_move(move);
        })
        .def("set_temperature", [](GomokuMCTS& self, float temperature) {
            self.mcts.set_temperature(temperature);
        })
        .def("set_num_simulations", [](GomokuMCTS& self, int num_simulations) {
            self.mcts.set_num_simulations(num_simulations);
        })
        .def("get_num_simulations", [](GomokuMCTS& self) {
            return self.mcts.get_num_simulations();
        })
        .def("get_c_puct", [](GomokuMCTS& self) {
            return self.mcts.get_c_puct();
        });
}