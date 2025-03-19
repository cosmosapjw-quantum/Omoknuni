#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "../core/game/gomoku.h"
#include "../core/game/attack_defense.h"

namespace py = pybind11;

PYBIND11_MODULE(bindings, m) {
    m.doc() = "Python bindings for the AlphaZero engine";
    
    // Bind the Gamestate class
    py::class_<Gamestate>(m, "Gamestate")
        .def(py::init<int, bool, bool, int, bool>(),
             py::arg("board_size") = 15,
             py::arg("use_renju") = false,
             py::arg("use_omok") = false,
             py::arg("seed") = 0,
             py::arg("use_pro_long_opening") = false,
             "Construct a new Gamestate with optional rule settings and board size")
        .def("copy", &Gamestate::copy, "Return a deep copy of the current game state")
        .def("make_move", &Gamestate::make_move, "Make a move at the specified action for the given player")
        .def("undo_move", &Gamestate::undo_move, "Undo the move at the specified action")
        .def("is_terminal", &Gamestate::is_terminal, "Check if the game is over (win or stalemate)")
        .def("get_winner", &Gamestate::get_winner, "Return the winner (1 for BLACK, 2 for WHITE, 0 if none)")
        .def("get_valid_moves", &Gamestate::get_valid_moves, "Return a list of valid moves")
        .def("get_board", &Gamestate::get_board, "Return a 2D vector representing the current board state")
        .def("apply_action", &Gamestate::apply_action, "Apply an action and return the new game state")
        .def("to_tensor", &Gamestate::to_tensor, "Convert the game state to a tensor for AI training")
        .def("get_action", &Gamestate::get_action, "Get the move that led from the current state to a child state")
        .def("is_five_in_a_row", &Gamestate::is_five_in_a_row, "Check if there is a five-in-a-row from the given cell")
        .def_readwrite("board_size", &Gamestate::board_size)
        .def_readwrite("current_player", &Gamestate::current_player)
        .def_readwrite("action", &Gamestate::action)
        .def_readwrite("black_first_stone", &Gamestate::black_first_stone)
        .def_readwrite("use_renju", &Gamestate::use_renju)
        .def_readwrite("use_omok", &Gamestate::use_omok)
        .def_readwrite("use_pro_long_opening", &Gamestate::use_pro_long_opening);
    
    // Bind the AttackDefenseModule class
    py::class_<AttackDefenseModule>(m, "AttackDefenseModule")
        .def(py::init<int>(), py::arg("board_size"))
        .def("__call__", [](AttackDefenseModule& self, 
                           py::array_t<float> board_np, 
                           py::array_t<int64_t> moves_np,
                           py::array_t<int64_t> player_np) {
            return self(board_np, moves_np, player_np);
        }, "Calculate attack and defense scores for the given board, moves, and player");
}