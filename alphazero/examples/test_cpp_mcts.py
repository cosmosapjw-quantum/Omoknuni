#!/usr/bin/env python3
"""
Test script for the C++ MCTS implementation.
"""

import sys
import os
import time
import torch
import numpy as np
from typing import Dict, Tuple
import argparse

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alphazero.python.games.gomoku import GomokuGame
from alphazero.python.mcts.mcts import MCTS  # Python MCTS
from alphazero.python.mcts.cpp_mcts_wrapper import CppMCTSWrapper  # C++ MCTS wrapper
from alphazero.python.models.simple_conv_net import SimpleConvNet


def print_board(board):
    """Print the board state in a more readable format."""
    symbols = {0: ".", 1: "●", 2: "○"}
    board_size = len(board)
    
    print("   ", end="")
    for i in range(board_size):
        print(f"{i:2d}", end=" ")
    print()
    
    for i in range(board_size):
        print(f"{i:2d} ", end="")
        for j in range(board_size):
            print(f" {symbols[board[i][j]]}", end=" ")
        print()


def performance_test(board_size=5, num_simulations=20, use_cuda=False, num_threads=1):
    """
    Compare the performance of Python MCTS and C++ MCTS with and without transposition table.
    
    Args:
        board_size: Size of the board
        num_simulations: Number of simulations to run
        use_cuda: Whether to use CUDA for the neural network
        num_threads: Number of threads for the C++ MCTS
    """
    print(f"Running performance test with board_size={board_size}, "
          f"num_simulations={num_simulations}, use_cuda={use_cuda}, num_threads={num_threads}")
    
    # Set device
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create neural network
    network = SimpleConvNet(
        board_size=board_size,
        input_channels=3,
        num_filters=64,
        num_residual_blocks=3
    ).to(device)
    
    # Create Gomoku game
    game = GomokuGame(board_size=board_size)
    
    # Play a few random moves to get to an interesting position
    for _ in range(min(5, board_size * board_size // 4)):
        legal_moves = game.get_legal_moves()
        move = np.random.choice(legal_moves)
        game.apply_move(move)
    
    # Create Python MCTS
    python_mcts = MCTS(
        game=game.clone(),
        evaluator=network.predict,
        c_puct=1.5,
        num_simulations=num_simulations,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25,
        temperature=1.0
    )
    
    # Try to create C++ MCTS implementations
    try:
        # Without transposition table
        cpp_mcts_without_tt = CppMCTSWrapper(
            game=game.clone(),
            evaluator=network.predict,
            c_puct=1.5,
            num_simulations=num_simulations,
            dirichlet_alpha=0.3,
            dirichlet_noise_weight=0.25,
            temperature=1.0,
            use_transposition_table=False,
            transposition_table_size=100000,
            num_threads=num_threads
        )
        
        # With transposition table
        cpp_mcts_with_tt = CppMCTSWrapper(
            game=game.clone(),
            evaluator=network.predict,
            c_puct=1.5,
            num_simulations=num_simulations,
            dirichlet_alpha=0.3,
            dirichlet_noise_weight=0.25,
            temperature=1.0,
            use_transposition_table=True,
            transposition_table_size=100000,
            num_threads=num_threads
        )
        
        cpp_available = True
    except ImportError:
        print("C++ MCTS not available. Only testing Python MCTS.")
        cpp_available = False
    
    # Print the board
    print("\nCurrent board:")
    print_board(game.state.get_board())
    print()
    
    # Test Python MCTS
    print("Running Python MCTS...")
    start_time = time.time()
    python_probs = python_mcts.search()
    python_time = time.time() - start_time
    python_move = python_mcts.select_move()
    
    print(f"Python MCTS took {python_time:.3f} seconds ({num_simulations / python_time:.1f} simulations/second)")
    
    # Print top 5 moves from Python MCTS
    print("Top 5 moves from Python MCTS:")
    top_moves = sorted(python_probs.items(), key=lambda x: x[1], reverse=True)[:5]
    for move, prob in top_moves:
        row, col = divmod(move, board_size)
        print(f"  ({row}, {col}): {prob:.3f}")
    
    # Test C++ MCTS without transposition table
    if cpp_available:
        print("\nRunning C++ MCTS without transposition table...")
        start_time = time.time()
        cpp_probs_without_tt = cpp_mcts_without_tt.search()
        cpp_time_without_tt = time.time() - start_time
        cpp_move_without_tt = cpp_mcts_without_tt.select_move()
        
        print(f"C++ MCTS without TT took {cpp_time_without_tt:.3f} seconds "
              f"({num_simulations / cpp_time_without_tt:.1f} simulations/second)")
        print(f"Speedup vs Python: {python_time / cpp_time_without_tt:.2f}x")
        
        # Print top 5 moves from C++ MCTS without TT
        print("Top 5 moves from C++ MCTS without TT:")
        top_moves = sorted(cpp_probs_without_tt.items(), key=lambda x: x[1], reverse=True)[:5]
        for move, prob in top_moves:
            row, col = divmod(move, board_size)
            print(f"  ({row}, {col}): {prob:.3f}")
        
        # Test C++ MCTS with transposition table
        print("\nRunning C++ MCTS with transposition table...")
        start_time = time.time()
        cpp_probs_with_tt = cpp_mcts_with_tt.search()
        cpp_time_with_tt = time.time() - start_time
        cpp_move_with_tt = cpp_mcts_with_tt.select_move()
        
        print(f"C++ MCTS with TT took {cpp_time_with_tt:.3f} seconds "
              f"({num_simulations / cpp_time_with_tt:.1f} simulations/second)")
        print(f"Speedup vs without TT: {cpp_time_without_tt / cpp_time_with_tt:.2f}x")
        print(f"Speedup vs Python: {python_time / cpp_time_with_tt:.2f}x")
        
        # Print top 5 moves from C++ MCTS with TT
        print("Top 5 moves from C++ MCTS with TT:")
        top_moves = sorted(cpp_probs_with_tt.items(), key=lambda x: x[1], reverse=True)[:5]
        for move, prob in top_moves:
            row, col = divmod(move, board_size)
            print(f"  ({row}, {col}): {prob:.3f}")
        
        # Compare all the moves
        py_row, py_col = divmod(python_move, board_size)
        cpp_row_without_tt, cpp_col_without_tt = divmod(cpp_move_without_tt, board_size)
        cpp_row_with_tt, cpp_col_with_tt = divmod(cpp_move_with_tt, board_size)
        
        print(f"\nPython MCTS selected move: ({py_row}, {py_col})")
        print(f"C++ MCTS without TT selected move: ({cpp_row_without_tt}, {cpp_col_without_tt})")
        print(f"C++ MCTS with TT selected move: ({cpp_row_with_tt}, {cpp_col_with_tt})")


def parse_args():
    parser = argparse.ArgumentParser(description="Test C++ MCTS performance")
    
    parser.add_argument("--board-size", type=int, default=6,
                        help="Size of the Gomoku board (default: 6)")
    parser.add_argument("--num-simulations", type=int, default=50,
                        help="Number of MCTS simulations (default: 50)")
    parser.add_argument("--num-threads", type=int, default=1,
                        help="Number of threads for C++ MCTS (default: 1)")
    parser.add_argument("--use-cuda", action="store_true",
                        help="Use CUDA if available")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    performance_test(
        board_size=args.board_size,
        num_simulations=args.num_simulations,
        use_cuda=args.use_cuda,
        num_threads=args.num_threads
    )


if __name__ == "__main__":
    main()