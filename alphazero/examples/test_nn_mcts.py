#!/usr/bin/env python3
"""
Test script for the MCTS implementation with the neural network evaluator,
using the C++ Gomoku implementation.
"""

import sys
import os
import time
import torch

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alphazero.python.games.gomoku import GomokuGame
from alphazero.python.mcts.mcts import MCTS
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


def main():
    # Parameters
    board_size = 15
    use_renju = False
    use_omok = False
    
    # Create a Gomoku game with C++ implementation
    print("Creating Gomoku game with C++ implementation...")
    game = GomokuGame(board_size=board_size, use_renju=use_renju, use_omok=use_omok)
    
    # Create neural network
    print("Creating neural network...")
    network = SimpleConvNet(
        board_size=board_size,
        input_channels=3,
        num_filters=64,
        num_residual_blocks=3
    )
    
    # Create MCTS with neural network evaluator
    print("Creating MCTS with neural network evaluator...")
    mcts = MCTS(
        game=game,
        evaluator=network.predict,
        c_puct=1.5,
        num_simulations=200,  # Adjust based on your computer's speed
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25,
        temperature=1.0
    )
    
    # Play a game using MCTS
    print("Playing a game using MCTS with neural network and C++ Gomoku...")
    
    # Print the initial board
    board = game.state.get_board()
    print("\nInitial board:")
    print_board(board)
    print()
    
    move_count = 0
    
    while not game.is_terminal() and move_count < 30:  # Limit to 30 moves for testing
        # Get current player
        current_player = game.get_current_player()
        player_name = "Black" if current_player == 1 else "White"
        
        # Use MCTS to select a move
        print(f"{player_name}'s turn, thinking...")
        start_time = time.time()
        move, probs = mcts.select_move(return_probs=True)
        elapsed_time = time.time() - start_time
        
        # Get move coordinates
        row, col = divmod(move, game.board_size)
        
        # Print top 5 moves with probabilities
        top_moves = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
        top_moves_str = ", ".join([f"({m//game.board_size}, {m%game.board_size}): {p:.2f}" for m, p in top_moves])
        
        print(f"{player_name} plays at ({row}, {col}) - move {move} (took {elapsed_time:.2f}s)")
        print(f"Top moves: {top_moves_str}")
        
        # Apply the move
        game.apply_move(move)
        
        # Update the MCTS with the move
        mcts.update_with_move(move)
        
        # Print the board
        board = game.state.get_board()
        print_board(board)
        print()
        
        move_count += 1
    
    # Game over or move limit reached
    if game.is_terminal():
        winner = game.get_winner()
        if winner == 0:
            print("Game ended in a draw.")
        elif winner == 1:
            print("Black wins!")
        else:
            print("White wins!")
    else:
        print("Move limit reached.")


if __name__ == "__main__":
    main()