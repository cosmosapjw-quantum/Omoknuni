#!/usr/bin/env python3
"""
Test script for the MCTS implementation with a random evaluator.
"""

import sys
import os
import random
import time
import numpy as np
from typing import Dict, Tuple

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alphazero.python.games.gomoku import GomokuGame
from alphazero.python.mcts.mcts import MCTS
from alphazero.python.games.game_base import GameWrapper


def random_evaluator(game_state: GameWrapper) -> Tuple[Dict[int, float], float]:
    """
    Random evaluator for testing. Returns random policy and value.
    
    Args:
        game_state: The game state to evaluate
        
    Returns:
        Tuple of (policy, value), where policy is a dict mapping moves to probabilities
    """
    # Get legal moves
    legal_moves = game_state.get_legal_moves()
    
    if not legal_moves:
        return {}, 0.0
    
    # Generate random probabilities for the legal moves
    probs = np.random.dirichlet([1.0] * len(legal_moves))
    
    # Create policy dictionary
    policy = {move: float(prob) for move, prob in zip(legal_moves, probs)}
    
    # Generate random value between -1 and 1
    value = random.uniform(-1.0, 1.0)
    
    return policy, value


def main():
    # Create a Gomoku game with a smaller board size
    board_size = 6  # Using a smaller board size
    num_sims = 50   # Fewer simulations
    
    print(f"Creating Gomoku game with board_size={board_size}...")
    game = GomokuGame(board_size=board_size)
    
    # Create MCTS with random evaluator
    print(f"Creating MCTS with random evaluator and {num_sims} simulations...")
    mcts = MCTS(
        game=game,
        evaluator=random_evaluator,
        c_puct=1.0,
        num_simulations=num_sims,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25,
        temperature=1.0
    )
    
    # Play a game using MCTS
    print("Playing a game using MCTS...")
    
    while not game.is_terminal():
        # Get current player
        current_player = game.get_current_player()
        player_name = "Black" if current_player == 1 else "White"
        
        # Use MCTS to select a move
        print(f"{player_name}'s turn, running search...")
        start_time = time.time()
        
        print("Starting search...")
        # First run the search
        probs = mcts.search()
        print("Search completed, selecting move...")
        # Then select the move
        move = mcts.select_move()
        
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
        game.print_board()
        print()
    
    # Game over, print the result
    winner = game.get_winner()
    if winner == 0:
        print("Game ended in a draw.")
    elif winner == 1:
        print("Black wins!")
    else:
        print("White wins!")


if __name__ == "__main__":
    main()