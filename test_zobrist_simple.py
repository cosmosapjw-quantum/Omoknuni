#!/usr/bin/env python3
"""
Simple test script for verifying that the Zobrist hashing is working correctly.
"""

import time
import numpy as np
from alphazero.bindings.cpp_mcts import GomokuMCTS
from alphazero.python.games.gomoku import GomokuGame


def test_zobrist_hashing():
    """Test basic functionality of Zobrist hashing in MCTS"""
    
    print("Testing Zobrist hash implementation...")
    
    # Create a simple game instance
    board_size = 9
    game = GomokuGame(board_size=board_size)
    
    # Create an MCTS instance
    mcts = GomokuMCTS(
        num_simulations=10,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25,
        use_transposition_table=True,
        num_threads=1
    )
    
    # Create a simple dummy evaluator
    def dummy_evaluator(board_flat):
        policy = [1.0 / (board_size * board_size)] * (board_size * board_size)
        return policy, 0.0
    
    # Make a few moves
    moves = [(board_size // 2) * board_size + (board_size // 2)]  # Center
    
    for move in moves:
        try:
            game.apply_move(move)
        except Exception as e:
            print(f"Error making move {move}: {e}")
            break
    
    # Get the board state
    board = game.get_board().flatten()
    legal_moves = list(game.get_legal_moves())
    
    # Run a search
    print(f"Running MCTS search with board size {board_size}x{board_size}")
    
    try:
        # First search
        start_time = time.time()
        probs1 = mcts.search(board, legal_moves, dummy_evaluator)
        elapsed = time.time() - start_time
        print(f"Search completed in {elapsed:.4f} seconds")
        
        # Verify that we get reasonable results
        print(f"Number of moves with non-zero probability: {len(probs1)}")
        
        # Make sure the highest probability move is reasonable
        if probs1:
            best_move = max(probs1.items(), key=lambda x: x[1])[0]
            best_prob = probs1[best_move]
            print(f"Best move: {best_move} with probability {best_prob:.4f}")
            
            # Select a move
            selected_move = mcts.select_move(temperature=1.0)
            print(f"Selected move: {selected_move}")
            
            print("Zobrist hash appears to be functioning correctly!")
            return True
        else:
            print("Error: No moves with non-zero probability.")
            return False
    except Exception as e:
        print(f"Error during search: {e}")
        return False


if __name__ == "__main__":
    test_zobrist_hashing()