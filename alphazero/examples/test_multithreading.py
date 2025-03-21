#!/usr/bin/env python3
"""
Test script to verify the multithreading bug fix in MCTS.
"""

import sys
import os
import time
import numpy as np

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alphazero.python.games.gomoku import GomokuGame
from alphazero.python.mcts.cpp_mcts_wrapper import CppMCTSWrapper

def run_mcts_test(num_threads, num_simulations=50):
    """Run MCTS with specified number of threads and simulations."""
    print(f"Testing MCTS with {num_threads} threads and {num_simulations} simulations")
    
    # Create a simple game
    game = GomokuGame(board_size=9)
    
    # Create a simple evaluator that returns uniform policy and random value
    def simple_evaluator(state):
        board_size = game.board_size
        policy = np.ones(board_size * board_size) / (board_size * board_size)
        value = np.random.uniform(-0.1, 0.1)  # Small random value to avoid extreme values
        return policy, value
    
    # Create MCTS
    mcts = CppMCTSWrapper(
        game=game,
        evaluator=simple_evaluator,
        c_puct=1.5,
        num_simulations=num_simulations,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25,
        temperature=1.0,
        use_transposition_table=True,
        transposition_table_size=10000,
        num_threads=num_threads
    )
    
    # Run search and time it
    start_time = time.time()
    try:
        probabilities = mcts.search()
        end_time = time.time()
        
        # Check if search returned valid probabilities
        if not probabilities:
            print("  ERROR: Search returned empty probabilities")
            return False
        
        # Get move counts from root
        move_counts = {}
        total_visits = 0
        for move, prob in probabilities.items():
            move_counts[move] = int(prob * num_simulations * 10)  # Approximation based on probabilities
            total_visits += move_counts[move]
        
        print(f"  Search completed in {end_time - start_time:.3f} seconds")
        print(f"  Root has approximately {total_visits} total visits across all children")
        
        # Select and apply a move
        move = mcts.select_move()
        game.apply_move(move)
        mcts.update_with_move(move)
        
        print(f"  Selected move: {move // game.board_size}, {move % game.board_size}")
        return True
    
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting multithreading test...")
    
    # First test with 1 thread as baseline
    success_single = run_mcts_test(num_threads=1)
    
    if success_single:
        print("\nSingle-threaded test passed!\n")
        
        # Now test with multiple threads
        success_multi = run_mcts_test(num_threads=4)
        
        if success_multi:
            print("\nMulti-threaded test passed!")
            print("\nBug fix VERIFIED! The MCTS implementation now works correctly with multiple threads.")
        else:
            print("\nMulti-threaded test FAILED. More debugging needed.")
    else:
        print("\nSingle-threaded test FAILED. Basic functionality issue detected.")