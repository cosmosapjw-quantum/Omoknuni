#!/usr/bin/env python3
"""
Test script for MCTS threading bug fix verification.
This script tests the MCTS implementation with multiple threads.
"""

import sys
import os
import time
import numpy as np
import traceback

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from alphazero.python.games.gomoku import GomokuGame
except ImportError as e:
    print(f"Error importing GomokuGame: {e}")
    sys.exit(1)

try:
    from alphazero.python.mcts.cpp_mcts_wrapper import CppMCTSWrapper
except ImportError as e:
    print(f"Error importing CppMCTSWrapper: {e}")
    sys.exit(1)

def dummy_evaluator(state_tensor):
    """A simple deterministic evaluator for testing."""
    # Return fixed policy and value to make tests predictable
    board_size = int(np.sqrt(len(state_tensor) / 3))  # Assuming 3 channels
    policy = np.ones(board_size * board_size) / (board_size * board_size)
    return policy, 0.0

def test_single_thread():
    """Basic test with a single thread to establish baseline."""
    print("Testing with 1 thread...")
    
    try:
        # Create game
        game = GomokuGame(board_size=5)
        
        # Create MCTS with 1 thread
        mcts = CppMCTSWrapper(
            game=game,
            evaluator=dummy_evaluator,
            c_puct=1.5,
            num_simulations=10,  # Small number for quick test
            dirichlet_alpha=0.3,
            dirichlet_noise_weight=0.25,
            temperature=1.0,
            use_transposition_table=True,
            transposition_table_size=1000,
            num_threads=1
        )
        
        # Run search
        print("Running search with 1 thread...")
        probs = mcts.search()
        move = mcts.select_move()
        print(f"Selected move: {move // 5}, {move % 5}")
        print("Single thread test successful")
        return True
    except Exception as e:
        print(f"Error in single thread test: {e}")
        traceback.print_exc()
        return False

def test_multi_thread():
    """Test with multiple threads to verify the bugfix."""
    print("\nTesting with 4 threads...")
    
    try:
        # Create game
        game = GomokuGame(board_size=5)
        
        # Create MCTS with 4 threads
        mcts = CppMCTSWrapper(
            game=game,
            evaluator=dummy_evaluator,
            c_puct=1.5,
            num_simulations=10,  # Small number for quick test
            dirichlet_alpha=0.3,
            dirichlet_noise_weight=0.25,
            temperature=1.0,
            use_transposition_table=True,
            transposition_table_size=1000,
            num_threads=4
        )
        
        # Run search
        print("Running search with 4 threads...")
        probs = mcts.search()
        move = mcts.select_move()
        print(f"Selected move: {move // 5}, {move % 5}")
        print("Multi-thread test successful")
        return True
    except Exception as e:
        print(f"Error in multi-thread test: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting MCTS threading tests...")
    
    # First test with single thread
    single_thread_success = test_single_thread()
    
    # Then test with multiple threads
    if single_thread_success:
        multi_thread_success = test_multi_thread()
        
        if multi_thread_success:
            print("\nAll tests passed! Bug fix verified.")
        else:
            print("\nMulti-threading test failed!")
    else:
        print("\nSingle thread test failed! Check basic functionality first.")