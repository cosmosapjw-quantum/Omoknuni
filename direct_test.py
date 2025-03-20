#!/usr/bin/env python3
"""
Direct test of MCTS implementation without dependencies.
"""

import sys
import os
import time
import numpy as np
from pprint import pprint

# Add package to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Try to import the required modules
try:
    from alphazero.bindings.cpp_mcts import GomokuMCTS as CppMCTS
    print("Successfully imported CppMCTS")
except ImportError as e:
    print(f"Error importing CppMCTS: {e}")
    sys.exit(1)

def simple_evaluator(board):
    """Simple evaluator that returns uniform policy and neutral value."""
    # Assuming a 5x5 board for simplicity
    board_size = 5
    policy = [1.0 / (board_size * board_size)] * (board_size * board_size)
    value = 0.0
    return policy, value

def main():
    """Test the MCTS implementation directly."""
    # First test with 1 thread
    try:
        print("\nTesting with 1 thread...")
        mcts = CppMCTS(
            num_simulations=50,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_noise_weight=0.25,
            virtual_loss_weight=1.0,
            use_transposition_table=True,
            transposition_table_size=10000,
            num_threads=1
        )
        
        # Create a simple board
        board_size = 5
        board = [0] * (board_size * board_size)
        
        # Run search
        start_time = time.time()
        probs = mcts.search(board, list(range(board_size * board_size)), simple_evaluator)
        end_time = time.time()
        
        print(f"Search completed in {end_time - start_time:.3f} seconds")
        print(f"Probabilities: {dict(sorted(probs.items())[:5])}")
        
        # Test successful
        print("Single-threaded test successful!")
    except Exception as e:
        print(f"Error in single-threaded test: {e}")
        import traceback
        traceback.print_exc()
        
    # Now test with multiple threads
    try:
        print("\nTesting with 4 threads...")
        mcts = CppMCTS(
            num_simulations=50,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_noise_weight=0.25,
            virtual_loss_weight=1.0,
            use_transposition_table=True,
            transposition_table_size=10000,
            num_threads=4
        )
        
        # Create a simple board
        board_size = 5
        board = [0] * (board_size * board_size)
        
        # Run search
        start_time = time.time()
        probs = mcts.search(board, list(range(board_size * board_size)), simple_evaluator)
        end_time = time.time()
        
        print(f"Search completed in {end_time - start_time:.3f} seconds")
        print(f"Probabilities: {dict(sorted(probs.items())[:5])}")
        
        # Test successful
        print("Multi-threaded test successful!")
    except Exception as e:
        print(f"Error in multi-threaded test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()