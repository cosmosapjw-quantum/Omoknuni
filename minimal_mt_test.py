#!/usr/bin/env python3
"""
Minimal test for multi-threaded MCTS
"""

import sys
import time

# Import the module
from alphazero.bindings.cpp_mcts import MCTS

# Define a simple evaluator
def simple_evaluator(state_tensor):
    print("Evaluator called")
    # Just return uniform policy and neutral value
    return [1.0 / len(state_tensor)] * len(state_tensor), 0.0

# Main function
def main():
    print("Creating single-threaded MCTS instance...")
    mcts_single = MCTS(
        num_simulations=5,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25,
        virtual_loss_weight=1.0,
        use_transposition_table=True,
        transposition_table_size=100,
        num_threads=1
    )
    print("Created single-threaded MCTS")
    
    # Simple state and moves
    state = [0.0] * 9  # 3x3 board
    moves = list(range(9))
    
    # Single-threaded test
    print("\nRunning single-threaded search...")
    start = time.time()
    result = mcts_single.search(state, moves, simple_evaluator)
    end = time.time()
    print(f"Single-threaded search completed in {end - start:.3f} seconds")
    print(f"Result: {result}")
    
    print("\nCreating multi-threaded MCTS instance...")
    mcts_multi = MCTS(
        num_simulations=5,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25,
        virtual_loss_weight=1.0,
        use_transposition_table=True,
        transposition_table_size=100,
        num_threads=4
    )
    print("Created multi-threaded MCTS")
    
    # Multi-threaded test
    print("\nRunning multi-threaded search...")
    start = time.time()
    result = mcts_multi.search(state, moves, simple_evaluator)
    end = time.time()
    print(f"Multi-threaded search completed in {end - start:.3f} seconds")
    print(f"Result: {result}")

if __name__ == "__main__":
    main()