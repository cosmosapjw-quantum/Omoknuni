#!/usr/bin/env python3
"""
Minimal test for Python MCTS bindings
"""

import sys
import time

print("Starting minimal Python MCTS test...")
print(f"Python version: {sys.version}")

try:
    print("\nStep 1: Importing the MCTS class...")
    from alphazero.bindings.cpp_mcts import MCTS
    print("Success: MCTS imported")
    
    print("\nStep 2: Creating MCTS instance...")
    mcts = MCTS(
        num_simulations=1,  # Just 1 simulation for speed
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25,
        virtual_loss_weight=1.0,
        use_transposition_table=False,  # Disable for simplicity
        transposition_table_size=10,
        num_threads=1  # Single-threaded for now
    )
    print("Success: MCTS instance created")
    
    print("\nStep 3: Preparing for search...")
    # Very small state
    state = [0.0] * 4  # Tiny 2x2 board
    legal_moves = [0, 1, 2, 3]  # All positions are legal
    
    # Super simple evaluator that just returns constants
    def debug_evaluator(state_tensor):
        print("  Evaluator called")
        return [0.25, 0.25, 0.25, 0.25], 0.0
    
    print("\nStep 4: Starting search...")
    start_time = time.time()
    probs = mcts.search(state, legal_moves, debug_evaluator)
    end_time = time.time()
    
    print(f"Success: Search completed in {end_time - start_time:.3f} seconds")
    print(f"Probabilities: {probs}")
    
    print("\nStep 5: Done!")
    print("All tests passed")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()