#!/usr/bin/env python3
"""
Multi-threaded test for Python MCTS bindings
"""

import sys
import time

print("Starting multi-threaded Python MCTS test...")
print(f"Python version: {sys.version}")

try:
    print("\nStep 1: Importing the MCTS class...")
    from alphazero.bindings.cpp_mcts import MCTS
    print("Success: MCTS imported")
    
    print("\nStep 2: Creating multi-threaded MCTS instance...")
    mcts = MCTS(
        num_simulations=10,  # More simulations for multi-threading
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25,
        virtual_loss_weight=1.0,
        use_transposition_table=True,  # Enable transposition table
        transposition_table_size=100,
        num_threads=4  # Multi-threaded
    )
    print("Success: Multi-threaded MCTS instance created")
    
    print("\nStep 3: Preparing for search...")
    # Small state
    state = [0.0] * 9  # 3x3 board
    legal_moves = list(range(9))  # All positions are legal
    
    # Evaluator with a small delay to simulate work
    # Use a list to allow modification inside the function
    eval_count = [0]
    def debug_evaluator(state_tensor):
        eval_count[0] += 1
        print(f"  Evaluator called ({eval_count[0]})")
        time.sleep(0.01)  # Small delay
        return [1.0/9] * 9, 0.0
    
    print("\nStep 4: Starting multi-threaded search...")
    start_time = time.time()
    probs = mcts.search(state, legal_moves, debug_evaluator)
    end_time = time.time()
    
    print(f"Success: Search completed in {end_time - start_time:.3f} seconds")
    print(f"Evaluator called {eval_count[0]} times")
    print(f"Probabilities: {dict(sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3])}")
    
    print("\nStep 5: Done!")
    print("All tests passed")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()