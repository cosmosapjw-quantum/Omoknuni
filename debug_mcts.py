#!/usr/bin/env python3
"""
Debug script for MCTS class to find where it's hanging.
"""

import sys
import time
import traceback

def main():
    print(f"Python version: {sys.version}")
    
    try:
        print("Importing MCTS class...")
        from alphazero.bindings.cpp_mcts import MCTS
        print("SUCCESS: MCTS class imported!")
        
        # Create instance with minimal settings and only 1 simulation
        print("Creating MCTS instance...")
        mcts = MCTS(
            num_simulations=1,  # Just do 1 simulation to see if it works at all
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_noise_weight=0.25,
            virtual_loss_weight=1.0,
            use_transposition_table=False,  # Disable transposition table for simplicity
            transposition_table_size=10,
            num_threads=1  # Just 1 thread for now
        )
        print("SUCCESS: MCTS instance created!")
        
        # Define a very simple evaluator that returns immediately
        def minimal_evaluator(state_tensor):
            print("Evaluator called!")
            return [1.0 / len(state_tensor)] * len(state_tensor), 0.0
        
        # Create a tiny state tensor
        print("Creating state tensor...")
        state_tensor = [0.0] * 9  # Just a 3x3 board
        
        # Define minimal legal moves
        print("Defining legal moves...")
        legal_moves = list(range(9))
        
        # Now try the search with timing
        print("\nStarting search...")
        start_time = time.time()
        
        try:
            print("Calling mcts.search()...")
            probs = mcts.search(state_tensor, legal_moves, minimal_evaluator)
            end_time = time.time()
            
            print(f"SEARCH COMPLETED in {end_time - start_time:.3f} seconds!")
            print(f"Probabilities: {probs}")
            
        except Exception as e:
            print(f"SEARCH FAILED: {e}")
            traceback.print_exc()

    except Exception as e:
        print(f"SETUP FAILED: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()