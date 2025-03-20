#!/usr/bin/env python3
"""
Test script for MCTS class from cpp_mcts module.
"""

import sys
import os
import traceback

def main():
    print(f"Python version: {sys.version}")
    
    # Try to import and use the MCTS class
    print("\nImporting MCTS class...")
    try:
        from alphazero.bindings.cpp_mcts import MCTS
        print("SUCCESS: MCTS class imported!")
        
        # Create an instance and check basic methods
        print("\nCreating MCTS instance...")
        mcts = MCTS(
            num_simulations=100,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_noise_weight=0.25,
            virtual_loss_weight=1.0,
            use_transposition_table=True,
            transposition_table_size=10000,
            num_threads=1
        )
        print("SUCCESS: MCTS instance created!")
        
        # Check some methods
        print("\nTesting basic methods...")
        print(f"Number of simulations: {mcts.get_num_simulations()}")
        mcts.set_num_simulations(200)
        print(f"Updated number of simulations: {mcts.get_num_simulations()}")
        
        # Let's try a simple search
        print("\nTrying a simple search...")
        
        # Define a simple evaluator function
        def simple_evaluator(state_tensor):
            # Return uniform policy and neutral value
            return [1.0 / 25] * 25, 0.0
        
        # Create a simple state tensor (5x5 empty board)
        state_tensor = [0.0] * 25
        
        # Define legal moves (all positions)
        legal_moves = list(range(25))
        
        # Try the search
        try:
            probs = mcts.search(state_tensor, legal_moves, simple_evaluator)
            print("SEARCH SUCCESS!")
            print(f"Probabilities: {dict(list(probs.items())[:5])}...")
        except Exception as e:
            print(f"SEARCH FAILED: {e}")
            traceback.print_exc()
        
    except Exception as e:
        print(f"FAILED: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()