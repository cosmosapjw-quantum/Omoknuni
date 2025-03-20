#!/usr/bin/env python3
"""
Corrected test script for MCTS class from cpp_mcts module.
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
        
        # Create an instance
        print("\nCreating MCTS instance...")
        mcts = MCTS(
            num_simulations=50,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_noise_weight=0.25,
            virtual_loss_weight=1.0,
            use_transposition_table=True,
            transposition_table_size=1000,
            num_threads=1
        )
        print("SUCCESS: MCTS instance created!")
        
        # Test search with single thread
        print("\nTesting single-threaded search...")
        test_search(mcts)
        
        # Now test with multiple threads
        print("\nCreating multi-threaded MCTS instance...")
        multi_mcts = MCTS(
            num_simulations=50,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_noise_weight=0.25,
            virtual_loss_weight=1.0,
            use_transposition_table=True,
            transposition_table_size=1000,
            num_threads=4
        )
        print("SUCCESS: Multi-threaded MCTS instance created!")
        
        # Test search with multiple threads
        print("\nTesting multi-threaded search...")
        test_search(multi_mcts)
        
        print("\nAll tests passed successfully!")

    except Exception as e:
        print(f"FAILED: {e}")
        traceback.print_exc()

def test_search(mcts):
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
        
        # Test move selection
        move = mcts.select_move()
        print(f"Selected move: {move}")
        
        # Test updating with the move
        mcts.update_with_move(move)
        print("Successfully updated with move")
        
        return True
    except Exception as e:
        print(f"SEARCH FAILED: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()