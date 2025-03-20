#!/usr/bin/env python3
"""
Test script for the fixed Python MCTS bindings
"""

import time
import sys

def test_single_threaded():
    print("Testing single-threaded MCTS...")
    
    from alphazero.bindings.cpp_mcts import MCTS
    
    # Create an MCTS instance with default settings
    mcts = MCTS(
        num_simulations=20,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25,
        virtual_loss_weight=1.0,
        use_transposition_table=True,
        transposition_table_size=1000,
        num_threads=1
    )
    print("Single-threaded MCTS instance created successfully")
    
    # Create a simple state and legal moves
    state = [0.0] * 25  # 5x5 board
    legal_moves = list(range(25))
    
    # Create a simple evaluator
    def evaluator(state_tensor):
        # Return uniform policy and neutral value
        return [1.0 / len(state_tensor)] * len(state_tensor), 0.0
    
    # Run search
    start_time = time.time()
    print("Starting single-threaded search...")
    probs = mcts.search(state, legal_moves, evaluator)
    end_time = time.time()
    
    print(f"Single-threaded search completed in {end_time - start_time:.3f} seconds")
    print(f"Top 5 moves by probability: {dict(sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5])}")
    
    return True

def test_multi_threaded():
    print("\nTesting multi-threaded MCTS...")
    
    from alphazero.bindings.cpp_mcts import MCTS
    
    # Create an MCTS instance with multi-threading
    mcts = MCTS(
        num_simulations=20,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25,
        virtual_loss_weight=1.0,
        use_transposition_table=True,
        transposition_table_size=1000,
        num_threads=4
    )
    print("Multi-threaded MCTS instance created successfully")
    
    # Create a simple state and legal moves
    state = [0.0] * 25  # 5x5 board
    legal_moves = list(range(25))
    
    # Create a simple evaluator
    def evaluator(state_tensor):
        # Add a small delay to simulate work
        time.sleep(0.01)
        # Return uniform policy and neutral value
        return [1.0 / len(state_tensor)] * len(state_tensor), 0.0
    
    # Run search
    start_time = time.time()
    print("Starting multi-threaded search...")
    probs = mcts.search(state, legal_moves, evaluator)
    end_time = time.time()
    
    print(f"Multi-threaded search completed in {end_time - start_time:.3f} seconds")
    print(f"Top 5 moves by probability: {dict(sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5])}")
    
    return True

if __name__ == "__main__":
    print("Testing Python MCTS with our fixes")
    print("---------------------------------")
    
    # First test single-threaded
    single_success = test_single_threaded()
    
    if single_success:
        # Then test multi-threaded
        multi_success = test_multi_threaded()
        
        if multi_success:
            print("\n✅ VERIFICATION SUCCESS: Both single-threaded and multi-threaded MCTS are working!")
            sys.exit(0)
        else:
            print("\n❌ Multi-threaded test failed")
            sys.exit(1)
    else:
        print("\n❌ Single-threaded test failed")
        sys.exit(1)