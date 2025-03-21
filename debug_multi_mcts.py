#!/usr/bin/env python3
"""
Debug script for MCTS class with multiple threads.
"""

import sys
import time
import traceback

# Import MCTS globally
print("About to import MCTS class...")
from alphazero.bindings.cpp_mcts import MCTS
print("Successfully imported MCTS class")

def main():
    print(f"Python version: {sys.version}")
    
    try:
        print("MCTS class imported successfully!")
        
        # First test with single thread as baseline
        print("\n=== Single-threaded test ===")
        test_mcts(num_threads=1)
        
        # Now test with multiple threads
        print("\n=== Multi-threaded test (4 threads) ===")
        test_mcts(num_threads=4)
        
    except Exception as e:
        print(f"SETUP FAILED: {e}")
        traceback.print_exc()

def test_mcts(num_threads):
    try:
        # Create instance
        print(f"Creating MCTS instance with {num_threads} threads...")
        mcts = MCTS(
            num_simulations=20,  # More simulations for multi-threading
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_noise_weight=0.25,
            virtual_loss_weight=1.0,
            use_transposition_table=True,  # Enable TT to test both fixes
            transposition_table_size=100,
            num_threads=num_threads
        )
        print("SUCCESS: MCTS instance created!")
        
        # Define a simple evaluator
        eval_count = 0
        def simple_evaluator(state_tensor):
            nonlocal eval_count
            eval_count += 1
            # Simulate some work
            time.sleep(0.01)
            return [1.0 / len(state_tensor)] * len(state_tensor), 0.0
        
        # Create a state tensor
        state_tensor = [0.0] * 25  # 5x5 board
        legal_moves = list(range(25))
        
        # Now try the search with timing
        print(f"Starting search with {num_threads} threads...")
        start_time = time.time()
        
        probs = mcts.search(state_tensor, legal_moves, simple_evaluator)
        end_time = time.time()
        
        print(f"SEARCH COMPLETED in {end_time - start_time:.3f} seconds!")
        print(f"Evaluator called {eval_count} times")
        print(f"Top 5 moves by probability: {dict(sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5])}")
        
        return True
        
    except Exception as e:
        print(f"TEST FAILED: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()