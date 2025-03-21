#!/usr/bin/env python3
"""
Benchmark script for comparing single-threaded and multi-threaded MCTS performance.
"""

import sys
import time
import numpy as np

print("Benchmarking MCTS performance with different thread counts")
print("--------------------------------------------------------")

try:
    print("\nImporting the improved MCTS module...")
    from alphazero.bindings.improved_cpp_mcts import MCTS
    print("Successfully imported MCTS module")
    
    # Testing parameters
    NUM_SIMULATIONS = 100  # More simulations to see the effect
    EVALUATOR_DELAY = 0.02  # 20ms delay to simulate a neural network evaluation
    
    # Create a game board
    board_size = 9
    board = np.zeros(board_size * board_size, dtype=np.int32)
    legal_moves = list(range(board_size * board_size))
    
    # Define an evaluator with a delay to simulate neural network evaluation
    def evaluator(state):
        # Add a delay to simulate neural network processing
        time.sleep(EVALUATOR_DELAY)
        # Return uniform policy and value 0
        policy = [1.0 / len(state)] * len(state)
        value = 0.0
        return policy, value
    
    # Run benchmarks with different thread counts
    thread_counts = [1, 2, 4, 8]
    results = {}
    
    for threads in thread_counts:
        print(f"\nTesting with {threads} thread(s)...")
        
        # Create MCTS instance
        mcts = MCTS(
            num_simulations=NUM_SIMULATIONS,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_noise_weight=0.25,
            virtual_loss_weight=1.0,
            use_transposition_table=True,
            transposition_table_size=1000,
            num_threads=threads
        )
        
        # Run search with timing
        start_time = time.time()
        mcts.search(board, legal_moves, evaluator)
        end_time = time.time()
        
        elapsed = end_time - start_time
        results[threads] = elapsed
        
        print(f"Completed in {elapsed:.3f} seconds")
    
    # Print summary
    print("\nPerformance summary:")
    print("--------------------")
    print(f"{'Threads':<10} {'Time (s)':<10} {'Speedup':<10}")
    baseline = results[1]  # Single-threaded time
    
    for threads in thread_counts:
        time_taken = results[threads]
        speedup = baseline / time_taken
        print(f"{threads:<10} {time_taken:<10.3f} {speedup:<10.2f}x")
    
    # Calculate theoretical maximum speedup based on Amdahl's law
    # Assuming the evaluator is the parallelizable part
    serial_fraction = 0.05  # 5% serial overhead estimate
    
    print("\nTheoretical vs. Actual speedup:")
    print("------------------------------")
    print(f"{'Threads':<10} {'Theoretical':<15} {'Actual':<10}")
    
    for threads in thread_counts:
        theoretical = 1 / (serial_fraction + (1 - serial_fraction) / threads)
        actual = baseline / results[threads]
        print(f"{threads:<10} {theoretical:<15.2f}x {actual:<10.2f}x")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()