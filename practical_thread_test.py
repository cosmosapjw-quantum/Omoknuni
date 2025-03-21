#!/usr/bin/env python3
"""
Practical threading test using the improved MCTS bindings
"""
import time
import numpy as np
import sys
import subprocess

# Rebuild the bindings to ensure we have the latest version
subprocess.check_call([sys.executable, 'improved_setup.py', 'build_ext', '--inplace'])

from alphazero.bindings.improved_cpp_mcts import MCTS

# Parameters for our test
BOARD_SIZE = 9
NUM_SIMULATIONS = 50
EVALUATOR_TIME = 0.01  # Simulating a 10ms neural network inference

# Create a test board
board = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.int32)
legal_moves = list(range(BOARD_SIZE * BOARD_SIZE))

# This evaluator simulates a neural network that takes some time to run
def neural_net_evaluator(state):
    # Simulate neural network inference time
    time.sleep(EVALUATOR_TIME)
    
    # Generate a slightly more realistic policy (not uniform)
    # This makes the central positions more likely
    policy = np.ones(len(state)) / len(state)
    for i in range(len(state)):
        row, col = i // BOARD_SIZE, i % BOARD_SIZE
        # Make central positions more attractive
        distance_from_center = abs(row - BOARD_SIZE // 2) + abs(col - BOARD_SIZE // 2)
        policy[i] *= (1.0 + 0.2 * (BOARD_SIZE - distance_from_center))
    
    # Normalize policy
    policy = policy / policy.sum()
    
    # Random value with slight bias
    value = np.random.normal(0.1, 0.2)
    
    return policy.tolist(), value

def test_mcts(num_threads):
    print(f"\nTesting MCTS with {num_threads} thread(s) and {NUM_SIMULATIONS} simulations...")
    
    # Create MCTS instance
    mcts = MCTS(
        num_simulations=NUM_SIMULATIONS,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25,
        virtual_loss_weight=1.0,
        use_transposition_table=True,
        transposition_table_size=10000,
        num_threads=num_threads
    )
    
    # Run search and time it
    start_time = time.time()
    probs = mcts.search(board, legal_moves, neural_net_evaluator)
    end_time = time.time()
    
    elapsed = end_time - start_time
    
    print(f"Search completed in {elapsed:.3f} seconds")
    
    # Print the top moves
    top_moves = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 moves:")
    for move, prob in top_moves:
        row, col = move // BOARD_SIZE, move % BOARD_SIZE
        print(f"  Position ({row}, {col}): {prob:.3f}")
    
    return elapsed

def main():
    print("Practical test of multithreaded MCTS with simulated neural network")
    print("===============================================================")
    
    # Test with different thread counts
    thread_counts = [1, 2, 4, 8]
    results = {}
    
    for threads in thread_counts:
        elapsed = test_mcts(threads)
        results[threads] = elapsed
    
    # Print summary
    print("\nPerformance summary:")
    print("--------------------")
    print(f"{'Threads':<10} {'Time (s)':<10} {'Speedup':<10}")
    baseline = results[1]
    
    for threads in thread_counts:
        time_taken = results[threads]
        speedup = baseline / time_taken
        print(f"{threads:<10} {time_taken:<10.3f} {speedup:<10.2f}x")
    
    print("\nTheoretical speedup with Amdahl's Law (assuming 5% serial overhead):")
    for threads in thread_counts:
        serial_fraction = 0.05
        theoretical = 1 / (serial_fraction + (1 - serial_fraction) / threads)
        print(f"{threads} threads: {theoretical:.2f}x")

if __name__ == "__main__":
    main()