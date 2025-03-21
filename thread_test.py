#!/usr/bin/env python3
"""
Simple thread test for the improved MCTS bindings
"""
import sys
import time
import numpy as np

# Build the improved bindings
import subprocess
subprocess.check_call([sys.executable, 'improved_setup.py', 'build_ext', '--inplace'])

from alphazero.bindings.improved_cpp_mcts import MCTS

# Create a small test
board_size = 5
board = np.zeros(board_size * board_size, dtype=np.int32)
legal_moves = list(range(board_size * board_size))

# Define a simple evaluator with delay
def evaluator(state):
    # Add a delay to make threading effects obvious
    time.sleep(0.1)  # 100ms delay
    return [1.0/len(state)] * len(state), 0.0

print("Running with 8 threads and 8 simulations...")
mcts = MCTS(
    num_simulations=8,  # Just enough to see threading effects
    num_threads=8,      # Use many threads
    use_transposition_table=False  # Disable for simplicity
)

# Run search
mcts.search(board, legal_moves, evaluator)
print("Done!")