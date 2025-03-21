#!/usr/bin/env python3
"""
Test script for the BatchedCppMCTSWrapper.
"""

import numpy as np
import time
import sys

print("Starting test...")

# Create a dummy neural network for testing
class DummyNN:
    def __init__(self, board_size):
        self.board_size = board_size

    def __call__(self, board):
        # Generate random policy
        policy = {i: np.random.random() for i in range(self.board_size * self.board_size)}
        # Normalize policy
        sum_policy = sum(policy.values())
        if sum_policy > 0:
            policy = {k: v / sum_policy for k, v in policy.items()}
        value = np.random.random() * 2 - 1  # Random value between -1 and 1
        return policy, value

# Function to create a batched evaluator
def create_batch_evaluator(nn):
    def batch_evaluator(games):
        results = []
        for game in games:
            policy, value = nn(game.get_board())
            results.append((policy, value))
        return results
    return batch_evaluator

# Try importing the required modules
try:
    from alphazero.python.games.gomoku import GomokuGame
    print("GomokuGame imported successfully")
except ImportError as e:
    print(f"Failed to import GomokuGame: {e}")
    sys.exit(1)

try:
    from alphazero.python.mcts.batched_cpp_mcts_wrapper import BatchedCppMCTSWrapper
    print("BatchedCppMCTSWrapper imported successfully")
except ImportError as e:
    print(f"Failed to import BatchedCppMCTSWrapper: {e}")
    sys.exit(1)

# Create a game
board_size = 9
print(f"Creating game with board size {board_size}")
game = GomokuGame(board_size=board_size)

# Create a neural network and batched evaluator
print("Creating neural network and batched evaluator")
nn = DummyNN(board_size)
batch_eval = create_batch_evaluator(nn)

# Create the batched MCTS wrapper
print("Creating batched MCTS wrapper")
try:
    mcts = BatchedCppMCTSWrapper(
        game=game,
        evaluator=batch_eval,
        num_simulations=50,
        num_threads=2,
        batch_size=8,
        max_wait_ms=10
    )
    print("Batched MCTS wrapper created successfully")
except Exception as e:
    print(f"Failed to create batched MCTS wrapper: {e}")
    sys.exit(1)

# Test the wrapper
print("Testing batched MCTS wrapper...")
for i in range(3):
    try:
        print(f"Starting move selection {i+1}...")
        start_time = time.time()
        move = mcts.select_move()
        elapsed = time.time() - start_time
        print(f"Selected move {move} in {elapsed:.3f} seconds")
        
        print(f"Applying move {move} to game")
        game.apply_move(move)
        
        print(f"Updating MCTS with move {move}")
        mcts.update_with_move(move)
        
        print(f"Move {i+1} completed successfully")
    except Exception as e:
        print(f"Error during move {i+1}: {e}")
        sys.exit(1)

print("Batched MCTS test completed successfully!")