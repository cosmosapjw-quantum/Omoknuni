#!/usr/bin/env python3
"""
Minimal test script for the MCTS implementation.
"""

import numpy as np
import time
import sys

# First, let's check if we can import the wrappers
print("Checking imports...")

try:
    from alphazero.python.mcts.improved_cpp_mcts_wrapper import ImprovedCppMCTSWrapper
    print("✓ ImprovedCppMCTSWrapper imported successfully")
    have_improved = True
except ImportError as e:
    print(f"✗ ImprovedCppMCTSWrapper import failed: {e}")
    have_improved = False

try:
    from alphazero.python.games.gomoku import GomokuGame
    print("✓ GomokuGame imported successfully")
    have_game = True
except ImportError as e:
    print(f"✗ GomokuGame import failed: {e}")
    have_game = False

# If we couldn't import the required modules, exit
if not (have_improved and have_game):
    print("Required modules not available. Exiting.")
    sys.exit(1)

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

def test_simple_mcts():
    print("\nRunning simple MCTS test...")
    
    # Create a game
    board_size = 9
    print(f"Creating game with board size {board_size}")
    game = GomokuGame(board_size=board_size)
    
    # Create a neural network for evaluation
    print("Creating neural network")
    nn = DummyNN(board_size)
    
    # Create MCTS
    print("Creating MCTS with 50 simulations and 1 thread")
    mcts = ImprovedCppMCTSWrapper(
        game=game,
        evaluator=nn,
        num_simulations=50,
        num_threads=1
    )
    
    # Play a few moves
    print("Playing moves...")
    for i in range(3):
        start_time = time.time()
        print(f"  Selecting move {i+1}...")
        move = mcts.select_move()
        elapsed = time.time() - start_time
        
        print(f"  Move {i+1}: Selected move {move} in {elapsed:.3f}s")
        
        print(f"  Applying move {move} to game")
        game.apply_move(move)
        
        print(f"  Updating MCTS with move {move}")
        mcts.update_with_move(move)
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_simple_mcts()