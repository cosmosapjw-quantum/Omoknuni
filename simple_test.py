#!/usr/bin/env python3
"""
Simple test script for the batched MCTS implementation.
"""

import numpy as np
import time
import os
import sys

# Import the necessary modules
import importlib

# Define the classes we might use
BatchedCppMCTSWrapper = None
ImprovedCppMCTSWrapper = None
CppMCTSWrapper = None

# Try to import batched wrapper
try:
    from alphazero.python.mcts.batched_cpp_mcts_wrapper import BatchedCppMCTSWrapper
    print("Batched MCTS available.")
except ImportError:
    print("Batched MCTS not available.")

# Try to import improved wrapper
try:
    from alphazero.python.mcts.improved_cpp_mcts_wrapper import ImprovedCppMCTSWrapper
    print("Improved MCTS available.")
except ImportError:
    print("Improved MCTS not available.")

# Try to import original wrapper
try:
    module = importlib.import_module("alphazero.python.mcts.cpp_mcts_wrapper")
    CppMCTSWrapper = getattr(module, "CppMCTSWrapper")
    print("Original MCTS available.")
except (ImportError, AttributeError):
    print("Original MCTS not available.")

# Import the game
try:
    from alphazero.python.games.gomoku import GomokuGame
except ImportError:
    print("Cannot find GomokuGame class.")
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

def test_improved_mcts():
    print("Testing improved MCTS wrapper...")
    
    # Check if improved or original MCTS is available
    if ImprovedCppMCTSWrapper is None and CppMCTSWrapper is None:
        print("No MCTS implementation available. Skipping test.")
        return
    
    # Create a game
    board_size = 9
    game = GomokuGame(board_size=board_size)
    
    # Create a neural network for evaluation
    nn = DummyNN(board_size)
    
    # Create MCTS
    if ImprovedCppMCTSWrapper is not None:
        mcts = ImprovedCppMCTSWrapper(
            game=game,
            evaluator=nn,
            num_simulations=50,
            num_threads=1
        )
        print("Using improved MCTS wrapper")
    else:
        mcts = CppMCTSWrapper(
            game=game,
            evaluator=nn,
            num_simulations=50,
            num_threads=1
        )
        print("Using original MCTS wrapper")
    
    # Play a few moves
    for i in range(3):
        start_time = time.time()
        move = mcts.select_move()
        elapsed = time.time() - start_time
        
        print(f"Move {i+1}: Selected move {move} in {elapsed:.3f}s")
        
        game.apply_move(move)
        mcts.update_with_move(move)
    
    print("MCTS test completed successfully!")

def test_batched_mcts():
    print("Testing batched MCTS wrapper...")
    
    # Check if batched MCTS is available
    if BatchedCppMCTSWrapper is None:
        print("Batched MCTS not available. Skipping test.")
        return
    
    # Create a game
    board_size = 9
    game = GomokuGame(board_size=board_size)
    
    # Create a neural network for batch evaluation
    nn = DummyNN(board_size)
    
    # Create a batched evaluator function
    def batch_evaluator(games):
        results = []
        for game in games:
            policy, value = nn(game.get_board())
            results.append((policy, value))
        return results
    
    # Create MCTS
    mcts = BatchedCppMCTSWrapper(
        game=game,
        evaluator=batch_evaluator,
        num_simulations=50,
        num_threads=2,
        batch_size=8,
        max_wait_ms=10
    )
    print("Using batched MCTS wrapper")
    
    # Play a few moves
    for i in range(3):
        start_time = time.time()
        move = mcts.select_move()
        elapsed = time.time() - start_time
        
        print(f"Move {i+1}: Selected move {move} in {elapsed:.3f}s")
        
        game.apply_move(move)
        mcts.update_with_move(move)
    
    print("Batched MCTS test completed successfully!")

if __name__ == "__main__":
    print("Running MCTS tests...")
    
    # Test the improved MCTS first
    test_improved_mcts()
    
    print("\n" + "-" * 50 + "\n")
    
    # Then test the batched MCTS
    test_batched_mcts()