#!/usr/bin/env python3
"""
Minimal test for the batched MCTS implementation.
This test focuses only on basic imports and initialization to debug memory issues.
"""
import numpy as np
import time
import sys
import os
import gc

print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

# First, let's try importing just the module without using it
try:
    print("\n=== Testing module import only ===")
    import alphazero.bindings.batched_cpp_mcts
    print("Module import successful")
except Exception as e:
    print(f"Module import failed: {e}")
    sys.exit(1)

# Now try getting the classes
try:
    print("\n=== Testing class imports ===")
    from alphazero.bindings.batched_cpp_mcts import BatchedGomokuMCTS, BatchEvaluator
    print("Class imports successful")
except Exception as e:
    print(f"Class imports failed: {e}")
    sys.exit(1)

# Initialize a game
try:
    print("\n=== Testing game initialization ===")
    from alphazero.python.games.gomoku import GomokuGame
    board_size = 9
    game = GomokuGame(board_size=board_size)
    print("Game created successfully")
except Exception as e:
    print(f"Game creation failed: {e}")
    sys.exit(1)

# Define a simple batch evaluation function
def simple_batch_evaluator(board_batch):
    print(f"Evaluator called with batch of size {len(board_batch)}")
    results = []
    for _ in range(len(board_batch)):
        # Create uniform policy and random value
        policy = [1.0 / (board_size * board_size)] * (board_size * board_size)
        value = 0.0
        results.append((policy, value))
    return results

# First test: Can we create a BatchedGomokuMCTS object?
try:
    print("\n=== Testing MCTS creation ===")
    mcts = BatchedGomokuMCTS(
        num_simulations=1,  # Minimal simulations
        num_threads=1,      # Single thread
        use_transposition_table=False  # Disable TT for simplicity
    )
    print("MCTS created successfully")
except Exception as e:
    print(f"MCTS creation failed: {e}")
    sys.exit(1)

# Second test: Can we get a board and legal moves?
try:
    print("\n=== Testing board extraction ===")
    board = np.array(game.get_board()).flatten().astype(np.int32)
    legal_moves = game.get_legal_moves()
    print(f"Board shape: {board.shape}, Legal moves: {len(legal_moves)}")
except Exception as e:
    print(f"Board extraction failed: {e}")
    sys.exit(1)

# Test 1: Minimal search with a single legal move
try:
    print("\n=== Test 1: Minimal search with single legal move ===")
    probs = mcts.search_batched(
        board,
        legal_moves[:1],  # Just use first legal move
        simple_batch_evaluator,
        batch_size=1,     # Batch size of 1
        max_wait_ms=1     # Minimal wait time
    )
    print("Search completed!")
    print(f"Result: {probs}")
except Exception as e:
    print(f"Test 1 failed: {e}")
    import traceback
    traceback.print_exc()

# Clean up between tests
del mcts
gc.collect()

# Test 2: Try with a few more legal moves but still single-threaded
try:
    print("\n=== Test 2: Search with multiple legal moves ===")
    mcts = BatchedGomokuMCTS(
        num_simulations=5,  # A few simulations
        num_threads=1,      # Single thread
        use_transposition_table=False  # Disable TT for simplicity
    )
    probs = mcts.search_batched(
        board,
        legal_moves[:5],  # Use 5 legal moves
        simple_batch_evaluator,
        batch_size=1,     # Batch size of 1
        max_wait_ms=1     # Minimal wait time
    )
    print("Search completed!")
    print(f"Result keys: {list(probs.keys())}")
except Exception as e:
    print(f"Test 2 failed: {e}")
    import traceback
    traceback.print_exc()

# Clean up between tests
del mcts
gc.collect()

# Test 3: Multiple legal moves with a larger batch size
try:
    print("\n=== Test 3: Search with larger batch size ===")
    mcts = BatchedGomokuMCTS(
        num_simulations=10,  # More simulations
        num_threads=1,       # Still single thread
        use_transposition_table=False
    )
    probs = mcts.search_batched(
        board,
        legal_moves[:10],  # Use 10 legal moves
        simple_batch_evaluator,
        batch_size=4,      # Larger batch size
        max_wait_ms=10     # Longer wait time
    )
    print("Search completed!")
    print(f"Result keys: {list(probs.keys())}")
except Exception as e:
    print(f"Test 3 failed: {e}")
    import traceback
    traceback.print_exc()

# Clean up between tests
del mcts
gc.collect()

# Test 4: Multi-threaded search
try:
    print("\n=== Test 4: Multi-threaded search ===")
    mcts = BatchedGomokuMCTS(
        num_simulations=10,  # More simulations
        num_threads=2,       # Two threads
        use_transposition_table=False
    )
    probs = mcts.search_batched(
        board,
        legal_moves[:10],  # Use 10 legal moves
        simple_batch_evaluator,
        batch_size=4,      # Larger batch size
        max_wait_ms=10     # Longer wait time
    )
    print("Search completed!")
    print(f"Result keys: {list(probs.keys())}")
except Exception as e:
    print(f"Test 4 failed: {e}")
    import traceback
    traceback.print_exc()

# Clean up
del mcts
gc.collect()

print("\n=== All tests completed ===")