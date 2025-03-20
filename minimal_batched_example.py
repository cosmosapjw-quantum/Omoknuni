#!/usr/bin/env python3
"""
Minimal example of using the batched MCTS implementation.
"""
import numpy as np
import time
import gc

from alphazero.bindings.batched_cpp_mcts import BatchedGomokuMCTS
from alphazero.python.games.gomoku import GomokuGame

print("Starting minimal batched MCTS example...")

# Create a game
board_size = 9
game = GomokuGame(board_size=board_size)
print(f"Created game with board size {board_size}")

# Create a minimal batch evaluator function
def minimal_batch_evaluator(board_batch):
    print(f"Evaluating batch of {len(board_batch)} boards")
    results = []
    
    for _ in range(len(board_batch)):
        # Create uniform policy
        policy = [1.0 / (board_size * board_size)] * (board_size * board_size)
        # Random value
        value = 0.0
        results.append((policy, value))
    
    return results

# Get the initial board state
board = np.array(game.get_board()).flatten().astype(np.int32)
legal_moves = game.get_legal_moves()
print(f"Board shape: {board.shape}, legal moves: {len(legal_moves)}")

# Take only first 5 legal moves for testing
limited_moves = legal_moves[:5]
print(f"Testing with {len(limited_moves)} legal moves: {limited_moves}")

# Create MCTS with minimal settings
print("Creating MCTS...")
mcts = BatchedGomokuMCTS(
    num_simulations=5,        # Very few simulations
    num_threads=1,            # Single thread
    use_transposition_table=False  # No transposition table
)
print("MCTS created")

# Run a search
print("Running search...")
try:
    probs = mcts.search_batched(
        board,
        limited_moves,
        minimal_batch_evaluator,
        batch_size=1,
        max_wait_ms=1
    )
    print(f"Search completed successfully")
    print(f"Probabilities: {probs}")
    
    # Select a move
    move = mcts.select_move(1.0)
    print(f"Selected move: {move}")
    
    # Clean up
    del mcts
    gc.collect()
    
    print("Example completed successfully")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()