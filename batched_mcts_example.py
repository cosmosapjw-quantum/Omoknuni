#!/usr/bin/env python3
"""
Example script demonstrating how to use the batched MCTS implementation 
directly with alphazero.bindings.batched_cpp_mcts

This example shows the most efficient approach by avoiding the Python wrapper
and directly using the C++ bindings, which minimizes GIL acquisitions and
maximizes GPU utilization.
"""

import numpy as np
import time
import gc

from alphazero.bindings.batched_cpp_mcts import BatchedGomokuMCTS
from alphazero.python.games.gomoku import GomokuGame

print("Starting batched MCTS example...")

# Create a game
board_size = 9
game = GomokuGame(board_size=board_size)
print(f"Created Gomoku game with board size {board_size}")

# Define a batch evaluation function that simulates a neural network
def batch_evaluator(board_batch):
    """
    This function demonstrates how to implement a batch evaluator for MCTS.
    In a real AlphaZero implementation, this would use your neural network
    to evaluate the board positions in a batch.
    
    Args:
        board_batch: A list of board states to evaluate
        
    Returns:
        A list of tuples (policy, value) for each board
    """
    print(f"Evaluating batch of {len(board_batch)} positions")
    
    # Start timing the batch evaluation
    start_time = time.time()
    
    results = []
    for board in board_batch:
        # In a real implementation, you would process all boards at once
        # with your neural network. Here we simulate with random values.
        
        # Create a policy (probabilities for each move)
        total_cells = board_size * board_size
        policy = np.zeros(total_cells)
        
        # Only assign probabilities to empty cells
        empty_indices = []
        for i in range(min(len(board), total_cells)):
            if board[i] == 0:  # Empty cell
                empty_indices.append(i)
        
        if empty_indices:
            # Assign random probabilities to empty cells
            for idx in empty_indices:
                policy[idx] = np.random.random()
                
            # Normalize probabilities to sum to 1
            policy_sum = np.sum(policy)
            if policy_sum > 0:
                policy = policy / policy_sum
        else:
            # If no empty cells, use uniform distribution
            policy = np.ones(total_cells) / total_cells
        
        # Convert to list for C++ compatibility
        policy_list = policy.tolist()
        
        # Generate a random value between -1 and 1
        value = np.random.uniform(-1, 1)
        
        results.append((policy_list, value))
    
    # Calculate time taken
    elapsed = time.time() - start_time
    print(f"Batch evaluation completed in {elapsed:.6f} seconds")
    
    return results

# Create the MCTS with batched search
print("Creating batched MCTS...")
mcts = BatchedGomokuMCTS(
    num_simulations=50,       # Number of simulations per search (reduced for this example)
    c_puct=1.5,               # Exploration constant
    dirichlet_alpha=0.03,     # Dirichlet noise alpha for root
    dirichlet_noise_weight=0.25,  # Weight of noise at root
    virtual_loss_weight=1.0,  # Weight of virtual loss (1.0 is standard)
    use_transposition_table=True,  # Use transposition table for state caching
    transposition_table_size=10000,  # Size of transposition table (reduced for this example)
    num_threads=2             # Number of parallel search threads (reduced for this example)
)
print("Batched MCTS created successfully")

# Play a few moves to demonstrate usage
print("\nStarting game simulation...")
for move_num in range(5):
    print(f"\nMove {move_num + 1}")
    
    # Get the current game state
    board = np.array(game.get_board()).flatten().astype(np.int32)
    legal_moves = game.get_legal_moves()
    
    print(f"Board has {len(legal_moves)} legal moves")
    
    # Run the batched search
    start_time = time.time()
    
    # Make sure the legal moves are valid
    filtered_legal_moves = [move for move in legal_moves if 0 <= move < board_size * board_size]
    
    # Use only the first 10 legal moves for this example to keep it fast
    limited_legal_moves = filtered_legal_moves[:10]
    print(f"Using {len(limited_legal_moves)} out of {len(filtered_legal_moves)} legal moves")
    
    probs = mcts.search_batched(
        board,
        limited_legal_moves,
        batch_evaluator,
        batch_size=4,      # Small batch size for this example
        max_wait_ms=1      # Minimal wait time for testing
    )
    elapsed = time.time() - start_time
    print(f"Search completed in {elapsed:.3f} seconds")
    
    # Select a move based on the search probabilities
    move = mcts.select_move(temperature=1.0)
    print(f"Selected move: {move} (row={move // board_size}, col={move % board_size})")
    
    # Apply the move and update the tree
    game.apply_move(move)
    mcts.update_with_move(move)
    
    # Print the current board
    print("Current board state:")
    print(np.array(game.get_board()).reshape(board_size, board_size))

print("\nExample completed successfully!")

# Clean up resources
del mcts
gc.collect()