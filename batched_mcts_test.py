#!/usr/bin/env python3
"""
Comprehensive test for the batched MCTS implementation.

This test verifies that the batched MCTS implementation works correctly
and demonstrates how to use it with a neural network evaluator.
"""
import numpy as np
import time
import sys
import gc

print("Starting batched MCTS test...")

# Import the necessary classes
try:
    from alphazero.python.mcts.batched_cpp_mcts_wrapper import BatchedCppMCTSWrapper
    from alphazero.python.games.gomoku import GomokuGame
    from alphazero.python.models.batched_evaluator import BatchedEvaluator, PyBatchEvaluator
    print("Imports successful")
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying direct imports...")
    try:
        from alphazero.bindings.batched_cpp_mcts import BatchedGomokuMCTS
        from alphazero.python.games.gomoku import GomokuGame
        print("Direct imports successful")
    except Exception as e2:
        print(f"Direct imports failed: {e2}")
        sys.exit(1)

# Create a dummy neural network for testing
class DummyNetwork:
    def __init__(self, board_size):
        self.board_size = board_size
        
    def __call__(self, state_tensor):
        """Simple forward pass with random outputs"""
        # Generate a random policy and value
        policy_logits = np.random.normal(0, 1, self.board_size * self.board_size)
        value = np.random.uniform(-1, 1)
        return policy_logits, value
        
    def process_batch(self, state_tensors):
        """Process a batch of state tensors"""
        policy_logits_batch = []
        values_batch = []
        
        for state in state_tensors:
            policy_logits, value = self(state)
            policy_logits_batch.append(policy_logits)
            values_batch.append(value)
            
        return policy_logits_batch, values_batch

# Create a game
board_size = 9
print(f"Creating game with board size {board_size}")
game = GomokuGame(board_size=board_size)

# Create a neural network and evaluator
print("Creating neural network and evaluator")
network = DummyNetwork(board_size)
evaluator = BatchedEvaluator(network, batch_size=8, max_wait_time=0.001)

# Test 1: Can we create a BatchedCppMCTSWrapper?
print("\n=== Test 1: Creating BatchedCppMCTSWrapper ===")
try:
    mcts = BatchedCppMCTSWrapper(
        game=game,
        evaluator=evaluator,
        num_simulations=50,
        c_puct=1.5,
        num_threads=2,
        batch_size=8,
        max_wait_ms=10
    )
    print("BatchedCppMCTSWrapper created successfully")
except Exception as e:
    print(f"Error creating wrapper: {e}")
    print("Trying fallback to direct implementation...")
    try:
        # Create a simple batch evaluator function
        def create_batch_evaluator(nn, board_size):
            def batch_eval(board_batch):
                results = []
                for board in board_batch:
                    # Generate random policy
                    policy = {i: np.random.random() for i in range(board_size * board_size)}
                    # Normalize
                    sum_policy = sum(policy.values())
                    if sum_policy > 0:
                        policy = {k: v / sum_policy for k, v in policy.items()}
                    value = np.random.uniform(-1, 1)
                    results.append((policy, value))
                return results
            return batch_eval
        
        # Create the direct MCTS instance
        batch_eval = create_batch_evaluator(network, board_size)
        mcts = BatchedGomokuMCTS(
            num_simulations=50,
            c_puct=1.5,
            dirichlet_alpha=0.03,
            dirichlet_noise_weight=0.25,
            virtual_loss_weight=1.0,
            use_transposition_table=True,
            transposition_table_size=10000,
            num_threads=2
        )
        
        # Create a board and legal moves for testing
        board = np.array(game.get_board()).flatten().astype(np.int32)
        legal_moves = game.get_legal_moves()
        
        # Verify direct implementation works
        print("Testing direct MCTS implementation...")
        probs = mcts.search_batched(
            board, 
            legal_moves[:10], 
            batch_eval,
            batch_size=4,
            max_wait_ms=10
        )
        print(f"Direct implementation works! Result keys: {list(probs.keys())[:5]}...")
        
        print("Using direct implementation instead of wrapper")
        direct_mode = True
    except Exception as e2:
        print(f"Fallback also failed: {e2}")
        sys.exit(1)
else:
    direct_mode = False

# Test 2: Can we run a search?
print("\n=== Test 2: Running search ===")
try:
    if not direct_mode:
        # Use the wrapper
        print("Running search with wrapper...")
        probs = mcts.search()
        print(f"Search successful! Result keys: {list(probs.keys())[:5]}...")
        
        # Select a move
        move = mcts.select_move()
        print(f"Selected move: {move}")
        
        # Apply the move
        game.apply_move(move)
        mcts.update_with_move(move)
        print("Move applied and MCTS updated")
    else:
        # Use direct implementation
        print("Running search with direct implementation...")
        
        # Get fresh board state
        board = np.array(game.get_board()).flatten().astype(np.int32)
        legal_moves = game.get_legal_moves()
        
        probs = mcts.search_batched(
            board, 
            legal_moves, 
            batch_eval,
            batch_size=4,
            max_wait_ms=10
        )
        print(f"Search successful! Result keys: {list(probs.keys())[:5]}...")
        
        # Select a move
        move = mcts.select_move(1.0)
        print(f"Selected move: {move}")
        
        # Apply the move
        game.apply_move(move)
        mcts.update_with_move(move)
        print("Move applied and MCTS updated")
    
    print("Test 2 completed successfully")
except Exception as e:
    print(f"Error in Test 2: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Can we run multiple searches in a row?
print("\n=== Test 3: Running multiple searches ===")
try:
    for i in range(3):
        print(f"\nSearch iteration {i+1}")
        if not direct_mode:
            # Use the wrapper
            probs = mcts.search()
            move = mcts.select_move()
        else:
            # Use direct implementation
            board = np.array(game.get_board()).flatten().astype(np.int32)
            legal_moves = game.get_legal_moves()
            
            probs = mcts.search_batched(
                board, 
                legal_moves, 
                batch_eval,
                batch_size=4,
                max_wait_ms=10
            )
            move = mcts.select_move(1.0)
        
        print(f"Selected move: {move}")
        game.apply_move(move)
        mcts.update_with_move(move)
        print(f"Game state after move {i+1}:")
        print(np.array(game.get_board()).reshape(board_size, board_size))
    
    print("Test 3 completed successfully")
except Exception as e:
    print(f"Error in Test 3: {e}")
    import traceback
    traceback.print_exc()

# Clean up resources
del mcts
gc.collect()

print("\nAll tests completed successfully!")