# Modified file: alphazero/examples/test_batch_evaluation.py

import sys
import os
import time
import torch
import torch.nn.functional as F
import numpy as np

# Add package to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alphazero.python.games.gomoku import GomokuGame
from alphazero.python.models.simple_conv_net import SimpleConvNet
from alphazero.python.mcts.batch_evaluator import BatchEvaluator

def test_batch_evaluation():
    # Create network
    board_size = 9
    network = SimpleConvNet(board_size=board_size, input_channels=3, num_filters=32)
    device = next(network.parameters()).device
    network.eval()
    
    # Generate random states
    num_states = 100
    game_states = []
    tensor_states = []
    
    print(f"Generating {num_states} random game states...")
    for _ in range(num_states):
        # Create random game state
        game = GomokuGame(board_size=board_size)
        
        # Make random moves
        num_moves = np.random.randint(0, 10)
        for _ in range(num_moves):
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break
            move = np.random.choice(legal_moves)
            game.apply_move(move)
        
        # Store game and tensor state
        game_states.append(game)
        tensor_states.append(game.get_state_tensor().flatten().tolist())
    
    # Create batch evaluator
    batch_evaluator = BatchEvaluator(network, batch_size=16)
    
    # Test sequential evaluation
    print("Testing sequential evaluation...")
    start_time = time.time()
    
    seq_policies = []
    seq_values = []
    
    for game in game_states:
        # Use the existing predict method
        policy, value = network.predict(game)
        seq_policies.append(policy)
        seq_values.append(value)
    
    seq_time = time.time() - start_time
    print(f"Sequential evaluation: {seq_time:.4f} seconds")
    
    # Test batch evaluation
    print("Testing batch evaluation...")
    start_time = time.time()
    
    batch_policies, batch_values = batch_evaluator.evaluate_batch(tensor_states)
    
    batch_time = time.time() - start_time
    print(f"Batch evaluation: {batch_time:.4f} seconds")
    
    # Calculate speedup
    speedup = seq_time / batch_time
    print(f"Speedup: {speedup:.2f}x")
    
    # Verify results are similar
    print("\nVerifying results...")
    
    # Convert the first few batch policy results back to dictionaries for comparison
    flat_policy = batch_policies[0]
    batch_policy_dict = {}
    for move, prob in enumerate(flat_policy):
        if prob > 0:
            batch_policy_dict[move] = prob
    
    # Normalize
    total = sum(batch_policy_dict.values())
    if total > 0:
        batch_policy_dict = {k: v / total for k, v in batch_policy_dict.items()}
    
    # Compare with sequential
    print(f"First policy (sequential): {list(seq_policies[0].items())[:5]}")
    print(f"First policy (batch): {list(batch_policy_dict.items())[:5]}")
    print(f"First value (sequential): {seq_values[0]}")
    print(f"First value (batch): {batch_values[0]}")
    
    return {
        "sequential_time": seq_time,
        "batch_time": batch_time,
        "speedup": speedup
    }

if __name__ == "__main__":
    test_batch_evaluation()