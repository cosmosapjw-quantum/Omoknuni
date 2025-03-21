# Create file: alphazero/examples/simple_benchmark_mcts.py

import sys
import os
import time
import torch
import numpy as np

# Add package to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alphazero.python.games.gomoku import GomokuGame
from alphazero.python.models.simple_conv_net import SimpleConvNet
from alphazero.python.mcts.mcts import MCTS
from alphazero.python.mcts.batch_evaluator import BatchEvaluator

# Create a wrapper for direct batch processing of states
class DirectEvaluator:
    def __init__(self, network, board_size):
        self.network = network
        self.board_size = board_size
        self.batch_evaluator = BatchEvaluator(network, batch_size=16)
    
    def __call__(self, game_state):
        """Standard interface for game state evaluation."""
        state_tensor = game_state.get_state_tensor()
        state_flat = state_tensor.flatten().tolist()
        
        # Use batch evaluation with batch size 1
        policies, values = self.batch_evaluator.evaluate_batch([state_flat])
        
        if not policies or not values:
            return {}, 0.0
        
        # Convert flat policy to dictionary
        policy_dict = {}
        legal_moves = game_state.get_legal_moves()
        
        for move in legal_moves:
            if 0 <= move < len(policies[0]):
                policy_dict[move] = policies[0][move]
        
        # Normalize
        total = sum(policy_dict.values())
        if total > 0:
            policy_dict = {k: v / total for k, v in policy_dict.items()}
        elif legal_moves:
            policy_dict = {move: 1.0 / len(legal_moves) for move in legal_moves}
        
        return policy_dict, values[0]

def test_mcts_performance():
    # Parameters
    board_size = 9
    num_simulations = 100
    num_moves = 3
    
    # Create neural network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    network = SimpleConvNet(
        board_size=board_size,
        input_channels=3,
        num_filters=32,
        num_residual_blocks=3
    ).to(device)
    network.eval()
    
    # Test standard evaluator
    print("\nTesting standard evaluation:")
    game = GomokuGame(board_size=board_size)
    
    # Standard MCTS
    standard_mcts = MCTS(
        game=game,
        evaluator=network.predict,
        c_puct=1.5,
        num_simulations=num_simulations,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25,
        temperature=1.0
    )
    
    start_time = time.time()
    for _ in range(num_moves):
        move = standard_mcts.select_move()
        game.apply_move(move)
        standard_mcts.update_with_move(move)
    standard_time = time.time() - start_time
    
    print(f"Standard MCTS: {standard_time:.3f} seconds for {num_moves} moves")
    print(f"Simulations per second: {(num_moves * num_simulations) / standard_time:.1f}")
    
    # Test batch evaluator
    print("\nTesting batch evaluation:")
    game = GomokuGame(board_size=board_size)
    
    # Create batch evaluator
    direct_evaluator = DirectEvaluator(network, board_size)
    
    # MCTS with batch evaluator
    batch_mcts = MCTS(
        game=game,
        evaluator=direct_evaluator,
        c_puct=1.5,
        num_simulations=num_simulations,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25,
        temperature=1.0
    )
    
    start_time = time.time()
    for _ in range(num_moves):
        move = batch_mcts.select_move()
        game.apply_move(move)
        batch_mcts.update_with_move(move)
    batch_time = time.time() - start_time
    
    print(f"Batch MCTS: {batch_time:.3f} seconds for {num_moves} moves")
    print(f"Simulations per second: {(num_moves * num_simulations) / batch_time:.1f}")
    
    # Calculate speedup
    speedup = standard_time / batch_time
    print(f"\nSpeedup: {speedup:.2f}x")

if __name__ == "__main__":
    test_mcts_performance()