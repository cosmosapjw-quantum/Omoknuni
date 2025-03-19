#!/usr/bin/env python3
"""
Test script to compare the performance of the original MCTS with the enhanced version.
"""

import sys
import os
import time
import torch
import numpy as np
from typing import Dict, Tuple
import argparse

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alphazero.python.games.gomoku import GomokuGame
from alphazero.python.mcts.mcts import MCTS
from alphazero.python.mcts.enhanced_mcts import EnhancedMCTS
from alphazero.python.models.simple_conv_net import SimpleConvNet
from alphazero.python.models.batched_evaluator import BatchedEvaluator


def print_board(board):
    """Print the board state in a more readable format."""
    symbols = {0: ".", 1: "●", 2: "○"}
    board_size = len(board)
    
    print("   ", end="")
    for i in range(board_size):
        print(f"{i:2d}", end=" ")
    print()
    
    for i in range(board_size):
        print(f"{i:2d} ", end="")
        for j in range(board_size):
            print(f" {symbols[board[i][j]]}", end=" ")
        print()


def play_with_mcts(game, network, num_simulations, use_enhanced, num_workers=1, use_batched=False):
    """
    Play a single move with MCTS or EnhancedMCTS.
    
    Args:
        game: Game state
        network: Neural network for evaluation
        num_simulations: Number of MCTS simulations
        use_enhanced: Whether to use EnhancedMCTS
        num_workers: Number of parallel workers
        use_batched: Whether to use BatchedEvaluator
        
    Returns:
        Tuple of (move, policy, elapsed_time)
    """
    # Create evaluator
    if use_batched:
        evaluator = BatchedEvaluator(network, batch_size=16)
        eval_fn = evaluator.evaluate
    else:
        eval_fn = network.predict
    
    # Create MCTS
    start_time = time.time()
    
    if use_enhanced:
        mcts = EnhancedMCTS(
            game=game,
            evaluator=eval_fn,
            c_puct=1.5,
            num_simulations=num_simulations,
            dirichlet_alpha=0.3,
            dirichlet_noise_weight=0.25,
            temperature=1.0,
            use_transposition_table=True,
            transposition_table_size=100000,
            num_workers=num_workers,
            rave_weight=0.1,
            use_progressive_widening=False
        )
    else:
        mcts = MCTS(
            game=game,
            evaluator=eval_fn,
            c_puct=1.5,
            num_simulations=num_simulations,
            dirichlet_alpha=0.3,
            dirichlet_noise_weight=0.25,
            temperature=1.0
        )
    
    # Select a move
    move, policy = mcts.select_move(return_probs=True)
    
    elapsed_time = time.time() - start_time
    
    # Shutdown batched evaluator if used
    if use_batched:
        evaluator.shutdown()
    
    return move, policy, elapsed_time


def run_tests(board_size=9, num_games=2, num_moves=5, num_simulations=400, use_cuda=False):
    """
    Run performance tests on different MCTS configurations.
    
    Args:
        board_size: Size of the game board
        num_games: Number of games to play
        num_moves: Number of moves to play per game
        num_simulations: Number of MCTS simulations per move
        use_cuda: Whether to use CUDA for neural network
    """
    # Set device
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create network
    network = SimpleConvNet(
        board_size=board_size,
        input_channels=3,
        num_filters=64,
        num_residual_blocks=3
    ).to(device)
    
    # Define configurations to test
    configs = [
        {"name": "Original MCTS", "enhanced": False, "workers": 1, "batched": False},
        {"name": "Enhanced MCTS", "enhanced": True, "workers": 1, "batched": False},
        {"name": "Enhanced MCTS + Batched", "enhanced": True, "workers": 1, "batched": True},
        {"name": "Enhanced MCTS + Parallel", "enhanced": True, "workers": 4, "batched": False},
        {"name": "Enhanced MCTS + Parallel + Batched", "enhanced": True, "workers": 4, "batched": True}
    ]
    
    # Run tests
    results = {config["name"]: {"times": [], "nodes_per_second": []} for config in configs}
    
    for game_idx in range(num_games):
        print(f"\n=== Game {game_idx+1}/{num_games} ===")
        
        # Create a new game
        game = GomokuGame(board_size=board_size)
        
        # Play a few random moves to get to an interesting position
        for _ in range(3):
            legal_moves = game.get_legal_moves()
            move = np.random.choice(legal_moves)
            game.apply_move(move)
        
        # Print initial board
        print("\nInitial board:")
        print_board(game.state.get_board())
        print()
        
        # Save initial state for each configuration
        initial_state = game.clone()
        
        # Test each configuration
        for config in configs:
            print(f"\nTesting {config['name']}...")
            game = initial_state.clone()
            
            for move_idx in range(num_moves):
                print(f"  Move {move_idx+1}/{num_moves}...")
                
                # Play a move with the current configuration
                move, policy, elapsed_time = play_with_mcts(
                    game=game,
                    network=network,
                    num_simulations=num_simulations,
                    use_enhanced=config["enhanced"],
                    num_workers=config["workers"],
                    use_batched=config["batched"]
                )
                
                # Apply the move
                game.apply_move(move)
                
                # Record results
                nodes_per_second = num_simulations / elapsed_time
                results[config["name"]]["times"].append(elapsed_time)
                results[config["name"]]["nodes_per_second"].append(nodes_per_second)
                
                print(f"  Took {elapsed_time:.3f}s ({nodes_per_second:.1f} nodes/s)")
    
    # Compute average results
    print("\n=== Results ===")
    for name, data in results.items():
        avg_time = np.mean(data["times"])
        avg_nodes_per_second = np.mean(data["nodes_per_second"])
        print(f"{name}:")
        print(f"  Avg. time per move: {avg_time:.3f}s")
        print(f"  Avg. nodes per second: {avg_nodes_per_second:.1f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Test enhanced MCTS performance")
    
    parser.add_argument("--board-size", type=int, default=9,
                        help="Size of the Gomoku board (default: 9)")
    parser.add_argument("--num-games", type=int, default=2,
                        help="Number of games to play (default: 2)")
    parser.add_argument("--num-moves", type=int, default=5,
                        help="Number of moves to play per game (default: 5)")
    parser.add_argument("--num-simulations", type=int, default=400,
                        help="Number of MCTS simulations per move (default: 400)")
    parser.add_argument("--use-cuda", action="store_true",
                        help="Use CUDA if available")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    run_tests(
        board_size=args.board_size,
        num_games=args.num_games,
        num_moves=args.num_moves,
        num_simulations=args.num_simulations,
        use_cuda=args.use_cuda
    )


if __name__ == "__main__":
    main()