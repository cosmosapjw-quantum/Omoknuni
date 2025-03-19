#!/usr/bin/env python3
"""
Training script for AlphaZero on Gomoku using the C++ implementation.
"""

import sys
import os
import argparse
import torch

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alphazero.python.games.gomoku import GomokuGame
from alphazero.python.models.simple_conv_net import SimpleConvNet
from alphazero.python.training.trainer import AlphaZeroTrainingPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Train AlphaZero on Gomoku using C++ implementation")
    
    parser.add_argument("--board-size", type=int, default=9,
                        help="Size of the Gomoku board (default: 9)")
    parser.add_argument("--use-renju", action="store_true",
                        help="Use Renju rules")
    parser.add_argument("--use-omok", action="store_true",
                        help="Use Omok rules")
    parser.add_argument("--use-pro-long-opening", action="store_true",
                        help="Use professional long opening rules")
    
    parser.add_argument("--num-filters", type=int, default=64,
                        help="Number of filters in the neural network (default: 64)")
    parser.add_argument("--num-residual-blocks", type=int, default=5,
                        help="Number of residual blocks in the neural network (default: 5)")
    
    parser.add_argument("--num-iterations", type=int, default=10,
                        help="Number of training iterations (default: 10)")
    parser.add_argument("--games-per-iteration", type=int, default=5,
                        help="Number of self-play games per iteration (default: 5)")
    parser.add_argument("--mcts-simulations", type=int, default=400,
                        help="Number of MCTS simulations per move (default: 400)")
    
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training (default: 64)")
    parser.add_argument("--epochs-per-iteration", type=int, default=5,
                        help="Number of epochs per training iteration (default: 5)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of workers for parallel self-play (default: 1)")
    parser.add_argument("--save-interval", type=int, default=1,
                        help="Interval at which to save checkpoints (default: 1)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints (default: 'checkpoints')")
    
    parser.add_argument("--use-cuda", action="store_true",
                        help="Use CUDA if available")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create neural network
    network = SimpleConvNet(
        board_size=args.board_size,
        input_channels=3,
        num_filters=args.num_filters,
        num_residual_blocks=args.num_residual_blocks
    ).to(device)
    
    # Game arguments
    game_args = {
        "board_size": args.board_size,
        "use_renju": args.use_renju,
        "use_omok": args.use_omok,
        "use_pro_long_opening": args.use_pro_long_opening
    }
    
    # MCTS arguments
    mcts_args = {
        "c_puct": 1.5,
        "num_simulations": args.mcts_simulations,
        "dirichlet_alpha": 0.3,
        "dirichlet_noise_weight": 0.25,
        "temperature": 1.0
    }
    
    # Trainer arguments
    trainer_args = {
        "lr": args.lr,
        "weight_decay": 1e-4,
        "value_loss_weight": 1.0,
        "policy_loss_weight": 1.0,
        "checkpoint_dir": args.checkpoint_dir
    }
    
    # Create training pipeline
    print("Creating training pipeline with C++ Gomoku implementation...")
    pipeline = AlphaZeroTrainingPipeline(
        game_class=GomokuGame,
        network=network,
        game_args=game_args,
        mcts_args=mcts_args,
        trainer_args=trainer_args
    )
    
    # Run training
    print(f"Starting training for {args.num_iterations} iterations...")
    pipeline.run_training(
        num_iterations=args.num_iterations,
        games_per_iteration=args.games_per_iteration,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        epochs_per_iteration=args.epochs_per_iteration,
        save_interval=args.save_interval
    )
    
    print("Training complete!")


if __name__ == "__main__":
    main()