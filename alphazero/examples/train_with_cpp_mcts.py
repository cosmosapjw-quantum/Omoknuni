#!/usr/bin/env python3
"""
Training script for AlphaZero on Gomoku using the C++ MCTS implementation.
"""

import sys
import os
import argparse
import torch
import numpy as np
from typing import Dict, Tuple, List, Any
import time
from tqdm import tqdm
import multiprocessing as mp

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alphazero.python.games.gomoku import GomokuGame
from alphazero.python.models.simple_conv_net import SimpleConvNet
from alphazero.python.training.trainer import AlphaZeroTrainer
from alphazero.python.training.self_play import GameRecord

try:
    from alphazero.python.mcts.cpp_mcts_wrapper import CppMCTSWrapper
    CPP_MCTS_AVAILABLE = True
except ImportError:
    print("Warning: C++ MCTS implementation not available. Using Python implementation.")
    from alphazero.python.mcts.mcts import MCTS
    CPP_MCTS_AVAILABLE = False


class CppMCTSTrainingPipeline:
    """
    Training pipeline using the C++ MCTS implementation.
    """
    def __init__(self,
                 game_class,
                 network: torch.nn.Module,
                 game_args: Dict[str, Any] = None,
                 mcts_args: Dict[str, Any] = None,
                 trainer_args: Dict[str, Any] = None):
        """
        Initialize the training pipeline.
        
        Args:
            game_class: Game class to use
            network: Neural network for evaluation and training
            game_args: Arguments for game initialization
            mcts_args: Arguments for MCTS initialization
            trainer_args: Arguments for trainer initialization
        """
        self.game_class = game_class
        self.network = network
        self.game_args = game_args or {}
        self.mcts_args = mcts_args or {}
        self.trainer_args = trainer_args or {}
        
        # Create trainer
        self.trainer = AlphaZeroTrainer(
            network=network,
            **self.trainer_args
        )
    
    def _play_game(self, game_idx: int, temperature_cutoff: int = 30) -> GameRecord:
        """
        Play a complete game using MCTS with the current network.
        
        Args:
            game_idx: Index of the game (for logging)
            temperature_cutoff: Move number after which temperature is set to ~0
        
        Returns:
            GameRecord containing the game data
        """
        # Create game instance
        game = self.game_class(**self.game_args)
        
        # Create MCTS instance
        if CPP_MCTS_AVAILABLE:
            mcts = CppMCTSWrapper(
                game=game,
                evaluator=self.network.predict,
                **self.mcts_args
            )
        else:
            mcts = MCTS(
                game=game,
                evaluator=self.network.predict,
                **self.mcts_args
            )
        
        # Initialize game record
        record = GameRecord()
        move_count = 0
        
        # Play until the game is over
        while not game.is_terminal():
            # Get the current state
            state_tensor = game.get_state_tensor()
            
            # Adjust temperature based on move count
            if move_count >= temperature_cutoff:
                mcts.temperature = 0.01  # Almost zero temperature for deterministic play
            
            # Run MCTS and select a move
            move, policy = mcts.select_move(return_probs=True)
            
            # Add state and policy to record
            record.add_step(state_tensor, policy)
            
            # Apply the move
            game.apply_move(move)
            
            # Update the MCTS tree
            mcts.update_with_move(move)
            
            move_count += 1
        
        # Game over, get the result
        winner = game.get_winner()
        current_player = game.get_current_player()
        
        # Add values to the record based on the game result
        record.add_values(winner, current_player)
        
        return record
    
    def generate_games(self, num_games: int, num_workers: int = 1) -> List[GameRecord]:
        """
        Generate multiple self-play games in parallel.
        
        Args:
            num_games: Number of games to generate
            num_workers: Number of parallel workers
        
        Returns:
            List of GameRecord objects
        """
        if num_workers > 1:
            with mp.Pool(num_workers) as pool:
                records = list(tqdm(
                    pool.starmap(self._play_game, [(i, 30) for i in range(num_games)]),
                    total=num_games,
                    desc="Generating games"
                ))
        else:
            records = []
            for i in tqdm(range(num_games), desc="Generating games"):
                records.append(self._play_game(i, 30))
        
        return records
    
    def run_iteration(self, num_games: int, num_workers: int = 1, batch_size: int = 128, epochs: int = 1) -> Dict[str, float]:
        """
        Run a single training iteration.
        
        Args:
            num_games: Number of self-play games to generate
            num_workers: Number of worker processes for self-play
            batch_size: Batch size for training
            epochs: Number of training epochs
        
        Returns:
            Dictionary of training metrics
        """
        # Generate games
        print(f"Generating {num_games} self-play games with {num_workers} workers...")
        game_records = self.generate_games(num_games, num_workers)
        
        # Train on games
        print("Training network on self-play data...")
        metrics = self.trainer.train_on_games(game_records, batch_size, epochs)
        
        return metrics
    
    def run_training(self, num_iterations: int, games_per_iteration: int, num_workers: int,
                    batch_size: int, epochs_per_iteration: int, save_interval: int):
        """
        Run the complete training pipeline.
        
        Args:
            num_iterations: Number of training iterations
            games_per_iteration: Number of self-play games per iteration
            num_workers: Number of worker processes for self-play
            batch_size: Batch size for training
            epochs_per_iteration: Number of training epochs per iteration
            save_interval: Interval at which to save checkpoints
        """
        for iteration in range(num_iterations):
            print(f"\n=== Iteration {iteration+1}/{num_iterations} ===")
            start_time = time.time()
            
            # Run iteration
            metrics = self.run_iteration(
                num_games=games_per_iteration,
                num_workers=num_workers,
                batch_size=batch_size,
                epochs=epochs_per_iteration
            )
            
            # Save checkpoint
            if (iteration + 1) % save_interval == 0:
                checkpoint_path = self.trainer.save_checkpoint(iteration + 1)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            # Print iteration summary
            elapsed_time = time.time() - start_time
            print(f"Iteration completed in {elapsed_time:.2f} seconds")
            print(f"Policy Loss: {metrics['policy_loss']:.4f}, "
                  f"Value Loss: {metrics['value_loss']:.4f}, "
                  f"Total Loss: {metrics['total_loss']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train AlphaZero on Gomoku with C++ MCTS")
    
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
                        help="Number of worker processes for self-play (default: 1)")
    parser.add_argument("--mcts-threads", type=int, default=1,
                        help="Number of threads per MCTS (default: 1)")
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
    
    # Add C++ specific arguments if available
    if CPP_MCTS_AVAILABLE:
        mcts_args.update({
            "use_transposition_table": True,
            "transposition_table_size": 100000,
            "num_threads": args.mcts_threads
        })
        print(f"Using C++ MCTS with {args.mcts_threads} threads")
    else:
        print("Using Python MCTS implementation")
    
    # Trainer arguments
    trainer_args = {
        "lr": args.lr,
        "weight_decay": 1e-4,
        "value_loss_weight": 1.0,
        "policy_loss_weight": 1.0,
        "checkpoint_dir": args.checkpoint_dir
    }
    
    # Create training pipeline
    pipeline = CppMCTSTrainingPipeline(
        game_class=GomokuGame,
        network=network,
        game_args=game_args,
        mcts_args=mcts_args,
        trainer_args=trainer_args
    )
    
    # Run training
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