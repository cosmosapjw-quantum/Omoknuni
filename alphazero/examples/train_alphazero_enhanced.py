#!/usr/bin/env python3
"""
Training script for AlphaZero on Gomoku using the enhanced MCTS implementation.
"""

import sys
import os
import argparse
import torch
import multiprocessing as mp
from typing import Dict, Any

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alphazero.python.games.gomoku import GomokuGame
from alphazero.python.models.simple_conv_net import SimpleConvNet
from alphazero.python.models.batched_evaluator import BatchedEvaluator
from alphazero.python.training.trainer import AlphaZeroTrainer
from alphazero.python.mcts.enhanced_mcts import EnhancedMCTS


class EnhancedTrainingPipeline:
    """
    Enhanced training pipeline for AlphaZero using the improved MCTS implementation.
    """
    def __init__(self,
                 game_class,
                 network: torch.nn.Module,
                 game_args: Dict[str, Any] = None,
                 mcts_args: Dict[str, Any] = None,
                 trainer_args: Dict[str, Any] = None,
                 use_batched_evaluator: bool = True,
                 batch_size: int = 16):
        """
        Initialize the training pipeline.
        
        Args:
            game_class: Game class to use
            network: Neural network for evaluation and training
            game_args: Arguments for game initialization
            mcts_args: Arguments for MCTS initialization
            trainer_args: Arguments for trainer initialization
            use_batched_evaluator: Whether to use batched evaluation
            batch_size: Batch size for batched evaluation
        """
        self.game_class = game_class
        self.network = network
        self.game_args = game_args or {}
        self.mcts_args = mcts_args or {}
        self.trainer_args = trainer_args or {}
        self.use_batched_evaluator = use_batched_evaluator
        self.batch_size = batch_size
        
        # Create trainer
        self.trainer = AlphaZeroTrainer(
            network=network,
            **self.trainer_args
        )
    
    def _get_evaluator(self):
        """Get the appropriate evaluator."""
        if self.use_batched_evaluator:
            return BatchedEvaluator(self.network, batch_size=self.batch_size)
        else:
            return self.network
    
    def generate_game(self, game_idx: int, temperature_cutoff: int = 30):
        """
        Generate a self-play game.
        
        Args:
            game_idx: Index of the game (for logging)
            temperature_cutoff: Move number after which temperature is set to ~0
        
        Returns:
            Tuple of (states, policies, values) for training
        """
        # Create game
        game = self.game_class(**self.game_args)
        
        # Create evaluator
        evaluator = self._get_evaluator()
        eval_fn = evaluator.evaluate if self.use_batched_evaluator else evaluator.predict
        
        # Create MCTS
        mcts = EnhancedMCTS(
            game=game,
            evaluator=eval_fn,
            **self.mcts_args
        )
        
        # Initialize data
        states = []
        policies = []
        move_count = 0
        
        print(f"Generating game {game_idx}...")
        
        # Play game
        while not game.is_terminal():
            # Adjust temperature based on move count
            if move_count >= temperature_cutoff:
                mcts.temperature = 0.01  # Almost zero temperature for deterministic play
            
            # Run MCTS search
            state_tensor = game.get_state_tensor()
            move, policy = mcts.select_move(return_probs=True)
            
            # Add to data
            states.append(state_tensor)
            policies.append(policy)
            
            # Apply move
            game.apply_move(move)
            
            # Update MCTS
            mcts.update_with_move(move)
            
            move_count += 1
        
        # Get values based on game result
        winner = game.get_winner()
        values = []
        
        for i in range(len(states)):
            player_at_step = 1 if i % 2 == 0 else 2
            
            if winner == 0:
                # Draw
                values.append(0.0)
            elif winner == player_at_step:
                # Player at this step won
                values.append(1.0)
            else:
                # Player at this step lost
                values.append(-1.0)
        
        # Shutdown batched evaluator if used
        if self.use_batched_evaluator:
            evaluator.shutdown()
        
        return states, policies, values
    
    def generate_games(self, num_games: int, num_workers: int = 1, temperature_cutoff: int = 30):
        """
        Generate multiple self-play games.
        
        Args:
            num_games: Number of games to generate
            num_workers: Number of parallel workers
            temperature_cutoff: Move number after which temperature is set to ~0
        
        Returns:
            List of (states, policies, values) tuples for training
        """
        if num_workers > 1:
            # Parallel generation using multiprocessing
            with mp.Pool(num_workers) as pool:
                results = pool.starmap(
                    self.generate_game,
                    [(i, temperature_cutoff) for i in range(num_games)]
                )
        else:
            # Sequential generation
            results = [self.generate_game(i, temperature_cutoff) for i in range(num_games)]
        
        return results
    
    def train_on_games(self, game_data, batch_size: int = 128, epochs: int = 1):
        """
        Train the network on a batch of games.
        
        Args:
            game_data: List of (states, policies, values) tuples
            batch_size: Batch size for training
            epochs: Number of epochs to train for
        
        Returns:
            Dictionary of training metrics
        """
        # Flatten the data
        all_states = []
        all_policies = []
        all_values = []
        
        # Create a list of game records in the format expected by the trainer
        from alphazero.python.training.self_play import GameRecord
        game_records = []
        
        for states, policies, values in game_data:
            record = GameRecord()
            record.states = states
            record.policies = policies
            record.values = values
            game_records.append(record)
            
        # Train the network
        metrics = self.trainer.train_on_games(
            game_records=game_records,
            batch_size=batch_size,
            epochs=epochs
        )
        
        return metrics
    
    def run_iteration(self, num_games: int, num_workers: int = 1, batch_size: int = 128, epochs: int = 1):
        """
        Run a single training iteration.
        
        Args:
            num_games: Number of self-play games to generate
            num_workers: Number of parallel workers
            batch_size: Batch size for training
            epochs: Number of epochs to train for
        
        Returns:
            Dictionary of training metrics
        """
        # Generate games
        print(f"Generating {num_games} self-play games with {num_workers} workers...")
        game_data = self.generate_games(num_games, num_workers)
        
        # Train on games
        print("Training network on self-play data...")
        metrics = self.train_on_games(game_data, batch_size, epochs)
        
        return metrics
    
    def run_training(self, num_iterations: int, games_per_iteration: int, num_workers: int,
                      batch_size: int, epochs_per_iteration: int, save_interval: int):
        """
        Run the complete training loop.
        
        Args:
            num_iterations: Number of training iterations
            games_per_iteration: Number of games per iteration
            num_workers: Number of parallel workers
            batch_size: Batch size for training
            epochs_per_iteration: Number of epochs per iteration
            save_interval: Interval for saving checkpoints
        """
        import time
        
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
            
            # Print metrics
            elapsed_time = time.time() - start_time
            print(f"Iteration completed in {elapsed_time:.1f}s")
            print(f"Policy loss: {metrics['policy_loss']:.4f}, Value loss: {metrics['value_loss']:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train AlphaZero with enhanced MCTS")
    
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
    parser.add_argument("--mcts-workers", type=int, default=1,
                        help="Number of worker threads for parallel MCTS (default: 1)")
    parser.add_argument("--save-interval", type=int, default=1,
                        help="Interval at which to save checkpoints (default: 1)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints (default: 'checkpoints')")
    
    parser.add_argument("--use-cuda", action="store_true",
                        help="Use CUDA if available")
    parser.add_argument("--use-batched-evaluator", action="store_true",
                        help="Use batched neural network evaluation")
    parser.add_argument("--use-transposition-table", action="store_true",
                        help="Use transposition table in MCTS")
    parser.add_argument("--use-progressive-widening", action="store_true",
                        help="Use progressive widening for large branching factors")
    
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
        "temperature": 1.0,
        "use_transposition_table": args.use_transposition_table,
        "transposition_table_size": 100000,
        "num_workers": args.mcts_workers,
        "rave_weight": 0.1 if args.mcts_workers > 1 else 0.0,
        "use_progressive_widening": args.use_progressive_widening
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
    print("Creating enhanced training pipeline...")
    pipeline = EnhancedTrainingPipeline(
        game_class=GomokuGame,
        network=network,
        game_args=game_args,
        mcts_args=mcts_args,
        trainer_args=trainer_args,
        use_batched_evaluator=args.use_batched_evaluator,
        batch_size=16
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