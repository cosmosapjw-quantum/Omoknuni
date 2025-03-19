import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from typing import List, Dict, Tuple, Any, Optional
from tqdm import tqdm

from alphazero.python.models.network_base import BaseNetwork
from alphazero.python.training.self_play import SelfPlay, GameRecord


class AlphaZeroTrainer:
    """
    Trainer for AlphaZero-style learning.
    """
    def __init__(self, 
                 network: BaseNetwork,
                 optimizer: torch.optim.Optimizer = None,
                 lr: float = 0.001,
                 weight_decay: float = 1e-4,
                 value_loss_weight: float = 1.0,
                 policy_loss_weight: float = 1.0,
                 checkpoint_dir: str = "checkpoints"):
        """
        Initialize the trainer.
        
        Args:
            network: Neural network to train
            optimizer: PyTorch optimizer (if None, Adam will be used)
            lr: Learning rate (if optimizer is None)
            weight_decay: Weight decay (if optimizer is None)
            value_loss_weight: Weight of the value loss in the total loss
            policy_loss_weight: Weight of the policy loss in the total loss
            checkpoint_dir: Directory to save checkpoints
        """
        self.network = network
        self.value_loss_weight = value_loss_weight
        self.policy_loss_weight = policy_loss_weight
        self.checkpoint_dir = checkpoint_dir
        
        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = optim.Adam(
                network.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            self.optimizer = optimizer
        
        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
    def train_on_games(self, 
                       game_records: List[GameRecord], 
                       batch_size: int = 128,
                       epochs: int = 1) -> Dict[str, float]:
        """
        Train the network on a batch of games.
        
        Args:
            game_records: List of GameRecord objects
            batch_size: Batch size for training
            epochs: Number of epochs to train for
            
        Returns:
            Dictionary of training metrics
        """
        # Prepare the training data
        states, policies, values = self._prepare_training_data(game_records)
        
        # Move to GPU if available
        device = next(self.network.parameters()).device
        
        # Convert to torch tensors
        states_tensor = torch.FloatTensor(states).to(device)
        
        # Convert policies to tensors (this depends on the specific format of your policy)
        policy_tensors = []
        board_size = int(np.sqrt(len(policies[0])))
        for policy in policies:
            policy_tensor = torch.zeros(board_size * board_size).to(device)
            for move, prob in policy.items():
                policy_tensor[move] = prob
            policy_tensors.append(policy_tensor)
        policies_tensor = torch.stack(policy_tensors)
        
        values_tensor = torch.FloatTensor(values).view(-1, 1).to(device)
        
        # Train for the specified number of epochs
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}
        
        for epoch in range(epochs):
            # Shuffle the data
            indices = torch.randperm(len(states_tensor))
            states_tensor = states_tensor[indices]
            policies_tensor = policies_tensor[indices]
            values_tensor = values_tensor[indices]
            
            # Train in batches
            num_batches = (len(states_tensor) + batch_size - 1) // batch_size
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_total_loss = 0.0
            
            for i in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}"):
                # Get batch
                start = i * batch_size
                end = min((i + 1) * batch_size, len(states_tensor))
                batch_states = states_tensor[start:end]
                batch_policies = policies_tensor[start:end]
                batch_values = values_tensor[start:end]
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                policy_logits, value = self.network(batch_states)
                
                # Calculate policy loss (cross-entropy)
                # Note: we use log_softmax + nll_loss which is equivalent to cross-entropy
                log_probs = nn.LogSoftmax(dim=1)(policy_logits)
                policy_loss = -torch.sum(batch_policies * log_probs) / len(batch_states)
                
                # Calculate value loss (MSE)
                value_loss = nn.MSELoss()(value, batch_values)
                
                # Total loss
                total_loss = self.policy_loss_weight * policy_loss + self.value_loss_weight * value_loss
                
                # Backward pass and optimize
                total_loss.backward()
                self.optimizer.step()
                
                # Update metrics
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_total_loss += total_loss.item()
            
            # Update metrics
            metrics["policy_loss"] = epoch_policy_loss / num_batches
            metrics["value_loss"] = epoch_value_loss / num_batches
            metrics["total_loss"] = epoch_total_loss / num_batches
            
            # Print metrics
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Policy Loss: {metrics['policy_loss']:.4f}, "
                  f"Value Loss: {metrics['value_loss']:.4f}, "
                  f"Total Loss: {metrics['total_loss']:.4f}")
        
        return metrics
    
    def _prepare_training_data(self, game_records: List[GameRecord]) -> Tuple[np.ndarray, List[Dict[int, float]], np.ndarray]:
        """
        Prepare training data from game records.
        
        Args:
            game_records: List of GameRecord objects
            
        Returns:
            Tuple of (states, policies, values) with states and values as numpy arrays
            and policies as a list of dictionaries
        """
        states = []
        policies = []
        values = []
        
        for record in game_records:
            record_states, record_policies, record_values = record.get_samples()
            states.extend(record_states)
            policies.extend(record_policies)
            values.extend(record_values)
        
        return np.array(states), policies, np.array(values)
    
    def save_checkpoint(self, iteration: int) -> str:
        """
        Save a checkpoint of the current network.
        
        Args:
            iteration: Current training iteration
            
        Returns:
            Path to the saved checkpoint
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_iter_{iteration}.pt")
        torch.save({
            "iteration": iteration,
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, checkpoint_path)
        
        return checkpoint_path
    
    def load_checkpoint(self, path: str) -> int:
        """
        Load a checkpoint.
        
        Args:
            path: Path to the checkpoint
            
        Returns:
            The iteration number of the loaded checkpoint
        """
        checkpoint = torch.load(path)
        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        iteration = checkpoint["iteration"]
        
        return iteration


class AlphaZeroTrainingPipeline:
    """
    Complete training pipeline for AlphaZero-style learning.
    """
    def __init__(self, 
                 game_class,
                 network: BaseNetwork,
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
        
        # Create self-play and trainer modules
        self.self_play = SelfPlay(
            game_class=game_class,
            network=network,
            game_args=game_args,
            mcts_args=mcts_args
        )
        
        self.trainer = AlphaZeroTrainer(
            network=network,
            **self.trainer_args
        )
    
    def run_iteration(self, 
                      num_games: int = 50, 
                      num_workers: int = 1,
                      batch_size: int = 128,
                      epochs: int = 1) -> Dict[str, float]:
        """
        Run a single training iteration.
        
        Args:
            num_games: Number of self-play games to generate
            num_workers: Number of parallel workers for self-play
            batch_size: Batch size for training
            epochs: Number of epochs to train for
            
        Returns:
            Dictionary of training metrics
        """
        # Generate games
        print(f"Generating {num_games} self-play games with {num_workers} workers...")
        game_records = self.self_play.generate_games(num_games, num_workers)
        
        # Train on games
        print("Training network on self-play data...")
        metrics = self.trainer.train_on_games(game_records, batch_size, epochs)
        
        return metrics
    
    def run_training(self, 
                     num_iterations: int = 100,
                     games_per_iteration: int = 50,
                     num_workers: int = 1,
                     batch_size: int = 128,
                     epochs_per_iteration: int = 1,
                     save_interval: int = 10) -> None:
        """
        Run the complete training loop.
        
        Args:
            num_iterations: Total number of training iterations
            games_per_iteration: Number of self-play games per iteration
            num_workers: Number of parallel workers for self-play
            batch_size: Batch size for training
            epochs_per_iteration: Number of epochs to train for per iteration
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
            
            # Save checkpoint if needed
            if (iteration + 1) % save_interval == 0:
                checkpoint_path = self.trainer.save_checkpoint(iteration + 1)
                print(f"Saved checkpoint to {checkpoint_path}")
            
            # Print iteration summary
            elapsed_time = time.time() - start_time
            print(f"Iteration completed in {elapsed_time:.2f} seconds")
            print(f"Policy Loss: {metrics['policy_loss']:.4f}, "
                  f"Value Loss: {metrics['value_loss']:.4f}, "
                  f"Total Loss: {metrics['total_loss']:.4f}")