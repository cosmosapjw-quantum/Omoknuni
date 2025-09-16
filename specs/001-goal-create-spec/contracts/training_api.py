"""
Training Pipeline API Contract
=============================

Self-play generation and neural network training interface.
Optimized for sample efficiency and training stability.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Iterator, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingExample:
    """Single training example from self-play."""

    state: np.ndarray          # Game position features (C, H, W)
    policy: np.ndarray         # MCTS visit count distribution (normalized)
    value: float               # Game outcome from position player's perspective
    game_type: str             # Game identifier ('gomoku', 'chess', 'go')
    move_number: int           # Move number in game
    game_id: str               # Unique game identifier


@dataclass
class GameResult:
    """Result of a complete self-play game."""

    winner: Optional[int]      # Winning player (0, 1) or None for draw
    move_count: int            # Total moves in game
    game_length_seconds: float # Wall-clock time for game
    examples: List[TrainingExample]  # Training positions from game
    final_board: str           # Human-readable final position
    metadata: Dict[str, Any]   # Additional game information


class SelfPlayGenerator(ABC):
    """Self-play game generator for training data."""

    @abstractmethod
    def __init__(self,
                 game_type: str,
                 model_path: str,
                 mcts_simulations: int = 800,
                 temperature_schedule: List[Tuple[int, float]] = None,
                 add_dirichlet_noise: bool = True,
                 num_threads: int = 8):
        """Initialize self-play generator.

        Args:
            game_type: Game to play ('gomoku', 'chess', 'go')
            model_path: Path to current neural network model
            mcts_simulations: MCTS simulations per move
            temperature_schedule: [(move_threshold, temperature), ...]
            add_dirichlet_noise: Add exploration noise at root
            num_threads: MCTS search threads
        """
        pass

    @abstractmethod
    def generate_game(self, game_id: str) -> GameResult:
        """Generate single self-play game.

        Args:
            game_id: Unique identifier for this game

        Returns:
            GameResult: Complete game with training examples
        """
        pass

    @abstractmethod
    def generate_games(self,
                      num_games: int,
                      parallel_games: int = 4) -> Iterator[GameResult]:
        """Generate multiple self-play games in parallel.

        Args:
            num_games: Total number of games to generate
            parallel_games: Number of concurrent games

        Yields:
            GameResult: Each completed game as it finishes
        """
        pass

    @abstractmethod
    def update_model(self, model_path: str) -> None:
        """Update neural network model for self-play.

        Args:
            model_path: Path to new model checkpoint
        """
        pass


class ExperienceBuffer(ABC):
    """Experience replay buffer for training data."""

    @abstractmethod
    def __init__(self,
                 buffer_path: Path,
                 max_examples: int = 1_000_000,
                 cache_size_mb: int = 512):
        """Initialize experience buffer.

        Args:
            buffer_path: Directory for memory-mapped storage
            max_examples: Maximum training examples to store
            cache_size_mb: RAM cache size in megabytes
        """
        pass

    @abstractmethod
    def add_games(self, games: List[GameResult]) -> None:
        """Add games to experience buffer.

        Args:
            games: List of completed self-play games
        """
        pass

    @abstractmethod
    def sample_batch(self,
                    batch_size: int,
                    game_types: Optional[List[str]] = None) -> List[TrainingExample]:
        """Sample training batch from buffer.

        Args:
            batch_size: Number of examples to sample
            game_types: Restrict to specific game types (None = all)

        Returns:
            List of training examples
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics.

        Returns:
            dict: Stats including size, distribution, memory usage
        """
        pass

    @abstractmethod
    def cleanup(self, keep_last_n: int = 100_000) -> None:
        """Remove old examples to manage storage.

        Args:
            keep_last_n: Number of most recent examples to retain
        """
        pass


class ModelTrainer(ABC):
    """Neural network model trainer with mixed precision."""

    @abstractmethod
    def __init__(self,
                 model_path: str,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 batch_size: int = 512,
                 use_mixed_precision: bool = True):
        """Initialize model trainer.

        Args:
            model_path: Path to model checkpoint to continue training
            learning_rate: Initial learning rate
            weight_decay: L2 regularization strength
            batch_size: Training batch size
            use_mixed_precision: Enable fp16 training
        """
        pass

    @abstractmethod
    def train_step(self,
                  batch: List[TrainingExample]) -> Dict[str, float]:
        """Single training step on batch.

        Args:
            batch: Training examples

        Returns:
            dict: Training metrics including losses and learning rate
        """
        pass

    @abstractmethod
    def validate(self,
                validation_data: List[TrainingExample]) -> Dict[str, float]:
        """Validate model on held-out data.

        Args:
            validation_data: Examples for validation

        Returns:
            dict: Validation metrics
        """
        pass

    @abstractmethod
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save model checkpoint.

        Args:
            checkpoint_path: Path for saved checkpoint
        """
        pass

    @abstractmethod
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training progress statistics.

        Returns:
            dict: Training stats including iteration count, loss history
        """
        pass


def generate_self_play_batch(game_type: str,
                           model_path: str,
                           num_games: int,
                           output_path: Path,
                           **generation_kwargs) -> List[GameResult]:
    """Generate batch of self-play games and save to disk.

    High-level interface for self-play data generation.

    Args:
        game_type: Game to play
        model_path: Current model checkpoint
        num_games: Number of games to generate
        output_path: Directory to save games
        **generation_kwargs: Additional arguments for generator

    Returns:
        List of generated games
    """
    # Contract test placeholder - implementation required
    raise NotImplementedError("Self-play batch generation implementation required")


def train_model_iteration(model_path: str,
                         experience_buffer: ExperienceBuffer,
                         num_train_steps: int = 1000,
                         validation_split: float = 0.1) -> Dict[str, float]:
    """Run one iteration of model training.

    Args:
        model_path: Path to model checkpoint
        experience_buffer: Training data source
        num_train_steps: Number of gradient steps
        validation_split: Fraction of data for validation

    Returns:
        dict: Training results and metrics
    """
    # Contract test placeholder - implementation required
    raise NotImplementedError("Model training iteration implementation required")


def evaluate_model_strength(old_model_path: str,
                          new_model_path: str,
                          game_type: str,
                          num_games: int = 100,
                          time_per_move: float = 1.0) -> Dict[str, Any]:
    """Evaluate new model against previous checkpoint.

    Plays games between old and new models to measure improvement.

    Args:
        old_model_path: Previous model checkpoint
        new_model_path: New model to evaluate
        game_type: Game for evaluation
        num_games: Number of evaluation games
        time_per_move: MCTS search time per move

    Returns:
        dict: Evaluation results including win rate, game statistics
    """
    # Contract test placeholder - implementation required
    raise NotImplementedError("Model evaluation implementation required")


def create_training_pipeline(config: Dict[str, Any]) -> Tuple[SelfPlayGenerator,
                                                             ExperienceBuffer,
                                                             ModelTrainer]:
    """Factory function to create complete training pipeline.

    Args:
        config: Training configuration dictionary

    Returns:
        tuple: (self_play_generator, experience_buffer, model_trainer)
    """
    # Contract test placeholder - implementation required
    raise NotImplementedError("Training pipeline factory implementation required")


class TrainingMetrics:
    """Training progress tracking and visualization."""

    def __init__(self, log_dir: Path):
        """Initialize metrics tracking.

        Args:
            log_dir: Directory for metric logs and plots
        """
        # Contract test placeholder - implementation required
        raise NotImplementedError("Training metrics implementation required")

    def log_training_step(self, metrics: Dict[str, float]) -> None:
        """Record training step metrics."""
        # Contract test placeholder - implementation required
        raise NotImplementedError("Training step logging implementation required")

    def log_evaluation(self, eval_results: Dict[str, Any]) -> None:
        """Record model evaluation results."""
        # Contract test placeholder - implementation required
        raise NotImplementedError("Evaluation logging implementation required")

    def generate_report(self) -> str:
        """Generate training progress report.

        Returns:
            str: Formatted training report
        """
        # Contract test placeholder - implementation required
        raise NotImplementedError("Report generation implementation required")