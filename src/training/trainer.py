"""
Neural Network Model Trainer Implementation
==========================================

Implements ModelTrainer contract with AdamW optimizer, cosine learning rate
scheduling, mixed precision training, and gradient clipping for training stability.

Features:
- AdamW optimizer with configurable weight decay
- Cosine annealing learning rate schedule with warm restarts
- Mixed precision training using PyTorch AMP for RTX 3060 Ti optimization
- Gradient clipping to prevent training instability
- Comprehensive training metrics and validation
- Support for all game types (Gomoku, Chess, Go)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import logging
import time
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from collections import defaultdict, deque

# Import contracts and model
import sys
sys.path.append('specs/001-goal-create-spec')
from contracts.training_api import ModelTrainer, TrainingExample

from src.neural.model import AlphaZeroNet, create_model_for_game

logger = logging.getLogger(__name__)


class AlphaZeroTrainer(ModelTrainer):
    """
    Neural network trainer for AlphaZero models.

    Implements the ModelTrainer contract with production-ready training features:
    - AdamW optimizer with weight decay regularization
    - Cosine annealing learning rate schedule
    - Mixed precision training for memory efficiency
    - Gradient clipping for training stability
    - Comprehensive metrics tracking
    """

    def __init__(self,
                 model_path: str,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 batch_size: int = 512,
                 use_mixed_precision: bool = True,
                 gradient_clip_norm: float = 1.0,
                 lr_schedule_t_max: int = 1000,
                 lr_min_ratio: float = 0.1):
        """
        Initialize AlphaZero model trainer.

        Args:
            model_path: Path to model checkpoint to continue training
            learning_rate: Initial learning rate for AdamW optimizer
            weight_decay: L2 regularization strength
            batch_size: Training batch size (should fit in GPU memory)
            use_mixed_precision: Enable fp16 training with automatic mixed precision
            gradient_clip_norm: Maximum gradient norm for clipping (0 to disable)
            lr_schedule_t_max: Period for cosine annealing schedule
            lr_min_ratio: Minimum learning rate as ratio of initial LR
        """
        self.model_path = model_path
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.use_mixed_precision = use_mixed_precision
        self.gradient_clip_norm = gradient_clip_norm
        self.lr_schedule_t_max = lr_schedule_t_max
        self.lr_min_ratio = lr_min_ratio

        # Device detection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Training on device: {self.device}")

        # Load model
        self.model = self._load_model()
        self.model.to(self.device)

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Learning rate scheduler - cosine annealing with warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=lr_schedule_t_max,
            eta_min=learning_rate * lr_min_ratio
        )

        # Mixed precision scaler
        self.scaler = GradScaler() if use_mixed_precision else None

        # Training state
        self.step_count = 0
        self.epoch_count = 0
        self.loss_history = deque(maxlen=1000)  # Keep last 1000 losses
        self.metrics_history = defaultdict(lambda: deque(maxlen=100))

        logger.info(f"Trainer initialized - Model: {self.model.__class__.__name__}, "
                   f"Parameters: {self._count_parameters():,}, "
                   f"Mixed Precision: {use_mixed_precision}")

    def _load_model(self) -> AlphaZeroNet:
        """Load model from checkpoint or create new model."""
        model_path = Path(self.model_path)

        if model_path.exists():
            logger.info(f"Loading existing model from {model_path}")
            try:
                # Try loading full model first with weights_only=False for our trusted models
                model = torch.load(model_path, map_location='cpu', weights_only=False)
                if isinstance(model, AlphaZeroNet):
                    return model

                # If state_dict, determine game type and create model
                state_dict = model if isinstance(model, dict) else model.state_dict()
                game_type = self._detect_game_type_from_state_dict(state_dict)
                model = create_model_for_game(game_type)
                model.load_state_dict(state_dict)
                logger.info(f"Loaded model for game type: {game_type}")
                return model

            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise RuntimeError(f"Could not load model from {model_path}: {e}")
        else:
            # Create new model - default to Gomoku if no existing model
            logger.info("Creating new Gomoku model")
            return create_model_for_game('gomoku')

    def _detect_game_type_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """Detect game type from model state dict input channels."""
        # Look for first conv layer to determine input channels
        for key, tensor in state_dict.items():
            if 'initial_conv' in key and 'weight' in key:
                input_channels = tensor.shape[1]
                if input_channels == 36:
                    return 'gomoku'
                elif input_channels == 30:
                    return 'chess'
                elif input_channels == 25:
                    return 'go'
                else:
                    logger.warning(f"Unknown input channels {input_channels}, defaulting to gomoku")
                    return 'gomoku'

        logger.warning("Could not detect game type from state dict, defaulting to gomoku")
        return 'gomoku'

    def _count_parameters(self) -> int:
        """Count trainable parameters in model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _prepare_batch(self, batch: List[TrainingExample]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert training examples to tensors.

        Args:
            batch: List of training examples

        Returns:
            tuple: (states, policies, values) as tensors
        """
        states = np.stack([example.state for example in batch])
        policies = np.stack([example.policy for example in batch])
        values = np.array([example.value for example in batch], dtype=np.float32)

        # Convert to tensors and move to device
        states_tensor = torch.from_numpy(states).float().to(self.device)
        policies_tensor = torch.from_numpy(policies).float().to(self.device)
        values_tensor = torch.from_numpy(values).float().to(self.device)

        return states_tensor, policies_tensor, values_tensor

    def _compute_loss(self,
                     policy_pred: torch.Tensor,
                     value_pred: torch.Tensor,
                     policy_target: torch.Tensor,
                     value_target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training losses.

        Args:
            policy_pred: Predicted policy logits (batch_size, num_actions)
            value_pred: Predicted values (batch_size,)
            policy_target: Target policy distribution (batch_size, num_actions)
            value_target: Target values (batch_size,)

        Returns:
            tuple: (total_loss, metrics_dict)
        """
        # Policy loss: cross-entropy with target distribution
        policy_loss = F.cross_entropy(policy_pred, policy_target, reduction='mean')

        # Value loss: mean squared error
        value_loss = F.mse_loss(value_pred.squeeze(), value_target, reduction='mean')

        # Total loss: weighted combination
        total_loss = policy_loss + value_loss

        # Additional metrics
        with torch.no_grad():
            # Policy accuracy (top-1)
            policy_pred_classes = torch.argmax(policy_pred, dim=1)
            policy_target_classes = torch.argmax(policy_target, dim=1)
            policy_accuracy = (policy_pred_classes == policy_target_classes).float().mean()

            # Value MAE
            value_mae = F.l1_loss(value_pred.squeeze(), value_target, reduction='mean')

            metrics = {
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'total_loss': total_loss.item(),
                'policy_accuracy': policy_accuracy.item(),
                'value_mae': value_mae.item(),
            }

        return total_loss, metrics

    def train_step(self, batch: List[TrainingExample]) -> Dict[str, float]:
        """
        Single training step on batch.

        Args:
            batch: Training examples

        Returns:
            dict: Training metrics including losses and learning rate
        """
        if len(batch) == 0:
            raise ValueError("Empty batch provided to train_step")

        self.model.train()
        start_time = time.time()

        # Prepare batch data
        states, policy_targets, value_targets = self._prepare_batch(batch)

        # Forward pass with mixed precision
        if self.use_mixed_precision and self.scaler is not None:
            with autocast():
                policy_pred, value_pred = self.model(states)
                loss, metrics = self._compute_loss(policy_pred, value_pred,
                                                 policy_targets, value_targets)

            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.gradient_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard precision training
            policy_pred, value_pred = self.model(states)
            loss, metrics = self._compute_loss(policy_pred, value_pred,
                                             policy_targets, value_targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

            # Optimizer step
            self.optimizer.step()

        # Learning rate scheduling
        self.scheduler.step()

        # Update training state
        self.step_count += 1
        self.loss_history.append(loss.item())
        for key, value in metrics.items():
            self.metrics_history[key].append(value)

        # Add additional metrics
        metrics.update({
            'learning_rate': self.scheduler.get_last_lr()[0],
            'step_time': time.time() - start_time,
            'batch_size': len(batch),
            'step_count': self.step_count,
        })

        # Add gradient norm if available
        if self.gradient_clip_norm > 0:
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            metrics['gradient_norm'] = total_norm

        return metrics

    def validate(self, validation_data: List[TrainingExample]) -> Dict[str, float]:
        """
        Validate model on held-out data.

        Args:
            validation_data: Examples for validation

        Returns:
            dict: Validation metrics
        """
        if len(validation_data) == 0:
            return {}

        self.model.eval()
        total_metrics = defaultdict(float)
        num_batches = 0

        with torch.no_grad():
            # Process validation data in batches
            for i in range(0, len(validation_data), self.batch_size):
                batch = validation_data[i:i + self.batch_size]
                states, policy_targets, value_targets = self._prepare_batch(batch)

                # Forward pass
                if self.use_mixed_precision:
                    with autocast():
                        policy_pred, value_pred = self.model(states)
                        _, batch_metrics = self._compute_loss(policy_pred, value_pred,
                                                            policy_targets, value_targets)
                else:
                    policy_pred, value_pred = self.model(states)
                    _, batch_metrics = self._compute_loss(policy_pred, value_pred,
                                                        policy_targets, value_targets)

                # Accumulate metrics
                for key, value in batch_metrics.items():
                    total_metrics[key] += value
                num_batches += 1

        # Average metrics across batches
        avg_metrics = {f'val_{key}': value / num_batches
                      for key, value in total_metrics.items()}

        # Store validation metrics
        for key, value in avg_metrics.items():
            self.metrics_history[key].append(value)

        return avg_metrics

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save model checkpoint.

        Args:
            checkpoint_path: Path for saved checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Save full model for easy loading
        torch.save(self.model, checkpoint_path)

        # Also save training state
        state_dict_path = checkpoint_path.with_suffix('.state.pth')
        training_state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'loss_history': list(self.loss_history),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
        }
        torch.save(training_state, state_dict_path)

        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load training state from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        state_dict_path = Path(checkpoint_path).with_suffix('.state.pth')

        if state_dict_path.exists():
            logger.info(f"Loading training state from {state_dict_path}")
            training_state = torch.load(state_dict_path, map_location=self.device, weights_only=False)

            self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
            self.scheduler.load_state_dict(training_state['scheduler_state_dict'])
            self.step_count = training_state.get('step_count', 0)
            self.epoch_count = training_state.get('epoch_count', 0)
            self.loss_history = deque(training_state.get('loss_history', []), maxlen=1000)

            if self.scaler and training_state.get('scaler_state_dict'):
                self.scaler.load_state_dict(training_state['scaler_state_dict'])

    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training progress statistics.

        Returns:
            dict: Training stats including iteration count, loss history
        """
        stats = {
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'current_lr': self.scheduler.get_last_lr()[0],
            'total_parameters': self._count_parameters(),
            'device': str(self.device),
            'mixed_precision': self.use_mixed_precision,
        }

        # Add recent loss statistics
        if self.loss_history:
            recent_losses = list(self.loss_history)
            stats.update({
                'recent_loss_mean': np.mean(recent_losses),
                'recent_loss_std': np.std(recent_losses),
                'recent_loss_min': np.min(recent_losses),
                'recent_loss_max': np.max(recent_losses),
                'loss_history_length': len(recent_losses),
            })

        # Add metrics history statistics
        for metric_name, history in self.metrics_history.items():
            if history:
                recent_values = list(history)
                stats[f'{metric_name}_mean'] = np.mean(recent_values)
                stats[f'{metric_name}_recent'] = recent_values[-1] if recent_values else 0.0

        return stats

    def reset_scheduler(self, t_max: Optional[int] = None) -> None:
        """
        Reset learning rate scheduler (useful for warm restarts).

        Args:
            t_max: New period for cosine annealing (None to keep current)
        """
        if t_max is not None:
            self.lr_schedule_t_max = t_max

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.lr_schedule_t_max,
            eta_min=self.learning_rate * self.lr_min_ratio
        )
        logger.info(f"Learning rate scheduler reset with T_max={self.lr_schedule_t_max}")