import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from alphazero.python.models.network_base import BaseNetwork


class SimpleConvNet(BaseNetwork):
    """
    A simple convolutional neural network for AlphaZero.
    
    This network has a structure inspired by AlphaZero's network architecture:
    1. Input: Game state tensor
    2. Initial convolution layer
    3. Several residual blocks
    4. Policy head: Produces move probabilities
    5. Value head: Produces state value estimate
    """
    def __init__(self, 
                 board_size: int = 15, 
                 input_channels: int = 3, 
                 num_filters: int = 128,
                 num_residual_blocks: int = 5):
        """
        Initialize the network.
        
        Args:
            board_size: Size of the game board
            input_channels: Number of input channels (game state tensor channels)
            num_filters: Number of filters in convolutional layers
            num_residual_blocks: Number of residual blocks
        """
        super(SimpleConvNet, self).__init__()
        
        self.board_size = board_size
        action_size = board_size * board_size
        
        # Initial convolutional layer
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_residual_blocks)
        ])
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, action_size)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()  # Tanh activation to ensure value is in [-1, 1]
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor with shape (batch_size, input_channels, board_size, board_size)
            
        Returns:
            Tuple of (policy_logits, value) tensors
        """
        # Initial convolutional layer
        x = self.conv_block(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head
        policy_logits = self.policy_head(x)
        
        # Value head
        value = self.value_head(x)
        
        return policy_logits, value


class ResidualBlock(nn.Module):
    """
    Residual block as used in AlphaZero network.
    
    Structure:
    1. Conv + BN + ReLU
    2. Conv + BN
    3. Skip connection
    4. ReLU
    """
    def __init__(self, num_filters: int):
        """
        Initialize the residual block.
        
        Args:
            num_filters: Number of convolutional filters
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        residual = x
        
        # First conv + bn + relu
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Second conv + bn
        out = self.bn2(self.conv2(out))
        
        # Skip connection and final ReLU
        out += residual
        out = F.relu(out)
        
        return out