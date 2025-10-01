"""
AlphaZero Neural Network Architecture
====================================

ResNet-based architecture with Squeeze-Excitation blocks for board game position evaluation.
Optimized for RTX 3060 Ti (8GB VRAM) with mixed precision support.

Architecture:
- Initial 3x3 conv layer (input_channels -> 256)
- 20 Residual blocks with SE attention (256 channels each)
- Policy head: 1x1 conv + linear layer
- Value head: 1x1 conv + global average pool + linear layers

Target: ~10M parameters, fits with batch size 64 in 8GB VRAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention.

    Computes channel-wise attention weights to recalibrate feature maps.

    Args:
        channels: Number of input channels
        reduction: Reduction ratio for bottleneck (default: 16)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        # Squeeze: Global average pooling
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # Excitation: Two fully connected layers with bottleneck
        reduced_channels = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using proper initialization."""
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='sigmoid')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SE block.

        Args:
            x: Input tensor (batch_size, channels, height, width)

        Returns:
            Recalibrated feature maps with same shape as input
        """
        batch_size, channels, _, _ = x.size()

        # Squeeze: Global context embedding
        y = self.global_avgpool(x)  # (B, C, 1, 1)
        y = y.view(batch_size, channels)  # (B, C)

        # Excitation: Channel-wise scaling
        y = F.relu(self.fc1(y))  # (B, C//r)
        y = torch.sigmoid(self.fc2(y))  # (B, C)
        y = y.view(batch_size, channels, 1, 1)  # (B, C, 1, 1)

        # Scale original features
        return x * y


class ResidualBlock(nn.Module):
    """Residual block with Squeeze-Excitation attention.

    Architecture: Conv-BN-ReLU-Conv-BN-SE + residual connection

    Args:
        channels: Number of input/output channels
        use_se: Whether to include SE attention (default: True)
    """

    def __init__(self, channels: int, use_se: bool = True):
        super().__init__()
        self.channels = channels
        self.use_se = use_se

        # First convolution
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        # Second convolution
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

        # Squeeze-Excitation block
        if use_se:
            self.se = SqueezeExcitation(channels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize convolution weights using He initialization."""
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        # Initialize BatchNorm
        for m in [self.bn1, self.bn2]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block.

        Args:
            x: Input tensor (batch_size, channels, height, width)

        Returns:
            Output tensor with same shape as input
        """
        identity = x

        # First conv-bn-relu
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second conv-bn
        out = self.conv2(out)
        out = self.bn2(out)

        # Squeeze-excitation attention
        if self.use_se:
            out = self.se(out)

        # Residual connection
        out += identity
        out = F.relu(out)

        return out


class PolicyHead(nn.Module):
    """Policy head for action probability prediction.

    Architecture: 1x1 conv -> flatten -> linear layer

    Args:
        input_channels: Number of input channels from backbone
        num_actions: Number of possible actions (board size squared)
        board_size: Optional board size tuple (height, width). If None, inferred from num_actions
    """

    def __init__(self, input_channels: int, num_actions: int, board_size: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.input_channels = input_channels
        self.num_actions = num_actions

        # Infer board size from num_actions if not provided
        if board_size is None:
            board_size = self._infer_board_size(num_actions)
        self.board_height, self.board_width = board_size

        # 1x1 convolution to reduce channels
        self.conv = nn.Conv2d(input_channels, 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(2)

        # Linear layer for final policy logits - computed lazily on first forward pass
        self.fc = None

        # Initialize weights for conv and bn layers
        self._init_conv_weights()

    def _infer_board_size(self, num_actions: int) -> Tuple[int, int]:
        """Infer board size from number of actions.

        Args:
            num_actions: Number of possible actions

        Returns:
            Tuple of (height, width)
        """
        # Handle common game types
        if num_actions == 225:  # Gomoku 15x15
            return (15, 15)
        elif num_actions == 361:  # Go 19x19
            return (19, 19)
        elif num_actions == 64:  # Chess board positions
            return (8, 8)
        elif num_actions in (4096, 20480):  # Chess with move encodings
            return (8, 8)  # Still use 8x8 board for spatial features
        else:
            # Default: assume square board
            board_size = int(math.sqrt(num_actions))
            if board_size * board_size == num_actions:
                return (board_size, board_size)
            else:
                # Fallback for non-square boards
                return (15, 15)  # Default to Gomoku size

    def _init_conv_weights(self):
        """Initialize weights for conv and bn layers."""
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def _init_fc_weights(self):
        """Initialize weights for the linear layer."""
        if self.fc is not None:
            nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='linear')
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through policy head.

        Args:
            x: Input features (batch_size, channels, height, width)

        Returns:
            Policy logits (batch_size, num_actions)
        """
        batch_size = x.size(0)

        # 1x1 conv and activation
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)

        # Flatten spatial dimensions
        out = out.view(batch_size, -1)

        # Lazy initialization of linear layer
        if self.fc is None:
            flattened_size = out.size(1)
            self.fc = nn.Linear(flattened_size, self.num_actions)
            # Move to same device as input
            self.fc = self.fc.to(out.device)
            self._init_fc_weights()

        # Final linear layer
        logits = self.fc(out)

        return logits


class ValueHead(nn.Module):
    """Value head for position evaluation.

    Architecture: 1x1 conv -> global avg pool -> linear layers -> tanh

    Args:
        input_channels: Number of input channels from backbone
    """

    def __init__(self, input_channels: int):
        super().__init__()
        self.input_channels = input_channels

        # 1x1 convolution to reduce channels
        self.conv = nn.Conv2d(input_channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)

        # Global average pooling
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(1, 256)
        self.fc2 = nn.Linear(256, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='tanh')
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through value head.

        Args:
            x: Input features (batch_size, channels, height, width)

        Returns:
            Value estimates (batch_size, 1) in range [-1, 1]
        """
        batch_size = x.size(0)

        # 1x1 conv and activation
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)

        # Global average pooling
        out = self.global_avgpool(out)  # (batch_size, 1, 1, 1)
        out = out.view(batch_size, -1)   # (batch_size, 1)

        # Fully connected layers
        out = F.relu(self.fc1(out))
        value = torch.tanh(self.fc2(out))

        return value


class AlphaZeroNet(nn.Module):
    """AlphaZero neural network with ResNet backbone and dual heads.

    Architecture optimized for board games with configurable input shapes.
    Designed to fit in 8GB VRAM with batch size 64.

    Args:
        input_channels: Number of input feature planes
        num_actions: Number of possible actions (typically board_size^2)
        num_blocks: Number of residual blocks (default: 20)
        hidden_channels: Number of channels in residual blocks (default: 256)
        use_se: Whether to use Squeeze-Excitation (default: True)
    """

    def __init__(
        self,
        input_channels: int,
        num_actions: int,
        num_blocks: int = 20,
        hidden_channels: int = 256,
        use_se: bool = True
    ):
        super().__init__()
        self.input_channels = input_channels
        self.num_actions = num_actions
        self.num_blocks = num_blocks
        self.hidden_channels = hidden_channels
        self.use_se = use_se

        # Initial convolution layer
        self.initial_conv = nn.Conv2d(
            input_channels, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.initial_bn = nn.BatchNorm2d(hidden_channels)

        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_channels, use_se=use_se)
            for _ in range(num_blocks)
        ])

        # Output heads
        self.policy_head = PolicyHead(hidden_channels, num_actions)
        self.value_head = ValueHead(hidden_channels)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize initial convolution weights."""
        nn.init.kaiming_normal_(self.initial_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.initial_bn.weight, 1)
        nn.init.constant_(self.initial_bn.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            x: Input features (batch_size, channels, height, width)

        Returns:
            tuple: (policy_logits, values)
                policy_logits: Action probabilities (batch_size, num_actions)
                values: Position values (batch_size, 1) in range [-1, 1]
        """
        # Initial convolution
        out = self.initial_conv(x)
        out = self.initial_bn(out)
        out = F.relu(out)

        # Residual tower
        for block in self.residual_blocks:
            out = block(out)

        # Dual heads
        policy_logits = self.policy_head(out)
        values = self.value_head(out)

        return policy_logits, values

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_memory_usage(self, batch_size: int, input_shape: Tuple[int, int, int]) -> dict:
        """Estimate GPU memory usage for inference.

        Args:
            batch_size: Batch size for inference
            input_shape: Input shape (channels, height, width)

        Returns:
            Dictionary with memory usage estimates in MB
        """
        # Parameter memory
        param_memory = sum(p.numel() * 4 for p in self.parameters()) / (1024 * 1024)  # 4 bytes per float32

        # Activation memory (rough estimate)
        c, h, w = input_shape
        activation_memory = batch_size * self.hidden_channels * h * w * 4 * (self.num_blocks + 2) / (1024 * 1024)

        # Output memory
        output_memory = batch_size * (self.num_actions + 1) * 4 / (1024 * 1024)

        total_memory = param_memory + activation_memory + output_memory

        return {
            'parameters_mb': param_memory,
            'activations_mb': activation_memory,
            'outputs_mb': output_memory,
            'total_mb': total_memory,
            'fits_8gb': total_memory < 7000,  # Conservative 7GB limit (leave 1GB safety)
            'optimal_batch_size': self._estimate_optimal_batch_size(input_shape)
        }

    def _estimate_optimal_batch_size(self, input_shape: Tuple[int, int, int]) -> int:
        """Estimate optimal batch size for RTX 3060 Ti.

        Args:
            input_shape: Input shape (channels, height, width)

        Returns:
            Recommended batch size for maximum GPU utilization
        """
        # Parameter memory (constant)
        param_memory_gb = sum(p.numel() * 4 for p in self.parameters()) / (1024**3)

        # Available memory for activations (7GB - parameters - 500MB safety)
        available_memory_gb = 7.0 - param_memory_gb - 0.5

        # Estimate activation memory per sample
        c, h, w = input_shape
        activation_per_sample_gb = (self.hidden_channels * h * w * 4 * (self.num_blocks + 2)) / (1024**3)

        # Calculate optimal batch size
        optimal_batch = int(available_memory_gb / activation_per_sample_gb)

        # Round down to nearest power of 2 and clamp to reasonable range
        optimal_batch = min(512, max(32, 2 ** int(optimal_batch.bit_length() - 1)))

        return optimal_batch


def create_model_for_game(game: str, **kwargs) -> AlphaZeroNet:
    """Factory function to create game-specific models.

    Args:
        game: Game type ('gomoku', 'chess', 'go')
        **kwargs: Additional model parameters

    Returns:
        Configured AlphaZeroNet model

    Raises:
        ValueError: If game type is not supported
    """
    game = game.lower()

    # Set optimized defaults for RTX 3060 Ti (8GB VRAM) - maximize usage
    default_kwargs = {
        'num_blocks': 20,      # Original spec - more representation capacity
        'hidden_channels': 256, # Original spec - better feature learning
        'use_se': True
    }
    default_kwargs.update(kwargs)

    # Game-specific configurations with ENHANCED feature planes
    if game == 'gomoku':
        return AlphaZeroNet(
            input_channels=36,  # Enhanced Gomoku: 36 planes with threat detection, run-length analysis
            num_actions=225,    # 15x15 board
            **default_kwargs
        )
    elif game == 'chess':
        return AlphaZeroNet(
            input_channels=30,  # Enhanced Chess: 30 planes with proper move history, castling, en passant
            num_actions=4096,   # 64 squares * 64 possible moves (simplified)
            **default_kwargs
        )
    elif game == 'go':
        return AlphaZeroNet(
            input_channels=25,  # Enhanced Go: 25 planes with proper move history separation
            num_actions=361,    # 19x19 board
            **default_kwargs
        )
    else:
        raise ValueError(f"Unsupported game type: {game}. Supported: 'gomoku', 'chess', 'go'")


def create_random_model(game: str, seed: Optional[int] = None) -> AlphaZeroNet:
    """Create a randomly initialized model for testing/baseline.

    Args:
        game: Game type
        seed: Random seed for reproducible initialization

    Returns:
        Randomly initialized model
    """
    if seed is not None:
        torch.manual_seed(seed)

    model = create_model_for_game(game)

    # Apply custom initialization if needed
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Xavier initialization for linear layers
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    return model


# Mixed precision compatibility
def enable_mixed_precision(model: AlphaZeroNet) -> AlphaZeroNet:
    """Enable mixed precision training compatibility.

    Args:
        model: AlphaZeroNet model

    Returns:
        Model with mixed precision optimizations
    """
    # Convert BatchNorm to FP32 for numerical stability
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.float()

    return model


# Model validation utilities
def validate_model_output(model: AlphaZeroNet, input_tensor: torch.Tensor) -> bool:
    """Validate model outputs have correct shapes and ranges.

    Args:
        model: AlphaZeroNet model
        input_tensor: Sample input tensor

    Returns:
        True if outputs are valid
    """
    model.eval()
    with torch.no_grad():
        policy_logits, values = model(input_tensor)

        # Check shapes
        batch_size = input_tensor.size(0)
        if policy_logits.shape != (batch_size, model.num_actions):
            return False
        if values.shape != (batch_size, 1):
            return False

        # Check value range
        if not (-1 <= values.min() <= values.max() <= 1):
            return False

        # Check for NaN/inf
        if torch.isnan(policy_logits).any() or torch.isnan(values).any():
            return False
        if torch.isinf(policy_logits).any() or torch.isinf(values).any():
            return False

    return True


if __name__ == "__main__":
    """Basic testing when run directly."""
    print("AlphaZero Model Architecture Test")
    print("=" * 40)

    # Test different game configurations
    games = ['gomoku', 'chess', 'go']

    for game in games:
        print(f"\nTesting {game.capitalize()} model:")
        model = create_model_for_game(game)

        # Get model info
        num_params = model.get_num_parameters()
        print(f"  Parameters: {num_params:,} (~{num_params/1e6:.1f}M)")

        # Test forward pass
        if game == 'gomoku':
            test_input = torch.randn(4, 36, 15, 15)
        elif game == 'chess':
            test_input = torch.randn(4, 12, 8, 8)
        else:  # go
            test_input = torch.randn(4, 17, 19, 19)

        policy_logits, values = model(test_input)
        print(f"  Policy shape: {policy_logits.shape}")
        print(f"  Value shape: {values.shape}")
        print(f"  Value range: [{values.min():.3f}, {values.max():.3f}]")

        # Memory estimation
        memory_info = model.get_memory_usage(64, test_input.shape[1:])
        optimal_batch = memory_info['optimal_batch_size']
        optimal_memory = model.get_memory_usage(optimal_batch, test_input.shape[1:])

        print(f"  Memory (batch=64): {memory_info['total_mb']:.1f}MB")
        print(f"  Optimal batch size: {optimal_batch}")
        print(f"  Memory (optimal): {optimal_memory['total_mb']:.1f}MB")
        print(f"  GPU utilization: {optimal_memory['total_mb']/8000*100:.1f}% of 8GB")

        # Validation
        is_valid = validate_model_output(model, test_input)
        print(f"  Output validation: {'✅' if is_valid else '❌'}")
