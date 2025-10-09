"""
DLPack Inference Bridge
=======================

Zero-copy inference bridge using DLPack tensors for C++ MCTS integration.
Implements BatchInferenceCallback interface with torch.from_dlpack() conversion.

This module is part of T008b in the MCTS throughput recovery project.
"""

import logging
import time
from typing import List, Tuple, Dict, Any, Optional
from threading import Lock

import numpy as np
import torch
import torch.nn as nn

try:
    import mcts_py
    HAS_MCTS_PY = True
except ImportError:
    HAS_MCTS_PY = False


class DLPackInferenceBridge:
    """Zero-copy inference bridge using DLPack tensors.

    Implements BatchInferenceCallback interface for C++ MCTS integration.
    Uses DLPack protocol to eliminate numpy copy overhead.

    Architecture:
    1. C++ provides list of IGameState objects
    2. Create DLPack tensor via mcts_py.create_batch_tensor_from_states()
    3. Convert to PyTorch via torch.from_dlpack() (zero-copy)
    4. Run neural network inference on GPU
    5. Extract policy/value and return to C++

    Args:
        model: PyTorch neural network model (nn.Module)
        device: Target device ('cpu', 'cuda', 'cuda:0', etc.)
        enable_fallback: Enable numpy fallback if DLPack fails
        warmup_iterations: Number of warmup batches for GPU
        use_mixed_precision: Enable FP16 mixed precision on CUDA (T008f)

    Example:
        >>> model = GomokuNet().cuda()
        >>> bridge = DLPackInferenceBridge(model, device='cuda', use_mixed_precision=True)
        >>> bridge.warmup(batch_size=64)
        >>> # Use with C++ coordinator
        >>> results = bridge.batch_inference(states)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        enable_fallback: bool = True,
        warmup_iterations: int = 5,
        use_mixed_precision: bool = True
    ):
        if not HAS_MCTS_PY:
            raise ImportError(
                "mcts_py module not available. "
                "DLPack inference requires C++ extensions to be built."
            )

        self.model = model
        self.device = torch.device(device)
        self.enable_fallback = enable_fallback
        self.warmup_iterations = warmup_iterations

        # T008f: Enable mixed precision for CUDA (FP16 with tensor cores)
        self.use_mixed_precision = use_mixed_precision and self.device.type == 'cuda'
        if self.use_mixed_precision:
            # Enable cudnn autotuner for better performance with tensor cores
            torch.backends.cudnn.benchmark = True

        # Move model to target device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Metrics tracking
        self._total_batches = 0
        self._total_states = 0
        self._dlpack_successes = 0
        self._fallback_uses = 0
        self._total_latency_ms = 0.0
        self._metrics_lock = Lock()

        # Logger
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"DLPackInferenceBridge initialized: device={device}, "
            f"fallback={enable_fallback}, mixed_precision={self.use_mixed_precision}"
        )

    def batch_inference(
        self,
        states: List
    ) -> List[Tuple[List[float], float]]:
        """Execute neural network inference for a batch of game states.

        This is the main entry point called by C++ BatchInferenceCoordinator.

        Flow:
        1. Validate inputs (non-empty, same game type)
        2. Create DLPack tensor from states
        3. Convert to PyTorch tensor (zero-copy)
        4. Transfer to GPU if needed (async copy)
        5. Run model forward pass
        6. Transfer results back to CPU
        7. Extract policy/value pairs
        8. Return as list of tuples

        Args:
            states: List of IGameState objects from C++

        Returns:
            List[(policy, value)] where:
                policy: List[float] - action probabilities
                value: float - position evaluation

        Raises:
            ValueError: If states is empty or contains mixed game types
            RuntimeError: If DLPack conversion fails and fallback disabled
        """
        start_time = time.perf_counter()

        # Validate inputs
        if not states or len(states) == 0:
            raise ValueError("states list cannot be empty")

        batch_size = len(states)

        try:
            # Try DLPack path (zero-copy)
            results = self._dlpack_inference(states)

            with self._metrics_lock:
                self._dlpack_successes += 1

        except Exception as e:
            if self.enable_fallback:
                self.logger.warning(
                    f"DLPack inference failed: {e}, using numpy fallback"
                )
                results = self._numpy_fallback_inference(states)

                with self._metrics_lock:
                    self._fallback_uses += 1
            else:
                raise RuntimeError(f"DLPack inference failed: {e}") from e

        # Update metrics
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        with self._metrics_lock:
            self._total_batches += 1
            self._total_states += batch_size
            self._total_latency_ms += elapsed_ms

        return results

    def _dlpack_inference(
        self,
        states: List
    ) -> List[Tuple[List[float], float]]:
        """DLPack zero-copy inference path.

        Args:
            states: List of IGameState objects

        Returns:
            List of (policy, value) tuples
        """
        # Create DLPack tensor from states (zero-copy)
        capsule = mcts_py.create_batch_tensor_from_states(states, use_cuda=False)
        features = torch.from_dlpack(capsule)

        # Transfer to target device
        if self.device.type == 'cuda':
            # Async transfer if pinned memory
            features_gpu = features.to(self.device, non_blocking=True)
        else:
            features_gpu = features

        # T008f: Run inference with FP16 mixed precision on CUDA
        with torch.no_grad():
            if self.use_mixed_precision:
                # Use automatic mixed precision (FP16) for 1.5-2× speedup on tensor cores
                with torch.cuda.amp.autocast():
                    policy_logits, value = self.model(features_gpu)
            else:
                # Standard FP32 inference
                policy_logits, value = self.model(features_gpu)

        # Apply softmax to get probabilities (always in FP32 for numerical stability)
        policy = torch.softmax(policy_logits.float(), dim=1)

        # Transfer results back to CPU (async)
        policy_cpu = policy.cpu()
        value_cpu = value.cpu()

        # Convert to Python lists
        results = []
        policy_np = policy_cpu.numpy()
        value_np = value_cpu.numpy()

        for i in range(len(states)):
            policy_list = policy_np[i].tolist()
            value_scalar = float(value_np[i])
            results.append((policy_list, value_scalar))

        return results

    def _numpy_fallback_inference(
        self,
        states: List
    ) -> List[Tuple[List[float], float]]:
        """Fallback to numpy array extraction.

        Args:
            states: List of IGameState objects

        Returns:
            List of (policy, value) tuples
        """
        batch_size = len(states)
        num_planes = states[0].get_num_feature_planes()
        board_size = states[0].get_board_size()

        # Allocate numpy array
        features_np = np.zeros(
            (batch_size, num_planes, board_size, board_size),
            dtype=np.float32
        )

        # Extract features for each state
        for i, state in enumerate(states):
            buffer = np.zeros(
                num_planes * board_size * board_size,
                dtype=np.float32
            )
            state.extract_features_to_buffer(buffer)
            features_np[i] = buffer.reshape(num_planes, board_size, board_size)

        # Convert to torch (with copy)
        features = torch.from_numpy(features_np).to(self.device)

        # T008f: Run inference with mixed precision if enabled
        with torch.no_grad():
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    policy_logits, value = self.model(features)
            else:
                policy_logits, value = self.model(features)

        # Apply softmax (always in FP32 for numerical stability)
        policy = torch.softmax(policy_logits.float(), dim=1)

        # Convert to Python lists
        results = []
        policy_np = policy.cpu().numpy()
        value_np = value.cpu().numpy()

        for i in range(batch_size):
            policy_list = policy_np[i].tolist()
            value_scalar = float(value_np[i])
            results.append((policy_list, value_scalar))

        return results

    def warmup(self, batch_size: int = 64, game_type: str = 'gomoku'):
        """Warm up GPU with dummy batches.

        Runs several dummy inference batches to:
        - Initialize CUDA kernels
        - Allocate GPU memory
        - Prime memory pools
        - Measure baseline latency

        Args:
            batch_size: Size of warmup batches
            game_type: Game type for dummy states ('gomoku', 'chess', 'go')
        """
        try:
            import alphazero_py

            # Create dummy states
            if game_type == 'gomoku':
                states = [alphazero_py.GomokuState() for _ in range(batch_size)]
            elif game_type == 'chess':
                states = [alphazero_py.ChessState() for _ in range(batch_size)]
            elif game_type == 'go':
                states = [alphazero_py.GoState() for _ in range(batch_size)]
            else:
                raise ValueError(f"Unknown game type: {game_type}")

            # Run warmup iterations
            self.logger.info(
                f"Warming up with {self.warmup_iterations} batches "
                f"(size={batch_size}, game={game_type})"
            )

            for i in range(self.warmup_iterations):
                self.batch_inference(states)

            self.logger.info("Warmup complete")

        except ImportError:
            self.logger.warning(
                "alphazero_py not available, skipping warmup"
            )

    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.

        Returns:
            Dictionary with:
                - total_batches: Total batch_inference calls
                - total_states: Total states evaluated
                - avg_batch_size: Average batch size
                - dlpack_successes: DLPack path used
                - fallback_uses: Numpy fallback used
                - avg_latency_ms: Average inference latency
                - dlpack_success_rate: Percentage of DLPack successes
        """
        with self._metrics_lock:
            if self._total_batches == 0:
                return {
                    'total_batches': 0,
                    'total_states': 0,
                    'avg_batch_size': 0.0,
                    'dlpack_successes': 0,
                    'fallback_uses': 0,
                    'avg_latency_ms': 0.0,
                    'dlpack_success_rate': 0.0
                }

            return {
                'total_batches': self._total_batches,
                'total_states': self._total_states,
                'avg_batch_size': self._total_states / self._total_batches,
                'dlpack_successes': self._dlpack_successes,
                'fallback_uses': self._fallback_uses,
                'avg_latency_ms': self._total_latency_ms / self._total_batches,
                'dlpack_success_rate': (
                    100.0 * self._dlpack_successes / self._total_batches
                )
            }

    def reset_metrics(self):
        """Reset all performance metrics."""
        with self._metrics_lock:
            self._total_batches = 0
            self._total_states = 0
            self._dlpack_successes = 0
            self._fallback_uses = 0
            self._total_latency_ms = 0.0
