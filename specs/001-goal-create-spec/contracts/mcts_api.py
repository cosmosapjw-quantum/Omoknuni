"""
MCTS Engine API Contract
========================

Core Monte Carlo Tree Search interface for high-performance board game AI.
All functions must be implemented to pass contract tests.
"""

import numpy as np
from typing import Optional, List, Tuple
from abc import ABC, abstractmethod


class GameState(ABC):
    """Abstract game state interface - must be implemented for each game."""

    @abstractmethod
    def apply_move_inplace(self, action: int) -> None:
        """Apply move directly to current state (no copy).

        Args:
            action: Integer action in game's action space

        Raises:
            ValueError: If action is illegal in current position
        """
        pass

    @abstractmethod
    def get_legal_moves(self) -> np.ndarray:
        """Get boolean mask of legal moves.

        Returns:
            np.ndarray: Boolean array where True indicates legal move
        """
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """Check if game is in terminal state.

        Returns:
            bool: True if game is finished
        """
        pass

    @abstractmethod
    def get_terminal_value(self) -> float:
        """Get terminal value from current player's perspective.

        Returns:
            float: Value in [-1, 1], where 1=win, 0=draw, -1=loss

        Raises:
            ValueError: If called on non-terminal state
        """
        pass

    @abstractmethod
    def extract_features(self) -> np.ndarray:
        """Extract neural network input features.

        Returns:
            np.ndarray: Feature tensor of shape (channels, height, width)
        """
        pass

    @abstractmethod
    def get_current_player(self) -> int:
        """Get current player to move.

        Returns:
            int: 0 or 1 indicating which player's turn
        """
        pass

    @abstractmethod
    def copy(self) -> 'GameState':
        """Create deep copy of game state.

        Returns:
            GameState: Independent copy of current state
        """
        pass


def search(state: GameState,
          num_simulations: int,
          cpuct: float = 1.25,
          num_threads: int = 8,
          add_dirichlet_noise: bool = False,
          random_seed: Optional[int] = None) -> np.ndarray:
    """Run MCTS search and return visit count distribution.

    This is the primary interface for the MCTS engine. Must achieve
    performance targets: 30-40k simulations/second including NN inference.

    Args:
        state: Game state to search from (not modified)
        num_simulations: Number of MCTS simulations to run
        cpuct: Exploration constant for PUCT formula
        num_threads: Number of search threads (recommend 8-10 for Ryzen 5900X)
        add_dirichlet_noise: Add noise at root for exploration (training only)
        random_seed: Fixed seed for deterministic behavior (testing only)

    Returns:
        np.ndarray: Visit count distribution over legal actions

    Raises:
        ValueError: If parameters are invalid
        RuntimeError: If search fails due to resource constraints
    """
    # Contract test placeholder - implementation required
    raise NotImplementedError("MCTS search implementation required")


def search_with_info(state: GameState,
                    num_simulations: int,
                    cpuct: float = 1.25,
                    num_threads: int = 8) -> Tuple[np.ndarray, dict]:
    """Run MCTS search with detailed performance information.

    Extended version of search() that returns additional metrics
    for performance monitoring and debugging.

    Args:
        state: Game state to search from
        num_simulations: Number of MCTS simulations to run
        cpuct: Exploration constant for PUCT formula
        num_threads: Number of search threads

    Returns:
        tuple: (visit_counts, info_dict)
            visit_counts: Visit distribution over actions
            info_dict: Performance metrics including:
                - 'simulations_per_second': float
                - 'gpu_utilization': float
                - 'average_batch_size': float
                - 'memory_usage_mb': float
                - 'thread_efficiency': List[float]
    """
    # Contract test placeholder - implementation required
    raise NotImplementedError("MCTS search with info implementation required")


def evaluate_position(state: GameState) -> Tuple[np.ndarray, float]:
    """Evaluate position using neural network (single inference).

    Direct neural network evaluation without MCTS search.
    Useful for position analysis and debugging.

    Args:
        state: Game state to evaluate

    Returns:
        tuple: (policy, value)
            policy: Probability distribution over actions
            value: Position value from current player's perspective [-1, 1]
    """
    # Contract test placeholder - implementation required
    raise NotImplementedError("Position evaluation implementation required")


def get_best_move(state: GameState,
                 num_simulations: int,
                 temperature: float = 0.0,
                 **search_kwargs) -> int:
    """Get best move from MCTS search.

    Convenience function that runs search and selects move based on
    visit counts and temperature parameter.

    Args:
        state: Game state to search from
        num_simulations: Number of MCTS simulations
        temperature: Sampling temperature (0.0 = greedy, 1.0 = proportional)
        **search_kwargs: Additional arguments passed to search()

    Returns:
        int: Selected action/move
    """
    # Contract test placeholder - implementation required
    raise NotImplementedError("Best move selection implementation required")


class MCTSEngine:
    """Stateful MCTS engine with persistent tree and configuration."""

    def __init__(self,
                 game_type: str,
                 model_path: str,
                 num_threads: int = 8,
                 max_tree_nodes: int = 50_000_000):
        """Initialize MCTS engine.

        Args:
            game_type: Game identifier ('gomoku', 'chess', 'go')
            model_path: Path to trained neural network model
            num_threads: Number of search threads
            max_tree_nodes: Maximum tree size in nodes
        """
        # Contract test placeholder - implementation required
        self.game_type = game_type
        self.model_path = model_path
        self.num_threads = num_threads
        self.max_tree_nodes = max_tree_nodes

    def search(self, state: GameState, num_simulations: int, **kwargs) -> np.ndarray:
        """Run MCTS search using persistent tree."""
        # Contract test placeholder - implementation required
        raise NotImplementedError("Engine search implementation required")

    def reset_tree(self) -> None:
        """Clear search tree and start fresh."""
        # Contract test placeholder - implementation required
        raise NotImplementedError("Tree reset implementation required")

    def get_tree_stats(self) -> dict:
        """Get current tree statistics.

        Returns:
            dict: Tree stats including node count, memory usage, etc.
        """
        # Contract test placeholder - implementation required
        raise NotImplementedError("Tree stats implementation required")