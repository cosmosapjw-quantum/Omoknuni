from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class GameWrapper(ABC):
    """
    Abstract base class for game wrappers.
    This provides a common interface for all games, regardless of their implementation.
    """
    
    @abstractmethod
    def __init__(self):
        """Initialize the game state"""
        pass
    
    @abstractmethod
    def get_legal_moves(self) -> List[int]:
        """Return a list of legal moves in the current state"""
        pass
    
    @abstractmethod
    def is_terminal(self) -> bool:
        """Check if the current state is terminal (game over)"""
        pass
    
    @abstractmethod
    def get_winner(self) -> int:
        """Return the winner (0 for draw, 1 for player 1, 2 for player 2)"""
        pass
    
    @abstractmethod
    def apply_move(self, move: int) -> None:
        """Apply a move to the current state"""
        pass
    
    @abstractmethod
    def get_state_tensor(self) -> np.ndarray:
        """Convert the current state to a tensor representation for the neural network"""
        pass
    
    @abstractmethod
    def clone(self):
        """Return a deep copy of the current game state"""
        pass
    
    @abstractmethod
    def get_current_player(self) -> int:
        """Return the current player (1 or 2)"""
        pass
    
    @abstractmethod
    def get_attack_defense_scores(self, moves: List[int]) -> Tuple[List[float], List[float]]:
        """
        Calculate attack and defense scores for the given moves
        
        Args:
            moves: List of moves to evaluate
            
        Returns:
            Tuple of (attack_scores, defense_scores)
        """
        pass