import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List

from alphazero.python.games.game_base import GameWrapper


class BaseNetwork(nn.Module):
    """
    Base class for neural networks used in AlphaZero.
    """
    def __init__(self):
        super(BaseNetwork, self).__init__()
    
    def forward(self, x):
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor
            
        Returns:
            policy_logits, value
        """
        raise NotImplementedError
    
    def process_batch(self, states: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a batch of game states.
        
        Args:
            states: List of game state tensors
            
        Returns:
            Tuple of (policy_logits, value) tensors
        """
        # Convert the list of states to a single tensor
        batch = np.array(states)
        batch_tensor = torch.FloatTensor(batch)
        
        # If CUDA is available, move to GPU
        if next(self.parameters()).is_cuda:
            batch_tensor = batch_tensor.cuda()
        
        # Forward pass
        policy_logits, value = self(batch_tensor)
        
        return policy_logits, value
    
    def predict(self, game_state: GameWrapper) -> Tuple[Dict[int, float], float]:
        """
        Predict policy and value for a single game state.
        
        Args:
            game_state: Game state to predict for
            
        Returns:
            Tuple of (policy, value), where policy is a dict mapping moves to probabilities
        """
        # Convert the game state to a tensor
        state_tensor = game_state.get_state_tensor()
        states = [state_tensor]
        
        # Forward pass
        policy_logits, value = self.process_batch(states)
        
        # Get the valid moves
        valid_moves = game_state.get_legal_moves()
        
        # Convert policy logits to probabilities
        policy = self._policy_to_probabilities(policy_logits[0], valid_moves, game_state.board_size)
        
        return policy, value.item()
    
    def _policy_to_probabilities(self, 
                                policy_logits: torch.Tensor, 
                                valid_moves: List[int], 
                                board_size: int) -> Dict[int, float]:
        """
        Convert policy logits to a dictionary of move probabilities.
        
        Args:
            policy_logits: Raw policy output from the network
            valid_moves: List of valid moves
            board_size: Size of the game board
            
        Returns:
            Dictionary mapping moves to probabilities
        """
        # Get probabilities by applying softmax
        policy = F.softmax(policy_logits, dim=0).detach().cpu().numpy()
        
        # Create a dictionary of move -> probability
        policy_dict = {}
        
        # Set probabilities for valid moves
        for move in valid_moves:
            policy_dict[move] = policy[move]
        
        # Normalize probabilities
        if sum(policy_dict.values()) > 0:
            factor = 1.0 / sum(policy_dict.values())
            policy_dict = {move: prob * factor for move, prob in policy_dict.items()}
        else:
            # If all probabilities are zero, use a uniform distribution
            policy_dict = {move: 1.0 / len(valid_moves) for move in valid_moves}
        
        return policy_dict
    
    def save(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model to
        """
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath: str) -> None:
        """
        Load the model from a file.
        
        Args:
            filepath: Path to load the model from
        """
        self.load_state_dict(torch.load(filepath))