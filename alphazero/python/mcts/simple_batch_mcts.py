# Create file: alphazero/python/mcts/simple_batch_mcts.py

import numpy as np
import torch
from typing import Dict, Tuple, Optional

from alphazero.python.games.game_base import GameWrapper
from alphazero.python.mcts.mcts import MCTS

class BatchEvaluator:
    """Simple evaluator that handles batch evaluation."""
    def __init__(self, network, board_size):
        """Initialize the batch evaluator."""
        self.network = network
        self.board_size = board_size
        self.device = next(network.parameters()).device
    
    def __call__(self, game_state):
        """Standard evaluator interface."""
        # Get the state tensor directly without any reshaping
        state_tensor = game_state.get_state_tensor()
        
        # Simply add batch dimension and process
        batch_tensor = torch.FloatTensor(state_tensor).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, value = self.network(batch_tensor)
        
        # Create policy dictionary
        policy_dict = {}
        legal_moves = game_state.get_legal_moves()
        
        # Get probabilities for legal moves
        probs = torch.softmax(policy_logits[0], dim=0).cpu().numpy()
        for move in legal_moves:
            policy_dict[move] = float(probs[move])
        
        # Normalize
        total = sum(policy_dict.values())
        if total > 0:
            policy_dict = {k: v/total for k, v in policy_dict.items()}
        elif legal_moves:
            # Fallback to uniform distribution
            policy_dict = {move: 1.0/len(legal_moves) for move in legal_moves}
        
        return policy_dict, float(value.item())

class SimpleBatchedMCTS:
    """
    A simple MCTS wrapper that uses batch evaluation.
    """
    def __init__(self, 
                 game: GameWrapper, 
                 evaluator,
                 c_puct: float = 1.5,
                 num_simulations: int = 800,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_noise_weight: float = 0.25,
                 temperature: float = 1.0):
        """
        Initialize the batched MCTS wrapper.
        
        Args:
            game: Game state
            evaluator: Neural network evaluator
            c_puct: Exploration constant
            num_simulations: Number of MCTS simulations
            dirichlet_alpha: Alpha parameter for Dirichlet noise
            dirichlet_noise_weight: Weight of Dirichlet noise
            temperature: Temperature for move selection
        """
        self.game = game
        
        # Create batch evaluator if network is available
        if hasattr(evaluator, 'network'):
            self.evaluator = BatchEvaluator(evaluator.network, game.board_size)
        else:
            self.evaluator = evaluator
        
        # Create standard MCTS with our evaluator
        self.mcts = MCTS(
            game=game,
            evaluator=self.evaluator,
            c_puct=c_puct,
            num_simulations=num_simulations,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_noise_weight=dirichlet_noise_weight,
            temperature=temperature
        )
    
    def select_move(self, temperature: Optional[float] = None, return_probs: bool = False):
        """
        Select a move using MCTS search.
        
        Args:
            temperature: Temperature for move selection
            return_probs: Whether to return probabilities as well
            
        Returns:
            The selected move, or a tuple of (move, probabilities) if return_probs is True
        """
        # Use provided temperature if available
        if temperature is not None:
            old_temp = self.mcts.temperature
            self.mcts.temperature = temperature
        
        # Select a move
        if return_probs:
            move, probs = self.mcts.select_move(return_probs=True)
            result = (move, probs)
        else:
            move = self.mcts.select_move()
            result = move
        
        # Restore original temperature
        if temperature is not None:
            self.mcts.temperature = old_temp
        
        return result
    
    def update_with_move(self, move: int) -> None:
        """
        Update the search tree with a move.
        
        Args:
            move: The move to update with
        """
        self.mcts.update_with_move(move)