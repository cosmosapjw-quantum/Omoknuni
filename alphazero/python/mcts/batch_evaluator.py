# Create file: alphazero/python/mcts/batch_evaluator.py

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

class BatchEvaluator:
    """
    Python batch evaluator for neural network inference.
    """
    def __init__(self, network, batch_size=16):
        """
        Initialize the batch evaluator.
        
        Args:
            network: PyTorch neural network model
            batch_size: Maximum batch size for evaluation
        """
        self.network = network
        self.batch_size = batch_size
        
        # Set network to evaluation mode
        self.network.eval()
    
    def evaluate_batch(self, states: List[List[float]]) -> Tuple[List[List[float]], List[float]]:
        """
        Evaluate a batch of states.
        
        Args:
            states: List of state tensors (flattened)
            
        Returns:
            Tuple of (policies, values)
        """
        with torch.no_grad():
            # Convert to PyTorch tensor
            batch = torch.FloatTensor(states)
            
            # Determine shape
            batch_size = batch.shape[0]
            
            # Assume 3-channel board
            channels = 3
            board_size = int(np.sqrt(batch.shape[1] / channels))
            
            # Reshape for network
            if board_size * board_size * channels == batch.shape[1]:
                batch = batch.view(batch_size, channels, board_size, board_size)
            
            # Move to device
            device = next(self.network.parameters()).device
            batch = batch.to(device)
            
            # Forward pass
            policy_logits, values = self.network(batch)
            
            # Convert to list format
            policies = []
            value_list = []
            
            for i in range(batch_size):
                # Convert policy logits to probabilities
                policy = F.softmax(policy_logits[i], dim=0).cpu().numpy().tolist()
                policies.append(policy)
                value_list.append(values[i].item())
            
            return policies, value_list