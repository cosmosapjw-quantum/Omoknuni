# Create file: alphazero/python/mcts/parallel_batch_mcts.py

import numpy as np
import torch
import multiprocessing as mp
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm

from alphazero.python.games.game_base import GameWrapper
from alphazero.python.mcts.mcts import MCTS

class ParallelBatchEvaluator:
    """
    Parallel batch evaluator that handles batch evaluation across multiple processes.
    """
    def __init__(self, network, board_size, num_workers=None, batch_size=64):
        """
        Initialize the parallel batch evaluator.
        
        Args:
            network: PyTorch neural network model
            board_size: Size of the game board
            num_workers: Number of worker processes (default: number of CPU cores)
            batch_size: Maximum batch size for evaluation
        """
        self.network = network
        self.board_size = board_size
        self.device = next(network.parameters()).device
        
        # Determine number of workers
        if num_workers is None:
            num_workers = mp.cpu_count()
        self.num_workers = max(1, num_workers)
        
        self.batch_size = batch_size
        
        # Set network to evaluation mode
        self.network.eval()
        
        # Create a process pool
        if self.num_workers > 1:
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                # Method already set, ignore
                pass
            
            self.pool = mp.Pool(self.num_workers)
        else:
            self.pool = None
    
    def __call__(self, game_state):
        """Standard evaluator interface."""
        # Get the state tensor directly 
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
    
    def evaluate_batch(self, states: List[np.ndarray]) -> Tuple[List[Dict[int, float]], List[float]]:
        """
        Parallel batch evaluation of states.
        
        Args:
            states: List of state tensors
            
        Returns:
            Tuple of (policies, values)
        """
        # Ensure states are numpy arrays
        states = [np.array(state) for state in states]
        
        if self.num_workers <= 1 or len(states) <= 1:
            # Fallback to sequential evaluation
            return self._sequential_batch_eval(states)
        
        try:
            # Split the batch into chunks for parallel processing
            policies = [None] * len(states)
            values = [None] * len(states)
            
            # Use pool map with a progress bar
            with tqdm(total=len(states), desc="Batch Evaluation", disable=len(states) < 10) as pbar:
                def update_progress(result):
                    pbar.update(1)
                    return result
                
                # Parallel evaluation
                results = []
                for i in range(0, len(states), self.batch_size):
                    batch = states[i:i+self.batch_size]
                    
                    # Submit batch for parallel processing
                    batch_results = [
                        self.pool.apply_async(
                            self._single_state_eval, 
                            args=(state,), 
                            callback=update_progress
                        ) 
                        for state in batch
                    ]
                    
                    results.extend(batch_results)
                
                # Collect results
                for i, result in enumerate(results):
                    policies[i], values[i] = result.get()
            
            return policies, values
        
        except Exception as e:
            print(f"Parallel evaluation error: {e}")
            # Fallback to sequential evaluation
            return self._sequential_batch_eval(states)
    
    def _single_state_eval(self, state: np.ndarray) -> Tuple[Dict[int, float], float]:
        """
        Evaluate a single state on the current device.
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (policy, value)
        """
        # Move tensor to device
        batch_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, value = self.network(batch_tensor)
        
        # Convert policy to dictionary of valid moves
        policy_dict = {}
        
        # Assuming global policy probabilities
        probs = torch.softmax(policy_logits[0], dim=0).cpu().numpy()
        
        # Fill policy dictionary with move probabilities
        total_prob = 0.0
        for move in range(len(probs)):
            if probs[move] > 0:
                policy_dict[move] = float(probs[move])
                total_prob += probs[move]
        
        # Normalize probabilities
        if total_prob > 0:
            policy_dict = {k: v/total_prob for k, v in policy_dict.items()}
        
        return policy_dict, float(value.item())
    
    def _sequential_batch_eval(self, states: List[np.ndarray]) -> Tuple[List[Dict[int, float]], List[float]]:
        """
        Sequential batch evaluation for fallback or small batches.
        
        Args:
            states: List of state tensors
            
        Returns:
            Tuple of (policies, values)
        """
        policies = []
        values = []
        
        for state in tqdm(states, desc="Batch Evaluation", disable=len(states) < 10):
            # Evaluate single state
            policy, value = self._single_state_eval(state)
            policies.append(policy)
            values.append(value)
        
        return policies, values
    
    def __del__(self):
        """
        Clean up the process pool when the object is deleted.
        """
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.close()
            self.pool.join()

class ParallelBatchedMCTS:
    """
    A parallel MCTS wrapper that uses batch evaluation.
    """
    def __init__(self, 
                 game: GameWrapper, 
                 evaluator,
                 num_workers: Optional[int] = None,
                 c_puct: float = 1.5,
                 num_simulations: int = 800,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_noise_weight: float = 0.25,
                 temperature: float = 1.0):
        """
        Initialize the parallel batched MCTS wrapper.
        
        Args:
            game: Game state
            evaluator: Neural network evaluator
            num_workers: Number of parallel workers
            c_puct: Exploration constant
            num_simulations: Number of MCTS simulations
            dirichlet_alpha: Alpha parameter for Dirichlet noise
            dirichlet_noise_weight: Weight of Dirichlet noise
            temperature: Temperature for move selection
        """
        self.game = game
        
        # Create parallel batch evaluator if network is available
        if hasattr(evaluator, 'network'):
            self.evaluator = ParallelBatchEvaluator(
                evaluator.network, 
                game.board_size, 
                num_workers=num_workers
            )
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