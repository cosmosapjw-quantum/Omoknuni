import torch
import numpy as np
import threading
import time
from typing import List, Dict, Tuple, Callable, Optional, Any, Union, Set, Deque
from collections import deque

from alphazero.python.games.game_base import GameWrapper
from alphazero.python.models.network_base import BaseNetwork


class BatchedEvaluator:
    """
    A batched evaluator for neural network inference.
    
    This class accumulates positions to evaluate and processes them in batches
    for more efficient neural network inference.
    """
    def __init__(self, 
                 network: BaseNetwork, 
                 batch_size: int = 16,
                 max_wait_time: float = 0.001):
        """
        Initialize the batched evaluator.
        
        Args:
            network: The neural network to use for evaluation
            batch_size: Maximum batch size for evaluation
            max_wait_time: Maximum time to wait for a full batch (seconds)
        """
        self.network = network
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        
        # Queue for positions to evaluate
        self.queue: Deque[Tuple[GameWrapper, Dict[str, Any]]] = deque()
        self.queue_lock = threading.Lock()
        self.queue_not_empty = threading.Condition(self.queue_lock)
        
        # Results
        self.results: Dict[int, Tuple[Dict[int, float], float]] = {}
        self.results_lock = threading.Lock()
        
        # Worker thread
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
    
    def evaluate(self, game_state: GameWrapper) -> Tuple[Dict[int, float], float]:
        """
        Evaluate a game state.
        
        This method adds the game state to the queue and waits for the result.
        
        Args:
            game_state: The game state to evaluate
            
        Returns:
            Tuple of (policy, value)
        """
        # Create a unique ID for this request
        request_id = id(game_state)
        
        # Add to queue
        with self.queue_lock:
            self.queue.append((game_state, {'id': request_id}))
            self.queue_not_empty.notify()
        
        # Wait for result
        while True:
            with self.results_lock:
                if request_id in self.results:
                    result = self.results[request_id]
                    del self.results[request_id]
                    return result
            
            # Sleep briefly to avoid consuming CPU
            time.sleep(0.0001)
    
    def _worker_loop(self) -> None:
        """
        Main loop for the worker thread.
        
        This method processes positions in batches.
        """
        while self.running:
            batch = []
            batch_ids = []
            
            # Get a batch of positions
            with self.queue_lock:
                # Wait until there's at least one position in the queue
                while len(self.queue) == 0:
                    self.queue_not_empty.wait()
                    if not self.running:
                        return
                
                # Get as many positions as possible, up to batch_size
                start_time = time.time()
                while len(self.queue) > 0 and len(batch) < self.batch_size:
                    game_state, metadata = self.queue.popleft()
                    batch.append(game_state)
                    batch_ids.append(metadata['id'])
                    
                    # If we haven't reached batch_size, check if we've waited long enough
                    if len(batch) < self.batch_size and len(self.queue) > 0:
                        elapsed = time.time() - start_time
                        if elapsed >= self.max_wait_time:
                            break
            
            # Process the batch
            if batch:
                # Convert each game state to a tensor
                states = [game.get_state_tensor() for game in batch]
                
                # Forward pass
                policy_logits, values = self.network.process_batch(states)
                
                # Extract results
                with self.results_lock:
                    for i, request_id in enumerate(batch_ids):
                        # Convert policy logits to probabilities
                        game = batch[i]
                        valid_moves = game.get_legal_moves()
                        board_size = getattr(game, 'board_size', int(np.sqrt(len(policy_logits[i]))))
                        policy = self._policy_to_probabilities(
                            policy_logits[i], valid_moves, board_size
                        )
                        
                        # Store result
                        self.results[request_id] = (policy, values[i].item())
    
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
        import torch.nn.functional as F
        
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
    
    def shutdown(self) -> None:
        """Shutdown the worker thread."""
        self.running = False
        with self.queue_lock:
            self.queue_not_empty.notify_all()
        self.worker_thread.join()