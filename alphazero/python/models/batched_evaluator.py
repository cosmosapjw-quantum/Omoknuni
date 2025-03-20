import torch
import numpy as np
import threading
import time
import queue
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


class PyBatchEvaluator:
    """
    Python implementation of batch evaluation for neural networks.
    
    This class provides similar functionality to the C++ BatchEvaluator class
    but operates entirely in Python. It collects positions and evaluates them
    in batches using a dedicated worker thread.
    """
    
    def __init__(self, 
                 model: Any,
                 batch_size: int = 16,
                 max_wait_ms: int = 10,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the batch evaluator.
        
        Args:
            model: The neural network model to evaluate positions
            batch_size: Maximum batch size
            max_wait_ms: Maximum time to wait before processing a non-full batch (ms)
            device: Device to run the model on ('cpu' or 'cuda')
        """
        self.model = model
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms / 1000.0  # Convert to seconds
        self.device = device
        
        # Request/response queues
        self.request_queue = queue.Queue()
        self.response_dict = {}
        self.response_lock = threading.Lock()
        
        # State
        self.next_request_id = 0
        self.running = False
        self.stop_requested = False
        
        # Statistics
        self.positions_evaluated = 0
        self.batches_processed = 0
    
    def start(self):
        """Start the batch evaluation worker thread."""
        if not self.running:
            self.stop_requested = False
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker_loop)
            self.worker_thread.daemon = True
            self.worker_thread.start()
    
    def stop(self):
        """Stop the batch evaluation worker thread."""
        if self.running:
            self.stop_requested = True
            if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=1.0)
            self.running = False
    
    def enqueue_position(self, position):
        """
        Enqueue a position for evaluation.
        
        Args:
            position: The position to evaluate (as expected by the model)
            
        Returns:
            int: A unique request ID used to retrieve the result
        """
        request_id = self.next_request_id
        self.next_request_id += 1
        
        self.request_queue.put((request_id, position))
        
        return request_id
    
    def get_result(self, request_id):
        """
        Get the result of a previously enqueued position.
        
        Args:
            request_id: The request ID returned by enqueue_position
            
        Returns:
            tuple: (policy, value) for the position
        """
        # Wait until the result is available
        while True:
            with self.response_lock:
                if request_id in self.response_dict:
                    result = self.response_dict[request_id]
                    del self.response_dict[request_id]
                    return result
            
            if self.stop_requested:
                return {}, 0.0
            
            time.sleep(0.001)  # Short sleep to avoid tight looping
    
    def get_stats(self):
        """
        Get statistics about the batch evaluator.
        
        Returns:
            tuple: (positions_evaluated, batches_processed)
        """
        return self.positions_evaluated, self.batches_processed
    
    def reset_stats(self):
        """Reset the statistics counters."""
        self.positions_evaluated = 0
        self.batches_processed = 0
    
    def _worker_loop(self):
        """Main worker loop that processes batches."""
        while not self.stop_requested:
            self._process_next_batch()
    
    def _process_next_batch(self):
        """Process the next batch of positions."""
        # Collect a batch of positions
        batch_ids = []
        batch_positions = []
        
        # Try to fill the batch
        try:
            # Always get at least one item, waiting if necessary
            item = self.request_queue.get(timeout=self.max_wait_ms)
            request_id, position = item
            batch_ids.append(request_id)
            batch_positions.append(position)
            self.request_queue.task_done()
            
            # Try to get more items without blocking
            while len(batch_positions) < self.batch_size:
                try:
                    item = self.request_queue.get_nowait()
                    request_id, position = item
                    batch_ids.append(request_id)
                    batch_positions.append(position)
                    self.request_queue.task_done()
                except queue.Empty:
                    break
        except queue.Empty:
            # No items available even after waiting
            return
        
        # Process the batch if we have positions
        if batch_positions:
            try:
                # Prepare the batch for the model
                with torch.no_grad():
                    # Convert to the format expected by the model
                    # This will depend on the specific model being used
                    if isinstance(batch_positions[0], np.ndarray):
                        batch_tensor = torch.from_numpy(np.stack(batch_positions)).to(self.device)
                    elif isinstance(batch_positions[0], torch.Tensor):
                        batch_tensor = torch.stack(batch_positions).to(self.device)
                    else:
                        # For other types, we'll need custom conversion
                        batch_tensor = self._convert_batch(batch_positions)
                    
                    # Call the model with the batch
                    policy_batch, value_batch = self.model(batch_tensor)
                
                # Store the results
                with self.response_lock:
                    for i, request_id in enumerate(batch_ids):
                        if i < len(policy_batch) and i < len(value_batch):
                            self.response_dict[request_id] = (policy_batch[i], value_batch[i])
                
                # Update statistics
                self.positions_evaluated += len(batch_positions)
                self.batches_processed += 1
            except Exception as e:
                print(f"Error in batch evaluation: {e}")
                
                # Create default results for all items in the batch
                with self.response_lock:
                    for request_id in batch_ids:
                        # Default policy is uniform random, value is 0
                        default_policy = {}
                        self.response_dict[request_id] = (default_policy, 0.0)
    
    def _convert_batch(self, batch_positions):
        """
        Convert a batch of positions to the format expected by the model.
        This method should be overridden for different model types.
        
        Args:
            batch_positions: List of positions
            
        Returns:
            torch.Tensor: Batch tensor for the model
        """
        # Default implementation assumes a numpy array representation
        return torch.from_numpy(np.stack(batch_positions)).to(self.device)


class BatchProcessor:
    """
    A simple utility to process data in batches.
    
    This class collects items and processes them in batches of a specified size,
    which is useful for operations that benefit from batching, like neural
    network inference.
    """
    
    def __init__(self, process_func: Callable, batch_size: int = 16):
        """
        Initialize the batch processor.
        
        Args:
            process_func: Function that processes a batch and returns results
            batch_size: Size of batches to process
        """
        self.process_func = process_func
        self.batch_size = batch_size
        self.items = []
        self.results = []
    
    def add(self, item):
        """
        Add an item to the batch.
        
        Args:
            item: The item to add
            
        Returns:
            bool: True if the batch was processed, False otherwise
        """
        self.items.append(item)
        
        # Process the batch if it's full
        if len(self.items) >= self.batch_size:
            self.process()
            return True
        
        return False
    
    def process(self):
        """
        Process the current batch.
        
        Returns:
            List: The results from processing the batch
        """
        if not self.items:
            return []
        
        # Process the batch
        results = self.process_func(self.items)
        
        # Store and return the results
        self.results.extend(results)
        self.items = []
        
        return results
    
    def get_results(self):
        """
        Get all results processed so far.
        
        Returns:
            List: All results
        """
        # Process any remaining items
        if self.items:
            self.process()
        
        # Get and clear the results
        results = self.results
        self.results = []
        
        return results