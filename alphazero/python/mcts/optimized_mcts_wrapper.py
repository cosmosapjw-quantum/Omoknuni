# File: alphazero/python/mcts/optimized_mcts_wrapper.py

import numpy as np
from typing import Dict, Tuple, List, Callable, Optional, Union
import threading
import queue
import time

from alphazero.python.games.game_base import GameWrapper
from alphazero.python.mcts.batch_evaluator import BatchEvaluator

try:
    from alphazero.bindings import cpp_mcts
    CPP_MCTS_AVAILABLE = True
except ImportError:
    CPP_MCTS_AVAILABLE = False
    print("Warning: C++ MCTS implementation not available")

class BatchedEvaluator:
    """
    Wrapper for direct batch processing of states.
    """
    def __init__(self, network, board_size, batch_size=16):
        """
        Initialize the batched evaluator.
        
        Args:
            network: Neural network model
            board_size: Size of the game board
            batch_size: Maximum size for evaluation batches
        """
        self.network = network
        self.board_size = board_size
        self.batch_evaluator = BatchEvaluator(network, batch_size)
    
    def __call__(self, game_state):
        """
        Standard interface for game state evaluation.
        
        Args:
            game_state: Game state to evaluate
            
        Returns:
            Tuple of (policy_dict, value)
        """
        state_tensor = game_state.get_state_tensor()
        state_flat = state_tensor.flatten().tolist()
        
        # Use batch evaluation with batch size 1
        policies, values = self.batch_evaluator.evaluate_batch([state_flat])
        
        if not policies or not values:
            return {}, 0.0
        
        # Convert flat policy to dictionary
        policy_dict = {}
        legal_moves = game_state.get_legal_moves()
        
        for move in legal_moves:
            if 0 <= move < len(policies[0]):
                policy_dict[move] = policies[0][move]
        
        # Normalize
        total = sum(policy_dict.values())
        if total > 0:
            policy_dict = {k: v / total for k, v in policy_dict.items()}
        elif legal_moves:
            policy_dict = {move: 1.0 / len(legal_moves) for move in legal_moves}
        
        return policy_dict, values[0]

class OptimizedMCTSWrapper:
    """
    Optimized MCTS implementation using batch evaluation.
    """
    def __init__(self, 
                 game: GameWrapper, 
                 evaluator: Callable,
                 c_puct: float = 1.5,
                 num_simulations: int = 800,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_noise_weight: float = 0.25,
                 temperature: float = 1.0,
                 use_transposition_table: bool = True,
                 transposition_table_size: int = 100000,
                 num_threads: int = 4,
                 batch_size: int = 16):
        """
        Initialize the optimized MCTS wrapper.
        
        Args:
            game: Game state
            evaluator: Neural network evaluator
            c_puct: Exploration constant
            num_simulations: Number of MCTS simulations
            dirichlet_alpha: Alpha parameter for Dirichlet noise
            dirichlet_noise_weight: Weight of Dirichlet noise
            temperature: Temperature for move selection
            use_transposition_table: Whether to use transposition table
            transposition_table_size: Size of transposition table
            num_threads: Number of simulation threads
            batch_size: Batch size for evaluation
        """
        self.game = game
        self.board_size = game.board_size
        self.temperature = temperature
        
        # Create batched evaluator if network is available
        if hasattr(evaluator, 'network'):
            self.evaluator = BatchedEvaluator(
                network=evaluator.network,
                board_size=self.board_size,
                batch_size=batch_size
            )
        else:
            # Use the provided evaluator directly
            self.evaluator = evaluator
        
        # Check if we're using the C++ MCTS implementation
        if CPP_MCTS_AVAILABLE:
            # Create C++ MCTS
            self.mcts = cpp_mcts.MCTS(
                c_puct=c_puct,
                num_simulations=num_simulations,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_noise_weight=dirichlet_noise_weight,
                virtual_loss_weight=1.0,
                use_transposition_table=use_transposition_table,
                transposition_table_size=transposition_table_size,
                num_threads=num_threads
            )
            self.using_cpp = True
        else:
            # Fall back to Python MCTS
            from alphazero.python.mcts.mcts import MCTS
            self.mcts = MCTS(
                game=game,
                evaluator=self.evaluator,
                c_puct=c_puct,
                num_simulations=num_simulations,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_noise_weight=dirichlet_noise_weight,
                temperature=temperature
            )
            self.using_cpp = False
    
    def select_move(self, temperature: Optional[float] = None, return_probs: bool = False) -> Union[int, Tuple[int, Dict[int, float]]]:
        """
        Select a move using MCTS search.
        
        Args:
            temperature: Temperature for move selection (overrides instance value if provided)
            return_probs: Whether to return probabilities as well
            
        Returns:
            The selected move, or a tuple of (move, probabilities) if return_probs is True
        """
        # Use provided or default temperature
        temp = temperature if temperature is not None else self.temperature
        
        if self.using_cpp:
            # Get current state and legal moves
            state_tensor = self.game.get_state_tensor().flatten().tolist()
            legal_moves = self.game.get_legal_moves()
            
            # Define evaluator function for C++ MCTS
            def cpp_evaluator(state):
                # Create a temporary array
                state_array = np.array(state)
                
                # Reshape to match expected shape if possible
                channels = 3  # Assuming 3 channels
                if len(state) == self.board_size * self.board_size * channels:
                    state_array = state_array.reshape(channels, self.board_size, self.board_size)
                
                # Create a temporary game state wrapper
                class TempGameState:
                    def __init__(self, tensor, board_size, legal_moves):
                        self.tensor = tensor
                        self.board_size = board_size
                        self._legal_moves = legal_moves
                    
                    def get_state_tensor(self):
                        return self.tensor
                    
                    def get_legal_moves(self):
                        return self._legal_moves
                
                # Create a temporary game state
                temp_game = TempGameState(
                    tensor=state_array,
                    board_size=self.board_size,
                    legal_moves=list(range(self.board_size * self.board_size))
                )
                
                # Get policy and value
                policy_dict, value = self.evaluator(temp_game)
                
                # Convert policy dictionary to array
                policy_array = np.zeros(self.board_size * self.board_size)
                for move, prob in policy_dict.items():
                    if 0 <= move < len(policy_array):
                        policy_array[move] = prob
                
                return policy_array.tolist(), value
            
            # Run search
            probs = self.mcts.search(state_tensor, legal_moves, cpp_evaluator)
            
            # Select move
            move = self.mcts.select_move(temp)
        else:
            # Using Python MCTS
            if temp != self.temperature:
                self.mcts.temperature = temp
            
            # Get move and probabilities
            if return_probs:
                move, probs = self.mcts.select_move(return_probs=True)
            else:
                move = self.mcts.select_move()
                probs = {}
        
        if return_probs:
            return move, probs
        return move
    
    def update_with_move(self, move: int) -> None:
        """
        Update the search tree with a move.
        
        Args:
            move: The move to update with
        """
        if self.using_cpp:
            self.mcts.update_with_move(move)
        else:
            # For Python MCTS
            self.game.apply_move(move)
            self.mcts.update_with_move(move)
    
    def set_temperature(self, temperature: float) -> None:
        """
        Set the temperature parameter.
        
        Args:
            temperature: Temperature for move selection
        """
        self.temperature = temperature
        if self.using_cpp:
            self.mcts.set_temperature(temperature)
        else:
            self.mcts.temperature = temperature

class ParallelOptimizedMCTSWrapper(OptimizedMCTSWrapper):
    """
    MCTS wrapper with parallel batch evaluation for even better performance.
    """
    def __init__(self, 
                 game: GameWrapper, 
                 evaluator: Callable,
                 c_puct: float = 1.5,
                 num_simulations: int = 800,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_noise_weight: float = 0.25,
                 temperature: float = 1.0,
                 use_transposition_table: bool = True,
                 transposition_table_size: int = 100000,
                 num_threads: int = 4,
                 batch_size: int = 16,
                 eval_queue_size: int = 1024):
        """
        Initialize the parallel optimized MCTS wrapper.
        
        This version uses a background thread to continuously process
        evaluation requests in batches for maximum throughput.
        """
        super().__init__(
            game=game,
            evaluator=evaluator,
            c_puct=c_puct,
            num_simulations=num_simulations,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_noise_weight=dirichlet_noise_weight,
            temperature=temperature,
            use_transposition_table=use_transposition_table,
            transposition_table_size=transposition_table_size,
            num_threads=num_threads,
            batch_size=batch_size
        )
        
        # Only setup parallel processing if we have a network
        if hasattr(evaluator, 'network'):
            # Create the batch evaluator
            self.batch_evaluator = BatchEvaluator(evaluator.network, batch_size)
            
            # Setup background evaluation thread
            self.eval_queue = queue.Queue(maxsize=eval_queue_size)
            self.eval_results = {}
            self.eval_lock = threading.Lock()
            self.stopping = False
            
            # Start evaluation thread
            self.eval_thread = threading.Thread(target=self._evaluation_worker)
            self.eval_thread.daemon = True
            self.eval_thread.start()
    
    def _evaluation_worker(self):
        """Background thread that processes evaluation requests in batches."""
        while not self.stopping:
            batch = []
            state_ids = []
            
            # Collect items up to batch size or wait timeout
            try:
                # Get at least one item
                state_id, state = self.eval_queue.get(timeout=0.001)
                batch.append(state)
                state_ids.append(state_id)
                
                # Try to get more items without blocking
                for _ in range(self.batch_size - 1):
                    try:
                        state_id, state = self.eval_queue.get_nowait()
                        batch.append(state)
                        state_ids.append(state_id)
                    except queue.Empty:
                        break
                
                # Process batch
                if batch:
                    # Use batch evaluation
                    policies, values = self.batch_evaluator.evaluate_batch(batch)
                    
                    # Store results
                    with self.eval_lock:
                        for i, state_id in enumerate(state_ids):
                            if i < len(policies) and i < len(values):
                                self.eval_results[state_id] = (policies[i], values[i])
                
                # Mark tasks as done
                for _ in range(len(batch)):
                    self.eval_queue.task_done()
                    
            except queue.Empty:
                # No items in queue, short sleep
                time.sleep(0.001)
            except Exception as e:
                print(f"Error in evaluation worker: {e}")
    
    def _get_cpp_evaluator(self):
        """
        Create an evaluator function for C++ MCTS that uses the background thread.
        """
        def evaluator_func(state):
            # Create unique ID for this state
            state_id = hash(tuple(state))
            
            # Check if already evaluated
            with self.eval_lock:
                if state_id in self.eval_results:
                    result = self.eval_results[state_id]
                    del self.eval_results[state_id]  # Clean up
                    return result
            
            # Queue for evaluation
            try:
                self.eval_queue.put((state_id, state), block=False)
            except queue.Full:
                # If queue is full, fall back to direct evaluation
                return super()._get_cpp_evaluator()(state)
            
            # Wait for result
            max_wait = 1.0  # Maximum wait time in seconds
            start_time = time.time()
            while time.time() - start_time < max_wait:
                with self.eval_lock:
                    if state_id in self.eval_results:
                        result = self.eval_results[state_id]
                        del self.eval_results[state_id]  # Clean up
                        return result
                
                # Short sleep to avoid busy waiting
                time.sleep(0.001)
            
            # If we timed out, fall back to direct evaluation
            return super()._get_cpp_evaluator()(state)
        
        return evaluator_func
    
    def select_move(self, temperature: Optional[float] = None, return_probs: bool = False) -> Union[int, Tuple[int, Dict[int, float]]]:
        """
        Select a move using parallel MCTS search.
        """
        if self.using_cpp and hasattr(self, 'batch_evaluator'):
            # Get current state and legal moves
            state_tensor = self.game.get_state_tensor().flatten().tolist()
            legal_moves = self.game.get_legal_moves()
            
            # Use provided or default temperature
            temp = temperature if temperature is not None else self.temperature
            
            # Run search with parallel evaluator
            probs = self.mcts.search(state_tensor, legal_moves, self._get_cpp_evaluator())
            
            # Select move
            move = self.mcts.select_move(temp)
            
            if return_probs:
                return move, probs
            return move
        else:
            # Fall back to standard implementation
            return super().select_move(temperature, return_probs)
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'stopping'):
            self.stopping = True
            if hasattr(self, 'eval_thread') and self.eval_thread.is_alive():
                self.eval_thread.join(timeout=1.0)