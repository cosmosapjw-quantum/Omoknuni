# Update file: alphazero/python/mcts/cpp_mcts_wrapper.py

import numpy as np
from typing import Dict, Tuple, Callable, Optional, Union

from alphazero.python.games.game_base import GameWrapper
from alphazero.python.mcts.batch_evaluator import BatchEvaluator

try:
    from alphazero.bindings import cpp_mcts
    from alphazero.bindings import batch_evaluator
    CPP_BATCH_AVAILABLE = True
except ImportError:
    CPP_BATCH_AVAILABLE = False

class CppMCTSWrapper:
    """
    Wrapper for C++ MCTS with batch evaluation.
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
        Initialize the C++ MCTS wrapper.
        """
        self.game = game
        self.temperature = temperature
        
        # Create batch evaluator
        if hasattr(evaluator, 'network'):
            # Create batch evaluator with network
            self.python_evaluator = BatchEvaluator(evaluator.network, batch_size)
        else:
            # Use evaluator directly
            self.python_evaluator = evaluator
        
        # Create MCTS instance
        self.mcts = cpp_mcts.MCTS(
            c_puct=c_puct,
            num_simulations=num_simulations,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_noise_weight=dirichlet_noise_weight,
            virtual_loss=1.0,
            use_transposition_table=use_transposition_table,
            transposition_table_size=transposition_table_size,
            num_threads=num_threads
        )
        
        # Create C++ batch evaluator if available
        if CPP_BATCH_AVAILABLE:
            self.cpp_evaluator = batch_evaluator.PyTorchBatchEvaluator(
                self.python_evaluator, batch_size)
    
    def select_move(self, temperature: Optional[float] = None, return_probs: bool = False):
        """
        Select a move using MCTS search.
        """
        # Get current state and legal moves
        state_tensor = self.game.get_state_tensor().flatten().tolist()
        legal_moves = self.game.get_legal_moves()
        
        # Use provided or default temperature
        temp = temperature if temperature is not None else self.temperature
        
        # Define evaluator function for C++ MCTS
        def evaluator_func(state):
            if CPP_BATCH_AVAILABLE:
                # Use C++ batch evaluator
                future = self.cpp_evaluator.submit(state)
                return future.get()
            else:
                # Fall back to Python evaluation
                policies, values = self.python_evaluator.evaluate_batch([state])
                if policies and values:
                    return policies[0], values[0]
                return [], 0.0
        
        # Run search
        probs = self.mcts.search(state_tensor, legal_moves, evaluator_func)
        
        # Select move
        move = self.mcts.select_move(temp)
        
        if return_probs:
            return move, probs
        return move
        
    def update_with_move(self, move: int) -> None:
        """
        Update the search tree with a move.
        """
        self.mcts.update_with_move(move)
    
    def set_temperature(self, temperature: float) -> None:
        """
        Set the temperature parameter.
        """
        self.temperature = temperature