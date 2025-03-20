"""
Improved Python wrapper for the C++ MCTS implementation with better GIL handling

This wrapper is designed to safely use multithreading from C++ without any
Python-side threading or GIL issues.
"""

import numpy as np
from typing import List, Dict, Tuple, Callable, Optional, Any, Union
import time
import threading
import sys
import os

try:
    # Try to import the improved version
    from alphazero.bindings.improved_cpp_mcts import GomokuMCTS as CppGomokuMCTS
    CPP_MCTS_AVAILABLE = True
    print("Using improved C++ MCTS implementation with proper GIL handling")
except ImportError:
    try:
        # Fall back to the original version
        from alphazero.bindings.cpp_mcts import GomokuMCTS as CppGomokuMCTS
        CPP_MCTS_AVAILABLE = True
        print("Warning: Using original C++ MCTS implementation. Multithreading may cause issues with the Python GIL.")
    except ImportError:
        print("Warning: C++ MCTS implementation not available. Using Python implementation instead.")
        CPP_MCTS_AVAILABLE = False

from alphazero.python.games.game_base import GameWrapper


class ImprovedCppMCTSWrapper:
    """
    Improved Python wrapper for the C++ MCTS implementation.
    This class provides a similar interface to the Python MCTS class,
    but delegates the actual search to the C++ implementation for better performance.
    
    Key improvements:
    - Better handling of Python's GIL when using multithreading
    - Thread-safe evaluator function handling
    - Better error reporting and recovery
    """
    
    def __init__(self, 
                 game: GameWrapper, 
                 evaluator: Callable[[GameWrapper], Tuple[Dict[int, float], float]],
                 c_puct: float = 1.5,
                 num_simulations: int = 800,
                 dirichlet_alpha: float = 0.3,
                 dirichlet_noise_weight: float = 0.25,
                 temperature: float = 1.0,
                 use_transposition_table: bool = True,
                 transposition_table_size: int = 1000000,
                 num_threads: int = 1):
        """
        Initialize the improved MCTS algorithm.
        
        Args:
            game: The game to search for moves
            evaluator: Function that evaluates a game state and returns (move_priors, value)
            c_puct: Exploration constant for UCB
            num_simulations: Number of simulations to run per search
            dirichlet_alpha: Alpha parameter for Dirichlet noise
            dirichlet_noise_weight: Weight of Dirichlet noise added to root prior probabilities
            temperature: Temperature parameter for move selection
            use_transposition_table: Whether to use a transposition table
            transposition_table_size: Maximum size of the transposition table
            num_threads: Number of threads for parallel search - C++ manages all threading
        """
        if not CPP_MCTS_AVAILABLE:
            raise ImportError("C++ MCTS implementation not available")
        
        self.game = game
        self.evaluator = evaluator
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_noise_weight = dirichlet_noise_weight
        self.num_threads = num_threads
        
        # Create the C++ MCTS object
        self.cpp_mcts = CppGomokuMCTS(
            num_simulations=num_simulations,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_noise_weight=dirichlet_noise_weight,
            virtual_loss_weight=1.0,  # Default value
            use_transposition_table=use_transposition_table,
            transposition_table_size=transposition_table_size,
            num_threads=num_threads
        )
        
        # Set the temperature
        self.cpp_mcts.set_temperature(temperature)
        
        # Track statistics
        self.eval_count = 0
        self.search_time = 0.0
        self.search_count = 0
    
    def search(self, game_state: Optional[GameWrapper] = None) -> Dict[int, float]:
        """
        Perform MCTS search from the given game state.
        All threading is managed by the C++ side.
        
        Args:
            game_state: The game state to start the search from (uses self.game if None)
            
        Returns:
            Dictionary mapping moves to search probabilities
        """
        if game_state is not None:
            self.game = game_state
        
        # Get the legal moves
        legal_moves = self.game.get_legal_moves()
        
        # Get the board as a numpy array
        board = np.array(self.game.get_board()).flatten()
        
        # Reset evaluation counter for this search
        self.eval_count = 0
        
        # Create a thread-safe wrapper function for the evaluator
        def evaluator_wrapper(board_flat):
            try:
                # Track number of evaluations
                self.eval_count += 1
                
                # Create a game copy for evaluation
                game_copy = self.game.clone()
                
                # Call the Python evaluator
                priors, value = self.evaluator(game_copy)
                
                # Convert the prior dictionary to a flat list
                board_size = self.game.board_size
                total_cells = board_size * board_size
                
                prior_list = [0.0] * total_cells
                for move, prior in priors.items():
                    if isinstance(move, int) and 0 <= move < total_cells:
                        prior_list[move] = prior
                
                return prior_list, value
            except Exception as e:
                print(f"Error in evaluator: {e}")
                # Return default values in case of error
                return [1.0 / len(board_flat)] * len(board_flat), 0.0
        
        # Run the C++ search
        try:
            start_time = time.time()
            
            # The C++ side will handle all threading and GIL management
            probabilities = self.cpp_mcts.search(board, legal_moves, evaluator_wrapper)
            
            elapsed = time.time() - start_time
            self.search_time += elapsed
            self.search_count += 1
            
            if self.search_count % 10 == 0:
                avg_time = self.search_time / self.search_count
                print(f"Average search time: {avg_time:.3f}s over {self.search_count} searches")
            
            return probabilities
        except Exception as e:
            print(f"Error in C++ search: {e}")
            # Return uniform probabilities as a fallback
            return {move: 1.0 / len(legal_moves) for move in legal_moves}
    
    def select_move(self, return_probs: bool = False) -> Union[int, Tuple[int, Dict[int, float]]]:
        """
        Select a move to play based on the search probabilities.
        
        Args:
            return_probs: Whether to return the search probabilities as well
            
        Returns:
            The selected move, or a tuple of (move, probabilities) if return_probs is True
        """
        # Get probabilities from search
        probs = self.search()
        
        try:
            # Select move using the C++ implementation
            move = self.cpp_mcts.select_move(self.temperature)
        except Exception as e:
            print(f"Error in C++ MCTS select_move: {e}")
            # Fallback to selecting a move based on the probabilities
            moves = list(probs.keys())
            probs_list = [probs[move] for move in moves]
            
            # Ensure probabilities sum to 1
            sum_probs = sum(probs_list)
            if abs(sum_probs - 1.0) > 1e-10 and sum_probs > 0:
                probs_list = [p / sum_probs for p in probs_list]
                
            # Handle the case where all probabilities are zero
            if all(p == 0 for p in probs_list) and moves:
                # Use uniform distribution as a fallback
                probs_list = [1.0 / len(moves)] * len(moves)
                
            move = np.random.choice(moves, p=probs_list)
        
        if return_probs:
            return move, probs
        return move
    
    def update_with_move(self, move: int) -> None:
        """
        Update the tree with the given move.
        
        Args:
            move: The move to update with
        """
        # Validate the move
        if not isinstance(move, int):
            print(f"Warning: Invalid move type {type(move)}, expected int")
            return
            
        board_size = self.game.board_size
        if move < 0 or move >= board_size * board_size:
            print(f"Warning: Move {move} is out of bounds for board size {board_size}")
            return
            
        # Try to update with the move, with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.cpp_mcts.update_with_move(move)
                return  # Success
            except Exception as e:
                print(f"Error in C++ MCTS update_with_move (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    # On the last attempt, reset the MCTS instead
                    print("Resetting MCTS tree due to persistent errors")
                    try:
                        # Create a new MCTS instance with the same parameters
                        self.cpp_mcts = CppGomokuMCTS(
                            num_simulations=self.cpp_mcts.get_num_simulations(),
                            c_puct=1.5,  # Default value
                            dirichlet_alpha=self.dirichlet_alpha,
                            dirichlet_noise_weight=self.dirichlet_noise_weight,
                            virtual_loss_weight=1.0,
                            use_transposition_table=True,
                            transposition_table_size=1000000,
                            num_threads=self.num_threads
                        )
                        self.cpp_mcts.set_temperature(self.temperature)
                    except Exception as reset_error:
                        print(f"Error resetting MCTS: {reset_error}")
                else:
                    # On earlier attempts, wait a bit before retrying
                    time.sleep(0.1)