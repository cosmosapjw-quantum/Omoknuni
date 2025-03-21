# Create file: alphazero/python/mcts/safe_cpp_mcts_wrapper.py

import numpy as np
from typing import List, Dict, Tuple, Callable, Optional, Any

try:
    from alphazero.bindings.cpp_mcts import GomokuMCTS as CppGomokuMCTS
    CPP_MCTS_AVAILABLE = True
except ImportError:
    print("Warning: C++ MCTS implementation not available")
    CPP_MCTS_AVAILABLE = False

from alphazero.python.games.game_base import GameWrapper


class SafeCppMCTSWrapper:
    """
    A wrapper around the C++ MCTS implementation with enhanced safety checks.
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
                 transposition_table_size: int = 100000,
                 num_threads: int = 1):
        """
        Initialize the C++ MCTS algorithm with safety checks.
        
        Args:
            game: The game to search for moves
            evaluator: Function that evaluates a game state and returns (policy, value)
            c_puct: Exploration constant for UCB
            num_simulations: Number of simulations to run per search
            dirichlet_alpha: Alpha parameter for Dirichlet noise
            dirichlet_noise_weight: Weight of Dirichlet noise added to root prior probabilities
            temperature: Temperature parameter for move selection
            use_transposition_table: Whether to use a transposition table
            transposition_table_size: Maximum size of the transposition table
            num_threads: Number of worker threads
        """
        if not CPP_MCTS_AVAILABLE:
            raise ImportError("C++ MCTS implementation not available")
        
        self.game = game
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_noise_weight = dirichlet_noise_weight
        self.temperature = temperature
        self.use_transposition_table = use_transposition_table
        self.transposition_table_size = transposition_table_size
        self.num_threads = num_threads
        
        # Create the C++ MCTS
        print(f"Creating C++ MCTS with {num_threads} thread(s)")
        self.cpp_mcts = CppGomokuMCTS(
            num_simulations=num_simulations,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_noise_weight=dirichlet_noise_weight,
            virtual_loss_weight=1.0,
            use_transposition_table=use_transposition_table,
            transposition_table_size=transposition_table_size,
            num_threads=num_threads
        )
        
        # Set the temperature
        self.cpp_mcts.set_temperature(temperature)
        
        # Store probabilities from the last search
        self.current_probabilities = {}
        
        # Store the last board state for validation
        self.last_board_state = None
    
    def search(self, game_state: Optional[GameWrapper] = None) -> Dict[int, float]:
        """
        Perform MCTS search from the given game state with safety checks.
        
        Args:
            game_state: The game state to start the search from (uses self.game if None)
            
        Returns:
            Dictionary mapping moves to search probabilities
        """
        if game_state is not None:
            self.game = game_state
        
        # Store the current board state for validation
        self.last_board_state = np.array(self.game.get_board())
            
        # Get the legal moves
        legal_moves = self.game.get_legal_moves()
        
        # Check if there are any legal moves
        if not legal_moves:
            print("No legal moves available")
            return {}
        
        # Get the board as a flat array
        board = self.game.get_board().flatten()
        
        # Create a wrapper around the evaluator
        def safe_evaluator(board_flat):
            try:
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
                
                # Ensure there's at least some non-zero value to prevent division by zero
                if sum(prior_list) == 0:
                    print("Warning: All zero priors, using uniform distribution")
                    # Use uniform distribution for legal moves
                    for move in legal_moves:
                        prior_list[move] = 1.0 / len(legal_moves)
                
                return prior_list, value
            except Exception as e:
                print(f"Error in evaluator: {e}")
                import traceback
                traceback.print_exc()
                
                # Return uniform distribution and random value as fallback
                board_size = self.game.board_size
                total_cells = board_size * board_size
                prior_list = [0.0] * total_cells
                for move in legal_moves:
                    prior_list[move] = 1.0 / len(legal_moves)
                return prior_list, 0.0
        
        # Run the C++ search
        try:
            self.cpp_mcts.set_num_simulations(self.num_simulations)
            probabilities = self.cpp_mcts.search(board, legal_moves, safe_evaluator)
            
            # Safety check: ensure returned probabilities are for legal moves only
            filtered_probs = {move: prob for move, prob in probabilities.items() if move in legal_moves}
            
            # Re-normalize if needed
            total_prob = sum(filtered_probs.values())
            if total_prob > 0:
                filtered_probs = {move: prob/total_prob for move, prob in filtered_probs.items()}
            
            # Store the probabilities
            self.current_probabilities = filtered_probs
            
            return filtered_probs
        except Exception as e:
            print(f"Error in C++ MCTS search: {e}")
            import traceback
            traceback.print_exc()
            
            # Return uniform probabilities as fallback
            uniform_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
            self.current_probabilities = uniform_probs
            return uniform_probs
    
    def select_move(self, return_probs: bool = False):
        """
        Select a move based on the search probabilities with safety checks.
        
        Args:
            return_probs: Whether to return the probabilities as well
            
        Returns:
            The selected move, or (move, probabilities) if return_probs is True
        """
        # Get legal moves
        legal_moves = self.game.get_legal_moves()
        
        if not legal_moves:
            print("No legal moves available for selection")
            return -1 if not return_probs else (-1, {})
        
        # Get probabilities
        if not self.current_probabilities:
            self.search()
        
        # Safety check: ensure we have probabilities for at least some legal moves
        has_legal_probs = any(move in legal_moves for move in self.current_probabilities)
        
        if not has_legal_probs:
            print("No probabilities for legal moves, falling back to uniform distribution")
            self.current_probabilities = {move: 1.0 / len(legal_moves) for move in legal_moves}
        
        # Select a move based on the probabilities
        try:
            # First try to use the C++ select_move
            move = self.cpp_mcts.select_move(self.temperature)
            
            # Verify the move is legal
            if move not in legal_moves:
                print(f"Warning: C++ selected illegal move {move}, selecting from legal moves instead")
                move = self._sample_move_from_probs(legal_moves)
        except Exception as e:
            print(f"Error in C++ select_move: {e}, falling back to Python selection")
            move = self._sample_move_from_probs(legal_moves)
        
        if return_probs:
            return move, self.current_probabilities
        return move
    
    def _sample_move_from_probs(self, legal_moves):
        """Sample a move from available probabilities, ensuring it's legal."""
        # Filter for legal moves
        legal_probs = {move: self.current_probabilities.get(move, 0) 
                      for move in legal_moves}
        
        # Ensure non-zero probabilities
        if sum(legal_probs.values()) <= 0:
            # Use uniform distribution if no legal moves have probabilities
            legal_probs = {move: 1.0 / len(legal_moves) for move in legal_moves}
        
        # Normalize
        total = sum(legal_probs.values())
        normalized_probs = {move: prob/total for move, prob in legal_probs.items()}
        
        # Create lists for sampling
        moves = list(normalized_probs.keys())
        probs = list(normalized_probs.values())
        
        if not moves:
            return np.random.choice(legal_moves)
        
        # Sample a move
        try:
            move = np.random.choice(moves, p=probs)
        except Exception as e:
            print(f"Error sampling move: {e}")
            move = np.random.choice(legal_moves)
        
        return move
    
    def update_with_move(self, move):
        """
        Update the MCTS tree with the given move.
        
        Args:
            move: The move to update with
        """
        try:
            # Verify that the board state hasn't changed unexpectedly
            if self.last_board_state is not None:
                current_board = np.array(self.game.get_board())
                board_matches = np.array_equal(self.last_board_state, current_board)
                
                if not board_matches:
                    print("Warning: Board state changed unexpectedly between search and update")
                    # We still continue with the update
            
            self.cpp_mcts.update_with_move(move)
        except Exception as e:
            print(f"Error in update_with_move: {e}")
            import traceback
            traceback.print_exc()
        
        # Update our board state record
        try:
            self.last_board_state = np.array(self.game.get_board())
        except Exception:
            self.last_board_state = None
        
        # Clear the probabilities
        self.current_probabilities = {}