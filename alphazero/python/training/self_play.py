import numpy as np
import time
import random
from typing import List, Dict, Tuple, Any, Optional
import multiprocessing as mp
from tqdm import tqdm

from alphazero.python.games.game_base import GameWrapper
from alphazero.python.mcts.mcts import MCTS


class GameRecord:
    """
    Record of a game played for training.
    
    Attributes:
        states: List of game state tensors
        policies: List of policy distributions from MCTS
        values: List of outcome values from the final game result
    """
    def __init__(self):
        self.states: List[np.ndarray] = []
        self.policies: List[Dict[int, float]] = []
        self.values: List[float] = []
    
    def add_step(self, state: np.ndarray, policy: Dict[int, float]) -> None:
        """
        Add a step to the game record.
        
        Args:
            state: Game state tensor
            policy: Policy distribution from MCTS
        """
        self.states.append(state)
        self.policies.append(policy)
    
    def add_values(self, result: int, current_player: int) -> None:
        """
        Add outcome values based on the game result.
        
        Args:
            result: Game result (0 for draw, 1 for black win, 2 for white win)
            current_player: Current player at the end of the game
        """
        # If the game is a draw, all values are 0
        if result == 0:
            self.values = [0.0] * len(self.states)
            return
        
        # Initialize values list with the same length as states
        self.values = [0.0] * len(self.states)
        
        # If we have a winner, we need to determine the value for each step
        for i in range(len(self.states)):
            # For even indices, the player is the first player (alternating)
            player_at_step = 1 if i % 2 == 0 else 2
            
            if result == player_at_step:
                # This player won
                self.values[i] = 1.0
            else:
                # This player lost
                self.values[i] = -1.0
                
        # Verify that values length matches states length
        if len(self.values) != len(self.states):
            raise ValueError(f"Length mismatch: values ({len(self.values)}) and states ({len(self.states)})")
    
    def get_samples(self) -> Tuple[List[np.ndarray], List[Dict[int, float]], List[float]]:
        """
        Get the training samples from this game.
        
        Returns:
            Tuple of (states, policies, values)
        """
        return self.states, self.policies, self.values


class SelfPlay:
    """
    Self-play module for generating training data.
    """
    def __init__(self, 
                 game_class,
                 network,
                 game_args: Dict[str, Any] = None,
                 mcts_args: Dict[str, Any] = None):
        """
        Initialize the self-play module.
        
        Args:
            game_class: Game class to use
            network: Neural network for evaluation
            game_args: Arguments for game initialization
            mcts_args: Arguments for MCTS initialization
        """
        self.game_class = game_class
        self.network = network
        self.game_args = game_args or {}
        self.mcts_args = mcts_args or {}
    
    def play_game(self, temperature_cutoff: int = 30) -> GameRecord:
        """
        Play a single game using MCTS with the current network.
        
        Args:
            temperature_cutoff: Move number after which temperature is set to ~0
            
        Returns:
            GameRecord object containing the game data
        """
        game = self.game_class(**self.game_args)
        
        # Create MCTS with default arguments
        mcts_args = {
            "c_puct": 1.5,
            "num_simulations": 800,
            "dirichlet_alpha": 0.3,
            "dirichlet_noise_weight": 0.25,
            "temperature": 1.0
        }
        mcts_args.update(self.mcts_args)
        
        mcts = MCTS(
            game=game,
            evaluator=self.network.predict,
            **mcts_args
        )
        
        record = GameRecord()
        move_count = 0
        
        while not game.is_terminal():
            # Adjust temperature based on move count
            if move_count >= temperature_cutoff:
                mcts.temperature = 0.01  # Almost zero temperature for deterministic play
            
            # Run MCTS search
            state_tensor = game.get_state_tensor()
            move, policy = mcts.select_move(return_probs=True)
            
            # Add to game record
            record.add_step(state_tensor, policy)
            
            # Apply the move
            game.apply_move(move)
            
            # Update MCTS with the move
            mcts.update_with_move(move)
            
            move_count += 1
        
        # Game over, get the result
        winner = game.get_winner()
        record.add_values(winner, game.get_current_player())
        
        return record
    
    def generate_games(self, num_games: int, num_workers: int = 1) -> List[GameRecord]:
        """
        Generate multiple games in parallel.
        
        Args:
            num_games: Number of games to generate
            num_workers: Number of parallel workers
            
        Returns:
            List of GameRecord objects
        """
        # Use multiprocessing only if more than one worker and enough games
        if num_workers > 1 and num_games >= num_workers:
            try:
                # Set a timeout for each game to prevent hanging
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                # Method already set, ignore
                pass
                
            records = []
            # Use context manager for proper cleanup
            with mp.Pool(num_workers) as pool:
                try:
                    # Track progress with tqdm
                    game_iter = pool.imap_unordered(self._play_game_wrapper, [None] * num_games)
                    for record in tqdm(game_iter, total=num_games, desc="Generating games"):
                        if record is not None:
                            records.append(record)
                except Exception as e:
                    print(f"Error in parallel game generation: {e}")
                    # Fallback to sequential mode if parallel fails
                    pool.terminate()
                    pool.join()
                    remaining = num_games - len(records)
                    if remaining > 0:
                        print(f"Falling back to sequential mode for {remaining} remaining games")
                        for _ in tqdm(range(remaining), desc="Generating additional games"):
                            try:
                                records.append(self.play_game())
                            except Exception as game_error:
                                print(f"Error generating game: {game_error}")
        else:
            # Sequential game generation
            records = []
            for _ in tqdm(range(num_games), desc="Generating games"):
                try:
                    records.append(self.play_game())
                except Exception as e:
                    print(f"Error generating game: {e}")
                    
        # Ensure we return valid records
        return [r for r in records if r is not None]
    
    def _play_game_wrapper(self, _) -> Optional[GameRecord]:
        """
        Wrapper for parallel game generation with error handling.
        
        Returns:
            GameRecord or None if an error occurred
        """
        try:
            return self.play_game()
        except Exception as e:
            print(f"Error in game generation: {e}")
            return None