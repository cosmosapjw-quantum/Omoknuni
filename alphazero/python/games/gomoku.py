import numpy as np
from typing import List, Tuple, Optional

from alphazero.python.games.game_base import GameWrapper

# Import the C++ modules
from alphazero.core.gomoku import Gamestate
from alphazero.core.attack_defense import AttackDefenseModule


class GomokuGame(GameWrapper):
    """Python wrapper for the C++ Gomoku implementation"""
    
    def __init__(self, board_size: int = 15, use_renju: bool = False, use_omok: bool = False, seed: int = 0, use_pro_long_opening: bool = False):
        """
        Initialize a new Gomoku game
        
        Args:
            board_size: Size of the board (default: 15x15)
            use_renju: Whether to use Renju rules (default: False)
            use_omok: Whether to use Omok rules (default: False)
            seed: Random seed for the game (default: 0)
            use_pro_long_opening: Whether to use professional long opening rules (default: False)
        """
        self.board_size = board_size
        self.use_renju = use_renju
        self.use_omok = use_omok
        self.use_pro_long_opening = use_pro_long_opening
        
        # Initialize the C++ game state
        self.state = Gamestate(board_size, use_renju, use_omok, seed, use_pro_long_opening)
        
        # Initialize attack/defense module
        self.attack_defense = AttackDefenseModule(board_size)
    
    def get_legal_moves(self) -> List[int]:
        """Return a list of legal moves in the current state"""
        return self.state.get_valid_moves()
    
    def is_terminal(self) -> bool:
        """Check if the current state is terminal (game over)"""
        return self.state.is_terminal()
    
    def get_winner(self) -> int:
        """Return the winner (0 for draw, 1 for black, 2 for white)"""
        return self.state.get_winner()
    
    def apply_move(self, move: int) -> None:
        """Apply a move to the current state"""
        self.state.make_move(move, self.state.current_player)
    
    def get_state_tensor(self) -> np.ndarray:
        """
        Convert the current state to a tensor representation for the neural network
        
        Returns:
            np.ndarray: A 3D tensor with shape (3, board_size, board_size)
                - Channel 0: Current player's stones (1 where stones exist, 0 elsewhere)
                - Channel 1: Opponent's stones (1 where stones exist, 0 elsewhere)
                - Channel 2: Current player indicator (1 everywhere if current player is Black, 0 if White)
        """
        tensor = self.state.to_tensor()
        return np.array(tensor, dtype=np.float32)
    
    def clone(self) -> 'GomokuGame':
        """Return a deep copy of the current game state"""
        new_game = GomokuGame(
            self.board_size, 
            self.use_renju, 
            self.use_omok, 
            0,  # Don't copy the seed
            self.use_pro_long_opening
        )
        new_game.state = self.state.copy()
        return new_game
    
    def get_current_player(self) -> int:
        """Return the current player (1 for Black, 2 for White)"""
        return self.state.current_player
    
    def is_five_in_a_row(self, action: int, player: int) -> bool:
        """
        Check if there is a five-in-a-row from the given action for the player
        
        Args:
            action: The action to check (or -1 to check the entire board)
            player: The player to check for (1 for Black, 2 for White)
            
        Returns:
            True if there is a five-in-a-row, False otherwise
        """
        return self.state.is_five_in_a_row(action, player)
    
    def undo_move(self, action: int) -> None:
        """
        Undo the move at the specified action
        
        Args:
            action: The action to undo
        """
        self.state.undo_move(action)
    
    def get_attack_defense_scores(self, moves: List[int]) -> Tuple[List[float], List[float]]:
        """
        Calculate attack and defense scores for the given moves
        
        Args:
            moves: List of moves to evaluate
            
        Returns:
            Tuple of (attack_scores, defense_scores)
        """
        if not moves:
            return [], []
        
        # Get the board as a numpy array
        board = np.array(self.state.get_board(), dtype=np.float32)
        
        # Reshape for the batch dimension
        batch_board = np.expand_dims(board, axis=0).repeat(len(moves), axis=0)
        
        # Create moves and player arrays
        moves_array = np.array(moves, dtype=np.int64)
        player_array = np.full(len(moves), self.state.current_player, dtype=np.int64)
        
        # Call the C++ implementation
        attack_scores, defense_scores = self.attack_defense(batch_board, moves_array, player_array)
        
        return attack_scores.tolist(), defense_scores.tolist()
    
    def get_board(self) -> np.ndarray:
        """
        Get the current board state as a 2D numpy array
        
        Returns:
            np.ndarray: 2D array where 0=empty, 1=black, 2=white
        """
        return np.array(self.state.get_board(), dtype=np.int32)
    
    def print_board(self) -> None:
        """Print the current board state to the console"""
        board = self.get_board()
        symbols = {0: ".", 1: "●", 2: "○"}
        
        print("   ", end="")
        for i in range(self.board_size):
            print(f"{i:2d}", end=" ")
        print()
        
        for i in range(self.board_size):
            print(f"{i:2d} ", end="")
            for j in range(self.board_size):
                print(f" {symbols[board[i][j]]}", end=" ")
            print()
                
    def reset(self) -> None:
        """Reset the game to the initial state"""
        # Reinitialize the C++ game state
        self.state = Gamestate(
            self.board_size, 
            self.use_renju, 
            self.use_omok, 
            0,  # Use default seed
            self.use_pro_long_opening
        )
        
    # These methods are now directly calling the C++ implementation