"""
Game Adapter Interface API Contract

This file defines the contract for the unified game interface that enables
polymorphic dispatch across all game implementations (Chess, Go, Gomoku).

The interface provides:
- GameFactory: Creates game instances with type detection
- GameRegistry: Manages game type registration
- GameSerializer: Handles state serialization/deserialization
- Unified GameState interface for all games

This contract must be implemented to enable the MCTS algorithm to work
with any game without knowing specific implementation details.
"""

from enum import Enum
from typing import List, Dict, Optional, Union, Any
from abc import ABC, abstractmethod


class GameType(Enum):
    """Enumeration of supported game types."""
    UNKNOWN = 0
    CHESS = 1
    GO = 2
    GOMOKU = 3


class GameResult(Enum):
    """Enumeration of possible game results."""
    ONGOING = 0
    WIN_PLAYER1 = 1
    WIN_PLAYER2 = 2
    DRAW = 3
    NO_RESULT = 4  # For Japanese Go rules: triple ko, etc.


class IGameState(ABC):
    """
    Abstract interface for all game state implementations.

    This interface defines the operations that all game implementations
    must provide for use with the MCTS algorithm.
    """

    @abstractmethod
    def get_legal_moves(self) -> List[int]:
        """
        Get all legal moves in the current state.

        Returns:
            List of legal action integers
        """
        raise NotImplementedError()

    @abstractmethod
    def is_legal_move(self, action: int) -> bool:
        """
        Check if a specific move is legal.

        Args:
            action: The action to check

        Returns:
            True if legal, False otherwise
        """
        raise NotImplementedError()

    @abstractmethod
    def make_move(self, action: int) -> None:
        """
        Execute a move, updating the game state.

        Args:
            action: The action to execute

        Raises:
            ValueError: If the action is illegal
        """
        raise NotImplementedError()

    @abstractmethod
    def undo_move(self) -> bool:
        """
        Undo the last move.

        Returns:
            True if a move was undone, False if no moves to undo
        """
        raise NotImplementedError()

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Check if the game state is terminal.

        Returns:
            True if terminal, False otherwise
        """
        raise NotImplementedError()

    @abstractmethod
    def get_game_result(self) -> GameResult:
        """
        Get the result of the game.

        Returns:
            Game result (should be ONGOING if not terminal)
        """
        raise NotImplementedError()

    @abstractmethod
    def get_current_player(self) -> int:
        """
        Get the current player.

        Returns:
            Current player (1 or 2)
        """
        raise NotImplementedError()

    @abstractmethod
    def get_board_size(self) -> int:
        """
        Get the board size.

        Returns:
            Board size (typically width/height)
        """
        raise NotImplementedError()

    @abstractmethod
    def get_action_space_size(self) -> int:
        """
        Get the action space size.

        Returns:
            Total number of possible actions
        """
        raise NotImplementedError()

    @abstractmethod
    def get_tensor_representation(self) -> List[List[List[float]]]:
        """
        Get tensor representation for neural network.

        Returns:
            3D tensor: [channels][height][width]
        """
        raise NotImplementedError()

    @abstractmethod
    def get_basic_tensor_representation(self) -> List[List[List[float]]]:
        """
        Get basic 18-channel AlphaZero tensor representation.

        Returns:
            3D tensor with 18 channels: [18][height][width]
        """
        raise NotImplementedError()

    @abstractmethod
    def get_enhanced_tensor_representation(self) -> List[List[List[float]]]:
        """
        Get enhanced tensor representation with additional features.

        Returns:
            3D tensor with game-specific enhanced features
        """
        raise NotImplementedError()

    @abstractmethod
    def get_hash(self) -> int:
        """
        Get hash for transposition table.

        Returns:
            64-bit hash of current state
        """
        raise NotImplementedError()

    @abstractmethod
    def clone(self) -> 'IGameState':
        """
        Create a deep copy of the current state.

        Returns:
            New copy of the game state
        """
        raise NotImplementedError()

    @abstractmethod
    def batch_clone(self, count: int) -> List['IGameState']:
        """
        Create multiple deep copies efficiently.

        Args:
            count: Number of clones to create

        Returns:
            List of game state clones
        """
        raise NotImplementedError()

    @abstractmethod
    def copy_from(self, source: 'IGameState') -> None:
        """
        Copy state from another game state instance.

        Args:
            source: The source state to copy from

        Raises:
            ValueError: If game types don't match
        """
        raise NotImplementedError()

    @abstractmethod
    def action_to_string(self, action: int) -> str:
        """
        Convert action to string representation.

        Args:
            action: The action to convert

        Returns:
            String representation (e.g., "e2e4" in chess)
        """
        raise NotImplementedError()

    @abstractmethod
    def string_to_action(self, move_str: str) -> Optional[int]:
        """
        Convert string representation to action.

        Args:
            move_str: String representation

        Returns:
            Action integer, or None if invalid
        """
        raise NotImplementedError()

    @abstractmethod
    def to_string(self) -> str:
        """
        Get string representation of the state.

        Returns:
            Human-readable representation
        """
        raise NotImplementedError()

    @abstractmethod
    def equals(self, other: 'IGameState') -> bool:
        """
        Check equality with another game state.

        Args:
            other: The other game state

        Returns:
            True if equal, False otherwise
        """
        raise NotImplementedError()

    @abstractmethod
    def get_move_history(self) -> List[int]:
        """
        Get the history of moves.

        Returns:
            List of actions that led to current state
        """
        raise NotImplementedError()

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate the game state for consistency.

        Returns:
            True if valid, False otherwise
        """
        raise NotImplementedError()

    @abstractmethod
    def get_bitboards(self) -> List[List[int]]:
        """
        Get bitboard representation.

        Returns:
            List of bitboards for each player
        """
        raise NotImplementedError()

    @abstractmethod
    def get_game_type(self) -> GameType:
        """
        Get the game type.

        Returns:
            Game type enum value
        """
        raise NotImplementedError()


class GameRegistry:
    """
    Singleton registry for game types and their factories.

    Manages registration of game types and provides centralized
    game instance creation without tight coupling.
    """

    _instance = None
    _factories = {}

    @classmethod
    def instance(cls) -> 'GameRegistry':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register_game(self, game_type: GameType, factory_func) -> None:
        """
        Register a game type with its factory function.

        Args:
            game_type: Game type to register
            factory_func: Function that creates new instances
        """
        self._factories[game_type] = factory_func

    def is_registered(self, game_type: GameType) -> bool:
        """
        Check if a game type is registered.

        Args:
            game_type: Game type to check

        Returns:
            True if registered, False otherwise
        """
        return game_type in self._factories

    def get_factory(self, game_type: GameType):
        """
        Get the factory function for a game type.

        Args:
            game_type: Game type

        Returns:
            Factory function

        Raises:
            ValueError: If type is not registered
        """
        if game_type not in self._factories:
            raise ValueError(f"Game type {game_type} is not registered")
        return self._factories[game_type]

    def get_registered_types(self) -> List[GameType]:
        """
        Get all registered game types.

        Returns:
            List of registered game types
        """
        return list(self._factories.keys())

    def clear(self) -> None:
        """Clear all registrations (mainly for testing)."""
        self._factories.clear()


class GameFactory:
    """
    Factory for creating game instances.

    Provides static methods to create game instances using the registry
    or directly with specific parameters.
    """

    @staticmethod
    def create_game(game_type: Union[GameType, str]) -> IGameState:
        """
        Create a game instance of the specified type.

        Args:
            game_type: Game type to create (enum or string)

        Returns:
            Game instance

        Raises:
            ValueError: If type is not registered
        """
        if isinstance(game_type, str):
            game_type = GameType[game_type.upper()]

        factory = GameRegistry.instance().get_factory(game_type)
        return factory()

    @staticmethod
    def create_chess(
        chess960: bool = False,
        fen: str = "",
        position_number: int = -1
    ) -> IGameState:
        """
        Create a chess game with specific options.

        Args:
            chess960: Whether to use Chess960 rules
            fen: Optional FEN string for initial position
            position_number: Chess960 position number (0-959)

        Returns:
            Chess game instance
        """
        raise NotImplementedError("create_chess must be implemented")

    @staticmethod
    def create_go(
        board_size: int = 19,
        rule_set: int = 0,  # 0=Chinese, 1=Japanese, 2=Korean
        custom_komi: float = -1.0
    ) -> IGameState:
        """
        Create a Go game with specific options.

        Args:
            board_size: Board size (9, 13, or 19)
            rule_set: Rule set to use
            custom_komi: Optional custom komi value

        Returns:
            Go game instance
        """
        raise NotImplementedError("create_go must be implemented")

    @staticmethod
    def create_gomoku(
        board_size: int = 15,
        use_renju: bool = False,
        use_omok: bool = False,
        seed: int = 0,
        use_pro_long_opening: bool = False
    ) -> IGameState:
        """
        Create a Gomoku game with specific options.

        Args:
            board_size: Board size (typically 15)
            use_renju: Whether to use Renju rules
            use_omok: Whether to use Omok rules
            seed: Random seed for initialization
            use_pro_long_opening: Whether to use pro-long opening restrictions

        Returns:
            Gomoku game instance
        """
        raise NotImplementedError("create_gomoku must be implemented")

    @staticmethod
    def create_game_from_moves(
        game_type: GameType,
        moves: str
    ) -> IGameState:
        """
        Create a game instance from a sequence of moves.

        Args:
            game_type: Game type
            moves: String containing move sequence

        Returns:
            Game instance with moves applied

        Raises:
            ValueError: If moves are invalid
        """
        raise NotImplementedError("create_game_from_moves must be implemented")

    @staticmethod
    def create_games(game_type: GameType, count: int) -> List[IGameState]:
        """
        Create multiple game instances efficiently.

        Args:
            game_type: Game type
            count: Number of instances to create

        Returns:
            List of game instances
        """
        return [GameFactory.create_game(game_type) for _ in range(count)]

    @staticmethod
    def detect_game_type(input_str: str) -> GameType:
        """
        Detect game type from state or move notation.

        Args:
            input_str: String containing game state or moves

        Returns:
            Detected game type
        """
        raise NotImplementedError("detect_game_type must be implemented")


class GameSerializer:
    """
    Game state serialization and deserialization.

    Handles saving and loading game states to/from various formats.
    """

    @staticmethod
    def serialize_game(game: IGameState) -> str:
        """
        Serialize a game state to string.

        Args:
            game: Game state to serialize

        Returns:
            Serialized string representation
        """
        raise NotImplementedError("serialize_game must be implemented")

    @staticmethod
    def deserialize_game(data: str) -> IGameState:
        """
        Deserialize a game state from string.

        Args:
            data: Serialized string representation

        Returns:
            Deserialized game state

        Raises:
            ValueError: If deserialization fails
        """
        raise NotImplementedError("deserialize_game must be implemented")

    @staticmethod
    def save_game(game: IGameState, filename: str) -> None:
        """
        Save game state to file.

        Args:
            game: Game state to save
            filename: Output filename

        Raises:
            IOError: If file cannot be written
        """
        raise NotImplementedError("save_game must be implemented")

    @staticmethod
    def load_game(filename: str) -> IGameState:
        """
        Load game state from file.

        Args:
            filename: Input filename

        Returns:
            Loaded game state

        Raises:
            IOError: If file cannot be read
            ValueError: If file cannot be parsed
        """
        raise NotImplementedError("load_game must be implemented")

    @staticmethod
    def export_to_standard_format(game: IGameState) -> str:
        """
        Export game to standard format (PGN/SGF/custom).

        Args:
            game: Game state to export

        Returns:
            String in standard format
        """
        raise NotImplementedError("export_to_standard_format must be implemented")


# Utility functions
def game_type_to_string(game_type: GameType) -> str:
    """Convert GameType enum to string."""
    return game_type.name


def string_to_game_type(type_str: str) -> GameType:
    """Convert string to GameType enum."""
    try:
        return GameType[type_str.upper()]
    except KeyError:
        return GameType.UNKNOWN


# Game adapter utilities
def are_states_equivalent(state1: IGameState, state2: IGameState) -> bool:
    """
    Check if two game states are equivalent.

    Args:
        state1: First game state
        state2: Second game state

    Returns:
        True if equivalent, False otherwise
    """
    return (
        state1.get_game_type() == state2.get_game_type() and
        state1.get_hash() == state2.get_hash() and
        state1.equals(state2)
    )


def get_game_statistics(game: IGameState) -> Dict[str, float]:
    """
    Get game statistics.

    Args:
        game: Game state to analyze

    Returns:
        Dictionary of statistic names to values
    """
    return {
        'move_count': len(game.get_move_history()),
        'legal_moves': len(game.get_legal_moves()),
        'board_size': game.get_board_size(),
        'action_space_size': game.get_action_space_size(),
        'current_player': game.get_current_player(),
        'is_terminal': 1.0 if game.is_terminal() else 0.0
    }


def validate_move_sequence(game: IGameState, moves: List[int]) -> bool:
    """
    Validate a sequence of moves.

    Args:
        game: Initial game state (will be cloned)
        moves: List of actions to validate

    Returns:
        True if all moves are legal in sequence, False otherwise
    """
    test_game = game.clone()

    for move in moves:
        if not test_game.is_legal_move(move):
            return False
        try:
            test_game.make_move(move)
        except ValueError:
            return False

    return True


def convert_action_format(
    game: IGameState,
    action: int,
    format_type: str
) -> str:
    """
    Convert between different action representations.

    Args:
        game: Game state for context
        action: Action to convert
        format_type: Target format ("string", "coordinate", "index")

    Returns:
        Converted action representation
    """
    if format_type == "string":
        return game.action_to_string(action)
    elif format_type == "index":
        return str(action)
    elif format_type == "coordinate":
        # Convert to coordinate notation (row, col)
        board_size = game.get_board_size()
        row = action // board_size
        col = action % board_size
        return f"({row},{col})"
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def get_game_complexity(game_type: GameType) -> Dict[str, float]:
    """
    Get game complexity metrics.

    Args:
        game_type: Game type

    Returns:
        Dictionary of complexity metrics
    """
    complexity_data = {
        GameType.CHESS: {
            'branching_factor': 35.0,
            'average_game_length': 40.0,
            'state_space_complexity': 47.0,  # log10
            'game_tree_complexity': 123.0   # log10
        },
        GameType.GO: {
            'branching_factor': 250.0,
            'average_game_length': 150.0,
            'state_space_complexity': 171.0,  # log10
            'game_tree_complexity': 360.0    # log10
        },
        GameType.GOMOKU: {
            'branching_factor': 200.0,
            'average_game_length': 30.0,
            'state_space_complexity': 105.0,  # log10
            'game_tree_complexity': 70.0     # log10
        }
    }

    return complexity_data.get(game_type, {
        'branching_factor': 0.0,
        'average_game_length': 0.0,
        'state_space_complexity': 0.0,
        'game_tree_complexity': 0.0
    })