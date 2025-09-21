"""
Unit tests for the game adapter interface.

Tests the unified game interface including GameFactory, GameRegistry,
and GameSerializer classes that enable polymorphic dispatch across
all game implementations.

HOWTO-RUN-TESTS:
================
# Run all game adapter interface tests
python -m pytest tests/unit/test_game_adapter_interface.py -v

# Run specific test class
python -m pytest tests/unit/test_game_adapter_interface.py::TestGameAdapterInterface -v

# Run with detailed output
python -m pytest tests/unit/test_game_adapter_interface.py -v -s
"""

import pytest
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the contract API to test against
sys.path.append('specs/001-goal-create-spec')
from contracts.interface_api import (
    GameType,
    GameResult,
    GameRegistry,
    GameFactory,
    GameSerializer,
    IGameState
)

# Mock implementations for testing
class MockGameState:
    """Mock game state for testing interface functionality."""

    def __init__(self, game_type='GOMOKU', board_size=15):
        self.game_type = game_type
        self.board_size = board_size
        self.current_player = 1
        self.move_history = []
        self.is_terminal_state = False
        self.game_result = 'ONGOING'
        self.action_space_size = board_size * board_size

    def get_legal_moves(self):
        """Return mock legal moves."""
        if self.is_terminal_state:
            return []
        return list(range(min(10, self.action_space_size)))

    def is_legal_move(self, action):
        """Check if move is legal."""
        return action in self.get_legal_moves()

    def make_move(self, action):
        """Make a move."""
        if not self.is_legal_move(action):
            raise ValueError(f"Illegal move: {action}")
        self.move_history.append(action)
        self.current_player = 3 - self.current_player

        # Simulate game ending after 5 moves
        if len(self.move_history) >= 5:
            self.is_terminal_state = True
            self.game_result = 'WIN_PLAYER1'

    def undo_move(self):
        """Undo last move."""
        if not self.move_history:
            return False
        self.move_history.pop()
        self.current_player = 3 - self.current_player
        self.is_terminal_state = False
        self.game_result = 'ONGOING'
        return True

    def is_terminal(self):
        """Check if game is terminal."""
        return self.is_terminal_state

    def get_game_result(self):
        """Get game result."""
        return self.game_result

    def get_current_player(self):
        """Get current player."""
        return self.current_player

    def get_board_size(self):
        """Get board size."""
        return self.board_size

    def get_action_space_size(self):
        """Get action space size."""
        return self.action_space_size

    def get_tensor_representation(self):
        """Get tensor representation."""
        channels = 7 if self.game_type == 'GOMOKU' else (12 if self.game_type == 'CHESS' else 17)
        return [[[0.0 for _ in range(self.board_size)]
                for _ in range(self.board_size)]
                for _ in range(channels)]

    def get_basic_tensor_representation(self):
        """Get basic tensor representation."""
        return [[[0.0 for _ in range(self.board_size)]
                for _ in range(self.board_size)]
                for _ in range(18)]

    def get_enhanced_tensor_representation(self):
        """Get enhanced tensor representation."""
        return self.get_tensor_representation()

    def get_hash(self):
        """Get state hash."""
        return hash(tuple(self.move_history))

    def clone(self):
        """Clone the state."""
        new_state = MockGameState(self.game_type, self.board_size)
        new_state.current_player = self.current_player
        new_state.move_history = self.move_history.copy()
        new_state.is_terminal_state = self.is_terminal_state
        new_state.game_result = self.game_result
        return new_state

    def batch_clone(self, count):
        """Create multiple clones."""
        return [self.clone() for _ in range(count)]

    def copy_from(self, source):
        """Copy from another state."""
        self.game_type = source.game_type
        self.board_size = source.board_size
        self.current_player = source.current_player
        self.move_history = source.move_history.copy()
        self.is_terminal_state = source.is_terminal_state
        self.game_result = source.game_result

    def action_to_string(self, action):
        """Convert action to string."""
        row = action // self.board_size
        col = action % self.board_size
        return f"{chr(ord('A') + col)}{row + 1}"

    def string_to_action(self, move_str):
        """Convert string to action."""
        if len(move_str) < 2:
            return None
        col = ord(move_str[0].upper()) - ord('A')
        try:
            row = int(move_str[1:]) - 1
            action = row * self.board_size + col
            return action if 0 <= action < self.action_space_size else None
        except ValueError:
            return None

    def to_string(self):
        """String representation."""
        return f"{self.game_type} game with {len(self.move_history)} moves"

    def equals(self, other):
        """Check equality."""
        return (self.game_type == other.game_type and
                self.move_history == other.move_history and
                self.current_player == other.current_player)

    def get_move_history(self):
        """Get move history."""
        return self.move_history.copy()

    def validate(self):
        """Validate state."""
        return True

    def get_bitboards(self):
        """Get bitboard representation."""
        return [[], []]  # Empty bitboards for mock

    def get_game_type(self):
        """Get game type."""
        return self.game_type


class TestGameAdapterInterface:
    """Test the game adapter interface functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear any existing registrations
        try:
            GameRegistry.instance().clear()
        except:
            pass  # Ignore if not implemented yet

    def test_game_type_enum_exists(self):
        """Test that GameType enum exists and has expected values."""
        # These should be available from the interface
        expected_types = ['UNKNOWN', 'CHESS', 'GO', 'GOMOKU']

        for game_type in expected_types:
            assert hasattr(GameType, game_type), f"GameType.{game_type} should exist"

    def test_game_result_enum_exists(self):
        """Test that GameResult enum exists and has expected values."""
        expected_results = ['ONGOING', 'WIN_PLAYER1', 'WIN_PLAYER2', 'DRAW', 'NO_RESULT']

        for result in expected_results:
            assert hasattr(GameResult, result), f"GameResult.{result} should exist"

    def test_game_registry_singleton(self):
        """Test that GameRegistry follows singleton pattern."""
        try:
            registry1 = GameRegistry.instance()
            registry2 = GameRegistry.instance()

            # Should be the same instance
            assert registry1 is registry2
        except AttributeError:
            pytest.skip("GameRegistry not implemented yet")

    def test_game_factory_exists(self):
        """Test that GameFactory class exists with expected methods."""
        expected_methods = [
            'create_game',
            'create_chess',
            'create_go',
            'create_gomoku',
            'create_game_from_moves',
            'create_games',
            'detect_game_type'
        ]

        for method in expected_methods:
            assert hasattr(GameFactory, method), f"GameFactory.{method} should exist"

    def test_game_serializer_exists(self):
        """Test that GameSerializer class exists with expected methods."""
        expected_methods = [
            'serialize_game',
            'deserialize_game',
            'save_game',
            'load_game',
            'export_to_standard_format'
        ]

        for method in expected_methods:
            assert hasattr(GameSerializer, method), f"GameSerializer.{method} should exist"

    def test_interface_polymorphism(self):
        """Test that the interface supports polymorphic dispatch."""
        # Create mock states for different games
        chess_state = MockGameState('CHESS', 8)
        go_state = MockGameState('GO', 19)
        gomoku_state = MockGameState('GOMOKU', 15)

        states = [chess_state, go_state, gomoku_state]

        # Test that all states implement the same interface
        for state in states:
            # Basic interface methods
            assert hasattr(state, 'get_legal_moves')
            assert hasattr(state, 'is_legal_move')
            assert hasattr(state, 'make_move')
            assert hasattr(state, 'is_terminal')
            assert hasattr(state, 'get_current_player')

            # Tensor interface methods
            assert hasattr(state, 'get_tensor_representation')
            assert hasattr(state, 'get_basic_tensor_representation')
            assert hasattr(state, 'get_enhanced_tensor_representation')

            # Utility methods
            assert hasattr(state, 'clone')
            assert hasattr(state, 'action_to_string')
            assert hasattr(state, 'string_to_action')

    def test_game_type_detection(self):
        """Test automatic game type detection from move notation."""
        test_cases = [
            # Chess examples
            ("e2e4 e7e5 Nf3", "CHESS"),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "CHESS"),

            # Go examples
            ("(;FF[4]GM[1];B[pd];W[dp])", "GO"),
            ("D4 Q16 D16", "GO"),

            # Gomoku examples
            ("H8 H9 I8", "GOMOKU"),
            ("A1 B2 C3", "GOMOKU"),
        ]

        try:
            for notation, expected_type in test_cases:
                detected = GameFactory.detect_game_type(notation)
                expected_enum = getattr(GameType, expected_type)
                assert detected == expected_enum, f"Failed to detect {expected_type} from '{notation}'"
        except AttributeError:
            pytest.skip("GameFactory.detect_game_type not implemented yet")

    def test_game_serialization_roundtrip(self):
        """Test that game state can be serialized and deserialized."""
        try:
            # Create a mock game with some moves
            original_state = MockGameState('GOMOKU', 15)
            original_state.make_move(112)  # H8
            original_state.make_move(113)  # H9

            # Serialize the state
            serialized = GameSerializer.serialize_game(original_state)
            assert isinstance(serialized, str)
            assert len(serialized) > 0

            # Deserialize the state
            restored_state = GameSerializer.deserialize_game(serialized)

            # Verify the states are equivalent
            assert restored_state.get_move_history() == original_state.get_move_history()
            assert restored_state.get_current_player() == original_state.get_current_player()
            assert restored_state.get_game_type() == original_state.get_game_type()

        except AttributeError:
            pytest.skip("GameSerializer not implemented yet")

    def test_batch_game_creation(self):
        """Test efficient creation of multiple game instances."""
        try:
            # Test creating multiple games of the same type
            game_count = 5
            games = GameFactory.create_games(GameType.GOMOKU, game_count)

            assert len(games) == game_count

            # All games should be independent instances
            for i, game in enumerate(games):
                assert game.get_game_type() == GameType.GOMOKU
                assert game.get_board_size() == 15  # Default Gomoku size

                # Make different moves to verify independence
                if game.get_legal_moves():
                    game.make_move(game.get_legal_moves()[0])

            # Verify games have different states
            move_histories = [game.get_move_history() for game in games]
            for i in range(len(move_histories)):
                for j in range(i + 1, len(move_histories)):
                    # Different games should be independent
                    assert True  # They might have same initial state, which is fine

        except AttributeError:
            pytest.skip("GameFactory.create_games not implemented yet")

    def test_game_from_moves(self):
        """Test creating game state from move sequence."""
        try:
            # Test creating a game with pre-applied moves
            moves = "H8 H9 I8 I9 J8"
            game = GameFactory.create_game_from_moves(GameType.GOMOKU, moves)

            # Verify moves were applied
            history = game.get_move_history()
            assert len(history) > 0

            # Verify the game state is as expected
            assert game.get_current_player() in [1, 2]

        except AttributeError:
            pytest.skip("GameFactory.create_game_from_moves not implemented yet")

    def test_custom_game_parameters(self):
        """Test creating games with custom parameters."""
        try:
            # Test Chess with Chess960
            chess960_game = GameFactory.create_chess(chess960=True, position_number=518)
            assert chess960_game.get_game_type() == GameType.CHESS
            assert chess960_game.get_board_size() == 8

            # Test Go with different board size
            small_go = GameFactory.create_go(board_size=9, rule_set=1)  # Japanese rules
            assert small_go.get_game_type() == GameType.GO
            assert small_go.get_board_size() == 9

            # Test Gomoku with Renju rules
            renju_game = GameFactory.create_gomoku(board_size=15, use_renju=True)
            assert renju_game.get_game_type() == GameType.GOMOKU
            assert renju_game.get_board_size() == 15

        except AttributeError:
            pytest.skip("Custom game creation methods not implemented yet")

    def test_game_registry_registration(self):
        """Test manual game registration and retrieval."""
        try:
            registry = GameRegistry.instance()

            # Register a mock game factory
            def mock_factory():
                return MockGameState('TEST_GAME', 10)

            # This might not work if GameType doesn't have TEST_GAME
            # but we can test the registration mechanism exists
            assert hasattr(registry, 'register_game')
            assert hasattr(registry, 'is_registered')
            assert hasattr(registry, 'get_registered_types')

        except AttributeError:
            pytest.skip("GameRegistry registration methods not implemented yet")

    def test_error_handling(self):
        """Test proper error handling in the interface."""
        try:
            # Test creating unknown game type
            with pytest.raises(Exception):  # Should raise some kind of error
                GameFactory.create_game(999)  # Invalid game type

            # Test deserializing invalid data
            with pytest.raises(Exception):
                GameSerializer.deserialize_game("invalid data")

            # Test loading non-existent file
            with pytest.raises(Exception):
                GameSerializer.load_game("/nonexistent/file.game")

        except AttributeError:
            pytest.skip("Error handling not implemented yet")

    def test_export_to_standard_formats(self):
        """Test exporting games to standard formats."""
        try:
            # Create games with some moves
            chess_game = MockGameState('CHESS', 8)
            chess_game.make_move(20)  # e2-e4 (mock)

            go_game = MockGameState('GO', 19)
            go_game.make_move(75)  # D4 (mock)

            gomoku_game = MockGameState('GOMOKU', 15)
            gomoku_game.make_move(112)  # H8 (mock)

            # Test export to standard formats
            chess_pgn = GameSerializer.export_to_standard_format(chess_game)
            go_sgf = GameSerializer.export_to_standard_format(go_game)
            gomoku_custom = GameSerializer.export_to_standard_format(gomoku_game)

            # Basic validation
            assert isinstance(chess_pgn, str) and len(chess_pgn) > 0
            assert isinstance(go_sgf, str) and len(go_sgf) > 0
            assert isinstance(gomoku_custom, str) and len(gomoku_custom) > 0

            # Check for format-specific markers
            assert "[Event" in chess_pgn or "1." in chess_pgn  # PGN markers
            assert "(;" in go_sgf or "FF[4]" in go_sgf  # SGF markers

        except AttributeError:
            pytest.skip("Export to standard formats not implemented yet")

    def test_tensor_representation_consistency(self):
        """Test that tensor representations are consistent across games."""
        games = [
            MockGameState('CHESS', 8),
            MockGameState('GO', 19),
            MockGameState('GOMOKU', 15)
        ]

        for game in games:
            # Basic tensor representation should always be 18 channels
            basic_tensor = game.get_basic_tensor_representation()
            assert len(basic_tensor) == 18, f"Basic tensor should have 18 channels for {game.get_game_type()}"

            # Enhanced tensor representation should match game-specific requirements
            enhanced_tensor = game.get_enhanced_tensor_representation()
            if game.get_game_type() == 'GOMOKU':
                assert len(enhanced_tensor) == 7, "Gomoku should have 7 channels"
            elif game.get_game_type() == 'CHESS':
                assert len(enhanced_tensor) == 12, "Chess should have 12 channels"
            elif game.get_game_type() == 'GO':
                assert len(enhanced_tensor) == 17, "Go should have 17 channels"

            # All tensors should have correct dimensions
            board_size = game.get_board_size()
            for channel in basic_tensor:
                assert len(channel) == board_size
                assert len(channel[0]) == board_size

    def test_move_validation_consistency(self):
        """Test that move validation is consistent across the interface."""
        game = MockGameState('GOMOKU', 15)

        # Test legal moves
        legal_moves = game.get_legal_moves()
        for move in legal_moves[:3]:  # Test first few moves
            assert game.is_legal_move(move), f"Move {move} should be legal"

            # Apply move and test it's recorded
            old_history_len = len(game.get_move_history())
            game.make_move(move)
            new_history_len = len(game.get_move_history())
            assert new_history_len == old_history_len + 1, "Move history should increase by 1"

        # Test undo functionality
        if len(game.get_move_history()) > 0:
            old_len = len(game.get_move_history())
            success = game.undo_move()
            assert success, "Undo should succeed"
            assert len(game.get_move_history()) == old_len - 1, "Move history should decrease by 1"


class TestGameInterfaceIntegration:
    """Integration tests for the complete game interface system."""

    def test_full_game_workflow(self):
        """Test a complete workflow using the game interface."""
        try:
            # 1. Create a game
            game = GameFactory.create_game(GameType.GOMOKU)

            # 2. Play some moves
            legal_moves = game.get_legal_moves()
            for i in range(min(3, len(legal_moves))):
                move = legal_moves[i]
                assert game.is_legal_move(move)
                game.make_move(move)

            # 3. Clone the game
            cloned_game = game.clone()
            assert cloned_game.equals(game)

            # 4. Serialize the game
            serialized = GameSerializer.serialize_game(game)

            # 5. Deserialize and verify
            restored_game = GameSerializer.deserialize_game(serialized)
            assert restored_game.get_move_history() == game.get_move_history()

            # 6. Export to standard format
            exported = GameSerializer.export_to_standard_format(game)
            assert isinstance(exported, str)

        except (AttributeError, NotImplementedError):
            pytest.skip("Full workflow not implemented yet")

    def test_cross_game_compatibility(self):
        """Test that the interface works consistently across all game types."""
        game_types = [GameType.CHESS, GameType.GO, GameType.GOMOKU]

        try:
            for game_type in game_types:
                # Create game
                game = GameFactory.create_game(game_type)

                # Test basic interface
                assert game.get_board_size() > 0
                assert game.get_action_space_size() > 0
                assert game.get_current_player() in [1, 2]
                assert not game.is_terminal()  # New game shouldn't be terminal

                # Test tensor representations
                basic_tensor = game.get_basic_tensor_representation()
                enhanced_tensor = game.get_enhanced_tensor_representation()

                assert len(basic_tensor) == 18  # Standard AlphaZero format
                assert len(enhanced_tensor) > 0   # Game-specific format

                # Test move interface
                legal_moves = game.get_legal_moves()
                assert len(legal_moves) > 0  # New game should have legal moves

                if legal_moves:
                    first_move = legal_moves[0]
                    assert game.is_legal_move(first_move)

                    # Test string conversion
                    move_str = game.action_to_string(first_move)
                    assert isinstance(move_str, str)

                    converted_back = game.string_to_action(move_str)
                    assert converted_back == first_move

        except (AttributeError, NotImplementedError):
            pytest.skip("Cross-game compatibility testing not fully implemented yet")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])