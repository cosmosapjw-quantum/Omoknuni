"""
Contract tests for MCTS Engine API.

Tests all functions and classes defined in contracts/mcts_api.py to ensure
API signatures match exactly and implementations raise NotImplementedError
until the actual MCTS engine is implemented.

These tests MUST FAIL initially with NotImplementedError - this is the
Test-Driven Development (TDD) approach for the MCTS implementation.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from typing import Tuple
from unittest.mock import Mock

# Add src and spec contract paths to import paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "specs" / "001-goal-create-spec"))

# Import the contract definitions
from contracts.mcts_api import (
    GameState,
    search,
    search_with_info,
    evaluate_position,
    get_best_move,
    MCTSEngine
)


class MockGameState(GameState):
    """Mock implementation of GameState for testing API contracts."""

    def __init__(self, is_terminal=False, terminal_value=0.0):
        self._is_terminal = is_terminal
        self._terminal_value = terminal_value
        self._current_player = 0

    def apply_move_inplace(self, action: int) -> None:
        """Mock implementation."""
        if action < 0:
            raise ValueError("Invalid action")

    def get_legal_moves(self) -> np.ndarray:
        """Mock implementation - returns 15x15 legal moves for Gomoku."""
        return np.ones(225, dtype=bool)  # All moves legal

    def is_terminal(self) -> bool:
        """Mock implementation."""
        return self._is_terminal

    def get_terminal_value(self) -> float:
        """Mock implementation."""
        if not self._is_terminal:
            raise ValueError("Not a terminal state")
        return self._terminal_value

    def extract_features(self) -> np.ndarray:
        """Mock implementation - returns Gomoku features."""
        return np.zeros((7, 15, 15), dtype=np.float32)

    def get_current_player(self) -> int:
        """Mock implementation."""
        return self._current_player

    def copy(self) -> 'GameState':
        """Mock implementation."""
        return MockGameState(self._is_terminal, self._terminal_value)


class TestGameStateContract:
    """Test GameState abstract base class contract."""

    def test_gamestate_is_abstract(self):
        """Test that GameState is an abstract base class."""
        assert hasattr(GameState, '__abstractmethods__')
        assert len(GameState.__abstractmethods__) > 0

    def test_gamestate_abstract_methods_exist(self):
        """Test that all required abstract methods are defined."""
        expected_methods = {
            'apply_move_inplace',
            'get_legal_moves',
            'is_terminal',
            'get_terminal_value',
            'extract_features',
            'get_current_player',
            'copy'
        }

        actual_methods = GameState.__abstractmethods__
        assert expected_methods == actual_methods

    def test_mock_gamestate_implementation(self):
        """Test that mock implementation works correctly."""
        state = MockGameState()

        # Test basic methods
        assert not state.is_terminal()
        assert state.get_current_player() == 0

        legal_moves = state.get_legal_moves()
        assert isinstance(legal_moves, np.ndarray)
        assert legal_moves.dtype == bool
        assert legal_moves.shape == (225,)  # 15x15 Gomoku

        features = state.extract_features()
        assert isinstance(features, np.ndarray)
        assert features.shape == (7, 15, 15)  # Gomoku feature planes

        # Test copy
        copied_state = state.copy()
        assert isinstance(copied_state, GameState)
        assert copied_state is not state

    def test_terminal_state_behavior(self):
        """Test terminal state behavior."""
        terminal_state = MockGameState(is_terminal=True, terminal_value=1.0)

        assert terminal_state.is_terminal()
        assert terminal_state.get_terminal_value() == 1.0

    def test_illegal_move_raises_error(self):
        """Test that illegal moves raise ValueError."""
        state = MockGameState()

        with pytest.raises(ValueError):
            state.apply_move_inplace(-1)  # Invalid action

    def test_terminal_value_on_non_terminal_raises_error(self):
        """Test that getting terminal value on non-terminal state raises error."""
        state = MockGameState(is_terminal=False)

        with pytest.raises(ValueError):
            state.get_terminal_value()


class TestSearchFunctionContract:
    """Test the main search() function contract."""

    def test_search_function_exists(self):
        """Test that search function is importable."""
        assert callable(search)

    def test_search_signature_parameters(self):
        """Test search function signature matches contract exactly."""
        import inspect

        sig = inspect.signature(search)
        params = sig.parameters

        # Check required parameters
        assert 'state' in params
        assert 'num_simulations' in params

        # Check optional parameters with defaults
        assert 'cpuct' in params
        assert params['cpuct'].default == 1.25

        assert 'num_threads' in params
        assert params['num_threads'].default == 8

        assert 'add_dirichlet_noise' in params
        assert params['add_dirichlet_noise'].default is False

        assert 'random_seed' in params
        assert params['random_seed'].default is None

    def test_search_not_implemented_error(self):
        """Test that search raises NotImplementedError (TDD requirement)."""
        state = MockGameState()

        with pytest.raises(NotImplementedError) as exc_info:
            search(state, num_simulations=100)

        assert "MCTS search implementation required" in str(exc_info.value)

    def test_search_parameter_types(self):
        """Test search function accepts correct parameter types."""
        state = MockGameState()

        # All these should raise NotImplementedError, not TypeError
        with pytest.raises(NotImplementedError):
            search(state, 100)  # Basic call

        with pytest.raises(NotImplementedError):
            search(state, 100, cpuct=1.5)  # Float cpuct

        with pytest.raises(NotImplementedError):
            search(state, 100, num_threads=12)  # Int threads

        with pytest.raises(NotImplementedError):
            search(state, 100, add_dirichlet_noise=True)  # Boolean noise

        with pytest.raises(NotImplementedError):
            search(state, 100, random_seed=42)  # Int seed


class TestSearchWithInfoContract:
    """Test the search_with_info() function contract."""

    def test_search_with_info_function_exists(self):
        """Test that search_with_info function is importable."""
        assert callable(search_with_info)

    def test_search_with_info_signature(self):
        """Test search_with_info function signature."""
        import inspect

        sig = inspect.signature(search_with_info)
        params = sig.parameters

        # Check required parameters
        assert 'state' in params
        assert 'num_simulations' in params

        # Check optional parameters
        assert 'cpuct' in params
        assert params['cpuct'].default == 1.25

        assert 'num_threads' in params
        assert params['num_threads'].default == 8

    def test_search_with_info_not_implemented(self):
        """Test that search_with_info raises NotImplementedError."""
        state = MockGameState()

        with pytest.raises(NotImplementedError) as exc_info:
            search_with_info(state, num_simulations=100)

        assert "MCTS search with info implementation required" in str(exc_info.value)

    def test_search_with_info_return_type_annotation(self):
        """Test return type annotation is correct."""
        import inspect

        sig = inspect.signature(search_with_info)
        return_annotation = sig.return_annotation

        # Should return Tuple[np.ndarray, dict]
        assert hasattr(return_annotation, '__origin__')
        assert return_annotation.__origin__ is tuple


class TestEvaluatePositionContract:
    """Test the evaluate_position() function contract."""

    def test_evaluate_position_function_exists(self):
        """Test that evaluate_position function is importable."""
        assert callable(evaluate_position)

    def test_evaluate_position_signature(self):
        """Test evaluate_position function signature."""
        import inspect

        sig = inspect.signature(evaluate_position)
        params = sig.parameters

        # Should have single required parameter
        assert len(params) == 1
        assert 'state' in params
        assert params['state'].default == inspect.Parameter.empty

    def test_evaluate_position_not_implemented(self):
        """Test that evaluate_position raises NotImplementedError."""
        state = MockGameState()

        with pytest.raises(NotImplementedError) as exc_info:
            evaluate_position(state)

        assert "Position evaluation implementation required" in str(exc_info.value)

    def test_evaluate_position_return_type_annotation(self):
        """Test return type annotation is correct."""
        import inspect

        sig = inspect.signature(evaluate_position)
        return_annotation = sig.return_annotation

        # Should return Tuple[np.ndarray, float]
        assert hasattr(return_annotation, '__origin__')
        assert return_annotation.__origin__ is tuple


class TestGetBestMoveContract:
    """Test the get_best_move() function contract."""

    def test_get_best_move_function_exists(self):
        """Test that get_best_move function is importable."""
        assert callable(get_best_move)

    def test_get_best_move_signature(self):
        """Test get_best_move function signature."""
        import inspect

        sig = inspect.signature(get_best_move)
        params = sig.parameters

        # Check required parameters
        assert 'state' in params
        assert 'num_simulations' in params

        # Check optional parameters
        assert 'temperature' in params
        assert params['temperature'].default == 0.0

        # Check **search_kwargs
        assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    def test_get_best_move_not_implemented(self):
        """Test that get_best_move raises NotImplementedError."""
        state = MockGameState()

        with pytest.raises(NotImplementedError) as exc_info:
            get_best_move(state, num_simulations=100)

        assert "Best move selection implementation required" in str(exc_info.value)

    def test_get_best_move_return_type(self):
        """Test get_best_move return type annotation."""
        import inspect

        sig = inspect.signature(get_best_move)
        return_annotation = sig.return_annotation

        # Should return int
        assert return_annotation is int


class TestMCTSEngineContract:
    """Test the MCTSEngine class contract."""

    def test_mcts_engine_class_exists(self):
        """Test that MCTSEngine class is importable."""
        assert MCTSEngine is not None
        assert isinstance(MCTSEngine, type)

    def test_mcts_engine_init_signature(self):
        """Test MCTSEngine.__init__ signature."""
        import inspect

        sig = inspect.signature(MCTSEngine.__init__)
        params = sig.parameters

        # Check required parameters (excluding self)
        assert 'game_type' in params
        assert 'model_path' in params

        # Check optional parameters with defaults
        assert 'num_threads' in params
        assert params['num_threads'].default == 8

        assert 'max_tree_nodes' in params
        assert params['max_tree_nodes'].default == 50_000_000

    def test_mcts_engine_initialization_basic(self):
        """Test basic MCTSEngine initialization."""
        engine = MCTSEngine(
            game_type="gomoku",
            model_path="/path/to/model.pth"
        )

        assert engine.game_type == "gomoku"
        assert engine.model_path == "/path/to/model.pth"
        assert engine.num_threads == 8  # Default
        assert engine.max_tree_nodes == 50_000_000  # Default

    def test_mcts_engine_initialization_with_params(self):
        """Test MCTSEngine initialization with custom parameters."""
        engine = MCTSEngine(
            game_type="chess",
            model_path="/custom/model.pth",
            num_threads=12,
            max_tree_nodes=100_000_000
        )

        assert engine.game_type == "chess"
        assert engine.model_path == "/custom/model.pth"
        assert engine.num_threads == 12
        assert engine.max_tree_nodes == 100_000_000

    def test_mcts_engine_search_method(self):
        """Test MCTSEngine.search method exists and raises NotImplementedError."""
        engine = MCTSEngine("gomoku", "/path/to/model.pth")
        state = MockGameState()

        assert hasattr(engine, 'search')
        assert callable(engine.search)

        with pytest.raises(NotImplementedError) as exc_info:
            engine.search(state, num_simulations=100)

        assert "Engine search implementation required" in str(exc_info.value)

    def test_mcts_engine_reset_tree_method(self):
        """Test MCTSEngine.reset_tree method exists and raises NotImplementedError."""
        engine = MCTSEngine("gomoku", "/path/to/model.pth")

        assert hasattr(engine, 'reset_tree')
        assert callable(engine.reset_tree)

        with pytest.raises(NotImplementedError) as exc_info:
            engine.reset_tree()

        assert "Tree reset implementation required" in str(exc_info.value)

    def test_mcts_engine_get_tree_stats_method(self):
        """Test MCTSEngine.get_tree_stats method exists and raises NotImplementedError."""
        engine = MCTSEngine("gomoku", "/path/to/model.pth")

        assert hasattr(engine, 'get_tree_stats')
        assert callable(engine.get_tree_stats)

        with pytest.raises(NotImplementedError) as exc_info:
            engine.get_tree_stats()

        assert "Tree stats implementation required" in str(exc_info.value)

    def test_mcts_engine_method_signatures(self):
        """Test MCTSEngine method signatures match contract."""
        import inspect

        # Test search method signature
        search_sig = inspect.signature(MCTSEngine.search)
        search_params = search_sig.parameters

        assert 'self' in search_params
        assert 'state' in search_params
        assert 'num_simulations' in search_params

        # Should accept **kwargs
        assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in search_params.values())

        # Test return annotation
        assert search_sig.return_annotation is np.ndarray

        # Test reset_tree signature
        reset_sig = inspect.signature(MCTSEngine.reset_tree)
        reset_params = reset_sig.parameters

        assert len(reset_params) == 1  # Only self
        assert 'self' in reset_params

        # Test get_tree_stats signature
        stats_sig = inspect.signature(MCTSEngine.get_tree_stats)
        stats_params = stats_sig.parameters

        assert len(stats_params) == 1  # Only self
        assert 'self' in stats_params
        assert stats_sig.return_annotation is dict


class TestAPIIntegration:
    """Integration tests for the complete MCTS API contract."""

    def test_all_api_functions_raise_not_implemented(self):
        """Test that all API functions raise NotImplementedError as required for TDD."""
        state = MockGameState()

        # Test all standalone functions
        with pytest.raises(NotImplementedError):
            search(state, 100)

        with pytest.raises(NotImplementedError):
            search_with_info(state, 100)

        with pytest.raises(NotImplementedError):
            evaluate_position(state)

        with pytest.raises(NotImplementedError):
            get_best_move(state, 100)

    def test_mcts_engine_all_methods_raise_not_implemented(self):
        """Test that all MCTSEngine methods raise NotImplementedError."""
        engine = MCTSEngine("gomoku", "/path/to/model.pth")
        state = MockGameState()

        with pytest.raises(NotImplementedError):
            engine.search(state, 100)

        with pytest.raises(NotImplementedError):
            engine.reset_tree()

        with pytest.raises(NotImplementedError):
            engine.get_tree_stats()

    def test_api_coverage_completeness(self):
        """Test that all functions and classes from contract are tested."""
        # This test ensures we have 100% coverage of the contract API

        # All functions should be tested
        functions_to_test = [
            search,
            search_with_info,
            evaluate_position,
            get_best_move
        ]

        for func in functions_to_test:
            assert callable(func)
            # Each should raise NotImplementedError
            state = MockGameState()
            with pytest.raises(NotImplementedError):
                if func == search or func == search_with_info or func == get_best_move:
                    func(state, 100)
                else:  # evaluate_position
                    func(state)

        # Classes should be tested
        assert MCTSEngine is not None
        assert GameState is not None

    def test_contract_consistency(self):
        """Test that contract definitions are internally consistent."""
        # All functions that take a state parameter should accept GameState
        state = MockGameState()

        # Should not raise TypeError about state parameter
        try:
            search(state, 100)
        except NotImplementedError:
            pass  # Expected
        except TypeError as e:
            if "state" in str(e):
                pytest.fail(f"search() rejected valid GameState: {e}")

        try:
            evaluate_position(state)
        except NotImplementedError:
            pass  # Expected
        except TypeError as e:
            if "state" in str(e):
                pytest.fail(f"evaluate_position() rejected valid GameState: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])