"""
Unit tests for model evaluation system.

Tests head-to-head model comparison, ELO rating calculation, and performance
measurement functionality.
"""

import pytest
import tempfile
import shutil
import json
import math
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

import numpy as np

# Import the module under test
from src.training.evaluator import (
    ModelEvaluator, EvaluationConfig, EvaluationResult,
    ELORatingSystem, StatisticalAnalyzer, RandomMoveGenerator,
    evaluate_model_strength, create_evaluator, save_evaluation_results
)

# Import contracts for test data
import sys
sys.path.append('specs/001-goal-create-spec')
from contracts.training_api import GameResult, TrainingExample


class TestEvaluationConfig:
    """Test evaluation configuration."""

    def test_default_config_creation(self):
        """Test creating default evaluation configuration."""
        config = EvaluationConfig()

        assert config.game_type == "gomoku"
        assert config.num_games == 100
        assert config.mcts_simulations == 800
        assert config.time_per_move == 1.0
        assert config.temperature == 0.1  # Low for evaluation
        assert config.add_dirichlet_noise == False  # No noise for evaluation

    def test_custom_config_creation(self):
        """Test creating custom evaluation configuration."""
        config = EvaluationConfig(
            game_type="chess",
            num_games=50,
            mcts_simulations=400,
            confidence_level=0.99
        )

        assert config.game_type == "chess"
        assert config.num_games == 50
        assert config.mcts_simulations == 400
        assert config.confidence_level == 0.99


class TestEvaluationResult:
    """Test evaluation result data structure."""

    def test_default_result_creation(self):
        """Test creating default evaluation result."""
        result = EvaluationResult(
            old_model_path="old.pth",
            new_model_path="new.pth",
            game_type="gomoku"
        )

        assert result.old_model_path == "old.pth"
        assert result.new_model_path == "new.pth"
        assert result.game_type == "gomoku"
        assert result.total_games == 0
        assert result.win_rate == 0.0
        assert result.elo_difference == 0.0

    def test_result_update(self):
        """Test updating evaluation result."""
        result = EvaluationResult(
            old_model_path="old.pth",
            new_model_path="new.pth",
            game_type="gomoku"
        )

        result.total_games = 100
        result.new_model_wins = 65
        result.old_model_wins = 30
        result.draws = 5
        result.win_rate = 0.65

        assert result.total_games == 100
        assert result.new_model_wins == 65
        assert result.win_rate == 0.65


class TestELORatingSystem:
    """Test ELO rating system."""

    def test_initial_rating(self):
        """Test initial rating assignment."""
        elo = ELORatingSystem()

        rating = elo.get_rating("new_model.pth")
        assert rating == 1500.0  # Default rating

    def test_expected_score_calculation(self):
        """Test expected score calculation."""
        elo = ELORatingSystem()

        # Equal ratings should give 0.5 expected score
        expected = elo._expected_score(1500.0, 1500.0)
        assert abs(expected - 0.5) < 1e-6

        # Higher rating should give higher expected score
        expected_higher = elo._expected_score(1600.0, 1500.0)
        assert expected_higher > 0.5

        # Lower rating should give lower expected score
        expected_lower = elo._expected_score(1400.0, 1500.0)
        assert expected_lower < 0.5

    def test_rating_update_win(self):
        """Test rating update after a win."""
        elo = ELORatingSystem()

        # Model A wins against equal opponent
        elo.update_ratings("model_a.pth", "model_b.pth", 1.0)

        new_a = elo.get_rating("model_a.pth")
        new_b = elo.get_rating("model_b.pth")

        # Winner should gain rating, loser should lose rating
        assert new_a > 1500.0
        assert new_b < 1500.0

    def test_rating_update_draw(self):
        """Test rating update after a draw."""
        elo = ELORatingSystem()

        # Draw between equal opponents
        elo.update_ratings("model_a.pth", "model_b.pth", 0.5)

        new_a = elo.get_rating("model_a.pth")
        new_b = elo.get_rating("model_b.pth")

        # Ratings should change very little for equal opponents
        assert abs(new_a - 1500.0) < 50  # Some change due to Glicko-2
        assert abs(new_b - 1500.0) < 50

    def test_rating_difference(self):
        """Test rating difference calculation."""
        elo = ELORatingSystem()

        # Create some ratings through games
        for _ in range(5):
            elo.update_game("strong.pth", "weak.pth", 1.0)

        diff = elo.get_rating_difference("strong.pth", "weak.pth")
        assert diff > 0  # Strong should be higher

    def test_multiple_games_rating_evolution(self):
        """Test rating evolution over multiple games."""
        elo = ELORatingSystem()

        # Simulate strong model beating weak model repeatedly
        for _ in range(10):
            elo.update_ratings("strong.pth", "weak.pth", 1.0)

        strong_rating = elo.get_rating("strong.pth")
        weak_rating = elo.get_rating("weak.pth")

        assert strong_rating > 1500.0
        assert weak_rating < 1500.0
        assert strong_rating - weak_rating > 50.0  # Significant difference


class TestRandomMoveGenerator:
    """Test random move generator for baselines."""

    def test_generator_initialization(self):
        """Test random move generator initialization."""
        generator = RandomMoveGenerator("gomoku")
        assert generator.game_type == "gomoku"

    def test_generate_game(self):
        """Test game generation with random moves."""
        generator = RandomMoveGenerator("gomoku")
        result = generator.generate_game("test_game")

        assert isinstance(result, GameResult)
        assert result.winner in [0, 1, None]  # Valid outcomes
        assert result.move_count > 0
        assert result.game_length_seconds > 0
        assert isinstance(result.examples, list)

    def test_multiple_games_different_outcomes(self):
        """Test that random generator produces varied outcomes."""
        generator = RandomMoveGenerator("gomoku")
        results = [generator.generate_game(f"game_{i}") for i in range(10)]

        # Should have some variation in outcomes
        winners = [r.winner for r in results]
        assert len(set(winners)) > 1  # Not all the same

    def test_different_game_types(self):
        """Test random generator with different game types."""
        for game_type in ["gomoku", "chess", "go"]:
            generator = RandomMoveGenerator(game_type)
            result = generator.generate_game(f"test_{game_type}")
            assert isinstance(result, GameResult)


class TestGlicko2AdvancedFeatures:
    """Test advanced Glicko-2 features."""

    def test_baseline_anchoring(self):
        """Test baseline anchoring system."""
        elo = ELORatingSystem()

        # Verify baselines exist and have been initialized
        assert 'random' in elo.players
        assert 'uniform' in elo.players
        # Note: random_anchor_target_elo = 0.0, so random should be at 0 ELO, not 1500
        assert elo.get_rating('random') == 0.0
        assert elo.get_rating('uniform') == 0.0  # Also recentered

    def test_rating_deviation_tracking(self):
        """Test that rating deviation (uncertainty) is tracked."""
        elo = ELORatingSystem()

        # Trigger player creation and capture initial RD value
        initial_record = elo.get_rating_record("test_model.pth")
        initial_rd = initial_record.rd_elo  # Capture the value, not the reference
        assert initial_rd > 300  # High uncertainty for new player

        # Create a similar-strength opponent for better RD reduction
        similar_opponent = elo.get_rating_record("similar_model.pth")

        # After games against similar opponent, RD should decrease
        for _ in range(10):  # More games needed for noticeable RD reduction
            elo.update_ratings("test_model.pth", "similar_model.pth", 0.5)  # Draws

        final_record = elo.get_rating_record("test_model.pth")
        assert final_record.rd_elo < initial_rd  # Uncertainty reduced

    def test_volatility_tracking(self):
        """Test that volatility is tracked and updated."""
        elo = ELORatingSystem()

        # Trigger player creation and check initial volatility
        initial_record = elo.get_rating_record("test_model.pth")
        initial_sigma = initial_record.sigma
        assert initial_sigma > 0  # Should have some volatility

        # Volatility should be updated after games
        elo.update_ratings("test_model.pth", "random", 1.0)
        updated_record = elo.get_rating_record("test_model.pth")
        assert updated_record.sigma >= 0  # Should still be non-negative
        # Volatility might change slightly, but should remain reasonable
        assert abs(updated_record.sigma - initial_sigma) < 0.1

    def test_recentering_mechanism(self):
        """Test anchored recentering to maintain scale."""
        elo = ELORatingSystem()

        # Create some artificial rating drift
        rec1 = elo.get_rating_record("test1.pth")  # Create players
        rec2 = elo.get_rating_record("test2.pth")
        rec1.mu_elo = 2000.0  # High rating
        rec2.mu_elo = 1000.0  # Low rating

        # Recenter
        elo.recenter_to_random_anchor()

        # Random baseline should be close to 0 (target anchor ELO)
        random_rating = elo.get_rating('random')
        assert abs(random_rating - 0.0) < 10  # Should be near baseline


class TestStatisticalAnalyzer:
    """Test statistical analysis functions."""

    def test_wilson_confidence_interval(self):
        """Test Wilson confidence interval calculation."""
        # Perfect win rate
        lower, upper = StatisticalAnalyzer.wilson_confidence_interval(100, 100, 0.95)
        assert lower > 0.9  # Should be high but not exactly 1.0
        assert upper <= 1.0

        # 50% win rate
        lower, upper = StatisticalAnalyzer.wilson_confidence_interval(50, 100, 0.95)
        assert lower < 0.5 < upper  # Should contain 0.5

        # Zero games
        lower, upper = StatisticalAnalyzer.wilson_confidence_interval(0, 0, 0.95)
        assert lower == 0.0
        assert upper == 0.0

    def test_binomial_test(self):
        """Test binomial significance test."""
        # Perfect win rate with many games - should be significant
        p_value = StatisticalAnalyzer.binomial_test(90, 100, 0.5)
        assert p_value < 0.01  # Highly significant

        # Exactly 50% win rate - should not be significant
        p_value = StatisticalAnalyzer.binomial_test(50, 100, 0.5)
        assert p_value > 0.05  # Not significant

        # Small sample - should return conservative p-value
        p_value = StatisticalAnalyzer.binomial_test(5, 10, 0.5)
        assert p_value >= 0.05  # Conservative for small samples

    def test_normal_cdf_approximation(self):
        """Test normal CDF approximation."""
        # Test some known values
        assert abs(StatisticalAnalyzer._normal_cdf(0.0) - 0.5) < 0.01
        assert StatisticalAnalyzer._normal_cdf(-3.0) < 0.01
        assert StatisticalAnalyzer._normal_cdf(3.0) > 0.99

    def test_z_score_lookup(self):
        """Test Z-score lookup for confidence levels."""
        # Test known confidence levels
        assert abs(StatisticalAnalyzer._get_z_score(0.95) - 1.96) < 0.01
        assert abs(StatisticalAnalyzer._get_z_score(0.99) - 2.576) < 0.01

        # Test default fallback
        assert StatisticalAnalyzer._get_z_score(0.97) == 1.96  # Should default to 95%


class TestModelEvaluator:
    """Test model evaluator."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_config(self):
        """Create mock evaluation configuration."""
        return EvaluationConfig(
            game_type="gomoku",
            num_games=10,  # Small for testing
            mcts_simulations=100,
            parallel_games=2,
            min_games_for_significance=5
        )

    @pytest.fixture
    def mock_game_results(self):
        """Create mock game results."""
        results = []
        for i in range(10):
            # New model wins 7/10 games
            outcome = "new_win" if i < 7 else "old_win"
            score = 1.0 if outcome == "new_win" else 0.0

            results.append({
                'game_id': f'test_game_{i}',
                'game_idx': i,
                'outcome': outcome,
                'new_model_score': score,
                'winner': 0 if outcome == "new_win" else 1,
                'move_count': 25 + i,
                'game_time': 1.0 + i * 0.1,
                'new_model_first': i % 2 == 0,
                'final_board': f'Mock board {i}'
            })
        return results

    def test_evaluator_initialization(self, mock_config):
        """Test model evaluator initialization."""
        evaluator = ModelEvaluator(mock_config)

        assert evaluator.config == mock_config
        assert evaluator.elo_system is not None
        assert isinstance(evaluator.elo_system, ELORatingSystem)

    def test_evaluator_with_custom_elo(self, mock_config):
        """Test evaluator with custom ELO system."""
        custom_elo = ELORatingSystem()
        evaluator = ModelEvaluator(mock_config, custom_elo)

        assert evaluator.elo_system == custom_elo
        assert isinstance(evaluator.elo_system, ELORatingSystem)

    @patch('src.training.evaluator.SelfPlayGameGenerator')
    def test_generator_creation(self, mock_generator_class, mock_config):
        """Test self-play generator creation for evaluation."""
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator

        evaluator = ModelEvaluator(mock_config)
        generator = evaluator._create_generator("test_model.pth")

        # Verify generator was created with correct parameters
        mock_generator_class.assert_called_once_with(
            game_type=mock_config.game_type,
            model_path="test_model.pth",
            mcts_simulations=mock_config.mcts_simulations,
            temperature_schedule=[(0, mock_config.temperature)],
            add_dirichlet_noise=mock_config.add_dirichlet_noise,
            num_threads=mock_config.num_threads
        )

        assert generator == mock_generator

    def test_analyze_results(self, mock_config, mock_game_results):
        """Test game results analysis."""
        evaluator = ModelEvaluator(mock_config)
        result = EvaluationResult(
            old_model_path="old.pth",
            new_model_path="new.pth",
            game_type="gomoku"
        )

        evaluator._analyze_results(result, mock_game_results)

        # Check basic statistics
        assert result.total_games == 10
        assert result.new_model_wins == 7
        assert result.old_model_wins == 3
        assert result.draws == 0
        assert result.win_rate == 0.7
        assert result.average_game_length == 29.5  # (25+26+...+34) / 10
        assert abs(result.average_game_time - 1.45) < 0.01  # (1.0+1.1+...+1.9) / 10

    def test_elo_rating_update(self, mock_config, mock_game_results):
        """Test ELO rating updates during evaluation."""
        evaluator = ModelEvaluator(mock_config)
        result = EvaluationResult(
            old_model_path="old.pth",
            new_model_path="new.pth",
            game_type="gomoku"
        )
        result.game_results = mock_game_results

        evaluator._update_elo_ratings(result)

        # New model should have higher rating after winning 7/10
        assert result.new_model_elo > result.old_model_elo
        assert result.elo_difference > 0

    def test_statistical_calculation(self, mock_config, mock_game_results):
        """Test statistical significance calculation."""
        evaluator = ModelEvaluator(mock_config)
        result = EvaluationResult(
            old_model_path="old.pth",
            new_model_path="new.pth",
            game_type="gomoku"
        )
        result.total_games = 10
        result.new_model_wins = 7

        evaluator._calculate_statistics(result)

        # Check confidence interval
        assert len(result.win_rate_confidence_interval) == 2
        lower, upper = result.win_rate_confidence_interval
        assert 0.0 <= lower <= 0.7 <= upper <= 1.0

        # Check significance test
        assert isinstance(result.p_value, float)
        assert 0.0 <= result.p_value <= 1.0
        assert isinstance(result.is_statistically_significant, bool)

    @patch('src.training.evaluator.SelfPlayGameGenerator')
    def test_single_game_play(self, mock_generator_class, mock_config):
        """Test single game play between models."""
        # Mock generators
        mock_generator = Mock()
        mock_game_result = GameResult(
            winner=0,
            move_count=30,
            game_length_seconds=2.0,
            examples=[],
            final_board="Mock final board",
            metadata={}
        )
        mock_generator.generate_game.return_value = mock_game_result

        evaluator = ModelEvaluator(mock_config)

        # Test game where new model plays first and wins
        game_result = evaluator._play_single_game(
            mock_generator, mock_generator, 0, new_model_first=True
        )

        assert game_result['outcome'] == 'new_win'
        assert game_result['new_model_score'] == 1.0
        assert game_result['winner'] == 0
        assert game_result['move_count'] == 30
        assert game_result['new_model_first'] == True

    @patch('src.training.evaluator.SelfPlayGameGenerator')
    def test_single_game_play_draw(self, mock_generator_class, mock_config):
        """Test single game play resulting in draw."""
        # Mock generator with draw result
        mock_generator = Mock()
        mock_game_result = GameResult(
            winner=None,  # Draw
            move_count=50,
            game_length_seconds=3.0,
            examples=[],
            final_board="Draw board",
            metadata={}
        )
        mock_generator.generate_game.return_value = mock_game_result

        evaluator = ModelEvaluator(mock_config)

        game_result = evaluator._play_single_game(
            mock_generator, mock_generator, 0, new_model_first=False
        )

        assert game_result['outcome'] == 'draw'
        assert game_result['new_model_score'] == 0.5
        assert game_result['winner'] is None

    @patch('src.training.evaluator.SelfPlayGameGenerator')
    def test_single_game_play_error_handling(self, mock_generator_class, mock_config):
        """Test error handling in single game play."""
        # Mock generator that raises exception
        mock_generator = Mock()
        mock_generator.generate_game.side_effect = Exception("Game generation failed")

        evaluator = ModelEvaluator(mock_config)

        game_result = evaluator._play_single_game(
            mock_generator, mock_generator, 0, new_model_first=True
        )

        # Should return draw as fallback
        assert game_result['outcome'] == 'draw'
        assert game_result['new_model_score'] == 0.5
        assert 'error' in game_result

    @patch('src.training.evaluator.ModelEvaluator._play_head_to_head_games')
    @patch('src.training.evaluator.SelfPlayGameGenerator')
    def test_full_evaluation(self, mock_generator_class, mock_play_games,
                           mock_config, mock_game_results):
        """Test full model evaluation process."""
        # Setup mocks
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        mock_play_games.return_value = mock_game_results

        evaluator = ModelEvaluator(mock_config)

        # Run evaluation
        result = evaluator.evaluate_models("old.pth", "new.pth")

        # Verify result
        assert result.old_model_path == "old.pth"
        assert result.new_model_path == "new.pth"
        assert result.total_games == 10
        assert result.new_model_wins == 7
        assert result.win_rate == 0.7
        assert result.evaluation_duration > 0

        # Verify generators were created and shutdown
        assert mock_generator_class.call_count == 2
        assert mock_generator.shutdown.call_count == 2


class TestContractFunction:
    """Test contract function compliance."""

    @patch('src.training.evaluator.ModelEvaluator')
    def test_evaluate_model_strength_contract(self, mock_evaluator_class):
        """Test contract function evaluate_model_strength."""
        # Mock evaluator and result
        mock_evaluator = Mock()
        mock_result = EvaluationResult(
            old_model_path="old.pth",
            new_model_path="new.pth",
            game_type="chess"
        )
        mock_result.total_games = 50
        mock_result.new_model_wins = 30
        mock_result.win_rate = 0.6
        mock_result.elo_difference = 50.0
        mock_result.evaluation_duration = 120.0
        mock_result.is_statistically_significant = True
        mock_result.p_value = 0.02
        mock_result.win_rate_confidence_interval = (0.45, 0.75)
        mock_result.evaluation_id = "test-123"
        mock_result.timestamp = "2025-09-24 12:00:00"

        mock_evaluator.evaluate_models.return_value = mock_result
        mock_evaluator_class.return_value = mock_evaluator

        # Call contract function
        result = evaluate_model_strength(
            old_model_path="old.pth",
            new_model_path="new.pth",
            game_type="chess",
            num_games=50,
            time_per_move=2.0
        )

        # Verify result format
        assert isinstance(result, dict)
        assert result['old_model_path'] == "old.pth"
        assert result['new_model_path'] == "new.pth"
        assert result['game_type'] == "chess"
        assert result['total_games'] == 50
        assert result['win_rate'] == 0.6
        assert result['elo_difference'] == 50.0
        assert 'evaluation_id' in result
        assert 'timestamp' in result

        # Verify evaluator was configured correctly
        mock_evaluator_class.assert_called_once()
        config_arg = mock_evaluator_class.call_args[0][0]
        assert config_arg.game_type == "chess"
        assert config_arg.num_games == 50
        assert config_arg.mcts_simulations == int(800 * 2.0)  # Scaled with time


class TestFactoryFunctions:
    """Test factory functions and utilities."""

    def test_create_evaluator(self):
        """Test evaluator factory function."""
        config_dict = {
            'game_type': 'go',
            'num_games': 200,
            'mcts_simulations': 1600,
            'confidence_level': 0.99
        }

        evaluator = create_evaluator(config_dict)

        assert isinstance(evaluator, ModelEvaluator)
        assert evaluator.config.game_type == 'go'
        assert evaluator.config.num_games == 200
        assert evaluator.config.mcts_simulations == 1600
        assert evaluator.config.confidence_level == 0.99

    def test_save_evaluation_results(self, tmp_path):
        """Test saving evaluation results to disk."""
        # Create test results
        results = [
            EvaluationResult(
                old_model_path="old1.pth",
                new_model_path="new1.pth",
                game_type="gomoku",
                evaluation_id="test-1",
                timestamp="2025-09-24 12:00:00"
            ),
            EvaluationResult(
                old_model_path="old2.pth",
                new_model_path="new2.pth",
                game_type="chess",
                evaluation_id="test-2",
                timestamp="2025-09-24 13:00:00"
            )
        ]

        # Populate some data
        results[0].total_games = 100
        results[0].new_model_wins = 60
        results[0].win_rate = 0.6

        # Save results
        save_evaluation_results(results, tmp_path)

        # Verify files were created
        result_files = list(tmp_path.glob("evaluation_*.json"))
        assert len(result_files) == 2

        # Verify content
        for result_file in result_files:
            with open(result_file, 'r') as f:
                data = json.load(f)

            assert 'old_model_path' in data
            assert 'new_model_path' in data
            assert 'evaluation_id' in data
            assert 'timestamp' in data


class TestIntegrationScenarios:
    """Integration test scenarios."""

    def test_complete_evaluation_workflow(self):
        """Test complete evaluation workflow with mocked components."""
        config = EvaluationConfig(
            game_type="gomoku",
            num_games=20,
            mcts_simulations=100
        )

        # Create evaluator with known ELO system
        elo_system = ELORatingSystem()
        evaluator = ModelEvaluator(config, elo_system)

        # Mock the game playing to return predictable results
        def mock_play_games(old_gen, new_gen):
            # Simulate new model winning 15/20 games
            results = []
            for i in range(20):
                outcome = "new_win" if i < 15 else "old_win"
                score = 1.0 if outcome == "new_win" else 0.0
                results.append({
                    'game_id': f'game_{i}',
                    'game_idx': i,
                    'outcome': outcome,
                    'new_model_score': score,
                    'winner': 0 if outcome == "new_win" else 1,
                    'move_count': 30,
                    'game_time': 1.0,
                    'new_model_first': i % 2 == 0,
                    'final_board': 'test'
                })
            return results

        with patch.object(evaluator, '_play_head_to_head_games', side_effect=mock_play_games):
            with patch.object(evaluator, '_create_generator', return_value=Mock()):
                result = evaluator.evaluate_models("old.pth", "new.pth")

        # Verify comprehensive result
        assert result.total_games == 20
        assert result.new_model_wins == 15
        assert result.win_rate == 0.75
        assert result.new_model_elo > result.old_model_elo
        # Check that we have a reasonable p-value (may or may not be significant with small sample)
        assert 0.0 <= result.p_value <= 1.0
        assert result.evaluation_duration > 0

    def test_edge_case_no_games(self):
        """Test edge case with zero games."""
        config = EvaluationConfig(num_games=0)
        evaluator = ModelEvaluator(config)

        result = EvaluationResult(
            old_model_path="old.pth",
            new_model_path="new.pth",
            game_type="gomoku"
        )

        # Test statistics calculation with no games
        evaluator._calculate_statistics(result)

        assert result.win_rate_confidence_interval == (0.0, 0.0)
        assert result.p_value == 1.0
        assert not result.is_statistically_significant

    def test_perfect_win_rate_statistics(self):
        """Test statistics with perfect win rate."""
        analyzer = StatisticalAnalyzer()

        # Perfect win rate should have high confidence
        lower, upper = analyzer.wilson_confidence_interval(100, 100, 0.95)
        assert lower > 0.95
        assert abs(upper - 1.0) < 1e-10  # Account for floating point precision

        # Should be highly significant
        p_value = analyzer.binomial_test(100, 100, 0.5)
        assert p_value < 0.001


if __name__ == '__main__':
    pytest.main([__file__])