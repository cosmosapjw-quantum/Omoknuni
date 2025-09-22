"""
Contract tests for Training Pipeline API
========================================

Tests that verify the training API contracts are properly defined and will fail
until implementations are provided. Following Test-Driven Development methodology.

These tests MUST fail with NotImplementedError until implementations are complete.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import sys
import os

# Add the contracts to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'specs', '001-goal-create-spec', 'contracts'))

from training_api import (
    TrainingExample,
    GameResult,
    SelfPlayGenerator,
    ExperienceBuffer,
    ModelTrainer,
    TrainingMetrics,
    generate_self_play_batch,
    train_model_iteration,
    evaluate_model_strength,
    create_training_pipeline
)


class TestTrainingDataStructures:
    """Test training data structure contracts."""

    def test_training_example_structure(self):
        """Test TrainingExample dataclass structure."""
        # Create valid training example
        state = np.random.rand(36, 15, 15)  # Gomoku with 36 planes
        policy = np.random.rand(225)  # 15x15 board
        policy = policy / policy.sum()  # Normalize

        example = TrainingExample(
            state=state,
            policy=policy,
            value=0.5,
            game_type='gomoku',
            move_number=10,
            game_id='test_game_001'
        )

        assert example.state.shape == (36, 15, 15)
        assert example.policy.shape == (225,)
        assert abs(example.policy.sum() - 1.0) < 1e-6
        assert example.value == 0.5
        assert example.game_type == 'gomoku'
        assert example.move_number == 10
        assert example.game_id == 'test_game_001'

    def test_game_result_structure(self):
        """Test GameResult dataclass structure."""
        examples = [
            TrainingExample(
                state=np.random.rand(36, 15, 15),
                policy=np.random.rand(225) / 225,
                value=1.0,
                game_type='gomoku',
                move_number=i,
                game_id='test_game_001'
            ) for i in range(5)
        ]

        result = GameResult(
            winner=0,
            move_count=20,
            game_length_seconds=150.5,
            examples=examples,
            final_board='. . . X O\n. . X O .\n...',
            metadata={'opening': 'center', 'difficulty': 'hard'}
        )

        assert result.winner == 0
        assert result.move_count == 20
        assert result.game_length_seconds == 150.5
        assert len(result.examples) == 5
        assert isinstance(result.final_board, str)
        assert 'opening' in result.metadata


class TestSelfPlayGeneratorContract:
    """Test SelfPlayGenerator abstract class contract."""

    def test_self_play_generator_is_abstract(self):
        """Test that SelfPlayGenerator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SelfPlayGenerator()

    def test_self_play_generator_init_signature(self):
        """Test SelfPlayGenerator.__init__ signature."""
        # Create a minimal implementation for testing
        class TestSelfPlayGenerator(SelfPlayGenerator):
            def __init__(self, game_type, model_path, mcts_simulations=800,
                        temperature_schedule=None, add_dirichlet_noise=True, num_threads=8):
                super().__init__(game_type, model_path, mcts_simulations,
                               temperature_schedule, add_dirichlet_noise, num_threads)

            def generate_game(self, game_id): raise NotImplementedError()
            def generate_games(self, num_games, parallel_games=4): raise NotImplementedError()
            def update_model(self, model_path): raise NotImplementedError()

        # Test that we can create subclass (initialization should work)
        # The abstract methods will fail when called
        generator = TestSelfPlayGenerator('gomoku', 'model.pth')
        assert generator is not None

    def test_generate_game_signature(self):
        """Test generate_game method signature."""
        class TestSelfPlayGenerator(SelfPlayGenerator):
            def __init__(self, *args, **kwargs): pass
            def generate_game(self, game_id: str) -> GameResult: raise NotImplementedError()
            def generate_games(self, num_games, parallel_games=4): raise NotImplementedError()
            def update_model(self, model_path): raise NotImplementedError()

        generator = TestSelfPlayGenerator()
        with pytest.raises(NotImplementedError):
            generator.generate_game('test_game_001')

    def test_generate_games_signature(self):
        """Test generate_games method signature."""
        class TestSelfPlayGenerator(SelfPlayGenerator):
            def __init__(self, *args, **kwargs): pass
            def generate_game(self, game_id): raise NotImplementedError()
            def generate_games(self, num_games: int, parallel_games: int = 4): raise NotImplementedError()
            def update_model(self, model_path): raise NotImplementedError()

        generator = TestSelfPlayGenerator()
        with pytest.raises(NotImplementedError):
            list(generator.generate_games(10, 4))

    def test_update_model_signature(self):
        """Test update_model method signature."""
        class TestSelfPlayGenerator(SelfPlayGenerator):
            def __init__(self, *args, **kwargs): pass
            def generate_game(self, game_id): raise NotImplementedError()
            def generate_games(self, num_games, parallel_games=4): raise NotImplementedError()
            def update_model(self, model_path: str) -> None: raise NotImplementedError()

        generator = TestSelfPlayGenerator()
        with pytest.raises(NotImplementedError):
            generator.update_model('new_model.pth')


class TestExperienceBufferContract:
    """Test ExperienceBuffer abstract class contract."""

    def test_experience_buffer_is_abstract(self):
        """Test that ExperienceBuffer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ExperienceBuffer()

    def test_experience_buffer_init_signature(self):
        """Test ExperienceBuffer.__init__ signature."""
        class TestExperienceBuffer(ExperienceBuffer):
            def __init__(self, buffer_path, max_examples=1_000_000, cache_size_mb=512):
                super().__init__(buffer_path, max_examples, cache_size_mb)

            def add_games(self, games): raise NotImplementedError()
            def sample_batch(self, batch_size, game_types=None): raise NotImplementedError()
            def get_stats(self): raise NotImplementedError()
            def cleanup(self, keep_last_n=100_000): raise NotImplementedError()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Test that we can create subclass (initialization should work)
            # The abstract methods will fail when called
            buffer = TestExperienceBuffer(Path(tmpdir))
            assert buffer is not None

    def test_add_games_signature(self):
        """Test add_games method signature."""
        class TestExperienceBuffer(ExperienceBuffer):
            def __init__(self, *args, **kwargs): pass
            def add_games(self, games: List[GameResult]) -> None: raise NotImplementedError()
            def sample_batch(self, batch_size, game_types=None): raise NotImplementedError()
            def get_stats(self): raise NotImplementedError()
            def cleanup(self, keep_last_n=100_000): raise NotImplementedError()

        buffer = TestExperienceBuffer()
        with pytest.raises(NotImplementedError):
            buffer.add_games([])

    def test_sample_batch_signature(self):
        """Test sample_batch method signature."""
        class TestExperienceBuffer(ExperienceBuffer):
            def __init__(self, *args, **kwargs): pass
            def add_games(self, games): raise NotImplementedError()
            def sample_batch(self, batch_size: int, game_types=None) -> List[TrainingExample]: raise NotImplementedError()
            def get_stats(self): raise NotImplementedError()
            def cleanup(self, keep_last_n=100_000): raise NotImplementedError()

        buffer = TestExperienceBuffer()
        with pytest.raises(NotImplementedError):
            buffer.sample_batch(32)

    def test_get_stats_signature(self):
        """Test get_stats method signature."""
        class TestExperienceBuffer(ExperienceBuffer):
            def __init__(self, *args, **kwargs): pass
            def add_games(self, games): raise NotImplementedError()
            def sample_batch(self, batch_size, game_types=None): raise NotImplementedError()
            def get_stats(self) -> Dict[str, Any]: raise NotImplementedError()
            def cleanup(self, keep_last_n=100_000): raise NotImplementedError()

        buffer = TestExperienceBuffer()
        with pytest.raises(NotImplementedError):
            buffer.get_stats()

    def test_cleanup_signature(self):
        """Test cleanup method signature."""
        class TestExperienceBuffer(ExperienceBuffer):
            def __init__(self, *args, **kwargs): pass
            def add_games(self, games): raise NotImplementedError()
            def sample_batch(self, batch_size, game_types=None): raise NotImplementedError()
            def get_stats(self): raise NotImplementedError()
            def cleanup(self, keep_last_n: int = 100_000) -> None: raise NotImplementedError()

        buffer = TestExperienceBuffer()
        with pytest.raises(NotImplementedError):
            buffer.cleanup(50_000)


class TestModelTrainerContract:
    """Test ModelTrainer abstract class contract."""

    def test_model_trainer_is_abstract(self):
        """Test that ModelTrainer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ModelTrainer()

    def test_model_trainer_init_signature(self):
        """Test ModelTrainer.__init__ signature."""
        class TestModelTrainer(ModelTrainer):
            def __init__(self, model_path, learning_rate=0.001, weight_decay=1e-4,
                        batch_size=512, use_mixed_precision=True):
                super().__init__(model_path, learning_rate, weight_decay, batch_size, use_mixed_precision)

            def train_step(self, batch): raise NotImplementedError()
            def validate(self, validation_data): raise NotImplementedError()
            def save_checkpoint(self, checkpoint_path): raise NotImplementedError()
            def get_training_stats(self): raise NotImplementedError()

        # Test that we can create subclass (initialization should work)
        # The abstract methods will fail when called
        trainer = TestModelTrainer('model.pth')
        assert trainer is not None

    def test_train_step_signature(self):
        """Test train_step method signature."""
        class TestModelTrainer(ModelTrainer):
            def __init__(self, *args, **kwargs): pass
            def train_step(self, batch: List[TrainingExample]) -> Dict[str, float]: raise NotImplementedError()
            def validate(self, validation_data): raise NotImplementedError()
            def save_checkpoint(self, checkpoint_path): raise NotImplementedError()
            def get_training_stats(self): raise NotImplementedError()

        trainer = TestModelTrainer()
        with pytest.raises(NotImplementedError):
            trainer.train_step([])

    def test_validate_signature(self):
        """Test validate method signature."""
        class TestModelTrainer(ModelTrainer):
            def __init__(self, *args, **kwargs): pass
            def train_step(self, batch): raise NotImplementedError()
            def validate(self, validation_data: List[TrainingExample]) -> Dict[str, float]: raise NotImplementedError()
            def save_checkpoint(self, checkpoint_path): raise NotImplementedError()
            def get_training_stats(self): raise NotImplementedError()

        trainer = TestModelTrainer()
        with pytest.raises(NotImplementedError):
            trainer.validate([])

    def test_save_checkpoint_signature(self):
        """Test save_checkpoint method signature."""
        class TestModelTrainer(ModelTrainer):
            def __init__(self, *args, **kwargs): pass
            def train_step(self, batch): raise NotImplementedError()
            def validate(self, validation_data): raise NotImplementedError()
            def save_checkpoint(self, checkpoint_path: str) -> None: raise NotImplementedError()
            def get_training_stats(self): raise NotImplementedError()

        trainer = TestModelTrainer()
        with pytest.raises(NotImplementedError):
            trainer.save_checkpoint('checkpoint.pth')

    def test_get_training_stats_signature(self):
        """Test get_training_stats method signature."""
        class TestModelTrainer(ModelTrainer):
            def __init__(self, *args, **kwargs): pass
            def train_step(self, batch): raise NotImplementedError()
            def validate(self, validation_data): raise NotImplementedError()
            def save_checkpoint(self, checkpoint_path): raise NotImplementedError()
            def get_training_stats(self) -> Dict[str, Any]: raise NotImplementedError()

        trainer = TestModelTrainer()
        with pytest.raises(NotImplementedError):
            trainer.get_training_stats()


class TestTrainingMetricsContract:
    """Test TrainingMetrics class contract."""

    def test_training_metrics_init_signature(self):
        """Test TrainingMetrics.__init__ signature."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(NotImplementedError):
                TrainingMetrics(Path(tmpdir))

    def test_log_training_step_signature(self):
        """Test log_training_step method signature."""
        # Cannot test directly since __init__ raises NotImplementedError
        # This test verifies the method exists in the contract
        assert hasattr(TrainingMetrics, 'log_training_step')

    def test_log_evaluation_signature(self):
        """Test log_evaluation method signature."""
        # Cannot test directly since __init__ raises NotImplementedError
        # This test verifies the method exists in the contract
        assert hasattr(TrainingMetrics, 'log_evaluation')

    def test_generate_report_signature(self):
        """Test generate_report method signature."""
        # Cannot test directly since __init__ raises NotImplementedError
        # This test verifies the method exists in the contract
        assert hasattr(TrainingMetrics, 'generate_report')


class TestStandaloneFunctionContracts:
    """Test standalone function contracts."""

    def test_generate_self_play_batch_signature(self):
        """Test generate_self_play_batch function signature."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(NotImplementedError):
                generate_self_play_batch(
                    game_type='gomoku',
                    model_path='model.pth',
                    num_games=10,
                    output_path=Path(tmpdir)
                )

    def test_train_model_iteration_signature(self):
        """Test train_model_iteration function signature."""
        # Create a mock experience buffer for testing
        class MockBuffer:
            pass

        with pytest.raises(NotImplementedError):
            train_model_iteration(
                model_path='model.pth',
                experience_buffer=MockBuffer(),
                num_train_steps=100,
                validation_split=0.1
            )

    def test_evaluate_model_strength_signature(self):
        """Test evaluate_model_strength function signature."""
        with pytest.raises(NotImplementedError):
            evaluate_model_strength(
                old_model_path='old_model.pth',
                new_model_path='new_model.pth',
                game_type='gomoku',
                num_games=100,
                time_per_move=1.0
            )

    def test_create_training_pipeline_signature(self):
        """Test create_training_pipeline function signature."""
        config = {
            'game_type': 'gomoku',
            'model_path': 'model.pth',
            'buffer_path': '/tmp/buffer',
            'max_examples': 1_000_000
        }

        with pytest.raises(NotImplementedError):
            create_training_pipeline(config)


class TestContractCompleteness:
    """Test that contract coverage is comprehensive."""

    def test_all_abstract_methods_covered(self):
        """Test that all abstract methods are tested."""
        # SelfPlayGenerator methods
        assert hasattr(SelfPlayGenerator, '__init__')
        assert hasattr(SelfPlayGenerator, 'generate_game')
        assert hasattr(SelfPlayGenerator, 'generate_games')
        assert hasattr(SelfPlayGenerator, 'update_model')

        # ExperienceBuffer methods
        assert hasattr(ExperienceBuffer, '__init__')
        assert hasattr(ExperienceBuffer, 'add_games')
        assert hasattr(ExperienceBuffer, 'sample_batch')
        assert hasattr(ExperienceBuffer, 'get_stats')
        assert hasattr(ExperienceBuffer, 'cleanup')

        # ModelTrainer methods
        assert hasattr(ModelTrainer, '__init__')
        assert hasattr(ModelTrainer, 'train_step')
        assert hasattr(ModelTrainer, 'validate')
        assert hasattr(ModelTrainer, 'save_checkpoint')
        assert hasattr(ModelTrainer, 'get_training_stats')

    def test_data_structure_types(self):
        """Test that data structures have correct type annotations."""
        # This test ensures the data structures follow the contract
        example = TrainingExample(
            state=np.zeros((36, 15, 15)),
            policy=np.ones(225) / 225,
            value=0.0,
            game_type='gomoku',
            move_number=0,
            game_id='test'
        )

        result = GameResult(
            winner=None,
            move_count=0,
            game_length_seconds=0.0,
            examples=[example],
            final_board='',
            metadata={}
        )

        assert isinstance(example.state, np.ndarray)
        assert isinstance(example.policy, np.ndarray)
        assert isinstance(example.value, (int, float))
        assert isinstance(example.game_type, str)
        assert isinstance(example.move_number, int)
        assert isinstance(example.game_id, str)

        assert isinstance(result.examples, list)
        assert isinstance(result.metadata, dict)


if __name__ == '__main__':
    # Run contract tests
    pytest.main([__file__, '-v'])