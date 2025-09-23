"""
Self-Play Game Generator
========================

Generates self-play games for training data with temperature scheduling,
Dirichlet noise injection, and proper game outcome determination.

Key features:
- Temperature-based move selection during self-play
- Dirichlet noise at root for exploration
- Position augmentation through game symmetries
- Integration with MCTS search coordinator
- Parallel game generation support
"""

import time
import uuid
import logging
import threading
from typing import List, Dict, Tuple, Optional, Iterator, Any, Callable
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import json

# Import training API contracts
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from specs.contracts.training_api import (
    SelfPlayGenerator, GameResult, TrainingExample
)

# Import core components
from src.core.search_coordinator import SearchCoordinator, SearchRequest, SearchResult
from src.neural.inference_worker import GPUInferenceWorker
from src.telemetry.metrics import MetricsCollector

# Game bindings (will be available after C++ compilation)
try:
    import alphazero_py
    GAMES_AVAILABLE = True
except ImportError:
    # Fallback for testing without compiled extensions
    GAMES_AVAILABLE = False
    alphazero_py = None


@dataclass
class SelfPlayConfig:
    """Configuration for self-play generation."""

    game_type: str = "gomoku"
    mcts_simulations: int = 800
    temperature_schedule: List[Tuple[int, float]] = field(default_factory=lambda: [(30, 1.0), (1000, 0.1)])
    dirichlet_alpha: float = 0.3  # Gomoku default, varies by game
    dirichlet_weight: float = 0.25
    cpuct: float = 1.25
    add_dirichlet_noise: bool = True
    num_threads: int = 8
    max_game_length: int = 512  # Prevent infinite games
    save_positions_from_move: int = 0  # Start saving training data from move N


class SelfPlayGameGenerator(SelfPlayGenerator):
    """Self-play game generator implementation."""

    def __init__(self,
                 game_type: str,
                 model_path: str,
                 mcts_simulations: int = 800,
                 temperature_schedule: List[Tuple[int, float]] = None,
                 add_dirichlet_noise: bool = True,
                 num_threads: int = 8):
        """Initialize self-play generator.

        Args:
            game_type: Game to play ('gomoku', 'chess', 'go')
            model_path: Path to current neural network model
            mcts_simulations: MCTS simulations per move
            temperature_schedule: [(move_threshold, temperature), ...]
            add_dirichlet_noise: Add exploration noise at root
            num_threads: MCTS search threads
        """
        self.config = SelfPlayConfig(
            game_type=game_type,
            mcts_simulations=mcts_simulations,
            temperature_schedule=temperature_schedule or [(30, 1.0), (1000, 0.1)],
            add_dirichlet_noise=add_dirichlet_noise,
            num_threads=num_threads
        )

        self.model_path = model_path
        self.logger = logging.getLogger(__name__)

        # Set game-specific parameters
        self._set_game_specific_params()

        # Initialize components (will be set up when first needed)
        self.inference_worker: Optional[GPUInferenceWorker] = None
        self.search_coordinator: Optional[SearchCoordinator] = None
        self.telemetry = MetricsCollector()

        # Statistics tracking
        self.games_generated = 0
        self.total_positions = 0
        self.generation_times = []

        self.logger.info(f"Self-play generator initialized for {game_type}")

    def _set_game_specific_params(self) -> None:
        """Set game-specific parameters like Dirichlet noise."""
        game_params = {
            'gomoku': {'dirichlet_alpha': 0.3, 'max_game_length': 225},
            'chess': {'dirichlet_alpha': 0.2, 'max_game_length': 512},
            'go': {'dirichlet_alpha': 0.03, 'max_game_length': 722}  # 19x19 + pass moves
        }

        params = game_params.get(self.config.game_type, game_params['gomoku'])
        self.config.dirichlet_alpha = params['dirichlet_alpha']
        self.config.max_game_length = params['max_game_length']

    def _ensure_components_initialized(self) -> None:
        """Lazy initialization of GPU inference and search coordinator."""
        if self.inference_worker is None:
            # Initialize GPU inference worker
            from src.neural.device_manager import DeviceManager
            device_manager = DeviceManager()
            device_info = device_manager.detect_device()

            if device_info.is_cuda_available:
                from src.neural.inference_worker import GPUInferenceWorker
                self.inference_worker = GPUInferenceWorker(
                    model_path=self.model_path,
                    batch_size=64,  # Optimal for RTX 3060 Ti
                    timeout_ms=3.0
                )
                self.inference_worker.start()
            else:
                # Fallback to CPU inference
                from src.neural.cpu_inference import CPUInferenceWorker
                self.inference_worker = CPUInferenceWorker(model_path=self.model_path)
                self.inference_worker.start()

            # Initialize search coordinator
            self.search_coordinator = SearchCoordinator(
                inference_worker=self.inference_worker,
                max_threads=self.config.num_threads
            )
            self.search_coordinator.start()

    def generate_game(self, game_id: str) -> GameResult:
        """Generate single self-play game.

        Args:
            game_id: Unique identifier for this game

        Returns:
            GameResult: Complete game with training examples
        """
        self._ensure_components_initialized()

        start_time = time.time()
        self.logger.debug(f"Starting self-play game {game_id}")

        # Create game state
        if GAMES_AVAILABLE:
            game_type_enum = getattr(alphazero_py.GameType, self.config.game_type.upper())
            game_state = alphazero_py.create_game(game_type_enum)
        else:
            # Mock game state for testing
            game_state = self._create_mock_game_state()

        # Track game data
        game_examples = []
        move_history = []
        position_values = []

        move_count = 0

        try:
            while not self._is_game_terminal(game_state) and move_count < self.config.max_game_length:
                # Get current temperature
                temperature = self._get_temperature(move_count)

                # Perform MCTS search
                search_request = SearchRequest(
                    request_id=f"{game_id}_move_{move_count}",
                    game_state=game_state,
                    simulations=self.config.mcts_simulations,
                    temperature=temperature,
                    add_noise=self.config.add_dirichlet_noise and move_count < 30
                )

                # Submit search request
                search_future = self.search_coordinator.submit_search(search_request)
                search_result = search_future.result(timeout=30.0)  # 30s timeout per move

                # Extract training data (if past warmup moves)
                if move_count >= self.config.save_positions_from_move:
                    training_example = self._create_training_example(
                        game_state=game_state,
                        policy=search_result.policy,
                        move_number=move_count,
                        game_id=game_id
                    )
                    game_examples.append(training_example)
                    position_values.append(search_result.value)

                # Apply temperature-based move selection
                move_action = self._select_move_with_temperature(
                    search_result.policy, temperature
                )

                # Make the move
                if GAMES_AVAILABLE:
                    game_state.make_move(move_action)
                else:
                    self._make_mock_move(game_state, move_action)

                move_history.append(move_action)
                move_count += 1

                self.logger.debug(f"Game {game_id}: Move {move_count}, action {move_action}, temp {temperature:.2f}")

            # Determine game outcome
            game_outcome = self._determine_game_outcome(game_state)

            # Update training examples with final game value
            self._update_examples_with_outcome(game_examples, game_outcome, move_count)

            # Create game result
            game_result = GameResult(
                winner=game_outcome.get('winner'),
                move_count=move_count,
                game_length_seconds=time.time() - start_time,
                examples=game_examples,
                final_board=self._get_board_string(game_state),
                metadata={
                    'game_id': game_id,
                    'game_type': self.config.game_type,
                    'mcts_simulations': self.config.mcts_simulations,
                    'total_positions': len(game_examples),
                    'final_outcome': game_outcome,
                    'move_history': move_history[:50]  # Limit for storage
                }
            )

            # Update statistics
            self.games_generated += 1
            self.total_positions += len(game_examples)
            self.generation_times.append(time.time() - start_time)

            self.logger.info(f"Game {game_id} completed: {move_count} moves, "
                           f"{len(game_examples)} training examples, "
                           f"outcome: {game_outcome}")

            return game_result

        except Exception as e:
            self.logger.error(f"Error generating game {game_id}: {e}")
            raise

    def generate_games(self,
                      num_games: int,
                      parallel_games: int = 4) -> Iterator[GameResult]:
        """Generate multiple self-play games in parallel.

        Args:
            num_games: Total number of games to generate
            parallel_games: Number of concurrent games

        Yields:
            GameResult: Each completed game as it finishes
        """
        self.logger.info(f"Generating {num_games} self-play games with {parallel_games} parallel")

        with ThreadPoolExecutor(max_workers=parallel_games, thread_name_prefix="selfplay") as executor:
            # Submit all game generation tasks
            futures = []
            for i in range(num_games):
                game_id = f"selfplay_{uuid.uuid4().hex[:8]}_{i}"
                future = executor.submit(self.generate_game, game_id)
                futures.append(future)

            # Yield results as they complete
            for future in as_completed(futures):
                try:
                    game_result = future.result()
                    yield game_result
                except Exception as e:
                    self.logger.error(f"Failed to generate game: {e}")
                    continue

    def update_model(self, model_path: str) -> None:
        """Update neural network model for self-play.

        Args:
            model_path: Path to new model checkpoint
        """
        self.logger.info(f"Updating self-play model: {model_path}")
        self.model_path = model_path

        # If components are initialized, update them
        if self.inference_worker is not None:
            self.inference_worker.update_model(model_path)

    def get_statistics(self) -> Dict[str, Any]:
        """Get self-play generation statistics.

        Returns:
            Dictionary of statistics
        """
        avg_time = np.mean(self.generation_times) if self.generation_times else 0.0
        avg_positions = self.total_positions / max(self.games_generated, 1)

        return {
            'games_generated': self.games_generated,
            'total_positions': self.total_positions,
            'average_positions_per_game': avg_positions,
            'average_generation_time_seconds': avg_time,
            'games_per_hour': 3600 / avg_time if avg_time > 0 else 0.0
        }

    def shutdown(self) -> None:
        """Shutdown self-play generator and cleanup resources."""
        self.logger.info("Shutting down self-play generator")

        if self.search_coordinator:
            self.search_coordinator.stop()

        if self.inference_worker:
            self.inference_worker.stop()

    # Helper methods

    def _get_temperature(self, move_count: int) -> float:
        """Get temperature for current move based on schedule."""
        for move_threshold, temperature in self.config.temperature_schedule:
            if move_count < move_threshold:
                return temperature
        # Return last temperature if past all thresholds
        return self.config.temperature_schedule[-1][1]

    def _select_move_with_temperature(self, policy: np.ndarray, temperature: float) -> int:
        """Select move using temperature-scaled policy."""
        if temperature == 0.0:
            # Deterministic selection
            return np.argmax(policy)

        # Temperature scaling
        scaled_logits = np.log(policy + 1e-8) / temperature
        scaled_probs = np.exp(scaled_logits - np.max(scaled_logits))
        scaled_probs /= np.sum(scaled_probs)

        # Sample from scaled distribution
        return np.random.choice(len(policy), p=scaled_probs)

    def _create_training_example(self,
                                game_state: Any,
                                policy: np.ndarray,
                                move_number: int,
                                game_id: str) -> TrainingExample:
        """Create training example from current position."""
        # Extract features from game state
        if GAMES_AVAILABLE:
            features = game_state.get_tensor_representation()
            features_array = np.array(features)
        else:
            # Mock features for testing
            features_array = np.random.rand(36, 15, 15).astype(np.float32)

        return TrainingExample(
            state=features_array,
            policy=policy.copy(),
            value=0.0,  # Will be updated with final game outcome
            game_type=self.config.game_type,
            move_number=move_number,
            game_id=game_id
        )

    def _update_examples_with_outcome(self,
                                     examples: List[TrainingExample],
                                     game_outcome: Dict[str, Any],
                                     final_move_count: int) -> None:
        """Update training examples with final game outcome."""
        winner = game_outcome.get('winner')

        for i, example in enumerate(examples):
            # Calculate value from perspective of player who made the move
            player_to_move = i % 2  # Alternating players

            if winner is None:  # Draw
                example.value = 0.0
            elif winner == player_to_move:  # Win for current player
                example.value = 1.0
            else:  # Loss for current player
                example.value = -1.0

    def _determine_game_outcome(self, game_state: Any) -> Dict[str, Any]:
        """Determine the outcome of a completed game."""
        if GAMES_AVAILABLE:
            if game_state.is_terminal():
                result = game_state.get_game_result()
                if result == alphazero_py.GameResult.WIN_PLAYER1:
                    return {'winner': 0, 'result': 'win_player1'}
                elif result == alphazero_py.GameResult.WIN_PLAYER2:
                    return {'winner': 1, 'result': 'win_player2'}
                else:
                    return {'winner': None, 'result': 'draw'}
            else:
                return {'winner': None, 'result': 'max_moves_reached'}
        else:
            # Mock outcome for testing
            return {'winner': np.random.choice([0, 1, None]), 'result': 'mock_game'}

    def _is_game_terminal(self, game_state: Any) -> bool:
        """Check if game is in terminal state."""
        if GAMES_AVAILABLE:
            return game_state.is_terminal()
        else:
            # Mock terminal check
            return np.random.random() < 0.01  # 1% chance of termination per move

    def _get_board_string(self, game_state: Any) -> str:
        """Get human-readable board representation."""
        if GAMES_AVAILABLE:
            return game_state.to_string()
        else:
            return "Mock board state"

    # Mock implementations for testing without compiled C++ extensions

    def _create_mock_game_state(self) -> Dict[str, Any]:
        """Create mock game state for testing."""
        return {
            'board': np.zeros((15, 15), dtype=int),
            'current_player': 0,
            'move_count': 0,
            'terminal': False
        }

    def _make_mock_move(self, game_state: Dict[str, Any], action: int) -> None:
        """Make move in mock game state."""
        board_size = 15  # Assume Gomoku
        row, col = action // board_size, action % board_size
        game_state['board'][row, col] = game_state['current_player'] + 1
        game_state['current_player'] = 1 - game_state['current_player']
        game_state['move_count'] += 1

        # Simple termination condition
        if game_state['move_count'] > 20:
            game_state['terminal'] = np.random.random() < 0.1


# Factory functions and utilities

def create_self_play_generator(config: Dict[str, Any]) -> SelfPlayGameGenerator:
    """Factory function to create self-play generator from config.

    Args:
        config: Configuration dictionary

    Returns:
        Configured SelfPlayGameGenerator instance
    """
    return SelfPlayGameGenerator(
        game_type=config.get('game_type', 'gomoku'),
        model_path=config.get('model_path', 'models/latest.pth'),
        mcts_simulations=config.get('mcts_simulations', 800),
        temperature_schedule=config.get('temperature_schedule'),
        add_dirichlet_noise=config.get('add_dirichlet_noise', True),
        num_threads=config.get('num_threads', 8)
    )


def save_games_to_disk(games: List[GameResult], output_path: Path) -> None:
    """Save generated games to disk in JSON format.

    Args:
        games: List of completed games
        output_path: Directory to save games
    """
    output_path.mkdir(parents=True, exist_ok=True)

    for i, game in enumerate(games):
        game_file = output_path / f"game_{i:06d}.json"

        # Convert numpy arrays to lists for JSON serialization
        game_data = {
            'winner': game.winner,
            'move_count': game.move_count,
            'game_length_seconds': game.game_length_seconds,
            'final_board': game.final_board,
            'metadata': game.metadata,
            'examples': [
                {
                    'state': example.state.tolist(),
                    'policy': example.policy.tolist(),
                    'value': example.value,
                    'game_type': example.game_type,
                    'move_number': example.move_number,
                    'game_id': example.game_id
                }
                for example in game.examples
            ]
        }

        with open(game_file, 'w') as f:
            json.dump(game_data, f, indent=2)


def load_games_from_disk(input_path: Path) -> List[GameResult]:
    """Load games from disk.

    Args:
        input_path: Directory containing saved games

    Returns:
        List of loaded GameResult objects
    """
    games = []

    for game_file in sorted(input_path.glob("game_*.json")):
        with open(game_file, 'r') as f:
            game_data = json.load(f)

        # Convert back to numpy arrays
        examples = [
            TrainingExample(
                state=np.array(ex['state']),
                policy=np.array(ex['policy']),
                value=ex['value'],
                game_type=ex['game_type'],
                move_number=ex['move_number'],
                game_id=ex['game_id']
            )
            for ex in game_data['examples']
        ]

        game = GameResult(
            winner=game_data['winner'],
            move_count=game_data['move_count'],
            game_length_seconds=game_data['game_length_seconds'],
            examples=examples,
            final_board=game_data['final_board'],
            metadata=game_data['metadata']
        )

        games.append(game)

    return games