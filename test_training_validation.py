#!/usr/bin/env python3
"""Validate that training can generate games successfully."""

import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress noise
logging.getLogger('device_manager').setLevel(logging.CRITICAL)
logging.getLogger('InferenceWorker').setLevel(logging.CRITICAL)
logging.getLogger('src.core.search_coordinator').setLevel(logging.CRITICAL)
logging.getLogger('AlphaZeroMCTS').setLevel(logging.CRITICAL)

from src.training.self_play import SelfPlayGameGenerator

def test_parallel_training_games():
    """Test generating games in parallel as in training."""
    print("=" * 80)
    print("TRAINING VALIDATION: Generate 4 games in parallel")
    print("=" * 80)

    model_path = "models/gomoku_48h/latest.pth"
    if not Path(model_path).exists():
        model_path = "models/gomoku_test.pth"

    print(f"Model: {model_path}")
    print(f"Config: 800 simulations, 8 threads, 2 parallel games")
    print()

    try:
        generator = SelfPlayGameGenerator(
            game_type="gomoku",
            model_path=model_path,
            mcts_simulations=800,  # Match training config
            num_threads=8
        )

        print("Generating 4 games with 2 parallel workers...")
        games_completed = 0

        for i, game_result in enumerate(generator.generate_games(num_games=4, parallel_games=2)):
            games_completed += 1
            print(f"✓ Game {games_completed}/4 completed:")
            print(f"    Moves: {game_result.move_count}")
            print(f"    Winner: {game_result.winner}")
            print(f"    Examples: {len(game_result.examples)}")
            print(f"    Time: {game_result.game_length_seconds:.1f}s")

        if games_completed == 4:
            print()
            print("=" * 80)
            print("✓ VALIDATION PASSED: All 4 games completed successfully")
            print("=" * 80)
            return True
        else:
            print()
            print("=" * 80)
            print(f"✗ VALIDATION FAILED: Only {games_completed}/4 games completed")
            print("=" * 80)
            return False

    except Exception as e:
        print()
        print("=" * 80)
        print(f"✗ VALIDATION FAILED: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_parallel_training_games()
    sys.exit(0 if success else 1)
