#!/usr/bin/env python3
"""Debug script to test self-play generation in isolation."""

import logging
import sys
import traceback
from pathlib import Path

# Configure logging to show all errors
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress verbose loggers
logging.getLogger('device_manager').setLevel(logging.ERROR)
logging.getLogger('InferenceWorker').setLevel(logging.ERROR)
logging.getLogger('src.core.search_coordinator').setLevel(logging.ERROR)
logging.getLogger('AlphaZeroMCTS').setLevel(logging.ERROR)

from src.training.self_play import SelfPlayGameGenerator

def test_single_game():
    """Test generating a single self-play game."""
    print("=" * 80)
    print("Testing Single Self-Play Game Generation")
    print("=" * 80)

    model_path = "models/gomoku_48h/latest.pth"

    if not Path(model_path).exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Using fallback model...")
        model_path = "models/gomoku_test.pth"

    print(f"Using model: {model_path}")
    print()

    try:
        # Create self-play generator
        print("Creating self-play generator...")
        generator = SelfPlayGameGenerator(
            game_type="gomoku",
            model_path=model_path,
            mcts_simulations=400,  # Realistic training value
            num_threads=8
        )
        print("✓ Self-play generator created")
        print()

        # Generate a single game
        print("Generating single game...")
        game_id = "test_game_001"

        try:
            game_result = generator.generate_game(game_id)
            print("✓ Game generated successfully!")
            print(f"  - Moves: {game_result.move_count}")
            print(f"  - Winner: {game_result.winner}")
            print(f"  - Training examples: {len(game_result.examples)}")
            print(f"  - Game time: {game_result.game_length_seconds:.2f}s")
            return True

        except Exception as e:
            print(f"✗ Game generation FAILED")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception message: {str(e)}")
            print(f"Exception repr: {repr(e)}")
            print("\nFull traceback:")
            traceback.print_exc()
            return False

    except Exception as e:
        print(f"✗ Failed to create self-play generator")
        print(f"Exception: {e}")
        traceback.print_exc()
        return False

def test_parallel_games():
    """Test generating multiple games in parallel."""
    print("\n" + "=" * 80)
    print("Testing Parallel Self-Play Game Generation")
    print("=" * 80)

    model_path = "models/gomoku_48h/latest.pth"
    if not Path(model_path).exists():
        model_path = "models/gomoku_test.pth"

    print(f"Using model: {model_path}")
    print()

    try:
        generator = SelfPlayGameGenerator(
            game_type="gomoku",
            model_path=model_path,
            mcts_simulations=100,
            num_threads=4
        )

        print("Generating 3 games in parallel...")
        games_completed = 0
        games_failed = 0

        for i, game_result in enumerate(generator.generate_games(num_games=3, parallel_games=2)):
            games_completed += 1
            print(f"  Game {i+1}: {game_result.move_count} moves, winner={game_result.winner}")

        print(f"\n✓ Completed {games_completed} games")

        if games_completed == 3:
            return True
        else:
            print(f"✗ Only completed {games_completed}/3 games")
            return False

    except Exception as e:
        print(f"✗ Parallel game generation failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Self-Play Debug Test Suite")
    print("=" * 80)
    print()

    # Test single game
    test1_passed = test_single_game()

    # Test parallel games if single game works
    test2_passed = False
    if test1_passed:
        test2_passed = test_parallel_games()
    else:
        print("\nSkipping parallel test due to single game failure")

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Single game test: {'PASS' if test1_passed else 'FAIL'}")
    print(f"Parallel game test: {'PASS' if test2_passed else 'FAIL' if test1_passed else 'SKIPPED'}")
    print()

    if test1_passed and test2_passed:
        print("✓ All tests PASSED")
        sys.exit(0)
    else:
        print("✗ Some tests FAILED")
        sys.exit(1)
