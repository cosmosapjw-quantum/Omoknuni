#!/usr/bin/env python3
"""Quick test of parallel MCTS performance improvement."""

import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
logging.getLogger('AlphaZeroMCTS').setLevel(logging.INFO)

from src.training.self_play import SelfPlayGameGenerator

model_path = "models/gomoku_48h/latest.pth"
if not Path(model_path).exists():
    model_path = "models/gomoku_test.pth"

print("=" * 80)
print("Testing Parallel MCTS Performance")
print("=" * 80)
print(f"Model: {model_path}")
print()

# Test with 100 simulations (quick test)
generator = SelfPlayGameGenerator(
    game_type="gomoku",
    model_path=model_path,
    mcts_simulations=100,
    num_threads=8,
    batch_size_max=96,
    inference_timeout_ms=4.0
)

print("Generating 1 test game with 100 simulations per move...")
print("First move will show parallel thread initialization")
print()

start = time.time()
try:
    game = generator.generate_game("perf_test_001")
    elapsed = time.time() - start

    print(f"\n✓ Game completed successfully!")
    print(f"  Moves: {game.move_count}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"  Time per move: {elapsed/game.move_count:.2f}s")
    print(f"  Simulations/second: {100 * game.move_count / elapsed:.0f}")
    print()

    if elapsed / game.move_count < 1.0:
        print("✓ Performance looks good (< 1s per move with 100 sims)")
    else:
        print("⚠ Still slow, but should be faster than before")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
