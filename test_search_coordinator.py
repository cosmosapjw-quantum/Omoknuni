#!/usr/bin/env python3
"""Test search coordinator in isolation to check for hanging."""

import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Reduce noise
logging.getLogger('device_manager').setLevel(logging.WARNING)
logging.getLogger('AlphaZeroMCTS').setLevel(logging.WARNING)

from src.core.search_coordinator import SearchCoordinator, SearchRequest
from src.neural.inference_worker import GPUInferenceWorker
from src.games.game_state import create_game_state

def test_search():
    """Test a single MCTS search."""
    print("Testing Search Coordinator")
    print("=" * 80)

    model_path = "models/gomoku_48h/latest.pth"
    if not Path(model_path).exists():
        model_path = "models/gomoku_test.pth"

    print(f"Model: {model_path}")
    print(f"Creating inference worker...")

    # Create inference worker
    worker = GPUInferenceWorker(
        model_path=model_path,
        batch_size=64,
        timeout_ms=3.0
    )
    worker.start()

    print(f"Creating search coordinator with 4 threads...")
    coordinator = SearchCoordinator(
        inference_worker=worker,
        max_threads=4
    )
    coordinator.start()

    print(f"Creating game state...")
    game_state = create_game_state("gomoku")

    print(f"Submitting search request for 100 simulations...")
    start = time.time()

    request = SearchRequest(
        request_id="test_001",
        game_state=game_state,
        simulations=100,
        temperature=1.0,
        add_noise=True
    )

    future = coordinator.submit_search(request)

    try:
        result = future.result(timeout=30.0)
        elapsed = time.time() - start

        print(f"✓ Search completed in {elapsed:.2f}s")
        print(f"  Policy shape: {result.policy.shape}")
        print(f"  Best move: {result.best_move}")
        print(f"  Visit count: {result.visit_count}")
        print(f"  Value: {result.value:.3f}")

        return True

    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ Search FAILED after {elapsed:.2f}s")
        print(f"  Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        coordinator.shutdown()
        worker.shutdown()

if __name__ == "__main__":
    success = test_search()
    exit(0 if success else 1)
