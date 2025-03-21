#!/usr/bin/env python3
"""
Test script for the batched MCTS implementation with leaf parallelization.

This script tests the performance of the batched MCTS implementation
by comparing it to the improved MCTS implementation and measuring
the speed and efficiency of neural network evaluation.
"""

import numpy as np
import time
import os
import sys
from tqdm import tqdm
import argparse

try:
    from alphazero.python.games.gomoku import GomokuGame
    from alphazero.python.mcts.improved_cpp_mcts_wrapper import ImprovedCppMCTSWrapper
    from alphazero.python.mcts.batched_cpp_mcts_wrapper import BatchedCppMCTSWrapper
    from alphazero.python.models.simple_conv_net import SimpleConvNet
except ImportError:
    # Add the project directory to the Python path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from alphazero.python.games.gomoku import GomokuGame
    from alphazero.python.mcts.improved_cpp_mcts_wrapper import ImprovedCppMCTSWrapper
    from alphazero.python.mcts.batched_cpp_mcts_wrapper import BatchedCppMCTSWrapper
    from alphazero.python.models.simple_conv_net import SimpleConvNet


def create_dummy_neural_network(board_size, device='cpu'):
    """Create a dummy neural network for testing."""
    class DummyNN:
        def __init__(self, board_size):
            self.board_size = board_size

        def __call__(self, board):
            # Return uniform policy and random value for testing
            policy = {i: 1.0 / (self.board_size * self.board_size) 
                     for i in range(self.board_size * self.board_size)}
            value = np.random.random() * 2 - 1  # Random value between -1 and 1
            return policy, value
    
    return DummyNN(board_size)


def create_batched_evaluator(neural_network, board_size):
    """Create a batched evaluator that can process multiple boards at once."""
    def batched_evaluator(games):
        # This would normally be a neural network batch evaluation
        results = []
        for game in games:
            policy, value = neural_network(game.get_board())
            results.append((policy, value))
        return results
    
    return batched_evaluator


def run_mcts_test(game_class, board_size, num_games, num_simulations, num_threads, 
                 use_batched=True, batch_size=16, max_wait_ms=10, verbose=True):
    """
    Run a test comparing different MCTS implementations.
    
    Args:
        game_class: The game class to use
        board_size: Size of the game board
        num_games: Number of games to play
        num_simulations: Number of simulations per move
        num_threads: Number of threads to use
        use_batched: Whether to use the batched MCTS implementation
        batch_size: Size of batches for neural network evaluation
        max_wait_ms: Maximum wait time for batch completion
        verbose: Whether to print detailed output
    
    Returns:
        Dictionary with performance metrics
    """
    # Create a neural network for evaluation
    neural_network = create_dummy_neural_network(board_size)
    
    # Function to evaluate a single game state
    def single_evaluator(game):
        return neural_network(game.get_board())
    
    # Function to evaluate a batch of game states
    batched_evaluator = create_batched_evaluator(neural_network, board_size)
    
    # Create initial game
    game = game_class(board_size=board_size)
    
    # Statistics
    total_time = 0.0
    total_moves = 0
    total_evals = 0
    
    # Create MCTS agent
    if use_batched:
        mcts = BatchedCppMCTSWrapper(
            game=game,
            evaluator=batched_evaluator,
            num_simulations=num_simulations,
            num_threads=num_threads,
            batch_size=batch_size,
            max_wait_ms=max_wait_ms
        )
        if verbose:
            print(f"Using batched MCTS with {num_threads} threads, batch size {batch_size}")
    else:
        mcts = ImprovedCppMCTSWrapper(
            game=game,
            evaluator=single_evaluator,
            num_simulations=num_simulations,
            num_threads=num_threads
        )
        if verbose:
            print(f"Using improved MCTS with {num_threads} threads")
    
    # Play games
    for game_idx in range(num_games):
        game.reset()
        moves_played = 0
        
        # Play until game is over or max moves reached
        while not game.is_terminal() and moves_played < 10:  # Limit to 10 moves for testing
            start_time = time.time()
            
            # Select a move
            move = mcts.select_move()
            
            elapsed = time.time() - start_time
            if verbose:
                print(f"Game {game_idx+1}, move {moves_played+1}: "
                      f"Selected move {move} in {elapsed:.3f}s")
            
            # Apply the move
            game.apply_move(move)
            mcts.update_with_move(move)
            
            # Update statistics
            total_time += elapsed
            total_moves += 1
            total_evals += mcts.eval_count
            
            moves_played += 1
    
    # Calculate performance metrics
    avg_time_per_move = total_time / total_moves if total_moves > 0 else 0
    avg_evals_per_move = total_evals / total_moves if total_moves > 0 else 0
    evals_per_second = total_evals / total_time if total_time > 0 else 0
    
    # Print summary
    if verbose:
        print("\nPerformance Summary:")
        print(f"Total moves: {total_moves}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Average time per move: {avg_time_per_move:.3f}s")
        print(f"Average evaluations per move: {avg_evals_per_move:.1f}")
        print(f"Evaluations per second: {evals_per_second:.1f}")
    
    return {
        "total_moves": total_moves,
        "total_time": total_time,
        "avg_time_per_move": avg_time_per_move,
        "avg_evals_per_move": avg_evals_per_move,
        "evals_per_second": evals_per_second
    }


def compare_implementations(game_class, board_size, num_games, num_simulations, num_threads_list, 
                          batch_sizes=None, verbose=True):
    """
    Compare different MCTS implementations with varying thread counts and batch sizes.
    
    Args:
        game_class: The game class to use
        board_size: Size of the game board
        num_games: Number of games to play
        num_simulations: Number of simulations per move
        num_threads_list: List of thread counts to test
        batch_sizes: List of batch sizes to test
        verbose: Whether to print detailed output
    """
    results = []
    
    # Test improved MCTS with different thread counts
    for num_threads in num_threads_list:
        if verbose:
            print(f"\n=== Testing Improved MCTS with {num_threads} threads ===")
        result = run_mcts_test(
            game_class=game_class,
            board_size=board_size,
            num_games=num_games,
            num_simulations=num_simulations,
            num_threads=num_threads,
            use_batched=False,
            verbose=verbose
        )
        result["implementation"] = "Improved MCTS"
        result["num_threads"] = num_threads
        result["batch_size"] = None
        results.append(result)
    
    # Test batched MCTS with different thread counts and batch sizes
    if batch_sizes is None:
        batch_sizes = [8, 16, 32]
    
    for num_threads in num_threads_list:
        for batch_size in batch_sizes:
            if verbose:
                print(f"\n=== Testing Batched MCTS with {num_threads} threads, "
                      f"batch size {batch_size} ===")
            result = run_mcts_test(
                game_class=game_class,
                board_size=board_size,
                num_games=num_games,
                num_simulations=num_simulations,
                num_threads=num_threads,
                use_batched=True,
                batch_size=batch_size,
                verbose=verbose
            )
            result["implementation"] = "Batched MCTS"
            result["num_threads"] = num_threads
            result["batch_size"] = batch_size
            results.append(result)
    
    # Print comparison
    if verbose:
        print("\n=== Performance Comparison ===")
        print(f"{'Implementation':<15} {'Threads':<8} {'Batch Size':<10} "
              f"{'Time/Move (s)':<14} {'Evals/Move':<12} {'Evals/Second':<12}")
        print("-" * 80)
        
        for result in results:
            impl = result["implementation"]
            threads = result["num_threads"]
            batch = result["batch_size"] if result["batch_size"] is not None else "N/A"
            time_per_move = result["avg_time_per_move"]
            evals_per_move = result["avg_evals_per_move"]
            evals_per_second = result["evals_per_second"]
            
            print(f"{impl:<15} {threads:<8} {batch:<10} "
                  f"{time_per_move:<14.3f} {evals_per_move:<12.1f} {evals_per_second:<12.1f}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test batched MCTS implementation")
    parser.add_argument("--board-size", type=int, default=9, help="Board size")
    parser.add_argument("--num-games", type=int, default=2, help="Number of games to play")
    parser.add_argument("--num-simulations", type=int, default=400, help="Number of simulations per move")
    parser.add_argument("--threads", type=str, default="1,2,4", help="Comma-separated list of thread counts")
    parser.add_argument("--batch-sizes", type=str, default="8,16,32", help="Comma-separated list of batch sizes")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    
    args = parser.parse_args()
    
    # Parse thread counts and batch sizes
    num_threads_list = [int(t) for t in args.threads.split(",")]
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    
    # Run comparison
    compare_implementations(
        game_class=GomokuGame,
        board_size=args.board_size,
        num_games=args.num_games,
        num_simulations=args.num_simulations,
        num_threads_list=num_threads_list,
        batch_sizes=batch_sizes,
        verbose=args.verbose
    )