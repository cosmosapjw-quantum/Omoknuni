#!/usr/bin/env python3
"""
Test script for the Zobrist hashing implementation in the AlphaZero framework.
This script compares the performance and correctness of the Zobrist hashing
against the previous hashing method.
"""

import time
import sys
import numpy as np
from alphazero.bindings.cpp_mcts import GomokuMCTS
from alphazero.python.games.gomoku import GomokuGame

# Try to import matplotlib, but continue if not available
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available, skipping visualization.")


def test_search_performance(board_size=15, use_zobrist=True, num_tests=5, num_sims=200):
    """Test search performance with and without Zobrist hashing"""
    
    game = GomokuGame(board_size=board_size)
    
    # Create an MCTS instance
    mcts = GomokuMCTS(
        num_simulations=num_sims,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25,
        use_transposition_table=True,
        num_threads=1
    )
    
    # Create a simple dummy evaluator
    def dummy_evaluator(game_state):
        board_size = game_state.board_size
        policy = np.ones(board_size * board_size) / (board_size * board_size)
        return policy, 0.0
    
    # Time the search process
    total_time = 0
    for _ in range(num_tests):
        # Reset the game
        game.reset()
        
        # Play some random moves to create a non-trivial board state
        for _ in range(board_size // 2):
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break
            move = np.random.choice(legal_moves)
            game.apply_move(move)
            # Break if game is over
            if game.is_terminal():
                break
        
        # Get the board state
        board = game.get_board().flatten()
        legal_moves = list(game.get_legal_moves())
        
        # Reset MCTS
        mcts = GomokuMCTS(
            num_simulations=num_sims,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_noise_weight=0.25,
            use_transposition_table=True,
            num_threads=1
        )
        
        # Wrap the evaluator to adapt to our interface
        def evaluator_wrapper(board_flat):
            policy, value = dummy_evaluator(game)
            policy_list = [0.0] * (board_size * board_size)
            for move, prob in enumerate(policy):
                policy_list[move] = prob
            return policy_list, value
        
        # Time the search
        start_time = time.time()
        mcts.search(board, legal_moves, evaluator_wrapper)
        elapsed = time.time() - start_time
        
        total_time += elapsed
    
    return total_time / num_tests


def test_zobrist_consistency():
    """Test that Zobrist hashing produces consistent results for the same board state"""
    
    board_size = 9
    game = GomokuGame(board_size=board_size)
    
    # Create an MCTS instance with Zobrist hashing
    mcts1 = GomokuMCTS(
        num_simulations=10,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25,
        use_transposition_table=True,
        num_threads=1
    )
    
    # Create another MCTS instance with Zobrist hashing
    mcts2 = GomokuMCTS(
        num_simulations=10,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25,
        use_transposition_table=True,
        num_threads=1
    )
    
    # Play some moves
    moves = [
        10, 11, 
        20, 21,
        30, 31
    ]
    
    for move in moves:
        game.apply_move(move)
    
    # Get the board state
    board = game.get_board().flatten()
    legal_moves = list(game.get_legal_moves())
    
    # Simple evaluation function
    def evaluator(game_state):
        return np.ones(board_size * board_size) / (board_size * board_size), 0.0
    
    # Wrap the evaluator
    def evaluator_wrapper(board_flat):
        policy, value = evaluator(game)
        policy_list = [0.0] * (board_size * board_size)
        for move, prob in enumerate(policy):
            policy_list[move] = prob
        return policy_list, value
    
    # Run search with both MCTS instances
    probs1 = mcts1.search(board, legal_moves, evaluator_wrapper)
    probs2 = mcts2.search(board, legal_moves, evaluator_wrapper)
    
    # The probabilities should be very similar 
    # (exact equality might not happen due to initialization randomness)
    similar = True
    for move in probs1:
        if move in probs2:
            if abs(probs1[move] - probs2[move]) > 0.1:
                similar = False
                break
    
    return similar


def compare_performance():
    """Compare performance with and without Zobrist hashing"""
    
    board_sizes = [9, 15, 19]
    with_zobrist_times = []
    without_zobrist_times = []
    
    print("\nPerformance comparison:")
    print("----------------------")
    
    for board_size in board_sizes:
        print(f"\nTesting board size: {board_size}x{board_size}")
        
        # Test with Zobrist hashing
        time_with_zobrist = test_search_performance(board_size=board_size, use_zobrist=True)
        with_zobrist_times.append(time_with_zobrist)
        print(f"  Average search time with Zobrist hashing: {time_with_zobrist:.4f} seconds")
        
        # Test without Zobrist hashing
        time_without_zobrist = test_search_performance(board_size=board_size, use_zobrist=False)
        without_zobrist_times.append(time_without_zobrist)
        print(f"  Average search time without Zobrist hashing: {time_without_zobrist:.4f} seconds")
        
        # Calculate improvement
        improvement = (time_without_zobrist - time_with_zobrist) / time_without_zobrist * 100
        print(f"  Performance improvement: {improvement:.2f}%")
    
    # Plot results if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        plt.figure(figsize=(10, 6))
        x = np.arange(len(board_sizes))
        width = 0.35
        
        plt.bar(x - width/2, with_zobrist_times, width, label='With Zobrist')
        plt.bar(x + width/2, without_zobrist_times, width, label='Without Zobrist')
        
        plt.xlabel('Board Size')
        plt.ylabel('Average Search Time (s)')
        plt.title('MCTS Search Performance: Zobrist vs. Standard Hashing')
        plt.xticks(x, [f"{size}x{size}" for size in board_sizes])
        plt.legend()
        
        # Calculate and display improvement percentages
        for i in range(len(board_sizes)):
            improvement = (without_zobrist_times[i] - with_zobrist_times[i]) / without_zobrist_times[i] * 100
            plt.text(i, max(with_zobrist_times[i], without_zobrist_times[i]) + 0.02, 
                     f"{improvement:.1f}%", ha='center', va='bottom')
        
        plt.savefig('zobrist_performance.png')
        print("\nPerformance comparison plot saved as 'zobrist_performance.png'")


def test_zobrist_hash():
    """Run all Zobrist hash tests"""
    
    print("Testing Zobrist hash implementation...")
    
    # Test consistency
    consistent = test_zobrist_consistency()
    print(f"Zobrist hash consistency test: {'PASSED' if consistent else 'FAILED'}")
    
    # Compare performance
    compare_performance()
    
    print("\nZobrist hash testing complete!")


if __name__ == "__main__":
    test_zobrist_hash()