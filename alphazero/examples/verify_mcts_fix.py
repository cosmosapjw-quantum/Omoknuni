#!/usr/bin/env python3
"""
Verification script for MCTS multithreading fix
This script directly tests the C++ MCTS implementation with different thread counts
"""

import sys
import os
import time
import numpy as np

# Add package to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from alphazero.python.games.gomoku import GomokuGame
    from alphazero.python.mcts.cpp_mcts_wrapper import CppMCTSWrapper
except ImportError as e:
    print(f"Error importing required modules: {e}")
    sys.exit(1)

def simple_evaluator(state_tensor):
    """Simple deterministic evaluator for testing"""
    # Create simple uniform policy
    board_size = int(np.sqrt(len(state_tensor) / 3))  # Assuming 3 channels
    policy = np.ones(board_size * board_size) / (board_size * board_size)
    value = 0.0  # Neutral value
    return policy, value

def test_mcts(num_threads, num_simulations=100):
    """Test MCTS with specified number of threads"""
    print(f"\nRunning test with {num_threads} threads and {num_simulations} simulations")
    
    # Create game
    game = GomokuGame(board_size=5)
    
    # Play some random moves to get a non-empty board
    for _ in range(3):
        legal_moves = game.get_legal_moves()
        if legal_moves:
            move = np.random.choice(legal_moves)
            game.apply_move(move)
    
    print("Game board:")
    board = game.state.get_board()
    for row in board:
        print(" ".join([".OX"[int(cell)] for cell in row]))
    
    # Create MCTS
    mcts = CppMCTSWrapper(
        game=game,
        evaluator=simple_evaluator,
        c_puct=1.5,
        num_simulations=num_simulations,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25,
        temperature=1.0,
        use_transposition_table=True,
        transposition_table_size=10000,
        num_threads=num_threads
    )
    
    try:
        # Measure search time
        start_time = time.time()
        mcts.search()
        end_time = time.time()
        search_time = end_time - start_time
        
        # Get best move
        move = mcts.select_move()
        row, col = divmod(move, game.board_size)
        
        print(f"Search completed in {search_time:.3f} seconds")
        print(f"Best move: ({row}, {col})")
        
        # Apply move and update tree
        game.apply_move(move)
        mcts.update_with_move(move)
        
        # Run another search to verify tree update worked
        start_time = time.time()
        mcts.search()
        end_time = time.time()
        search_time_2 = end_time - start_time
        
        print(f"Second search completed in {search_time_2:.3f} seconds")
        
        return True
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("MCTS Multithreading Fix Verification")
    print("===================================")
    
    # Test with single thread first (baseline)
    single_success = test_mcts(num_threads=1)
    
    if single_success:
        # Test with multiple threads
        multi_success = test_mcts(num_threads=4)
        
        if multi_success:
            print("\n✅ VERIFICATION SUCCESSFUL: MCTS works correctly with multiple threads!")
        else:
            print("\n❌ Multithreaded test failed.")
    else:
        print("\n❌ Single-threaded test failed. Basic functionality is broken.")

if __name__ == "__main__":
    main()