# Create file: alphazero/examples/parallel_batch_test.py

import sys
import os
import time
import torch
import multiprocessing as mp

# Add package to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alphazero.python.games.gomoku import GomokuGame
from alphazero.python.models.simple_conv_net import SimpleConvNet
from alphazero.python.mcts.parallel_batch_mcts import ParallelBatchedMCTS
from alphazero.python.models.network_base import BaseNetwork

def print_board(board):
    """Print the board in a readable format."""
    symbols = {0: ".", 1: "●", 2: "○"}
    board_size = len(board)
    
    print("   ", end="")
    for i in range(board_size):
        print(f"{i:2d}", end=" ")
    print()
    
    for i in range(board_size):
        print(f"{i:2d} ", end="")
        for j in range(board_size):
            print(f" {symbols[board[i][j]]}", end=" ")
        print()

def test_parallel_batch():
    """Test the parallel batched MCTS implementation."""
    # Print multiprocessing method and available CPU cores
    print(f"Multiprocessing start method: {mp.get_start_method()}")
    print(f"Number of CPU cores: {mp.cpu_count()}")
    
    # Create game
    board_size = 9
    game = GomokuGame(board_size=board_size)
    
    # Create neural network
    print("Creating neural network...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = SimpleConvNet(
        board_size=board_size,
        input_channels=3,
        num_filters=32,
        num_residual_blocks=3
    ).to(device)
    network.eval()
    
    # Wrapper network to use with predict method
    class NetworkWrapper(BaseNetwork):
        def __init__(self, network):
            super().__init__()
            self.network = network
    
    wrapped_network = NetworkWrapper(network)
    wrapped_network.predict = lambda game_state: network.predict(game_state)
    
    # Test different numbers of workers
    worker_options = [4, 8, 12]
    
    for num_workers in worker_options:
        print(f"\n{'='*40}")
        print(f"Testing with {num_workers} worker(s)")
        print(f"{'='*40}")
        
        # Create parallel batched MCTS
        print("Creating parallel batched MCTS...")
        mcts = ParallelBatchedMCTS(
            game=game,
            evaluator=wrapped_network,
            num_workers=num_workers,
            num_simulations=50  # Small number for quick testing
        )
        
        # Initial board
        print("\nInitial board:")
        board = game.get_board()
        print_board(board)
        
        # Play a few moves
        for i in range(5):
            print(f"\nMove {i+1}:")
            
            # Get current player
            player = "Black" if game.get_current_player() == 1 else "White"
            print(f"Current player: {player}")
            
            # Select move
            start_time = time.time()
            move, probs = mcts.select_move(return_probs=True)
            elapsed = time.time() - start_time
            
            # Convert move to coordinates
            row, col = divmod(move, board_size)
            print(f"Selected move: ({row}, {col}) in {elapsed:.2f} seconds")
            
            # Show top moves
            top_moves = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            print("Top moves considered:")
            for m, p in top_moves:
                r, c = divmod(m, board_size)
                print(f"  ({r}, {c}): {p:.3f}")
            
            # Apply move
            game.apply_move(move)
            mcts.update_with_move(move)
            
            # Show updated board
            board = game.get_board()
            print("\nBoard after move:")
            print_board(board)
        
        print(f"\nTest with {num_workers} worker(s) completed successfully!")

if __name__ == "__main__":
    test_parallel_batch()
    # Run the test