# Create file: alphazero/examples/test_optimized_self_play.py

import sys
import os
import time
import torch
import numpy as np
from tqdm import tqdm

# Add package to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alphazero.python.games.gomoku import GomokuGame
from alphazero.python.models.simple_conv_net import SimpleConvNet
from alphazero.python.mcts.mcts import MCTS
from alphazero.python.mcts.optimized_mcts_wrapper import OptimizedMCTSWrapper, ParallelOptimizedMCTSWrapper

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

def play_self_play_game(mcts_type, num_moves=10, num_simulations=100, board_size=9, 
                        use_parallel=False, print_boards=True):
    """
    Play a self-play game using the specified MCTS implementation.
    
    Args:
        mcts_type: 'standard' or 'optimized'
        num_moves: Maximum number of moves to play
        num_simulations: Number of MCTS simulations per move
        board_size: Size of the board
        use_parallel: Whether to use the parallel version of optimized MCTS
        print_boards: Whether to print board state after each move
        
    Returns:
        Dictionary of game statistics
    """
    # Create game and neural network
    game = GomokuGame(board_size=board_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = SimpleConvNet(
        board_size=board_size,
        input_channels=3,
        num_filters=32,
        num_residual_blocks=3
    ).to(device)
    network.eval()  # Set to evaluation mode
    
    # Create MCTS
    if mcts_type == 'standard':
        mcts = MCTS(
            game=game,
            evaluator=network.predict,
            c_puct=1.5,
            num_simulations=num_simulations,
            dirichlet_alpha=0.3,
            dirichlet_noise_weight=0.25,
            temperature=1.0
        )
    else:  # optimized
        if use_parallel:
            mcts = ParallelOptimizedMCTSWrapper(
                game=game,
                evaluator=network.predict,
                c_puct=1.5,
                num_simulations=num_simulations,
                dirichlet_alpha=0.3,
                dirichlet_noise_weight=0.25,
                temperature=1.0,
                num_threads=4,
                batch_size=16
            )
        else:
            mcts = OptimizedMCTSWrapper(
                game=game,
                evaluator=network.predict,
                c_puct=1.5,
                num_simulations=num_simulations,
                dirichlet_alpha=0.3,
                dirichlet_noise_weight=0.25,
                temperature=1.0,
                num_threads=4,
                batch_size=16
            )
    
    # Initial board
    if print_boards:
        print(f"\nInitial board ({mcts_type} MCTS):")
        board = game.get_board()
        print_board(board)
    
    # Play moves
    move_times = []
    moves_played = 0
    
    for i in range(num_moves):
        if game.is_terminal():
            break
        
        # Get player
        player = "Black" if game.get_current_player() == 1 else "White"
        
        # Select move with timing
        start_time = time.time()
        move, probs = mcts.select_move(return_probs=True)
        move_time = time.time() - start_time
        move_times.append(move_time)
        
        # Apply move
        row, col = divmod(move, board_size)
        
        if print_boards:
            print(f"\nMove {i+1}: {player} plays at ({row}, {col}) - took {move_time:.3f}s")
            
            # Print top 3 move probabilities
            top_moves = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            print("Top moves considered:")
            for m, p in top_moves:
                r, c = divmod(m, board_size)
                print(f"  ({r}, {c}): {p:.3f}")
        
        game.apply_move(move)
        moves_played += 1
        
        # Update MCTS tree
        mcts.update_with_move(move)
        
        if print_boards:
            board = game.get_board()
            print_board(board)
    
    # Clean up resources
    if mcts_type == 'optimized' and use_parallel:
        mcts.close()
    
    # Print final state
    if print_boards:
        if game.is_terminal():
            winner = game.get_winner()
            if winner == 0:
                print("\nGame ended in a draw.")
            else:
                print(f"\nPlayer {winner} wins!")
        else:
            print("\nReached move limit.")
    
    return {
        "mcts_type": mcts_type,
        "moves_played": moves_played,
        "avg_move_time": sum(move_times) / max(len(move_times), 1),
        "total_time": sum(move_times),
        "simulations_per_second": (moves_played * num_simulations) / max(sum(move_times), 0.001)
    }

def compare_mcts_implementations():
    """Compare standard and optimized MCTS implementations."""
    # Settings
    num_moves = 5
    num_simulations = 200
    board_size = 9
    
    print("Comparing MCTS implementations:")
    print(f"- Board size: {board_size}x{board_size}")
    print(f"- Simulations per move: {num_simulations}")
    print(f"- Moves per game: {num_moves}")
    
    # Standard MCTS
    print("\nTesting standard MCTS...")
    standard_stats = play_self_play_game(
        mcts_type='standard',
        num_moves=num_moves,
        num_simulations=num_simulations,
        board_size=board_size,
        print_boards=False
    )
    
    # Optimized MCTS
    print("\nTesting optimized MCTS...")
    optimized_stats = play_self_play_game(
        mcts_type='optimized',
        num_moves=num_moves,
        num_simulations=num_simulations,
        board_size=board_size,
        print_boards=False
    )
    
    # Parallel optimized MCTS
    print("\nTesting parallel optimized MCTS...")
    parallel_stats = play_self_play_game(
        mcts_type='optimized',
        num_moves=num_moves,
        num_simulations=num_simulations,
        board_size=board_size,
        use_parallel=True,
        print_boards=False
    )
    
    # Print comparison
    print("\n=== Performance Comparison ===")
    print(f"Standard MCTS: {standard_stats['avg_move_time']:.3f}s per move, "
          f"{standard_stats['simulations_per_second']:.1f} sim/s")
    
    print(f"Optimized MCTS: {optimized_stats['avg_move_time']:.3f}s per move, "
          f"{optimized_stats['simulations_per_second']:.1f} sim/s")
    
    print(f"Parallel Optimized MCTS: {parallel_stats['avg_move_time']:.3f}s per move, "
          f"{parallel_stats['simulations_per_second']:.1f} sim/s")
    
    # Calculate speedups
    std_sps = standard_stats['simulations_per_second']
    opt_speedup = optimized_stats['simulations_per_second'] / std_sps
    par_speedup = parallel_stats['simulations_per_second'] / std_sps
    
    print("\n=== Speedup vs Standard MCTS ===")
    print(f"Optimized MCTS: {opt_speedup:.2f}x")
    print(f"Parallel Optimized MCTS: {par_speedup:.2f}x")
    
    return standard_stats, optimized_stats, parallel_stats

def play_interactive_game():
    """Play an interactive game against the optimized MCTS AI."""
    # Settings
    board_size = 9
    num_simulations = 400
    
    # Create game and neural network
    game = GomokuGame(board_size=board_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = SimpleConvNet(
        board_size=board_size,
        input_channels=3,
        num_filters=32,
        num_residual_blocks=3
    ).to(device)
    network.eval()
    
    # Create MCTS
    mcts = OptimizedMCTSWrapper(
        game=game,
        evaluator=network.predict,
        c_puct=1.5,
        num_simulations=num_simulations,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25,
        temperature=0.5,  # Lower temperature for more deterministic play
        num_threads=4,
        batch_size=16
    )
    
    # Initial board
    print("\nWelcome to Gomoku! You'll play as Black (●), AI as White (○).")
    print("Enter moves as 'row,col' (e.g., '3,4')")
    board = game.get_board()
    print_board(board)
    
    # Game loop
    while not game.is_terminal():
        current_player = game.get_current_player()
        
        if current_player == 1:  # Human (Black)
            legal_moves = game.get_legal_moves()
            
            if not legal_moves:
                print("No legal moves available. Game over.")
                break
            
            while True:
                try:
                    move_input = input("\nYour move (row,col): ")
                    
                    if move_input.lower() in ('q', 'quit', 'exit'):
                        print("Quitting game.")
                        return
                    
                    parts = move_input.split(',')
                    if len(parts) != 2:
                        print("Invalid input. Use format 'row,col' (e.g., '3,4')")
                        continue
                    
                    row = int(parts[0].strip())
                    col = int(parts[1].strip())
                    
                    if not (0 <= row < board_size and 0 <= col < board_size):
                        print(f"Position out of bounds. Must be between 0 and {board_size-1}")
                        continue
                    
                    move = row * board_size + col
                    
                    if move not in legal_moves:
                        print("Invalid move. Position is occupied or not allowed.")
                        continue
                    
                    # Valid move
                    break
                    
                except ValueError:
                    print("Invalid input. Please enter numbers.")
            
            # Apply human move
            game.apply_move(move)
            mcts.update_with_move(move)
            
        else:  # AI (White)
            print("\nAI is thinking...")
            
            # Select move with timing
            start_time = time.time()
            move = mcts.select_move()
            move_time = time.time() - start_time
            
            # Apply AI move
            row, col = divmod(move, board_size)
            print(f"AI plays at ({row}, {col}) - took {move_time:.1f}s")
            
            game.apply_move(move)
            mcts.update_with_move(move)
        
        # Print updated board
        board = game.get_board()
        print_board(board)
    
    # Game over
    winner = game.get_winner()
    if winner == 0:
        print("\nGame ended in a draw.")
    elif winner == 1:
        print("\nYou win! Congratulations!")
    else:
        print("\nAI wins! Better luck next time.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test optimized MCTS self-play")
    parser.add_argument("--mode", choices=["self", "compare", "interactive"], default="self",
                        help="Test mode: self-play, comparison, or interactive game")
    parser.add_argument("--use-parallel", action="store_true",
                        help="Use parallel version of optimized MCTS")
    parser.add_argument("--num-moves", type=int, default=5,
                        help="Number of moves to play in self-play mode")
    parser.add_argument("--simulations", type=int, default=200,
                        help="Number of MCTS simulations per move")
    parser.add_argument("--board-size", type=int, default=9,
                        help="Board size for the game")
    
    args = parser.parse_args()
    
    if args.mode == "self":
        # Test single self-play game with full output
        play_self_play_game(
            mcts_type='optimized',
            num_moves=args.num_moves,
            num_simulations=args.simulations,
            board_size=args.board_size,
            use_parallel=args.use_parallel,
            print_boards=True
        )
    elif args.mode == "compare":
        # Compare different MCTS implementations
        compare_mcts_implementations()
    elif args.mode == "interactive":
        # Play an interactive game against the AI
        play_interactive_game()