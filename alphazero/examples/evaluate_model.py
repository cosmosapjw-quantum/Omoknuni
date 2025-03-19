#!/usr/bin/env python3
"""
Script to evaluate a trained AlphaZero model by playing against itself or a random agent.
"""

import sys
import os
import argparse
import torch
import random
import time
from tqdm import tqdm
import numpy as np

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alphazero.python.games.gomoku import GomokuGame
from alphazero.python.mcts.mcts import MCTS
from alphazero.python.models.simple_conv_net import SimpleConvNet


def print_board(board):
    """Print the board state in a more readable format."""
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


def random_agent(game):
    """Simple random agent for testing."""
    legal_moves = game.get_legal_moves()
    return random.choice(legal_moves) if legal_moves else -1


def mcts_agent(game, network, mcts_simulations=800, c_puct=1.5, temperature=0.1):
    """MCTS agent using the provided network."""
    mcts = MCTS(
        game=game,
        evaluator=network.predict,
        c_puct=c_puct,
        num_simulations=mcts_simulations,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.1,  # Less exploration for evaluation
        temperature=temperature
    )
    
    move = mcts.select_move()
    return move


def play_game(board_size, use_renju, use_omok, use_pro_long_opening, 
              player1, player2, player1_args=None, player2_args=None, 
              display=False, move_limit=None):
    """
    Play a game between two agents.
    
    Args:
        board_size: Size of the Gomoku board
        use_renju: Whether to use Renju rules
        use_omok: Whether to use Omok rules
        use_pro_long_opening: Whether to use professional long opening rules
        player1: Function that takes a game and returns a move
        player2: Function that takes a game and returns a move
        player1_args: Arguments to pass to player1
        player2_args: Arguments to pass to player2
        display: Whether to display the game
        move_limit: Maximum number of moves to play
        
    Returns:
        Winner of the game (0 for draw, 1 for black/player1, 2 for white/player2)
    """
    game = GomokuGame(
        board_size=board_size,
        use_renju=use_renju,
        use_omok=use_omok,
        use_pro_long_opening=use_pro_long_opening
    )
    
    player1_args = player1_args or {}
    player2_args = player2_args or {}
    
    if display:
        print("Initial board:")
        board = game.state.get_board()
        print_board(board)
        print()
    
    move_count = 0
    
    while not game.is_terminal():
        # Check move limit
        if move_limit is not None and move_count >= move_limit:
            if display:
                print("Move limit reached.")
            return 0  # Draw
        
        # Get current player
        current_player = game.get_current_player()
        
        # Get move from the appropriate agent
        start_time = time.time()
        if current_player == 1:  # Black
            move = player1(game, **player1_args)
        else:  # White
            move = player2(game, **player2_args)
        elapsed_time = time.time() - start_time
        
        if move == -1:
            if display:
                print(f"Player {current_player} has no valid moves.")
            return 3 - current_player  # Opponent wins
        
        # Get move coordinates
        row, col = divmod(move, game.board_size)
        
        if display:
            player_name = "Black" if current_player == 1 else "White"
            print(f"{player_name} plays at ({row}, {col}) - move {move} (took {elapsed_time:.2f}s)")
        
        # Apply the move
        game.apply_move(move)
        
        if display:
            board = game.state.get_board()
            print_board(board)
            print()
        
        move_count += 1
    
    # Game over
    winner = game.get_winner()
    
    if display:
        if winner == 0:
            print("Game ended in a draw.")
        elif winner == 1:
            print("Black wins!")
        else:
            print("White wins!")
    
    return winner


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AlphaZero model")
    
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    
    parser.add_argument("--board-size", type=int, default=9,
                        help="Size of the Gomoku board (default: 9)")
    parser.add_argument("--use-renju", action="store_true",
                        help="Use Renju rules")
    parser.add_argument("--use-omok", action="store_true",
                        help="Use Omok rules")
    parser.add_argument("--use-pro-long-opening", action="store_true",
                        help="Use professional long opening rules")
    
    parser.add_argument("--num-filters", type=int, default=64,
                        help="Number of filters in the neural network (default: 64)")
    parser.add_argument("--num-residual-blocks", type=int, default=5,
                        help="Number of residual blocks in the neural network (default: 5)")
    
    parser.add_argument("--mcts-simulations", type=int, default=800,
                        help="Number of MCTS simulations per move (default: 800)")
    parser.add_argument("--opponent", choices=["random", "self"], default="random",
                        help="Opponent type: random or self (default: random)")
    parser.add_argument("--opponent-simulations", type=int, default=200,
                        help="Number of MCTS simulations for opponent if 'self' (default: 200)")
    
    parser.add_argument("--num-games", type=int, default=10,
                        help="Number of evaluation games to play (default: 10)")
    parser.add_argument("--move-limit", type=int, default=None,
                        help="Maximum number of moves per game (default: None)")
    parser.add_argument("--display", action="store_true",
                        help="Display the games")
    
    parser.add_argument("--use-cuda", action="store_true",
                        help="Use CUDA if available")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create neural network
    print("Creating neural network...")
    network = SimpleConvNet(
        board_size=args.board_size,
        input_channels=3,
        num_filters=args.num_filters,
        num_residual_blocks=args.num_residual_blocks
    ).to(device)
    
    # Load model checkpoint
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        network.load_state_dict(checkpoint["model_state_dict"])
    else:
        network.load_state_dict(checkpoint)
    
    # Set up player functions
    player1 = mcts_agent  # AlphaZero agent always plays as Black
    player1_args = {
        "network": network,
        "mcts_simulations": args.mcts_simulations,
        "c_puct": 1.5,
        "temperature": 0.1
    }
    
    if args.opponent == "random":
        player2 = random_agent
        player2_args = {}
        print("Opponent: Random agent")
    else:  # self
        player2 = mcts_agent
        player2_args = {
            "network": network,
            "mcts_simulations": args.opponent_simulations,
            "c_puct": 1.5,
            "temperature": 0.1
        }
        print(f"Opponent: Self-play ({args.opponent_simulations} simulations)")
    
    # Play evaluation games
    print(f"Playing {args.num_games} evaluation games...")
    results = []
    
    for i in tqdm(range(args.num_games)):
        if args.display:
            print(f"\n=== Game {i+1}/{args.num_games} ===\n")
        
        winner = play_game(
            board_size=args.board_size,
            use_renju=args.use_renju,
            use_omok=args.use_omok,
            use_pro_long_opening=args.use_pro_long_opening,
            player1=player1,
            player2=player2,
            player1_args=player1_args,
            player2_args=player2_args,
            display=args.display,
            move_limit=args.move_limit
        )
        
        results.append(winner)
    
    # Calculate statistics
    black_wins = results.count(1)
    white_wins = results.count(2)
    draws = results.count(0)
    
    print("\n=== Evaluation Results ===")
    print(f"Total games: {args.num_games}")
    print(f"Black (AlphaZero) wins: {black_wins} ({black_wins/args.num_games*100:.1f}%)")
    print(f"White (Opponent) wins: {white_wins} ({white_wins/args.num_games*100:.1f}%)")
    print(f"Draws: {draws} ({draws/args.num_games*100:.1f}%)")


if __name__ == "__main__":
    main()