#!/usr/bin/env python3
"""
Script to play Gomoku against a trained AlphaZero model.
"""

import sys
import os
import argparse
import torch
import time
import re

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


def human_player(game):
    """
    Get a move from the human player.
    
    Args:
        game: Game state
        
    Returns:
        The selected move, or -1 if no valid move
    """
    board_size = game.board_size
    valid_moves = game.get_legal_moves()
    
    if not valid_moves:
        return -1
    
    while True:
        try:
            move_input = input("Enter your move (row,col or 'quit'): ")
            
            if move_input.lower() == 'quit':
                print("Exiting game.")
                sys.exit(0)
            
            # Try to parse as row,col
            match = re.match(r'(\d+)[,\s]+(\d+)', move_input)
            if match:
                row = int(match.group(1))
                col = int(match.group(2))
                
                # Check if within bounds
                if not (0 <= row < board_size and 0 <= col < board_size):
                    print(f"Invalid coordinates. Must be between 0 and {board_size-1}.")
                    continue
                
                move = row * board_size + col
                
                # Check if move is valid
                if move not in valid_moves:
                    print("Invalid move. That position is already occupied or not allowed.")
                    continue
                
                return move
            else:
                print("Invalid input format. Use 'row,col' (e.g., '7,8').")
        
        except ValueError:
            print("Invalid input. Please enter numbers for row and column.")
        except Exception as e:
            print(f"Error: {str(e)}")


def ai_player(game, network, mcts_simulations=800, c_puct=1.5, temperature=0.1, dirichlet_noise=False):
    """
    Get a move from the AI player using MCTS with the provided network.
    
    Args:
        game: Game state
        network: Neural network for evaluation
        mcts_simulations: Number of MCTS simulations
        c_puct: Exploration constant
        temperature: Temperature for move selection
        dirichlet_noise: Whether to add Dirichlet noise to the root
        
    Returns:
        The selected move, or -1 if no valid move
    """
    valid_moves = game.get_legal_moves()
    
    if not valid_moves:
        return -1
    
    mcts = MCTS(
        game=game,
        evaluator=network.predict,
        c_puct=c_puct,
        num_simulations=mcts_simulations,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25 if dirichlet_noise else 0.0,
        temperature=temperature
    )
    
    # Get move probabilities
    move, probs = mcts.select_move(return_probs=True)
    
    # Print top 5 moves considered by the AI
    top_moves = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]
    print("AI top moves:")
    for m, p in top_moves:
        row, col = divmod(m, game.board_size)
        print(f"  ({row}, {col}): {p:.3f}")
    
    return move


def parse_args():
    parser = argparse.ArgumentParser(description="Play Gomoku against AlphaZero")
    
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    
    parser.add_argument("--board-size", type=int, default=15,
                        help="Size of the Gomoku board (default: 15)")
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
    
    parser.add_argument("--mcts-simulations", type=int, default=1600,
                        help="Number of MCTS simulations per AI move (default: 1600)")
    parser.add_argument("--ai-first", action="store_true",
                        help="AI plays first (black)")
    
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
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            network.load_state_dict(checkpoint["model_state_dict"])
        else:
            network.load_state_dict(checkpoint)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Using untrained network instead.")
    
    # Create game
    game = GomokuGame(
        board_size=args.board_size,
        use_renju=args.use_renju,
        use_omok=args.use_omok,
        use_pro_long_opening=args.use_pro_long_opening
    )
    
    # Set up players
    if args.ai_first:
        player1 = lambda g: ai_player(g, network, args.mcts_simulations)
        player2 = human_player
        print("AI plays as Black (first), you play as White (second)")
    else:
        player1 = human_player
        player2 = lambda g: ai_player(g, network, args.mcts_simulations)
        print("You play as Black (first), AI plays as White (second)")
    
    # Print rules
    print("\n=== Game Rules ===")
    if args.use_renju:
        print("Using Renju rules: Black has restrictions on overlines and forbidden moves")
    elif args.use_omok:
        print("Using Omok rules: Other specific restrictions apply")
    else:
        print("Using standard Gomoku rules: Five-in-a-row wins")
    
    if args.use_pro_long_opening:
        print("Using professional long opening rules")
    print()
    
    # Print initial board
    board = game.state.get_board()
    print("Initial board:")
    print_board(board)
    print()
    
    # Main game loop
    while not game.is_terminal():
        # Get current player
        current_player = game.get_current_player()
        player_name = "Black" if current_player == 1 else "White"
        player_symbol = "●" if current_player == 1 else "○"
        
        print(f"{player_name}'s turn ({player_symbol})")
        
        # Get move from the appropriate player
        start_time = time.time()
        if current_player == 1:  # Black
            move = player1(game)
        else:  # White
            move = player2(game)
        elapsed_time = time.time() - start_time
        
        if move == -1:
            print(f"{player_name} has no valid moves.")
            break
        
        # Get move coordinates
        row, col = divmod(move, game.board_size)
        print(f"{player_name} plays at ({row}, {col}) - took {elapsed_time:.2f}s")
        
        # Apply the move
        game.apply_move(move)
        
        # Print the updated board
        board = game.state.get_board()
        print_board(board)
        print()
    
    # Game over
    winner = game.get_winner()
    
    if winner == 0:
        print("Game ended in a draw.")
    elif winner == 1:
        print("Black wins!")
    else:
        print("White wins!")


if __name__ == "__main__":
    main()