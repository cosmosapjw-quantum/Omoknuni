#!/usr/bin/env python3
"""
A simple example to test the Gomoku wrapper.
"""

import sys
import os
import random

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from alphazero.python.games.gomoku import GomokuGame

def main():
    # Create a Gomoku game with default settings
    game = GomokuGame(board_size=15, use_renju=False, use_omok=False)
    
    # Play a random game
    while not game.is_terminal():
        # Get legal moves
        legal_moves = game.get_legal_moves()
        
        if not legal_moves:
            print("No legal moves available.")
            break
        
        # Select a random move
        move = random.choice(legal_moves)
        
        # Get current player
        current_player = game.get_current_player()
        player_name = "Black" if current_player == 1 else "White"
        
        # Get move coordinates
        row, col = divmod(move, game.board_size)
        
        print(f"{player_name} plays at ({row}, {col}) - move {move}")
        
        # Apply the move
        game.apply_move(move)
        
        # Print the board
        game.print_board()
        print()
    
    # Game over, print the result
    winner = game.get_winner()
    if winner == 0:
        print("Game ended in a draw.")
    elif winner == 1:
        print("Black wins!")
    else:
        print("White wins!")


if __name__ == "__main__":
    main()