#!/usr/bin/env python3
"""
Check if the project is properly set up by trying to import the modules.
"""

import sys
import os

# Ensure the package is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    print("Checking project setup...")
    
    try:
        print("Trying to import Gomoku game wrapper...")
        from alphazero.python.games.gomoku import GomokuGame
        print("✓ Successfully imported Gomoku game wrapper")
    except ImportError as e:
        print(f"✗ Failed to import Gomoku game wrapper: {e}")
    
    try:
        print("Trying to import C++ modules directly...")
        
        try:
            from alphazero.core.gomoku import Gamestate
            print("✓ Successfully imported Gamestate from C++")
        except ImportError as e:
            print(f"✗ Failed to import Gamestate: {e}")
        
        try:
            from alphazero.core.attack_defense import AttackDefenseModule
            print("✓ Successfully imported AttackDefenseModule from C++")
        except ImportError as e:
            print(f"✗ Failed to import AttackDefenseModule: {e}")
    except Exception as e:
        print(f"✗ An error occurred while importing C++ modules: {e}")
    
    print("\nSetup check complete.")


if __name__ == "__main__":
    main()