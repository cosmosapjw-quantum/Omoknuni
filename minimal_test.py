#!/usr/bin/env python3
"""
Minimal test script to debug Python module loading issues.
"""

import sys
import os
import traceback

def main():
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if shared libraries exist
    bindings_dir = os.path.join(os.getcwd(), "alphazero", "bindings")
    print(f"\nChecking bindings directory: {bindings_dir}")
    if os.path.exists(bindings_dir):
        print("Directory exists.")
        print("Contents:")
        for f in os.listdir(bindings_dir):
            print(f"  - {f}")
    else:
        print("Directory doesn't exist!")
    
    # Try to load the module
    print("\nTrying to import cpp_mcts module...")
    try:
        import alphazero.bindings.cpp_mcts
        print("SUCCESS: Module imported!")
        print(f"Module contents: {dir(alphazero.bindings.cpp_mcts)}")
    except ImportError as e:
        print(f"FAILED: ImportError - {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()