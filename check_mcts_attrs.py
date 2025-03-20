#!/usr/bin/env python3
"""
Check available attributes of the MCTS class.
"""

import sys
from alphazero.bindings.cpp_mcts import MCTS

def main():
    print("Available attributes and methods in MCTS class:")
    for attr in dir(MCTS):
        if not attr.startswith('__'):
            print(f"  - {attr}")
    
    # Create an instance and check its attributes
    print("\nCreating MCTS instance...")
    mcts = MCTS(
        num_simulations=100,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25,
        virtual_loss_weight=1.0,
        use_transposition_table=True,
        transposition_table_size=10000,
        num_threads=1
    )
    
    # Let's check the instance attributes
    print("\nInstance attributes:")
    for attr in dir(mcts):
        if not attr.startswith('__'):
            print(f"  - {attr}")

if __name__ == "__main__":
    main()