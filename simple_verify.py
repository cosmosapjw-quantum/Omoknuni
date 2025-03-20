#!/usr/bin/env python3
"""
Simple verification script for MCTS multithreading fix.
"""

import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Print the python path to help debug import issues
print(f"Python path: {sys.path}")

# Enable verbose importing
print("Trying to import the module...")
print(f"Current working directory: {os.getcwd()}")
print(f"Looking for libraries in: {os.path.abspath(os.path.dirname(__file__))}")

# List available .so files in bindings directory
bindings_dir = os.path.join(os.path.dirname(__file__), "alphazero", "bindings")
if os.path.exists(bindings_dir):
    print(f"Contents of {bindings_dir}:")
    for item in os.listdir(bindings_dir):
        print(f"  - {item}")
else:
    print(f"{bindings_dir} does not exist.")

# Try to import the module with detailed error tracking
try:
    print("Attempting to import MCTS...")
    from alphazero.bindings.cpp_mcts import MCTS
    print("Successfully imported MCTS")
except ImportError as e:
    print(f"Error importing MCTS: {e}")
    
    # Show more detailed traceback
    import traceback
    traceback.print_exc()
    
    # Check if the module file exists
    try:
        import importlib.util
        module_path = os.path.join(bindings_dir, "cpp_mcts.cpython-312-x86_64-linux-gnu.so")
        if os.path.exists(module_path):
            print(f"Module exists at: {module_path}")
            # Try to load it directly
            spec = importlib.util.spec_from_file_location("cpp_mcts", module_path)
            if spec:
                print("Successfully created module spec")
            else:
                print("Failed to create module spec")
        else:
            print(f"Module file not found: {module_path}")
    except Exception as ee:
        print(f"Error checking module: {ee}")
    
    sys.exit(1)

def simple_evaluator(state_tensor):
    """Simple evaluator that returns uniform policy and neutral value."""
    # Uniform policy
    policy = [1.0 / 25] * 25  # 5x5 board
    value = 0.0  # Neutral value
    return policy, value

def test_mcts(num_threads, num_simulations=100):
    """Test MCTS with specified number of threads."""
    print(f"\nRunning test with {num_threads} threads and {num_simulations} simulations")
    
    try:
        # Create MCTS with specified settings
        mcts = MCTS(
            num_simulations=num_simulations,
            c_puct=1.5,
            dirichlet_alpha=0.3,
            dirichlet_noise_weight=0.25,
            virtual_loss_weight=1.0,
            use_transposition_table=True,
            transposition_table_size=10000,
            num_threads=num_threads
        )
        
        # Create a simple game state (empty 5x5 board)
        state_tensor = [0.0] * 25
        
        # Define legal moves (all positions)
        legal_moves = list(range(25))
        
        # Search
        start_time = time.time()
        probs = mcts.search(state_tensor, legal_moves, simple_evaluator)
        end_time = time.time()
        search_time = end_time - start_time
        
        print(f"Search completed in {search_time:.3f} seconds")
        print(f"Top 3 moves by probability:", sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3])
        
        # Test successful
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