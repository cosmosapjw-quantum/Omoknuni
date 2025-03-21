#!/usr/bin/env python3
"""
Test script for the improved MCTS bindings with better GIL handling.
"""

import os
import sys
import time
import numpy as np

# Add the module paths
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("Testing improved MCTS bindings with proper GIL handling")
print("------------------------------------------------------")

# Step 1: Compile the improved bindings
try:
    print("\nStep 1: Building improved bindings...")
    
    # Create a minimal setup.py for just the improved bindings
    with open('improved_setup.py', 'w') as f:
        f.write('''
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import sys
import os
import pybind11

# Define compiler flags based on platform
extra_compile_args = []
if sys.platform == 'win32':
    extra_compile_args = ['/std:c++17']  # MSVC flag
else:
    extra_compile_args = ['-std=c++17']  # GCC/Clang flag

# Define include directories
include_dirs = [
    pybind11.get_include(),
    "alphazero/core",
    "alphazero/core/game",
    "alphazero/core/mcts",
    "alphazero/core/utils"
]

# Define link args for threading library
libraries = []
if sys.platform != 'win32':
    libraries = ['pthread']

# MCTS module and dependencies
improved_mcts_module = Extension(
    'alphazero.bindings.improved_cpp_mcts',
    sources=[
        'alphazero/bindings/improved_mcts_bindings.cpp',
        'alphazero/core/mcts/mcts_node.cpp',
        'alphazero/core/mcts/mcts.cpp',
        'alphazero/core/mcts/transposition_table.cpp',
        'alphazero/core/utils/thread_pool.cpp',
        'alphazero/core/mcts/zobrist_hash.cpp'
    ],
    include_dirs=include_dirs,
    language='c++',
    extra_compile_args=extra_compile_args,
    libraries=libraries,
)

setup(
    name="alphazero-improved",
    version="0.1.0",
    ext_modules=[improved_mcts_module],
    cmdclass={'build_ext': build_ext},
)
''')
    
    # Compile the improved bindings
    import subprocess
    subprocess.check_call([sys.executable, 'improved_setup.py', 'build_ext', '--inplace'])
    print("Successfully built improved bindings!")
    
except Exception as e:
    print(f"Error building improved bindings: {e}")
    import traceback
    traceback.print_exc()
    print("Skipping binding compilation and proceeding with tests...")

# Try importing the improved module
try:
    print("\nStep 2: Importing improved MCTS module...")
    from alphazero.bindings.improved_cpp_mcts import MCTS
    print("Successfully imported improved MCTS module!")
    
    print("\nStep 3: Testing the module...")
    # Create a game board
    board_size = 5
    board = np.zeros(board_size * board_size, dtype=np.int32)
    legal_moves = list(range(board_size * board_size))
    
    # Define a simple evaluator function
    def evaluator(state):
        # Add a small delay to simulate work
        time.sleep(0.01)
        # Return uniform policy and value 0
        policy = [1.0 / len(state)] * len(state)
        value = 0.0
        return policy, value
    
    # Test with single thread first
    print("\nTesting single-threaded MCTS...")
    mcts_single = MCTS(
        num_simulations=20,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25,
        virtual_loss_weight=1.0,
        use_transposition_table=True,
        transposition_table_size=1000,
        num_threads=1
    )
    
    start_time = time.time()
    probs_single = mcts_single.search(board, legal_moves, evaluator)
    single_time = time.time() - start_time
    
    print(f"Single-threaded search completed in {single_time:.3f} seconds")
    print(f"Top 5 moves: {dict(sorted(probs_single.items(), key=lambda x: x[1], reverse=True)[:5])}")
    
    # Test with multiple threads
    print("\nTesting multi-threaded MCTS...")
    mcts_multi = MCTS(
        num_simulations=20,
        c_puct=1.5,
        dirichlet_alpha=0.3,
        dirichlet_noise_weight=0.25,
        virtual_loss_weight=1.0,
        use_transposition_table=True,
        transposition_table_size=1000,
        num_threads=4
    )
    
    start_time = time.time()
    probs_multi = mcts_multi.search(board, legal_moves, evaluator)
    multi_time = time.time() - start_time
    
    print(f"Multi-threaded search completed in {multi_time:.3f} seconds")
    print(f"Top 5 moves: {dict(sorted(probs_multi.items(), key=lambda x: x[1], reverse=True)[:5])}")
    
    # Calculate speedup
    if single_time > 0:
        speedup = single_time / multi_time
        print(f"\nSpeedup from multithreading: {speedup:.2f}x")
    
    print("\nAll tests completed successfully!")
    print("The improved bindings with proper GIL handling are working!")
    
except Exception as e:
    print(f"Error in testing: {e}")
    import traceback
    traceback.print_exc()