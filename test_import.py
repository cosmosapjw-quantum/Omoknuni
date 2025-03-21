#!/usr/bin/env python3
"""
Test importing the batched module.
"""

try:
    from alphazero.bindings.batched_cpp_mcts import BatchedGomokuMCTS
    print('BatchedGomokuMCTS imported successfully')
except ImportError as e:
    print(f'Failed to import BatchedGomokuMCTS: {e}')

try:
    from alphazero.bindings.improved_cpp_mcts import GomokuMCTS
    print('GomokuMCTS from improved_cpp_mcts imported successfully')
except ImportError as e:
    print(f'Failed to import GomokuMCTS from improved_cpp_mcts: {e}')

try:
    from alphazero.bindings.cpp_mcts import GomokuMCTS
    print('GomokuMCTS from cpp_mcts imported successfully')
except ImportError as e:
    print(f'Failed to import GomokuMCTS from cpp_mcts: {e}')