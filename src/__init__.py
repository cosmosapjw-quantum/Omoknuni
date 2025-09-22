"""
Omoknuni - High-Performance AlphaZero Engine for board games
"""

try:
    # Import the C++ extension module (correct path from scikit-build-core)
    from .alphazero_py import GomokuState, ChessState, GoState
    from . import alphazero_py

    __all__ = ['alphazero_py', 'GomokuState', 'ChessState', 'GoState']

except ImportError as e:
    raise ImportError(f"Cannot import alphazero_py module. Build may be required: {e}")

__version__ = "0.1.0"