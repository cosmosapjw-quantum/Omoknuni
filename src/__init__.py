"""
Omoknuni - High-Performance AlphaZero Engine for board games
"""

try:
    # Try to import the C++ extension module
    # This will work when the module is installed properly
    from .alphazero_py import GomokuState, ChessState, GoState
    from . import alphazero_py

    __all__ = ['alphazero_py', 'GomokuState', 'ChessState', 'GoState']

except ImportError as e:
    # If the C++ module is not available as a submodule, that's okay
    # Scripts can still try to import it directly from the build directory
    __all__ = []

__version__ = "0.1.0"