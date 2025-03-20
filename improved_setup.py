
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
