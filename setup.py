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

# Define the extension modules
gomoku_module = Extension(
    'alphazero.core.gomoku',
    sources=['alphazero/core/game/gomoku.cpp'],
    include_dirs=include_dirs,
    language='c++',
    extra_compile_args=extra_compile_args,
    libraries=libraries,
)

attack_defense_module = Extension(
    'alphazero.core.attack_defense',
    sources=['alphazero/core/game/attack_defense.cpp'],
    include_dirs=include_dirs,
    language='c++',
    extra_compile_args=extra_compile_args,
    libraries=libraries,
)

# MCTS module and dependencies
mcts_module = Extension(
    'alphazero.bindings.cpp_mcts',
    sources=[
        'alphazero/bindings/mcts_bindings.cpp',
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

<<<<<<< HEAD
# New batch evaluator module
batch_evaluator_module = Extension(
    'alphazero.bindings.batch_evaluator',
    sources=[
        'alphazero/bindings/batch_evaluator_bindings.cpp'
=======
# Improved MCTS module with better GIL handling
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

# Batched MCTS module with leaf parallelization
batched_mcts_module = Extension(
    'alphazero.bindings.batched_cpp_mcts',
    sources=[
        'alphazero/bindings/batched_mcts_bindings.cpp',
        'alphazero/core/mcts/mcts_node.cpp',
        'alphazero/core/mcts/mcts.cpp',
        'alphazero/core/mcts/transposition_table.cpp',
        'alphazero/core/mcts/batch_evaluator.cpp',
        'alphazero/core/utils/thread_pool.cpp',
        'alphazero/core/mcts/zobrist_hash.cpp'
>>>>>>> 42bb511ab1410a992c3fb9fc8a11235d555aea77
    ],
    include_dirs=include_dirs,
    language='c++',
    extra_compile_args=extra_compile_args,
    libraries=libraries,
)

setup(
    name="alphazero",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AlphaZero-style board game AI",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/alphazero",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.7.0",
        "pybind11>=2.6.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.50.0",
    ],
<<<<<<< HEAD
    ext_modules=[
        gomoku_module, 
        attack_defense_module, 
        mcts_module, 
        batch_evaluator_module
    ],
=======
    ext_modules=[gomoku_module, attack_defense_module, mcts_module, improved_mcts_module, batched_mcts_module],
>>>>>>> 42bb511ab1410a992c3fb9fc8a11235d555aea77
    cmdclass={'build_ext': build_ext},
)