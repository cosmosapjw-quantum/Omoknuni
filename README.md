# Omoknuni: AlphaZero for Gomoku

A modular, high-performance AlphaZero-style AI system for Gomoku/Renju with C++ game logic and MCTS implementation. The project features both Python and C++ implementations of the Monte Carlo Tree Search algorithm with a focus on performance and scalability.

## Features

- Game-agnostic framework for AlphaZero-style learning and play
- High-performance C++ implementation of Gomoku/Renju game rules
- Attack and defense score computation for Gomoku in C++
- Python-based MCTS with neural network integration
- High-performance C++ MCTS implementation with Python bindings
- Zobrist hashing for efficient board state representation
- Transposition table for MCTS optimization
- Multithreaded MCTS with virtual loss
- Enhanced MCTS with RAVE (Rapid Action Value Estimation)
- Flexible neural network architectures in PyTorch
- Complete training and evaluation pipeline

## Project Structure

Below is the complete structure of the Omoknuni project, detailing each folder and file with their roles:

### Top-Level Files
- `README.md`: Project documentation
- `bugfixes.txt`: Documentation of recent bug fixes
- `CMakeLists.txt`: Main CMake build configuration file
- `__init__.py`: Python package initialization

### `/alphazero` - Core Package

#### `/alphazero/core` - C++ Implementation
- **`/game`**: C++ implementations of game rules
  - `gomoku.h` & `gomoku.cpp`: Core Gomoku game logic with rule variations (standard, Renju, Omok)
  - `attack_defense.h` & `attack_defense.cpp`: Attack and defense score computation for move evaluation
  - `CMakeLists.txt`: Build configuration for game logic

- **`/mcts`**: C++ implementation of Monte Carlo Tree Search
  - `mcts.h` & `mcts.cpp`: High-performance MCTS implementation
  - `mcts_node.h` & `mcts_node.cpp`: Node representation for MCTS tree
  - `transposition_table.h` & `transposition_table.cpp`: Optimized transposition table with LRU caching
  - `zobrist_hash.h` & `zobrist_hash.cpp`: Zobrist hashing for efficient state representation
  - `CMakeLists.txt`: Build configuration for MCTS

- **`/utils`**: Utility functions for C++ code
  - `thread_pool.h` & `thread_pool.cpp`: Thread pool for parallel MCTS simulations
  - `CMakeLists.txt`: Build configuration for utilities

- `CMakeLists.txt`: Main build configuration for C++ components

#### `/alphazero/python` - Python Implementation

- **`/games`**: Game interface wrappers
  - `game_base.py`: Abstract base class for game wrappers
  - `gomoku.py`: Python wrapper for C++ Gomoku implementation

- **`/mcts`**: Python MCTS implementations
  - `mcts.py`: Basic MCTS implementation
  - `enhanced_mcts.py`: Advanced MCTS with transposition table, virtual loss, and RAVE
  - `cpp_mcts_wrapper.py`: Wrapper for C++ MCTS implementation
  - `transposition_table.py`: Python transposition table for MCTS optimization

- **`/models`**: Neural network models for AlphaZero
  - `network_base.py`: Base class for all neural network models
  - `simple_conv_net.py`: Convolutional neural network for board evaluation
  - `batched_evaluator.py`: Efficient batch evaluation for neural networks

- **`/training`**: Training pipeline components
  - `trainer.py`: Main training loop and model optimization
  - `self_play.py`: Self-play game generation for training data

- **`/utils`**: Python utility functions
  - `tree_visualization.py`: Visualization tools for MCTS trees

- `__init__.py`: Python package initialization

#### `/alphazero/bindings` - C++/Python Bindings
- `mcts_bindings.cpp`: Bindings for C++ MCTS implementation
- `game_bindings.cpp`: Bindings for C++ game implementations
- `CMakeLists.txt`: Build configuration for Python bindings

#### `/alphazero/examples` - Example Scripts
- `check_setup.py`: Verify the setup and dependencies
- `evaluate_model.py`: Evaluate trained models against different opponents
- `human_vs_ai.py`: Interface for playing against trained models
- `test_cpp_mcts.py`: Test C++ MCTS implementation
- `test_enhanced_mcts.py`: Test enhanced MCTS features
- `test_gomoku.py`: Test Gomoku game implementation
- `test_mcts.py`: Test basic MCTS implementation
- `test_nn_mcts.py`: Test neural network integration with MCTS
- `test_zobrist.py`: Test and benchmark Zobrist hashing
- `test_zobrist_simple.py`: Simple tests for Zobrist hashing
- `train_alphazero.py`: Basic AlphaZero training script
- `train_alphazero_enhanced.py`: Enhanced training with additional optimizations
- `train_with_cpp_mcts.py`: Training using the C++ MCTS implementation

## Key Features Explained

### 1. Gomoku Game Implementation
- **Multiple Rule Variations:**
  - Standard Gomoku: Five or more in a row wins
  - Renju Rules: Black has restrictions on overlines and forbidden moves (double-three, double-four)
  - Omok Rules: Another variant with specific restrictions
  - Professional Long Opening: Special opening rules used in professional games
- **Efficient Bitboard Representation:** Uses bitboards for fast move validation and pattern detection
- **Attack-Defense Scoring:** Evaluates moves based on offensive and defensive potential

### 2. Monte Carlo Tree Search (MCTS)
- **Multiple Implementations:**
  - Basic Python MCTS
  - Enhanced Python MCTS with optimizations
  - High-performance C++ MCTS with Python bindings
- **Optimizations:**
  - Transposition Table: Reuses evaluations for identical board positions
  - Zobrist Hashing: Efficient board state representation for transposition tables
  - Virtual Loss: Prevents thread collision in parallel search
  - RAVE (Rapid Action Value Estimation): Accelerates search with move urgency evaluation
  - Progressive Widening: Handles large branching factors by focusing on promising moves

### 3. Neural Network Integration
- **Flexible Architecture:**
  - Residual convolutional networks for board evaluation
  - Policy head for move probability prediction
  - Value head for position evaluation
- **Batch Processing:** Efficient batch evaluation for improved performance

### 4. Training Pipeline
- **Self-Play Game Generation:**
  - Multithreaded self-play for fast data generation
  - Temperature-based move selection for exploration
  - Dirichlet noise for root node exploration
- **Model Training:**
  - Combined policy and value loss optimization
  - Checkpoint management and model versioning
  - Early stopping and best model tracking

### 5. Parallelization and Performance
- **Multithreaded MCTS:**
  - Thread pool for parallel simulations
  - Thread-safe transposition table
  - Virtual loss for efficient thread coordination
- **C++/Python Integration:**
  - High-performance core in C++
  - Flexible experimentation in Python
  - Optimized bindings for minimal overhead

## Installation

### Prerequisites

- Python 3.8+ with pip
- C++17 compatible compiler (MSVC on Windows, GCC/Clang on Linux/Mac)
- PyTorch
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd Omoknuni

# Install the package in development mode
pip install -e .
```

For Windows users, see [WINDOWS_BUILD.md](WINDOWS_BUILD.md) for specific build instructions.

## Usage

### Testing the C++ Gomoku Implementation

```bash
python alphazero/examples/test_gomoku.py
```

### Testing Zobrist Hashing Performance

```bash
python alphazero/examples/test_zobrist.py
```

### Playing with MCTS and Neural Network

```bash
python alphazero/examples/test_nn_mcts.py
```

### Training AlphaZero

```bash
python alphazero/examples/train_alphazero.py \
    --board-size 9 \
    --num-iterations 20 \
    --games-per-iteration 10 \
    --mcts-simulations 400 \
    --num-workers 4 \
    --batch-size 64 \
    --epochs-per-iteration 5 \
    --checkpoint-dir checkpoints
```

For faster training, start with a smaller board size like 9x9.

### Training with C++ MCTS

For higher performance training with the C++ MCTS implementation:

```bash
python alphazero/examples/train_with_cpp_mcts.py \
    --board-size 9 \
    --num-iterations 20 \
    --games-per-iteration 10 \
    --mcts-simulations 800 \
    --num-workers 4 \
    --batch-size 64 \
    --epochs-per-iteration 5 \
    --checkpoint-dir checkpoints
```

### Evaluating a Trained Model

```bash
python alphazero/examples/evaluate_model.py \
    --model-path checkpoints/model_iter_20.pt \
    --board-size 9 \
    --num-games 20 \
    --mcts-simulations 800 \
    --opponent random
```

### Playing Against AlphaZero

```bash
python alphazero/examples/human_vs_ai.py \
    --model-path checkpoints/model_iter_20.pt \
    --board-size 15 \
    --mcts-simulations 1600
```

## Gomoku Rules Options

The implementation supports several rule variations:

- **Standard Gomoku**: The simplest rule set, where five or more in a row wins.
- **Renju Rules**: Black has restrictions on overlines and forbidden moves.
- **Omok Rules**: Another rule variation with specific restrictions.
- **Professional Long Opening**: Special opening rules used in professional games.

To use these rules, add the corresponding flags to the command:

```bash
--use-renju --use-pro-long-opening
```

## Advanced Training Options

For advanced training with larger boards or more complex models:

```bash
python alphazero/examples/train_alphazero_enhanced.py \
    --board-size 15 \
    --num-filters 128 \
    --num-residual-blocks 10 \
    --num-iterations 100 \
    --games-per-iteration 50 \
    --mcts-simulations 800 \
    --batch-size 128 \
    --epochs-per-iteration 10 \
    --num-workers 8 \
    --use-transposition-table \
    --use-cuda
```

## Recent Updates

Recent commits have focused on:
- Implementation of Zobrist hashing for efficient board state representation
- Optimization of MCTS transposition table
- Fixing MCTS debugging issues
- Performance optimizations for large board sizes

## License

[MIT License](LICENSE)