# AlphaZero Engine

A modular, high-performance AlphaZero-style AI system that can be applied to various board games.

## Features

- Game-agnostic framework for AlphaZero-style learning and play
- High-performance MCTS implementation in C++ with Python bindings
- Flexible neural network architectures in PyTorch
- Efficient parallelization for tree search and neural network inference

## Supported Games

- Gomoku/Renju (primary)
- Chess (planned)
- Go (future extension)

## Requirements

- C++17 compiler
- Python 3.8+
- PyTorch
- CUDA-capable GPU (recommended)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/alphazero.git
cd alphazero

# Install the package
pip install -e .
```

## Usage

See the examples directory for usage examples.

## License

[MIT License](LICENSE)