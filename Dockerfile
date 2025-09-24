# Multi-stage Dockerfile for High-Performance AlphaZero Engine
# Optimized for CUDA 12.x, Python 3.12, and C++ extensions

# =============================================================================
# Build Stage - Compile C++ extensions and install dependencies
# =============================================================================
FROM nvidia/cuda:12.2-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_ARCHITECTURES="75;86;89"

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    build-essential \
    cmake \
    git \
    pkg-config \
    libomp-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip and install build dependencies
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support first (for stable versions)
RUN pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121

# Set build directory
WORKDIR /app

# Copy build configuration files
COPY requirements.txt pyproject.toml CMakeLists.txt ./
COPY cpp_extensions/ ./cpp_extensions/
COPY src/ ./src/

# Install Python dependencies
RUN pip install -r requirements.txt

# Set C++ compiler flags for optimal performance
ENV CFLAGS="-O3 -march=x86-64-v3 -fopenmp -DNDEBUG"
ENV CXXFLAGS="-O3 -march=x86-64-v3 -fopenmp -DNDEBUG"

# Build C++ extensions with optimization
RUN pip install -e . --config-settings build-dir=build

# =============================================================================
# Runtime Stage - Minimal production image
# =============================================================================
FROM nvidia/cuda:12.2-runtime-ubuntu22.04 AS runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-venv \
    libomp16 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash --uid 1001 alphazero
USER alphazero
WORKDIR /home/alphazero

# Copy virtual environment from builder
COPY --from=builder --chown=alphazero:alphazero /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=alphazero:alphazero . .

# Create necessary directories
RUN mkdir -p results checkpoints training_logs training_data evaluation_results

# Set default configuration
ENV OMOKNUNI_CONFIG_PATH=/home/alphazero/config/default.yaml
ENV OMOKNUNI_DATA_PATH=/home/alphazero/training_data
ENV OMOKNUNI_LOG_LEVEL=INFO

# Health check to ensure the system is working
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import src.alphazero_py; import torch; print(f'CUDA available: {torch.cuda.is_available()}'); exit(0)" || exit 1

# Default command - run basic system check
CMD ["python", "-c", "import src.alphazero_py; print('AlphaZero Engine initialized successfully')"]

# =============================================================================
# Development Stage - Full development environment
# =============================================================================
FROM builder AS development

# Install development dependencies
RUN pip install pytest pytest-cov pytest-xdist black flake8 mypy isort pre-commit

# Install Jupyter for interactive development
RUN pip install jupyter jupyterlab

# Set development environment variables
ENV OMOKNUNI_ENV=development
ENV OMOKNUNI_LOG_LEVEL=DEBUG

# Expose Jupyter port
EXPOSE 8888

# Development user setup
RUN useradd --create-home --shell /bin/bash --uid 1001 dev
USER dev
WORKDIR /home/dev

# Copy application code
COPY --chown=dev:dev . .

# Default development command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# =============================================================================
# Training Stage - Optimized for model training
# =============================================================================
FROM runtime AS training

# Install additional training dependencies
USER root
RUN apt-get update && apt-get install -y \
    htop \
    nvtop \
    && rm -rf /var/lib/apt/lists/*

USER alphazero

# Create training-specific directories with proper permissions
RUN mkdir -p /home/alphazero/models /home/alphazero/experiments /home/alphazero/tensorboard_logs

# Set training environment variables
ENV OMOKNUNI_ENV=training
ENV OMOKNUNI_ENABLE_TENSORBOARD=true
ENV CUDA_LAUNCH_BLOCKING=0

# Expose TensorBoard port
EXPOSE 6006

# Training health check - ensure GPU memory is available
HEALTHCHECK --interval=60s --timeout=15s --start-period=120s --retries=2 \
    CMD python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() > 0; print('GPU training ready')" || exit 1

# Default training command
CMD ["python", "-m", "src.training.training_loop", "--config", "config/default.yaml"]