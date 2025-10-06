# Quick Start Guide: MCTS Throughput Recovery

## Prerequisites

### Hardware Requirements
- **CPU**: AMD Ryzen 5900X or equivalent (12+ cores recommended)
- **GPU**: NVIDIA RTX 3060 Ti or better (8GB VRAM minimum)
- **RAM**: 32GB DDR4
- **OS**: Ubuntu 22.04 LTS

### Software Requirements
```bash
# System packages
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    python3.12-dev \
    libomp-dev \
    cuda-toolkit-12-1

# Python environment
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# Core dependencies
pip install -r requirements.txt
pip install pybind11[global]>=2.10.0  # For DLPack support
```

## Build Instructions

### Step 1: Configure Build Environment

```bash
# Set optimization flags for Ryzen 5900X
export CFLAGS="-O3 -march=znver3 -mtune=znver3 -fopenmp"
export CXXFLAGS="-O3 -march=znver3 -mtune=znver3 -fopenmp -std=c++17"

# Enable CUDA if available
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Step 2: Apply Optimization Patches

```bash
# Create feature branch
git checkout -b feature/004-mcts-throughput-recovery

# Apply patches in order (if provided as patch files)
git apply patches/001-wu-uct-virtual-loss.patch
git apply patches/002-lock-free-queue.patch
git apply patches/003-dlpack-bridge.patch
git apply patches/004-thread-affinity.patch
```

### Step 3: Build C++ Extensions

```bash
# Clean previous builds
rm -rf build/
pip uninstall mcts_py -y

# Build with optimizations
python -m pip install -e . \
    --config-settings build-dir=build \
    --config-settings cmake.define.ENABLE_WU_UCT=ON \
    --config-settings cmake.define.ENABLE_LOCK_FREE=ON \
    --config-settings cmake.define.ENABLE_DLPACK=ON \
    --config-settings cmake.define.ENABLE_THREAD_AFFINITY=ON

# Verify build
python -c "import mcts_py; print(mcts_py.__version__)"
```

## Validation Steps

### Step 1: Unit Tests

```bash
# Test individual optimizations
python -m pytest tests/unit/test_wu_uct.py -v
python -m pytest tests/unit/test_lock_free_queue.py -v
python -m pytest tests/unit/test_dlpack_bridge.py -v
python -m pytest tests/unit/test_thread_affinity.py -v

# Test thread safety
python -m pytest tests/unit/test_thread_safety.py -v --thread-sanitizer
```

### Step 2: Integration Tests

```bash
# Test complete pipeline
python -m pytest tests/integration/test_optimized_pipeline.py -v

# Verify backward compatibility
python -m pytest tests/integration/test_compatibility.py -v

# Test quality preservation
python -m pytest tests/integration/test_search_quality.py -v
```

### Step 3: Performance Benchmarks

```bash
# Baseline measurement (before optimizations)
git checkout main
python scripts/benchmark_throughput.py --tag baseline --game gomoku

# Optimized measurement (after optimizations)
git checkout feature/004-mcts-throughput-recovery
python scripts/benchmark_throughput.py --tag optimized --game gomoku

# Generate comparison report
python scripts/compare_benchmarks.py baseline optimized
```

## Configuration

### Create Optimized Configuration

Create `config/optimized.yaml`:

```yaml
# Optimized configuration for Ryzen 5900X + RTX 3060 Ti
mcts:
  # Tree settings
  num_simulations: 800
  c_puct: 1.25

  # Threading
  num_threads: 8  # One CCD on Ryzen 5900X
  thread_affinity: true
  affinity_mode: "ccd0"  # Use first CCD

  # Virtual loss (WU-UCT style)
  virtual_loss_mode: "wu_uct"
  virtual_loss_magnitude: 1.0
  busy_edge_masking: true

  # Memory
  tree_pool_size: 10000000  # 10M nodes
  use_memory_arenas: true
  arena_block_size: 1048576  # 1MB

  # Queue settings
  queue_type: "lock_free"
  queue_capacity: 4096

inference:
  # Batching
  min_batch_size: 32
  max_batch_size: 64
  batch_timeout_ms: 1.0

  # GPU settings
  device: "cuda:0"
  use_mixed_precision: true
  use_pinned_memory: true
  pinned_pool_size: 33554432  # 32MB

  # DLPack
  use_dlpack: true

  # Persistent thread
  use_persistent_thread: true
  inference_thread_affinity: "ccd1"

neural:
  # Model settings (unchanged)
  blocks: 20
  channels: 256
  se_ratio: 8
```

### Apply Configuration

```bash
# Run with optimized configuration
python scripts/test_mcts.py \
    --config config/optimized.yaml \
    --game gomoku \
    --simulations 10000
```

## Performance Validation

### Expected Metrics

Run the validation script to verify optimizations:

```bash
python scripts/validate_optimizations.py
```

Expected output:
```
MCTS Throughput Recovery - Validation Report
============================================

Throughput:
  Baseline:   3,831 sims/sec
  Optimized: 26,142 sims/sec
  Improvement: 6.82x ✓

GPU Utilization:
  Baseline:  32.8%
  Optimized: 85.3% ✓

Collision Rate:
  Baseline:  18.2%
  Optimized:  4.7% ✓

Batch Efficiency:
  Average Size: 48.3 / 64 (75.5%) ✓
  Timeout Rate: 22.3%
  Size Trigger: 77.7%

Thread Efficiency (8 threads):
  Baseline:  42.3%
  Optimized: 78.6% ✓

Search Quality:
  Policy Agreement: 96.2% ✓
  Value MSE: 0.008 ✓
  Win Rate vs Baseline: 99.8% ✓

All validation checks PASSED!
```

### Troubleshooting

#### Low Throughput (<20k sims/sec)

1. **Check GPU utilization**:
   ```bash
   nvidia-smi dmon -s u -i 0
   ```
   Should show 80%+ utilization

2. **Verify thread affinity**:
   ```bash
   taskset -cp $(pgrep -f "python.*mcts")
   ```
   Should show threads bound to cores 0-5 (CCD0)

3. **Check queue metrics**:
   ```bash
   python scripts/monitor_queue.py
   ```
   Look for high contention or overflow

#### High Collision Rate (>10%)

1. **Increase virtual loss**:
   ```yaml
   virtual_loss_magnitude: 1.5  # Increase from 1.0
   ```

2. **Reduce thread count**:
   ```yaml
   num_threads: 6  # Reduce from 8
   ```

3. **Enable busy-edge masking**:
   ```yaml
   busy_edge_masking: true  # Must be enabled
   ```

#### Memory Issues

1. **Monitor memory usage**:
   ```bash
   python scripts/monitor_memory.py --duration 60
   ```

2. **Reduce pool sizes if needed**:
   ```yaml
   tree_pool_size: 5000000  # Reduce from 10M
   pinned_pool_size: 16777216  # Reduce from 32MB
   ```

## A/B Testing

### Compare Search Quality

```bash
# Run tournament between versions
python scripts/tournament.py \
    --player1 baseline \
    --player2 optimized \
    --games 100 \
    --game gomoku

# Expected: Optimized wins 50±5% (no strength change)
```

### Statistical Validation

```bash
# Detailed quality analysis
python scripts/analyze_quality.py \
    --baseline-dir runs/baseline/ \
    --optimized-dir runs/optimized/ \
    --metrics "policy_kl,value_mse,visit_distribution"
```

## Production Deployment

### Step 1: Extended Validation

```bash
# 24-hour soak test
python scripts/soak_test.py \
    --config config/optimized.yaml \
    --duration 86400 \
    --game gomoku

# Should complete with:
# - No memory leaks
# - No crashes
# - Stable throughput
```

### Step 2: Gradual Rollout

```python
# config/rollout.yaml
rollout:
  enabled: true
  optimization_percentage: 10  # Start with 10% using optimizations

  # Gradual increase
  schedule:
    - day: 1
      percentage: 10
    - day: 3
      percentage: 25
    - day: 7
      percentage: 50
    - day: 14
      percentage: 100
```

### Step 3: Monitoring

```bash
# Start monitoring dashboard
python scripts/monitoring/dashboard.py \
    --port 8080 \
    --config config/optimized.yaml
```

Access at `http://localhost:8080` to monitor:
- Real-time throughput
- GPU utilization
- Collision rates
- Queue depths
- Memory usage

## Rollback Procedure

If issues occur:

```bash
# Quick rollback to baseline
git checkout main
pip install -e . --force-reinstall

# Or disable specific optimizations
export DISABLE_WU_UCT=1
export DISABLE_LOCK_FREE=1
export DISABLE_DLPACK=1

python scripts/test_mcts.py --config config/baseline.yaml
```

## Benchmarking Different Games

### Gomoku (15×15)
```bash
python scripts/benchmark_game.py --game gomoku --board-size 15
# Expected: 26,000+ sims/sec
```

### Chess
```bash
python scripts/benchmark_game.py --game chess
# Expected: 22,000+ sims/sec (more complex eval)
```

### Go (19×19)
```bash
python scripts/benchmark_game.py --game go --board-size 19
# Expected: 18,000+ sims/sec (larger board)
```

## Next Steps

After successful validation:

1. **Merge to main**:
   ```bash
   git checkout main
   git merge feature/004-mcts-throughput-recovery
   git push origin main
   ```

2. **Update documentation**:
   ```bash
   python scripts/generate_docs.py --include-benchmarks
   ```

3. **Train new models**:
   ```bash
   python src/training/train.py \
       --config config/optimized.yaml \
       --game gomoku \
       --iterations 100
   ```

4. **Share results**:
   - Update README.md with performance numbers
   - Create blog post about optimizations
   - Submit PR to main repository

## Support

For issues or questions:
- Check `docs/troubleshooting.md`
- Review logs in `logs/mcts_optimization.log`
- Run diagnostics: `python scripts/diagnose.py`

Remember: The optimizations are designed to be transparent to the MCTS algorithm itself - search quality should remain identical while throughput improves dramatically.