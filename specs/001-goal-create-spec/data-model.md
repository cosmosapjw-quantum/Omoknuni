# Data Model: High-Performance AlphaZero Engine

## Core MCTS Tree Structure

### MCTSNode (Structure-of-Arrays Layout)
**Purpose**: Represents individual nodes in the Monte Carlo Tree Search

**Fields**:
- `visit_count: float32` - Number of times node has been visited (N)
- `total_value: float32` - Accumulated value from all visits (W)
- `prior_prob: float32` - Neural network policy probability (P)
- `virtual_loss: float32` - Temporary loss to prevent thread collisions (VL)
- `parent_index: int32` - Index of parent node (-1 for root)
- `first_child_index: int32` - Index of first child in children array (-1 if unexpanded)
- `num_children: uint16` - Number of child nodes
- `flags: uint8` - Packed boolean flags (expanded, terminal, current_player)

**Memory Layout**:
```
Alignment: 64-byte boundaries for SIMD operations
Size per node: 27 bytes achieved (target was <64 bytes)
Max nodes: 50M (configurable, 270MB for 10M nodes actual)
Index space: int32 supports 2B nodes
Node allocation: 330M allocations/second via free list
```

**Validation Rules**:
- `visit_count >= 0`
- `total_value` in range [-visit_count, visit_count]
- `prior_prob` in range [0, 1]
- `virtual_loss >= 0`
- `parent_index < current_node_index` (DAG property)
- `first_child_index > current_node_index` or -1

**Node Pool Management**:
- **Pre-allocation**: All nodes allocated at tree creation time
- **Free List**: O(1) reuse of deallocated nodes via vector<NodeIndex>
- **Contiguous Allocation**: Multi-child expansions get adjacent indices
- **High Water Mark**: `next_free_index` tracks allocated index range
- **Bounds Checking**: `is_valid_index()` validates against allocated range

**State Transitions**:
1. **Creation**: All fields zero-initialized except parent linkage
2. **Expansion**: Set first_child_index, num_children, expanded flag
3. **Selection**: Apply virtual_loss temporarily
4. **Backup**: Increment visit_count, update total_value, remove virtual_loss
5. **Terminal**: Set terminal flag, compute exact terminal_value
6. **Deallocation**: Add to free list for O(1) reuse

### GameState Interface
**Purpose**: Abstract game representation supporting multiple board games

**Core Fields**:
- `board_representation: varies` - Game-specific board state
- `current_player: int8` - Active player (0 or 1)
- `move_history: int16[]` - Sequence of moves leading to this state
- `legal_moves_cache: uint8[]` - Cached legal move mask
- `hash_value: uint64` - Position hash for transposition table

**Game-Specific Implementations**:

#### GomokuState
- `board: uint8[15][15]` - 0=empty, 1=player1, 2=player2
- `last_move: int16` - Most recent move coordinate
- `move_count: uint8` - Total moves played

#### ChessState
- `pieces: uint64[12]` - Bitboards for each piece type/color
- `castling_rights: uint8` - 4 bits for K/Q side castling
- `en_passant_square: uint8` - Target square for en passant capture
- `halfmove_clock: uint8` - Moves since pawn move or capture
- `fullmove_number: uint16` - Game move counter

#### GoState
- `stones: uint8[][]` - 2D array sized for board (9x9 to 19x19)
- `captured_stones: uint16[2]` - Capture count per player
- `ko_square: int16` - Coordinate of ko-prohibited square
- `pass_count: uint8` - Consecutive passes (2 = game end)

**Validation Rules**:
- `current_player` in {0, 1}
- `legal_moves_cache` updated when board changes
- `hash_value` updated incrementally with moves

## Neural Network Data Structures

### InferenceBatch
**Purpose**: Batch of positions for GPU neural network inference

**Fields**:
- `positions: float32[B,C,H,W]` - Batched feature planes
- `batch_size: int32` - Actual batch size (≤ max_batch_size)
- `game_types: uint8[B]` - Game identifier per position
- `metadata: BatchMetadata` - Timing and queue information

**Constraints**:
- `batch_size` in range [1, 256] (GPU memory limited)
- `positions` aligned to 16-byte boundaries
- Feature planes `C` varies by game (7 for Gomoku, 12 for Chess, 17 for Go)

### InferenceResult
**Purpose**: Neural network output for batched positions

**Fields**:
- `policies: float32[B,A]` - Policy probabilities per position
- `values: float32[B]` - Position values from current player's perspective
- `processing_time_ms: float32` - GPU inference duration
- `batch_utilization: float32` - Actual batch size / max batch size

**Validation Rules**:
- `policies[i]` sum to 1.0 for each position i
- `values[i]` in range [-1, 1]
- `processing_time_ms >= 0`

## Training Pipeline Data

### ExperienceBuffer
**Purpose**: Stores self-play game data for neural network training

**Fields**:
- `states: float32[N,C,H,W]` - Game positions as feature planes
- `policies: float32[N,A]` - MCTS visit count distributions (targets)
- `outcomes: float32[N]` - Game outcomes from position player's perspective
- `game_metadata: GameMetadata[N]` - Game ID, move number, etc.

**Storage Strategy**:
- **Memory-mapped files**: Scale beyond RAM for large datasets
- **Parquet format**: Compression and schema evolution support
- **RAM cache**: LRU cache for recently accessed experiences
- **Rotation policy**: Keep last 1M experiences, archive older data

**Sampling Strategy**:
- **Uniform random**: Each experience equally likely
- **Recency weighting**: Slight preference for recent games
- **Balance by outcome**: Ensure wins/losses/draws represented

### TrainingMetrics
**Purpose**: Track neural network training progress and stability

**Fields**:
- `policy_loss: float32` - KL divergence between predicted and target policies
- `value_loss: float32` - MSE between predicted and actual game outcomes
- `policy_entropy: float32` - Diversity measure of policy predictions
- `value_accuracy: float32` - Sign agreement between predicted and actual outcomes
- `gradient_norm: float32` - L2 norm of gradients (stability indicator)
- `learning_rate: float32` - Current adaptive learning rate

**Quality Indicators**:
- Policy loss: Should decrease monotonically, typical range 0.5-2.0
- Value loss: Should decrease monotonically, typical range 0.1-0.8
- Policy entropy: Should decrease gradually, indicating specialization
- Value accuracy: Should increase, target >75% sign agreement
- Gradient norm: Should remain stable, <10 indicates good conditioning

## Performance Telemetry

### SearchMetrics
**Purpose**: Monitor MCTS search performance and efficiency

**Fields**:
- `simulations_per_second: float32` - Total simulation rate including NN inference
- `cpu_utilization: float32` - Percentage of CPU cores active
- `gpu_utilization: float32` - Percentage of GPU compute utilization
- `average_batch_size: float32` - Mean positions per inference batch
- `memory_usage_mb: float32` - Peak resident memory usage
- `thread_efficiency: float32[]` - Per-thread work distribution

**Performance Targets**:
- `simulations_per_second`: 30,000-40,000 (including neural network time)
- `cpu_utilization`: 85%-95% (optimal thread saturation)
- `gpu_utilization`: 80%-92% (realistic GPU efficiency)
- `average_batch_size`: 32-64 (efficient GPU occupancy)
- `memory_usage_mb`: <1,024 (tree memory constraint)

### SystemHealthMetrics
**Purpose**: Monitor system stability and resource usage

**Fields**:
- `memory_growth_rate_mb_per_hour: float32` - Memory leak detection
- `gpu_memory_peak_mb: float32` - Peak VRAM usage monitoring
- `thread_contention_ratio: float32` - Atomic operation efficiency
- `inference_queue_depth: float32` - GPU workload backlog
- `error_rates: ErrorMetrics` - Exception and failure tracking

**Health Thresholds**:
- Memory growth rate: <10 MB/hour (no significant leaks)
- GPU memory peak: <6,800 MB (85% of 8GB VRAM)
- Thread contention: <10% (efficient atomic operations)
- Inference queue depth: <100 (no persistent backlog)

---

*Data model designed for cache efficiency, thread safety, and hardware optimization on Ryzen 5900X + RTX 3060 Ti architecture.*