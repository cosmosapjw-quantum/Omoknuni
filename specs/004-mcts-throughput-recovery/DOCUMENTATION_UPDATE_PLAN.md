# Documentation Update Plan: Spec 004 Comprehensive Revision

**Date**: 2025-10-14
**Purpose**: Consolidate, update, and align all Spec 004 documentation with review.txt findings

---

## Analysis Summary

### Current State (15 documents, 15,245 total lines)

**Active Documents** (keep and update):
- ✅ `spec.md` (503 lines) - v2.0, authoritative functional spec
- ✅ `plan.md` (2,398 lines) - v1.0, authoritative technical plan
- ✅ `tasks.md` (2,747 lines) - v1.0, authoritative task breakdown
- ✅ `CONSTITUTION.md` (729 lines) - v2.0, immutable constraints
- ✅ `CLARIFICATIONS.md` (633 lines) - Resolved ambiguities
- ✅ `ACCEPTANCE_CHECKLIST.md` (860 lines) - Validation checklist
- ✅ `TRACEABILITY_MATRIX.md` (524 lines) - Cross-check analysis
- ✅ `data-model.md` (542 lines) - Memory layout specs
- ✅ `research.md` (521 lines) - Architecture decisions
- ✅ `quickstart.md` (454 lines) - Build/validation guide
- ✅ `README.md` (383 lines) - Spec overview
- ✅ `CHECKLIST_SUMMARY.md` (264 lines) - Quick reference

**Obsolete Documents** (delete):
- ❌ `SPECIFICATION.md` (1,210 lines) - Superseded by spec.md v2.0
- ❌ `TASKS.md` (1,751 lines) - Superseded by tasks.md v1.0
- ❌ `TECHNICAL_PLAN.md` (1,034 lines) - Superseded by plan.md v1.0
- ❌ `PR_CHECKLIST.md` (509 lines) - Superseded by ACCEPTANCE_CHECKLIST.md
- ❌ `REVIEW-RESPONSE.md` (183 lines) - One-time response, archive only

**Missing Content** (from review.txt):
- 🆕 Lightweight NN redesign (review.txt lines 621-1396)
- 🆕 Precompute legal moves optimization (review.txt lines 189-201)
- 🆕 DLPack fast path validation (review.txt lines 260-280)
- 🆕 Thread affinity implementation details (review.txt lines 244-250)

---

## Gap Analysis from review.txt

### GAP 1: Lightweight NN Architecture (MAJOR - 775 lines in review.txt)

**Location**: review.txt lines 621-1396
**Content**:
- RepVGG/DBB reparameterizable blocks + ECA attention
- Ghost bottleneck residuals (30-70% speedup)
- ShuffleNetV2 channel-split (15-35% speedup)
- Two-tier evaluator cascade (1.5-2.5× throughput)
- Early-exit heads with confidence gating (1.2-1.6× speedup)
- Auxiliary tactical heads (threat detection)
- Game-specific configurations (Gomoku/Chess/Go)

**Status**: NOT in any spec document
**Action**: Create `NEURAL_NETWORK_OPTIMIZATION.md` (Phase 7)

### GAP 2: Precompute Legal Moves (MEDIUM - review.txt lines 189-201)

**Recommendation**: Store `legal_moves` in `InferenceRequest` to avoid state access during expansion
**Current Coverage**:
- ❌ spec.md: Not mentioned
- ❌ plan.md: Not mentioned
- ❌ tasks.md: Not mentioned
- ✅ TRACEABILITY_MATRIX.md: Identified as GAP 2, deferred to Phase 2

**Action**: Add to plan.md Section B2.4, tasks.md as T030a

### GAP 3: DLPack Fast Path Validation (LOW - review.txt lines 260-280)

**Recommendation**: Verify `DLPackInferenceBridge` active, ensure no NumPy conversion fallback
**Current Coverage**:
- ⚠️ spec.md FR1.5: Mentions "streamline batch interface" (generic)
- ⚠️ plan.md B5: Recommends "verify DLPackInferenceBridge active" (no task)
- ❌ tasks.md: No explicit task
- ✅ TRACEABILITY_MATRIX.md: Identified as GAP 1, deferred

**Action**: Add to plan.md Section B5.4, tasks.md as T014a

### GAP 4: Thread Affinity Implementation (LOW - review.txt lines 244-250)

**Recommendation**: Explicit core pinning to physical cores 0-11, avoid SMT siblings
**Current Coverage**:
- ⚠️ spec.md FR1.3: Mentions "thread affinity" (no code)
- ⚠️ plan.md B3.4: Recommends "tuning" (no implementation)
- ❌ tasks.md T013: Validation only, no implementation details
- ✅ TRACEABILITY_MATRIX.md: Identified as GAP 3

**Action**: Add implementation code to plan.md Section B3.4

---

## Update Actions

### Action 1: Delete Obsolete Documents

```bash
# Backup first
mkdir -p specs/004-mcts-throughput-recovery/archive/
mv specs/004-mcts-throughput-recovery/{SPECIFICATION.md,TASKS.md,TECHNICAL_PLAN.md,PR_CHECKLIST.md,REVIEW-RESPONSE.md} \
   specs/004-mcts-throughput-recovery/archive/

# Create archive README
echo "# Archived Documents

These documents have been superseded by newer versions:
- SPECIFICATION.md → spec.md v2.0
- TASKS.md → tasks.md v1.0
- TECHNICAL_PLAN.md → plan.md v1.0
- PR_CHECKLIST.md → ACCEPTANCE_CHECKLIST.md
- REVIEW-RESPONSE.md → Historical record only

Archived: 2025-10-14
" > specs/004-mcts-throughput-recovery/archive/README.md
```

### Action 2: Update spec.md

**Add Section 6: Advanced Optimizations (Optional Phases)**

```markdown
## 6. Advanced Optimizations (Optional Phases)

### 6.1 Precompute Legal Moves (Phase 2, Post-8k)

**FR6.1: Legal Move Precomputation**:
- Store `std::vector<Move> legal_moves` in `InferenceRequest`
- Populate in `ContinuousSimulationRunner` before queue submit
- Use stored moves in `expand_node_with_result()` (skip `getLegalMoves()` call)
- **Expected Impact**: 10-20% reduction in expansion time
- **Validation**: Micro-benchmark shows expand_node time reduction

### 6.2 DLPack Fast Path Validation (Phase 2, If Needed)

**FR6.2: DLPack Zero-Copy Verification**:
- Instrument `PyBatchInferenceCallback` to log conversion path
- Verify `isinstance(bridge, DLPackInferenceBridge) == True` at runtime
- Measure batch callback time (target <0.5ms)
- **Acceptance**: Fast path confirmed, no NumPy conversion fallback

### 6.3 Lightweight Neural Network (Phase 7, Future)

**FR6.3: Neural Network Architecture Optimization**:
- RepVGG/ECA reparameterizable blocks (train multi-branch, fuse for inference)
- Ghost bottleneck residuals (30-70% FLOP reduction)
- Early-exit heads with confidence gating (1.2-1.6× speedup)
- Two-tier evaluator cascade (1.5-2.5× throughput)
- **Target**: 1.5-3.5× model speedup without quality loss
- **Validation**: ELO ≥ baseline, throughput 12k-22k sims/sec (post-model optimization)
- **Reference**: See `NEURAL_NETWORK_OPTIMIZATION.md` for detailed design
```

### Action 3: Update plan.md

**Section B2.4: Precompute Legal Moves (Phase 2)**

```markdown
#### B2.4 Precompute Legal Moves (Optional Phase 2)

**Problem**: `expand_node_with_result()` calls `state.getLegalMoves()` during expansion
**Target**: Eliminate state access by precomputing moves

**Implementation**:

**File**: `cpp_extensions/mcts/async_inference_queue.hpp`
**Add to InferenceRequest**:
```cpp
struct InferenceRequest {
    int node_index;
    std::unique_ptr<IGameState> state;
    std::vector<Move> legal_moves;  // NEW: Precomputed moves
    int current_player;              // NEW: Precomputed player
    // ... existing fields
};
```

**File**: `cpp_extensions/mcts/continuous_simulation_runner.cpp`
**Precompute before submit**:
```cpp
// Before enqueuing request
auto legal_moves = current_state->getLegalMoves();
int current_player = current_state->getCurrentPlayer();

InferenceRequest req;
req.node_index = leaf_index;
req.state = std::move(current_state);
req.legal_moves = std::move(legal_moves);  // Store
req.current_player = current_player;       // Store

queue->submit_request(std::move(req));
```

**File**: `cpp_extensions/mcts/tree.cpp`
**Use stored moves**:
```cpp
void MCTSTree::expand_node_with_result(const InferenceRequest& req, ...) {
    // OLD: auto legal_moves = req.state->getLegalMoves();
    // NEW: Use precomputed
    const auto& legal_moves = req.legal_moves;
    int player = req.current_player;

    // ... rest of expansion logic
}
```

**Expected Impact**: 10-20% reduction in expansion time (micro-benchmark)
```

**Section B5.4: DLPack Fast Path Verification (Optional)**

```markdown
#### B5.4 DLPack Fast Path Verification (Phase 2, If Needed)

**Problem**: Uncertain if `DLPackInferenceBridge` is active or falling back to NumPy conversion
**Target**: Confirm zero-copy path, measure batch callback overhead

**Instrumentation**:

**File**: `src/core/mcts.py`
**Add logging to PyBatchInferenceCallback**:
```python
def _py_batch_inference_callback(game_states, inference_fn):
    import time
    start = time.perf_counter()

    # Log conversion path
    if isinstance(inference_fn, DLPackInferenceBridge):
        logger.debug(f"✅ DLPack fast path active (batch_size={len(game_states)})")
        results = inference_fn.batch_inference(game_states)
    else:
        logger.warning(f"⚠️ NumPy fallback path (batch_size={len(game_states)})")
        # ... NumPy conversion

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.debug(f"Batch callback time: {elapsed_ms:.2f}ms")

    return results
```

**Validation**:
- Run benchmark with `LOG_LEVEL=DEBUG`
- Grep for "DLPack fast path active" (should be 100% of batches)
- Measure callback time: target <0.5ms per batch

**Acceptance**: Fast path confirmed, callback overhead <1% of total time
```

**Section B3.4: Thread Affinity Implementation (Enhanced)**

```markdown
#### B3.4 Thread Affinity Tuning (Enhanced with Implementation)

**Problem**: Current `ThreadAffinityManager` uses `thread_id % 24` heuristic, may not distribute optimally
**Target**: Explicit core pinning to physical cores 0-11, avoid SMT siblings

**Current Implementation**:
**File**: `cpp_extensions/mcts/thread_affinity.cpp` (assumed to exist)

**Enhanced Implementation**:
```cpp
#include <pthread.h>
#include <sched.h>

class ThreadAffinityManager {
public:
    static void pin_to_physical_core(int thread_id, int num_physical_cores = 12) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);

        // Map thread to physical core (avoid SMT siblings)
        int core_id = thread_id % num_physical_cores;
        CPU_SET(core_id, &cpuset);

        // Apply affinity
        int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        if (result != 0) {
            std::cerr << "Warning: Failed to set thread affinity for thread "
                      << thread_id << " to core " << core_id << std::endl;
        } else {
            std::cout << "Thread " << thread_id << " pinned to core " << core_id << std::endl;
        }
    }

    // Ryzen 5900X specific: CCD0 (cores 0-5), CCD1 (cores 6-11)
    static void pin_to_ccd(int thread_id) {
        int core_id = thread_id % 12;  // 12 physical cores
        pin_to_physical_core(thread_id, 12);
    }
};
```

**Integration**:
**File**: `cpp_extensions/mcts/continuous_simulation_runner.cpp`
```cpp
void ContinuousSimulationRunner::worker_thread(int thread_id) {
    // Pin thread to physical core at startup
    ThreadAffinityManager::pin_to_ccd(thread_id);

    // ... rest of worker loop
}
```

**Validation** (in T013):
- Verify pinning with `lscpu --extended` or `taskset -cp <pid>`
- Measure cache misses with `perf stat -e cache-misses`
- Expected: ~15% cache miss reduction vs. no affinity

**Evidence**: Thread affinity confirmed via `lscpu`, cache misses reduced
```

### Action 4: Update tasks.md

**Add T014a: DLPack Fast Path Verification (0.25 days)**

```markdown
### T014a: DLPack Fast Path Verification (Optional Phase 2)

**Goal**: Verify DLPack zero-copy path is active, measure batch callback overhead

**Scope**: 0.25 day

**Dependencies**: T014 (comprehensive throughput benchmarks)

**Files to Modify**:
- `src/core/mcts.py` (add logging to `_py_batch_inference_callback`)

**Implementation**:
```python
# src/core/mcts.py
def _py_batch_inference_callback(game_states, inference_fn):
    import time
    start = time.perf_counter()

    if isinstance(inference_fn, DLPackInferenceBridge):
        conversion_path = "dlpack_fast"
        results = inference_fn.batch_inference(game_states)
    else:
        conversion_path = "numpy_fallback"
        # NumPy conversion

    elapsed_ms = (time.perf_counter() - start) * 1000

    # Log telemetry
    telemetry.log_batch_callback(
        conversion_path=conversion_path,
        batch_size=len(game_states),
        callback_time_ms=elapsed_ms
    )

    return results
```

**Acceptance Tests**:
1. Unit test: Verify logging works
2. Integration test: Run benchmark with `LOG_LEVEL=DEBUG`, grep for "dlpack_fast"
3. Performance test: Callback time <0.5ms mean

**Done Means**:
- ✅ DLPack fast path confirmed in 100% of batches (no NumPy fallback)
- ✅ Batch callback overhead <1% of total time
- ✅ Telemetry field `conversion_path == "dlpack_fast"`

**Rollback**: N/A (diagnostic only, no functional changes)
```

**Add T030a: Precompute Legal Moves (1.0 day)**

```markdown
### T030a: Precompute Legal Moves in InferenceRequest (Phase 2)

**Goal**: Eliminate state access during expansion by precomputing legal moves and current player

**Scope**: 1.0 day

**Dependencies**: T009 (state pooling integration complete)

**Files to Modify**:
1. `cpp_extensions/mcts/async_inference_queue.hpp`
2. `cpp_extensions/mcts/continuous_simulation_runner.cpp`
3. `cpp_extensions/mcts/tree.cpp`

**Implementation**:

**Step 1: Extend InferenceRequest**
```cpp
// async_inference_queue.hpp
struct InferenceRequest {
    int node_index;
    std::unique_ptr<IGameState> state;
    std::vector<Move> legal_moves;  // NEW
    int current_player;              // NEW
    // ... existing fields
};
```

**Step 2: Precompute in Runner**
```cpp
// continuous_simulation_runner.cpp
void ContinuousSimulationRunner::submit_to_queue(...) {
    // Precompute before state ownership transfer
    auto legal_moves = current_state->getLegalMoves();
    int player = current_state->getCurrentPlayer();

    InferenceRequest req;
    req.node_index = leaf_index;
    req.state = std::move(current_state);  // Transfer ownership
    req.legal_moves = std::move(legal_moves);
    req.current_player = player;

    queue_->submit_request(std::move(req));
}
```

**Step 3: Use Precomputed in Expansion**
```cpp
// tree.cpp
void MCTSTree::expand_node_with_result(const InferenceRequest& req, ...) {
    // OLD: auto legal_moves = req.state->getLegalMoves();
    // NEW: Use precomputed
    const auto& legal_moves = req.legal_moves;
    int player = req.current_player;

    // ... allocate children, set priors
}
```

**Acceptance Tests**:
1. Unit test: `InferenceRequest` serialization/deserialization
2. Parity test: Expansion results identical with/without precomputation
3. Performance test: Expansion time reduced 10-20% (micro-benchmark)

**Done Means**:
- ✅ `getLegalMoves()` NOT called during expansion (instrumentation confirms)
- ✅ Expansion time reduced ≥10% (measured with 10k expansions)
- ✅ All parity tests pass (visit counts, Q-values identical)

**Rollback**: Feature flag `PRECOMPUTE_LEGAL_MOVES` (default: false)
```

### Action 5: Create NEURAL_NETWORK_OPTIMIZATION.md

```markdown
# Neural Network Architecture Optimization (Phase 7)

**Version**: 1.0
**Date**: 2025-10-14
**Status**: FUTURE (Post-8k Throughput Target)
**Authority**: Derived from review.txt lines 621-1396

---

## Executive Summary

This document specifies neural network architecture optimizations to achieve **1.5-3.5× model speedup** beyond the 8,000 sims/sec MCTS target, enabling **12k-22k sims/sec** total throughput on RTX 3060 Ti hardware **without sacrificing superhuman play quality**.

**Key Insight from review.txt**: GPU is only 32.8% of total time. Doubling NN speed (via lighter architecture) directly raises throughput, making 10k+ sims/sec achievable.

---

## 1. Architecture Options (Ranked by Safety)

### Option A: RepVGG/ECA ResNet (Safest, +25-50% Speed)

**Concept**: Train with multi-branch residuals (3×3 + 1×1 + identity), **fuse to single 3×3 conv** at inference.

**Design**:
- **Training Block**: Conv3×3 + Conv1×1 + identity, BN each, **ECA** after sum
- **Inference Block**: Fused single Conv3×3 + BN (branches merged)
- **ECA (Efficient Channel Attention)**: 1-D conv on global-avg pooled channels (near-zero overhead vs. SE)

**Expected Gains**:
- Throughput: **+25-50% vs. current ResNet+SE** at same depth/width
- Strength: Minimal loss (often improves due to richer training graph)
- FLOPS: ~30% reduction (fewer operations, better memory bandwidth)

**Implementation** (Gomoku 15×15):
```python
# Entry/Exit blocks
blocks = [
    RepECA(C=64, use_eca=True),  # ×2 blocks
    RepECA(C=64, use_eca=True),
]

# Middle blocks (standard residual)
for _ in range(8):
    blocks.append(RepECA(C=64, use_eca=True))

# Export for inference
for block in blocks:
    if isinstance(block, RepECA):
        block.switch_to_deploy()  # Fuse branches → single conv
```

**Validation**:
- Train 10k games, export fused model
- Measure: FP16 inference time per batch-64
- Target: 30.7ms → <20ms (1.5× speedup)
- Quality: ELO ≥ baseline (1000-game match)

---

### Option B: Ghost Bottleneck + ShuffleV2 (Maximum Speed, +40-80%)

**Concept**: Replace heavy convs with **Ghost modules** (generate many feature maps from few intrinsic maps) + **ShuffleNetV2 channel-split** bottlenecks.

**Design**:
- **Entry 2 blocks**: RepECA (clean features)
- **Middle 6-8 blocks**: Ghost bottlenecks (FLOP reduction)
- **Exit 2 blocks**: RepECA (policy/value quality)
- **Middle alternative**: ShuffleV2 units (memory bandwidth reduction)

**Ghost Module**:
```python
class GhostModule(nn.Module):
    def __init__(self, in_ch, out_ch, ratio=2):
        mid = out_ch // ratio
        self.primary = Conv1×1(in_ch, mid) + BN + ReLU
        self.cheap = DWConv3×3(mid, out_ch - mid, groups=mid) + BN + ReLU

    def forward(self, x):
        y = self.primary(x)
        z = self.cheap(y)
        return torch.cat([y, z], dim=1)  # Concatenate
```

**Expected Gains**:
- Throughput: **+40-80% vs. RepECA** (model-only)
- Strength: Small dent, mitigated by auxiliary tactical heads (below)
- FLOPS: 50-60% reduction

**Risk Mitigation**:
- Add **auxiliary tactical heads** (threat detection) to preserve pattern recognition
- Use **self-distillation** (EMA teacher) to stabilize training

---

### Option C: Two-Tier Evaluator Cascade (×1.5-2.5 Throughput)

**Concept**: Run **micro-net** first (C=24-32, B=2-3); if decisive, accept output; else escalate to main net.

**Gate Logic**:
```python
if entropy(micro_policy) < τ or threat_detected:
    return micro_result  # Accept (30-60% of positions)
else:
    return main_net(state)  # Escalate (40-70%)
```

**Micro-Net Design** (Gomoku):
- Stem: Conv3×3, C=32
- Body: 2× RepECA blocks, C=32
- Heads: Policy (grid 15×15), Value (tanh)
- Size: ~0.1M params (vs. 2M main net)

**Expected Gains**:
- Throughput: **×1.5-2.5** end-to-end (many trivial positions skip main net)
- Strength: Preserve superhuman (conservative gate, MCTS absorbs noise)

**Calibration**:
- Set τ (entropy threshold) via validation: 95% top-move agreement with main net
- Threat detection: Immediate five, open four (aux head output)

---

### Option D: Early-Exit Heads (×1.2-1.6 Throughput, Stackable)

**Concept**: Add **auxiliary policy/value heads** at earlier depths (block 3, block 6). If confident, stop forward.

**Implementation**:
```python
class FastMCTSNet(nn.Module):
    def __init__(self, ...):
        self.blocks = nn.ModuleList([...])  # 12 blocks
        self.early_exit_3 = PolicyValueHead(C=64)
        self.early_exit_6 = PolicyValueHead(C=64)
        self.main_head = PolicyValueHead(C=64)

    def forward(self, x, inference=True):
        for i, block in enumerate(self.blocks):
            x = block(x)
            if inference and i == 2:  # Block 3
                policy, value = self.early_exit_3(x)
                if abs(value) > 0.9 or entropy(policy) < 0.7:
                    return policy, value  # Exit early
            # Similar check at block 6

        return self.main_head(x)  # Full forward
```

**Expected Gains**:
- Throughput: **×1.2-1.6** average (position-dependent)
- Strength: Negligible impact (exits only when confident)

**Training**:
- Attach deep supervision (small λ=0.1 on aux losses)
- Freeze thresholds at inference to hit target sims/sec

---

## 2. Auxiliary Tactical Heads (Strength Preservation)

**Problem**: Lighter architectures (Ghost, Shuffle) risk losing tactical pattern recognition.

**Solution**: Add **multi-label classifiers** that predict:
- Immediate five / open four / open three (per player)
- Run-length-to-five maps (discretized bins)

**Implementation**:
```python
class ThreatHead(nn.Module):
    def __init__(self, in_ch):
        self.conv = Conv1×1(in_ch, 6)  # 6 threat types

    def forward(self, x):
        return torch.sigmoid(self.conv(x))  # Multi-label

# Loss
loss = L_policy + L_value + 0.1 * L_threats
```

**Effect**:
- Small nets learn **threat semantics** faster
- Recovers "tactical bite" lost by Ghost/Shuffle substitutions
- Boosts cascade gate reliability (threat flags guide early-exit)

---

## 3. Game-Specific Configurations

### Gomoku Freestyle (Lighter, Faster)

```python
net = FastMCTSNet(
    in_planes=36,
    trunk_channels=64,
    blocks=(2, 8, 2),  # Entry, middle, exit
    middle_kind="ghost",
    board_size=(15, 15),
    early_exit_at=[4, 8],
    exit_entropy_thr=0.75,
    exit_value_thr=0.90,
)
```

**Expected**: 1.4-1.8× model speedup → 12-14k sims/sec total

### Renju/Omok (Pattern-Strong)

```python
net = FastMCTSNet(
    in_planes=36,
    trunk_channels=64,
    blocks=(2, 8, 2),
    middle_kind="ghost",
    middle_schedule=["ghost"]*6 + ["shuffle"]*2,  # Hybrid
    board_size=(15, 15),
    early_exit_at=[6],  # Conservative
    exit_entropy_thr=0.65,
    exit_value_thr=0.92,
    aux_threat_heads=True,  # Enable tactical heads
)
```

**Expected**: 1.3-1.6× model speedup + aux heads → superhuman retained

### Chess 8×8

```python
net = FastMCTSNet(
    in_planes=112,
    trunk_channels=64,
    blocks=(2, 8, 2),
    middle_kind="ghost",
    policy_kind="flat",
    action_dim=4672,
    early_exit_at=[6],
    exit_entropy_thr=0.90,  # Very conservative
    exit_value_thr=0.95,
)
```

**Expected**: 1.2-1.5× model speedup (chess benefits from richer priors)

### Go 9×9

```python
net = FastMCTSNet(
    in_planes=17,
    trunk_channels=64,
    blocks=(2, 8, 2),
    middle_kind="ghost",
    board_size=(9, 9),
    early_exit_at=[4, 8],
    exit_entropy_thr=0.85,
    exit_value_thr=0.92,
)
```

**Expected**: 1.5-2.0× model speedup (9×9 is compute-light)

---

## 4. Implementation Checklist

**Phase 7 Tasks** (Future):

- [ ] **T031**: Implement RepECA blocks with BN folding
- [ ] **T032**: Implement Ghost bottleneck module
- [ ] **T033**: Implement early-exit heads with confidence gating
- [ ] **T034**: Implement two-tier cascade (micro-net + main net)
- [ ] **T035**: Implement auxiliary threat heads
- [ ] **T036**: Train and validate all architectures (ELO ≥ baseline)
- [ ] **T037**: Benchmark throughput (target 12k-22k sims/sec)

**Training Protocol**:
```python
# Losses
L = L_policy + L_value + 0.1*L_threats + 0.1*L_aux_early_exits

# Self-distillation (optional)
teacher = EMA(model, decay=0.999)
L += 0.2 * KL(policy || teacher_policy) + 0.1 * MSE(value - teacher_value)

# Data augmentation
augment = dihedral_symmetry + noise_planes_dropout

# Export
model.switch_to_deploy()  # Fuse RepECA branches
torch.save(model, "model_optimized.pth")
```

**A/B Validation**:
- Run 1000-game match vs. current SE-ResNet at fixed visits
- Verify ELO ≥ baseline (or within -10 ELO if sims/sec ≥1.7×)

---

## 5. Expected Performance (3060 Ti, FP16, Batching)

| Architecture | Model Speed | MCTS Speed (Current: 8k) | Total Throughput |
|--------------|-------------|--------------------------|------------------|
| Baseline (SE-ResNet) | 1.0× | 8k sims/sec | 8k sims/sec |
| (A) RepECA | 1.25-1.5× | 8k × 1.25 = 10k | **10-12k sims/sec** |
| (B) Ghost+Shuffle | 1.4-1.8× | 8k × 1.5 = 12k | **12-14k sims/sec** |
| (C) Cascade (on top of A/B) | ×1.5-2.5 | 12k × 1.8 = 21.6k | **18-22k sims/sec** |
| (D) Early-Exit (on top of A/B) | ×1.2-1.6 | 12k × 1.3 = 15.6k | **15-18k sims/sec** |

**Combined (B+C)**: **18-22k sims/sec** realistic with superhuman quality preserved

---

## 6. References

All content derived from review.txt lines 621-1396:
- RepVGG/DBB reparameterization: lines 1267-1279
- Ghost bottlenecks: lines 1281-1289
- ShuffleNetV2: lines 1291-1300
- Two-tier cascade: lines 1302-1310
- Early-exit heads: lines 1313-1320
- Auxiliary tactical heads: lines 1322-1335
- Game configs: lines 1090-1244
- Training checklist: lines 1377-1385

**Status**: Documented for Phase 7 (post-8k target achieved)
```

### Action 6: Update README.md

**Update document index and status**:
```markdown
## Document Index

### Active Specifications (v2.0)
- ✅ **spec.md** - Functional requirements (503 lines, v2.0)
- ✅ **CONSTITUTION.md** - Immutable constraints (729 lines, v2.0)
- ✅ **plan.md** - Technical implementation plan (2,398 lines, v1.0)
- ✅ **tasks.md** - Detailed task breakdown (2,747 lines, v1.0)

### Supporting Documents
- ✅ **CLARIFICATIONS.md** - Resolved ambiguities (633 lines)
- ✅ **ACCEPTANCE_CHECKLIST.md** - Validation checklist (860 lines)
- ✅ **TRACEABILITY_MATRIX.md** - Cross-check analysis (524 lines)
- ✅ **data-model.md** - Memory layout specifications (542 lines)
- ✅ **research.md** - Architecture decisions (521 lines)
- ✅ **quickstart.md** - Build and validation guide (454 lines)

### Future Phases
- 🔮 **NEURAL_NETWORK_OPTIMIZATION.md** - NN architecture redesign (Phase 7)

### Archived (Superseded)
- 📦 **archive/SPECIFICATION.md** - Old spec (superseded by spec.md v2.0)
- 📦 **archive/TASKS.md** - Old tasks (superseded by tasks.md v1.0)
- 📦 **archive/TECHNICAL_PLAN.md** - Old plan (superseded by plan.md v1.0)
```

---

## Execution Plan

1. ✅ **Create this plan document** (DOCUMENTATION_UPDATE_PLAN.md)
2. ⏳ **Delete obsolete files** (move to archive/)
3. ⏳ **Update spec.md** (add Section 6: Advanced Optimizations)
4. ⏳ **Update plan.md** (add B2.4, B3.4, B5.4)
5. ⏳ **Update tasks.md** (add T014a, T030a)
6. ⏳ **Create NEURAL_NETWORK_OPTIMIZATION.md** (Phase 7 spec)
7. ⏳ **Update README.md** (document index)
8. ⏳ **Verify all cross-references** (links, section numbers)

---

**Estimated Time**: 2-3 hours
**Impact**: Complete documentation alignment with review.txt
**Result**: Single source of truth, no contradictions, all gaps resolved
