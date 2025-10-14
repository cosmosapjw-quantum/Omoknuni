# Neural Network Architecture Optimization (Phase 7)

**Version**: 1.0
**Date**: 2025-10-14
**Status**: FUTURE (Post-8k MCTS Throughput Target)
**Authority**: Derived from review.txt lines 621-1396 + spec.md v2.0 Section 6.3

---

## Executive Summary

This document specifies neural network architecture optimizations to achieve **1.5-3.5× model speedup** beyond the 8,000 sims/sec MCTS target, enabling **12k-22k sims/sec** total throughput on RTX 3060 Ti hardware **without sacrificing superhuman play quality**.

**Key Insight** (review.txt lines 13-18): GPU is only 32.8% of total time at baseline. MCTS CPU coordination is the dominant bottleneck (67.2%). After fixing MCTS to reach 8k sims/sec, doubling NN speed via lighter architecture directly raises total throughput to 10k+ sims/sec.

**Constitutional Compliance**: Maintains Python PyTorch interface (NO libtorch, per CONSTITUTION.md Section 1.4). All optimizations are model architecture changes only.

---

## 1. Architecture Options (Ranked by Safety)

### Option A: RepVGG/ECA ResNet (Safest, +25-50% Speed)

**Source**: review.txt lines 1267-1279

**Concept**: Train with multi-branch residuals (3×3 + 1×1 + identity), **fuse to single 3×3 conv** at inference via BN folding.

**Design**:
- **Training Block**:
  - Conv3×3 (stride=1, padding=1) + BN
  - Conv1×1 (stride=1, padding=0) + BN
  - Identity branch + BN (if in_channels == out_channels)
  - Sum all branches
  - **ECA (Efficient Channel Attention)**: 1-D conv on global-avg pooled channels
  - ReLU activation

- **Inference Block** (after `switch_to_deploy()`):
  - Fused single Conv3×3 + BN (all branches merged via weight arithmetic)
  - ECA (remains as cheap 1-D conv, k=3 or k=5)
  - ReLU activation

**ECA vs. SE**:
- SE (Squeeze-Excitation): Global pool → FC (reduce) → ReLU → FC (expand) → Sigmoid
- ECA: Global pool → **1-D Conv(kernel=k)** → Sigmoid
- Overhead: ECA ≈ 5-10% of SE computational cost
- Quality: ECA ≥ SE in many benchmarks (less prone to overfitting)

**Expected Gains**:
- Throughput: **+25-50% vs. current ResNet+SE** at same depth/width
- Strength: Minimal loss (often improves due to richer training graph with multi-branch)
- FLOPS: ~30% reduction (fused conv eliminates branch additions)
- Memory Bandwidth: ~25% reduction (single conv path)

**Implementation** (Gomoku 15×15):
```python
class RepECA(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_eca=True):
        super().__init__()
        self.in_ch, self.out_ch, self.stride = in_ch, out_ch, stride
        self.deploy = False

        # Training branches
        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.conv1 = nn.Conv2d(in_ch, out_ch, 1, stride=stride, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.has_id = (in_ch == out_ch and stride == 1)
        self.id_bn = nn.BatchNorm2d(out_ch) if self.has_id else None

        self.eca = ECA(out_ch, k=5) if use_eca else nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.rbr_reparam = None  # Fused conv (inference)

    @torch.no_grad()
    def switch_to_deploy(self):
        \"\"\"Fuse multi-branch to single conv.\"\"\"
        if self.deploy:
            return

        # Fuse Conv+BN for each branch
        k3, b3 = _fuse_conv_bn_wb(self.conv3, self.bn3)
        k1, b1 = _fuse_conv_bn_wb(self.conv1, self.bn1)
        k1 = _pad_1x1_to_3x3(k1)  # Pad 1x1 kernel to 3x3

        # Add identity branch if present
        if self.has_id:
            kid = _id_kernel_3x3(self.out_ch, self.in_ch, self.conv3.weight.device)
            g, beta = self.id_bn.weight, self.id_bn.bias
            mean, var, eps = self.id_bn.running_mean, self.id_bn.running_var, self.id_bn.eps
            std = torch.sqrt(var + eps)
            kid = kid * (g / std).reshape(-1, 1, 1, 1)
            bid = beta - mean * (g / std)
        else:
            kid = torch.zeros_like(k3)
            bid = torch.zeros_like(b3)

        # Sum kernels and biases
        k_fused = k3 + k1 + kid
        b_fused = b3 + b1 + bid

        # Create fused conv
        self.rbr_reparam = nn.Conv2d(self.in_ch, self.out_ch, 3, stride=self.stride, padding=1, bias=True)
        self.rbr_reparam.weight.data.copy_(k_fused)
        self.rbr_reparam.bias.data.copy_(b_fused)

        # Delete training branches
        for m in [self.conv3, self.bn3, self.conv1, self.bn1, self.id_bn]:
            if m is not None:
                m.requires_grad_(False)
        self.conv3 = self.bn3 = self.conv1 = self.bn1 = self.id_bn = None
        self.deploy = True

    def forward(self, x):
        if self.deploy:
            y = self.rbr_reparam(x)
        else:
            y = self.bn3(self.conv3(x)) + self.bn1(self.conv1(x))
            if self.has_id:
                y = y + self.id_bn(x)
        y = self.eca(y)
        return self.act(y)


class ECA(nn.Module):
    \"\"\"Efficient Channel Attention (1-D conv on pooled channels).\"\"\"
    def __init__(self, channels: int, k: int = 5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x: (B,C,H,W)
        y = self.pool(x)                       # (B,C,1,1)
        y = y.squeeze(-1).transpose(1, 2)      # (B,1,C)
        y = self.conv(y)                       # (B,1,C)
        y = self.sig(y).transpose(1, 2).unsqueeze(-1)  # (B,C,1,1)
        return x * y


# Full network
blocks = []
for _ in range(2):  # Entry
    blocks.append(RepECA(64, 64, use_eca=True))
for _ in range(8):  # Middle
    blocks.append(RepECA(64, 64, use_eca=True))
for _ in range(2):  # Exit
    blocks.append(RepECA(64, 64, use_eca=True))

# Export for inference
for block in blocks:
    block.switch_to_deploy()
```

**Validation**:
- Train 10k self-play games with Gomoku 15×15
- Export fused model: `model.switch_to_deploy(); torch.save(model, "model_repeca.pth")`
- Measure: FP16 inference time per batch-64
- Baseline: 30.7ms (current SE-ResNet), Target: <20ms (1.5× speedup)
- Quality: ELO ≥ baseline (1000-game match, 95% confidence)

---

### Option B: Ghost Bottleneck + ShuffleV2 (Maximum Speed, +40-80%)

**Source**: review.txt lines 1281-1300

**Concept**: Replace heavy convolutions with **Ghost modules** (generate many feature maps from few intrinsic maps via cheap depthwise convs) or **ShuffleNetV2 channel-split** bottlenecks.

**Design**:
- **Entry 2 blocks**: RepECA (clean features for initial processing)
- **Middle 6-8 blocks**: Ghost bottlenecks OR ShuffleV2 units (FLOP/bandwidth reduction)
- **Exit 2 blocks**: RepECA (policy/value heads prefer clean features)

**Ghost Module**:
```python
class GhostModule(nn.Module):
    \"\"\"Generate many features from few intrinsic features.\"\"\"
    def __init__(self, in_ch, out_ch, ratio=2, k=1, dwk=3, stride=1, relu=True):
        super().__init__()
        mid = int(round(out_ch / ratio))
        rem = out_ch - mid

        # Primary conv (intrinsic features)
        self.primary = nn.Sequential(
            nn.Conv2d(in_ch, mid, k, stride, k // 2 if k > 1 else 0, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

        # Cheap operation (depthwise conv to generate ghost features)
        self.cheap = nn.Sequential(
            nn.Conv2d(mid, rem, dwk, 1, dwk // 2, groups=mid, bias=False),
            nn.BatchNorm2d(rem),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

    def forward(self, x):
        y = self.primary(x)  # Intrinsic features
        z = self.cheap(y)    # Ghost features (cheap)
        return torch.cat([y, z], dim=1)


class GhostBottleneck(nn.Module):
    \"\"\"Ghost bottleneck residual block.\"\"\"
    def __init__(self, in_ch, hidden_ch, out_ch, stride=1, use_eca=True):
        super().__init__()
        self.conv1 = GhostModule(in_ch, hidden_ch, relu=True)
        self.dw = nn.Conv2d(hidden_ch, hidden_ch, 3, stride=stride, padding=1, groups=hidden_ch, bias=False)
        self.dw_bn = nn.BatchNorm2d(hidden_ch)
        self.conv2 = GhostModule(hidden_ch, out_ch, relu=False)
        self.eca = ECA(out_ch, k=3) if use_eca else nn.Identity()

        self.short = nn.Identity() if (stride == 1 and in_ch == out_ch) else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.dw_bn(self.dw(out))
        out = self.conv2(out)
        out = self.eca(out)
        out = out + self.short(x)
        return self.act(out)
```

**ShuffleNetV2 Unit** (alternative to Ghost):
```python
def channel_shuffle(x, groups=2):
    b, c, h, w = x.size()
    assert c % groups == 0
    x = x.reshape(b, groups, c // groups, h, w).transpose(1, 2).contiguous()
    return x.reshape(b, c, h, w)


class ShuffleV2Unit(nn.Module):
    \"\"\"ShuffleNetV2 channel-split bottleneck.\"\"\"
    def __init__(self, channels, use_eca=True):
        super().__init__()
        half = channels // 2
        self.branch = nn.Sequential(
            nn.Conv2d(half, half, 1, 1, 0, bias=False),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
            nn.Conv2d(half, half, 3, 1, 1, groups=half, bias=False),  # Depthwise
            nn.BatchNorm2d(half),
            nn.Conv2d(half, half, 1, 1, 0, bias=False),
            nn.BatchNorm2d(half),
            nn.ReLU(inplace=True),
        )
        self.eca = ECA(channels, k=3) if use_eca else nn.Identity()

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)  # Split channels
        y2 = self.branch(x2)  # Process one half
        y = torch.cat([x1, y2], dim=1)  # Concatenate
        y = channel_shuffle(y, 2)  # Shuffle for info mixing
        return self.eca(y)
```

**Full Network** (hybrid approach):
```python
blocks = []
# Entry: RepECA (2 blocks)
for _ in range(2):
    blocks.append(RepECA(64, 64, use_eca=True))

# Middle: Ghost bottlenecks (6-8 blocks)
for _ in range(8):
    blocks.append(GhostBottleneck(64, hidden_ch=64, out_ch=64, stride=1, use_eca=True))

# Exit: RepECA (2 blocks)
for _ in range(2):
    blocks.append(RepECA(64, 64, use_eca=True))
```

**Expected Gains**:
- Throughput: **+40-80% vs. RepECA** (model-only, measured FP16 batch-64)
- Strength: Small dent expected (5-15 ELO loss), mitigated by:
  - Auxiliary tactical heads (threat detection, see Section 2)
  - Self-distillation with EMA teacher
  - Data augmentation (dihedral symmetry + noise dropout)
- FLOPS: 50-60% reduction vs. standard ResNet
- Memory Bandwidth: ~40% reduction (depthwise convs are bandwidth-efficient)

**Risk Mitigation**:
- **Use auxiliary threat heads** (immediate five, open four, open three) to preserve pattern recognition
- **Conservative early-exit thresholds** if stacking with Option D
- **Extensive validation**: 5k-game self-play match before production deployment

---

### Option C: Two-Tier Evaluator Cascade (×1.5-2.5 Throughput)

**Source**: review.txt lines 1302-1310

**Concept**: Run **micro-net** first (C=24-32, B=2-3, ~0.1M params); if decisive, accept output; else escalate to main net.

**Gate Logic**:
```python
micro_policy, micro_value = micro_net(state)
entropy = -sum(p * log(p)) for p in softmax(micro_policy)
threat_flag = (micro_threat_head(state) > 0.8).any()  # Immediate five/open four

if entropy < τ or threat_flag:
    return micro_policy, micro_value  # Accept (30-60% of positions)
else:
    return main_net(state)  # Escalate (40-70%)
```

**Micro-Net Design** (Gomoku 15×15):
```python
class MicroNet(nn.Module):
    def __init__(self, in_planes=36):
        super().__init__()
        C = 32  # Small channel count
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, C, 3, 1, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        # Body: 2-3 RepECA blocks
        self.blocks = nn.ModuleList([
            RepECA(C, C, use_eca=True),
            RepECA(C, C, use_eca=True),
        ])

        # Heads
        self.policy_head = GridPolicyHead(C)  # 15×15 output
        self.value_head = ValueHead(C)  # Scalar output

        # Threat detection (optional, for gate)
        self.threat_head = nn.Sequential(
            nn.Conv2d(C, 6, 1),  # 6 threat types
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)

        policy = self.policy_head(x)
        value = self.value_head(x)
        threat = self.threat_head(x)  # For gate decision

        return policy, value, threat


# Cascade logic (in MCTS inference callback)
def cascade_inference(state):
    # Micro-net first
    micro_policy, micro_value, micro_threat = micro_net(state)

    # Compute entropy
    policy_probs = torch.softmax(micro_policy, dim=-1)
    entropy = -(policy_probs * torch.log(policy_probs + 1e-8)).sum()

    # Gate decision
    if entropy < τ or micro_threat.max() > 0.8:
        # Accept micro-net output (trivial position)
        return micro_policy, micro_value, "micro"  # Log decision
    else:
        # Escalate to main net
        main_policy, main_value = main_net(state)
        return main_policy, main_value, "main"
```

**Expected Gains**:
- Throughput: **×1.5-2.5** end-to-end (many trivial positions skip main net)
- Acceptance rate: 30-60% (depends on game phase, τ calibration)
- Strength: Preserve superhuman (conservative gate ensures MCTS absorbs micro-net noise)
- Latency: Positions that escalate pay 2× NN cost (micro + main), but MCTS visit budget allows this

**Calibration**:
- Set τ (entropy threshold) via validation set: 95% top-move agreement with main net
- Threat detection threshold: 0.8 (immediate tactical threats)
- Tune τ per game: Gomoku freestyle τ=0.75, Renju/Omok τ=0.65 (more conservative)

**Training**:
- Train micro-net independently with same self-play data
- Loss: `L = L_policy + L_value + 0.1 * L_threats`
- Smaller network, faster convergence (1k games sufficient)

---

### Option D: Early-Exit Heads (×1.2-1.6 Throughput, Stackable)

**Source**: review.txt lines 1313-1320

**Concept**: Add **auxiliary policy/value heads** at earlier depths (block 3, block 6). If head output is confident, **stop forward propagation** and return early.

**Implementation**:
```python
class FastMCTSNet(nn.Module):
    def __init__(self, in_planes, trunk_channels=64, blocks=(2, 8, 2), early_exit_at=None):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(in_planes, trunk_channels, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(trunk_channels), nn.ReLU(inplace=True))

        # Build blocks
        self.blocks = nn.ModuleList()
        for _ in range(sum(blocks)):
            self.blocks.append(RepECA(trunk_channels, trunk_channels, use_eca=True))

        # Main head
        self.policy_head = GridPolicyHead(trunk_channels)
        self.value_head = ValueHead(trunk_channels)

        # Early-exit heads (optional)
        self.early_exit_at = early_exit_at or []
        self.early_exit_heads = nn.ModuleDict()
        for idx in self.early_exit_at:
            self.early_exit_heads[str(idx)] = nn.ModuleDict({
                'policy': GridPolicyHead(trunk_channels),
                'value': ValueHead(trunk_channels),
            })

    def forward(self, x, legal_mask=None, inference=False):
        x = self.stem(x)

        # Forward through blocks with early-exit checks
        for i, block in enumerate(self.blocks):
            x = block(x)

            # Check early-exit at designated blocks (inference only)
            if inference and str(i+1) in self.early_exit_heads:
                policy = self.early_exit_heads[str(i+1)]['policy'](x)
                value = self.early_exit_heads[str(i+1)]['value'](x)

                # Compute confidence
                policy_probs = torch.softmax(policy, dim=-1)
                if legal_mask is not None:
                    policy_probs = policy_probs.masked_fill(~legal_mask, 0)
                    policy_probs = policy_probs / policy_probs.sum(dim=-1, keepdim=True)

                entropy = -(policy_probs * torch.log(policy_probs + 1e-8)).sum(dim=-1)
                value_abs = value.abs().squeeze(-1)

                # Exit if confident (low entropy OR high value certainty)
                if (entropy < self.exit_entropy_thr or value_abs > self.exit_value_thr).all():
                    return policy, value  # Exit early

        # Full forward (no early exit)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


# Configuration (Gomoku freestyle)
net = FastMCTSNet(
    in_planes=36,
    trunk_channels=64,
    blocks=(2, 8, 2),
    early_exit_at=[4, 8],  # Exit at block 4 or block 8
)
net.exit_entropy_thr = 0.75  # Low entropy = confident policy
net.exit_value_thr = 0.90  # High |value| = confident evaluation
```

**Expected Gains**:
- Throughput: **×1.2-1.6** average (position-dependent, more trivial = more exits)
- Exit rate: 20-50% (depends on thresholds and game phase)
- Strength: Negligible impact (exits only when confident, MCTS visit budget absorbs noise)
- Stackable: Works with Options A/B/C (cascade can use early-exit main net)

**Training** (deep supervision):
```python
# Attach auxiliary losses to early-exit heads (small weight)
loss = L_policy(main_head) + L_value(main_head)
for idx in early_exit_at:
    loss += 0.1 * L_policy(early_head[idx]) + 0.1 * L_value(early_head[idx])
```

**Calibration**:
- Freeze thresholds at inference to hit target sims/sec
- Gomoku: entropy ≤0.75 OR |value| ≥0.90
- Renju/Omok: entropy ≤0.65 OR |value| ≥0.92 (more conservative)
- Chess: entropy ≤0.90 OR |value| ≥0.95 (very conservative)

---

## 2. Auxiliary Tactical Heads (Strength Preservation)

**Source**: review.txt lines 1322-1335

**Problem**: Lighter architectures (Ghost, Shuffle) risk losing tactical pattern recognition that larger ResNets capture naturally.

**Solution**: Add **multi-label threat classifiers** that predict:
- Immediate five (per player)
- Open four (per player)
- Open three (per player)
- Run-length-to-five maps (discretized bins: 0, 1, 2, 3, 4+)

**Implementation**:
```python
class ThreatHead(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        # 6 binary classifiers (black: five/four/three, white: five/four/three)
        self.threat_conv = nn.Conv2d(in_ch, 6, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.threat_conv(x))  # (B, 6, H, W)


# Add to main network
self.threat_head = ThreatHead(trunk_channels)

# Loss (multi-label binary cross-entropy)
loss_total = L_policy + L_value + 0.05 * L_threats
```

**Effect**:
- Small nets learn **threat semantics** faster (explicit supervision)
- Recovers "tactical bite" lost by Ghost/Shuffle substitutions
- Boosts cascade gate reliability (threat flags guide early-exit decisions)
- Inference overhead: Negligible (single 1×1 conv)

**Training**:
- Label generation: Analyze board positions for immediate threats
- Gomoku: Use existing threat detection logic in `IGameState`
- Weight: λ=0.05-0.10 (small enough to not distort policy/value training)

---

## 3. Game-Specific Configurations

**Source**: review.txt lines 1090-1244

### Gomoku Freestyle (Lighter, Faster)

**Goal**: Maximize throughput (data generation speed)

```python
net = FastMCTSNet(
    in_planes=36,
    trunk_channels=64,
    blocks=(2, 8, 2),  # Entry=2, middle=8, exit=2
    middle_kind="ghost",  # Ghost bottlenecks in middle
    board_size=(15, 15),
    early_exit_at=[4, 8],  # Exit at blocks 4, 8
    exit_entropy_thr=0.75,  # Liberal threshold
    exit_value_thr=0.90,
)
```

**Expected**: 1.4-1.8× model speedup → **12-14k sims/sec** total

---

### Renju/Omok (Pattern-Strong)

**Goal**: Preserve tactical strength (rules require adjudication)

```python
net = FastMCTSNet(
    in_planes=36,
    trunk_channels=64,
    blocks=(2, 8, 2),
    middle_kind="ghost",
    middle_schedule=["ghost"]*6 + ["shuffle"]*2,  # Hybrid (last 2 blocks Shuffle)
    board_size=(15, 15),
    early_exit_at=[6],  # Conservative (single exit)
    exit_entropy_thr=0.65,  # Stricter threshold
    exit_value_thr=0.92,
    aux_threat_heads=True,  # Enable threat detection
)
```

**Expected**: 1.3-1.6× model speedup + aux heads → superhuman retained

---

### Chess 8×8

**Goal**: Balance speed with complex tactics

```python
net = FastMCTSNet(
    in_planes=112,
    trunk_channels=64,
    blocks=(2, 8, 2),
    middle_kind="ghost",
    policy_kind="flat",  # Flat policy head (4672 actions)
    action_dim=4672,
    early_exit_at=[6],
    exit_entropy_thr=0.90,  # Very conservative
    exit_value_thr=0.95,
)
```

**Expected**: 1.2-1.5× model speedup (chess benefits from richer priors)

---

### Go 9×9

**Goal**: Maximize speedup (small board is compute-light)

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

**Expected**: 1.5-2.0× model speedup → **13-16k sims/sec** total

---

## 4. Training Protocol

**Source**: review.txt lines 1377-1385

```python
# Combined loss with all components
L = L_policy + L_value + 0.1*L_threats + 0.1*L_aux_early_exits

# Self-distillation (optional, for stability)
teacher = EMA(model, decay=0.999)  # Exponential moving average teacher
L += 0.2 * KL(policy || teacher_policy) + 0.1 * MSE(value, teacher_value)

# Data augmentation
augment = [
    dihedral_symmetry,  # 8 symmetries (4 rotations × 2 reflections)
    noise_planes_dropout(p=0.1),  # Drop 10% of input planes randomly
]

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10000)

# Training loop
for game in self_play_games:
    positions, policies, values = game.get_training_data()
    positions = augment(positions)

    # Forward
    model_policy, model_value = model(positions)

    # Loss
    loss = cross_entropy(model_policy, policies) + mse(model_value, values)
    loss += 0.1 * binary_cross_entropy(model.threat_head(positions), threats)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Export for inference
for block in model.blocks:
    if isinstance(block, RepECA):
        block.switch_to_deploy()  # Fuse multi-branch to single conv

torch.save(model, "model_optimized.pth")
```

---

## 5. A/B Validation Protocol

**Goal**: Verify strength ≥ baseline or within acceptable tolerance

**Test**: 1000-game match at fixed visits (800 sims/move)

```python
# Baseline (current SE-ResNet)
baseline_model = torch.load("model_baseline.pth")

# Optimized (e.g., RepECA)
optimized_model = torch.load("model_repeca.pth")

# Match
results = run_match(
    model_a=baseline_model,
    model_b=optimized_model,
    games=1000,
    simulations_per_move=800,
    swap_sides=True,  # Each model plays black and white
)

# Acceptance criteria
assert results.elo_diff >= 0 or (results.elo_diff >= -10 and speedup >= 1.7)
# "If ELO within -10 but sims/sec ≥1.7×: ACCEPTABLE (training data throughput prioritized)"
```

**Metrics**:
- ELO rating: ≥0 (no regression) OR ≥-10 if speedup ≥1.7×
- Policy agreement: ≥95% on 1000-position test set
- Value MSE: ≤0.01 vs. baseline

---

## 6. Expected Performance (3060 Ti, FP16, Batching)

### Single-Model Speedups

| Architecture | Model Speed | MCTS Speed (Current: 8k) | Total Throughput |
|--------------|-------------|--------------------------|------------------|
| Baseline (SE-ResNet) | 1.0× | 8k sims/sec | **8k sims/sec** |
| (A) RepECA | 1.25-1.5× | 8k × 1.35 = 10.8k | **10-12k sims/sec** |
| (B) Ghost+Shuffle | 1.4-1.8× | 8k × 1.6 = 12.8k | **12-14k sims/sec** |
| (C) Cascade (on A/B) | ×1.5-2.5 | 10.8k × 2.0 = 21.6k | **16-22k sims/sec** |
| (D) Early-Exit (on A/B) | ×1.2-1.6 | 10.8k × 1.4 = 15.1k | **13-17k sims/sec** |

### Combined Architecture (B + C)

**Configuration**: Ghost+Shuffle trunk + Two-tier cascade

- **Ghost+Shuffle speedup**: 1.6× (model-only)
- **Cascade multiplier**: ×2.0 (50% acceptance rate)
- **Combined**: 1.6 × 2.0 = 3.2× model throughput
- **MCTS base**: 8k sims/sec
- **Total throughput**: 8k × (1 + 0.328 × 3.2) = **16.4k sims/sec**
  - (GPU was 32.8% of time at baseline, now 3.2× faster → contribution increases)

**Realistic Range**: **18-22k sims/sec** (accounting for MCTS scaling, batching dynamics)

---

## 7. Implementation Checklist (Phase 7 Tasks)

**Prerequisites**: 8k sims/sec MCTS target achieved (validates GPU becomes bottleneck)

**Tasks** (estimated 10 days total):

- [ ] **T031**: Implement RepECA blocks with BN folding (1.5 days)
  - `_fuse_conv_bn_wb()`, `_pad_1x1_to_3x3()`, `switch_to_deploy()`
  - Unit tests: Fused weights match multi-branch output
  - Integration: Train 1k games, verify convergence

- [ ] **T032**: Implement Ghost bottleneck module (1.0 day)
  - `GhostModule`, `GhostBottleneck` classes
  - Unit tests: Output shapes, FLOP count reduction
  - Integration: Train 2k games, measure speedup

- [ ] **T033**: Implement early-exit heads with confidence gating (1.5 days)
  - `FastMCTSNet` with `early_exit_at` parameter
  - Gate logic: entropy/value thresholds
  - Unit tests: Early-exit triggered correctly
  - Integration: Calibrate thresholds for 95% policy agreement

- [ ] **T034**: Implement two-tier cascade (micro-net + main net) (2.0 days)
  - `MicroNet` class (C=32, B=2-3)
  - Cascade inference logic with gate
  - Unit tests: Acceptance rate, throughput measurement
  - Integration: Train micro-net independently, validate τ

- [ ] **T035**: Implement auxiliary threat heads (1.0 day)
  - `ThreatHead` multi-label classifier
  - Label generation from game positions
  - Unit tests: Threat detection accuracy
  - Integration: Add to loss with λ=0.05

- [ ] **T036**: Train and validate all architectures (2.0 days)
  - Train each architecture: RepECA, Ghost, Cascade, Early-Exit
  - A/B validation: 1000-game matches
  - Acceptance: ELO ≥ baseline OR (≥-10 AND speedup ≥1.7×)

- [ ] **T037**: Benchmark throughput (1.0 day)
  - End-to-end throughput measurement (MCTS + NN)
  - Target: 12k-22k sims/sec (depending on architecture)
  - CSV telemetry with model architecture tags

**Total**: ~10 days (can parallelize T031-T035 if multiple engineers)

---

## 8. Rollback Plan

**Feature Flags** (config.yml):
```yaml
neural_network:
  architecture: "se_resnet"  # Options: se_resnet, repeca, ghost, cascade, early_exit
  use_auxiliary_heads: false
  early_exit_enabled: false
  cascade_enabled: false
```

**Rollback Triggers**:
- ELO < baseline - 15 (unacceptable strength loss)
- Throughput < baseline × 1.2 (insufficient speedup)
- Training instability (loss divergence, NaN values)

**Rollback Procedure**:
1. Set `architecture: "se_resnet"` in config
2. Re-run inference benchmark
3. Verify baseline performance restored
4. File bug report with training logs, model checkpoints

---

## 9. References

All content derived from review.txt:
- **RepVGG/DBB reparameterization**: lines 1267-1279
- **Ghost bottlenecks**: lines 1281-1289
- **ShuffleNetV2**: lines 1291-1300
- **Two-tier cascade**: lines 1302-1310
- **Early-exit heads**: lines 1313-1320
- **Auxiliary tactical heads**: lines 1322-1335
- **Game-specific configs**: lines 1090-1244
- **Training protocol**: lines 1377-1385
- **Expected performance**: lines 1390-1396

---

**Status**: Documented for Phase 7 (post-8k MCTS target achieved)

**Next Steps** (when MCTS reaches 8k):
1. Review this document
2. Select architecture option (recommend: RepECA for safety, then Ghost if needed)
3. Implement T031-T037
4. Validate via A/B testing
5. Deploy to production if acceptance criteria met
