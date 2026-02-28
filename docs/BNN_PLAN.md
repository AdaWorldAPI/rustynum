# BNN Plan — Aligned with Existing Optimizations

> **Date**: 2026-02-28
> **Branch**: `claude/implement-neural-network-research-owTKn`
> **Prerequisite**: Read `docs/optimizations.md` first

---

## 0. What Already Exists (DO NOT REINVENT)

Before writing a single line, understand what's already built and tested:

### Existing BNN Infrastructure (rustynum-core)

| File | Lines | Tests | What It Does |
|------|-------|-------|-------------|
| `bnn.rs` | 556 | 11 | `BnnDotResult`, `BnnNeuron`, `BnnLayer`, `bnn_dot()`, `bnn_dot_3ch()`, `bnn_batch_dot()` |
| `graph_hv.rs` | 840 | 25 | `GraphHV` (3×16,384-bit), `bundle()` (4-bit carry-save adder), `bundle_into()` (stochastic EMA), `encode_edge()`, `decode_edge_source()`, `circular_shift()`, `decay()` |
| `cam_index.rs` | 420 | 8 | Multi-probe LSH index: 4 tables × 64-bit hash signatures, O(log N) candidate retrieval + exact Hamming verification |
| `dn_tree.rs` | 598 | 9 | Quaternary plasticity tree: beam search with partial Hamming, BTSP gating, arena-layout SoA summaries |

**Total: 2,414 lines, 53 tests, all passing.**

### Existing Search/Filtering Pipeline (DO NOT REPLACE)

| Component | File | Lines | What It Does |
|-----------|------|-------|-------------|
| HDR Cascade | `simd.rs:1206-1644` | ~440 | 3-stroke: Belichtungsmesser warmup → full Hamming → PreciseMode tail |
| K0/K1/K2 | `kernels.rs` | ~600 | Fixed-size cascade: 64-bit probe → 512-bit stats → full EnergyConflict |
| Horizontal Sweep | `horizontal_sweep.rs` | ~350 | 90° word-by-word scan, progressive early exit, safety margin 1.5× |
| Hybrid Pipeline | `hybrid.rs` | ~500 | Tier 0 VNNI prefilter → K0 → K1 → K2 → BF16 tail with awareness |

### Existing SIMD Dispatch (REUSE, DON'T DUPLICATE)

| Function | Signature | What BNN Uses It For |
|----------|-----------|---------------------|
| `select_hamming_fn()` | `→ fn(&[u8],&[u8])→u64` | XOR + POPCNT = the BNN dot product core |
| `select_dot_i8_fn()` | `→ fn(&[i8],&[i8])→i32` | VNNI INT8 correlation for PreciseMode |
| `k0_probe()` | `(&[u64],&[u64])→u32` | 64-bit fast reject before full BNN dot |
| `k1_stats()` | `(&[u64],&[u64])→u32` | 512-bit intermediate check |
| `k2_exact()` | `(&[u64],&[u64])→EnergyConflict` | Full distance + energy decomposition |

**The BNN dot product IS the Hamming complement.** `bnn_dot()` already calls
`select_hamming_fn()`. There is no separate XNOR kernel — the identity
`XNOR_popcount(a,b) = TOTAL_BITS - XOR_popcount(a,b)` is used. This is correct
and maximally efficient.

---

## 1. Gap Analysis: What's Missing vs What's Done

### Already Done (no work needed)

| Capability | Status | Location |
|-----------|--------|----------|
| Binary dot product (XNOR+POPCNT) | DONE | `bnn.rs:bnn_dot()` via `select_hamming_fn()` |
| 3-channel correlation | DONE | `bnn.rs:bnn_dot_3ch()` |
| Batch inference + top-K | DONE | `bnn.rs:bnn_batch_dot()` |
| BNN neuron (forward + plasticity) | DONE | `bnn.rs:BnnNeuron::forward()` |
| BNN dense layer | DONE | `bnn.rs:BnnLayer` |
| Winner-take-all | DONE | `bnn.rs:BnnLayer::winner()` |
| Graph encoding (bind/permute/bundle) | DONE | `graph_hv.rs` |
| Majority-vote bundle (N≤15 adder tree) | DONE | `graph_hv.rs:bundle()` |
| Stochastic EMA learning | DONE | `graph_hv.rs:bundle_into()` |
| Probabilistic decay (LTP/LTD) | DONE | `graph_hv.rs:GraphHV::decay()` |
| LSH index for approximate NN | DONE | `cam_index.rs:CamIndex` |
| Hierarchical routing tree | DONE | `dn_tree.rs:DNTree` |
| SIMD dispatch (AVX-512/AVX2/scalar) | DONE | `simd.rs:select_hamming_fn()` |

### Genuine Gaps (new code needed)

| Gap | Impact | Lines Est. | Priority |
|-----|--------|-----------|----------|
| **G1: Multi-layer BNN network** | Can't stack layers for depth | ~150 | P0 |
| **G2: Batch-accelerated layer forward** | `BnnLayer::forward()` is sequential per-neuron | ~80 | P0 |
| **G3: BNN → K0/K1/K2 cascade wiring** | `bnn_batch_dot()` doesn't use kernel cascade for early exit | ~100 | P1 |
| **G4: BNN → HDR cascade wiring** | No 3-stroke search over BNN weight banks | ~60 | P1 |
| **G5: Binary convolution** | No spatial/temporal convolution primitive | ~200 | P2 |
| **G6: Training signal via awareness** | BF16 awareness (crystallized/tensioned) not fed back to BNN learning | ~120 | P2 |
| **G7: Amplitude correction (RIF-Net BIR-EWM)** | `BnnNeuron.bias` exists but no batch normalization or element-wise multiply | ~150 | P2 |

---

## 2. Implementation Plan (Ordered by Impact, Aligned with Existing Code)

### Phase 1: Wire BNN to Existing Cascade (P0 — highest impact, least code)

The single highest-impact change is making `bnn_batch_dot()` use the K0/K1/K2
cascade instead of brute-force scanning. This is 95% wiring, 5% new code.

#### Task 1.1: `bnn_cascade_search()` — Wire BNN to K0/K1/K2

**File**: `rustynum-core/src/bnn.rs` (add ~100 lines)

```
bnn_cascade_search(query: &Fingerprint<256>, weights: &[Fingerprint<256>], top_k: usize)
  → Vec<(usize, BnnDotResult)>

Implementation:
  1. Convert query/weights to &[u64] (zero-copy via .words)
  2. For each weight:
     a. K0 probe: k0_probe(query.words, weight.words) — reject if > proportional threshold
     b. K1 stats: k1_stats(query.words, weight.words) — reject survivors
     c. K2 exact: k2_exact(query.words, weight.words) — full EnergyConflict
  3. Convert EnergyConflict → BnnDotResult (Hamming → XNOR score)
  4. Sort + truncate to top_k
```

**Why this is high impact**: `bnn_batch_dot()` currently does N full 16,384-bit
Hamming computations. K0/K1/K2 eliminates ~95% of candidates after touching
only 64-512 bits. For 10,000 weight vectors, this means ~500 full computations
instead of 10,000.

**Reuses**: `k0_probe()`, `k1_stats()`, `k2_exact()`, `SliceGate` from `kernels.rs`.
Zero new SIMD code.

#### Task 1.2: `BnnNetwork` — Multi-layer stacking

**File**: `rustynum-core/src/bnn.rs` (add ~150 lines)

```rust
pub struct BnnNetwork {
    layers: Vec<BnnLayer>,
}

impl BnnNetwork {
    pub fn new(layer_sizes: &[usize], rng: &mut SplitMix64) -> Self;

    /// Forward pass: feed input through all layers sequentially.
    /// Each layer's output activation becomes the next layer's input.
    /// The "output" is the activation of the last layer's winner neuron.
    pub fn forward(
        &mut self,
        input: &Fingerprint<256>,
        learn: bool,
        learning_rate: f64,
        rng: &mut SplitMix64,
    ) -> (usize, BnnDotResult);

    /// Predict: forward without learning, return winner index per layer.
    pub fn predict(&self, input: &Fingerprint<256>) -> Vec<(usize, BnnDotResult)>;
}
```

**Why**: Currently `BnnLayer` exists but there's no way to stack layers for depth.
A 3-layer BNN (input → hidden → output) is the minimal useful architecture.
Each layer's winner activation (a `Fingerprint<256>`) feeds into the next layer.

#### Task 1.3: Batch-parallel `BnnLayer::forward_batch()`

**File**: `rustynum-core/src/bnn.rs` (add ~80 lines)

```rust
impl BnnLayer {
    /// Forward pass on multiple inputs simultaneously.
    /// Uses split_at_mut for parallel output, no Arc<Mutex>.
    pub fn forward_batch(
        &mut self,
        inputs: &[Fingerprint<256>],
        learn: bool,
        learning_rate: f64,
        rng: &mut SplitMix64,
    ) -> Vec<Vec<f32>>;
}
```

**Why**: Current `forward()` processes one input at a time. Batch processing
enables CPU-level parallelism via `split_at_mut` (the pattern already used
in `rustyblas` for GEMM tiling).

---

### Phase 2: Wire BNN to HDR Cascade (P1 — accelerates large-scale inference)

#### Task 2.1: `bnn_hdr_search()` — 3-Stroke over weight banks

**File**: `rustynum-core/src/bnn.rs` (add ~60 lines)

```
bnn_hdr_search(query: &Fingerprint<256>, weights: &[u8], weight_len: usize, top_k: usize)
  → Vec<(usize, BnnDotResult)>

Implementation:
  1. Wrap weights as flat u8 buffer (same layout as hdr_cascade_search)
  2. Call hdr_cascade_search() with PreciseMode::Off (or ::Vnni for cosine)
  3. Convert HdrResult → BnnDotResult
```

**Why**: The HDR cascade's Belichtungsmesser (1/16th sampling + statistical
warmup) kills 98% of candidates before full computation. For a BNN with
100,000 weight vectors, this means ~2,000 full computations instead of
100,000.

**Reuses**: `hdr_cascade_search()` from `simd.rs`. Zero new SIMD code.

#### Task 2.2: BNN → CamIndex + DNTree integration

**File**: `rustynum-core/src/bnn.rs` (add ~100 lines)

```rust
impl BnnLayer {
    /// Build a CAM index over all neuron weights for O(log N) lookup.
    pub fn build_cam_index(&self, seed: u64) -> CamIndex;

    /// Forward using CAM index: O(log N) instead of O(N) per input.
    pub fn forward_cam(
        &self,
        input: &Fingerprint<256>,
        cam: &CamIndex,
        top_k: usize,
    ) -> Vec<(usize, BnnDotResult)>;
}
```

**Why**: CamIndex provides O(log N) candidate retrieval via LSH. For a BNN
layer with 10,000 neurons, brute-force winner-take-all is O(10,000). With
CamIndex, it's O(log 10,000 × window_size) ≈ O(400). 25× speedup.

**Reuses**: `CamIndex` from `cam_index.rs`. Zero new data structure code.

---

### Phase 3: Training Signal via Awareness (P2 — enables learning from BF16 decomposition)

#### Task 3.1: Awareness-guided learning rate

**File**: `rustynum-core/src/bnn.rs` (add ~120 lines)

The BF16 awareness substrate classifies each dimension as
crystallized/tensioned/uncertain/noise. This can guide per-dimension
learning rates:

```
crystallized (00) → freeze (learning_rate × 0.01)
tensioned    (01) → amplify (learning_rate × 5.0)
uncertain    (10) → normal (learning_rate × 1.0)
noise        (11) → ignore (learning_rate × 0.0)
```

**File changes**:
```rust
impl BnnNeuron {
    /// Forward with awareness-guided per-dimension learning.
    /// Uses BF16 decomposition of the weight-input difference to modulate
    /// which bits flip during plasticity.
    pub fn forward_aware(
        &mut self,
        input: &Fingerprint<256>,
        awareness: &[u8],  // 2 bits per dim, packed 4 per byte
        rng: &mut SplitMix64,
    ) -> f32;
}
```

**Reuses**: `bf16_hamming_fn()` from `bf16_hamming.rs` for awareness computation.
The 2-bit per-dimension classification already exists.

#### Task 3.2: Binary convolution (spatial)

**File**: `rustynum-core/src/bnn.rs` (add ~200 lines)

For spatial patterns, a binary convolution slides a kernel Fingerprint across
a sequence of Fingerprints:

```rust
pub fn bnn_conv1d(
    input: &[Fingerprint<256>],
    kernel: &Fingerprint<256>,
    stride: usize,
) -> Vec<BnnDotResult>;
```

Each output position is `bnn_dot(input[i], kernel)`. This is embarrassingly
parallel and each dot product reuses `select_hamming_fn()`.

For 2D convolution over GraphHV channels:

```rust
pub fn bnn_conv1d_3ch(
    input: &[GraphHV],
    kernel: &GraphHV,
    stride: usize,
) -> Vec<BnnDotResult>;
```

---

## 3. What NOT To Do

| Anti-Pattern | Why |
|-------------|-----|
| Write a separate XNOR+POPCNT kernel | `bnn_dot()` already uses `select_hamming_fn()` — same VPOPCNTDQ path |
| Build a separate search index | CamIndex + DNTree already exist with 17 tests |
| Rewrite the cascade pipeline | K0/K1/K2 + HDR cascade already exist with 25+ tests |
| Add `Arc<Mutex>` for parallel BNN | Use `split_at_mut` (pattern from rustyblas GEMM) |
| Store BNN weights as `Vec<u8>` | They're already `Fingerprint<256>` — zero-copy to `&[u64]` |
| Import external BNN libraries | Pure compute, no dependencies (rustynum law) |
| Do IO in BNN functions | Pure compute only (rustynum law) |

---

## 4. Estimated Impact

| Phase | New Lines | Tests | Speedup | What Unlocks |
|-------|-----------|-------|---------|-------------|
| Phase 1 | ~330 | ~15 | 20× batch inference (K0/K1/K2 cascade) | Multi-layer networks, batch processing |
| Phase 2 | ~160 | ~8 | 25× large-scale search (CamIndex + HDR) | O(log N) neuron lookup, 100K+ weight banks |
| Phase 3 | ~320 | ~10 | N/A (quality, not speed) | Awareness-guided learning, spatial patterns |

**Total new code: ~810 lines + ~33 tests.**

Compare with what already exists: **2,414 lines + 53 tests.**

The ratio is ~1:3 — for every 1 line of new code, 3 lines of existing
infrastructure are reused. This is the correct ratio for a wiring task.

---

## 5. File Change Map

| File | Action | Lines Changed |
|------|--------|--------------|
| `rustynum-core/src/bnn.rs` | ADD functions | +810 |
| `rustynum-core/src/lib.rs` | ADD exports | +5 |
| `rustynum-core/src/bnn.rs` (tests) | ADD tests | +300 |

**No other files modified.** All existing infrastructure is consumed via
public API calls. No modifications to `kernels.rs`, `simd.rs`, `hybrid.rs`,
`cam_index.rs`, `dn_tree.rs`, `graph_hv.rs`, or `horizontal_sweep.rs`.

---

## 6. Implementation Log (Session-Resilient)

> **Read this every session.** Tracks exactly what was done and what remains.

### 2026-02-28: Phase 1 + Phase 2 + Phase 3 (partial) — DONE

**Branch**: `claude/implement-neural-network-research-owTKn`

**Files modified**:
- `rustynum-core/src/bnn.rs` — +430 lines of implementation + +280 lines of tests
- `rustynum-core/src/lib.rs` — +5 lines of exports

**Functions added to `bnn.rs`**:

| Function | Lines | Phase | What It Does |
|----------|-------|-------|-------------|
| `bnn_cascade_search()` | 60 | P1-1.1 | Wire BNN to K0/K1/K2 kernel pipeline |
| `bnn_cascade_search_with_energy()` | 50 | P1-1.1 | Same + EnergyConflict decomposition |
| `BnnNetwork::new/forward/predict/depth` | 80 | P1-1.2 | Multi-layer BNN stacking |
| `bnn_hdr_search()` | 50 | P2-2.1 | Wire BNN to 3-stroke HDR cascade (cfg avx512/avx2) |
| `BnnLayer::build_cam_index()` | 12 | P2-2.2 | Build LSH index over neuron weights |
| `BnnLayer::winner_cam()` | 30 | P2-2.2 | O(log N) winner-take-all via CamIndex |
| `bnn_conv1d()` | 8 | P3-3.2 | 1D binary convolution over Fingerprint sequence |
| `bnn_conv1d_3ch()` | 8 | P3-3.2 | 1D binary convolution over GraphHV sequence |
| `bnn_conv1d_cascade()` | 12 | P3-3.2 | Cascade-accelerated 1D convolution |

**Types added**:
- `BnnCascadeResult` — cascade matches + PipelineStats
- `BnnEnergyResult` — match + EnergyConflict + KernelStage
- `BnnNetwork` — multi-layer BNN

**Tests added**: 14 new tests (25 total BNN tests, 309 total rustynum-core tests)

| Test | What It Verifies |
|------|-----------------|
| `test_cascade_search_finds_exact_match` | K0/K1/K2 cascade finds planted exact match |
| `test_cascade_search_zero_false_negatives` | Cascade finds all planted near-matches |
| `test_cascade_search_with_energy` | Energy decomposition correct for exact match |
| `test_cascade_search_empty` | Empty input returns empty result |
| `test_network_creation` | Layer sizes correct |
| `test_network_forward` | Forward pass produces valid output |
| `test_network_predict` | Predict returns per-layer results |
| `test_network_learning_changes_state` | Plasticity updates neuron state |
| `test_build_cam_index` | CAM index has correct count |
| `test_winner_cam_finds_match` | LSH finds planted match in shortlist |
| `test_conv1d_basic` | Exact match at correct position |
| `test_conv1d_stride` | Stride produces correct output count |
| `test_conv1d_3ch` | 3-channel convolution works |
| `test_conv1d_cascade` | Cascade-accelerated conv finds exact match |

### Still TODO

- [ ] **P3-3.1: Awareness-guided learning** (`forward_aware`) — per-dimension
  learning rate modulation from BF16 awareness (crystallized/tensioned/uncertain/noise)
- [ ] **P3-3.3: Amplitude correction** (RIF-Net BIR-EWM) — batch normalization
  + element-wise multiply for gradient preservation

---

## 7. Verification Strategy

```bash
# After each phase:
cargo test -p rustynum-core -- bnn     # BNN-specific tests
cargo test -p rustynum-core            # Full crate (ensure nothing breaks)
cargo clippy -p rustynum-core -- -D warnings  # Lint

# Regression: existing tests must not change
# Expected: 244+ existing tests pass, new tests added on top
```

---

*This plan was designed to maximize impact by wiring existing infrastructure
rather than building from scratch. The 2,414 lines of existing BNN/HDC code
and the 1,890+ lines of search pipeline code are the foundation — not the
competition.*

---

## 8. Architectural Constraint: BNN vs DN-Tree Recall (SESSION-CRITICAL)

> **Date**: 2026-02-28
> **Severity**: Cognitive failure — this section documents a fundamental
> misunderstanding by Claude that must not recur.

### The Problem That Was Misunderstood

BNN (Binary Neural Network) provides neural **plasticity** — learning,
weight updating, XNOR+POPCNT similarity. It does NOT provide O(1)
**recall** of a specific tree/branch/twig/leaf path.

Two Claude sessions attempted to solve O(1) recall by:
1. Proposing CamIndex overlays on DN-tree leaves
2. Proposing K0/K1 Belichtungsmesser cascades inside traverse()
3. Modifying `rustynum-core/src/dn_tree.rs` and `kernels.rs` directly

All three approaches were wrong because the O(1) recall **already exists**
in the DN-tree by construction.

### DN = Distinguished Name — The Key IS The Path

The DN-tree is a **Distinguished Name** tree. The prototype index (proto_idx)
encodes the full tree path via quaternary decomposition:

```
proto_idx = 1500, num_prototypes = 4096

Level 0: root [0, 4096)
Level 1: 1500 / 1024 = 1   → branch [1024, 2048)    digit: 1
Level 2: (1500-1024) / 256 = 1  → twig [1280, 1536)  digit: 1
Level 3: (1500-1280) / 64 = 3   → leaf [1472, 1536)   digit: 3

DN path: [1, 1, 3]
```

This is pure integer division. O(1) per level, O(depth) total.
No hash table. No CamIndex. No tree walk. The index IS the address.

`select_child()` in `dn_tree.rs` already implements exactly this:
```rust
let offset = proto_idx - node.range_lo;
(offset / quarter_size).min(3)
```

The "full tree in one key" — each quaternary digit selects the child
at that level. The proto_idx is the distinguished name.

### Three Distinct Operations — Never Confuse Them

| Operation | Mechanism | Cost | When Used |
|-----------|-----------|------|-----------|
| **Recall** (known key → path) | DN quaternary decomposition | O(depth) arithmetic | `update()` — walk to leaf via proto_idx |
| **Search** (unknown query → matching leaves) | Beam search + partial Hamming | O(depth × beam × bits) | `traverse()` — similarity-based descent |
| **Plasticity** (learn from access) | BNN `bundle_into()` | O(depth × 49152 bits) | `update()` — bundle HV into path nodes |

**Recall** is what the DN-tree gives for free. It is the integer
decomposition of the key. No search structure needed.

**Search** is what `traverse()` does — when you have a query HV and
need to find which leaves are most similar. This is where
Belichtungsmesser K0/K1 filtering helps (reduce cost per level).

**Plasticity** is what BNN provides — stochastic bundling, learning
rate modulation, BTSP gating.

### What CamIndex Actually Solves

CamIndex is for a DIFFERENT problem: approximate nearest neighbor
search over a FLAT collection of HVs. It provides O(L × (log N + W))
lookup via multi-probe LSH.

CamIndex belongs in BNN layers (`BnnLayer::build_cam_index()`) for
O(log N) winner-take-all over neuron weights. It does NOT belong as
a layer on top of DN-tree — the tree's own addressing scheme is
strictly superior for structural recall.

### One Binary, One Bindspace — No Crate Privacy Walls

All crates compile into ONE binary. There is no runtime isolation
between crates. `Fingerprint<256>` = `Overlay` = 2048 bytes — they
are the SAME memory viewed through different type lenses. Everything
is written to bindspace.

DN-tree summaries, BNN weights, CamIndex prototypes — all live on
the same `Fingerprint<256>` surface. Zero-copy. Same pointer.

The Rust module `pub`/private distinction is compile-time hygiene,
not an architectural boundary. When code needs access to DN-tree
internals, it imports `rustynum_core::dn_tree` and uses the types
directly. The crate boundary does NOT mean "can't touch" — it means
"organized separately, built together."

`rustynum-bnn` exists for BNN-specific logic (Belichtungsmesser,
TraversalStats, signal quality) that doesn't belong in core but
runs in the same binary and reads the same bindspace surface.

### What rustynum-bnn Provides

The `belichtungsmesser` module in `rustynum-bnn` provides K0/K1
probe functions and TraversalStats for use **outside** the DN-tree
traverse loop — by callers who build their own search over DN-tree
summaries via the public `summary(node_idx)` API.

```
rustynum-bnn::belichtungsmesser
  ├── k0_probe_conflict(a, b) → u32       Pure XOR+POPCNT
  ├── k1_stats_conflict(a, b) → u32       8-word XOR+POPCNT
  ├── TraversalStats                       Welford auto-adjust
  ├── signal_quality(summary) → f32        Noise floor detection
  ├── classify_hdr(k1, stats) → u8         HDR class from K1
  ├── bf16_refine_cold(q, c) → u8          BF16 range awareness
  ├── filter_children(...)  → Vec<ChildScore>  Full K0→K1→BF16 cascade
  └── hdr_beam_width(scores, base) → usize  HDR-aware beam width
```

These are standalone compute functions. They do not import, modify,
or wrap DN-tree internals.
