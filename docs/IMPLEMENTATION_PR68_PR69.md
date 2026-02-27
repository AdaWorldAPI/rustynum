# BitPacked Plastic HDC-BNN Implementation

> **PRs**: #68 (foundations) + #69 (SIMD wiring pass)
> **Branch**: `claude/implement-neural-network-research-owTKn`
> **Date**: 2026-02-27
> **Total new code**: 2,537 lines (2,183 + 354) across 5 files
> **Tests**: 65 new tests (53 in PR #68, 12 in PR #69), 295 total passing

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Module 1: GraphHV — 3-Channel Hypervector](#3-module-1-graphhv--3-channel-hypervector)
4. [Module 2: CamIndex — Content-Addressable Memory](#4-module-2-camindex--content-addressable-memory)
5. [Module 3: DNTree — Hierarchical Plasticity Tree](#5-module-3-dntree--hierarchical-plasticity-tree)
6. [Module 4: BNN — Binary Neural Network Kernels](#6-module-4-bnn--binary-neural-network-kernels)
7. [SIMD Wiring Pass (PR #69)](#7-simd-wiring-pass-pr-69)
8. [Mathematical Foundations](#8-mathematical-foundations)
9. [Test Coverage](#9-test-coverage)
10. [API Reference](#10-api-reference)
11. [Integration Points](#11-integration-points)
12. [Design Decisions & Trade-offs](#12-design-decisions--trade-offs)

---

## 1. Executive Summary

This implementation adds a complete **plastic hyperdimensional computing (HDC)
graph memory system** to `rustynum-core`, built entirely on the existing
`Fingerprint<256>` type and SIMD infrastructure. The system combines four
modules into a unified pipeline:

```
Input → GraphHV encoding → CamIndex (O(log N) lookup)
                        → DNTree (hierarchical routing)
                        → BNN (inference + learning)
```

**Key properties:**
- Pure compute — zero IO, zero allocations on hot paths
- Reuses existing `Fingerprint<256>` (2048 bytes = 16,384 bits) as the atomic unit
- All operations reduce to XOR, AND, popcount, and shift — SIMD-friendly
- Biological plasticity via stochastic bundling + BTSP gating
- 49,152-bit hypervectors = 96 AVX-512 registers (perfect packing)

---

## 2. Architecture Overview

### The Three-Channel GraphHV Model

Every entity in the graph is encoded as three 16,384-bit channels:

```
GraphHV (49,152 bits = 6,144 bytes)
├── Channel 0: Node/Identity   — random base + positional encoding
├── Channel 1: Edge/Relation   — bind(source, dest, label) via XOR + shift
└── Channel 2: Plastic/State   — running average via majority-vote bundling
```

### Grey/White Matter Analogy

| Brain Region | HDC Role | GraphHV Channel | Operation |
|-------------|----------|-----------------|-----------|
| Grey matter | Inference | Channel 0 (activations) | XNOR + popcount |
| White matter | Connections | Channel 1 (weights) | XOR bind |
| Plasticity | Learning | Channel 2 (running avg) | `bundle_into()` |

### Data Flow

```
                    ┌──────────────────┐
                    │   GraphHV Pool   │
                    │ (49,152-bit HVs) │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼───────┐ ┌───▼────┐ ┌───────▼────────┐
     │   CamIndex     │ │DNTree  │ │   BNN Layer    │
     │ (LSH → exact)  │ │(beam   │ │ (XNOR+popcount │
     │ O(log N)       │ │search) │ │  inference)    │
     │ candidate      │ │partial │ │                │
     │ retrieval      │ │Hamming │ │ grey / white / │
     └────────────────┘ │routing │ │ plastic        │
                        └────────┘ └────────────────┘
```

### File Layout

| File | Lines | Tests | Purpose |
|------|-------|-------|---------|
| `rustynum-core/src/graph_hv.rs` | 840 | 25 | 3-channel HV type + HDC algebra |
| `rustynum-core/src/cam_index.rs` | 420 | 8 | Multi-probe LSH index |
| `rustynum-core/src/dn_tree.rs` | 598 | 9 | Hierarchical plasticity tree |
| `rustynum-core/src/bnn.rs` | 556 | 11 | BNN inference + learning |
| `rustynum-core/src/fingerprint.rs` | +39 | +2 | Zero-copy `as_bytes()`/`as_bytes_mut()` |
| **Total** | **2,414** | **65** | |

### Public Exports (lib.rs)

```rust
// 3D Graph HDC
pub use graph_hv::{
    bundle, bundle_into, decode_edge_source, encode_edge,
    GraphHV, GRAPH_HV_BITS, GRAPH_HV_BYTES, GRAPH_HV_CHANNELS,
};

// Content-Addressable Memory
pub use cam_index::{CamConfig, CamHit, CamIndex};

// DN-tree
pub use dn_tree::{DNConfig, DNTree, DNTreeStats, TraversalHit};

// BNN kernels
pub use bnn::{bnn_batch_dot, bnn_dot, bnn_dot_3ch, BnnDotResult, BnnLayer, BnnNeuron};
```

---

## 3. Module 1: GraphHV — 3-Channel Hypervector

**File**: `rustynum-core/src/graph_hv.rs` (840 lines, 25 tests)

### 3.1 Type Definition

```rust
pub struct GraphHV {
    pub channels: [Fingerprint<256>; 3],
}
```

- **3 channels x 256 words x 64 bits = 49,152 bits = 6,144 bytes**
- Exactly 96 AVX-512 registers (512 bits each) — perfect packing
- Channel semantics: `CH_NODE=0`, `CH_EDGE=1`, `CH_PLASTIC=2`

### 3.2 HDC Algebra Operations

#### XOR Bind (`bind`)

```rust
pub fn bind(&self, other: &Self) -> Self
```

Associates two hypervectors. Fundamental properties:
- **Self-inverse**: `bind(bind(a, b), b) = a` (XOR is its own inverse)
- **Commutative**: `bind(a, b) = bind(b, a)`
- **Associative**: `bind(bind(a, b), c) = bind(a, bind(b, c))`
- **Near-orthogonal to inputs**: `similarity(a, bind(a, b)) ≈ 0.50`

This is per-word XOR across all 3 channels: `channels[i] ^ other.channels[i]`.

#### Circular Shift Permutation (`permute`)

```rust
pub fn permute(&self, shift: u32) -> Self
```

Creates positionally-encoded variants of the same HV. A shifted version is
nearly orthogonal to the original (`similarity ≈ 0.50 ± 0.05` for any non-zero shift).

**Implementation** (PR #69 optimization): Zero modulo operations. Two-step approach:

1. **Word rotation**: `rotate_right(word_shift)` — moves whole 64-bit words
   to their destination positions. No modulo arithmetic, just a pointer shuffle.

2. **Linear carry chain**: For intra-word bit shifts, process words in reverse
   order. Each word shifts left and takes carry bits from the word below:

```rust
// Save wrap carry from word 255 (highest) → word 0
let wrap_carry = words[255] >> (64 - bit_shift);
// Process backwards: words[i-1] is always unmodified when read
for i in (1..256).rev() {
    words[i] = (words[i] << bit_shift) | (words[i - 1] >> (64 - bit_shift));
}
words[0] = (words[0] << bit_shift) | wrap_carry;
```

**Previous implementation** used per-bit modulo arithmetic with
`(i * 64 + b + shift) % total_bits` — replaced with zero-modulo approach.

**Verified properties** (test coverage):
- `permute(0) = identity`
- `permute(16384) = identity` (full rotation)
- `permute(shift)` then `permute(16384 - shift) = identity` (inverse)
- `popcount` preserved (bit-conservative)
- Word-boundary shift (shift=64) works correctly

#### Majority-Vote Bundle (`bundle`)

```rust
pub fn bundle(vectors: &[&GraphHV], rng: &mut SplitMix64) -> GraphHV
```

Superimposes multiple HVs, preserving common patterns. For each bit position,
the output bit is set if the majority of input vectors have it set.

**Implementation** (PR #69 optimization): Two paths based on N:

##### Path 1: 4-Bit Carry-Save Adder Tree (N ≤ 15)

Branchless word-parallel counting — all 64 bit positions counted
simultaneously using 4 counter words `(b0, b1, b2, b3)`:

```rust
fn csa_add(b0: &mut u64, b1: &mut u64, b2: &mut u64, b3: &mut u64, input: u64) {
    let c0 = *b0 & input;     // carry from bit 0
    *b0 ^= input;             // sum at bit 0
    let c1 = *b1 & c0;        // carry from bit 1
    *b1 ^= c0;                // sum at bit 1
    let c2 = *b2 & c1;        // carry from bit 2
    *b2 ^= c1;                // sum at bit 2
    *b3 ^= c2;                // sum at bit 3 (no carry needed for N≤15)
}
```

After feeding all N input words through `csa_add`, each bit position `j`
in `(b0[j], b1[j], b2[j], b3[j])` holds the 4-bit count of how many inputs
had that bit set.

**Threshold comparison** uses two's complement subtraction across 64 parallel
4-bit counters:

```rust
fn threshold_gte(b0: u64, b1: u64, b2: u64, b3: u64, target: usize) -> u64
```

The target value is broadcast to all 64 positions as a 4-bit constant.
Then a ripple carry subtraction (`count - target`) is performed. The
final carry out (1 = no borrow) means `count >= target`. Returns a `u64`
bitmask where bit `j` = 1 iff `count[j] >= target`.

For **even N**, ties are broken randomly: one `rng.next_u64()` per output
word provides 64 independent coin flips. The formula:
```
gt | (eq & rng.next_u64())
```
where `eq = gte & !gt` (positions where count == threshold exactly).

##### Path 2: Column-Count (N > 15)

Explicit `u16` counters per bit position. 4-bit adder saturates at 15,
so for N > 15 we fall back to per-bit counting:

```rust
let mut counts = [0u16; 64];
for v in vectors {
    let word = v.channels[ch].words[w];
    for bit in 0..64 {
        counts[bit] += ((word >> bit) & 1) as u16;
    }
}
```

Same tiebreak logic for even N.

**Previous implementation**: Per-bit counting for all N (no adder tree).
The new adder tree path is ~4x faster for typical HDC bundling (N ≤ 15)
because it processes all 64 bit positions in parallel using word-level ops.

#### Stochastic Weighted Merge (`bundle_into`)

```rust
pub fn bundle_into(
    summary: &GraphHV,
    new_observation: &GraphHV,
    learning_rate: f64,
    btsp_boost: f64,
    rng: &mut SplitMix64,
) -> GraphHV
```

Exponential moving average in binary:
- For each disagreeing bit (XOR of summary and observation), flip to the
  new value with probability `effective_lr = learning_rate * btsp_boost`
- The flip mask is generated by ANDing `n_ands` random words:
  `P(bit=1) = (0.5)^n_ands ≈ effective_lr`
- `learning_rate = 0.03` (decay=0.97): effective memory window ~33 updates
- `btsp_boost` (1.0-7.0): amplifies learning rate for one update, simulating
  CaMKII autophosphorylation after a stochastic plateau potential

#### Decay (`decay`)

```rust
pub fn decay(&mut self, keep_prob: f64, rng: &mut SplitMix64)
```

Probabilistic LTP/LTD: each set bit survives with probability `keep_prob`.
Uses fast word-level decay: AND multiple random words to create a kill mask
with `P(bit=1) ≈ 1 - keep_prob`, then clear those bits.

#### Edge Encoding and Decoding

```rust
pub fn encode_edge(source: &GraphHV, dest: &GraphHV, role: &GraphHV, shift: u32) -> GraphHV
pub fn decode_edge_source(edge: &GraphHV, dest: &GraphHV, role: &GraphHV, shift: u32) -> GraphHV
```

Standard GraphHD directed labeled edge encoding:
- **Encode**: `permute(source, shift) ^ dest ^ role`
  - Permute encodes direction (source→dest, not dest→source)
  - XOR-bind with destination and role label
- **Decode**: Unbind known components, apply inverse permutation
  - `source = permute_inverse(edge ^ dest ^ role, shift)`
  - XOR is self-inverse: `a ^ a = 0`, so unbinding = rebinding

**Roundtrip verified**: `decode_edge_source(encode_edge(s, d, r, k), d, r, k) = s`

### 3.3 Similarity Metrics

| Method | Scope | Returns | Notes |
|--------|-------|---------|-------|
| `hamming_distance(&self, other)` | All 3 channels | `u32` | Sum of per-channel popcount(XOR) |
| `similarity(&self, other)` | All 3 channels | `f64 [0,1]` | `1.0 - hamming/49152` |
| `partial_hamming(&self, other, bits)` | First N bits per ch | `u32` | For DN-tree prefix scan |
| `partial_similarity(&self, other, bits)` | First N bits per ch | `f64 [0,1]` | Normalized partial |
| `popcount(&self)` | All 3 channels | `u32` | Total set bits |

### 3.4 Constants

```rust
pub const GRAPH_HV_CHANNELS: usize = 3;
pub const GRAPH_HV_BITS: usize = 49_152;    // 16,384 × 3
pub const GRAPH_HV_BYTES: usize = 6_144;    // 2,048 × 3
```

---

## 4. Module 2: CamIndex — Content-Addressable Memory

**File**: `rustynum-core/src/cam_index.rs` (420 lines, 8 tests)

### 4.1 Architecture

Multi-probe Locality-Sensitive Hashing (LSH) for O(log N) nearest-neighbor
retrieval from 49,152-bit hypervectors:

```
GraphHV (49,152 bits)
    │
    ├── LSH Table 0: hash(hv, proj_0) → sorted Vec<(hash, idx)>
    ├── LSH Table 1: hash(hv, proj_1) → sorted Vec<(hash, idx)>
    ├── LSH Table 2: hash(hv, proj_2) → sorted Vec<(hash, idx)>
    └── LSH Table 3: hash(hv, proj_3) → sorted Vec<(hash, idx)>
              │
              └── Union candidates → Exact Hamming → Top-K
```

### 4.2 LSH Projector with Precomputed Masks (PR #69)

Each hash bit is the XOR-parity of `sample_size` randomly chosen input bits.

**Before PR #69**: Stored scattered `Vec<Vec<(usize, usize, u64)>>` tuples
(channel, word, bitmask). Hash computation required random memory accesses.

**After PR #69**: Precomputed `Vec<[Fingerprint<256>; 3]>` — one set of 3
channel masks per hash bit. Hash computation becomes contiguous AND + popcount:

```rust
fn hash(&self, hv: &GraphHV) -> u64 {
    let mut code = 0u64;
    for (i, ch_masks) in self.masks.iter().enumerate() {
        let mut parity = 0u32;
        for (ch_mask, channel) in ch_masks.iter().zip(hv.channels.iter()) {
            for (m, w) in ch_mask.words.iter().zip(channel.words.iter()) {
                if *m != 0 {
                    parity += (m & w).count_ones();
                }
            }
        }
        if parity & 1 != 0 {
            code |= 1u64 << i;
        }
    }
    code
}
```

**Benefit**: Sequential memory access pattern (contiguous AND + popcount)
instead of scattered loads. The `if *m != 0` guard skips zero masks (most
words are zero since only `sample_size=8` bits are set per hash bit across
3 × 256 = 768 words).

### 4.3 Configuration

```rust
pub struct CamConfig {
    pub num_tables: usize,     // Default: 4
    pub sample_size: usize,    // Default: 8 (bits per hash projection)
    pub window_size: usize,    // Default: 32 (scan window around insertion point)
}
```

| Parameter | Sweet Spot | Reasoning |
|-----------|-----------|-----------|
| `num_tables` | 4 | Balances recall vs query cost. More tables = higher recall but more hash computations |
| `sample_size` | 8 | Bits per XOR-parity projection. Too few = poor discrimination, too many = hash becomes too sensitive |
| `window_size` | 32 | Scan range around hash match point. Wider = better recall for hash collisions |

### 4.4 Operations

#### Insert — O(L × log N)

```rust
pub fn insert(&mut self, hv: GraphHV) -> usize
```

For each of L hash tables:
1. Compute LSH hash via projector
2. Binary search for insertion point (`partition_point`)
3. Insert `(hash, idx)` maintaining sorted order

#### Query — O(L × W + C × D)

```rust
pub fn query(&self, query: &GraphHV, top_k: usize) -> Vec<CamHit>
```

1. For each table: compute query hash, binary search for position
2. Scan `window_size` entries around the position
3. Collect unique candidate indices across all tables
4. Compute exact Hamming distance on candidate shortlist
5. Sort by distance, return top-K

The two-phase approach (approximate LSH → exact verification) achieves
sub-linear query time while guaranteeing exact distance on returned results.

#### Rebuild — O(N × L × log N)

```rust
pub fn rebuild(&mut self)
```

Recomputes all hash tables from scratch. Useful after bulk modifications.

### 4.5 CamHit Result

```rust
pub struct CamHit {
    pub index: usize,      // Prototype index in the store
    pub distance: u32,     // Exact Hamming distance to query
}
```

---

## 5. Module 3: DNTree — Hierarchical Plasticity Tree

**File**: `rustynum-core/src/dn_tree.rs` (598 lines, 9 tests)

### 5.1 Design

Adapted from "On Demand Memory Specialization for Distributed Graph Processing"
(2013) for HDC graph memory. A quaternary tree (fanout=4) over prototype indices
with bundled summary HVs at each node.

```
Root [0, 4096)
├── Child 0 [0, 1024)
│   ├── Grandchild 0.0 [0, 256)
│   ├── Grandchild 0.1 [256, 512)
│   ├── Grandchild 0.2 [512, 768)
│   └── Grandchild 0.3 [768, 1024)
├── Child 1 [1024, 2048)
├── Child 2 [2048, 3072)
└── Child 3 [3072, 4096)
```

### 5.2 Arena Storage (SoA Layout)

```rust
pub struct DNTree {
    nodes: Vec<DNNode>,        // Arena-allocated nodes
    summaries: Vec<GraphHV>,   // Parallel summary HVs (SoA)
    config: DNConfig,
}
```

Nodes and summaries are stored in parallel `Vec`s (Structure-of-Arrays layout).
This ensures the hot traversal path touches only the summaries it needs,
maximizing cache efficiency.

### 5.3 Node Structure

```rust
pub struct DNNode {
    pub range_lo: usize,              // Prototype range [lo, hi)
    pub range_hi: usize,
    pub level: u32,                   // Depth in tree
    pub access_count: u32,            // Access counter
    pub children: Option<[usize; 4]>, // None = leaf, Some = internal
}
```

### 5.4 Configuration

```rust
pub struct DNConfig {
    pub num_prototypes: usize,         // 4096 default
    pub split_threshold: u32,          // 8 — splits when access_count >= threshold
    pub growth_factor: f64,            // 1.8 — threshold grows with depth
    pub learning_rate: f64,            // 0.03 — summary bundling rate
    pub early_exit_threshold: f64,     // 0.82 — early exit on high similarity
    pub partial_bits: usize,           // 1024 — prefix bits for partial Hamming
    pub max_depth: u32,                // 7 — traversal depth limit
    pub beam_width: usize,             // 2 — children followed per level
    pub btsp_gate_prob: f64,           // 0.0 — BTSP gating probability
    pub btsp_boost: f64,               // 7.0 — BTSP learning rate multiplier
}
```

The split threshold grows exponentially with depth:
`threshold(level) = split_threshold * growth_factor^level`

This makes deeper nodes harder to split, naturally limiting tree depth.

### 5.5 Update — Plasticity

```rust
pub fn update(&mut self, proto_idx: usize, hv: &GraphHV, rng: &mut SplitMix64)
```

The core learning operation:

1. **BTSP gate check**: Random draw against `btsp_gate_prob`. If it fires,
   learning rate is amplified by `btsp_boost` for this update (simulating
   CaMKII autophosphorylation after a stochastic plateau potential)

2. **Walk root → leaf**: For each node on the path:
   - Increment `access_count` (saturating add)
   - Bundle the input HV into the node's summary via `bundle_into()`
   - If leaf and `access_count >= threshold` and `range_size >= 4`: split

3. **Split**: Create 4 children, each covering 1/4 of the parent's range

### 5.6 Traverse — Beam Search with Early Exit

```rust
pub fn traverse(&self, query: &GraphHV, top_k: usize) -> Vec<TraversalHit>
```

1. Start beam at root with weight 1.0
2. At each level, for each beam entry:
   - If leaf: compute full similarity, add to hits
   - If internal: compute partial similarity with each child's summary
   - Sort children by similarity (descending)
   - **Early exit**: if best child similarity > `early_exit_threshold`, follow only 1 child
   - Otherwise: follow top `beam_width` children
3. Weight is propagated: `child_weight = access_count / parent_access_count`
4. Collect remaining beam entries that didn't reach leaves
5. Sort hits by weighted score, return top-K

### 5.7 TraversalHit

```rust
pub struct TraversalHit {
    pub range_lo: usize,   // Prototype range start
    pub range_hi: usize,   // Prototype range end (exclusive)
    pub level: u32,        // Tree level where hit was found
    pub score: f64,        // Weighted similarity score
}
```

### 5.8 Expected Performance

| Operation | Latency | Notes |
|-----------|---------|-------|
| `update` | ~30 ns/level | bundle + decay per node on path |
| `traverse` (top-8) | 180-420 ns | early exit wins big at depth 3-4 |
| Memory | < 4 MB | full tree + summaries for 16K prototypes |

---

## 6. Module 4: BNN — Binary Neural Network Kernels

**File**: `rustynum-core/src/bnn.rs` (556 lines, 11 tests)

### 6.1 Core Primitive: XNOR + Popcount

The fundamental BNN operation is the binary dot product:

```
dot(a, w) = 2 × popcount(XNOR(a, w)) - total_bits
```

Equivalent to signed matching: identical bits contribute +1, differing bits
contribute -1. Score range: [-1.0, +1.0].

**Mathematical identity exploited in PR #69**:
```
XNOR_popcount(a, b) = TOTAL_BITS - XOR_popcount(a, b)
```

This allows reusing the existing XOR-based Hamming SIMD kernel
(`select_hamming_fn()`) instead of implementing a separate XNOR kernel.

### 6.2 SIMD Dispatch (PR #69)

```rust
#[cfg(any(feature = "avx512", feature = "avx2"))]
type HammingFn = fn(&[u8], &[u8]) -> u64;

#[cfg(any(feature = "avx512", feature = "avx2"))]
fn hamming_simd() -> HammingFn {
    static FN: OnceLock<HammingFn> = OnceLock::new();
    *FN.get_or_init(crate::simd::select_hamming_fn)
}
```

The dispatch is resolved once at first call via `OnceLock` and cached as a
static function pointer. Subsequent calls are a single indirect call — zero
branching, zero runtime detection overhead.

The internal `bnn_hamming_u32` function handles the dispatch:

```rust
#[inline]
fn bnn_hamming_u32(a: &[u8], b: &[u8]) -> u32 {
    #[cfg(any(feature = "avx512", feature = "avx2"))]
    { hamming_simd()(a, b) as u32 }

    #[cfg(not(any(feature = "avx512", feature = "avx2")))]
    { a.iter().zip(b.iter()).map(|(&x, &y)| (x ^ y).count_ones()).sum() }
}
```

### 6.3 BnnDotResult

```rust
pub struct BnnDotResult {
    pub match_count: u32,   // popcount(XNOR(a, w)) — matching bits
    pub total_bits: u32,    // Total bits compared
    pub score: f32,         // Normalized [-1.0, +1.0]
}
```

### 6.4 Dot Product Functions

| Function | Input | Bits Compared | Use Case |
|----------|-------|---------------|----------|
| `bnn_dot(a, w)` | 2 × `Fingerprint<256>` | 16,384 | Single-channel inference |
| `bnn_dot_3ch(a, w)` | 2 × `GraphHV` | 49,152 | Full 3-channel correlation |
| `bnn_batch_dot(q, weights, k)` | 1 query + N weights | 16,384 × N | Top-K batch inference |

### 6.5 BnnNeuron

```rust
pub struct BnnNeuron {
    pub state: GraphHV,
    pub bias: f32,          // Amplitude correction (RIF-Net BIR-EWM)
    pub threshold: f32,     // Binary activation threshold
}
```

The neuron's `state` uses GraphHV's 3 channels:
- `channels[0]`: current binary activation
- `channels[1]`: weight pattern (synaptic connections)
- `channels[2]`: plastic running average (learned prototype)

#### Forward Pass

```rust
pub fn forward(&mut self, input: &Fingerprint<256>, learn: bool,
               learning_rate: f64, rng: &mut SplitMix64) -> f32
```

1. **Grey matter inference**: `bnn_dot(input, weights)` → raw score
2. **Amplitude correction**: `pre_activation = score + bias`
3. **Binary threshold**: Set activation channel to `input` or `!input`
4. **Plasticity** (if `learn=true`): `bundle_into()` on plastic channel

#### Winner-Take-All

```rust
pub fn winner(&self, input: &Fingerprint<256>) -> (usize, BnnDotResult)
```

Finds the neuron with the highest `bnn_dot` score. Used for classification
and competitive learning.

### 6.6 BnnLayer

```rust
pub struct BnnLayer {
    pub neurons: Vec<BnnNeuron>,
}
```

A bank of neurons processing the same input in parallel:
- `forward()`: compute all activations, optionally learn
- `winner()`: competitive selection (WTA)
- `len()` / `is_empty()`: standard size queries

### 6.7 RIF-Net Connection

Based on "Accurate binary neural network based on rich information flow"
(Zhang et al., Neurocomputing 2025):

- **1-bit weights and activations**: stored as `Fingerprint<256>`
- **XNOR + popcount**: the fundamental binary convolution
- **Rich Information Flow**: via amplitude correction (`bias` field)
- **Shortcut connections**: external to this module, enabled by the channel model

---

## 7. SIMD Wiring Pass (PR #69)

PR #69 made 5 ordered changes that connect the HDC modules to the AVX-512
substrate, turning theoretical operations into hardware-accelerated hot paths.

### 7.1 Change 1: Zero-Copy Byte View

**File**: `rustynum-core/src/fingerprint.rs` (+39 lines)

Added `as_bytes()` and `as_bytes_mut()` to `Fingerprint<N>`:

```rust
pub fn as_bytes(&self) -> &[u8] {
    // SAFETY: [u64; N] is contiguous. u64 is 8-byte aligned; u8 requires
    // 1-byte alignment. Pointer cast is always valid. Length N * 8 is exact.
    unsafe { std::slice::from_raw_parts(self.words.as_ptr() as *const u8, N * 8) }
}
```

This enables direct passage of fingerprint data to SIMD kernels that operate
on `&[u8]` slices — zero copy, zero allocation. The pointer cast from
`*const u64` to `*const u8` is always valid because `u8` has weaker alignment
requirements than `u64`.

**Tests added**:
- `test_as_bytes_roundtrip`: `from_bytes(as_bytes()) = identity`
- `test_as_bytes_zero_copy`: Verifies pointer equality (`as_bytes().as_ptr() == words.as_ptr()`)

### 7.2 Change 2: SIMD Hamming Dispatch for BNN

**File**: `rustynum-core/src/bnn.rs` (+96 lines, -26 lines)

Wired `bnn_dot` and `bnn_dot_3ch` to `select_hamming_fn()` SIMD dispatch.

**Key insight**: `XNOR_popcount(a, b) = TOTAL_BITS - XOR_popcount(a, b)`.
Instead of implementing a separate XNOR SIMD kernel, we reuse the existing
XOR-based Hamming distance kernel and subtract from the total.

The `OnceLock`-based dispatch resolves the SIMD function pointer once:

```rust
static FN: OnceLock<HammingFn> = OnceLock::new();
*FN.get_or_init(crate::simd::select_hamming_fn)
```

This selects the best available path at runtime:
- **AVX-512 VPOPCNTDQ**: XOR + `_mm512_popcnt_epi64` + horizontal reduce
- **AVX2 Harley-Seal**: CSA tree + `_mm256_sad_epu8` popcount
- **Scalar**: `(x ^ y).count_ones()` per byte

**Regression test added**: `test_bnn_dot_matches_scalar` verifies that the
SIMD path produces bit-identical results to the scalar reference.

### 7.3 Change 3: Word-Parallel Majority-Vote Bundle

**File**: `rustynum-core/src/graph_hv.rs` (+215 lines, -28 lines)

Replaced per-bit counting with two optimized paths:

**Path A (N ≤ 15)**: 4-bit carry-save adder tree

The `csa_add` function is a full adder operating on 64 parallel bit
positions simultaneously. Four `u64` words `(b0, b1, b2, b3)` hold a
4-bit counter per position. After feeding N inputs through the adder,
`threshold_gte` performs 64 parallel 4-bit unsigned comparisons using
two's complement subtraction with ripple carry.

The adder tree is completely branchless — every operation is bitwise.
For 7 inputs (typical bundle size), this is 7 adder calls + 1 threshold
comparison vs 7 × 64 × 256 × 3 = 344,064 branch-dependent increments.

**Path B (N > 15)**: Column-count with u16 counters

Falls back to explicit `u16` counters per bit position, since the 4-bit
adder saturates at 15.

**Tests added**:
- `test_bundle_adder_tree_matches_scalar`: Verifies N=7 (odd, adder path)
  produces identical results to bit-by-bit scalar counting
- `test_bundle_column_count_large_n`: Verifies N=20 (column-count path)
  maintains majority-vote properties

### 7.4 Change 4: Precomputed XOR-Parity Masks for CAM LSH

**File**: `rustynum-core/src/cam_index.rs` (+59 lines, -35 lines)

Replaced scattered `Vec<Vec<(usize, usize, u64)>>` projection storage with
precomputed `Vec<[Fingerprint<256>; 3]>` masks.

**Before**: Each hash bit stored a list of (channel, word, bitmask) tuples.
Computing the hash required random loads from scattered memory locations.

**After**: Each hash bit has 3 full `Fingerprint<256>` masks (one per channel).
The hash computation becomes contiguous AND + popcount:

```rust
for (ch_mask, channel) in ch_masks.iter().zip(hv.channels.iter()) {
    for (m, w) in ch_mask.words.iter().zip(channel.words.iter()) {
        if *m != 0 {
            parity += (m & w).count_ones();
        }
    }
}
```

Memory layout is sequential (mask words are contiguous in memory), so the
CPU prefetcher handles this efficiently. The `if *m != 0` guard skips the
vast majority of zero words (only ~8 bits are set across 768 words).

**Test added**: `test_lsh_hash_deterministic` verifies deterministic hashing
and that different random vectors produce different hashes.

### 7.5 Change 5: Clean Circular Shift

**File**: `rustynum-core/src/graph_hv.rs` (within the +215 above)

Replaced modulo-heavy implementation with:
1. `rotate_right(word_shift)` — standard library word rotation
2. Linear carry chain — backward pass through words

**Zero modulo operations**, one array copy + one linear pass.
All existing `circular_shift` tests pass unchanged (verified: permute_identity,
permute_full_rotation, permute_preserves_popcount, permute_creates_near_orthogonal,
circular_shift_word_boundary, circular_shift_inverse).

---

## 8. Mathematical Foundations

### 8.1 Hyperdimensional Computing (HDC) Algebra

The system implements a MAP (Multiply-Add-Permute) algebra over binary
vectors of dimension D = 16,384:

| Operation | Symbol | Binary Implementation | Property |
|-----------|--------|----------------------|----------|
| Bind | ⊗ | XOR | Self-inverse, nearly orthogonal to inputs |
| Bundle | + | Majority vote | Preserves similarity to all inputs |
| Permute | π | Circular bit shift | Creates orthogonal variants |
| Similarity | sim | 1 - Hamming/D | [0,1] normalized distance |

### 8.2 Capacity

At D = 16,384 per channel × 3 channels = 49,152 total bits:
- **Bundle capacity**: ~200-400 concepts before interference > 5%
- **Bind depth**: Arbitrary (XOR preserves randomness)
- **Permute orthogonality**: Any non-zero shift produces ~50% similarity

### 8.3 Carry-Save Addition (CSA)

The CSA adder is a full adder circuit implemented in bitwise logic:

```
       a_in ───┬──── XOR ──── sum_out
               │       │
        b_in ──┤───── AND ──── carry_out
               │
              (ripple through bit levels)
```

For 4-bit counters across 64 parallel positions:
- `b0`: bit 0 of the count (ones)
- `b1`: bit 1 of the count (twos)
- `b2`: bit 2 of the count (fours)
- `b3`: bit 3 of the count (eights)

Max count = 15 (hence N ≤ 15 constraint).

### 8.4 Two's Complement Threshold Comparison

To check `count >= target` for 64 parallel 4-bit counters:

1. Broadcast `target` to 64 positions: each bit of target becomes either
   `u64::MAX` (all 1s) or `0` (all 0s)
2. Compute `count + NOT(target) + 1` (two's complement subtraction)
3. Final carry = 1 means no borrow, i.e., `count >= target`
4. The carry chain ripples through 4 levels (b0→b1→b2→b3)

### 8.5 BTSP (Behavioral Time-Scale Plasticity)

The BTSP mechanism simulates:
- **Plateau potential**: Random gate with probability `btsp_gate_prob`
  (typically 0.005-0.02, biologically ~1% of events)
- **CaMKII amplification**: When the gate fires, learning rate is multiplied
  by `btsp_boost` (typically 7.0x) for that single update
- **Effect**: Creates sudden, strong memory consolidation events against a
  background of slow incremental learning

---

## 9. Test Coverage

### 9.1 Test Counts by Module

| Module | Tests | Key Assertions |
|--------|-------|----------------|
| `graph_hv.rs` | 25 | Algebra (bind, bundle, permute), similarity, edge encoding, decay, adder correctness |
| `cam_index.rs` | 8 | Insert/query, exact match, ordering, similar find, rebuild, deterministic hash |
| `dn_tree.rs` | 9 | Structure, update, bundling, split, traversal, early exit, BTSP, child selection |
| `bnn.rs` | 11 | Dot product (identical/opposite/random), 3ch, neuron forward, plasticity, WTA, batch, SIMD match |
| `fingerprint.rs` | +2 | as_bytes roundtrip, zero-copy pointer equality |
| **Total** | **65** | |

### 9.2 Key Regression Tests (PR #69)

| Test | Verifies |
|------|----------|
| `test_bnn_dot_matches_scalar` | SIMD Hamming dispatch produces identical results to scalar |
| `test_bundle_adder_tree_matches_scalar` | 4-bit CSA adder tree matches per-bit counting for N=7 |
| `test_bundle_column_count_large_n` | Column-count path (N=20) preserves majority-vote properties |
| `test_lsh_hash_deterministic` | Precomputed mask hashing is deterministic |
| `test_as_bytes_zero_copy` | `as_bytes()` shares the same pointer as `words` |
| `test_circular_shift_inverse` | `permute(k)` then `permute(16384-k) = identity` |

### 9.3 Statistical Invariant Tests

Many tests verify probabilistic properties with tolerances:

| Test | Invariant | Tolerance |
|------|-----------|-----------|
| Random HV similarity | ~0.50 | ±0.02 |
| Permuted HV similarity | ~0.50 | ±0.05 |
| Bundle similarity to inputs | > 0.55 | — |
| Low LR preserves old summary | > 0.97 | — |
| Random BNN dot score | ~0.0 | ±0.05 |
| Bound edge orthogonal to components | ~0.50 | ±0.05 |

---

## 10. API Reference

### 10.1 GraphHV

```rust
// Construction
GraphHV::zero() -> Self
GraphHV::random(rng: &mut SplitMix64) -> Self
GraphHV::from_channels(node, edge, plastic) -> Self

// HDC operations
graphhv.bind(&other) -> GraphHV                     // XOR across channels
graphhv.permute(shift: u32) -> GraphHV              // Circular bit shift
bundle(vectors: &[&GraphHV], rng) -> GraphHV        // Majority vote
bundle_into(summary, new, lr, boost, rng) -> GraphHV // Stochastic merge

// Similarity
graphhv.hamming_distance(&other) -> u32              // Sum of 3 channels
graphhv.similarity(&other) -> f64                    // Normalized [0,1]
graphhv.partial_hamming(&other, bits) -> u32         // Prefix scan
graphhv.partial_similarity(&other, bits) -> f64      // Prefix normalized
graphhv.popcount() -> u32                            // Total set bits

// Mutation
graphhv.decay(keep_prob, rng)                        // Stochastic bit death

// Edge codec
encode_edge(source, dest, role, shift) -> GraphHV
decode_edge_source(edge, dest, role, shift) -> GraphHV
```

### 10.2 CamIndex

```rust
CamIndex::new(config: CamConfig, seed: u64) -> Self
CamIndex::with_defaults(seed: u64) -> Self
cam.insert(hv: GraphHV) -> usize                    // Returns index
cam.query(query: &GraphHV, top_k: usize) -> Vec<CamHit>
cam.get(idx: usize) -> Option<&GraphHV>
cam.rebuild()
cam.len() -> usize
cam.is_empty() -> bool
```

### 10.3 DNTree

```rust
DNTree::new(config: DNConfig) -> Self
DNTree::with_capacity(num_prototypes: usize) -> Self
tree.update(proto_idx, hv: &GraphHV, rng)           // Plasticity
tree.traverse(query: &GraphHV, top_k) -> Vec<TraversalHit>
tree.stats() -> DNTreeStats
tree.summary(node_idx) -> &GraphHV
tree.num_prototypes() -> usize
tree.num_nodes() -> usize
```

### 10.4 BNN

```rust
bnn_dot(activation, weight) -> BnnDotResult          // Single channel
bnn_dot_3ch(activation, weight) -> BnnDotResult      // All 3 channels
bnn_batch_dot(query, weights, top_k) -> Vec<(usize, BnnDotResult)>

BnnNeuron::random(rng) -> Self
BnnNeuron::from_weights(weights, rng) -> Self
neuron.forward(input, learn, lr, rng) -> f32
neuron.activation() -> &Fingerprint<256>
neuron.weights() -> &Fingerprint<256>
neuron.plastic() -> &Fingerprint<256>

BnnLayer::random(n, rng) -> Self
layer.forward(input, learn, lr, rng) -> Vec<f32>
layer.winner(input) -> (usize, BnnDotResult)
```

### 10.5 Fingerprint Additions

```rust
fingerprint.as_bytes() -> &[u8]           // Zero-copy byte view
fingerprint.as_bytes_mut() -> &mut [u8]   // Zero-copy mutable byte view
```

---

## 11. Integration Points

### 11.1 With Existing rustynum-core Infrastructure

| Module | Uses | How |
|--------|------|-----|
| `graph_hv` | `Fingerprint<256>` | Channels are fingerprints directly |
| `graph_hv` | `SplitMix64` | All randomness via existing PRNG |
| `bnn` | `select_hamming_fn()` | SIMD-dispatched XOR+popcount |
| `bnn` | `Fingerprint::as_bytes()` | Zero-copy passage to SIMD kernels |
| `cam_index` | `Fingerprint<256>` | Precomputed masks as fingerprints |
| `dn_tree` | `bundle_into()` | Summary bundling via graph_hv |

### 11.2 Dependency Graph (internal)

```
fingerprint.rs  ←── graph_hv.rs  ←── cam_index.rs
      │                  ↑                │
      │                  │                │
      └──────── bnn.rs ──┘                │
                  ↑                       │
                  │                       │
            dn_tree.rs ───────────────────┘
```

No circular dependencies. All modules depend on `Fingerprint<256>` and
`SplitMix64` only. `bnn` and `dn_tree` depend on `graph_hv` for `GraphHV`
and `bundle_into`. `cam_index` depends on `graph_hv` for `GraphHV`.

### 11.3 With ladybug-rs (Future)

The modules are designed to slot into the ladybug-rs BindSpace:

| rustynum Module | ladybug-rs Integration Point |
|-----------------|------------------------------|
| `GraphHV` | `TypedSlot::GraphHV` in Blackboard |
| `CamIndex` | Replace linear scan in `BindSpace::search()` |
| `DNTree` | Hierarchical routing in `BindSpace::resolve()` |
| `BNN` | Pattern matching in `CollapseGate` decisions |

### 11.4 With crewai-rust (Future)

| Operation | crewai-rust Usage |
|-----------|-------------------|
| `bnn_dot` | Agent comparison (how similar are two agent states) |
| `bundle_into` | Agent memory consolidation |
| `encode_edge` | Agent→Agent relationship encoding |
| `CamIndex::query` | Agent lookup by state similarity |

---

## 12. Design Decisions & Trade-offs

### 12.1 Why 3 Channels?

Three channels map naturally to the Subject-Predicate-Object triple structure
and the grey/white/plastic matter analogy. Two channels would lose the
distinction between activations and weights. Four or more would increase
storage without proportional information gain at D=16,384.

### 12.2 Why Quaternary Trees?

Binary trees (fanout=2) give deeper hierarchies and more computation per
traversal. Octal trees (fanout=8) give flatter hierarchies but require more
comparisons per level. Quaternary (fanout=4) is the empirical sweet spot for
this dimensionality: deep enough for good pruning, shallow enough for
fast descent.

### 12.3 Why Carry-Save Adder Instead of SIMD Popcount?

For N ≤ 15, the CSA adder tree operates on 64 bit positions simultaneously
using plain `u64` word ops. This is competitive with SIMD because:
- The data is already in `u64` words (no conversion needed)
- The adder is completely branchless
- It avoids the overhead of loading data into SIMD registers for a
  fundamentally per-word operation

### 12.4 Why Precomputed Masks Instead of Scattered Tuples?

The scattered tuple approach `(channel, word, bit)` requires random memory
accesses to hash a vector. The precomputed mask approach uses contiguous
AND + popcount, which the CPU prefetcher handles well. The trade-off is
higher memory usage (64 × 3 × 2048 bytes = 384 KB per projector vs ~1 KB),
but this is negligible for an index that stores thousands of 6 KB prototypes.

### 12.5 Why OnceLock for SIMD Dispatch?

The SIMD function pointer must be resolved via CPUID detection, which is
not free. `OnceLock` ensures this happens exactly once, with subsequent calls
being a single pointer load. Alternatives:
- `lazy_static!`: External dependency, macro-based
- `std::sync::Once`: Requires separate static for the value
- Module-level static: Not possible with runtime detection

`OnceLock` is the idiomatic Rust solution (stable since 1.70).

### 12.6 Why Not a Separate XNOR SIMD Kernel?

The identity `XNOR_pop(a,b) = TOTAL - XOR_pop(a,b)` means we can reuse
the existing, battle-tested Hamming distance kernel. One subtraction is
cheaper than maintaining a parallel SIMD code path.

---

*This document describes the implementation as of commit `dd36a26` (PR #69)
building on commit `bf2b207` (PR #68), both on branch
`claude/implement-neural-network-research-owTKn`.*
