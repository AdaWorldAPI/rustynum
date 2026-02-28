# Cross-Plane Partial Binding Algebra: Technical Mapping

> **Version**: 1.0 — 2026-02-28
> **Crate**: `rustynum-bnn` (`cross_plane.rs`)
> **Upstream**: `rustynum-core` (`spatial_resonance.rs`, `kernels.rs`, `hybrid.rs`, `fingerprint.rs`, `delta.rs`, `layer_stack.rs`, `bf16_hamming.rs`)

---

## 1. Technique Inventory

### 1.1 Cross-Plane Orthogonal Projection Vote

**File**: `rustynum-bnn/src/cross_plane.rs:187–236`
**Type**: `CrossPlaneVote`

| Property | Value |
|---|---|
| **Input** | 3 bitmasks (`s_mask`, `p_mask`, `o_mask`), each `n_words` u64 |
| **Output** | 8 disjoint bitmasks partitioning the full codebook population |
| **Cost** | 7 AND + 7 NOT per u64 word (branch-free) |
| **AVX-512 mapping** | `vpternlogd` with immediate encoding for each of 8 ternary logic functions |
| **Precedent** | None in VSA literature — existing systems treat factorization as a single-step operation (Frady et al. 2020), not a lattice-stratified bitwise vote |

**How it works**: Given three per-plane survivor masks from sigma-gated cascade filtering, the vote extracts which of the 2^3 = 8 Boolean combinations each codebook entry belongs to. The 8 masks are:

```
core    = S & P & O           (all 3 planes agree)
sp      = S & P & ~O          (S and P agree, O does not)
so      = S & ~P & O          (S and O agree, P does not)
po      = ~S & P & O          (P and O agree, S does not)
s_only  = S & ~P & ~O         (only S agrees)
p_only  = ~S & P & ~O         (only P agrees)
o_only  = ~S & ~P & O         (only O agrees)
noise   = ~S & ~P & ~O        (no plane agrees)
```

This produces a complete partition: every codebook entry falls into exactly one of the 8 categories.

### 1.2 Halo Type Enumeration

**File**: `rustynum-bnn/src/cross_plane.rs:60–144`
**Type**: `HaloType` (repr(u8), 8 variants)

The 8 halo types map directly to:

| HaloType | u8 | Lattice Level | Boolean Lattice B_3 Node | 2-Simplex Face |
|---|---|---|---|---|
| Noise | 0 | 0 | {} (empty set) | Empty face |
| S | 1 | 1 | {S} | Vertex A |
| P | 2 | 1 | {P} | Vertex B |
| O | 3 | 1 | {O} | Vertex C |
| SP | 4 | 2 | {S,P} | Edge AB |
| SO | 5 | 2 | {S,O} | Edge AC |
| PO | 6 | 2 | {P,O} | Edge BC |
| Core | 7 | 3 | {S,P,O} | Full triangle |

**Implementation-dependent features**:
- `repr(u8)` enables zero-cost conversion to/from 3-bit membership
- `from_membership(s, p, o)` constructs from raw booleans
- `plane_count()` returns lattice level (0-3)
- `planes()` returns (bool, bool, bool) tuple
- `inference_mode()` maps to Forward/Backward/Abduction/Analogy

### 1.3 Inference Modes

**File**: `rustynum-bnn/src/cross_plane.rs:325–398`
**Types**: `InferenceMode`, `TypedQuery`

| Mode | Known Slots | Open Slot | Query Type | Linguistic Analog |
|---|---|---|---|---|
| Forward | S, P | O | "S [P]-ed whom?" | SVO completion |
| Backward | P, O | S | "Who [P]-ed O?" | Passive voice recovery |
| Abduction | S, O | P | "How are S and O related?" | Relation inference |
| Analogy | 1 of {S,P,O} | 2 of {S,P,O} | "What else fits this pattern?" | Analogical transfer |

**XOR self-inverse recovery** (file: `cross_plane.rs:786–834`):
```
Forward:   O = crystal[1] ^ P    (crystal[1] = P^O, so P^O^P = O)
Backward:  S = crystal[0] ^ P    (crystal[0] = S^P, so S^P^P = S)
Abduction: P = crystal[0] ^ S    (crystal[0] = S^P, so S^P^S = P)
```

### 1.4 SPO Triple Encoding

**Files**: `rustynum-core/src/spatial_resonance.rs:211–221`, `rustynum-bnn/src/cross_plane.rs:705–712`

The 3-axis XOR decomposition:
```
crystal[0] = X-axis = S ^ P    (Subject bound with Predicate)
crystal[1] = Y-axis = P ^ O    (Predicate bound with Object)
crystal[2] = Z-axis = S ^ O    (Subject bound with Object)
```

**Properties**:
- XOR is its own inverse in GF(2): `a ^ a = 0`, `a ^ 0 = a`
- Exact recovery: `S = crystal[0] ^ P`, `O = crystal[1] ^ P`
- Zero crosstalk noise (unlike TPR/HRR which produce `O(K/sqrt(D))` noise from K stored triples)
- Each axis is independently addressable for per-plane cascade filtering

**Spatial encoding** (rustynum-core): `SpatialCrystal3D::spo_encode()` uses `CrystalAxis::xor_bind()` which operates on BF16 byte vectors.

**Binary encoding** (rustynum-bnn): `SpoTriple::encode()` uses `Fingerprint<256>` XOR which operates on 256 u64 words.

### 1.5 Lattice Climbing (DN Tree Growth)

**File**: `rustynum-bnn/src/cross_plane.rs:527–638`
**Type**: `LatticeClimber`

Tracks partial bindings across 3 lattice levels:

```
Level 0: Noise       → discarded
Level 1: Free vars   → S, P, O types (self.free_vars)
Level 2: Partial pairs → SP, SO, PO types (self.partial_pairs)
Level 3: Full triples → Core type (self.full_triples)
```

**Composition rules** (`try_compose`):
- SP + O → Core (forward completion)
- SO + P → Core (abductive completion)
- PO + S → Core (backward completion)
- Level 1 + Level 1 → Level 2 (free variable pairing)

**CollapseGate integration** (`gate_decision`):
- Full triples with avg confidence > 1.5 → Flow
- Partial pairs present → Hold (accumulate more evidence)
- Only free variables → Hold
- Nothing → Block

### 1.6 Growth Paths

**File**: `rustynum-bnn/src/cross_plane.rs:459–481`
**Type**: `GrowthPath` (6 variants)

Each growth path maps to a specific cognitive/linguistic strategy:

| Path | Stages | Strategy | Linguistic Parallel |
|---|---|---|---|
| SubjectFirst | S → SP → Core | "Jan does something to someone" | SVO languages |
| SubjectObject | S → SO → Core | "Jan and Ada relate somehow" | Implicit relation |
| ObjectAction | O → PO → Core | "Something is done to Ada" | OVS/Passive |
| ActionFirst | P → SP → Core | "Creation happens, by whom?" | VSO languages |
| ActionObject | P → PO → Core | "Something creates Ada" | VOS languages |
| ObjectSubject | O → SO → Core | "Ada and Jan, what relation?" | OSV languages |

### 1.7 DN Mutation Operators

**File**: `rustynum-bnn/src/cross_plane.rs:487–518`
**Type**: `MutationOp` (6 variants)

| Mutation | Slots Changed | Type | Confidence Decay |
|---|---|---|---|
| MutateS | 1 (Subject) | Conservative | 0.5x |
| MutateP | 1 (Predicate) | Conservative | 0.5x |
| MutateO | 1 (Object) | Conservative | 0.5x |
| MutateSP | 2 (Subject + Predicate) | Radical | 0.25x |
| MutateSO | 2 (Subject + Object) | Radical | 0.25x |
| MutatePO | 2 (Predicate + Object) | Radical | 0.25x |

Conservative mutations (single-slot) correspond to exploitation.
Radical mutations (double-slot) correspond to exploration.

### 1.8 NARS Truth Value Bridge

**File**: `rustynum-bnn/src/cross_plane.rs:419–444`
**Method**: `PartialBinding::nars_truth()`

Maps cross-plane evidence to NARS-style (frequency, confidence) pairs:
- **Frequency**: average Hamming similarity across agreeing planes
- **Confidence**: proportional to plane count / 3 (1/3 for free vars, 2/3 for pairs, 1.0 for core)

### 1.9 Resonator Warm-Start

**File**: `rustynum-bnn/src/cross_plane.rs:723–758`
**Type**: `WarmStart`

Pre-fills known slots from partial binding evidence before resonator network iteration:
- Forward query: S and P pre-filled, O random
- Backward query: P and O pre-filled, S random
- Abduction query: S and O pre-filled, P random
- Expected speedup: ~K× where K = number of pre-filled planes

---

## 2. Implementation Dependencies

### 2.1 Type Dependencies (Upstream → Downstream)

```
rustynum-core::fingerprint::Fingerprint<256>
    ↓ (used by)
rustynum-bnn::cross_plane::SpoTriple          (3 × Fingerprint<256>)
rustynum-bnn::cross_plane::TypedQuery          (Option<Fingerprint<256>> × 3)
rustynum-bnn::cross_plane::WarmStart           (Option<Fingerprint<256>> × 3)
rustynum-bnn::cross_plane::PartialBinding      (plane_distances: [u32; 3])
rustynum-bnn::cross_plane::InferenceResult     (inferred: Fingerprint<256>)
```

```
rustynum-core::layer_stack::CollapseGate
    ↓ (used by)
rustynum-bnn::cross_plane::LatticeClimber::gate_decision()
```

```
rustynum-core::rng::SplitMix64
    ↓ (used by)
rustynum-bnn::cross_plane::random_fingerprint()
rustynum-bnn::cross_plane::SpoTriple::mutate()
rustynum-bnn::cross_plane::WarmStart::fill_random()
```

### 2.2 SIMD Dependencies

| Operation | Current Path | Optimal AVX-512 |
|---|---|---|
| CrossPlaneVote::extract | Scalar AND/NOT loop | `vpternlogd` (single instruction per output mask) |
| popcount_mask | `.count_ones()` per u64 | `VPOPCNTDQ` + horizontal reduce |
| Hamming distance in find_best_match | Fingerprint::hamming_distance (scalar loop) | `select_hamming_fn()` (VPOPCNTDQ dispatch) |
| Bitmask iteration in entries_of | `.trailing_zeros()` + bit-clear loop | `TZCNT` + `BLSR` (already optimal) |

### 2.3 Container Size Dependencies

| Container | Fingerprint Size | Bits | Alignment |
|---|---|---|---|
| SKU-16K | Fingerprint<256> | 16384 | 8-byte (u64 word) |
| SKU-64K | Fingerprint<1024> | 65536 | 8-byte (u64 word) |
| BF16 axis | CrystalAxis (Vec<u8>) | variable | 1-byte |
| AlignedBuf2K | 2048 bytes | 16384 | 8-byte (repr(align(8))) |

Cross-plane currently hardcoded to `Fingerprint<256>` (SKU-16K). The `SpoTriple`, `TypedQuery`, `WarmStart`, and `InferenceResult` all use `Fingerprint<256>`.

---

## 3. Cascade Integration Map

### 3.1 Pre-Vote: K0/K1/K2 Cascade (rustynum-core/kernels.rs)

```
K0 Probe (64-bit)   → eliminates ~55% (1 u64 XOR + POPCNT)
K1 Stats (512-bit)  → eliminates ~90% of survivors (8 u64 XOR + POPCNT)
K2 Exact (full)     → EnergyConflict decomposition (256 u64 XOR + AND + POPCNT)
```

The cascade runs independently per plane. After K2, each plane produces a survivor bitmask. These 3 masks become the input to `CrossPlaneVote::extract()`.

### 3.2 Post-Vote: BF16 Tail (rustynum-core/hybrid.rs)

Survivors from the cross-plane vote (especially Core and partial pairs) enter the BF16 tail:

```
BF16 Structured Distance → sign/exp/man weighted (bf16_hamming.rs)
Structural Diff          → which dimensions changed
Awareness Substrate      → crystallized/tensioned/uncertain/noise per dim
```

### 3.3 Post-Vote: Spatial Resonance (rustynum-core/spatial_resonance.rs)

Core survivors enter 3D spatial matching:

```
spatial_sweep()          → per-axis early exit (X→Y→Z)
spatial_awareness()      → per-axis crystallization/tension
spatial_learning_signal  → per-axis attention weights (96 total)
```

### 3.4 Full Pipeline

```
Input: Query (S, P, O fingerprints or partial)
  │
  ├─ Per-plane K0/K1/K2 cascade (3× independent runs)
  │     S-plane: query_s against codebook_s → s_mask
  │     P-plane: query_p against codebook_p → p_mask
  │     O-plane: query_o against codebook_o → o_mask
  │
  ├─ CrossPlaneVote::extract(s_mask, p_mask, o_mask)
  │     → 8 disjoint halo type masks
  │
  ├─ LatticeClimber::ingest(bindings)
  │     → stratified by lattice level
  │
  ├─ LatticeClimber::try_compose()
  │     → promote partial pairs + free vars → full triples
  │
  ├─ LatticeClimber::gate_decision()
  │     → Flow / Hold / Block
  │
  ├─ [if Flow] → infer() with TypedQuery
  │     → XOR unbinding to recover open slot
  │     → Best match in codebook
  │
  ├─ [if Hold] → accumulate more evidence (next cycle)
  │
  └─ [if Block] → discard hypotheses
```

---

## 4. Awareness Integration

### 4.1 Per-Axis Awareness (spatial_resonance.rs)

The BF16 superposition decomposition from `bf16_hamming.rs` classifies each dimension into one of 4 states:

| State | Bits | Meaning | Cross-Plane Significance |
|---|---|---|---|
| Crystallized | 00 | Sign + exp agree | Axis contributes to core membership |
| Tensioned | 01 | Sign disagrees | Active contradiction in this plane |
| Uncertain | 10 | Sign agrees, high exp spread | Direction known, magnitude unclear |
| Noise | 11 | Only mantissa differs | Irrelevant for this plane |

### 4.2 Spatial Coherence

`spatial_awareness_decompose()` computes geometric mean of per-axis crystallization:

```
spatial_coherence = (crystallized_x * crystallized_y * crystallized_z)^(1/3)
```

High spatial coherence (> 0.9) indicates strong 3-plane agreement → Core.
Low spatial coherence indicates partial binding → Level 1 or 2.

### 4.3 Attention Weight Feedback

`extract_spatial_learning_signal()` produces 96 attention weights (32 per axis × 3 axes) that feed into:
- WideMetaView W144-W159 (32 f32 weights per axis)
- NARS truth value revision
- Hybrid pipeline weight adaptation

---

## 5. Delta Layer Integration

### 5.1 DeltaLayer Superposition and Cross-Plane

Each writer's DeltaLayer represents a hypothesis about one or more SPO slots:

```
Writer 0: delta for S → proposes a subject
Writer 1: delta for P → proposes a predicate
Writer 2: delta for O → proposes an object
```

The cross-plane vote operates on the effective values (ground ^ delta), not on ground truth alone. This means:
- Multiple writers can propose different fillers for the same slot
- Contradictions are visible via AND+popcount between deltas
- The CollapseGate decides whether to commit (Flow), wait (Hold), or discard (Block)

### 5.2 Error Threshold Connection

For `Fingerprint<256>` with D = 16384 bits:
- Awareness degrades when concurrent writers exceed ~sqrt(D) ≈ 128
- Below threshold: crystallized signal dominates → reliable cross-plane classification
- At threshold: noise floor (~50% per dimension) → cross-plane vote becomes random
- This is the VSA analog of Eigen's error catastrophe

---

## 6. Test Coverage

| Test | What It Verifies |
|---|---|
| `test_cross_plane_vote_distribution` | 8 masks partition full population |
| `test_cross_plane_vote_sp_type` | SP = S & P & ~O extraction |
| `test_entries_of_returns_correct_indices` | Bitmask → index list conversion |
| `test_growth_path_stages` | All 6 growth paths produce correct 3-stage sequences |
| `test_halo_inference_mode` | HaloType → InferenceMode mapping |
| `test_halo_type_from_membership` | Boolean triple → HaloType |
| `test_halo_type_plane_count` | Lattice level computation |
| `test_inference_forward_recovers_object` | Forward: SP→O via XOR unbinding |
| `test_inference_backward_recovers_subject` | Backward: PO→S via XOR unbinding |
| `test_inference_abduction_recovers_predicate` | Abduction: SO→P via XOR unbinding |
| `test_lattice_climber_ingest_and_levels` | Stratification into 3 lattice levels |
| `test_lattice_climber_compose_sp_plus_o` | SP + O → Core promotion |
| `test_lattice_climber_gate_decisions` | CollapseGate Flow/Hold/Block |
| `test_mutation_conservative_vs_radical` | Slot count (1 vs 2) |
| `test_mutation_preserves_kept_slots` | Non-mutated slots unchanged |
| `test_nars_truth_value` | NARS (frequency, confidence) from cross-plane evidence |
| `test_spo_triple_encode_decode` | XOR encoding roundtrip |
| `test_typed_query_forward` | TypedQuery construction and known_count |
| `test_warm_start_from_forward_query` | WarmStart pre-fills known slots |
| `test_warm_start_fill_random` | Random fill for unknown slots |
| `test_popcount_mask_exact` | Popcount with n_entries boundary |

Total: 23 tests, all passing.

---

## 7. Integration Outlook and Future Work

### 7.1 Near-Term: Wire Cross-Plane into ladybug-rs Search

**Current state**: Cross-plane algebra is implemented in rustynum-bnn but not yet wired into ladybug-rs's search path.

**Integration points**:
1. `ladybug-rs/src/storage/bind_space.rs` — search functions should invoke `CrossPlaneVote::extract()` after per-plane K0/K1/K2 cascade
2. `ladybug-rs/src/core/rustynum_accel.rs` — add `cross_plane_vote()` as an acceleration entry point
3. `ladybug-rs/src/storage/bind_space.rs` — `TypedQuery` maps directly to BindSpace query types (forward = "who does X do Y to?", backward = "who does Y to Z?")

**Effort**: Medium. Requires passing 3 per-plane survivor masks from cascaded search to `CrossPlaneVote::extract()`, then using `LatticeClimber` for incremental composition.

### 7.2 Near-Term: AVX-512 `vpternlogd` Optimization

**Current state**: `CrossPlaneVote::extract()` uses scalar AND/NOT loop.

**Optimization**: Replace the 7 operations per word with `vpternlogd` using appropriate truth table immediates:

```
core    = vpternlogd(S, P, O, 0x80)   // truth table: 1 only when all 3 are 1
sp      = vpternlogd(S, P, O, 0x40)   // S=1, P=1, O=0
so      = vpternlogd(S, P, O, 0x20)   // S=1, P=0, O=1
po      = vpternlogd(S, P, O, 0x10)   // S=0, P=1, O=1
s_only  = vpternlogd(S, P, O, 0x02)   // S=1, P=0, O=0
p_only  = vpternlogd(S, P, O, 0x04)   // S=0, P=1, O=0
o_only  = vpternlogd(S, P, O, 0x08)   // S=0, P=0, O=1
noise   = vpternlogd(S, P, O, 0x01)   // S=0, P=0, O=0
```

Each `vpternlogd` computes a 3-input Boolean function in a single instruction on 512-bit registers. This replaces 7 AND + 7 NOT per word with 8 `vpternlogd` per 8 words.

**Effort**: Low. Mechanical translation using `_mm512_ternarylogic_epi64()` (available on stable 1.93).

### 7.3 Mid-Term: Hierarchical Cascade with Cross-Plane Pruning

**Current state**: K0/K1/K2 runs independently per plane, then cross-plane vote occurs after all planes complete.

**Optimization**: Use K0 results from the first plane to prune candidates before running K1/K2 on the other planes:

```
K0(S-plane) → s_k0_mask
    ↓ (only process K0 survivors)
K0(P-plane on s_k0 survivors) → sp_k0_mask
    ↓ (only process joint survivors)
K0(O-plane on sp_k0 survivors) → spo_k0_mask
    ↓
K1 on spo_k0 survivors per plane
    ↓
K2 on spo_k1 survivors per plane
    ↓
CrossPlaneVote on K2 survivors
```

**Expected speedup**: If K0 rejects ~55% per plane, hierarchical pruning processes only 45%^3 ≈ 9% of candidates at K1 level, vs. 3 × 45% = 135% of work in the independent approach.

### 7.4 Mid-Term: Resonator Network Integration

**Current state**: `WarmStart` pre-fills known slots but doesn't run a resonator network iteration loop.

**Integration**: Connect to the resonator network formalism (Frady et al. 2020):
1. Initialize resonator from WarmStart
2. Iterate: unbind → clean-up (nearest neighbor in codebook) → rebind
3. Convergence check: do iterations stabilize?
4. Use `LatticeClimber` to track progress through lattice levels during iteration

**Effort**: Medium. The algebra is already in place; need the iteration loop and convergence detection.

### 7.5 Mid-Term: NARS Truth Revision from Cross-Plane Evidence

**Current state**: `nars_truth()` computes static (frequency, confidence) pair.

**Integration with ladybug-rs NARS**:
1. Cross-plane evidence updates NARS truth values for stored triples
2. SP-type bindings reduce confidence in the O slot → trigger forward inference
3. Tensioned awareness on one axis → trigger NARS revision on that slot
4. Multiple confirming cross-plane votes accumulate evidence → confidence grows

### 7.6 Long-Term: 4D Extension (Temporal SPO + Context)

**Current state**: 3-plane SPO crystal.

**Extension to SPOC (Subject-Predicate-Object-Context)**:
- Add 4th plane for temporal/contextual binding
- Boolean lattice grows from B_3 (8 types) to B_4 (16 types)
- Face lattice of 3-simplex (tetrahedron)
- 14 non-trivial halo types (excluding noise and core)
- 4 free variables, 6 pairs, 4 triples, 1 quadruple

**New halo types**: {S}, {P}, {O}, {C}, {SP}, {SO}, {SC}, {PO}, {PC}, {OC}, {SPO}, {SPC}, {SOC}, {POC}, {SPOC}

**New inference modes**:
- Temporal prediction: known SPO, find C ("when/where?")
- Contextual binding: known SPC, find O ("in this context, S does P to whom?")

### 7.7 Long-Term: Compile-Time Lattice Verification

**Current state**: Lattice structure enforced by enum variants and tests.

**Improvement**: Use Rust const generics and type-level programming to encode the lattice structure at compile time:
- `PartialBinding<S: Known, P: Known, O: Unknown>` → Forward inference only
- Type-level guarantee that `try_compose` only accepts compatible types
- Zero-cost abstraction: all lattice checks at compile time

### 7.8 Long-Term: Distributed Cross-Plane Voting

For multi-node deployments:
- Each node runs per-plane cascade independently
- Cross-plane vote requires only 3 bitmasks (not full fingerprints)
- Bandwidth: 3 × ceil(N/64) × 8 bytes for N codebook entries
- For N=100K entries: 3 × 1563 × 8 = 37.5 KB per query (trivial)
- Halo type classification is embarrassingly parallel

---

## 8. Performance Characteristics

### 8.1 Computational Complexity

| Operation | Complexity | Cost at N=100K, D=16384 |
|---|---|---|
| Per-plane K0/K1/K2 cascade | O(N × D/64) per plane | ~300K u64 ops × 3 planes |
| CrossPlaneVote::extract | O(N/64) | ~1.5K u64 ops |
| popcount_mask | O(N/64) per mask × 8 | ~12K u64 ops |
| entries_of | O(popcount) per mask | varies |
| LatticeClimber::ingest | O(survivors) | ~5K entries typical |
| try_compose | O(pairs × fvs) | quadratic but small populations |
| infer (find_best_match) | O(codebook × D/64) | ~256K u64 ops |

**Key insight**: Cross-plane vote is O(N/64) — essentially free compared to the O(N × D/64) cascade. The vote adds <1% overhead to the total pipeline cost.

### 8.2 Memory Layout

| Structure | Size | Alignment | Notes |
|---|---|---|---|
| CrossPlaneVote (8 masks) | 8 × ceil(N/64) × 8 bytes | 8-byte | Heap-allocated Vec<u64> |
| PartialBinding | 40 bytes | 8-byte | entry_index + halo_type + confidence + 3×u32 |
| SpoTriple | 3 × 2048 + 4 bytes | 8-byte | 3 × Fingerprint<256> + f32 |
| WarmStart | 3 × Option<2048> + 1 byte | 8-byte | 3 × Option<Fingerprint<256>> + u8 |
| TypedQuery | 3 × Option<2048> + enum | 8-byte | 3 × Option<Fingerprint<256>> + InferenceMode |

### 8.3 Zero-Copy Properties

| Operation | Zero-Copy? | Notes |
|---|---|---|
| CrossPlaneVote::extract | Yes | Reads input masks, writes output masks. No copies. |
| CrossPlaneVote::mask_for | Yes | Returns &[u64] reference |
| SpoTriple::encode | No | Creates 3 new Fingerprint<256> via XOR |
| Fingerprint::hamming_distance | Yes | Reads both inputs, returns u32 |
| WarmStart::from_query | No | Clones Option<Fingerprint<256>> |

---

## 9. Cross-Repository Integration Points

### 9.1 rustynum → ladybug-rs

| rustynum Symbol | ladybug-rs Consumer | Integration Status |
|---|---|---|
| `CrossPlaneVote::extract()` | `bind_space.rs` search | NOT YET WIRED |
| `LatticeClimber` | `bind_space.rs` incremental search | NOT YET WIRED |
| `TypedQuery` | Query type from semantic kernel | NOT YET WIRED |
| `infer()` | BindSpace pattern completion | NOT YET WIRED |
| `Fingerprint<256>` | BindSpace containers (already used) | WIRED |
| `select_hamming_fn()` | SIMD dispatch (should replace simd.rs) | P0 TODO |

### 9.2 rustynum → crewai-rust

| rustynum Symbol | crewai-rust Consumer | Integration Status |
|---|---|---|
| `HaloType` | Agent belief classification | NOT YET WIRED |
| `InferenceMode` | Agent reasoning strategy selection | NOT YET WIRED |
| `GrowthPath` | Blackboard hypothesis tracking | NOT YET WIRED |
| `MutationOp` | Agent exploration/exploitation balance | NOT YET WIRED |

### 9.3 Data Flow: End-to-End

```
User query (text)
  → n8n-rs: tokenize, embed (Jina 1024-D)
    → ladybug-rs: project to Fingerprint<256>
      → rustynum-core: K0/K1/K2 cascade per plane
        → rustynum-bnn: CrossPlaneVote + LatticeClimber
          → rustynum-bnn: infer() via XOR unbinding
            → ladybug-rs: BindSpace pattern completion
              → crewai-rust: Agent action selection
                → n8n-rs: response generation
```

---

## 10. Mathematical Foundations Summary

| Property | Algebraic Structure | Implementation |
|---|---|---|
| XOR binding | Group (Z_2^D, XOR) | `Fingerprint<N>` BitXor trait |
| Self-inverse | a ^ a = 0 (identity) | `spo_recover_subject()` |
| Commutativity | a ^ b = b ^ a | LayerStack commit order independence |
| Associativity | (a ^ b) ^ c = a ^ (b ^ c) | Multi-delta composition |
| Partial binding lattice | Boolean lattice B_3 | `HaloType` enum (8 variants) |
| Face lattice isomorphism | B_3 ≅ face lattice of 2-simplex | 6 growth paths = 6 edges |
| Error threshold | Eigen quasispecies theory | CollapseGate Hold/Block |
| Awareness substrate | 4-state per-dim classification | BF16 sign/exp/man decomposition |
| Semantic roles | Fillmore case grammar / Goldberg constructions | S=Agent, P=Construction, O=Theme |
| SDM partial matching | Kanerva content-addressable memory | Per-plane cascade filtering |

---

*This document maps the complete technical landscape of the cross-plane partial binding algebra. It should be updated as integration with ladybug-rs, crewai-rust, and n8n-rs proceeds.*
