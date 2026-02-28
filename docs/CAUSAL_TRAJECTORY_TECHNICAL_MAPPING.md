# Causal Trajectory Hydration: Technical Mapping

> **Version**: 1.0 — 2026-02-28
> **Crate**: `rustynum-bnn` (`causal_trajectory.rs`)
> **Upstream**: `rustynum-core` (`fingerprint.rs`, `layer_stack.rs`), `rustynum-bnn` (`cross_plane.rs`, `rif_net_integration.rs`)

---

## 1. Technique Inventory

### 1.1 Resonator Snapshot Recording

**File**: `causal_trajectory.rs:72–114`
**Type**: `ResonatorSnapshot`

| Field | Type | Size | Purpose |
|---|---|---|---|
| `iter` | u16 | 2B | Iteration counter |
| `s_est` | Fingerprint<256> | 2048B | Subject estimate (16384 bits) |
| `p_est` | Fingerprint<256> | 2048B | Predicate estimate |
| `o_est` | Fingerprint<256> | 2048B | Object estimate |
| `s_mask` | Vec<u64> | N/64 words | S-plane cascade survivor bitmask |
| `p_mask` | Vec<u64> | N/64 words | P-plane cascade survivor bitmask |
| `o_mask` | Vec<u64> | N/64 words | O-plane cascade survivor bitmask |
| `n_entries` | usize | 8B | Codebook population size |
| `delta_s` | u32 | 4B | Hamming(s_est[t], s_est[t-1]) |
| `delta_p` | u32 | 4B | Hamming(p_est[t], p_est[t-1]) |
| `delta_o` | u32 | 4B | Hamming(o_est[t], o_est[t-1]) |

**Total per snapshot**: ~6 KB fixed + 3 × ceil(N/64) × 8 bytes for masks.
For N=100K: ~6 KB + 37.5 KB = ~44 KB per iteration.
For 10 iterations: ~440 KB total trajectory.

**Methods**:
- `total_delta()`: Sum of all plane convergence deltas
- `converged(threshold)`: True when all deltas below threshold
- `cross_plane_vote()`: Extract typed halo via `CrossPlaneVote::extract()`

### 1.2 RIF Diff — Causal Change Recording

**File**: `causal_trajectory.rs:126–179`
**Type**: `RifDiff`

XOR between snapshots at iterations t and t-2. Records WHAT CHANGED.

| Operation | Input | Output | Cost |
|---|---|---|---|
| `RifDiff::compute(earlier, later)` | 2 × ResonatorSnapshot | RifDiff | 3 × 256 XOR + 3 × 256 POPCNT |

Each RifDiff stores:
- Per-plane XOR fingerprints (s_diff, p_diff, o_diff): the bitwise delta
- Per-plane activity counts (popcount of each diff): how much changed
- Dominant plane detection: which plane had the most activity

**Why t and t-2 (not t-1)**: Following Zhang et al. 2025's RIF shortcut pattern. Non-adjacent comparison prevents the trivial "everything changes a little each step" pattern from masking larger structural shifts. The 2-step gap acts as a low-pass filter on convergence dynamics.

### 1.3 EWM Correction — Causal Saliency Map

**File**: `causal_trajectory.rs:198–243`
**Type**: `EwmCorrection`

Per-iteration, per-word L1 correction magnitude. Records WHERE the resonator worked hardest.

| Field | Type | Granularity | Purpose |
|---|---|---|---|
| `s_correction[256]` | [u32; 256] | 1 value per u64 word | Popcount of XOR delta per S-plane word |
| `p_correction[256]` | [u32; 256] | 1 value per u64 word | Popcount of XOR delta per P-plane word |
| `o_correction[256]` | [u32; 256] | 1 value per u64 word | Popcount of XOR delta per O-plane word |

**CausalSaliency** (file: `causal_trajectory.rs:256–293`):

Computed from a sliding window of EwmCorrections. Classifies each word as:

| Classification | Detection Rule | Causal Meaning | DN Tree Action |
|---|---|---|---|
| **Crystallizing** | correction[t] + 2 < correction[0] | Evidence SUPPORTING factorization | Increase NARS frequency |
| **Dissolving** | correction[t] > correction[0] + 2 | Evidence UNDERMINING factorization | Decrease NARS confidence |
| **Contested** | ≥ 2 direction changes in window | COMPETING hypotheses | Speciation: CONTRADICTS edge |

The +2 threshold prevents noise from triggering false transitions.

### 1.4 BPReLU Causal Arrow — Directionality

**File**: `causal_trajectory.rs:366–439`
**Type**: `CausalArrow`

Uses the BPReLU's asymmetric slopes (α_pos=1.0, α_neg=0.25) to distinguish:

| Direction | Meaning | BPReLU Mode | Pearl Analog |
|---|---|---|---|
| **Forward** | Commitment DROVE context change | α_pos dominates | P(effect \| do(cause)) — interventional |
| **Backward** | Context OVERRODE commitment | α_neg dominates | P(cause \| effect) — observational |
| **Symmetric** | No clear direction | α_pos ≈ α_neg | Correlation without causation |
| **Contested** | Dimensions split forward/backward | Both strong | Mixed causation |

**Algorithm** (`plane_asymmetry`, line 445):
1. XOR the two estimates → diff fingerprint
2. Popcount → changed_bits
3. Normalize to stability: `1.0 - 2.0 × changed_bits / total_bits`
4. Forward activation: `BPReLU(stability)` — responds to PRESENCE (stable → forward)
5. Backward activation: `BPReLU(-stability)` — responds to ABSENCE (unstable → backward)
6. Classify: ratio > 0.7 → Forward, ratio < 0.3 → Backward, else Symmetric/Contested

**Per-plane decomposition**: Each CausalArrow has independent s_direction, p_direction, o_direction plus an aggregate `overall` direction.

### 1.5 Causal Chain — Convergence Genealogy

**File**: `causal_trajectory.rs:478–547`
**Type**: `CausalChain` (contains `Vec<CausalLink>`)

Stacks RIF diffs to extract which plane converged FIRST and which followed.

**Detection rule** (from windowed RIF diffs):
- Plane active EARLY + quiet LATE → **stabilized first** → CAUSE
- Plane quiet EARLY + active LATE → **responded later** → EFFECT
- "Active" = popcount > 2× the other window's popcount

**The 6 possible causal links map to the 6 halo types**:

| Causal Link | Halo Type | Linguistic Parallel |
|---|---|---|
| S stabilized → P responded | SP | "Subject identified → action follows" |
| S stabilized → O responded | SO | "Subject found → object found via relation" |
| P stabilized → S responded | SP (reverse) | "Action chosen → actor inferred" |
| P stabilized → O responded | PO | "Action identified → target follows" |
| O stabilized → S responded | SO (reverse) | "Object found → agent inferred" |
| O stabilized → P responded | PO (reverse) | "Object given → relation inferred" |

### 1.6 Halo Transitions — Lattice Level Movements

**File**: `causal_trajectory.rs:553–600`
**Function**: `detect_halo_transitions()`

Compares cross-plane votes between adjacent snapshots to find entries that moved between halo types.

| Transition | Level Delta | NARS Effect | Sigma Edge |
|---|---|---|---|
| Noise → S (or P, O) | +1 | f=0.8, c=0.5 (Supports) | Entry → Plane (Enables) |
| S → SP (or SO) | +1 | f=0.8, c=0.5 (Supports) | Plane → Plane (Causes) |
| SP → Core | +1 | f=0.8, c=0.7 (Supports) | Plane → Core (Causes) |
| Core → SP (or SO, PO) | -1 | f=0.2, c=0.5 (Undermines) | Core → Plane (Undermines) |
| SP → S | -1 | f=0.2, c=0.5 (Undermines) | Plane → Plane (Undermines) |

### 1.7 NARS Truth Value Engine

**File**: `causal_trajectory.rs:608–680`
**Type**: `NarsTruth`

Full NARS inference rule implementation:

| Rule | Formula | Use Case |
|---|---|---|
| **Revision** | w₁=c₁/(1-c₁), f_new=(w₁f₁+w₂f₂)/(w₁+w₂), c_new=w_total/(w_total+1) | Combining two observations of same relation |
| **Deduction** | f=f₁·f₂, c=f·c₁·c₂/(f·c₁·c₂+k) | A→B, B→C ⊢ A→C (transitive) |
| **Abduction** | f=f₁, c=f₁·f₂·c₁·c₂/(f₁·f₂·c₁·c₂+k) | A→B, C→B ⊢ A→C (same effect) |
| **Induction** | f=f₂, c=f₁·f₂·c₁·c₂/(f₁·f₂·c₁·c₂+k) | A→B, A→C ⊢ B→C (same cause) |

The `k` parameter (set to 1e-3) controls the confidence horizon — how much evidence is needed to reach high confidence.

### 1.8 Sigma Graph Edges

**File**: `causal_trajectory.rs:710–735`
**Types**: `SigmaEdge`, `SigmaNode`

Output of the causal trajectory engine — growth instructions for the DN tree.

| SigmaNode Variant | What It Represents |
|---|---|
| `Entry(usize)` | A specific codebook entry |
| `Plane(DominantPlane)` | All entries in S/P/O plane |
| `HaloGroup(HaloType)` | All entries of a specific halo type |

| CausalRelation | When Generated | NARS Confidence |
|---|---|---|
| `Causes` | CausalChain link or Forward CausalArrow | 0.5–0.6 |
| `Enables` | Halo promotion (level +1) | 0.5 |
| `Contradicts` | Contested saliency or Contested CausalArrow | 0.3 |
| `Supports` | Halo promotion with high confidence | 0.5–0.9 |
| `Undermines` | Halo demotion | 0.5–0.9 |

### 1.9 CausalTrajectory — The Full Pipeline

**File**: `causal_trajectory.rs:741–870`
**Type**: `CausalTrajectory`

The main orchestrator. Two entry points:

1. **`record_iteration(snapshot)`** — called per resonator iteration:
   - EwmCorrection from prev→curr (if prev exists)
   - CausalArrow from prev→curr (if prev exists)
   - RifDiff from t-2→t (if t-2 exists)
   - HaloTransitions from prev→curr → NARS statements

2. **`finalize()`** — called after convergence or max iterations:
   - CausalChain from stacked RifDiffs → SigmaEdges
   - CausalArrow analysis → SigmaEdges
   - CausalSaliency from EwmCorrections → NARS statements
   - CollapseGate decision from accumulated evidence

**Gate decision logic**:
- `Flow`: converged (all deltas < 100) AND supports > undermines + contradicts
- `Block`: contradicts > supports
- `Hold`: everything else

---

## 2. Dependency Graph

### 2.1 Type Dependencies

```
rustynum-core::fingerprint::Fingerprint<256>
    ↓ (used by every struct in causal_trajectory)
    ResonatorSnapshot (3 × Fingerprint<256> for estimates)
    RifDiff (3 × Fingerprint<256> for diffs)
    EwmCorrection (3 × [u32; 256] from per-word popcount)

rustynum-core::layer_stack::CollapseGate
    ↓
    CausalTrajectory::gate_decision() → CollapseGate

rustynum-bnn::cross_plane::CrossPlaneVote
    ↓
    ResonatorSnapshot::cross_plane_vote() → CrossPlaneVote
    detect_halo_transitions() uses CrossPlaneVote::entries_of()

rustynum-bnn::cross_plane::HaloType
    ↓
    HaloTransition (from/to fields)
    SigmaNode::HaloGroup(HaloType)
    halo_to_dominant_plane() mapping

rustynum-bnn::cross_plane::InferenceMode
    ↓
    NarsCausalStatement.inference_mode

rustynum-bnn::rif_net_integration::BPReLU
    ↓
    CausalArrow::compute() uses BPReLU::default() for asymmetry
```

### 2.2 Data Flow (Per Iteration)

```
ResonatorSnapshot[t]
  │
  ├─ EwmCorrection::compute(snap[t-1], snap[t])
  │    └→ per_word_popcount(XOR of estimates)
  │         └→ [u32; 256] × 3 planes
  │
  ├─ CausalArrow::compute(snap[t-1], snap[t])
  │    └→ plane_asymmetry(prev_est, curr_est, BPReLU)
  │         └→ classify_asymmetry(fwd, bwd) → CausalDirection
  │
  ├─ RifDiff::compute(snap[t-2], snap[t])      [if t ≥ 2]
  │    └→ XOR + popcount per plane
  │
  └─ detect_halo_transitions(snap[t-1], snap[t])
       └→ CrossPlaneVote::extract() on both
            └→ entries_of() per HaloType
                 └→ compare membership → HaloTransition
                      └→ generate_nars_from_transition()
                           └→ NarsCausalStatement (Supports/Undermines)
```

### 2.3 Data Flow (Finalization)

```
CausalTrajectory::finalize()
  │
  ├─ CausalChain::from_rif_diffs(all RifDiffs)
  │    └→ windowed comparison of activity
  │         └→ CausalLink (cause_plane → effect_plane)
  │              └→ SigmaEdge (Causes)
  │
  ├─ For each CausalArrow:
  │    └→ Per-plane direction analysis
  │         ├→ Forward → SigmaEdge (Causes, source→next)
  │         ├→ Backward → SigmaEdge (Causes, next→source)
  │         └→ Contested → SigmaEdge (Contradicts)
  │
  └─ CausalSaliency::from_ewm_window(all EwmCorrections)
       └→ Per-plane contested count > 10
            └→ NarsCausalStatement (Contradicts)
```

---

## 3. SIMD Optimization Opportunities

| Operation | Current | Optimal AVX-512 | Speedup |
|---|---|---|---|
| `per_word_popcount` (256 words) | `.count_ones()` loop | VPOPCNTDQ on 8 words/cycle | ~32× |
| XOR for RifDiff/EwmCorrection | Fingerprint BitXor loop | `_mm512_xor_si512` on 8 words/cycle | ~32× |
| `detect_halo_transitions` | entries_of per HaloType | `vpternlogd` extraction (already in CrossPlaneVote) | ~8× |
| `classify_word_trend` | scalar comparison loop | SIMD compare + blend | ~16× |

**Critical path**: The per-iteration cost is dominated by 3 XOR + 3 POPCNT (for EwmCorrection) plus 3 XOR + 3 POPCNT (for RifDiff). Total: 6 × 256 word operations = 1536 word ops. At 8 words/cycle on AVX-512: ~192 cycles = ~48 ns. Negligible compared to the cascade search it instruments.

---

## 4. Memory Layout

| Structure | Size (Fixed) | Size (N=100K) | Alignment |
|---|---|---|---|
| ResonatorSnapshot | 6160 B | 6160 + 37.5 KB | 8-byte |
| RifDiff | 6156 B | 6156 B (fixed) | 8-byte |
| EwmCorrection | 3074 B | 3074 B (fixed) | 4-byte |
| CausalArrow | 52 B | 52 B (fixed) | 4-byte |
| NarsTruth | 8 B | 8 B (fixed) | 4-byte |
| NarsCausalStatement | 28 B | 28 B (fixed) | 4-byte |
| SigmaEdge | 36 B | 36 B (fixed) | 4-byte |
| CausalTrajectory (10 iter) | ~100 KB | ~540 KB | 8-byte |

---

## 5. Integration Plan

### 5.1 Near-Term: Wire into ladybug-rs Resonator Loop

**Current state**: `CausalTrajectory` is a standalone data structure. Not yet called from the resonator loop.

**Integration**:
1. `ladybug-rs/src/core/resonator.rs` — after each unbind→project→rebind step, call `traj.record_iteration(snapshot)`
2. After convergence or max iterations, call `traj.finalize()`
3. Feed `traj.sigma_edges` into the Sigma Graph via `bind_space.rs` edge creation
4. Feed `traj.gate_decision()` into the CollapseGate for commit/hold/block

**Effort**: Medium. The snapshot construction requires extracting s_mask/p_mask/o_mask from the cascade search, which are already computed but not currently surfaced.

### 5.2 Near-Term: Connect NARS Statements to Truth Table

**Current state**: `NarsCausalStatement` is generated but not consumed.

**Integration**:
1. `ladybug-rs/src/storage/bind_space.rs` — stored SPO triples have NARS truth values
2. After finalization, iterate `traj.nars_statements` and call `truth.revise(new_truth)` on matching triples
3. Use `CausalRelation::Supports` to increase frequency, `Undermines` to decrease it

### 5.3 Mid-Term: Trajectory Compression for Long Resonator Runs

**Current state**: Stores all snapshots in memory.

**Optimization**: For long-running resonators (>20 iterations):
1. Keep only the last 5 snapshots (sliding window)
2. Aggregate EwmCorrections into running CausalSaliency
3. Merge CausalLinks with same cause/effect planes (revision rule)
4. Estimated memory reduction: 4× for 20-iteration runs

### 5.4 Mid-Term: Streaming Sigma Edge Emission

**Current state**: `finalize()` is batch — all edges emitted at once.

**Improvement**: Emit SigmaEdges during `record_iteration()` when confidence exceeds threshold:
1. Halo promotions with level_delta ≥ 2 → immediate edge emission
2. CausalArrow with Forward/Backward strength > 0.8 → immediate emission
3. Reduces finalization latency for real-time systems

### 5.5 Mid-Term: Multi-Trajectory Comparison

**Future**: Compare trajectories from different queries to detect shared causal structure:
1. Two queries that converge via the same CausalChain → structural similarity
2. Shared SigmaEdges across trajectories → reinforce DN tree edges (NARS revision)
3. Divergent trajectories → speciation (DN tree branching)

### 5.6 Long-Term: Temporal Context Extension (SPOC)

**Current state**: 3-plane SPO.

**Extension to SPOC** (Subject-Predicate-Object-Context):
- Add C-plane to ResonatorSnapshot (4th Fingerprint<256>)
- 4-plane cross-plane vote: B_4 = 16 halo types
- CausalArrow gets 4th direction (c_direction)
- CausalChain has 12 possible links (4×3 directional pairs)
- RifDiff has 4th diff plane

### 5.7 Long-Term: Causal DAG Extraction

**Current state**: CausalChain is a sequence of pairwise links.

**Extension**: Build a full DAG from the trajectory:
1. CausalLinks define directed edges
2. NARS truth values define edge weights
3. Cycle detection → contradiction (CollapseGate::Block)
4. Topological sort → causal ordering
5. Export as Neo4j Cypher for persistent storage

---

## 6. Test Coverage

| Test | What It Verifies |
|---|---|
| `test_nars_revision` | Equal-weight revision gives mean frequency, increased confidence |
| `test_nars_deduction` | Transitive inference produces high frequency |
| `test_nars_truth_bounds` | Frequency and confidence clamped to [0,1] |
| `test_rif_diff_identical` | Self-diff has zero activity |
| `test_rif_diff_different` | Different snapshots have nonzero activity |
| `test_ewm_correction_self_is_zero` | Self-correction is zero |
| `test_ewm_correction_nonzero` | Different snapshots have nonzero correction |
| `test_causal_arrow_identical_is_forward` | Identical snapshots → Forward (stable) |
| `test_causal_arrow_has_three_planes` | Three per-plane magnitudes |
| `test_causal_chain_empty` | Empty diffs → empty chain |
| `test_causal_chain_detects_stabilization` | S active→quiet, P quiet→active detects S→P |
| `test_saliency_needs_minimum_corrections` | < 2 corrections → all zero counts |
| `test_saliency_detects_crystallizing` | Decreasing correction → crystallizing classification |
| `test_halo_transition_detection` | Different masks produce transitions |
| `test_trajectory_record_and_finalize` | 5 iterations: 4 EWM, 4 arrows, 3 RIF diffs |
| `test_trajectory_gate_decision_empty` | Empty trajectory → Block |
| `test_trajectory_gate_decision_converged` | Converged + supports → Flow |
| `test_trajectory_gate_decision_contradicted` | Contradicts > supports → Block |
| `test_snapshot_converged` | Threshold logic correct |
| `test_classify_asymmetry_forward` | High forward ratio → Forward |
| `test_classify_asymmetry_backward` | High backward ratio → Backward |
| `test_classify_asymmetry_symmetric` | Zero inputs → Symmetric |
| `test_per_word_popcount` | Correct popcount per word |

Total: 23 tests, all passing.

---

## 7. Mathematical Foundation Summary

| Property | Structure | Implementation |
|---|---|---|
| XOR as causal diff | GF(2)^D group | `RifDiff::compute()`: XOR + popcount |
| EWM as saliency | Per-word correction magnitude | `EwmCorrection`: popcount of per-word XOR |
| BPReLU as do-calculus | Interventional vs observational | `CausalArrow`: asymmetric slope application |
| NARS revision | Bayesian-like evidence fusion | `NarsTruth::revise()`: weighted mean + confidence growth |
| NARS deduction | Syllogistic chaining | `NarsTruth::deduction()`: f₁·f₂ with confidence decay |
| Lattice climbing | Boolean lattice B_3 | `HaloTransition`: level_delta = to.level - from.level |
| Error threshold | Eigen quasispecies | `gate_decision()`: Block when contradicts > supports |
| Causal ordering | Temporal precedence | `CausalChain`: early-active + late-quiet = cause |
| Convergence genealogy | DAG of plane stabilization | `CausalChain::root_cause()`: first stabilizing plane |

---

*This document maps the complete technical landscape of the causal trajectory hydration engine. It should be updated as integration with ladybug-rs and crewai-rust proceeds.*
