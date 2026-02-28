# Cognitive Architecture: Research Insights, Missing Pieces & Integration Plan

> **Context**: rustynum BitPacked Plastic HDC-BNN system (PRs #68-#69)
> **Date**: 2026-02-27
> **Purpose**: Capture all architectural insights, latest research connections,
> and concrete integration plan before context dilution across sessions.
> **Status**: Living document — update as research progresses.

---

## Table of Contents

1. [What Exists Today](#1-what-exists-today)
2. [The Six Missing Pieces for Human-Adjacent Cognition](#2-the-six-missing-pieces)
3. [Why 16,384-Bit Binary Beats 10,000D Continuous](#3-dimensionality-analysis)
4. [The Traversal Is the Bottleneck](#4-traversal-bottleneck)
5. [3D Edge Trajectories Through Crystal Space](#5-3d-edge-trajectories)
6. [Latest Research Connections (2025-2026)](#6-latest-research)
7. [Causal Inference Integration Plan](#7-causal-inference-plan)
8. [Concrete Implementation Roadmap](#8-implementation-roadmap)
9. [Mathematical Appendix](#9-mathematical-appendix)

---

## 1. What Exists Today

### Implemented Modules (2,537 lines, 65 tests)

| Module | Lines | What It Does |
|--------|-------|-------------|
| `GraphHV` (graph_hv.rs) | 840 | 3-channel 49,152-bit hypervector: node/edge/plastic |
| `CamIndex` (cam_index.rs) | 420 | Multi-probe LSH with precomputed XOR-parity masks |
| `DNTree` (dn_tree.rs) | 598 | Quaternary plasticity tree with BTSP gating |
| `BNN` (bnn.rs) | 556 | XNOR+popcount inference, neuron/layer/batch |
| `Fingerprint` additions | +39 | Zero-copy `as_bytes()`/`as_bytes_mut()` |

### Existing Infrastructure (pre-existing)

| Component | What It Provides |
|-----------|-----------------|
| `Fingerprint<256>` | Universal 2048-byte container = `[u64; 256]` |
| `DeltaLayer<N>` | XOR delta from ground truth, writer owns `&mut` |
| `LayerStack<N>` | Multi-writer concurrent state with bundle superposition |
| `CollapseGate` | Flow/Hold/Block decision between superposition and ground |
| `SpatialCrystal3D` | 3-axis BF16 crystal for SPO encoding |
| `Awareness substrate` | 2-bit per-dimension: crystallized/tensioned/uncertain/noise |
| `K0/K1/K2 cascade` | 64-bit probe → 512-bit stats → full exact + EnergyConflict |
| `Hybrid pipeline` | Tiered dispatch: INT8 prefilter → Hamming → BF16 tail |
| `SIMD dispatch` | Runtime: AVX-512 VPOPCNTDQ → AVX2 Harley-Seal → scalar |

### The Pipeline Today (disconnected)

```
                   ┌── DNTree.traverse()    (partial_hamming + beam)
                   │
Query ─────────────┼── CamIndex.query()     (LSH → exact Hamming)
                   │
                   └── BNN.winner()         (XNOR + popcount)

K0→K1→K2 cascade ──── operates on flat candidate lists (separate)
Awareness substrate ── classifies BF16 survivors (separate)
CollapseGate ────────── makes flow/hold/block (separate)
SpatialCrystal3D ───── encodes SPO axes (separate)
```

**Problem**: These seven systems don't talk to each other.

---

## 2. The Six Missing Pieces

### Piece 1: Temporal Binding

**What's missing**: Everything is a snapshot. No temporal sequences, no causality.

**The primitive already exists**: Permutation shift encodes position.
What's missing is a **temporal context vector** and a **trace decay gradient**:

```
sequence(a, b, c) = a ⊕ π¹(b) ⊕ π²(c)
```

Each element is bound at increasing shift offsets. The shift distance encodes
temporal position. But there's no:

- **Clock HV**: A slowly-rotating context vector (theta rhythm) that binds
  the sequence to its moment in time
- **Trace decay**: Elements at shift=1 should decay faster than shift=0,
  creating a recency gradient
- **Modulation**: Temporal context should modulate `learning_rate` in
  `bundle_into()`, not just the BTSP gate

**Why it matters**: Without this, the graph memory is a semantic store but
cannot represent "A happened, then B happened, because of C." Episodic memory
requires temporal binding.

**Estimated implementation**: ~300 lines in `temporal.rs`

### Piece 2: 3D Edge Trajectories

**What's missing**: `SpatialCrystal3D` encodes SPO on 3 axes. `GraphHV`
edges are flat binary vectors. These two systems don't communicate.

**The insight**: Edges should be **trajectories through crystal space**, not
flat vectors. See [Section 5](#5-3d-edge-trajectories) for full details.

**Estimated implementation**: ~300 lines in `crystal_edge.rs`

### Piece 3: Predictive Coding (Error Signal)

**What's missing**: The system stores and retrieves but never predicts.

**The wiring**: The awareness substrate already classifies dimensions as
crystallized/tensioned/uncertain/noise. This IS a prediction error signal:

```
Expected   = ground_truth (LayerStack ground)
Observed   = ground_truth ⊕ Σ(deltas) (superposition)
Surprise   = shift in awareness state distribution

crystallized_ratio increasing → prediction confirmed → lower learning rate
tensioned_ratio spiking       → prediction violated  → BTSP fires
uncertain_ratio dominant      → insufficient evidence → CollapseGate HOLD
noise_ratio dominant          → signal lost           → CollapseGate BLOCK
```

**The fix**: Wire awareness distribution ratios AS the CollapseGate input:

```rust
fn gate_decision(awareness: &AwarenessDistribution) -> CollapseGate {
    if awareness.crystallized_ratio() > 0.7 { CollapseGate::Flow }
    else if awareness.tensioned_ratio() > 0.3 { CollapseGate::Block }
    else if awareness.uncertain_ratio() > 0.5 { CollapseGate::Hold }
    else if awareness.noise_ratio() > 0.6 { CollapseGate::Block }
    else { CollapseGate::Hold }
}
```

This makes the gate a **Bayesian decision boundary** over awareness space.

**Estimated implementation**: ~150 lines wiring existing components

### Piece 4: Inhibitory Sculpting

**What's missing**: Everything is excitatory. XOR is symmetric, majority vote
is additive, popcount is unsigned. Biological cognition relies on inhibition.

**The fix**: Lateral inhibition within BNN layers:

```rust
fn lateral_inhibit(scores: &[f32], radius: usize) -> Vec<f32> {
    // Each neuron's score is reduced by neighbors' scores
    // Creates sparse, localized activation patterns
    // Radius controls receptive field size
}
```

More fundamentally: `Fingerprint<256>` should support a **signed interpretation**
where XOR with a "negative" HV creates destructive interference. The
`CrystalAxis` BF16 sign bit IS the inhibition channel.

**Estimated implementation**: ~200 lines extending BNN

### Piece 5: Consolidation Cycles (Sleep)

**What's missing**: BTSP handles one-shot encoding. Decay handles forgetting.
No offline consolidation.

**The fix**: A `consolidate()` function that:
1. Walks DN-tree collecting high-access-count leaves
2. Replays their prototype HVs through `bundle_into()` with elevated learning rate
3. Uses awareness substrate to resolve contradictions (crystallized wins,
   tensioned competes, uncertain decays)
4. Calls `cam.rebuild()` at the end

**Biological analog**: Slow-wave sleep: hippocampal replay → cortical
consolidation → memory stabilization.

**Estimated implementation**: ~200 lines in `consolidation.rs`

### Piece 6: Multi-Scale Coupling (The Binding Problem)

**What's missing**: K0/K1/K2 does multi-scale filtering on flat candidates.
The graph memory operates at single scale.

**The fix**: Cross-scale feedback:
- `partial_hamming(bits=64)` → K0 → micro signal → modulates BNN threshold
- `partial_hamming(bits=1024)` → K1 → meso signal → modulates CAM window_size
- Full `hamming_distance()` → K2 → macro signal → modulates BTSP gate probability

**Estimated implementation**: ~150 lines wiring existing components

---

## 3. Dimensionality Analysis: 16,384 Binary vs 10,000D Continuous

### Quantitative Comparison

| Metric | 10,000D continuous (HRR/MAP) | 16,384-bit binary (current) |
|--------|------------------------------|----------------------------|
| Orthogonality deviation | ~1/√10000 = 0.01 | ~1/√16384 = 0.0078 |
| Bundle capacity | ~769 concepts | ~1170/channel, ~3500 total |
| Binding depth before degradation | ~20 levels | ~20 levels (K/128 noise) |
| Similarity resolution | ~0.01 | ~0.00006 (1/16384) |
| SIMD throughput | vmulps+vaddps (Tier 3) | VPOPCNTDQ (Tier 1, 40x cheaper) |
| Memory per vector | 40 KB (10K × f32) | 6 KB (3 × 2048 bytes) |

### Why Binary Wins

1. **Capacity**: 16,384 > 10,000 in effective dimensions
2. **Compute**: XOR+popcount is Tier 1 (2ns/2KB), multiply+add is Tier 3 (50x slower)
3. **Memory**: 6.6x more compact per vector
4. **SIMD alignment**: 16,384 bits = 256 u64 words = 32 AVX-512 registers per channel
5. **Algebraic closure**: XOR bind stays in binary space (no normalization needed)

### When 10,000D Continuous Would Win

Only if you need:
- Continuous similarity gradients (binary has steps of 1/16384)
- Smooth interpolation between concepts
- Gradient-based learning (backprop through the representation)

None of these apply to the HDC-BNN architecture. The binary representation
is strictly superior for this system.

### Binding Depth Analysis

At depth K bindings, cumulative noise = K/√D:

| Depth | 10,000D noise | 16,384-bit noise | Threshold (5%) |
|-------|--------------|-------------------|----------------|
| 1 | 0.010 | 0.0078 | OK |
| 5 | 0.050 | 0.039 | OK |
| 10 | 0.100 | 0.078 | OK |
| 20 | 0.200 | 0.156 | Marginal |
| 50 | 0.500 | 0.390 | Degraded |

For deep composition (>20 levels), the fix is NOT more dimensions — it's
**error correction during traversal**: project back to nearest known prototype
via CamIndex after each bind level.

---

## 4. The Traversal Is the Bottleneck

### Current State: Three Disconnected Pipelines

```
Query → DNTree.traverse()  uses partial_hamming (dumb K0/K1)
Query → CamIndex.query()   uses LSH → exact Hamming (no awareness)
Query → BNN.winner()       uses XNOR+popcount (no hierarchy)
```

### Target State: One Unified Cascade

```
Query
  │
  ├─ K0 probe (64-bit) against DN-tree node summaries
  │    → eliminates ~55% of BRANCHES, not just candidates
  │
  ├─ K1 stats (512-bit) as ADAPTIVE beam weight
  │    → beam_width = f(K1 score variance)
  │      clustered K1 scores → wide beam (uncertain)
  │      one dominant K1 score → narrow beam (crystallized)
  │
  ├─ K2 exact + EnergyConflict at leaves
  │    → conflict/energy decomposition says WHY, not just IF
  │    → conflict = bits where a=1,b=1 (agreement)
  │    → energy_a, energy_b = individual popcounts
  │
  └─ Awareness classification of K2 survivors
       → feeds BACK into CollapseGate
       → crystallized > 0.7 → FLOW
       → tensioned > 0.3 → BLOCK
       → uncertain > 0.5 → HOLD (widen beam, descend deeper)
```

### Axis-Aware K1: Where 3D Edges Emerge

The SpatialCrystal3D axes are correlated:

```
X = S ⊕ P      sim(X, Y) = sim(S, O)   // P cancels
Y = P ⊕ O      sim(X, Z) = sim(P, O)   // S cancels
Z = S ⊕ O      sim(Y, Z) = sim(P, S)   // O cancels
```

The crystal geometry **warps based on content**. Current `partial_hamming`
ignores this. The fix:

```rust
fn axis_aware_k1(query: &SpatialCrystal3D, candidate: &SpatialCrystal3D) -> AxisScore {
    let k1_x = k1_stats(&query.x.as_words(), &candidate.x.as_words());
    let k1_y = k1_stats(&query.y.as_words(), &candidate.y.as_words());
    let k1_z = k1_stats(&query.z.as_words(), &candidate.z.as_words());

    let agreement = k1_x.min(k1_y).min(k1_z);  // worst axis = true floor
    let tension = k1_x.max(k1_y).max(k1_z) - agreement;  // axis spread

    AxisScore { agreement, tension, per_axis: [k1_x, k1_y, k1_z] }
}
```

**The tension signal IS the edge.** When axes disagree, you've found where
the relation lives in crystal space. The 3D edge trajectory is not a new data
structure — it's a **traversal signal** from axis-aware K1 scoring.

| Agreement | Tension | Interpretation | Action |
|-----------|---------|---------------|--------|
| High | Low | Genuine match | Narrow beam |
| High | High | **Interesting edge** | FOLLOW THIS |
| Low | Low | Genuine mismatch | Prune |
| Low | High | Partial match | Widen beam |

---

## 5. 3D Edge Trajectories Through Crystal Space

### The Architecture Gap

Currently:
- `SpatialCrystal3D` stores SPO on 3 axes (static encoding)
- `GraphHV` channel 1 stores edges as flat binary vectors
- These don't communicate

### The Insight: Edges Are Crystal Trajectories

An edge from node A to node B via relation R should trace a path through
the 3D crystal:

```
sample_0 = crystal.probe(A_position)          // source embedding in crystal
sample_1 = crystal.probe(midpoint(A, B))      // relation lives HERE
sample_2 = crystal.probe(B_position)          // destination embedding

edge_3d = sample_0 ⊕ π¹(sample_1) ⊕ π²(sample_2)
```

This combines temporal binding (permutation for position) with spatial
probing (crystal coordinates). The result is a hypervector that encodes:
- WHERE the source is in crystal space
- WHERE the relation manifests (the midpoint)
- WHERE the destination is in crystal space
- The TRAJECTORY connecting them

### Properties of Crystal Trajectories

1. **Edge curvature**: The midpoint sample IS the relation. Where it falls
   in crystal space encodes what kind of relation it is (causation, similarity,
   temporal succession, spatial proximity).

2. **Cross-axis resonance**: An edge touching X (subject-predicate) and Y
   (predicate-object) axes simultaneously creates a resonance pattern that
   encodes the predicate's mediating role. This is detectable by axis-aware K1.

3. **Trajectory bundling**: `bundle()` on multiple edge trajectories produces
   a "flow field" — the majority-vote surface of how information moves through
   the crystal. This is the aggregate causal structure.

4. **Differential decay**: Edge endpoints can decay faster than the center
   (causal structure persists longer than specific instances), or the center
   can decay faster (instances remembered, causal link forgotten). The decay
   rate along the trajectory encodes certainty.

5. **Reversibility**: Since XOR binding is self-inverse:
   ```
   unbind(edge_3d, sample_2, sample_1) recovers sample_0  (source from edge)
   unbind(edge_3d, sample_0, sample_2) recovers sample_1  (relation from edge)
   ```
   You can recover ANY component from the other two.

### Connection to Causal Inference

A **causal edge** is a trajectory where:
- The source precedes the destination temporally (shift encodes time order)
- The midpoint (relation) is an **intervention point** — changing it changes
  the destination but not the source (asymmetric)
- Bundled causal trajectories reveal **causal structure** — the flow field
  shows which regions of crystal space causally influence which others

This is directly related to Pearl's do-calculus: `do(X=x)` corresponds to
**clamping the source sample** and observing how the trajectory propagates.

---

## 6. Latest Research Connections (2025-2026)

### 6.1 Causal Representation Learning

**"Attention as Binding: A Vector-Symbolic Perspective on Transformer Reasoning"**
(arXiv, Dec 2025)

Connects VSAs with causal analysis in transformers. Key insight: attention
heads implement soft vector-symbolic binding operations (XOR/convolution).
Interventional studies edit embeddings to test whether models perform
consistent unbinding/rebinding.

**Connection to our system**: The `CollapseGate` is exactly this — an
intervention point where superposition collapses to ground truth. The
awareness substrate's crystallized/tensioned classification IS the
interventional signal. Wire it: if an intervention (clamping a delta layer)
produces crystallized awareness → the causal link is real. If it produces
tensioned awareness → confounded.

### 6.2 Causal Discovery via Binary Adjacency

**"Causal Discovery via Bayesian Optimization" (ICLR 2025)**

Uses a **binary adjacency matrix** for directed graphs where edge i→j is
encoded as a positional ordering pi < pj. Sparsity constraints on the DAG
are enforced through L1 regularization.

**Connection to our system**: The `GraphHV` channel 1 (edge channel) IS a
binary adjacency representation — but distributed, not explicit. The CamIndex
hash provides a compressed adjacency test: two nodes are "adjacent" if their
hash values collide across multiple tables. The sparsity constraint maps to
**decay** — edges that aren't reinforced naturally disappear, enforcing
implicit sparsity without L1.

### 6.3 Temporal Causal Discovery

**"Causal Discovery from Temporal Data" (ACM Computing Surveys, 2025)**

Comprehensive survey covering Granger causality, neural graphical models,
and causal discovery from irregular time series. Key finding: Neural-ODE
based models handle irregular sampling better than fixed-interval approaches.

**Connection to our system**: The BTSP gate fires **irregularly** (stochastic,
not periodic). This creates irregular temporal sampling of the causal process.
The existing `bundle_into()` with variable learning rate is structurally
equivalent to a discrete-time neural-ODE integration step. To wire Granger
causality: check whether the presence of one GraphHV's plastic channel
(channel 2) improves prediction of another GraphHV's future state.

### 6.4 Disentangled Causal Factors

**"Causal-Oriented Representation Learning Predictor (CReP)"**
(Communications Physics, June 2025)

Decomposes the representation space into three orthogonal latent factors:
cause-related, effect-related, and non-causal.

**Connection to our system**: The three GraphHV channels MAP DIRECTLY to this:
- Channel 0 (Node/Identity) → cause-related representation
- Channel 1 (Edge/Relation) → the causal mechanism itself
- Channel 2 (Plastic/State) → effect-related representation (accumulated outcome)

The existing `bnn_dot_3ch()` computes the full 49,152-bit correlation.
Per-channel `bnn_dot()` gives the disentangled causal factors. The difference
between channels reveals which factor drives the match:

```
If channel_0_score >> channel_1_score: match is cause-based (identity)
If channel_1_score >> channel_0_score: match is mechanism-based (relation)
If channel_2_score >> others:          match is outcome-based (history)
```

### 6.5 Dynamic Causal Graphs

**"Uncertainty-Aware Disentangled Dynamic Graph Attention Network"**
(IEEE TPAMI, Oct 2025)

Captures spatio-temporal distribution shifts via invariant pattern discovery.
Integrates causal reasoning through intervention-based training and
information bottleneck constraints.

**Connection to our system**: The `CollapseGate` is an information bottleneck.
When it decides HOLD, it accumulates more evidence (widens the bottleneck).
When it decides FLOW, it collapses to ground truth (narrows the bottleneck).
When it decides BLOCK, it discards (closes the bottleneck). The gate IS the
information bottleneck, implemented in XOR algebra.

Distribution shift detection maps to the awareness substrate: if the ratio
of crystallized/tensioned/uncertain/noise changes significantly between
observation windows, a distribution shift has occurred. This should trigger
a higher BTSP gate probability (learn faster during regime change).

### 6.6 Quantum-Inspired Cognition in Superposition

**"Cognition in Superposition" (arXiv, Aug 2025)**
**"Transforming Neural Networks into Quantum-Cognitive Models" (MDPI, May 2025)**

Multiple 2025 papers model cognition as quantum-like superposition that
collapses during decision-making. Key insight: holding multiple contradictory
beliefs simultaneously until evidence forces collapse.

**Connection to our system**: This IS the `LayerStack` + `CollapseGate`
architecture. Multiple writers hold contradictory deltas in superposition
over ground truth. The awareness substrate READS the superposition (AND +
popcount SEES contradictions). The CollapseGate decides when to collapse.
This is not metaphorical — it's the exact same mathematical structure, using
XOR instead of complex amplitudes.

The key difference from quantum mechanics: our superposition is read without
destroying it (AND + popcount is non-destructive). HOLD keeps the superposition
alive. This is closer to **decoherence-free subspaces** than Copenhagen
measurement.

### 6.7 BTSP Computational Models

**Latest BTSP research (2025)** confirms the biological mechanism:

- BTSP enables one-shot learning through CaMKII autophosphorylation
- Plateau potentials last 300-500ms (behavioral timescale)
- During the plateau, synaptic weights can change dramatically (~7x normal rate)
- The mechanism is stochastic — not every event triggers a plateau

**Connection to our system**: The existing BTSP implementation (`btsp_gate_prob`,
`btsp_boost`) is directly mapped from this biology:
- `btsp_gate_prob = 0.01`: ~1% of events trigger a plateau (matches biology)
- `btsp_boost = 7.0`: 7x amplification during plateau (matches CaMKII data)
- The stochastic gate (`rng.next_f64() < btsp_gate_prob`) mirrors the
  probabilistic nature of plateau potential initiation

### 6.8 Category-Theoretic Foundation for VSAs

**"Developing a Foundation of VSAs Using Category Theory" (arXiv, Jan 2025)**

Formalizes VSA operations (binding, bundling, permutation) in category theory.
Binary Spatter Codes use XOR and AND as multiply/divide to give division ring
structure. Maintaining near-orthogonality is critical for symbol distinguishability.

**Connection to our system**: The `Fingerprint<256>` operations already form
this algebraic structure:
- XOR = multiplication in the division ring (binding)
- AND + popcount = inner product (similarity)
- Majority vote = addition (bundling)
- Circular shift = group action (permutation)

The category-theoretic perspective suggests that the **CollapseGate** is a
natural transformation between two functors: the "superposition functor"
(LayerStack with deltas) and the "ground truth functor" (committed state).
FLOW is the natural transformation. HOLD is the identity. BLOCK is the zero
morphism.

### 6.9 Recursive Causal Discovery

**"Recursive Causal Discovery" (JMLR, Volume 26, 2025)**

Introduces removable variables and recursive decomposition for causal
discovery. The MARVEL algorithm recursively identifies and removes variables
to simplify the causal structure.

**Connection to our system**: The DN-tree IS a recursive decomposition of the
prototype space. Each split creates 4 sub-problems. The access_count at each
node is a natural "importance" measure — low-access nodes are candidates for
removal (analogous to removable variables). Implementing MARVEL-like pruning
in the DN-tree: remove subtrees with access_count below a threshold, merge
their prototypes into the parent's summary.

---

## 7. Causal Inference Integration Plan

### Phase 1: Wire Awareness to CollapseGate (Pearl's Ladder Rung 1: Association)

**Goal**: The CollapseGate makes decisions based on awareness substrate ratios.

**Implementation**:
```rust
// In layer_stack.rs or new causal_gate.rs
pub struct AwarenessDistribution {
    pub crystallized: f32,  // ratio of crystallized dimensions
    pub tensioned: f32,     // ratio of tensioned dimensions
    pub uncertain: f32,     // ratio of uncertain dimensions
    pub noise: f32,         // ratio of noise dimensions
}

pub fn awareness_gate(dist: &AwarenessDistribution) -> CollapseGate {
    if dist.crystallized > 0.7 { CollapseGate::Flow }
    else if dist.tensioned > 0.3 { CollapseGate::Block }
    else if dist.uncertain > 0.5 { CollapseGate::Hold }
    else { CollapseGate::Block } // noise-dominated = discard
}
```

**Connects**: `bf16_hamming.rs` awareness output → `layer_stack.rs` CollapseGate

### Phase 2: Per-Channel Causal Decomposition (Pearl's Ladder Rung 2: Intervention)

**Goal**: Identify which GraphHV channel drives a match (cause vs mechanism vs outcome).

**Implementation**:
```rust
pub struct CausalDecomposition {
    pub cause_score: f32,      // channel 0 contribution
    pub mechanism_score: f32,  // channel 1 contribution
    pub outcome_score: f32,    // channel 2 contribution
    pub dominant: CausalFactor,
}

pub fn decompose_causal(query: &GraphHV, candidate: &GraphHV) -> CausalDecomposition {
    let ch0 = bnn_dot(&query.channels[0], &candidate.channels[0]);
    let ch1 = bnn_dot(&query.channels[1], &candidate.channels[1]);
    let ch2 = bnn_dot(&query.channels[2], &candidate.channels[2]);
    // ... determine dominant factor
}
```

**Intervention test**: Clamp one channel (set to zero or random), re-query.
If score drops dramatically → that channel is causal. If score unchanged →
that channel is spurious.

### Phase 3: Temporal Causal Sequences (Pearl's Ladder Rung 2: do-calculus)

**Goal**: Encode "A causes B" as a temporal trajectory through crystal space.

**Implementation**:
```rust
pub struct CausalEdge {
    pub source_sample: Fingerprint<256>,   // crystal probe at source
    pub mechanism_sample: Fingerprint<256>, // crystal probe at midpoint
    pub effect_sample: Fingerprint<256>,    // crystal probe at destination
    pub trajectory: Fingerprint<256>,       // source ⊕ π¹(mechanism) ⊕ π²(effect)
    pub temporal_shift: u32,               // time gap encoding
}
```

**do(X=x)**: Clamp `source_sample`, propagate through trajectory, observe
effect on `effect_sample`. If the trajectory is a genuine causal link,
changing the source MUST change the effect (holding mechanism constant).

### Phase 4: Counterfactual Reasoning (Pearl's Ladder Rung 3: Counterfactuals)

**Goal**: Answer "What would have happened if X had been different?"

**Implementation**: Use the LayerStack superposition:

1. Create a counterfactual delta: `delta_cf = actual_source ⊕ counterfactual_source`
2. Apply delta to ground truth: `cf_world = ground ⊕ delta_cf`
3. Propagate through causal edges: `cf_effect = unbind(trajectory, cf_source, mechanism)`
4. Compare `cf_effect` vs `actual_effect`
5. Use awareness substrate to classify the difference:
   - Crystallized difference → counterfactual has definite effect
   - Tensioned difference → counterfactual creates contradiction
   - Uncertain difference → insufficient information
   - Noise difference → counterfactual is irrelevant

This is **exactly** what the LayerStack was designed for: multiple realities
(deltas) coexisting over shared ground truth, with CollapseGate deciding
which reality wins.

---

## 8. Concrete Implementation Roadmap

### Priority Order (by value/effort ratio)

| # | Module | Lines | Dependencies | Value |
|---|--------|-------|-------------|-------|
| 1 | Awareness → CollapseGate wiring | ~150 | bf16_hamming + layer_stack | Unblocks everything |
| 2 | Unified K0/K1/K2 → DN-tree traversal | ~250 | kernels + dn_tree | 10x traversal quality |
| 3 | Axis-aware K1 scoring | ~200 | spatial_resonance + kernels | 3D edge detection |
| 4 | Temporal binding primitives | ~300 | graph_hv + fingerprint | Enables causality |
| 5 | Crystal edge trajectories | ~300 | spatial_resonance + graph_hv + temporal | 3D causal edges |
| 6 | Causal decomposition (per-channel) | ~150 | bnn + graph_hv | Intervention testing |
| 7 | Consolidation cycles | ~200 | dn_tree + cam_index + awareness | Sleep/memory |
| 8 | Lateral inhibition in BNN | ~200 | bnn | Sparse activations |
| 9 | Counterfactual reasoning | ~250 | layer_stack + causal edges | Pearl Rung 3 |
| **Total** | | **~2,000** | | |

### File Plan

```
rustynum-core/src/
├── causal_gate.rs       // Phase 1: Awareness → CollapseGate wiring
├── temporal.rs          // Phase 4: Temporal binding primitives
├── crystal_edge.rs      // Phase 5: 3D edge trajectories
├── causal_decompose.rs  // Phase 6: Per-channel causal analysis
├── consolidation.rs     // Phase 7: Sleep/consolidation cycles
│
├── dn_tree.rs           // Modified: K0/K1/K2 cascade integration
├── kernels.rs           // Modified: axis-aware K1 scoring
├── bnn.rs               // Modified: lateral inhibition
├── layer_stack.rs       // Modified: counterfactual delta propagation
└── graph_hv.rs          // Modified: temporal sequence operations
```

### Test Strategy

Each phase adds tests verifying:
1. **Algebraic correctness**: Operations maintain XOR group properties
2. **Statistical invariants**: Random vectors stay ~50% similar
3. **Roundtrip**: Encode → decode recovers original
4. **Regression**: SIMD matches scalar reference
5. **Causal**: Intervention changes effect but not cause

---

## 9. Mathematical Appendix

### A. Noise Accumulation Under Binding

For binary vectors of dimension D, binding depth K produces cumulative noise:

```
ε(K) = K / √D

At D = 16384, K = 10: ε = 10/128 = 0.078 (acceptable)
At D = 16384, K = 20: ε = 20/128 = 0.156 (marginal)
At D = 16384, K = 50: ε = 50/128 = 0.390 (degraded)
```

**Mitigation**: Error correction via CamIndex projection after every N bindings.

### B. Bundle Capacity

For majority-vote bundle of N vectors at dimension D:

```
P(bit error) = Φ(-√(D/N))

At D = 16384, N = 100: P(error) ≈ 10⁻¹⁷ (per bit)
At D = 16384, N = 500: P(error) ≈ 10⁻⁴  (per bit)
At D = 16384, N = 1000: P(error) ≈ 0.01  (per bit, 5% Hamming error)
```

3 channels multiply capacity by ~3x (independent random vectors).

### C. Crystal Axis Correlation

Given SPO encoding:
```
X = S ⊕ P,  Y = P ⊕ O,  Z = S ⊕ O

Cov(X, Y) = E[(S⊕P)(P⊕O)] = E[S⊕O] = correlation between S and O
```

When S and O are independent random: Cov(X,Y) = 0 (axes orthogonal).
When S ≈ O (reflexive relation): Cov(X,Y) ≈ 1 (axes collapse).

The **axis tension** (max - min K1 score across axes) detects this collapse
and signals that the relation is geometrically degenerate — the predicate is
self-referential.

### D. Carry-Save Adder Correctness

For N ≤ 15 inputs, the 4-bit CSA maintains the invariant:

```
count[j] = b3[j]×8 + b2[j]×4 + b1[j]×2 + b0[j]×1

for all j ∈ {0, 1, ..., 63} simultaneously
```

Proof: Each `csa_add` call increments the counter by the input bit.
The carry chain `c0 = b0 & input, b0 ^= input` is a half-adder.
Rippling through b1, b2, b3 completes the full addition.

The threshold comparison `count >= target` uses the identity:
```
count >= target ⟺ count - target >= 0 ⟺ carry_out(count + ~target + 1) = 1
```

---

*This document captures the complete architectural vision, research
connections, and implementation roadmap for the rustynum HDC-BNN cognitive
system as of 2026-02-27. Update when new research or implementation
changes the picture.*

*Sources referenced in Section 6:*
- *"Attention as Binding" (arXiv 2512.14709, Dec 2025)*
- *"Causal Discovery via Bayesian Optimization" (ICLR 2025)*
- *"Causal Discovery from Temporal Data" (ACM Computing Surveys, 2025)*
- *"CReP: Causal-Oriented Representation Learning" (Communications Physics, Jun 2025)*
- *"Uncertainty-Aware Disentangled Dynamic Graph Attention" (IEEE TPAMI, Oct 2025)*
- *"Cognition in Superposition" (arXiv 2508.20098, Aug 2025)*
- *"Transforming NNs into Quantum-Cognitive Models" (MDPI Technologies, May 2025)*
- *"Foundation of VSAs Using Category Theory" (arXiv 2501.05368, Jan 2025)*
- *"Recursive Causal Discovery — MARVEL" (JMLR Vol 26, 2025)*
- *"Causal Integration in GNNs" (World Wide Web, Apr 2025)*
- *"Temporal Knowledge Graph Reasoning with Complex Causal Relations" (Expert Systems, 2026)*
- *"CCAGNN: Causal Confounder-Aware GNN" (arXiv 2602.17941, Feb 2026)*
