# Two Punktlandung Sessions: deepmsm + jaxpi → ladybug-rs

## Status: What's Already Landed

### DeepNSM: ✅ HARVESTED (165KB, 6 files)
```
src/grammar/nsm.rs              (12KB)  65 primes, NSMField, from_text(), to_fingerprint()
src/grammar/causality.rs        (11KB)  CausalityFlow, agent/action/patient parsing
src/grammar/qualia.rs           (21KB)  Qualia triangle grammar
src/grammar/triangle.rs         (9KB)   Triangle geometry
src/spo/nsm_substrate.rs        (27KB)  NsmCodebook, MetaConcept, MetacognitiveSubstrate
src/spo/deepnsm_integration.rs  (26KB)  ExplicationParser, DeepNsmCodebook, legality_score()
```
**No further integration needed.** The pipeline from paper → primes → role-bind → fingerprint is implemented.

### Existing Infrastructure Relevant to deepmsm + jaxpi
```
src/cognitive/collapse_gate.rs  (18KB)  GateState(FLOW/HOLD/BLOCK), evaluate_gate(), sd-based gating
src/spo/causal_trajectory.rs    (30KB)  HydratedEdge, CausalArrow, inference modes
src/learning/causal_ops.rs      (22KB)  CausalEngine, GraphEdge, CausalEdgeType
src/learning/causal_bridge.rs   (18KB)  Bridge between learning and causal subsystems
src/search/causal.rs            (37KB)  Causal search engine
src/mul/gate.rs                 (3.6KB) Simpler gate (wraps collapse_gate)
src/mul/homeostasis.rs          (5KB)   Stable equilibrium detection
src/mul/hysteresis.rs           (3.3KB) Path-dependent state tracking
src/core/scent.rs               (18KB)  Scent field navigation
src/graph/spo/scent.rs          (7KB)   SPO-specific scent masks
src/query/scent_scan.rs         (32KB)  Scent-based query
```

---

## Session A: deepmsm Punktlandung

**Repo source:** `AdaWorldAPI/deepmsm` (Python, ~200KB meaningful code)
**Target:** NOT a transcode. A **math extraction** into 4 new Rust modules.

### What deepmsm has that ladybug DOESN'T

| deepmsm concept | What it computes | ladybug equivalent | Gap |
|---|---|---|---|
| `U_layer` (reweighting) | Learned stationary distribution π | `collapse_gate.rs` has sd-threshold gating | Gap: no LEARNED weights, only static σ-threshold |
| `S_layer` (transition matrix) | Symmetric K where K[i,j] = P(j|i) | `causal_trajectory.rs` has directed edges | Gap: no transition MATRIX, only individual edges |
| `Coarse_grain` (softmax N→M) | Hierarchical state compression | CLAM (geometric, not learned) | Gap: geometric split, not data-driven softmax |
| `vampe_loss_rev()` | Quality score for decomposition | AccumulatedHarvest (heuristic) | Gap: no variational score, no eigenvalue analysis |
| `get_transition_matrix()` | K from observation pairs (x_0, x_t) | Nothing | Gap: **completely missing** |
| `timescales()` | -τ/ln(λ) from eigenvalues of K | Nothing | Gap: **completely missing** |
| CK test K(2τ)=K(τ)² | Markov consistency check | Nothing | Gap: **completely missing** |
| Attention `Mask` | Per-position importance weights | `scent.rs` (static masks) | Gap: static, not learned from transitions |

### The 4 New Modules

```
src/nars/transition_matrix.rs  (~300 lines)
  - TransitionMatrix: σ-band × σ-band counts
  - observe_transition(from_band, to_band): increment count
  - normalize() → row-stochastic K
  - eigendecomposition() → eigenvalues + eigenvectors (small matrix: 3-6 bands)
  - implied_timescales(tau) → Vec<f32>
  - ck_test(K_tau, K_2tau) → consistency score per entry
  - Wire: called from spo_harvest on every band-crossing event

src/nars/vampe.rs  (~200 lines)
  - vampe_score(K) → f32 (Σ λ_i²)
  - spectral_gap(eigenvalues) → suggested_num_bands
  - calibrate_thresholds(vampe, current_thresholds) → new_thresholds
  - Wire: called periodically from hdr_cascade during compaction

src/learning/attention_mask.rs  (~250 lines)
  - NibbleAttention: HashMap<(BandId, BandId), [f32; 4096]>
  - observe_transition(from, to, from_band, to_band): XOR diff → EMA update
  - masked_distance(a, b, transition) → weighted Hamming
  - top_k_positions(transition, k) → most informative nibble positions
  - Wire: called from scent.rs instead of static masks

src/nars/causal_weighting.rs  (~150 lines)  [SHARED with jaxpi session]
  - CausalWeighting: num_stages, tolerance, momentum
  - update(residuals): lower-triangular exponential decay
  - weight_revision(stage, truth) → weighted TruthValue
  - gate_state(stage) → GateState
  - Wire: wraps existing collapse_gate.rs evaluate_gate()
```

### Session Plan

```
1. Clone deepmsm to filesystem, read deepmsm.py + helper.py fully         (10 min)
2. Extract the 4 mathematical kernels (exact formulas, not Python code)     (20 min)
3. Write transition_matrix.rs with eigendecomposition                       (45 min)
   - Use nalgebra for small-matrix eigensolve (3-6 × 3-6)
   - Test: synthetic Markov chain with known eigenvalues
4. Write vampe.rs with spectral gap analysis                                (30 min)
   - Test: synthetic clusters, verify band count recovery
5. Write attention_mask.rs with EMA-learned nibble weights                  (30 min)
   - Test: synthetic transition data, verify mask learns changing positions
6. Write causal_weighting.rs with lower-triangular decay                    (20 min)
   - Test: causal chain A→B→C, verify late evidence gated
7. Wire transition_matrix into spo_harvest.rs band-crossing events          (20 min)
8. Wire vampe into hdr_cascade.rs compaction cycle                          (15 min)
9. Wire attention_mask into scent.rs                                        (15 min)
10. Wire causal_weighting into collapse_gate.rs                             (15 min)
11. Integration test: full pipeline synthetic → encode → factorize →
    transition matrix → VAMPE calibrate → attention learn → causal gate     (30 min)
12. Push PR                                                                  (10 min)
```

**Total: ~4 hours. ~900 lines new code + ~200 lines wiring.**

### Dependencies to Add to Cargo.toml
```toml
[dependencies]
nalgebra = "0.33"  # Small-matrix eigendecomposition (only for 3-6 × 3-6 transition matrices)
```
No other new deps. Everything else uses existing ladybug primitives.

---

## Session B: jaxpi Punktlandung

**Repo source:** `AdaWorldAPI/jaxpi` (Python/JAX, ~22KB meaningful code in jaxpi/)
**Target:** NOT a transcode. An **algorithm extraction** into 3 new Rust modules.

### What jaxpi has that ladybug DOESN'T

| jaxpi concept | What it computes | ladybug equivalent | Gap |
|---|---|---|---|
| Causal temporal weighting | Lower-triangular exponential decay on time chunks | CollapseGate (binary FLOW/BLOCK) | Gap: no GRADUATED weighting, no cumulative residual tracking |
| NTK gradient balancing | Equalize gradient norms across loss terms | Nothing | Gap: **completely missing** — 8 Faktorzerlegung terms unbalanced |
| `grad_norm` weighting | w_i = mean_norm / grad_norm_i | Nothing | Gap: **completely missing** |
| Weight momentum (EMA) | Running average of weights (0.9) | NARS has temporal decay but not on WEIGHTS | Gap: partial — need to apply EMA to term weights |
| Fourier feature embeddings | cos/sin projection for multi-scale | Nothing | Gap: interesting but Phase 2, not critical path |
| Random weight factorization | W = diag(g)·V decomposition | SPO planes ARE a factorization | Gap: conceptual — planes exist but per-bit importance doesn't |
| Period embeddings | Trainable periodic boundary | Cyclic detection doesn't exist | Gap: nice-to-have, not critical |

### The 3 New Modules

```
src/nars/causal_weighting.rs  (~150 lines)  [SHARED with deepmsm session]
  - (same as deepmsm session — build once, use in both)

src/spo/term_balancing.rs  (~200 lines)
  - TermBalancer: running_norms[8], weights[8], momentum
  - update(terms: &FactorialTerms): EMA of magnitudes → equalize weights
  - balanced_terms(terms) → FactorialTerms with weights applied
  - diagnostic(): which terms are currently dominant, which suppressed
  - Wire: called in spo_harvest.rs after raw 8-term computation, before NARS

src/spo/residual_monitor.rs  (~200 lines)
  - ResidualMonitor: per-stage running residual
  - observe(stage, predicted_truth, observed_truth): r = |predicted - observed|
  - cumulative_residual(up_to_stage) → f32
  - feeds causal_weighting.update()
  - convergence_rate() → f32 (how fast residual is decreasing)
  - Wire: called from nars/inference.rs after every revision
```

### Session Plan

```
1. Clone jaxpi, read models.py + archs.py fully                            (10 min)
2. Extract the 3 algorithms (exact math, not JAX/Flax code)                (15 min)
3. causal_weighting.rs — IF not already built in deepmsm session            (20 min)
   - (if deepmsm session was first, this is already done)
4. Write term_balancing.rs                                                  (30 min)
   - Test: synthetic 8-term vector with 100× imbalance
   - Verify balancing makes all terms detectable
5. Write residual_monitor.rs                                                (30 min)
   - Test: synthetic convergence curve
   - Verify cumulative residual feeds causal_weighting correctly
6. Wire term_balancing into spo_harvest.rs                                  (15 min)
7. Wire residual_monitor into nars/inference.rs                             (15 min)
8. Wire causal_weighting into collapse_gate.rs (if not already done)        (10 min)
9. Integration test: stream evidence → factorize → balance terms →
    monitor residual → causal weight → gate → converge                      (30 min)
10. Benchmark: with/without balancing, count false discovery rate            (20 min)
11. Push PR                                                                  (10 min)
```

**Total: ~3 hours. ~550 lines new code + ~100 lines wiring.**
**(~2 hours if causal_weighting.rs already built in deepmsm session)**

### Dependencies
None new. Pure arithmetic on existing types.

---

## Session Order

**deepmsm FIRST** because:
1. It creates `causal_weighting.rs` which jaxpi session reuses
2. It creates `transition_matrix.rs` which provides the eigenvalue infrastructure jaxpi's residual monitor uses
3. VAMPE calibration provides the σ-band boundaries that term balancing operates within

**jaxpi SECOND** because:
1. It consumes `causal_weighting.rs` (already exists)
2. Term balancing needs to know the σ-bands (from VAMPE)
3. Residual monitor needs transition matrix (from deepmsm session)

```
Session A (deepmsm): 4 hours → 4 new modules, 900 lines
   ↓ causal_weighting.rs + transition_matrix.rs exist
Session B (jaxpi):   2-3 hours → 2 new modules, 400 lines
   ↓ term_balancing.rs + residual_monitor.rs exist
TOTAL: 6-7 hours → 6 new modules, ~1300 lines, zero new concepts needed
```

---

## What This Does NOT Include

- **Fourier features for nibble frequency analysis** — interesting but Phase 2. Needs real corpus data to calibrate frequencies. Park it.
- **Learned coarse-graining replacing CLAM** — requires training infrastructure. The attention mask is the lightweight version. Park it.
- **Period embeddings for cyclic detection** — nice-to-have. Not critical path. Park it.
- **Kolmogorov superposition theorem connection** — pure theory work. Needs the benchmark suite (Phase 6) to ground it empirically. Park it for the paper.
- **Weight factorization (per-bit importance)** — conceptually the attention mask does this. Explicit W=diag(g)·V factorization is a refinement. Park it.

## What This DOES Include

After both sessions, ladybug-rs gains:
1. **Principled σ-band calibration** (VAMPE replaces hand-tuned thresholds)
2. **Markov consistency validation** (CK test catches broken assumptions)
3. **Learned attention masks** (scent becomes data-driven, not static)
4. **Temporal causal ordering** (CollapseGate backed by PDE theory)
5. **Balanced Faktorzerlegung** (all 8 terms get equal attention)
6. **Convergence monitoring** (know when the graph is "done")

These six capabilities are EXACTLY what the four-AI validation demanded:
- ChatGPT: "correlated evidence" → causal weighting
- ChatGPT: "false discovery" → term balancing
- All four: "orthogonality faithfulness" → VAMPE validates the decomposition is real
- Gemini: "temporal precedence" → causal ordering
- Grok: "publish exact formula" → residual monitor gives the formula a quality metric
