# σ-Significance Scoring & Per-Word Histogram: Technical Mapping

> **Version**: 1.0 — 2026-03-01
> **Commit**: `b8eca06` (branch `claude/implement-neural-network-research-owTKn`)
> **Crates modified**: `rustynum-core` (kernels.rs, hybrid.rs, lib.rs), `rustynum-bnn` (causal_trajectory.rs, cross_plane.rs)

---

## 1. Technique Inventory

### 1.1 σ-Significance Scoring (kernels.rs:398–553)

**Problem solved**: The existing 3-tier HDR scoring (`hot/mid/cold`) uses ad-hoc
fraction-of-D thresholds (10%/30%/49% of total bits). These thresholds have no
statistical grounding — they don't tell you whether a match is statistically
significant or just noise.

**Solution**: Map the raw Hamming distance to standard statistical significance
using the known noise distribution for balanced binary vectors:

```
For D-bit balanced vectors:
  Expected noise distance: μ = D/2
  Standard deviation:      σ = √(D/4)
  z-score:                 z = (μ - observed) / σ
```

For SKU-16K (D = 16384): μ = 8192, σ = 64.

| σ-Level | Distance Threshold (SKU-16K) | p-value | Significance |
|---|---|---|---|
| < 1.5σ | > 8096 | > 0.067 | Noise |
| 1.5–2.0σ | 8064–8096 | 0.023–0.067 | Hint |
| 2.0–2.5σ | 8032–8064 | 0.006–0.023 | Evidence (95% CI) |
| 2.5–3.0σ | 8000–8032 | 0.001–0.006 | Strong (99% CI) |
| > 3.0σ | < 8000 | < 0.001 | Discovery (99.7%+) |

**Types**:

| Type | File:Line | Size | Purpose |
|---|---|---|---|
| `SignificanceLevel` | kernels.rs:398 | 1 byte (enum) | 5-tier: Noise/Hint/Evidence/Strong/Discovery |
| `SigmaScore` | kernels.rs:427 | 12 bytes | sigma (f32) + level (enum) + p_value (f32) |
| `SigmaGate` | kernels.rs:442 | 28 bytes | Precomputed u32 thresholds per tier |

**Functions**:

| Function | File:Line | Input | Output | Cost |
|---|---|---|---|---|
| `SigmaGate::new(total_bits)` | kernels.rs:461 | usize | SigmaGate | O(1), once at init |
| `SigmaGate::sku_16k()` | kernels.rs:485 | — | SigmaGate | O(1) |
| `SigmaGate::sku_64k()` | kernels.rs:491 | — | SigmaGate | O(1) |
| `score_sigma(ec, gate)` | kernels.rs:504 | EnergyConflict + SigmaGate | SigmaScore | 4 u32 compares + 1 f32 div |

**Critical design choice**: The tier classification uses **integer comparisons only**
(precomputed in `SigmaGate`). The f32 z-score and p-value are informational — the
hot path is branchless integer logic.

```rust
// Hot path: integer-only tier decision (no floats)
let level = if conflict < gate.discovery {      // < 8000
    SignificanceLevel::Discovery
} else if conflict < gate.strong {              // < 8032
    SignificanceLevel::Strong
} else if conflict < gate.evidence {            // < 8064
    SignificanceLevel::Evidence
} else if conflict < gate.hint {                // < 8096
    SignificanceLevel::Hint
} else {
    SignificanceLevel::Noise
};
```

### 1.2 Per-Word Popcount Histogram (kernels.rs:555–701)

**Problem solved**: `k2_exact()` computes per-word `count_ones()` for each of the
256 u64 words (SKU-16K) but immediately sums them into a single `conflict: u32`.
The positional information is discarded.

**Solution**: `k2_exact_histogram()` — identical computation, but stores each
per-word popcount before summing.

**Types**:

| Type | File:Line | Size | Purpose |
|---|---|---|---|
| `K2Histogram` | kernels.rs:565 | 24 B + Vec(512 B for SKU-16K) | EnergyConflict + per-word Vec<u16> |

**Functions**:

| Function | File:Line | Input | Output | Cost |
|---|---|---|---|---|
| `k2_exact_histogram()` | kernels.rs:629 | &[u64], &[u64], n_words | K2Histogram | Same as k2_exact + 256 u16 stores |
| `K2Histogram::max_word_conflict()` | kernels.rs:576 | &self | u16 | O(n_words) |
| `K2Histogram::hottest_word()` | kernels.rs:582 | &self | usize | O(n_words) |
| `K2Histogram::hot_word_count(threshold)` | kernels.rs:593 | &self, u16 | usize | O(n_words) |
| `K2Histogram::variance()` | kernels.rs:604 | &self | f32 | O(n_words) |

**Key insight**: `variance()` distinguishes localized from distributed disagreement:
- **High variance** = a few words disagree strongly → structural difference (e.g., one
  semantic field differs, rest agrees)
- **Low variance** = all words disagree equally → noise-like difference (random
  perturbation)

This is the positional information that embeddings (e.g., Jina 1024-D) encode:
early dimensions capture broad semantics, later dimensions capture fine-grained
distinctions. A per-word histogram reveals WHERE the vectors disagree.

### 1.3 Stripe Histogram & Shift Detection (causal_trajectory.rs:1123–1370)

**Problem solved**: The causal trajectory records per-iteration dynamics, but doesn't
track how the POPULATION of candidates shifts across σ-bands over time.

**Solution**: `StripeHistogram` tracks candidate counts in 6 half-σ bands per plane.
`ShiftDetector` compares consecutive time windows to detect distributional migration.

**Types**:

| Type | File:Line | Size | Purpose |
|---|---|---|---|
| `StripeHistogram` | causal_trajectory.rs:1144 | 24 bytes | 6 × u32 counters for 0.5σ bands |
| `ShiftDirection` | causal_trajectory.rs:1224 | 1 byte (enum) | TowardFoveal/TowardNoise/Bimodal/Stable |
| `ShiftSignal` | causal_trajectory.rs:1238 | ~32 bytes | Direction + per-plane + magnitude |
| `ShiftDetector` | causal_trajectory.rs:1260 | ~168 bytes | Current + previous histograms + window count |

**Functions**:

| Function | File:Line | Input | Output | Cost |
|---|---|---|---|---|
| `StripeHistogram::record(sigma)` | causal_trajectory.rs:1172 | f32 | &mut self | O(1), 5 compares |
| `StripeHistogram::center_of_mass()` | causal_trajectory.rs:1197 | &self | f32 | O(1), 6 multiply-adds |
| `ShiftDetector::record(plane, sigma)` | causal_trajectory.rs:1284 | usize, f32 | &mut self | O(1) |
| `ShiftDetector::advance_window()` | causal_trajectory.rs:1291 | &mut self | — | O(1), memcpy |
| `ShiftDetector::detect_shift()` | causal_trajectory.rs:1299 | &self | Option<ShiftSignal> | O(1) |
| `ShiftDetector::gate_bias()` | causal_trajectory.rs:1348 | &self | Option<CollapseGate> | O(1) |

**The stripe bands**:

| Stripe | σ Range | Statistical Meaning |
|---|---|---|
| `below_1s` | < 1.0σ | Deep noise floor |
| `s1_to_s15` | 1.0–1.5σ | Emerging from noise |
| `s15_to_s2` | 1.5–2.0σ | Hint (interesting, not conclusive) |
| `s2_to_s25` | 2.0–2.5σ | Evidence (p < 0.05, 95% CI) |
| `s25_to_s3` | 2.5–3.0σ | Strong (p < 0.01, 99% CI) |
| `above_3s` | > 3.0σ | Discovery (foveal quality) |

**Center-of-mass** uses bin centers [0.5, 1.25, 1.75, 2.25, 2.75, 3.25] as weights.
A rising center-of-mass means the codebook is improving (more candidates reaching
higher significance). A falling center means the codebook is going stale.

### 1.4 Per-Plane σ-Significance for Cross-Plane Vote (cross_plane.rs:967–1074)

**Problem solved**: The cross-plane B_3 lattice classifies candidates into 8 halo
types based on per-plane survivor masks. But the survivor masks are binary
(pass/fail) — they don't capture the DEGREE of significance per plane.

**Solution**: `PlaneSignificance` stores per-plane `SigmaScore`. The halo type is
derived from per-plane significance thresholds instead of binary masks.

**Types**:

| Type | File:Line | Size | Purpose |
|---|---|---|---|
| `PlaneSignificance` | cross_plane.rs:978 | 36 bytes | 3 × SigmaScore (S, P, O) |
| Added to `PartialBinding` | cross_plane.rs:417 | +3 bytes | `plane_sigma: [SignificanceLevel; 3]` |

**Functions**:

| Function | File:Line | Input | Output | Cost |
|---|---|---|---|---|
| `PlaneSignificance::halo_type(min_level)` | cross_plane.rs:991 | &self, SignificanceLevel | HaloType | O(1) |
| `PlaneSignificance::min_level()` | cross_plane.rs:999 | &self | SignificanceLevel | O(1) |
| `PlaneSignificance::max_level()` | cross_plane.rs:1004 | &self | SignificanceLevel | O(1) |
| `PlaneSignificance::levels()` | cross_plane.rs:1009 | &self | [SignificanceLevel; 3] | O(1) |
| `classify_with_sigma()` | cross_plane.rs:1019 | entry_index, distances, gate, min | PartialBinding | O(1) |

### 1.5 EwmTier ← SignificanceLevel Mapping (causal_trajectory.rs:62–78)

**Problem solved**: `EwmTier` (Crystallized/Confident/Transitional/Noise) had no
constructor from the kernel pipeline's output.

**Solution**: `impl From<SignificanceLevel> for EwmTier` with semantic alignment:

| SignificanceLevel | EwmTier | Semantic |
|---|---|---|
| Discovery (> 3σ) | Crystallized | Settled knowledge, high causal weight |
| Strong (2.5–3σ) | Confident | Familiar territory, amplify |
| Evidence (2–2.5σ) | Transitional | Under active revision |
| Hint (1.5–2σ) | Transitional | Under active revision |
| Noise (< 1.5σ) | Noise | Irrelevant, zero causal weight |

---

## 2. Dependency Graph

```
rustynum-core/kernels.rs
  │
  ├── SignificanceLevel (enum, derives Ord)
  ├── SigmaScore { sigma, level, p_value }
  ├── SigmaGate { discovery, strong, evidence, hint, mu, sigma_unit, total_bits }
  ├── score_sigma(ec, gate) → SigmaScore
  ├── K2Histogram { energy, word_conflicts }
  ├── k2_exact_histogram(query, candidate, n_words) → K2Histogram
  │
  ├── KernelResult.sigma: SigmaScore  ←── new field
  └── kernel_pipeline() calls score_sigma() after score_hdr()
         │
         ▼
rustynum-core/hybrid.rs
  │
  └── HybridScore.sigma: SigmaScore  ←── new field, propagated from KernelResult
         │
         ▼
rustynum-bnn/causal_trajectory.rs
  │
  ├── impl From<SignificanceLevel> for EwmTier  ←── bridge
  ├── StripeHistogram { below_1s, ..., above_3s }
  ├── ShiftDirection (enum)
  ├── ShiftSignal { direction, plane_directions, com_delta, magnitude }
  └── ShiftDetector { current, previous, window_count }
         │
         ▼
rustynum-bnn/cross_plane.rs
  │
  ├── PlaneSignificance { s, p, o: SigmaScore }
  ├── PartialBinding.plane_sigma: [SignificanceLevel; 3]  ←── new field
  └── classify_with_sigma(entry, distances, gate, min) → PartialBinding
```

**Direction of dependency** (preserved — rustynum LAW):
```
rustynum-core (types + compute)  ←── LEAF, no upstream deps
    ↑
rustynum-bnn (BNN inference)     ←── imports from rustynum-core only
```

No dependency violations. No IO. No allocations beyond Vec<u16> in K2Histogram.

---

## 3. Integration with Existing Pipeline

### 3.1 K0 → K1 → K2 → σ Flow

```
K0 probe (64-bit)        → bool (reject/pass)
K1 stats (512-bit)       → bool (reject/pass)
K2 exact (full width)    → EnergyConflict { conflict, energy_a, energy_b, agreement }
                              │
                              ├── score_hdr(ec, gate) → HdrScore { hot, mid, cold }  [EXISTING]
                              └── score_sigma(ec, σ_gate) → SigmaScore { sigma, level, p_value }  [NEW]
                                      │
                                      ├── stored in KernelResult.sigma
                                      └── propagated to HybridScore.sigma
```

**Zero overhead on rejection path**: `score_sigma()` only runs on K2 survivors (~5%).
The 4 integer comparisons add < 1 ns per survivor.

### 3.2 Horizontal Sweep Integration (NOT YET WIRED)

The horizontal sweep (`rustynum-arrow/horizontal_sweep.rs`) returns
`Vec<(usize, u64)>` with raw Hamming distances. To add σ-scoring:

```rust
// Post-sweep: convert raw distances to σ-scores
let sigma_gate = SigmaGate::sku_16k();
let sigma_results: Vec<_> = sweep_result.hits.iter().map(|(idx, dist)| {
    let ec = EnergyConflict { conflict: *dist as u32, ..Default::default() };
    (*idx, score_sigma(&ec, &sigma_gate))
}).collect();
```

Cost: O(n_survivors) × 4 comparisons = negligible.

### 3.3 Causal Trajectory Integration

The `ShiftDetector` plugs into the `CausalTrajectory::record_iteration()` loop:

```
Resonator iteration t
  ├── Record snapshot (existing)
  ├── EWM correction (existing)
  ├── BPReLU arrow (existing)
  ├── RIF diff (existing)
  ├── Halo transitions (existing)
  │
  └── [NEW] ShiftDetector.record(plane, sigma) for each candidate scored this iteration
      └── At window boundaries: ShiftDetector.advance_window()
      └── ShiftDetector.detect_shift() → ShiftSignal
           └── ShiftDetector.gate_bias() → Optional CollapseGate bias
```

The shift signal feeds back to `CausalTrajectory::gate_decision()`:
- **TowardFoveal** → bias FLOW (world is clarifying)
- **TowardNoise** → bias HOLD (ground is moving)
- **Bimodal** → bias HOLD (world is splitting)
- **Stable** → no bias (use existing NARS balance)

### 3.4 Cross-Plane Lattice Integration

`classify_with_sigma()` replaces the binary survivor mask approach:

```
Before: per-plane Hamming → binary pass/fail → 8 halo types
After:  per-plane Hamming → SigmaScore per plane → per-plane SignificanceLevel
        → halo_type(min_level) → 8 halo types WITH statistical grounding
```

The `min_level` parameter controls the halo classification sensitivity:
- `min_level = Evidence` (default): 95% CI per plane → conservative
- `min_level = Hint`: catches more candidates but with lower confidence
- `min_level = Discovery`: extremely selective, only foveal-quality matches

---

## 4. Performance Analysis

### 4.1 Cycle Budget

| Operation | Cycles | Notes |
|---|---|---|
| `SigmaGate::new()` | ~20 | Once at init (sqrt + 4 multiplies) |
| `score_sigma()` | ~5 | 4 u32 comparisons + 1 f32 division |
| `k2_exact_histogram()` | Same as k2_exact + ~64 | 256 u16 stores (1/4 cycle each via store buffer) |
| `StripeHistogram::record()` | ~3 | 5 f32 comparisons |
| `ShiftDetector::detect_shift()` | ~30 | 3 × center_of_mass + comparisons |

**Total overhead per K2 survivor**: ~5 cycles for σ-scoring.
**Reference**: K2 exact costs ~256 cycles (256 words × XOR + POPCNT).
**Overhead**: 5/256 = **2%**.

### 4.2 Memory Budget

| Structure | Size | When Allocated |
|---|---|---|
| `SigmaGate` | 28 bytes | Once at init, lives in stack |
| `SigmaScore` | 12 bytes | Per K2 result, in KernelResult |
| `K2Histogram.word_conflicts` | 512 bytes (SKU-16K) | Per histogram call, heap |
| `StripeHistogram` | 24 bytes | 6 per ShiftDetector (3 current + 3 previous) |
| `ShiftDetector` | ~168 bytes | One per CausalTrajectory |
| `PlaneSignificance` | 36 bytes | Per candidate in cross-plane vote |
| `PartialBinding.plane_sigma` | 3 bytes | Per PartialBinding |

**Total new memory per query**: 12 bytes × n_survivors + 168 bytes (detector) ≈ negligible.

### 4.3 Comparison: σ-Scoring vs Cosine

| Metric | σ-Scoring (this work) | Cosine Similarity |
|---|---|---|
| Distance compute | XOR + POPCNT (12 cycles for 2KB) | MUL + ADD + SQRT (~3100 cycles) |
| σ-classification | 4 u32 compares (5 cycles) | Not applicable |
| Total per candidate | ~17 cycles | ~3100 cycles |
| **Speedup** | **182×** | Baseline |
| Statistical grounding | Yes (z-score, p-value) | No (arbitrary threshold) |
| Positional information | Yes (per-word histogram) | No (scalar dot product) |
| Energy decomposition | Yes (EnergyConflict) | No |

---

## 5. SIMD Acceleration Opportunities

### 5.1 Vectorized σ-Classification (Future)

For batch scoring (e.g., horizontal sweep survivors), the 4 comparisons per
candidate can be vectorized:

```
AVX-512: 16 × u32 per register
_mm512_cmplt_epu32_mask(conflicts, discovery_broadcast)  → mask16 of Discovery hits
_mm512_cmplt_epu32_mask(conflicts, strong_broadcast)     → mask16 of Strong hits
...etc.
```

This would score 16 candidates in 4 instructions = 0.25 instructions per candidate.

### 5.2 Histogram via VPOPCNTDQ (Future)

The per-word popcount in `k2_exact_histogram()` currently uses scalar `count_ones()`.
With AVX-512 VPOPCNTDQ:

```
8 words per VPOPCNTDQ instruction
256 words / 8 = 32 instructions for histogram
Store: _mm512_cvtepi64_epi16 (pack 8 × u64 → 8 × u16) = 32 pack + 32 store = 64 instructions
```

vs scalar: 256 × count_ones + 256 stores = ~512 instructions.
**4× speedup** on the histogram path.

### 5.3 Horizontal Sweep + σ Fusion (Future)

The horizontal sweep already computes per-word Hamming distances. Adding
per-word histogram is zero-cost during the sweep — the per-word distances
ARE the histogram. The σ-classification can run after each checkpoint:

```
After examining 8 words: partial_sigma = (mu_scaled - accumulated) / sigma_scaled
If partial_sigma < hint_threshold → early reject with σ-grounded confidence
```

This gives **σ-grounded early exit** instead of arbitrary threshold early exit.

---

## 6. Test Coverage

### 6.1 σ-Significance Tests (kernels.rs, 18 new tests)

| Test | What It Verifies |
|---|---|
| `test_sigma_gate_sku_16k` | μ=8192, σ=64, discovery=8000 for SKU-16K |
| `test_sigma_gate_sku_64k` | μ=32768, σ=128 for SKU-64K |
| `test_sigma_score_noise` | distance=8192 → σ≈0 → Noise |
| `test_sigma_score_discovery` | distance=8000 → σ≈3.0 → Discovery |
| `test_sigma_score_evidence` | distance=8064 → σ≈2.0 → Evidence |
| `test_sigma_score_strong` | distance=8032 → σ≈2.5 → Strong |
| `test_sigma_score_hint` | distance=8096 → σ≈1.5 → Hint |
| `test_sigma_score_anti_correlated` | distance=8500 → σ<0 → Noise, p=1.0 |
| `test_sigma_exact_match_is_discovery` | distance=0 → σ=128 → Discovery |
| `test_sigma_ordering` | Noise < Hint < Evidence < Strong < Discovery |
| `test_k2_histogram_matches_k2_exact` | Histogram aggregate = k2_exact result |
| `test_k2_histogram_per_word_sum` | Sum of word_conflicts = energy.conflict |
| `test_k2_histogram_zero_on_identical` | Identical vectors → all zeros |
| `test_k2_histogram_all_ones_vs_zeros` | 0xFF... vs 0x00... → all words = 64 |
| `test_k2_histogram_variance_localized` | Localized diff → high variance |
| `test_k2_histogram_hot_word_count` | Count of words above threshold |
| `test_k2_histogram_64k` | Correct length for SKU-64K (1024 words) |
| `test_pipeline_results_have_sigma` | kernel_pipeline output includes SigmaScore |

### 6.2 Stripe & Shift Tests (causal_trajectory.rs, 13 new tests)

| Test | What It Verifies |
|---|---|
| `test_ewm_tier_from_significance_discovery` | Discovery → Crystallized |
| `test_ewm_tier_from_significance_strong` | Strong → Confident |
| `test_ewm_tier_from_significance_evidence` | Evidence → Transitional |
| `test_ewm_tier_from_significance_hint` | Hint → Transitional |
| `test_ewm_tier_from_significance_noise` | Noise → Noise |
| `test_stripe_histogram_record` | Values binned to correct stripes |
| `test_stripe_histogram_center_of_mass` | Weighted average is correct |
| `test_stripe_histogram_empty` | Empty histogram → CoM = 0.0 |
| `test_shift_detector_no_data` | No windows → None |
| `test_shift_detector_toward_foveal` | Rising σ → TowardFoveal |
| `test_shift_detector_toward_noise` | Falling σ → TowardNoise |
| `test_shift_detector_stable` | Same distribution → Stable |
| `test_shift_detector_gate_bias` | TowardFoveal → Flow, TowardNoise → Hold |

### 6.3 Test Totals

| Crate | Before | New Tests | After |
|---|---|---|---|
| rustynum-core | ~131 | 18 | ~149 |
| rustynum-bnn | ~87 | 13 | ~100 |
| **Total** | ~218 | **31** | **~249** |

---

## 7. Integration Plan

### Phase 1: Current (DONE — commit b8eca06)

- [x] `SignificanceLevel` enum with Ord derivation
- [x] `SigmaScore` struct with sigma, level, p_value
- [x] `SigmaGate` with precomputed integer thresholds
- [x] `score_sigma()` with integer-only tier decision
- [x] `K2Histogram` with per-word conflict vector
- [x] `k2_exact_histogram()` with 4x-unrolled per-word stores
- [x] `KernelResult.sigma` field
- [x] `HybridScore.sigma` field
- [x] `From<SignificanceLevel> for EwmTier`
- [x] `StripeHistogram` with 6 half-σ bands
- [x] `ShiftDetector` with center-of-mass shift analysis
- [x] `PlaneSignificance` for per-plane σ-scoring
- [x] `classify_with_sigma()` for σ-grounded halo classification
- [x] `PartialBinding.plane_sigma` field
- [x] 31 new tests, all passing

### Phase 2: Near-Term (Next Session)

- [ ] Wire `ShiftDetector` into `CausalTrajectory::record_iteration()`
- [ ] Add σ-based early exit to horizontal sweep (σ-grounded instead of arbitrary threshold)
- [ ] Wire `classify_with_sigma()` into LatticeClimber as alternative to binary mask approach
- [ ] Add `K2Histogram` option to `kernel_pipeline()` for selected survivors
- [ ] Benchmark: σ-significance recall@10 vs HDR recall@10 on Jina embeddings

### Phase 3: Mid-Term

- [ ] AVX-512 vectorized batch `score_sigma()` (16 candidates per instruction)
- [ ] Fuse per-word histogram into VPOPCNTDQ path (zero overhead)
- [ ] σ-grounded Tier 0 prefilter threshold (currently uses arbitrary INT8 dot product cutoff)
- [ ] Wire stripe shift detection into ladybug-rs CollapseGate via rustynum_accel

### Phase 4: Long-Term

- [ ] Adaptive `SigmaGate` that adjusts for non-balanced vectors (skewed energy distributions)
- [ ] Per-word σ-scoring (not just per-container): significance of individual words
- [ ] Multi-resolution histogram: word → cacheline → quarter → half → full vector
- [ ] σ-grounded BF16 tail threshold (currently uses fixed weights)

---

## 8. Mathematical Foundations

### 8.1 The Noise Model

For D-bit balanced random binary vectors (each bit independently Bernoulli(0.5)):

```
Hamming distance H(a, b) ~ Binomial(D, 0.5)
E[H] = D/2
Var[H] = D/4
σ[H] = √(D/4) = √D / 2
```

By CLT (D = 16384 >> 30), H is approximately Gaussian:

```
H ≈ N(D/2, D/4)
```

The z-score z = (D/2 - H) / √(D/4) follows N(0,1) under the null hypothesis
that a and b are independent random vectors.

### 8.2 Significance of the σ-Bands

| Band | z-range | One-tailed p | Two-tailed p | Interpretation |
|---|---|---|---|---|
| Noise | z < 1.5 | > 0.067 | > 0.134 | Not statistically significant |
| Hint | 1.5 ≤ z < 2.0 | 0.023–0.067 | 0.046–0.134 | Marginal significance |
| Evidence | 2.0 ≤ z < 2.5 | 0.006–0.023 | 0.012–0.046 | Conventional 95% CI |
| Strong | 2.5 ≤ z < 3.0 | 0.001–0.006 | 0.002–0.012 | 99% CI |
| Discovery | z ≥ 3.0 | < 0.001 | < 0.002 | Particle physics standard |

### 8.3 SKU-Specific Thresholds

| SKU | D | μ | σ | Discovery (< 3σ) | Strong (< 2.5σ) | Evidence (< 2σ) | Hint (< 1.5σ) |
|---|---|---|---|---|---|---|---|
| 16K | 16384 | 8192 | 64 | < 8000 | < 8032 | < 8064 | < 8096 |
| 64K | 65536 | 32768 | 128 | < 32384 | < 32448 | < 32512 | < 32576 |

### 8.4 The Per-Word Variance Insight

For a random perturbation (noise), the per-word conflict distribution is
approximately uniform — each word has ~32 bits flipped (half of 64). The
variance of a uniform(0, 64) distribution is σ² = 64²/12 ≈ 341.

For a structured difference (e.g., one semantic field differs), the distribution
is bimodal: some words have ~0 conflict, others have ~64. This gives variance
close to 64² × p(1-p) where p is the fraction of differing words.

**Threshold**: variance > 400 suggests structured (localized) disagreement.
Variance < 300 suggests noise-like (distributed) disagreement.

---

## 9. Cross-Repo Impact

### 9.1 ladybug-rs

`ladybug-rs/src/core/rustynum_accel.rs` calls `rustynum_core::kernel_pipeline()`.
The added `sigma` field in `KernelResult` is immediately available:

```rust
let (results, stats) = kernel_pipeline(...);
for r in &results {
    match r.sigma.level {
        SignificanceLevel::Discovery => /* foveal quality, commit */,
        SignificanceLevel::Strong => /* high confidence, bias toward commit */,
        SignificanceLevel::Evidence => /* worth investigating further */,
        SignificanceLevel::Hint => /* interesting, accumulate evidence */,
        SignificanceLevel::Noise => /* not significant, skip */,
    }
}
```

### 9.2 crewai-rust

The `ShiftDetector.gate_bias()` output can inform Blackboard agent decisions:
- **TowardFoveal** → agents can commit to decisions faster
- **TowardNoise** → agents should gather more evidence before acting
- **Bimodal** → world is splitting, consider spawning sub-agents for each branch

### 9.3 n8n-rs

The `SigmaScore.p_value` can be included in workflow node outputs as a
standard statistical confidence measure, compatible with external analytics
pipelines that expect p-values.
