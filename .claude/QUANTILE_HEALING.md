# QUANTILE_HEALING.md

## Quantile-Aware Self-Healing Precision

**Status:** Architecture spec. Builds on SESSION_D_LENS_CORRECTION.md.
**Prereqs:** Session C (ReservoirSample, Welford), Session D (gamma, cushion, fold).

---

## THE CORE IDEA

The cascade makes binary decisions (reject/pass) but the CONFIDENCE is continuous.
A candidate at the boundary of rejection gets the same treatment as one deep in noise.
Self-healing reads EXACTLY enough additional bytes to resolve ambiguity — no more.

---

## UNCERTAINTY QUANTIFICATION

Sampling N bits from a vector of T total bits. The projection uncertainty:

```
σ_est = (T/N) × √(N × p × (1-p))

where p = projected_distance / T

At 1/16 sample (N = 1024, T = 16384):
  σ_est ≈ 16 × √(1024 × 0.5 × 0.5) = 16 × 16 = 256

At 1/4 sample (N = 4096):
  σ_est ≈ 4 × √(4096 × 0.5 × 0.5) = 4 × 32 = 128

At full (N = T):
  σ_est = 0 (exact)

Uncertainty shrinks as √N. Each additional byte HEALS a quantifiable amount.
```

---

## BOUNDARY PRESSURE

The density of candidates AT each band boundary determines healing priority:

```
Dense boundary (high pressure):
  Many candidates cluster near it.
  A small distance change flips the band assignment.
  This boundary is DOING WORK. It needs precision.
  → Read more bytes to resolve.

Sparse boundary (low pressure):
  Few candidates near it. Natural gap in the distribution.
  The boundary is FREE. No precision needed.
  → Accept the projection, save bytes.
```

Pressure is measured from the ReservoirSample:

```rust
fn boundary_pressure(&self, boundary: u32, window: u32) -> u32 {
    self.reservoir.samples.iter()
        .filter(|&&d| d >= boundary.saturating_sub(window)
                    && d <= boundary.saturating_add(window))
        .count() as u32
}
```

---

## ADAPTIVE HEALING

```rust
fn healing_target(
    &self,
    projected: u32,       // current projected distance
    sigma_est: u32,       // current estimation uncertainty
    bytes_read: usize,    // bytes consumed so far
    total_bytes: usize,   // full vector size
) -> usize {
    // Find nearest boundary and its pressure
    let (nearest_dist, pressure) = self.nearest_boundary_pressure(projected);
    
    // Far from any boundary → confident, no healing needed
    if nearest_dist > sigma_est * 3 { return bytes_read; }
    
    // Urgency = pressure × proximity
    let urgency = pressure as u64 * sigma_est as u64 / nearest_dist.max(1) as u64;
    
    let additional = match urgency {
        0..=50 => 0,                     // accept uncertainty
        51..=200 => 64,                  // one cache line
        201..=500 => 256,                // moderate healing
        501..=1000 => 512,               // significant healing
        _ => total_bytes - bytes_read,   // go to full
    };
    
    // SIMD-align to 64 bytes
    ((bytes_read + additional + 63) & !63).min(total_bytes)
}
```

---

## SELF-ORGANIZING BOUNDARY FOLD

Boundaries MOVE toward density troughs. Reduces the need for healing
by eliminating ambiguity at the source:

```
BEFORE FOLD:
  Many candidates at boundary b₁ → high pressure → lots of healing needed

FOLD:
  b₁ slides toward nearest density trough (minimum pressure position)

AFTER FOLD:
  Few candidates at new b₁ → low pressure → minimal healing
  The fold ELIMINATED the uncertainty instead of resolving it.
```

The fold search uses golden-section search for optimal trough finding
(unimodal density between peaks, φ-convergent):

```rust
fn fold_boundaries(&mut self) {
    if self.reservoir.len() < 500 { return; }
    let mut sorted = self.reservoir.samples.clone();
    sorted.sort_unstable();
    
    for i in 0..4 {
        let lo = self.bands[i].saturating_sub(self.sigma);
        let hi = self.bands[i].saturating_add(self.sigma);
        self.bands[i] = golden_section_min_density(&sorted, lo, hi, self.sigma / 4);
        self.pressure[i] = self.boundary_pressure(self.bands[i], self.sigma / 4);
    }
    // Ensure monotonicity
    for i in 1..4 {
        if self.bands[i] <= self.bands[i-1] { self.bands[i] = self.bands[i-1] + 1; }
    }
}
```

---

## GAMMA + CUSHION LENS CORRECTION

Before classifying, correct the raw distance for distribution skew and kurtosis:

```
Raw distance → γ correction (fix asymmetry) → κ correction (fix tail weight) → band lookup

γ from Pearson's skewness: γ = 1.0 + skew × 0.15, clamped [0.5, 2.0]
κ from excess kurtosis: κ = kurtosis/300, normalized so 1.0 = normal

γ correction: 256-entry integer LUT, rebuilt on calibration
κ correction: cubic deviation term, ~5 integer ops

Together they make σ-based bands work on ANY distribution shape.
The empirical quantile fallback becomes a diagnostic, not the primary path.
```

---

## THE FULL HEALING PIPELINE

```
OBSERVE → CALIBRATE → QUERY → HEAL → OBSERVE (feedback loop)

1. Reservoir observes actual distances from confirmed matches.
2. Distribution shape updates (skewness → γ, kurtosis → κ).
3. Boundaries fold to density troughs.
4. Next query uses corrected distances + folded bands.
5. Uncertain comparisons heal with adaptive byte allocation.
6. Healing results feed back into reservoir.

The cascade LEARNS what precision it needs.
Tight clusters → less precision needed → fewer bytes read → faster.
Wide spread → more precision needed → more bytes → accurate.
Bimodal → detect the gap → fold boundary into trough → no false negatives.

Convergence: minimum bytes per comparison for target confidence.
```

---

## CONNECTION TO PLANE ALPHA CHANNEL

```
CASCADE LEVEL:                    PLANE LEVEL:
projected near boundary           acc[k] near threshold
  → pressure high                   → alpha uncertain
  → needs more bytes                → needs more encounters
  → heal by reading                 → heal by accumulating

projected far from boundary       |acc[k]| >> threshold
  → pressure low                    → alpha committed
  → accept projection               → bit is defined
  → save bandwidth                   → stable

SAME PRINCIPLE AT TWO SCALES:
  Cascade: uncertainty about a COMPARISON (heals by reading more bytes)
  Plane:   uncertainty about a BIT POSITION (heals by accumulating encounters)
```
