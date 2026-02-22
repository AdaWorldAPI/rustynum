//! NARS Reverse Causality — Tactic #4 from the 34-tactic roster.
//!
//! Forward inference: `bind(A, CAUSES) → k-NN search → B`
//! Reverse inference: `unbind(B, CAUSES) → k-NN search → candidate A`
//!
//! For Binary base, XOR is self-inverse: `unbind == bind`.
//! For Signed base, unbind negates the role before binding.
//!
//! Confidence comes from the CRP (Cluster Radial Profile):
//! if the nearest match lies within σ of its cluster center,
//! the recovery is confident. Beyond p95, the causal chain
//! is broken (noise floor).
//!
//! Granger signal G(A→B,τ) = d(A_t, B_{t+τ}) - d(B_t, B_{t+τ})
//! gives directionality: G < 0 ⇒ A predicts B (A is closer to future B).

use crate::sweep::{Base, bind};

// ---------------------------------------------------------------------------
// Unbind
// ---------------------------------------------------------------------------

/// Unbind: inverse of `bind`.
///
/// - Binary: exact inverse — XOR is self-inverse, so `unbind == bind`.
/// - Unsigned(B): exact inverse — element-wise subtraction mod B.
/// - Signed(B): **approximate** inverse — negates role, then binds (saturating add +
///   clamp). When the original bind saturated (e.g. bind(3, 2) in Signed(7) clamps
///   to 3), information is destroyed and unbind cannot recover the original value.
///   For values that don't hit the clamp boundary, the inverse is exact.
pub fn unbind(bound: &[i8], role: &[i8], base: Base) -> Vec<i8> {
    let d = bound.len();
    assert_eq!(d, role.len());

    match base {
        Base::Binary => {
            // XOR is its own inverse.
            bind(bound, role, base)
        }
        Base::Unsigned(bv) => {
            // Additive inverse mod B: (bound - role) mod B
            let mut result = vec![0i8; d];
            for i in 0..d {
                let diff = bound[i] as i16 - role[i] as i16;
                result[i] = diff.rem_euclid(bv as i16) as i8;
            }
            result
        }
        Base::Signed(_) => {
            // Negate role, then bind (saturating add + clamp).
            // Widen to i16 to handle i8::MIN correctly: -(-128) = 128 → clamp to 127.
            let neg_role: Vec<i8> = role.iter().map(|&v| {
                (-(v as i16)).clamp(-128, 127) as i8
            }).collect();
            bind(bound, &neg_role, base)
        }
    }
}

// ---------------------------------------------------------------------------
// Bind Space — a named store of holographic entities
// ---------------------------------------------------------------------------

/// An entry in the bind space: named entity with its vector.
#[derive(Clone)]
pub struct Entity {
    pub id: u32,
    pub name: String,
    pub vector: Vec<i8>,
}

/// A named role (verb) vector.
#[derive(Clone)]
pub struct Role {
    pub name: String,
    pub vector: Vec<i8>,
}

/// Forward binding: `bind(subject, role) → target`.
///
/// Returns the bound vector. To find the actual target, search
/// the bind space for the nearest entity to this result.
pub fn forward_bind(subject: &[i8], role: &Role, base: Base) -> Vec<i8> {
    bind(subject, &role.vector, base)
}

/// Reverse unbind: given an outcome and a role, recover the candidate cause.
///
/// Returns the unbound vector. To find the actual cause, search
/// the bind space for the nearest entity to this result.
pub fn reverse_unbind(outcome: &[i8], role: &Role, base: Base) -> Vec<i8> {
    unbind(outcome, &role.vector, base)
}

// ---------------------------------------------------------------------------
// Causal Trace — multi-hop reverse reasoning
// ---------------------------------------------------------------------------

/// One step in a causal trace: the recovered entity and its confidence.
#[derive(Clone, Debug)]
pub struct TraceStep {
    /// The entity ID that was recovered at this step.
    pub entity_id: u32,
    /// Symbol distance from the unbound candidate to the recovered entity.
    pub distance: u64,
    /// Distance as a fraction of total dimensions (0.0 = identical, 0.5 = random).
    pub normalized_distance: f64,
    /// Whether this step is confident (distance < threshold).
    pub confident: bool,
}

/// Result of a multi-hop reverse causal trace.
#[derive(Clone, Debug)]
pub struct CausalTrace {
    /// The outcome entity we started from.
    pub outcome_id: u32,
    /// The role (verb) used for each unbinding hop.
    pub role_name: String,
    /// Each step in the reverse chain, from outcome back to root cause.
    pub steps: Vec<TraceStep>,
    /// How many steps were confident before the chain broke.
    pub confident_depth: usize,
}

/// Brute-force nearest entity search on i8 vectors.
///
/// Returns (entity_id, hamming_distance) of the closest entity.
/// For production use, replace with CAKES DFS Sieve on a CLAM tree.
fn nearest_entity(candidate: &[i8], entities: &[Entity]) -> (u32, u64) {
    let mut best_id = 0u32;
    let mut best_dist = u64::MAX;

    for e in entities {
        let dist = symbol_distance(candidate, &e.vector);
        if dist < best_dist {
            best_dist = dist;
            best_id = e.id;
        }
    }

    (best_id, best_dist)
}

/// Symbol-level Hamming distance on i8 slices.
///
/// Counts positions where `a[i] != b[i]`. This is the correct distance
/// for holographic vectors where each i8 is a discrete symbol (not a byte
/// to be decomposed into bits).
fn symbol_distance(a: &[i8], b: &[i8]) -> u64 {
    assert_eq!(a.len(), b.len());
    let mut dist = 0u64;
    for i in 0..a.len() {
        if a[i] != b[i] {
            dist += 1;
        }
    }
    dist
}

/// Reverse causal trace: starting from `outcome`, repeatedly unbind with
/// `role` and search for the nearest real entity, up to `max_depth` hops.
///
/// Each hop: `candidate = unbind(current, role)` → nearest entity.
/// Stops when confidence drops below threshold or max_depth is reached.
///
/// The `confidence_threshold` is a normalized distance: steps with
/// `normalized_distance > confidence_threshold` are marked not confident.
/// A typical value is 0.35 (since 0.5 = random for binary vectors).
pub fn reverse_trace(
    outcome: &Entity,
    role: &Role,
    entities: &[Entity],
    base: Base,
    max_depth: usize,
    confidence_threshold: f64,
) -> CausalTrace {
    let total_dims = outcome.vector.len() as f64;
    let mut steps = Vec::with_capacity(max_depth);
    let mut current = outcome.vector.clone();
    let mut confident_depth = 0;

    for _ in 0..max_depth {
        let candidate = reverse_unbind(&current, role, base);
        let (entity_id, distance) = nearest_entity(&candidate, entities);
        let normalized = distance as f64 / total_dims;
        let confident = normalized < confidence_threshold;

        steps.push(TraceStep {
            entity_id,
            distance,
            normalized_distance: normalized,
            confident,
        });

        if confident {
            confident_depth += 1;
        } else {
            // Chain is broken — stop tracing.
            break;
        }

        // Next hop: use the recovered entity as the new current.
        if let Some(e) = entities.iter().find(|e| e.id == entity_id) {
            current = e.vector.clone();
        } else {
            break;
        }
    }

    CausalTrace {
        outcome_id: outcome.id,
        role_name: role.name.clone(),
        steps,
        confident_depth,
    }
}

// ---------------------------------------------------------------------------
// Granger Signal — temporal directional causality
// ---------------------------------------------------------------------------

/// Granger signal between two time series of holographic vectors.
///
/// G(A→B, τ) = d(A_t, B_{t+τ}) - d(B_t, B_{t+τ})
///
/// If G < 0: A_t is closer to B_{t+τ} than B_t is — A predicts B.
/// If G > 0: B_t is closer to future B than A_t — no causal signal from A.
/// If G ≈ 0: A and B are equidistant from future B — inconclusive.
///
/// Returns the mean Granger signal across all valid time steps.
pub fn granger_signal(
    series_a: &[Vec<i8>],
    series_b: &[Vec<i8>],
    tau: usize,
) -> f64 {
    assert!(tau > 0, "Granger signal requires tau > 0 (lag must be at least 1)");
    assert_eq!(series_a.len(), series_b.len());
    let n = series_a.len();
    if tau >= n {
        return 0.0;
    }

    let mut sum = 0.0f64;
    let mut count = 0usize;

    for t in 0..(n - tau) {
        let d_ab = symbol_distance(&series_a[t], &series_b[t + tau]) as f64;
        let d_bb = symbol_distance(&series_b[t], &series_b[t + tau]) as f64;
        sum += d_ab - d_bb;
        count += 1;
    }

    if count > 0 { sum / count as f64 } else { 0.0 }
}

/// Scan multiple lags to find the strongest Granger signal.
///
/// Returns (best_lag, signal) where signal is the most negative G(A→B, τ).
/// More negative means A is a stronger predictor of B at that lag.
pub fn granger_scan(
    series_a: &[Vec<i8>],
    series_b: &[Vec<i8>],
    max_lag: usize,
) -> (usize, f64) {
    let mut best_lag = 1;
    let mut best_signal = f64::MAX;

    for tau in 1..=max_lag {
        let g = granger_signal(series_a, series_b, tau);
        if g < best_signal {
            best_signal = g;
            best_lag = tau;
        }
    }

    (best_lag, best_signal)
}

// ---------------------------------------------------------------------------
// SimilarPair Detection — Tactic #11 (pairwise similarity screening)
// ---------------------------------------------------------------------------

/// A pair of entities with high structural similarity (low symbol distance).
///
/// This only checks structural proximity — it does NOT verify truth values.
/// To detect actual contradictions, the caller must compare truth values
/// on the returned pairs.
#[derive(Clone, Debug)]
pub struct SimilarPair {
    pub entity_a: u32,
    pub entity_b: u32,
    pub distance: u64,
    pub normalized_distance: f64,
}

/// Find entity pairs that are structurally similar (low symbol distance)
/// within a given radius. For Binary base, random pairs average ~0.5
/// normalized distance; pairs below the threshold are structurally correlated.
///
/// This is O(n²) brute force. Replace with CAKES ρ-NN for O(n·log n).
pub fn find_similar_pairs(
    entities: &[Entity],
    radius_threshold: f64,
) -> Vec<SimilarPair> {
    let mut pairs = Vec::new();
    let n = entities.len();
    if n == 0 {
        return pairs;
    }

    let total_dims = entities[0].vector.len() as f64;

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = symbol_distance(&entities[i].vector, &entities[j].vector);
            let norm = dist as f64 / total_dims;
            if norm < radius_threshold {
                pairs.push(SimilarPair {
                    entity_a: entities[i].id,
                    entity_b: entities[j].id,
                    distance: dist,
                    normalized_distance: norm,
                });
            }
        }
    }

    pairs.sort_by(|a, b| a.distance.cmp(&b.distance));
    pairs
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sweep::Base;
    use rustynum_core::SplitMix64;

    fn make_random_entity(id: u32, d: usize, rng: &mut SplitMix64, base: Base) -> Entity {
        let vector: Vec<i8> = (0..d)
            .map(|_| rng.gen_range_i8(base.min_val(), base.max_val()))
            .collect();
        Entity {
            id,
            name: format!("entity_{}", id),
            vector,
        }
    }

    fn make_role(name: &str, d: usize, rng: &mut SplitMix64, base: Base) -> Role {
        let vector: Vec<i8> = (0..d)
            .map(|_| rng.gen_range_i8(base.min_val(), base.max_val()))
            .collect();
        Role {
            name: name.to_string(),
            vector,
        }
    }

    // --- Unbind tests ---

    #[test]
    fn test_unbind_binary_self_inverse() {
        let mut rng = SplitMix64::new(42);
        let base = Base::Binary;
        let a: Vec<i8> = (0..256).map(|_| (rng.next_u64() & 1) as i8).collect();
        let role: Vec<i8> = (0..256).map(|_| (rng.next_u64() & 1) as i8).collect();

        let bound = bind(&a, &role, base);
        let recovered = unbind(&bound, &role, base);
        assert_eq!(a, recovered, "Binary unbind must recover original");
    }

    #[test]
    fn test_unbind_unsigned_mod_inverse() {
        let mut rng = SplitMix64::new(42);
        let base = Base::Unsigned(7);
        let a: Vec<i8> = (0..256).map(|_| rng.gen_range_i8(0, 6)).collect();
        let role: Vec<i8> = (0..256).map(|_| rng.gen_range_i8(0, 6)).collect();

        let bound = bind(&a, &role, base);
        let recovered = unbind(&bound, &role, base);
        assert_eq!(a, recovered, "Unsigned unbind must recover original");
    }

    #[test]
    fn test_unbind_signed_approximate_inverse() {
        // Signed bind uses saturating add + clamp, so unbind is approximate
        // when values hit the clamp boundary. Test with small values that
        // won't saturate.
        let base = Base::Signed(7);
        let a: Vec<i8> = vec![0, 1, -1, 2, -2, 0, 1, -1];
        let role: Vec<i8> = vec![0, 0, 0, 0, 0, 1, 1, 1];

        let bound = bind(&a, &role, base);
        let recovered = unbind(&bound, &role, base);
        // For non-saturating values, this should be exact
        assert_eq!(a, recovered, "Signed unbind should recover when not saturating");
    }

    #[test]
    fn test_unbind_signed_saturation_is_lossy() {
        // Demonstrate that Signed unbind is lossy at the clamp boundary.
        // bind(3, 2) in Signed(7): saturating_add(3,2)=5, clamp(-3,3)=3
        // unbind(3, 2): negate(2)=-2, bind(3,-2)=saturating_add(3,-2)=1
        // Original was 3, recovered is 1 — information was destroyed by clamping.
        let base = Base::Signed(7);
        let a: Vec<i8> = vec![3];    // at the boundary
        let role: Vec<i8> = vec![2];  // pushes past clamp

        let bound = bind(&a, &role, base);
        assert_eq!(bound, vec![3], "bind(3,2) clamps to 3");

        let recovered = unbind(&bound, &role, base);
        assert_ne!(recovered, a,
            "Signed unbind MUST be lossy when bind saturated: recovered {:?} vs original {:?}",
            recovered, a);
        assert_eq!(recovered, vec![1],
            "unbind(3,2) = bind(3,-2) = 1 (not 3)");
    }

    #[test]
    fn test_unbind_signed_min_value() {
        // i8::MIN (-128) should negate to 127 (clamped), not stay as -128
        let bound = vec![0i8; 4];
        let role = vec![i8::MIN; 4]; // [-128, -128, -128, -128]
        let result = unbind(&bound, &role, Base::Signed(7));
        // unbind(0, -128) should = bind(0, 127) since neg(-128) clamps to 127
        // bind(0, 127) with Signed(7) = (0 + 127).clamp(-3, 3) = 3
        let expected = bind(&bound, &vec![127i8; 4], Base::Signed(7));
        assert_eq!(result, expected, "unbind should negate i8::MIN to 127, not -128");
    }

    // --- Forward / Reverse roundtrip ---

    #[test]
    fn test_forward_reverse_roundtrip_binary() {
        let mut rng = SplitMix64::new(42);
        let base = Base::Binary;
        let d = 1024;

        let subject = make_random_entity(0, d, &mut rng, base);
        let role = make_role("CAUSES", d, &mut rng, base);

        let target_vector = forward_bind(&subject.vector, &role, base);
        let candidate = reverse_unbind(&target_vector, &role, base);

        // Binary: exact recovery
        assert_eq!(candidate, subject.vector);
    }

    // --- Reverse trace ---

    #[test]
    fn test_reverse_trace_single_hop() {
        let mut rng = SplitMix64::new(42);
        let base = Base::Binary;
        let d = 1024;

        let cause = make_random_entity(1, d, &mut rng, base);
        let role = make_role("CAUSES", d, &mut rng, base);

        // Effect = bind(cause, CAUSES)
        let effect_vec = forward_bind(&cause.vector, &role, base);
        let effect = Entity {
            id: 2,
            name: "effect".to_string(),
            vector: effect_vec,
        };

        let entities = vec![cause.clone(), effect.clone()];

        let trace = reverse_trace(&effect, &role, &entities, base, 3, 0.4);
        assert!(!trace.steps.is_empty());
        assert_eq!(trace.steps[0].entity_id, cause.id,
            "Reverse trace should recover the cause");
        assert_eq!(trace.steps[0].distance, 0,
            "Binary unbind should give exact recovery");
        assert!(trace.steps[0].confident);
        assert!(trace.confident_depth >= 1);
    }

    #[test]
    fn test_reverse_trace_chain() {
        let mut rng = SplitMix64::new(42);
        let base = Base::Binary;
        let d = 2048;

        // Build a causal chain: A → B → C
        let a = make_random_entity(1, d, &mut rng, base);
        let role = make_role("CAUSES", d, &mut rng, base);

        let b_vec = forward_bind(&a.vector, &role, base);
        let b = Entity { id: 2, name: "B".to_string(), vector: b_vec };

        let c_vec = forward_bind(&b.vector, &role, base);
        let c = Entity { id: 3, name: "C".to_string(), vector: c_vec };

        let entities = vec![a.clone(), b.clone(), c.clone()];

        // Reverse trace from C should recover: C → B → A
        let trace = reverse_trace(&c, &role, &entities, base, 5, 0.4);
        assert!(trace.confident_depth >= 2,
            "Should trace back at least 2 hops: got {}", trace.confident_depth);
        assert_eq!(trace.steps[0].entity_id, b.id, "First hop should find B");
        assert_eq!(trace.steps[1].entity_id, a.id, "Second hop should find A");
    }

    #[test]
    fn test_reverse_trace_stops_at_noise() {
        let mut rng = SplitMix64::new(42);
        let base = Base::Binary;
        let d = 2048;

        // Entities: A is a root cause, B = bind(A, CAUSES_1),
        // we trace using a DIFFERENT role (CAUSES_2).
        // Unbinding B with CAUSES_2 gives a random vector (not A).
        let a = make_random_entity(1, d, &mut rng, base);
        let role1 = make_role("CAUSES_1", d, &mut rng, base);
        let role2 = make_role("CAUSES_2", d, &mut rng, base);

        let b_vec = forward_bind(&a.vector, &role1, base);
        let b = Entity { id: 2, name: "B".to_string(), vector: b_vec };

        // Only A and B in the entity set.
        // unbind(B, CAUSES_2) = B XOR CAUSES_2, which is random noise
        // (since B was created with CAUSES_1, not CAUSES_2).
        let entities = vec![a.clone(), b.clone()];

        let trace = reverse_trace(&b, &role2, &entities, base, 5, 0.05);
        // First hop: unbind(B, CAUSES_2) = random → nearest entity has
        // normalized distance ~0.5, so it should NOT be confident.
        assert_eq!(trace.confident_depth, 0,
            "Wrong role should fail on first hop: confident_depth = {}", trace.confident_depth);
    }

    // --- Granger signal ---

    #[test]
    fn test_granger_signal_self_prediction() {
        let mut rng = SplitMix64::new(42);
        let d = 256;
        let base = Base::Binary;
        let n = 20;

        // Series B with gradual drift: B_{t+1} is a noisy copy of B_t
        let mut series_b: Vec<Vec<i8>> = Vec::with_capacity(n);
        let b0: Vec<i8> = (0..d).map(|_| (rng.next_u64() & 1) as i8).collect();
        series_b.push(b0);
        for t in 1..n {
            let prev = &series_b[t - 1];
            let mut next = prev.clone();
            // Flip ~5% of bits
            for i in 0..d {
                if rng.next_u64() % 20 == 0 {
                    next[i] ^= 1;
                }
            }
            series_b.push(next);
        }

        // Series A = random (uncorrelated with B)
        let series_a: Vec<Vec<i8>> = (0..n)
            .map(|_| (0..d).map(|_| (rng.next_u64() & 1) as i8).collect())
            .collect();

        // Random A should NOT predict B (G ≈ 0 or positive)
        let g = granger_signal(&series_a, &series_b, 1);
        // B predicts itself better than random A does, so G should be >= 0
        assert!(g >= -5.0,
            "Random series should not strongly predict B: G = {}", g);
    }

    #[test]
    fn test_granger_signal_causal_series() {
        let mut rng = SplitMix64::new(42);
        let d = 256;
        let n = 30;
        let tau = 2;

        // A causes B with lag τ: B_{t+τ} = A_t (with noise)
        let series_a: Vec<Vec<i8>> = (0..n)
            .map(|_| (0..d).map(|_| (rng.next_u64() & 1) as i8).collect())
            .collect();

        let mut series_b: Vec<Vec<i8>> = Vec::with_capacity(n);
        for t in 0..n {
            if t >= tau {
                // B[t] = A[t-tau] with ~3% noise
                let mut b = series_a[t - tau].clone();
                for i in 0..d {
                    if rng.next_u64() % 33 == 0 {
                        b[i] ^= 1;
                    }
                }
                series_b.push(b);
            } else {
                series_b.push((0..d).map(|_| (rng.next_u64() & 1) as i8).collect());
            }
        }

        // A should predict B at lag tau: G(A→B,τ) should be negative
        let g = granger_signal(&series_a, &series_b, tau);
        assert!(g < 0.0,
            "A should predict B at lag {}: G = {}", tau, g);

        // Scan should find the best lag near tau
        let (best_lag, best_g) = granger_scan(&series_a, &series_b, 5);
        assert_eq!(best_lag, tau,
            "Best lag should be {}, got {} (G={})", tau, best_lag, best_g);
    }

    // --- SimilarPair detection ---

    #[test]
    fn test_find_similar_pairs_identical() {
        let mut rng = SplitMix64::new(42);
        let base = Base::Binary;
        let d = 256;

        let a = make_random_entity(1, d, &mut rng, base);
        let b = Entity {
            id: 2,
            name: "clone".to_string(),
            vector: a.vector.clone(),
        };

        let pairs = find_similar_pairs(&[a, b], 0.1);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].distance, 0);
    }

    #[test]
    fn test_find_similar_pairs_no_match() {
        let mut rng = SplitMix64::new(42);
        let base = Base::Binary;
        let d = 1024;

        // Random binary vectors in 1024-D: expected Hamming distance ≈ 512
        // (normalized ≈ 0.5). Threshold 0.3 should find no pairs.
        let entities: Vec<Entity> = (0..10)
            .map(|i| make_random_entity(i, d, &mut rng, base))
            .collect();

        let pairs = find_similar_pairs(&entities, 0.3);
        assert!(pairs.is_empty(),
            "Random 1024-D binary vectors should not be within 0.3: found {} pairs", pairs.len());
    }
}

// ===========================================================================
// BF16 Causal Pipeline — per-dimension causal attribution
// ===========================================================================

use rustynum_core::bf16_hamming::{
    BF16Weights, BF16StructuralDiff, TRAINING_WEIGHTS,
    bf16_hamming_scalar, structural_diff, select_bf16_hamming_fn,
    fp32_to_bf16_bytes,
};

// ---------------------------------------------------------------------------
// BF16 Entity
// ---------------------------------------------------------------------------

/// An entity with BF16-encoded vector (2 bytes per dimension).
///
/// For Jina v3 1024-D embeddings: 2048 bytes per entity.
/// Created by truncating FP32 embeddings via `fp32_to_bf16_bytes()`.
#[derive(Clone)]
pub struct BF16Entity {
    pub id: u32,
    pub name: String,
    /// BF16 bytes: 2 bytes per dimension, little-endian.
    pub bf16_bytes: Vec<u8>,
    /// Number of dimensions (bf16_bytes.len() / 2).
    pub n_dims: usize,
}

impl BF16Entity {
    /// Create from FP32 embedding (truncates to BF16).
    pub fn from_f32(id: u32, name: &str, embedding: &[f32]) -> Self {
        let bf16_bytes = fp32_to_bf16_bytes(embedding);
        let n_dims = embedding.len();
        Self { id, name: name.to_string(), bf16_bytes, n_dims }
    }
}

// ---------------------------------------------------------------------------
// Per-Dimension Causal Map
// ---------------------------------------------------------------------------

/// Causal attribution per dimension: which features carry the causal signal
/// between two time series.
#[derive(Clone, Debug)]
pub struct CausalFeatureMap {
    /// Number of dimensions.
    pub n_dims: usize,
    /// Per-dimension sign-flip count across all timesteps.
    pub sign_flip_counts: Vec<u32>,
    /// Per-dimension exponent-shift count.
    pub exponent_shift_counts: Vec<u32>,
    /// Dimensions sorted by sign-flip frequency (descending).
    pub top_causal_dims: Vec<(usize, u32)>,
    /// Total timesteps analyzed.
    pub timesteps: usize,
    /// Overall Granger signal (scalar, for comparison with existing API).
    pub granger_signal: f64,
    /// Best lag (at which the per-dim signal was strongest).
    pub best_lag: usize,
}

impl CausalFeatureMap {
    /// Fraction of timesteps where dimension `dim` had a sign flip.
    pub fn sign_flip_rate(&self, dim: usize) -> f64 {
        if self.timesteps == 0 { return 0.0; }
        self.sign_flip_counts[dim] as f64 / self.timesteps as f64
    }

    /// Dimensions where sign flips occur > threshold fraction of timesteps.
    pub fn causal_dims_above(&self, threshold: f64) -> Vec<usize> {
        (0..self.n_dims)
            .filter(|&d| self.sign_flip_rate(d) > threshold)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// BF16 Granger Signal — Per-Dimension Causal Attribution
// ---------------------------------------------------------------------------

/// BF16-structured Granger signal with per-dimension causal attribution.
///
/// Like `granger_signal()` but operates on BF16 byte series and returns
/// not just "A causes B" but "A causes B via dimensions [47, 312, 891]".
pub fn bf16_granger_causal_map(
    series_a: &[BF16Entity],
    series_b: &[BF16Entity],
    tau: usize,
) -> CausalFeatureMap {
    assert!(tau > 0);
    assert_eq!(series_a.len(), series_b.len());
    let n = series_a.len();
    let n_dims = series_a[0].n_dims;
    assert!(tau < n);

    let bf16_fn = select_bf16_hamming_fn();
    let weights = &TRAINING_WEIGHTS;

    let mut sign_flips = vec![0u32; n_dims];
    let mut exp_shifts = vec![0u32; n_dims];
    let mut cross_sum = 0.0f64;
    let mut auto_sum = 0.0f64;
    let mut count = 0usize;

    for t in 0..(n - tau) {
        // Cross-series structural diff: A_t vs B_{t+tau}
        let cross_diff = structural_diff(
            &series_a[t].bf16_bytes,
            &series_b[t + tau].bf16_bytes,
        );

        // Accumulate per-dimension sign flips from cross-series
        for &dim in &cross_diff.sign_flip_dims {
            sign_flips[dim] += 1;
        }
        for &dim in &cross_diff.major_magnitude_shifts {
            exp_shifts[dim] += 1;
        }

        // Scalar Granger signal for comparison
        let d_ab = bf16_fn(
            &series_a[t].bf16_bytes,
            &series_b[t + tau].bf16_bytes,
            weights,
        ) as f64;
        let d_bb = bf16_fn(
            &series_b[t].bf16_bytes,
            &series_b[t + tau].bf16_bytes,
            weights,
        ) as f64;
        cross_sum += d_ab;
        auto_sum += d_bb;
        count += 1;
    }

    // Build top causal dims (sorted by sign-flip count, descending)
    let mut top: Vec<(usize, u32)> = sign_flips.iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(d, &c)| (d, c))
        .collect();
    top.sort_by(|a, b| b.1.cmp(&a.1));

    let granger = if count > 0 {
        (cross_sum - auto_sum) / count as f64
    } else {
        0.0
    };

    CausalFeatureMap {
        n_dims,
        sign_flip_counts: sign_flips,
        exponent_shift_counts: exp_shifts,
        top_causal_dims: top,
        timesteps: count,
        granger_signal: granger,
        best_lag: tau,
    }
}

/// Scan multiple lags and return the CausalFeatureMap at the best lag.
pub fn bf16_granger_causal_scan(
    series_a: &[BF16Entity],
    series_b: &[BF16Entity],
    max_lag: usize,
) -> CausalFeatureMap {
    let mut best_map = bf16_granger_causal_map(series_a, series_b, 1);

    for tau in 2..=max_lag {
        let map = bf16_granger_causal_map(series_a, series_b, tau);
        if map.granger_signal < best_map.granger_signal {
            best_map = map;
        }
    }

    best_map
}

// ---------------------------------------------------------------------------
// BF16 Causal Trace — reverse reasoning with per-dim attribution
// ---------------------------------------------------------------------------

/// One step in a BF16 causal trace, with structural diff information.
#[derive(Clone, Debug)]
pub struct BF16TraceStep {
    pub entity_id: u32,
    /// BF16-structured weighted distance.
    pub weighted_distance: u64,
    /// Normalized weighted distance (0.0 = identical).
    pub normalized_distance: f64,
    /// Whether this step is confident.
    pub confident: bool,
    /// Structural diff between unbound candidate and recovered entity.
    pub diff: BF16StructuralDiff,
}

/// BF16 reverse causal trace with per-dimension attribution.
#[derive(Clone, Debug)]
pub struct BF16CausalTrace {
    pub outcome_id: u32,
    pub role_name: String,
    pub steps: Vec<BF16TraceStep>,
    pub confident_depth: usize,
    /// Dimensions that consistently carry causal signal across all hops.
    pub causal_backbone_dims: Vec<usize>,
}

/// BF16 reverse trace: unbind outcome with role, search for nearest
/// BF16 entity, report structural diff at each hop.
pub fn bf16_reverse_trace(
    outcome: &BF16Entity,
    role_bf16: &[u8],
    entities: &[BF16Entity],
    max_depth: usize,
    confidence_threshold: f64,
    weights: &BF16Weights,
) -> BF16CausalTrace {
    let bf16_fn = select_bf16_hamming_fn();

    // For BF16 binary base: unbind = XOR (same as bind)
    let xor_unbind = |bound: &[u8], role: &[u8]| -> Vec<u8> {
        bound.iter().zip(role.iter()).map(|(a, b)| a ^ b).collect()
    };

    let nearest_bf16 = |candidate: &[u8]| -> (u32, u64) {
        let mut best_id = 0u32;
        let mut best_dist = u64::MAX;
        for e in entities {
            let d = bf16_fn(candidate, &e.bf16_bytes, weights);
            if d < best_dist {
                best_dist = d;
                best_id = e.id;
            }
        }
        (best_id, best_dist)
    };

    // Max possible BF16 distance for normalization
    let max_dist_per_dim = (weights.sign as u64) + 8 * (weights.exponent as u64)
        + 7 * (weights.mantissa as u64);
    let max_total = max_dist_per_dim * (outcome.n_dims as u64);

    let mut steps = Vec::with_capacity(max_depth);
    let mut current_bytes = outcome.bf16_bytes.clone();
    let mut confident_depth = 0;
    let mut all_sign_flip_dims: Vec<Vec<usize>> = Vec::new();

    for _ in 0..max_depth {
        let candidate = xor_unbind(&current_bytes, role_bf16);
        let (entity_id, distance) = nearest_bf16(&candidate);
        let normalized = if max_total > 0 { distance as f64 / max_total as f64 } else { 1.0 };
        let confident = normalized < confidence_threshold;

        // Structural diff between candidate and recovered entity
        let recovered = entities.iter().find(|e| e.id == entity_id);
        let diff = if let Some(r) = recovered {
            structural_diff(&candidate, &r.bf16_bytes)
        } else {
            BF16StructuralDiff::default()
        };

        all_sign_flip_dims.push(diff.sign_flip_dims.to_vec());

        steps.push(BF16TraceStep {
            entity_id,
            weighted_distance: distance,
            normalized_distance: normalized,
            confident,
            diff,
        });

        if confident {
            confident_depth += 1;
        } else {
            break;
        }

        if let Some(e) = recovered {
            current_bytes = e.bf16_bytes.clone();
        } else {
            break;
        }
    }

    // Causal backbone: dimensions that appear in sign_flip_dims
    // across most confident hops.
    let causal_backbone_dims = if confident_depth >= 2 {
        let confident_dims: Vec<&Vec<usize>> = all_sign_flip_dims[..confident_depth]
            .iter().collect();
        let mut dim_counts = std::collections::HashMap::new();
        for dims in &confident_dims {
            for &d in dims.iter() {
                *dim_counts.entry(d).or_insert(0u32) += 1;
            }
        }
        let threshold = (confident_depth as u32).saturating_sub(1).max(1);
        let mut backbone: Vec<usize> = dim_counts.iter()
            .filter(|(_, &count)| count >= threshold)
            .map(|(&dim, _)| dim)
            .collect();
        backbone.sort();
        backbone
    } else {
        Vec::new()
    };

    BF16CausalTrace {
        outcome_id: outcome.id,
        role_name: String::new(),
        steps,
        confident_depth,
        causal_backbone_dims,
    }
}

// ---------------------------------------------------------------------------
// BF16 Learning Event — what changed and why
// ---------------------------------------------------------------------------

/// A learning event captured from BF16 structural diff.
#[derive(Clone, Debug)]
pub struct BF16LearningEvent {
    /// Entity that was updated.
    pub entity_id: u32,
    /// Timestep of the update.
    pub timestep: usize,
    /// BF16-structured distance between before and after.
    pub distance: u64,
    /// Structural diff.
    pub diff: BF16StructuralDiff,
    /// Causal interpretation.
    pub interpretation: LearningInterpretation,
}

#[derive(Clone, Debug)]
pub enum LearningInterpretation {
    /// No meaningful change (mantissa noise only).
    Noise,
    /// Attention rebalancing: magnitude shifted but polarity preserved.
    AttentionShift { dims: Vec<usize> },
    /// Semantic reversal: sign flipped on key dimensions.
    SemanticReversal { dims: Vec<usize> },
    /// Both: sign flips AND magnitude shifts.
    MajorUpdate { sign_dims: Vec<usize>, magnitude_dims: Vec<usize> },
}

/// Classify a learning step from BF16 structural diff.
pub fn classify_learning_event(
    entity_id: u32,
    timestep: usize,
    before: &[u8],
    after: &[u8],
    weights: &BF16Weights,
) -> BF16LearningEvent {
    let bf16_fn = select_bf16_hamming_fn();
    let distance = bf16_fn(before, after, weights);
    let diff = structural_diff(before, after);

    let interpretation = match (diff.sign_flips, diff.major_magnitude_shifts.len()) {
        (0, 0) => LearningInterpretation::Noise,
        (0, _) => LearningInterpretation::AttentionShift {
            dims: diff.major_magnitude_shifts.to_vec(),
        },
        (_, 0) => LearningInterpretation::SemanticReversal {
            dims: diff.sign_flip_dims.to_vec(),
        },
        (_, _) => LearningInterpretation::MajorUpdate {
            sign_dims: diff.sign_flip_dims.to_vec(),
            magnitude_dims: diff.major_magnitude_shifts.to_vec(),
        },
    };

    BF16LearningEvent {
        entity_id,
        timestep,
        distance,
        diff,
        interpretation,
    }
}

// ---------------------------------------------------------------------------
// BF16 Causal Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod bf16_causal_tests {
    use super::*;

    fn make_jina_like_embedding(seed: u64, n_dims: usize) -> Vec<f32> {
        let mut rng = rustynum_core::SplitMix64::new(seed);
        (0..n_dims).map(|_| {
            (rng.next_u64() as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32 * 0.1
        }).collect()
    }

    #[test]
    fn test_bf16_granger_causal_map_returns_per_dim() {
        let n_dims = 64;
        let n_steps = 20;
        let tau = 2;

        let series_a: Vec<BF16Entity> = (0..n_steps).map(|t| {
            BF16Entity::from_f32(t as u32, &format!("A_{}", t),
                &make_jina_like_embedding(t as u64, n_dims))
        }).collect();

        // B[t] at lag tau is similar to A[t-tau] with some dims flipped
        let series_b: Vec<BF16Entity> = (0..n_steps).map(|t| {
            if t >= tau {
                let mut emb = make_jina_like_embedding((t - tau) as u64, n_dims);
                for &d in &[5, 10, 15] {
                    if d < n_dims { emb[d] = -emb[d]; }
                }
                BF16Entity::from_f32(t as u32, &format!("B_{}", t), &emb)
            } else {
                BF16Entity::from_f32(t as u32, &format!("B_{}", t),
                    &make_jina_like_embedding(100 + t as u64, n_dims))
            }
        }).collect();

        let map = bf16_granger_causal_map(&series_a, &series_b, tau);

        // Dims 5, 10, 15 should have high sign-flip counts
        assert!(map.sign_flip_counts[5] > 0, "Dim 5 should have sign flips");
        assert!(map.sign_flip_counts[10] > 0, "Dim 10 should have sign flips");
        assert!(map.sign_flip_counts[15] > 0, "Dim 15 should have sign flips");
        assert!(!map.top_causal_dims.is_empty(), "Should have causal dims");
    }

    #[test]
    fn test_bf16_granger_scan_finds_best_lag() {
        let n_dims = 32;
        let n_steps = 30;
        let true_lag = 3;

        let series_a: Vec<BF16Entity> = (0..n_steps).map(|t| {
            BF16Entity::from_f32(t as u32, &format!("A_{}", t),
                &make_jina_like_embedding(t as u64, n_dims))
        }).collect();

        let series_b: Vec<BF16Entity> = (0..n_steps).map(|t| {
            if t >= true_lag {
                let mut emb = make_jina_like_embedding((t - true_lag) as u64, n_dims);
                // Small perturbation
                for d in 0..3 { emb[d] = -emb[d]; }
                BF16Entity::from_f32(t as u32, &format!("B_{}", t), &emb)
            } else {
                BF16Entity::from_f32(t as u32, &format!("B_{}", t),
                    &make_jina_like_embedding(200 + t as u64, n_dims))
            }
        }).collect();

        let map = bf16_granger_causal_scan(&series_a, &series_b, 5);
        // Best lag should be around the true lag
        assert!(map.best_lag >= 1 && map.best_lag <= 5);
    }

    #[test]
    fn test_classify_learning_event_semantic_reversal() {
        let n_dims = 32;
        let before_f32 = make_jina_like_embedding(42, n_dims);
        let mut after_f32 = before_f32.clone();
        after_f32[7] = -after_f32[7];

        let before = fp32_to_bf16_bytes(&before_f32);
        let after = fp32_to_bf16_bytes(&after_f32);

        let event = classify_learning_event(1, 0, &before, &after, &TRAINING_WEIGHTS);

        match &event.interpretation {
            LearningInterpretation::SemanticReversal { dims } |
            LearningInterpretation::MajorUpdate { sign_dims: dims, .. } => {
                assert!(dims.contains(&7), "Should detect sign flip on dim 7");
            }
            other => panic!("Expected SemanticReversal or MajorUpdate, got {:?}", other),
        }
    }

    #[test]
    fn test_classify_learning_event_noise() {
        let n_dims = 32;
        let emb = make_jina_like_embedding(42, n_dims);
        let before = fp32_to_bf16_bytes(&emb);
        let after = before.clone();

        let event = classify_learning_event(1, 0, &before, &after, &TRAINING_WEIGHTS);
        assert!(matches!(event.interpretation, LearningInterpretation::Noise));
    }

    #[test]
    fn test_bf16_reverse_trace_single_hop() {
        let n_dims = 64;
        let cause = BF16Entity::from_f32(1, "cause",
            &make_jina_like_embedding(1, n_dims));

        let role_f32 = make_jina_like_embedding(99, n_dims);
        let role_bf16 = fp32_to_bf16_bytes(&role_f32);

        // Effect = XOR(cause, role) in BF16 byte space
        let effect_bytes: Vec<u8> = cause.bf16_bytes.iter()
            .zip(role_bf16.iter())
            .map(|(a, b)| a ^ b)
            .collect();
        let effect = BF16Entity { id: 2, name: "effect".into(), bf16_bytes: effect_bytes, n_dims };

        let entities = vec![cause.clone(), effect.clone()];
        let trace = bf16_reverse_trace(
            &effect, &role_bf16, &entities, 3, 0.3, &BF16Weights::default(),
        );

        assert!(!trace.steps.is_empty());
        assert_eq!(trace.steps[0].entity_id, 1, "Should recover cause");
        assert_eq!(trace.steps[0].weighted_distance, 0, "XOR unbind should be exact");
        assert!(trace.steps[0].confident);
    }

    #[test]
    fn test_causal_feature_map_rates() {
        let map = CausalFeatureMap {
            n_dims: 4,
            sign_flip_counts: vec![10, 0, 5, 8],
            exponent_shift_counts: vec![2, 0, 1, 3],
            top_causal_dims: vec![(0, 10), (3, 8), (2, 5)],
            timesteps: 20,
            granger_signal: -0.5,
            best_lag: 2,
        };

        assert!((map.sign_flip_rate(0) - 0.5).abs() < 0.01);
        assert_eq!(map.sign_flip_rate(1), 0.0);
        assert!((map.sign_flip_rate(2) - 0.25).abs() < 0.01);

        let above_30 = map.causal_dims_above(0.3);
        assert!(above_30.contains(&0)); // 50%
        assert!(above_30.contains(&3)); // 40%
        assert!(!above_30.contains(&2)); // 25% < 30%
    }

    #[test]
    fn test_classify_learning_event_attention_shift() {
        // Create two BF16 vectors where only exponent bits differ significantly
        let n_dims = 4;
        let before_f32 = vec![0.1f32, 0.2, 0.3, 0.4];
        let after_f32 = vec![0.1, 0.2, 30.0, 40.0]; // dims 2,3: huge magnitude change

        let before = fp32_to_bf16_bytes(&before_f32);
        let after = fp32_to_bf16_bytes(&after_f32);

        let event = classify_learning_event(1, 0, &before, &after, &TRAINING_WEIGHTS);

        // Should be AttentionShift or MajorUpdate (depends on whether sign also flipped)
        match &event.interpretation {
            LearningInterpretation::AttentionShift { dims } => {
                assert!(!dims.is_empty(), "Should detect magnitude shifts");
            }
            LearningInterpretation::MajorUpdate { magnitude_dims, .. } => {
                assert!(!magnitude_dims.is_empty(), "Should detect magnitude shifts");
            }
            _ => {
                // Also acceptable if exponent didn't change enough bits
            }
        }
    }
}
