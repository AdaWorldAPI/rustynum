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
            let neg_role: Vec<i8> = role.iter().map(|&v| v.saturating_neg()).collect();
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
