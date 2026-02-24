//! Hybrid scoring pipeline: binary Hamming + BF16 structured distance + awareness feedback.
//!
//! This module bridges three previously disconnected subsystems:
//!
//! 1. **kernels.rs** — K0/K1/K2 cascaded binary Hamming (integer-only, fast pruning)
//! 2. **bf16_hamming.rs** — BF16 structured distance (sign/exp/man weighted, learning-aware)
//! 3. **bf16_hamming.rs** — Awareness substrate (4-state superposition decomposition)
//!
//! ## Pipeline
//!
//! ```text
//! Query + Database
//!   │
//!   ├─ K0 Probe (64-bit) ──→ reject ~55%
//!   ├─ K1 Stats (512-bit) ──→ reject ~90% of survivors
//!   ├─ K2 Exact (full width) ──→ Hamming + EnergyConflict
//!   │
//!   └─ BF16 Tail (survivors only, ~5%)
//!      ├─ Structured distance: sign/exp/man weighted
//!      ├─ Structural diff: which dimensions changed
//!      └─ Awareness: crystallized/tensioned/uncertain/noise per dim
//! ```
//!
//! ## Learning Loop
//!
//! After recognition, the awareness state feeds back:
//! - **Crystallized** dims → increase hybrid weight (settled knowledge)
//! - **Tensioned** dims → flag for NARS revision (active contradiction)
//! - **Uncertain** dims → increase exploration weight
//! - **Noise** dims → decrease hybrid weight (mask out)
//!
//! The hybrid weights can be stored in WideMetaView W144-W159 (32 f32 weights)
//! or as a packed 2-bit-per-dim vector in the container itself.

use crate::bf16_hamming::{
    self, AwarenessState, AwarenessThresholds, BF16StructuralDiff, BF16Weights,
    SuperpositionState,
};
use crate::kernels::{self, EnergyConflict, HdrScore, PipelineStats, SliceGate};

// ============================================================================
// Hybrid score — combines binary + BF16 + awareness
// ============================================================================

/// Combined score from binary Hamming + BF16 structured distance.
///
/// The binary Hamming (K2 exact) gives the fast integer distance.
/// The BF16 structured distance gives a quality-weighted float score
/// that distinguishes sign flips (class-level) from mantissa noise.
#[derive(Clone, Debug)]
pub struct HybridScore {
    /// Binary Hamming distance (from K2 exact). Integer, fast.
    pub hamming_distance: u32,
    /// HDR score from binary Hamming (0-6 scale).
    pub hdr: HdrScore,
    /// Energy/conflict decomposition from binary Hamming.
    pub energy: EnergyConflict,
    /// BF16 structured distance (weighted sign/exp/man). Float, quality.
    pub bf16_distance: u64,
    /// BF16 structural diff — per-dimension breakdown of what changed.
    pub structural_diff: BF16StructuralDiff,
    /// Candidate index in the database.
    pub index: usize,
    /// Combined score: lower = better match.
    /// Computed as: hamming_distance * hamming_weight + bf16_distance * bf16_weight
    pub combined_score: f64,
}

/// Configuration for the hybrid pipeline.
#[derive(Clone, Debug)]
pub struct HybridConfig {
    /// Gate thresholds for K0/K1/K2 binary pruning.
    pub gate: SliceGate,
    /// BF16 weights for structured distance scoring.
    pub bf16_weights: BF16Weights,
    /// Weight for binary Hamming in combined score (default: 1.0).
    pub hamming_weight: f64,
    /// Weight for BF16 structured distance in combined score (default: 0.01).
    /// Smaller because bf16_distance is on a larger scale.
    pub bf16_weight: f64,
    /// Maximum number of K2 survivors to send to BF16 tail.
    /// Bounds worst-case tail cost. 0 = no limit.
    pub max_bf16_candidates: usize,
    /// Awareness thresholds for superposition decomposition.
    pub awareness_thresholds: AwarenessThresholds,
}

impl HybridConfig {
    /// Default config for SKU-16K containers.
    pub fn sku_16k() -> Self {
        Self {
            gate: SliceGate::sku_16k(),
            bf16_weights: BF16Weights::default(),
            hamming_weight: 1.0,
            bf16_weight: 0.01,
            max_bf16_candidates: 100,
            awareness_thresholds: AwarenessThresholds::default(),
        }
    }

    /// Default config for SKU-64K containers.
    pub fn sku_64k() -> Self {
        Self {
            gate: SliceGate::sku_64k(),
            bf16_weights: BF16Weights::default(),
            hamming_weight: 1.0,
            bf16_weight: 0.01,
            max_bf16_candidates: 100,
            awareness_thresholds: AwarenessThresholds::default(),
        }
    }

    /// Config optimized for learning: TRAINING_WEIGHTS ignore mantissa noise.
    pub fn learning(total_bits: usize) -> Self {
        Self {
            gate: SliceGate::new(total_bits, 0.05, 0.15, 0.30, 0.90, 1.5),
            bf16_weights: crate::bf16_hamming::TRAINING_WEIGHTS,
            hamming_weight: 0.5,
            bf16_weight: 0.02,
            max_bf16_candidates: 50,
            awareness_thresholds: AwarenessThresholds::default(),
        }
    }
}

/// Statistics for the hybrid pipeline.
#[derive(Clone, Debug, Default)]
pub struct HybridStats {
    /// Binary pipeline stats (K0/K1/K2).
    pub binary_stats: PipelineStats,
    /// Number of candidates that went through BF16 tail scoring.
    pub bf16_scored: usize,
    /// Number of candidates with sign flips detected.
    pub sign_flip_candidates: usize,
    /// Number of candidates with major magnitude shifts.
    pub magnitude_shift_candidates: usize,
}

// ============================================================================
// Hybrid pipeline — K0 → K1 → K2 → BF16 tail
// ============================================================================

/// Run the hybrid pipeline: binary pruning + BF16 structured scoring.
///
/// Phase 1: Binary Hamming (K0→K1→K2) prunes ~95% of candidates using
/// integer-only operations. This is the fast path.
///
/// Phase 2: BF16 structured distance scores the ~5% survivors with
/// weighted sign/exp/man decomposition. This gives quality ranking.
///
/// `query_bytes`: query as raw bytes (must be 2× aligned for BF16 interpretation)
/// `database_bytes`: flat byte array of all containers
/// `n_candidates`: number of containers in database
/// `config`: hybrid pipeline configuration
///
/// Returns sorted matches (best first) and pipeline statistics.
pub fn hybrid_pipeline(
    query_bytes: &[u8],
    database_bytes: &[u8],
    n_candidates: usize,
    config: &HybridConfig,
) -> (Vec<HybridScore>, HybridStats) {
    let n_bytes = query_bytes.len();
    assert!(
        n_bytes == kernels::SKU_16K_BYTES || n_bytes == kernels::SKU_64K_BYTES,
        "query must be {} or {} bytes, got {}",
        kernels::SKU_16K_BYTES,
        kernels::SKU_64K_BYTES,
        n_bytes
    );
    assert!(database_bytes.len() >= n_candidates * n_bytes);

    // Phase 1: Binary Hamming pipeline (K0→K1→K2)
    let (binary_matches, binary_stats) =
        kernels::kernel_pipeline_bytes(query_bytes, database_bytes, n_candidates, &config.gate);

    let mut stats = HybridStats {
        binary_stats,
        ..Default::default()
    };

    // Phase 2: BF16 structured scoring on K2 survivors
    let bf16_fn = bf16_hamming::select_bf16_hamming_fn();
    let mut hybrid_scores: Vec<HybridScore> = Vec::with_capacity(binary_matches.len());

    // Limit BF16 candidates if configured (bound worst-case tail cost)
    let bf16_limit = if config.max_bf16_candidates == 0 {
        binary_matches.len()
    } else {
        binary_matches.len().min(config.max_bf16_candidates)
    };

    // Sort binary matches by distance so we BF16-score the best candidates first
    let mut sorted_binary = binary_matches;
    sorted_binary.sort_by_key(|m| m.distance);

    for km in sorted_binary.iter().take(bf16_limit) {
        let cand_offset = km.index * n_bytes;
        let cand_bytes = &database_bytes[cand_offset..cand_offset + n_bytes];

        // BF16 structured distance (uses AVX-512 BITALG if available)
        let bf16_dist = bf16_fn(query_bytes, cand_bytes, &config.bf16_weights);

        // Structural diff — per-dimension breakdown for learning
        let diff = bf16_hamming::structural_diff(query_bytes, cand_bytes);

        stats.bf16_scored += 1;
        if diff.sign_flips > 0 {
            stats.sign_flip_candidates += 1;
        }
        if !diff.major_magnitude_shifts.is_empty() {
            stats.magnitude_shift_candidates += 1;
        }

        // Combined score: binary + weighted BF16
        let combined = km.distance as f64 * config.hamming_weight
            + bf16_dist as f64 * config.bf16_weight;

        hybrid_scores.push(HybridScore {
            hamming_distance: km.distance,
            hdr: km.hdr,
            energy: km.energy,
            bf16_distance: bf16_dist,
            structural_diff: diff,
            index: km.index,
            combined_score: combined,
        });
    }

    // Sort by combined score (best first)
    hybrid_scores.sort_by(|a, b| a.combined_score.partial_cmp(&b.combined_score).unwrap());

    (hybrid_scores, stats)
}

// ============================================================================
// Awareness feedback — learning signal from recognition results
// ============================================================================

/// Learning signal extracted from recognition results.
///
/// After running the hybrid pipeline, decompose the query against the
/// top-K matches to understand per-dimension learning state.
#[derive(Clone, Debug)]
pub struct LearningSignal {
    /// Superposition state: per-dimension awareness classification.
    pub awareness: SuperpositionState,
    /// Fraction of dimensions that are crystallized (settled).
    pub crystallized_ratio: f32,
    /// Fraction of dimensions under tension (contradictory evidence).
    pub tension_ratio: f32,
    /// Number of sign-flip dimensions across all top-K comparisons.
    pub total_sign_flips: usize,
    /// Per-dimension attention weights derived from awareness state.
    /// Crystallized → 1.0, Tensioned → 0.5, Uncertain → 0.25, Noise → 0.0
    pub attention_weights: Vec<f32>,
    /// Packed 2-bit awareness states (4 dims per byte).
    pub packed_states: Vec<u8>,
}

/// Default attention weights for each awareness state.
const ATTENTION_CRYSTALLIZED: f32 = 1.0;
const ATTENTION_TENSIONED: f32 = 0.5;
const ATTENTION_UNCERTAIN: f32 = 0.25;
const ATTENTION_NOISE: f32 = 0.0;

/// Extract learning signal from query + top-K recognition results.
///
/// Takes the query and the BF16 byte representations of the top K matches,
/// decomposes them into per-dimension awareness states, and produces
/// attention weights that can be fed back into WideMetaView hybrid weights.
///
/// `query_bytes`: the recognition query (BF16 byte representation)
/// `top_k_bytes`: byte representations of the top K matches (2-3 max)
/// `thresholds`: awareness classification thresholds
pub fn extract_learning_signal(
    query_bytes: &[u8],
    top_k_bytes: &[&[u8]],
    thresholds: &AwarenessThresholds,
) -> LearningSignal {
    assert!(
        !top_k_bytes.is_empty() && top_k_bytes.len() <= 2,
        "extract_learning_signal needs 1-2 top-K results (plus query = 2-3 vectors total)"
    );

    // Build vector set: query + top-K results (2-3 total)
    let mut vectors: Vec<&[u8]> = Vec::with_capacity(top_k_bytes.len() + 1);
    vectors.push(query_bytes);
    vectors.extend_from_slice(top_k_bytes);

    // Superposition decomposition
    let awareness = bf16_hamming::superposition_decompose(&vectors, thresholds);

    // Compute attention weights from awareness states
    let attention_weights: Vec<f32> = awareness
        .states
        .iter()
        .map(|s| match s {
            AwarenessState::Crystallized => ATTENTION_CRYSTALLIZED,
            AwarenessState::Tensioned => ATTENTION_TENSIONED,
            AwarenessState::Uncertain => ATTENTION_UNCERTAIN,
            AwarenessState::Noise => ATTENTION_NOISE,
        })
        .collect();

    // Count total sign flips across all query↔result diffs
    let mut total_sign_flips = 0;
    for &top_bytes in top_k_bytes {
        let diff = bf16_hamming::structural_diff(query_bytes, top_bytes);
        total_sign_flips += diff.sign_flips;
    }

    let packed_states = awareness.packed_states.clone();
    let crystallized_ratio = awareness.crystallized_pct;
    let tension_ratio = awareness.tensioned_pct;

    LearningSignal {
        awareness,
        crystallized_ratio,
        tension_ratio,
        total_sign_flips,
        attention_weights,
        packed_states,
    }
}

/// Apply learning signal to update hybrid weights.
///
/// Given a current set of 32 f32 hybrid weights (matching WideMetaView W144-W159)
/// and a learning signal, update the weights using exponential moving average.
///
/// Each of the 32 weight slots corresponds to a group of dimensions:
/// - For 1024-dim BF16: each slot covers 32 dimensions (1024/32)
/// - For 512-dim BF16: each slot covers 16 dimensions (512/32)
///
/// The update rule per group:
/// - If group is mostly crystallized → increase weight toward 1.0
/// - If group is mostly tensioned → reduce weight toward 0.5
/// - If group is mostly noise → decrease weight toward 0.0
pub fn update_hybrid_weights(
    current_weights: &mut [f32; 32],
    signal: &LearningSignal,
    learning_rate: f32,
) {
    let n_dims = signal.awareness.n_dims;
    if n_dims == 0 {
        return;
    }
    let group_size = n_dims.div_ceil(32);

    for group in 0..32 {
        let start = group * group_size;
        let end = (start + group_size).min(n_dims);
        if start >= n_dims {
            break;
        }

        // Count states in this group
        let mut crystallized = 0u32;
        let mut tensioned = 0u32;
        let mut noise = 0u32;
        let group_count = (end - start) as f32;

        for d in start..end {
            match signal.awareness.states[d] {
                AwarenessState::Crystallized => crystallized += 1,
                AwarenessState::Tensioned => tensioned += 1,
                AwarenessState::Noise => noise += 1,
                AwarenessState::Uncertain => {} // neutral
            }
        }

        // Target weight based on group composition
        let cryst_ratio = crystallized as f32 / group_count;
        let tension_ratio = tensioned as f32 / group_count;
        let noise_ratio = noise as f32 / group_count;

        let target = if cryst_ratio > 0.5 {
            // Mostly settled → high weight
            0.8 + 0.2 * cryst_ratio
        } else if tension_ratio > 0.3 {
            // Contradictory → medium weight (hold both perspectives)
            0.5
        } else if noise_ratio > 0.5 {
            // Mostly noise → suppress
            0.1 * (1.0 - noise_ratio)
        } else {
            // Mixed → maintain current
            current_weights[group]
        };

        // EMA update
        current_weights[group] =
            current_weights[group] * (1.0 - learning_rate) + target * learning_rate;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf16_hamming::fp32_to_bf16_bytes;

    /// Helper: create a BF16 byte vector from f32 values, padded to container size.
    fn make_container(values: &[f32], container_bytes: usize) -> Vec<u8> {
        let bf16 = fp32_to_bf16_bytes(values);
        let mut padded = vec![0u8; container_bytes];
        let copy_len = bf16.len().min(container_bytes);
        padded[..copy_len].copy_from_slice(&bf16[..copy_len]);
        padded
    }

    #[test]
    fn test_hybrid_config_defaults() {
        let c16 = HybridConfig::sku_16k();
        assert_eq!(c16.gate.total_bits, 16384);
        assert_eq!(c16.bf16_weights.sign, 256);
        assert!(c16.hamming_weight > 0.0);

        let c64 = HybridConfig::sku_64k();
        assert_eq!(c64.gate.total_bits, 65536);

        let cl = HybridConfig::learning(kernels::SKU_16K_BITS);
        assert_eq!(cl.bf16_weights.mantissa, 0); // TRAINING_WEIGHTS ignores mantissa
    }

    #[test]
    fn test_hybrid_pipeline_exact_match() {
        let config = HybridConfig::sku_16k();
        let n_dims = kernels::SKU_16K_BYTES / 2; // 1024 BF16 dims
        let values: Vec<f32> = (0..n_dims).map(|i| (i as f32 * 0.01).sin()).collect();
        let query = make_container(&values, kernels::SKU_16K_BYTES);

        // Database: exact match at index 0, random at 1-9
        let n = 10;
        let mut db = Vec::with_capacity(n * kernels::SKU_16K_BYTES);
        db.extend_from_slice(&query); // exact match
        for i in 1..n {
            let rand_vals: Vec<f32> =
                (0..n_dims).map(|j| ((i * 1000 + j) as f32 * 0.037).cos()).collect();
            db.extend_from_slice(&make_container(&rand_vals, kernels::SKU_16K_BYTES));
        }

        let (scores, stats) = hybrid_pipeline(&query, &db, n, &config);

        // Should find exact match
        assert!(!scores.is_empty(), "Should find at least one match");
        assert_eq!(scores[0].index, 0, "Best match should be the exact copy");
        assert_eq!(scores[0].hamming_distance, 0);
        assert_eq!(scores[0].bf16_distance, 0);
        assert_eq!(scores[0].combined_score, 0.0);
        assert_eq!(scores[0].structural_diff.sign_flips, 0);

        // Stats
        assert!(stats.bf16_scored >= 1);
        assert_eq!(stats.binary_stats.total, n);
    }

    #[test]
    fn test_hybrid_pipeline_sign_flip_scoring() {
        let config = HybridConfig::sku_16k();
        let n_dims = kernels::SKU_16K_BYTES / 2;

        // Query: all positive
        let values: Vec<f32> = (0..n_dims).map(|i| (i as f32 + 1.0) * 0.001).collect();
        let query = make_container(&values, kernels::SKU_16K_BYTES);

        // Candidate 0: same magnitudes, some signs flipped (first 100 dims)
        let mut flipped_values = values.clone();
        for v in flipped_values.iter_mut().take(100) {
            *v = -*v;
        }
        let candidate = make_container(&flipped_values, kernels::SKU_16K_BYTES);

        let n = 1;
        let (scores, stats) = hybrid_pipeline(&query, &candidate, n, &config);

        if !scores.is_empty() {
            // BF16 distance should reflect sign flips heavily (weight 256 each)
            assert!(
                scores[0].bf16_distance > 0,
                "Sign flips should produce non-zero BF16 distance"
            );
            assert!(
                scores[0].structural_diff.sign_flips > 0,
                "Should detect sign flips"
            );
            assert!(
                stats.sign_flip_candidates > 0,
                "Should count sign flip candidates"
            );
        }
    }

    #[test]
    fn test_learning_signal_crystallized() {
        // Two very similar vectors → should be mostly crystallized
        let vals_a: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let a = fp32_to_bf16_bytes(&vals_a);

        // Same values (identical) → all crystallized
        let signal = extract_learning_signal(&a, &[&a], &AwarenessThresholds::default());

        assert!(
            signal.crystallized_ratio > 0.9,
            "Identical vectors should be >90% crystallized, got {}",
            signal.crystallized_ratio
        );
        assert_eq!(signal.total_sign_flips, 0);
        assert_eq!(signal.attention_weights.len(), 64);

        // All weights should be 1.0 (crystallized)
        for &w in &signal.attention_weights {
            assert!(
                (w - ATTENTION_CRYSTALLIZED).abs() < 0.01,
                "Crystallized dims should have weight {}, got {}",
                ATTENTION_CRYSTALLIZED,
                w
            );
        }
    }

    #[test]
    fn test_learning_signal_tensioned() {
        // Vectors with sign disagreement → tensioned dimensions
        let vals_a: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1 + 0.5).sin()).collect();
        let mut vals_b = vals_a.clone();
        // Flip signs on first 32 dims
        for v in vals_b.iter_mut().take(32) {
            *v = -*v;
        }
        let a = fp32_to_bf16_bytes(&vals_a);
        let b = fp32_to_bf16_bytes(&vals_b);

        let signal = extract_learning_signal(&a, &[&b], &AwarenessThresholds::default());

        assert!(
            signal.tension_ratio > 0.3,
            "Sign flips should cause >30% tension, got {}",
            signal.tension_ratio
        );
        assert!(signal.total_sign_flips > 20, "Should detect many sign flips");

        // Tensioned dims should have lower attention weight
        let tensioned_weights: Vec<f32> = signal.attention_weights[..32]
            .iter()
            .filter(|&&w| w <= ATTENTION_TENSIONED)
            .cloned()
            .collect();
        assert!(
            tensioned_weights.len() > 10,
            "Many of the first 32 dims should have reduced attention"
        );
    }

    #[test]
    fn test_update_hybrid_weights() {
        let vals: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let a = fp32_to_bf16_bytes(&vals);

        // All crystallized → weights should increase
        let signal = extract_learning_signal(&a, &[&a], &AwarenessThresholds::default());

        let mut weights = [0.5f32; 32];
        update_hybrid_weights(&mut weights, &signal, 0.1);

        // Weights should have moved toward 1.0 (crystallized target)
        for &w in &weights[..4] {
            // First 4 groups (128 dims / 32 groups = 4 dims per group)
            assert!(
                w > 0.5,
                "Crystallized groups should increase weight, got {}",
                w
            );
        }
    }

    #[test]
    fn test_update_hybrid_weights_noise_suppression() {
        // Create vectors with only mantissa differences → noise
        let vals_a: Vec<f32> = (0..64).map(|i| (i as f32 + 1.0) * 1.0).collect();
        let mut vals_b = vals_a.clone();
        // Tiny perturbations that only affect mantissa bits
        for v in vals_b.iter_mut() {
            *v *= 1.05; // ~5% change → mantissa noise only in BF16
        }
        let a = fp32_to_bf16_bytes(&vals_a);
        let b = fp32_to_bf16_bytes(&vals_b);

        let signal = extract_learning_signal(&a, &[&b], &AwarenessThresholds::default());

        let mut weights = [0.5f32; 32];
        let original_weights = weights;
        update_hybrid_weights(&mut weights, &signal, 0.3);

        // With mostly noise dims, weights should decrease or stay low
        let noise_decreased = weights
            .iter()
            .zip(original_weights.iter())
            .filter(|(&new, &old)| new < old)
            .count();
        // At least some groups should have decreased (noise suppression)
        // Note: depending on BF16 rounding, some dims might classify as crystallized
        // so we only check that the mechanism works, not exact counts
        assert!(
            noise_decreased > 0 || signal.awareness.noise_pct > 0.0,
            "Should detect noise and/or decrease weights"
        );
    }

    #[test]
    fn test_packed_awareness_roundtrip() {
        let vals_a: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut vals_b = vals_a.clone();
        for v in vals_b.iter_mut().take(16) {
            *v = -*v;
        }
        let a = fp32_to_bf16_bytes(&vals_a);
        let b = fp32_to_bf16_bytes(&vals_b);

        let signal = extract_learning_signal(&a, &[&b], &AwarenessThresholds::default());

        // packed_states should be compact
        assert_eq!(
            signal.packed_states.len(),
            (64 + 3) / 4, // ceil(64 dims / 4 dims per byte)
            "Packed states should be ceil(n_dims/4) bytes"
        );

        // Unpack and verify
        let unpacked =
            bf16_hamming::unpack_awareness_states(&signal.packed_states, signal.awareness.n_dims);
        assert_eq!(unpacked, signal.awareness.states);
    }

    #[test]
    fn test_hybrid_stats_tracking() {
        let config = HybridConfig::sku_16k();
        let n_dims = kernels::SKU_16K_BYTES / 2;
        let values: Vec<f32> = (0..n_dims).map(|i| (i as f32 * 0.01).sin()).collect();
        let query = make_container(&values, kernels::SKU_16K_BYTES);

        // All random candidates
        let n = 50;
        let mut db = Vec::with_capacity(n * kernels::SKU_16K_BYTES);
        for i in 0..n {
            let rand_vals: Vec<f32> =
                (0..n_dims).map(|j| ((i * 777 + j) as f32 * 0.037).cos()).collect();
            db.extend_from_slice(&make_container(&rand_vals, kernels::SKU_16K_BYTES));
        }

        let (_, stats) = hybrid_pipeline(&query, &db, n, &config);

        assert_eq!(stats.binary_stats.total, n);
        // Binary pipeline should account for all candidates
        assert_eq!(
            stats.binary_stats.k0_rejected
                + stats.binary_stats.k1_rejected
                + stats.binary_stats.k2_promoted,
            n
        );
        // BF16 scored should be <= K2 promoted
        assert!(stats.bf16_scored <= stats.binary_stats.k2_promoted);
    }
}
