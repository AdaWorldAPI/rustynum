//! Tail backend trait — performance-portable scoring for K2 survivors.
//!
//! Follows the libCEED pattern: the orchestration layer (hybrid.rs) is pure Rust,
//! backends are isolated behind a trait. Unsafe and FFI live only inside backend
//! implementations. The trait boundary enforces:
//!
//! 1. **Pure functions** — no persistent state across calls
//! 2. **No mutation** — input slices are borrowed immutably
//! 3. **No leaked ownership** — results are owned structs, not references
//!
//! ```text
//! hybrid.rs (orchestration, safe Rust)
//!   │
//!   ├── TailBackend::score()        ← trait call, dispatched at runtime
//!   │     │
//!   │     ├── PopcntBackend         ← AVX-512 BITALG / scalar (bf16_hamming.rs)
//!   │     ├── XsmmBackend           ← LIBXSMM FFI (feature = "libxsmm")
//!   │     └── FallbackBackend       ← pure Rust, no SIMD
//!   │
//!   └── TailBackend::score_batch()  ← batch scoring for fused GEMM+reduction
//! ```
//!
//! ## Why only the tail?
//!
//! K0/K1/K2 are integer-only XOR+POPCNT — one implementation fits all.
//! The tail is where BF16 structured distance, GEMM, and awareness decomposition
//! happen. That's where backends diverge: BITALG vs AMX vs LIBXSMM vs GPU.

use crate::bf16_hamming::{AwarenessThresholds, BF16StructuralDiff, BF16Weights};

// ============================================================================
// Tail score — the output contract (owned, no references)
// ============================================================================

/// Result of tail scoring a single candidate.
///
/// This is the backend's return type — fully owned, no borrows.
/// The orchestration layer can store, sort, and combine these freely.
#[derive(Clone, Debug)]
pub struct TailScore {
    /// BF16 structured distance (weighted sign/exp/man).
    pub bf16_distance: u64,
    /// Structural diff — per-dimension breakdown.
    pub structural_diff: BF16StructuralDiff,
}

/// Result of batch tail scoring (multiple candidates at once).
///
/// Backends that support fused GEMM+reduction (e.g. LIBXSMM) return
/// distances for all candidates in one call, avoiding per-candidate overhead.
#[derive(Clone, Debug)]
pub struct BatchTailScore {
    /// Per-candidate BF16 distances, indexed by position in input slice.
    pub distances: Vec<u64>,
    /// Per-candidate structural diffs.
    pub diffs: Vec<BF16StructuralDiff>,
}

/// Compact tail score — distance + minimal counters, no structural diff.
///
/// For ranking-only consumers that don't need the full per-dimension breakdown.
/// Backends implement this directly (not by calling score_batch and discarding).
#[derive(Clone, Copy, Debug)]
pub struct CompactTailScore {
    /// BF16 structured distance (weighted sign/exp/man).
    pub bf16_distance: u64,
    /// Number of sign-flipped dimensions (class-level signal).
    pub sign_flips: u16,
    /// Number of exponent bits changed (magnitude signal).
    pub exponent_bits_changed: u16,
}

// ============================================================================
// TailBackend trait — the contract
// ============================================================================

/// Performance-portable tail scoring backend.
///
/// Implementations must be:
/// - **Stateless**: no mutable fields, no persistent borrows
/// - **Thread-safe**: `Send + Sync` (backends may be called from rayon)
/// - **Pure**: same inputs → same outputs, no side effects
///
/// The `&self` receiver exists only for dispatch — backends should hold
/// only configuration (weights, thresholds), never data references.
pub trait TailBackend: Send + Sync {
    /// Human-readable name for diagnostics and benchmarking.
    fn name(&self) -> &'static str;

    /// Score a single candidate against the query.
    ///
    /// `query_bytes` and `candidate_bytes` are BF16 byte slices (2 bytes/dim).
    /// Both must have the same length. No mutation, no retained references.
    fn score(
        &self,
        query_bytes: &[u8],
        candidate_bytes: &[u8],
        weights: &BF16Weights,
    ) -> TailScore;

    /// Score multiple candidates in one call.
    ///
    /// Default implementation calls `score()` in a loop.
    /// Backends with fused GEMM+reduction (LIBXSMM, AMX) override this
    /// to avoid per-candidate dispatch overhead.
    ///
    /// `candidate_slices`: contiguous byte array, each candidate is `query_bytes.len()` bytes.
    /// `n_candidates`: number of candidates in the batch.
    fn score_batch(
        &self,
        query_bytes: &[u8],
        candidate_slices: &[u8],
        n_candidates: usize,
        weights: &BF16Weights,
    ) -> BatchTailScore {
        let stride = query_bytes.len();
        let mut distances = Vec::with_capacity(n_candidates);
        let mut diffs = Vec::with_capacity(n_candidates);

        for i in 0..n_candidates {
            let offset = i * stride;
            let cand = &candidate_slices[offset..offset + stride];
            let s = self.score(query_bytes, cand, weights);
            distances.push(s.bf16_distance);
            diffs.push(s.structural_diff);
        }

        BatchTailScore { distances, diffs }
    }

    /// Compact batch scoring — distance + minimal counters only.
    ///
    /// Each backend implements this directly. Does NOT call `score_batch()`
    /// and discard fields — the compact path skips structural diff computation.
    ///
    /// `candidate_slices`: contiguous byte array, each candidate is `query_bytes.len()` bytes.
    /// `n_candidates`: number of candidates in the batch.
    fn score_batch_compact(
        &self,
        query_bytes: &[u8],
        candidate_slices: &[u8],
        n_candidates: usize,
        weights: &BF16Weights,
    ) -> Vec<CompactTailScore> {
        let stride = query_bytes.len();
        let mut results = Vec::with_capacity(n_candidates);
        for i in 0..n_candidates {
            let offset = i * stride;
            let cand = &candidate_slices[offset..offset + stride];
            results.push(self.score_compact(query_bytes, cand, weights));
        }
        results
    }

    /// Compact single scoring — distance + minimal counters only.
    ///
    /// Backends should override if they can skip work compared to full `score()`.
    fn score_compact(
        &self,
        query_bytes: &[u8],
        candidate_bytes: &[u8],
        weights: &BF16Weights,
    ) -> CompactTailScore {
        // Default: compute distance + count sign/exp flips without full structural_diff
        compact_score_from_bytes(query_bytes, candidate_bytes, weights)
    }

    /// Whether this backend supports efficient batch scoring.
    ///
    /// If true, callers should prefer `score_batch()` over repeated `score()`.
    /// Backends with fused GEMM (LIBXSMM) return true.
    fn supports_batch(&self) -> bool {
        false
    }

    /// Awareness decomposition for learning feedback.
    ///
    /// Default implementation uses `bf16_hamming::superposition_decompose`.
    /// Backends may override for fused scoring+awareness in one pass.
    fn awareness(
        &self,
        vectors: &[&[u8]],
        thresholds: &AwarenessThresholds,
    ) -> crate::bf16_hamming::SuperpositionState {
        crate::bf16_hamming::superposition_decompose(vectors, thresholds)
    }
}

// ============================================================================
// Auto-detection — runtime backend selection
// ============================================================================

/// Select the best available tail backend for this CPU.
///
/// Priority order:
/// 1. LIBXSMM (if feature enabled and library available)
/// 2. AVX-512 BITALG POPCNT backend (if CPU supports it)
/// 3. Pure Rust scalar fallback
///
/// The result is `Box<dyn TailBackend>` — one allocation at init time,
/// then zero-cost trait dispatch on every call.
///
/// Note: The GEMM backend is NOT auto-selected because it uses a different
/// distance metric (dot-product gap vs. weighted Hamming). Use `gemm_backend()`
/// to explicitly select it.
pub fn auto_detect() -> Box<dyn TailBackend> {
    // Priority 1: LIBXSMM (feature-gated)
    #[cfg(feature = "libxsmm")]
    {
        if let Some(backend) = crate::backends::xsmm::XsmmBackend::try_new() {
            return Box::new(backend);
        }
    }

    // Priority 2: AVX-512 BITALG (runtime CPUID check)
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512bitalg") {
            return Box::new(crate::backends::popcnt::PopcntBackend::avx512());
        }
    }

    // Priority 3: Scalar fallback (always available)
    Box::new(crate::backends::popcnt::PopcntBackend::scalar())
}

/// Create a GEMM-based tail backend for batch-optimized scoring.
///
/// Uses f32 dot product (via BF16→f32 conversion + SIMD) as the batch distance
/// metric. The distance is `(query_self_dot - dot(query, candidate)) * scale`,
/// which gives lower distance for more similar vectors.
///
/// Single-candidate scoring (`score()`) uses standard BF16 structured Hamming
/// distance for backward compatibility.
///
/// This backend reports `supports_batch() = true` and is optimized for cases
/// where many K2 survivors need scoring simultaneously.
pub fn gemm_backend() -> Box<dyn TailBackend> {
    Box::new(crate::backends::gemm::GemmBackend::new())
}

/// Create a GEMM-based tail backend with custom scale factor.
///
/// See `gemm_backend()` for details. The scale factor controls the
/// f32→u64 distance quantization resolution.
pub fn gemm_backend_with_scale(scale: f32) -> Box<dyn TailBackend> {
    Box::new(crate::backends::gemm::GemmBackend::with_scale(scale))
}

// ============================================================================
// Compact scoring helper — lightweight distance + counters, no full diff
// ============================================================================

/// Compute compact score directly from BF16 bytes.
///
/// Counts sign flips and exponent bit changes without building the full
/// BF16StructuralDiff (no per-dimension lists, no magnitude shift detection).
/// This is the "zero-allocation" path for ranking-only consumers.
pub fn compact_score_from_bytes(
    query_bytes: &[u8],
    candidate_bytes: &[u8],
    weights: &BF16Weights,
) -> CompactTailScore {
    assert_eq!(query_bytes.len(), candidate_bytes.len());
    assert!(query_bytes.len() % 2 == 0);

    let n_dims = query_bytes.len() / 2;
    let mut distance: u64 = 0;
    let mut sign_flips: u16 = 0;
    let mut exponent_bits: u16 = 0;

    for d in 0..n_dims {
        let i = d * 2;
        let va = u16::from_le_bytes([query_bytes[i], query_bytes[i + 1]]);
        let vb = u16::from_le_bytes([candidate_bytes[i], candidate_bytes[i + 1]]);
        let xor = va ^ vb;

        let sign = (xor >> 15) & 1;
        let exp_pop = ((xor >> 7) & 0xFF).count_ones() as u16;
        let man_pop = (xor & 0x7F).count_ones() as u16;

        distance += sign as u64 * weights.sign as u64
            + exp_pop as u64 * weights.exponent as u64
            + man_pop as u64 * weights.mantissa as u64;

        sign_flips += sign as u16;
        exponent_bits += exp_pop;
    }

    CompactTailScore {
        bf16_distance: distance,
        sign_flips,
        exponent_bits_changed: exponent_bits,
    }
}

// ============================================================================
// Capabilities — programmatic backend introspection
// ============================================================================

/// Runtime capabilities report for the tail scoring engine.
///
/// Deterministic, serializable, not coupled to logging.
/// Use `capabilities()` to get the current report.
#[derive(Clone, Debug)]
pub struct Capabilities {
    /// Selected backend name (e.g. "popcnt-avx512", "popcnt-scalar", "libxsmm").
    pub backend_name: String,
    /// Whether the libxsmm feature was enabled at compile time.
    pub libxsmm_compiled: bool,
    /// Whether the libxsmm backend is available at runtime.
    pub libxsmm_available: bool,
    /// Whether the libxsmm backend was selected (available ≠ selected).
    pub libxsmm_selected: bool,
    /// AVX-512 foundation detected at runtime.
    pub avx512f: bool,
    /// AVX-512 VPOPCNTDQ detected (fast Hamming).
    pub avx512_vpopcntdq: bool,
    /// AVX-512 BITALG detected (BF16 structured distance).
    pub avx512_bitalg: bool,
    /// AVX-512 VNNI detected (INT8 dot product).
    pub avx512_vnni: bool,
    /// AVX-512 BF16 detected (native BF16 dot product — distinct from BITALG).
    pub avx512_bf16: bool,
    /// AMX tile support detected.
    pub amx_tile: bool,
    /// AMX INT8 support detected.
    pub amx_int8: bool,
    /// AMX BF16 support detected.
    pub amx_bf16: bool,
    /// Whether batch scoring is efficient for the selected backend.
    pub supports_batch: bool,
}

/// Get capabilities report for the current runtime environment.
///
/// Probes CPU features and determines backend selection.
/// The report is deterministic for a given CPU + compile features.
pub fn capabilities() -> Capabilities {
    let backend = auto_detect();
    let name = backend.name().to_string();
    let supports_batch = backend.supports_batch();
    let libxsmm_selected = name == "libxsmm";

    #[cfg(target_arch = "x86_64")]
    let (avx512f, vpopcntdq, bitalg, vnni, bf16_feat, amx_tile, amx_int8, amx_bf16) = (
        is_x86_feature_detected!("avx512f"),
        is_x86_feature_detected!("avx512vpopcntdq"),
        is_x86_feature_detected!("avx512bitalg"),
        is_x86_feature_detected!("avx512vnni"),
        // avx512bf16 may not be detectable on all toolchains
        false, // conservative: report false until we confirm detection works
        false, // AMX detection requires specific CPUID checks
        false,
        false,
    );

    #[cfg(not(target_arch = "x86_64"))]
    let (avx512f, vpopcntdq, bitalg, vnni, bf16_feat, amx_tile, amx_int8, amx_bf16) =
        (false, false, false, false, false, false, false, false);

    let libxsmm_compiled = cfg!(feature = "libxsmm");
    let libxsmm_available = libxsmm_compiled && libxsmm_selected; // if selected, it's available

    Capabilities {
        backend_name: name,
        libxsmm_compiled,
        libxsmm_available,
        libxsmm_selected,
        avx512f,
        avx512_vpopcntdq: vpopcntdq,
        avx512_bitalg: bitalg,
        avx512_vnni: vnni,
        avx512_bf16: bf16_feat,
        amx_tile,
        amx_int8,
        amx_bf16,
        supports_batch,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf16_hamming::fp32_to_bf16_bytes;

    #[test]
    fn test_auto_detect_returns_backend() {
        let backend = auto_detect();
        let name = backend.name();
        assert!(
            name == "popcnt-avx512" || name == "popcnt-scalar" || name == "libxsmm",
            "Unexpected backend name: {}",
            name
        );
    }

    #[test]
    fn test_tail_score_identical_is_zero() {
        let backend = auto_detect();
        let vals: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let bytes = fp32_to_bf16_bytes(&vals);
        let weights = BF16Weights::default();

        let score = backend.score(&bytes, &bytes, &weights);
        assert_eq!(score.bf16_distance, 0);
        assert_eq!(score.structural_diff.sign_flips, 0);
    }

    #[test]
    fn test_tail_score_sign_flip_detected() {
        let backend = auto_detect();
        let a: Vec<f32> = (0..64).map(|i| (i as f32 + 1.0) * 0.01).collect();
        let mut b = a.clone();
        b[0] = -b[0]; // flip sign of dim 0
        let a_bytes = fp32_to_bf16_bytes(&a);
        let b_bytes = fp32_to_bf16_bytes(&b);
        let weights = BF16Weights::default();

        let score = backend.score(&a_bytes, &b_bytes, &weights);
        assert!(score.bf16_distance >= 256, "Sign flip should cost >= 256");
        assert!(score.structural_diff.sign_flips >= 1);
    }

    #[test]
    fn test_batch_scoring_matches_individual() {
        let backend = auto_detect();
        let weights = BF16Weights::default();

        let query: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();
        let query_bytes = fp32_to_bf16_bytes(&query);

        // 3 candidates
        let c0: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).cos()).collect();
        let c1: Vec<f32> = (0..32).map(|i| (i as f32 * 0.2).sin()).collect();
        let c2 = query.clone(); // exact match

        let c0_bytes = fp32_to_bf16_bytes(&c0);
        let c1_bytes = fp32_to_bf16_bytes(&c1);
        let c2_bytes = fp32_to_bf16_bytes(&c2);

        // Individual scores
        let s0 = backend.score(&query_bytes, &c0_bytes, &weights);
        let s1 = backend.score(&query_bytes, &c1_bytes, &weights);
        let s2 = backend.score(&query_bytes, &c2_bytes, &weights);

        // Batch score
        let mut batch_data = Vec::new();
        batch_data.extend_from_slice(&c0_bytes);
        batch_data.extend_from_slice(&c1_bytes);
        batch_data.extend_from_slice(&c2_bytes);

        let batch = backend.score_batch(&query_bytes, &batch_data, 3, &weights);

        assert_eq!(batch.distances[0], s0.bf16_distance);
        assert_eq!(batch.distances[1], s1.bf16_distance);
        assert_eq!(batch.distances[2], s2.bf16_distance);
        assert_eq!(batch.distances[2], 0); // exact match
    }

    // ====================================================================
    // Compact scoring tests
    // ====================================================================

    #[test]
    fn test_compact_score_identical_is_zero() {
        let backend = auto_detect();
        let vals: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let bytes = fp32_to_bf16_bytes(&vals);
        let weights = BF16Weights::default();

        let compact = backend.score_compact(&bytes, &bytes, &weights);
        assert_eq!(compact.bf16_distance, 0);
        assert_eq!(compact.sign_flips, 0);
        assert_eq!(compact.exponent_bits_changed, 0);
    }

    #[test]
    fn test_compact_score_matches_full_distance() {
        let backend = auto_detect();
        let weights = BF16Weights::default();

        let a: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).cos()).collect();
        let a_bytes = fp32_to_bf16_bytes(&a);
        let b_bytes = fp32_to_bf16_bytes(&b);

        let full = backend.score(&a_bytes, &b_bytes, &weights);
        let compact = backend.score_compact(&a_bytes, &b_bytes, &weights);

        // Distance must match exactly
        assert_eq!(
            full.bf16_distance, compact.bf16_distance,
            "Compact distance must equal full distance"
        );
        // Sign flips must match
        assert_eq!(
            full.structural_diff.sign_flips as u16, compact.sign_flips,
            "Compact sign_flips must match full"
        );
    }

    #[test]
    fn test_compact_batch_matches_individual() {
        let backend = auto_detect();
        let weights = BF16Weights::default();

        let query: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();
        let query_bytes = fp32_to_bf16_bytes(&query);

        let c0: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).cos()).collect();
        let c1: Vec<f32> = (0..32).map(|i| (i as f32 * 0.2).sin()).collect();
        let c2 = query.clone();

        let c0_bytes = fp32_to_bf16_bytes(&c0);
        let c1_bytes = fp32_to_bf16_bytes(&c1);
        let c2_bytes = fp32_to_bf16_bytes(&c2);

        // Individual compact scores
        let s0 = backend.score_compact(&query_bytes, &c0_bytes, &weights);
        let s1 = backend.score_compact(&query_bytes, &c1_bytes, &weights);
        let s2 = backend.score_compact(&query_bytes, &c2_bytes, &weights);

        // Batch compact
        let mut batch_data = Vec::new();
        batch_data.extend_from_slice(&c0_bytes);
        batch_data.extend_from_slice(&c1_bytes);
        batch_data.extend_from_slice(&c2_bytes);

        let batch = backend.score_batch_compact(&query_bytes, &batch_data, 3, &weights);

        assert_eq!(batch[0].bf16_distance, s0.bf16_distance);
        assert_eq!(batch[1].bf16_distance, s1.bf16_distance);
        assert_eq!(batch[2].bf16_distance, s2.bf16_distance);
        assert_eq!(batch[2].bf16_distance, 0); // exact match
    }

    // ====================================================================
    // Ordering invariant tests
    // ====================================================================

    #[test]
    fn test_ordering_invariant_stable_across_backends() {
        // The invariant: same inputs → same ordering when sorted by (distance, index).
        // "Backend swap doesn't change math."
        let scalar = crate::backends::popcnt::PopcntBackend::scalar();
        let dispatched = crate::backends::popcnt::PopcntBackend::avx512();
        let weights = BF16Weights::default();

        let query: Vec<f32> = (0..64).map(|i| (i as f32 * 0.17).sin()).collect();
        let query_bytes = fp32_to_bf16_bytes(&query);

        // 20 candidates with varying distances
        let n = 20;
        let stride = query_bytes.len();
        let mut batch_data = Vec::with_capacity(n * stride);
        for i in 0..n {
            let vals: Vec<f32> = (0..64)
                .map(|j| ((i * 100 + j) as f32 * 0.037).cos())
                .collect();
            batch_data.extend_from_slice(&fp32_to_bf16_bytes(&vals));
        }

        let scores_scalar = scalar.score_batch(&query_bytes, &batch_data, n, &weights);
        let scores_dispatched = dispatched.score_batch(&query_bytes, &batch_data, n, &weights);

        // Sort by (distance, index) — deterministic tiebreak
        let mut order_scalar: Vec<(u64, usize)> = scores_scalar
            .distances
            .iter()
            .enumerate()
            .map(|(i, &d)| (d, i))
            .collect();
        order_scalar.sort();

        let mut order_dispatched: Vec<(u64, usize)> = scores_dispatched
            .distances
            .iter()
            .enumerate()
            .map(|(i, &d)| (d, i))
            .collect();
        order_dispatched.sort();

        assert_eq!(
            order_scalar, order_dispatched,
            "Ordering must be identical across backends when sorted by (distance, index)"
        );
    }

    #[test]
    fn test_ordering_invariant_with_forced_ties() {
        // Candidates with identical distances: ordering by index must be stable.
        let backend = auto_detect();
        let weights = BF16Weights::default();

        let query: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let query_bytes = fp32_to_bf16_bytes(&query);

        // 5 candidates: indices 0,1,2 are identical copies of a distant vector.
        // Indices 3,4 are exact matches.
        let distant: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * -0.5).collect();
        let distant_bytes = fp32_to_bf16_bytes(&distant);

        let n = 5;
        let stride = query_bytes.len();
        let mut batch_data = Vec::with_capacity(n * stride);
        batch_data.extend_from_slice(&distant_bytes); // 0: distant
        batch_data.extend_from_slice(&distant_bytes); // 1: distant (same distance as 0)
        batch_data.extend_from_slice(&distant_bytes); // 2: distant (same distance as 0)
        batch_data.extend_from_slice(&query_bytes); // 3: exact match
        batch_data.extend_from_slice(&query_bytes); // 4: exact match (same distance as 3)

        let scores = backend.score_batch(&query_bytes, &batch_data, n, &weights);

        // Sort by (distance, index)
        let mut order: Vec<(u64, usize)> = scores
            .distances
            .iter()
            .enumerate()
            .map(|(i, &d)| (d, i))
            .collect();
        order.sort();

        // Exact matches (distance=0) should come first, ordered by index
        assert_eq!(order[0], (0, 3), "First should be index 3 (exact match)");
        assert_eq!(order[1], (0, 4), "Second should be index 4 (exact match)");

        // Distant candidates should be tied, ordered by index
        let distant_dist = order[2].0;
        assert_eq!(
            order[2],
            (distant_dist, 0),
            "Tied distant: index 0 first"
        );
        assert_eq!(
            order[3],
            (distant_dist, 1),
            "Tied distant: index 1 second"
        );
        assert_eq!(
            order[4],
            (distant_dist, 2),
            "Tied distant: index 2 third"
        );
    }

    #[test]
    fn test_ordering_compact_matches_full() {
        // Compact ordering must match full ordering.
        let backend = auto_detect();
        let weights = BF16Weights::default();

        let query: Vec<f32> = (0..32).map(|i| (i as f32 * 0.13).sin()).collect();
        let query_bytes = fp32_to_bf16_bytes(&query);

        let n = 15;
        let stride = query_bytes.len();
        let mut batch_data = Vec::with_capacity(n * stride);
        for i in 0..n {
            let vals: Vec<f32> = (0..32)
                .map(|j| ((i * 200 + j) as f32 * 0.041).cos())
                .collect();
            batch_data.extend_from_slice(&fp32_to_bf16_bytes(&vals));
        }

        let full = backend.score_batch(&query_bytes, &batch_data, n, &weights);
        let compact = backend.score_batch_compact(&query_bytes, &batch_data, n, &weights);

        // Distances must be identical
        for i in 0..n {
            assert_eq!(
                full.distances[i], compact[i].bf16_distance,
                "Distance mismatch at index {}: full={} compact={}",
                i, full.distances[i], compact[i].bf16_distance
            );
        }

        // Ordering must be identical
        let mut order_full: Vec<(u64, usize)> = full
            .distances
            .iter()
            .enumerate()
            .map(|(i, &d)| (d, i))
            .collect();
        order_full.sort();

        let mut order_compact: Vec<(u64, usize)> = compact
            .iter()
            .enumerate()
            .map(|(i, c)| (c.bf16_distance, i))
            .collect();
        order_compact.sort();

        assert_eq!(
            order_full, order_compact,
            "Full and compact must produce identical orderings"
        );
    }

    // ====================================================================
    // Capabilities tests
    // ====================================================================

    #[test]
    fn test_capabilities_deterministic() {
        let caps1 = capabilities();
        let caps2 = capabilities();

        assert_eq!(caps1.backend_name, caps2.backend_name);
        assert_eq!(caps1.avx512f, caps2.avx512f);
        assert_eq!(caps1.avx512_vpopcntdq, caps2.avx512_vpopcntdq);
        assert_eq!(caps1.avx512_bitalg, caps2.avx512_bitalg);
        assert_eq!(caps1.libxsmm_compiled, caps2.libxsmm_compiled);
        assert_eq!(caps1.supports_batch, caps2.supports_batch);
    }

    #[test]
    fn test_capabilities_backend_name_valid() {
        let caps = capabilities();
        assert!(
            caps.backend_name == "popcnt-avx512"
                || caps.backend_name == "popcnt-scalar"
                || caps.backend_name == "libxsmm",
            "Unexpected backend: {}",
            caps.backend_name
        );
    }

    #[test]
    fn test_capabilities_debug_printable() {
        // Capabilities must be Debug-printable (programmatic truth)
        let caps = capabilities();
        let debug = format!("{:?}", caps);
        assert!(debug.contains("backend_name"));
        assert!(debug.contains("avx512f"));
    }

    #[test]
    fn test_capabilities_consistency() {
        // If BITALG is detected, AVX-512F must also be present
        let caps = capabilities();
        if caps.avx512_bitalg {
            assert!(
                caps.avx512f,
                "BITALG implies AVX-512F, but avx512f=false"
            );
        }
        // If libxsmm is selected, it must be compiled
        if caps.libxsmm_selected {
            assert!(
                caps.libxsmm_compiled,
                "Selected XSMM without compile feature"
            );
        }
    }

    // ====================================================================
    // Microbench: tail throughput
    // ====================================================================
    //
    // Run with: cargo test --release -p rustynum-core -- bench_tail --ignored --nocapture
    //
    // Measures ns/candidate and candidates/sec for the popcnt backend.
    // Anti-DCE: accumulates checksum of returned distances.
    // Reuses buffers across iterations (no allocator noise).
    // Warmup: 5 unmeasured iterations before timing.

    /// Helper: deterministic pseudo-random f32 vector.
    fn pseudo_f32_vec(seed: usize, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((seed * 7919 + i * 6271) as f32 * 0.000137).sin())
            .collect()
    }

    fn run_bench(backend: &dyn TailBackend, n_candidates: usize, n_dims: usize) {
        let weights = BF16Weights::default();
        let query = fp32_to_bf16_bytes(&pseudo_f32_vec(42, n_dims));
        let stride = query.len();

        // Pre-allocate candidate buffer (reused, no allocator noise)
        let mut batch_data = vec![0u8; n_candidates * stride];
        for i in 0..n_candidates {
            let vals = pseudo_f32_vec(i + 1000, n_dims);
            let bytes = fp32_to_bf16_bytes(&vals);
            batch_data[i * stride..(i + 1) * stride].copy_from_slice(&bytes);
        }

        // Warmup (5 unmeasured iterations)
        let mut checksum: u64 = 0;
        for _ in 0..5 {
            let batch = backend.score_batch(&query, &batch_data, n_candidates, &weights);
            checksum = checksum.wrapping_add(batch.distances.iter().sum::<u64>());
        }

        // Timed iterations
        let iterations = 100;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let batch = backend.score_batch(&query, &batch_data, n_candidates, &weights);
            // Anti-DCE: compiler can't optimize away if we use the result
            checksum = checksum.wrapping_add(batch.distances.iter().sum::<u64>());
        }
        let elapsed = start.elapsed();

        let total_candidates = iterations * n_candidates;
        let ns_per_candidate = elapsed.as_nanos() as f64 / total_candidates as f64;
        let candidates_per_sec = total_candidates as f64 / elapsed.as_secs_f64();
        let bytes_per_sec = candidates_per_sec * stride as f64;

        eprintln!(
            "  N={:<6} dims={:<6} | {:.1} ns/cand | {:.0} cand/s | {:.1} MB/s | checksum={}",
            n_candidates,
            n_dims,
            ns_per_candidate,
            candidates_per_sec,
            bytes_per_sec / 1_000_000.0,
            checksum,
        );
    }

    fn run_bench_compact(backend: &dyn TailBackend, n_candidates: usize, n_dims: usize) {
        let weights = BF16Weights::default();
        let query = fp32_to_bf16_bytes(&pseudo_f32_vec(42, n_dims));
        let stride = query.len();

        let mut batch_data = vec![0u8; n_candidates * stride];
        for i in 0..n_candidates {
            let vals = pseudo_f32_vec(i + 1000, n_dims);
            let bytes = fp32_to_bf16_bytes(&vals);
            batch_data[i * stride..(i + 1) * stride].copy_from_slice(&bytes);
        }

        // Warmup
        let mut checksum: u64 = 0;
        for _ in 0..5 {
            let compact = backend.score_batch_compact(&query, &batch_data, n_candidates, &weights);
            checksum = checksum.wrapping_add(compact.iter().map(|c| c.bf16_distance).sum::<u64>());
        }

        let iterations = 100;
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            let compact = backend.score_batch_compact(&query, &batch_data, n_candidates, &weights);
            checksum = checksum.wrapping_add(compact.iter().map(|c| c.bf16_distance).sum::<u64>());
        }
        let elapsed = start.elapsed();

        let total_candidates = iterations * n_candidates;
        let ns_per_candidate = elapsed.as_nanos() as f64 / total_candidates as f64;
        let candidates_per_sec = total_candidates as f64 / elapsed.as_secs_f64();

        eprintln!(
            "  N={:<6} dims={:<6} | {:.1} ns/cand | {:.0} cand/s | (compact) checksum={}",
            n_candidates,
            n_dims,
            ns_per_candidate,
            candidates_per_sec,
            checksum,
        );
    }

    #[test]
    #[ignore] // dev-only: cargo test --release -p rustynum-core -- bench_tail_full --ignored --nocapture
    fn bench_tail_full_scoring() {
        let caps = capabilities();
        eprintln!("\n=== Tail Backend Benchmark (full) ===");
        eprintln!("Backend: {}", caps.backend_name);
        eprintln!("AVX-512F={} BITALG={} VPOPCNTDQ={}", caps.avx512f, caps.avx512_bitalg, caps.avx512_vpopcntdq);

        let backend = auto_detect();
        let dims = 1024; // Jina-standard: 1024 BF16 dims = 2048 bytes

        for &n in &[7, 8, 64, 256, 1024] {
            run_bench(backend.as_ref(), n, dims);
        }
        eprintln!("=====================================\n");
    }

    #[test]
    #[ignore] // dev-only: cargo test --release -p rustynum-core -- bench_tail_compact --ignored --nocapture
    fn bench_tail_compact_scoring() {
        let caps = capabilities();
        eprintln!("\n=== Tail Backend Benchmark (compact) ===");
        eprintln!("Backend: {}", caps.backend_name);

        let backend = auto_detect();
        let dims = 1024;

        for &n in &[7, 8, 64, 256, 1024] {
            run_bench_compact(backend.as_ref(), n, dims);
        }
        eprintln!("=========================================\n");
    }
}
