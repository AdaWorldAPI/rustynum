//! BF16 GEMM-based tail backend — batch scoring via f32 dot products.
//!
//! Uses BF16→f32 conversion + SIMD dot product to score multiple candidates
//! in a single batch. The distance metric is a quantized dot-product gap:
//!
//! ```text
//! distance = (query_self_dot - dot(query, candidate)) * scale
//! ```
//!
//! - Exact match → distance = 0
//! - Random → distance ~ half of query energy * scale
//! - Anti-match → distance ~ double query energy * scale
//!
//! For single-candidate scoring (`score()`), delegates to the standard BF16
//! structured Hamming distance (same as PopcntBackend) for consistency with
//! the per-candidate path.
//!
//! The batch path (`score_batch()`) provides the efficiency gain:
//! 1. Bulk BF16→f32 conversion for query (once) and candidates
//! 2. SIMD f32 dot products for all candidates
//! 3. Per-candidate structural diff on raw BF16 bytes
//!
//! ## When to use
//!
//! - Many K2 survivors (>= 8 candidates in the BF16 tail)
//! - When dot-product similarity is a better ranking signal than bit Hamming
//! - When you want `supports_batch() = true` without LIBXSMM

use crate::bf16_hamming::{self, BF16Weights};
use crate::tail_backend::{
    compact_score_from_bytes, BatchTailScore, CompactTailScore, TailBackend, TailScore,
};

/// Default scale factor for f32→u64 distance quantization.
///
/// With 1024 BF16 dims of typical magnitude ~1.0, max dot product ≈ 1024.
/// Scale 1000 gives distance range ~0 to ~2,048,000 which fits u64 comfortably
/// and preserves fine-grained ranking resolution.
const DEFAULT_SCALE: f32 = 1000.0;

/// BF16 GEMM-based tail backend.
///
/// Provides batch-optimized scoring using f32 dot product similarity.
/// Single-candidate scoring uses standard BF16 structured Hamming distance
/// for backward compatibility with the per-candidate path.
pub struct GemmBackend {
    /// Scale factor for f32→u64 distance quantization.
    scale: f32,
    /// Dispatched BF16 hamming function for individual scoring.
    hamming_fn: bf16_hamming::BF16HammingFn,
}

// Safety: GemmBackend holds only a function pointer, a float, and no mutable state.
unsafe impl Send for GemmBackend {}
unsafe impl Sync for GemmBackend {}

impl Default for GemmBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GemmBackend {
    /// Create a new GEMM backend with default scale factor.
    pub fn new() -> Self {
        Self {
            scale: DEFAULT_SCALE,
            hamming_fn: bf16_hamming::select_bf16_hamming_fn(),
        }
    }

    /// Create a GEMM backend with custom scale factor.
    ///
    /// Larger scale → finer distance resolution but larger u64 values.
    /// Default: 1000.0.
    pub fn with_scale(scale: f32) -> Self {
        Self {
            scale,
            hamming_fn: bf16_hamming::select_bf16_hamming_fn(),
        }
    }
}

/// Scalar f32 dot product (always available, no SIMD feature gates).
///
/// Used as the baseline implementation. When SIMD features are compiled,
/// `simd_dot_f32()` is preferred in the batch path.
#[inline]
#[cfg_attr(any(feature = "avx512", feature = "avx2"), allow(dead_code))]
fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    // 4x manual unroll for ILP
    let n = a.len();
    let full_quads = n / 4;
    let mut acc0: f32 = 0.0;
    let mut acc1: f32 = 0.0;
    let mut acc2: f32 = 0.0;
    let mut acc3: f32 = 0.0;

    for q in 0..full_quads {
        let base = q * 4;
        acc0 += a[base] * b[base];
        acc1 += a[base + 1] * b[base + 1];
        acc2 += a[base + 2] * b[base + 2];
        acc3 += a[base + 3] * b[base + 3];
    }

    for i in (full_quads * 4)..n {
        acc0 += a[i] * b[i];
    }

    acc0 + acc1 + acc2 + acc3
}

/// Best-available f32 dot product — dispatches to SIMD when compiled.
#[inline]
fn dot_f32_dispatch(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(any(feature = "avx512", feature = "avx2"))]
    {
        crate::simd::dot_f32(a, b)
    }
    #[cfg(not(any(feature = "avx512", feature = "avx2")))]
    {
        dot_f32_scalar(a, b)
    }
}

/// Bulk BF16 bytes → f32 conversion optimized for batch processing.
///
/// Same as `bf16_hamming::bf16_bytes_to_fp32` but pre-allocates into an
/// existing buffer to avoid per-candidate allocation.
#[inline]
fn bf16_bytes_to_f32_into(bytes: &[u8], out: &mut Vec<f32>) {
    let n_dims = bytes.len() / 2;
    out.clear();
    out.reserve(n_dims);
    for chunk in bytes.chunks_exact(2) {
        let bf16 = u16::from_le_bytes([chunk[0], chunk[1]]);
        let bits = (bf16 as u32) << 16;
        out.push(f32::from_bits(bits));
    }
}

impl TailBackend for GemmBackend {
    fn name(&self) -> &'static str {
        "gemm-bf16"
    }

    fn score(
        &self,
        query_bytes: &[u8],
        candidate_bytes: &[u8],
        weights: &BF16Weights,
    ) -> TailScore {
        // Single candidate: use standard BF16 structured Hamming distance
        // (same as PopcntBackend) for compatibility with non-batch consumers.
        let bf16_distance = (self.hamming_fn)(query_bytes, candidate_bytes, weights);
        let structural_diff = bf16_hamming::structural_diff(query_bytes, candidate_bytes);

        TailScore {
            bf16_distance,
            structural_diff,
        }
    }

    fn score_batch(
        &self,
        query_bytes: &[u8],
        candidate_slices: &[u8],
        n_candidates: usize,
        _weights: &BF16Weights,
    ) -> BatchTailScore {
        let stride = query_bytes.len();

        // Convert query BF16 → f32 (done once, amortized over batch)
        let query_f32 = bf16_hamming::bf16_bytes_to_fp32(query_bytes);

        // Query self-dot: reference for distance computation
        let query_self_dot = dot_f32_dispatch(&query_f32, &query_f32);

        let mut distances = Vec::with_capacity(n_candidates);
        let mut diffs = Vec::with_capacity(n_candidates);

        // Reusable buffer for candidate f32 conversion (no per-candidate allocation)
        let mut cand_f32 = Vec::with_capacity(query_f32.len());

        for i in 0..n_candidates {
            let offset = i * stride;
            let cand_bytes = &candidate_slices[offset..offset + stride];

            // BF16 → f32 (reuses buffer)
            bf16_bytes_to_f32_into(cand_bytes, &mut cand_f32);

            // Dot product distance: gap from self-similarity
            let dot = dot_f32_dispatch(&query_f32, &cand_f32);
            let raw_distance = (query_self_dot - dot) * self.scale;
            // Clamp: negative distances (candidate more similar than self) → 0
            let distance = raw_distance.max(0.0) as u64;

            distances.push(distance);

            // Structural diff on raw BF16 bytes (can't be GEMMified)
            diffs.push(bf16_hamming::structural_diff(query_bytes, cand_bytes));
        }

        BatchTailScore { distances, diffs }
    }

    fn score_compact(
        &self,
        query_bytes: &[u8],
        candidate_bytes: &[u8],
        weights: &BF16Weights,
    ) -> CompactTailScore {
        compact_score_from_bytes(query_bytes, candidate_bytes, weights)
    }

    fn score_batch_compact(
        &self,
        query_bytes: &[u8],
        candidate_slices: &[u8],
        n_candidates: usize,
        weights: &BF16Weights,
    ) -> Vec<CompactTailScore> {
        // Compact path: no structural diff allocation, use standard Hamming distance
        let stride = query_bytes.len();
        let mut results = Vec::with_capacity(n_candidates);
        for i in 0..n_candidates {
            let offset = i * stride;
            let cand = &candidate_slices[offset..offset + stride];
            results.push(compact_score_from_bytes(query_bytes, cand, weights));
        }
        results
    }

    fn supports_batch(&self) -> bool {
        true
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
    fn test_gemm_backend_name() {
        let backend = GemmBackend::new();
        assert_eq!(backend.name(), "gemm-bf16");
        assert!(backend.supports_batch());
    }

    #[test]
    fn test_gemm_score_identical_is_zero() {
        let backend = GemmBackend::new();
        let vals: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let bytes = fp32_to_bf16_bytes(&vals);
        let weights = BF16Weights::default();

        // Single score: uses Hamming distance (same as PopcntBackend)
        let score = backend.score(&bytes, &bytes, &weights);
        assert_eq!(score.bf16_distance, 0);
        assert_eq!(score.structural_diff.sign_flips, 0);
    }

    #[test]
    fn test_gemm_batch_identical_is_zero() {
        let backend = GemmBackend::new();
        let vals: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let bytes = fp32_to_bf16_bytes(&vals);
        let weights = BF16Weights::default();

        // Batch with 1 candidate = exact match
        let batch = backend.score_batch(&bytes, &bytes, 1, &weights);
        assert_eq!(batch.distances[0], 0);
        assert_eq!(batch.diffs[0].sign_flips, 0);
    }

    #[test]
    fn test_gemm_batch_ordering_preserved() {
        let backend = GemmBackend::new();
        let weights = BF16Weights::default();

        let query: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();
        let query_bytes = fp32_to_bf16_bytes(&query);

        // Candidate 0: distant (cosine-like)
        let c0: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).cos()).collect();
        // Candidate 1: exact match
        let c1 = query.clone();
        // Candidate 2: negated (anti-match)
        let c2: Vec<f32> = query.iter().map(|v| -v).collect();

        let c0_bytes = fp32_to_bf16_bytes(&c0);
        let c1_bytes = fp32_to_bf16_bytes(&c1);
        let c2_bytes = fp32_to_bf16_bytes(&c2);

        let mut batch_data = Vec::new();
        batch_data.extend_from_slice(&c0_bytes);
        batch_data.extend_from_slice(&c1_bytes);
        batch_data.extend_from_slice(&c2_bytes);

        let batch = backend.score_batch(&query_bytes, &batch_data, 3, &weights);

        // Exact match (index 1) should have distance 0
        assert_eq!(batch.distances[1], 0, "Exact match should have distance 0");

        // Anti-match (index 2) should have highest distance
        assert!(
            batch.distances[2] > batch.distances[0],
            "Anti-match should be farther than random: anti={} random={}",
            batch.distances[2],
            batch.distances[0]
        );

        // Random (index 0) should be between exact and anti
        assert!(
            batch.distances[0] > batch.distances[1],
            "Random should be farther than exact: random={} exact={}",
            batch.distances[0],
            batch.distances[1]
        );
    }

    #[test]
    fn test_gemm_batch_structural_diffs_match_individual() {
        // Structural diffs from batch must match individual score() diffs
        let backend = GemmBackend::new();
        let weights = BF16Weights::default();

        let query: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();
        let query_bytes = fp32_to_bf16_bytes(&query);

        let c0: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).cos()).collect();
        let c1: Vec<f32> = (0..32).map(|i| (i as f32 * 0.2).sin()).collect();

        let c0_bytes = fp32_to_bf16_bytes(&c0);
        let c1_bytes = fp32_to_bf16_bytes(&c1);

        // Individual scores
        let s0 = backend.score(&query_bytes, &c0_bytes, &weights);
        let s1 = backend.score(&query_bytes, &c1_bytes, &weights);

        // Batch
        let mut batch_data = Vec::new();
        batch_data.extend_from_slice(&c0_bytes);
        batch_data.extend_from_slice(&c1_bytes);

        let batch = backend.score_batch(&query_bytes, &batch_data, 2, &weights);

        // Structural diffs must match exactly
        assert_eq!(
            batch.diffs[0].sign_flips, s0.structural_diff.sign_flips,
            "Structural diff sign_flips mismatch for candidate 0"
        );
        assert_eq!(
            batch.diffs[1].sign_flips, s1.structural_diff.sign_flips,
            "Structural diff sign_flips mismatch for candidate 1"
        );
    }

    #[test]
    fn test_gemm_compact_matches_full_distance() {
        let backend = GemmBackend::new();
        let weights = BF16Weights::default();

        let a: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).cos()).collect();
        let a_bytes = fp32_to_bf16_bytes(&a);
        let b_bytes = fp32_to_bf16_bytes(&b);

        let full = backend.score(&a_bytes, &b_bytes, &weights);
        let compact = backend.score_compact(&a_bytes, &b_bytes, &weights);

        // Both use Hamming-based distance (single-candidate path)
        assert_eq!(
            full.bf16_distance, compact.bf16_distance,
            "Compact distance must equal full Hamming distance"
        );
    }

    #[test]
    fn test_gemm_scale_affects_batch_distance() {
        let weights = BF16Weights::default();

        let query: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).sin()).collect();
        let query_bytes = fp32_to_bf16_bytes(&query);

        let candidate: Vec<f32> = (0..32).map(|i| (i as f32 * 0.1).cos()).collect();
        let cand_bytes = fp32_to_bf16_bytes(&candidate);

        let backend_1x = GemmBackend::with_scale(1000.0);
        let backend_2x = GemmBackend::with_scale(2000.0);

        let d1 = backend_1x.score_batch(&query_bytes, &cand_bytes, 1, &weights);
        let d2 = backend_2x.score_batch(&query_bytes, &cand_bytes, 1, &weights);

        // 2x scale should give ~2x distance (not exact due to float→u64 truncation)
        let ratio = d2.distances[0] as f64 / d1.distances[0].max(1) as f64;
        assert!(
            (ratio - 2.0).abs() < 0.1,
            "2x scale should give ~2x distance, ratio={}",
            ratio
        );
    }

    #[test]
    fn test_dot_f32_scalar_basic() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let dot = dot_f32_scalar(&a, &b);
        assert!((dot - 70.0).abs() < 0.01, "dot = {}", dot);
    }

    #[test]
    fn test_dot_f32_scalar_self_dot() {
        let a = vec![1.0, 1.0, 1.0, 1.0];
        let dot = dot_f32_scalar(&a, &a);
        assert!((dot - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_dot_f32_dispatch_matches_scalar() {
        let a: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).cos()).collect();

        let scalar = dot_f32_scalar(&a, &b);
        let dispatch = dot_f32_dispatch(&a, &b);

        assert!(
            (scalar - dispatch).abs() < 0.1,
            "scalar={} dispatch={}",
            scalar,
            dispatch
        );
    }

    #[test]
    fn test_bf16_bytes_to_f32_into_reuses_buffer() {
        let vals: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes = fp32_to_bf16_bytes(&vals);
        let mut buf = Vec::new();

        // First call
        bf16_bytes_to_f32_into(&bytes, &mut buf);
        assert_eq!(buf.len(), 4);
        assert!((buf[0] - 1.0).abs() < 0.02);

        // Second call reuses buffer
        let vals2: Vec<f32> = vec![5.0, 6.0];
        let bytes2 = fp32_to_bf16_bytes(&vals2);
        bf16_bytes_to_f32_into(&bytes2, &mut buf);
        assert_eq!(buf.len(), 2);
        assert!((buf[0] - 5.0).abs() < 0.02);
    }
}
