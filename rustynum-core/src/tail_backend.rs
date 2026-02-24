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
}
