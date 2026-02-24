//! POPCNT-based tail backend — wraps bf16_hamming.rs dispatch.
//!
//! Two variants selected at construction time:
//! - `avx512()`: AVX-512 BITALG path (Ice Lake+)
//! - `scalar()`: Pure Rust fallback
//!
//! No state, no FFI, no owned buffers. Just a function pointer
//! and the trait implementation.

use crate::bf16_hamming::{self, BF16Weights};
use crate::tail_backend::{TailBackend, TailScore};

/// POPCNT-based tail backend.
///
/// Wraps the existing `bf16_hamming::select_bf16_hamming_fn()` dispatch
/// behind the `TailBackend` trait. Zero overhead: the function pointer
/// is resolved once at construction, then called directly.
pub struct PopcntBackend {
    /// Dispatched function pointer (AVX-512 or scalar).
    hamming_fn: bf16_hamming::BF16HammingFn,
    /// Human-readable name for diagnostics.
    name: &'static str,
}

// Safety: PopcntBackend holds only a function pointer and a static str.
// Function pointers are Send+Sync. No mutable state.
unsafe impl Send for PopcntBackend {}
unsafe impl Sync for PopcntBackend {}

impl PopcntBackend {
    /// Construct with AVX-512 BITALG path.
    ///
    /// Caller must have verified CPU support before calling this.
    /// Use `auto_detect()` for safe runtime selection.
    pub fn avx512() -> Self {
        Self {
            hamming_fn: bf16_hamming::select_bf16_hamming_fn(),
            name: "popcnt-avx512",
        }
    }

    /// Construct with scalar fallback.
    pub fn scalar() -> Self {
        Self {
            hamming_fn: bf16_hamming::bf16_hamming_scalar,
            name: "popcnt-scalar",
        }
    }
}

impl TailBackend for PopcntBackend {
    fn name(&self) -> &'static str {
        self.name
    }

    fn score(
        &self,
        query_bytes: &[u8],
        candidate_bytes: &[u8],
        weights: &BF16Weights,
    ) -> TailScore {
        let bf16_distance = (self.hamming_fn)(query_bytes, candidate_bytes, weights);
        let structural_diff = bf16_hamming::structural_diff(query_bytes, candidate_bytes);

        TailScore {
            bf16_distance,
            structural_diff,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf16_hamming::fp32_to_bf16_bytes;

    #[test]
    fn test_scalar_backend_identical() {
        let backend = PopcntBackend::scalar();
        assert_eq!(backend.name(), "popcnt-scalar");

        let vals: Vec<f32> = (0..16).map(|i| (i as f32 * 0.5).sin()).collect();
        let bytes = fp32_to_bf16_bytes(&vals);
        let score = backend.score(&bytes, &bytes, &BF16Weights::default());

        assert_eq!(score.bf16_distance, 0);
        assert_eq!(score.structural_diff.sign_flips, 0);
    }

    #[test]
    fn test_scalar_backend_sign_flip() {
        let backend = PopcntBackend::scalar();
        let a = fp32_to_bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let b = fp32_to_bf16_bytes(&[-1.0, 2.0, 3.0, -4.0]);
        let score = backend.score(&a, &b, &BF16Weights::default());

        assert!(score.bf16_distance >= 512); // at least 2 sign flips * 256
        assert_eq!(score.structural_diff.sign_flips, 2);
    }

    #[test]
    fn test_avx512_backend_matches_scalar() {
        // This tests that the dispatched variant matches scalar
        let scalar = PopcntBackend::scalar();
        let dispatched = PopcntBackend::avx512(); // may be AVX-512 or scalar depending on CPU

        let a: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let b: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).cos()).collect();
        let a_bytes = fp32_to_bf16_bytes(&a);
        let b_bytes = fp32_to_bf16_bytes(&b);
        let w = BF16Weights::default();

        let s_scalar = scalar.score(&a_bytes, &b_bytes, &w);
        let s_dispatched = dispatched.score(&a_bytes, &b_bytes, &w);

        assert_eq!(
            s_scalar.bf16_distance, s_dispatched.bf16_distance,
            "Dispatched backend must match scalar"
        );
        assert_eq!(
            s_scalar.structural_diff.sign_flips,
            s_dispatched.structural_diff.sign_flips,
        );
    }
}
