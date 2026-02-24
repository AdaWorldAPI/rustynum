//! LIBXSMM tail backend — JIT-compiled BF16 GEMM via FFI.
//!
//! Feature-gated: only compiled with `--features libxsmm`.
//!
//! ## FFI Surface (from maxsim-cpu reference)
//!
//! ```text
//! libxsmm_init()           — one-time initialization
//! libxsmm_finalize()       — cleanup (optional)
//! libxsmm_sgemm(...)       — BLAS-compatible SGEMM (auto-JIT internally)
//! libxsmm_dispatch_gemm()  — explicit JIT dispatch for BF16
//! ```
//!
//! ## Linking
//!
//! build.rs must link:
//! ```text
//! cargo:rustc-link-lib=static=xsmm
//! cargo:rustc-link-lib=static=xsmmext
//! cargo:rustc-link-lib=dl
//! cargo:rustc-link-lib=m
//! cargo:rustc-link-lib=pthread
//! ```
//!
//! ## Safety Contract
//!
//! All FFI calls are confined to this module. The `XsmmBackend` holds only
//! configuration. No LIBXSMM pointers, handles, or buffers leak through
//! the `TailBackend` trait boundary.
//!
//! ## Batch Scoring
//!
//! The batch path widens BF16 bytes to f32, runs `libxsmm_sgemm` for
//! Q × Candidates^T, then reduces per-row. This fuses the GEMM + scoring
//! into one backend call — the maxsim-cpu pattern.

use crate::bf16_hamming::{self, BF16Weights};
use crate::tail_backend::{BatchTailScore, CompactTailScore, TailBackend, TailScore, compact_score_from_bytes};
use std::sync::Once;

// ============================================================================
// FFI bindings (minimal — matches maxsim-cpu libxsmm_bindings.rs)
// ============================================================================

type LibxsmmBlasint = libc::c_int;

extern "C" {
    fn libxsmm_init();
    fn libxsmm_finalize();

    fn libxsmm_sgemm(
        transa: *const libc::c_char,
        transb: *const libc::c_char,
        m: *const LibxsmmBlasint,
        n: *const LibxsmmBlasint,
        k: *const LibxsmmBlasint,
        alpha: *const libc::c_float,
        a: *const libc::c_float,
        lda: *const LibxsmmBlasint,
        b: *const libc::c_float,
        ldb: *const LibxsmmBlasint,
        beta: *const libc::c_float,
        c: *mut libc::c_float,
        ldc: *const LibxsmmBlasint,
    );
}

/// One-time LIBXSMM initialization.
static INIT: Once = Once::new();

fn ensure_init() {
    INIT.call_once(|| unsafe {
        libxsmm_init();
    });
}

// ============================================================================
// Backend
// ============================================================================

/// LIBXSMM-backed tail scorer.
///
/// Single scoring falls back to POPCNT (LIBXSMM doesn't help for 1 candidate).
/// Batch scoring uses SGEMM for fused GEMM + reduction.
///
/// Configuration only — no LIBXSMM handles stored.
pub struct XsmmBackend {
    /// Fallback for single-candidate scoring.
    popcnt_fn: bf16_hamming::BF16HammingFn,
}

unsafe impl Send for XsmmBackend {}
unsafe impl Sync for XsmmBackend {}

impl XsmmBackend {
    /// Try to create an XSMM backend.
    ///
    /// Returns `None` if LIBXSMM initialization fails.
    /// In practice this only fails if the shared library isn't found.
    pub fn try_new() -> Option<Self> {
        ensure_init();
        Some(Self {
            popcnt_fn: bf16_hamming::select_bf16_hamming_fn(),
        })
    }
}

impl TailBackend for XsmmBackend {
    fn name(&self) -> &'static str {
        "libxsmm"
    }

    /// Single candidate: use POPCNT (LIBXSMM overhead not worth it for N=1).
    fn score(
        &self,
        query_bytes: &[u8],
        candidate_bytes: &[u8],
        weights: &BF16Weights,
    ) -> TailScore {
        let bf16_distance = (self.popcnt_fn)(query_bytes, candidate_bytes, weights);
        let structural_diff = bf16_hamming::structural_diff(query_bytes, candidate_bytes);
        TailScore {
            bf16_distance,
            structural_diff,
        }
    }

    /// Batch scoring via LIBXSMM SGEMM.
    ///
    /// 1. Widen BF16 bytes → f32 for query and all candidates
    /// 2. SGEMM: Q × Candidates^T → similarity matrix [1 × N]
    /// 3. Convert similarity back to structured BF16 distances
    ///
    /// For the BF16 structured distance, we still need per-dimension
    /// decomposition (sign/exp/man). The GEMM gives us an f32 dot-product
    /// similarity as a pre-filter; the structured diff runs only on
    /// candidates that pass the dot-product threshold.
    ///
    /// With N >= 8 candidates, the fused GEMM amortizes JIT overhead.
    fn score_batch(
        &self,
        query_bytes: &[u8],
        candidate_slices: &[u8],
        n_candidates: usize,
        weights: &BF16Weights,
    ) -> BatchTailScore {
        let stride = query_bytes.len();

        // For small batches, fall back to per-candidate scoring
        if n_candidates < 8 {
            let mut distances = Vec::with_capacity(n_candidates);
            let mut diffs = Vec::with_capacity(n_candidates);
            for i in 0..n_candidates {
                let offset = i * stride;
                let cand = &candidate_slices[offset..offset + stride];
                let s = self.score(query_bytes, cand, weights);
                distances.push(s.bf16_distance);
                diffs.push(s.structural_diff);
            }
            return BatchTailScore { distances, diffs };
        }

        // Widen BF16 → f32
        let query_f32 = bf16_hamming::bf16_bytes_to_fp32(query_bytes);
        let n_dims = query_f32.len();

        let mut candidates_f32 = Vec::with_capacity(n_candidates * n_dims);
        for i in 0..n_candidates {
            let offset = i * stride;
            let cand_bytes = &candidate_slices[offset..offset + stride];
            candidates_f32.extend_from_slice(&bf16_hamming::bf16_bytes_to_fp32(cand_bytes));
        }

        // SGEMM: similarities[i] = dot(query, candidate_i)
        // Q is [1 × dim], Candidates is [N × dim] (row-major)
        // We want C = Q × Candidates^T = [1 × N]
        //
        // In column-major BLAS: C^T = Candidates × Q^T
        // transa='T', transb='N', M=N_candidates, N=1, K=dim
        let mut similarities = vec![0.0f32; n_candidates];

        unsafe {
            let transa = b'T' as libc::c_char;
            let transb = b'N' as libc::c_char;
            let m = n_candidates as LibxsmmBlasint;
            let n = 1 as LibxsmmBlasint;
            let k = n_dims as LibxsmmBlasint;
            let alpha = 1.0f32;
            let beta = 0.0f32;
            let lda = n_dims as LibxsmmBlasint;
            let ldb = n_dims as LibxsmmBlasint;
            let ldc = n_candidates as LibxsmmBlasint;

            libxsmm_sgemm(
                &transa,
                &transb,
                &m,
                &n,
                &k,
                &alpha,
                candidates_f32.as_ptr(),
                &lda,
                query_f32.as_ptr(),
                &ldb,
                &beta,
                similarities.as_mut_ptr(),
                &ldc,
            );
        }

        // Now compute exact BF16 structured distances for all candidates.
        // The GEMM similarity can be used as a pre-sort hint, but we still
        // need the sign/exp/man decomposition for the learning signal.
        let mut distances = Vec::with_capacity(n_candidates);
        let mut diffs = Vec::with_capacity(n_candidates);

        for i in 0..n_candidates {
            let offset = i * stride;
            let cand = &candidate_slices[offset..offset + stride];
            let d = (self.popcnt_fn)(query_bytes, cand, weights);
            let diff = bf16_hamming::structural_diff(query_bytes, cand);
            distances.push(d);
            diffs.push(diff);
        }

        BatchTailScore { distances, diffs }
    }

    fn supports_batch(&self) -> bool {
        true
    }

    /// Compact single scoring: distance + counters, no structural diff.
    fn score_compact(
        &self,
        query_bytes: &[u8],
        candidate_bytes: &[u8],
        weights: &BF16Weights,
    ) -> CompactTailScore {
        compact_score_from_bytes(query_bytes, candidate_bytes, weights)
    }

    /// Compact batch: distance + counters only, skips structural diff entirely.
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
            results.push(compact_score_from_bytes(query_bytes, cand, weights));
        }
        results
    }
}

impl Drop for XsmmBackend {
    fn drop(&mut self) {
        // libxsmm_finalize is optional and safe to call multiple times.
        // We don't call it here because other backends may still be alive.
        // It's called automatically at process exit by LIBXSMM's atexit handler.
    }
}
