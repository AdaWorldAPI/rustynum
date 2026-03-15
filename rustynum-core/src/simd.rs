//! SIMD dispatch: detect once, dispatch forever.
//!
//! One `LazyLock` detects the CPU tier at first call.
//! Every function is one line: `dispatch!(name(args) -> ret);`
//! Adding a new function = adding one line. That's it.
//!
//! Build targets:
//!   cargo build                              → auto (runtime picks best)
//!   RUSTFLAGS="-C target-cpu=native" cargo b → dedicated (compiler picks best)
//!   cargo build --target aarch64-*           → ARM (scalar only)

use std::sync::LazyLock;

// ─── Tier detection: happens ONCE, at first access ─────────────────

#[derive(Clone, Copy, PartialEq)]
enum Tier { Avx512, Avx2, Scalar }

static TIER: LazyLock<Tier> = LazyLock::new(|| {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") { return Tier::Avx512; }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Tier::Avx2;
        }
    }
    Tier::Scalar
});

#[inline(always)]
fn tier() -> Tier { *TIER }

// ─── The macro: one line per function ──────────────────────────────

/// Default dispatch: same function name exists in all three modules.
/// `dispatch!(dot_f32(a: &[f32], b: &[f32]) -> f32);`
/// expands to a pub fn that matches on tier() and calls the right module.
macro_rules! dispatch {
    // Same name in all three modules
    (
        $(#[$meta:meta])*
        $name:ident( $($arg:ident : $ty:ty),* $(,)? ) -> $ret:ty
    ) => {
        $(#[$meta])*
        #[inline]
        pub fn $name( $($arg : $ty),* ) -> $ret {
            match tier() {
                Tier::Avx512 => unsafe { crate::simd_avx512::$name($($arg),*) },
                Tier::Avx2   => crate::simd_avx2::$name($($arg),*),
                Tier::Scalar => crate::scalar_fns::$name($($arg),*),
            }
        }
    };
    // Same name, no return type (returns ())
    (
        $(#[$meta:meta])*
        $name:ident( $($arg:ident : $ty:ty),* $(,)? )
    ) => {
        $(#[$meta])*
        #[inline]
        pub fn $name( $($arg : $ty),* ) {
            match tier() {
                Tier::Avx512 => unsafe { crate::simd_avx512::$name($($arg),*) },
                Tier::Avx2   => crate::simd_avx2::$name($($arg),*),
                Tier::Scalar => crate::scalar_fns::$name($($arg),*),
            }
        }
    };
    // Custom: different paths per tier (for functions missing AVX2 impl)
    (
        $(#[$meta:meta])*
        $name:ident( $($arg:ident : $ty:ty),* $(,)? ) -> $ret:ty
        { $a512:path, $a2:path, $sc:path }
    ) => {
        $(#[$meta])*
        #[inline]
        pub fn $name( $($arg : $ty),* ) -> $ret {
            match tier() {
                Tier::Avx512 => unsafe { $a512($($arg),*) },
                Tier::Avx2   => $a2($($arg),*),
                Tier::Scalar => $sc($($arg),*),
            }
        }
    };
    // Custom, no return type
    (
        $(#[$meta:meta])*
        $name:ident( $($arg:ident : $ty:ty),* $(,)? )
        { $a512:path, $a2:path, $sc:path }
    ) => {
        $(#[$meta])*
        #[inline]
        pub fn $name( $($arg : $ty),* ) {
            match tier() {
                Tier::Avx512 => unsafe { $a512($($arg),*) },
                Tier::Avx2   => $a2($($arg),*),
                Tier::Scalar => $sc($($arg),*),
            }
        }
    };
}

// ─── BLAS-1 ────────────────────────────────────────────────────────

dispatch!(dot_f32(a: &[f32], b: &[f32]) -> f32);
dispatch!(dot_f64(a: &[f64], b: &[f64]) -> f64);
dispatch!(axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]));
dispatch!(axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]));
dispatch!(scal_f32(alpha: f32, x: &mut [f32]));
dispatch!(scal_f64(alpha: f64, x: &mut [f64]));
dispatch!(asum_f32(x: &[f32]) -> f32);
dispatch!(asum_f64(x: &[f64]) -> f64);
dispatch!(nrm2_f32(x: &[f32]) -> f32);
dispatch!(nrm2_f64(x: &[f64]) -> f64);

// iamax: no AVX2 impl, falls to scalar
dispatch!(iamax_f32(x: &[f32]) -> (usize, f32)
    { crate::simd_avx512::iamax_f32, crate::scalar_fns::iamax_f32, crate::scalar_fns::iamax_f32 });
dispatch!(iamax_f64(x: &[f64]) -> (usize, f64)
    { crate::simd_avx512::iamax_f64, crate::scalar_fns::iamax_f64, crate::scalar_fns::iamax_f64 });

// ─── Element-wise f32 ──────────────────────────────────────────────

dispatch!(add_f32_scalar(a: &[f32], scalar: f32) -> Vec<f32>
    { crate::simd_avx512::add_f32_scalar, crate::scalar_fns::add_f32_scalar, crate::scalar_fns::add_f32_scalar });
dispatch!(sub_f32_scalar(a: &[f32], scalar: f32) -> Vec<f32>
    { crate::simd_avx512::sub_f32_scalar, crate::scalar_fns::sub_f32_scalar, crate::scalar_fns::sub_f32_scalar });
dispatch!(mul_f32_scalar(a: &[f32], scalar: f32) -> Vec<f32>
    { crate::simd_avx512::mul_f32_scalar, crate::scalar_fns::mul_f32_scalar, crate::scalar_fns::mul_f32_scalar });
dispatch!(div_f32_scalar(a: &[f32], scalar: f32) -> Vec<f32>
    { crate::simd_avx512::div_f32_scalar, crate::scalar_fns::div_f32_scalar, crate::scalar_fns::div_f32_scalar });
dispatch!(add_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32>
    { crate::simd_avx512::add_f32_vec, crate::scalar_fns::add_f32_vec, crate::scalar_fns::add_f32_vec });
dispatch!(sub_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32>
    { crate::simd_avx512::sub_f32_vec, crate::scalar_fns::sub_f32_vec, crate::scalar_fns::sub_f32_vec });
dispatch!(mul_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32>
    { crate::simd_avx512::mul_f32_vec, crate::scalar_fns::mul_f32_vec, crate::scalar_fns::mul_f32_vec });
dispatch!(div_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32>
    { crate::simd_avx512::div_f32_vec, crate::scalar_fns::div_f32_vec, crate::scalar_fns::div_f32_vec });

// ─── Element-wise f64 ──────────────────────────────────────────────

dispatch!(add_f64_scalar(a: &[f64], scalar: f64) -> Vec<f64>
    { crate::simd_avx512::add_f64_scalar, crate::scalar_fns::add_f64_scalar, crate::scalar_fns::add_f64_scalar });
dispatch!(sub_f64_scalar(a: &[f64], scalar: f64) -> Vec<f64>
    { crate::simd_avx512::sub_f64_scalar, crate::scalar_fns::sub_f64_scalar, crate::scalar_fns::sub_f64_scalar });
dispatch!(mul_f64_scalar(a: &[f64], scalar: f64) -> Vec<f64>
    { crate::simd_avx512::mul_f64_scalar, crate::scalar_fns::mul_f64_scalar, crate::scalar_fns::mul_f64_scalar });
dispatch!(div_f64_scalar(a: &[f64], scalar: f64) -> Vec<f64>
    { crate::simd_avx512::div_f64_scalar, crate::scalar_fns::div_f64_scalar, crate::scalar_fns::div_f64_scalar });
dispatch!(add_f64_vec(a: &[f64], b: &[f64]) -> Vec<f64>
    { crate::simd_avx512::add_f64_vec, crate::scalar_fns::add_f64_vec, crate::scalar_fns::add_f64_vec });
dispatch!(sub_f64_vec(a: &[f64], b: &[f64]) -> Vec<f64>
    { crate::simd_avx512::sub_f64_vec, crate::scalar_fns::sub_f64_vec, crate::scalar_fns::sub_f64_vec });
dispatch!(mul_f64_vec(a: &[f64], b: &[f64]) -> Vec<f64>
    { crate::simd_avx512::mul_f64_vec, crate::scalar_fns::mul_f64_vec, crate::scalar_fns::mul_f64_vec });
dispatch!(div_f64_vec(a: &[f64], b: &[f64]) -> Vec<f64>
    { crate::simd_avx512::div_f64_vec, crate::scalar_fns::div_f64_vec, crate::scalar_fns::div_f64_vec });

// ─── GEMM ───────────────────────────────────────────────────────────

// Goto BLAS style blocked SGEMM: C = alpha * A * B + C
// Row-major layout. A is m x k (stride lda), B is k x n (stride ldb),
// C is m x n (stride ldc). Beta already applied by caller.
// On AVX-512 CPUs, uses a packed 6x16 microkernel with FMA.
// Falls back to scalar on other architectures.
dispatch!(sgemm_blocked(m: usize, n: usize, k: usize, alpha: f32, a: &[f32], lda: usize, b: &[f32], ldb: usize, c: &mut [f32], ldc: usize)
    { crate::simd_avx512::sgemm_blocked, crate::simd_avx2::sgemm_blocked, crate::scalar_fns::sgemm_blocked });

// Goto BLAS style blocked DGEMM: C = alpha * A * B + C
// Row-major layout. A is m x k (stride lda), B is k x n (stride ldb),
// C is m x n (stride ldc). Beta already applied by caller.
// On AVX-512 CPUs, uses a packed 6x8 microkernel with FMA.
// Falls back to scalar on other architectures.
dispatch!(dgemm_blocked(m: usize, n: usize, k: usize, alpha: f64, a: &[f64], lda: usize, b: &[f64], ldb: usize, c: &mut [f64], ldc: usize)
    { crate::simd_avx512::dgemm_blocked, crate::simd_avx2::dgemm_blocked, crate::scalar_fns::dgemm_blocked });

// ─── Hamming / bitops ──────────────────────────────────────────────

dispatch!(hamming_distance(a: &[u8], b: &[u8]) -> u64);
dispatch!(popcount(a: &[u8]) -> u64);
dispatch!(dot_i8(a: &[u8], b: &[u8]) -> i64);

// ─── Functions that return fn pointers (for hot-loop callers) ──────
//
// These are special: callers do `let f = select_hamming_fn()` then
// call `f` millions of times. The fn pointer IS the dispatch.

pub fn select_hamming_fn() -> fn(&[u8], &[u8]) -> u64 {
    match tier() {
        Tier::Avx512 => |a, b| unsafe { crate::simd_avx512::hamming_distance(a, b) },
        Tier::Avx2   => crate::simd_avx2::hamming_distance,
        Tier::Scalar => crate::scalar_fns::hamming_distance,
    }
}

pub fn select_dot_i8_fn() -> fn(&[u8], &[u8]) -> i64 {
    match tier() {
        Tier::Avx512 => |a, b| unsafe { crate::simd_avx512::dot_i8(a, b) },
        Tier::Avx2   => crate::simd_avx2::dot_i8,
        Tier::Scalar => crate::scalar_fns::dot_i8,
    }
}

// ─── Batch / top-k ─────────────────────────────────────────────────

dispatch!(hamming_batch(query: &[u8], database: &[u8], num_rows: usize, row_bytes: usize) -> Vec<u64>);

/// Top-k nearest neighbors by Hamming distance.
///
/// Returns `(indices, distances)` of the `k` closest rows in `database` to `query`.
/// Uses partial sort — O(n*k) but avoids full sort for small k.
pub fn hamming_top_k(
    query: &[u8],
    database: &[u8],
    num_rows: usize,
    row_bytes: usize,
    k: usize,
) -> (Vec<usize>, Vec<u64>) {
    match tier() {
        Tier::Avx512 => unsafe {
            crate::simd_avx512::hamming_top_k(query, database, num_rows, row_bytes, k)
        },
        Tier::Avx2 => crate::simd_avx2::hamming_top_k(query, database, num_rows, row_bytes, k),
        Tier::Scalar => crate::scalar_fns::hamming_top_k(query, database, num_rows, row_bytes, k),
    }
}

// ─── Re-exports that consumers expect ──────────────────────────────

// Constants (used by GEMM and consumer crates)
pub const F32_LANES: usize = 16;
pub const F64_LANES: usize = 8;
pub const U8_LANES: usize = 64;

pub const SGEMM_MR: usize = 6;
pub const SGEMM_NR: usize = 16;
pub const DGEMM_MR: usize = 6;
pub const DGEMM_NR: usize = 8;

pub const SGEMM_KC: usize = 256;
pub const SGEMM_MC: usize = 128;
pub const SGEMM_NC: usize = 1024;
pub const DGEMM_KC: usize = 256;
pub const DGEMM_MC: usize = 96;
pub const DGEMM_NC: usize = 2048;

// Cache blocking parameters (legacy, kept for backward compat)
pub const L1_BLOCK: usize = 8192;
pub const L2_BLOCK: usize = 65536;
pub const L3_BLOCK: usize = 2_097_152;

// ─── Deprecated backward-compat types ──────────────────────────────

/// Result from HDR cascade search.
///
/// Deprecated: use [`crate::hdr::RankedHit`] instead.
#[deprecated(since = "0.4.0", note = "use hdr::RankedHit")]
#[derive(Debug, Clone)]
pub struct HdrResult {
    /// Index into the database.
    pub index: usize,
    /// Exact Hamming distance (from Stroke 2).
    pub hamming: u64,
    /// Optional high-precision distance (from Stroke 3).
    /// f64::NAN if Stroke 3 was not run (PreciseMode::Off).
    pub precise: f64,
}

/// Deprecated: use [`crate::hdr::PreciseMode`] instead.
#[deprecated(since = "0.4.0", note = "use hdr::PreciseMode")]
pub type PreciseMode = crate::hdr::PreciseMode;

/// Deprecated: use [`crate::hdr::Cascade::query()`] instead.
#[deprecated(since = "0.4.0", note = "use hdr::Cascade::query()")]
#[allow(deprecated)]
pub fn hdr_cascade_search(
    query: &[u8],
    database: &[u8],
    vec_bytes: usize,
    num_vectors: usize,
    threshold: u64,
    precise_mode: PreciseMode,
) -> Vec<HdrResult> {
    let cascade = crate::hdr::Cascade::from_threshold(threshold, vec_bytes);
    let ranked = cascade.query(query, database, vec_bytes, num_vectors, precise_mode);
    ranked
        .into_iter()
        .map(|r| HdrResult {
            index: r.index,
            hamming: r.hamming,
            precise: r.precise,
        })
        .collect()
}

// ─── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(deprecated)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_f32() {
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..100).map(|i| (i * 2) as f32).collect();
        let result = dot_f32(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!(
            (result - expected).abs() < 1.0,
            "dot_f32 mismatch: {} vs {}",
            result,
            expected
        );
    }

    #[test]
    fn test_dot_f64() {
        let a: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..100).map(|i| (i * 2) as f64).collect();
        let result = dot_f64(&a, &b);
        let expected: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_axpy_f32() {
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut y: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        axpy_f32(2.0, &x, &mut y);
        assert_eq!(y, vec![12.0, 24.0, 36.0, 48.0]);
    }

    #[test]
    fn test_scal_f32() {
        let mut x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        scal_f32(3.0, &mut x);
        assert_eq!(x, vec![3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_asum_f32() {
        let x = vec![-1.0f32, 2.0, -3.0, 4.0];
        assert_eq!(asum_f32(&x), 10.0);
    }

    #[test]
    fn test_nrm2_f32() {
        let x = vec![3.0f32, 4.0];
        assert!((nrm2_f32(&x) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_iamax_f32_basic() {
        let x = vec![1.0f32, -5.0, 3.0, 2.0];
        let (idx, val) = iamax_f32(&x);
        assert_eq!(idx, 1); // |-5.0| = 5.0 is the largest
        assert!((val - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_iamax_f32_large() {
        // Test with > 16 elements (exercises SIMD path)
        let mut x = vec![0.1f32; 100];
        x[73] = -99.0;
        let (idx, val) = iamax_f32(&x);
        assert_eq!(idx, 73);
        assert!((val - 99.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_iamax_f32_all_negative() {
        let x = vec![-3.0f32, -1.0, -7.0, -2.0];
        let (idx, val) = iamax_f32(&x);
        assert_eq!(idx, 2); // |-7.0| = 7.0
        assert!((val - 7.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_iamax_f64_basic() {
        let x = vec![1.0f64, 2.0, -10.0, 4.0];
        let (idx, val) = iamax_f64(&x);
        assert_eq!(idx, 2);
        assert!((val - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_iamax_f64_large() {
        let mut x = vec![0.01f64; 200];
        x[150] = 42.0;
        let (idx, val) = iamax_f64(&x);
        assert_eq!(idx, 150);
        assert!((val - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_iamax_empty() {
        let x: Vec<f32> = vec![];
        let (idx, val) = iamax_f32(&x);
        assert_eq!(idx, 0);
        assert_eq!(val, 0.0);
    }

    #[test]
    fn test_iamax_single() {
        let x = vec![-3.125f32];
        let (idx, val) = iamax_f32(&x);
        assert_eq!(idx, 0);
        assert!((val - 3.125).abs() < f32::EPSILON);
    }

    #[test]
    fn test_hamming_distance_identical() {
        let a = vec![0xFFu8; 2048]; // 2KB CogRecord container
        let b = vec![0xFFu8; 2048];
        assert_eq!(hamming_distance(&a, &b), 0);
    }

    #[test]
    fn test_hamming_distance_all_different() {
        let a = vec![0x00u8; 64];
        let b = vec![0xFFu8; 64];
        // 64 bytes × 8 bits = 512 differing bits
        assert_eq!(hamming_distance(&a, &b), 512);
    }

    #[test]
    fn test_hamming_distance_known() {
        // Single byte difference: 0b10101010 ^ 0b01010101 = 0b11111111 → 8 bits
        let mut a = vec![0u8; 100];
        let mut b = vec![0u8; 100];
        a[0] = 0b10101010;
        b[0] = 0b01010101;
        a[50] = 0b11110000;
        b[50] = 0b00001111;
        assert_eq!(hamming_distance(&a, &b), 16); // 8 + 8
    }

    #[test]
    fn test_hamming_distance_2kb() {
        // Simulate CogRecord: 2KB containers with ~25% bit difference
        let a: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let b: Vec<u8> = (0..2048).map(|i| ((i + 1) % 256) as u8).collect();
        let dist = hamming_distance(&a, &b);
        // Verify against scalar reference
        let expected: u64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x ^ y).count_ones() as u64)
            .sum();
        assert_eq!(dist, expected);
    }

    #[test]
    fn test_hamming_batch() {
        let query = vec![0xAAu8; 16];
        let mut database = vec![0u8; 16 * 4]; // 4 rows of 16 bytes
                                              // Row 0: identical → 0
        database[..16].fill(0xAA);
        // Row 1: all different → 16*8 = 128
        database[16..32].fill(0x55);
        // Row 2: half different → 64
        database[32..40].fill(0xAA);
        database[40..48].fill(0x55);
        // Row 3: one byte different → 8
        database[48..64].fill(0xAA);
        database[48] = 0x55;

        let distances = hamming_batch(&query, &database, 4, 16);
        assert_eq!(distances[0], 0);
        assert_eq!(distances[1], 128);
        assert_eq!(distances[2], 64);
        assert_eq!(distances[3], 8);
    }

    #[test]
    fn test_hamming_top_k() {
        let query = vec![0xAAu8; 16];
        let mut database = vec![0xAAu8; 16 * 5]; // 5 rows, all identical
                                                 // Make rows 1 and 3 more different
        database[16] = 0x00; // row 1: 1 byte diff → 4 bits
        database[48] = 0x00;
        database[49] = 0x00; // row 3: 2 bytes diff → 8 bits

        let (indices, distances) = hamming_top_k(&query, &database, 5, 16, 3);
        assert_eq!(indices.len(), 3);
        // Top 3 should be the 3 rows with 0 distance
        assert_eq!(distances[0], 0);
        assert_eq!(distances[1], 0);
        assert_eq!(distances[2], 0);
    }

    // ---- select_hamming_fn / select_dot_i8_fn ----

    #[test]
    fn test_select_hamming_fn() {
        let f = select_hamming_fn();
        let a = vec![0xFFu8; 128];
        let b = vec![0x00u8; 128];
        assert_eq!(f(&a, &b), 128 * 8);
    }

    #[test]
    fn test_select_dot_i8_fn() {
        let f = select_dot_i8_fn();
        let a = vec![1u8; 64];
        let b = vec![1u8; 64];
        assert_eq!(f(&a, &b), 64); // 1*1 * 64
    }

    // ---- HDR Cascade Search ----

    #[test]
    fn test_hdr_cascade_basic() {
        let vec_len = 2048;
        let query = vec![0xAAu8; vec_len];
        let mut db = Vec::new();
        db.extend(vec![0xAA; vec_len]); // vec 0: identical
        db.extend(vec![0x55; vec_len]); // vec 1: maximally different
        db.extend(vec![0xAA; vec_len]); // vec 2: identical
        db.extend(vec![0x00; vec_len]); // vec 3: very different

        let results = hdr_cascade_search(&query, &db, vec_len, 4, 100, PreciseMode::Off);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].index, 0);
        assert_eq!(results[0].hamming, 0);
        assert_eq!(results[1].index, 2);
        assert_eq!(results[1].hamming, 0);
    }

    #[test]
    fn test_hdr_warmup_sigma() {
        let vec_len = 2048;
        let num_random = 1000;
        let num_close = 5;
        let total = num_random + num_close;
        let mut db = vec![0u8; vec_len * total];

        // Fill with pseudo-random data
        for (i, byte) in db[..num_random * vec_len].iter_mut().enumerate() {
            *byte = ((i * 7 + 13) % 256) as u8;
        }

        let query = vec![0xAA; vec_len];
        // Plant 5 close matches (copy query with small perturbation)
        for m in 0..num_close {
            let base = (num_random + m) * vec_len;
            db[base..base + vec_len].copy_from_slice(&query);
            // Flip ~50 bytes spread evenly
            for j in 0..50 {
                db[base + j * 40] ^= 0xFF;
            }
        }

        let results = hdr_cascade_search(&query, &db, vec_len, total, 500, PreciseMode::Off);
        assert!(
            results.len() >= num_close,
            "Expected at least {} matches, got {}",
            num_close,
            results.len()
        );
        // All planted matches should appear
        let indices: Vec<usize> = results.iter().map(|r| r.index).collect();
        for m in 0..num_close {
            assert!(
                indices.contains(&(num_random + m)),
                "Missing planted match {}",
                num_random + m
            );
        }
    }

    #[test]
    fn test_hdr_precision_tier() {
        let vec_len = 2048;
        let query = vec![0xAA; vec_len];
        let mut db = Vec::new();
        db.extend(vec![0xAA; vec_len]); // identical → cosine ≈ 1.0
        db.extend(vec![0x55; vec_len]); // maximally different (won't survive)

        let results = hdr_cascade_search(&query, &db, vec_len, 2, 100, PreciseMode::Vnni);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 0);
        assert!(results[0].precise.is_finite());
        // Cosine of identical vectors should be ~1.0
        assert!(
            (results[0].precise - 1.0).abs() < 0.01,
            "Expected cosine ~1.0, got {}",
            results[0].precise
        );
    }

    #[test]
    fn test_hdr_precision_tier_ranks() {
        let vec_len = 2048;
        let query = vec![0xAA; vec_len];

        // Two candidates with similar Hamming but different patterns
        let mut cand_a = query.clone();
        let mut cand_b = query.clone();
        // Flip same NUMBER of bits but different byte positions
        for byte in &mut cand_a[..30] {
            *byte ^= 0xFF;
        }
        for byte in &mut cand_b[500..530] {
            *byte ^= 0xFF;
        }

        let mut db = Vec::new();
        db.extend_from_slice(&cand_a);
        db.extend_from_slice(&cand_b);

        let results = hdr_cascade_search(&query, &db, vec_len, 2, 300, PreciseMode::Vnni);
        assert_eq!(results.len(), 2);
        // Both have same Hamming distance (30*8 = 240 bits)
        assert_eq!(results[0].hamming, results[1].hamming);
        // Precision tier provides cosine ranking
        assert!(results[0].precise.is_finite());
        assert!(results[1].precise.is_finite());
    }

    // ---- PreciseMode::F32 tests ----

    #[test]
    fn test_hdr_f32_dequantize_identical() {
        let vec_len = 2048;
        // Uniform u8 vectors — when dequantized with scale=1.0, zp=128,
        // value 200 → f32 = 1.0*(200-128) = 72.0
        let query = vec![200u8; vec_len];
        let mut db = Vec::new();
        db.extend(vec![200u8; vec_len]); // identical
        db.extend(vec![56u8; vec_len]); // opposite sign: 56-128 = -72

        let results = hdr_cascade_search(
            &query,
            &db,
            vec_len,
            2,
            20000,
            PreciseMode::F32 {
                scale: 1.0,
                zero_point: 128,
            },
        );
        // Both survive Hamming threshold (generous)
        assert!(!results.is_empty());
        // Identical vector should have cosine ~1.0
        let ident = results.iter().find(|r| r.index == 0).unwrap();
        assert!(
            (ident.precise - 1.0).abs() < 0.01,
            "Expected cosine ~1.0, got {}",
            ident.precise
        );
    }

    #[test]
    fn test_hdr_f32_dequantize_cosine_ranking() {
        let vec_len = 256;
        // Query: constant 200 → dequantized = 72.0
        let query = vec![200u8; vec_len];
        // Close match: 190 → dequantized = 62.0 (positive, high cosine)
        let close = vec![190u8; vec_len];
        // Far match: 80 → dequantized = -48.0 (negative, low cosine)
        let far = vec![80u8; vec_len];

        let mut db = Vec::new();
        db.extend_from_slice(&far);
        db.extend_from_slice(&close);

        let results = hdr_cascade_search(
            &query,
            &db,
            vec_len,
            2,
            50000,
            PreciseMode::F32 {
                scale: 1.0,
                zero_point: 128,
            },
        );
        assert_eq!(results.len(), 2);
        // Sorted by cosine descending — close should be first
        assert_eq!(results[0].index, 1, "Close match should rank first");
        assert!(results[0].precise > results[1].precise);
    }

    // ---- PreciseMode::BF16 tests ----

    #[test]
    fn test_hdr_bf16_falls_through_to_f32() {
        let vec_len = 256;
        let query = vec![200u8; vec_len];
        let db = vec![200u8; vec_len]; // single identical vector

        let results_f32 = hdr_cascade_search(
            &query,
            &db,
            vec_len,
            1,
            50000,
            PreciseMode::F32 {
                scale: 1.0,
                zero_point: 128,
            },
        );
        let results_bf16 = hdr_cascade_search(
            &query,
            &db,
            vec_len,
            1,
            50000,
            PreciseMode::BF16 {
                scale: 1.0,
                zero_point: 128,
            },
        );
        // BF16 currently uses same f32 path, so results should be identical
        assert_eq!(results_f32.len(), results_bf16.len());
        if !results_f32.is_empty() {
            assert!((results_f32[0].precise - results_bf16[0].precise).abs() < 1e-6);
        }
    }

    // ---- PreciseMode::DeltaXor tests ----

    #[test]
    fn test_hdr_delta_xor_pure_hamming() {
        // With delta_weight=0.0, DeltaXor degenerates to pure Hamming ranking
        let vec_len = 2048;
        let query = vec![0xAA; vec_len];
        let mut db = Vec::new();
        db.extend(vec![0xAA; vec_len]); // identical: hamming=0

        let results = hdr_cascade_search(
            &query,
            &db,
            vec_len,
            1,
            100,
            PreciseMode::DeltaXor { delta_weight: 0.0 },
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].hamming, 0);
        // With w=0: blended = hamming_norm * 1 + 0 = 0, precise = 1 - 0 = 1.0
        assert!(
            (results[0].precise - 1.0).abs() < 0.01,
            "Expected ~1.0 with w=0.0, got {}",
            results[0].precise
        );
    }

    #[test]
    fn test_hdr_delta_xor_blended() {
        let vec_len = 2048;
        let query = vec![0xAA; vec_len];

        let mut close = query.clone();
        // Flip 20 bytes → 160 bit hamming
        for i in 0..20 {
            close[i * 100] ^= 0xFF;
        }

        let mut db = Vec::new();
        db.extend_from_slice(&close);

        let results = hdr_cascade_search(
            &query,
            &db,
            vec_len,
            1,
            500,
            PreciseMode::DeltaXor { delta_weight: 0.3 },
        );
        assert_eq!(results.len(), 1);
        assert!(results[0].precise.is_finite());
        // Should be between 0 and 1 for a close-but-not-identical match
        assert!(
            results[0].precise > 0.0 && results[0].precise < 1.0,
            "Expected blended in (0,1), got {}",
            results[0].precise
        );
    }

    // ---- PreciseMode::BF16Hamming tests ----

    #[test]
    fn test_hdr_bf16_hamming_identical() {
        // BF16 vectors: 1024 dims × 2 bytes = 2048 bytes per vector
        let vec_len = 2048;
        let query = vec![0x3F; vec_len]; // All same BF16 value
        let mut db = Vec::new();
        db.extend_from_slice(&query); // vec 0: identical

        let weights = crate::bf16_hamming::BF16Weights::default();
        let results = hdr_cascade_search(
            &query,
            &db,
            vec_len,
            1,
            10000,
            PreciseMode::BF16Hamming { weights },
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].hamming, 0);
        // Identical vectors: BF16 distance = 0, similarity = 1.0
        assert!(
            (results[0].precise - 1.0).abs() < 0.01,
            "Expected ~1.0 for identical, got {}",
            results[0].precise
        );
    }

    #[test]
    fn test_hdr_bf16_hamming_ranking() {
        // Test that BF16Hamming correctly ranks: closer vectors get higher similarity
        let vec_len = 2048;
        let query = vec![0x00; vec_len];

        // vec 0: 1 byte flipped → small distance
        let mut close = vec![0x00; vec_len];
        close[0] = 0x80; // sign flip on dim 0

        // vec 1: many bytes flipped → large distance
        let far = vec![0xFF; vec_len];
        let _ = &far; // all bits differ from query

        let mut db = Vec::new();
        db.extend_from_slice(&close);
        db.extend_from_slice(&far);

        let weights = crate::bf16_hamming::BF16Weights::default();
        let results = hdr_cascade_search(
            &query,
            &db,
            vec_len,
            2,
            u64::MAX,
            PreciseMode::BF16Hamming { weights },
        );
        assert_eq!(results.len(), 2);
        // Results sorted by precise (descending), so closer should be first
        assert!(
            results[0].precise > results[1].precise,
            "Close vector ({}) should rank higher than far vector ({})",
            results[0].precise,
            results[1].precise
        );
    }

    // ---- GEMM tests ----

    #[test]
    fn test_sgemm_blocked_identity() {
        // A = 4x4 identity, B = 4x4 with known values, C should equal alpha * B
        let m = 4;
        let n = 4;
        let k = 4;
        let alpha = 1.0f32;
        let a = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let b = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let mut c = vec![0.0f32; m * n];
        sgemm_blocked(m, n, k, alpha, &a, k, &b, n, &mut c, n);
        for i in 0..m * n {
            assert!(
                (c[i] - b[i]).abs() < 1e-4,
                "sgemm identity mismatch at {}: {} vs {}",
                i, c[i], b[i]
            );
        }
    }

    #[test]
    fn test_sgemm_blocked_alpha() {
        let m = 2;
        let n = 2;
        let k = 2;
        let alpha = 2.5f32;
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 4];
        sgemm_blocked(m, n, k, alpha, &a, k, &b, n, &mut c, n);
        // C = alpha * A * B
        // A*B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        // alpha * A*B = [[47.5, 55.0], [107.5, 125.0]]
        let expected = vec![47.5, 55.0, 107.5, 125.0];
        for i in 0..4 {
            assert!(
                (c[i] - expected[i]).abs() < 1e-3,
                "sgemm alpha mismatch at {}: {} vs {}",
                i, c[i], expected[i]
            );
        }
    }

    #[test]
    fn test_sgemm_blocked_non_square() {
        // Non-square: 3x5 = 3x4 * 4x5
        let m = 3;
        let n = 5;
        let k = 4;
        let alpha = 1.0f32;
        let a: Vec<f32> = (0..m * k).map(|i| (i + 1) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i + 1) as f32) * 0.1).collect();
        let mut c = vec![0.0f32; m * n];
        sgemm_blocked(m, n, k, alpha, &a, k, &b, n, &mut c, n);

        // Verify against naive
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    expected[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }
        for i in 0..m * n {
            assert!(
                (c[i] - expected[i]).abs() < 1e-3,
                "sgemm non-square mismatch at {}: {} vs {}",
                i, c[i], expected[i]
            );
        }
    }

    #[test]
    fn test_sgemm_blocked_large() {
        // Test with dimensions that exercise blocking (larger than MR/NR)
        let m = 50;
        let n = 40;
        let k = 30;
        let alpha = 1.0f32;
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) * 0.3).collect();
        let mut c = vec![0.0f32; m * n];
        sgemm_blocked(m, n, k, alpha, &a, k, &b, n, &mut c, n);

        // Verify against naive
        let mut expected = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    expected[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }
        for i in 0..m * n {
            assert!(
                (c[i] - expected[i]).abs() < 1e-1,
                "sgemm large mismatch at {}: {} vs {}",
                i, c[i], expected[i]
            );
        }
    }

    #[test]
    fn test_dgemm_blocked_identity() {
        let m = 4;
        let n = 4;
        let k = 4;
        let alpha = 1.0f64;
        let a = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let b = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let mut c = vec![0.0f64; m * n];
        dgemm_blocked(m, n, k, alpha, &a, k, &b, n, &mut c, n);
        for i in 0..m * n {
            assert!(
                (c[i] - b[i]).abs() < 1e-10,
                "dgemm identity mismatch at {}: {} vs {}",
                i, c[i], b[i]
            );
        }
    }

    #[test]
    fn test_dgemm_blocked_large() {
        let m = 50;
        let n = 40;
        let k = 30;
        let alpha = 1.0f64;
        let a: Vec<f64> = (0..m * k).map(|i| ((i % 7) as f64 - 3.0) * 0.5).collect();
        let b: Vec<f64> = (0..k * n).map(|i| ((i % 11) as f64 - 5.0) * 0.3).collect();
        let mut c = vec![0.0f64; m * n];
        dgemm_blocked(m, n, k, alpha, &a, k, &b, n, &mut c, n);

        let mut expected = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    expected[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }
        for i in 0..m * n {
            assert!(
                (c[i] - expected[i]).abs() < 1e-6,
                "dgemm large mismatch at {}: {} vs {}",
                i, c[i], expected[i]
            );
        }
    }
}
