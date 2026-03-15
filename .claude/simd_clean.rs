//! SIMD dispatch: detect once, dispatch forever.
//!
//! One `OnceLock` detects the CPU tier at first call.
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
        Tier::Avx2   => crate::scalar_fns::dot_i8,  // TODO: add simd_avx2::dot_i8
        Tier::Scalar => crate::scalar_fns::dot_i8,
    }
}

// ─── Batch / top-k (delegate to tier-specific impls) ───────────────

dispatch!(hamming_batch(query: &[u8], database: &[u8], num_rows: usize, row_bytes: usize) -> Vec<u64>);

// hamming_top_k has a complex signature — explicit dispatch
pub fn hamming_top_k(
    query: &[u8],
    database: &[u8],
    num_rows: usize,
    row_bytes: usize,
    k: usize,
    threshold: u64,
) -> Vec<(usize, u64)> {
    match tier() {
        Tier::Avx512 => unsafe {
            crate::simd_avx512::hamming_top_k(query, database, num_rows, row_bytes, k, threshold)
        },
        Tier::Avx2 => crate::simd_avx2::hamming_top_k(query, database, num_rows, row_bytes, k, threshold),
        Tier::Scalar => crate::scalar_fns::hamming_top_k(query, database, num_rows, row_bytes, k, threshold),
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

// HDR cascade search — already has its own dispatch internally
pub use crate::hdr::hdr_cascade_search;
