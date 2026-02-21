//! Shared SIMD primitive operations used across all rustynum crates.
//!
//! These are low-level building blocks — the actual BLAS/LAPACK/FFT
//! implementations in rustyblas and rustymkl compose these primitives.

use std::simd::f32x16;
use std::simd::f64x8;
use std::simd::num::SimdFloat;

// ============================================================================
// AVX-512 lane counts
// ============================================================================

/// f32 lanes per AVX-512 register (512 / 32 = 16).
pub const F32_LANES: usize = 16;
/// f64 lanes per AVX-512 register (512 / 64 = 8).
pub const F64_LANES: usize = 8;
/// u8 lanes per AVX-512 register (512 / 8 = 64).
pub const U8_LANES: usize = 64;
/// i32 lanes per AVX-512 register (512 / 32 = 16).
pub const I32_LANES: usize = 16;
/// i64 lanes per AVX-512 register (512 / 64 = 8).
pub const I64_LANES: usize = 8;

// ============================================================================
// GEMM microkernel tile sizes (for cache-blocked GEMM)
// ============================================================================

/// GEMM microkernel: rows of A processed per iteration.
/// For f32: 6 rows × 16 columns = 6 zmm registers for C tile.
pub const SGEMM_MR: usize = 6;
/// GEMM microkernel: columns of B processed per iteration.
/// 16 = one full f32x16 register width.
pub const SGEMM_NR: usize = 16;

/// GEMM microkernel for f64: 6 rows × 8 columns.
pub const DGEMM_MR: usize = 6;
/// 8 = one full f64x8 register width.
pub const DGEMM_NR: usize = 8;

// ============================================================================
// Cache blocking parameters
// ============================================================================

/// L1 cache block size (elements). ~32KB / 4 bytes = 8192 f32 elements.
pub const L1_BLOCK: usize = 8192;
/// L2 cache block size (elements). ~256KB / 4 bytes = 65536 f32 elements.
pub const L2_BLOCK: usize = 65536;
/// L3 cache block size (elements). ~8MB / 4 bytes.
pub const L3_BLOCK: usize = 2_097_152;

/// Block size for K dimension in GEMM (fits in L1).
pub const SGEMM_KC: usize = 256;
/// Block size for M dimension in GEMM (fits in L2).
pub const SGEMM_MC: usize = 128;
/// Block size for N dimension in GEMM (fits in L3).
pub const SGEMM_NC: usize = 4096;

/// Block size for K dimension in DGEMM.
pub const DGEMM_KC: usize = 256;
/// Block size for M dimension in DGEMM.
pub const DGEMM_MC: usize = 96;
/// Block size for N dimension in DGEMM.
pub const DGEMM_NC: usize = 2048;

// ============================================================================
// SIMD dot product primitives
// ============================================================================

/// SIMD f32 dot product using f32x16 (AVX-512).
/// 4x unrolled for maximum ILP.
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let chunks = len / F32_LANES;

    let mut acc0 = f32x16::splat(0.0);
    let mut acc1 = f32x16::splat(0.0);
    let mut acc2 = f32x16::splat(0.0);
    let mut acc3 = f32x16::splat(0.0);

    let full_iters = chunks / 4;
    for i in 0..full_iters {
        let base = i * 4 * F32_LANES;
        let a0 = f32x16::from_slice(&a[base..]);
        let b0 = f32x16::from_slice(&b[base..]);
        acc0 += a0 * b0;

        let a1 = f32x16::from_slice(&a[base + F32_LANES..]);
        let b1 = f32x16::from_slice(&b[base + F32_LANES..]);
        acc1 += a1 * b1;

        let a2 = f32x16::from_slice(&a[base + 2 * F32_LANES..]);
        let b2 = f32x16::from_slice(&b[base + 2 * F32_LANES..]);
        acc2 += a2 * b2;

        let a3 = f32x16::from_slice(&a[base + 3 * F32_LANES..]);
        let b3 = f32x16::from_slice(&b[base + 3 * F32_LANES..]);
        acc3 += a3 * b3;
    }

    // Remaining full SIMD chunks
    for i in (full_iters * 4)..chunks {
        let base = i * F32_LANES;
        let av = f32x16::from_slice(&a[base..]);
        let bv = f32x16::from_slice(&b[base..]);
        acc0 += av * bv;
    }

    let mut sum = (acc0 + acc1 + acc2 + acc3).reduce_sum();

    // Scalar tail
    for i in (chunks * F32_LANES)..len {
        sum += a[i] * b[i];
    }

    sum
}

/// SIMD f64 dot product using f64x8 (AVX-512).
/// 4x unrolled.
#[inline]
pub fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let chunks = len / F64_LANES;

    let mut acc0 = f64x8::splat(0.0);
    let mut acc1 = f64x8::splat(0.0);
    let mut acc2 = f64x8::splat(0.0);
    let mut acc3 = f64x8::splat(0.0);

    let full_iters = chunks / 4;
    for i in 0..full_iters {
        let base = i * 4 * F64_LANES;
        let a0 = f64x8::from_slice(&a[base..]);
        let b0 = f64x8::from_slice(&b[base..]);
        acc0 += a0 * b0;

        let a1 = f64x8::from_slice(&a[base + F64_LANES..]);
        let b1 = f64x8::from_slice(&b[base + F64_LANES..]);
        acc1 += a1 * b1;

        let a2 = f64x8::from_slice(&a[base + 2 * F64_LANES..]);
        let b2 = f64x8::from_slice(&b[base + 2 * F64_LANES..]);
        acc2 += a2 * b2;

        let a3 = f64x8::from_slice(&a[base + 3 * F64_LANES..]);
        let b3 = f64x8::from_slice(&b[base + 3 * F64_LANES..]);
        acc3 += a3 * b3;
    }

    for i in (full_iters * 4)..chunks {
        let base = i * F64_LANES;
        let av = f64x8::from_slice(&a[base..]);
        let bv = f64x8::from_slice(&b[base..]);
        acc0 += av * bv;
    }

    let mut sum = (acc0 + acc1 + acc2 + acc3).reduce_sum();

    for i in (chunks * F64_LANES)..len {
        sum += a[i] * b[i];
    }

    sum
}

/// SIMD f32 axpy: y[i] += alpha * x[i]
#[inline]
pub fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    let len = x.len();
    let chunks = len / F32_LANES;
    let alpha_v = f32x16::splat(alpha);

    for i in 0..chunks {
        let base = i * F32_LANES;
        let xv = f32x16::from_slice(&x[base..]);
        let mut yv = f32x16::from_slice(&y[base..]);
        yv += alpha_v * xv;
        yv.copy_to_slice(&mut y[base..base + F32_LANES]);
    }

    for i in (chunks * F32_LANES)..len {
        y[i] += alpha * x[i];
    }
}

/// SIMD f64 axpy: y[i] += alpha * x[i]
#[inline]
pub fn axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    let len = x.len();
    let chunks = len / F64_LANES;
    let alpha_v = f64x8::splat(alpha);

    for i in 0..chunks {
        let base = i * F64_LANES;
        let xv = f64x8::from_slice(&x[base..]);
        let mut yv = f64x8::from_slice(&y[base..]);
        yv += alpha_v * xv;
        yv.copy_to_slice(&mut y[base..base + F64_LANES]);
    }

    for i in (chunks * F64_LANES)..len {
        y[i] += alpha * x[i];
    }
}

/// SIMD f32 scale: x[i] *= alpha
#[inline]
pub fn scal_f32(alpha: f32, x: &mut [f32]) {
    let len = x.len();
    let chunks = len / F32_LANES;
    let alpha_v = f32x16::splat(alpha);

    for i in 0..chunks {
        let base = i * F32_LANES;
        let xv = f32x16::from_slice(&x[base..]);
        let result = alpha_v * xv;
        result.copy_to_slice(&mut x[base..base + F32_LANES]);
    }

    for i in (chunks * F32_LANES)..len {
        x[i] *= alpha;
    }
}

/// SIMD f64 scale: x[i] *= alpha
#[inline]
pub fn scal_f64(alpha: f64, x: &mut [f64]) {
    let len = x.len();
    let chunks = len / F64_LANES;
    let alpha_v = f64x8::splat(alpha);

    for i in 0..chunks {
        let base = i * F64_LANES;
        let xv = f64x8::from_slice(&x[base..]);
        let result = alpha_v * xv;
        result.copy_to_slice(&mut x[base..base + F64_LANES]);
    }

    for i in (chunks * F64_LANES)..len {
        x[i] *= alpha;
    }
}

/// SIMD f32 sum of absolute values (asum / L1 norm).
#[inline]
pub fn asum_f32(x: &[f32]) -> f32 {
    let len = x.len();
    let chunks = len / F32_LANES;
    let mut acc = f32x16::splat(0.0);

    for i in 0..chunks {
        let base = i * F32_LANES;
        let xv = f32x16::from_slice(&x[base..]);
        acc += xv.abs();
    }

    let mut sum = acc.reduce_sum();
    for i in (chunks * F32_LANES)..len {
        sum += x[i].abs();
    }
    sum
}

/// SIMD f64 sum of absolute values (asum / L1 norm).
#[inline]
pub fn asum_f64(x: &[f64]) -> f64 {
    let len = x.len();
    let chunks = len / F64_LANES;
    let mut acc = f64x8::splat(0.0);

    for i in 0..chunks {
        let base = i * F64_LANES;
        let xv = f64x8::from_slice(&x[base..]);
        acc += xv.abs();
    }

    let mut sum = acc.reduce_sum();
    for i in (chunks * F64_LANES)..len {
        sum += x[i].abs();
    }
    sum
}

/// SIMD f32 L2 norm (nrm2).
#[inline]
pub fn nrm2_f32(x: &[f32]) -> f32 {
    let len = x.len();
    let chunks = len / F32_LANES;
    let mut acc = f32x16::splat(0.0);

    for i in 0..chunks {
        let base = i * F32_LANES;
        let xv = f32x16::from_slice(&x[base..]);
        acc += xv * xv;
    }

    let mut sum = acc.reduce_sum();
    for i in (chunks * F32_LANES)..len {
        sum += x[i] * x[i];
    }
    sum.sqrt()
}

/// SIMD f64 L2 norm (nrm2).
#[inline]
pub fn nrm2_f64(x: &[f64]) -> f64 {
    let len = x.len();
    let chunks = len / F64_LANES;
    let mut acc = f64x8::splat(0.0);

    for i in 0..chunks {
        let base = i * F64_LANES;
        let xv = f64x8::from_slice(&x[base..]);
        acc += xv * xv;
    }

    let mut sum = acc.reduce_sum();
    for i in (chunks * F64_LANES)..len {
        sum += x[i] * x[i];
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_f32() {
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..100).map(|i| (i * 2) as f32).collect();
        let result = dot_f32(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        assert!((result - expected).abs() < 1.0, "dot_f32 mismatch: {} vs {}", result, expected);
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
}
