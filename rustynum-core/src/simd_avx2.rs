//! AVX2 SIMD primitives (256-bit): f32x8, f64x4, u8x32.
//!
//! Same API surface as simd.rs (AVX-512) but with half-width vectors.
//! Selected at compile time via `--features avx2 --no-default-features`.
//!
//! Targets: Intel Meteor Lake (U9 185H), Alder Lake, AMD Zen 2+, etc.
//! These CPUs have AVX2 + AVX-VNNI (256-bit) but no AVX-512.

use std::simd::f32x8;
use std::simd::f64x4;
use std::simd::num::SimdFloat;

// ============================================================================
// AVX2 lane counts (half of AVX-512)
// ============================================================================

pub const F32_LANES: usize = 8;
pub const F64_LANES: usize = 4;
pub const U8_LANES: usize = 32;
pub const I32_LANES: usize = 8;
pub const I64_LANES: usize = 4;

// ============================================================================
// GEMM microkernel tile sizes for AVX2
// ============================================================================

/// GEMM microkernel: 6 rows x 8 columns (f32x8).
pub const SGEMM_MR: usize = 6;
pub const SGEMM_NR: usize = 8;

/// DGEMM microkernel: 4 rows x 4 columns (f64x4).
pub const DGEMM_MR: usize = 4;
pub const DGEMM_NR: usize = 4;

// ============================================================================
// Cache blocking parameters (same cache hierarchy, smaller tiles)
// ============================================================================

pub const L1_BLOCK: usize = 8192;
pub const L2_BLOCK: usize = 65536;
pub const L3_BLOCK: usize = 2_097_152;

pub const SGEMM_KC: usize = 256;
pub const SGEMM_MC: usize = 128;
pub const SGEMM_NC: usize = 2048;

pub const DGEMM_KC: usize = 256;
pub const DGEMM_MC: usize = 96;
pub const DGEMM_NC: usize = 1024;

// ============================================================================
// SIMD dot product (AVX2: f32x8, 4x unrolled)
// ============================================================================

#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let chunks = len / F32_LANES;

    let mut acc0 = f32x8::splat(0.0);
    let mut acc1 = f32x8::splat(0.0);
    let mut acc2 = f32x8::splat(0.0);
    let mut acc3 = f32x8::splat(0.0);

    let full_iters = chunks / 4;
    for i in 0..full_iters {
        let base = i * 4 * F32_LANES;
        acc0 += f32x8::from_slice(&a[base..]) * f32x8::from_slice(&b[base..]);
        acc1 += f32x8::from_slice(&a[base + F32_LANES..]) * f32x8::from_slice(&b[base + F32_LANES..]);
        acc2 += f32x8::from_slice(&a[base + 2 * F32_LANES..]) * f32x8::from_slice(&b[base + 2 * F32_LANES..]);
        acc3 += f32x8::from_slice(&a[base + 3 * F32_LANES..]) * f32x8::from_slice(&b[base + 3 * F32_LANES..]);
    }

    for i in (full_iters * 4)..chunks {
        let base = i * F32_LANES;
        acc0 += f32x8::from_slice(&a[base..]) * f32x8::from_slice(&b[base..]);
    }

    let mut sum = (acc0 + acc1 + acc2 + acc3).reduce_sum();
    for i in (chunks * F32_LANES)..len {
        sum += a[i] * b[i];
    }
    sum
}

#[inline]
pub fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let chunks = len / F64_LANES;

    let mut acc0 = f64x4::splat(0.0);
    let mut acc1 = f64x4::splat(0.0);
    let mut acc2 = f64x4::splat(0.0);
    let mut acc3 = f64x4::splat(0.0);

    let full_iters = chunks / 4;
    for i in 0..full_iters {
        let base = i * 4 * F64_LANES;
        acc0 += f64x4::from_slice(&a[base..]) * f64x4::from_slice(&b[base..]);
        acc1 += f64x4::from_slice(&a[base + F64_LANES..]) * f64x4::from_slice(&b[base + F64_LANES..]);
        acc2 += f64x4::from_slice(&a[base + 2 * F64_LANES..]) * f64x4::from_slice(&b[base + 2 * F64_LANES..]);
        acc3 += f64x4::from_slice(&a[base + 3 * F64_LANES..]) * f64x4::from_slice(&b[base + 3 * F64_LANES..]);
    }

    for i in (full_iters * 4)..chunks {
        let base = i * F64_LANES;
        acc0 += f64x4::from_slice(&a[base..]) * f64x4::from_slice(&b[base..]);
    }

    let mut sum = (acc0 + acc1 + acc2 + acc3).reduce_sum();
    for i in (chunks * F64_LANES)..len {
        sum += a[i] * b[i];
    }
    sum
}

// ============================================================================
// SIMD axpy, scal, asum, nrm2 (AVX2)
// ============================================================================

#[inline]
pub fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    debug_assert_eq!(x.len(), y.len());
    let len = x.len();
    let chunks = len / F32_LANES;
    let alpha_v = f32x8::splat(alpha);

    for i in 0..chunks {
        let base = i * F32_LANES;
        let xv = f32x8::from_slice(&x[base..]);
        let mut yv = f32x8::from_slice(&y[base..]);
        yv += alpha_v * xv;
        yv.copy_to_slice(&mut y[base..base + F32_LANES]);
    }
    for i in (chunks * F32_LANES)..len {
        y[i] += alpha * x[i];
    }
}

#[inline]
pub fn axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    let len = x.len();
    let chunks = len / F64_LANES;
    let alpha_v = f64x4::splat(alpha);

    for i in 0..chunks {
        let base = i * F64_LANES;
        let xv = f64x4::from_slice(&x[base..]);
        let mut yv = f64x4::from_slice(&y[base..]);
        yv += alpha_v * xv;
        yv.copy_to_slice(&mut y[base..base + F64_LANES]);
    }
    for i in (chunks * F64_LANES)..len {
        y[i] += alpha * x[i];
    }
}

#[inline]
pub fn scal_f32(alpha: f32, x: &mut [f32]) {
    let len = x.len();
    let chunks = len / F32_LANES;
    let alpha_v = f32x8::splat(alpha);

    for i in 0..chunks {
        let base = i * F32_LANES;
        let xv = f32x8::from_slice(&x[base..]);
        (alpha_v * xv).copy_to_slice(&mut x[base..base + F32_LANES]);
    }
    for i in (chunks * F32_LANES)..len {
        x[i] *= alpha;
    }
}

#[inline]
pub fn scal_f64(alpha: f64, x: &mut [f64]) {
    let len = x.len();
    let chunks = len / F64_LANES;
    let alpha_v = f64x4::splat(alpha);

    for i in 0..chunks {
        let base = i * F64_LANES;
        let xv = f64x4::from_slice(&x[base..]);
        (alpha_v * xv).copy_to_slice(&mut x[base..base + F64_LANES]);
    }
    for i in (chunks * F64_LANES)..len {
        x[i] *= alpha;
    }
}

#[inline]
pub fn asum_f32(x: &[f32]) -> f32 {
    let len = x.len();
    let chunks = len / F32_LANES;
    let mut acc = f32x8::splat(0.0);

    for i in 0..chunks {
        let base = i * F32_LANES;
        acc += f32x8::from_slice(&x[base..]).abs();
    }

    let mut sum = acc.reduce_sum();
    for i in (chunks * F32_LANES)..len {
        sum += x[i].abs();
    }
    sum
}

#[inline]
pub fn asum_f64(x: &[f64]) -> f64 {
    let len = x.len();
    let chunks = len / F64_LANES;
    let mut acc = f64x4::splat(0.0);

    for i in 0..chunks {
        let base = i * F64_LANES;
        acc += f64x4::from_slice(&x[base..]).abs();
    }

    let mut sum = acc.reduce_sum();
    for i in (chunks * F64_LANES)..len {
        sum += x[i].abs();
    }
    sum
}

#[inline]
pub fn nrm2_f32(x: &[f32]) -> f32 {
    let len = x.len();
    let chunks = len / F32_LANES;
    let mut acc = f32x8::splat(0.0);

    for i in 0..chunks {
        let base = i * F32_LANES;
        let xv = f32x8::from_slice(&x[base..]);
        acc += xv * xv;
    }

    let mut sum = acc.reduce_sum();
    for i in (chunks * F32_LANES)..len {
        sum += x[i] * x[i];
    }
    sum.sqrt()
}

#[inline]
pub fn nrm2_f64(x: &[f64]) -> f64 {
    let len = x.len();
    let chunks = len / F64_LANES;
    let mut acc = f64x4::splat(0.0);

    for i in 0..chunks {
        let base = i * F64_LANES;
        let xv = f64x4::from_slice(&x[base..]);
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
        assert!((result - expected).abs() < 1.0, "dot_f32: {} vs {}", result, expected);
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
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut y = vec![10.0f32, 20.0, 30.0, 40.0];
        axpy_f32(2.0, &x, &mut y);
        assert_eq!(y, vec![12.0, 24.0, 36.0, 48.0]);
    }

    #[test]
    fn test_scal_f32() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
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
