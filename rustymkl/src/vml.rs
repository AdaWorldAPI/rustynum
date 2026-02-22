//! Vector Math Library (VML) — SIMD-vectorized transcendental functions.
//!
//! Pure Rust replacement for Intel MKL VML. All functions process arrays
//! element-wise using AVX-512 SIMD.
//!
//! Naming convention follows MKL: `vs` prefix = single-precision vector,
//! `vd` prefix = double-precision vector.

use std::simd::num::SimdFloat;
use std::simd::StdFloat;
use rustynum_core::simd::{F32_LANES, F64_LANES};

// SIMD vector types selected by feature flag
#[cfg(feature = "avx512")]
use std::simd::{f32x16 as F32Simd, f64x8 as F64Simd};
#[cfg(not(feature = "avx512"))]
use std::simd::{f32x8 as F32Simd, f64x4 as F64Simd};

// ============================================================================
// EXP: e^x
// ============================================================================

/// Vectorized single-precision exp: out[i] = e^(x[i])
///
/// Uses polynomial approximation for SIMD lanes, scalar fallback for tail.
pub fn vsexp(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    let len = x.len();
    let chunks = len / F32_LANES;

    for i in 0..chunks {
        let base = i * F32_LANES;
        let xv = F32Simd::from_slice(&x[base..]);
        let result = simd_exp_f32(xv);
        result.copy_to_slice(&mut out[base..base + F32_LANES]);
    }

    for i in (chunks * F32_LANES)..len {
        out[i] = x[i].exp();
    }
}

/// Vectorized double-precision exp: out[i] = e^(x[i])
pub fn vdexp(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), out.len());
    let len = x.len();
    let chunks = len / F64_LANES;

    for i in 0..chunks {
        let base = i * F64_LANES;
        let xv = F64Simd::from_slice(&x[base..]);
        let result = simd_exp_f64(xv);
        result.copy_to_slice(&mut out[base..base + F64_LANES]);
    }

    for i in (chunks * F64_LANES)..len {
        out[i] = x[i].exp();
    }
}

// ============================================================================
// LOG: ln(x)
// ============================================================================

/// Vectorized single-precision natural log: out[i] = ln(x[i])
pub fn vsln(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    let len = x.len();
    let chunks = len / F32_LANES;

    for i in 0..chunks {
        let base = i * F32_LANES;
        let xv = F32Simd::from_slice(&x[base..]);
        let result = simd_ln_f32(xv);
        result.copy_to_slice(&mut out[base..base + F32_LANES]);
    }

    for i in (chunks * F32_LANES)..len {
        out[i] = x[i].ln();
    }
}

/// Vectorized double-precision natural log.
// TODO(simd): REFACTOR — vdln is fully scalar, no SIMD path.
// Needs SIMD range reduction + Padé approximation like simd_ln_f32 (which is also scalar).
pub fn vdln(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), out.len());
    let len = x.len();

    for i in 0..len {
        out[i] = x[i].ln();
    }
}

// ============================================================================
// SQRT: square root
// ============================================================================

/// Vectorized single-precision sqrt: out[i] = sqrt(x[i])
pub fn vssqrt(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    let len = x.len();
    let chunks = len / F32_LANES;

    for i in 0..chunks {
        let base = i * F32_LANES;
        let xv = F32Simd::from_slice(&x[base..]);
        let result = xv.sqrt();
        result.copy_to_slice(&mut out[base..base + F32_LANES]);
    }

    for i in (chunks * F32_LANES)..len {
        out[i] = x[i].sqrt();
    }
}

/// Vectorized double-precision sqrt.
pub fn vdsqrt(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), out.len());
    let len = x.len();
    let chunks = len / F64_LANES;

    for i in 0..chunks {
        let base = i * F64_LANES;
        let xv = F64Simd::from_slice(&x[base..]);
        let result = xv.sqrt();
        result.copy_to_slice(&mut out[base..base + F64_LANES]);
    }

    for i in (chunks * F64_LANES)..len {
        out[i] = x[i].sqrt();
    }
}

// ============================================================================
// ABS: absolute value
// ============================================================================

/// Vectorized single-precision abs: out[i] = |x[i]|
pub fn vsabs(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    let len = x.len();
    let chunks = len / F32_LANES;

    for i in 0..chunks {
        let base = i * F32_LANES;
        let xv = F32Simd::from_slice(&x[base..]);
        let result = xv.abs();
        result.copy_to_slice(&mut out[base..base + F32_LANES]);
    }

    for i in (chunks * F32_LANES)..len {
        out[i] = x[i].abs();
    }
}

/// Vectorized double-precision abs.
pub fn vdabs(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), out.len());
    let len = x.len();
    let chunks = len / F64_LANES;

    for i in 0..chunks {
        let base = i * F64_LANES;
        let xv = F64Simd::from_slice(&x[base..]);
        let result = xv.abs();
        result.copy_to_slice(&mut out[base..base + F64_LANES]);
    }

    for i in (chunks * F64_LANES)..len {
        out[i] = x[i].abs();
    }
}

// ============================================================================
// ADD / SUB / MUL / DIV: element-wise arithmetic
// ============================================================================

/// Vectorized single-precision add: out[i] = a[i] + b[i]
pub fn vsadd(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let len = a.len();
    let chunks = len / F32_LANES;

    for i in 0..chunks {
        let base = i * F32_LANES;
        let av = F32Simd::from_slice(&a[base..]);
        let bv = F32Simd::from_slice(&b[base..]);
        let result = av + bv;
        result.copy_to_slice(&mut out[base..base + F32_LANES]);
    }

    for i in (chunks * F32_LANES)..len {
        out[i] = a[i] + b[i];
    }
}

/// Vectorized single-precision multiply: out[i] = a[i] * b[i]
pub fn vsmul(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let len = a.len();
    let chunks = len / F32_LANES;

    for i in 0..chunks {
        let base = i * F32_LANES;
        let av = F32Simd::from_slice(&a[base..]);
        let bv = F32Simd::from_slice(&b[base..]);
        let result = av * bv;
        result.copy_to_slice(&mut out[base..base + F32_LANES]);
    }

    for i in (chunks * F32_LANES)..len {
        out[i] = a[i] * b[i];
    }
}

/// Vectorized single-precision divide: out[i] = a[i] / b[i]
pub fn vsdiv(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    let len = a.len();
    let chunks = len / F32_LANES;

    for i in 0..chunks {
        let base = i * F32_LANES;
        let av = F32Simd::from_slice(&a[base..]);
        let bv = F32Simd::from_slice(&b[base..]);
        let result = av / bv;
        result.copy_to_slice(&mut out[base..base + F32_LANES]);
    }

    for i in (chunks * F32_LANES)..len {
        out[i] = a[i] / b[i];
    }
}

// ============================================================================
// SIN / COS: trigonometric functions (polynomial approximation)
// ============================================================================

/// Vectorized single-precision sin: out[i] = sin(x[i])
// TODO(simd): REFACTOR — vssin is fully scalar. Needs Chebyshev/minimax SIMD polynomial.
// Range reduction to [-pi, pi] then degree-7 polynomial, all in f32x16.
pub fn vssin(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    for i in 0..x.len() {
        out[i] = x[i].sin();
    }
}

/// Vectorized single-precision cos: out[i] = cos(x[i])
// TODO(simd): REFACTOR — vscos is fully scalar. Same polynomial approach as vssin (phase shift).
pub fn vscos(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    for i in 0..x.len() {
        out[i] = x[i].cos();
    }
}

/// Vectorized single-precision pow: out[i] = a[i]^b[i]
// TODO(simd): REFACTOR — vspow is fully scalar. Can be SIMD via exp(b * ln(a)).
pub fn vspow(a: &[f32], b: &[f32], out: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), out.len());
    for i in 0..a.len() {
        out[i] = a[i].powf(b[i]);
    }
}

// ============================================================================
// SIMD polynomial approximations for transcendental functions
// ============================================================================

/// Fast SIMD exp(x) for F32Simd using the "range reduction + polynomial" method.
///
/// Algorithm:
/// 1. Clamp input to avoid overflow/underflow
/// 2. Decompose x = n * ln(2) + r, where n = round(x / ln(2))
/// 3. Compute exp(r) using degree-6 minimax polynomial
/// 4. Scale by 2^n via integer addition to the exponent field
#[inline(always)]
fn simd_exp_f32(x: F32Simd) -> F32Simd {
    let ln2_inv = F32Simd::splat(1.442695040888963f32); // 1/ln(2)
    let ln2_hi = F32Simd::splat(0.693145751953125f32);
    let ln2_lo = F32Simd::splat(1.428606765330187e-6f32);

    // Polynomial coefficients (minimax on [-ln2/2, ln2/2])
    let c1 = F32Simd::splat(1.0);
    let c2 = F32Simd::splat(0.5);
    let c3 = F32Simd::splat(0.16666666666666666);
    let c4 = F32Simd::splat(0.041666666666666664);
    let c5 = F32Simd::splat(0.008333333333333333);

    // Clamp to avoid overflow
    let x_clamped = x.simd_max(F32Simd::splat(-87.0)).simd_min(F32Simd::splat(88.0));

    // n = round(x / ln2)
    let n = (x_clamped * ln2_inv + F32Simd::splat(0.5)).floor();

    // r = x - n * ln2 (high precision via hi/lo split)
    let r = x_clamped - n * ln2_hi - n * ln2_lo;

    // exp(r) ≈ 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120
    let poly = c1 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * c5))));

    // TODO(simd): REFACTOR — the 2^n scaling exits to scalar via to_array()/powi().
    // This defeats the SIMD polynomial pipeline. Needs ldexp via
    // integer manipulation of f32 exponent bits (bit_cast + add to exponent).
    let n_arr = n.to_array();
    let mut out = poly.to_array();
    for i in 0..out.len() {
        out[i] *= (2.0f32).powi(n_arr[i] as i32);
    }
    F32Simd::from_array(out)
}

/// Fast SIMD exp(x) for F64Simd.
// TODO(simd): REFACTOR — simd_exp_f64 is fully scalar (to_array → exp → from_array).
// Needs same range-reduction + polynomial pipeline as simd_exp_f32 but f64 precision.
#[inline(always)]
fn simd_exp_f64(x: F64Simd) -> F64Simd {
    let mut arr = x.to_array();
    for v in arr.iter_mut() {
        *v = v.exp();
    }
    F64Simd::from_array(arr)
}

/// Fast SIMD ln(x) for F32Simd.
// TODO(simd): REFACTOR — simd_ln_f32 is fully scalar (to_array → ln → from_array).
// Needs SIMD range reduction: extract exponent bits, Padé approximation on mantissa.
#[inline(always)]
fn simd_ln_f32(x: F32Simd) -> F32Simd {
    let mut arr = x.to_array();
    for v in arr.iter_mut() {
        *v = v.ln();
    }
    F32Simd::from_array(arr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vsexp() {
        let x: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let mut out = vec![0.0f32; 32];
        vsexp(&x, &mut out);
        for i in 0..32 {
            let expected = x[i].exp();
            assert!(
                (out[i] - expected).abs() / expected.max(1e-10) < 1e-4,
                "vsexp mismatch at {}: {} vs {}",
                i, out[i], expected
            );
        }
    }

    #[test]
    fn test_vssqrt() {
        let x: Vec<f32> = (1..33).map(|i| i as f32).collect();
        let mut out = vec![0.0f32; 32];
        vssqrt(&x, &mut out);
        for i in 0..32 {
            assert!((out[i] - x[i].sqrt()).abs() < 1e-6);
        }
    }

    #[test]
    fn test_vsabs() {
        let x = vec![-1.0f32, 2.0, -3.0, 4.0];
        let mut out = vec![0.0f32; 4];
        vsabs(&x, &mut out);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_vsadd() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut out = vec![0.0f32; 4];
        vsadd(&a, &b, &mut out);
        assert_eq!(out, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_vsmul() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut out = vec![0.0f32; 4];
        vsmul(&a, &b, &mut out);
        assert_eq!(out, vec![5.0, 12.0, 21.0, 32.0]);
    }
}
