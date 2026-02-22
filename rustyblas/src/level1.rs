//! BLAS Level 1: Vector-vector operations.
//!
//! All operations use AVX-512 SIMD via `rustynum_core::simd` primitives.

// TODO(simd): REFACTOR — all strided (incx/incy > 1) fallback paths are scalar loops.
// Affected: sdot, ddot, saxpy, daxpy, sscal, dscal, snrm2, dnrm2, sasum, dasum.
// Also: isamax/idamax are fully scalar (argmax has no SIMD path yet).
// Fix: SIMD gather/scatter for strided access, or gather into contiguous buffer → SIMD → scatter.

use rustynum_core::simd;

// ============================================================================
// DOT: inner product
// ============================================================================

/// Single-precision dot product: result = x^T * y
#[inline]
pub fn sdot(n: usize, x: &[f32], incx: usize, y: &[f32], incy: usize) -> f32 {
    if incx == 1 && incy == 1 {
        simd::dot_f32(&x[..n], &y[..n])
    } else {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += x[i * incx] * y[i * incy];
        }
        sum
    }
}

/// Double-precision dot product: result = x^T * y
#[inline]
pub fn ddot(n: usize, x: &[f64], incx: usize, y: &[f64], incy: usize) -> f64 {
    if incx == 1 && incy == 1 {
        simd::dot_f64(&x[..n], &y[..n])
    } else {
        let mut sum = 0.0f64;
        for i in 0..n {
            sum += x[i * incx] * y[i * incy];
        }
        sum
    }
}

// ============================================================================
// AXPY: y = alpha * x + y
// ============================================================================

/// Single-precision axpy: y := alpha * x + y
#[inline]
pub fn saxpy(n: usize, alpha: f32, x: &[f32], incx: usize, y: &mut [f32], incy: usize) {
    if alpha == 0.0 {
        return;
    }
    if incx == 1 && incy == 1 {
        simd::axpy_f32(alpha, &x[..n], &mut y[..n]);
    } else {
        for i in 0..n {
            y[i * incy] += alpha * x[i * incx];
        }
    }
}

/// Double-precision axpy: y := alpha * x + y
#[inline]
pub fn daxpy(n: usize, alpha: f64, x: &[f64], incx: usize, y: &mut [f64], incy: usize) {
    if alpha == 0.0 {
        return;
    }
    if incx == 1 && incy == 1 {
        simd::axpy_f64(alpha, &x[..n], &mut y[..n]);
    } else {
        for i in 0..n {
            y[i * incy] += alpha * x[i * incx];
        }
    }
}

// ============================================================================
// SCAL: x = alpha * x
// ============================================================================

/// Single-precision scal: x := alpha * x
#[inline]
pub fn sscal(n: usize, alpha: f32, x: &mut [f32], incx: usize) {
    if incx == 1 {
        simd::scal_f32(alpha, &mut x[..n]);
    } else {
        for i in 0..n {
            x[i * incx] *= alpha;
        }
    }
}

/// Double-precision scal: x := alpha * x
#[inline]
pub fn dscal(n: usize, alpha: f64, x: &mut [f64], incx: usize) {
    if incx == 1 {
        simd::scal_f64(alpha, &mut x[..n]);
    } else {
        for i in 0..n {
            x[i * incx] *= alpha;
        }
    }
}

// ============================================================================
// NRM2: Euclidean norm
// ============================================================================

/// Single-precision nrm2: ||x||_2
#[inline]
pub fn snrm2(n: usize, x: &[f32], incx: usize) -> f32 {
    if incx == 1 {
        simd::nrm2_f32(&x[..n])
    } else {
        let mut sum = 0.0f32;
        for i in 0..n {
            let v = x[i * incx];
            sum += v * v;
        }
        sum.sqrt()
    }
}

/// Double-precision nrm2: ||x||_2
#[inline]
pub fn dnrm2(n: usize, x: &[f64], incx: usize) -> f64 {
    if incx == 1 {
        simd::nrm2_f64(&x[..n])
    } else {
        let mut sum = 0.0f64;
        for i in 0..n {
            let v = x[i * incx];
            sum += v * v;
        }
        sum.sqrt()
    }
}

// ============================================================================
// ASUM: sum of absolute values
// ============================================================================

/// Single-precision asum: sum(|x_i|)
#[inline]
pub fn sasum(n: usize, x: &[f32], incx: usize) -> f32 {
    if incx == 1 {
        simd::asum_f32(&x[..n])
    } else {
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += x[i * incx].abs();
        }
        sum
    }
}

/// Double-precision asum: sum(|x_i|)
#[inline]
pub fn dasum(n: usize, x: &[f64], incx: usize) -> f64 {
    if incx == 1 {
        simd::asum_f64(&x[..n])
    } else {
        let mut sum = 0.0f64;
        for i in 0..n {
            sum += x[i * incx].abs();
        }
        sum
    }
}

// ============================================================================
// IAMAX: index of max absolute value
// ============================================================================

/// Single-precision iamax: index of max |x_i|
#[inline]
pub fn isamax(n: usize, x: &[f32], incx: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut max_idx = 0;
    let mut max_val = x[0].abs();
    for i in 1..n {
        let v = x[i * incx].abs();
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx
}

/// Double-precision iamax: index of max |x_i|
#[inline]
pub fn idamax(n: usize, x: &[f64], incx: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut max_idx = 0;
    let mut max_val = x[0].abs();
    for i in 1..n {
        let v = x[i * incx].abs();
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    max_idx
}

// ============================================================================
// COPY: x -> y
// ============================================================================

/// Single-precision copy: y := x
#[inline]
pub fn scopy(n: usize, x: &[f32], incx: usize, y: &mut [f32], incy: usize) {
    if incx == 1 && incy == 1 {
        y[..n].copy_from_slice(&x[..n]);
    } else {
        for i in 0..n {
            y[i * incy] = x[i * incx];
        }
    }
}

/// Double-precision copy: y := x
#[inline]
pub fn dcopy(n: usize, x: &[f64], incx: usize, y: &mut [f64], incy: usize) {
    if incx == 1 && incy == 1 {
        y[..n].copy_from_slice(&x[..n]);
    } else {
        for i in 0..n {
            y[i * incy] = x[i * incx];
        }
    }
}

// ============================================================================
// SWAP: x <-> y
// ============================================================================

/// Single-precision swap: x <-> y
#[inline]
pub fn sswap(n: usize, x: &mut [f32], incx: usize, y: &mut [f32], incy: usize) {
    if incx == 1 && incy == 1 {
        x[..n].swap_with_slice(&mut y[..n]);
    } else {
        for i in 0..n {
            let tmp = x[i * incx];
            x[i * incx] = y[i * incy];
            y[i * incy] = tmp;
        }
    }
}

/// Double-precision swap: x <-> y
#[inline]
pub fn dswap(n: usize, x: &mut [f64], incx: usize, y: &mut [f64], incy: usize) {
    if incx == 1 && incy == 1 {
        x[..n].swap_with_slice(&mut y[..n]);
    } else {
        for i in 0..n {
            let tmp = x[i * incx];
            x[i * incx] = y[i * incy];
            y[i * incy] = tmp;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdot() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let y = vec![5.0f32, 6.0, 7.0, 8.0];
        assert_eq!(sdot(4, &x, 1, &y, 1), 70.0);
    }

    #[test]
    fn test_ddot() {
        let x = vec![1.0f64, 2.0, 3.0];
        let y = vec![4.0f64, 5.0, 6.0];
        assert_eq!(ddot(3, &x, 1, &y, 1), 32.0);
    }

    #[test]
    fn test_sdot_strided() {
        let x = vec![1.0f32, 0.0, 2.0, 0.0, 3.0];
        let y = vec![4.0f32, 0.0, 5.0, 0.0, 6.0];
        assert_eq!(sdot(3, &x, 2, &y, 2), 32.0);
    }

    #[test]
    fn test_saxpy() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut y = vec![10.0f32, 20.0, 30.0, 40.0];
        saxpy(4, 2.0, &x, 1, &mut y, 1);
        assert_eq!(y, vec![12.0, 24.0, 36.0, 48.0]);
    }

    #[test]
    fn test_sscal() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        sscal(4, 3.0, &mut x, 1);
        assert_eq!(x, vec![3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_snrm2() {
        let x = vec![3.0f32, 4.0];
        assert!((snrm2(2, &x, 1) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_sasum() {
        let x = vec![-1.0f32, 2.0, -3.0, 4.0];
        assert_eq!(sasum(4, &x, 1), 10.0);
    }

    #[test]
    fn test_isamax() {
        let x = vec![1.0f32, -5.0, 3.0, -2.0];
        assert_eq!(isamax(4, &x, 1), 1);
    }

    #[test]
    fn test_scopy() {
        let x = vec![1.0f32, 2.0, 3.0];
        let mut y = vec![0.0f32; 3];
        scopy(3, &x, 1, &mut y, 1);
        assert_eq!(y, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sswap() {
        let mut x = vec![1.0f32, 2.0, 3.0];
        let mut y = vec![4.0f32, 5.0, 6.0];
        sswap(3, &mut x, 1, &mut y, 1);
        assert_eq!(x, vec![4.0, 5.0, 6.0]);
        assert_eq!(y, vec![1.0, 2.0, 3.0]);
    }
}
