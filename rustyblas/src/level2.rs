//! BLAS Level 2: Matrix-vector operations.
//!
//! All operations support both row-major and column-major layouts
//! via the CBLAS-style `Layout` parameter.

// TODO(simd): REFACTOR — most Level 2 operations have scalar inner loops:
// - sgemv/dgemv: only incx==1 RowMajor NoTrans/ColMajor Trans paths use SIMD dot;
//   all strided, transpose, and ColMajor NoTrans paths are scalar.
// - sger/dger: fully scalar rank-1 update loops.
// - ssymv/dsymv: fully scalar symmetric MV (triangular iteration).
// - strmv: fully scalar triangular MV.
// - strsv: fully scalar triangular solve (sequential dependencies — partial SIMD only).
// Fix: vectorize contiguous inner loops (j-loops) with SIMD; strided paths need gather/scatter.

use rustynum_core::layout::{Layout, Transpose, Uplo};
use rustynum_core::simd;

// ============================================================================
// GEMV: General matrix-vector multiply
// y := alpha * op(A) * x + beta * y
// ============================================================================

/// Single-precision GEMV: y := alpha * op(A) * x + beta * y
///
/// # Arguments
/// * `layout` - Row-major or column-major
/// * `trans` - Whether to transpose A
/// * `m` - Rows of A
/// * `n` - Columns of A
/// * `alpha` - Scalar multiplier for A*x
/// * `a` - Matrix A (m x n)
/// * `lda` - Leading dimension of A
/// * `x` - Input vector
/// * `incx` - Stride of x
/// * `beta` - Scalar multiplier for y
/// * `y` - Output vector (modified in place)
/// * `incy` - Stride of y
pub fn sgemv(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    x: &[f32],
    incx: usize,
    beta: f32,
    y: &mut [f32],
    incy: usize,
) {
    let (rows, _cols) = match trans {
        Transpose::NoTrans => (m, n),
        _ => (n, m),
    };

    // Scale y by beta
    if beta == 0.0 {
        for i in 0..rows {
            y[i * incy] = 0.0;
        }
    } else if beta != 1.0 {
        for i in 0..rows {
            y[i * incy] *= beta;
        }
    }

    if alpha == 0.0 {
        return;
    }

    match (layout, trans) {
        (Layout::RowMajor, Transpose::NoTrans) => {
            // y[i] += alpha * sum_j(A[i,j] * x[j])
            for i in 0..m {
                let row_start = i * lda;
                if incx == 1 {
                    let dot = simd::dot_f32(&a[row_start..row_start + n], &x[..n]);
                    y[i * incy] += alpha * dot;
                } else {
                    let mut sum = 0.0f32;
                    for j in 0..n {
                        sum += a[row_start + j] * x[j * incx];
                    }
                    y[i * incy] += alpha * sum;
                }
            }
        }
        (Layout::RowMajor, _) => {
            // Transpose: y[j] += alpha * sum_i(A[i,j] * x[i])
            for i in 0..m {
                let row_start = i * lda;
                let xi = alpha * x[i * incx];
                for j in 0..n {
                    y[j * incy] += xi * a[row_start + j];
                }
            }
        }
        (Layout::ColMajor, Transpose::NoTrans) => {
            // Column-major, no trans: y[i] += alpha * sum_j(A[i + j*lda] * x[j])
            for j in 0..n {
                let col_start = j * lda;
                let xj = alpha * x[j * incx];
                for i in 0..m {
                    y[i * incy] += xj * a[col_start + i];
                }
            }
        }
        (Layout::ColMajor, _) => {
            // Column-major, trans: y[j] += alpha * sum_i(A[i + j*lda] * x[i])
            for j in 0..n {
                let col_start = j * lda;
                if incx == 1 {
                    let mut sum = 0.0f32;
                    for i in 0..m {
                        sum += a[col_start + i] * x[i];
                    }
                    y[j * incy] += alpha * sum;
                } else {
                    let mut sum = 0.0f32;
                    for i in 0..m {
                        sum += a[col_start + i] * x[i * incx];
                    }
                    y[j * incy] += alpha * sum;
                }
            }
        }
    }
}

/// Double-precision GEMV: y := alpha * op(A) * x + beta * y
pub fn dgemv(
    layout: Layout,
    trans: Transpose,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    incx: usize,
    beta: f64,
    y: &mut [f64],
    incy: usize,
) {
    let (rows, _cols) = match trans {
        Transpose::NoTrans => (m, n),
        _ => (n, m),
    };

    if beta == 0.0 {
        for i in 0..rows {
            y[i * incy] = 0.0;
        }
    } else if beta != 1.0 {
        for i in 0..rows {
            y[i * incy] *= beta;
        }
    }

    if alpha == 0.0 {
        return;
    }

    match (layout, trans) {
        (Layout::RowMajor, Transpose::NoTrans) => {
            for i in 0..m {
                let row_start = i * lda;
                if incx == 1 {
                    let dot = simd::dot_f64(&a[row_start..row_start + n], &x[..n]);
                    y[i * incy] += alpha * dot;
                } else {
                    let mut sum = 0.0f64;
                    for j in 0..n {
                        sum += a[row_start + j] * x[j * incx];
                    }
                    y[i * incy] += alpha * sum;
                }
            }
        }
        (Layout::RowMajor, _) => {
            for i in 0..m {
                let row_start = i * lda;
                let xi = alpha * x[i * incx];
                for j in 0..n {
                    y[j * incy] += xi * a[row_start + j];
                }
            }
        }
        (Layout::ColMajor, Transpose::NoTrans) => {
            for j in 0..n {
                let col_start = j * lda;
                let xj = alpha * x[j * incx];
                for i in 0..m {
                    y[i * incy] += xj * a[col_start + i];
                }
            }
        }
        (Layout::ColMajor, _) => {
            for j in 0..n {
                let col_start = j * lda;
                let mut sum = 0.0f64;
                for i in 0..m {
                    sum += a[col_start + i] * x[i * incx];
                }
                y[j * incy] += alpha * sum;
            }
        }
    }
}

// ============================================================================
// GER: rank-1 update  A := alpha * x * y^T + A
// ============================================================================

/// Single-precision GER: A := alpha * x * y^T + A
pub fn sger(
    layout: Layout,
    m: usize,
    n: usize,
    alpha: f32,
    x: &[f32],
    incx: usize,
    y: &[f32],
    incy: usize,
    a: &mut [f32],
    lda: usize,
) {
    if alpha == 0.0 {
        return;
    }

    match layout {
        Layout::RowMajor => {
            for i in 0..m {
                let xi = alpha * x[i * incx];
                let row_start = i * lda;
                for j in 0..n {
                    a[row_start + j] += xi * y[j * incy];
                }
            }
        }
        Layout::ColMajor => {
            for j in 0..n {
                let yj = alpha * y[j * incy];
                let col_start = j * lda;
                for i in 0..m {
                    a[col_start + i] += x[i * incx] * yj;
                }
            }
        }
    }
}

/// Double-precision GER: A := alpha * x * y^T + A
pub fn dger(
    layout: Layout,
    m: usize,
    n: usize,
    alpha: f64,
    x: &[f64],
    incx: usize,
    y: &[f64],
    incy: usize,
    a: &mut [f64],
    lda: usize,
) {
    if alpha == 0.0 {
        return;
    }

    match layout {
        Layout::RowMajor => {
            for i in 0..m {
                let xi = alpha * x[i * incx];
                let row_start = i * lda;
                for j in 0..n {
                    a[row_start + j] += xi * y[j * incy];
                }
            }
        }
        Layout::ColMajor => {
            for j in 0..n {
                let yj = alpha * y[j * incy];
                let col_start = j * lda;
                for i in 0..m {
                    a[col_start + i] += x[i * incx] * yj;
                }
            }
        }
    }
}

// ============================================================================
// SYMV: Symmetric matrix-vector multiply
// y := alpha * A * x + beta * y  (A is symmetric)
// ============================================================================

/// Single-precision SYMV: y := alpha * A * x + beta * y (A symmetric)
pub fn ssymv(
    layout: Layout,
    uplo: Uplo,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    x: &[f32],
    incx: usize,
    beta: f32,
    y: &mut [f32],
    incy: usize,
) {
    // Scale y by beta
    if beta == 0.0 {
        for i in 0..n {
            y[i * incy] = 0.0;
        }
    } else if beta != 1.0 {
        for i in 0..n {
            y[i * incy] *= beta;
        }
    }

    if alpha == 0.0 {
        return;
    }

    // For symmetric, we process only the stored triangle
    for i in 0..n {
        let xi = x[i * incx];
        let mut sum = 0.0f32;
        match (layout, uplo) {
            (Layout::RowMajor, Uplo::Upper) | (Layout::ColMajor, Uplo::Lower) => {
                // Diagonal
                sum += a[i * lda + i] * xi;
                // Off-diagonal: both directions
                for j in (i + 1)..n {
                    let aij = a[i * lda + j];
                    sum += aij * x[j * incx];
                    y[j * incy] += alpha * aij * xi;
                }
            }
            (Layout::RowMajor, Uplo::Lower) | (Layout::ColMajor, Uplo::Upper) => {
                // Off-diagonal: below
                for j in 0..i {
                    let aij = a[i * lda + j];
                    sum += aij * x[j * incx];
                    y[j * incy] += alpha * aij * xi;
                }
                // Diagonal
                sum += a[i * lda + i] * xi;
            }
        }
        y[i * incy] += alpha * sum;
    }
}

/// Double-precision SYMV: y := alpha * A * x + beta * y (A symmetric)
pub fn dsymv(
    layout: Layout,
    uplo: Uplo,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    x: &[f64],
    incx: usize,
    beta: f64,
    y: &mut [f64],
    incy: usize,
) {
    if beta == 0.0 {
        for i in 0..n {
            y[i * incy] = 0.0;
        }
    } else if beta != 1.0 {
        for i in 0..n {
            y[i * incy] *= beta;
        }
    }

    if alpha == 0.0 {
        return;
    }

    for i in 0..n {
        let xi = x[i * incx];
        let mut sum = 0.0f64;
        match (layout, uplo) {
            (Layout::RowMajor, Uplo::Upper) | (Layout::ColMajor, Uplo::Lower) => {
                sum += a[i * lda + i] * xi;
                for j in (i + 1)..n {
                    let aij = a[i * lda + j];
                    sum += aij * x[j * incx];
                    y[j * incy] += alpha * aij * xi;
                }
            }
            (Layout::RowMajor, Uplo::Lower) | (Layout::ColMajor, Uplo::Upper) => {
                for j in 0..i {
                    let aij = a[i * lda + j];
                    sum += aij * x[j * incx];
                    y[j * incy] += alpha * aij * xi;
                }
                sum += a[i * lda + i] * xi;
            }
        }
        y[i * incy] += alpha * sum;
    }
}

// ============================================================================
// TRMV: Triangular matrix-vector multiply  x := op(A) * x
// ============================================================================

/// Single-precision TRMV: x := op(A) * x (A triangular)
pub fn strmv(
    layout: Layout,
    uplo: Uplo,
    trans: Transpose,
    diag: rustynum_core::layout::Diag,
    n: usize,
    a: &[f32],
    lda: usize,
    x: &mut [f32],
    incx: usize,
) {
    let unit = diag == rustynum_core::layout::Diag::Unit;

    match (layout, uplo, trans) {
        (Layout::RowMajor, Uplo::Upper, Transpose::NoTrans) => {
            for i in 0..n {
                let mut sum = if unit { x[i * incx] } else { a[i * lda + i] * x[i * incx] };
                for j in (i + 1)..n {
                    sum += a[i * lda + j] * x[j * incx];
                }
                x[i * incx] = sum;
            }
        }
        (Layout::RowMajor, Uplo::Lower, Transpose::NoTrans) => {
            for i in (0..n).rev() {
                let mut sum = if unit { x[i * incx] } else { a[i * lda + i] * x[i * incx] };
                for j in 0..i {
                    sum += a[i * lda + j] * x[j * incx];
                }
                x[i * incx] = sum;
            }
        }
        _ => {
            // Transpose cases: swap upper/lower semantics
            let effective_uplo = match (uplo, trans) {
                (Uplo::Upper, Transpose::Trans | Transpose::ConjTrans) => Uplo::Lower,
                (Uplo::Lower, Transpose::Trans | Transpose::ConjTrans) => Uplo::Upper,
                (u, _) => u,
            };
            // Recurse with flipped params
            strmv(layout, effective_uplo, Transpose::NoTrans, diag, n, a, lda, x, incx);
        }
    }
}

// ============================================================================
// TRSV: Triangular solve  x := op(A)^{-1} * x
// ============================================================================

/// Single-precision TRSV: x := A^{-1} * x (A triangular, row-major)
pub fn strsv(
    layout: Layout,
    uplo: Uplo,
    trans: Transpose,
    diag: rustynum_core::layout::Diag,
    n: usize,
    a: &[f32],
    lda: usize,
    x: &mut [f32],
    incx: usize,
) {
    let unit = diag == rustynum_core::layout::Diag::Unit;

    match (layout, uplo, trans) {
        (Layout::RowMajor, Uplo::Lower, Transpose::NoTrans) => {
            // Forward substitution
            for i in 0..n {
                let mut sum = x[i * incx];
                for j in 0..i {
                    sum -= a[i * lda + j] * x[j * incx];
                }
                x[i * incx] = if unit { sum } else { sum / a[i * lda + i] };
            }
        }
        (Layout::RowMajor, Uplo::Upper, Transpose::NoTrans) => {
            // Back substitution
            for i in (0..n).rev() {
                let mut sum = x[i * incx];
                for j in (i + 1)..n {
                    sum -= a[i * lda + j] * x[j * incx];
                }
                x[i * incx] = if unit { sum } else { sum / a[i * lda + i] };
            }
        }
        _ => {
            // For transpose, swap upper/lower
            let effective_uplo = match (uplo, trans) {
                (Uplo::Upper, Transpose::Trans | Transpose::ConjTrans) => Uplo::Lower,
                (Uplo::Lower, Transpose::Trans | Transpose::ConjTrans) => Uplo::Upper,
                (u, _) => u,
            };
            strsv(layout, effective_uplo, Transpose::NoTrans, diag, n, a, lda, x, incx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgemv_rowmajor_notrans() {
        // A = [[1, 2], [3, 4]], x = [1, 1], y should be [3, 7]
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let x = vec![1.0f32, 1.0];
        let mut y = vec![0.0f32; 2];
        sgemv(Layout::RowMajor, Transpose::NoTrans, 2, 2, 1.0, &a, 2, &x, 1, 0.0, &mut y, 1);
        assert_eq!(y, vec![3.0, 7.0]);
    }

    #[test]
    fn test_sgemv_with_alpha_beta() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let x = vec![1.0f32, 1.0];
        let mut y = vec![10.0f32, 20.0];
        // y = 2.0 * A * x + 3.0 * y = 2*[3,7] + 3*[10,20] = [6+30, 14+60] = [36, 74]
        sgemv(Layout::RowMajor, Transpose::NoTrans, 2, 2, 2.0, &a, 2, &x, 1, 3.0, &mut y, 1);
        assert_eq!(y, vec![36.0, 74.0]);
    }

    #[test]
    fn test_sgemv_trans() {
        // A = [[1, 2], [3, 4]], x = [1, 1], A^T * x = [4, 6]
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let x = vec![1.0f32, 1.0];
        let mut y = vec![0.0f32; 2];
        sgemv(Layout::RowMajor, Transpose::Trans, 2, 2, 1.0, &a, 2, &x, 1, 0.0, &mut y, 1);
        assert_eq!(y, vec![4.0, 6.0]);
    }

    #[test]
    fn test_sger() {
        let x = vec![1.0f32, 2.0];
        let y = vec![3.0f32, 4.0];
        let mut a = vec![0.0f32; 4];
        sger(Layout::RowMajor, 2, 2, 1.0, &x, 1, &y, 1, &mut a, 2);
        assert_eq!(a, vec![3.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_strsv_lower() {
        // L = [[2, 0], [1, 3]], b = [4, 7]
        // Forward sub: x0 = 4/2 = 2, x1 = (7 - 1*2)/3 = 5/3
        let a = vec![2.0f32, 0.0, 1.0, 3.0];
        let mut x = vec![4.0f32, 7.0];
        strsv(
            Layout::RowMajor, Uplo::Lower, Transpose::NoTrans,
            rustynum_core::layout::Diag::NonUnit, 2, &a, 2, &mut x, 1,
        );
        assert!((x[0] - 2.0).abs() < 1e-6);
        assert!((x[1] - 5.0 / 3.0).abs() < 1e-5);
    }
}
