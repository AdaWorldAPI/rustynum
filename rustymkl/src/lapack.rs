//! LAPACK routines — pure Rust, built on top of rustyblas.
//!
//! Covers the essential decompositions and solvers:
//!
//! - **LU factorization** (`sgetrf` / `dgetrf`) with partial pivoting
//! - **LU solve** (`sgetrs` / `dgetrs`) — solve A*X = B using LU factors
//! - **Cholesky factorization** (`spotrf` / `dpotrf`)
//! - **Cholesky solve** (`spotrs` / `dpotrs`)
//! - **QR factorization** (`sgeqrf` / `dgeqrf`)
//! - **Triangular solve** — delegates to rustyblas `strsm`/`dtrsm`
//!
//! All operations work on flat row-major or column-major arrays.
//! Uses rustyblas Level 1/2/3 primitives internally.

// TODO(simd): REFACTOR — all LAPACK routines are fully scalar.
// - sgetrf/dgetrf: trailing submatrix update (rank-1) is scalar — can use BLAS sger/dger.
//   Pivot search is scalar argmax — can use isamax/idamax.
// - sgetrs/dgetrs: forward/back substitution is scalar — can use BLAS strsm.
// - spotrf/dpotrf: inner products are scalar — can use BLAS sdot/ddot for column norms.
// - spotrs/dpotrs: triangular solves are scalar — can use BLAS strsm.
// - sgeqrf/dgeqrf: Householder reflector application is scalar — can use BLAS sger/dger.
// Many have sequential data dependencies (k-loop), but inner j-loops are parallelizable.
// Fix: delegate inner loops to BLAS Level 1/2 routines which already have SIMD paths.

use rustynum_core::layout::Layout;

// ============================================================================
// LU Factorization: PA = LU
// ============================================================================

/// Single-precision LU factorization with partial pivoting.
///
/// Factors the M x N matrix A into P * A = L * U.
/// - A is overwritten with L (unit lower) and U (upper).
/// - `ipiv` receives the pivot indices (length min(m, n)).
///
/// Returns 0 on success, > 0 if U is singular (U[info][info] == 0).
pub fn sgetrf(
    layout: Layout,
    m: usize,
    n: usize,
    a: &mut [f32],
    lda: usize,
    ipiv: &mut [usize],
) -> i32 {
    let min_mn = m.min(n);

    for k in 0..min_mn {
        // Find pivot: max |A[i, k]| for i in k..m
        let mut max_val = 0.0f32;
        let mut max_idx = k;
        for i in k..m {
            let val = a[layout.index(i, k, lda)].abs();
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        ipiv[k] = max_idx;

        if max_val == 0.0 {
            return (k + 1) as i32; // Singular
        }

        // Swap rows k and max_idx
        if max_idx != k {
            for j in 0..n {
                let idx_k = layout.index(k, j, lda);
                let idx_p = layout.index(max_idx, j, lda);
                a.swap(idx_k, idx_p);
            }
        }

        // Compute multipliers and update
        let pivot = a[layout.index(k, k, lda)];
        for i in (k + 1)..m {
            let idx = layout.index(i, k, lda);
            a[idx] /= pivot;
        }

        // Update trailing submatrix: A[i,j] -= A[i,k] * A[k,j]
        for i in (k + 1)..m {
            let lik = a[layout.index(i, k, lda)];
            for j in (k + 1)..n {
                let ukj = a[layout.index(k, j, lda)];
                let idx = layout.index(i, j, lda);
                a[idx] -= lik * ukj;
            }
        }
    }

    0
}

/// Double-precision LU factorization with partial pivoting.
pub fn dgetrf(
    layout: Layout,
    m: usize,
    n: usize,
    a: &mut [f64],
    lda: usize,
    ipiv: &mut [usize],
) -> i32 {
    let min_mn = m.min(n);

    for k in 0..min_mn {
        let mut max_val = 0.0f64;
        let mut max_idx = k;
        for i in k..m {
            let val = a[layout.index(i, k, lda)].abs();
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        ipiv[k] = max_idx;

        if max_val == 0.0 {
            return (k + 1) as i32;
        }

        if max_idx != k {
            for j in 0..n {
                let idx_k = layout.index(k, j, lda);
                let idx_p = layout.index(max_idx, j, lda);
                a.swap(idx_k, idx_p);
            }
        }

        let pivot = a[layout.index(k, k, lda)];
        for i in (k + 1)..m {
            let idx = layout.index(i, k, lda);
            a[idx] /= pivot;
        }

        for i in (k + 1)..m {
            let lik = a[layout.index(i, k, lda)];
            for j in (k + 1)..n {
                let ukj = a[layout.index(k, j, lda)];
                let idx = layout.index(i, j, lda);
                a[idx] -= lik * ukj;
            }
        }
    }

    0
}

// ============================================================================
// LU Solve: solve A * X = B using LU factors from sgetrf/dgetrf
// ============================================================================

/// Single-precision LU solve: solve A * X = B.
///
/// A must already be LU-factored by `sgetrf`.
/// B is overwritten with the solution X.
///
/// - `n` - order of matrix A
/// - `nrhs` - number of right-hand side columns
pub fn sgetrs(
    layout: Layout,
    n: usize,
    nrhs: usize,
    a: &[f32],
    lda: usize,
    ipiv: &[usize],
    b: &mut [f32],
    ldb: usize,
) {
    // Apply row interchanges to B
    for k in 0..n {
        if ipiv[k] != k {
            for j in 0..nrhs {
                let idx_k = layout.index(k, j, ldb);
                let idx_p = layout.index(ipiv[k], j, ldb);
                b.swap(idx_k, idx_p);
            }
        }
    }

    // Forward substitution: L * Y = P * B
    for k in 0..n {
        for i in (k + 1)..n {
            let lik = a[layout.index(i, k, lda)];
            for j in 0..nrhs {
                let bkj = b[layout.index(k, j, ldb)];
                let idx = layout.index(i, j, ldb);
                b[idx] -= lik * bkj;
            }
        }
    }

    // Back substitution: U * X = Y
    for k in (0..n).rev() {
        let ukk = a[layout.index(k, k, lda)];
        for j in 0..nrhs {
            let idx = layout.index(k, j, ldb);
            b[idx] /= ukk;
        }
        for i in 0..k {
            let uik = a[layout.index(i, k, lda)];
            for j in 0..nrhs {
                let bkj = b[layout.index(k, j, ldb)];
                let idx = layout.index(i, j, ldb);
                b[idx] -= uik * bkj;
            }
        }
    }
}

/// Double-precision LU solve.
pub fn dgetrs(
    layout: Layout,
    n: usize,
    nrhs: usize,
    a: &[f64],
    lda: usize,
    ipiv: &[usize],
    b: &mut [f64],
    ldb: usize,
) {
    for k in 0..n {
        if ipiv[k] != k {
            for j in 0..nrhs {
                let idx_k = layout.index(k, j, ldb);
                let idx_p = layout.index(ipiv[k], j, ldb);
                b.swap(idx_k, idx_p);
            }
        }
    }

    for k in 0..n {
        for i in (k + 1)..n {
            let lik = a[layout.index(i, k, lda)];
            for j in 0..nrhs {
                let bkj = b[layout.index(k, j, ldb)];
                let idx = layout.index(i, j, ldb);
                b[idx] -= lik * bkj;
            }
        }
    }

    for k in (0..n).rev() {
        let ukk = a[layout.index(k, k, lda)];
        for j in 0..nrhs {
            let idx = layout.index(k, j, ldb);
            b[idx] /= ukk;
        }
        for i in 0..k {
            let uik = a[layout.index(i, k, lda)];
            for j in 0..nrhs {
                let bkj = b[layout.index(k, j, ldb)];
                let idx = layout.index(i, j, ldb);
                b[idx] -= uik * bkj;
            }
        }
    }
}

// ============================================================================
// Cholesky Factorization: A = L * L^T (or U^T * U)
// ============================================================================

/// Single-precision Cholesky factorization.
///
/// Factors symmetric positive-definite A into L * L^T (lower) or U^T * U (upper).
/// The specified triangle of A is overwritten with the factor.
///
/// Returns 0 on success, > 0 if A is not positive definite.
pub fn spotrf(
    layout: Layout,
    uplo: rustynum_core::layout::Uplo,
    n: usize,
    a: &mut [f32],
    lda: usize,
) -> i32 {
    match uplo {
        rustynum_core::layout::Uplo::Lower => {
            for j in 0..n {
                let mut sum = a[layout.index(j, j, lda)];
                for k in 0..j {
                    let ljk = a[layout.index(j, k, lda)];
                    sum -= ljk * ljk;
                }
                if sum <= 0.0 {
                    return (j + 1) as i32;
                }
                let ljj = sum.sqrt();
                a[layout.index(j, j, lda)] = ljj;

                for i in (j + 1)..n {
                    let mut sum = a[layout.index(i, j, lda)];
                    for k in 0..j {
                        sum -= a[layout.index(i, k, lda)] * a[layout.index(j, k, lda)];
                    }
                    a[layout.index(i, j, lda)] = sum / ljj;
                }
            }
        }
        rustynum_core::layout::Uplo::Upper => {
            for j in 0..n {
                let mut sum = a[layout.index(j, j, lda)];
                for k in 0..j {
                    let ukj = a[layout.index(k, j, lda)];
                    sum -= ukj * ukj;
                }
                if sum <= 0.0 {
                    return (j + 1) as i32;
                }
                let ujj = sum.sqrt();
                a[layout.index(j, j, lda)] = ujj;

                for i in (j + 1)..n {
                    let mut sum = a[layout.index(j, i, lda)];
                    for k in 0..j {
                        sum -= a[layout.index(k, j, lda)] * a[layout.index(k, i, lda)];
                    }
                    a[layout.index(j, i, lda)] = sum / ujj;
                }
            }
        }
    }
    0
}

/// Double-precision Cholesky factorization.
pub fn dpotrf(
    layout: Layout,
    uplo: rustynum_core::layout::Uplo,
    n: usize,
    a: &mut [f64],
    lda: usize,
) -> i32 {
    match uplo {
        rustynum_core::layout::Uplo::Lower => {
            for j in 0..n {
                let mut sum = a[layout.index(j, j, lda)];
                for k in 0..j {
                    let ljk = a[layout.index(j, k, lda)];
                    sum -= ljk * ljk;
                }
                if sum <= 0.0 {
                    return (j + 1) as i32;
                }
                let ljj = sum.sqrt();
                a[layout.index(j, j, lda)] = ljj;

                for i in (j + 1)..n {
                    let mut sum = a[layout.index(i, j, lda)];
                    for k in 0..j {
                        sum -= a[layout.index(i, k, lda)] * a[layout.index(j, k, lda)];
                    }
                    a[layout.index(i, j, lda)] = sum / ljj;
                }
            }
        }
        rustynum_core::layout::Uplo::Upper => {
            for j in 0..n {
                let mut sum = a[layout.index(j, j, lda)];
                for k in 0..j {
                    let ukj = a[layout.index(k, j, lda)];
                    sum -= ukj * ukj;
                }
                if sum <= 0.0 {
                    return (j + 1) as i32;
                }
                let ujj = sum.sqrt();
                a[layout.index(j, j, lda)] = ujj;

                for i in (j + 1)..n {
                    let mut sum = a[layout.index(j, i, lda)];
                    for k in 0..j {
                        sum -= a[layout.index(k, j, lda)] * a[layout.index(k, i, lda)];
                    }
                    a[layout.index(j, i, lda)] = sum / ujj;
                }
            }
        }
    }
    0
}

// ============================================================================
// Cholesky Solve: A * X = B using Cholesky factors
// ============================================================================

/// Single-precision Cholesky solve: solve A * X = B.
/// A must already be Cholesky-factored by `spotrf`.
pub fn spotrs(
    layout: Layout,
    uplo: rustynum_core::layout::Uplo,
    n: usize,
    nrhs: usize,
    a: &[f32],
    lda: usize,
    b: &mut [f32],
    ldb: usize,
) {
    match uplo {
        rustynum_core::layout::Uplo::Lower => {
            // Forward substitution: L * Y = B
            for k in 0..n {
                let lkk = a[layout.index(k, k, lda)];
                for j in 0..nrhs {
                    let idx = layout.index(k, j, ldb);
                    b[idx] /= lkk;
                }
                for i in (k + 1)..n {
                    let lik = a[layout.index(i, k, lda)];
                    for j in 0..nrhs {
                        let bkj = b[layout.index(k, j, ldb)];
                        let idx = layout.index(i, j, ldb);
                        b[idx] -= lik * bkj;
                    }
                }
            }
            // Back substitution: L^T * X = Y
            for k in (0..n).rev() {
                let lkk = a[layout.index(k, k, lda)];
                for j in 0..nrhs {
                    let idx = layout.index(k, j, ldb);
                    b[idx] /= lkk;
                }
                for i in 0..k {
                    let lki = a[layout.index(k, i, lda)];
                    for j in 0..nrhs {
                        let bkj = b[layout.index(k, j, ldb)];
                        let idx = layout.index(i, j, ldb);
                        b[idx] -= lki * bkj;
                    }
                }
            }
        }
        rustynum_core::layout::Uplo::Upper => {
            // Forward substitution: U^T * Y = B
            for k in 0..n {
                let ukk = a[layout.index(k, k, lda)];
                for j in 0..nrhs {
                    let idx = layout.index(k, j, ldb);
                    b[idx] /= ukk;
                }
                for i in (k + 1)..n {
                    let uki = a[layout.index(k, i, lda)];
                    for j in 0..nrhs {
                        let bkj = b[layout.index(k, j, ldb)];
                        let idx = layout.index(i, j, ldb);
                        b[idx] -= uki * bkj;
                    }
                }
            }
            // Back substitution: U * X = Y
            for k in (0..n).rev() {
                let ukk = a[layout.index(k, k, lda)];
                for j in 0..nrhs {
                    let idx = layout.index(k, j, ldb);
                    b[idx] /= ukk;
                }
                for i in 0..k {
                    let uik = a[layout.index(i, k, lda)];
                    for j in 0..nrhs {
                        let bkj = b[layout.index(k, j, ldb)];
                        let idx = layout.index(i, j, ldb);
                        b[idx] -= uik * bkj;
                    }
                }
            }
        }
    }
}

// ============================================================================
// QR Factorization: A = Q * R (Householder reflections)
// ============================================================================

/// Single-precision QR factorization using Householder reflections.
///
/// Factors M x N matrix A into Q * R.
/// - A is overwritten: upper triangle = R, below diagonal = Householder vectors.
/// - `tau` receives the Householder scalar factors (length min(m, n)).
///
/// Returns 0 on success.
pub fn sgeqrf(
    layout: Layout,
    m: usize,
    n: usize,
    a: &mut [f32],
    lda: usize,
    tau: &mut [f32],
) -> i32 {
    let min_mn = m.min(n);

    for k in 0..min_mn {
        // Compute Householder reflector for column k
        let mut norm_sq = 0.0f32;
        for i in k..m {
            let v = a[layout.index(i, k, lda)];
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();

        if norm == 0.0 {
            tau[k] = 0.0;
            continue;
        }

        let akk = a[layout.index(k, k, lda)];
        let sign = if akk >= 0.0 { 1.0 } else { -1.0 };
        let alpha = -sign * norm;

        // tau = 2 / (v^T * v) where v[k] = akk - alpha, v[i] = a[i,k] for i > k
        let beta = akk - alpha;
        if beta.abs() < 1e-30 {
            tau[k] = 0.0;
            a[layout.index(k, k, lda)] = alpha;
            continue;
        }

        // Scale v so v[k] = 1
        for i in (k + 1)..m {
            let idx = layout.index(i, k, lda);
            a[idx] /= beta;
        }

        tau[k] = -beta / alpha;
        a[layout.index(k, k, lda)] = alpha;

        // Apply reflector to trailing columns: A[k:m, k+1:n] -= tau * v * v^T * A[k:m, k+1:n]
        for j in (k + 1)..n {
            // Compute w = v^T * A[:, j]
            let mut w = a[layout.index(k, j, lda)]; // v[k] = 1
            for i in (k + 1)..m {
                w += a[layout.index(i, k, lda)] * a[layout.index(i, j, lda)];
            }
            w *= tau[k];

            // Update A[:, j] -= w * v
            a[layout.index(k, j, lda)] -= w;
            for i in (k + 1)..m {
                let vik = a[layout.index(i, k, lda)];
                let idx = layout.index(i, j, lda);
                a[idx] -= w * vik;
            }
        }
    }

    0
}

/// Double-precision QR factorization.
pub fn dgeqrf(
    layout: Layout,
    m: usize,
    n: usize,
    a: &mut [f64],
    lda: usize,
    tau: &mut [f64],
) -> i32 {
    let min_mn = m.min(n);

    for k in 0..min_mn {
        let mut norm_sq = 0.0f64;
        for i in k..m {
            let v = a[layout.index(i, k, lda)];
            norm_sq += v * v;
        }
        let norm = norm_sq.sqrt();

        if norm == 0.0 {
            tau[k] = 0.0;
            continue;
        }

        let akk = a[layout.index(k, k, lda)];
        let sign = if akk >= 0.0 { 1.0 } else { -1.0 };
        let alpha = -sign * norm;

        let beta = akk - alpha;
        if beta.abs() < 1e-60 {
            tau[k] = 0.0;
            a[layout.index(k, k, lda)] = alpha;
            continue;
        }

        for i in (k + 1)..m {
            let idx = layout.index(i, k, lda);
            a[idx] /= beta;
        }

        tau[k] = -beta / alpha;
        a[layout.index(k, k, lda)] = alpha;

        for j in (k + 1)..n {
            let mut w = a[layout.index(k, j, lda)];
            for i in (k + 1)..m {
                w += a[layout.index(i, k, lda)] * a[layout.index(i, j, lda)];
            }
            w *= tau[k];

            a[layout.index(k, j, lda)] -= w;
            for i in (k + 1)..m {
                let vik = a[layout.index(i, k, lda)];
                let idx = layout.index(i, j, lda);
                a[idx] -= w * vik;
            }
        }
    }

    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgetrf_and_solve() {
        // A = [[2, 1], [1, 3]]
        let mut a = vec![2.0f32, 1.0, 1.0, 3.0];
        let mut ipiv = vec![0usize; 2];
        let info = sgetrf(Layout::RowMajor, 2, 2, &mut a, 2, &mut ipiv);
        assert_eq!(info, 0, "LU factorization should succeed");

        // Solve A * x = b where b = [5, 7]
        // Expected: x = [1.6, 1.8]
        let mut b = vec![5.0f32, 7.0];
        sgetrs(Layout::RowMajor, 2, 1, &a, 2, &ipiv, &mut b, 1);
        assert!((b[0] - 1.6).abs() < 1e-5, "x[0] = {}", b[0]);
        assert!((b[1] - 1.8).abs() < 1e-5, "x[1] = {}", b[1]);
    }

    #[test]
    fn test_dgetrf_3x3() {
        // A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]
        let mut a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0];
        let mut ipiv = vec![0usize; 3];
        let info = dgetrf(Layout::RowMajor, 3, 3, &mut a, 3, &mut ipiv);
        assert_eq!(info, 0);
    }

    #[test]
    fn test_spotrf_cholesky() {
        // A = [[4, 2], [2, 3]] (symmetric positive definite)
        let mut a = vec![4.0f32, 2.0, 2.0, 3.0];
        let info = spotrf(Layout::RowMajor, rustynum_core::layout::Uplo::Lower, 2, &mut a, 2);
        assert_eq!(info, 0);

        // L[0,0] = sqrt(4) = 2
        assert!((a[0] - 2.0).abs() < 1e-6);
        // L[1,0] = 2/2 = 1
        assert!((a[2] - 1.0).abs() < 1e-6);
        // L[1,1] = sqrt(3 - 1) = sqrt(2)
        assert!((a[3] - 2.0f32.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_spotrf_not_positive_definite() {
        // A = [[1, 2], [2, 1]] — not positive definite
        let mut a = vec![1.0f32, 2.0, 2.0, 1.0];
        let info = spotrf(Layout::RowMajor, rustynum_core::layout::Uplo::Lower, 2, &mut a, 2);
        assert!(info > 0, "Should detect non-positive-definite matrix");
    }

    #[test]
    fn test_sgeqrf() {
        // A = [[1, 1], [0, 1], [1, 0]] (3x2)
        let mut a = vec![1.0f32, 1.0, 0.0, 1.0, 1.0, 0.0];
        let mut tau = vec![0.0f32; 2];
        let info = sgeqrf(Layout::RowMajor, 3, 2, &mut a, 2, &mut tau);
        assert_eq!(info, 0);
        // R should be in upper triangle
        // R[0,0] should be -sqrt(2) (or sqrt(2) depending on sign convention)
        assert!(a[0].abs() > 1.0, "R[0,0] should be nonzero: {}", a[0]);
    }
}
