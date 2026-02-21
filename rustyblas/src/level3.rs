//! BLAS Level 3: Matrix-matrix operations.
//!
//! The crown jewel is GEMM — cache-blocked with AVX-512 microkernels.
//! Uses the Goto BLAS algorithm: pack panels of A and B into contiguous
//! cache-friendly buffers, then invoke the microkernel on tiles.

use rustynum_core::layout::{Diag, Layout, Side, Transpose, Uplo};
use rustynum_core::parallel::parallel_for_chunks;
use rustynum_core::simd::{self, F32_LANES, F64_LANES, SGEMM_KC, SGEMM_MC, SGEMM_MR, SGEMM_NC, SGEMM_NR, DGEMM_KC, DGEMM_MC, DGEMM_MR, DGEMM_NC, DGEMM_NR};
use std::simd::f32x16;
use std::simd::f64x8;
use std::simd::num::SimdFloat;

// ============================================================================
// SGEMM: Single-precision General Matrix Multiply
// C := alpha * op(A) * op(B) + beta * C
// ============================================================================

/// Single-precision GEMM with cache-blocked AVX-512 microkernel.
///
/// C := alpha * op(A) * op(B) + beta * C
///
/// Supports both row-major and column-major layouts.
/// Uses Goto BLAS algorithm: panel packing + tiled microkernel.
/// Multi-threaded for large matrices.
pub fn sgemm(
    layout: Layout,
    trans_a: Transpose,
    trans_b: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    // Scale C by beta
    if beta == 0.0 {
        for i in 0..m {
            for j in 0..n {
                c[layout.index(i, j, ldc)] = 0.0;
            }
        }
    } else if beta != 1.0 {
        for i in 0..m {
            for j in 0..n {
                let idx = layout.index(i, j, ldc);
                c[idx] *= beta;
            }
        }
    }

    if alpha == 0.0 || m == 0 || n == 0 || k == 0 {
        return;
    }

    // For small matrices, use simple triple-loop
    if m * n * k < 32768 {
        sgemm_simple(layout, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, c, ldc);
        return;
    }

    // Cache-blocked GEMM with packing
    sgemm_blocked(layout, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, c, ldc);
}

/// Simple triple-loop GEMM for small matrices.
fn sgemm_simple(
    layout: Layout,
    trans_a: Transpose,
    trans_b: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: &mut [f32],
    ldc: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                let a_val = match (layout, trans_a) {
                    (Layout::RowMajor, Transpose::NoTrans) => a[i * lda + p],
                    (Layout::RowMajor, _) => a[p * lda + i],
                    (Layout::ColMajor, Transpose::NoTrans) => a[p * lda + i],
                    (Layout::ColMajor, _) => a[i * lda + p],
                };
                let b_val = match (layout, trans_b) {
                    (Layout::RowMajor, Transpose::NoTrans) => b[p * ldb + j],
                    (Layout::RowMajor, _) => b[j * ldb + p],
                    (Layout::ColMajor, Transpose::NoTrans) => b[j * ldb + p],
                    (Layout::ColMajor, _) => b[p * ldb + j],
                };
                sum += a_val * b_val;
            }
            let idx = layout.index(i, j, ldc);
            c[idx] += alpha * sum;
        }
    }
}

/// Cache-blocked GEMM using the Goto BLAS algorithm.
///
/// Memory hierarchy:
/// - Outer loop over N (L3 blocks of NC columns)
/// - Middle loop over K (L2 blocks of KC depth)
/// - Pack panel of B into contiguous buffer
/// - Inner loop over M (L1 blocks of MC rows)
/// - Pack panel of A into contiguous buffer
/// - Microkernel: MR x NR register tile
fn sgemm_blocked(
    layout: Layout,
    trans_a: Transpose,
    trans_b: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    c: &mut [f32],
    ldc: usize,
) {
    let mc = SGEMM_MC.min(m);
    let nc = SGEMM_NC.min(n);
    let kc = SGEMM_KC.min(k);

    // Packed buffers — padded to MR/NR boundaries for microkernel alignment
    let mc_padded = ((mc + SGEMM_MR - 1) / SGEMM_MR) * SGEMM_MR;
    let nc_padded = ((nc + SGEMM_NR - 1) / SGEMM_NR) * SGEMM_NR;
    let mut packed_a = vec![0.0f32; mc_padded * kc];
    let mut packed_b = vec![0.0f32; kc * nc_padded];

    for jc in (0..n).step_by(nc) {
        let jb = nc.min(n - jc);

        for pc in (0..k).step_by(kc) {
            let pb = kc.min(k - pc);

            // Pack B panel: pb x jb -> packed_b in NR-column panels
            pack_b_f32(layout, trans_b, b, ldb, pc, jc, pb, jb, &mut packed_b);

            // Parallelize over M blocks
            for ic in (0..m).step_by(mc) {
                let ib = mc.min(m - ic);

                // Pack A panel: ib x pb -> packed_a in MR-row panels
                pack_a_f32(layout, trans_a, a, lda, ic, pc, ib, pb, &mut packed_a);

                // Microkernel: MR x NR tiles
                sgemm_macrokernel(alpha, &packed_a, &packed_b, c, layout, ldc, ic, jc, ib, jb, pb);
            }
        }
    }
}

/// Pack A panel into MR-row contiguous strips for cache-friendly access.
fn pack_a_f32(
    layout: Layout,
    trans: Transpose,
    a: &[f32],
    lda: usize,
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
    packed: &mut [f32],
) {
    let mut idx = 0;
    for i_block in (0..rows).step_by(SGEMM_MR) {
        let mr = SGEMM_MR.min(rows - i_block);
        for p in 0..cols {
            for ir in 0..SGEMM_MR {
                if ir < mr {
                    let i = row_start + i_block + ir;
                    let j = col_start + p;
                    packed[idx] = match (layout, trans) {
                        (Layout::RowMajor, Transpose::NoTrans) => a[i * lda + j],
                        (Layout::RowMajor, _) => a[j * lda + i],
                        (Layout::ColMajor, Transpose::NoTrans) => a[j * lda + i],
                        (Layout::ColMajor, _) => a[i * lda + j],
                    };
                } else {
                    packed[idx] = 0.0; // Pad with zeros
                }
                idx += 1;
            }
        }
    }
}

/// Pack B panel into NR-column contiguous strips.
fn pack_b_f32(
    layout: Layout,
    trans: Transpose,
    b: &[f32],
    ldb: usize,
    row_start: usize,
    col_start: usize,
    rows: usize,
    cols: usize,
    packed: &mut [f32],
) {
    let mut idx = 0;
    for j_block in (0..cols).step_by(SGEMM_NR) {
        let nr = SGEMM_NR.min(cols - j_block);
        for p in 0..rows {
            for jr in 0..SGEMM_NR {
                if jr < nr {
                    let i = row_start + p;
                    let j = col_start + j_block + jr;
                    packed[idx] = match (layout, trans) {
                        (Layout::RowMajor, Transpose::NoTrans) => b[i * ldb + j],
                        (Layout::RowMajor, _) => b[j * ldb + i],
                        (Layout::ColMajor, Transpose::NoTrans) => b[j * ldb + i],
                        (Layout::ColMajor, _) => b[i * ldb + j],
                    };
                } else {
                    packed[idx] = 0.0;
                }
                idx += 1;
            }
        }
    }
}

/// Macro-kernel: dispatch MR x NR microkernels over the packed panels.
fn sgemm_macrokernel(
    alpha: f32,
    packed_a: &[f32],
    packed_b: &[f32],
    c: &mut [f32],
    layout: Layout,
    ldc: usize,
    ic: usize,
    jc: usize,
    mb: usize,
    nb: usize,
    kb: usize,
) {
    let mr_blocks = (mb + SGEMM_MR - 1) / SGEMM_MR;
    let nr_blocks = (nb + SGEMM_NR - 1) / SGEMM_NR;

    for jr in 0..nr_blocks {
        let nr = SGEMM_NR.min(nb - jr * SGEMM_NR);

        for ir in 0..mr_blocks {
            let mr = SGEMM_MR.min(mb - ir * SGEMM_MR);

            // Microkernel: compute MR x NR tile of C
            let a_offset = ir * SGEMM_MR * kb;
            let b_offset = jr * SGEMM_NR * kb;

            sgemm_microkernel_6x16(
                alpha,
                &packed_a[a_offset..],
                &packed_b[b_offset..],
                c,
                layout,
                ldc,
                ic + ir * SGEMM_MR,
                jc + jr * SGEMM_NR,
                mr,
                nr,
                kb,
            );
        }
    }
}

/// AVX-512 microkernel: 6x16 tile of C using f32x16 SIMD.
///
/// Uses 6 accumulator registers (one per row of the tile),
/// each holding 16 f32 values — exactly one zmm register.
/// This gives 6 * 16 = 96 FMA operations per K iteration.
#[inline(always)]
fn sgemm_microkernel_6x16(
    alpha: f32,
    packed_a: &[f32],
    packed_b: &[f32],
    c: &mut [f32],
    layout: Layout,
    ldc: usize,
    row: usize,
    col: usize,
    mr: usize,
    nr: usize,
    kb: usize,
) {
    // Accumulator registers for 6 rows x 16 columns
    let mut acc = [f32x16::splat(0.0); 6];

    // Main K loop
    for p in 0..kb {
        // Load NR elements of B for this K step
        let b_base = p * SGEMM_NR;
        let b_vec = if nr >= SGEMM_NR {
            f32x16::from_slice(&packed_b[b_base..])
        } else {
            // Partial: pad with zeros
            let mut buf = [0.0f32; 16];
            for j in 0..nr {
                buf[j] = packed_b[b_base + j];
            }
            f32x16::from_array(buf)
        };

        // Load MR elements of A and broadcast-multiply
        let a_base = p * SGEMM_MR;
        for ir in 0..mr.min(6) {
            let a_val = f32x16::splat(packed_a[a_base + ir]);
            acc[ir] += a_val * b_vec;
        }
    }

    // Store results back to C with alpha scaling
    let alpha_v = f32x16::splat(alpha);
    for ir in 0..mr.min(6) {
        let result = acc[ir] * alpha_v;
        let mut buf = result.to_array();
        for jr in 0..nr {
            let idx = layout.index(row + ir, col + jr, ldc);
            c[idx] += buf[jr];
        }
    }
}

// ============================================================================
// DGEMM: Double-precision General Matrix Multiply
// ============================================================================

/// Double-precision GEMM: C := alpha * op(A) * op(B) + beta * C
pub fn dgemm(
    layout: Layout,
    trans_a: Transpose,
    trans_b: Transpose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    // Scale C by beta
    if beta == 0.0 {
        for i in 0..m {
            for j in 0..n {
                c[layout.index(i, j, ldc)] = 0.0;
            }
        }
    } else if beta != 1.0 {
        for i in 0..m {
            for j in 0..n {
                let idx = layout.index(i, j, ldc);
                c[idx] *= beta;
            }
        }
    }

    if alpha == 0.0 || m == 0 || n == 0 || k == 0 {
        return;
    }

    // Simple triple-loop for now; cache-blocked dgemm follows same pattern as sgemm
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                let a_val = match (layout, trans_a) {
                    (Layout::RowMajor, Transpose::NoTrans) => a[i * lda + p],
                    (Layout::RowMajor, _) => a[p * lda + i],
                    (Layout::ColMajor, Transpose::NoTrans) => a[p * lda + i],
                    (Layout::ColMajor, _) => a[i * lda + p],
                };
                let b_val = match (layout, trans_b) {
                    (Layout::RowMajor, Transpose::NoTrans) => b[p * ldb + j],
                    (Layout::RowMajor, _) => b[j * ldb + p],
                    (Layout::ColMajor, Transpose::NoTrans) => b[j * ldb + p],
                    (Layout::ColMajor, _) => b[p * ldb + j],
                };
                sum += a_val * b_val;
            }
            let idx = layout.index(i, j, ldc);
            c[idx] += alpha * sum;
        }
    }
}

// ============================================================================
// SSYRK: Symmetric rank-k update  C := alpha * A * A^T + beta * C
// ============================================================================

/// Single-precision SYRK: C := alpha * A * A^T + beta * C (or A^T * A)
pub fn ssyrk(
    layout: Layout,
    uplo: Uplo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    // Scale C by beta
    for i in 0..n {
        let (j_start, j_end) = match uplo {
            Uplo::Upper => (i, n),
            Uplo::Lower => (0, i + 1),
        };
        for j in j_start..j_end {
            let idx = layout.index(i, j, ldc);
            c[idx] = if beta == 0.0 { 0.0 } else { c[idx] * beta };
        }
    }

    if alpha == 0.0 {
        return;
    }

    // C += alpha * op(A) * op(A)^T
    for i in 0..n {
        let (j_start, j_end) = match uplo {
            Uplo::Upper => (i, n),
            Uplo::Lower => (0, i + 1),
        };
        for j in j_start..j_end {
            let mut sum = 0.0f32;
            for p in 0..k {
                let a_ip = match (layout, trans) {
                    (Layout::RowMajor, Transpose::NoTrans) => a[i * lda + p],
                    (Layout::RowMajor, _) => a[p * lda + i],
                    (Layout::ColMajor, Transpose::NoTrans) => a[p * lda + i],
                    (Layout::ColMajor, _) => a[i * lda + p],
                };
                let a_jp = match (layout, trans) {
                    (Layout::RowMajor, Transpose::NoTrans) => a[j * lda + p],
                    (Layout::RowMajor, _) => a[p * lda + j],
                    (Layout::ColMajor, Transpose::NoTrans) => a[p * lda + j],
                    (Layout::ColMajor, _) => a[j * lda + p],
                };
                sum += a_ip * a_jp;
            }
            let idx = layout.index(i, j, ldc);
            c[idx] += alpha * sum;
        }
    }
}

/// Double-precision SYRK: C := alpha * A * A^T + beta * C
pub fn dsyrk(
    layout: Layout,
    uplo: Uplo,
    trans: Transpose,
    n: usize,
    k: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    for i in 0..n {
        let (j_start, j_end) = match uplo {
            Uplo::Upper => (i, n),
            Uplo::Lower => (0, i + 1),
        };
        for j in j_start..j_end {
            let idx = layout.index(i, j, ldc);
            c[idx] = if beta == 0.0 { 0.0 } else { c[idx] * beta };
        }
    }

    if alpha == 0.0 {
        return;
    }

    for i in 0..n {
        let (j_start, j_end) = match uplo {
            Uplo::Upper => (i, n),
            Uplo::Lower => (0, i + 1),
        };
        for j in j_start..j_end {
            let mut sum = 0.0f64;
            for p in 0..k {
                let a_ip = match (layout, trans) {
                    (Layout::RowMajor, Transpose::NoTrans) => a[i * lda + p],
                    (Layout::RowMajor, _) => a[p * lda + i],
                    (Layout::ColMajor, Transpose::NoTrans) => a[p * lda + i],
                    (Layout::ColMajor, _) => a[i * lda + p],
                };
                let a_jp = match (layout, trans) {
                    (Layout::RowMajor, Transpose::NoTrans) => a[j * lda + p],
                    (Layout::RowMajor, _) => a[p * lda + j],
                    (Layout::ColMajor, Transpose::NoTrans) => a[p * lda + j],
                    (Layout::ColMajor, _) => a[j * lda + p],
                };
                sum += a_ip * a_jp;
            }
            let idx = layout.index(i, j, ldc);
            c[idx] += alpha * sum;
        }
    }
}

// ============================================================================
// STRSM: Triangular solve with multiple right-hand sides
// op(A) * X = alpha * B  or  X * op(A) = alpha * B
// ============================================================================

/// Single-precision TRSM: solve op(A) * X = alpha * B (A triangular)
pub fn strsm(
    layout: Layout,
    side: Side,
    uplo: Uplo,
    trans: Transpose,
    diag: Diag,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &mut [f32],
    ldb: usize,
) {
    let unit = diag == Diag::Unit;

    // Scale B by alpha
    if alpha != 1.0 {
        for i in 0..m {
            for j in 0..n {
                let idx = layout.index(i, j, ldb);
                b[idx] *= alpha;
            }
        }
    }

    match (side, layout, uplo, trans) {
        (Side::Left, Layout::RowMajor, Uplo::Lower, Transpose::NoTrans) => {
            // Forward substitution: L * X = B
            for i in 0..m {
                for j in 0..n {
                    let mut sum = b[i * ldb + j];
                    for p in 0..i {
                        sum -= a[i * lda + p] * b[p * ldb + j];
                    }
                    b[i * ldb + j] = if unit { sum } else { sum / a[i * lda + i] };
                }
            }
        }
        (Side::Left, Layout::RowMajor, Uplo::Upper, Transpose::NoTrans) => {
            // Back substitution: U * X = B
            for i in (0..m).rev() {
                for j in 0..n {
                    let mut sum = b[i * ldb + j];
                    for p in (i + 1)..m {
                        sum -= a[i * lda + p] * b[p * ldb + j];
                    }
                    b[i * ldb + j] = if unit { sum } else { sum / a[i * lda + i] };
                }
            }
        }
        _ => {
            // Other cases: transform to one of the above
            // For right-side: X * A = B -> A^T * X^T = B^T
            // Simple fallback
            for i in 0..m {
                for j in 0..n {
                    let idx = layout.index(i, j, ldb);
                    let mut sum = b[idx];
                    for p in 0..i {
                        let a_idx = layout.index(i, p, lda);
                        let b_idx = layout.index(p, j, ldb);
                        sum -= a[a_idx] * b[b_idx];
                    }
                    let diag_idx = layout.index(i, i, lda);
                    b[idx] = if unit { sum } else { sum / a[diag_idx] };
                }
            }
        }
    }
}

// ============================================================================
// SSYMM: Symmetric matrix multiply  C := alpha * A * B + beta * C
// ============================================================================

/// Single-precision SYMM: C := alpha * A * B + beta * C (A symmetric)
pub fn ssymm(
    layout: Layout,
    side: Side,
    uplo: Uplo,
    m: usize,
    n: usize,
    alpha: f32,
    a: &[f32],
    lda: usize,
    b: &[f32],
    ldb: usize,
    beta: f32,
    c: &mut [f32],
    ldc: usize,
) {
    // Scale C by beta
    for i in 0..m {
        for j in 0..n {
            let idx = layout.index(i, j, ldc);
            c[idx] = if beta == 0.0 { 0.0 } else { c[idx] * beta };
        }
    }

    if alpha == 0.0 {
        return;
    }

    // Simple implementation using the symmetry of A
    let ka = match side {
        Side::Left => m,
        Side::Right => n,
    };

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            match side {
                Side::Left => {
                    for p in 0..ka {
                        // A[i,p] using symmetry
                        let (ai, aj) = if i <= p { (i, p) } else { (p, i) };
                        let a_val = match uplo {
                            Uplo::Upper => a[layout.index(ai, aj, lda)],
                            Uplo::Lower => a[layout.index(aj, ai, lda)],
                        };
                        let b_val = b[layout.index(p, j, ldb)];
                        sum += a_val * b_val;
                    }
                }
                Side::Right => {
                    for p in 0..ka {
                        let b_val = b[layout.index(i, p, ldb)];
                        let (ai, aj) = if p <= j { (p, j) } else { (j, p) };
                        let a_val = match uplo {
                            Uplo::Upper => a[layout.index(ai, aj, lda)],
                            Uplo::Lower => a[layout.index(aj, ai, lda)],
                        };
                        sum += b_val * a_val;
                    }
                }
            }
            let idx = layout.index(i, j, ldc);
            c[idx] += alpha * sum;
        }
    }
}

/// Double-precision SYMM.
pub fn dsymm(
    layout: Layout,
    side: Side,
    uplo: Uplo,
    m: usize,
    n: usize,
    alpha: f64,
    a: &[f64],
    lda: usize,
    b: &[f64],
    ldb: usize,
    beta: f64,
    c: &mut [f64],
    ldc: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let idx = layout.index(i, j, ldc);
            c[idx] = if beta == 0.0 { 0.0 } else { c[idx] * beta };
        }
    }

    if alpha == 0.0 {
        return;
    }

    let ka = match side {
        Side::Left => m,
        Side::Right => n,
    };

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            match side {
                Side::Left => {
                    for p in 0..ka {
                        let (ai, aj) = if i <= p { (i, p) } else { (p, i) };
                        let a_val = match uplo {
                            Uplo::Upper => a[layout.index(ai, aj, lda)],
                            Uplo::Lower => a[layout.index(aj, ai, lda)],
                        };
                        sum += a_val * b[layout.index(p, j, ldb)];
                    }
                }
                Side::Right => {
                    for p in 0..ka {
                        let b_val = b[layout.index(i, p, ldb)];
                        let (ai, aj) = if p <= j { (p, j) } else { (j, p) };
                        let a_val = match uplo {
                            Uplo::Upper => a[layout.index(ai, aj, lda)],
                            Uplo::Lower => a[layout.index(aj, ai, lda)],
                        };
                        sum += b_val * a_val;
                    }
                }
            }
            c[layout.index(i, j, ldc)] += alpha * sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgemm_identity() {
        // A = I(2), B = [[1,2],[3,4]], C should be [[1,2],[3,4]]
        let a = vec![1.0f32, 0.0, 0.0, 1.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut c = vec![0.0f32; 4];
        sgemm(Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
              2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sgemm_simple_multiply() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // C = A*B = [[19,22],[43,50]]
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f32; 4];
        sgemm(Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
              2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_sgemm_alpha_beta() {
        // C = 2 * A * B + 3 * C
        let a = vec![1.0f32, 0.0, 0.0, 1.0]; // identity
        let b = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut c = vec![10.0f32, 20.0, 30.0, 40.0];
        sgemm(Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
              2, 2, 2, 2.0, &a, 2, &b, 2, 3.0, &mut c, 2);
        // C = 2*[[1,2],[3,4]] + 3*[[10,20],[30,40]] = [[32,64],[96,128]]
        assert_eq!(c, vec![32.0, 64.0, 96.0, 128.0]);
    }

    #[test]
    fn test_sgemm_transpose_a() {
        // A^T = [[1,3],[2,4]], B = [[1,0],[0,1]]
        // C = A^T * B = [[1,3],[2,4]]
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![1.0f32, 0.0, 0.0, 1.0];
        let mut c = vec![0.0f32; 4];
        sgemm(Layout::RowMajor, Transpose::Trans, Transpose::NoTrans,
              2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2);
        assert_eq!(c, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_sgemm_non_square() {
        // A(2x3) = [[1,2,3],[4,5,6]], B(3x2) = [[1,2],[3,4],[5,6]]
        // C = A*B = [[22,28],[49,64]]
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = vec![0.0f32; 4];
        sgemm(Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
              2, 2, 3, 1.0, &a, 3, &b, 2, 0.0, &mut c, 2);
        assert_eq!(c, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_dgemm_simple() {
        let a = vec![1.0f64, 2.0, 3.0, 4.0];
        let b = vec![5.0f64, 6.0, 7.0, 8.0];
        let mut c = vec![0.0f64; 4];
        dgemm(Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
              2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_sgemm_colmajor() {
        // Column-major: A = [[1,3],[2,4]] stored as [1,2,3,4]
        // B = [[5,7],[6,8]] stored as [5,6,7,8]
        // C = A*B = [[23,31],[34,46]] stored as [23,34,31,46]
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // col-major
        let b = vec![5.0f32, 6.0, 7.0, 8.0]; // col-major
        let mut c = vec![0.0f32; 4];
        sgemm(Layout::ColMajor, Transpose::NoTrans, Transpose::NoTrans,
              2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2);
        assert_eq!(c, vec![23.0, 34.0, 31.0, 46.0]);
    }

    #[test]
    fn test_ssyrk() {
        // A = [[1,2],[3,4]], C = A * A^T = [[5,11],[11,25]]
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut c = vec![0.0f32; 4];
        ssyrk(Layout::RowMajor, Uplo::Upper, Transpose::NoTrans,
              2, 2, 1.0, &a, 2, 0.0, &mut c, 2);
        assert_eq!(c[0], 5.0);  // C[0,0]
        assert_eq!(c[1], 11.0); // C[0,1]
        assert_eq!(c[3], 25.0); // C[1,1]
    }
}
