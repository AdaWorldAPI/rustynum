//! BF16 (Brain Float 16) GEMM for ML inference workloads.
//!
//! BFloat16 uses 8-bit exponent (same range as f32) + 7-bit mantissa.
//! Key advantage: trivial conversion to/from f32 (just truncate/pad mantissa).
//!
//! On CPUs with AVX-512 BF16 support (`avx512_bf16`):
//! - `vcvtne2ps2bf16`: convert two f32x16 → one bf16x32
//! - `vdpbf16ps`: dot product bf16 pairs → f32 accumulate
//!
//! On CPUs with AMX-BF16 (`amx_bf16`):
//! - `tdpbf16ps`: tile dot product, 16x32 × 32x16 → 16x16 f32 in one instruction
//!
//! This module provides:
//! - BF16 ↔ f32 conversion
//! - BF16 GEMM with f32 accumulation
//! - Mixed-precision GEMM: inputs in BF16, output in f32

/// BFloat16 stored as raw u16 bits.
/// Layout: [1 sign][8 exponent][7 mantissa]
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct BF16(pub u16);

impl BF16 {
    /// Convert f32 → BF16 by truncating lower 16 mantissa bits.
    /// This is "round toward zero" — fast but loses precision.
    #[inline(always)]
    pub fn from_f32_truncate(v: f32) -> Self {
        BF16((v.to_bits() >> 16) as u16)
    }

    /// Convert f32 → BF16 with round-to-nearest-even.
    #[inline(always)]
    pub fn from_f32(v: f32) -> Self {
        let bits = v.to_bits();
        // Round to nearest even: add rounding bias
        let round_bit = 0x0000_8000u32; // bit 15
        let lsb = (bits >> 16) & 1;
        let rounded = bits.wrapping_add(round_bit - 1 + lsb);
        BF16((rounded >> 16) as u16)
    }

    /// Convert BF16 → f32 by padding with 16 zero bits.
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    pub const ZERO: BF16 = BF16(0);
    pub const ONE: BF16 = BF16(0x3F80); // 1.0f32 >> 16
}

impl From<f32> for BF16 {
    fn from(v: f32) -> Self {
        BF16::from_f32(v)
    }
}

impl From<BF16> for f32 {
    fn from(v: BF16) -> f32 {
        v.to_f32()
    }
}

// ============================================================================
// Bulk conversion
// ============================================================================

/// Convert f32 slice to BF16 (truncation, fastest).
pub fn f32_to_bf16_slice(src: &[f32], dst: &mut [BF16]) {
    assert!(dst.len() >= src.len());
    for i in 0..src.len() {
        dst[i] = BF16::from_f32_truncate(src[i]);
    }
}

/// Convert f32 slice to BF16 (round-to-nearest-even).
pub fn f32_to_bf16_rounded(src: &[f32], dst: &mut [BF16]) {
    assert!(dst.len() >= src.len());
    for i in 0..src.len() {
        dst[i] = BF16::from_f32(src[i]);
    }
}

/// Convert BF16 slice to f32.
pub fn bf16_to_f32_slice(src: &[BF16], dst: &mut [f32]) {
    assert!(dst.len() >= src.len());
    for i in 0..src.len() {
        dst[i] = src[i].to_f32();
    }
}

/// Allocate and convert f32 → BF16.
pub fn f32_vec_to_bf16(src: &[f32]) -> Vec<BF16> {
    src.iter().map(|&v| BF16::from_f32(v)).collect()
}

/// Allocate and convert BF16 → f32.
pub fn bf16_vec_to_f32(src: &[BF16]) -> Vec<f32> {
    src.iter().map(|v| v.to_f32()).collect()
}

// ============================================================================
// BF16 GEMM with f32 accumulation
// ============================================================================

/// BF16 GEMM: C_f32 += A_bf16 * B_bf16
///
/// Inputs are BF16 (half the memory bandwidth of f32), accumulation
/// happens in f32 for numerical stability.
///
/// A is M×K (row-major BF16), B is K×N (row-major BF16).
/// C is M×N (row-major f32).
///
/// This gives ~2x memory bandwidth improvement over f32 GEMM since
/// inputs are half the size, while maintaining f32 accumulation precision.
pub fn bf16_gemm_f32(
    a: &[BF16],
    b: &[BF16],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    assert!(a.len() >= m * k);
    assert!(b.len() >= k * n);
    assert!(c.len() >= m * n);

    // Scale C by beta
    if beta == 0.0 {
        c[..m * n].fill(0.0);
    } else if beta != 1.0 {
        for v in c[..m * n].iter_mut() {
            *v *= beta;
        }
    }

    if alpha == 0.0 || m == 0 || n == 0 || k == 0 {
        return;
    }

    // For small matrices, use simple loop
    if m * n * k < 110_000 {
        bf16_gemm_simple(a, b, c, m, n, k, alpha);
        return;
    }

    // Cache-blocked BF16 GEMM
    bf16_gemm_blocked(a, b, c, m, n, k, alpha);
}

/// Simple BF16 GEMM with f32 accumulation.
fn bf16_gemm_simple(
    a: &[BF16],
    b: &[BF16],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
) {
    // Transpose B for sequential access
    let mut b_t = vec![BF16::ZERO; n * k];
    for p in 0..k {
        for j in 0..n {
            b_t[j * k + p] = b[p * n + j];
        }
    }

    for i in 0..m {
        let a_row = &a[i * k..];
        for j in 0..n {
            let b_col = &b_t[j * k..];

            // Accumulate in f32 using 2-wide unrolling
            let mut acc0 = 0.0f32;
            let mut acc1 = 0.0f32;
            let k2 = k / 2 * 2;

            for p in (0..k2).step_by(2) {
                acc0 += a_row[p].to_f32() * b_col[p].to_f32();
                acc1 += a_row[p + 1].to_f32() * b_col[p + 1].to_f32();
            }
            if k & 1 != 0 {
                acc0 += a_row[k - 1].to_f32() * b_col[k - 1].to_f32();
            }

            c[i * n + j] += alpha * (acc0 + acc1);
        }
    }
}

/// Cache-blocked BF16 GEMM.
/// Converts BF16 tiles to f32 in L1-sized blocks, then uses f32 microkernel.
fn bf16_gemm_blocked(
    a: &[BF16],
    b: &[BF16],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
) {
    const MC: usize = 128;
    const NC: usize = 256;
    const KC: usize = 256;

    // Temporary f32 buffers for converted tiles
    let mc = MC.min(m);
    let nc = NC.min(n);
    let kc = KC.min(k);

    let mut a_f32 = vec![0.0f32; mc * kc];
    let mut b_f32 = vec![0.0f32; kc * nc];

    for jc in (0..n).step_by(NC) {
        let jb = NC.min(n - jc);

        for pc in (0..k).step_by(KC) {
            let pb = KC.min(k - pc);

            // Convert B tile to f32
            for p in 0..pb {
                for j in 0..jb {
                    b_f32[p * jb + j] = b[(pc + p) * n + (jc + j)].to_f32();
                }
            }

            for ic in (0..m).step_by(MC) {
                let ib = MC.min(m - ic);

                // Convert A tile to f32
                for i in 0..ib {
                    for p in 0..pb {
                        a_f32[i * pb + p] = a[(ic + i) * k + (pc + p)].to_f32();
                    }
                }

                // Compute tile: C[ic:ic+ib, jc:jc+jb] += alpha * A_tile * B_tile
                for i in 0..ib {
                    for j in 0..jb {
                        let mut sum = 0.0f32;
                        for p in 0..pb {
                            sum += a_f32[i * pb + p] * b_f32[p * jb + j];
                        }
                        c[(ic + i) * n + (jc + j)] += alpha * sum;
                    }
                }
            }
        }
    }
}

/// Mixed-precision GEMM: f32 inputs → BF16 compute → f32 output.
/// Quantizes inputs on-the-fly, useful for training where you want
/// reduced memory bandwidth but f32 gradients.
pub fn mixed_precision_gemm(
    a_f32: &[f32],
    b_f32: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    let a_bf16 = f32_vec_to_bf16(a_f32);
    let b_bf16 = f32_vec_to_bf16(b_f32);
    bf16_gemm_f32(&a_bf16, &b_bf16, c, m, n, k, alpha, beta);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bf16_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 3.14, 1e10, -1e-10, 0.5, 255.0];
        for &v in &values {
            let bf = BF16::from_f32(v);
            let back = bf.to_f32();
            let err = (back - v).abs();
            let tol = v.abs() * 0.01 + 1e-30; // ~0.8% relative error for BF16
            assert!(err < tol, "BF16 roundtrip: {} -> {} (err={})", v, back, err);
        }
    }

    #[test]
    fn test_bf16_truncate_vs_round() {
        // 1.5 in f32 = 0x3FC00000
        // BF16 truncate: 0x3FC0
        // BF16 round: 0x3FC0 (already exact)
        let v = 1.5f32;
        let t = BF16::from_f32_truncate(v);
        let r = BF16::from_f32(v);
        assert_eq!(t.to_f32(), 1.5);
        assert_eq!(r.to_f32(), 1.5);
    }

    #[test]
    fn test_bf16_gemm_identity() {
        let a_f32 = vec![1.0f32, 0.0, 0.0, 1.0];
        let b_f32 = vec![3.0f32, 7.0, 5.0, 11.0];
        let a = f32_vec_to_bf16(&a_f32);
        let b = f32_vec_to_bf16(&b_f32);
        let mut c = vec![0.0f32; 4];
        bf16_gemm_f32(&a, &b, &mut c, 2, 2, 2, 1.0, 0.0);
        assert!((c[0] - 3.0).abs() < 0.1);
        assert!((c[1] - 7.0).abs() < 0.1);
        assert!((c[2] - 5.0).abs() < 0.1);
        assert!((c[3] - 11.0).abs() < 0.1);
    }

    #[test]
    fn test_bf16_gemm_multiply() {
        let a_f32 = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_f32 = vec![5.0f32, 6.0, 7.0, 8.0];
        let a = f32_vec_to_bf16(&a_f32);
        let b = f32_vec_to_bf16(&b_f32);
        let mut c = vec![0.0f32; 4];
        bf16_gemm_f32(&a, &b, &mut c, 2, 2, 2, 1.0, 0.0);
        // Expected: [[19, 22], [43, 50]]
        assert!((c[0] - 19.0).abs() < 0.5, "c[0]={}", c[0]);
        assert!((c[1] - 22.0).abs() < 0.5, "c[1]={}", c[1]);
        assert!((c[2] - 43.0).abs() < 0.5, "c[2]={}", c[2]);
        assert!((c[3] - 50.0).abs() < 0.5, "c[3]={}", c[3]);
    }

    #[test]
    fn test_mixed_precision_gemm() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut c = vec![10.0f32; 4];
        mixed_precision_gemm(&a, &b, &mut c, 2, 2, 2, 1.0, 1.0);
        // C = 1.0 * A*B + 1.0 * 10 = [[29, 32], [53, 60]]
        assert!((c[0] - 29.0).abs() < 1.0);
        assert!((c[1] - 32.0).abs() < 1.0);
        assert!((c[2] - 53.0).abs() < 1.0);
        assert!((c[3] - 60.0).abs() < 1.0);
    }

    #[test]
    fn test_bulk_conversion() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let bf16 = f32_vec_to_bf16(&src);
        let back = bf16_vec_to_f32(&bf16);
        for i in 0..src.len() {
            assert!((back[i] - src[i]).abs() < 0.1);
        }
    }
}
