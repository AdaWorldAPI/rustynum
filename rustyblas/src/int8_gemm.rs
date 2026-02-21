//! INT8 Quantized GEMM with AVX-512 VNNI acceleration.
//!
//! Implements quantized matrix multiplication for inference workloads:
//!   C_f32 = scale_a * scale_b * (A_u8 - zp_a) * (B_i8 - zp_b)
//!
//! Uses AVX-512 VNNI `vpdpbusd` instruction: u8 × i8 → i32 accumulate,
//! processing 64 multiply-adds per instruction (16 groups of 4).
//!
//! Quantization schemes supported:
//! - **Symmetric**: zero_point = 0, scale = max(|x|) / 127
//! - **Asymmetric**: zero_point ≠ 0, scale = (max - min) / 255
//! - **Per-tensor**: single scale/zp for entire tensor
//! - **Per-channel**: per-row scale/zp for A, per-column for B

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// ============================================================================
// Quantization parameters
// ============================================================================

/// Quantization parameters for a tensor.
#[derive(Clone, Debug)]
pub struct QuantParams {
    /// Scale factor: real_value = scale * (quantized_value - zero_point)
    pub scale: f32,
    /// Zero point offset (0 for symmetric quantization)
    pub zero_point: i32,
}

/// Per-channel quantization: one scale/zp per row or column.
#[derive(Clone, Debug)]
pub struct PerChannelQuantParams {
    pub scales: Vec<f32>,
    pub zero_points: Vec<i32>,
}

// ============================================================================
// Quantize / Dequantize
// ============================================================================

/// Quantize f32 tensor to u8 (asymmetric, per-tensor).
/// Returns (quantized_data, params).
pub fn quantize_f32_to_u8(data: &[f32]) -> (Vec<u8>, QuantParams) {
    if data.is_empty() {
        return (vec![], QuantParams { scale: 1.0, zero_point: 0 });
    }

    let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let scale = if max_val == min_val {
        1.0
    } else {
        (max_val - min_val) / 255.0
    };
    let zero_point = (-min_val / scale).round() as i32;
    let zero_point = zero_point.clamp(0, 255);

    let quantized: Vec<u8> = data
        .iter()
        .map(|&v| ((v / scale).round() as i32 + zero_point).clamp(0, 255) as u8)
        .collect();

    (quantized, QuantParams { scale, zero_point })
}

/// Quantize f32 tensor to i8 (symmetric, per-tensor).
/// Returns (quantized_data, params).
pub fn quantize_f32_to_i8(data: &[f32]) -> (Vec<i8>, QuantParams) {
    if data.is_empty() {
        return (vec![], QuantParams { scale: 1.0, zero_point: 0 });
    }

    let abs_max = data
        .iter()
        .copied()
        .fold(0.0f32, |acc, v| acc.max(v.abs()));

    let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };

    let quantized: Vec<i8> = data
        .iter()
        .map(|&v| (v / scale).round().clamp(-128.0, 127.0) as i8)
        .collect();

    (quantized, QuantParams { scale, zero_point: 0 })
}

/// Per-channel quantization: quantize each row of an MxK matrix to i8.
pub fn quantize_per_channel_i8(
    data: &[f32],
    rows: usize,
    cols: usize,
) -> (Vec<i8>, PerChannelQuantParams) {
    let mut quantized = vec![0i8; rows * cols];
    let mut scales = Vec::with_capacity(rows);
    let mut zero_points = Vec::with_capacity(rows);

    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        let abs_max = row.iter().copied().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };

        for c in 0..cols {
            quantized[r * cols + c] = (row[c] / scale).round().clamp(-128.0, 127.0) as i8;
        }

        scales.push(scale);
        zero_points.push(0);
    }

    (quantized, PerChannelQuantParams { scales, zero_points })
}

// ============================================================================
// INT8 GEMM: AVX-512 VNNI path
// ============================================================================

/// INT8 GEMM using AVX-512 VNNI: C_i32 += A_u8 * B_i8
///
/// A is M×K (u8, row-major), B is K×N (i8, row-major).
/// C is M×N (i32, row-major) — accumulated dot products.
///
/// This is the raw integer multiply-accumulate. To get f32 output,
/// use `int8_gemm_f32` which applies dequantization.
///
/// On VNNI-capable CPUs, `vpdpbusd` processes 64 u8×i8→i32 MACs per
/// instruction (16 lanes × 4 pairs each). For a 512×512 GEMM, this
/// gives ~4x throughput vs f32 GEMM.
pub fn int8_gemm_i32(
    a: &[u8],   // M × K, row-major
    b: &[i8],   // K × N, row-major
    c: &mut [i32], // M × N, row-major
    m: usize,
    n: usize,
    k: usize,
) {
    assert!(a.len() >= m * k);
    assert!(b.len() >= k * n);
    assert!(c.len() >= m * n);

    c[..m * n].fill(0);

    // Transpose B to column-major for sequential access: B_t[j][p]
    let mut b_t = vec![0i8; n * k];
    for p in 0..k {
        for j in 0..n {
            b_t[j * k + p] = b[p * n + j];
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512bw") {
            unsafe {
                int8_gemm_vnni(a, &b_t, c, m, n, k);
            }
            return;
        }
    }

    // Scalar fallback
    int8_gemm_scalar(a, &b_t, c, m, n, k);
}

/// Scalar INT8 GEMM fallback (B already transposed).
fn int8_gemm_scalar(a: &[u8], b_t: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut acc: i32 = 0;
            for p in 0..k {
                acc += a[i * k + p] as i32 * b_t[j * k + p] as i32;
            }
            c[i * n + j] = acc;
        }
    }
}

/// AVX-512 VNNI INT8 GEMM using `vpdpbusd`.
///
/// Safety: requires avx512vnni + avx512bw. B_t is K-transposed (column-major).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
unsafe fn int8_gemm_vnni(a: &[u8], b_t: &[i8], c: &mut [i32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        let a_row = &a[i * k..];

        for j in 0..n {
            let b_col = &b_t[j * k..];
            let mut acc = _mm512_setzero_si512();

            // Main loop: 64 bytes per iteration using VNNI
            // vpdpbusd processes 4 bytes per 32-bit lane × 16 lanes = 64 bytes
            let mut p = 0;
            while p + 64 <= k {
                let a_vec = _mm512_loadu_si512(a_row[p..].as_ptr() as *const __m512i);
                let b_vec = _mm512_loadu_si512(b_col[p..].as_ptr() as *const __m512i);
                acc = _mm512_dpbusd_epi32(acc, a_vec, b_vec);
                p += 64;
            }

            // Handle remaining bytes with zero-padded buffer
            if p < k {
                let mut a_buf = [0u8; 64];
                let mut b_buf = [0u8; 64]; // store as u8, reinterpret as i8
                let remaining = k - p;
                a_buf[..remaining].copy_from_slice(&a_row[p..p + remaining]);
                for idx in 0..remaining {
                    b_buf[idx] = b_col[p + idx] as u8;
                }

                let a_vec = _mm512_loadu_si512(a_buf.as_ptr() as *const __m512i);
                let b_vec = _mm512_loadu_si512(b_buf.as_ptr() as *const __m512i);
                acc = _mm512_dpbusd_epi32(acc, a_vec, b_vec);
            }

            // Horizontal reduction of 16 i32 lanes
            c[i * n + j] = _mm512_reduce_add_epi32(acc);
        }
    }
}

// ============================================================================
// Dequantized INT8 GEMM → f32 output
// ============================================================================

/// Quantized GEMM with f32 output:
///   C_f32[i][j] = scale_a * scale_b * sum_k( (A_u8[i][k] - zp_a) * (B_i8[k][j]) )
///
/// For symmetric B quantization (zp_b = 0), this simplifies to:
///   C_f32[i][j] = scale_a * scale_b * (dot(A_u8[i], B_i8[j]) - zp_a * sum(B_i8[j]))
pub fn int8_gemm_f32(
    a: &[u8],
    b: &[i8],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    scale_a: f32,
    zero_point_a: i32,
    scale_b: f32,
) {
    // Compute raw i32 GEMM
    let mut c_i32 = vec![0i32; m * n];
    int8_gemm_i32(a, b, &mut c_i32, m, n, k);

    // If zero_point_a != 0, compute column sums of B for correction
    let combined_scale = scale_a * scale_b;

    if zero_point_a == 0 {
        // Simple dequantization
        for i in 0..m * n {
            c[i] = c_i32[i] as f32 * combined_scale;
        }
    } else {
        // Need correction: C -= zp_a * col_sums(B)
        let mut b_col_sums = vec![0i32; n];
        for p in 0..k {
            for j in 0..n {
                b_col_sums[j] += b[p * n + j] as i32;
            }
        }

        for i in 0..m {
            for j in 0..n {
                let raw = c_i32[i * n + j] - zero_point_a * b_col_sums[j];
                c[i * n + j] = raw as f32 * combined_scale;
            }
        }
    }
}

/// Per-channel quantized GEMM: each row of A and each column of B
/// has its own scale factor.
pub fn int8_gemm_per_channel_f32(
    a: &[u8],
    b: &[i8],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    a_scales: &[f32],    // M scales
    a_zero_points: &[i32], // M zero points
    b_scales: &[f32],    // N scales
) {
    let mut c_i32 = vec![0i32; m * n];
    int8_gemm_i32(a, b, &mut c_i32, m, n, k);

    // Pre-compute B column sums for zero-point correction
    let mut b_col_sums = vec![0i32; n];
    let has_zp = a_zero_points.iter().any(|&zp| zp != 0);
    if has_zp {
        for p in 0..k {
            for j in 0..n {
                b_col_sums[j] += b[p * n + j] as i32;
            }
        }
    }

    for i in 0..m {
        for j in 0..n {
            let raw = if has_zp {
                c_i32[i * n + j] - a_zero_points[i] * b_col_sums[j]
            } else {
                c_i32[i * n + j]
            };
            c[i * n + j] = raw as f32 * a_scales[i] * b_scales[j];
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_f32_to_u8() {
        let data = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        let (q, params) = quantize_f32_to_u8(&data);
        assert_eq!(q.len(), 5);
        // Dequantize and verify
        for i in 0..5 {
            let deq = params.scale * (q[i] as i32 - params.zero_point) as f32;
            assert!((deq - data[i]).abs() < 0.02, "mismatch at {}: {} vs {}", i, deq, data[i]);
        }
    }

    #[test]
    fn test_quantize_f32_to_i8() {
        let data = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        let (q, params) = quantize_f32_to_i8(&data);
        assert_eq!(params.zero_point, 0); // Symmetric
        for i in 0..5 {
            let deq = params.scale * q[i] as f32;
            assert!((deq - data[i]).abs() < 0.02, "mismatch at {}: {} vs {}", i, deq, data[i]);
        }
    }

    #[test]
    fn test_int8_gemm_basic() {
        // A(2x3, u8) * B(3x2, i8) = C(2x2, i32)
        let a: Vec<u8> = vec![1, 2, 3, 4, 5, 6];
        let b: Vec<i8> = vec![1, 2, 3, 4, 5, 6];
        let mut c = vec![0i32; 4];
        int8_gemm_i32(&a, &b, &mut c, 2, 2, 3);
        // Row 0: 1*1+2*3+3*5=22, 1*2+2*4+3*6=28
        // Row 1: 4*1+5*3+6*5=49, 4*2+5*4+6*6=64
        assert_eq!(c, vec![22, 28, 49, 64]);
    }

    #[test]
    fn test_int8_gemm_f32_symmetric() {
        // Use symmetric quantization (zp=0)
        let a: Vec<u8> = vec![10, 20, 30, 40];
        let b: Vec<i8> = vec![1, 2, 3, 4];
        let mut c = vec![0.0f32; 4];
        int8_gemm_f32(&a, &b, &mut c, 2, 2, 2, 0.1, 0, 0.01, );
        // Raw: [10*1+20*3=70, 10*2+20*4=100, 30*1+40*3=150, 30*2+40*4=220]
        // Scaled: 0.1 * 0.01 * raw = 0.001 * raw
        assert!((c[0] - 0.070).abs() < 1e-5);
        assert!((c[1] - 0.100).abs() < 1e-5);
        assert!((c[2] - 0.150).abs() < 1e-5);
        assert!((c[3] - 0.220).abs() < 1e-5);
    }

    #[test]
    fn test_int8_gemm_large() {
        // 64x64 GEMM to exercise VNNI path
        let m = 64;
        let k = 64;
        let n = 64;
        let a: Vec<u8> = (0..m * k).map(|i| (i % 200) as u8).collect();
        let b: Vec<i8> = (0..k * n).map(|i| ((i % 200) as i8).wrapping_sub(100)).collect();
        let mut c = vec![0i32; m * n];
        int8_gemm_i32(&a, &b, &mut c, m, n, k);

        // Verify against scalar reference
        let mut c_ref = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i32;
                for p in 0..k {
                    acc += a[i * k + p] as i32 * b[p * n + j] as i32;
                }
                c_ref[i * n + j] = acc;
            }
        }
        assert_eq!(c, c_ref);
    }

    #[test]
    fn test_quantize_roundtrip() {
        // Quantize f32 -> i8 -> multiply -> dequantize, compare to f32 multiply
        let m = 8;
        let k = 16;
        let n = 8;
        let a_f32: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.01) - 0.5).collect();
        let b_f32: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.02) - 0.8).collect();

        // Quantize
        let (a_q, a_params) = quantize_f32_to_u8(&a_f32);
        let (b_q, b_params) = quantize_f32_to_i8(&b_f32);

        // Quantized GEMM
        let mut c_q = vec![0.0f32; m * n];
        int8_gemm_f32(
            &a_q, &b_q, &mut c_q, m, n, k,
            a_params.scale, a_params.zero_point, b_params.scale,
        );

        // Reference f32 GEMM
        let mut c_ref = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    c_ref[i * n + j] += a_f32[i * k + p] * b_f32[p * n + j];
                }
            }
        }

        // Quantization introduces error — allow ~10% relative error
        for i in 0..m * n {
            let err = (c_q[i] - c_ref[i]).abs();
            let tol = c_ref[i].abs() * 0.15 + 0.05; // relative + absolute tolerance
            assert!(
                err < tol,
                "Quantized GEMM error at {}: {} vs {} (err={})",
                i, c_q[i], c_ref[i], err
            );
        }
    }
}
