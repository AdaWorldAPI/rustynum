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
    assert_eq!(a.len(), b.len());
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
    assert_eq!(a.len(), b.len());
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
    assert_eq!(x.len(), y.len());
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
    assert_eq!(x.len(), y.len());
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

// ============================================================================
// VPOPCNTDQ: Hardware-accelerated Hamming distance for HDC/CogRecord
// ============================================================================

/// Hamming distance between two byte arrays (number of differing bits).
///
/// Computes `popcount(a XOR b)` — the total bit-level difference.
/// Runtime dispatches to VPOPCNTDQ (64 bytes/iter, ~8x scalar) when available,
/// otherwise falls back to scalar POPCNT on u64 chunks (~4x naive).
///
/// For a 2KB CogRecord container: 32 VPOPCNTDQ iterations vs 256 scalar POPCNTs.
#[inline]
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512f") {
            return unsafe { hamming_vpopcntdq(a, b) };
        }
    }

    hamming_scalar_popcnt(a, b)
}

/// Batch Hamming distance: compute distances from `query` to each row in `database`.
///
/// `database` is a flat byte array with `num_rows` rows of `row_bytes` each.
/// Returns a Vec of Hamming distances, one per row.
///
/// 4x unrolled for ILP — processes 4 database rows per outer iteration.
#[inline]
pub fn hamming_batch(query: &[u8], database: &[u8], num_rows: usize, row_bytes: usize) -> Vec<u64> {
    assert_eq!(query.len(), row_bytes);
    assert_eq!(database.len(), num_rows * row_bytes);

    let mut distances = vec![0u64; num_rows];

    // 4x unrolled
    let full = num_rows / 4;
    for i in 0..full {
        let base = i * 4;
        distances[base] = hamming_distance(query, &database[base * row_bytes..(base + 1) * row_bytes]);
        distances[base + 1] = hamming_distance(query, &database[(base + 1) * row_bytes..(base + 2) * row_bytes]);
        distances[base + 2] = hamming_distance(query, &database[(base + 2) * row_bytes..(base + 3) * row_bytes]);
        distances[base + 3] = hamming_distance(query, &database[(base + 3) * row_bytes..(base + 4) * row_bytes]);
    }
    for i in (full * 4)..num_rows {
        distances[i] = hamming_distance(query, &database[i * row_bytes..(i + 1) * row_bytes]);
    }

    distances
}

/// Top-k nearest neighbors by Hamming distance.
///
/// Returns `(indices, distances)` of the `k` closest rows in `database` to `query`.
/// Uses partial sort — O(n*k) but avoids full sort for small k.
pub fn hamming_top_k(query: &[u8], database: &[u8], num_rows: usize, row_bytes: usize, k: usize) -> (Vec<usize>, Vec<u64>) {
    let distances = hamming_batch(query, database, num_rows, row_bytes);
    let k = k.min(num_rows);

    // Build index array and partial sort by distance
    let mut indices: Vec<usize> = (0..num_rows).collect();
    indices.select_nth_unstable_by_key(k.saturating_sub(1), |&i| distances[i]);
    indices.truncate(k);
    indices.sort_unstable_by_key(|&i| distances[i]);

    let top_distances: Vec<u64> = indices.iter().map(|&i| distances[i]).collect();
    (indices, top_distances)
}

/// VPOPCNTDQ fast path: 64 bytes per iteration.
///
/// XOR → VPOPCNTDQ → accumulate → horizontal sum.
/// Throughput: ~8x scalar POPCNT for aligned data.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
#[inline]
unsafe fn hamming_vpopcntdq(a: &[u8], b: &[u8]) -> u64 {
    use core::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 64; // 512 bits = 64 bytes per __m512i

    let mut total = _mm512_setzero_si512();

    for i in 0..chunks {
        let base = i * 64;
        let av = _mm512_loadu_si512(a[base..].as_ptr() as *const __m512i);
        let bv = _mm512_loadu_si512(b[base..].as_ptr() as *const __m512i);
        let xored = _mm512_xor_si512(av, bv);
        let popcnt = _mm512_popcnt_epi64(xored);
        total = _mm512_add_epi64(total, popcnt);
    }

    // Horizontal sum: store 8 × i64 and sum
    let mut vals = [0i64; 8];
    _mm512_storeu_si512(vals.as_mut_ptr() as *mut __m512i, total);
    let mut sum: u64 = vals.iter().map(|&v| v as u64).sum();

    // Scalar tail
    for i in (chunks * 64)..len {
        sum += (a[i] ^ b[i]).count_ones() as u64;
    }

    sum
}

/// Portable fallback: scalar POPCNT on u64 chunks.
///
/// Processes 8 bytes per iteration using hardware POPCNT instruction
/// (available on all modern x86_64). ~4x faster than byte-by-byte.
#[inline]
fn hamming_scalar_popcnt(a: &[u8], b: &[u8]) -> u64 {
    let len = a.len();
    let u64_chunks = len / 8;
    let mut sum: u64 = 0;

    for i in 0..u64_chunks {
        let base = i * 8;
        let a_u64 = u64::from_le_bytes([
            a[base], a[base+1], a[base+2], a[base+3],
            a[base+4], a[base+5], a[base+6], a[base+7],
        ]);
        let b_u64 = u64::from_le_bytes([
            b[base], b[base+1], b[base+2], b[base+3],
            b[base+4], b[base+5], b[base+6], b[base+7],
        ]);
        sum += (a_u64 ^ b_u64).count_ones() as u64;
    }

    for i in (u64_chunks * 8)..len {
        sum += (a[i] ^ b[i]).count_ones() as u64;
    }

    sum
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

    #[test]
    fn test_hamming_distance_identical() {
        let a = vec![0xFFu8; 2048]; // 2KB CogRecord container
        let b = vec![0xFFu8; 2048];
        assert_eq!(hamming_distance(&a, &b), 0);
    }

    #[test]
    fn test_hamming_distance_all_different() {
        let a = vec![0x00u8; 64];
        let b = vec![0xFFu8; 64];
        // 64 bytes × 8 bits = 512 differing bits
        assert_eq!(hamming_distance(&a, &b), 512);
    }

    #[test]
    fn test_hamming_distance_known() {
        // Single byte difference: 0b10101010 ^ 0b01010101 = 0b11111111 → 8 bits
        let mut a = vec![0u8; 100];
        let mut b = vec![0u8; 100];
        a[0] = 0b10101010;
        b[0] = 0b01010101;
        a[50] = 0b11110000;
        b[50] = 0b00001111;
        assert_eq!(hamming_distance(&a, &b), 16); // 8 + 8
    }

    #[test]
    fn test_hamming_distance_2kb() {
        // Simulate CogRecord: 2KB containers with ~25% bit difference
        let a: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let b: Vec<u8> = (0..2048).map(|i| ((i + 1) % 256) as u8).collect();
        let dist = hamming_distance(&a, &b);
        // Verify against scalar reference
        let expected: u64 = a.iter().zip(b.iter())
            .map(|(&x, &y)| (x ^ y).count_ones() as u64).sum();
        assert_eq!(dist, expected);
    }

    #[test]
    fn test_hamming_batch() {
        let query = vec![0xAAu8; 16];
        let mut database = vec![0u8; 16 * 4]; // 4 rows of 16 bytes
        // Row 0: identical → 0
        for i in 0..16 { database[i] = 0xAA; }
        // Row 1: all different → 16*8 = 128
        for i in 16..32 { database[i] = 0x55; }
        // Row 2: half different → 64
        for i in 32..40 { database[i] = 0xAA; }
        for i in 40..48 { database[i] = 0x55; }
        // Row 3: one byte different → 8
        for i in 48..64 { database[i] = 0xAA; }
        database[48] = 0x55;

        let distances = hamming_batch(&query, &database, 4, 16);
        assert_eq!(distances[0], 0);
        assert_eq!(distances[1], 128);
        assert_eq!(distances[2], 64);
        assert_eq!(distances[3], 8);
    }

    #[test]
    fn test_hamming_top_k() {
        let query = vec![0xAAu8; 16];
        let mut database = vec![0xAAu8; 16 * 5]; // 5 rows, all identical
        // Make rows 1 and 3 more different
        database[16] = 0x00; // row 1: 1 byte diff → 4 bits
        database[48] = 0x00; database[49] = 0x00; // row 3: 2 bytes diff → 8 bits

        let (indices, distances) = hamming_top_k(&query, &database, 5, 16, 3);
        assert_eq!(indices.len(), 3);
        // Top 3 should be the 3 rows with 0 distance
        assert_eq!(distances[0], 0);
        assert_eq!(distances[1], 0);
        assert_eq!(distances[2], 0);
    }
}
