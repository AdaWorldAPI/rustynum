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
/// 3-tier runtime dispatch:
///   1. AVX-512 VPOPCNTDQ: 64 bytes/iter, ~8x scalar (server CPUs, Ice Lake+)
///   2. AVX2 Harley-Seal vpshufb popcount: 32 bytes/iter, ~4x scalar (all CPUs since 2013)
///   3. Scalar POPCNT on u64 chunks: 8 bytes/iter (universal fallback)
///
/// For a 2KB CogRecord container: 32 VPOPCNTDQ or 64 AVX2 iterations.
#[inline]
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512f") {
            return unsafe { hamming_vpopcntdq(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { hamming_avx2(a, b) };
        }
    }

    hamming_scalar_popcnt(a, b)
}

/// Select the best Hamming distance function for this CPU.
/// Call ONCE, use many times — avoids redundant CPUID checks.
///
/// Dispatch chain: VPOPCNTDQ (AVX-512) → Harley-Seal (AVX2) → scalar POPCNT.
#[inline]
pub fn select_hamming_fn() -> fn(&[u8], &[u8]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512f") {
            return hamming_vpopcntdq_safe;
        }
        if is_x86_feature_detected!("avx2") {
            return hamming_avx2_safe;
        }
    }
    hamming_scalar_popcnt
}

/// Select the best int8 dot product function for this CPU.
/// Call ONCE, use many times — avoids redundant CPUID checks.
///
/// Dispatch chain: AVX-512 VNNI (VPDPBUSD) → scalar.
#[inline]
pub fn select_dot_i8_fn() -> fn(&[u8], &[u8]) -> i64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512f") {
            return dot_i8_vnni_safe;
        }
    }
    dot_i8_scalar
}

/// Safe wrapper for dot_i8_vnni (coerces to fn pointer).
#[cfg(target_arch = "x86_64")]
fn dot_i8_vnni_safe(a: &[u8], b: &[u8]) -> i64 {
    unsafe { dot_i8_vnni(a, b) }
}

/// Batch Hamming distance: compute distances from `query` to each row in `database`.
///
/// `database` is a flat byte array with `num_rows` rows of `row_bytes` each.
/// Returns a Vec of Hamming distances, one per row.
///
/// CPUID is checked once at batch level, not per-row.
/// 4x unrolled for ILP — processes 4 database rows per outer iteration.
#[inline]
pub fn hamming_batch(query: &[u8], database: &[u8], num_rows: usize, row_bytes: usize) -> Vec<u64> {
    assert_eq!(query.len(), row_bytes);
    assert_eq!(database.len(), num_rows * row_bytes);

    let mut distances = vec![0u64; num_rows];

    // Dispatch once at batch level — avoids N redundant CPUID checks.
    let hamming_fn = select_hamming_fn();

    let full = num_rows / 4;
    for i in 0..full {
        let base = i * 4;
        distances[base]     = hamming_fn(query, &database[base * row_bytes..(base + 1) * row_bytes]);
        distances[base + 1] = hamming_fn(query, &database[(base + 1) * row_bytes..(base + 2) * row_bytes]);
        distances[base + 2] = hamming_fn(query, &database[(base + 2) * row_bytes..(base + 3) * row_bytes]);
        distances[base + 3] = hamming_fn(query, &database[(base + 3) * row_bytes..(base + 4) * row_bytes]);
    }
    for i in (full * 4)..num_rows {
        distances[i] = hamming_fn(query, &database[i * row_bytes..(i + 1) * row_bytes]);
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

/// Safe wrapper for hamming_vpopcntdq (coerces to fn pointer).
#[cfg(target_arch = "x86_64")]
fn hamming_vpopcntdq_safe(a: &[u8], b: &[u8]) -> u64 {
    unsafe { hamming_vpopcntdq(a, b) }
}

/// Safe wrapper for hamming_avx2 (coerces to fn pointer).
#[cfg(target_arch = "x86_64")]
fn hamming_avx2_safe(a: &[u8], b: &[u8]) -> u64 {
    unsafe { hamming_avx2(a, b) }
}

/// AVX2 Hamming distance using Harley-Seal vpshufb popcount.
/// ~4x faster than scalar on any machine with AVX2 (2013+).
///
/// Uses nibble lookup table (vpshufb) for byte-level popcount,
/// then vpsadbw for horizontal sum within each 256-bit lane.
/// Processes in blocks of 8 to avoid u8 accumulator saturation.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hamming_avx2(a: &[u8], b: &[u8]) -> u64 {
    use core::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 32; // 256 bits = 32 bytes per __m256i

    // Nibble lookup table for popcount
    let low_mask = _mm256_set1_epi8(0x0f);
    let lookup = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
    );

    let mut total = _mm256_setzero_si256();

    // Process in blocks of 8 to avoid u8 counter saturation (max 255)
    let blocks = chunks / 8;
    for block in 0..blocks {
        let mut local = _mm256_setzero_si256();
        for i in 0..8 {
            let idx = (block * 8 + i) * 32;
            let av = _mm256_loadu_si256(a[idx..].as_ptr() as *const __m256i);
            let bv = _mm256_loadu_si256(b[idx..].as_ptr() as *const __m256i);
            let xored = _mm256_xor_si256(av, bv);

            let lo = _mm256_and_si256(xored, low_mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16(xored, 4), low_mask);
            let popcnt_lo = _mm256_shuffle_epi8(lookup, lo);
            let popcnt_hi = _mm256_shuffle_epi8(lookup, hi);
            local = _mm256_add_epi8(local, popcnt_lo);
            local = _mm256_add_epi8(local, popcnt_hi);
        }
        // Widen u8 counts to u64 via vpsadbw
        total = _mm256_add_epi64(total, _mm256_sad_epu8(local, _mm256_setzero_si256()));
    }

    // Remaining full chunks
    let mut local = _mm256_setzero_si256();
    for i in (blocks * 8)..chunks {
        let idx = i * 32;
        let av = _mm256_loadu_si256(a[idx..].as_ptr() as *const __m256i);
        let bv = _mm256_loadu_si256(b[idx..].as_ptr() as *const __m256i);
        let xored = _mm256_xor_si256(av, bv);
        let lo = _mm256_and_si256(xored, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(xored, 4), low_mask);
        local = _mm256_add_epi8(local, _mm256_shuffle_epi8(lookup, lo));
        local = _mm256_add_epi8(local, _mm256_shuffle_epi8(lookup, hi));
    }
    total = _mm256_add_epi64(total, _mm256_sad_epu8(local, _mm256_setzero_si256()));

    // Horizontal sum 4 × i64
    let mut vals = [0i64; 4];
    _mm256_storeu_si256(vals.as_mut_ptr() as *mut __m256i, total);
    let mut sum = (vals[0] + vals[1] + vals[2] + vals[3]) as u64;

    // Scalar tail
    for i in (chunks * 32)..len {
        sum += (a[i] ^ b[i]).count_ones() as u64;
    }

    sum
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

// ============================================================================
// POPCOUNT: standalone bit counting (no XOR)
// ============================================================================

/// Count total set bits in a byte slice.
///
/// 3-tier dispatch:
/// 1. VPOPCNTDQ (AVX-512): 64 bytes/iteration, 8 u64 lanes, zero waste on 2KB containers
/// 2. AVX2 Harley-Seal: vpshufb nibble popcount, 32 bytes/iteration
/// 3. Scalar POPCNT: u64::count_ones(), 8 bytes/iteration
///
/// For 2048-byte containers: 32 VPOPCNTDQ iterations (2048/64), matching u64x8 width exactly.
pub fn popcount(a: &[u8]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512f") {
            return unsafe { popcount_vpopcntdq(a) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { popcount_avx2(a) };
        }
    }
    popcount_scalar(a)
}

/// VPOPCNTDQ popcount: 64 bytes per iteration (8 × u64 lanes).
///
/// For 2048-byte containers = 32 iterations, matching u64x8 width exactly — zero overhead.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
#[inline]
unsafe fn popcount_vpopcntdq(a: &[u8]) -> u64 {
    use core::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 64; // 512 bits = 64 bytes = 8 × u64

    let mut total = _mm512_setzero_si512();

    for i in 0..chunks {
        let base = i * 64;
        let v = _mm512_loadu_si512(a[base..].as_ptr() as *const __m512i);
        let popcnt = _mm512_popcnt_epi64(v);
        total = _mm512_add_epi64(total, popcnt);
    }

    // Horizontal sum: 8 × i64
    let mut vals = [0i64; 8];
    _mm512_storeu_si512(vals.as_mut_ptr() as *mut __m512i, total);
    let mut sum: u64 = vals.iter().map(|&v| v as u64).sum();

    // Scalar tail
    for i in (chunks * 64)..len {
        sum += a[i].count_ones() as u64;
    }

    sum
}

/// AVX2 Harley-Seal popcount: vpshufb nibble lookup, 32 bytes per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn popcount_avx2(a: &[u8]) -> u64 {
    use core::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 32;

    let low_mask = _mm256_set1_epi8(0x0f);
    let lookup = _mm256_setr_epi8(
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
    );

    let mut total = _mm256_setzero_si256();

    // Process in blocks of 8 to avoid u8 saturation (max 8*32*4 = 1024 bits per byte < 255)
    let blocks = chunks / 8;
    for block in 0..blocks {
        let mut local = _mm256_setzero_si256();
        for i in 0..8 {
            let idx = (block * 8 + i) * 32;
            let v = _mm256_loadu_si256(a[idx..].as_ptr() as *const __m256i);
            let lo = _mm256_and_si256(v, low_mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
            let cnt = _mm256_add_epi8(
                _mm256_shuffle_epi8(lookup, lo),
                _mm256_shuffle_epi8(lookup, hi),
            );
            local = _mm256_add_epi8(local, cnt);
        }
        let sad = _mm256_sad_epu8(local, _mm256_setzero_si256());
        total = _mm256_add_epi64(total, sad);
    }

    // Remaining chunks
    if blocks * 8 < chunks {
        let mut local = _mm256_setzero_si256();
        for i in blocks * 8..chunks {
            let idx = i * 32;
            let v = _mm256_loadu_si256(a[idx..].as_ptr() as *const __m256i);
            let lo = _mm256_and_si256(v, low_mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
            let cnt = _mm256_add_epi8(
                _mm256_shuffle_epi8(lookup, lo),
                _mm256_shuffle_epi8(lookup, hi),
            );
            local = _mm256_add_epi8(local, cnt);
        }
        let sad = _mm256_sad_epu8(local, _mm256_setzero_si256());
        total = _mm256_add_epi64(total, sad);
    }

    // Horizontal sum
    let arr: [i64; 4] = std::mem::transmute(total);
    let mut sum: u64 = arr.iter().map(|&v| v as u64).sum();

    // Scalar tail
    for i in (chunks * 32)..len {
        sum += a[i].count_ones() as u64;
    }

    sum
}

/// Scalar popcount fallback: u64::count_ones() with 4x unrolling.
#[inline]
fn popcount_scalar(a: &[u8]) -> u64 {
    let len = a.len();
    let u64_chunks = len / 8;
    let full_quads = u64_chunks / 4;
    let mut sum: u64 = 0;

    for q in 0..full_quads {
        let base = q * 32;
        let w0 = u64::from_ne_bytes(a[base..base + 8].try_into().unwrap());
        let w1 = u64::from_ne_bytes(a[base + 8..base + 16].try_into().unwrap());
        let w2 = u64::from_ne_bytes(a[base + 16..base + 24].try_into().unwrap());
        let w3 = u64::from_ne_bytes(a[base + 24..base + 32].try_into().unwrap());
        sum += w0.count_ones() as u64
            + w1.count_ones() as u64
            + w2.count_ones() as u64
            + w3.count_ones() as u64;
    }

    for i in full_quads * 4..u64_chunks {
        let base = i * 8;
        let w = u64::from_ne_bytes(a[base..base + 8].try_into().unwrap());
        sum += w.count_ones() as u64;
    }

    for i in u64_chunks * 8..len {
        sum += a[i].count_ones() as u64;
    }

    sum
}

// ============================================================================
// VNNI: INT8 dot product (VPDPBUSD) for embedding containers
// ============================================================================

/// Signed int8 dot product: treats both byte slices as signed i8, returns sum of products.
///
/// Runtime dispatches to AVX-512 VNNI (`VPDPBUSD`) when available.
/// VPDPBUSD is unsigned×signed, so signed×signed uses the XOR-0x80 correction:
///   signed_result = dpbusd(a XOR 0x80, b) − 128 × sum(b)
///
/// For a 2KB CogRecord container: 32 VPDPBUSD iterations vs 2048 scalar multiplies.
#[inline]
pub fn dot_i8(a: &[u8], b: &[u8]) -> i64 {
    assert_eq!(a.len(), b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512f") {
            return unsafe { dot_i8_vnni(a, b) };
        }
    }

    dot_i8_scalar(a, b)
}

/// VPDPBUSD fast path: 64 bytes per iteration.
///
/// Uses unsigned×signed multiply-accumulate with bias correction for signed×signed:
///   a_unsigned = a_signed XOR 0x80  (shifts signed range [−128,127] to unsigned [0,255])
///   dpbusd_result = Σ(a_unsigned × b_signed)
///   signed_result = dpbusd_result − 128 × Σ(b_signed)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vnni")]
#[inline]
unsafe fn dot_i8_vnni(a: &[u8], b: &[u8]) -> i64 {
    use core::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 64;

    // Bias mask: XOR with 0x80 per byte to convert signed→unsigned
    let bias = _mm512_set1_epi32(0x80808080u32 as i32);
    // Ones vector for computing sum(b) via VPDPBUSD(ones, b)
    let ones = _mm512_set1_epi32(0x01010101u32 as i32);

    let mut acc = _mm512_setzero_si512();
    let mut b_sum = _mm512_setzero_si512();

    for i in 0..chunks {
        let base = i * 64;
        let av = _mm512_loadu_si512(a[base..].as_ptr() as *const __m512i);
        let bv = _mm512_loadu_si512(b[base..].as_ptr() as *const __m512i);

        // Convert a from signed to unsigned-with-bias
        let av_u = _mm512_xor_si512(av, bias);

        // VPDPBUSD: acc += Σ(a_unsigned[j] × b_signed[j]) per 4-byte group
        acc = _mm512_dpbusd_epi32(acc, av_u, bv);

        // Accumulate sum(b_signed) for correction
        b_sum = _mm512_dpbusd_epi32(b_sum, ones, bv);
    }

    // Horizontal sum of acc (16 × i32 → i64)
    let mut acc_vals = [0i32; 16];
    _mm512_storeu_si512(acc_vals.as_mut_ptr() as *mut __m512i, acc);
    let total_biased: i64 = acc_vals.iter().map(|&v| v as i64).sum();

    // Horizontal sum of b_sum
    let mut bsum_vals = [0i32; 16];
    _mm512_storeu_si512(bsum_vals.as_mut_ptr() as *mut __m512i, b_sum);
    let total_b: i64 = bsum_vals.iter().map(|&v| v as i64).sum();

    // Correction: biased_result = signed_result + 128 * sum(b)
    let mut result = total_biased - 128 * total_b;

    // Scalar tail
    for i in (chunks * 64)..len {
        result += (a[i] as i8 as i64) * (b[i] as i8 as i64);
    }

    result
}

/// Portable fallback: scalar int8 dot product.
#[inline]
fn dot_i8_scalar(a: &[u8], b: &[u8]) -> i64 {
    let len = a.len();
    let chunks = len / 32;
    let mut total: i64 = 0;

    for c in 0..chunks {
        let base = c * 32;
        let mut acc: i32 = 0;
        for i in 0..32 {
            acc += (a[base + i] as i8 as i32) * (b[base + i] as i8 as i32);
        }
        total += acc as i64;
    }

    for i in (chunks * 32)..len {
        total += (a[i] as i8 as i64) * (b[i] as i8 as i64);
    }

    total
}

// ============================================================================
// SIMD element-wise operations (f32/f64)
// ============================================================================

/// SIMD f32 element-wise add with scalar: out[i] = a[i] + scalar
#[inline]
pub fn add_f32_scalar(a: &[f32], scalar: f32) -> Vec<f32> {
    let len = a.len();
    let mut out = vec![0.0f32; len];
    let chunks = len / F32_LANES;
    let sv = f32x16::splat(scalar);
    for i in 0..chunks {
        let base = i * F32_LANES;
        let av = f32x16::from_slice(&a[base..]);
        (av + sv).copy_to_slice(&mut out[base..base + F32_LANES]);
    }
    for i in (chunks * F32_LANES)..len {
        out[i] = a[i] + scalar;
    }
    out
}

/// SIMD f32 element-wise subtract scalar: out[i] = a[i] - scalar
#[inline]
pub fn sub_f32_scalar(a: &[f32], scalar: f32) -> Vec<f32> {
    let len = a.len();
    let mut out = vec![0.0f32; len];
    let chunks = len / F32_LANES;
    let sv = f32x16::splat(scalar);
    for i in 0..chunks {
        let base = i * F32_LANES;
        let av = f32x16::from_slice(&a[base..]);
        (av - sv).copy_to_slice(&mut out[base..base + F32_LANES]);
    }
    for i in (chunks * F32_LANES)..len {
        out[i] = a[i] - scalar;
    }
    out
}

/// SIMD f32 element-wise multiply with scalar: out[i] = a[i] * scalar
#[inline]
pub fn mul_f32_scalar(a: &[f32], scalar: f32) -> Vec<f32> {
    let len = a.len();
    let mut out = vec![0.0f32; len];
    let chunks = len / F32_LANES;
    let sv = f32x16::splat(scalar);
    for i in 0..chunks {
        let base = i * F32_LANES;
        let av = f32x16::from_slice(&a[base..]);
        (av * sv).copy_to_slice(&mut out[base..base + F32_LANES]);
    }
    for i in (chunks * F32_LANES)..len {
        out[i] = a[i] * scalar;
    }
    out
}

/// SIMD f32 element-wise divide by scalar: out[i] = a[i] / scalar
#[inline]
pub fn div_f32_scalar(a: &[f32], scalar: f32) -> Vec<f32> {
    let len = a.len();
    let mut out = vec![0.0f32; len];
    let chunks = len / F32_LANES;
    let sv = f32x16::splat(scalar);
    for i in 0..chunks {
        let base = i * F32_LANES;
        let av = f32x16::from_slice(&a[base..]);
        (av / sv).copy_to_slice(&mut out[base..base + F32_LANES]);
    }
    for i in (chunks * F32_LANES)..len {
        out[i] = a[i] / scalar;
    }
    out
}

/// SIMD f32 element-wise vector add: out[i] = a[i] + b[i]
#[inline]
pub fn add_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut out = vec![0.0f32; len];
    let chunks = len / F32_LANES;
    for i in 0..chunks {
        let base = i * F32_LANES;
        let av = f32x16::from_slice(&a[base..]);
        let bv = f32x16::from_slice(&b[base..]);
        (av + bv).copy_to_slice(&mut out[base..base + F32_LANES]);
    }
    for i in (chunks * F32_LANES)..len {
        out[i] = a[i] + b[i];
    }
    out
}

/// SIMD f32 element-wise vector subtract: out[i] = a[i] - b[i]
#[inline]
pub fn sub_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut out = vec![0.0f32; len];
    let chunks = len / F32_LANES;
    for i in 0..chunks {
        let base = i * F32_LANES;
        let av = f32x16::from_slice(&a[base..]);
        let bv = f32x16::from_slice(&b[base..]);
        (av - bv).copy_to_slice(&mut out[base..base + F32_LANES]);
    }
    for i in (chunks * F32_LANES)..len {
        out[i] = a[i] - b[i];
    }
    out
}

/// SIMD f32 element-wise vector multiply: out[i] = a[i] * b[i]
#[inline]
pub fn mul_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut out = vec![0.0f32; len];
    let chunks = len / F32_LANES;
    for i in 0..chunks {
        let base = i * F32_LANES;
        let av = f32x16::from_slice(&a[base..]);
        let bv = f32x16::from_slice(&b[base..]);
        (av * bv).copy_to_slice(&mut out[base..base + F32_LANES]);
    }
    for i in (chunks * F32_LANES)..len {
        out[i] = a[i] * b[i];
    }
    out
}

/// SIMD f32 element-wise vector divide: out[i] = a[i] / b[i]
#[inline]
pub fn div_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut out = vec![0.0f32; len];
    let chunks = len / F32_LANES;
    for i in 0..chunks {
        let base = i * F32_LANES;
        let av = f32x16::from_slice(&a[base..]);
        let bv = f32x16::from_slice(&b[base..]);
        (av / bv).copy_to_slice(&mut out[base..base + F32_LANES]);
    }
    for i in (chunks * F32_LANES)..len {
        out[i] = a[i] / b[i];
    }
    out
}

/// SIMD f64 element-wise add with scalar: out[i] = a[i] + scalar
#[inline]
pub fn add_f64_scalar(a: &[f64], scalar: f64) -> Vec<f64> {
    let len = a.len();
    let mut out = vec![0.0f64; len];
    let chunks = len / F64_LANES;
    let sv = f64x8::splat(scalar);
    for i in 0..chunks {
        let base = i * F64_LANES;
        let av = f64x8::from_slice(&a[base..]);
        (av + sv).copy_to_slice(&mut out[base..base + F64_LANES]);
    }
    for i in (chunks * F64_LANES)..len {
        out[i] = a[i] + scalar;
    }
    out
}

/// SIMD f64 element-wise subtract scalar: out[i] = a[i] - scalar
#[inline]
pub fn sub_f64_scalar(a: &[f64], scalar: f64) -> Vec<f64> {
    let len = a.len();
    let mut out = vec![0.0f64; len];
    let chunks = len / F64_LANES;
    let sv = f64x8::splat(scalar);
    for i in 0..chunks {
        let base = i * F64_LANES;
        let av = f64x8::from_slice(&a[base..]);
        (av - sv).copy_to_slice(&mut out[base..base + F64_LANES]);
    }
    for i in (chunks * F64_LANES)..len {
        out[i] = a[i] - scalar;
    }
    out
}

/// SIMD f64 element-wise multiply with scalar: out[i] = a[i] * scalar
#[inline]
pub fn mul_f64_scalar(a: &[f64], scalar: f64) -> Vec<f64> {
    let len = a.len();
    let mut out = vec![0.0f64; len];
    let chunks = len / F64_LANES;
    let sv = f64x8::splat(scalar);
    for i in 0..chunks {
        let base = i * F64_LANES;
        let av = f64x8::from_slice(&a[base..]);
        (av * sv).copy_to_slice(&mut out[base..base + F64_LANES]);
    }
    for i in (chunks * F64_LANES)..len {
        out[i] = a[i] * scalar;
    }
    out
}

/// SIMD f64 element-wise divide by scalar: out[i] = a[i] / scalar
#[inline]
pub fn div_f64_scalar(a: &[f64], scalar: f64) -> Vec<f64> {
    let len = a.len();
    let mut out = vec![0.0f64; len];
    let chunks = len / F64_LANES;
    let sv = f64x8::splat(scalar);
    for i in 0..chunks {
        let base = i * F64_LANES;
        let av = f64x8::from_slice(&a[base..]);
        (av / sv).copy_to_slice(&mut out[base..base + F64_LANES]);
    }
    for i in (chunks * F64_LANES)..len {
        out[i] = a[i] / scalar;
    }
    out
}

/// SIMD f64 element-wise vector add: out[i] = a[i] + b[i]
#[inline]
pub fn add_f64_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut out = vec![0.0f64; len];
    let chunks = len / F64_LANES;
    for i in 0..chunks {
        let base = i * F64_LANES;
        let av = f64x8::from_slice(&a[base..]);
        let bv = f64x8::from_slice(&b[base..]);
        (av + bv).copy_to_slice(&mut out[base..base + F64_LANES]);
    }
    for i in (chunks * F64_LANES)..len {
        out[i] = a[i] + b[i];
    }
    out
}

/// SIMD f64 element-wise vector subtract: out[i] = a[i] - b[i]
#[inline]
pub fn sub_f64_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut out = vec![0.0f64; len];
    let chunks = len / F64_LANES;
    for i in 0..chunks {
        let base = i * F64_LANES;
        let av = f64x8::from_slice(&a[base..]);
        let bv = f64x8::from_slice(&b[base..]);
        (av - bv).copy_to_slice(&mut out[base..base + F64_LANES]);
    }
    for i in (chunks * F64_LANES)..len {
        out[i] = a[i] - b[i];
    }
    out
}

/// SIMD f64 element-wise vector multiply: out[i] = a[i] * b[i]
#[inline]
pub fn mul_f64_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut out = vec![0.0f64; len];
    let chunks = len / F64_LANES;
    for i in 0..chunks {
        let base = i * F64_LANES;
        let av = f64x8::from_slice(&a[base..]);
        let bv = f64x8::from_slice(&b[base..]);
        (av * bv).copy_to_slice(&mut out[base..base + F64_LANES]);
    }
    for i in (chunks * F64_LANES)..len {
        out[i] = a[i] * b[i];
    }
    out
}

/// SIMD f64 element-wise vector divide: out[i] = a[i] / b[i]
#[inline]
pub fn div_f64_vec(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut out = vec![0.0f64; len];
    let chunks = len / F64_LANES;
    for i in 0..chunks {
        let base = i * F64_LANES;
        let av = f64x8::from_slice(&a[base..]);
        let bv = f64x8::from_slice(&b[base..]);
        (av / bv).copy_to_slice(&mut out[base..base + F64_LANES]);
    }
    for i in (chunks * F64_LANES)..len {
        out[i] = a[i] / b[i];
    }
    out
}

// ============================================================================
// 3-Stroke Adaptive HDR Cascade Search
// ============================================================================

/// Result from HDR cascade search.
#[derive(Debug, Clone)]
pub struct HdrResult {
    /// Index into the database.
    pub index: usize,
    /// Exact Hamming distance (from Stroke 2).
    pub hamming: u64,
    /// Optional high-precision distance (from Stroke 3, if BF16/cosine enabled).
    /// f64::NAN if Stroke 3 was not run.
    pub precise: f64,
}

/// 3-Stroke Adaptive HDR Cascade Search.
///
/// Operates on raw `&[u8]` slices — the HDC/NumArray layer wraps this.
///
/// ## Stroke 1 (Belichtungsmesser)
///
/// Partial popcount on first `vec_bytes/16` bytes of each vector.
/// Warmup on first 128 candidates to estimate population μ and σ.
/// Reject candidates where scaled estimate > threshold + 3σ.
/// Kills ~98% of candidates.
///
/// ## Stroke 2 (Full resolution)
///
/// Incremental full Hamming on survivors (remaining bytes only).
/// Reject candidates where full distance > threshold.
/// Kills ~90% of Stroke 1 survivors.
///
/// ## Stroke 3 (HDR precision)
///
/// VNNI-accelerated dot_i8 → cosine similarity on finalists.
/// Only runs if `precise_tier` is true.
///
/// ## Performance model (100K × 2KB vectors, threshold=500)
///
/// | Stroke | Candidates | Ops/candidate | Total ops |
/// |--------|-----------|---------------|-----------|
/// | 1      | 100,000   | 2 VPOPCNTDQ  | 200,000   |
/// | 2      | ~2,000    | 32 VPOPCNTDQ | 64,000    |
/// | 3      | ~200      | 32 VNNI      | 6,400     |
/// | **Total** |        |              | **270,400** |
/// | Brute force |      |              | 3,200,000 |
/// | **Speedup** |      |              | **~12×**  |
pub fn hdr_cascade_search(
    query: &[u8],
    database: &[u8],
    vec_bytes: usize,
    num_vectors: usize,
    threshold: u64,
    precise_tier: bool,
) -> Vec<HdrResult> {
    assert_eq!(query.len(), vec_bytes);
    assert_eq!(database.len(), vec_bytes * num_vectors);

    let hamming_fn = select_hamming_fn();

    // ─── Configuration ───
    let s1_bytes = (vec_bytes / 16).max(64).min(vec_bytes); // At least 64 bytes (1 VPOPCNTDQ)
    let scale1 = (vec_bytes as f64) / (s1_bytes as f64);
    let warmup_n = 128.min(num_vectors);

    // For small vectors, skip cascade entirely
    if vec_bytes < 128 {
        let mut results = Vec::new();
        for i in 0..num_vectors {
            let base = i * vec_bytes;
            let d = hamming_fn(query, &database[base..base + vec_bytes]);
            if d <= threshold {
                results.push(HdrResult { index: i, hamming: d, precise: f64::NAN });
            }
        }
        if precise_tier && !results.is_empty() {
            apply_precision_tier(query, database, vec_bytes, &mut results);
        }
        return results;
    }

    // ════════════════════════════════════════════════════════
    // STROKE 1: Partial popcount with σ warmup
    // ════════════════════════════════════════════════════════

    // Compute σ_est: the estimation error of partial sampling at the threshold boundary.
    //
    // For a true Hamming distance d from N*8 total bits, sampling s bytes gives
    // d̂ = popcount(xor_prefix) × (N/s). The standard deviation of d̂ is:
    //   σ(d̂) = N × √(8 × p × (1−p) / s)
    // where p = d/(N×8) is the bit-flip probability.
    //
    // We compute σ at the threshold boundary (p = threshold / total_bits).
    // Additionally, we use warmup to get the population distribution for a tighter
    // secondary σ if the population is well-separated from the threshold.
    let query_prefix = &query[..s1_bytes];
    let total_bits = (vec_bytes * 8) as f64;
    let p_thresh = (threshold as f64 / total_bits).clamp(0.001, 0.999);
    let sigma_est = (vec_bytes as f64) * (8.0 * p_thresh * (1.0 - p_thresh) / s1_bytes as f64).sqrt();

    // Warmup: sample first warmup_n candidates to check if population σ is even wider
    let mut warmup_dists = Vec::with_capacity(warmup_n);
    for i in 0..warmup_n {
        let base = i * vec_bytes;
        let cand_prefix = &database[base..base + s1_bytes];
        let d = hamming_fn(query_prefix, cand_prefix);
        let estimate = (d as f64 * scale1) as u64;
        warmup_dists.push(estimate);
    }

    let var: f64 = {
        let mu: f64 = warmup_dists.iter().map(|&d| d as f64).sum::<f64>() / warmup_n as f64;
        warmup_dists.iter()
            .map(|&d| { let diff = d as f64 - mu; diff * diff })
            .sum::<f64>() / warmup_n as f64
    };
    let sigma_pop = var.sqrt();

    // Use the LARGER of estimation error σ and population σ.
    // σ_est captures the statistical uncertainty of partial sampling.
    // σ_pop captures the actual spread in the database.
    let sigma = sigma_est.max(sigma_pop).max(1.0);

    // Reject threshold: threshold + 3σ (Belichtungsmesser: one measurement, applied to all)
    let s1_reject = threshold as f64 + 3.0 * sigma;

    // Phase 1b: Scan ALL candidates (re-checking warmup is fine — tiny cost)
    let mut survivors: Vec<(usize, u64)> = Vec::with_capacity(num_vectors / 20);

    for i in 0..num_vectors {
        let base = i * vec_bytes;
        let cand_prefix = &database[base..base + s1_bytes];
        let d = hamming_fn(query_prefix, cand_prefix);
        let estimate = (d as f64 * scale1) as u64;

        if (estimate as f64) <= s1_reject {
            survivors.push((i, d)); // carry partial distance for incremental Stroke 2
        }
    }

    // ════════════════════════════════════════════════════════
    // STROKE 2: Full Hamming on survivors (incremental)
    // ════════════════════════════════════════════════════════

    let mut finalists: Vec<HdrResult> = Vec::with_capacity(survivors.len() / 5 + 1);
    let query_rest = &query[s1_bytes..];

    for &(idx, d_prefix) in &survivors {
        let base = idx * vec_bytes;
        // Incremental: compute remaining bytes only
        let d_rest = hamming_fn(query_rest, &database[base + s1_bytes..base + vec_bytes]);
        let d_full = d_prefix + d_rest;

        if d_full <= threshold {
            finalists.push(HdrResult {
                index: idx,
                hamming: d_full,
                precise: f64::NAN,
            });
        }
    }

    // ════════════════════════════════════════════════════════
    // STROKE 3: High-precision distance (optional VNNI cosine)
    // ════════════════════════════════════════════════════════

    if precise_tier && !finalists.is_empty() {
        apply_precision_tier(query, database, vec_bytes, &mut finalists);
    }

    finalists
}

/// Stroke 3: compute VNNI dot_i8 → cosine similarity for finalists.
/// Sorts by cosine descending (best first).
fn apply_precision_tier(
    query: &[u8],
    database: &[u8],
    vec_bytes: usize,
    finalists: &mut Vec<HdrResult>,
) {
    let dot_fn = select_dot_i8_fn();

    let query_norm_sq = dot_fn(query, query);
    let query_norm = (query_norm_sq as f64).sqrt();

    if query_norm == 0.0 {
        for r in finalists.iter_mut() {
            r.precise = 0.0;
        }
        return;
    }

    for r in finalists.iter_mut() {
        let base = r.index * vec_bytes;
        let candidate = &database[base..base + vec_bytes];

        let dot = dot_fn(query, candidate);
        let cand_norm_sq = dot_fn(candidate, candidate);
        let cand_norm = (cand_norm_sq as f64).sqrt();

        r.precise = if cand_norm > 0.0 {
            dot as f64 / (query_norm * cand_norm)
        } else {
            0.0
        };
    }

    // Sort by cosine descending (most similar first)
    finalists.sort_unstable_by(|a, b| {
        b.precise.partial_cmp(&a.precise).unwrap_or(std::cmp::Ordering::Equal)
    });
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

    // ---- select_hamming_fn / select_dot_i8_fn ----

    #[test]
    fn test_select_hamming_fn() {
        let f = select_hamming_fn();
        let a = vec![0xFFu8; 128];
        let b = vec![0x00u8; 128];
        assert_eq!(f(&a, &b), 128 * 8);
    }

    #[test]
    fn test_select_dot_i8_fn() {
        let f = select_dot_i8_fn();
        let a = vec![1u8; 64];
        let b = vec![1u8; 64];
        assert_eq!(f(&a, &b), 64); // 1*1 * 64
    }

    // ---- HDR Cascade Search ----

    #[test]
    fn test_hdr_cascade_basic() {
        let vec_len = 2048;
        let query = vec![0xAAu8; vec_len];
        let mut db = Vec::new();
        db.extend(vec![0xAA; vec_len]); // vec 0: identical
        db.extend(vec![0x55; vec_len]); // vec 1: maximally different
        db.extend(vec![0xAA; vec_len]); // vec 2: identical
        db.extend(vec![0x00; vec_len]); // vec 3: very different

        let results = hdr_cascade_search(&query, &db, vec_len, 4, 100, false);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].index, 0);
        assert_eq!(results[0].hamming, 0);
        assert_eq!(results[1].index, 2);
        assert_eq!(results[1].hamming, 0);
    }

    #[test]
    fn test_hdr_warmup_sigma() {
        let vec_len = 2048;
        let num_random = 1000;
        let num_close = 5;
        let total = num_random + num_close;
        let mut db = vec![0u8; vec_len * total];

        // Fill with pseudo-random data
        for i in 0..num_random * vec_len {
            db[i] = ((i * 7 + 13) % 256) as u8;
        }

        let query = vec![0xAA; vec_len];
        // Plant 5 close matches (copy query with small perturbation)
        for m in 0..num_close {
            let base = (num_random + m) * vec_len;
            db[base..base + vec_len].copy_from_slice(&query);
            // Flip ~50 bytes spread evenly
            for j in 0..50 {
                db[base + j * 40] ^= 0xFF;
            }
        }

        let results = hdr_cascade_search(&query, &db, vec_len, total, 500, false);
        assert!(results.len() >= num_close, "Expected at least {} matches, got {}", num_close, results.len());
        // All planted matches should appear
        let indices: Vec<usize> = results.iter().map(|r| r.index).collect();
        for m in 0..num_close {
            assert!(indices.contains(&(num_random + m)), "Missing planted match {}", num_random + m);
        }
    }

    #[test]
    fn test_hdr_precision_tier() {
        let vec_len = 2048;
        let query = vec![0xAA; vec_len];
        let mut db = Vec::new();
        db.extend(vec![0xAA; vec_len]); // identical → cosine ≈ 1.0
        db.extend(vec![0x55; vec_len]); // maximally different (won't survive)

        let results = hdr_cascade_search(&query, &db, vec_len, 2, 100, true);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 0);
        assert!(results[0].precise.is_finite());
        // Cosine of identical vectors should be ~1.0
        assert!((results[0].precise - 1.0).abs() < 0.01,
            "Expected cosine ~1.0, got {}", results[0].precise);
    }

    #[test]
    fn test_hdr_precision_tier_ranks() {
        let vec_len = 2048;
        let query = vec![0xAA; vec_len];

        // Two candidates with similar Hamming but different patterns
        let mut cand_a = query.clone();
        let mut cand_b = query.clone();
        // Flip same NUMBER of bits but different byte positions
        for i in 0..30 { cand_a[i] ^= 0xFF; }
        for i in 500..530 { cand_b[i] ^= 0xFF; }

        let mut db = Vec::new();
        db.extend_from_slice(&cand_a);
        db.extend_from_slice(&cand_b);

        let results = hdr_cascade_search(&query, &db, vec_len, 2, 300, true);
        assert_eq!(results.len(), 2);
        // Both have same Hamming distance (30*8 = 240 bits)
        assert_eq!(results[0].hamming, results[1].hamming);
        // Precision tier provides cosine ranking
        assert!(results[0].precise.is_finite());
        assert!(results[1].precise.is_finite());
    }
}
