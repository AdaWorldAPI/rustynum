//! Scalar fallback implementations for every SIMD operation.
//! These run on ANY architecture. LLVM may auto-vectorize.

// ─── BLAS-1 ────────────────────────────────────────────────────────

pub fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn dot_f64_scalar(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn axpy_f32_scalar(alpha: f32, x: &[f32], y: &mut [f32]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

pub fn axpy_f64_scalar(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

pub fn scal_f32_scalar(alpha: f32, x: &mut [f32]) {
    for v in x.iter_mut() {
        *v *= alpha;
    }
}

pub fn scal_f64_scalar(alpha: f64, x: &mut [f64]) {
    for v in x.iter_mut() {
        *v *= alpha;
    }
}

pub fn asum_f32_scalar(x: &[f32]) -> f32 {
    x.iter().map(|v| v.abs()).sum()
}

pub fn asum_f64_scalar(x: &[f64]) -> f64 {
    x.iter().map(|v| v.abs()).sum()
}

pub fn nrm2_f32_scalar(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

pub fn nrm2_f64_scalar(x: &[f64]) -> f64 {
    x.iter().map(|v| v * v).sum::<f64>().sqrt()
}

pub fn iamax_f32_scalar(x: &[f32]) -> (usize, f32) {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        let abs = v.abs();
        if abs > max_val {
            max_val = abs;
            max_idx = i;
        }
    }
    (max_idx, x.get(max_idx).copied().unwrap_or(0.0))
}

pub fn iamax_f64_scalar(x: &[f64]) -> (usize, f64) {
    let mut max_idx = 0;
    let mut max_val = f64::NEG_INFINITY;
    for (i, &v) in x.iter().enumerate() {
        let abs = v.abs();
        if abs > max_val {
            max_val = abs;
            max_idx = i;
        }
    }
    (max_idx, x.get(max_idx).copied().unwrap_or(0.0))
}

// ─── Element-wise f32 ──────────────────────────────────────────────

pub fn add_f32_scalar_fn(a: &[f32], scalar: f32) -> Vec<f32> {
    a.iter().map(|&v| v + scalar).collect()
}

pub fn sub_f32_scalar_fn(a: &[f32], scalar: f32) -> Vec<f32> {
    a.iter().map(|&v| v - scalar).collect()
}

pub fn mul_f32_scalar_fn(a: &[f32], scalar: f32) -> Vec<f32> {
    a.iter().map(|&v| v * scalar).collect()
}

pub fn div_f32_scalar_fn(a: &[f32], scalar: f32) -> Vec<f32> {
    a.iter().map(|&v| v / scalar).collect()
}

pub fn add_f32_vec_fn(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

pub fn sub_f32_vec_fn(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}

pub fn mul_f32_vec_fn(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
}

pub fn div_f32_vec_fn(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x / y).collect()
}

// ─── Element-wise f64 ──────────────────────────────────────────────

pub fn add_f64_scalar_fn(a: &[f64], scalar: f64) -> Vec<f64> {
    a.iter().map(|&v| v + scalar).collect()
}

pub fn sub_f64_scalar_fn(a: &[f64], scalar: f64) -> Vec<f64> {
    a.iter().map(|&v| v - scalar).collect()
}

pub fn mul_f64_scalar_fn(a: &[f64], scalar: f64) -> Vec<f64> {
    a.iter().map(|&v| v * scalar).collect()
}

pub fn div_f64_scalar_fn(a: &[f64], scalar: f64) -> Vec<f64> {
    a.iter().map(|&v| v / scalar).collect()
}

pub fn add_f64_vec_fn(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

pub fn sub_f64_vec_fn(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}

pub fn mul_f64_vec_fn(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect()
}

pub fn div_f64_vec_fn(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x / y).collect()
}

// ─── Hamming / bitops ──────────────────────────────────────────────

pub fn hamming_scalar(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let u64_chunks = len / 8;
    let mut sum: u64 = 0;
    for i in 0..u64_chunks {
        let base = i * 8;
        let wa = u64::from_ne_bytes(a[base..base + 8].try_into().unwrap());
        let wb = u64::from_ne_bytes(b[base..base + 8].try_into().unwrap());
        sum += (wa ^ wb).count_ones() as u64;
    }
    for i in (u64_chunks * 8)..len {
        sum += (a[i] ^ b[i]).count_ones() as u64;
    }
    sum
}

pub fn popcount_scalar(a: &[u8]) -> u64 {
    let len = a.len();
    let u64_chunks = len / 8;
    let mut sum: u64 = 0;
    for i in 0..u64_chunks {
        let base = i * 8;
        let w = u64::from_ne_bytes(a[base..base + 8].try_into().unwrap());
        sum += w.count_ones() as u64;
    }
    for &byte in &a[u64_chunks * 8..] {
        sum += byte.count_ones() as u64;
    }
    sum
}

pub fn dot_i8_scalar(a: &[u8], b: &[u8]) -> i64 {
    assert_eq!(a.len(), b.len());
    let mut total: i64 = 0;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        total += (ai as i8 as i64) * (bi as i8 as i64);
    }
    total
}

// ─── Plain-name aliases (for dispatch! macro) ────────────────────────

pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 { dot_f32_scalar(a, b) }
pub fn dot_f64(a: &[f64], b: &[f64]) -> f64 { dot_f64_scalar(a, b) }
pub fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) { axpy_f32_scalar(alpha, x, y) }
pub fn axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) { axpy_f64_scalar(alpha, x, y) }
pub fn scal_f32(alpha: f32, x: &mut [f32]) { scal_f32_scalar(alpha, x) }
pub fn scal_f64(alpha: f64, x: &mut [f64]) { scal_f64_scalar(alpha, x) }
pub fn asum_f32(x: &[f32]) -> f32 { asum_f32_scalar(x) }
pub fn asum_f64(x: &[f64]) -> f64 { asum_f64_scalar(x) }
pub fn nrm2_f32(x: &[f32]) -> f32 { nrm2_f32_scalar(x) }
pub fn nrm2_f64(x: &[f64]) -> f64 { nrm2_f64_scalar(x) }
pub fn iamax_f32(x: &[f32]) -> (usize, f32) { iamax_f32_scalar(x) }
pub fn iamax_f64(x: &[f64]) -> (usize, f64) { iamax_f64_scalar(x) }

pub fn add_f32_scalar(a: &[f32], scalar: f32) -> Vec<f32> { add_f32_scalar_fn(a, scalar) }
pub fn sub_f32_scalar(a: &[f32], scalar: f32) -> Vec<f32> { sub_f32_scalar_fn(a, scalar) }
pub fn mul_f32_scalar(a: &[f32], scalar: f32) -> Vec<f32> { mul_f32_scalar_fn(a, scalar) }
pub fn div_f32_scalar(a: &[f32], scalar: f32) -> Vec<f32> { div_f32_scalar_fn(a, scalar) }
pub fn add_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32> { add_f32_vec_fn(a, b) }
pub fn sub_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32> { sub_f32_vec_fn(a, b) }
pub fn mul_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32> { mul_f32_vec_fn(a, b) }
pub fn div_f32_vec(a: &[f32], b: &[f32]) -> Vec<f32> { div_f32_vec_fn(a, b) }

pub fn add_f64_scalar(a: &[f64], scalar: f64) -> Vec<f64> { add_f64_scalar_fn(a, scalar) }
pub fn sub_f64_scalar(a: &[f64], scalar: f64) -> Vec<f64> { sub_f64_scalar_fn(a, scalar) }
pub fn mul_f64_scalar(a: &[f64], scalar: f64) -> Vec<f64> { mul_f64_scalar_fn(a, scalar) }
pub fn div_f64_scalar(a: &[f64], scalar: f64) -> Vec<f64> { div_f64_scalar_fn(a, scalar) }
pub fn add_f64_vec(a: &[f64], b: &[f64]) -> Vec<f64> { add_f64_vec_fn(a, b) }
pub fn sub_f64_vec(a: &[f64], b: &[f64]) -> Vec<f64> { sub_f64_vec_fn(a, b) }
pub fn mul_f64_vec(a: &[f64], b: &[f64]) -> Vec<f64> { mul_f64_vec_fn(a, b) }
pub fn div_f64_vec(a: &[f64], b: &[f64]) -> Vec<f64> { div_f64_vec_fn(a, b) }

pub fn hamming_distance(a: &[u8], b: &[u8]) -> u64 { hamming_scalar(a, b) }
pub fn popcount(a: &[u8]) -> u64 { popcount_scalar(a) }
pub fn dot_i8(a: &[u8], b: &[u8]) -> i64 { dot_i8_scalar(a, b) }

pub fn hamming_batch(query: &[u8], database: &[u8], num_rows: usize, row_bytes: usize) -> Vec<u64> {
    assert_eq!(query.len(), row_bytes);
    assert_eq!(database.len(), num_rows * row_bytes);
    let mut distances = vec![0u64; num_rows];
    for i in 0..num_rows {
        distances[i] = hamming_scalar(query, &database[i * row_bytes..(i + 1) * row_bytes]);
    }
    distances
}

pub fn hamming_top_k(
    query: &[u8],
    database: &[u8],
    num_rows: usize,
    row_bytes: usize,
    k: usize,
) -> (Vec<usize>, Vec<u64>) {
    let distances = hamming_batch(query, database, num_rows, row_bytes);
    let k = k.min(num_rows);
    let mut indices: Vec<usize> = (0..num_rows).collect();
    indices.select_nth_unstable_by_key(k.saturating_sub(1), |&i| distances[i]);
    indices.truncate(k);
    indices.sort_unstable_by_key(|&i| distances[i]);
    let top_distances: Vec<u64> = indices.iter().map(|&i| distances[i]).collect();
    (indices, top_distances)
}

// ─── GEMM — scalar blocked fallback ─────────────────────────────────

/// Scalar blocked SGEMM: C = alpha * A * B + C
///
/// Row-major layout. A is m x k (stride lda), B is k x n (stride ldb),
/// C is m x n (stride ldc). Beta already applied by caller.
pub fn sgemm_blocked(
    m: usize, n: usize, k: usize,
    alpha: f32, a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    c: &mut [f32], ldc: usize,
) {
    // Simple ijk loop — LLVM may auto-vectorize
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * lda + p] * b[p * ldb + j];
            }
            c[i * ldc + j] += alpha * sum;
        }
    }
}

/// Scalar blocked DGEMM: C = alpha * A * B + C
///
/// Row-major layout. A is m x k (stride lda), B is k x n (stride ldb),
/// C is m x n (stride ldc). Beta already applied by caller.
pub fn dgemm_blocked(
    m: usize, n: usize, k: usize,
    alpha: f64, a: &[f64], lda: usize,
    b: &[f64], ldb: usize,
    c: &mut [f64], ldc: usize,
) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                sum += a[i * lda + p] * b[p * ldb + j];
            }
            c[i * ldc + j] += alpha * sum;
        }
    }
}
