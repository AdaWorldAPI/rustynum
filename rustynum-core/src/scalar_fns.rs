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
