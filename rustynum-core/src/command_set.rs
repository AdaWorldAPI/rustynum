//! CPU Command Set: detect once, dispatch forever.
//!
//! ONE `OnceLock` lookup at first call resolves the entire SIMD backend.
//! Every public function in `simd.rs` becomes a one-line delegate.
//! No per-call `is_x86_feature_detected!`. Zero dispatch overhead after init.

use std::sync::OnceLock;

// ─── Function pointer types ────────────────────────────────────────

// BLAS-1
type DotF32 = fn(&[f32], &[f32]) -> f32;
type DotF64 = fn(&[f64], &[f64]) -> f64;
type AxpyF32 = fn(f32, &[f32], &mut [f32]);
type AxpyF64 = fn(f64, &[f64], &mut [f64]);
type ScalF32 = fn(f32, &mut [f32]);
type ScalF64 = fn(f64, &mut [f64]);
type AsumF32 = fn(&[f32]) -> f32;
type AsumF64 = fn(&[f64]) -> f64;
type Nrm2F32 = fn(&[f32]) -> f32;
type Nrm2F64 = fn(&[f64]) -> f64;
type IamaxF32 = fn(&[f32]) -> (usize, f32);
type IamaxF64 = fn(&[f64]) -> (usize, f64);

// Element-wise (scalar and vec variants share the same type per element type)
type EwScalarF32 = fn(&[f32], f32) -> Vec<f32>;
type EwScalarF64 = fn(&[f64], f64) -> Vec<f64>;
type EwVecF32 = fn(&[f32], &[f32]) -> Vec<f32>;
type EwVecF64 = fn(&[f64], &[f64]) -> Vec<f64>;

// Hamming / bitops
type Hamming = fn(&[u8], &[u8]) -> u64;
type Popcount = fn(&[u8]) -> u64;
type DotI8 = fn(&[u8], &[u8]) -> i64;

// ─── The Command Set ───────────────────────────────────────────────

pub struct CommandSet {
    // BLAS-1
    pub dot_f32: DotF32,
    pub dot_f64: DotF64,
    pub axpy_f32: AxpyF32,
    pub axpy_f64: AxpyF64,
    pub scal_f32: ScalF32,
    pub scal_f64: ScalF64,
    pub asum_f32: AsumF32,
    pub asum_f64: AsumF64,
    pub nrm2_f32: Nrm2F32,
    pub nrm2_f64: Nrm2F64,
    pub iamax_f32: IamaxF32,
    pub iamax_f64: IamaxF64,

    // Element-wise f32
    pub add_f32_scalar: EwScalarF32,
    pub sub_f32_scalar: EwScalarF32,
    pub mul_f32_scalar: EwScalarF32,
    pub div_f32_scalar: EwScalarF32,
    pub add_f32_vec: EwVecF32,
    pub sub_f32_vec: EwVecF32,
    pub mul_f32_vec: EwVecF32,
    pub div_f32_vec: EwVecF32,

    // Element-wise f64
    pub add_f64_scalar: EwScalarF64,
    pub sub_f64_scalar: EwScalarF64,
    pub mul_f64_scalar: EwScalarF64,
    pub div_f64_scalar: EwScalarF64,
    pub add_f64_vec: EwVecF64,
    pub sub_f64_vec: EwVecF64,
    pub mul_f64_vec: EwVecF64,
    pub div_f64_vec: EwVecF64,

    // Hamming / bitops
    pub hamming: Hamming,
    pub popcount: Popcount,
    pub dot_i8: DotI8,
}

// ─── Detection + resolution ────────────────────────────────────────

static OPS: OnceLock<CommandSet> = OnceLock::new();

/// Get the command set for this CPU. Detected once, cached forever.
#[inline]
pub fn ops() -> &'static CommandSet {
    OPS.get_or_init(detect)
}

fn detect() -> CommandSet {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return avx512_command_set();
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return avx2_command_set();
        }
    }
    scalar_command_set()
}

// ─── AVX-512 command set ───────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn avx512_command_set() -> CommandSet {
    // Import the _avx512 functions from simd.rs (the existing code, wrapped)
    use super::simd_avx512_fns::*;

    CommandSet {
        dot_f32: dot_f32_avx512,
        dot_f64: dot_f64_avx512,
        axpy_f32: axpy_f32_avx512,
        axpy_f64: axpy_f64_avx512,
        scal_f32: scal_f32_avx512,
        scal_f64: scal_f64_avx512,
        asum_f32: asum_f32_avx512,
        asum_f64: asum_f64_avx512,
        nrm2_f32: nrm2_f32_avx512,
        nrm2_f64: nrm2_f64_avx512,
        iamax_f32: iamax_f32_avx512,
        iamax_f64: iamax_f64_avx512,

        add_f32_scalar: add_f32_scalar_avx512,
        sub_f32_scalar: sub_f32_scalar_avx512,
        mul_f32_scalar: mul_f32_scalar_avx512,
        div_f32_scalar: div_f32_scalar_avx512,
        add_f32_vec: add_f32_vec_avx512,
        sub_f32_vec: sub_f32_vec_avx512,
        mul_f32_vec: mul_f32_vec_avx512,
        div_f32_vec: div_f32_vec_avx512,

        add_f64_scalar: add_f64_scalar_avx512,
        sub_f64_scalar: sub_f64_scalar_avx512,
        mul_f64_scalar: mul_f64_scalar_avx512,
        div_f64_scalar: div_f64_scalar_avx512,
        add_f64_vec: add_f64_vec_avx512,
        sub_f64_vec: sub_f64_vec_avx512,
        mul_f64_vec: mul_f64_vec_avx512,
        div_f64_vec: div_f64_vec_avx512,

        hamming: hamming_avx512,
        popcount: popcount_avx512,
        dot_i8: dot_i8_avx512,
    }
}

// ─── AVX2 command set ──────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
fn avx2_command_set() -> CommandSet {
    use super::simd_avx2;
    use super::scalar_fns::*;

    CommandSet {
        // BLAS-1: AVX2 implementations exist in simd_avx2.rs
        dot_f32: simd_avx2::dot_f32,
        dot_f64: simd_avx2::dot_f64,
        axpy_f32: simd_avx2::axpy_f32,
        axpy_f64: simd_avx2::axpy_f64,
        scal_f32: simd_avx2::scal_f32,
        scal_f64: simd_avx2::scal_f64,
        asum_f32: simd_avx2::asum_f32,
        asum_f64: simd_avx2::asum_f64,
        nrm2_f32: simd_avx2::nrm2_f32,
        nrm2_f64: simd_avx2::nrm2_f64,

        // iamax: no AVX2 impl → scalar
        iamax_f32: iamax_f32_scalar,
        iamax_f64: iamax_f64_scalar,

        // Element-wise: no AVX2 impl → scalar (LLVM auto-vectorizes to AVX2)
        add_f32_scalar: add_f32_scalar_fn,
        sub_f32_scalar: sub_f32_scalar_fn,
        mul_f32_scalar: mul_f32_scalar_fn,
        div_f32_scalar: div_f32_scalar_fn,
        add_f32_vec: add_f32_vec_fn,
        sub_f32_vec: sub_f32_vec_fn,
        mul_f32_vec: mul_f32_vec_fn,
        div_f32_vec: div_f32_vec_fn,

        add_f64_scalar: add_f64_scalar_fn,
        sub_f64_scalar: sub_f64_scalar_fn,
        mul_f64_scalar: mul_f64_scalar_fn,
        div_f64_scalar: div_f64_scalar_fn,
        add_f64_vec: add_f64_vec_fn,
        sub_f64_vec: sub_f64_vec_fn,
        mul_f64_vec: mul_f64_vec_fn,
        div_f64_vec: div_f64_vec_fn,

        // Hamming: AVX2 versions exist in simd_avx2.rs
        hamming: simd_avx2::hamming_distance,
        popcount: popcount_scalar,  // simd_avx2 doesn't export standalone popcount
        dot_i8: dot_i8_scalar,      // no AVX2 dot_i8 yet (Session F gap)
    }
}

// ─── Scalar command set ────────────────────────────────────────────

fn scalar_command_set() -> CommandSet {
    use super::scalar_fns::*;

    CommandSet {
        dot_f32: dot_f32_scalar,
        dot_f64: dot_f64_scalar,
        axpy_f32: axpy_f32_scalar,
        axpy_f64: axpy_f64_scalar,
        scal_f32: scal_f32_scalar,
        scal_f64: scal_f64_scalar,
        asum_f32: asum_f32_scalar,
        asum_f64: asum_f64_scalar,
        nrm2_f32: nrm2_f32_scalar,
        nrm2_f64: nrm2_f64_scalar,
        iamax_f32: iamax_f32_scalar,
        iamax_f64: iamax_f64_scalar,

        add_f32_scalar: add_f32_scalar_fn,
        sub_f32_scalar: sub_f32_scalar_fn,
        mul_f32_scalar: mul_f32_scalar_fn,
        div_f32_scalar: div_f32_scalar_fn,
        add_f32_vec: add_f32_vec_fn,
        sub_f32_vec: sub_f32_vec_fn,
        mul_f32_vec: mul_f32_vec_fn,
        div_f32_vec: div_f32_vec_fn,

        add_f64_scalar: add_f64_scalar_fn,
        sub_f64_scalar: sub_f64_scalar_fn,
        mul_f64_scalar: mul_f64_scalar_fn,
        div_f64_scalar: div_f64_scalar_fn,
        add_f64_vec: add_f64_vec_fn,
        sub_f64_vec: sub_f64_vec_fn,
        mul_f64_vec: mul_f64_vec_fn,
        div_f64_vec: div_f64_vec_fn,

        hamming: hamming_scalar,
        popcount: popcount_scalar,
        dot_i8: dot_i8_scalar,
    }
}
