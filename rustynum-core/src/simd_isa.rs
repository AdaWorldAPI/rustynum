//! ISA trait: bridge between our stable types and std::simd portable_simd.
//!
//! Our `simd_avx512.rs` IS the stable portable_simd — method names match.
//! This trait lets kernels compile against either backend via feature flag.
//!
//! - **Default (stable)**: `simd_avx512.rs` types. Production. Ships everywhere.
//! - **Optional (nightly `--features portable_simd`)**: `std::simd`. All architectures.

/// Instruction-set abstraction. Kernels parameterised by `<I: Isa>` compile
/// against either our stable AVX-512 wrappers or nightly portable_simd.
pub trait Isa: Copy + 'static {
    type F32: Copy
        + std::ops::Add<Output = Self::F32>
        + std::ops::Sub<Output = Self::F32>
        + std::ops::Mul<Output = Self::F32>;
    type F64: Copy
        + std::ops::Add<Output = Self::F64>
        + std::ops::Sub<Output = Self::F64>
        + std::ops::Mul<Output = Self::F64>;
    type U8: Copy;

    const F32_LANES: usize;
    const LABEL: &'static str;

    // Construction
    fn f32_splat(val: f32) -> Self::F32;
    fn f32_zero() -> Self::F32;
    fn f32_load(slice: &[f32]) -> Self::F32;
    fn f32_store(val: Self::F32, slice: &mut [f32]);

    fn f64_splat(val: f64) -> Self::F64;
    fn f64_zero() -> Self::F64;

    // Arithmetic
    fn f32_fmadd(a: Self::F32, b: Self::F32, c: Self::F32) -> Self::F32;
    fn f32_reduce_sum(a: Self::F32) -> f32;
    fn f64_reduce_sum(a: Self::F64) -> f64;

    // Bitwise / popcount
    fn u8_xor(a: Self::U8, b: Self::U8) -> Self::U8;
    fn u8_popcnt_bytes(a: &[u8]) -> u64;
}

// ═══════════════════════════════════════════════════════════════════════════
// STABLE (default): forward to our simd_avx512.rs wrappers
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(not(feature = "portable_simd"))]
mod stable_impl {
    use super::Isa;
    use crate::simd_avx512::{F32x16, F64x8, U8x64};

    #[derive(Clone, Copy)]
    pub struct Native;

    impl Isa for Native {
        type F32 = F32x16;
        type F64 = F64x8;
        type U8 = U8x64;

        const F32_LANES: usize = 16;
        const LABEL: &'static str = "avx512-stable";

        #[inline(always)]
        fn f32_splat(v: f32) -> Self::F32 {
            F32x16::splat(v)
        }
        #[inline(always)]
        fn f32_zero() -> Self::F32 {
            F32x16::splat(0.0)
        }
        #[inline(always)]
        fn f32_load(s: &[f32]) -> Self::F32 {
            F32x16::from_slice(s)
        }
        #[inline(always)]
        fn f32_store(v: Self::F32, s: &mut [f32]) {
            v.copy_to_slice(s);
        }

        #[inline(always)]
        fn f64_splat(v: f64) -> Self::F64 {
            F64x8::splat(v)
        }
        #[inline(always)]
        fn f64_zero() -> Self::F64 {
            F64x8::splat(0.0)
        }

        #[inline(always)]
        fn f32_fmadd(a: Self::F32, b: Self::F32, c: Self::F32) -> Self::F32 {
            a.mul_add(b, c)
        }
        #[inline(always)]
        fn f32_reduce_sum(a: Self::F32) -> f32 {
            a.reduce_sum()
        }
        #[inline(always)]
        fn f64_reduce_sum(a: Self::F64) -> f64 {
            a.reduce_sum()
        }

        #[inline(always)]
        fn u8_xor(a: Self::U8, b: Self::U8) -> Self::U8 {
            a ^ b
        }
        #[inline(always)]
        fn u8_popcnt_bytes(a: &[u8]) -> u64 {
            // Delegate to the runtime-dispatched popcount in simd.rs
            #[cfg(feature = "avx512")]
            {
                crate::simd::popcount(a)
            }
            #[cfg(not(feature = "avx512"))]
            {
                a.iter().map(|b| b.count_ones() as u64).sum()
            }
        }
    }
}

#[cfg(not(feature = "portable_simd"))]
pub use stable_impl::Native;

// ═══════════════════════════════════════════════════════════════════════════
// NIGHTLY (opt-in --features portable_simd): forward to std::simd
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(feature = "portable_simd")]
mod portable_impl {
    use super::Isa;
    use std::simd::prelude::*;
    use std::simd::*;

    #[derive(Clone, Copy)]
    pub struct Native;

    impl Isa for Native {
        type F32 = f32x16;
        type F64 = f64x8;
        type U8 = u8x64;

        const F32_LANES: usize = 16;
        const LABEL: &'static str = "portable_simd";

        #[inline(always)]
        fn f32_splat(v: f32) -> Self::F32 {
            f32x16::splat(v)
        }
        #[inline(always)]
        fn f32_zero() -> Self::F32 {
            f32x16::splat(0.0)
        }
        #[inline(always)]
        fn f32_load(s: &[f32]) -> Self::F32 {
            f32x16::from_slice(s)
        }
        #[inline(always)]
        fn f32_store(v: Self::F32, s: &mut [f32]) {
            v.copy_to_slice(s);
        }

        #[inline(always)]
        fn f64_splat(v: f64) -> Self::F64 {
            f64x8::splat(v)
        }
        #[inline(always)]
        fn f64_zero() -> Self::F64 {
            f64x8::splat(0.0)
        }

        #[inline(always)]
        fn f32_fmadd(a: Self::F32, b: Self::F32, c: Self::F32) -> Self::F32 {
            // SimdFloat::mul_add
            a.mul_add(b, c)
        }
        #[inline(always)]
        fn f32_reduce_sum(a: Self::F32) -> f32 {
            a.reduce_sum()
        }
        #[inline(always)]
        fn f64_reduce_sum(a: Self::F64) -> f64 {
            a.reduce_sum()
        }

        #[inline(always)]
        fn u8_xor(a: Self::U8, b: Self::U8) -> Self::U8 {
            a ^ b
        }
        #[inline(always)]
        fn u8_popcnt_bytes(a: &[u8]) -> u64 {
            // portable_simd doesn't have direct VPOPCNTDQ; fall back to scalar
            a.iter().map(|b| b.count_ones() as u64).sum()
        }
    }
}

#[cfg(feature = "portable_simd")]
pub use portable_impl::Native;

/// Runtime ISA dispatch at kernel entry points.
///
/// Selects the best available ISA at runtime (AVX-512 → AVX2+FMA → scalar).
/// Each closure receives no ISA parameter — use [`Native`] inside, which
/// resolves at compile time via the `avx512` / `portable_simd` feature.
#[inline]
pub fn dispatch<R>(
    f_avx512: impl FnOnce() -> R,
    f_avx2: impl FnOnce() -> R,
    f_scalar: impl FnOnce() -> R,
) -> R {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return f_avx512();
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return f_avx2();
        }
    }
    f_scalar()
}
