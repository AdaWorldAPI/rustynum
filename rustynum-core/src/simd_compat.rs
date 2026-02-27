//! AVX-512 SIMD compatibility layer — stable Rust std::arch wrappers.
//!
//! Drop-in replacement for `std::simd` portable_simd types. Provides the same
//! API surface (methods, operators, type names) backed by `std::arch::x86_64`
//! intrinsics. All intrinsics used here are stable on Rust 1.89+.
//!
//! # Types
//!
//! | Compat type | portable_simd equiv | Backing type | Width |
//! |-------------|--------------------|--------------| ------|
//! | `F32x16`    | `f32x16`           | `__m512`     | 512b  |
//! | `F64x8`     | `f64x8`            | `__m512d`    | 512b  |
//! | `U8x64`     | `u8x64`            | `__m512i`    | 512b  |
//! | `I32x16`    | `i32x16`           | `__m512i`    | 512b  |
//! | `I64x8`     | `i64x8`            | `__m512i`    | 512b  |
//! | `U32x16`    | `u32x16`           | `__m512i`    | 512b  |
//! | `U64x8`     | `u64x8`            | `__m512i`    | 512b  |
//!
//! # Migration guide
//!
//! ```rust,ignore
//! // Before (nightly):
//! use std::simd::f32x16;
//! use std::simd::num::SimdFloat;
//!
//! // After (stable 1.93):
//! use rustynum_core::simd_compat::f32x16;
//! // No trait imports needed — all methods are inherent.
//! ```

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use core::fmt;
use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign,
    Mul, MulAssign, Neg, Not, Shl, Shr, Sub, SubAssign,
};

// ============================================================================
// Operator macros — reduce boilerplate for the 7 wrapper types
// ============================================================================

macro_rules! impl_bin_op {
    ($ty:ident, $trait:ident, $method:ident, $intr:path) => {
        impl $trait for $ty {
            type Output = Self;
            #[inline(always)]
            fn $method(self, rhs: Self) -> Self {
                Self(unsafe { $intr(self.0, rhs.0) })
            }
        }
    };
}

macro_rules! impl_assign_op {
    ($ty:ident, $trait:ident, $method:ident, $intr:path) => {
        impl $trait for $ty {
            #[inline(always)]
            fn $method(&mut self, rhs: Self) {
                self.0 = unsafe { $intr(self.0, rhs.0) };
            }
        }
    };
}

// ============================================================================
// F32x16 — 16 × f32 in one AVX-512 register (__m512)
// ============================================================================

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F32x16(pub __m512);

impl F32x16 {
    pub const LANES: usize = 16;

    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        Self(unsafe { _mm512_set1_ps(v) })
    }

    #[inline(always)]
    pub fn from_slice(s: &[f32]) -> Self {
        assert!(s.len() >= 16);
        Self(unsafe { _mm512_loadu_ps(s.as_ptr()) })
    }

    #[inline(always)]
    pub fn from_array(arr: [f32; 16]) -> Self {
        Self(unsafe { _mm512_loadu_ps(arr.as_ptr()) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 16] {
        let mut arr = [0.0f32; 16];
        unsafe { _mm512_storeu_ps(arr.as_mut_ptr(), self.0) };
        arr
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [f32]) {
        assert!(s.len() >= 16);
        unsafe { _mm512_storeu_ps(s.as_mut_ptr(), self.0) };
    }

    // --- Reductions ---

    #[inline(always)]
    pub fn reduce_sum(self) -> f32 {
        unsafe { _mm512_reduce_add_ps(self.0) }
    }

    #[inline(always)]
    pub fn reduce_min(self) -> f32 {
        unsafe { _mm512_reduce_min_ps(self.0) }
    }

    #[inline(always)]
    pub fn reduce_max(self) -> f32 {
        unsafe { _mm512_reduce_max_ps(self.0) }
    }

    // --- Element-wise min/max/clamp ---

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(unsafe { _mm512_min_ps(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(unsafe { _mm512_max_ps(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
        self.simd_max(lo).simd_min(hi)
    }

    // --- Math (StdFloat equivalents) ---

    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        Self(unsafe { _mm512_fmadd_ps(self.0, b.0, c.0) })
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(unsafe { _mm512_sqrt_ps(self.0) })
    }

    /// Round to nearest integer (ties to even).
    #[inline(always)]
    pub fn round(self) -> Self {
        // IMM8: bits[1:0]=0 (nearest), bit[3]=1 (suppress exceptions) = 0x08
        Self(unsafe { _mm512_roundscale_ps::<0x08>(self.0) })
    }

    /// Floor (round toward negative infinity).
    #[inline(always)]
    pub fn floor(self) -> Self {
        // IMM8: bits[1:0]=1 (floor), bit[3]=1 (suppress exceptions) = 0x09
        Self(unsafe { _mm512_roundscale_ps::<0x09>(self.0) })
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        unsafe {
            let mask = _mm512_set1_epi32(0x7FFF_FFFFi32);
            Self(_mm512_castsi512_ps(_mm512_and_si512(
                _mm512_castps_si512(self.0),
                mask,
            )))
        }
    }

    // --- Bit reinterpretation ---

    #[inline(always)]
    pub fn to_bits(self) -> U32x16 {
        U32x16(unsafe { _mm512_castps_si512(self.0) })
    }

    #[inline(always)]
    pub fn from_bits(bits: U32x16) -> Self {
        Self(unsafe { _mm512_castsi512_ps(bits.0) })
    }

    // --- Type casts ---

    /// Truncating cast f32→i32 (equivalent to `portable_simd .cast::<i32>()`).
    #[inline(always)]
    pub fn cast_i32(self) -> I32x16 {
        I32x16(unsafe { _mm512_cvttps_epi32(self.0) })
    }

    // --- Comparisons (return typed masks) ---

    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> F32Mask16 {
        F32Mask16(unsafe { _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> F32Mask16 {
        F32Mask16(unsafe { _mm512_cmp_ps_mask::<_CMP_NEQ_UQ>(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> F32Mask16 {
        F32Mask16(unsafe { _mm512_cmp_ps_mask::<_CMP_LT_OS>(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_le(self, other: Self) -> F32Mask16 {
        F32Mask16(unsafe { _mm512_cmp_ps_mask::<_CMP_LE_OS>(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> F32Mask16 {
        // GT(a, b) = LT(b, a)
        other.simd_lt(self)
    }

    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> F32Mask16 {
        // GE(a, b) = LE(b, a)
        other.simd_le(self)
    }
}

impl_bin_op!(F32x16, Add, add, _mm512_add_ps);
impl_bin_op!(F32x16, Sub, sub, _mm512_sub_ps);
impl_bin_op!(F32x16, Mul, mul, _mm512_mul_ps);
impl_bin_op!(F32x16, Div, div, _mm512_div_ps);
impl_assign_op!(F32x16, AddAssign, add_assign, _mm512_add_ps);
impl_assign_op!(F32x16, SubAssign, sub_assign, _mm512_sub_ps);
impl_assign_op!(F32x16, MulAssign, mul_assign, _mm512_mul_ps);
impl_assign_op!(F32x16, DivAssign, div_assign, _mm512_div_ps);

impl Neg for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe {
            let sign = _mm512_set1_epi32(i32::MIN); // 0x80000000
            Self(_mm512_castsi512_ps(_mm512_xor_si512(
                _mm512_castps_si512(self.0),
                sign,
            )))
        }
    }
}

impl fmt::Debug for F32x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F32x16({:?})", self.to_array())
    }
}

impl PartialEq for F32x16 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// F32Mask16 — 16-bit mask from f32 comparisons
// ============================================================================

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct F32Mask16(pub __mmask16);

impl F32Mask16 {
    /// Select: for each lane, if mask bit is 1 → true_val, else false_val.
    #[inline(always)]
    pub fn select(self, true_val: F32x16, false_val: F32x16) -> F32x16 {
        // _mm512_mask_blend_ps(k, a, b): if k[i] then b[i] else a[i]
        F32x16(unsafe { _mm512_mask_blend_ps(self.0, false_val.0, true_val.0) })
    }
}

// ============================================================================
// F64x8 — 8 × f64 in one AVX-512 register (__m512d)
// ============================================================================

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F64x8(pub __m512d);

impl F64x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn splat(v: f64) -> Self {
        Self(unsafe { _mm512_set1_pd(v) })
    }

    #[inline(always)]
    pub fn from_slice(s: &[f64]) -> Self {
        assert!(s.len() >= 8);
        Self(unsafe { _mm512_loadu_pd(s.as_ptr()) })
    }

    #[inline(always)]
    pub fn from_array(arr: [f64; 8]) -> Self {
        Self(unsafe { _mm512_loadu_pd(arr.as_ptr()) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [f64; 8] {
        let mut arr = [0.0f64; 8];
        unsafe { _mm512_storeu_pd(arr.as_mut_ptr(), self.0) };
        arr
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [f64]) {
        assert!(s.len() >= 8);
        unsafe { _mm512_storeu_pd(s.as_mut_ptr(), self.0) };
    }

    #[inline(always)]
    pub fn reduce_sum(self) -> f64 {
        unsafe { _mm512_reduce_add_pd(self.0) }
    }

    #[inline(always)]
    pub fn reduce_min(self) -> f64 {
        unsafe { _mm512_reduce_min_pd(self.0) }
    }

    #[inline(always)]
    pub fn reduce_max(self) -> f64 {
        unsafe { _mm512_reduce_max_pd(self.0) }
    }

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(unsafe { _mm512_min_pd(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(unsafe { _mm512_max_pd(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
        self.simd_max(lo).simd_min(hi)
    }

    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        Self(unsafe { _mm512_fmadd_pd(self.0, b.0, c.0) })
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        Self(unsafe { _mm512_sqrt_pd(self.0) })
    }

    #[inline(always)]
    pub fn round(self) -> Self {
        Self(unsafe { _mm512_roundscale_pd::<0x08>(self.0) })
    }

    #[inline(always)]
    pub fn floor(self) -> Self {
        Self(unsafe { _mm512_roundscale_pd::<0x09>(self.0) })
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        unsafe {
            let mask = _mm512_set1_epi64(0x7FFF_FFFF_FFFF_FFFFi64);
            Self(_mm512_castsi512_pd(_mm512_and_si512(
                _mm512_castpd_si512(self.0),
                mask,
            )))
        }
    }

    #[inline(always)]
    pub fn to_bits(self) -> U64x8 {
        U64x8(unsafe { _mm512_castpd_si512(self.0) })
    }

    #[inline(always)]
    pub fn from_bits(bits: U64x8) -> Self {
        Self(unsafe { _mm512_castsi512_pd(bits.0) })
    }

    // --- Comparisons ---

    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> F64Mask8 {
        F64Mask8(unsafe { _mm512_cmp_pd_mask::<_CMP_EQ_OQ>(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> F64Mask8 {
        F64Mask8(unsafe { _mm512_cmp_pd_mask::<_CMP_NEQ_UQ>(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> F64Mask8 {
        F64Mask8(unsafe { _mm512_cmp_pd_mask::<_CMP_LT_OS>(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_le(self, other: Self) -> F64Mask8 {
        F64Mask8(unsafe { _mm512_cmp_pd_mask::<_CMP_LE_OS>(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> F64Mask8 {
        other.simd_lt(self)
    }

    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> F64Mask8 {
        other.simd_le(self)
    }
}

impl_bin_op!(F64x8, Add, add, _mm512_add_pd);
impl_bin_op!(F64x8, Sub, sub, _mm512_sub_pd);
impl_bin_op!(F64x8, Mul, mul, _mm512_mul_pd);
impl_bin_op!(F64x8, Div, div, _mm512_div_pd);
impl_assign_op!(F64x8, AddAssign, add_assign, _mm512_add_pd);
impl_assign_op!(F64x8, SubAssign, sub_assign, _mm512_sub_pd);
impl_assign_op!(F64x8, MulAssign, mul_assign, _mm512_mul_pd);
impl_assign_op!(F64x8, DivAssign, div_assign, _mm512_div_pd);

impl Neg for F64x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe {
            let sign = _mm512_set1_epi64(i64::MIN); // 0x8000000000000000
            Self(_mm512_castsi512_pd(_mm512_xor_si512(
                _mm512_castpd_si512(self.0),
                sign,
            )))
        }
    }
}

impl fmt::Debug for F64x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F64x8({:?})", self.to_array())
    }
}

impl PartialEq for F64x8 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// F64Mask8 — 8-bit mask from f64 comparisons
// ============================================================================

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct F64Mask8(pub __mmask8);

impl F64Mask8 {
    #[inline(always)]
    pub fn select(self, true_val: F64x8, false_val: F64x8) -> F64x8 {
        F64x8(unsafe { _mm512_mask_blend_pd(self.0, false_val.0, true_val.0) })
    }
}

// ============================================================================
// U8x64 — 64 × u8 in one AVX-512 register (__m512i)
// ============================================================================

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct U8x64(pub __m512i);

impl U8x64 {
    pub const LANES: usize = 64;

    #[inline(always)]
    pub fn splat(v: u8) -> Self {
        Self(unsafe { _mm512_set1_epi8(v as i8) })
    }

    #[inline(always)]
    pub fn from_slice(s: &[u8]) -> Self {
        assert!(s.len() >= 64);
        Self(unsafe { _mm512_loadu_si512(s.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn from_array(arr: [u8; 64]) -> Self {
        Self(unsafe { _mm512_loadu_si512(arr.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [u8; 64] {
        let mut arr = [0u8; 64];
        unsafe { _mm512_storeu_si512(arr.as_mut_ptr() as *mut _, self.0) };
        arr
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u8]) {
        assert!(s.len() >= 64);
        unsafe { _mm512_storeu_si512(s.as_mut_ptr() as *mut _, self.0) };
    }

    /// Wrapping sum of all 64 bytes → u8 (matches portable_simd semantics).
    #[inline(always)]
    pub fn reduce_sum(self) -> u8 {
        unsafe {
            // SAD against zero sums groups of 8 bytes → 8 × u64
            let sad = _mm512_sad_epu8(self.0, _mm512_setzero_si512());
            _mm512_reduce_add_epi64(sad) as u8
        }
    }

    /// Minimum of all 64 bytes.
    #[inline(always)]
    pub fn reduce_min(self) -> u8 {
        // Tree reduction: 512→256→128→scalar
        let arr = self.to_array();
        let mut m = arr[0];
        for i in 1..64 {
            if arr[i] < m {
                m = arr[i];
            }
        }
        m
    }

    /// Maximum of all 64 bytes.
    #[inline(always)]
    pub fn reduce_max(self) -> u8 {
        let arr = self.to_array();
        let mut m = arr[0];
        for i in 1..64 {
            if arr[i] > m {
                m = arr[i];
            }
        }
        m
    }

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(unsafe { _mm512_min_epu8(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(unsafe { _mm512_max_epu8(self.0, other.0) })
    }
}

// u8 add/sub use AVX-512BW instructions
impl_bin_op!(U8x64, Add, add, _mm512_add_epi8);
impl_bin_op!(U8x64, Sub, sub, _mm512_sub_epi8);
impl_assign_op!(U8x64, AddAssign, add_assign, _mm512_add_epi8);
impl_assign_op!(U8x64, SubAssign, sub_assign, _mm512_sub_epi8);

// u8 multiply — no single instruction; widen to u16, multiply, truncate back.
impl Mul for U8x64 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            // Split into lower/upper 32-byte halves
            let a_lo = _mm512_castsi512_si256(self.0);
            let a_hi = _mm512_extracti64x4_epi64::<1>(self.0);
            let b_lo = _mm512_castsi512_si256(rhs.0);
            let b_hi = _mm512_extracti64x4_epi64::<1>(rhs.0);

            // Zero-extend u8→u16 (256→512 bits, 32 elements each)
            let a16_lo = _mm512_cvtepu8_epi16(a_lo);
            let a16_hi = _mm512_cvtepu8_epi16(a_hi);
            let b16_lo = _mm512_cvtepu8_epi16(b_lo);
            let b16_hi = _mm512_cvtepu8_epi16(b_hi);

            // Multiply as u16 (wrapping at 16-bit)
            let prod_lo = _mm512_mullo_epi16(a16_lo, b16_lo);
            let prod_hi = _mm512_mullo_epi16(a16_hi, b16_hi);

            // Truncate u16→u8 (keep low byte)
            let packed_lo = _mm512_cvtepi16_epi8(prod_lo);
            let packed_hi = _mm512_cvtepi16_epi8(prod_hi);

            Self(_mm512_inserti64x4::<1>(
                _mm512_castsi256_si512(packed_lo),
                packed_hi,
            ))
        }
    }
}

impl MulAssign for U8x64 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

// Bitwise ops for u8
impl_bin_op!(U8x64, BitAnd, bitand, _mm512_and_si512);
impl_bin_op!(U8x64, BitXor, bitxor, _mm512_xor_si512);
impl_bin_op!(U8x64, BitOr, bitor, _mm512_or_si512);
impl_assign_op!(U8x64, BitAndAssign, bitand_assign, _mm512_and_si512);
impl_assign_op!(U8x64, BitXorAssign, bitxor_assign, _mm512_xor_si512);
impl_assign_op!(U8x64, BitOrAssign, bitor_assign, _mm512_or_si512);

impl Not for U8x64 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let all_ones = _mm512_set1_epi8(-1);
            Self(_mm512_xor_si512(self.0, all_ones))
        }
    }
}

impl fmt::Debug for U8x64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "U8x64({:?})", &self.to_array()[..])
    }
}

impl PartialEq for U8x64 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// I32x16 — 16 × i32 in one AVX-512 register (__m512i)
// ============================================================================

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct I32x16(pub __m512i);

impl I32x16 {
    pub const LANES: usize = 16;

    #[inline(always)]
    pub fn splat(v: i32) -> Self {
        Self(unsafe { _mm512_set1_epi32(v) })
    }

    #[inline(always)]
    pub fn from_slice(s: &[i32]) -> Self {
        assert!(s.len() >= 16);
        Self(unsafe { _mm512_loadu_si512(s.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn from_array(arr: [i32; 16]) -> Self {
        Self(unsafe { _mm512_loadu_si512(arr.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i32; 16] {
        let mut arr = [0i32; 16];
        unsafe { _mm512_storeu_si512(arr.as_mut_ptr() as *mut _, self.0) };
        arr
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i32]) {
        assert!(s.len() >= 16);
        unsafe { _mm512_storeu_si512(s.as_mut_ptr() as *mut _, self.0) };
    }

    #[inline(always)]
    pub fn reduce_sum(self) -> i32 {
        unsafe { _mm512_reduce_add_epi32(self.0) }
    }

    #[inline(always)]
    pub fn reduce_min(self) -> i32 {
        unsafe { _mm512_reduce_min_epi32(self.0) }
    }

    #[inline(always)]
    pub fn reduce_max(self) -> i32 {
        unsafe { _mm512_reduce_max_epi32(self.0) }
    }

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(unsafe { _mm512_min_epi32(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(unsafe { _mm512_max_epi32(self.0, other.0) })
    }

    /// Cast i32→f32 (equivalent to `portable_simd .cast::<f32>()`).
    #[inline(always)]
    pub fn cast_f32(self) -> F32x16 {
        F32x16(unsafe { _mm512_cvtepi32_ps(self.0) })
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { _mm512_abs_epi32(self.0) })
    }
}

impl_bin_op!(I32x16, Add, add, _mm512_add_epi32);
impl_bin_op!(I32x16, Sub, sub, _mm512_sub_epi32);
impl_assign_op!(I32x16, AddAssign, add_assign, _mm512_add_epi32);
impl_assign_op!(I32x16, SubAssign, sub_assign, _mm512_sub_epi32);

// i32 multiply: _mm512_mullo_epi32 (AVX-512F)
impl_bin_op!(I32x16, Mul, mul, _mm512_mullo_epi32);
impl_assign_op!(I32x16, MulAssign, mul_assign, _mm512_mullo_epi32);

// i32 divide: no SIMD instruction — array fallback
impl Div for I32x16 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let a = self.to_array();
        let b = rhs.to_array();
        let mut c = [0i32; 16];
        for i in 0..16 {
            c[i] = a[i] / b[i];
        }
        Self::from_array(c)
    }
}

impl DivAssign for I32x16 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

// Bitwise
impl_bin_op!(I32x16, BitAnd, bitand, _mm512_and_si512);
impl_bin_op!(I32x16, BitXor, bitxor, _mm512_xor_si512);
impl_bin_op!(I32x16, BitOr, bitor, _mm512_or_si512);
impl_assign_op!(I32x16, BitAndAssign, bitand_assign, _mm512_and_si512);
impl_assign_op!(I32x16, BitXorAssign, bitxor_assign, _mm512_xor_si512);
impl_assign_op!(I32x16, BitOrAssign, bitor_assign, _mm512_or_si512);

impl Not for I32x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let all_ones = _mm512_set1_epi32(-1);
            Self(_mm512_xor_si512(self.0, all_ones))
        }
    }
}

impl Neg for I32x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe { Self(_mm512_sub_epi32(_mm512_setzero_si512(), self.0)) }
    }
}

impl fmt::Debug for I32x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "I32x16({:?})", self.to_array())
    }
}

impl PartialEq for I32x16 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// I64x8 — 8 × i64 in one AVX-512 register (__m512i)
// ============================================================================

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct I64x8(pub __m512i);

impl I64x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn splat(v: i64) -> Self {
        Self(unsafe { _mm512_set1_epi64(v) })
    }

    #[inline(always)]
    pub fn from_slice(s: &[i64]) -> Self {
        assert!(s.len() >= 8);
        Self(unsafe { _mm512_loadu_si512(s.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn from_array(arr: [i64; 8]) -> Self {
        Self(unsafe { _mm512_loadu_si512(arr.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i64; 8] {
        let mut arr = [0i64; 8];
        unsafe { _mm512_storeu_si512(arr.as_mut_ptr() as *mut _, self.0) };
        arr
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [i64]) {
        assert!(s.len() >= 8);
        unsafe { _mm512_storeu_si512(s.as_mut_ptr() as *mut _, self.0) };
    }

    #[inline(always)]
    pub fn reduce_sum(self) -> i64 {
        unsafe { _mm512_reduce_add_epi64(self.0) }
    }

    #[inline(always)]
    pub fn reduce_min(self) -> i64 {
        unsafe { _mm512_reduce_min_epi64(self.0) }
    }

    #[inline(always)]
    pub fn reduce_max(self) -> i64 {
        unsafe { _mm512_reduce_max_epi64(self.0) }
    }

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        Self(unsafe { _mm512_min_epi64(self.0, other.0) })
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        Self(unsafe { _mm512_max_epi64(self.0, other.0) })
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(unsafe { _mm512_abs_epi64(self.0) })
    }
}

impl_bin_op!(I64x8, Add, add, _mm512_add_epi64);
impl_bin_op!(I64x8, Sub, sub, _mm512_sub_epi64);
impl_assign_op!(I64x8, AddAssign, add_assign, _mm512_add_epi64);
impl_assign_op!(I64x8, SubAssign, sub_assign, _mm512_sub_epi64);

// i64 multiply: _mm512_mullo_epi64 (AVX-512DQ — available on all server CPUs)
impl Mul for I64x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // Fallback: array-based multiply (AVX-512DQ _mm512_mullo_epi64 may
        // not be available on all targets)
        let a = self.to_array();
        let b = rhs.to_array();
        let mut c = [0i64; 8];
        for i in 0..8 {
            c[i] = a[i].wrapping_mul(b[i]);
        }
        Self::from_array(c)
    }
}

impl MulAssign for I64x8 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

// i64 divide: no SIMD instruction — array fallback
impl Div for I64x8 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let a = self.to_array();
        let b = rhs.to_array();
        let mut c = [0i64; 8];
        for i in 0..8 {
            c[i] = a[i] / b[i];
        }
        Self::from_array(c)
    }
}

impl DivAssign for I64x8 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

// Bitwise
impl_bin_op!(I64x8, BitAnd, bitand, _mm512_and_si512);
impl_bin_op!(I64x8, BitXor, bitxor, _mm512_xor_si512);
impl_bin_op!(I64x8, BitOr, bitor, _mm512_or_si512);
impl_assign_op!(I64x8, BitAndAssign, bitand_assign, _mm512_and_si512);
impl_assign_op!(I64x8, BitXorAssign, bitxor_assign, _mm512_xor_si512);
impl_assign_op!(I64x8, BitOrAssign, bitor_assign, _mm512_or_si512);

impl Not for I64x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let all_ones = _mm512_set1_epi64(-1);
            Self(_mm512_xor_si512(self.0, all_ones))
        }
    }
}

impl Neg for I64x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe { Self(_mm512_sub_epi64(_mm512_setzero_si512(), self.0)) }
    }
}

impl fmt::Debug for I64x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "I64x8({:?})", self.to_array())
    }
}

impl PartialEq for I64x8 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// U32x16 — 16 × u32 in one AVX-512 register (__m512i)
// Used primarily for bit manipulation in transcendental functions (vml.rs).
// ============================================================================

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct U32x16(pub __m512i);

impl U32x16 {
    pub const LANES: usize = 16;

    #[inline(always)]
    pub fn splat(v: u32) -> Self {
        Self(unsafe { _mm512_set1_epi32(v as i32) })
    }

    #[inline(always)]
    pub fn from_slice(s: &[u32]) -> Self {
        assert!(s.len() >= 16);
        Self(unsafe { _mm512_loadu_si512(s.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn from_array(arr: [u32; 16]) -> Self {
        Self(unsafe { _mm512_loadu_si512(arr.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 16] {
        let mut arr = [0u32; 16];
        unsafe { _mm512_storeu_si512(arr.as_mut_ptr() as *mut _, self.0) };
        arr
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u32]) {
        assert!(s.len() >= 16);
        unsafe { _mm512_storeu_si512(s.as_mut_ptr() as *mut _, self.0) };
    }

    #[inline(always)]
    pub fn reduce_sum(self) -> u32 {
        unsafe { _mm512_reduce_add_epi32(self.0) as u32 }
    }
}

impl_bin_op!(U32x16, Add, add, _mm512_add_epi32);
impl_bin_op!(U32x16, Sub, sub, _mm512_sub_epi32);
impl_bin_op!(U32x16, Mul, mul, _mm512_mullo_epi32);
impl_assign_op!(U32x16, AddAssign, add_assign, _mm512_add_epi32);

// Bitwise
impl_bin_op!(U32x16, BitAnd, bitand, _mm512_and_si512);
impl_bin_op!(U32x16, BitXor, bitxor, _mm512_xor_si512);
impl_bin_op!(U32x16, BitOr, bitor, _mm512_or_si512);

impl Not for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let all_ones = _mm512_set1_epi32(-1);
            Self(_mm512_xor_si512(self.0, all_ones))
        }
    }
}

// Shift operators for U32x16 (per-element variable shift)
impl Shr<Self> for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_srlv_epi32(self.0, rhs.0) })
    }
}

impl Shl<Self> for U32x16 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_sllv_epi32(self.0, rhs.0) })
    }
}

impl fmt::Debug for U32x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "U32x16({:?})", self.to_array())
    }
}

impl PartialEq for U32x16 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// U64x8 — 8 × u64 in one AVX-512 register (__m512i)
// Used primarily for bit manipulation in transcendental functions and HDC.
// ============================================================================

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct U64x8(pub __m512i);

impl U64x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn splat(v: u64) -> Self {
        Self(unsafe { _mm512_set1_epi64(v as i64) })
    }

    #[inline(always)]
    pub fn from_slice(s: &[u64]) -> Self {
        assert!(s.len() >= 8);
        Self(unsafe { _mm512_loadu_si512(s.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn from_array(arr: [u64; 8]) -> Self {
        Self(unsafe { _mm512_loadu_si512(arr.as_ptr() as *const _) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [u64; 8] {
        let mut arr = [0u64; 8];
        unsafe { _mm512_storeu_si512(arr.as_mut_ptr() as *mut _, self.0) };
        arr
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [u64]) {
        assert!(s.len() >= 8);
        unsafe { _mm512_storeu_si512(s.as_mut_ptr() as *mut _, self.0) };
    }

    #[inline(always)]
    pub fn reduce_sum(self) -> u64 {
        unsafe { _mm512_reduce_add_epi64(self.0) as u64 }
    }
}

impl_bin_op!(U64x8, Add, add, _mm512_add_epi64);
impl_bin_op!(U64x8, Sub, sub, _mm512_sub_epi64);
impl_assign_op!(U64x8, AddAssign, add_assign, _mm512_add_epi64);

// Bitwise
impl_bin_op!(U64x8, BitAnd, bitand, _mm512_and_si512);
impl_bin_op!(U64x8, BitXor, bitxor, _mm512_xor_si512);
impl_bin_op!(U64x8, BitOr, bitor, _mm512_or_si512);
impl_assign_op!(U64x8, BitAndAssign, bitand_assign, _mm512_and_si512);
impl_assign_op!(U64x8, BitXorAssign, bitxor_assign, _mm512_xor_si512);
impl_assign_op!(U64x8, BitOrAssign, bitor_assign, _mm512_or_si512);

impl Not for U64x8 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        unsafe {
            let all_ones = _mm512_set1_epi64(-1);
            Self(_mm512_xor_si512(self.0, all_ones))
        }
    }
}

// Shift operators for U64x8 (per-element variable shift)
impl Shr<Self> for U64x8 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_srlv_epi64(self.0, rhs.0) })
    }
}

impl Shl<Self> for U64x8 {
    type Output = Self;
    #[inline(always)]
    fn shl(self, rhs: Self) -> Self {
        Self(unsafe { _mm512_sllv_epi64(self.0, rhs.0) })
    }
}

impl fmt::Debug for U64x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "U64x8({:?})", self.to_array())
    }
}

impl PartialEq for U64x8 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// ============================================================================
// Type aliases — lowercase names matching portable_simd convention
// ============================================================================

#[allow(non_camel_case_types)]
pub type f32x16 = F32x16;
#[allow(non_camel_case_types)]
pub type f64x8 = F64x8;
#[allow(non_camel_case_types)]
pub type u8x64 = U8x64;
#[allow(non_camel_case_types)]
pub type i32x16 = I32x16;
#[allow(non_camel_case_types)]
pub type i64x8 = I64x8;
#[allow(non_camel_case_types)]
pub type u32x16 = U32x16;
#[allow(non_camel_case_types)]
pub type u64x8 = U64x8;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32x16_basic() {
        let a = F32x16::splat(1.0);
        let b = F32x16::splat(2.0);
        let c = a + b;
        assert!((c.reduce_sum() - 48.0).abs() < 1e-6); // 16 × 3.0
    }

    #[test]
    fn f32x16_from_slice() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let v = F32x16::from_slice(&data);
        let arr = v.to_array();
        assert_eq!(arr[0], 0.0);
        assert_eq!(arr[15], 15.0);
    }

    #[test]
    fn f32x16_reduce() {
        let data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        let v = F32x16::from_slice(&data);
        assert!((v.reduce_sum() - 136.0).abs() < 1e-4); // sum(1..=16) = 136
        assert!((v.reduce_min() - 1.0).abs() < 1e-6);
        assert!((v.reduce_max() - 16.0).abs() < 1e-6);
    }

    #[test]
    fn f32x16_math() {
        let v = F32x16::splat(4.0);
        assert!((v.sqrt().reduce_sum() - 32.0).abs() < 1e-4); // 16 × 2.0
        assert!((v.abs().reduce_sum() - 64.0).abs() < 1e-4);
        let neg = F32x16::splat(-3.5);
        assert!((neg.abs().reduce_sum() - 56.0).abs() < 1e-4); // 16 × 3.5
        assert!((neg.round().reduce_sum() + 64.0).abs() < 1e-4); // 16 × -4.0
        assert!((neg.floor().reduce_sum() + 64.0).abs() < 1e-4); // 16 × -4.0
    }

    #[test]
    fn f32x16_fma() {
        let a = F32x16::splat(2.0);
        let b = F32x16::splat(3.0);
        let c = F32x16::splat(1.0);
        // fma: a*b + c = 2*3+1 = 7
        let result = a.mul_add(b, c);
        assert!((result.reduce_sum() - 112.0).abs() < 1e-4); // 16 × 7
    }

    #[test]
    fn f32x16_comparison_select() {
        let a = F32x16::from_array([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let threshold = F32x16::splat(8.5);
        let mask = a.simd_lt(threshold);
        let result = mask.select(F32x16::splat(1.0), F32x16::splat(0.0));
        assert!((result.reduce_sum() - 8.0).abs() < 1e-6); // 8 values < 8.5
    }

    #[test]
    fn f64x8_basic() {
        let a = F64x8::splat(1.0);
        let b = F64x8::splat(2.0);
        let c = a + b;
        assert!((c.reduce_sum() - 24.0).abs() < 1e-10); // 8 × 3.0
    }

    #[test]
    fn u8x64_bitwise() {
        let a = U8x64::splat(0xF0);
        let b = U8x64::splat(0x0F);
        assert_eq!((a & b).to_array()[0], 0x00);
        assert_eq!((a | b).to_array()[0], 0xFF);
        assert_eq!((a ^ b).to_array()[0], 0xFF);
        assert_eq!((!a).to_array()[0], 0x0F);
    }

    #[test]
    fn i32x16_basic() {
        let a = I32x16::splat(10);
        let b = I32x16::splat(3);
        assert_eq!((a + b).reduce_sum(), 16 * 13);
        assert_eq!((a * b).reduce_sum(), 16 * 30);
    }

    #[test]
    fn i64x8_basic() {
        let a = I64x8::splat(100);
        let b = I64x8::splat(50);
        assert_eq!((a + b).reduce_sum(), 8 * 150);
        assert_eq!((a - b).reduce_sum(), 8 * 50);
    }

    #[test]
    fn u32x16_from_bits_roundtrip() {
        let f = F32x16::splat(1.0);
        let bits = f.to_bits();
        let f2 = F32x16::from_bits(bits);
        assert_eq!(f, f2);
    }

    #[test]
    fn u64x8_from_array() {
        let arr = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let v = U64x8::from_array(arr);
        assert_eq!(v.to_array(), arr);
    }

    #[test]
    fn cast_f32_i32_roundtrip() {
        let f = F32x16::splat(42.7);
        let i = f.cast_i32(); // truncating: 42.7 → 42
        assert_eq!(i.reduce_sum(), 16 * 42);
        let back = i.cast_f32();
        assert!((back.reduce_sum() - 16.0 * 42.0).abs() < 1e-4);
    }
}
