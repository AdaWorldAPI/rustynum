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
//! use rustynum_core::simd_avx512::f32x16;
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

impl Default for F32x16 {
    #[inline(always)]
    fn default() -> Self {
        Self(unsafe { _mm512_setzero_ps() })
    }
}

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

impl Default for F64x8 {
    #[inline(always)]
    fn default() -> Self {
        Self(unsafe { _mm512_setzero_pd() })
    }
}

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
        for &val in arr.iter().skip(1) {
            if val < m {
                m = val;
            }
        }
        m
    }

    /// Maximum of all 64 bytes.
    #[inline(always)]
    pub fn reduce_max(self) -> u8 {
        let arr = self.to_array();
        let mut m = arr[0];
        for &val in arr.iter().skip(1) {
            if val > m {
                m = val;
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
// AVX2 wrapper types — 256-bit (F32x8, F64x4)
// ============================================================================
// Same pattern as AVX-512 wrappers above. Used by simd_avx2.rs when
// compiling with --features avx2 --no-default-features.
// All intrinsics are stable std::arch::x86_64 (avx/avx2).

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F32x8(pub __m256);

impl F32x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        Self(unsafe { _mm256_set1_ps(v) })
    }

    #[inline(always)]
    pub fn from_slice(s: &[f32]) -> Self {
        assert!(s.len() >= 8);
        Self(unsafe { _mm256_loadu_ps(s.as_ptr()) })
    }

    #[inline(always)]
    pub fn from_array(a: [f32; 8]) -> Self {
        Self(unsafe { _mm256_loadu_ps(a.as_ptr()) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 8] {
        let mut out = [0.0f32; 8];
        unsafe { _mm256_storeu_ps(out.as_mut_ptr(), self.0) };
        out
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [f32]) {
        assert!(s.len() >= 8);
        unsafe { _mm256_storeu_ps(s.as_mut_ptr(), self.0) };
    }

    #[inline(always)]
    pub fn reduce_sum(self) -> f32 {
        unsafe {
            // Extract upper 128 and add to lower 128
            let hi = _mm256_extractf128_ps(self.0, 1);
            let lo = _mm256_castps256_ps128(self.0);
            let sum128 = _mm_add_ps(lo, hi);
            // Horizontal reduce 4 floats
            let hi64 = _mm_movehl_ps(sum128, sum128);
            let sum64 = _mm_add_ps(sum128, hi64);
            let hi32 = _mm_shuffle_ps(sum64, sum64, 0x55);
            let sum32 = _mm_add_ss(sum64, hi32);
            _mm_cvtss_f32(sum32)
        }
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        // Clear sign bit: AND with 0x7FFFFFFF
        unsafe {
            let mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFFi32));
            Self(_mm256_and_ps(self.0, mask))
        }
    }
}

impl Add for F32x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_add_ps(self.0, rhs.0) })
    }
}

impl AddAssign for F32x8 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = unsafe { _mm256_add_ps(self.0, rhs.0) };
    }
}

impl Mul for F32x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_mul_ps(self.0, rhs.0) })
    }
}

impl MulAssign for F32x8 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = unsafe { _mm256_mul_ps(self.0, rhs.0) };
    }
}

impl fmt::Debug for F32x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F32x8({:?})", self.to_array())
    }
}

impl PartialEq for F32x8 {
    fn eq(&self, other: &Self) -> bool {
        self.to_array() == other.to_array()
    }
}

// --- F64x4 (AVX2: 4 × f64) ---

#[derive(Copy, Clone)]
#[repr(transparent)]
pub struct F64x4(pub __m256d);

impl F64x4 {
    pub const LANES: usize = 4;

    #[inline(always)]
    pub fn splat(v: f64) -> Self {
        Self(unsafe { _mm256_set1_pd(v) })
    }

    #[inline(always)]
    pub fn from_slice(s: &[f64]) -> Self {
        assert!(s.len() >= 4);
        Self(unsafe { _mm256_loadu_pd(s.as_ptr()) })
    }

    #[inline(always)]
    pub fn from_array(a: [f64; 4]) -> Self {
        Self(unsafe { _mm256_loadu_pd(a.as_ptr()) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [f64; 4] {
        let mut out = [0.0f64; 4];
        unsafe { _mm256_storeu_pd(out.as_mut_ptr(), self.0) };
        out
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [f64]) {
        assert!(s.len() >= 4);
        unsafe { _mm256_storeu_pd(s.as_mut_ptr(), self.0) };
    }

    #[inline(always)]
    pub fn reduce_sum(self) -> f64 {
        unsafe {
            let hi = _mm256_extractf128_pd(self.0, 1);
            let lo = _mm256_castpd256_pd128(self.0);
            let sum128 = _mm_add_pd(lo, hi);
            let hi64 = _mm_unpackhi_pd(sum128, sum128);
            let sum64 = _mm_add_sd(sum128, hi64);
            _mm_cvtsd_f64(sum64)
        }
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        unsafe {
            let mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFFi64));
            Self(_mm256_and_pd(self.0, mask))
        }
    }
}

impl Add for F64x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_add_pd(self.0, rhs.0) })
    }
}

impl AddAssign for F64x4 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        self.0 = unsafe { _mm256_add_pd(self.0, rhs.0) };
    }
}

impl Mul for F64x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { _mm256_mul_pd(self.0, rhs.0) })
    }
}

impl MulAssign for F64x4 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 = unsafe { _mm256_mul_pd(self.0, rhs.0) };
    }
}

impl fmt::Debug for F64x4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F64x4({:?})", self.to_array())
    }
}

impl PartialEq for F64x4 {
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

// AVX2 aliases (256-bit)
#[allow(non_camel_case_types)]
pub type f32x8 = F32x8;
#[allow(non_camel_case_types)]
pub type f64x4 = F64x4;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Skip test at runtime if AVX-512 is not available on this CPU.
    macro_rules! require_avx512 {
        () => {
            if !is_x86_feature_detected!("avx512f") {
                eprintln!("skipping: AVX-512 not available on this CPU");
                return;
            }
        };
    }

    #[test]
    fn f32x16_basic() {
        require_avx512!();
        let a = F32x16::splat(1.0);
        let b = F32x16::splat(2.0);
        let c = a + b;
        assert!((c.reduce_sum() - 48.0).abs() < 1e-6); // 16 × 3.0
    }

    #[test]
    fn f32x16_from_slice() {
        require_avx512!();
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let v = F32x16::from_slice(&data);
        let arr = v.to_array();
        assert_eq!(arr[0], 0.0);
        assert_eq!(arr[15], 15.0);
    }

    #[test]
    fn f32x16_reduce() {
        require_avx512!();
        let data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
        let v = F32x16::from_slice(&data);
        assert!((v.reduce_sum() - 136.0).abs() < 1e-4); // sum(1..=16) = 136
        assert!((v.reduce_min() - 1.0).abs() < 1e-6);
        assert!((v.reduce_max() - 16.0).abs() < 1e-6);
    }

    #[test]
    fn f32x16_math() {
        require_avx512!();
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
        require_avx512!();
        let a = F32x16::splat(2.0);
        let b = F32x16::splat(3.0);
        let c = F32x16::splat(1.0);
        // fma: a*b + c = 2*3+1 = 7
        let result = a.mul_add(b, c);
        assert!((result.reduce_sum() - 112.0).abs() < 1e-4); // 16 × 7
    }

    #[test]
    fn f32x16_comparison_select() {
        require_avx512!();
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
        require_avx512!();
        let a = F64x8::splat(1.0);
        let b = F64x8::splat(2.0);
        let c = a + b;
        assert!((c.reduce_sum() - 24.0).abs() < 1e-10); // 8 × 3.0
    }

    #[test]
    fn u8x64_bitwise() {
        require_avx512!();
        let a = U8x64::splat(0xF0);
        let b = U8x64::splat(0x0F);
        assert_eq!((a & b).to_array()[0], 0x00);
        assert_eq!((a | b).to_array()[0], 0xFF);
        assert_eq!((a ^ b).to_array()[0], 0xFF);
        assert_eq!((!a).to_array()[0], 0x0F);
    }

    #[test]
    fn i32x16_basic() {
        require_avx512!();
        let a = I32x16::splat(10);
        let b = I32x16::splat(3);
        assert_eq!((a + b).reduce_sum(), 16 * 13);
        assert_eq!((a * b).reduce_sum(), 16 * 30);
    }

    #[test]
    fn i64x8_basic() {
        require_avx512!();
        let a = I64x8::splat(100);
        let b = I64x8::splat(50);
        assert_eq!((a + b).reduce_sum(), 8 * 150);
        assert_eq!((a - b).reduce_sum(), 8 * 50);
    }

    #[test]
    fn u32x16_from_bits_roundtrip() {
        require_avx512!();
        let f = F32x16::splat(1.0);
        let bits = f.to_bits();
        let f2 = F32x16::from_bits(bits);
        assert_eq!(f, f2);
    }

    #[test]
    fn u64x8_from_array() {
        require_avx512!();
        let arr = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let v = U64x8::from_array(arr);
        assert_eq!(v.to_array(), arr);
    }

    #[test]
    fn cast_f32_i32_roundtrip() {
        require_avx512!();
        let f = F32x16::splat(42.7);
        let i = f.cast_i32(); // truncating: 42.7 → 42
        assert_eq!(i.reduce_sum(), 16 * 42);
        let back = i.cast_f32();
        assert!((back.reduce_sum() - 16.0 * 42.0).abs() < 1e-4);
    }
}

// ============================================================================
// Module-level AVX-512 kernel functions (for dispatch! macro)
// ============================================================================
//
// Each function uses the wrapper types above (F32x16, F64x8, etc.)
// and is guarded by #[target_feature(enable = "avx512f")].

const F32_LANES: usize = 16;
const F64_LANES: usize = 8;

// ─── BLAS-1 ────────────────────────────────────────────────────────

/// # Safety
/// Caller must ensure AVX-512F is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / F32_LANES;
    let mut acc0 = f32x16::splat(0.0);
    let mut acc1 = f32x16::splat(0.0);
    let mut acc2 = f32x16::splat(0.0);
    let mut acc3 = f32x16::splat(0.0);
    let full_iters = chunks / 4;
    for i in 0..full_iters {
        let base = i * 4 * F32_LANES;
        acc0 += f32x16::from_slice(&a[base..]) * f32x16::from_slice(&b[base..]);
        acc1 += f32x16::from_slice(&a[base + F32_LANES..]) * f32x16::from_slice(&b[base + F32_LANES..]);
        acc2 += f32x16::from_slice(&a[base + 2 * F32_LANES..]) * f32x16::from_slice(&b[base + 2 * F32_LANES..]);
        acc3 += f32x16::from_slice(&a[base + 3 * F32_LANES..]) * f32x16::from_slice(&b[base + 3 * F32_LANES..]);
    }
    for i in (full_iters * 4)..chunks {
        let base = i * F32_LANES;
        acc0 += f32x16::from_slice(&a[base..]) * f32x16::from_slice(&b[base..]);
    }
    let mut sum = (acc0 + acc1 + acc2 + acc3).reduce_sum();
    for i in (chunks * F32_LANES)..len {
        sum += a[i] * b[i];
    }
    sum
}

/// # Safety
/// Caller must ensure AVX-512F is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    let chunks = len / F64_LANES;
    let mut acc0 = f64x8::splat(0.0);
    let mut acc1 = f64x8::splat(0.0);
    let mut acc2 = f64x8::splat(0.0);
    let mut acc3 = f64x8::splat(0.0);
    let full_iters = chunks / 4;
    for i in 0..full_iters {
        let base = i * 4 * F64_LANES;
        acc0 += f64x8::from_slice(&a[base..]) * f64x8::from_slice(&b[base..]);
        acc1 += f64x8::from_slice(&a[base + F64_LANES..]) * f64x8::from_slice(&b[base + F64_LANES..]);
        acc2 += f64x8::from_slice(&a[base + 2 * F64_LANES..]) * f64x8::from_slice(&b[base + 2 * F64_LANES..]);
        acc3 += f64x8::from_slice(&a[base + 3 * F64_LANES..]) * f64x8::from_slice(&b[base + 3 * F64_LANES..]);
    }
    for i in (full_iters * 4)..chunks {
        let base = i * F64_LANES;
        acc0 += f64x8::from_slice(&a[base..]) * f64x8::from_slice(&b[base..]);
    }
    let mut sum = (acc0 + acc1 + acc2 + acc3).reduce_sum();
    for i in (chunks * F64_LANES)..len {
        sum += a[i] * b[i];
    }
    sum
}

/// # Safety
/// Caller must ensure AVX-512F is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) {
    let len = x.len();
    let chunks = len / F32_LANES;
    let alpha_v = f32x16::splat(alpha);
    for i in 0..chunks {
        let base = i * F32_LANES;
        let mut yv = f32x16::from_slice(&y[base..]);
        yv += alpha_v * f32x16::from_slice(&x[base..]);
        yv.copy_to_slice(&mut y[base..base + F32_LANES]);
    }
    for i in (chunks * F32_LANES)..len {
        y[i] += alpha * x[i];
    }
}

/// # Safety
/// Caller must ensure AVX-512F is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn axpy_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    let len = x.len();
    let chunks = len / F64_LANES;
    let alpha_v = f64x8::splat(alpha);
    for i in 0..chunks {
        let base = i * F64_LANES;
        let mut yv = f64x8::from_slice(&y[base..]);
        yv += alpha_v * f64x8::from_slice(&x[base..]);
        yv.copy_to_slice(&mut y[base..base + F64_LANES]);
    }
    for i in (chunks * F64_LANES)..len {
        y[i] += alpha * x[i];
    }
}

/// # Safety
/// Caller must ensure AVX-512F is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn scal_f32(alpha: f32, x: &mut [f32]) {
    let len = x.len();
    let chunks = len / F32_LANES;
    let alpha_v = f32x16::splat(alpha);
    for i in 0..chunks {
        let base = i * F32_LANES;
        let result = alpha_v * f32x16::from_slice(&x[base..]);
        result.copy_to_slice(&mut x[base..base + F32_LANES]);
    }
    for xi in &mut x[chunks * F32_LANES..] {
        *xi *= alpha;
    }
}

/// # Safety
/// Caller must ensure AVX-512F is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn scal_f64(alpha: f64, x: &mut [f64]) {
    let len = x.len();
    let chunks = len / F64_LANES;
    let alpha_v = f64x8::splat(alpha);
    for i in 0..chunks {
        let base = i * F64_LANES;
        let result = alpha_v * f64x8::from_slice(&x[base..]);
        result.copy_to_slice(&mut x[base..base + F64_LANES]);
    }
    for xi in &mut x[chunks * F64_LANES..] {
        *xi *= alpha;
    }
}

/// # Safety
/// Caller must ensure AVX-512F is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn asum_f32(x: &[f32]) -> f32 {
    let len = x.len();
    let chunks = len / F32_LANES;
    let mut acc = f32x16::splat(0.0);
    for i in 0..chunks {
        let base = i * F32_LANES;
        acc += f32x16::from_slice(&x[base..]).abs();
    }
    let mut sum = acc.reduce_sum();
    for &xi in &x[chunks * F32_LANES..] {
        sum += xi.abs();
    }
    sum
}

/// # Safety
/// Caller must ensure AVX-512F is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn asum_f64(x: &[f64]) -> f64 {
    let len = x.len();
    let chunks = len / F64_LANES;
    let mut acc = f64x8::splat(0.0);
    for i in 0..chunks {
        let base = i * F64_LANES;
        acc += f64x8::from_slice(&x[base..]).abs();
    }
    let mut sum = acc.reduce_sum();
    for &xi in &x[chunks * F64_LANES..] {
        sum += xi.abs();
    }
    sum
}

/// # Safety
/// Caller must ensure AVX-512F is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn nrm2_f32(x: &[f32]) -> f32 {
    let len = x.len();
    let chunks = len / F32_LANES;
    let mut acc = f32x16::splat(0.0);
    for i in 0..chunks {
        let base = i * F32_LANES;
        let xv = f32x16::from_slice(&x[base..]);
        acc += xv * xv;
    }
    let mut sum = acc.reduce_sum();
    for &xi in &x[chunks * F32_LANES..] {
        sum += xi * xi;
    }
    sum.sqrt()
}

/// # Safety
/// Caller must ensure AVX-512F is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn nrm2_f64(x: &[f64]) -> f64 {
    let len = x.len();
    let chunks = len / F64_LANES;
    let mut acc = f64x8::splat(0.0);
    for i in 0..chunks {
        let base = i * F64_LANES;
        let xv = f64x8::from_slice(&x[base..]);
        acc += xv * xv;
    }
    let mut sum = acc.reduce_sum();
    for &xi in &x[chunks * F64_LANES..] {
        sum += xi * xi;
    }
    sum.sqrt()
}

/// # Safety
/// Caller must ensure AVX-512F is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn iamax_f32(x: &[f32]) -> (usize, f32) {
    if x.is_empty() { return (0, 0.0); }
    let len = x.len();
    let chunks = len / F32_LANES;
    let mut global_max = 0.0f32;
    let mut global_idx = 0usize;
    for c in 0..chunks {
        let base = c * F32_LANES;
        let v = f32x16::from_slice(&x[base..]);
        let abs_v = v.abs();
        let chunk_max = abs_v.reduce_max();
        if chunk_max > global_max {
            let arr = abs_v.to_array();
            for (lane, &val) in arr.iter().enumerate() {
                if val >= chunk_max - f32::EPSILON {
                    global_max = val;
                    global_idx = base + lane;
                    break;
                }
            }
        }
    }
    let tail_start = chunks * F32_LANES;
    for (j, &xi) in x[tail_start..].iter().enumerate() {
        let v = xi.abs();
        if v > global_max {
            global_max = v;
            global_idx = tail_start + j;
        }
    }
    (global_idx, global_max)
}

/// # Safety
/// Caller must ensure AVX-512F is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn iamax_f64(x: &[f64]) -> (usize, f64) {
    if x.is_empty() { return (0, 0.0); }
    let len = x.len();
    let chunks = len / F64_LANES;
    let mut global_max = 0.0f64;
    let mut global_idx = 0usize;
    for c in 0..chunks {
        let base = c * F64_LANES;
        let v = f64x8::from_slice(&x[base..]);
        let abs_v = v.abs();
        let chunk_max = abs_v.reduce_max();
        if chunk_max > global_max {
            let arr = abs_v.to_array();
            for (lane, &val) in arr.iter().enumerate() {
                if val >= chunk_max - f64::EPSILON {
                    global_max = val;
                    global_idx = base + lane;
                    break;
                }
            }
        }
    }
    let tail_start = chunks * F64_LANES;
    for (j, &xi) in x[tail_start..].iter().enumerate() {
        let v = xi.abs();
        if v > global_max {
            global_max = v;
            global_idx = tail_start + j;
        }
    }
    (global_idx, global_max)
}

// ─── Element-wise f32 ──────────────────────────────────────────────

macro_rules! elementwise_scalar_avx512_f32 {
    ($name:ident, $op:tt) => {
        /// # Safety
        /// Caller must ensure AVX-512F is available at runtime.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f")]
        pub unsafe fn $name(a: &[f32], scalar: f32) -> Vec<f32> {
            let len = a.len();
            let mut out = vec![0.0f32; len];
            let chunks = len / F32_LANES;
            let sv = f32x16::splat(scalar);
            for i in 0..chunks {
                let base = i * F32_LANES;
                let av = f32x16::from_slice(&a[base..]);
                (av $op sv).copy_to_slice(&mut out[base..base + F32_LANES]);
            }
            for i in (chunks * F32_LANES)..len {
                out[i] = a[i] $op scalar;
            }
            out
        }
    };
}

elementwise_scalar_avx512_f32!(add_f32_scalar, +);
elementwise_scalar_avx512_f32!(sub_f32_scalar, -);
elementwise_scalar_avx512_f32!(mul_f32_scalar, *);
elementwise_scalar_avx512_f32!(div_f32_scalar, /);

macro_rules! elementwise_vec_avx512_f32 {
    ($name:ident, $op:tt) => {
        /// # Safety
        /// Caller must ensure AVX-512F is available at runtime.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f")]
        pub unsafe fn $name(a: &[f32], b: &[f32]) -> Vec<f32> {
            let len = a.len();
            let mut out = vec![0.0f32; len];
            let chunks = len / F32_LANES;
            for i in 0..chunks {
                let base = i * F32_LANES;
                let av = f32x16::from_slice(&a[base..]);
                let bv = f32x16::from_slice(&b[base..]);
                (av $op bv).copy_to_slice(&mut out[base..base + F32_LANES]);
            }
            for i in (chunks * F32_LANES)..len {
                out[i] = a[i] $op b[i];
            }
            out
        }
    };
}

elementwise_vec_avx512_f32!(add_f32_vec, +);
elementwise_vec_avx512_f32!(sub_f32_vec, -);
elementwise_vec_avx512_f32!(mul_f32_vec, *);
elementwise_vec_avx512_f32!(div_f32_vec, /);

// ─── Element-wise f64 ──────────────────────────────────────────────

macro_rules! elementwise_scalar_avx512_f64 {
    ($name:ident, $op:tt) => {
        /// # Safety
        /// Caller must ensure AVX-512F is available at runtime.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f")]
        pub unsafe fn $name(a: &[f64], scalar: f64) -> Vec<f64> {
            let len = a.len();
            let mut out = vec![0.0f64; len];
            let chunks = len / F64_LANES;
            let sv = f64x8::splat(scalar);
            for i in 0..chunks {
                let base = i * F64_LANES;
                let av = f64x8::from_slice(&a[base..]);
                (av $op sv).copy_to_slice(&mut out[base..base + F64_LANES]);
            }
            for i in (chunks * F64_LANES)..len {
                out[i] = a[i] $op scalar;
            }
            out
        }
    };
}

elementwise_scalar_avx512_f64!(add_f64_scalar, +);
elementwise_scalar_avx512_f64!(sub_f64_scalar, -);
elementwise_scalar_avx512_f64!(mul_f64_scalar, *);
elementwise_scalar_avx512_f64!(div_f64_scalar, /);

macro_rules! elementwise_vec_avx512_f64 {
    ($name:ident, $op:tt) => {
        /// # Safety
        /// Caller must ensure AVX-512F is available at runtime.
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx512f")]
        pub unsafe fn $name(a: &[f64], b: &[f64]) -> Vec<f64> {
            let len = a.len();
            let mut out = vec![0.0f64; len];
            let chunks = len / F64_LANES;
            for i in 0..chunks {
                let base = i * F64_LANES;
                let av = f64x8::from_slice(&a[base..]);
                let bv = f64x8::from_slice(&b[base..]);
                (av $op bv).copy_to_slice(&mut out[base..base + F64_LANES]);
            }
            for i in (chunks * F64_LANES)..len {
                out[i] = a[i] $op b[i];
            }
            out
        }
    };
}

elementwise_vec_avx512_f64!(add_f64_vec, +);
elementwise_vec_avx512_f64!(sub_f64_vec, -);
elementwise_vec_avx512_f64!(mul_f64_vec, *);
elementwise_vec_avx512_f64!(div_f64_vec, /);

// ─── Hamming / bitops (VPOPCNTDQ) ─────────────────────────────────

/// # Safety
/// Caller must ensure AVX-512F and AVX-512 VPOPCNTDQ are available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
pub unsafe fn hamming_distance(a: &[u8], b: &[u8]) -> u64 {
    use core::arch::x86_64::*;
    let len = a.len();
    let chunks = len / 64;
    let mut total = _mm512_setzero_si512();
    for i in 0..chunks {
        let base = i * 64;
        let av = _mm512_loadu_si512(a[base..].as_ptr() as *const __m512i);
        let bv = _mm512_loadu_si512(b[base..].as_ptr() as *const __m512i);
        let xored = _mm512_xor_si512(av, bv);
        total = _mm512_add_epi64(total, _mm512_popcnt_epi64(xored));
    }
    let mut vals = [0i64; 8];
    _mm512_storeu_si512(vals.as_mut_ptr() as *mut __m512i, total);
    let mut sum: u64 = vals.iter().map(|&v| v as u64).sum();
    for i in (chunks * 64)..len {
        sum += (a[i] ^ b[i]).count_ones() as u64;
    }
    sum
}

/// # Safety
/// Caller must ensure AVX-512F and AVX-512 VPOPCNTDQ are available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
pub unsafe fn popcount(a: &[u8]) -> u64 {
    use core::arch::x86_64::*;
    let len = a.len();
    let chunks = len / 64;
    let mut total = _mm512_setzero_si512();
    for i in 0..chunks {
        let base = i * 64;
        let v = _mm512_loadu_si512(a[base..].as_ptr() as *const __m512i);
        total = _mm512_add_epi64(total, _mm512_popcnt_epi64(v));
    }
    let mut vals = [0i64; 8];
    _mm512_storeu_si512(vals.as_mut_ptr() as *mut __m512i, total);
    let mut sum: u64 = vals.iter().map(|&v| v as u64).sum();
    for &byte in &a[chunks * 64..] {
        sum += byte.count_ones() as u64;
    }
    sum
}

/// # Safety
/// Caller must ensure AVX-512F and AVX-512 VNNI are available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512vnni")]
pub unsafe fn dot_i8(a: &[u8], b: &[u8]) -> i64 {
    use core::arch::x86_64::*;
    let len = a.len();
    let chunks = len / 64;
    let bias = _mm512_set1_epi32(0x80808080u32 as i32);
    let ones = _mm512_set1_epi32(0x01010101u32 as i32);
    let mut acc = _mm512_setzero_si512();
    let mut b_sum = _mm512_setzero_si512();
    for i in 0..chunks {
        let base = i * 64;
        let av = _mm512_loadu_si512(a[base..].as_ptr() as *const __m512i);
        let bv = _mm512_loadu_si512(b[base..].as_ptr() as *const __m512i);
        let av_u = _mm512_xor_si512(av, bias);
        acc = _mm512_dpbusd_epi32(acc, av_u, bv);
        b_sum = _mm512_dpbusd_epi32(b_sum, ones, bv);
    }
    let mut acc_vals = [0i32; 16];
    _mm512_storeu_si512(acc_vals.as_mut_ptr() as *mut __m512i, acc);
    let total_biased: i64 = acc_vals.iter().map(|&v| v as i64).sum();
    let mut bsum_vals = [0i32; 16];
    _mm512_storeu_si512(bsum_vals.as_mut_ptr() as *mut __m512i, b_sum);
    let total_b: i64 = bsum_vals.iter().map(|&v| v as i64).sum();
    let mut result = total_biased - 128 * total_b;
    for i in (chunks * 64)..len {
        result += (a[i] as i8 as i64) * (b[i] as i8 as i64);
    }
    result
}

// ─── AVX-512 BW hamming (vpshufb) — no VPOPCNTDQ required ─────────

/// AVX-512 BW hamming using 512-bit vpshufb — 64 bytes per iteration.
/// Works on any CPU with avx512bw (no VPOPCNTDQ required).
///
/// # Safety
/// Caller must ensure AVX-512 BW is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
pub unsafe fn hamming_distance_bw(a: &[u8], b: &[u8]) -> u64 {
    use core::arch::x86_64::*;
    let n = a.len().min(b.len());
    let mut total = 0u64;

    // vpshufb LUT: popcount of each nibble (replicated across 64B)
    let lookup = _mm512_set4_epi32(
        0x04030302_i32, 0x03020201_i32, 0x03020201_i32, 0x02010100_i32,
    );
    let low_mask = _mm512_set1_epi8(0x0f);
    let mut acc = _mm512_setzero_si512();
    let mut i = 0;
    let mut inner_count = 0u32;

    while i + 64 <= n {
        let va = _mm512_loadu_si512(a.as_ptr().add(i) as *const _);
        let vb = _mm512_loadu_si512(b.as_ptr().add(i) as *const _);
        let xor = _mm512_xor_si512(va, vb);

        let lo = _mm512_and_si512(xor, low_mask);
        let hi = _mm512_and_si512(_mm512_srli_epi16(xor, 4), low_mask);
        let popcnt_lo = _mm512_shuffle_epi8(lookup, lo);
        let popcnt_hi = _mm512_shuffle_epi8(lookup, hi);
        acc = _mm512_add_epi8(acc, _mm512_add_epi8(popcnt_lo, popcnt_hi));

        i += 64;
        inner_count += 1;
        // Flush u8 accumulators before overflow (max 255/8 ≈ 31 iterations)
        if inner_count >= 30 {
            let sad = _mm512_sad_epu8(acc, _mm512_setzero_si512());
            total += _mm512_reduce_add_epi64(sad) as u64;
            acc = _mm512_setzero_si512();
            inner_count = 0;
        }
    }

    if inner_count > 0 {
        let sad = _mm512_sad_epu8(acc, _mm512_setzero_si512());
        total += _mm512_reduce_add_epi64(sad) as u64;
    }

    // Remainder
    while i < n {
        total += (a[i] ^ b[i]).count_ones() as u64;
        i += 1;
    }
    total
}

/// AVX-512 BW popcount using 512-bit vpshufb — 64 bytes per iteration.
///
/// # Safety
/// Caller must ensure AVX-512 BW is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
pub unsafe fn popcount_bw(a: &[u8]) -> u64 {
    use core::arch::x86_64::*;
    let n = a.len();
    let mut total = 0u64;

    let lookup = _mm512_set4_epi32(
        0x04030302_i32, 0x03020201_i32, 0x03020201_i32, 0x02010100_i32,
    );
    let low_mask = _mm512_set1_epi8(0x0f);
    let mut acc = _mm512_setzero_si512();
    let mut i = 0;
    let mut inner_count = 0u32;

    while i + 64 <= n {
        let va = _mm512_loadu_si512(a.as_ptr().add(i) as *const _);
        let lo = _mm512_and_si512(va, low_mask);
        let hi = _mm512_and_si512(_mm512_srli_epi16(va, 4), low_mask);
        let popcnt_lo = _mm512_shuffle_epi8(lookup, lo);
        let popcnt_hi = _mm512_shuffle_epi8(lookup, hi);
        acc = _mm512_add_epi8(acc, _mm512_add_epi8(popcnt_lo, popcnt_hi));

        i += 64;
        inner_count += 1;
        if inner_count >= 30 {
            let sad = _mm512_sad_epu8(acc, _mm512_setzero_si512());
            total += _mm512_reduce_add_epi64(sad) as u64;
            acc = _mm512_setzero_si512();
            inner_count = 0;
        }
    }

    if inner_count > 0 {
        let sad = _mm512_sad_epu8(acc, _mm512_setzero_si512());
        total += _mm512_reduce_add_epi64(sad) as u64;
    }

    while i < n {
        total += a[i].count_ones() as u64;
        i += 1;
    }
    total
}

/// AVX-512 BW batch hamming using vpshufb — for CPUs without VPOPCNTDQ.
///
/// # Safety
/// Caller must ensure AVX-512 BW is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
pub unsafe fn hamming_batch_bw(query: &[u8], database: &[u8], num_rows: usize, row_bytes: usize) -> Vec<u64> {
    let mut distances = vec![0u64; num_rows];
    let full = num_rows / 4;
    for i in 0..full {
        let base = i * 4;
        distances[base] = hamming_distance_bw(query, &database[base * row_bytes..(base + 1) * row_bytes]);
        distances[base + 1] = hamming_distance_bw(query, &database[(base + 1) * row_bytes..(base + 2) * row_bytes]);
        distances[base + 2] = hamming_distance_bw(query, &database[(base + 2) * row_bytes..(base + 3) * row_bytes]);
        distances[base + 3] = hamming_distance_bw(query, &database[(base + 3) * row_bytes..(base + 4) * row_bytes]);
    }
    for i in (full * 4)..num_rows {
        distances[i] = hamming_distance_bw(query, &database[i * row_bytes..(i + 1) * row_bytes]);
    }
    distances
}

/// AVX-512 BW top-k using vpshufb — for CPUs without VPOPCNTDQ.
///
/// # Safety
/// Caller must ensure AVX-512 BW is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512bw")]
pub unsafe fn hamming_top_k_bw(
    query: &[u8],
    database: &[u8],
    num_rows: usize,
    row_bytes: usize,
    k: usize,
) -> (Vec<usize>, Vec<u64>) {
    let distances = hamming_batch_bw(query, database, num_rows, row_bytes);
    let k = k.min(num_rows);
    let mut indices: Vec<usize> = (0..num_rows).collect();
    indices.select_nth_unstable_by_key(k.saturating_sub(1), |&i| distances[i]);
    indices.truncate(k);
    indices.sort_unstable_by_key(|&i| distances[i]);
    let top_distances: Vec<u64> = indices.iter().map(|&i| distances[i]).collect();
    (indices, top_distances)
}

// ─── Batch / top-k ─────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
/// # Safety
/// Caller must ensure AVX-512F and AVX-512 VPOPCNTDQ are available at runtime.
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
pub unsafe fn hamming_batch(query: &[u8], database: &[u8], num_rows: usize, row_bytes: usize) -> Vec<u64> {
    let mut distances = vec![0u64; num_rows];
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

#[cfg(target_arch = "x86_64")]
/// # Safety
/// Caller must ensure AVX-512F and AVX-512 VPOPCNTDQ are available at runtime.
#[target_feature(enable = "avx512f,avx512vpopcntdq")]
pub unsafe fn hamming_top_k(
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

// ═══════════════════════════════════════════════════════════════════
// GEMM — Goto BLAS packed microkernel (AVX-512)
// ═══════════════════════════════════════════════════════════════════

// Tile parameters for AVX-512:
// MR=6 rows x NR=16 cols -> 6 zmm registers for C tile
// KC chosen to fit A_panel + B_panel + C_tile in L1 (32KB)
const SGEMM_MR: usize = 6;
const SGEMM_NR: usize = 16;
const SGEMM_KC: usize = 256; // 6*256*4 + 256*16*4 + 6*16*4 = 6K+16K+384 ~ 22KB < 32KB L1
const SGEMM_MC: usize = 72;  // 12 micro-panels of MR=6
const SGEMM_NC: usize = 256; // 16 micro-panels of NR=16

const DGEMM_MR: usize = 6;
const DGEMM_NR: usize = 8;
const DGEMM_KC: usize = 192;
const DGEMM_MC: usize = 72;
const DGEMM_NC: usize = 128;

/// Pack a panel of A (mc x kc) into column-major MR-wide strips.
/// Layout: for each k, for each MR-block of rows, store MR contiguous values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn pack_a_f32(a: &[f32], lda: usize, mc: usize, kc: usize, i_start: usize, k_start: usize, buf: &mut [f32]) {
    let mut idx = 0;
    let mut ii = 0;
    while ii + SGEMM_MR <= mc {
        for p in 0..kc {
            for ir in 0..SGEMM_MR {
                buf[idx] = a[(i_start + ii + ir) * lda + (k_start + p)];
                idx += 1;
            }
        }
        ii += SGEMM_MR;
    }
    // Remainder rows (< MR): zero-pad
    if ii < mc {
        let rem = mc - ii;
        for p in 0..kc {
            for ir in 0..SGEMM_MR {
                buf[idx] = if ir < rem { a[(i_start + ii + ir) * lda + (k_start + p)] } else { 0.0 };
                idx += 1;
            }
        }
    }
}

/// Pack a panel of B (kc x nc) into row-major NR-wide strips.
/// Layout: for each k, for each NR-block of cols, store NR contiguous values.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn pack_b_f32(b: &[f32], ldb: usize, kc: usize, nc: usize, k_start: usize, j_start: usize, buf: &mut [f32]) {
    let mut idx = 0;
    let mut jj = 0;
    while jj + SGEMM_NR <= nc {
        for p in 0..kc {
            for jr in 0..SGEMM_NR {
                buf[idx] = b[(k_start + p) * ldb + (j_start + jj + jr)];
                idx += 1;
            }
        }
        jj += SGEMM_NR;
    }
    // Remainder cols (< NR): zero-pad
    if jj < nc {
        let rem = nc - jj;
        for p in 0..kc {
            for jr in 0..SGEMM_NR {
                buf[idx] = if jr < rem { b[(k_start + p) * ldb + (j_start + jj + jr)] } else { 0.0 };
                idx += 1;
            }
        }
    }
}

/// AVX-512 microkernel: C[MR x NR] += A_packed[MR x kc] * B_packed[kc x NR]
///
/// Uses 6 zmm accumulators (one per MR row), each holding NR=16 floats.
/// Inner loop: broadcast a[ir] from A_packed, FMA with NR-wide B_packed row.
/// This is the Goto BLAS GEBP inner kernel.
///
/// # Safety
/// Caller must ensure AVX-512F is available and all slice bounds are valid.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn sgemm_ukernel_6x16(
    kc: usize,
    alpha: f32,
    a_packed: &[f32], // MR * kc elements, MR-strided
    b_packed: &[f32], // kc * NR elements, NR-strided
    c: &mut [f32],    // MR rows of C (scattered by ldc)
    ldc: usize,
    mr_eff: usize,    // effective rows (may be < MR at edge)
    nr_eff: usize,    // effective cols (may be < NR at edge)
) {
    use core::arch::x86_64::*;

    // 6 accumulators for C tile rows
    let mut c0 = _mm512_setzero_ps();
    let mut c1 = _mm512_setzero_ps();
    let mut c2 = _mm512_setzero_ps();
    let mut c3 = _mm512_setzero_ps();
    let mut c4 = _mm512_setzero_ps();
    let mut c5 = _mm512_setzero_ps();

    // Main GEBP loop: for each k, load NR-wide B row, broadcast each A element
    for p in 0..kc {
        let b_off = p * SGEMM_NR;
        let bv = _mm512_loadu_ps(b_packed[b_off..].as_ptr());

        let a_off = p * SGEMM_MR;
        c0 = _mm512_fmadd_ps(_mm512_set1_ps(a_packed[a_off]), bv, c0);
        c1 = _mm512_fmadd_ps(_mm512_set1_ps(a_packed[a_off + 1]), bv, c1);
        c2 = _mm512_fmadd_ps(_mm512_set1_ps(a_packed[a_off + 2]), bv, c2);
        c3 = _mm512_fmadd_ps(_mm512_set1_ps(a_packed[a_off + 3]), bv, c3);
        c4 = _mm512_fmadd_ps(_mm512_set1_ps(a_packed[a_off + 4]), bv, c4);
        c5 = _mm512_fmadd_ps(_mm512_set1_ps(a_packed[a_off + 5]), bv, c5);
    }

    // Scale by alpha
    let alpha_v = _mm512_set1_ps(alpha);
    c0 = _mm512_mul_ps(c0, alpha_v);
    c1 = _mm512_mul_ps(c1, alpha_v);
    c2 = _mm512_mul_ps(c2, alpha_v);
    c3 = _mm512_mul_ps(c3, alpha_v);
    c4 = _mm512_mul_ps(c4, alpha_v);
    c5 = _mm512_mul_ps(c5, alpha_v);

    // Store: add to C (beta already applied by caller)
    let rows = [c0, c1, c2, c3, c4, c5];
    for ir in 0..mr_eff {
        let row_ptr = c[ir * ldc..].as_mut_ptr();
        if nr_eff == SGEMM_NR {
            // SAFETY: full NR-wide store, bounds guaranteed by caller
            let cv = _mm512_loadu_ps(row_ptr);
            _mm512_storeu_ps(row_ptr, _mm512_add_ps(cv, rows[ir]));
        } else {
            // SAFETY: masked store for edge tiles
            let mask: u16 = (1u32 << nr_eff) as u16 - 1;
            let cv = _mm512_maskz_loadu_ps(mask, row_ptr);
            _mm512_mask_storeu_ps(row_ptr, mask, _mm512_add_ps(cv, rows[ir]));
        }
    }
}

/// Goto BLAS style blocked SGEMM with packing and AVX-512 microkernel.
///
/// C = alpha * A * B + beta * C  (beta already applied by caller)
///
/// 5-loop structure: KC -> MC -> NC -> MR x NR microkernel
///
/// # Safety
/// Caller must ensure AVX-512F is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn sgemm_blocked(
    m: usize, n: usize, k: usize,
    alpha: f32, a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    c: &mut [f32], ldc: usize,
) {
    // Pack buffers — allocated once, reused across tiles
    let mut a_packed = vec![0.0f32; SGEMM_MC * SGEMM_KC];
    let mut b_packed = vec![0.0f32; SGEMM_KC * SGEMM_NC];

    // Loop 1: KC blocks
    let mut kk = 0;
    while kk < k {
        let kc = SGEMM_KC.min(k - kk);

        // Loop 2: NC blocks
        let mut jj = 0;
        while jj < n {
            let nc = SGEMM_NC.min(n - jj);

            // Pack B panel (kc x nc)
            pack_b_f32(b, ldb, kc, nc, kk, jj, &mut b_packed);

            // Loop 3: MC blocks
            let mut ii = 0;
            while ii < m {
                let mc = SGEMM_MC.min(m - ii);

                // Pack A panel (mc x kc)
                pack_a_f32(a, lda, mc, kc, ii, kk, &mut a_packed);

                // Loop 4+5: micro-tiles MR x NR
                let mut ir = 0;
                while ir < mc {
                    let mr_eff = SGEMM_MR.min(mc - ir);

                    let mut jr = 0;
                    while jr < nc {
                        let nr_eff = SGEMM_NR.min(nc - jr);

                        let a_off = (ir / SGEMM_MR) * (SGEMM_MR * kc);
                        let b_off = (jr / SGEMM_NR) * (SGEMM_NR * kc);

                        // SAFETY: tier() verified AVX-512F, buffers sized correctly
                        sgemm_ukernel_6x16(
                            kc, alpha,
                            &a_packed[a_off..],
                            &b_packed[b_off..],
                            &mut c[(ii + ir) * ldc + (jj + jr)..],
                            ldc, mr_eff, nr_eff,
                        );

                        jr += SGEMM_NR;
                    }
                    ir += SGEMM_MR;
                }

                ii += mc;
            }
            jj += nc;
        }
        kk += kc;
    }
}

// --- DGEMM (f64) blocked ---

/// Pack a panel of A (mc x kc) into column-major MR-wide strips (f64).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn pack_a_f64(a: &[f64], lda: usize, mc: usize, kc: usize, i_start: usize, k_start: usize, buf: &mut [f64]) {
    let mut idx = 0;
    let mut ii = 0;
    while ii + DGEMM_MR <= mc {
        for p in 0..kc {
            for ir in 0..DGEMM_MR {
                buf[idx] = a[(i_start + ii + ir) * lda + (k_start + p)];
                idx += 1;
            }
        }
        ii += DGEMM_MR;
    }
    if ii < mc {
        let rem = mc - ii;
        for p in 0..kc {
            for ir in 0..DGEMM_MR {
                buf[idx] = if ir < rem { a[(i_start + ii + ir) * lda + (k_start + p)] } else { 0.0 };
                idx += 1;
            }
        }
    }
}

/// Pack a panel of B (kc x nc) into row-major NR-wide strips (f64).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn pack_b_f64(b: &[f64], ldb: usize, kc: usize, nc: usize, k_start: usize, j_start: usize, buf: &mut [f64]) {
    let mut idx = 0;
    let mut jj = 0;
    while jj + DGEMM_NR <= nc {
        for p in 0..kc {
            for jr in 0..DGEMM_NR {
                buf[idx] = b[(k_start + p) * ldb + (j_start + jj + jr)];
                idx += 1;
            }
        }
        jj += DGEMM_NR;
    }
    if jj < nc {
        let rem = nc - jj;
        for p in 0..kc {
            for jr in 0..DGEMM_NR {
                buf[idx] = if jr < rem { b[(k_start + p) * ldb + (j_start + jj + jr)] } else { 0.0 };
                idx += 1;
            }
        }
    }
}

/// AVX-512 microkernel: C[6x8] += A_packed[6xkc] * B_packed[kcx8] (f64)
///
/// # Safety
/// Caller must ensure AVX-512F is available and all slice bounds are valid.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
unsafe fn dgemm_ukernel_6x8(
    kc: usize,
    alpha: f64,
    a_packed: &[f64],
    b_packed: &[f64],
    c: &mut [f64],
    ldc: usize,
    mr_eff: usize,
    nr_eff: usize,
) {
    use core::arch::x86_64::*;

    let mut c0 = _mm512_setzero_pd();
    let mut c1 = _mm512_setzero_pd();
    let mut c2 = _mm512_setzero_pd();
    let mut c3 = _mm512_setzero_pd();
    let mut c4 = _mm512_setzero_pd();
    let mut c5 = _mm512_setzero_pd();

    for p in 0..kc {
        let b_off = p * DGEMM_NR;
        let bv = _mm512_loadu_pd(b_packed[b_off..].as_ptr());

        let a_off = p * DGEMM_MR;
        c0 = _mm512_fmadd_pd(_mm512_set1_pd(a_packed[a_off]), bv, c0);
        c1 = _mm512_fmadd_pd(_mm512_set1_pd(a_packed[a_off + 1]), bv, c1);
        c2 = _mm512_fmadd_pd(_mm512_set1_pd(a_packed[a_off + 2]), bv, c2);
        c3 = _mm512_fmadd_pd(_mm512_set1_pd(a_packed[a_off + 3]), bv, c3);
        c4 = _mm512_fmadd_pd(_mm512_set1_pd(a_packed[a_off + 4]), bv, c4);
        c5 = _mm512_fmadd_pd(_mm512_set1_pd(a_packed[a_off + 5]), bv, c5);
    }

    let alpha_v = _mm512_set1_pd(alpha);
    c0 = _mm512_mul_pd(c0, alpha_v);
    c1 = _mm512_mul_pd(c1, alpha_v);
    c2 = _mm512_mul_pd(c2, alpha_v);
    c3 = _mm512_mul_pd(c3, alpha_v);
    c4 = _mm512_mul_pd(c4, alpha_v);
    c5 = _mm512_mul_pd(c5, alpha_v);

    let rows = [c0, c1, c2, c3, c4, c5];
    for ir in 0..mr_eff {
        let row_ptr = c[ir * ldc..].as_mut_ptr();
        if nr_eff == DGEMM_NR {
            // SAFETY: full NR-wide store, bounds guaranteed by caller
            let cv = _mm512_loadu_pd(row_ptr);
            _mm512_storeu_pd(row_ptr, _mm512_add_pd(cv, rows[ir]));
        } else {
            // SAFETY: masked store for edge tiles
            let mask: u8 = (1u16 << nr_eff) as u8 - 1;
            let cv = _mm512_maskz_loadu_pd(mask, row_ptr);
            _mm512_mask_storeu_pd(row_ptr, mask, _mm512_add_pd(cv, rows[ir]));
        }
    }
}

/// Goto BLAS style blocked DGEMM with packing and AVX-512 microkernel.
///
/// C = alpha * A * B + beta * C  (beta already applied by caller)
///
/// # Safety
/// Caller must ensure AVX-512F is available at runtime.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn dgemm_blocked(
    m: usize, n: usize, k: usize,
    alpha: f64, a: &[f64], lda: usize,
    b: &[f64], ldb: usize,
    c: &mut [f64], ldc: usize,
) {
    let mut a_packed = vec![0.0f64; DGEMM_MC * DGEMM_KC];
    let mut b_packed = vec![0.0f64; DGEMM_KC * DGEMM_NC];

    let mut kk = 0;
    while kk < k {
        let kc = DGEMM_KC.min(k - kk);

        let mut jj = 0;
        while jj < n {
            let nc = DGEMM_NC.min(n - jj);
            pack_b_f64(b, ldb, kc, nc, kk, jj, &mut b_packed);

            let mut ii = 0;
            while ii < m {
                let mc = DGEMM_MC.min(m - ii);
                pack_a_f64(a, lda, mc, kc, ii, kk, &mut a_packed);

                let mut ir = 0;
                while ir < mc {
                    let mr_eff = DGEMM_MR.min(mc - ir);
                    let mut jr = 0;
                    while jr < nc {
                        let nr_eff = DGEMM_NR.min(nc - jr);
                        let a_off = (ir / DGEMM_MR) * (DGEMM_MR * kc);
                        let b_off = (jr / DGEMM_NR) * (DGEMM_NR * kc);

                        // SAFETY: tier() verified AVX-512F, buffers sized correctly
                        dgemm_ukernel_6x8(
                            kc, alpha,
                            &a_packed[a_off..],
                            &b_packed[b_off..],
                            &mut c[(ii + ir) * ldc + (jj + jr)..],
                            ldc, mr_eff, nr_eff,
                        );

                        jr += DGEMM_NR;
                    }
                    ir += DGEMM_MR;
                }
                ii += mc;
            }
            jj += nc;
        }
        kk += kc;
    }
}
