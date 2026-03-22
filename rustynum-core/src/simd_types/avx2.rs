//! AVX2 backend: F32x16 = [F32x8; 2], F64x8 = [F64x4; 2].
//!
//! Two AVX2 operations per logical wide operation. Safe on any x86_64 CPU
//! with AVX2 + FMA (Haswell and later, ~2013).
//!
//! All intrinsics are stable `std::arch::x86_64` — no nightly required.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// We use the existing F32x8 / F64x4 wrappers from simd_avx512.rs as the
// half-width building blocks. They wrap __m256 / __m256d directly.
use crate::simd_avx512 as hw;

// ============================================================================
// F32x16 — 16 x f32, backed by [F32x8; 2] (two __m256)
// ============================================================================

#[derive(Copy, Clone)]
pub struct F32x16(pub hw::F32x8, pub hw::F32x8);

impl Default for F32x16 {
    #[inline(always)]
    fn default() -> Self {
        Self(hw::F32x8::splat(0.0), hw::F32x8::splat(0.0))
    }
}

impl F32x16 {
    pub const LANES: usize = 16;

    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        Self(hw::F32x8::splat(v), hw::F32x8::splat(v))
    }

    #[inline(always)]
    pub fn from_slice(s: &[f32]) -> Self {
        assert!(s.len() >= 16);
        Self(
            hw::F32x8::from_slice(&s[..8]),
            hw::F32x8::from_slice(&s[8..16]),
        )
    }

    #[inline(always)]
    pub fn from_array(arr: [f32; 16]) -> Self {
        let mut lo = [0.0f32; 8];
        let mut hi = [0.0f32; 8];
        lo.copy_from_slice(&arr[..8]);
        hi.copy_from_slice(&arr[8..16]);
        Self(hw::F32x8::from_array(lo), hw::F32x8::from_array(hi))
    }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 16] {
        let lo = self.0.to_array();
        let hi = self.1.to_array();
        let mut out = [0.0f32; 16];
        out[..8].copy_from_slice(&lo);
        out[8..].copy_from_slice(&hi);
        out
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [f32]) {
        assert!(s.len() >= 16);
        self.0.copy_to_slice(&mut s[..8]);
        self.1.copy_to_slice(&mut s[8..16]);
    }

    // --- Reductions ---

    #[inline(always)]
    pub fn reduce_sum(self) -> f32 {
        self.0.reduce_sum() + self.1.reduce_sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> f32 {
        let lo = self.0.to_array();
        let hi = self.1.to_array();
        let lo_min = lo.iter().copied().fold(f32::INFINITY, f32::min);
        let hi_min = hi.iter().copied().fold(f32::INFINITY, f32::min);
        lo_min.min(hi_min)
    }

    #[inline(always)]
    pub fn reduce_max(self) -> f32 {
        let lo = self.0.to_array();
        let hi = self.1.to_array();
        let lo_max = lo.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let hi_max = hi.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        lo_max.max(hi_max)
    }

    // --- Element-wise min/max/clamp ---

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        unsafe {
            Self(
                hw::F32x8(_mm256_min_ps(self.0.0, other.0.0)),
                hw::F32x8(_mm256_min_ps(self.1.0, other.1.0)),
            )
        }
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        unsafe {
            Self(
                hw::F32x8(_mm256_max_ps(self.0.0, other.0.0)),
                hw::F32x8(_mm256_max_ps(self.1.0, other.1.0)),
            )
        }
    }

    #[inline(always)]
    pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
        self.simd_max(lo).simd_min(hi)
    }

    // --- Math ---

    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        unsafe {
            Self(
                hw::F32x8(_mm256_fmadd_ps(self.0.0, b.0.0, c.0.0)),
                hw::F32x8(_mm256_fmadd_ps(self.1.0, b.1.0, c.1.0)),
            )
        }
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        unsafe {
            Self(
                hw::F32x8(_mm256_sqrt_ps(self.0.0)),
                hw::F32x8(_mm256_sqrt_ps(self.1.0)),
            )
        }
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(self.0.abs(), self.1.abs())
    }

    #[inline(always)]
    pub fn round(self) -> Self {
        unsafe {
            Self(
                hw::F32x8(_mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(self.0.0)),
                hw::F32x8(_mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(self.1.0)),
            )
        }
    }

    #[inline(always)]
    pub fn floor(self) -> Self {
        unsafe {
            Self(
                hw::F32x8(_mm256_floor_ps(self.0.0)),
                hw::F32x8(_mm256_floor_ps(self.1.0)),
            )
        }
    }

    // --- Comparisons ---

    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> F32Mask16 {
        unsafe {
            let lo = _mm256_cmp_ps::<_CMP_LT_OS>(self.0.0, other.0.0);
            let hi = _mm256_cmp_ps::<_CMP_LT_OS>(self.1.0, other.1.0);
            let lo_mask = _mm256_movemask_ps(lo) as u16;
            let hi_mask = (_mm256_movemask_ps(hi) as u16) << 8;
            F32Mask16(lo_mask | hi_mask)
        }
    }

    #[inline(always)]
    pub fn simd_le(self, other: Self) -> F32Mask16 {
        unsafe {
            let lo = _mm256_cmp_ps::<_CMP_LE_OS>(self.0.0, other.0.0);
            let hi = _mm256_cmp_ps::<_CMP_LE_OS>(self.1.0, other.1.0);
            let lo_mask = _mm256_movemask_ps(lo) as u16;
            let hi_mask = (_mm256_movemask_ps(hi) as u16) << 8;
            F32Mask16(lo_mask | hi_mask)
        }
    }

    #[inline(always)]
    pub fn simd_gt(self, other: Self) -> F32Mask16 {
        other.simd_lt(self)
    }

    #[inline(always)]
    pub fn simd_ge(self, other: Self) -> F32Mask16 {
        other.simd_le(self)
    }

    #[inline(always)]
    pub fn simd_eq(self, other: Self) -> F32Mask16 {
        unsafe {
            let lo = _mm256_cmp_ps::<_CMP_EQ_OQ>(self.0.0, other.0.0);
            let hi = _mm256_cmp_ps::<_CMP_EQ_OQ>(self.1.0, other.1.0);
            let lo_mask = _mm256_movemask_ps(lo) as u16;
            let hi_mask = (_mm256_movemask_ps(hi) as u16) << 8;
            F32Mask16(lo_mask | hi_mask)
        }
    }

    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> F32Mask16 {
        unsafe {
            let lo = _mm256_cmp_ps::<_CMP_NEQ_UQ>(self.0.0, other.0.0);
            let hi = _mm256_cmp_ps::<_CMP_NEQ_UQ>(self.1.0, other.1.0);
            let lo_mask = _mm256_movemask_ps(lo) as u16;
            let hi_mask = (_mm256_movemask_ps(hi) as u16) << 8;
            F32Mask16(lo_mask | hi_mask)
        }
    }
}

// --- Arithmetic operators ---

impl Add for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl Sub for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            Self(
                hw::F32x8(_mm256_sub_ps(self.0.0, rhs.0.0)),
                hw::F32x8(_mm256_sub_ps(self.1.0, rhs.1.0)),
            )
        }
    }
}

impl Mul for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0, self.1 * rhs.1)
    }
}

impl Div for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe {
            Self(
                hw::F32x8(_mm256_div_ps(self.0.0, rhs.0.0)),
                hw::F32x8(_mm256_div_ps(self.1.0, rhs.1.0)),
            )
        }
    }
}

impl AddAssign for F32x16 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) { self.0 += rhs.0; self.1 += rhs.1; }
}

impl SubAssign for F32x16 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for F32x16 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) { self.0 *= rhs.0; self.1 *= rhs.1; }
}

impl DivAssign for F32x16 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Neg for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe {
            let sign = _mm256_set1_epi32(i32::MIN);
            Self(
                hw::F32x8(_mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(self.0.0), sign))),
                hw::F32x8(_mm256_castsi256_ps(_mm256_xor_si256(_mm256_castps_si256(self.1.0), sign))),
            )
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
pub struct F32Mask16(pub u16);

impl F32Mask16 {
    /// Select: for each lane, if mask bit is 1 -> true_val, else false_val.
    #[inline(always)]
    pub fn select(self, true_val: F32x16, false_val: F32x16) -> F32x16 {
        unsafe {
            // Convert low 8 bits and high 8 bits to __m256 blend masks
            let lo_mask = self.bits_to_m256(self.0 & 0xFF);
            let hi_mask = self.bits_to_m256((self.0 >> 8) & 0xFF);
            F32x16(
                hw::F32x8(_mm256_blendv_ps(false_val.0.0, true_val.0.0, lo_mask)),
                hw::F32x8(_mm256_blendv_ps(false_val.1.0, true_val.1.0, hi_mask)),
            )
        }
    }

    /// Convert 8 mask bits to a __m256 where each lane is all-ones or all-zeros.
    #[inline(always)]
    unsafe fn bits_to_m256(self, bits: u16) -> __m256 {
        // Set each lane to -1 (all-ones float) or 0 based on mask bit
        _mm256_castsi256_ps(_mm256_set_epi32(
            if bits & 0x80 != 0 { -1i32 } else { 0 },
            if bits & 0x40 != 0 { -1i32 } else { 0 },
            if bits & 0x20 != 0 { -1i32 } else { 0 },
            if bits & 0x10 != 0 { -1i32 } else { 0 },
            if bits & 0x08 != 0 { -1i32 } else { 0 },
            if bits & 0x04 != 0 { -1i32 } else { 0 },
            if bits & 0x02 != 0 { -1i32 } else { 0 },
            if bits & 0x01 != 0 { -1i32 } else { 0 },
        ))
    }
}

// ============================================================================
// F64x8 — 8 x f64, backed by [F64x4; 2] (two __m256d)
// ============================================================================

#[derive(Copy, Clone)]
pub struct F64x8(pub hw::F64x4, pub hw::F64x4);

impl Default for F64x8 {
    #[inline(always)]
    fn default() -> Self {
        Self(hw::F64x4::splat(0.0), hw::F64x4::splat(0.0))
    }
}

impl F64x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn splat(v: f64) -> Self {
        Self(hw::F64x4::splat(v), hw::F64x4::splat(v))
    }

    #[inline(always)]
    pub fn from_slice(s: &[f64]) -> Self {
        assert!(s.len() >= 8);
        Self(
            hw::F64x4::from_slice(&s[..4]),
            hw::F64x4::from_slice(&s[4..8]),
        )
    }

    #[inline(always)]
    pub fn from_array(arr: [f64; 8]) -> Self {
        let mut lo = [0.0f64; 4];
        let mut hi = [0.0f64; 4];
        lo.copy_from_slice(&arr[..4]);
        hi.copy_from_slice(&arr[4..8]);
        Self(hw::F64x4::from_array(lo), hw::F64x4::from_array(hi))
    }

    #[inline(always)]
    pub fn to_array(self) -> [f64; 8] {
        let lo = self.0.to_array();
        let hi = self.1.to_array();
        let mut out = [0.0f64; 8];
        out[..4].copy_from_slice(&lo);
        out[4..].copy_from_slice(&hi);
        out
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [f64]) {
        assert!(s.len() >= 8);
        self.0.copy_to_slice(&mut s[..4]);
        self.1.copy_to_slice(&mut s[4..8]);
    }

    #[inline(always)]
    pub fn reduce_sum(self) -> f64 {
        self.0.reduce_sum() + self.1.reduce_sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> f64 {
        let lo = self.0.to_array();
        let hi = self.1.to_array();
        let lo_min = lo.iter().copied().fold(f64::INFINITY, f64::min);
        let hi_min = hi.iter().copied().fold(f64::INFINITY, f64::min);
        lo_min.min(hi_min)
    }

    #[inline(always)]
    pub fn reduce_max(self) -> f64 {
        let lo = self.0.to_array();
        let hi = self.1.to_array();
        let lo_max = lo.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let hi_max = hi.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        lo_max.max(hi_max)
    }

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        unsafe {
            Self(
                hw::F64x4(_mm256_min_pd(self.0.0, other.0.0)),
                hw::F64x4(_mm256_min_pd(self.1.0, other.1.0)),
            )
        }
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        unsafe {
            Self(
                hw::F64x4(_mm256_max_pd(self.0.0, other.0.0)),
                hw::F64x4(_mm256_max_pd(self.1.0, other.1.0)),
            )
        }
    }

    #[inline(always)]
    pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
        self.simd_max(lo).simd_min(hi)
    }

    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        unsafe {
            Self(
                hw::F64x4(_mm256_fmadd_pd(self.0.0, b.0.0, c.0.0)),
                hw::F64x4(_mm256_fmadd_pd(self.1.0, b.1.0, c.1.0)),
            )
        }
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        unsafe {
            Self(
                hw::F64x4(_mm256_sqrt_pd(self.0.0)),
                hw::F64x4(_mm256_sqrt_pd(self.1.0)),
            )
        }
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        Self(self.0.abs(), self.1.abs())
    }
}

impl Add for F64x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl Sub for F64x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            Self(
                hw::F64x4(_mm256_sub_pd(self.0.0, rhs.0.0)),
                hw::F64x4(_mm256_sub_pd(self.1.0, rhs.1.0)),
            )
        }
    }
}

impl Mul for F64x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0, self.1 * rhs.1)
    }
}

impl Div for F64x8 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe {
            Self(
                hw::F64x4(_mm256_div_pd(self.0.0, rhs.0.0)),
                hw::F64x4(_mm256_div_pd(self.1.0, rhs.1.0)),
            )
        }
    }
}

impl AddAssign for F64x8 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) { self.0 += rhs.0; self.1 += rhs.1; }
}

impl SubAssign for F64x8 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}

impl MulAssign for F64x8 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) { self.0 *= rhs.0; self.1 *= rhs.1; }
}

impl DivAssign for F64x8 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) { *self = *self / rhs; }
}

impl Neg for F64x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe {
            let sign = _mm256_set1_epi64x(i64::MIN);
            Self(
                hw::F64x4(_mm256_castsi256_pd(_mm256_xor_si256(_mm256_castpd_si256(self.0.0), sign))),
                hw::F64x4(_mm256_castsi256_pd(_mm256_xor_si256(_mm256_castpd_si256(self.1.0), sign))),
            )
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
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32x16_splat_and_sum() {
        let v = F32x16::splat(3.0);
        assert!((v.reduce_sum() - 48.0).abs() < 1e-6);
    }

    #[test]
    fn f32x16_fma() {
        let a = F32x16::splat(2.0);
        let b = F32x16::splat(3.0);
        let c = F32x16::splat(1.0);
        let r = a.mul_add(b, c);
        assert!((r.reduce_sum() - 112.0).abs() < 1e-4); // 16 * 7
    }

    #[test]
    fn f32x16_from_slice_roundtrip() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let v = F32x16::from_slice(&data);
        let arr = v.to_array();
        assert_eq!(arr[0], 0.0);
        assert_eq!(arr[15], 15.0);
    }

    #[test]
    fn f32x16_add_sub_mul_div() {
        let a = F32x16::splat(6.0);
        let b = F32x16::splat(2.0);
        assert!(((a + b).reduce_sum() - 128.0).abs() < 1e-4);
        assert!(((a - b).reduce_sum() - 64.0).abs() < 1e-4);
        assert!(((a * b).reduce_sum() - 192.0).abs() < 1e-4);
        assert!(((a / b).reduce_sum() - 48.0).abs() < 1e-4);
    }

    #[test]
    fn f32x16_mask_select() {
        let a = F32x16::from_array([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ]);
        let threshold = F32x16::splat(8.5);
        let mask = a.simd_lt(threshold);
        let result = mask.select(F32x16::splat(1.0), F32x16::splat(0.0));
        assert!((result.reduce_sum() - 8.0).abs() < 1e-6);
    }

    #[test]
    fn f64x8_splat_and_sum() {
        let v = F64x8::splat(3.0);
        assert!((v.reduce_sum() - 24.0).abs() < 1e-10);
    }

    #[test]
    fn f64x8_fma() {
        let a = F64x8::splat(2.0);
        let b = F64x8::splat(3.0);
        let c = F64x8::splat(1.0);
        let r = a.mul_add(b, c);
        assert!((r.reduce_sum() - 56.0).abs() < 1e-10);
    }

    #[test]
    fn f64x8_from_slice_roundtrip() {
        let data: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let v = F64x8::from_slice(&data);
        let arr = v.to_array();
        assert_eq!(arr[0], 0.0);
        assert_eq!(arr[7], 7.0);
    }
}
