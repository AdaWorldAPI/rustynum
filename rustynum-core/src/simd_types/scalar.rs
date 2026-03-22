//! Scalar fallback: F32x16 = [f32; 16], F64x8 = [f64; 8].
//!
//! Correct on every architecture. LLVM may auto-vectorize the loops.
//! This is NOT a separate code path — kernel code using these types
//! compiles identically to kernel code using the AVX-512 or NEON backends.

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ============================================================================
// F32x16 — 16 x f32, backed by [f32; 16]
// ============================================================================

#[derive(Copy, Clone)]
#[repr(align(64))]
pub struct F32x16(pub [f32; 16]);

impl Default for F32x16 {
    #[inline(always)]
    fn default() -> Self {
        Self([0.0; 16])
    }
}

impl F32x16 {
    pub const LANES: usize = 16;

    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        Self([v; 16])
    }

    #[inline(always)]
    pub fn from_slice(s: &[f32]) -> Self {
        assert!(s.len() >= 16);
        let mut arr = [0.0f32; 16];
        arr.copy_from_slice(&s[..16]);
        Self(arr)
    }

    #[inline(always)]
    pub fn from_array(arr: [f32; 16]) -> Self {
        Self(arr)
    }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 16] {
        self.0
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [f32]) {
        assert!(s.len() >= 16);
        s[..16].copy_from_slice(&self.0);
    }

    // --- Reductions ---

    #[inline(always)]
    pub fn reduce_sum(self) -> f32 {
        self.0.iter().sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> f32 {
        self.0.iter().copied().fold(f32::INFINITY, f32::min)
    }

    #[inline(always)]
    pub fn reduce_max(self) -> f32 {
        self.0.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    // --- Element-wise min/max/clamp ---

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        let mut out = [0.0f32; 16];
        for i in 0..16 {
            out[i] = self.0[i].min(other.0[i]);
        }
        Self(out)
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        let mut out = [0.0f32; 16];
        for i in 0..16 {
            out[i] = self.0[i].max(other.0[i]);
        }
        Self(out)
    }

    #[inline(always)]
    pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
        self.simd_max(lo).simd_min(hi)
    }

    // --- Math ---

    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        let mut out = [0.0f32; 16];
        for i in 0..16 {
            out[i] = self.0[i].mul_add(b.0[i], c.0[i]);
        }
        Self(out)
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        let mut out = [0.0f32; 16];
        for i in 0..16 {
            out[i] = self.0[i].sqrt();
        }
        Self(out)
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        let mut out = [0.0f32; 16];
        for i in 0..16 {
            out[i] = self.0[i].abs();
        }
        Self(out)
    }

    #[inline(always)]
    pub fn round(self) -> Self {
        let mut out = [0.0f32; 16];
        for i in 0..16 {
            out[i] = self.0[i].round();
        }
        Self(out)
    }

    #[inline(always)]
    pub fn floor(self) -> Self {
        let mut out = [0.0f32; 16];
        for i in 0..16 {
            out[i] = self.0[i].floor();
        }
        Self(out)
    }

    // --- Comparisons ---

    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> F32Mask16 {
        let mut bits = 0u16;
        for i in 0..16 {
            if self.0[i] < other.0[i] {
                bits |= 1 << i;
            }
        }
        F32Mask16(bits)
    }

    #[inline(always)]
    pub fn simd_le(self, other: Self) -> F32Mask16 {
        let mut bits = 0u16;
        for i in 0..16 {
            if self.0[i] <= other.0[i] {
                bits |= 1 << i;
            }
        }
        F32Mask16(bits)
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
        let mut bits = 0u16;
        for i in 0..16 {
            if self.0[i] == other.0[i] {
                bits |= 1 << i;
            }
        }
        F32Mask16(bits)
    }

    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> F32Mask16 {
        let mut bits = 0u16;
        for i in 0..16 {
            if self.0[i] != other.0[i] {
                bits |= 1 << i;
            }
        }
        F32Mask16(bits)
    }
}

// --- Arithmetic operators ---

impl Add for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let mut out = [0.0f32; 16];
        for i in 0..16 { out[i] = self.0[i] + rhs.0[i]; }
        Self(out)
    }
}

impl Sub for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        let mut out = [0.0f32; 16];
        for i in 0..16 { out[i] = self.0[i] - rhs.0[i]; }
        Self(out)
    }
}

impl Mul for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let mut out = [0.0f32; 16];
        for i in 0..16 { out[i] = self.0[i] * rhs.0[i]; }
        Self(out)
    }
}

impl Div for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let mut out = [0.0f32; 16];
        for i in 0..16 { out[i] = self.0[i] / rhs.0[i]; }
        Self(out)
    }
}

impl AddAssign for F32x16 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) { for i in 0..16 { self.0[i] += rhs.0[i]; } }
}

impl SubAssign for F32x16 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) { for i in 0..16 { self.0[i] -= rhs.0[i]; } }
}

impl MulAssign for F32x16 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) { for i in 0..16 { self.0[i] *= rhs.0[i]; } }
}

impl DivAssign for F32x16 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) { for i in 0..16 { self.0[i] /= rhs.0[i]; } }
}

impl Neg for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        let mut out = [0.0f32; 16];
        for i in 0..16 { out[i] = -self.0[i]; }
        Self(out)
    }
}

impl fmt::Debug for F32x16 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F32x16({:?})", self.0)
    }
}

impl PartialEq for F32x16 {
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
}

// ============================================================================
// F32Mask16 — 16-bit mask from f32 comparisons
// ============================================================================

#[derive(Copy, Clone, Debug)]
pub struct F32Mask16(pub u16);

impl F32Mask16 {
    #[inline(always)]
    pub fn select(self, true_val: F32x16, false_val: F32x16) -> F32x16 {
        let mut out = [0.0f32; 16];
        for i in 0..16 {
            out[i] = if (self.0 >> i) & 1 == 1 { true_val.0[i] } else { false_val.0[i] };
        }
        F32x16(out)
    }
}

// ============================================================================
// F64x8 — 8 x f64, backed by [f64; 8]
// ============================================================================

#[derive(Copy, Clone)]
#[repr(align(64))]
pub struct F64x8(pub [f64; 8]);

impl Default for F64x8 {
    #[inline(always)]
    fn default() -> Self {
        Self([0.0; 8])
    }
}

impl F64x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn splat(v: f64) -> Self {
        Self([v; 8])
    }

    #[inline(always)]
    pub fn from_slice(s: &[f64]) -> Self {
        assert!(s.len() >= 8);
        let mut arr = [0.0f64; 8];
        arr.copy_from_slice(&s[..8]);
        Self(arr)
    }

    #[inline(always)]
    pub fn from_array(arr: [f64; 8]) -> Self {
        Self(arr)
    }

    #[inline(always)]
    pub fn to_array(self) -> [f64; 8] {
        self.0
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [f64]) {
        assert!(s.len() >= 8);
        s[..8].copy_from_slice(&self.0);
    }

    #[inline(always)]
    pub fn reduce_sum(self) -> f64 {
        self.0.iter().sum()
    }

    #[inline(always)]
    pub fn reduce_min(self) -> f64 {
        self.0.iter().copied().fold(f64::INFINITY, f64::min)
    }

    #[inline(always)]
    pub fn reduce_max(self) -> f64 {
        self.0.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        let mut out = [0.0f64; 8];
        for i in 0..8 { out[i] = self.0[i].min(other.0[i]); }
        Self(out)
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        let mut out = [0.0f64; 8];
        for i in 0..8 { out[i] = self.0[i].max(other.0[i]); }
        Self(out)
    }

    #[inline(always)]
    pub fn simd_clamp(self, lo: Self, hi: Self) -> Self {
        self.simd_max(lo).simd_min(hi)
    }

    #[inline(always)]
    pub fn mul_add(self, b: Self, c: Self) -> Self {
        let mut out = [0.0f64; 8];
        for i in 0..8 { out[i] = self.0[i].mul_add(b.0[i], c.0[i]); }
        Self(out)
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        let mut out = [0.0f64; 8];
        for i in 0..8 { out[i] = self.0[i].sqrt(); }
        Self(out)
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        let mut out = [0.0f64; 8];
        for i in 0..8 { out[i] = self.0[i].abs(); }
        Self(out)
    }
}

impl Add for F64x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let mut out = [0.0f64; 8];
        for i in 0..8 { out[i] = self.0[i] + rhs.0[i]; }
        Self(out)
    }
}

impl Sub for F64x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        let mut out = [0.0f64; 8];
        for i in 0..8 { out[i] = self.0[i] - rhs.0[i]; }
        Self(out)
    }
}

impl Mul for F64x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let mut out = [0.0f64; 8];
        for i in 0..8 { out[i] = self.0[i] * rhs.0[i]; }
        Self(out)
    }
}

impl Div for F64x8 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let mut out = [0.0f64; 8];
        for i in 0..8 { out[i] = self.0[i] / rhs.0[i]; }
        Self(out)
    }
}

impl AddAssign for F64x8 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) { for i in 0..8 { self.0[i] += rhs.0[i]; } }
}

impl SubAssign for F64x8 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) { for i in 0..8 { self.0[i] -= rhs.0[i]; } }
}

impl MulAssign for F64x8 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) { for i in 0..8 { self.0[i] *= rhs.0[i]; } }
}

impl DivAssign for F64x8 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) { for i in 0..8 { self.0[i] /= rhs.0[i]; } }
}

impl Neg for F64x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        let mut out = [0.0f64; 8];
        for i in 0..8 { out[i] = -self.0[i]; }
        Self(out)
    }
}

impl fmt::Debug for F64x8 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "F64x8({:?})", self.0)
    }
}

impl PartialEq for F64x8 {
    fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
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
    fn f32x16_add_sub_mul_div() {
        let a = F32x16::splat(6.0);
        let b = F32x16::splat(2.0);
        assert!((a + b).reduce_sum() - 128.0 < 1e-4);
        assert!((a - b).reduce_sum() - 64.0 < 1e-4);
        assert!((a * b).reduce_sum() - 192.0 < 1e-4);
        assert!((a / b).reduce_sum() - 48.0 < 1e-4);
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
        assert!((r.reduce_sum() - 56.0).abs() < 1e-10); // 8 * 7
    }
}
