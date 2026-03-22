//! NEON backend: F32x16 = [float32x4_t; 4], F64x8 = [float64x2_t; 4].
//!
//! Four NEON operations per logical wide operation. NEON is baseline on
//! aarch64 — always available, no runtime detection needed.
//!
//! All intrinsics are stable `std::arch::aarch64` — no nightly required.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

// ============================================================================
// F32x16 — 16 x f32, backed by [float32x4_t; 4] (four NEON registers)
// ============================================================================

#[derive(Copy, Clone)]
pub struct F32x16(
    pub float32x4_t,
    pub float32x4_t,
    pub float32x4_t,
    pub float32x4_t,
);

impl Default for F32x16 {
    #[inline(always)]
    fn default() -> Self {
        unsafe {
            let z = vdupq_n_f32(0.0);
            Self(z, z, z, z)
        }
    }
}

impl F32x16 {
    pub const LANES: usize = 16;

    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        unsafe {
            let s = vdupq_n_f32(v);
            Self(s, s, s, s)
        }
    }

    #[inline(always)]
    pub fn from_slice(s: &[f32]) -> Self {
        assert!(s.len() >= 16);
        unsafe {
            Self(
                vld1q_f32(s.as_ptr()),
                vld1q_f32(s.as_ptr().add(4)),
                vld1q_f32(s.as_ptr().add(8)),
                vld1q_f32(s.as_ptr().add(12)),
            )
        }
    }

    #[inline(always)]
    pub fn from_array(arr: [f32; 16]) -> Self {
        Self::from_slice(&arr)
    }

    #[inline(always)]
    pub fn to_array(self) -> [f32; 16] {
        let mut out = [0.0f32; 16];
        unsafe {
            vst1q_f32(out.as_mut_ptr(), self.0);
            vst1q_f32(out.as_mut_ptr().add(4), self.1);
            vst1q_f32(out.as_mut_ptr().add(8), self.2);
            vst1q_f32(out.as_mut_ptr().add(12), self.3);
        }
        out
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [f32]) {
        assert!(s.len() >= 16);
        unsafe {
            vst1q_f32(s.as_mut_ptr(), self.0);
            vst1q_f32(s.as_mut_ptr().add(4), self.1);
            vst1q_f32(s.as_mut_ptr().add(8), self.2);
            vst1q_f32(s.as_mut_ptr().add(12), self.3);
        }
    }

    // --- Reductions ---

    #[inline(always)]
    pub fn reduce_sum(self) -> f32 {
        unsafe {
            let s01 = vaddq_f32(self.0, self.1);
            let s23 = vaddq_f32(self.2, self.3);
            let s = vaddq_f32(s01, s23);
            vaddvq_f32(s)
        }
    }

    #[inline(always)]
    pub fn reduce_min(self) -> f32 {
        unsafe {
            let m01 = vminq_f32(self.0, self.1);
            let m23 = vminq_f32(self.2, self.3);
            let m = vminq_f32(m01, m23);
            vminvq_f32(m)
        }
    }

    #[inline(always)]
    pub fn reduce_max(self) -> f32 {
        unsafe {
            let m01 = vmaxq_f32(self.0, self.1);
            let m23 = vmaxq_f32(self.2, self.3);
            let m = vmaxq_f32(m01, m23);
            vmaxvq_f32(m)
        }
    }

    // --- Element-wise min/max/clamp ---

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        unsafe {
            Self(
                vminq_f32(self.0, other.0),
                vminq_f32(self.1, other.1),
                vminq_f32(self.2, other.2),
                vminq_f32(self.3, other.3),
            )
        }
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        unsafe {
            Self(
                vmaxq_f32(self.0, other.0),
                vmaxq_f32(self.1, other.1),
                vmaxq_f32(self.2, other.2),
                vmaxq_f32(self.3, other.3),
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
                vfmaq_f32(c.0, self.0, b.0),
                vfmaq_f32(c.1, self.1, b.1),
                vfmaq_f32(c.2, self.2, b.2),
                vfmaq_f32(c.3, self.3, b.3),
            )
        }
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        unsafe {
            Self(
                vsqrtq_f32(self.0),
                vsqrtq_f32(self.1),
                vsqrtq_f32(self.2),
                vsqrtq_f32(self.3),
            )
        }
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        unsafe {
            Self(
                vabsq_f32(self.0),
                vabsq_f32(self.1),
                vabsq_f32(self.2),
                vabsq_f32(self.3),
            )
        }
    }

    #[inline(always)]
    pub fn round(self) -> Self {
        unsafe {
            Self(
                vrndnq_f32(self.0),
                vrndnq_f32(self.1),
                vrndnq_f32(self.2),
                vrndnq_f32(self.3),
            )
        }
    }

    #[inline(always)]
    pub fn floor(self) -> Self {
        unsafe {
            Self(
                vrndmq_f32(self.0),
                vrndmq_f32(self.1),
                vrndmq_f32(self.2),
                vrndmq_f32(self.3),
            )
        }
    }

    // --- Comparisons ---

    #[inline(always)]
    pub fn simd_lt(self, other: Self) -> F32Mask16 {
        unsafe {
            let c0 = vcltq_f32(self.0, other.0);
            let c1 = vcltq_f32(self.1, other.1);
            let c2 = vcltq_f32(self.2, other.2);
            let c3 = vcltq_f32(self.3, other.3);
            F32Mask16::from_neon_masks(c0, c1, c2, c3)
        }
    }

    #[inline(always)]
    pub fn simd_le(self, other: Self) -> F32Mask16 {
        unsafe {
            let c0 = vcleq_f32(self.0, other.0);
            let c1 = vcleq_f32(self.1, other.1);
            let c2 = vcleq_f32(self.2, other.2);
            let c3 = vcleq_f32(self.3, other.3);
            F32Mask16::from_neon_masks(c0, c1, c2, c3)
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
            let c0 = vceqq_f32(self.0, other.0);
            let c1 = vceqq_f32(self.1, other.1);
            let c2 = vceqq_f32(self.2, other.2);
            let c3 = vceqq_f32(self.3, other.3);
            F32Mask16::from_neon_masks(c0, c1, c2, c3)
        }
    }

    #[inline(always)]
    pub fn simd_ne(self, other: Self) -> F32Mask16 {
        unsafe {
            let c0 = vmvnq_u32(vceqq_f32(self.0, other.0));
            let c1 = vmvnq_u32(vceqq_f32(self.1, other.1));
            let c2 = vmvnq_u32(vceqq_f32(self.2, other.2));
            let c3 = vmvnq_u32(vceqq_f32(self.3, other.3));
            F32Mask16::from_neon_masks(c0, c1, c2, c3)
        }
    }
}

// --- Arithmetic operators ---

impl Add for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            Self(
                vaddq_f32(self.0, rhs.0),
                vaddq_f32(self.1, rhs.1),
                vaddq_f32(self.2, rhs.2),
                vaddq_f32(self.3, rhs.3),
            )
        }
    }
}

impl Sub for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            Self(
                vsubq_f32(self.0, rhs.0),
                vsubq_f32(self.1, rhs.1),
                vsubq_f32(self.2, rhs.2),
                vsubq_f32(self.3, rhs.3),
            )
        }
    }
}

impl Mul for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            Self(
                vmulq_f32(self.0, rhs.0),
                vmulq_f32(self.1, rhs.1),
                vmulq_f32(self.2, rhs.2),
                vmulq_f32(self.3, rhs.3),
            )
        }
    }
}

impl Div for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe {
            Self(
                vdivq_f32(self.0, rhs.0),
                vdivq_f32(self.1, rhs.1),
                vdivq_f32(self.2, rhs.2),
                vdivq_f32(self.3, rhs.3),
            )
        }
    }
}

impl AddAssign for F32x16 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl SubAssign for F32x16 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}

impl MulAssign for F32x16 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}

impl DivAssign for F32x16 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) { *self = *self / rhs; }
}

impl Neg for F32x16 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        unsafe {
            Self(
                vnegq_f32(self.0),
                vnegq_f32(self.1),
                vnegq_f32(self.2),
                vnegq_f32(self.3),
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
    /// Build a 16-bit mask from four NEON uint32x4_t comparison results.
    #[inline(always)]
    unsafe fn from_neon_masks(
        c0: uint32x4_t,
        c1: uint32x4_t,
        c2: uint32x4_t,
        c3: uint32x4_t,
    ) -> Self {
        // Extract high bit of each 32-bit lane -> 4 bits per register
        let b0 = Self::neon_mask4(c0);
        let b1 = Self::neon_mask4(c1);
        let b2 = Self::neon_mask4(c2);
        let b3 = Self::neon_mask4(c3);
        Self(b0 | (b1 << 4) | (b2 << 8) | (b3 << 12))
    }

    /// Extract 4 mask bits from a uint32x4_t (each lane is all-ones or all-zeros).
    #[inline(always)]
    unsafe fn neon_mask4(v: uint32x4_t) -> u16 {
        // Shift each lane right by 31 to get 0 or 1
        let shifted = vshrq_n_u32::<31>(v);
        let lane0 = vgetq_lane_u32::<0>(shifted) as u16;
        let lane1 = vgetq_lane_u32::<1>(shifted) as u16;
        let lane2 = vgetq_lane_u32::<2>(shifted) as u16;
        let lane3 = vgetq_lane_u32::<3>(shifted) as u16;
        lane0 | (lane1 << 1) | (lane2 << 2) | (lane3 << 3)
    }

    /// Select: for each lane, if mask bit is 1 -> true_val, else false_val.
    #[inline(always)]
    pub fn select(self, true_val: F32x16, false_val: F32x16) -> F32x16 {
        unsafe {
            let m0 = self.bits_to_neon_mask(self.0 & 0xF);
            let m1 = self.bits_to_neon_mask((self.0 >> 4) & 0xF);
            let m2 = self.bits_to_neon_mask((self.0 >> 8) & 0xF);
            let m3 = self.bits_to_neon_mask((self.0 >> 12) & 0xF);
            F32x16(
                vbslq_f32(m0, true_val.0, false_val.0),
                vbslq_f32(m1, true_val.1, false_val.1),
                vbslq_f32(m2, true_val.2, false_val.2),
                vbslq_f32(m3, true_val.3, false_val.3),
            )
        }
    }

    /// Convert 4 mask bits to a uint32x4_t where each lane is all-ones or all-zeros.
    #[inline(always)]
    unsafe fn bits_to_neon_mask(self, bits: u16) -> uint32x4_t {
        let arr: [u32; 4] = [
            if bits & 1 != 0 { u32::MAX } else { 0 },
            if bits & 2 != 0 { u32::MAX } else { 0 },
            if bits & 4 != 0 { u32::MAX } else { 0 },
            if bits & 8 != 0 { u32::MAX } else { 0 },
        ];
        vld1q_u32(arr.as_ptr())
    }
}

// ============================================================================
// F64x8 — 8 x f64, backed by [float64x2_t; 4] (four NEON registers)
// ============================================================================

#[derive(Copy, Clone)]
pub struct F64x8(
    pub float64x2_t,
    pub float64x2_t,
    pub float64x2_t,
    pub float64x2_t,
);

impl Default for F64x8 {
    #[inline(always)]
    fn default() -> Self {
        unsafe {
            let z = vdupq_n_f64(0.0);
            Self(z, z, z, z)
        }
    }
}

impl F64x8 {
    pub const LANES: usize = 8;

    #[inline(always)]
    pub fn splat(v: f64) -> Self {
        unsafe {
            let s = vdupq_n_f64(v);
            Self(s, s, s, s)
        }
    }

    #[inline(always)]
    pub fn from_slice(s: &[f64]) -> Self {
        assert!(s.len() >= 8);
        unsafe {
            Self(
                vld1q_f64(s.as_ptr()),
                vld1q_f64(s.as_ptr().add(2)),
                vld1q_f64(s.as_ptr().add(4)),
                vld1q_f64(s.as_ptr().add(6)),
            )
        }
    }

    #[inline(always)]
    pub fn from_array(arr: [f64; 8]) -> Self {
        Self::from_slice(&arr)
    }

    #[inline(always)]
    pub fn to_array(self) -> [f64; 8] {
        let mut out = [0.0f64; 8];
        unsafe {
            vst1q_f64(out.as_mut_ptr(), self.0);
            vst1q_f64(out.as_mut_ptr().add(2), self.1);
            vst1q_f64(out.as_mut_ptr().add(4), self.2);
            vst1q_f64(out.as_mut_ptr().add(6), self.3);
        }
        out
    }

    #[inline(always)]
    pub fn copy_to_slice(self, s: &mut [f64]) {
        assert!(s.len() >= 8);
        unsafe {
            vst1q_f64(s.as_mut_ptr(), self.0);
            vst1q_f64(s.as_mut_ptr().add(2), self.1);
            vst1q_f64(s.as_mut_ptr().add(4), self.2);
            vst1q_f64(s.as_mut_ptr().add(6), self.3);
        }
    }

    #[inline(always)]
    pub fn reduce_sum(self) -> f64 {
        unsafe {
            let s01 = vaddq_f64(self.0, self.1);
            let s23 = vaddq_f64(self.2, self.3);
            let s = vaddq_f64(s01, s23);
            vaddvq_f64(s)
        }
    }

    #[inline(always)]
    pub fn reduce_min(self) -> f64 {
        unsafe {
            let m01 = vminq_f64(self.0, self.1);
            let m23 = vminq_f64(self.2, self.3);
            let m = vminq_f64(m01, m23);
            vminvq_f64(m)
        }
    }

    #[inline(always)]
    pub fn reduce_max(self) -> f64 {
        unsafe {
            let m01 = vmaxq_f64(self.0, self.1);
            let m23 = vmaxq_f64(self.2, self.3);
            let m = vmaxq_f64(m01, m23);
            vmaxvq_f64(m)
        }
    }

    #[inline(always)]
    pub fn simd_min(self, other: Self) -> Self {
        unsafe {
            Self(
                vminq_f64(self.0, other.0),
                vminq_f64(self.1, other.1),
                vminq_f64(self.2, other.2),
                vminq_f64(self.3, other.3),
            )
        }
    }

    #[inline(always)]
    pub fn simd_max(self, other: Self) -> Self {
        unsafe {
            Self(
                vmaxq_f64(self.0, other.0),
                vmaxq_f64(self.1, other.1),
                vmaxq_f64(self.2, other.2),
                vmaxq_f64(self.3, other.3),
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
                vfmaq_f64(c.0, self.0, b.0),
                vfmaq_f64(c.1, self.1, b.1),
                vfmaq_f64(c.2, self.2, b.2),
                vfmaq_f64(c.3, self.3, b.3),
            )
        }
    }

    #[inline(always)]
    pub fn sqrt(self) -> Self {
        unsafe {
            Self(
                vsqrtq_f64(self.0),
                vsqrtq_f64(self.1),
                vsqrtq_f64(self.2),
                vsqrtq_f64(self.3),
            )
        }
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        unsafe {
            Self(
                vabsq_f64(self.0),
                vabsq_f64(self.1),
                vabsq_f64(self.2),
                vabsq_f64(self.3),
            )
        }
    }
}

impl Add for F64x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        unsafe {
            Self(
                vaddq_f64(self.0, rhs.0),
                vaddq_f64(self.1, rhs.1),
                vaddq_f64(self.2, rhs.2),
                vaddq_f64(self.3, rhs.3),
            )
        }
    }
}

impl Sub for F64x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        unsafe {
            Self(
                vsubq_f64(self.0, rhs.0),
                vsubq_f64(self.1, rhs.1),
                vsubq_f64(self.2, rhs.2),
                vsubq_f64(self.3, rhs.3),
            )
        }
    }
}

impl Mul for F64x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        unsafe {
            Self(
                vmulq_f64(self.0, rhs.0),
                vmulq_f64(self.1, rhs.1),
                vmulq_f64(self.2, rhs.2),
                vmulq_f64(self.3, rhs.3),
            )
        }
    }
}

impl Div for F64x8 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        unsafe {
            Self(
                vdivq_f64(self.0, rhs.0),
                vdivq_f64(self.1, rhs.1),
                vdivq_f64(self.2, rhs.2),
                vdivq_f64(self.3, rhs.3),
            )
        }
    }
}

impl AddAssign for F64x8 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl SubAssign for F64x8 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}

impl MulAssign for F64x8 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
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
            Self(
                vnegq_f64(self.0),
                vnegq_f64(self.1),
                vnegq_f64(self.2),
                vnegq_f64(self.3),
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
