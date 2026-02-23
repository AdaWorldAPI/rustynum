//! XOR Delta Layer: borrow-free holographic storage.
//!
//! Ground truth is `&self` forever — readers never need `&mut`. Writers
//! own a private `DeltaLayer` containing their XOR delta from ground truth.
//!
//! ```text
//! effective_value = ground_truth ^ delta
//!
//! To write new_value:  delta = ground ^ new_value
//!                      ground ^ delta = ground ^ ground ^ new_value = new_value
//! ```
//!
//! No `RefCell`, no `UnsafeCell`, no runtime borrow checking.
//! XOR's self-inverse property makes the algebra handle isolation.

use crate::fingerprint::Fingerprint;

/// A single XOR delta overlay on immutable ground truth.
///
/// The ground truth is always borrowed as `&self` — no mutation needed.
/// The DeltaLayer owns its delta as a plain field — `&mut self` is just
/// regular Rust, no interior mutability tricks required.
pub struct DeltaLayer<const N: usize> {
    delta: Fingerprint<N>,
    writer_id: u32,
}

impl<const N: usize> DeltaLayer<N> {
    /// Create a clean (zero-delta) layer for the given writer.
    #[inline]
    pub fn new(writer_id: u32) -> Self {
        Self {
            delta: Fingerprint::zero(),
            writer_id,
        }
    }

    /// Read the effective value: `ground ^ delta`.
    #[inline]
    pub fn read(&self, ground: &Fingerprint<N>) -> Fingerprint<N> {
        ground ^ &self.delta
    }

    /// Write a new effective value.
    ///
    /// After this call, `self.read(ground)` will return `new_value`.
    ///
    /// Algebraically: `delta' = ground ^ new_value`, because
    /// `ground ^ delta' = ground ^ ground ^ new_value = new_value`.
    #[inline]
    pub fn write(&mut self, ground: &Fingerprint<N>, new_value: &Fingerprint<N>) {
        self.delta = ground ^ new_value;
    }

    /// Apply a targeted XOR patch to the effective value.
    ///
    /// `effective' = effective ^ patch`, implemented as `delta ^= patch`.
    #[inline]
    pub fn xor_patch(&mut self, patch: &Fingerprint<N>) {
        self.delta ^= patch;
    }

    /// Returns true if this layer has no changes (delta is zero).
    #[inline]
    pub fn is_clean(&self) -> bool {
        self.delta.is_zero()
    }

    /// Number of bits that differ from ground truth.
    #[inline]
    pub fn changed_bits(&self) -> u32 {
        self.delta.popcount()
    }

    /// Borrow the raw delta (for serialization, wire transfer, etc.).
    #[inline]
    pub fn delta(&self) -> &Fingerprint<N> {
        &self.delta
    }

    /// The writer ID that owns this layer.
    #[inline]
    pub fn writer_id(&self) -> u32 {
        self.writer_id
    }

    /// Conflict detection: do two deltas modify the same bits?
    /// O(N/64) via AND + popcount. SIMD-friendly.
    #[inline]
    pub fn conflicts_with(&self, other: &DeltaLayer<N>) -> bool {
        (&self.delta & &other.delta).popcount() > 0
    }

    /// Count the number of conflicting bits between two deltas.
    #[inline]
    pub fn conflict_bits(&self, other: &DeltaLayer<N>) -> u32 {
        (&self.delta & &other.delta).popcount()
    }

    /// Collapse this layer into the ground truth, producing a new ground truth.
    ///
    /// Consumes the layer — after collapse, create a fresh `DeltaLayer::new()`.
    #[inline]
    pub fn collapse(self, ground: &Fingerprint<N>) -> Fingerprint<N> {
        ground ^ &self.delta
    }
}

impl<const N: usize> Default for DeltaLayer<N> {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_fp() -> Fingerprint<4> {
        Fingerprint {
            words: [
                0xDEAD_BEEF_CAFE_BABE,
                0x1234_5678_9ABC_DEF0,
                0xFEDC_BA98_7654_3210,
                0x0123_4567_89AB_CDEF,
            ],
        }
    }

    fn other_fp() -> Fingerprint<4> {
        Fingerprint {
            words: [
                0x1111_1111_1111_1111,
                0x2222_2222_2222_2222,
                0x3333_3333_3333_3333,
                0x4444_4444_4444_4444,
            ],
        }
    }

    #[test]
    fn test_clean_layer_reads_ground() {
        let ground = sample_fp();
        let layer = DeltaLayer::<4>::new(0);
        assert_eq!(layer.read(&ground), ground);
        assert!(layer.is_clean());
    }

    #[test]
    fn test_write_then_read() {
        let ground = sample_fp();
        let new_val = other_fp();
        let mut layer = DeltaLayer::<4>::new(0);
        layer.write(&ground, &new_val);
        assert_eq!(layer.read(&ground), new_val);
        assert!(!layer.is_clean());
    }

    #[test]
    fn test_write_ground_is_clean() {
        let ground = sample_fp();
        let mut layer = DeltaLayer::<4>::new(0);
        layer.write(&ground, &ground);
        assert!(layer.is_clean());
    }

    #[test]
    fn test_xor_patch_self_inverse() {
        let ground = Fingerprint::<4>::zero();
        let mut layer = DeltaLayer::<4>::new(0);
        let patch = Fingerprint {
            words: [0xFF, 0, 0, 0],
        };
        layer.xor_patch(&patch);
        let result = layer.read(&ground);
        assert_eq!(result.words[0], 0xFF);
        // Patch again: self-inverse
        layer.xor_patch(&patch);
        assert!(layer.is_clean());
    }

    #[test]
    fn test_collapse() {
        let ground = sample_fp();
        let new_val = other_fp();
        let mut layer = DeltaLayer::<4>::new(0);
        layer.write(&ground, &new_val);
        let collapsed = layer.collapse(&ground);
        assert_eq!(collapsed, new_val);
    }

    #[test]
    fn test_no_conflict_independent_writes() {
        let ground = Fingerprint::<4>::zero();
        let mut a = DeltaLayer::<4>::new(0);
        let mut b = DeltaLayer::<4>::new(1);

        // Writer A sets bit 0
        let mut desired_a = Fingerprint::<4>::zero();
        desired_a.words[0] = 1;
        a.write(&ground, &desired_a);

        // Writer B sets bit 64
        let mut desired_b = Fingerprint::<4>::zero();
        desired_b.words[1] = 1;
        b.write(&ground, &desired_b);

        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn test_conflict_detected_overlapping_writes() {
        let ground = Fingerprint::<4>::zero();
        let mut a = DeltaLayer::<4>::new(0);
        let mut b = DeltaLayer::<4>::new(1);

        let mut desired = Fingerprint::<4>::zero();
        desired.words[0] = 1;
        a.write(&ground, &desired);
        b.write(&ground, &desired);

        assert!(a.conflicts_with(&b));
    }

    #[test]
    fn test_changed_bits() {
        let ground = Fingerprint::<2> {
            words: [0x00, 0x00],
        };
        let mut layer = DeltaLayer::<2>::new(0);
        let new_val = Fingerprint::<2> {
            words: [0xFF, 0x00],
        };
        layer.write(&ground, &new_val);
        assert_eq!(layer.changed_bits(), 8);
    }
}
