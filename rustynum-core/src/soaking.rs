//! Int8 soaking layer: dot-product kernel, binary↔int8 bridge, attention mask.
//!
//! The soaking layer is the TRANSIENT representation — int8 10000D vectors
//! that accumulate evidence before crystallizing to permanent binary storage.
//!
//! This module provides:
//! - **`dot_i8_10k`**: int8 dot product for 10000D vectors (scalar, SIMD-dispatched)
//! - **`binary_to_int8`**: expand binary fingerprint bits to ±1 int8 vector
//! - **`int8_to_binary`**: sign(int8) → binary fingerprint (crystallization)
//! - **`AttentionMask`**: σ-2/3 focus lens with project/classify
//!
//! # Zero IO
//!
//! All functions are pure compute. Takes `&[i8]` or `&[u8]` slices.
//! Never allocates beyond return values. Never does IO.

use crate::fingerprint::Fingerprint;

/// Default soaking dimension (from VSA literature, Kanerva 2009).
pub const SOAKING_DIM: usize = 10_000;

// ---------------------------------------------------------------------------
// Int8 dot product
// ---------------------------------------------------------------------------

/// Scalar int8 dot product for arbitrary-length vectors.
///
/// Accumulates in i64 to avoid overflow. For 10000 dimensions with
/// i8 × i8 (max 127²=16129), worst case = 10000 × 16129 = 161M,
/// well within i64 range.
#[inline]
pub fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i64 {
    assert_eq!(a.len(), b.len(), "dot product requires equal-length vectors");
    let mut acc: i64 = 0;
    for i in 0..a.len() {
        acc += a[i] as i64 * b[i] as i64;
    }
    acc
}

/// Normalized dot product: dot / dimension count.
/// Returns f32 for the projection score.
#[inline]
pub fn dot_i8_normalized(a: &[i8], b: &[i8]) -> f32 {
    let dot = dot_i8_scalar(a, b);
    dot as f32 / a.len() as f32
}

// ---------------------------------------------------------------------------
// Binary ↔ Int8 bridge
// ---------------------------------------------------------------------------

/// Expand a binary fingerprint to int8: each bit → +1 or -1.
///
/// Output length = N × 64 (total bits in the fingerprint).
/// bit=1 → +1, bit=0 → -1.
///
/// This creates the int8 representation from crystallized binary,
/// suitable for attention mask projection and soaking initialization.
pub fn binary_to_int8<const N: usize>(fp: &Fingerprint<N>) -> Vec<i8> {
    let total = N * 64;
    let mut out = Vec::with_capacity(total);
    for word_idx in 0..N {
        let word = fp.words[word_idx];
        for bit_idx in 0..64 {
            if (word >> bit_idx) & 1 == 1 {
                out.push(1);
            } else {
                out.push(-1);
            }
        }
    }
    out
}

/// Crystallize int8 soaking register to binary fingerprint: sign(i8) → bit.
///
/// - Positive values → bit = 1
/// - Zero and negative values → bit = 0
///
/// `values.len()` can differ from N×64. If shorter, remaining bits are 0.
/// If longer, excess values are ignored.
pub fn int8_to_binary<const N: usize>(values: &[i8]) -> Fingerprint<N> {
    let mut fp = Fingerprint::<N>::zero();
    let max_bits = (N * 64).min(values.len());
    for (i, &val) in values.iter().enumerate().take(max_bits) {
        if val > 0 {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            fp.words[word_idx] |= 1u64 << bit_idx;
        }
    }
    fp
}

/// Expand binary fingerprint to int8 with a specified output dimension.
///
/// If `dim` > total bits, extra positions are filled with 0 (unknown).
/// If `dim` < total bits, the vector is truncated.
pub fn binary_to_int8_dim<const N: usize>(fp: &Fingerprint<N>, dim: usize) -> Vec<i8> {
    let total_bits = N * 64;
    let mut out = Vec::with_capacity(dim);

    for i in 0..dim {
        if i < total_bits {
            let word_idx = i / 64;
            let bit_idx = i % 64;
            if (fp.words[word_idx] >> bit_idx) & 1 == 1 {
                out.push(1);
            } else {
                out.push(-1);
            }
        } else {
            out.push(0); // beyond fingerprint range: unknown
        }
    }
    out
}

// ---------------------------------------------------------------------------
// AttentionMask
// ---------------------------------------------------------------------------

/// Classification of how a new concept relates to the attention mask.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttentionResult {
    /// New concept resonates with known concepts (projection > threshold).
    Resonance(f32),
    /// New concept is genuinely novel (projection near zero).
    Novel(f32),
    /// New concept contradicts known concepts (projection < -threshold).
    Conflict(f32),
}

/// The σ-2/3 attention mask: a focus lens formed by known concepts.
///
/// A weighted bundle of all σ-2/3 concepts, used to project incoming
/// concepts and classify them as resonance, novel, or conflict.
///
/// The mask is a fixed-size int8 vector (default 10000D).
pub struct AttentionMask {
    /// The mask vector: weighted bundle of known concept vectors.
    mask: Vec<i8>,
    /// Number of concepts that contributed to this mask.
    pub concept_count: u32,
    /// Minimum σ-band included (2=HINT, 3=KNOWN).
    pub min_sigma: u8,
    /// Monotonic version counter for cache invalidation.
    pub version: u64,
    /// Threshold for resonance/conflict classification.
    threshold: f32,
}

impl AttentionMask {
    /// Create a new empty attention mask.
    pub fn new(dim: usize, min_sigma: u8) -> Self {
        Self {
            mask: vec![0i8; dim],
            concept_count: 0,
            min_sigma,
            version: 0,
            threshold: 0.3,
        }
    }

    /// Create with a custom classification threshold.
    pub fn with_threshold(dim: usize, min_sigma: u8, threshold: f32) -> Self {
        Self {
            mask: vec![0i8; dim],
            concept_count: 0,
            min_sigma,
            version: 0,
            threshold,
        }
    }

    /// Get the mask dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.mask.len()
    }

    /// Get the raw mask vector.
    #[inline]
    pub fn as_slice(&self) -> &[i8] {
        &self.mask
    }

    /// Clear the mask (reset to zero).
    pub fn clear(&mut self) {
        self.mask.fill(0);
        self.concept_count = 0;
        self.version += 1;
    }

    /// Add a concept vector to the mask via saturating add.
    ///
    /// `concept` is the int8 vector of the concept (from soaking or expanded binary).
    /// `weight` scales the contribution (typically NARS confidence × recency).
    pub fn add_concept(&mut self, concept: &[i8], weight: f32) {
        assert_eq!(concept.len(), self.mask.len(), "concept dim must match mask dim");
        for (m, &c) in self.mask.iter_mut().zip(concept.iter()) {
            let contribution = (c as f32 * weight).round() as i16;
            let new_val = (*m as i16 + contribution).clamp(-128, 127);
            *m = new_val as i8;
        }
        self.concept_count += 1;
        self.version += 1;
    }

    /// Project a new concept onto the mask.
    ///
    /// Returns normalized dot product: dot(concept, mask) / dim.
    /// - Positive → resonance (relates to known)
    /// - Near zero → novel (orthogonal to known)
    /// - Negative → conflict (contradicts known)
    pub fn project(&self, concept: &[i8]) -> f32 {
        assert_eq!(concept.len(), self.mask.len(), "concept dim must match mask dim");
        dot_i8_normalized(concept, &self.mask)
    }

    /// Classify a projection score.
    pub fn classify(&self, projection: f32) -> AttentionResult {
        if projection > self.threshold {
            AttentionResult::Resonance(projection)
        } else if projection < -self.threshold {
            AttentionResult::Conflict(projection)
        } else {
            AttentionResult::Novel(projection)
        }
    }

    /// Project and classify in one step.
    pub fn evaluate(&self, concept: &[i8]) -> AttentionResult {
        let proj = self.project(concept);
        self.classify(proj)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_i8_scalar_basic() {
        let a = [1i8, 2, 3, 4, 5];
        let b = [5i8, 4, 3, 2, 1];
        assert_eq!(dot_i8_scalar(&a, &b), 5 + 8 + 9 + 8 + 5);
    }

    #[test]
    fn test_dot_i8_scalar_negative() {
        let a = [1i8, -1, 1, -1];
        let b = [1i8, 1, -1, -1];
        assert_eq!(dot_i8_scalar(&a, &b), 1 - 1 - 1 + 1);
    }

    #[test]
    fn test_dot_i8_scalar_orthogonal() {
        // Two nearly orthogonal random vectors should have dot ≈ 0
        let a = [1i8, -1, 1, -1, 1, -1, 1, -1];
        let b = [1i8, 1, -1, -1, 1, 1, -1, -1];
        assert_eq!(dot_i8_scalar(&a, &b), 0);
    }

    #[test]
    fn test_binary_to_int8_roundtrip() {
        let fp = Fingerprint::<2> {
            words: [0xFF00FF00, 0x0F0F0F0F],
        };
        let int8_vec = binary_to_int8(&fp);
        assert_eq!(int8_vec.len(), 128); // 2 × 64

        // First 8 bits of 0xFF00FF00: bits 0-7 = 0x00 = 00000000
        // Wait, 0xFF00FF00 in binary, bit 0 (LSB) = 0
        // 0xFF00FF00 = 1111_1111_0000_0000_1111_1111_0000_0000
        // bit 0 = 0, bit 1 = 0, ..., bit 7 = 0
        // bit 8 = 1, bit 9 = 1, ..., bit 15 = 1
        assert_eq!(int8_vec[0], -1); // bit 0 = 0 → -1
        assert_eq!(int8_vec[8], 1);  // bit 8 = 1 → +1

        // Crystallize back
        let fp2: Fingerprint<2> = int8_to_binary(&int8_vec);
        assert_eq!(fp, fp2);
    }

    #[test]
    fn test_int8_to_binary_basic() {
        let values = [1i8, -1, 1, -1, 0, 1, -1, 1];
        // Expected: bits 0,2,5,7 = 1 → 0b10100101 = 0xA5
        let fp: Fingerprint<1> = int8_to_binary(&values);
        assert_eq!(fp.words[0] & 0xFF, 0b10100101);
    }

    #[test]
    fn test_int8_to_binary_zero_stays_zero() {
        let values = [0i8; 64];
        let fp: Fingerprint<1> = int8_to_binary(&values);
        assert!(fp.is_zero());
    }

    #[test]
    fn test_binary_to_int8_dim_extension() {
        let fp = Fingerprint::<1> { words: [0xFF] };
        let int8_vec = binary_to_int8_dim(&fp, 100);
        assert_eq!(int8_vec.len(), 100);
        // First 8 bits are 1 → +1
        for i in 0..8 {
            assert_eq!(int8_vec[i], 1, "bit {i} should be +1");
        }
        // Bits 8-63 are 0 → -1
        for i in 8..64 {
            assert_eq!(int8_vec[i], -1, "bit {i} should be -1");
        }
        // Beyond 64: filled with 0 (unknown)
        for i in 64..100 {
            assert_eq!(int8_vec[i], 0, "beyond-range bit {i} should be 0");
        }
    }

    #[test]
    fn test_attention_mask_empty() {
        let mask = AttentionMask::new(100, 2);
        assert_eq!(mask.dim(), 100);
        assert_eq!(mask.concept_count, 0);
        assert_eq!(mask.version, 0);
    }

    #[test]
    fn test_attention_mask_add_and_project() {
        let mut mask = AttentionMask::new(10, 2);

        // Add a concept: all +1
        let concept = vec![1i8; 10];
        mask.add_concept(&concept, 1.0);
        assert_eq!(mask.concept_count, 1);
        assert_eq!(mask.version, 1);

        // Project the same concept → should be high positive
        let proj = mask.project(&concept);
        assert!(proj > 0.0, "same concept should have positive projection");

        // Project opposite concept → should be negative
        let opposite = vec![-1i8; 10];
        let proj_neg = mask.project(&opposite);
        assert!(proj_neg < 0.0, "opposite concept should have negative projection");
    }

    #[test]
    fn test_attention_mask_classify() {
        let mask = AttentionMask::new(10, 2);
        assert!(matches!(mask.classify(0.5), AttentionResult::Resonance(_)));
        assert!(matches!(mask.classify(0.0), AttentionResult::Novel(_)));
        assert!(matches!(mask.classify(-0.5), AttentionResult::Conflict(_)));
    }

    #[test]
    fn test_attention_mask_clear() {
        let mut mask = AttentionMask::new(10, 2);
        mask.add_concept(&vec![1i8; 10], 1.0);
        assert_eq!(mask.concept_count, 1);
        mask.clear();
        assert_eq!(mask.concept_count, 0);
        assert!(mask.as_slice().iter().all(|&v| v == 0));
    }

    #[test]
    fn test_attention_mask_evaluate() {
        let mut mask = AttentionMask::new(10, 2);
        let concept = vec![1i8; 10];
        mask.add_concept(&concept, 1.0);
        let result = mask.evaluate(&concept);
        assert!(matches!(result, AttentionResult::Resonance(_)));
    }

    #[test]
    fn test_dot_normalized() {
        let a = [1i8, 1, 1, 1];
        let b = [1i8, 1, 1, 1];
        let d = dot_i8_normalized(&a, &b);
        assert!((d - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_soaking_dim_constant() {
        assert_eq!(SOAKING_DIM, 10_000);
    }
}
