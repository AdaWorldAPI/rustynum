//! Plane: the i8 accumulator substrate for holographic cognition.
//!
//! One dimension of cognition. 16,384 bits of signal (L1 cache resident).
//! The i8 accumulator IS the ground truth. Everything else is derived:
//!   bits = sign(acc), alpha = |acc| > threshold, truth = alpha density.
//!
//! NaN is structurally impossible: i8 saturating arithmetic, no floats.
//! Width mismatch handled gracefully: compare on shorter prefix, alpha=0 on remainder.
//!
//! SIMD wired through: all bulk operations delegate to [`crate::simd`].

use crate::fingerprint::Fingerprint;
use crate::simd;
use std::cell::RefCell;

// ============================================================================
// Accumulator — 64-byte aligned for AVX-512
// ============================================================================

/// 16,384 i8 slots. 16 KB. Fits L1 cache. 64-byte aligned for AVX-512 loads.
#[repr(C, align(64))]
pub struct Acc16K {
    pub values: [i8; 16384],
}

impl Default for Acc16K {
    fn default() -> Self {
        Self {
            values: [0i8; 16384],
        }
    }
}

impl Clone for Acc16K {
    fn clone(&self) -> Self {
        let mut new = Self::default();
        new.values.copy_from_slice(&self.values);
        new
    }
}

// ============================================================================
// Plane — the core type
// ============================================================================

/// One dimension of cognition. 16,384 bits (256 × u64 words).
///
/// Standard size: `Plane` = 16 KB accumulator = L1 cache resident.
/// The i8 accumulator is the ONLY stored state. `bits` and `alpha` are cached
/// derivations recomputed lazily from `acc` when dirty.
///
/// NaN impossible by construction:
/// - Accumulator: i8 saturating arithmetic
/// - Truth: u16 scaled \[0, 65535\], not float
/// - Distance: enum with Measured or Incomparable, not f32
pub struct Plane {
    /// Raw i8 accumulator. 16384 positions. 64-byte aligned.
    acc: Box<Acc16K>,
    /// Cached data bits derived from sign(acc).
    bits: Fingerprint<256>,
    /// Cached alpha mask derived from |acc| > threshold.
    alpha: Fingerprint<256>,
    /// Whether bits/alpha cache needs refresh from acc.
    dirty: bool,
    /// How many encounters shaped this plane.
    encounters: u32,
}

impl Clone for Plane {
    fn clone(&self) -> Self {
        Self {
            acc: self.acc.clone(),
            bits: self.bits.clone(),
            alpha: self.alpha.clone(),
            dirty: self.dirty,
            encounters: self.encounters,
        }
    }
}

/// Total bits in a standard Plane.
pub const PLANE_BITS: usize = 16384;
/// Total bytes for the fingerprint view.
pub const PLANE_BYTES: usize = 2048;

impl Plane {
    /// Bits in this plane.
    pub const BITS: usize = PLANE_BITS;
    /// Bytes in the fingerprint representation.
    pub const BYTES: usize = PLANE_BYTES;

    /// New empty plane. No evidence. Maximum uncertainty.
    pub fn new() -> Self {
        Self {
            acc: Box::new(Acc16K::default()),
            bits: Fingerprint::zero(),
            alpha: Fingerprint::zero(),
            dirty: false,
            encounters: 0,
        }
    }

    /// Number of encounters that shaped this plane.
    #[inline]
    pub fn encounters(&self) -> u32 {
        self.encounters
    }

    /// Cached data bits (sign of accumulator). Refreshes cache if dirty.
    #[inline]
    pub fn bits(&mut self) -> &Fingerprint<256> {
        self.ensure_cache();
        &self.bits
    }

    /// Cached alpha mask (|acc| > threshold). Refreshes cache if dirty.
    #[inline]
    pub fn alpha(&mut self) -> &Fingerprint<256> {
        self.ensure_cache();
        &self.alpha
    }

    /// Non-mutable byte view of bits (assumes cache is fresh).
    pub(crate) fn bits_bytes_ref(&self) -> &[u8] {
        self.bits.as_bytes()
    }

    /// Non-mutable byte view of alpha (assumes cache is fresh).
    pub(crate) fn alpha_bytes_ref(&self) -> &[u8] {
        self.alpha.as_bytes()
    }

    // ========================================================================
    // Encounter — evidence arrives
    // ========================================================================

    /// Feed raw bit evidence into the accumulator.
    /// Each bit position: acc\[k\] += if bit_k set { +1 } else { -1 }, saturating.
    #[allow(clippy::needless_range_loop)] // k indexes both acc[] and bit_bytes[k/8]
    pub fn encounter_bits(&mut self, evidence: &Fingerprint<256>) {
        let bit_bytes = evidence.as_bytes();
        let acc = &mut self.acc.values;

        for k in 0..Self::BITS {
            let byte_idx = k / 8;
            let bit_idx = k % 8;
            let is_set = (bit_bytes[byte_idx] >> bit_idx) & 1 == 1;
            if is_set {
                acc[k] = acc[k].saturating_add(1);
            } else {
                acc[k] = acc[k].saturating_sub(1);
            }
        }

        self.encounters += 1;
        self.dirty = true;
    }

    /// Encounter toward another Plane's bits.
    /// acc[k] += other.bits()[k] ? +1 : -1
    /// This is the DreamerV3 STE gradient in integer form.
    pub fn encounter_toward(&mut self, other: &mut Plane) {
        other.ensure_cache();
        let fp = other.bits.clone();
        self.encounter_bits(&fp);
    }

    /// Encounter AWAY from another Plane's bits.
    /// acc[k] += other.bits()[k] ? -1 : +1
    /// Anti-learning: punish this pattern.
    pub fn encounter_away(&mut self, other: &mut Plane) {
        other.ensure_cache();
        let inverted = !&other.bits;
        self.encounter_bits(&inverted);
    }

    /// RL reward encounter: +reward reinforces, -reward punishes.
    /// reward_sign: >= 0 = encounter_toward, < 0 = encounter_away
    pub fn reward_encounter(&mut self, evidence: &mut Plane, reward_sign: i8) {
        if reward_sign >= 0 {
            self.encounter_toward(evidence);
        } else {
            self.encounter_away(evidence);
        }
    }

    /// Feed a blake3-expanded text encounter into the accumulator.
    /// Hashes text → 32 bytes, then XOR-folds/repeats to fill 16K bits.
    pub fn encounter(&mut self, text: &str) {
        let fp = Self::text_to_fingerprint(text);
        self.encounter_bits(&fp);
    }

    /// Expand text to a full 16K fingerprint via blake3 XOR-fold.
    /// Deterministic: same text → same fingerprint.
    fn text_to_fingerprint(text: &str) -> Fingerprint<256> {
        let hash = blake3::hash(text.as_bytes());
        let seed = hash.as_bytes();

        // Use blake3 in keyed mode to generate enough bytes.
        // 256 words × 8 bytes = 2048 bytes needed.
        let mut output = vec![0u8; PLANE_BYTES];
        let mut hasher = blake3::Hasher::new_keyed(seed);
        hasher.update(text.as_bytes());
        let mut reader = hasher.finalize_xof();
        reader.fill(&mut output);

        Fingerprint::from_bytes(&output)
    }

    // ========================================================================
    // Cache refresh
    // ========================================================================

    /// Recompute bits and alpha from the accumulator.
    #[allow(clippy::needless_range_loop)] // k indexes acc[], bits.words[k/64], alpha.words[k/64]
    pub(crate) fn ensure_cache(&mut self) {
        if !self.dirty {
            return;
        }

        let threshold = self.alpha_threshold();
        let acc = &self.acc.values;

        for k in 0..Self::BITS {
            let word = k / 64;
            let bit = k % 64;

            // Data: sign of accumulator (positive → 1, zero/negative → 0)
            if acc[k] > 0 {
                self.bits.words[word] |= 1u64 << bit;
            } else {
                self.bits.words[word] &= !(1u64 << bit);
            }

            // Alpha: magnitude above threshold
            if acc[k].unsigned_abs() > threshold {
                self.alpha.words[word] |= 1u64 << bit;
            } else {
                self.alpha.words[word] &= !(1u64 << bit);
            }
        }

        self.dirty = false;
    }

    /// Adaptive threshold for alpha. More encounters → higher bar.
    fn alpha_threshold(&self) -> u8 {
        match self.encounters {
            0..=1 => 0,
            2..=5 => self.encounters as u8 / 2,
            6..=20 => self.encounters as u8 * 2 / 5,
            _ => {
                // Integer square root approximation. No float.
                let isqrt = integer_sqrt(self.encounters);
                ((isqrt * 4) / 5).min(127) as u8
            }
        }
    }

    // ========================================================================
    // Distance — SIMD-accelerated, alpha-aware
    // ========================================================================

    /// Distance to another Plane. SIMD-accelerated. Alpha-aware.
    ///
    /// Never panics. Never returns NaN. Returns `Distance::Incomparable` if
    /// there's no shared alpha region.
    pub fn distance(&mut self, other: &mut Plane) -> Distance {
        self.ensure_cache();
        other.ensure_cache();

        distance_slices(
            self.bits_bytes_ref(),
            self.alpha_bytes_ref(),
            other.bits_bytes_ref(),
            other.alpha_bytes_ref(),
        )
    }

    // ========================================================================
    // Truth — integer NARS truth
    // ========================================================================

    /// NARS truth derived from accumulator state.
    /// Pure integer arithmetic. NaN impossible.
    pub fn truth(&mut self) -> Truth {
        self.ensure_cache();

        let total_bits = Self::BITS as u32;
        let defined = simd::popcount(self.alpha.as_bytes()) as u32;

        // bits AND alpha → positive defined bits
        let mut buf = vec![0u8; Self::BYTES];
        let bits_bytes = self.bits.as_bytes();
        let alpha_bytes = self.alpha.as_bytes();
        for i in 0..Self::BYTES {
            buf[i] = bits_bytes[i] & alpha_bytes[i];
        }
        let positive = simd::popcount(&buf) as u32;

        // Integer-only scaling. No division by zero.
        let frequency = if defined == 0 {
            32768u16 // no evidence → 0.5 (maximum uncertainty)
        } else {
            ((positive as u64 * 65535) / defined as u64) as u16
        };

        let confidence = ((defined as u64 * 65535) / total_bits as u64) as u16;

        Truth {
            frequency,
            confidence,
            evidence: self.encounters,
        }
    }

    /// Access the raw accumulator.
    #[inline]
    pub fn acc(&self) -> &[i8; 16384] {
        &self.acc.values
    }
}

impl Default for Plane {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Distance — the enum, not a float
// ============================================================================

/// Distance result. No floats. No NaN. Ever.
/// The caller computes a ratio if they need one.
#[derive(Debug, Clone, Copy)]
pub enum Distance {
    /// Enough shared alpha to compare.
    Measured {
        /// Bits that disagree on shared-alpha positions.
        disagreement: u32,
        /// Bits where both planes have alpha=1.
        overlap: u32,
        /// Bits where one side has alpha=0 (uncertainty penalty).
        penalty: u32,
    },
    /// Not enough shared alpha to compare meaningfully.
    /// This is NOT an error. It's honest: "I can't tell."
    Incomparable,
}

impl Distance {
    /// Normalized distance as f32. ONLY place float appears.
    /// Returns None if Incomparable. Never NaN.
    #[inline]
    pub fn normalized(&self) -> Option<f32> {
        match self {
            Distance::Measured {
                disagreement,
                overlap,
                penalty,
            } => {
                let denom = overlap + penalty;
                if denom == 0 {
                    return None;
                }
                Some((*disagreement + *penalty) as f32 / denom as f32)
            }
            Distance::Incomparable => None,
        }
    }

    /// Is this closer than a threshold? Pure integer comparison. No float.
    #[inline]
    pub fn closer_than(&self, max_disagreement: u32) -> bool {
        match self {
            Distance::Measured { disagreement, .. } => *disagreement <= max_disagreement,
            Distance::Incomparable => false,
        }
    }

    /// Raw disagreement count. None if incomparable.
    #[inline]
    pub fn raw(&self) -> Option<u32> {
        match self {
            Distance::Measured { disagreement, .. } => Some(*disagreement),
            Distance::Incomparable => None,
        }
    }
}

// ============================================================================
// Truth — integer NARS truth value
// ============================================================================

/// NARS truth from accumulator state. Integer arithmetic only.
/// Frequency and confidence are u16 scaled \[0, 65535\] = \[0.0, 1.0\].
/// No float. No NaN. No division by zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Truth {
    /// Frequency: positive defined bits / total defined bits.
    /// Scaled: 0 = never true, 65535 = always true.
    pub frequency: u16,
    /// Confidence: defined bits / total bits.
    /// Scaled: 0 = no evidence, 65535 = fully defined.
    pub confidence: u16,
    /// Raw encounter count.
    pub evidence: u32,
}

impl Truth {
    /// Frequency as f32 \[0.0, 1.0\]. Guaranteed finite (u16 / 65535).
    #[inline]
    pub fn frequency_f32(&self) -> f32 {
        self.frequency as f32 / 65535.0
    }

    /// Confidence as f32 \[0.0, 1.0\].
    #[inline]
    pub fn confidence_f32(&self) -> f32 {
        self.confidence as f32 / 65535.0
    }

    /// Expectation: c * (f - 0.5) + 0.5. Integer version, returns u16 scaled.
    #[inline]
    pub fn expectation(&self) -> u16 {
        let f = self.frequency as i32;
        let c = self.confidence as i32;
        let centered = f - 32768;
        let weighted = (c * centered) / 65535;
        (weighted + 32768).clamp(0, 65535) as u16
    }

    /// Revision: combine two independent evidence sources. Integer only.
    pub fn revision(&self, other: &Truth) -> Truth {
        let total_evidence = self.evidence.saturating_add(other.evidence);
        if total_evidence == 0 {
            return Truth {
                frequency: 32768,
                confidence: 0,
                evidence: 0,
            };
        }

        let f = ((self.frequency as u64 * self.evidence as u64)
            + (other.frequency as u64 * other.evidence as u64))
            / total_evidence as u64;
        let c = ((self.confidence as u64 * self.evidence as u64)
            + (other.confidence as u64 * other.evidence as u64))
            / total_evidence as u64;

        Truth {
            frequency: f.min(65535) as u16,
            confidence: c.min(65535) as u16,
            evidence: total_evidence,
        }
    }
}

// ============================================================================
// Free functions
// ============================================================================

/// Pre-allocated scratch buffers for distance computation. 64-byte aligned for AVX-512.
/// One per thread. Allocated once, reused every distance call. No zeroing needed
/// because every byte is overwritten before popcount.
#[repr(C, align(64))]
pub struct DistanceScratch {
    masked_xor: [u8; PLANE_BYTES],
    shared_alpha: [u8; PLANE_BYTES],
    not_alpha: [u8; PLANE_BYTES],
}

impl DistanceScratch {
    fn new() -> Self {
        Self {
            masked_xor: [0u8; PLANE_BYTES],
            shared_alpha: [0u8; PLANE_BYTES],
            not_alpha: [0u8; PLANE_BYTES],
        }
    }
}

thread_local! {
    static SCRATCH: RefCell<DistanceScratch> = RefCell::new(DistanceScratch::new());
}

/// Core distance computation on raw byte slices. Handles any width.
/// Used by Plane::distance and Node::distance.
///
/// Zero-allocation: uses thread-local 64-byte-aligned scratch buffers.
/// XOR+AND+NOT into pre-warmed cache lines, then SIMD popcount via VPOPCNTDQ.
pub fn distance_slices(a_bits: &[u8], a_alpha: &[u8], b_bits: &[u8], b_alpha: &[u8]) -> Distance {
    let shared_len = a_bits
        .len()
        .min(b_bits.len())
        .min(a_alpha.len())
        .min(b_alpha.len());

    if shared_len == 0 {
        return Distance::Incomparable;
    }

    let a = &a_bits[..shared_len];
    let b = &b_bits[..shared_len];
    let aa = &a_alpha[..shared_len];
    let ba = &b_alpha[..shared_len];

    let (disagreement, overlap, penalty) = if shared_len <= PLANE_BYTES {
        // Fast path: thread-local scratch. No allocation. No zeroing.
        // Buffers are 64-byte aligned and cache-hot from the last call.
        SCRATCH.with(|cell| {
            let scratch = &mut *cell.borrow_mut();

            for i in 0..shared_len {
                let xor = a[i] ^ b[i];
                scratch.shared_alpha[i] = aa[i] & ba[i];
                scratch.masked_xor[i] = xor & scratch.shared_alpha[i];
                scratch.not_alpha[i] = !aa[i];
            }

            // SIMD popcount via simd.rs — AVX-512 VPOPCNTDQ → AVX2 → scalar.
            (
                simd::popcount(&scratch.masked_xor[..shared_len]) as u32,
                simd::popcount(&scratch.shared_alpha[..shared_len]) as u32,
                simd::popcount(&scratch.not_alpha[..shared_len]) as u32,
            )
        })
    } else {
        // Fallback for oversized slices: heap allocation.
        let mut shared_alpha_buf = vec![0u8; shared_len];
        let mut masked_xor_buf = vec![0u8; shared_len];
        let mut not_alpha_buf = vec![0u8; shared_len];

        for i in 0..shared_len {
            let xor = a[i] ^ b[i];
            shared_alpha_buf[i] = aa[i] & ba[i];
            masked_xor_buf[i] = xor & shared_alpha_buf[i];
            not_alpha_buf[i] = !aa[i];
        }

        (
            simd::popcount(&masked_xor_buf) as u32,
            simd::popcount(&shared_alpha_buf) as u32,
            simd::popcount(&not_alpha_buf) as u32,
        )
    };

    // Penalty for width mismatch region (wider plane's extra bits).
    let extra_bits = (a_bits.len().max(b_bits.len()) - shared_len) * 8;
    let total_penalty = penalty + extra_bits as u32;

    if overlap == 0 {
        return Distance::Incomparable;
    }

    Distance::Measured {
        disagreement,
        overlap,
        penalty: total_penalty,
    }
}

/// Integer square root via Newton's method. No float.
#[inline]
fn integer_sqrt(n: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_new_is_empty() {
        let p = Plane::new();
        assert_eq!(p.encounters(), 0);
        assert!(p.bits.is_zero());
        assert!(p.alpha.is_zero());
    }

    #[test]
    fn plane_encounter_builds_signal() {
        let mut p = Plane::new();
        p.encounter("hello");
        p.encounter("hello");
        p.encounter("hello");
        let t = p.truth();
        assert!(t.confidence > 0); // three encounters → some bits defined
        assert_eq!(t.evidence, 3);
    }

    #[test]
    fn plane_nan_impossible() {
        let mut empty = Plane::new();
        let t = empty.truth();
        assert_eq!(t.frequency, 32768); // 0.5 = maximum uncertainty
        assert_eq!(t.confidence, 0); // no evidence

        let mut other = Plane::new();
        let d = empty.distance(&mut other);
        assert!(matches!(d, Distance::Incomparable)); // not NaN, not panic
    }

    #[test]
    fn plane_encounter_bits_direct() {
        let mut p = Plane::new();
        let all_ones = Fingerprint::<256>::ones();
        p.encounter_bits(&all_ones);
        p.encounter_bits(&all_ones);
        let t = p.truth();
        assert_eq!(t.evidence, 2);
        // All bits positive with high confidence after 2 encounters of all-ones
        assert!(t.frequency > 32768);
    }

    #[test]
    fn distance_measured_between_similar_planes() {
        let mut a = Plane::new();
        let mut b = Plane::new();
        // Same evidence → should be close
        a.encounter("hello world");
        a.encounter("hello world");
        a.encounter("hello world");
        b.encounter("hello world");
        b.encounter("hello world");
        b.encounter("hello world");

        let d = a.distance(&mut b);
        match d {
            Distance::Measured {
                disagreement,
                overlap,
                ..
            } => {
                assert!(overlap > 0);
                assert_eq!(disagreement, 0); // same input → identical
            }
            Distance::Incomparable => panic!("expected Measured"),
        }
    }

    #[test]
    fn distance_closer_than() {
        let d = Distance::Measured {
            disagreement: 100,
            overlap: 8000,
            penalty: 200,
        };
        assert!(d.closer_than(100));
        assert!(d.closer_than(200));
        assert!(!d.closer_than(50));

        assert!(!Distance::Incomparable.closer_than(100));
    }

    #[test]
    fn distance_normalized() {
        let d = Distance::Measured {
            disagreement: 100,
            overlap: 900,
            penalty: 100,
        };
        let n = d.normalized().unwrap();
        // (100 + 100) / (900 + 100) = 0.2
        assert!((n - 0.2).abs() < 0.001);

        assert!(Distance::Incomparable.normalized().is_none());
    }

    #[test]
    fn truth_revision_integer_only() {
        let t1 = Truth {
            frequency: 60000,
            confidence: 50000,
            evidence: 10,
        };
        let t2 = Truth {
            frequency: 30000,
            confidence: 40000,
            evidence: 5,
        };
        let revised = t1.revision(&t2);
        // Weighted average: (60000*10 + 30000*5) / 15 = 50000
        assert_eq!(revised.frequency, 50000);
        assert_eq!(revised.evidence, 15);
    }

    #[test]
    fn truth_expectation_no_confidence() {
        let t = Truth {
            frequency: 60000,
            confidence: 0,
            evidence: 0,
        };
        assert_eq!(t.expectation(), 32768); // no confidence → 0.5
    }

    #[test]
    fn truth_expectation_full_confidence() {
        let t = Truth {
            frequency: 65535,
            confidence: 65535,
            evidence: 100,
        };
        // c=1.0, f=1.0 → expectation = 1.0 → 65535
        assert!(t.expectation() >= 65534); // allow rounding
    }

    #[test]
    fn integer_sqrt_correct() {
        assert_eq!(integer_sqrt(0), 0);
        assert_eq!(integer_sqrt(1), 1);
        assert_eq!(integer_sqrt(4), 2);
        assert_eq!(integer_sqrt(9), 3);
        assert_eq!(integer_sqrt(100), 10);
        assert_eq!(integer_sqrt(101), 10);
    }

    #[test]
    fn encounter_toward_reinforces() {
        let mut learner = Plane::new();
        let mut teacher = Plane::new();
        teacher.encounter("pattern A");
        teacher.encounter("pattern A");
        teacher.encounter("pattern A");

        // Learn toward the teacher 3 times
        learner.encounter_toward(&mut teacher);
        learner.encounter_toward(&mut teacher);
        learner.encounter_toward(&mut teacher);

        let d = learner.distance(&mut teacher);
        match d {
            Distance::Measured { disagreement, overlap, .. } => {
                assert!(overlap > 0);
                assert_eq!(disagreement, 0, "encounter_toward should match teacher bits");
            }
            Distance::Incomparable => panic!("expected Measured after encounter_toward"),
        }
    }

    #[test]
    fn encounter_away_punishes() {
        let mut learner = Plane::new();
        let mut target = Plane::new();
        target.encounter("pattern B");
        target.encounter("pattern B");
        target.encounter("pattern B");

        // Learn away from target 3 times
        learner.encounter_away(&mut target);
        learner.encounter_away(&mut target);
        learner.encounter_away(&mut target);

        let d = learner.distance(&mut target);
        match d {
            Distance::Measured { disagreement, overlap, .. } => {
                assert!(overlap > 0);
                // encounter_away inverts: all shared bits should disagree
                assert_eq!(disagreement, overlap, "encounter_away should maximally disagree");
            }
            Distance::Incomparable => panic!("expected Measured after encounter_away"),
        }
    }

    #[test]
    fn reward_encounter_positive_reinforces() {
        let mut learner = Plane::new();
        let mut evidence = Plane::new();
        evidence.encounter("reward signal");
        evidence.encounter("reward signal");
        evidence.encounter("reward signal");

        learner.reward_encounter(&mut evidence, 1);
        learner.reward_encounter(&mut evidence, 1);
        learner.reward_encounter(&mut evidence, 1);

        let d = learner.distance(&mut evidence);
        match d {
            Distance::Measured { disagreement, .. } => {
                assert_eq!(disagreement, 0, "positive reward should reinforce");
            }
            _ => panic!("expected Measured"),
        }
    }

    #[test]
    fn reward_encounter_negative_punishes() {
        let mut learner = Plane::new();
        let mut evidence = Plane::new();
        evidence.encounter("bad pattern");
        evidence.encounter("bad pattern");
        evidence.encounter("bad pattern");

        learner.reward_encounter(&mut evidence, -1);
        learner.reward_encounter(&mut evidence, -1);
        learner.reward_encounter(&mut evidence, -1);

        let d = learner.distance(&mut evidence);
        match d {
            Distance::Measured { disagreement, overlap, .. } => {
                assert_eq!(disagreement, overlap, "negative reward should punish");
            }
            _ => panic!("expected Measured"),
        }
    }

    #[test]
    fn encounter_toward_away_cancel() {
        let mut learner = Plane::new();
        let mut target = Plane::new();
        target.encounter("cancel test");
        target.encounter("cancel test");
        target.encounter("cancel test");

        // One toward, one away → should roughly cancel
        learner.encounter_toward(&mut target);
        learner.encounter_away(&mut target);

        // Accumulator should be near zero (all +1 then -1)
        let acc = learner.acc();
        let sum: i64 = acc.iter().map(|&v| v as i64).sum();
        assert_eq!(sum, 0, "toward + away should cancel to zero accumulator");
    }
}
