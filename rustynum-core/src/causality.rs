//! Causality decomposition: RGB (causing) ↔ CMYK (experiencing) from BF16 sign patterns.
//!
//! The 16 phenomenological dimensions in the qualia corpus have a subset that
//! flips sign between causing and experiencing: **warmth**, **social**, **sacredness**.
//! These are the "causality axis." When all three flip sign in a BF16 comparison,
//! the agent transitions from experiencing (CMYK/absorbing) to causing (RGB/emitting)
//! or vice versa.
//!
//! The BF16 sign bit (weight 256) is already the strongest signal in the distance
//! metric. This module extracts the *meaning* of that signal: not just "these differ"
//! but "the agent switched from receiving warmth to projecting coldness."
//!
//! # NARS Truth Values
//!
//! The awareness substrate (Crystallized/Tensioned/Uncertain/Noise) maps directly
//! to NARS `<frequency, confidence>`:
//!
//! | Awareness | NARS | Meaning |
//! |-----------|------|---------|
//! | Crystallized% | frequency | How much is settled (positive evidence) |
//! | 1 - Noise% | confidence | How much is signal (vs irrelevant noise) |
//! | Tensioned% | 1 - frequency | Active contradiction (negative evidence) |
//!
//! # SPO Integration
//!
//! When encoding a qualia SPO triple:
//! - Subject (X-axis): WHO — the agent's identity fingerprint
//! - Predicate (Y-axis): DOES WHAT — **CAUSES** (RGB) or **EXPERIENCES** (CMYK)
//! - Object (Z-axis): TO WHOM/WHAT — the target qualia coordinate
//!
//! The predicate axis carries the causality direction. The BF16 sign flip on
//! causality dimensions of the Y-axis IS the difference between
//! `(Agent, CAUSES, Grief)` and `(Agent, EXPERIENCES, Grief)`.
//!
//! The NARS truth value on the Y-axis tells you confidence in the causality direction.
//! High Tensioned% on Y = "uncertain whether causing or experiencing" = low frequency.

use crate::bf16_hamming::{PackedQualia, SuperpositionState};
#[cfg(test)]
use crate::bf16_hamming::AwarenessState;
use crate::spatial_resonance::{CrystalAxis, SpatialCrystal3D};

// ============================================================================
// Qualia dimension indices (match PackedQualia.resonance[0..16])
// ============================================================================

/// Named indices into the 16-dimensional qualia vector.
///
/// These match the order in `qualia_219.json` vector fields and
/// `PackedQualia.resonance[0..16]` (after Nib4 encoding).
pub mod qualia_dim {
    pub const BRIGHTNESS: usize = 0;
    pub const VALENCE: usize = 1;
    pub const DOMINANCE: usize = 2;
    pub const AROUSAL: usize = 3;
    pub const WARMTH: usize = 4;
    pub const CLARITY: usize = 5;
    pub const SOCIAL: usize = 6;
    pub const NOSTALGIA: usize = 7;
    pub const SACREDNESS: usize = 8;
    pub const DESIRE: usize = 9;
    pub const TENSION: usize = 10;
    pub const AWE: usize = 11;
    pub const GRIEF: usize = 12;
    pub const HOPE: usize = 13;
    pub const EDGE: usize = 14;
    pub const RESOLUTION_HUNGER: usize = 15;
}

/// Dimensions that flip sign between RGB (causing) and CMYK (experiencing).
///
/// These are the "causality axis" — sign reversal on ALL of these means
/// the agent switched from experiencing to causing (or vice versa).
///
/// - **warmth**: positive=receiving warmth(CMYK), negative=cold/projecting(RGB)
/// - **social**: positive=communal(CMYK), negative=antisocial/imposing(RGB)
/// - **sacredness**: positive=receiving sacred(CMYK), negative=profane/violating(RGB)
pub const CAUSALITY_DIMS: [usize; 3] = [
    qualia_dim::WARMTH,
    qualia_dim::SOCIAL,
    qualia_dim::SACREDNESS,
];

// ============================================================================
// CausalityDirection
// ============================================================================

/// Whether the agent is causing (RGB/emitting) or experiencing (CMYK/absorbing).
///
/// In the qualia corpus, the `shame` field (17th dimension, intensity bit)
/// encodes this directly: shame ≤ 0 → RGB (causing), shame > 0 → CMYK (caused).
/// At the BF16 level, the sign pattern on causality dimensions gives the same
/// signal without needing the explicit shame field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CausalityDirection {
    /// RGB: agent is causing/emitting the feeling.
    /// Warmth < 0, social < 0, sacredness < 0 (cold, antisocial, profane).
    Causing,
    /// CMYK: agent is experiencing/absorbing the feeling.
    /// Warmth ≥ 0, social ≥ 0, sacredness ≥ 0 (warm, communal, sacred).
    Experiencing,
}

impl CausalityDirection {
    /// Detect causality direction from a PackedQualia's resonance shape.
    ///
    /// Majority vote across causality dimensions:
    /// if 2+ of warmth/social/sacredness are negative → Causing (RGB).
    pub fn from_qualia(q: &PackedQualia) -> Self {
        let negative_count = CAUSALITY_DIMS
            .iter()
            .filter(|&&dim| q.resonance[dim] < 0)
            .count();
        if negative_count >= 2 {
            CausalityDirection::Causing
        } else {
            CausalityDirection::Experiencing
        }
    }

    /// Flip: Causing ↔ Experiencing.
    #[inline]
    pub fn flip(self) -> Self {
        match self {
            Self::Causing => Self::Experiencing,
            Self::Experiencing => Self::Causing,
        }
    }
}

// ============================================================================
// NARS Truth Value
// ============================================================================

/// NARS truth value: `<frequency, confidence>`.
///
/// Derived from the BF16 awareness substrate:
/// - **frequency** = crystallized% (settled knowledge / total evidence)
/// - **confidence** = 1 - noise% (meaningful signal / total signal)
///
/// The revision rule for two truth values <f1,c1> and <f2,c2>:
/// ```text
/// f_revised = (f1 × c1 + f2 × c2) / (c1 + c2)
/// c_revised = (c1 + c2) / (c1 + c2 + k)   where k = evidential horizon
/// ```
///
/// rustynum computes the truth values; ladybug-rs applies the revision rule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NarsTruthValue {
    /// Frequency: proportion of evidence that is positive (0.0..1.0).
    /// Maps to Crystallized%: how much is settled agreement.
    pub frequency: f32,
    /// Confidence: strength of evidence (0.0..1.0).
    /// Maps to 1 - Noise%: how much of the signal is meaningful.
    pub confidence: f32,
}

impl NarsTruthValue {
    /// Create from raw values, clamping to [0, 1].
    pub fn new(frequency: f32, confidence: f32) -> Self {
        Self {
            frequency: frequency.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Extract NARS truth from a SuperpositionState (awareness substrate).
    ///
    /// - frequency = crystallized% (settled = positive evidence)
    /// - confidence = 1 - noise% (signal = meaningful evidence)
    pub fn from_awareness(awareness: &SuperpositionState) -> Self {
        Self {
            frequency: awareness.crystallized_pct,
            confidence: 1.0 - awareness.noise_pct,
        }
    }

    /// Uninformative prior: <0.5, 0.0> (no evidence).
    pub fn ignorance() -> Self {
        Self {
            frequency: 0.5,
            confidence: 0.0,
        }
    }

    /// Expectation: f × c + 0.5 × (1 - c).
    /// The best single-number summary of a NARS truth value.
    #[inline]
    pub fn expectation(self) -> f32 {
        self.frequency * self.confidence + 0.5 * (1.0 - self.confidence)
    }
}

// ============================================================================
// CausalityDecomposition
// ============================================================================

/// Result of decomposing BF16 distance into causality components.
///
/// Separates sign flips into:
/// - **causality flips**: on warmth/social/sacredness (the direction switch)
/// - **content flips**: on all other dimensions (what changed, not the direction)
///
/// This is the "soul coordinate" insight: the BF16 sign bit (weight 256) carries
/// two kinds of information. On causality dims, it means "who is doing this."
/// On content dims, it means "what is being done."
#[derive(Clone, Debug)]
pub struct CausalityDecomposition {
    /// Sign flips on causality dimensions (warmth, social, sacredness).
    pub causality_flips: usize,
    /// Total causality dimensions checked (always 3).
    pub causality_dims_total: usize,
    /// Sign flips on non-causality dimensions (content change).
    pub content_flips: usize,
    /// Total non-causality dimensions (always 13).
    pub content_dims_total: usize,
    /// Causality direction of source (A).
    pub source_direction: CausalityDirection,
    /// Causality direction of target (B).
    pub target_direction: CausalityDirection,
    /// Whether a full causality reversal occurred (RGB↔CMYK).
    pub reversed: bool,
    /// NARS truth value for this comparison (from awareness if available).
    pub nars_truth: NarsTruthValue,
}

/// Decompose the difference between two PackedQualia into causality components.
///
/// Separates sign patterns into "who is doing this" (causality dims) and
/// "what is being done" (content dims). Optionally enriches with NARS truth
/// from the awareness substrate.
pub fn causality_decompose(
    a: &PackedQualia,
    b: &PackedQualia,
    awareness: Option<&SuperpositionState>,
) -> CausalityDecomposition {
    let mut causality_flips = 0;
    let mut content_flips = 0;

    for dim in 0..16 {
        let a_neg = a.resonance[dim] < 0;
        let b_neg = b.resonance[dim] < 0;
        if a_neg != b_neg {
            if CAUSALITY_DIMS.contains(&dim) {
                causality_flips += 1;
            } else {
                content_flips += 1;
            }
        }
    }

    let source_direction = CausalityDirection::from_qualia(a);
    let target_direction = CausalityDirection::from_qualia(b);

    let nars_truth = awareness
        .map(NarsTruthValue::from_awareness)
        .unwrap_or(NarsTruthValue::ignorance());

    CausalityDecomposition {
        causality_flips,
        causality_dims_total: CAUSALITY_DIMS.len(),
        content_flips,
        content_dims_total: 16 - CAUSALITY_DIMS.len(),
        source_direction,
        target_direction,
        reversed: source_direction != target_direction,
        nars_truth,
    }
}

// ============================================================================
// SPO Causal Encoding
// ============================================================================

/// Encode an SPO triple with explicit causality direction.
///
/// The predicate axis (Y) carries the causality signal. When the direction
/// is `Causing` (RGB), the sign bits on causality dimensions of the Y-axis
/// are flipped — encoding that the agent is CAUSING rather than EXPERIENCING.
///
/// ```text
/// (Agent, EXPERIENCES, Grief):
///   Y-axis = P ⊕ O  (standard binding, warmth/social/sacredness positive)
///
/// (Agent, CAUSES, Grief):
///   Y-axis = P ⊕ O ⊕ causality_mask  (sign-flipped on causality dims)
/// ```
///
/// The BF16 sign flip (weight 256) on 3 causality dims contributes 3 × 256 = 768
/// to the distance between CAUSES and EXPERIENCES predicates. This is a strong,
/// measurable signal that the awareness substrate will classify as Tensioned
/// when comparing a causing-state against an experiencing-state.
pub fn spo_encode_causal(
    subject: &CrystalAxis,
    predicate: &CrystalAxis,
    object: &CrystalAxis,
    direction: CausalityDirection,
) -> SpatialCrystal3D {
    let base = SpatialCrystal3D::spo_encode(subject, predicate, object);

    match direction {
        CausalityDirection::Experiencing => base,
        CausalityDirection::Causing => {
            // Flip sign bits on causality dimensions of Y-axis (predicate).
            // BF16 sign bit is bit 15 of u16 = bit 7 of high byte.
            let mut y_data = base.y.data.clone();
            for &dim in &CAUSALITY_DIMS {
                if dim < base.y.n_dims {
                    let hi_byte = dim * 2 + 1;
                    if hi_byte < y_data.len() {
                        y_data[hi_byte] ^= 0x80; // flip BF16 sign bit
                    }
                }
            }
            SpatialCrystal3D::new(
                base.x,
                CrystalAxis {
                    data: y_data,
                    n_dims: base.y.n_dims,
                },
                base.z,
            )
        }
    }
}

/// Build a causality mask for BF16 vectors.
///
/// Returns a byte vector where the sign bit (bit 15 of each BF16 pair) is set
/// on causality dimensions and clear everywhere else. XOR with this mask
/// flips the causality direction without touching content dimensions.
pub fn causality_mask_bf16(n_dims: usize) -> Vec<u8> {
    let mut mask = vec![0u8; n_dims * 2];
    for &dim in &CAUSALITY_DIMS {
        if dim < n_dims {
            mask[dim * 2 + 1] = 0x80; // sign bit of high byte
        }
    }
    mask
}

/// Extract per-axis NARS truth values from a SpatialAwareness decomposition.
///
/// Returns (subject_truth, predicate_truth, object_truth).
/// The predicate truth (Y-axis) is the confidence in causality direction.
pub fn spatial_nars_truth(
    awareness: &crate::spatial_resonance::SpatialAwareness,
) -> (NarsTruthValue, NarsTruthValue, NarsTruthValue) {
    (
        NarsTruthValue::from_awareness(&awareness.x_awareness),
        NarsTruthValue::from_awareness(&awareness.y_awareness),
        NarsTruthValue::from_awareness(&awareness.z_awareness),
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_prosocial() -> PackedQualia {
        // CMYK: warm, social, sacred (all positive on causality dims)
        PackedQualia::new(
            [
                50,  // brightness
                60,  // valence
                40,  // dominance
                30,  // arousal
                70,  // warmth (positive = CMYK)
                80,  // clarity
                60,  // social (positive = CMYK)
                20,  // nostalgia
                50,  // sacredness (positive = CMYK)
                30,  // desire
                20,  // tension
                40,  // awe
                10,  // grief
                60,  // hope
                15,  // edge
                40,  // resolution_hunger
            ],
            1.0,
        )
    }

    fn make_dark() -> PackedQualia {
        // RGB: cold, antisocial, profane (negative on causality dims)
        PackedQualia::new(
            [
                10,   // brightness
                30,   // valence (positive — schadenfreude has pleasure)
                90,   // dominance
                50,   // arousal
                -80,  // warmth (negative = RGB)
                40,   // clarity
                -60,  // social (negative = RGB)
                0,    // nostalgia
                -70,  // sacredness (negative = RGB)
                60,   // desire
                20,   // tension
                0,    // awe
                0,    // grief
                0,    // hope
                85,   // edge
                50,   // resolution_hunger
            ],
            1.0,
        )
    }

    #[test]
    fn test_causality_direction_prosocial() {
        let q = make_prosocial();
        assert_eq!(
            CausalityDirection::from_qualia(&q),
            CausalityDirection::Experiencing,
        );
    }

    #[test]
    fn test_causality_direction_dark() {
        let q = make_dark();
        assert_eq!(
            CausalityDirection::from_qualia(&q),
            CausalityDirection::Causing,
        );
    }

    #[test]
    fn test_causality_decompose_reversal() {
        let prosocial = make_prosocial();
        let dark = make_dark();
        let decomp = causality_decompose(&prosocial, &dark, None);

        assert_eq!(decomp.causality_flips, 3); // warmth, social, sacredness all flip
        assert!(decomp.reversed);
        assert_eq!(decomp.source_direction, CausalityDirection::Experiencing);
        assert_eq!(decomp.target_direction, CausalityDirection::Causing);
        // Content dims: same sign (both positive), different magnitude
        // No content flips in this case — causality is pure direction change
        assert_eq!(decomp.content_flips, 0);
    }

    #[test]
    fn test_causality_decompose_same_direction() {
        let a = make_prosocial();
        let b = make_prosocial();
        let decomp = causality_decompose(&a, &b, None);

        assert_eq!(decomp.causality_flips, 0);
        assert!(!decomp.reversed);
        assert_eq!(decomp.source_direction, CausalityDirection::Experiencing);
        assert_eq!(decomp.target_direction, CausalityDirection::Experiencing);
    }

    #[test]
    fn test_nars_truth_from_awareness() {
        let awareness = SuperpositionState {
            n_dims: 4,
            sign_consensus: vec![255, 255, 0, 255],
            exp_spread: vec![0, 0, 0, 0],
            mantissa_noise: vec![false, false, false, true],
            states: vec![
                AwarenessState::Crystallized,
                AwarenessState::Crystallized,
                AwarenessState::Tensioned,
                AwarenessState::Noise,
            ],
            packed_states: vec![0b00_00_01_11],
            crystallized_pct: 0.5,  // 2/4
            tensioned_pct: 0.25,    // 1/4
            uncertain_pct: 0.0,
            noise_pct: 0.25,        // 1/4
        };

        let truth = NarsTruthValue::from_awareness(&awareness);
        assert!((truth.frequency - 0.5).abs() < 0.01);      // 50% crystallized
        assert!((truth.confidence - 0.75).abs() < 0.01);     // 75% signal (1 - 25% noise)
        assert!((truth.expectation() - 0.5).abs() < 0.01);   // 0.5×0.75 + 0.5×0.25 = 0.5
    }

    #[test]
    fn test_nars_truth_ignorance() {
        let ign = NarsTruthValue::ignorance();
        assert_eq!(ign.frequency, 0.5);
        assert_eq!(ign.confidence, 0.0);
        assert!((ign.expectation() - 0.5).abs() < 0.001); // no evidence = 0.5
    }

    #[test]
    fn test_nars_truth_certain() {
        let certain = NarsTruthValue::new(1.0, 1.0);
        assert!((certain.expectation() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_causality_mask() {
        let mask = causality_mask_bf16(16);
        assert_eq!(mask.len(), 32); // 16 dims × 2 bytes

        // Only causality dims should have sign bit set
        for dim in 0..16 {
            let hi = mask[dim * 2 + 1];
            if CAUSALITY_DIMS.contains(&dim) {
                assert_eq!(hi, 0x80, "dim {dim} should have sign bit in mask");
            } else {
                assert_eq!(hi, 0x00, "dim {dim} should be clear in mask");
            }
        }
    }

    #[test]
    fn test_spo_causal_encoding_differs() {
        let subject = CrystalAxis::from_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
        let predicate = CrystalAxis::from_f32(&[0.5; 16]);
        let object = CrystalAxis::from_f32(&[0.1; 16]);

        let experiencing = spo_encode_causal(
            &subject, &predicate, &object,
            CausalityDirection::Experiencing,
        );
        let causing = spo_encode_causal(
            &subject, &predicate, &object,
            CausalityDirection::Causing,
        );

        // X and Z axes should be identical (same subject and object)
        assert_eq!(experiencing.x.data, causing.x.data, "X-axis should not change");
        assert_eq!(experiencing.z.data, causing.z.data, "Z-axis should not change");

        // Y-axis should differ (causality direction is in predicate)
        assert_ne!(experiencing.y.data, causing.y.data, "Y-axis should differ for different causality");
    }

    #[test]
    fn test_spo_causal_distance() {
        // The distance between CAUSES and EXPERIENCES on the Y-axis should be
        // approximately 3 × 256 = 768 (3 sign flips × sign weight 256)
        let subject = CrystalAxis::from_f32(&[1.0; 16]);
        let predicate = CrystalAxis::from_f32(&[1.0; 16]);
        let object = CrystalAxis::from_f32(&[1.0; 16]);

        let exp = spo_encode_causal(
            &subject, &predicate, &object,
            CausalityDirection::Experiencing,
        );
        let cau = spo_encode_causal(
            &subject, &predicate, &object,
            CausalityDirection::Causing,
        );

        let y_dist = exp.y.distance(&cau.y, &crate::bf16_hamming::BF16Weights::default());

        // 3 sign flips × 256 weight = 768 minimum
        // Actual may be slightly different due to XOR binding affecting other bits
        assert!(
            y_dist >= 768,
            "Y-axis distance should be >= 768 (3 × sign_weight), got {y_dist}"
        );
    }

    #[test]
    fn test_causality_direction_flip() {
        assert_eq!(CausalityDirection::Causing.flip(), CausalityDirection::Experiencing);
        assert_eq!(CausalityDirection::Experiencing.flip(), CausalityDirection::Causing);
    }

    #[test]
    fn test_spatial_nars_truth() {
        use crate::bf16_hamming::AwarenessThresholds;
        use crate::spatial_resonance::{spatial_awareness_decompose, SpatialCrystal3D};

        let a = SpatialCrystal3D::from_f32(
            &[1.0, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
        );
        // Identical → all crystallized → frequency=1.0, confidence=1.0
        let awareness = spatial_awareness_decompose(&a, &a, &AwarenessThresholds::default());
        let (s_truth, p_truth, o_truth) = spatial_nars_truth(&awareness);

        // Identical crystals → everything crystallized → high frequency
        assert!(s_truth.frequency > 0.8, "subject truth should be high: {}", s_truth.frequency);
        assert!(p_truth.frequency > 0.8, "predicate truth should be high: {}", p_truth.frequency);
        assert!(o_truth.frequency > 0.8, "object truth should be high: {}", o_truth.frequency);
    }
}
