//! BF16 3D Spatial Resonance — Crystal4K-aligned axis model.
//!
//! Mirrors ladybug-rs's Crystal4K holographic coordinate system (3 × 10Kbit
//! axis projections) but operates on BF16-structured vectors with
//! sign/exponent/mantissa-aware distance.
//!
//! ## Architecture
//!
//! Crystal4K encodes 3D spatial structure as three orthogonal fingerprint
//! projections (X, Y, Z). This module provides the BF16 counterpart:
//! each axis is a BF16 vector where the 1+7+8 prefix decomposition
//! gives per-dimension awareness of:
//!
//! - **Sign** (1 bit): spatial polarity along this axis
//! - **Exponent** (7 bits): magnitude/scale along this axis
//! - **Mantissa** (8 bits): fine detail (often noise)
//!
//! ## SPO Grammar Integration
//!
//! The SPO crystal (Subject-Predicate-Object) uses 3D grid hashing:
//! - hash(S) → x, hash(P) → y, hash(O) → z
//!
//! This module provides BF16-accelerated distance for SPO triple comparison:
//! - Subject distance on X-axis (who?)
//! - Predicate distance on Y-axis (does what?)
//! - Object distance on Z-axis (to whom?)
//!
//! ## Semantic Kernel Wiring
//!
//! The semantic kernel in ladybug-rs maps all operations to BindSpace addresses.
//! This module provides the hardware-accelerated scoring layer that the kernel
//! dispatches to for spatial queries: `resonate()`, `xor_bind()`, `collapse()`.
//!
//! ## Hardware
//!
//! Uses AVX-512 BITALG for BF16 structured distance via `select_bf16_hamming_fn()`.
//! Falls back to scalar for non-AVX-512 CPUs. The VNNI path (`dot_i8`) is used
//! for int8 embedding dot products on the EMBED container axis.

use crate::bf16_hamming::{
    self, AwarenessState, AwarenessThresholds, BF16StructuralDiff, BF16Weights, SuperpositionState,
};

// ============================================================================
// Crystal Axis — one BF16 vector projection along X, Y, or Z
// ============================================================================

/// A single axis of a 3D BF16 crystal.
///
/// Stores a BF16 byte vector representing one spatial dimension.
/// The sign/exponent/mantissa decomposition maps to spatial semantics:
/// - Sign = direction (positive/negative along axis)
/// - Exponent = magnitude (how far along the axis)
/// - Mantissa = detail (fine position within the scale)
#[derive(Clone, Debug)]
pub struct CrystalAxis {
    /// Raw BF16 bytes (2 bytes per dimension, little-endian).
    pub data: Vec<u8>,
    /// Number of BF16 dimensions.
    pub n_dims: usize,
}

impl CrystalAxis {
    /// Create a crystal axis from BF16 bytes.
    pub fn from_bf16_bytes(data: Vec<u8>) -> Self {
        assert!(
            data.len().is_multiple_of(2),
            "BF16 data must be even number of bytes"
        );
        let n_dims = data.len() / 2;
        Self { data, n_dims }
    }

    /// Create a crystal axis from f32 values (truncated to BF16).
    pub fn from_f32(values: &[f32]) -> Self {
        let data = bf16_hamming::fp32_to_bf16_bytes(values);
        let n_dims = values.len();
        Self { data, n_dims }
    }

    /// Create a zero axis with given dimension count.
    pub fn zero(n_dims: usize) -> Self {
        Self {
            data: vec![0u8; n_dims * 2],
            n_dims,
        }
    }

    /// BF16 structured Hamming distance to another axis.
    ///
    /// Uses AVX-512 BITALG when available via `select_bf16_hamming_fn()`.
    pub fn distance(&self, other: &CrystalAxis, weights: &BF16Weights) -> u64 {
        assert_eq!(self.n_dims, other.n_dims);
        let f = bf16_hamming::select_bf16_hamming_fn();
        f(&self.data, &other.data, weights)
    }

    /// Structural diff — which dimensions changed sign/exponent/mantissa.
    pub fn structural_diff(&self, other: &CrystalAxis) -> BF16StructuralDiff {
        assert_eq!(self.data.len(), other.data.len());
        bf16_hamming::structural_diff(&self.data, &other.data)
    }

    /// XOR-bind two axes (for SPO triple encoding: S ⊕ P, etc).
    pub fn xor_bind(&self, other: &CrystalAxis) -> CrystalAxis {
        assert_eq!(self.data.len(), other.data.len());
        let data: Vec<u8> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a ^ b)
            .collect();
        CrystalAxis {
            data,
            n_dims: self.n_dims,
        }
    }
}

// ============================================================================
// SpatialCrystal3D — the 3-axis BF16 crystal
// ============================================================================

/// A 3D BF16 crystal: three orthogonal axis projections.
///
/// Mirrors Crystal4K's (X, Y, Z) structure but in BF16 space.
/// For SPO triples: X = Subject, Y = Predicate, Z = Object.
#[derive(Clone, Debug)]
pub struct SpatialCrystal3D {
    /// X-axis projection (Subject / spatial X).
    pub x: CrystalAxis,
    /// Y-axis projection (Predicate / spatial Y).
    pub y: CrystalAxis,
    /// Z-axis projection (Object / spatial Z).
    pub z: CrystalAxis,
}

impl SpatialCrystal3D {
    /// Create a 3D crystal from three BF16 axis vectors.
    pub fn new(x: CrystalAxis, y: CrystalAxis, z: CrystalAxis) -> Self {
        Self { x, y, z }
    }

    /// Create from three f32 vectors (one per axis).
    pub fn from_f32(x: &[f32], y: &[f32], z: &[f32]) -> Self {
        Self {
            x: CrystalAxis::from_f32(x),
            y: CrystalAxis::from_f32(y),
            z: CrystalAxis::from_f32(z),
        }
    }

    /// Create from a flat byte slice (3 × n_bytes, concatenated X|Y|Z).
    pub fn from_flat_bytes(bytes: &[u8], axis_bytes: usize) -> Self {
        assert_eq!(bytes.len(), axis_bytes * 3);
        Self {
            x: CrystalAxis::from_bf16_bytes(bytes[..axis_bytes].to_vec()),
            y: CrystalAxis::from_bf16_bytes(bytes[axis_bytes..axis_bytes * 2].to_vec()),
            z: CrystalAxis::from_bf16_bytes(bytes[axis_bytes * 2..].to_vec()),
        }
    }

    /// Number of BF16 dimensions per axis.
    pub fn dims_per_axis(&self) -> usize {
        self.x.n_dims
    }

    /// Total BF16 bytes across all 3 axes.
    pub fn total_bytes(&self) -> usize {
        self.x.data.len() + self.y.data.len() + self.z.data.len()
    }

    /// 3D BF16 distance: sum of per-axis structured distances.
    ///
    /// This is the L1 (Manhattan) distance in BF16-structured space.
    /// Each axis uses the full sign/exponent/mantissa weighting.
    pub fn distance_l1(&self, other: &SpatialCrystal3D, weights: &BF16Weights) -> u64 {
        self.x.distance(&other.x, weights)
            + self.y.distance(&other.y, weights)
            + self.z.distance(&other.z, weights)
    }

    /// 3D BF16 distance: max of per-axis distances (Chebyshev / L-infinity).
    ///
    /// Useful for SPO queries: "find triples where ANY axis is close".
    pub fn distance_linf(&self, other: &SpatialCrystal3D, weights: &BF16Weights) -> u64 {
        let dx = self.x.distance(&other.x, weights);
        let dy = self.y.distance(&other.y, weights);
        let dz = self.z.distance(&other.z, weights);
        dx.max(dy).max(dz)
    }

    /// Per-axis distances (for decomposed scoring).
    pub fn axis_distances(
        &self,
        other: &SpatialCrystal3D,
        weights: &BF16Weights,
    ) -> SpatialDistances {
        SpatialDistances {
            x: self.x.distance(&other.x, weights),
            y: self.y.distance(&other.y, weights),
            z: self.z.distance(&other.z, weights),
        }
    }

    /// SPO-encode: S ⊕ P on X-axis, P ⊕ O on Y-axis, S ⊕ O on Z-axis.
    ///
    /// This mirrors the SPO crystal encoding in ladybug-rs:
    /// `S ⊕ ROLE_S ⊕ P ⊕ ROLE_P ⊕ O ⊕ ROLE_O`
    /// but decomposed per axis for BF16-structured distance.
    pub fn spo_encode(
        subject: &CrystalAxis,
        predicate: &CrystalAxis,
        object: &CrystalAxis,
    ) -> Self {
        Self {
            x: subject.xor_bind(predicate), // S ⊕ P
            y: predicate.xor_bind(object),  // P ⊕ O
            z: subject.xor_bind(object),    // S ⊕ O
        }
    }

    /// Recover subject from SPO encoding given predicate.
    /// S = X ⊕ P (since X = S ⊕ P, XOR is self-inverse).
    pub fn spo_recover_subject(&self, predicate: &CrystalAxis) -> CrystalAxis {
        self.x.xor_bind(predicate)
    }

    /// Recover object from SPO encoding given predicate.
    /// O = Y ⊕ P (since Y = P ⊕ O).
    pub fn spo_recover_object(&self, predicate: &CrystalAxis) -> CrystalAxis {
        self.y.xor_bind(predicate)
    }

    /// Serialize to flat bytes (X|Y|Z concatenated).
    pub fn to_flat_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.total_bytes());
        bytes.extend_from_slice(&self.x.data);
        bytes.extend_from_slice(&self.y.data);
        bytes.extend_from_slice(&self.z.data);
        bytes
    }
}

/// Per-axis spatial distances.
#[derive(Clone, Copy, Debug)]
pub struct SpatialDistances {
    /// X-axis distance (Subject similarity for SPO).
    pub x: u64,
    /// Y-axis distance (Predicate similarity for SPO).
    pub y: u64,
    /// Z-axis distance (Object similarity for SPO).
    pub z: u64,
}

impl SpatialDistances {
    /// L1 (Manhattan) total.
    pub fn l1(&self) -> u64 {
        self.x + self.y + self.z
    }

    /// L-infinity (Chebyshev) — max axis.
    pub fn linf(&self) -> u64 {
        self.x.max(self.y).max(self.z)
    }

    /// Dominant axis (which axis contributes most distance).
    pub fn dominant_axis(&self) -> SpatialAxis {
        if self.x >= self.y && self.x >= self.z {
            SpatialAxis::X
        } else if self.y >= self.z {
            SpatialAxis::Y
        } else {
            SpatialAxis::Z
        }
    }

    /// Axis balance: 1.0 = perfectly balanced, 0.0 = one axis dominates.
    pub fn balance(&self) -> f32 {
        let total = self.l1() as f32;
        if total == 0.0 {
            return 1.0;
        }
        let max = self.linf() as f32;
        1.0 - (max / total - 1.0 / 3.0) * 1.5 // normalize so balanced=1, single-axis=0
    }
}

/// Which spatial axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpatialAxis {
    X,
    Y,
    Z,
}

// ============================================================================
// Spatial Awareness — 3D awareness decomposition
// ============================================================================

/// Per-axis awareness state from BF16 superposition decomposition.
#[derive(Clone, Debug)]
pub struct SpatialAwareness {
    /// X-axis awareness (Subject axis for SPO).
    pub x_awareness: SuperpositionState,
    /// Y-axis awareness (Predicate axis for SPO).
    pub y_awareness: SuperpositionState,
    /// Z-axis awareness (Object axis for SPO).
    pub z_awareness: SuperpositionState,
    /// Per-axis crystallization ratios.
    pub crystallized: [f32; 3],
    /// Per-axis tension ratios.
    pub tensioned: [f32; 3],
    /// Overall spatial coherence (product of axis crystallization).
    pub spatial_coherence: f32,
}

/// Decompose two 3D crystals into per-axis awareness.
///
/// This extends the 1D superposition decomposition from bf16_hamming.rs
/// to 3 independent axes. Each axis gets its own awareness classification
/// (crystallized/tensioned/uncertain/noise per dimension).
pub fn spatial_awareness_decompose(
    a: &SpatialCrystal3D,
    b: &SpatialCrystal3D,
    thresholds: &AwarenessThresholds,
) -> SpatialAwareness {
    let x_awareness = bf16_hamming::superposition_decompose(&[&a.x.data, &b.x.data], thresholds);
    let y_awareness = bf16_hamming::superposition_decompose(&[&a.y.data, &b.y.data], thresholds);
    let z_awareness = bf16_hamming::superposition_decompose(&[&a.z.data, &b.z.data], thresholds);

    let crystallized = [
        x_awareness.crystallized_pct,
        y_awareness.crystallized_pct,
        z_awareness.crystallized_pct,
    ];
    let tensioned = [
        x_awareness.tensioned_pct,
        y_awareness.tensioned_pct,
        z_awareness.tensioned_pct,
    ];

    // Spatial coherence: geometric mean of crystallization across axes
    let spatial_coherence = (crystallized[0] * crystallized[1] * crystallized[2]).cbrt();

    SpatialAwareness {
        x_awareness,
        y_awareness,
        z_awareness,
        crystallized,
        tensioned,
        spatial_coherence,
    }
}

// ============================================================================
// Spatial Learning Signal — 3D feedback for NARS
// ============================================================================

/// Learning signal from 3D spatial recognition.
///
/// Extends the 1D LearningSignal with per-axis decomposition.
/// This feeds into ladybug-rs's feedback.rs to update NARS truth values
/// with spatial context: "which axis changed?" → "which part of the SPO
/// triple needs revision?"
#[derive(Clone, Debug)]
pub struct SpatialLearningSignal {
    /// Per-axis structural diffs (sign flips, exp shifts, mantissa noise).
    pub x_diff: BF16StructuralDiff,
    pub y_diff: BF16StructuralDiff,
    pub z_diff: BF16StructuralDiff,

    /// Per-axis awareness decomposition.
    pub awareness: SpatialAwareness,

    /// Which axis has the most sign flips (= which SPO component changed class).
    pub dominant_change_axis: SpatialAxis,
    /// Total sign flips across all axes.
    pub total_sign_flips: usize,
    /// Total exponent shifts across all axes.
    pub total_exp_shifts: usize,

    /// Per-axis attention weights (32 per axis × 3 axes = 96 total).
    /// First 32: X-axis, next 32: Y-axis, last 32: Z-axis.
    pub attention_weights_3d: Vec<f32>,
}

/// Extract a 3D spatial learning signal from two crystals.
///
/// This is the spatial equivalent of `extract_learning_signal()` in hybrid.rs,
/// but decomposed per axis for Crystal4K / SPO grammar integration.
pub fn extract_spatial_learning_signal(
    query: &SpatialCrystal3D,
    result: &SpatialCrystal3D,
    thresholds: &AwarenessThresholds,
) -> SpatialLearningSignal {
    let x_diff = query.x.structural_diff(&result.x);
    let y_diff = query.y.structural_diff(&result.y);
    let z_diff = query.z.structural_diff(&result.z);

    let awareness = spatial_awareness_decompose(query, result, thresholds);

    // Dominant change axis: which axis has most sign flips
    let flips = [x_diff.sign_flips, y_diff.sign_flips, z_diff.sign_flips];
    let dominant_change_axis = if flips[0] >= flips[1] && flips[0] >= flips[2] {
        SpatialAxis::X
    } else if flips[1] >= flips[2] {
        SpatialAxis::Y
    } else {
        SpatialAxis::Z
    };

    let total_sign_flips = x_diff.sign_flips + y_diff.sign_flips + z_diff.sign_flips;
    let total_exp_shifts =
        x_diff.exponent_bits_changed + y_diff.exponent_bits_changed + z_diff.exponent_bits_changed;

    // Per-axis attention weights from awareness states
    let mut attention_weights_3d = Vec::with_capacity(96);
    for axis_awareness in [
        &awareness.x_awareness,
        &awareness.y_awareness,
        &awareness.z_awareness,
    ] {
        let n_dims = axis_awareness.n_dims;
        let group_size = n_dims.div_ceil(32).max(1);
        for group in 0..32 {
            let start = group * group_size;
            let end = (start + group_size).min(n_dims);
            if start >= n_dims {
                attention_weights_3d.push(0.5); // default for unused groups
                continue;
            }
            let mut crystallized = 0u32;
            let mut tensioned = 0u32;
            let mut noise = 0u32;
            let group_count = (end - start) as f32;
            for d in start..end {
                match axis_awareness.states[d] {
                    AwarenessState::Crystallized => crystallized += 1,
                    AwarenessState::Tensioned => tensioned += 1,
                    AwarenessState::Noise => noise += 1,
                    AwarenessState::Uncertain => {}
                }
            }
            let weight = if crystallized as f32 / group_count > 0.5 {
                0.8 + 0.2 * (crystallized as f32 / group_count)
            } else if tensioned as f32 / group_count > 0.3 {
                0.5
            } else if noise as f32 / group_count > 0.5 {
                0.1
            } else {
                0.5
            };
            attention_weights_3d.push(weight);
        }
    }

    SpatialLearningSignal {
        x_diff,
        y_diff,
        z_diff,
        awareness,
        dominant_change_axis,
        total_sign_flips,
        total_exp_shifts,
        attention_weights_3d,
    }
}

// ============================================================================
// Spatial Sweep — 3D batch search
// ============================================================================

/// Result of a spatial sweep (one match).
#[derive(Clone, Debug)]
pub struct SpatialMatch {
    /// Index in the database.
    pub index: usize,
    /// Per-axis distances.
    pub distances: SpatialDistances,
    /// L1 total distance.
    pub total_distance: u64,
    /// Dominant axis (which axis contributed most distance).
    pub dominant_axis: SpatialAxis,
}

/// Sweep a 3D crystal database for matches within threshold.
///
/// Uses per-axis early exit: if X-axis distance alone exceeds threshold,
/// skip Y and Z axes entirely.
pub fn spatial_sweep(
    query: &SpatialCrystal3D,
    database: &[SpatialCrystal3D],
    threshold: u64,
    weights: &BF16Weights,
    limit: usize,
) -> Vec<SpatialMatch> {
    let per_axis_threshold = threshold; // each axis can use up to full threshold

    let mut matches = Vec::new();

    for (idx, candidate) in database.iter().enumerate() {
        // Early exit on X-axis
        let dx = query.x.distance(&candidate.x, weights);
        if dx > per_axis_threshold {
            continue;
        }

        // Early exit on X+Y
        let dy = query.y.distance(&candidate.y, weights);
        if dx + dy > threshold {
            continue;
        }

        // Full 3D distance
        let dz = query.z.distance(&candidate.z, weights);
        let total = dx + dy + dz;

        if total <= threshold {
            let distances = SpatialDistances {
                x: dx,
                y: dy,
                z: dz,
            };
            matches.push(SpatialMatch {
                index: idx,
                distances,
                total_distance: total,
                dominant_axis: distances.dominant_axis(),
            });
        }

        if limit > 0 && matches.len() >= limit {
            break;
        }
    }

    matches.sort_by_key(|m| m.total_distance);
    matches
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_axis(values: &[f32]) -> CrystalAxis {
        CrystalAxis::from_f32(values)
    }

    #[test]
    fn test_crystal_axis_distance_identical() {
        let a = make_axis(&[1.0, 2.0, 3.0, 4.0]);
        let d = a.distance(&a, &BF16Weights::default());
        assert_eq!(d, 0, "identical axes should have zero distance");
    }

    #[test]
    fn test_crystal_axis_distance_sign_flip() {
        let a = make_axis(&[1.0, 2.0, 3.0, 4.0]);
        let b = make_axis(&[-1.0, -2.0, 3.0, 4.0]);
        let d = a.distance(&b, &BF16Weights::default());
        assert!(
            d >= 512,
            "two sign flips should give >= 512 distance, got {}",
            d
        );
    }

    #[test]
    fn test_crystal_axis_xor_bind_self_inverse() {
        let a = make_axis(&[1.0, 2.0, 3.0]);
        let b = make_axis(&[4.0, 5.0, 6.0]);
        let bound = a.xor_bind(&b);
        let recovered = bound.xor_bind(&b);
        assert_eq!(a.data, recovered.data, "XOR bind should be self-inverse");
    }

    #[test]
    fn test_spatial_crystal_3d_distance() {
        let a = SpatialCrystal3D::from_f32(
            &[1.0, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
        );
        let d = a.distance_l1(&a, &BF16Weights::default());
        assert_eq!(d, 0, "identical crystals should have zero L1 distance");
    }

    #[test]
    fn test_spatial_crystal_3d_axis_distances() {
        let a = SpatialCrystal3D::from_f32(&[1.0, 2.0], &[3.0, 4.0], &[5.0, 6.0]);
        let b = SpatialCrystal3D::from_f32(
            &[-1.0, 2.0], // X: sign flip
            &[3.0, 4.0],  // Y: identical
            &[5.0, 6.0],  // Z: identical
        );
        let dists = a.axis_distances(&b, &BF16Weights::default());
        assert!(dists.x > 0, "X should have distance from sign flip");
        assert_eq!(dists.y, 0, "Y should be identical");
        assert_eq!(dists.z, 0, "Z should be identical");
        assert_eq!(dists.dominant_axis(), SpatialAxis::X);
    }

    #[test]
    fn test_spo_encode_recover() {
        let subject = make_axis(&[1.0, 2.0, 3.0, 4.0]);
        let predicate = make_axis(&[5.0, 6.0, 7.0, 8.0]);
        let object = make_axis(&[9.0, 10.0, 11.0, 12.0]);

        let encoded = SpatialCrystal3D::spo_encode(&subject, &predicate, &object);
        let recovered_s = encoded.spo_recover_subject(&predicate);
        let recovered_o = encoded.spo_recover_object(&predicate);

        assert_eq!(
            subject.data, recovered_s.data,
            "subject should be recoverable"
        );
        assert_eq!(
            object.data, recovered_o.data,
            "object should be recoverable"
        );
    }

    #[test]
    fn test_spatial_awareness_decompose() {
        let a = SpatialCrystal3D::from_f32(
            &[1.0, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
        );
        // Identical → all crystallized
        let awareness = spatial_awareness_decompose(&a, &a, &AwarenessThresholds::default());
        assert!(
            awareness.spatial_coherence > 0.9,
            "identical crystals should have high coherence: {}",
            awareness.spatial_coherence
        );
        for &c in &awareness.crystallized {
            assert!(c > 0.9, "each axis should be >90% crystallized: {}", c);
        }
    }

    #[test]
    fn test_spatial_learning_signal() {
        let query = SpatialCrystal3D::from_f32(
            &[1.0, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
        );
        let result = SpatialCrystal3D::from_f32(
            &[-1.0, -2.0, 3.0, 4.0],  // X: sign flips
            &[5.0, 6.0, 7.0, 8.0],    // Y: identical
            &[9.0, 10.0, 11.0, 12.0], // Z: identical
        );

        let signal =
            extract_spatial_learning_signal(&query, &result, &AwarenessThresholds::default());

        assert!(signal.x_diff.sign_flips > 0, "X should have sign flips");
        assert_eq!(signal.y_diff.sign_flips, 0, "Y should have no sign flips");
        assert_eq!(signal.z_diff.sign_flips, 0, "Z should have no sign flips");
        assert_eq!(signal.dominant_change_axis, SpatialAxis::X);
        assert_eq!(signal.attention_weights_3d.len(), 96); // 32 per axis × 3
    }

    #[test]
    fn test_spatial_sweep_finds_match() {
        let query = SpatialCrystal3D::from_f32(
            &[1.0, 2.0, 3.0, 4.0],
            &[5.0, 6.0, 7.0, 8.0],
            &[9.0, 10.0, 11.0, 12.0],
        );

        let database = vec![
            query.clone(), // exact match at index 0
            SpatialCrystal3D::from_f32(
                &[-100.0, -200.0, -300.0, -400.0],
                &[-500.0, -600.0, -700.0, -800.0],
                &[-900.0, -1000.0, -1100.0, -1200.0],
            ), // very different
        ];

        let matches = spatial_sweep(&query, &database, 100, &BF16Weights::default(), 10);
        assert!(!matches.is_empty(), "should find at least one match");
        assert_eq!(matches[0].index, 0, "best match should be index 0");
        assert_eq!(
            matches[0].total_distance, 0,
            "exact match should have 0 distance"
        );
    }

    #[test]
    fn test_spatial_sweep_early_exit() {
        let query = SpatialCrystal3D::from_f32(&[1.0; 64], &[1.0; 64], &[1.0; 64]);

        // All candidates very far away on X → should exit early without computing Y, Z
        let database: Vec<SpatialCrystal3D> = (0..100)
            .map(|_| SpatialCrystal3D::from_f32(&[-100.0; 64], &[1.0; 64], &[1.0; 64]))
            .collect();

        let matches = spatial_sweep(&query, &database, 100, &BF16Weights::default(), 10);
        assert!(
            matches.is_empty(),
            "all candidates should be rejected by X-axis early exit"
        );
    }

    #[test]
    fn test_flat_bytes_roundtrip() {
        let crystal =
            SpatialCrystal3D::from_f32(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]);
        let bytes = crystal.to_flat_bytes();
        let back = SpatialCrystal3D::from_flat_bytes(&bytes, crystal.x.data.len());
        assert_eq!(crystal.x.data, back.x.data);
        assert_eq!(crystal.y.data, back.y.data);
        assert_eq!(crystal.z.data, back.z.data);
    }

    #[test]
    fn test_spatial_distances_balance() {
        let balanced = SpatialDistances {
            x: 100,
            y: 100,
            z: 100,
        };
        assert!(
            balanced.balance() > 0.9,
            "equal distances should be balanced"
        );

        let imbalanced = SpatialDistances { x: 300, y: 0, z: 0 };
        assert!(
            imbalanced.balance() < 0.5,
            "single-axis distance should be imbalanced"
        );
    }
}
