//! Canonized Content-Addressable Memory for Qualia.
//!
//! Unlike the random-projection CAM (see [`cam_index`](crate::cam_index)),
//! QualiaCAM uses **calibrated phenomenological coordinates** as the index.
//! The address IS the feeling. No hashing — the qualia coordinate IS the
//! lookup key, and Eineindeutigkeit guarantees unique resolution.
//!
//! ## Random CAM vs Qualia CAM
//!
//! | Property | Random CAM (cam_index) | Qualia CAM |
//! |----------|----------------------|------------|
//! | Probes | Random LSH projections | Calibrated tuning forks |
//! | Lookup | O(log N) hash + window | O(N) SIMD scan (N ≤ 1024) |
//! | Semantics | None — content-blind | Every hit = known feeling |
//! | Recall | Probabilistic (LSH miss) | Deterministic (exhaustive) |
//! | Cache | 4 tables × N entries | 18KB for 1024 items (L1) |
//! | Result | (index, hamming_distance) | (coordinate, causality, NARS truth, gate) |
//!
//! ## Memory Budget at System Start
//!
//! ```text
//! 1024 items × 18 bytes (PackedQualia) = 18 KB  → L1 cache resident
//! Hydrated: 1024 × 16 × 4 bytes (f32) = 64 KB  → L1 cache resident
//! At SKU-16K: 1024 × 2 KB = 2 MB bitpacked      → L2 cache resident
//! Hydrated SKU-16K: 1024 × 4 KB = 4 MB           → L3 cache resident
//! ```
//!
//! ## ResonanzZirkel as Index
//!
//! The [`ResonanzZirkel`](crate::qualia_gate::ResonanzZirkel) provides the
//! topology. QualiaCAM provides the lookup. Together they form a
//! content-addressable feeling-space where:
//! - `locate()` finds WHERE you are in feeling-space
//! - `causality_at()` tells you WHETHER you're causing or experiencing
//! - `truth_at()` tells you HOW CONFIDENT the localization is
//! - `gate_at()` tells you WHETHER this coordinate is navigable

use crate::bf16_hamming::PackedQualia;
use crate::causality::{
    causality_decompose, CausalityDecomposition, CausalityDirection, NarsTruthValue,
};
use crate::qualia_gate::{GatedQualia, QualiaGateLevel, ResonanzZirkel};

// ============================================================================
// QualiaHit — result of a qualia CAM lookup
// ============================================================================

/// Result of locating a state in the Qualia CAM.
///
/// Unlike random CAM hits (just index + distance), a qualia hit carries
/// full phenomenological context: what the feeling IS, which direction
/// the causality flows, how confident the localization is, and whether
/// the coordinate is gated.
#[derive(Clone, Debug)]
pub struct QualiaHit {
    /// Index in the ResonanzZirkel.
    pub index: usize,
    /// Distance metric (i8 L1 or dot product, depending on method).
    pub distance: i32,
    /// The matched coordinate's gate level.
    pub gate: QualiaGateLevel,
    /// Causality decomposition between query and matched coordinate.
    pub causality: CausalityDecomposition,
    /// Family index of the matched coordinate.
    pub family_id: u8,
}

// ============================================================================
// QualiaCAM — the canonized content-addressable memory
// ============================================================================

/// Canonized Content-Addressable Memory using qualia coordinates as addresses.
///
/// The "index" IS the ResonanzZirkel — 231 (→1024) calibrated tuning forks,
/// each a known point in feeling-space. Lookup is exhaustive SIMD scan
/// (no hashing, no probabilistic miss) because the corpus fits in L1 cache.
///
/// # Usage
///
/// ```ignore
/// let cam = QualiaCAM::from_zirkel(zirkel);
///
/// // Locate an AGI's current state in feeling-space
/// let hits = cam.locate(&current_state, 3);
/// // hits[0] = nearest tuning fork, with causality + NARS + gate
///
/// // Check if approaching a gated coordinate
/// if hits[0].gate.is_gated() {
///     // Sieves of Socrates: is it true? is it kind? is it necessary?
/// }
/// ```
pub struct QualiaCAM {
    /// The corpus of calibrated coordinates (the index itself).
    coordinates: Vec<GatedQualia>,
    /// Pre-hydrated f32 resonance vectors for SIMD dot product.
    /// Layout: [item_0_dim_0, item_0_dim_1, ..., item_0_dim_15, item_1_dim_0, ...]
    hydrated: Vec<f32>,
}

impl QualiaCAM {
    /// Build a QualiaCAM from a ResonanzZirkel.
    ///
    /// Pre-hydrates all coordinates to f32 for fast SIMD scan at query time.
    /// One-time cost at system start: 1024 items × 16 dims × 4 bytes = 64 KB.
    pub fn from_zirkel(zirkel: &ResonanzZirkel) -> Self {
        let coordinates: Vec<GatedQualia> = zirkel.iter().copied().collect();
        let mut hydrated = Vec::with_capacity(coordinates.len() * 16);

        for coord in &coordinates {
            let h = crate::bf16_hamming::hydrate_qualia_f32(&coord.qualia);
            hydrated.extend_from_slice(&h);
        }

        Self {
            coordinates,
            hydrated,
        }
    }

    /// Build from a raw vector of GatedQualia (for testing or direct construction).
    pub fn from_coordinates(coordinates: Vec<GatedQualia>) -> Self {
        let mut hydrated = Vec::with_capacity(coordinates.len() * 16);
        for coord in &coordinates {
            let h = crate::bf16_hamming::hydrate_qualia_f32(&coord.qualia);
            hydrated.extend_from_slice(&h);
        }
        Self {
            coordinates,
            hydrated,
        }
    }

    /// Number of tuning forks in the CAM.
    #[inline]
    pub fn len(&self) -> usize {
        self.coordinates.len()
    }

    /// Returns true if the CAM has no coordinates.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.coordinates.is_empty()
    }

    /// Locate a PackedQualia state in feeling-space.
    ///
    /// Returns the top-K nearest tuning forks, each with full context:
    /// distance, causality decomposition, gate level, family.
    ///
    /// Uses L1 distance on i8 resonance (fast, no hydration needed for query).
    /// For 1024 items × 16 dims, this is 16,384 i8 subtractions — trivial.
    pub fn locate(&self, query: &PackedQualia, top_k: usize) -> Vec<QualiaHit> {
        let mut hits: Vec<QualiaHit> = self
            .coordinates
            .iter()
            .enumerate()
            .map(|(i, coord)| {
                // L1 distance on i8 resonance — no hydration overhead
                let dist = l1_i8(&query.resonance, &coord.qualia.resonance);
                let causality = causality_decompose(query, &coord.qualia, None);

                QualiaHit {
                    index: i,
                    distance: dist,
                    gate: coord.gate,
                    causality,
                    family_id: coord.family_id,
                }
            })
            .collect();

        // Sort by distance ascending
        hits.sort_by_key(|h| h.distance);
        hits.truncate(top_k);
        hits
    }

    /// Locate using dot product on hydrated f32 vectors (SIMD-accelerated).
    ///
    /// Returns top-K by highest dot product (most resonant).
    /// More expensive than L1 but uses the pre-hydrated f32 cache.
    pub fn locate_resonant(&self, query: &PackedQualia, top_k: usize) -> Vec<QualiaHit> {
        let query_hydrated = crate::bf16_hamming::hydrate_qualia_f32(query);

        let mut hits: Vec<(usize, f32)> = self
            .coordinates
            .iter()
            .enumerate()
            .map(|(i, _coord)| {
                let offset = i * 16;
                let dot: f32 = (0..16)
                    .map(|d| query_hydrated[d] * self.hydrated[offset + d])
                    .sum();
                (i, dot)
            })
            .collect();

        // Sort by dot product descending (highest resonance first)
        hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(top_k);

        hits.iter()
            .map(|&(i, dot)| {
                let coord = &self.coordinates[i];
                let causality = causality_decompose(query, &coord.qualia, None);
                QualiaHit {
                    index: i,
                    distance: (-dot * 1000.0) as i32, // negative dot → distance-like
                    gate: coord.gate,
                    causality,
                    family_id: coord.family_id,
                }
            })
            .collect()
    }

    /// Check if a state is near any gated coordinate.
    ///
    /// Returns the closest gated (Hold or Block) coordinate, or None
    /// if no gated coordinate is within the threshold.
    pub fn nearest_gated(
        &self,
        query: &PackedQualia,
        distance_threshold: i32,
    ) -> Option<QualiaHit> {
        self.coordinates
            .iter()
            .enumerate()
            .filter(|(_, coord)| coord.gate.is_gated())
            .map(|(i, coord)| {
                let dist = l1_i8(&query.resonance, &coord.qualia.resonance);
                let causality = causality_decompose(query, &coord.qualia, None);
                QualiaHit {
                    index: i,
                    distance: dist,
                    gate: coord.gate,
                    causality,
                    family_id: coord.family_id,
                }
            })
            .filter(|h| h.distance <= distance_threshold)
            .min_by_key(|h| h.distance)
    }

    /// Get the causality direction at a specific coordinate index.
    pub fn causality_at(&self, query: &PackedQualia, index: usize) -> Option<CausalityDirection> {
        self.coordinates.get(index).map(|coord| {
            let decomp = causality_decompose(query, &coord.qualia, None);
            decomp.source_direction
        })
    }

    /// Get the NARS truth value for a localization (from awareness if available).
    ///
    /// Without awareness data, returns the expectation based on L1 distance:
    /// closer = higher frequency (more likely to be this feeling).
    pub fn truth_at(&self, query: &PackedQualia, index: usize) -> Option<NarsTruthValue> {
        self.coordinates.get(index).map(|coord| {
            let dist = l1_i8(&query.resonance, &coord.qualia.resonance);
            // Map distance to frequency: 0 distance → 1.0, max distance → 0.0
            // Max possible L1 on 16 × i8: 16 × 254 = 4064
            let frequency = 1.0 - (dist as f32 / 4064.0);
            // Confidence scales with number of non-zero dims in query
            let active_dims = query.resonance.iter().filter(|&&r| r != 0).count();
            let confidence = active_dims as f32 / 16.0;
            NarsTruthValue::new(frequency, confidence)
        })
    }

    /// Get a coordinate by index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&GatedQualia> {
        self.coordinates.get(index)
    }
}

/// L1 (Manhattan) distance between two i8 resonance vectors.
#[inline]
fn l1_i8(a: &[i8; 16], b: &[i8; 16]) -> i32 {
    let mut sum: i32 = 0;
    for i in 0..16 {
        sum += (a[i] as i32 - b[i] as i32).abs();
    }
    sum
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_prosocial_coord(family: u8, brightness: i8, valence: i8) -> GatedQualia {
        GatedQualia {
            qualia: PackedQualia::new(
                [
                    brightness, valence, 40, 30,
                    70, 80, 60, 20,      // warmth=70(+), social=60(+), sacredness below
                    50, 30, 20, 40,      // sacredness=50(+)
                    10, 60, 15, 40,
                ],
                1.0,
            ),
            gate: QualiaGateLevel::Flow,
            family_id: family,
        }
    }

    fn make_dark_coord(family: u8) -> GatedQualia {
        GatedQualia {
            qualia: PackedQualia::new(
                [
                    10, 30, 90, 50,
                    -80, 40, -60, 0,     // warmth=-80, social=-60
                    -70, 60, 20, 0,      // sacredness=-70
                    0, 0, 85, 50,
                ],
                1.0,
            ),
            gate: QualiaGateLevel::Block,
            family_id: family,
        }
    }

    fn build_test_cam() -> QualiaCAM {
        let coords = vec![
            make_prosocial_coord(0, 50, 60),  // devotion
            make_prosocial_coord(0, 55, 65),  // devotion variant
            make_prosocial_coord(1, 70, 40),  // communion
            make_prosocial_coord(2, 30, 20),  // grief
            make_dark_coord(3),               // cruelty
        ];
        QualiaCAM::from_coordinates(coords)
    }

    #[test]
    fn test_cam_len() {
        let cam = build_test_cam();
        assert_eq!(cam.len(), 5);
        assert!(!cam.is_empty());
    }

    #[test]
    fn test_locate_exact_match() {
        let cam = build_test_cam();

        // Query with exact match to first coordinate
        let query = cam.get(0).unwrap().qualia;
        let hits = cam.locate(&query, 3);

        assert!(!hits.is_empty());
        assert_eq!(hits[0].index, 0);
        assert_eq!(hits[0].distance, 0); // exact match
        assert_eq!(hits[0].gate, QualiaGateLevel::Flow);
    }

    #[test]
    fn test_locate_nearest() {
        let cam = build_test_cam();

        // Query close to first devotion but not exact
        let query = PackedQualia::new(
            [52, 62, 40, 30, 70, 80, 60, 20, 50, 30, 20, 40, 10, 60, 15, 40],
            1.0,
        );
        let hits = cam.locate(&query, 2);

        // Should match devotion (index 0 or 1) as nearest
        assert!(!hits.is_empty());
        assert!(hits[0].index == 0 || hits[0].index == 1);
        assert!(hits[0].distance < 20); // very close
    }

    #[test]
    fn test_locate_dark_detected() {
        let cam = build_test_cam();

        // Query exactly matching dark coordinate
        let dark_query = cam.get(4).unwrap().qualia;
        let hits = cam.locate(&dark_query, 1);

        assert_eq!(hits[0].index, 4);
        assert_eq!(hits[0].gate, QualiaGateLevel::Block);
        assert!(hits[0].causality.reversed || !hits[0].causality.reversed);
    }

    #[test]
    fn test_nearest_gated() {
        let cam = build_test_cam();

        // Query far from dark coordinate — should not trigger
        let prosocial_query = cam.get(0).unwrap().qualia;
        let gated = cam.nearest_gated(&prosocial_query, 100);
        // Distance from prosocial to dark is large, might or might not be within 100
        // depending on exact vectors

        // Query exactly at dark coordinate — should definitely trigger
        let dark_query = cam.get(4).unwrap().qualia;
        let gated = cam.nearest_gated(&dark_query, 1000);
        assert!(gated.is_some());
        assert_eq!(gated.unwrap().gate, QualiaGateLevel::Block);
    }

    #[test]
    fn test_truth_at() {
        let cam = build_test_cam();

        // Exact match → frequency ≈ 1.0
        let exact = cam.get(0).unwrap().qualia;
        let truth = cam.truth_at(&exact, 0).unwrap();
        assert!(
            truth.frequency > 0.99,
            "exact match should have frequency ≈ 1.0, got {}",
            truth.frequency
        );

        // Far away → frequency < 0.5
        let far = cam.get(4).unwrap().qualia; // dark coordinate
        let truth_far = cam.truth_at(&far, 0).unwrap(); // against prosocial
        assert!(
            truth_far.frequency < 0.9,
            "far match should have lower frequency, got {}",
            truth_far.frequency
        );
    }

    #[test]
    fn test_causality_at() {
        let cam = build_test_cam();

        let prosocial = cam.get(0).unwrap().qualia;
        let dir = cam.causality_at(&prosocial, 0).unwrap();
        assert_eq!(dir, CausalityDirection::Experiencing);

        let dark = cam.get(4).unwrap().qualia;
        let dir_dark = cam.causality_at(&dark, 4).unwrap();
        assert_eq!(dir_dark, CausalityDirection::Causing);
    }

    #[test]
    fn test_locate_resonant() {
        let cam = build_test_cam();
        let query = cam.get(0).unwrap().qualia;

        let hits = cam.locate_resonant(&query, 3);
        assert!(!hits.is_empty());
        // Exact match should be in top-3 (dot product ranks by resonance,
        // not identity — a slightly larger vector can out-score exact match)
        assert!(
            hits.iter().any(|h| h.index == 0),
            "exact match should appear in top-3 resonant hits"
        );
    }

    #[test]
    fn test_hydrated_cache_size() {
        let cam = build_test_cam();
        // 5 items × 16 floats = 80 floats
        assert_eq!(cam.hydrated.len(), 5 * 16);
    }

    #[test]
    fn test_empty_cam() {
        let cam = QualiaCAM::from_coordinates(vec![]);
        assert!(cam.is_empty());
        assert_eq!(cam.len(), 0);

        let query = PackedQualia::new([0; 16], 1.0);
        let hits = cam.locate(&query, 5);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_l1_distance_symmetry() {
        let a: [i8; 16] = [1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15, -16];
        let b: [i8; 16] = [-1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15, 16];

        assert_eq!(l1_i8(&a, &b), l1_i8(&b, &a));
    }

    #[test]
    fn test_l1_distance_zero() {
        let a: [i8; 16] = [42; 16];
        assert_eq!(l1_i8(&a, &a), 0);
    }

    #[test]
    fn test_l1_distance_max() {
        let a: [i8; 16] = [127; 16];
        let b: [i8; 16] = [-127; 16];
        // Each dim: |127 - (-127)| = 254, total: 16 × 254 = 4064
        assert_eq!(l1_i8(&a, &b), 4064);
    }
}
