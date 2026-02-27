//! 3-channel 16,384-bit hypervector for graph HDC (Hyperdimensional Computing).
//!
//! Each entity in the HDC graph memory is encoded as a triplet of 16,384-bit
//! binary vectors (exactly `Fingerprint<256>` = 256 × u64 = 2048 bytes):
//!
//! | Channel | Role              | Usage                                           |
//! |---------|-------------------|-------------------------------------------------|
//! | 0       | Node/Identity     | Random base + position (spatial if needed)      |
//! | 1       | Edge/Relation     | bind(source, dest, label) via XOR + shift        |
//! | 2       | Plastic/State     | Bundled history + decay (majority vote)          |
//!
//! Total: 49,152 bits = 6,144 bytes = 96 AVX-512 registers (perfect packing).
//!
//! ## Key Operations
//!
//! - **Bind** (XOR): associate two HVs — nearly orthogonal to both inputs
//! - **Bundle** (majority vote): superimpose multiple HVs — preserves similarity
//! - **Permute** (circular shift): position encoding — creates orthogonal variants
//! - **Similarity** (normalized Hamming): distance metric across all 3 channels
//!
//! ## Capacity
//!
//! At D=16,384 × 3 channels, the system supports ~200-400 bundled concepts
//! before interference exceeds 5% (well above typical graph neighborhoods).

use crate::fingerprint::Fingerprint;
use crate::rng::SplitMix64;

/// Number of channels in a graph hypervector.
pub const GRAPH_HV_CHANNELS: usize = 3;

/// Total bits across all channels.
pub const GRAPH_HV_BITS: usize = 16_384 * GRAPH_HV_CHANNELS; // 49,152

/// Total bytes across all channels.
pub const GRAPH_HV_BYTES: usize = 2048 * GRAPH_HV_CHANNELS; // 6,144

/// Channel indices for semantic clarity.
pub const CH_NODE: usize = 0;
pub const CH_EDGE: usize = 1;
pub const CH_PLASTIC: usize = 2;

/// A 3-channel 16,384-bit hypervector for graph HDC.
///
/// Each channel is a `Fingerprint<256>` (16,384 bits = 2,048 bytes).
/// Together they form a 49,152-bit representation suitable for encoding
/// graph nodes, edges, and plastic state in a single holographic vector.
#[derive(Clone, PartialEq, Eq)]
pub struct GraphHV {
    pub channels: [Fingerprint<256>; 3],
}

#[allow(clippy::needless_range_loop)]
impl GraphHV {
    /// Zero hypervector (identity element for XOR bind).
    pub fn zero() -> Self {
        Self {
            channels: [
                Fingerprint::zero(),
                Fingerprint::zero(),
                Fingerprint::zero(),
            ],
        }
    }

    /// Generate a random hypervector using the given PRNG.
    pub fn random(rng: &mut SplitMix64) -> Self {
        Self {
            channels: [
                random_fingerprint(rng),
                random_fingerprint(rng),
                random_fingerprint(rng),
            ],
        }
    }

    /// Create from three pre-existing fingerprints (node, edge, plastic).
    pub fn from_channels(
        node: Fingerprint<256>,
        edge: Fingerprint<256>,
        plastic: Fingerprint<256>,
    ) -> Self {
        Self {
            channels: [node, edge, plastic],
        }
    }

    /// XOR bind: associate two hypervectors.
    ///
    /// The result is nearly orthogonal to both inputs (fundamental HDC property).
    /// Bind is self-inverse: `bind(bind(a, b), b) = a`.
    pub fn bind(&self, other: &Self) -> Self {
        Self {
            channels: [
                &self.channels[0] ^ &other.channels[0],
                &self.channels[1] ^ &other.channels[1],
                &self.channels[2] ^ &other.channels[2],
            ],
        }
    }

    /// Circular bit-shift permutation for positional encoding.
    ///
    /// Shifts all bits in each channel left by `shift` positions (circular).
    /// Creates orthogonal variants of the same HV — essential for encoding
    /// position in sequences and graph structures.
    pub fn permute(&self, shift: u32) -> Self {
        Self {
            channels: [
                circular_shift(&self.channels[0], shift),
                circular_shift(&self.channels[1], shift),
                circular_shift(&self.channels[2], shift),
            ],
        }
    }

    /// Hamming distance across all 3 channels (sum of per-channel distances).
    #[inline]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        self.channels[0].hamming_distance(&other.channels[0])
            + self.channels[1].hamming_distance(&other.channels[1])
            + self.channels[2].hamming_distance(&other.channels[2])
    }

    /// Normalized Hamming similarity in [0.0, 1.0] across all channels.
    ///
    /// 1.0 = identical, 0.0 = maximally different.
    #[inline]
    pub fn similarity(&self, other: &Self) -> f64 {
        1.0 - self.hamming_distance(other) as f64 / GRAPH_HV_BITS as f64
    }

    /// Partial Hamming distance on the first `bits` of each channel.
    ///
    /// Used for fast hierarchical descent in DN-tree traversal.
    /// `bits` is clamped to [64, 16384] and rounded up to a multiple of 64.
    #[inline]
    pub fn partial_hamming(&self, other: &Self, bits: usize) -> u32 {
        let words = bits.clamp(64, 16_384).div_ceil(64);
        let words = words.min(256);
        let mut dist = 0u32;
        for ch in 0..3 {
            for w in 0..words {
                dist += (self.channels[ch].words[w] ^ other.channels[ch].words[w]).count_ones();
            }
        }
        dist
    }

    /// Partial similarity on first `bits` per channel, normalized to [0.0, 1.0].
    #[inline]
    pub fn partial_similarity(&self, other: &Self, bits: usize) -> f64 {
        let words = bits.clamp(64, 16_384).div_ceil(64);
        let words = words.min(256);
        let total_bits = words * 64 * 3;
        1.0 - self.partial_hamming(other, bits) as f64 / total_bits as f64
    }

    /// Total popcount across all channels.
    #[inline]
    pub fn popcount(&self) -> u32 {
        self.channels[0].popcount()
            + self.channels[1].popcount()
            + self.channels[2].popcount()
    }

    /// Returns true if all bits in all channels are zero.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.channels[0].is_zero() && self.channels[1].is_zero() && self.channels[2].is_zero()
    }

    /// Probabilistic bit decay on all channels.
    ///
    /// Each set bit survives with probability `keep_prob` (typically 0.96-0.98).
    /// Implements biological LTP/LTD (long-term potentiation/depression).
    ///
    /// Uses fast word-level decay: generates a kill mask by ANDing multiple
    /// random words, then clears those bits from the original.
    pub fn decay(&mut self, keep_prob: f64, rng: &mut SplitMix64) {
        let kill_prob = 1.0 - keep_prob.clamp(0.0, 1.0);
        if kill_prob < 1e-10 {
            return;
        }
        // n_ands random u64s ANDed together: P(bit=1) = (0.5)^n_ands ≈ kill_prob
        let n_ands = (-kill_prob.log2()).round().max(1.0) as u32;

        for ch in 0..3 {
            for w in 0..256 {
                if self.channels[ch].words[w] == 0 {
                    continue;
                }
                let mut kill_mask = rng.next_u64();
                for _ in 1..n_ands {
                    kill_mask &= rng.next_u64();
                }
                // Only clear bits that were already set
                self.channels[ch].words[w] &= !(kill_mask & self.channels[ch].words[w]);
            }
        }
    }
}

/// Majority-vote bundle of multiple graph hypervectors.
///
/// For each bit position across all channels, if more than half the input
/// vectors have the bit set, the output bit is set. Ties are broken randomly
/// by the provided RNG (true HDC randomized tiebreak).
///
/// This is the fundamental "superposition" operation: the result is similar
/// to all inputs, preserving the most common patterns.
pub fn bundle(vectors: &[&GraphHV], rng: &mut SplitMix64) -> GraphHV {
    if vectors.is_empty() {
        return GraphHV::zero();
    }
    if vectors.len() == 1 {
        return vectors[0].clone();
    }

    let n = vectors.len();
    let threshold = n / 2;
    let is_even = n.is_multiple_of(2);
    let mut result = GraphHV::zero();

    for ch in 0..3 {
        for w in 0..256 {
            let mut word = 0u64;
            for bit in 0..64 {
                let mask = 1u64 << bit;
                let count: usize = vectors
                    .iter()
                    .filter(|v| v.channels[ch].words[w] & mask != 0)
                    .count();
                if count > threshold
                    || (is_even && count == threshold && rng.next_u64() & 1 == 1)
                {
                    word |= mask;
                }
            }
            result.channels[ch].words[w] = word;
        }
    }

    result
}

/// Stochastic weighted merge of old summary and new observation.
///
/// For each bit:
/// - If old and new agree, the bit is unchanged
/// - If they disagree, the bit flips to the new value with probability
///   `learning_rate` (= 1 - decay, typically 0.03 for decay=0.97)
///
/// This provides exponential moving average behavior:
/// - learning_rate = 0.03 (decay = 0.97): effective memory window ~33 updates
/// - learning_rate = 0.10: effective memory window ~10 updates
///
/// ## BTSP-Enhanced Mode
///
/// When `btsp_boost > 1.0`, the learning rate is amplified for this update,
/// simulating CaMKII autophosphorylation boost after a stochastic plateau
/// potential. Typical range: 1.0 (normal) to 7.0 (strong BTSP).
pub fn bundle_into(
    summary: &GraphHV,
    new_observation: &GraphHV,
    learning_rate: f64,
    btsp_boost: f64,
    rng: &mut SplitMix64,
) -> GraphHV {
    let effective_lr = (learning_rate * btsp_boost).clamp(0.001, 0.999);
    // n_ands such that (0.5)^n_ands ≈ effective_lr
    let n_ands = (-effective_lr.log2()).round().max(1.0) as u32;

    let mut result = summary.clone();

    for ch in 0..3 {
        for w in 0..256 {
            let disagree = summary.channels[ch].words[w] ^ new_observation.channels[ch].words[w];
            if disagree == 0 {
                continue;
            }
            // Generate flip mask: each bit has ~effective_lr probability of being 1
            let mut flip_mask = rng.next_u64();
            for _ in 1..n_ands {
                flip_mask &= rng.next_u64();
            }
            // Flip only disagreeing bits (with probability ≈ effective_lr)
            result.channels[ch].words[w] ^= disagree & flip_mask;
        }
    }

    result
}

/// Encode a graph edge: `bind(permute(source, shift), dest, role)`.
///
/// Standard GraphHD encoding for directed labeled edges:
/// - Permute source to encode direction (source->dest, not dest->source)
/// - XOR-bind with destination and role label
pub fn encode_edge(
    source: &GraphHV,
    dest: &GraphHV,
    role: &GraphHV,
    shift: u32,
) -> GraphHV {
    source.permute(shift).bind(dest).bind(role)
}

/// Decode/unbind a graph edge component.
///
/// Given an encoded edge and two of the three components (source, dest, role),
/// recover the third via XOR unbinding (XOR is self-inverse).
pub fn decode_edge_source(
    edge: &GraphHV,
    dest: &GraphHV,
    role: &GraphHV,
    shift: u32,
) -> GraphHV {
    // edge = permute(source, shift) ^ dest ^ role
    // permute(source, shift) = edge ^ dest ^ role
    // source = permute_inverse(edge ^ dest ^ role, shift)
    let shifted_source = edge.bind(dest).bind(role);
    // Inverse permute = shift by (BITS - shift)
    let inverse_shift = (16_384 - (shift % 16_384) as usize) as u32;
    shifted_source.permute(inverse_shift)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Generate a random `Fingerprint<256>` from the PRNG.
fn random_fingerprint(rng: &mut SplitMix64) -> Fingerprint<256> {
    let mut words = [0u64; 256];
    for w in words.iter_mut() {
        *w = rng.next_u64();
    }
    Fingerprint::from_words(words)
}

/// Circular left-shift a `Fingerprint<256>` by `shift` bit positions.
///
/// Bits shifted past the MSB wrap around to the LSB.
#[allow(clippy::needless_range_loop)]
fn circular_shift(fp: &Fingerprint<256>, shift: u32) -> Fingerprint<256> {
    let total_bits = 256 * 64;
    let shift = (shift as usize) % total_bits;
    if shift == 0 {
        return fp.clone();
    }

    let word_shift = shift / 64;
    let bit_shift = shift % 64;
    let mut words = [0u64; 256];

    if bit_shift == 0 {
        // Pure word rotation
        for i in 0..256 {
            words[(i + word_shift) % 256] = fp.words[i];
        }
    } else {
        // Word rotation + intra-word bit shift
        for i in 0..256 {
            let dst = (i + word_shift) % 256;
            let dst_next = (dst + 1) % 256;
            words[dst] |= fp.words[i] << bit_shift;
            words[dst_next] |= fp.words[i] >> (64 - bit_shift);
        }
    }

    Fingerprint::from_words(words)
}

impl std::fmt::Debug for GraphHV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GraphHV[node_pop={}, edge_pop={}, plastic_pop={}]",
            self.channels[0].popcount(),
            self.channels[1].popcount(),
            self.channels[2].popcount(),
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> SplitMix64 {
        SplitMix64::new(42)
    }

    #[test]
    fn test_zero_identity_bind() {
        let mut rng = make_rng();
        let a = GraphHV::random(&mut rng);
        let zero = GraphHV::zero();
        assert_eq!(a.bind(&zero), a);
    }

    #[test]
    fn test_bind_self_inverse() {
        let mut rng = make_rng();
        let a = GraphHV::random(&mut rng);
        let result = a.bind(&a);
        assert!(result.is_zero());
    }

    #[test]
    fn test_bind_commutative() {
        let mut rng = make_rng();
        let a = GraphHV::random(&mut rng);
        let b = GraphHV::random(&mut rng);
        assert_eq!(a.bind(&b), b.bind(&a));
    }

    #[test]
    fn test_bind_associative() {
        let mut rng = make_rng();
        let a = GraphHV::random(&mut rng);
        let b = GraphHV::random(&mut rng);
        let c = GraphHV::random(&mut rng);
        let ab_c = a.bind(&b).bind(&c);
        let a_bc = a.bind(&b.bind(&c));
        assert_eq!(ab_c, a_bc);
    }

    #[test]
    fn test_permute_preserves_popcount() {
        let mut rng = make_rng();
        let a = GraphHV::random(&mut rng);
        let shifted = a.permute(7);
        assert_eq!(a.popcount(), shifted.popcount());
    }

    #[test]
    fn test_permute_identity() {
        let mut rng = make_rng();
        let a = GraphHV::random(&mut rng);
        assert_eq!(a.permute(0), a);
    }

    #[test]
    fn test_permute_full_rotation() {
        let mut rng = make_rng();
        let a = GraphHV::random(&mut rng);
        // Full rotation by total bits should return to original
        assert_eq!(a.permute(16_384), a);
    }

    #[test]
    fn test_permute_creates_near_orthogonal() {
        let mut rng = make_rng();
        let a = GraphHV::random(&mut rng);
        let shifted = a.permute(7);
        // Shifted version should be ~50% similar (near-orthogonal)
        let sim = a.similarity(&shifted);
        assert!(
            (sim - 0.5).abs() < 0.05,
            "Expected ~0.50 similarity, got {:.4}",
            sim
        );
    }

    #[test]
    fn test_hamming_self_zero() {
        let mut rng = make_rng();
        let a = GraphHV::random(&mut rng);
        assert_eq!(a.hamming_distance(&a), 0);
    }

    #[test]
    fn test_random_near_orthogonal() {
        let mut rng = make_rng();
        let a = GraphHV::random(&mut rng);
        let b = GraphHV::random(&mut rng);
        // Two random HVs should be ~50% similar
        let sim = a.similarity(&b);
        assert!(
            (sim - 0.5).abs() < 0.02,
            "Expected ~0.50 similarity, got {:.4}",
            sim
        );
    }

    #[test]
    fn test_partial_hamming_subset_of_full() {
        let mut rng = make_rng();
        let a = GraphHV::random(&mut rng);
        let b = GraphHV::random(&mut rng);
        let partial = a.partial_hamming(&b, 1024);
        let full = a.hamming_distance(&b);
        assert!(partial <= full);
    }

    #[test]
    fn test_bundle_majority() {
        let mut rng = make_rng();
        let a = GraphHV::random(&mut rng);
        let b = GraphHV::random(&mut rng);
        let c = GraphHV::random(&mut rng);

        // Bundle 3 vectors: result should be more similar to each than random
        let bundled = bundle(&[&a, &b, &c], &mut rng);
        let sim_a = bundled.similarity(&a);
        let sim_b = bundled.similarity(&b);
        let sim_c = bundled.similarity(&c);

        // Each should be > 0.5 (more similar than random)
        assert!(
            sim_a > 0.55,
            "Bundle should be similar to input a: {:.4}",
            sim_a
        );
        assert!(
            sim_b > 0.55,
            "Bundle should be similar to input b: {:.4}",
            sim_b
        );
        assert!(
            sim_c > 0.55,
            "Bundle should be similar to input c: {:.4}",
            sim_c
        );
    }

    #[test]
    fn test_bundle_single_identity() {
        let mut rng = make_rng();
        let a = GraphHV::random(&mut rng);
        let bundled = bundle(&[&a], &mut rng);
        assert_eq!(bundled, a);
    }

    #[test]
    fn test_bundle_into_learns() {
        let mut rng = make_rng();
        let old_summary = GraphHV::random(&mut rng);
        let new_obs = GraphHV::random(&mut rng);

        // After stochastic merge with learning_rate=0.5, result should be
        // somewhat between old and new
        let merged = bundle_into(&old_summary, &new_obs, 0.5, 1.0, &mut rng);

        let sim_old = merged.similarity(&old_summary);
        let sim_new = merged.similarity(&new_obs);
        // Both should be > 0.5 (closer than random)
        assert!(
            sim_old > 0.55,
            "Merged should retain old info: {:.4}",
            sim_old
        );
        assert!(
            sim_new > 0.55,
            "Merged should incorporate new info: {:.4}",
            sim_new
        );
    }

    #[test]
    fn test_bundle_into_low_lr_preserves_old() {
        let mut rng = make_rng();
        let old_summary = GraphHV::random(&mut rng);
        let new_obs = GraphHV::random(&mut rng);

        // Very low learning rate: almost no change
        let merged = bundle_into(&old_summary, &new_obs, 0.01, 1.0, &mut rng);
        let sim_old = merged.similarity(&old_summary);
        assert!(
            sim_old > 0.97,
            "Low LR should preserve old summary: {:.4}",
            sim_old
        );
    }

    #[test]
    fn test_bundle_into_btsp_boost() {
        let mut rng = SplitMix64::new(123);
        let old_summary = GraphHV::random(&mut rng);
        let new_obs = GraphHV::random(&mut rng);

        // Normal update
        let mut rng1 = SplitMix64::new(999);
        let normal = bundle_into(&old_summary, &new_obs, 0.03, 1.0, &mut rng1);

        // BTSP-boosted update (7x learning rate)
        let mut rng2 = SplitMix64::new(999);
        let boosted = bundle_into(&old_summary, &new_obs, 0.03, 7.0, &mut rng2);

        // Boosted should be more similar to new observation
        let sim_normal = normal.similarity(&new_obs);
        let sim_boosted = boosted.similarity(&new_obs);
        assert!(
            sim_boosted > sim_normal,
            "BTSP boost should increase learning: normal={:.4}, boosted={:.4}",
            sim_normal,
            sim_boosted
        );
    }

    #[test]
    fn test_decay_reduces_popcount() {
        let mut rng = make_rng();
        let mut a = GraphHV::random(&mut rng);
        let original_pop = a.popcount();

        a.decay(0.90, &mut rng); // aggressive decay
        let decayed_pop = a.popcount();
        assert!(
            decayed_pop < original_pop,
            "Decay should reduce popcount: {} -> {}",
            original_pop,
            decayed_pop
        );
    }

    #[test]
    fn test_edge_encoding_roundtrip() {
        let mut rng = make_rng();
        let source = GraphHV::random(&mut rng);
        let dest = GraphHV::random(&mut rng);
        let role = GraphHV::random(&mut rng);
        let shift = 7;

        let encoded = encode_edge(&source, &dest, &role, shift);

        // Recover source via unbinding
        let recovered = decode_edge_source(&encoded, &dest, &role, shift);
        assert_eq!(recovered, source);
    }

    #[test]
    fn test_edge_encoding_orthogonal_to_components() {
        let mut rng = make_rng();
        let source = GraphHV::random(&mut rng);
        let dest = GraphHV::random(&mut rng);
        let role = GraphHV::random(&mut rng);

        let encoded = encode_edge(&source, &dest, &role, 7);

        // Encoded edge should be ~orthogonal to each component
        assert!(
            (encoded.similarity(&source) - 0.5).abs() < 0.05,
            "Edge should be orthogonal to source"
        );
        assert!(
            (encoded.similarity(&dest) - 0.5).abs() < 0.05,
            "Edge should be orthogonal to dest"
        );
    }

    #[test]
    fn test_similarity_range() {
        let mut rng = make_rng();
        let a = GraphHV::random(&mut rng);
        let b = GraphHV::random(&mut rng);
        let sim = a.similarity(&b);
        assert!((0.0..=1.0).contains(&sim));
        assert!((a.similarity(&a) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_constants() {
        assert_eq!(GRAPH_HV_BITS, 49_152);
        assert_eq!(GRAPH_HV_BYTES, 6_144);
        assert_eq!(GRAPH_HV_CHANNELS, 3);
    }

    #[test]
    fn test_circular_shift_word_boundary() {
        // Test shift by exactly 64 bits (one word)
        let mut rng = make_rng();
        let a = GraphHV::random(&mut rng);
        let shifted = a.permute(64);
        assert_eq!(a.popcount(), shifted.popcount());
        assert_ne!(a, shifted);
    }

    #[test]
    fn test_circular_shift_inverse() {
        let mut rng = make_rng();
        let a = GraphHV::random(&mut rng);
        let shift = 137u32;
        let shifted = a.permute(shift);
        let back = shifted.permute(16_384 - shift);
        assert_eq!(a, back);
    }
}
