//! BF16-structured Hamming distance.
//!
//! Operates on raw byte pairs interpreted as BF16 (16-bit brain float).
//! XOR + masked popcount with per-field weighting:
//!   sign (bit 15): weight 256
//!   exponent (bits 14-7): weight 16 per flipped bit
//!   mantissa (bits 6-0): weight 1 per flipped bit
//!
//! Storage: 2 bytes per dimension (same as BF16).
//! For 1024-D Jina embedding: 2KB per vector.

use smallvec::SmallVec;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/// BF16 field weights.
/// These can be tuned for specific embedding models.
#[derive(Clone, Copy, Debug)]
pub struct BF16Weights {
    /// Weight for sign flip (default: 256).
    pub sign: u16,
    /// Weight per exponent bit flip (default: 16).
    pub exponent: u16,
    /// Weight per mantissa bit flip (default: 1).
    pub mantissa: u16,
}

impl BF16Weights {
    /// Create custom weights with overflow validation.
    ///
    /// The AVX-512 path accumulates `sign + 8×exponent + 7×mantissa` per BF16
    /// pair in a u16 lane before widening to u32. This must not exceed 65535.
    ///
    /// Panics if the per-element maximum exceeds u16 range.
    pub fn new(sign: u16, exponent: u16, mantissa: u16) -> Self {
        let max_per_elem = sign as u32 + 8 * exponent as u32 + 7 * mantissa as u32;
        assert!(
            max_per_elem <= 65535,
            "BF16Weights overflow: sign({}) + 8×exp({}) + 7×man({}) = {} > 65535. \
             The AVX-512 path would silently wrap in u16 lanes.",
            sign,
            exponent,
            mantissa,
            max_per_elem,
        );
        Self {
            sign,
            exponent,
            mantissa,
        }
    }
}

impl Default for BF16Weights {
    fn default() -> Self {
        // 256 + 8×16 + 7×1 = 391 — safe
        Self {
            sign: 256,
            exponent: 16,
            mantissa: 1,
        }
    }
}

impl PartialEq for BF16Weights {
    fn eq(&self, other: &Self) -> bool {
        self.sign == other.sign
            && self.exponent == other.exponent
            && self.mantissa == other.mantissa
    }
}

impl Eq for BF16Weights {}

/// Jina-optimized weights.
/// Jina v3 embeddings are normalized, so sign flips are the strongest signal.
/// Exponent changes are rare in normalized vectors (most values have similar magnitude).
/// Mantissa captures fine-grained similarity.
pub const JINA_WEIGHTS: BF16Weights = BF16Weights {
    sign: 256,
    exponent: 32,
    mantissa: 1,
};

/// Training-optimized weights.
/// During learning, we care about WHAT changed:
/// - Sign flips = dimension polarity changed (class-level signal)
/// - Exponent shifts = feature emphasis changed (attention-level signal)
/// - Mantissa changes = gradient noise (ignore)
pub const TRAINING_WEIGHTS: BF16Weights = BF16Weights {
    sign: 1024,
    exponent: 64,
    mantissa: 0,
};

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------

/// Scalar BF16-structured Hamming distance.
///
/// `a` and `b` are byte slices of BF16 values (2 bytes per dimension, little-endian).
/// Returns weighted distance.
pub fn bf16_hamming_scalar(a: &[u8], b: &[u8], weights: &BF16Weights) -> u64 {
    assert_eq!(a.len(), b.len());
    assert!(
        a.len().is_multiple_of(2),
        "BF16 data must be even number of bytes"
    );

    let mut total: u64 = 0;

    for i in (0..a.len()).step_by(2) {
        let va = u16::from_le_bytes([a[i], a[i + 1]]);
        let vb = u16::from_le_bytes([b[i], b[i + 1]]);
        let xor = va ^ vb;

        // Sign bit (bit 15)
        let sign_diff = (xor >> 15) & 1;

        // Exponent bits (bits 14-7)
        let exp_diff = (xor >> 7) & 0xFF;
        let exp_popcount = exp_diff.count_ones() as u16;

        // Mantissa bits (bits 6-0)
        let man_diff = xor & 0x7F;
        let man_popcount = man_diff.count_ones() as u16;

        total += (sign_diff * weights.sign
            + exp_popcount * weights.exponent
            + man_popcount * weights.mantissa) as u64;
    }

    total
}

// ---------------------------------------------------------------------------
// AVX-512 implementation
// ---------------------------------------------------------------------------

/// AVX-512 BF16-structured Hamming distance.
///
/// Processes 32 BF16 pairs (64 bytes) per iteration using:
/// - VPXORD for XOR
/// - VPSRLW + VPANDQ for field extraction
/// - VPOPCNTB for per-byte popcount
/// - VPADDW for accumulation
///
/// Requires: AVX-512BW + AVX-512BITALG (Ice Lake+)
///
/// Note: `_mm512_popcnt_epi8` is AVX-512 BITALG, not VPOPCNTDQ.
/// VPOPCNTDQ provides 32/64-bit lane popcount; per-byte popcount is BITALG.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512bitalg")]
unsafe fn bf16_hamming_avx512(a: &[u8], b: &[u8], weights: &BF16Weights) -> u64 {
    use std::arch::x86_64::*;

    let len = a.len();
    assert_eq!(len, b.len());
    let chunks = len / 64;

    let w_sign = _mm512_set1_epi16(weights.sign as i16);
    let w_exp = _mm512_set1_epi16(weights.exponent as i16);
    let w_man = _mm512_set1_epi16(weights.mantissa as i16);
    let mask_0xff = _mm512_set1_epi16(0x00FF);
    let mask_0x7f = _mm512_set1_epi16(0x007F);
    let one = _mm512_set1_epi16(1);

    // Accumulate in 32-bit to avoid u16 overflow.
    //
    // Per-element max (default weights): 256 + 8×16 + 7×1 = 391
    // 32 BF16 pairs per 512-bit chunk → 12,512 per chunk.
    // For 1024-D (32 chunks): 32 × 12,512 = 400,384 — fits u32 (max 4.29B).
    // Even with TRAINING_WEIGHTS (1024/64/0): 1024 + 8×64 = 1,536 per elem,
    // 32 × 1,536 × 32 chunks = 1,572,864 — still fits comfortably in u32.
    //
    // Overflow budget: safe for vectors up to ~2.7M dimensions with default
    // weights, or ~87K dimensions with TRAINING_WEIGHTS. Beyond that, widen
    // to u64 accumulators or add periodic reduction.
    let mut acc_lo = _mm512_setzero_si512();
    let mut acc_hi = _mm512_setzero_si512();

    for c in 0..chunks {
        let base = c * 64;
        let va = _mm512_loadu_si512(a.as_ptr().add(base) as *const __m512i);
        let vb = _mm512_loadu_si512(b.as_ptr().add(base) as *const __m512i);
        let xor = _mm512_xor_si512(va, vb);

        // Sign extraction: shift right 15, mask to 1, multiply by weight
        let signs = _mm512_and_si512(_mm512_srli_epi16(xor, 15), one);
        let sign_weighted = _mm512_mullo_epi16(signs, w_sign);

        // Exponent extraction: shift right 7, mask to 0xFF, popcount, weight
        let exp_shifted = _mm512_and_si512(_mm512_srli_epi16(xor, 7), mask_0xff);
        let exp_popcnt = _mm512_and_si512(_mm512_popcnt_epi8(exp_shifted), mask_0xff);
        let exp_weighted = _mm512_mullo_epi16(exp_popcnt, w_exp);

        // Mantissa extraction: mask to 0x7F, popcount, weight
        let man_masked = _mm512_and_si512(xor, mask_0x7f);
        let man_popcnt = _mm512_and_si512(_mm512_popcnt_epi8(man_masked), mask_0xff);
        let man_weighted = _mm512_mullo_epi16(man_popcnt, w_man);

        // Sum per-element: sign + exp + man (u16, max ~391 with default weights)
        let per_elem =
            _mm512_add_epi16(_mm512_add_epi16(sign_weighted, exp_weighted), man_weighted);

        // Widen u16 lanes to u32 and accumulate in two halves.
        // BF16 pairs are 2 bytes each so u16 lane boundaries coincide with
        // BF16 element boundaries — even lanes are elements 0,2,4,...
        // and odd lanes (shifted right 16 within each 32-bit word) are 1,3,5,...
        let even = _mm512_and_si512(per_elem, _mm512_set1_epi32(0x0000FFFF));
        let odd = _mm512_srli_epi32(per_elem, 16);
        acc_lo = _mm512_add_epi32(acc_lo, even);
        acc_hi = _mm512_add_epi32(acc_hi, odd);
    }

    // Horizontal sum
    let sum_lo = _mm512_reduce_add_epi32(acc_lo) as u64;
    let sum_hi = _mm512_reduce_add_epi32(acc_hi) as u64;
    let mut total = sum_lo + sum_hi;

    // Scalar tail for remaining bytes
    let tail_start = chunks * 64;
    if tail_start < len {
        total += bf16_hamming_scalar(&a[tail_start..], &b[tail_start..], weights);
    }

    total
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// Function pointer type for BF16 Hamming implementations.
pub type BF16HammingFn = fn(&[u8], &[u8], &BF16Weights) -> u64;

/// Select the fastest BF16-structured Hamming implementation for this CPU.
///
/// The result is cached in a `OnceLock` — the CPUID probe runs at most once.
/// Safe to call in hot loops.
pub fn select_bf16_hamming_fn() -> BF16HammingFn {
    static FN: OnceLock<BF16HammingFn> = OnceLock::new();
    *FN.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512bitalg") {
                // SAFETY: CPU feature detection guarantees AVX-512BW + BITALG are available.
                // The closure delegates to bf16_hamming_avx512 which requires these features.
                return |a, b, w| unsafe { bf16_hamming_avx512(a, b, w) };
            }
        }
        bf16_hamming_scalar
    })
}

// ---------------------------------------------------------------------------
// FP32 ↔ BF16 truncation
// ---------------------------------------------------------------------------

/// Truncate FP32 slice to BF16 bytes (little-endian).
///
/// This is the "quantization" step — but it's just dropping the low 16 bits
/// of each f32. No scale, no zero_point, no calibration.
///
/// Output: 2 bytes per float, half the size of input.
pub fn fp32_to_bf16_bytes(floats: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(floats.len() * 2);
    for &f in floats {
        let bits = f.to_bits();
        let bf16 = (bits >> 16) as u16;
        out.extend_from_slice(&bf16.to_le_bytes());
    }
    out
}

/// Widen BF16 bytes back to FP32.
///
/// Lossless in the BF16 → FP32 direction (just adds zero mantissa bits).
pub fn bf16_bytes_to_fp32(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len().is_multiple_of(2));
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let bf16 = u16::from_le_bytes([chunk[0], chunk[1]]);
        let bits = (bf16 as u32) << 16;
        out.push(f32::from_bits(bits));
    }
    out
}

// ---------------------------------------------------------------------------
// Structural diff — for learning/training
// ---------------------------------------------------------------------------

/// Structural diff between two BF16 vectors.
///
/// Returns per-dimension breakdown of what changed:
/// sign flips, exponent changes, mantissa changes.
/// This is the "structural gradient" — actionable learning signal.
///
/// Uses `SmallVec<[usize; 32]>` for dimension indices to avoid heap allocation
/// in the common case (≤32 sign flips or magnitude shifts per diff).
#[derive(Clone, Debug, Default)]
pub struct BF16StructuralDiff {
    /// Number of dimensions where the sign flipped.
    pub sign_flips: usize,
    /// Number of exponent bits that changed (total across all dims).
    pub exponent_bits_changed: usize,
    /// Number of mantissa bits that changed (total across all dims).
    pub mantissa_bits_changed: usize,
    /// Indices of dimensions where sign flipped.
    pub sign_flip_dims: SmallVec<[usize; 32]>,
    /// Indices of dimensions with exponent change >= 2 bits (magnitude shift >= 4x).
    pub major_magnitude_shifts: SmallVec<[usize; 32]>,
}

pub fn structural_diff(a: &[u8], b: &[u8]) -> BF16StructuralDiff {
    assert_eq!(a.len(), b.len());
    assert!(a.len().is_multiple_of(2));

    let mut diff = BF16StructuralDiff::default();
    let n_dims = a.len() / 2;

    for dim in 0..n_dims {
        let i = dim * 2;
        let va = u16::from_le_bytes([a[i], a[i + 1]]);
        let vb = u16::from_le_bytes([b[i], b[i + 1]]);
        let xor = va ^ vb;

        // Sign
        if xor & 0x8000 != 0 {
            diff.sign_flips += 1;
            diff.sign_flip_dims.push(dim);
        }

        // Exponent
        let exp_diff = ((xor >> 7) & 0xFF).count_ones() as usize;
        diff.exponent_bits_changed += exp_diff;
        if exp_diff >= 2 {
            diff.major_magnitude_shifts.push(dim);
        }

        // Mantissa
        diff.mantissa_bits_changed += (xor & 0x7F).count_ones() as usize;
    }

    diff
}

// ---------------------------------------------------------------------------
// Awareness substrate — 4-state superposition decomposition
// ---------------------------------------------------------------------------

/// Four-state awareness classification per dimension.
/// Encodes as 2 bits: 00=Crystallized, 01=Tensioned, 10=Uncertain, 11=Noise
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AwarenessState {
    /// All vectors agree on sign, low exponent spread. Settled knowledge.
    Crystallized = 0,
    /// Sign disagreement. Active contradiction — hold both.
    Tensioned = 1,
    /// All agree on sign, high exponent spread. Direction known, intensity unknown.
    Uncertain = 2,
    /// Only mantissa differs. Irrelevant noise — mask out.
    Noise = 3,
}

/// Result of decomposing 2-3 superposed BF16 vectors.
/// Per-dimension sign consensus, exponent spread, and 4-state classification.
#[derive(Clone, Debug)]
pub struct SuperpositionState {
    /// Number of dimensions
    pub n_dims: usize,
    /// Per-dimension sign consensus ratio: 1.0 = unanimous, 0.5 = split
    /// Stored as u8 quantized: 0=0.0, 255=1.0
    pub sign_consensus: Vec<u8>,
    /// Per-dimension exponent spread (XOR popcount across pairs, max 8)
    pub exp_spread: Vec<u8>,
    /// Per-dimension mantissa noise flag (true = mantissa disagrees)
    pub mantissa_noise: Vec<bool>,
    /// Per-dimension 4-state classification
    pub states: Vec<AwarenessState>,
    /// Packed 2-bit-per-dimension state vector (ceil(N/4) bytes)
    pub packed_states: Vec<u8>,
    /// Aggregate statistics
    pub crystallized_pct: f32,
    pub tensioned_pct: f32,
    pub uncertain_pct: f32,
    pub noise_pct: f32,
}

/// Thresholds for 4-state awareness classification.
#[derive(Clone, Copy, Debug)]
pub struct AwarenessThresholds {
    /// Sign consensus ratio below this = tensioned (default: 0.75 = 192/255)
    pub sign_consensus_threshold: u8,
    /// Exponent spread above this = uncertain (default: 2 bits)
    pub exp_spread_threshold: u8,
    /// Mantissa popcount above this = noise (default: 1 bit)
    pub mantissa_noise_threshold: u8,
}

impl Default for AwarenessThresholds {
    fn default() -> Self {
        Self {
            sign_consensus_threshold: 192,
            exp_spread_threshold: 2,
            mantissa_noise_threshold: 1,
        }
    }
}

/// Pack awareness states into a 2-bit-per-dimension byte vector.
/// 4 states per byte, MSB first:
/// `byte = (state[4i] << 6) | (state[4i+1] << 4) | (state[4i+2] << 2) | state[4i+3]`
pub fn pack_awareness_states(states: &[AwarenessState]) -> Vec<u8> {
    let n_bytes = states.len().div_ceil(4);
    let mut packed = vec![0u8; n_bytes];
    for (i, &s) in states.iter().enumerate() {
        let byte_idx = i / 4;
        let shift = 6 - (i % 4) * 2;
        packed[byte_idx] |= (s as u8) << shift;
    }
    packed
}

/// Unpack 2-bit-per-dimension packed states back to AwarenessState vector.
pub fn unpack_awareness_states(packed: &[u8], n_dims: usize) -> Vec<AwarenessState> {
    let mut states = Vec::with_capacity(n_dims);
    for i in 0..n_dims {
        let byte_idx = i / 4;
        let shift = 6 - (i % 4) * 2;
        let val = (packed[byte_idx] >> shift) & 0x03;
        states.push(match val {
            0 => AwarenessState::Crystallized,
            1 => AwarenessState::Tensioned,
            2 => AwarenessState::Uncertain,
            3 => AwarenessState::Noise,
            _ => unreachable!(),
        });
    }
    states
}

/// Decompose 2-3 superposed BF16 vectors into per-dimension awareness states.
///
/// For each dimension, classifies the relationship between the vectors:
/// - **Crystallized**: sign agreement + low exponent spread → settled knowledge
/// - **Tensioned**: sign disagreement → active contradiction, hold both
/// - **Uncertain**: sign agreement + high exponent spread → direction known, intensity unknown
/// - **Noise**: only mantissa differs → irrelevant noise, mask out
pub fn superposition_decompose(
    vectors: &[&[u8]],
    thresholds: &AwarenessThresholds,
) -> SuperpositionState {
    let n_vecs = vectors.len();
    assert!(
        (2..=3).contains(&n_vecs),
        "superposition_decompose requires 2-3 vectors, got {}",
        n_vecs
    );

    let byte_len = vectors[0].len();
    for v in vectors.iter() {
        assert_eq!(
            v.len(),
            byte_len,
            "All vectors must have the same byte length"
        );
    }
    assert!(
        byte_len.is_multiple_of(2),
        "BF16 data must be an even number of bytes"
    );

    let n_dims = byte_len / 2;
    let n_pairs = n_vecs * (n_vecs - 1) / 2; // 1 for N=2, 3 for N=3

    let mut sign_consensus = Vec::with_capacity(n_dims);
    let mut exp_spread = Vec::with_capacity(n_dims);
    let mut mantissa_noise = Vec::with_capacity(n_dims);
    let mut states = Vec::with_capacity(n_dims);

    for d in 0..n_dims {
        let offset = d * 2;

        // Extract u16 for each vector at this dimension
        let vals: SmallVec<[u16; 3]> = vectors
            .iter()
            .map(|v| u16::from_le_bytes([v[offset], v[offset + 1]]))
            .collect();

        // Compute pairwise metrics
        let mut sign_agreements: u32 = 0;
        let mut max_exp_popcount: u8 = 0;
        let mut has_mantissa_noise = false;

        for i in 0..n_vecs {
            for j in (i + 1)..n_vecs {
                let xor = vals[i] ^ vals[j];

                // Sign bit (bit 15): do they agree?
                if xor & 0x8000 == 0 {
                    sign_agreements += 1;
                }

                // Exponent (bits 14-7): XOR popcount
                let exp_xor = (xor >> 7) & 0xFF;
                let exp_pc = exp_xor.count_ones() as u8;
                if exp_pc > max_exp_popcount {
                    max_exp_popcount = exp_pc;
                }

                // Mantissa (bits 6-0): XOR popcount > threshold?
                let man_xor = xor & 0x7F;
                let man_pc = man_xor.count_ones() as u8;
                if man_pc > thresholds.mantissa_noise_threshold {
                    has_mantissa_noise = true;
                }
            }
        }

        // Sign consensus: map agreements/n_pairs to 0..255
        let consensus: u8 = if n_pairs == 1 {
            if sign_agreements == 1 {
                255
            } else {
                0
            }
        } else {
            // n_pairs == 3
            match sign_agreements {
                3 => 255,
                2 => 170,
                1 => 85,
                0 => 0,
                _ => unreachable!(),
            }
        };

        // Classify
        let state = if consensus < thresholds.sign_consensus_threshold {
            AwarenessState::Tensioned
        } else if max_exp_popcount > thresholds.exp_spread_threshold {
            AwarenessState::Uncertain
        } else if has_mantissa_noise && max_exp_popcount == 0 && consensus == 255 {
            AwarenessState::Noise
        } else {
            AwarenessState::Crystallized
        };

        sign_consensus.push(consensus);
        exp_spread.push(max_exp_popcount);
        mantissa_noise.push(has_mantissa_noise);
        states.push(state);
    }

    // Pack states
    let packed_states = pack_awareness_states(&states);

    // Aggregate percentages
    let mut counts = [0u32; 4];
    for &s in &states {
        counts[s as usize] += 1;
    }
    let n = n_dims as f32;
    let crystallized_pct = counts[0] as f32 / n;
    let tensioned_pct = counts[1] as f32 / n;
    let uncertain_pct = counts[2] as f32 / n;
    let noise_pct = counts[3] as f32 / n;

    SuperpositionState {
        n_dims,
        sign_consensus,
        exp_spread,
        mantissa_noise,
        states,
        packed_states,
        crystallized_pct,
        tensioned_pct,
        uncertain_pct,
        noise_pct,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identical_vectors_distance_zero() {
        let a = fp32_to_bf16_bytes(&[1.0, -2.0, 0.5, 100.0]);
        let d = bf16_hamming_scalar(&a, &a, &BF16Weights::default());
        assert_eq!(d, 0);
    }

    #[test]
    fn test_sign_flip_dominates() {
        let a = fp32_to_bf16_bytes(&[1.0]);
        let b = fp32_to_bf16_bytes(&[-1.0]);
        let w = BF16Weights::default();
        let d = bf16_hamming_scalar(&a, &b, &w);
        // Sign flip (1 bit * 256) + exponent same + mantissa same
        assert!(
            d >= 256,
            "Sign flip should contribute at least 256, got {}",
            d
        );
    }

    #[test]
    fn test_small_magnitude_change_is_cheap() {
        let a = fp32_to_bf16_bytes(&[1.0]);
        let b = fp32_to_bf16_bytes(&[1.001]);
        let w = BF16Weights::default();
        let d = bf16_hamming_scalar(&a, &b, &w);
        // Only mantissa bits differ — should be very small
        assert!(
            d < 8,
            "Small float change should be < 8 weighted distance, got {}",
            d
        );
    }

    #[test]
    fn test_large_magnitude_change_is_medium() {
        let a = fp32_to_bf16_bytes(&[1.0]);
        let b = fp32_to_bf16_bytes(&[256.0]);
        let w = BF16Weights::default();
        let d = bf16_hamming_scalar(&a, &b, &w);
        // Exponent changed significantly, no sign flip
        assert!(
            d > 16 && d < 256,
            "Large magnitude should be between exp and sign weight, got {}",
            d
        );
    }

    #[test]
    fn test_fp32_bf16_roundtrip() {
        let orig = vec![1.0f32, -0.5, 100.0, 0.001, -0.0];
        let bf16 = fp32_to_bf16_bytes(&orig);
        let back = bf16_bytes_to_fp32(&bf16);
        for (o, b) in orig.iter().zip(back.iter()) {
            if o.is_finite() && *o != 0.0 {
                let rel_err = ((o - b) / o).abs();
                assert!(
                    rel_err < 0.01,
                    "Roundtrip error too large: {} -> {} (err {})",
                    o,
                    b,
                    rel_err
                );
            }
        }
    }

    #[test]
    fn test_fp32_bf16_roundtrip_special() {
        // Test special values
        let orig = vec![0.0f32, -0.0, f32::INFINITY, f32::NEG_INFINITY];
        let bf16 = fp32_to_bf16_bytes(&orig);
        let back = bf16_bytes_to_fp32(&bf16);
        assert_eq!(back[0].to_bits(), 0.0f32.to_bits()); // +0
        assert!(back[2].is_infinite() && back[2] > 0.0); // +inf
        assert!(back[3].is_infinite() && back[3] < 0.0); // -inf
    }

    #[test]
    fn test_structural_diff_detects_sign_flip() {
        let a = fp32_to_bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let b = fp32_to_bf16_bytes(&[1.0, -2.0, 3.0, -4.0]);
        let diff = structural_diff(&a, &b);
        assert_eq!(diff.sign_flips, 2);
        assert_eq!(diff.sign_flip_dims.as_slice(), &[1, 3]);
    }

    #[test]
    fn test_structural_diff_identical() {
        let a = fp32_to_bf16_bytes(&[1.0, 2.0, 3.0]);
        let diff = structural_diff(&a, &a);
        assert_eq!(diff.sign_flips, 0);
        assert_eq!(diff.exponent_bits_changed, 0);
        assert_eq!(diff.mantissa_bits_changed, 0);
    }

    #[test]
    fn test_jina_1024d_smoke() {
        // Smoke test: 1024-D Jina-like embeddings
        let a: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.01 + 0.001).sin()).collect();
        let a_bf16 = fp32_to_bf16_bytes(&a);
        let b_bf16 = fp32_to_bf16_bytes(&b);
        let d = bf16_hamming_scalar(&a_bf16, &b_bf16, &JINA_WEIGHTS);
        // Very similar vectors — distance should be small relative to max
        let max_possible: u64 = 1024 * (256 + 8 * 32 + 7);
        assert!(
            d < max_possible / 10,
            "Similar vectors should be < 10% of max distance, got {}",
            d
        );
    }

    #[test]
    fn test_training_weights_ignore_mantissa() {
        let a = fp32_to_bf16_bytes(&[1.0]);
        let b = fp32_to_bf16_bytes(&[1.007]);
        let d = bf16_hamming_scalar(&a, &b, &TRAINING_WEIGHTS);
        assert_eq!(
            d, 0,
            "Training weights with mantissa=0 should ignore mantissa-only changes"
        );
    }

    #[test]
    fn test_dispatch_selects_function() {
        let f = select_bf16_hamming_fn();
        let a = fp32_to_bf16_bytes(&[1.0, -2.0, 3.0, 4.0]);
        let b = fp32_to_bf16_bytes(&[1.0, 2.0, 3.0, -4.0]);
        let w = BF16Weights::default();
        let d = f(&a, &b, &w);
        let d_scalar = bf16_hamming_scalar(&a, &b, &w);
        assert_eq!(d, d_scalar, "Dispatched function must match scalar");
    }

    #[test]
    fn test_avx512_matches_scalar() {
        #[cfg(target_arch = "x86_64")]
        {
            if !is_x86_feature_detected!("avx512bw") || !is_x86_feature_detected!("avx512bitalg") {
                return;
            }
            // 512 dims = 1024 bytes = 16 full AVX-512 chunks, no tail
            let a: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).sin()).collect();
            let b: Vec<f32> = (0..512).map(|i| (i as f32 * 0.1).cos()).collect();
            let a_bf16 = fp32_to_bf16_bytes(&a);
            let b_bf16 = fp32_to_bf16_bytes(&b);
            let w = BF16Weights::default();
            let scalar = bf16_hamming_scalar(&a_bf16, &b_bf16, &w);
            let simd = unsafe { bf16_hamming_avx512(&a_bf16, &b_bf16, &w) };
            assert_eq!(scalar, simd, "AVX-512 result must exactly match scalar");
        }
    }

    #[test]
    fn test_avx512_with_tail() {
        #[cfg(target_arch = "x86_64")]
        {
            if !is_x86_feature_detected!("avx512bw") || !is_x86_feature_detected!("avx512bitalg") {
                return;
            }
            // 37 dims = 74 bytes = 1 chunk (64 bytes) + 10 byte tail
            let a: Vec<f32> = (0..37).map(|i| (i as f32 * 0.3).sin()).collect();
            let b: Vec<f32> = (0..37).map(|i| (i as f32 * 0.3 + 0.5).cos()).collect();
            let a_bf16 = fp32_to_bf16_bytes(&a);
            let b_bf16 = fp32_to_bf16_bytes(&b);
            let w = JINA_WEIGHTS;
            let scalar = bf16_hamming_scalar(&a_bf16, &b_bf16, &w);
            let simd = unsafe { bf16_hamming_avx512(&a_bf16, &b_bf16, &w) };
            assert_eq!(scalar, simd, "AVX-512 with tail must match scalar");
        }
    }

    #[test]
    fn test_zero_vectors() {
        let a = vec![0u8; 2048];
        let b = vec![0u8; 2048];
        let d = bf16_hamming_scalar(&a, &b, &BF16Weights::default());
        assert_eq!(d, 0);
    }

    #[test]
    fn test_max_distance() {
        // All bits differ: 0x0000 vs 0xFFFF per element
        let a = vec![0u8; 4]; // 2 dims, all zero
        let b = vec![0xFFu8; 4]; // 2 dims, all ones
        let w = BF16Weights::default();
        let d = bf16_hamming_scalar(&a, &b, &w);
        // Per dim: sign(1*256) + exp(8*16) + man(7*1) = 256 + 128 + 7 = 391
        assert_eq!(d, 2 * 391);
    }

    #[test]
    fn test_weights_ordering() {
        // Sign flip should always cost more than any exponent change
        let a = fp32_to_bf16_bytes(&[1.0]);
        let sign_flip = fp32_to_bf16_bytes(&[-1.0]);
        let big_exp = fp32_to_bf16_bytes(&[1024.0]); // 10 exponent doublings from 1.0
        let w = BF16Weights::default();

        let d_sign = bf16_hamming_scalar(&a, &sign_flip, &w);
        let d_exp = bf16_hamming_scalar(&a, &big_exp, &w);
        assert!(
            d_sign > d_exp,
            "Sign flip ({}) should cost more than exponent shift ({})",
            d_sign,
            d_exp
        );
    }

    #[test]
    fn test_weights_new_valid() {
        let w = BF16Weights::new(256, 16, 1);
        assert_eq!(w.sign, 256);
        assert_eq!(w.exponent, 16);
        assert_eq!(w.mantissa, 1);
    }

    #[test]
    fn test_weights_new_max_safe() {
        // Exactly at the limit: 65535 = sign + 8*exp + 7*man
        // e.g. 0 + 8*8191 + 7*1 = 65535
        let w = BF16Weights::new(0, 8191, 1);
        assert_eq!(w.exponent, 8191);
    }

    #[test]
    #[should_panic(expected = "BF16Weights overflow")]
    fn test_weights_new_overflow_panics() {
        // 4096 + 8*4096 + 7*4096 = 4096 + 32768 + 28672 = 65536 — wraps!
        BF16Weights::new(4096, 4096, 4096);
    }

    // -----------------------------------------------------------------------
    // Awareness substrate tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_superposition_identical_vectors_all_crystallized() {
        let v = fp32_to_bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let state = superposition_decompose(&[&v, &v], &AwarenessThresholds::default());
        assert_eq!(state.n_dims, 4);
        assert!(state.crystallized_pct > 0.99);
        for s in &state.states {
            assert_eq!(*s, AwarenessState::Crystallized);
        }
    }

    #[test]
    fn test_superposition_sign_flips_are_tensioned() {
        let a = fp32_to_bf16_bytes(&[1.0, 2.0, 3.0, 4.0]);
        let b = fp32_to_bf16_bytes(&[-1.0, -2.0, 3.0, 4.0]);
        let state = superposition_decompose(&[&a, &b], &AwarenessThresholds::default());
        assert_eq!(state.states[0], AwarenessState::Tensioned);
        assert_eq!(state.states[1], AwarenessState::Tensioned);
        // Dims 2,3 should NOT be tensioned
        assert_ne!(state.states[2], AwarenessState::Tensioned);
        assert_ne!(state.states[3], AwarenessState::Tensioned);
    }

    #[test]
    fn test_superposition_magnitude_shift_is_uncertain() {
        let a = fp32_to_bf16_bytes(&[1.0, 1.0]);
        let b = fp32_to_bf16_bytes(&[1.0, 256.0]); // same sign, large exponent shift
        let state = superposition_decompose(&[&a, &b], &AwarenessThresholds::default());
        assert_eq!(state.states[0], AwarenessState::Crystallized);
        assert_eq!(state.states[1], AwarenessState::Uncertain);
    }

    #[test]
    fn test_superposition_mantissa_only_is_noise() {
        let a = fp32_to_bf16_bytes(&[1.0]);
        let b = fp32_to_bf16_bytes(&[1.09375]); // only mantissa differs (2 bits in BF16 mantissa)
        let state = superposition_decompose(&[&a, &b], &AwarenessThresholds::default());
        assert_eq!(state.states[0], AwarenessState::Noise);
    }

    #[test]
    fn test_superposition_three_vectors() {
        let a = fp32_to_bf16_bytes(&[1.0, -2.0, 3.0]);
        let b = fp32_to_bf16_bytes(&[1.0, 2.0, 3.0]);
        let c = fp32_to_bf16_bytes(&[-1.0, 2.0, 3.0]);
        let state = superposition_decompose(&[&a, &b, &c], &AwarenessThresholds::default());
        // dim 0: a=+, b=+, c=- -> 1/3 disagree -> tensioned (consensus ~170/255)
        // dim 1: a=-, b=+, c=+ -> 1/3 disagree -> tensioned
        assert_eq!(state.states[0], AwarenessState::Tensioned);
        assert_eq!(state.states[1], AwarenessState::Tensioned);
        // dim 2: all agree -> crystallized (if exp spread is low)
        assert_eq!(state.states[2], AwarenessState::Crystallized);
    }

    #[test]
    fn test_pack_unpack_awareness_states() {
        let states = vec![
            AwarenessState::Crystallized,
            AwarenessState::Tensioned,
            AwarenessState::Uncertain,
            AwarenessState::Noise,
            AwarenessState::Crystallized,
        ];
        let packed = pack_awareness_states(&states);
        let unpacked = unpack_awareness_states(&packed, states.len());
        assert_eq!(states, unpacked);
    }

    #[test]
    fn test_superposition_aggregate_percentages() {
        let a = fp32_to_bf16_bytes(&[1.0, -2.0, 3.0, 4.0]);
        let b = fp32_to_bf16_bytes(&[-1.0, 2.0, 3.0, 4.0]);
        let state = superposition_decompose(&[&a, &b], &AwarenessThresholds::default());
        let total =
            state.crystallized_pct + state.tensioned_pct + state.uncertain_pct + state.noise_pct;
        assert!(
            (total - 1.0).abs() < 0.01,
            "Percentages should sum to ~1.0, got {}",
            total
        );
    }
}
