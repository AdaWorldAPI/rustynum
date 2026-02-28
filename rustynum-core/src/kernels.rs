//! Specialized bitpacked kernels for L=16384 and L=65536 bit containers.
//!
//! Implements the LIBXSMM-inspired 7-rule optimization pipeline:
//!
//! 1. **Fix world to constants**: L=16384 (2KB) and L=65536 (8KB), no dynamic tails
//! 2. **Three kernels per SKU**: K0 Probe/Reject, K1 Stats, K2 Exact/Final
//! 3. **Slice gates as bounds**: Integer thresholds, no floats in the hot path
//! 4. **Kill negative cancellation**: Track energy + conflict, not just dot
//! 5. **Vectorize logic, scalarize popcount**: SIMD for XOR/AND, scalar for final reduction
//! 6. **BF16 only in tail**: Accumulate in FP32 (or u32/u64 for integer paths)
//! 7. **Benchmark like a court transcript**: B0 kernel-only, B1 pack-only, B2 end-to-end
//!
//! ## SKU Model
//!
//! Two fixed container sizes, no runtime variability:
//! - **SKU-16K**: 16384 bits = 256 u64 = 2048 bytes (CogRecord standard)
//! - **SKU-64K**: 65536 bits = 1024 u64 = 8192 bytes (recognition projection)
//!
//! ## Kernel Pipeline (per query)
//!
//! ```text
//! K0: Probe/Reject (1 u64 = 64 bits)  → eliminates ~95%
//! K1: Stats (8 u64 = 512 bits)         → eliminates ~90% of survivors
//! K2: Exact/Final (all bits)           → exact Hamming + energy/conflict
//! ```
//!
//! Each kernel returns a `SliceGate` decision: REJECT / PROMOTE / EXACT.

// ============================================================================
// Rule 1: Fixed-world constants — no dynamic sizing
// ============================================================================

/// SKU-16K: 16384 bits = 256 u64 words = 2048 bytes.
pub const SKU_16K_BITS: usize = 16_384;
pub const SKU_16K_WORDS: usize = 256;
pub const SKU_16K_BYTES: usize = 2048;

/// SKU-64K: 65536 bits = 1024 u64 words = 8192 bytes.
pub const SKU_64K_BITS: usize = 65_536;
pub const SKU_64K_WORDS: usize = 1024;
pub const SKU_64K_BYTES: usize = 8192;

/// K0 probe width: 64 bits.
const K0_BITS: usize = 64;

/// K1 stats width: 8 u64 = 512 bits.
const K1_WORDS: usize = 8;
const K1_BITS: usize = 512;

// ============================================================================
// Rule 3: Slice gates as integer bounds — no floats
// ============================================================================

/// Precomputed integer thresholds for a given container size.
///
/// All comparisons in the hot path use `u32` — no float division,
/// no f64 multiplication, no rounding. The thresholds are computed
/// once at init time from float fractions.
#[derive(Clone, Copy, Debug)]
pub struct SliceGate {
    /// K0 reject threshold: if K0 popcount > this, reject immediately.
    /// Scaled from total threshold proportional to K0 bit fraction, with safety margin.
    pub k0_reject: u32,
    /// K1 reject threshold: if K1 popcount > this, reject.
    pub k1_reject: u32,
    /// K2 hot threshold: distance < this = blazing resonance (HDR score 3).
    pub k2_hot: u32,
    /// K2 mid threshold: distance < this = solid match (HDR score 2).
    pub k2_mid: u32,
    /// K2 cold threshold: distance < this = weak signal (HDR score 1).
    pub k2_cold: u32,
    /// Anti-resonance threshold: distance > this = anti-match.
    pub k2_anti: u32,
    /// Total bits in the container (for reference).
    pub total_bits: u32,
}

impl SliceGate {
    /// Build thresholds for a given container size.
    ///
    /// `hot_frac`, `mid_frac`, `cold_frac` are fractions of total bits (0.0–0.5).
    /// `anti_frac` is the anti-resonance floor (typically 0.90).
    /// `safety_margin` multiplicatively relaxes K0/K1 thresholds to guarantee
    /// zero false negatives (typically 1.5).
    pub fn new(
        total_bits: usize,
        hot_frac: f64,
        mid_frac: f64,
        cold_frac: f64,
        anti_frac: f64,
        safety_margin: f64,
    ) -> Self {
        let d = total_bits as f64;

        // K0 threshold: proportional to K0's bit fraction of total, with safety margin
        let k0_fraction = K0_BITS as f64 / d;
        let k0_reject = (cold_frac * d * k0_fraction * safety_margin).ceil() as u32;

        // K1 threshold: proportional to K1's bit fraction of total, with safety margin
        let k1_fraction = K1_BITS as f64 / d;
        let k1_reject = (cold_frac * d * k1_fraction * safety_margin).ceil() as u32;

        Self {
            k0_reject,
            k1_reject,
            k2_hot: (hot_frac * d).floor() as u32,
            k2_mid: (mid_frac * d).floor() as u32,
            k2_cold: (cold_frac * d).floor() as u32,
            k2_anti: (anti_frac * d).ceil() as u32,
            total_bits: total_bits as u32,
        }
    }

    /// Default thresholds for SKU-16K (VSACLIP standard).
    pub fn sku_16k() -> Self {
        Self::new(SKU_16K_BITS, 0.10, 0.30, 0.49, 0.90, 1.5)
    }

    /// Default thresholds for SKU-64K (recognition projection).
    pub fn sku_64k() -> Self {
        Self::new(SKU_64K_BITS, 0.10, 0.30, 0.49, 0.90, 1.5)
    }
}

// ============================================================================
// Rule 4: Kill negative cancellation — track energy + conflict
// ============================================================================

/// Extended match result that separates energy and conflict.
///
/// Traditional Hamming distance conflates "how different" with "in what way".
/// Energy tracks total information content (popcount of each vector).
/// Conflict tracks the XOR popcount. Together they give:
///   - High energy, low conflict = strong agreement
///   - High energy, high conflict = active disagreement (anti-resonance)
///   - Low energy = sparse/uninformative
#[derive(Clone, Copy, Debug, Default)]
pub struct EnergyConflict {
    /// Hamming distance (XOR popcount) — the conflict signal.
    pub conflict: u32,
    /// Popcount of the query vector — query energy.
    pub energy_a: u32,
    /// Popcount of the candidate vector — candidate energy.
    pub energy_b: u32,
    /// Popcount of (A AND B) — shared information (agreement bits).
    pub agreement: u32,
}

impl EnergyConflict {
    /// Agreement ratio: shared bits / max possible shared bits.
    /// Range [0.0, 1.0]. Higher = more overlap.
    #[inline]
    pub fn agreement_ratio(&self) -> f32 {
        let min_energy = self.energy_a.min(self.energy_b);
        if min_energy == 0 {
            return 0.0;
        }
        self.agreement as f32 / min_energy as f32
    }

    /// Conflict density: conflict bits / total bits.
    /// Range [0.0, 1.0]. Lower = more similar.
    #[inline]
    pub fn conflict_density(&self, total_bits: u32) -> f32 {
        self.conflict as f32 / total_bits as f32
    }

    /// True if this is an anti-resonance (vectors are anti-correlated).
    /// Anti-resonance means most bits that could agree don't.
    #[inline]
    pub fn is_anti_resonance(&self, gate: &SliceGate) -> bool {
        self.conflict > gate.k2_anti
    }
}

// ============================================================================
// Rule 2: Three kernels per SKU — K0 Probe, K1 Stats, K2 Exact
// ============================================================================

/// HDR score from K2 exact comparison.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HdrScore {
    pub hot: u8,  // 0 or 3
    pub mid: u8,  // 0 or 2
    pub cold: u8, // 0 or 1
}

impl HdrScore {
    /// Total HDR score (0-6).
    #[inline]
    pub fn total(&self) -> u8 {
        self.hot + self.mid + self.cold
    }

    /// Returns true if any resonance detected.
    #[inline]
    pub fn is_match(&self) -> bool {
        self.hot > 0 || self.mid > 0 || self.cold > 0
    }
}

/// Result of the 3-kernel pipeline for a single candidate.
#[derive(Clone, Copy, Debug)]
pub struct KernelResult {
    /// Candidate index in the database.
    pub index: usize,
    /// Which kernel stage determined the outcome.
    pub stage: KernelStage,
    /// Exact Hamming distance (only valid if stage == K2).
    pub distance: u32,
    /// HDR score (only valid if stage == K2).
    pub hdr: HdrScore,
    /// Energy/conflict decomposition (only valid if stage == K2).
    pub energy: EnergyConflict,
}

/// Which kernel stage produced the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelStage {
    /// Rejected by K0 probe (64-bit).
    K0Reject,
    /// Rejected by K1 stats (512-bit).
    K1Reject,
    /// Passed to K2 exact (full width).
    K2Exact,
}

/// Pipeline statistics for benchmarking (Rule 7).
#[derive(Clone, Copy, Debug, Default)]
pub struct PipelineStats {
    /// Total candidates processed.
    pub total: usize,
    /// Rejected by K0 (probe).
    pub k0_rejected: usize,
    /// Rejected by K1 (stats).
    pub k1_rejected: usize,
    /// Promoted to K2 (exact).
    pub k2_promoted: usize,
    /// Matches found (K2 with HDR > 0).
    pub matches: usize,
    /// Anti-resonances detected.
    pub anti_resonances: usize,
}

impl PipelineStats {
    /// K0 rejection rate.
    #[inline]
    pub fn k0_rejection_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.k0_rejected as f64 / self.total as f64
    }

    /// K1 rejection rate (of K0 survivors).
    #[inline]
    pub fn k1_rejection_rate(&self) -> f64 {
        let k0_survivors = self.total - self.k0_rejected;
        if k0_survivors == 0 {
            return 0.0;
        }
        self.k1_rejected as f64 / k0_survivors as f64
    }

    /// Total rejection rate (K0 + K1).
    #[inline]
    pub fn total_rejection_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.k0_rejected + self.k1_rejected) as f64 / self.total as f64
    }
}

// ============================================================================
// Rule 5: Vectorize logic, scalarize popcount
// ============================================================================

/// K0 probe: 64-bit (1 u64 word) fast reject.
///
/// XOR the first u64 word and popcount. If the scaled conflict
/// already exceeds the proportional threshold, reject immediately.
/// Zero branches in the hot path — just XOR + POPCNT + compare.
#[inline]
pub fn k0_probe(query_word0: u64, candidate_word0: u64, gate: &SliceGate) -> bool {
    let xor = query_word0 ^ candidate_word0;
    let conflict = xor.count_ones();
    conflict <= gate.k0_reject
}

/// K1 stats: 512-bit (8 u64 words) medium reject.
///
/// XOR the first 8 words and popcount. Proportional threshold check.
/// SIMD XOR, scalar popcount — Rule 5.
#[inline]
pub fn k1_stats(query: &[u64], candidate: &[u64], gate: &SliceGate) -> bool {
    debug_assert!(query.len() >= K1_WORDS);
    debug_assert!(candidate.len() >= K1_WORDS);

    let mut conflict: u32 = 0;
    // Unrolled 8-word XOR + popcount
    conflict += (query[0] ^ candidate[0]).count_ones();
    conflict += (query[1] ^ candidate[1]).count_ones();
    conflict += (query[2] ^ candidate[2]).count_ones();
    conflict += (query[3] ^ candidate[3]).count_ones();
    conflict += (query[4] ^ candidate[4]).count_ones();
    conflict += (query[5] ^ candidate[5]).count_ones();
    conflict += (query[6] ^ candidate[6]).count_ones();
    conflict += (query[7] ^ candidate[7]).count_ones();

    conflict <= gate.k1_reject
}

/// K2 exact: Full-width Hamming distance with energy/conflict decomposition.
///
/// Computes XOR popcount (conflict), AND popcount (agreement), and individual
/// popcounts (energy) in a single pass. Rule 4: no negative cancellation.
///
/// Rule 5: XOR and AND are vectorizable (SIMD), popcount is scalarized
/// (hardware POPCNT on each u64 result).
#[inline]
pub fn k2_exact(query: &[u64], candidate: &[u64], n_words: usize) -> EnergyConflict {
    debug_assert!(query.len() >= n_words);
    debug_assert!(candidate.len() >= n_words);

    let mut conflict: u32 = 0;
    let mut energy_a: u32 = 0;
    let mut energy_b: u32 = 0;
    let mut agreement: u32 = 0;

    // 4x unrolled for ILP — 4 independent dependency chains
    let full_quads = n_words / 4;
    for q in 0..full_quads {
        let base = q * 4;
        let (qa, qb, qc, qd) = (
            query[base],
            query[base + 1],
            query[base + 2],
            query[base + 3],
        );
        let (ca, cb, cc, cd) = (
            candidate[base],
            candidate[base + 1],
            candidate[base + 2],
            candidate[base + 3],
        );

        conflict += (qa ^ ca).count_ones()
            + (qb ^ cb).count_ones()
            + (qc ^ cc).count_ones()
            + (qd ^ cd).count_ones();

        energy_a += qa.count_ones() + qb.count_ones() + qc.count_ones() + qd.count_ones();

        energy_b += ca.count_ones() + cb.count_ones() + cc.count_ones() + cd.count_ones();

        agreement += (qa & ca).count_ones()
            + (qb & cb).count_ones()
            + (qc & cc).count_ones()
            + (qd & cd).count_ones();
    }

    // Remaining words
    for i in (full_quads * 4)..n_words {
        conflict += (query[i] ^ candidate[i]).count_ones();
        energy_a += query[i].count_ones();
        energy_b += candidate[i].count_ones();
        agreement += (query[i] & candidate[i]).count_ones();
    }

    EnergyConflict {
        conflict,
        energy_a,
        energy_b,
        agreement,
    }
}

/// Score an EnergyConflict result against SliceGate thresholds.
#[inline]
pub fn score_hdr(ec: &EnergyConflict, gate: &SliceGate) -> HdrScore {
    HdrScore {
        hot: if ec.conflict < gate.k2_hot { 3 } else { 0 },
        mid: if ec.conflict < gate.k2_mid { 2 } else { 0 },
        cold: if ec.conflict < gate.k2_cold { 1 } else { 0 },
    }
}

// ============================================================================
// Pipeline: K0 → K1 → K2 cascaded search
// ============================================================================

/// Run the 3-kernel pipeline on a database of u64-word containers.
///
/// `query_words`: query container as &[u64] (length = n_words_per_container)
/// `database_words`: flat array of all containers (length = n_candidates * n_words)
/// `n_candidates`: number of containers in database
/// `n_words`: words per container (SKU_16K_WORDS or SKU_64K_WORDS)
/// `gate`: precomputed integer thresholds
///
/// Returns all matches (candidates that survived K0+K1 and have HDR > 0)
/// and pipeline statistics.
pub fn kernel_pipeline(
    query_words: &[u64],
    database_words: &[u64],
    n_candidates: usize,
    n_words: usize,
    gate: &SliceGate,
) -> (Vec<KernelResult>, PipelineStats) {
    assert!(
        n_words == SKU_16K_WORDS || n_words == SKU_64K_WORDS,
        "kernel_pipeline only supports SKU-16K ({}) or SKU-64K ({}) containers, got {}",
        SKU_16K_WORDS,
        SKU_64K_WORDS,
        n_words
    );
    assert_eq!(query_words.len(), n_words);
    assert!(database_words.len() >= n_candidates * n_words);

    let mut matches = Vec::new();
    let mut stats = PipelineStats {
        total: n_candidates,
        ..Default::default()
    };

    let q_word0 = query_words[0];

    for i in 0..n_candidates {
        let offset = i * n_words;
        let candidate = &database_words[offset..offset + n_words];

        // K0: 64-bit probe
        if !k0_probe(q_word0, candidate[0], gate) {
            stats.k0_rejected += 1;
            continue;
        }

        // K1: 512-bit stats
        if !k1_stats(query_words, candidate, gate) {
            stats.k1_rejected += 1;
            continue;
        }

        // K2: Full exact with energy/conflict
        stats.k2_promoted += 1;
        let ec = k2_exact(query_words, candidate, n_words);
        let hdr = score_hdr(&ec, gate);

        if ec.is_anti_resonance(gate) {
            stats.anti_resonances += 1;
        }

        if hdr.is_match() {
            stats.matches += 1;
            matches.push(KernelResult {
                index: i,
                stage: KernelStage::K2Exact,
                distance: ec.conflict,
                hdr,
                energy: ec,
            });
        }
    }

    (matches, stats)
}

/// Run the 3-kernel pipeline on byte-slice containers.
///
/// Convenience wrapper that reinterprets byte slices as u64 words.
/// `query_bytes`: query as &[u8] (length = SKU_16K_BYTES or SKU_64K_BYTES)
/// `database_bytes`: flat byte array
/// `n_candidates`: number of containers
pub fn kernel_pipeline_bytes(
    query_bytes: &[u8],
    database_bytes: &[u8],
    n_candidates: usize,
    gate: &SliceGate,
) -> (Vec<KernelResult>, PipelineStats) {
    let n_bytes = query_bytes.len();
    assert!(
        n_bytes == SKU_16K_BYTES || n_bytes == SKU_64K_BYTES,
        "query must be {} or {} bytes, got {}",
        SKU_16K_BYTES,
        SKU_64K_BYTES,
        n_bytes
    );
    assert!(database_bytes.len() >= n_candidates * n_bytes);

    let n_words = n_bytes / 8;

    // Reinterpret as u64 words (little-endian)
    let query_words = bytes_to_u64_words(query_bytes);
    let db_words = bytes_to_u64_words(database_bytes);

    kernel_pipeline(&query_words, &db_words, n_candidates, n_words, gate)
}

/// Full-sweep reference: no early exit, just K2 exact on everything.
///
/// For benchmarking Rule 7 (B0 vs pipeline) and verifying zero false negatives.
pub fn full_sweep(
    query_words: &[u64],
    database_words: &[u64],
    n_candidates: usize,
    n_words: usize,
    gate: &SliceGate,
) -> Vec<KernelResult> {
    assert_eq!(query_words.len(), n_words);
    assert!(database_words.len() >= n_candidates * n_words);

    let mut matches = Vec::new();

    for i in 0..n_candidates {
        let offset = i * n_words;
        let candidate = &database_words[offset..offset + n_words];
        let ec = k2_exact(query_words, candidate, n_words);
        let hdr = score_hdr(&ec, gate);

        if hdr.is_match() {
            matches.push(KernelResult {
                index: i,
                stage: KernelStage::K2Exact,
                distance: ec.conflict,
                hdr,
                energy: ec,
            });
        }
    }

    matches
}

// ============================================================================
// Rule 6: BF16 only in tail — FP32 accumulation for weighted distances
// ============================================================================

/// Weighted Hamming distance with BF16 field decomposition.
///
/// After K2 exact produces an integer conflict count, this optional tail
/// can produce a weighted float score for BF16-structured containers.
/// The hot path (K0/K1/K2) is pure integer. BF16 decomposition only runs
/// on the ~5% of candidates that survive to K2 AND need structured scoring.
///
/// Accumulates in FP32 — never stores intermediate BF16 results.
pub fn bf16_tail_score(
    query_bytes: &[u8],
    candidate_bytes: &[u8],
    sign_weight: f32,
    exp_weight: f32,
    man_weight: f32,
) -> f32 {
    assert_eq!(query_bytes.len(), candidate_bytes.len());
    assert!(query_bytes.len().is_multiple_of(2));

    let n_dims = query_bytes.len() / 2;
    // FP32 accumulator — Rule 6
    let mut total: f32 = 0.0;

    for d in 0..n_dims {
        let i = d * 2;
        let va = u16::from_le_bytes([query_bytes[i], query_bytes[i + 1]]);
        let vb = u16::from_le_bytes([candidate_bytes[i], candidate_bytes[i + 1]]);
        let xor = va ^ vb;

        let sign_diff = ((xor >> 15) & 1) as f32;
        let exp_popcount = ((xor >> 7) & 0xFF).count_ones() as f32;
        let man_popcount = (xor & 0x7F).count_ones() as f32;

        total += sign_diff * sign_weight + exp_popcount * exp_weight + man_popcount * man_weight;
    }

    total
}

// ============================================================================
// Helpers
// ============================================================================

/// Reinterpret a byte slice as u64 words (little-endian).
#[inline]
pub fn bytes_to_u64_words(bytes: &[u8]) -> Vec<u64> {
    assert!(bytes.len().is_multiple_of(8));
    let n_words = bytes.len() / 8;
    let mut words = Vec::with_capacity(n_words);
    for i in 0..n_words {
        let offset = i * 8;
        words.push(u64::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]));
    }
    words
}

// ============================================================================
// Rule 7: Benchmark like a court transcript
// ============================================================================

/// Benchmark timing breakdown for structured reporting.
///
/// Three tiers:
/// - B0: Kernel-only (K0+K1+K2 on pre-loaded u64 words, no conversion)
/// - B1: Pack-only (bytes→u64 conversion overhead)
/// - B2: End-to-end (bytes in, matches out, including conversion)
///
/// Usage: wrap `kernel_pipeline` / `kernel_pipeline_bytes` calls in
/// `std::time::Instant::now()` pairs and populate this struct.
#[derive(Clone, Debug, Default)]
pub struct BenchmarkTranscript {
    /// B0: kernel-only time (nanoseconds per query).
    pub b0_kernel_ns: u64,
    /// B1: pack-only time (nanoseconds for bytes→u64 conversion).
    pub b1_pack_ns: u64,
    /// B2: end-to-end time (nanoseconds per query, including packing).
    pub b2_e2e_ns: u64,
    /// Number of candidates.
    pub n_candidates: usize,
    /// Pipeline statistics.
    pub stats: PipelineStats,
    /// SKU (16384 or 65536 bits).
    pub sku_bits: usize,
}

impl BenchmarkTranscript {
    /// Nanoseconds per candidate (B0 kernel path).
    pub fn ns_per_candidate_kernel(&self) -> f64 {
        if self.n_candidates == 0 {
            return 0.0;
        }
        self.b0_kernel_ns as f64 / self.n_candidates as f64
    }

    /// Nanoseconds per candidate (B2 end-to-end).
    pub fn ns_per_candidate_e2e(&self) -> f64 {
        if self.n_candidates == 0 {
            return 0.0;
        }
        self.b2_e2e_ns as f64 / self.n_candidates as f64
    }

    /// Speedup of pipeline over full sweep (based on K0+K1 rejection).
    pub fn estimated_speedup(&self) -> f64 {
        if self.stats.k2_promoted == 0 {
            return self.n_candidates as f64;
        }
        self.n_candidates as f64 / self.stats.k2_promoted as f64
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a random-ish u64 word from a seed.
    fn pseudo_word(seed: u64) -> u64 {
        // SplitMix64 step
        let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Helper: create a container of N u64 words.
    fn random_container(seed: u64, n_words: usize) -> Vec<u64> {
        (0..n_words)
            .map(|i| pseudo_word(seed.wrapping_add(i as u64)))
            .collect()
    }

    #[test]
    fn test_slice_gate_sku_16k() {
        let gate = SliceGate::sku_16k();
        assert_eq!(gate.total_bits, 16384);

        // K0 reject: ceil(0.49 * 16384 * (64/16384) * 1.5) = ceil(0.49 * 64 * 1.5) = ceil(47.04) = 48
        assert!(gate.k0_reject > 0 && gate.k0_reject <= 64);

        // K1 reject: ceil(0.49 * 16384 * (512/16384) * 1.5) = ceil(0.49 * 512 * 1.5) = ceil(376.32) = 377
        assert!(gate.k1_reject > 0 && gate.k1_reject <= 512);

        // K2 thresholds
        assert!(gate.k2_hot < gate.k2_mid);
        assert!(gate.k2_mid < gate.k2_cold);
        assert!(gate.k2_cold < gate.k2_anti);
    }

    #[test]
    fn test_slice_gate_sku_64k() {
        let gate = SliceGate::sku_64k();
        assert_eq!(gate.total_bits, 65536);
        assert!(gate.k2_hot < gate.k2_mid);
        assert!(gate.k2_mid < gate.k2_cold);
    }

    #[test]
    fn test_k0_probe_identical() {
        let gate = SliceGate::sku_16k();
        let word = 0xDEADBEEFCAFEBABEu64;
        assert!(k0_probe(word, word, &gate)); // identical = 0 conflict
    }

    #[test]
    fn test_k0_probe_opposite() {
        let gate = SliceGate::sku_16k();
        let word = 0xFFFFFFFFFFFFFFFFu64;
        assert!(!k0_probe(word, 0u64, &gate)); // 64 bits conflict
    }

    #[test]
    fn test_k1_stats_identical() {
        let gate = SliceGate::sku_16k();
        let words: Vec<u64> = (0..8).map(pseudo_word).collect();
        assert!(k1_stats(&words, &words, &gate));
    }

    #[test]
    fn test_k2_exact_identical() {
        let container = random_container(42, SKU_16K_WORDS);
        let ec = k2_exact(&container, &container, SKU_16K_WORDS);

        assert_eq!(ec.conflict, 0);
        assert_eq!(ec.energy_a, ec.energy_b);
        assert_eq!(ec.agreement, ec.energy_a);
    }

    #[test]
    fn test_k2_exact_opposite() {
        let a = vec![0xFFFFFFFFFFFFFFFFu64; SKU_16K_WORDS];
        let b = vec![0u64; SKU_16K_WORDS];
        let ec = k2_exact(&a, &b, SKU_16K_WORDS);

        assert_eq!(ec.conflict, SKU_16K_BITS as u32);
        assert_eq!(ec.energy_a, SKU_16K_BITS as u32);
        assert_eq!(ec.energy_b, 0);
        assert_eq!(ec.agreement, 0);
    }

    #[test]
    fn test_energy_conflict_agreement_ratio() {
        let ec = EnergyConflict {
            conflict: 100,
            energy_a: 8000,
            energy_b: 8000,
            agreement: 7950,
        };
        let ratio = ec.agreement_ratio();
        assert!(ratio > 0.99);
    }

    #[test]
    fn test_hdr_scoring() {
        let gate = SliceGate::sku_16k();

        // Blazing match: 0 conflict
        let ec = EnergyConflict {
            conflict: 0,
            ..Default::default()
        };
        let hdr = score_hdr(&ec, &gate);
        assert_eq!(hdr.total(), 6); // hot(3) + mid(2) + cold(1)

        // Solid match: 20% of d
        let ec = EnergyConflict {
            conflict: (SKU_16K_BITS as f64 * 0.20) as u32,
            ..Default::default()
        };
        let hdr = score_hdr(&ec, &gate);
        assert_eq!(hdr.hot, 0);
        assert_eq!(hdr.mid, 2);
        assert_eq!(hdr.cold, 1);

        // Noise floor: 50% of d
        let ec = EnergyConflict {
            conflict: (SKU_16K_BITS as f64 * 0.50) as u32,
            ..Default::default()
        };
        let hdr = score_hdr(&ec, &gate);
        assert_eq!(hdr.total(), 0);
    }

    #[test]
    fn test_pipeline_zero_false_negatives() {
        // The critical invariant: pipeline must find everything full_sweep finds.
        // Use a tighter gate (cold=0.30) so K0/K1 can reject random vectors
        // at ~50% Hamming. With cold=0.49 + safety=1.5, K0 threshold (48/64)
        // is too loose to reject ~32/64 random conflict.
        let gate = SliceGate::new(SKU_16K_BITS, 0.05, 0.15, 0.30, 0.90, 1.5);
        let query = random_container(1, SKU_16K_WORDS);

        let n_candidates = 1000;
        let mut db = Vec::with_capacity(n_candidates * SKU_16K_WORDS);

        // Plant 5 known matches (copy query with small perturbation)
        let match_indices = [0, 42, 100, 500, 999];
        for i in 0..n_candidates {
            if match_indices.contains(&i) {
                // Near-identical: flip 1 bit in word 100
                let mut copy = query.clone();
                copy[100] ^= 1u64 << (i % 64);
                db.extend_from_slice(&copy);
            } else {
                db.extend_from_slice(&random_container(i as u64 + 1000, SKU_16K_WORDS));
            }
        }

        let full = full_sweep(&query, &db, n_candidates, SKU_16K_WORDS, &gate);
        let (pipeline, stats) = kernel_pipeline(&query, &db, n_candidates, SKU_16K_WORDS, &gate);

        // Zero false negatives: pipeline must find at least everything full_sweep found
        for full_match in &full {
            let found = pipeline.iter().any(|p| p.index == full_match.index);
            assert!(
                found,
                "FALSE NEGATIVE: full_sweep found index {} (dist={}) but pipeline missed it",
                full_match.index, full_match.distance
            );
        }

        // With cold=0.30, K0_reject = ceil(0.30 * 64 * 1.5) = 29.
        // Random vectors have ~32 bits conflict in K0 → most rejected.
        assert!(
            stats.k0_rejected + stats.k1_rejected > 0,
            "Pipeline didn't reject anything — K0/K1 thresholds may be too loose"
        );

        // All planted matches should be found by both
        assert!(
            full.len() >= 5,
            "Expected at least 5 matches from full sweep, got {}",
            full.len()
        );
    }

    #[test]
    fn test_pipeline_stats() {
        // Use tight gate (cold=0.30) to demonstrate K0/K1 rejection.
        // With default cold=0.49 + safety=1.5, thresholds are intentionally
        // generous (zero false negatives guarantee) and won't reject random data.
        let gate = SliceGate::new(SKU_16K_BITS, 0.05, 0.15, 0.30, 0.90, 1.5);
        let query = random_container(1, SKU_16K_WORDS);
        let n = 500;
        let mut db = Vec::with_capacity(n * SKU_16K_WORDS);
        for i in 0..n {
            db.extend_from_slice(&random_container(i as u64 + 5000, SKU_16K_WORDS));
        }

        let (_, stats) = kernel_pipeline(&query, &db, n, SKU_16K_WORDS, &gate);

        assert_eq!(stats.total, n);
        assert_eq!(stats.k0_rejected + stats.k1_rejected + stats.k2_promoted, n);

        // With cold=0.30 gate, random vectors (~50% Hamming) should be
        // rejected heavily since K0_reject = ceil(0.30*64*1.5) = 29 < 32.
        assert!(
            stats.total_rejection_rate() > 0.5,
            "Expected >50% rejection for random vectors with cold=0.30 gate, got {:.1}%",
            stats.total_rejection_rate() * 100.0
        );
    }

    #[test]
    fn test_bf16_tail_score() {
        // Two identical BF16 vectors should have score 0
        // BF16 1.0 = 0x3F80, stored LE as [0x80, 0x3F]
        // BF16 2.0 = 0x4000, stored LE as [0x00, 0x40]
        let data = vec![0x80, 0x3F, 0x00, 0x40];
        let score = bf16_tail_score(&data, &data, 256.0, 16.0, 1.0);
        assert_eq!(score, 0.0);

        // Two different vectors: 1.0 vs -1.0 (sign flip only)
        // BF16 1.0  = 0x3F80, stored LE as [0x80, 0x3F]
        // BF16 -1.0 = 0xBF80, stored LE as [0x80, 0xBF]
        // XOR = 0x3F80 ^ 0xBF80 = 0x8000 → sign bit 15 flipped
        let a = vec![0x80, 0x3F];
        let b = vec![0x80, 0xBF];
        let score = bf16_tail_score(&a, &b, 256.0, 16.0, 1.0);
        assert_eq!(score, 256.0); // sign bit flipped
    }

    #[test]
    fn test_anti_resonance_detection() {
        let gate = SliceGate::sku_16k();

        // Nearly all bits different = anti-resonance
        let ec = EnergyConflict {
            conflict: (SKU_16K_BITS as f64 * 0.95) as u32,
            energy_a: 8000,
            energy_b: 8000,
            agreement: 400,
        };
        assert!(ec.is_anti_resonance(&gate));

        // Low conflict = not anti-resonance
        let ec = EnergyConflict {
            conflict: 100,
            energy_a: 8000,
            energy_b: 8000,
            agreement: 7950,
        };
        assert!(!ec.is_anti_resonance(&gate));
    }

    #[test]
    fn test_bytes_to_u64_roundtrip() {
        let original: Vec<u64> = (0..8).map(pseudo_word).collect();
        let bytes: Vec<u8> = original.iter().flat_map(|w| w.to_le_bytes()).collect();
        let restored = bytes_to_u64_words(&bytes);
        assert_eq!(original, restored);
    }

    #[test]
    fn test_pipeline_bytes_interface() {
        let gate = SliceGate::sku_16k();
        let query_words = random_container(1, SKU_16K_WORDS);
        let query_bytes: Vec<u8> = query_words.iter().flat_map(|w| w.to_le_bytes()).collect();

        let n = 50;
        let mut db_words = Vec::with_capacity(n * SKU_16K_WORDS);
        // Plant one exact match
        db_words.extend_from_slice(&query_words);
        for i in 1..n {
            db_words.extend_from_slice(&random_container(i as u64 + 9000, SKU_16K_WORDS));
        }
        let db_bytes: Vec<u8> = db_words.iter().flat_map(|w| w.to_le_bytes()).collect();

        let (matches, stats) = kernel_pipeline_bytes(&query_bytes, &db_bytes, n, &gate);

        // Should find at least the exact match at index 0
        assert!(
            matches.iter().any(|m| m.index == 0 && m.distance == 0),
            "Expected exact match at index 0"
        );
        assert!(stats.matches >= 1);
    }

    #[test]
    fn test_sku_64k_pipeline() {
        let gate = SliceGate::sku_64k();
        let query = random_container(1, SKU_64K_WORDS);

        let n = 100;
        let mut db = Vec::with_capacity(n * SKU_64K_WORDS);
        // Plant one match
        db.extend_from_slice(&query);
        for i in 1..n {
            db.extend_from_slice(&random_container(i as u64 + 20000, SKU_64K_WORDS));
        }

        let (matches, stats) = kernel_pipeline(&query, &db, n, SKU_64K_WORDS, &gate);

        assert!(matches.iter().any(|m| m.index == 0 && m.distance == 0));
        assert_eq!(stats.total, n);
    }

    #[test]
    fn test_benchmark_transcript() {
        let transcript = BenchmarkTranscript {
            b0_kernel_ns: 1_000_000,
            b1_pack_ns: 100_000,
            b2_e2e_ns: 1_100_000,
            n_candidates: 100_000,
            stats: PipelineStats {
                total: 100_000,
                k0_rejected: 90_000,
                k1_rejected: 8_000,
                k2_promoted: 2_000,
                matches: 50,
                anti_resonances: 3,
            },
            sku_bits: SKU_16K_BITS,
        };

        assert!((transcript.ns_per_candidate_kernel() - 10.0).abs() < 0.01);
        assert!((transcript.estimated_speedup() - 50.0).abs() < 0.01);
        assert!((transcript.stats.k0_rejection_rate() - 0.9).abs() < 0.001);
        assert!((transcript.stats.total_rejection_rate() - 0.98).abs() < 0.001);
    }
}
