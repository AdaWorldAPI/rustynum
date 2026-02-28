//! K0/K1 Belichtungsmesser (Exposure Meter) for DN-tree traversal acceleration.
//!
//! Pre-filters children before the expensive `partial_similarity()` call.
//! Uses the same XOR+POPCNT that the K0/K1/K2 kernel pipeline already uses,
//! but as standalone probe functions with auto-adjusting thresholds.
//!
//! ## Progressive Stichprobe (4 Tiers × σ Gates)
//!
//! ```text
//! Tier  Bytes  Bits   σ-gate  Reject%  Cost   Measurement
//! ────  ─────  ─────  ──────  ───────  ─────  ───────────────────────
//! K0    8      64     1σ      ~84%     ~1ns   Spot meter (1 word)
//! K1    64     512    2σ      ~97.5%   ~4ns   Zone meter (8 words)
//! BF16  64     512    3σ      ~99.7%   ~20ns  Range awareness (sign/exp/man)
//! Full  2048   16384  exact   100%     ~48ns  Complete Stichprobe (leaves)
//! ```
//!
//! ## Key Insight: XOR+POPCNT+Threshold IS Already a Sigmoid
//!
//! The rejection probability as a function of true similarity p:
//! ```text
//! P(reject | p) = Φ((threshold - D(1-p)) / σ)
//! ```
//! where Φ is the normal CDF. No artificial activation function needed.

use rustynum_core::bf16_hamming::structural_diff;
use rustynum_core::graph_hv::GraphHV;

// ============================================================================
// K0/K1 Probe Functions
// ============================================================================

/// K0 probe: XOR popcount for a single u64 word pair.
///
/// Returns the raw conflict count (0–64). The natural sigmoid gate:
/// `P(reject|p) = Φ((threshold - 64(1-p)) / σ)`.
///
/// Cost: ~1ns (1 XOR + 1 POPCNT).
#[inline]
pub fn k0_probe_conflict(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// K1 stats: total XOR popcount for the first 8 u64 words.
///
/// Returns raw conflict count (0–512). Zone meter for DN-tree traversal.
/// 8 XOR+POPCNT operations, ~4ns.
#[inline]
pub fn k1_stats_conflict(a: &[u64], b: &[u64]) -> u32 {
    debug_assert!(a.len() >= 8);
    debug_assert!(b.len() >= 8);
    (a[0] ^ b[0]).count_ones()
        + (a[1] ^ b[1]).count_ones()
        + (a[2] ^ b[2]).count_ones()
        + (a[3] ^ b[3]).count_ones()
        + (a[4] ^ b[4]).count_ones()
        + (a[5] ^ b[5]).count_ones()
        + (a[6] ^ b[6]).count_ones()
        + (a[7] ^ b[7]).count_ones()
}

// ============================================================================
// TraversalStats — Welford's Online Algorithm
// ============================================================================

/// Running statistics for auto-adjusting σ-based traversal thresholds.
///
/// Uses Welford's online algorithm for numerically stable mean/variance.
/// K0 gate at 1σ (84% rejection), K1 gate at 2σ (97.5% rejection).
/// Thresholds self-calibrate to actual data distribution after warmup (n > 100).
pub struct TraversalStats {
    k0_mean: f32,
    k0_m2: f32,
    k1_mean: f32,
    k1_m2: f32,
    n: u32,
}

impl TraversalStats {
    pub fn new() -> Self {
        Self {
            k0_mean: 0.0,
            k0_m2: 0.0,
            k1_mean: 0.0,
            k1_m2: 0.0,
            n: 0,
        }
    }

    /// Update K0 running statistics (Welford step).
    #[inline]
    pub fn update_k0(&mut self, conflict: u32) {
        self.n += 1;
        let x = conflict as f32;
        let delta = x - self.k0_mean;
        self.k0_mean += delta / self.n as f32;
        let delta2 = x - self.k0_mean;
        self.k0_m2 += delta * delta2;
    }

    /// Update K1 running statistics (Welford step).
    /// Must be called after `update_k0` (shares the same n counter).
    #[inline]
    pub fn update_k1(&mut self, conflict: u32) {
        let x = conflict as f32;
        let delta = x - self.k1_mean;
        self.k1_mean += delta / self.n as f32;
        let delta2 = x - self.k1_mean;
        self.k1_m2 += delta * delta2;
    }

    /// K0 threshold: mean + 1σ (84% rejection of random baseline).
    /// Returns warmup default (36) until n > 100 observations.
    /// Warmup default: E[Binomial(64, 0.5)] + σ = 32 + 4 = 36.
    #[inline]
    pub fn k0_threshold(&self) -> u32 {
        if self.n < 100 {
            return 36;
        }
        let std = (self.k0_m2 / self.n as f32).sqrt();
        (self.k0_mean + 1.0 * std).ceil() as u32
    }

    /// K1 threshold: mean + 2σ (97.5% rejection).
    /// Returns warmup default (279) until n > 100 observations.
    /// Warmup default: E[Binomial(512, 0.5)] + 2σ = 256 + 2×11.3 ≈ 279.
    #[inline]
    pub fn k1_threshold(&self) -> u32 {
        if self.n < 100 {
            return 279;
        }
        let std = (self.k1_m2 / self.n as f32).sqrt();
        (self.k1_mean + 2.0 * std).ceil() as u32
    }

    /// K1 hot threshold: mean - 3σ (confident match).
    #[inline]
    pub fn k1_hot_threshold(&self) -> u32 {
        if self.n < 100 {
            return 200;
        }
        let std = (self.k1_m2 / self.n as f32).sqrt();
        (self.k1_mean - 3.0 * std).max(0.0).floor() as u32
    }

    /// K1 mid threshold: mean - 1σ (solid match).
    #[inline]
    pub fn k1_mid_threshold(&self) -> u32 {
        if self.n < 100 {
            return 245;
        }
        let std = (self.k1_m2 / self.n as f32).sqrt();
        (self.k1_mean - 1.0 * std).max(0.0).floor() as u32
    }

    /// Number of observations processed.
    #[inline]
    pub fn count(&self) -> u32 {
        self.n
    }

    /// Current K0 mean.
    #[inline]
    pub fn k0_mean(&self) -> f32 {
        self.k0_mean
    }

    /// Current K1 mean.
    #[inline]
    pub fn k1_mean(&self) -> f32 {
        self.k1_mean
    }
}

impl Default for TraversalStats {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Signal Quality — Noise Floor Detection
// ============================================================================

/// Per-word popcount variance on identity channel (channel 0).
///
/// High variance = concentrated signal (good summary).
/// Low variance (< 20) = noise floor (degraded by orthogonal superposition).
/// Random vector: E[variance] ≈ 16 (Var(Binomial(64, 0.5)) = 16).
pub fn signal_quality(summary: &GraphHV) -> f32 {
    let words = &summary.channels[0].words;
    let n = words.len() as f32;
    let mut sum = 0u32;
    let mut sum_sq = 0u64;
    for w in words {
        let pc = w.count_ones();
        sum += pc;
        sum_sq += (pc as u64) * (pc as u64);
    }
    let mean = sum as f32 / n;
    sum_sq as f32 / n - mean * mean
}

// ============================================================================
// HDR Classification
// ============================================================================

/// HDR classification for a K1 conflict score.
///
/// Returns:
/// - 3 (hot): k1 < hot_threshold (μ - 3σ) → confident, beam=1
/// - 2 (mid): k1 < mid_threshold (μ - 1σ) → solid, beam=config
/// - 1 (cold): k1 < k1_threshold (μ + 2σ) → uncertain, beam=config+1
/// - 0 (dark): k1 >= k1_threshold → skip
#[inline]
pub fn classify_hdr(k1: u32, stats: &TraversalStats) -> u8 {
    if k1 < stats.k1_hot_threshold() {
        3
    } else if k1 < stats.k1_mid_threshold() {
        2
    } else if k1 < stats.k1_threshold() {
        1
    } else {
        0
    }
}

/// BF16 range awareness for cold (uncertain) K1 survivors.
///
/// Reinterprets first 64 bytes of identity channel as 32 BF16 dimensions.
/// Returns adjusted HDR class:
/// - Low sign_flips + low exp → crystallized → promote to mid (2)
/// - High sign_flips → tensioned → stay cold (1)
pub fn bf16_refine_cold(query: &GraphHV, candidate: &GraphHV) -> u8 {
    let q_bytes = &query.channels[0].as_bytes()[..64];
    let c_bytes = &candidate.channels[0].as_bytes()[..64];
    let diff = structural_diff(q_bytes, c_bytes);
    if diff.sign_flips < 4 && diff.exponent_bits_changed < 4 {
        2 // crystallized → promote to mid
    } else {
        1 // tensioned or uncertain → stay cold
    }
}

// ============================================================================
// Accelerated Traversal
// ============================================================================

/// Result of K0/K1 cascade filtering on a DN-tree child.
#[derive(Clone, Copy, Debug)]
pub struct ChildScore {
    /// Child node index in the DN-tree arena.
    pub node_idx: usize,
    /// K1 conflict score (lower = more similar).
    pub k1_conflict: u32,
    /// HDR class: 3=hot, 2=mid, 1=cold, 0=dark (rejected).
    pub hdr: u8,
}

/// Filter DN-tree children using K0/K1 Belichtungsmesser cascade.
///
/// Takes query identity words and candidate summary identity words.
/// Returns surviving children with K1 scores and HDR classification.
///
/// This function is designed to be called from outside `rustynum-core`
/// using the public DN-tree API (`DNTree::summary()`, etc.).
pub fn filter_children(
    query_word0: u64,
    query_words: &[u64],
    query_hv: &GraphHV,
    children: &[(usize, &GraphHV)], // (node_idx, summary)
    stats: &mut TraversalStats,
) -> Vec<ChildScore> {
    let mut scores = Vec::new();

    for &(node_idx, summary) in children {
        // K0: 1σ sigmoid gate (1 XOR+POPCNT, ~1ns)
        let k0 = k0_probe_conflict(query_word0, summary.channels[0].words[0]);
        stats.update_k0(k0);
        if k0 > stats.k0_threshold() {
            continue;
        }

        // K1: 2σ sigmoid gate (8 XOR+POPCNT, ~4ns)
        let k1 = k1_stats_conflict(query_words, &summary.channels[0].words);
        stats.update_k1(k1);

        // HDR classification
        let mut hdr = classify_hdr(k1, stats);
        if hdr == 0 {
            continue; // dark → skip
        }

        // BF16 range awareness for cold survivors only
        if hdr == 1 {
            hdr = bf16_refine_cold(query_hv, summary);
        }

        scores.push(ChildScore {
            node_idx,
            k1_conflict: k1,
            hdr,
        });
    }

    // Sort by K1 conflict ascending (most similar first)
    scores.sort_by_key(|s| s.k1_conflict);
    scores
}

/// Determine beam width from HDR classification of filtered children.
///
/// - Hot (3): beam=1 (confident → narrow)
/// - Mid (2): beam=base_beam_width
/// - Cold (1): beam=base_beam_width+1 (uncertain → widen)
#[inline]
pub fn hdr_beam_width(scores: &[ChildScore], base_beam_width: usize) -> usize {
    let max_hdr = scores.iter().map(|s| s.hdr).max().unwrap_or(0);
    match max_hdr {
        3 => 1,
        2 => base_beam_width,
        _ => base_beam_width + 1,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rustynum_core::graph_hv::GraphHV;
    use rustynum_core::rng::SplitMix64;

    #[test]
    fn test_k0_probe_conflict_identical() {
        assert_eq!(k0_probe_conflict(0, 0), 0);
        assert_eq!(k0_probe_conflict(u64::MAX, u64::MAX), 0);
    }

    #[test]
    fn test_k0_probe_conflict_opposite() {
        assert_eq!(k0_probe_conflict(u64::MAX, 0), 64);
        assert_eq!(k0_probe_conflict(0, u64::MAX), 64);
    }

    #[test]
    fn test_k1_stats_conflict_identical() {
        let words: Vec<u64> = (0..8)
            .map(|i| 0xDEADBEEF_CAFEBABE_u64.wrapping_add(i))
            .collect();
        assert_eq!(k1_stats_conflict(&words, &words), 0);
    }

    #[test]
    fn test_k1_stats_conflict_opposite() {
        let a = vec![u64::MAX; 8];
        let b = vec![0u64; 8];
        assert_eq!(k1_stats_conflict(&a, &b), 512);
    }

    #[test]
    fn test_traversal_stats_warmup_defaults() {
        let stats = TraversalStats::new();
        assert_eq!(stats.k0_threshold(), 36);
        assert_eq!(stats.k1_threshold(), 279);
        assert_eq!(stats.count(), 0);
    }

    #[test]
    fn test_traversal_stats_converges() {
        let mut stats = TraversalStats::new();
        // Feed random baseline: Binomial(64, 0.5) → mean≈32, σ≈4
        for i in 0..200 {
            let conflict = 28 + (i % 9); // values 28-36, centered around 32
            stats.update_k0(conflict);
            stats.update_k1(conflict * 8); // scale for K1
        }
        // After warmup, K0 threshold should be near mean + 1σ ≈ 36
        let t = stats.k0_threshold();
        assert!(t >= 32 && t <= 40, "K0 threshold should be ~36, got {}", t);
    }

    #[test]
    fn test_signal_quality_random_vs_concentrated() {
        let mut rng = SplitMix64::new(42);
        let random = GraphHV::random(&mut rng);
        let q_rand = signal_quality(&random);
        // Random: variance ≈ 16 (Binomial(64, 0.5))
        assert!(
            q_rand < 25.0,
            "Random vector signal quality should be < 25, got {}",
            q_rand
        );

        // Concentrated: all bits set in first half, zero in second half
        let mut concentrated = GraphHV::zero();
        for i in 0..128 {
            concentrated.channels[0].words[i] = u64::MAX;
        }
        let q_conc = signal_quality(&concentrated);
        assert!(
            q_conc > 500.0,
            "Concentrated vector signal quality should be > 500, got {}",
            q_conc
        );
    }

    #[test]
    fn test_classify_hdr() {
        let mut stats = TraversalStats::new();
        // Feed baseline data to get past warmup
        for i in 0..200 {
            let k = 240 + (i % 40); // 240-279
            stats.update_k0(k / 8);
            stats.update_k1(k);
        }

        // Hot: way below mean - 3σ
        assert_eq!(classify_hdr(100, &stats), 3);

        // Dark: above k1_threshold
        assert_eq!(classify_hdr(500, &stats), 0);
    }

    #[test]
    fn test_filter_children_empty() {
        let mut stats = TraversalStats::new();
        let mut rng = SplitMix64::new(42);
        let query = GraphHV::random(&mut rng);
        let children: Vec<(usize, &GraphHV)> = vec![];
        let scores = filter_children(
            query.channels[0].words[0],
            &query.channels[0].words,
            &query,
            &children,
            &mut stats,
        );
        assert!(scores.is_empty());
    }

    #[test]
    fn test_filter_children_finds_similar() {
        let mut stats = TraversalStats::new();
        let mut rng = SplitMix64::new(42);
        let query = GraphHV::random(&mut rng);

        // Create a near-identical summary (flip 1 bit)
        let mut similar = query.clone();
        similar.channels[0].words[100] ^= 1;

        // Create a random (dissimilar) summary
        let dissimilar = GraphHV::random(&mut rng);

        let children = vec![(0, &similar), (1, &dissimilar)];
        let scores = filter_children(
            query.channels[0].words[0],
            &query.channels[0].words,
            &query,
            &children,
            &mut stats,
        );

        // The similar child should survive
        assert!(
            scores.iter().any(|s| s.node_idx == 0),
            "Near-identical child should survive K0/K1 cascade"
        );
    }

    #[test]
    fn test_hdr_beam_width() {
        let hot = vec![ChildScore {
            node_idx: 0,
            k1_conflict: 10,
            hdr: 3,
        }];
        assert_eq!(hdr_beam_width(&hot, 2), 1);

        let mid = vec![ChildScore {
            node_idx: 0,
            k1_conflict: 200,
            hdr: 2,
        }];
        assert_eq!(hdr_beam_width(&mid, 2), 2);

        let cold = vec![ChildScore {
            node_idx: 0,
            k1_conflict: 270,
            hdr: 1,
        }];
        assert_eq!(hdr_beam_width(&cold, 2), 3);
    }
}
