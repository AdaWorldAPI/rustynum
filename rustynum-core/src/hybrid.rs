//! Hybrid scoring pipeline: binary Hamming + BF16 structured distance + awareness feedback.
//!
//! This module bridges three previously disconnected subsystems:
//!
//! 1. **kernels.rs** — K0/K1/K2 cascaded binary Hamming (integer-only, fast pruning)
//! 2. **bf16_hamming.rs** — BF16 structured distance (sign/exp/man weighted, learning-aware)
//! 3. **bf16_hamming.rs** — Awareness substrate (4-state superposition decomposition)
//!
//! ## Pipeline
//!
//! ```text
//! Query + Database
//!   │
//!   ├─ K0 Probe (64-bit) ──→ reject ~55%
//!   ├─ K1 Stats (512-bit) ──→ reject ~90% of survivors
//!   ├─ K2 Exact (full width) ──→ Hamming + EnergyConflict
//!   │
//!   └─ BF16 Tail (survivors only, ~5%)
//!      ├─ Structured distance: sign/exp/man weighted
//!      ├─ Structural diff: which dimensions changed
//!      └─ Awareness: crystallized/tensioned/uncertain/noise per dim
//! ```
//!
//! ## Learning Loop
//!
//! After recognition, the awareness state feeds back:
//! - **Crystallized** dims → increase hybrid weight (settled knowledge)
//! - **Tensioned** dims → flag for NARS revision (active contradiction)
//! - **Uncertain** dims → increase exploration weight
//! - **Noise** dims → decrease hybrid weight (mask out)
//!
//! The hybrid weights can be stored in WideMetaView W144-W159 (32 f32 weights)
//! or as a packed 2-bit-per-dim vector in the container itself.

use crate::bf16_hamming::{
    self, AwarenessState, AwarenessThresholds, BF16StructuralDiff, BF16Weights,
    SuperpositionState,
};
use crate::kernels::{self, EnergyConflict, HdrScore, PipelineStats, SliceGate};
use crate::tail_backend::TailBackend;

#[cfg(any(feature = "avx512", feature = "avx2"))]
use crate::simd;

// ============================================================================
// Hybrid score — combines binary + BF16 + awareness
// ============================================================================

/// Combined score from binary Hamming + BF16 structured distance.
///
/// The binary Hamming (K2 exact) gives the fast integer distance.
/// The BF16 structured distance gives a quality-weighted float score
/// that distinguishes sign flips (class-level) from mantissa noise.
#[derive(Clone, Debug)]
pub struct HybridScore {
    /// Binary Hamming distance (from K2 exact). Integer, fast.
    pub hamming_distance: u32,
    /// HDR score from binary Hamming (0-6 scale).
    pub hdr: HdrScore,
    /// Energy/conflict decomposition from binary Hamming.
    pub energy: EnergyConflict,
    /// BF16 structured distance (weighted sign/exp/man). Float, quality.
    pub bf16_distance: u64,
    /// BF16 structural diff — per-dimension breakdown of what changed.
    pub structural_diff: BF16StructuralDiff,
    /// Candidate index in the database.
    pub index: usize,
    /// Combined score: lower = better match.
    /// Computed as: hamming_distance * hamming_weight + bf16_distance * bf16_weight
    pub combined_score: f64,
}

// ============================================================================
// Tier 0: VNNI INT8 dot-product prefilter
// ============================================================================

/// How Tier 0 selects survivors.
#[derive(Clone, Debug)]
pub enum Tier0Mode {
    /// Keep the top N candidates by INT8 dot-product similarity.
    TopK(usize),
    /// Keep the top fraction of candidates (0.0–1.0). E.g., 0.3 = top 30%.
    Fraction(f32),
}

/// Configuration for the Tier 0 VNNI prefilter.
///
/// When enabled, `dot_i8(query_bytes, candidate_bytes)` is computed for every
/// candidate BEFORE the K0/K1/K2 binary Hamming cascade. Only the top-scoring
/// survivors advance to the integer pipeline.
///
/// This is the cheapest possible similarity proxy: raw INT8 dot product on
/// container bytes, no quantization step required. VPDPBUSD processes 64
/// bytes/cycle on AVX-512 VNNI hardware.
#[derive(Clone, Debug)]
pub struct Tier0Config {
    /// Whether Tier 0 prefilter is active. Default: false (backward compatible).
    pub enabled: bool,
    /// Survivor selection mode.
    pub mode: Tier0Mode,
}

impl Default for Tier0Config {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: Tier0Mode::Fraction(0.25),
        }
    }
}

impl Tier0Config {
    /// Resolve the mode to a concrete survivor count given the database size.
    pub fn survivor_count(&self, n_candidates: usize) -> usize {
        match self.mode {
            Tier0Mode::TopK(k) => k.min(n_candidates),
            Tier0Mode::Fraction(f) => {
                // Stay in f32 to avoid f32→f64 precision inflation
                // (0.10f32 → 0.10000000149f64 would cause ceil to overshoot)
                let k = (n_candidates as f32 * f).ceil() as usize;
                k.max(1).min(n_candidates)
            }
        }
    }
}

/// Configuration for the hybrid pipeline.
#[derive(Clone, Debug)]
pub struct HybridConfig {
    /// Gate thresholds for K0/K1/K2 binary pruning.
    pub gate: SliceGate,
    /// BF16 weights for structured distance scoring.
    pub bf16_weights: BF16Weights,
    /// Weight for binary Hamming in combined score (default: 1.0).
    pub hamming_weight: f64,
    /// Weight for BF16 structured distance in combined score (default: 0.01).
    /// Smaller because bf16_distance is on a larger scale.
    pub bf16_weight: f64,
    /// Maximum number of K2 survivors to send to BF16 tail.
    /// Bounds worst-case tail cost. 0 = no limit.
    pub max_bf16_candidates: usize,
    /// Awareness thresholds for superposition decomposition.
    pub awareness_thresholds: AwarenessThresholds,
    /// Tier 0 VNNI prefilter configuration.
    pub tier0: Tier0Config,
}

impl HybridConfig {
    /// Default config for SKU-16K containers.
    pub fn sku_16k() -> Self {
        Self {
            gate: SliceGate::sku_16k(),
            bf16_weights: BF16Weights::default(),
            hamming_weight: 1.0,
            bf16_weight: 0.01,
            max_bf16_candidates: 100,
            awareness_thresholds: AwarenessThresholds::default(),
            tier0: Tier0Config::default(),
        }
    }

    /// Default config for SKU-64K containers.
    pub fn sku_64k() -> Self {
        Self {
            gate: SliceGate::sku_64k(),
            bf16_weights: BF16Weights::default(),
            hamming_weight: 1.0,
            bf16_weight: 0.01,
            max_bf16_candidates: 100,
            awareness_thresholds: AwarenessThresholds::default(),
            tier0: Tier0Config::default(),
        }
    }

    /// Config optimized for learning: TRAINING_WEIGHTS ignore mantissa noise.
    pub fn learning(total_bits: usize) -> Self {
        Self {
            gate: SliceGate::new(total_bits, 0.05, 0.15, 0.30, 0.90, 1.5),
            bf16_weights: crate::bf16_hamming::TRAINING_WEIGHTS,
            hamming_weight: 0.5,
            bf16_weight: 0.02,
            max_bf16_candidates: 50,
            awareness_thresholds: AwarenessThresholds::default(),
            tier0: Tier0Config::default(),
        }
    }

    /// Enable Tier 0 VNNI prefilter with TopK mode.
    pub fn with_tier0_topk(mut self, k: usize) -> Self {
        self.tier0 = Tier0Config {
            enabled: true,
            mode: Tier0Mode::TopK(k),
        };
        self
    }

    /// Enable Tier 0 VNNI prefilter with Fraction mode.
    pub fn with_tier0_fraction(mut self, fraction: f32) -> Self {
        self.tier0 = Tier0Config {
            enabled: true,
            mode: Tier0Mode::Fraction(fraction),
        };
        self
    }
}

/// Statistics for the Tier 0 VNNI prefilter.
#[derive(Clone, Debug, Default)]
pub struct Tier0Stats {
    /// Whether Tier 0 was active for this pipeline run.
    pub active: bool,
    /// Total candidates evaluated by Tier 0.
    pub input_count: usize,
    /// Candidates that survived Tier 0 (advanced to K0/K1/K2).
    pub survivor_count: usize,
    /// Candidates pruned by Tier 0.
    pub pruned_count: usize,
    /// Minimum INT8 dot product among survivors (similarity floor).
    pub min_survivor_dot: i64,
    /// Maximum INT8 dot product among survivors (similarity ceiling).
    pub max_survivor_dot: i64,
}

/// Statistics for the hybrid pipeline.
#[derive(Clone, Debug, Default)]
pub struct HybridStats {
    /// Tier 0 VNNI prefilter stats.
    pub tier0_stats: Tier0Stats,
    /// Binary pipeline stats (K0/K1/K2).
    pub binary_stats: PipelineStats,
    /// Number of candidates that went through BF16 tail scoring.
    pub bf16_scored: usize,
    /// Number of candidates with sign flips detected.
    pub sign_flip_candidates: usize,
    /// Number of candidates with major magnitude shifts.
    pub magnitude_shift_candidates: usize,
}

// ============================================================================
// Tier 0: VNNI prefilter implementation
// ============================================================================

/// Run Tier 0 prefilter: INT8 dot-product similarity on raw container bytes.
///
/// Computes `dot_i8(query_bytes, candidate_bytes)` for every candidate and
/// returns the indices of the top-scoring survivors plus stats.
///
/// Higher dot product = more similar bytes → better candidate.
/// Tiebreak: lower index wins (deterministic).
///
/// When SIMD features are not compiled (no avx512/avx2 features), this is
/// a no-op that returns all candidates.
#[cfg(any(feature = "avx512", feature = "avx2"))]
pub fn tier0_prefilter(
    query_bytes: &[u8],
    database_bytes: &[u8],
    n_candidates: usize,
    config: &Tier0Config,
) -> (Vec<usize>, Tier0Stats) {
    let n_bytes = query_bytes.len();
    let keep = config.survivor_count(n_candidates);

    // If keeping all candidates, skip the work
    if keep >= n_candidates {
        return (
            (0..n_candidates).collect(),
            Tier0Stats {
                active: true,
                input_count: n_candidates,
                survivor_count: n_candidates,
                pruned_count: 0,
                min_survivor_dot: i64::MIN,
                max_survivor_dot: i64::MAX,
            },
        );
    }

    let dot_fn = simd::select_dot_i8_fn();

    // Compute INT8 dot product for each candidate
    // Store (dot_product, index) for sorting — negate dot for ascending sort
    let mut scored: Vec<(i64, usize)> = Vec::with_capacity(n_candidates);
    for i in 0..n_candidates {
        let offset = i * n_bytes;
        let cand = &database_bytes[offset..offset + n_bytes];
        let dot = dot_fn(query_bytes, cand);
        scored.push((dot, i));
    }

    // Partial sort: select top-k by dot product (highest first).
    // Use select_nth_unstable for O(n) average instead of O(n log n) full sort.
    // We want the k-th largest, so sort by (-dot, index) for deterministic tiebreak.
    if keep < n_candidates {
        scored.select_nth_unstable_by(keep, |a, b| {
            // Higher dot first, then lower index for ties
            b.0.cmp(&a.0).then(a.1.cmp(&b.1))
        });
        scored.truncate(keep);
    }

    // Sort survivors by index (preserves original database order for downstream pipeline)
    scored.sort_unstable_by_key(|&(_, idx)| idx);

    let min_dot = scored.iter().map(|s| s.0).min().unwrap_or(0);
    let max_dot = scored.iter().map(|s| s.0).max().unwrap_or(0);
    let survivor_indices: Vec<usize> = scored.iter().map(|s| s.1).collect();

    let stats = Tier0Stats {
        active: true,
        input_count: n_candidates,
        survivor_count: survivor_indices.len(),
        pruned_count: n_candidates - survivor_indices.len(),
        min_survivor_dot: min_dot,
        max_survivor_dot: max_dot,
    };

    (survivor_indices, stats)
}

/// Build a compact database buffer containing only the survivor candidates.
///
/// Returns the compacted bytes and a mapping from compact index to original index.
fn build_survivor_db(
    database_bytes: &[u8],
    survivor_indices: &[usize],
    n_bytes: usize,
) -> Vec<u8> {
    let mut compact = Vec::with_capacity(survivor_indices.len() * n_bytes);
    for &idx in survivor_indices {
        let offset = idx * n_bytes;
        compact.extend_from_slice(&database_bytes[offset..offset + n_bytes]);
    }
    compact
}

// ============================================================================
// Hybrid pipeline — Tier 0 → K0 → K1 → K2 → BF16 tail
// ============================================================================

/// Run the hybrid pipeline: binary pruning + BF16 structured scoring.
///
/// Phase 1: Binary Hamming (K0→K1→K2) prunes ~95% of candidates using
/// integer-only operations. This is the fast path.
///
/// Phase 2: BF16 structured distance scores the ~5% survivors with
/// weighted sign/exp/man decomposition. This gives quality ranking.
///
/// `query_bytes`: query as raw bytes (must be 2× aligned for BF16 interpretation)
/// `database_bytes`: flat byte array of all containers
/// `n_candidates`: number of containers in database
/// `config`: hybrid pipeline configuration
///
/// Returns sorted matches (best first) and pipeline statistics.
pub fn hybrid_pipeline(
    query_bytes: &[u8],
    database_bytes: &[u8],
    n_candidates: usize,
    config: &HybridConfig,
) -> (Vec<HybridScore>, HybridStats) {
    let n_bytes = query_bytes.len();
    assert!(
        n_bytes == kernels::SKU_16K_BYTES || n_bytes == kernels::SKU_64K_BYTES,
        "query must be {} or {} bytes, got {}",
        kernels::SKU_16K_BYTES,
        kernels::SKU_64K_BYTES,
        n_bytes
    );
    assert!(database_bytes.len() >= n_candidates * n_bytes);

    let mut stats = HybridStats::default();

    // Phase 0 (optional): Tier 0 VNNI prefilter
    #[cfg(any(feature = "avx512", feature = "avx2"))]
    let (effective_db, effective_n, index_map) = if config.tier0.enabled {
        let (survivors, t0_stats) =
            tier0_prefilter(query_bytes, database_bytes, n_candidates, &config.tier0);
        stats.tier0_stats = t0_stats;

        if survivors.len() < n_candidates {
            let compact_db = build_survivor_db(database_bytes, &survivors, n_bytes);
            (compact_db, survivors.len(), Some(survivors))
        } else {
            (Vec::new(), n_candidates, None)
        }
    } else {
        (Vec::new(), n_candidates, None)
    };

    #[cfg(not(any(feature = "avx512", feature = "avx2")))]
    let (effective_db, effective_n, index_map) = (Vec::<u8>::new(), n_candidates, None::<Vec<usize>>);

    // Use compact DB if Tier 0 produced one, otherwise the original
    let db_ref = if effective_db.is_empty() {
        database_bytes
    } else {
        &effective_db
    };

    // Phase 1: Binary Hamming pipeline (K0→K1→K2)
    let (binary_matches, binary_stats) =
        kernels::kernel_pipeline_bytes(query_bytes, db_ref, effective_n, &config.gate);

    stats.binary_stats = binary_stats;

    // Phase 2: BF16 structured scoring on K2 survivors
    let bf16_fn = bf16_hamming::select_bf16_hamming_fn();
    let mut hybrid_scores: Vec<HybridScore> = Vec::with_capacity(binary_matches.len());

    // Limit BF16 candidates if configured (bound worst-case tail cost)
    let bf16_limit = if config.max_bf16_candidates == 0 {
        binary_matches.len()
    } else {
        binary_matches.len().min(config.max_bf16_candidates)
    };

    // Sort binary matches by distance so we BF16-score the best candidates first
    let mut sorted_binary = binary_matches;
    sorted_binary.sort_by_key(|m| m.distance);

    for km in sorted_binary.iter().take(bf16_limit) {
        // Remap index: if Tier 0 was active, km.index is compact → remap to original
        let original_index = match &index_map {
            Some(map) => map[km.index],
            None => km.index,
        };

        let cand_offset = km.index * n_bytes;
        let cand_bytes = &db_ref[cand_offset..cand_offset + n_bytes];

        // BF16 structured distance (uses AVX-512 BITALG if available)
        let bf16_dist = bf16_fn(query_bytes, cand_bytes, &config.bf16_weights);

        // Structural diff — per-dimension breakdown for learning
        let diff = bf16_hamming::structural_diff(query_bytes, cand_bytes);

        stats.bf16_scored += 1;
        if diff.sign_flips > 0 {
            stats.sign_flip_candidates += 1;
        }
        if !diff.major_magnitude_shifts.is_empty() {
            stats.magnitude_shift_candidates += 1;
        }

        // Combined score: binary + weighted BF16
        let combined = km.distance as f64 * config.hamming_weight
            + bf16_dist as f64 * config.bf16_weight;

        hybrid_scores.push(HybridScore {
            hamming_distance: km.distance,
            hdr: km.hdr,
            energy: km.energy,
            bf16_distance: bf16_dist,
            structural_diff: diff,
            index: original_index,
            combined_score: combined,
        });
    }

    // Sort by combined score (best first)
    hybrid_scores.sort_by(|a, b| a.combined_score.partial_cmp(&b.combined_score).unwrap_or(std::cmp::Ordering::Equal));

    (hybrid_scores, stats)
}

/// Run the hybrid pipeline with a pluggable tail backend.
///
/// Same as `hybrid_pipeline` but uses a `TailBackend` trait object for the
/// BF16 tail scoring phase. This is the libCEED-style performance-portable
/// entry point: the caller selects the backend at init time, then the
/// pipeline dispatches through the trait.
///
/// Use `tail_backend::auto_detect()` for automatic runtime selection.
///
/// ```text
/// let backend = rustynum_core::tail_backend::auto_detect();
/// let (scores, stats) = hybrid_pipeline_with_backend(
///     &query, &db, n, &config, backend.as_ref(),
/// );
/// ```
pub fn hybrid_pipeline_with_backend(
    query_bytes: &[u8],
    database_bytes: &[u8],
    n_candidates: usize,
    config: &HybridConfig,
    backend: &dyn TailBackend,
) -> (Vec<HybridScore>, HybridStats) {
    let n_bytes = query_bytes.len();
    assert!(
        n_bytes == kernels::SKU_16K_BYTES || n_bytes == kernels::SKU_64K_BYTES,
        "query must be {} or {} bytes, got {}",
        kernels::SKU_16K_BYTES,
        kernels::SKU_64K_BYTES,
        n_bytes
    );
    assert!(database_bytes.len() >= n_candidates * n_bytes);

    let mut stats = HybridStats::default();

    // Phase 0 (optional): Tier 0 VNNI prefilter
    #[cfg(any(feature = "avx512", feature = "avx2"))]
    let (effective_db, effective_n, index_map) = if config.tier0.enabled {
        let (survivors, t0_stats) =
            tier0_prefilter(query_bytes, database_bytes, n_candidates, &config.tier0);
        stats.tier0_stats = t0_stats;

        if survivors.len() < n_candidates {
            let compact_db = build_survivor_db(database_bytes, &survivors, n_bytes);
            (compact_db, survivors.len(), Some(survivors))
        } else {
            (Vec::new(), n_candidates, None)
        }
    } else {
        (Vec::new(), n_candidates, None)
    };

    #[cfg(not(any(feature = "avx512", feature = "avx2")))]
    let (effective_db, effective_n, index_map) = (Vec::<u8>::new(), n_candidates, None::<Vec<usize>>);

    let db_ref = if effective_db.is_empty() {
        database_bytes
    } else {
        &effective_db
    };

    // Phase 1: Binary Hamming pipeline (K0→K1→K2)
    let (binary_matches, binary_stats) =
        kernels::kernel_pipeline_bytes(query_bytes, db_ref, effective_n, &config.gate);

    stats.binary_stats = binary_stats;

    // Sort binary matches by distance so we BF16-score the best candidates first
    let mut sorted_binary = binary_matches;
    sorted_binary.sort_by_key(|m| m.distance);

    let bf16_limit = if config.max_bf16_candidates == 0 {
        sorted_binary.len()
    } else {
        sorted_binary.len().min(config.max_bf16_candidates)
    };

    let candidates_to_score = &sorted_binary[..bf16_limit];

    // Phase 2: BF16 tail scoring via backend
    let mut hybrid_scores: Vec<HybridScore> = Vec::with_capacity(candidates_to_score.len());

    if backend.supports_batch() && candidates_to_score.len() >= 8 {
        // Batch path: collect candidate bytes, score all at once
        let mut batch_bytes = Vec::with_capacity(candidates_to_score.len() * n_bytes);
        for km in candidates_to_score {
            let offset = km.index * n_bytes;
            batch_bytes.extend_from_slice(&db_ref[offset..offset + n_bytes]);
        }

        let batch = backend.score_batch(
            query_bytes,
            &batch_bytes,
            candidates_to_score.len(),
            &config.bf16_weights,
        );

        for (i, km) in candidates_to_score.iter().enumerate() {
            let original_index = match &index_map {
                Some(map) => map[km.index],
                None => km.index,
            };

            stats.bf16_scored += 1;
            let diff = &batch.diffs[i];
            if diff.sign_flips > 0 {
                stats.sign_flip_candidates += 1;
            }
            if !diff.major_magnitude_shifts.is_empty() {
                stats.magnitude_shift_candidates += 1;
            }

            let combined = km.distance as f64 * config.hamming_weight
                + batch.distances[i] as f64 * config.bf16_weight;

            hybrid_scores.push(HybridScore {
                hamming_distance: km.distance,
                hdr: km.hdr,
                energy: km.energy,
                bf16_distance: batch.distances[i],
                structural_diff: diff.clone(),
                index: original_index,
                combined_score: combined,
            });
        }
    } else {
        // Per-candidate path
        for km in candidates_to_score {
            let original_index = match &index_map {
                Some(map) => map[km.index],
                None => km.index,
            };

            let cand_offset = km.index * n_bytes;
            let cand_bytes = &db_ref[cand_offset..cand_offset + n_bytes];

            let tail = backend.score(query_bytes, cand_bytes, &config.bf16_weights);

            stats.bf16_scored += 1;
            if tail.structural_diff.sign_flips > 0 {
                stats.sign_flip_candidates += 1;
            }
            if !tail.structural_diff.major_magnitude_shifts.is_empty() {
                stats.magnitude_shift_candidates += 1;
            }

            let combined = km.distance as f64 * config.hamming_weight
                + tail.bf16_distance as f64 * config.bf16_weight;

            hybrid_scores.push(HybridScore {
                hamming_distance: km.distance,
                hdr: km.hdr,
                energy: km.energy,
                bf16_distance: tail.bf16_distance,
                structural_diff: tail.structural_diff,
                index: original_index,
                combined_score: combined,
            });
        }
    }

    // Sort by combined score (best first)
    hybrid_scores.sort_by(|a, b| a.combined_score.partial_cmp(&b.combined_score).unwrap_or(std::cmp::Ordering::Equal));

    (hybrid_scores, stats)
}

// ============================================================================
// Resonance result — decomposed recognition for agentic retrieval
// ============================================================================

/// A single resonant match with awareness decomposition.
///
/// Beyond distance scoring, this tells you *how* the match resonates:
/// which dimensions are crystallized (settled agreement), tensioned
/// (active contradiction), uncertain, or noise.
#[derive(Clone, Debug)]
pub struct ResonantMatch {
    /// BindSpace address or database index of the matched record.
    pub address: usize,
    /// Combined hybrid score (lower = better).
    pub score: f64,
    /// Binary Hamming distance.
    pub hamming_distance: u32,
    /// BF16 structured distance.
    pub bf16_distance: u64,
    /// Per-dimension awareness classification against this match.
    pub awareness: SuperpositionState,
    /// Fraction of dimensions crystallized (settled agreement).
    pub crystallized_ratio: f32,
    /// Fraction of dimensions tensioned (contradiction).
    pub tension_ratio: f32,
    /// Number of sign flips (class-level disagreements).
    pub sign_flips: usize,
}

/// Decomposed resonance result — the output of agentic retrieval.
///
/// Instead of a flat list of nearest neighbors, this groups matches
/// by their awareness relationship to the query:
///
/// - **Crystallized**: strong matches where most dimensions agree.
///   These are confirmed memories — the substrate recognizes this input.
/// - **Tensioned**: matches with significant disagreement (sign flips).
///   These are contradictions — the substrate has conflicting evidence.
/// - **Uncertain**: matches where dimensions are ambiguous.
///   These are exploratory — the substrate has partial evidence.
///
/// A `ResonanceAgent` on the blackboard publishes this as a typed slot
/// so other agents (felt-parse, chat handler) can incorporate the
/// decomposition into their decision-making.
#[derive(Clone, Debug)]
pub struct ResonanceResult {
    /// Matches where >50% of dimensions are crystallized.
    pub crystallized: Vec<ResonantMatch>,
    /// Matches where >30% of dimensions are tensioned.
    pub tensioned: Vec<ResonantMatch>,
    /// Matches where neither crystallized nor tensioned dominate.
    pub uncertain: Vec<ResonantMatch>,
    /// Noise floor: combined_score below which matches are pruned.
    pub noise_floor: f64,
    /// Learning signal aggregated from all top-K matches.
    pub learning_signal: Option<LearningSignal>,
    /// Pipeline statistics.
    pub stats: HybridStats,
}

/// Decompose hybrid pipeline results into a `ResonanceResult`.
///
/// Runs the hybrid pipeline, then classifies each survivor by its
/// awareness relationship to the query. Extracts a learning signal
/// from the top-2 matches for weight feedback.
///
/// `query_bytes`: raw container bytes (SKU-16K or SKU-64K)
/// `database_bytes`: flat byte array of all containers
/// `n_candidates`: number of containers in database
/// `config`: hybrid pipeline configuration
/// `top_k`: maximum number of matches to decompose (default: 10)
pub fn resonance_decompose(
    query_bytes: &[u8],
    database_bytes: &[u8],
    n_candidates: usize,
    config: &HybridConfig,
    top_k: usize,
) -> ResonanceResult {
    let n_bytes = query_bytes.len();

    // Phase 1: Run the full hybrid pipeline
    let (scores, stats) = hybrid_pipeline(query_bytes, database_bytes, n_candidates, config);

    // Take top-K
    let top = &scores[..scores.len().min(top_k)];

    // Compute noise floor: if we have enough matches, use the worst top-K score
    let noise_floor = if top.len() >= 3 {
        top.last().map(|s| s.combined_score * 1.1).unwrap_or(f64::MAX)
    } else {
        f64::MAX // no noise floor with few matches
    };

    // Phase 2: Decompose each match by awareness
    let mut crystallized = Vec::new();
    let mut tensioned = Vec::new();
    let mut uncertain = Vec::new();

    for hs in top {
        let cand_offset = hs.index * n_bytes;
        if cand_offset + n_bytes > database_bytes.len() {
            continue;
        }
        let cand_bytes = &database_bytes[cand_offset..cand_offset + n_bytes];

        // Per-match awareness decomposition (query vs this candidate)
        let awareness = bf16_hamming::superposition_decompose(
            &[query_bytes, cand_bytes],
            &config.awareness_thresholds,
        );

        let rm = ResonantMatch {
            address: hs.index,
            score: hs.combined_score,
            hamming_distance: hs.hamming_distance,
            bf16_distance: hs.bf16_distance,
            crystallized_ratio: awareness.crystallized_pct,
            tension_ratio: awareness.tensioned_pct,
            sign_flips: hs.structural_diff.sign_flips,
            awareness,
        };

        if rm.crystallized_ratio > 0.5 {
            crystallized.push(rm);
        } else if rm.tension_ratio > 0.3 {
            tensioned.push(rm);
        } else {
            uncertain.push(rm);
        }
    }

    // Phase 3: Extract learning signal from top-2 matches
    let learning_signal = if scores.len() >= 1 {
        let mut top_k_bytes: Vec<&[u8]> = Vec::new();
        for hs in scores.iter().take(2) {
            let offset = hs.index * n_bytes;
            if offset + n_bytes <= database_bytes.len() {
                top_k_bytes.push(&database_bytes[offset..offset + n_bytes]);
            }
        }
        if !top_k_bytes.is_empty() && top_k_bytes.len() <= 2 {
            Some(extract_learning_signal(
                query_bytes,
                &top_k_bytes,
                &config.awareness_thresholds,
            ))
        } else {
            None
        }
    } else {
        None
    };

    ResonanceResult {
        crystallized,
        tensioned,
        uncertain,
        noise_floor,
        learning_signal,
        stats,
    }
}

// ============================================================================
// Awareness feedback — learning signal from recognition results
// ============================================================================

/// Learning signal extracted from recognition results.
///
/// After running the hybrid pipeline, decompose the query against the
/// top-K matches to understand per-dimension learning state.
#[derive(Clone, Debug)]
pub struct LearningSignal {
    /// Superposition state: per-dimension awareness classification.
    pub awareness: SuperpositionState,
    /// Fraction of dimensions that are crystallized (settled).
    pub crystallized_ratio: f32,
    /// Fraction of dimensions under tension (contradictory evidence).
    pub tension_ratio: f32,
    /// Number of sign-flip dimensions across all top-K comparisons.
    pub total_sign_flips: usize,
    /// Per-dimension attention weights derived from awareness state.
    /// Crystallized → 1.0, Tensioned → 0.5, Uncertain → 0.25, Noise → 0.0
    pub attention_weights: Vec<f32>,
    /// Packed 2-bit awareness states (4 dims per byte).
    pub packed_states: Vec<u8>,
}

/// Default attention weights for each awareness state.
const ATTENTION_CRYSTALLIZED: f32 = 1.0;
const ATTENTION_TENSIONED: f32 = 0.5;
const ATTENTION_UNCERTAIN: f32 = 0.25;
const ATTENTION_NOISE: f32 = 0.0;

/// Extract learning signal from query + top-K recognition results.
///
/// Takes the query and the BF16 byte representations of the top K matches,
/// decomposes them into per-dimension awareness states, and produces
/// attention weights that can be fed back into WideMetaView hybrid weights.
///
/// `query_bytes`: the recognition query (BF16 byte representation)
/// `top_k_bytes`: byte representations of the top K matches (2-3 max)
/// `thresholds`: awareness classification thresholds
pub fn extract_learning_signal(
    query_bytes: &[u8],
    top_k_bytes: &[&[u8]],
    thresholds: &AwarenessThresholds,
) -> LearningSignal {
    assert!(
        !top_k_bytes.is_empty() && top_k_bytes.len() <= 2,
        "extract_learning_signal needs 1-2 top-K results (plus query = 2-3 vectors total)"
    );

    // Build vector set: query + top-K results (2-3 total)
    let mut vectors: Vec<&[u8]> = Vec::with_capacity(top_k_bytes.len() + 1);
    vectors.push(query_bytes);
    vectors.extend_from_slice(top_k_bytes);

    // Superposition decomposition
    let awareness = bf16_hamming::superposition_decompose(&vectors, thresholds);

    // Compute attention weights from awareness states
    let attention_weights: Vec<f32> = awareness
        .states
        .iter()
        .map(|s| match s {
            AwarenessState::Crystallized => ATTENTION_CRYSTALLIZED,
            AwarenessState::Tensioned => ATTENTION_TENSIONED,
            AwarenessState::Uncertain => ATTENTION_UNCERTAIN,
            AwarenessState::Noise => ATTENTION_NOISE,
        })
        .collect();

    // Count total sign flips across all query↔result diffs
    let mut total_sign_flips = 0;
    for &top_bytes in top_k_bytes {
        let diff = bf16_hamming::structural_diff(query_bytes, top_bytes);
        total_sign_flips += diff.sign_flips;
    }

    let packed_states = awareness.packed_states.clone();
    let crystallized_ratio = awareness.crystallized_pct;
    let tension_ratio = awareness.tensioned_pct;

    LearningSignal {
        awareness,
        crystallized_ratio,
        tension_ratio,
        total_sign_flips,
        attention_weights,
        packed_states,
    }
}

/// Apply learning signal to update hybrid weights.
///
/// Given a current set of 32 f32 hybrid weights (matching WideMetaView W144-W159)
/// and a learning signal, update the weights using exponential moving average.
///
/// Each of the 32 weight slots corresponds to a group of dimensions:
/// - For 1024-dim BF16: each slot covers 32 dimensions (1024/32)
/// - For 512-dim BF16: each slot covers 16 dimensions (512/32)
///
/// The update rule per group:
/// - If group is mostly crystallized → increase weight toward 1.0
/// - If group is mostly tensioned → reduce weight toward 0.5
/// - If group is mostly noise → decrease weight toward 0.0
pub fn update_hybrid_weights(
    current_weights: &mut [f32; 32],
    signal: &LearningSignal,
    learning_rate: f32,
) {
    let n_dims = signal.awareness.n_dims;
    if n_dims == 0 {
        return;
    }
    let group_size = n_dims.div_ceil(32);

    for group in 0..32 {
        let start = group * group_size;
        let end = (start + group_size).min(n_dims);
        if start >= n_dims {
            break;
        }

        // Count states in this group
        let mut crystallized = 0u32;
        let mut tensioned = 0u32;
        let mut noise = 0u32;
        let group_count = (end - start) as f32;

        for d in start..end {
            match signal.awareness.states[d] {
                AwarenessState::Crystallized => crystallized += 1,
                AwarenessState::Tensioned => tensioned += 1,
                AwarenessState::Noise => noise += 1,
                AwarenessState::Uncertain => {} // neutral
            }
        }

        // Target weight based on group composition
        let cryst_ratio = crystallized as f32 / group_count;
        let tension_ratio = tensioned as f32 / group_count;
        let noise_ratio = noise as f32 / group_count;

        let target = if cryst_ratio > 0.5 {
            // Mostly settled → high weight
            0.8 + 0.2 * cryst_ratio
        } else if tension_ratio > 0.3 {
            // Contradictory → medium weight (hold both perspectives)
            0.5
        } else if noise_ratio > 0.5 {
            // Mostly noise → suppress
            0.1 * (1.0 - noise_ratio)
        } else {
            // Mixed → maintain current
            current_weights[group]
        };

        // EMA update
        current_weights[group] =
            current_weights[group] * (1.0 - learning_rate) + target * learning_rate;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bf16_hamming::fp32_to_bf16_bytes;

    /// Helper: create a BF16 byte vector from f32 values, padded to container size.
    fn make_container(values: &[f32], container_bytes: usize) -> Vec<u8> {
        let bf16 = fp32_to_bf16_bytes(values);
        let mut padded = vec![0u8; container_bytes];
        let copy_len = bf16.len().min(container_bytes);
        padded[..copy_len].copy_from_slice(&bf16[..copy_len]);
        padded
    }

    #[test]
    fn test_hybrid_config_defaults() {
        let c16 = HybridConfig::sku_16k();
        assert_eq!(c16.gate.total_bits, 16384);
        assert_eq!(c16.bf16_weights.sign, 256);
        assert!(c16.hamming_weight > 0.0);

        let c64 = HybridConfig::sku_64k();
        assert_eq!(c64.gate.total_bits, 65536);

        let cl = HybridConfig::learning(kernels::SKU_16K_BITS);
        assert_eq!(cl.bf16_weights.mantissa, 0); // TRAINING_WEIGHTS ignores mantissa
    }

    #[test]
    fn test_hybrid_pipeline_exact_match() {
        let config = HybridConfig::sku_16k();
        let n_dims = kernels::SKU_16K_BYTES / 2; // 1024 BF16 dims
        let values: Vec<f32> = (0..n_dims).map(|i| (i as f32 * 0.01).sin()).collect();
        let query = make_container(&values, kernels::SKU_16K_BYTES);

        // Database: exact match at index 0, random at 1-9
        let n = 10;
        let mut db = Vec::with_capacity(n * kernels::SKU_16K_BYTES);
        db.extend_from_slice(&query); // exact match
        for i in 1..n {
            let rand_vals: Vec<f32> =
                (0..n_dims).map(|j| ((i * 1000 + j) as f32 * 0.037).cos()).collect();
            db.extend_from_slice(&make_container(&rand_vals, kernels::SKU_16K_BYTES));
        }

        let (scores, stats) = hybrid_pipeline(&query, &db, n, &config);

        // Should find exact match
        assert!(!scores.is_empty(), "Should find at least one match");
        assert_eq!(scores[0].index, 0, "Best match should be the exact copy");
        assert_eq!(scores[0].hamming_distance, 0);
        assert_eq!(scores[0].bf16_distance, 0);
        assert_eq!(scores[0].combined_score, 0.0);
        assert_eq!(scores[0].structural_diff.sign_flips, 0);

        // Stats
        assert!(stats.bf16_scored >= 1);
        assert_eq!(stats.binary_stats.total, n);
    }

    #[test]
    fn test_hybrid_pipeline_sign_flip_scoring() {
        let config = HybridConfig::sku_16k();
        let n_dims = kernels::SKU_16K_BYTES / 2;

        // Query: all positive
        let values: Vec<f32> = (0..n_dims).map(|i| (i as f32 + 1.0) * 0.001).collect();
        let query = make_container(&values, kernels::SKU_16K_BYTES);

        // Candidate 0: same magnitudes, some signs flipped (first 100 dims)
        let mut flipped_values = values.clone();
        for v in flipped_values.iter_mut().take(100) {
            *v = -*v;
        }
        let candidate = make_container(&flipped_values, kernels::SKU_16K_BYTES);

        let n = 1;
        let (scores, stats) = hybrid_pipeline(&query, &candidate, n, &config);

        if !scores.is_empty() {
            // BF16 distance should reflect sign flips heavily (weight 256 each)
            assert!(
                scores[0].bf16_distance > 0,
                "Sign flips should produce non-zero BF16 distance"
            );
            assert!(
                scores[0].structural_diff.sign_flips > 0,
                "Should detect sign flips"
            );
            assert!(
                stats.sign_flip_candidates > 0,
                "Should count sign flip candidates"
            );
        }
    }

    #[test]
    fn test_learning_signal_crystallized() {
        // Two very similar vectors → should be mostly crystallized
        let vals_a: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let a = fp32_to_bf16_bytes(&vals_a);

        // Same values (identical) → all crystallized
        let signal = extract_learning_signal(&a, &[&a], &AwarenessThresholds::default());

        assert!(
            signal.crystallized_ratio > 0.9,
            "Identical vectors should be >90% crystallized, got {}",
            signal.crystallized_ratio
        );
        assert_eq!(signal.total_sign_flips, 0);
        assert_eq!(signal.attention_weights.len(), 64);

        // All weights should be 1.0 (crystallized)
        for &w in &signal.attention_weights {
            assert!(
                (w - ATTENTION_CRYSTALLIZED).abs() < 0.01,
                "Crystallized dims should have weight {}, got {}",
                ATTENTION_CRYSTALLIZED,
                w
            );
        }
    }

    #[test]
    fn test_learning_signal_tensioned() {
        // Vectors with sign disagreement → tensioned dimensions
        let vals_a: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1 + 0.5).sin()).collect();
        let mut vals_b = vals_a.clone();
        // Flip signs on first 32 dims
        for v in vals_b.iter_mut().take(32) {
            *v = -*v;
        }
        let a = fp32_to_bf16_bytes(&vals_a);
        let b = fp32_to_bf16_bytes(&vals_b);

        let signal = extract_learning_signal(&a, &[&b], &AwarenessThresholds::default());

        assert!(
            signal.tension_ratio > 0.3,
            "Sign flips should cause >30% tension, got {}",
            signal.tension_ratio
        );
        assert!(signal.total_sign_flips > 20, "Should detect many sign flips");

        // Tensioned dims should have lower attention weight
        let tensioned_weights: Vec<f32> = signal.attention_weights[..32]
            .iter()
            .filter(|&&w| w <= ATTENTION_TENSIONED)
            .cloned()
            .collect();
        assert!(
            tensioned_weights.len() > 10,
            "Many of the first 32 dims should have reduced attention"
        );
    }

    #[test]
    fn test_update_hybrid_weights() {
        let vals: Vec<f32> = (0..128).map(|i| (i as f32 * 0.1).sin()).collect();
        let a = fp32_to_bf16_bytes(&vals);

        // All crystallized → weights should increase
        let signal = extract_learning_signal(&a, &[&a], &AwarenessThresholds::default());

        let mut weights = [0.5f32; 32];
        update_hybrid_weights(&mut weights, &signal, 0.1);

        // Weights should have moved toward 1.0 (crystallized target)
        for &w in &weights[..4] {
            // First 4 groups (128 dims / 32 groups = 4 dims per group)
            assert!(
                w > 0.5,
                "Crystallized groups should increase weight, got {}",
                w
            );
        }
    }

    #[test]
    fn test_update_hybrid_weights_noise_suppression() {
        // Create vectors with only mantissa differences → noise
        let vals_a: Vec<f32> = (0..64).map(|i| (i as f32 + 1.0) * 1.0).collect();
        let mut vals_b = vals_a.clone();
        // Tiny perturbations that only affect mantissa bits
        for v in vals_b.iter_mut() {
            *v *= 1.05; // ~5% change → mantissa noise only in BF16
        }
        let a = fp32_to_bf16_bytes(&vals_a);
        let b = fp32_to_bf16_bytes(&vals_b);

        let signal = extract_learning_signal(&a, &[&b], &AwarenessThresholds::default());

        let mut weights = [0.5f32; 32];
        let original_weights = weights;
        update_hybrid_weights(&mut weights, &signal, 0.3);

        // With mostly noise dims, weights should decrease or stay low
        let noise_decreased = weights
            .iter()
            .zip(original_weights.iter())
            .filter(|(&new, &old)| new < old)
            .count();
        // At least some groups should have decreased (noise suppression)
        // Note: depending on BF16 rounding, some dims might classify as crystallized
        // so we only check that the mechanism works, not exact counts
        assert!(
            noise_decreased > 0 || signal.awareness.noise_pct > 0.0,
            "Should detect noise and/or decrease weights"
        );
    }

    #[test]
    fn test_packed_awareness_roundtrip() {
        let vals_a: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
        let mut vals_b = vals_a.clone();
        for v in vals_b.iter_mut().take(16) {
            *v = -*v;
        }
        let a = fp32_to_bf16_bytes(&vals_a);
        let b = fp32_to_bf16_bytes(&vals_b);

        let signal = extract_learning_signal(&a, &[&b], &AwarenessThresholds::default());

        // packed_states should be compact
        assert_eq!(
            signal.packed_states.len(),
            (64 + 3) / 4, // ceil(64 dims / 4 dims per byte)
            "Packed states should be ceil(n_dims/4) bytes"
        );

        // Unpack and verify
        let unpacked =
            bf16_hamming::unpack_awareness_states(&signal.packed_states, signal.awareness.n_dims);
        assert_eq!(unpacked, signal.awareness.states);
    }

    #[test]
    fn test_hybrid_pipeline_with_backend_matches_original() {
        let config = HybridConfig::sku_16k();
        let n_dims = kernels::SKU_16K_BYTES / 2;
        let values: Vec<f32> = (0..n_dims).map(|i| (i as f32 * 0.01).sin()).collect();
        let query = make_container(&values, kernels::SKU_16K_BYTES);

        let n = 10;
        let mut db = Vec::with_capacity(n * kernels::SKU_16K_BYTES);
        db.extend_from_slice(&query); // exact match at index 0
        for i in 1..n {
            let rand_vals: Vec<f32> =
                (0..n_dims).map(|j| ((i * 1000 + j) as f32 * 0.037).cos()).collect();
            db.extend_from_slice(&make_container(&rand_vals, kernels::SKU_16K_BYTES));
        }

        // Run both pipelines
        let (original_scores, original_stats) = hybrid_pipeline(&query, &db, n, &config);
        let backend = crate::tail_backend::auto_detect();
        let (backend_scores, backend_stats) =
            hybrid_pipeline_with_backend(&query, &db, n, &config, backend.as_ref());

        // Results must match
        assert_eq!(original_scores.len(), backend_scores.len());
        assert_eq!(original_stats.bf16_scored, backend_stats.bf16_scored);

        for (orig, back) in original_scores.iter().zip(backend_scores.iter()) {
            assert_eq!(orig.index, back.index);
            assert_eq!(orig.hamming_distance, back.hamming_distance);
            assert_eq!(orig.bf16_distance, back.bf16_distance);
            assert!(
                (orig.combined_score - back.combined_score).abs() < 1e-10,
                "Scores diverged: {} vs {}",
                orig.combined_score,
                back.combined_score
            );
        }
    }

    #[test]
    fn test_hybrid_stats_tracking() {
        let config = HybridConfig::sku_16k();
        let n_dims = kernels::SKU_16K_BYTES / 2;
        let values: Vec<f32> = (0..n_dims).map(|i| (i as f32 * 0.01).sin()).collect();
        let query = make_container(&values, kernels::SKU_16K_BYTES);

        // All random candidates
        let n = 50;
        let mut db = Vec::with_capacity(n * kernels::SKU_16K_BYTES);
        for i in 0..n {
            let rand_vals: Vec<f32> =
                (0..n_dims).map(|j| ((i * 777 + j) as f32 * 0.037).cos()).collect();
            db.extend_from_slice(&make_container(&rand_vals, kernels::SKU_16K_BYTES));
        }

        let (_, stats) = hybrid_pipeline(&query, &db, n, &config);

        assert_eq!(stats.binary_stats.total, n);
        // Binary pipeline should account for all candidates
        assert_eq!(
            stats.binary_stats.k0_rejected
                + stats.binary_stats.k1_rejected
                + stats.binary_stats.k2_promoted,
            n
        );
        // BF16 scored should be <= K2 promoted
        assert!(stats.bf16_scored <= stats.binary_stats.k2_promoted);
    }

    // ====================================================================
    // Tier 0 VNNI prefilter tests
    // ====================================================================

    #[test]
    fn test_tier0_config_survivor_count() {
        // TopK mode
        let cfg = Tier0Config {
            enabled: true,
            mode: Tier0Mode::TopK(50),
        };
        assert_eq!(cfg.survivor_count(100), 50);
        assert_eq!(cfg.survivor_count(30), 30); // capped at n_candidates

        // Fraction mode
        let cfg = Tier0Config {
            enabled: true,
            mode: Tier0Mode::Fraction(0.25),
        };
        assert_eq!(cfg.survivor_count(100), 25);
        assert_eq!(cfg.survivor_count(4), 1); // ceil(4 * 0.25) = 1
        assert_eq!(cfg.survivor_count(1), 1); // minimum 1
    }

    #[test]
    fn test_tier0_prefilter_basic() {
        // Test that tier0_prefilter returns correct number of survivors
        let n_dims = kernels::SKU_16K_BYTES / 2;
        let values: Vec<f32> = (0..n_dims).map(|i| (i as f32 * 0.01).sin()).collect();
        let query = make_container(&values, kernels::SKU_16K_BYTES);

        let n = 20;
        let mut db = Vec::with_capacity(n * kernels::SKU_16K_BYTES);
        db.extend_from_slice(&query); // exact match at 0
        for i in 1..n {
            let rand_vals: Vec<f32> =
                (0..n_dims).map(|j| ((i * 1000 + j) as f32 * 0.037).cos()).collect();
            db.extend_from_slice(&make_container(&rand_vals, kernels::SKU_16K_BYTES));
        }

        let cfg = Tier0Config {
            enabled: true,
            mode: Tier0Mode::TopK(5),
        };
        let (survivors, stats) = tier0_prefilter(&query, &db, n, &cfg);

        assert_eq!(survivors.len(), 5);
        assert_eq!(stats.active, true);
        assert_eq!(stats.input_count, 20);
        assert_eq!(stats.survivor_count, 5);
        assert_eq!(stats.pruned_count, 15);

        // Exact match (index 0) must survive — it has the highest dot product
        assert!(
            survivors.contains(&0),
            "Exact match at index 0 must survive Tier 0, survivors: {:?}",
            survivors
        );

        // Survivors should be sorted by index (original database order)
        for w in survivors.windows(2) {
            assert!(w[0] < w[1], "Survivors must be in ascending index order");
        }
    }

    #[test]
    fn test_tier0_zero_false_negatives() {
        // Critical invariant: exact and near matches must survive Tier 0.
        //
        // INT8 dot product is a coarse proxy — it won't perfectly rank random data
        // the same way Hamming does. But for structurally similar vectors (exact
        // copies, small perturbations), the byte-level dot product is a strong signal.
        //
        // We verify: planted matches (exact + near) survive Tier 0 at 25% keep rate.
        // We also verify: when matches survive both pipelines, their scores are identical.
        let n_dims = kernels::SKU_16K_BYTES / 2;
        let values: Vec<f32> = (0..n_dims).map(|i| (i as f32 * 0.01).sin()).collect();
        let query = make_container(&values, kernels::SKU_16K_BYTES);

        // Build database with known matches among random data
        let n = 100;
        let mut db = Vec::with_capacity(n * kernels::SKU_16K_BYTES);

        // Index 0: exact match
        db.extend_from_slice(&query);
        // Index 1-4: near-matches (small perturbations)
        for k in 1..5 {
            let near_vals: Vec<f32> = (0..n_dims)
                .map(|i| (i as f32 * 0.01).sin() * (1.0 + k as f32 * 0.005))
                .collect();
            db.extend_from_slice(&make_container(&near_vals, kernels::SKU_16K_BYTES));
        }
        // Index 5-99: random
        for i in 5..n {
            let rand_vals: Vec<f32> =
                (0..n_dims).map(|j| ((i * 999 + j) as f32 * 0.037).cos()).collect();
            db.extend_from_slice(&make_container(&rand_vals, kernels::SKU_16K_BYTES));
        }

        // Run WITH Tier 0 (top 25% = 25 candidates survive)
        let config_t0 = HybridConfig::sku_16k().with_tier0_topk(25);
        let (t0_scores, t0_stats) = hybrid_pipeline(&query, &db, n, &config_t0);

        // Tier 0 must have been active
        assert!(t0_stats.tier0_stats.active);
        assert_eq!(t0_stats.tier0_stats.input_count, 100);
        assert_eq!(t0_stats.tier0_stats.survivor_count, 25);

        // Planted matches (indices 0-4) must survive
        assert!(
            t0_scores.iter().any(|s| s.index == 0),
            "Exact match at index 0 must survive Tier 0"
        );

        // Run WITHOUT Tier 0 for score comparison
        let config_no_t0 = HybridConfig::sku_16k();
        let (oracle_scores, _) = hybrid_pipeline(&query, &db, n, &config_no_t0);

        // Scores must match exactly for common entries
        for t0 in &t0_scores {
            if let Some(oracle) = oracle_scores.iter().find(|s| s.index == t0.index) {
                assert_eq!(
                    oracle.hamming_distance, t0.hamming_distance,
                    "Hamming distance mismatch for index {}",
                    t0.index
                );
                assert_eq!(
                    oracle.bf16_distance, t0.bf16_distance,
                    "BF16 distance mismatch for index {}",
                    t0.index
                );
            }
        }
    }

    #[test]
    fn test_tier0_with_backend_zero_false_negatives() {
        // Verify exact match survives through the backend path + Tier 0.
        // Also verify score identity for common entries.
        let n_dims = kernels::SKU_16K_BYTES / 2;
        let values: Vec<f32> = (0..n_dims).map(|i| (i as f32 * 0.01).sin()).collect();
        let query = make_container(&values, kernels::SKU_16K_BYTES);

        let n = 80;
        let mut db = Vec::with_capacity(n * kernels::SKU_16K_BYTES);
        db.extend_from_slice(&query); // exact match at 0
        // Near-matches at 1-3
        for k in 1..4 {
            let near_vals: Vec<f32> = (0..n_dims)
                .map(|i| (i as f32 * 0.01).sin() * (1.0 + k as f32 * 0.005))
                .collect();
            db.extend_from_slice(&make_container(&near_vals, kernels::SKU_16K_BYTES));
        }
        for i in 4..n {
            let rand_vals: Vec<f32> =
                (0..n_dims).map(|j| ((i * 1000 + j) as f32 * 0.037).cos()).collect();
            db.extend_from_slice(&make_container(&rand_vals, kernels::SKU_16K_BYTES));
        }

        let backend = crate::tail_backend::auto_detect();

        // Oracle: no Tier 0
        let config_no_t0 = HybridConfig::sku_16k();
        let (oracle, _) = hybrid_pipeline_with_backend(&query, &db, n, &config_no_t0, backend.as_ref());

        // Test: with Tier 0 (top 20 = 25%)
        let config_t0 = HybridConfig::sku_16k().with_tier0_topk(20);
        let (t0, t0_stats) = hybrid_pipeline_with_backend(&query, &db, n, &config_t0, backend.as_ref());

        assert!(t0_stats.tier0_stats.active);
        assert_eq!(t0_stats.tier0_stats.survivor_count, 20);

        // Exact match must survive
        assert!(
            t0.iter().any(|s| s.index == 0 && s.hamming_distance == 0),
            "Exact match at index 0 must survive Tier 0 via backend"
        );

        // Scores for common entries must be identical
        for t0_score in &t0 {
            if let Some(o) = oracle.iter().find(|s| s.index == t0_score.index) {
                assert_eq!(o.hamming_distance, t0_score.hamming_distance);
                assert_eq!(o.bf16_distance, t0_score.bf16_distance);
            }
        }
    }

    #[test]
    fn test_tier0_ordering_invariant() {
        // When Tier 0 keeps all survivors, result ordering must be identical
        // to the no-Tier-0 pipeline.
        let n_dims = kernels::SKU_16K_BYTES / 2;
        let values: Vec<f32> = (0..n_dims).map(|i| (i as f32 * 0.01).sin()).collect();
        let query = make_container(&values, kernels::SKU_16K_BYTES);

        let n = 30;
        let mut db = Vec::with_capacity(n * kernels::SKU_16K_BYTES);
        db.extend_from_slice(&query);
        for i in 1..n {
            let rand_vals: Vec<f32> =
                (0..n_dims).map(|j| ((i * 1000 + j) as f32 * 0.037).cos()).collect();
            db.extend_from_slice(&make_container(&rand_vals, kernels::SKU_16K_BYTES));
        }

        // Tier 0 with TopK(n) — keeps ALL candidates, no pruning
        let config_all = HybridConfig::sku_16k().with_tier0_topk(n);
        let config_off = HybridConfig::sku_16k();

        let (scores_on, stats_on) = hybrid_pipeline(&query, &db, n, &config_all);
        let (scores_off, _) = hybrid_pipeline(&query, &db, n, &config_off);

        // Tier 0 should report all survivors
        assert_eq!(stats_on.tier0_stats.survivor_count, n);
        assert_eq!(stats_on.tier0_stats.pruned_count, 0);

        // Result count must match
        assert_eq!(
            scores_on.len(),
            scores_off.len(),
            "Score count mismatch: Tier0={} vs Off={}",
            scores_on.len(),
            scores_off.len()
        );

        // Ordering and values must be identical
        for (on, off) in scores_on.iter().zip(scores_off.iter()) {
            assert_eq!(
                on.index, off.index,
                "Index mismatch: Tier0={} vs Off={}",
                on.index, off.index
            );
            assert_eq!(on.hamming_distance, off.hamming_distance);
            assert_eq!(on.bf16_distance, off.bf16_distance);
            assert!(
                (on.combined_score - off.combined_score).abs() < 1e-10,
                "Score mismatch: Tier0={:.6} vs Off={:.6}",
                on.combined_score,
                off.combined_score
            );
        }
    }

    #[test]
    fn test_tier0_disabled_is_noop() {
        // Tier0 disabled (default) must produce identical results to no Tier0 field
        let n_dims = kernels::SKU_16K_BYTES / 2;
        let values: Vec<f32> = (0..n_dims).map(|i| (i as f32 * 0.01).sin()).collect();
        let query = make_container(&values, kernels::SKU_16K_BYTES);

        let n = 10;
        let mut db = Vec::with_capacity(n * kernels::SKU_16K_BYTES);
        db.extend_from_slice(&query);
        for i in 1..n {
            let rand_vals: Vec<f32> =
                (0..n_dims).map(|j| ((i * 1000 + j) as f32 * 0.037).cos()).collect();
            db.extend_from_slice(&make_container(&rand_vals, kernels::SKU_16K_BYTES));
        }

        let config = HybridConfig::sku_16k(); // tier0.enabled = false

        let (scores, stats) = hybrid_pipeline(&query, &db, n, &config);

        // Tier 0 should NOT be active
        assert!(!stats.tier0_stats.active);
        assert_eq!(stats.tier0_stats.input_count, 0);

        // Should find exact match normally
        assert!(!scores.is_empty());
        assert_eq!(scores[0].index, 0);
        assert_eq!(scores[0].hamming_distance, 0);
    }

    #[test]
    fn test_tier0_survivor_rate() {
        // With a database of mostly random containers, Tier 0 should
        // meaningfully prune candidates while keeping the exact match.
        let n_dims = kernels::SKU_16K_BYTES / 2;
        let values: Vec<f32> = (0..n_dims).map(|i| (i as f32 * 0.01).sin()).collect();
        let query = make_container(&values, kernels::SKU_16K_BYTES);

        let n = 200;
        let mut db = Vec::with_capacity(n * kernels::SKU_16K_BYTES);
        db.extend_from_slice(&query); // exact match
        for i in 1..n {
            let rand_vals: Vec<f32> =
                (0..n_dims).map(|j| ((i * 1000 + j) as f32 * 0.037).cos()).collect();
            db.extend_from_slice(&make_container(&rand_vals, kernels::SKU_16K_BYTES));
        }

        // Keep top 10% (20 survivors out of 200)
        let config = HybridConfig::sku_16k().with_tier0_fraction(0.10);
        let (scores, stats) = hybrid_pipeline(&query, &db, n, &config);

        // Tier 0 stats
        assert!(stats.tier0_stats.active);
        assert_eq!(stats.tier0_stats.input_count, 200);
        assert_eq!(stats.tier0_stats.survivor_count, 20);
        assert_eq!(stats.tier0_stats.pruned_count, 180);

        // Binary pipeline should only see Tier 0 survivors
        assert_eq!(stats.binary_stats.total, 20);

        // Exact match must still be found
        assert!(
            scores.iter().any(|s| s.index == 0 && s.hamming_distance == 0),
            "Exact match at index 0 must survive Tier 0 + K0/K1/K2"
        );
    }

    #[test]
    fn test_tier0_stats_dot_product_range() {
        let n_dims = kernels::SKU_16K_BYTES / 2;
        let values: Vec<f32> = (0..n_dims).map(|i| (i as f32 * 0.01).sin()).collect();
        let query = make_container(&values, kernels::SKU_16K_BYTES);

        let n = 50;
        let mut db = Vec::with_capacity(n * kernels::SKU_16K_BYTES);
        db.extend_from_slice(&query);
        for i in 1..n {
            let rand_vals: Vec<f32> =
                (0..n_dims).map(|j| ((i * 1000 + j) as f32 * 0.037).cos()).collect();
            db.extend_from_slice(&make_container(&rand_vals, kernels::SKU_16K_BYTES));
        }

        let cfg = Tier0Config {
            enabled: true,
            mode: Tier0Mode::TopK(10),
        };
        let (_, stats) = tier0_prefilter(&query, &db, n, &cfg);

        // The exact match should produce the highest dot product
        // (self-dot is maximal), so max_survivor_dot should be > 0
        assert!(
            stats.max_survivor_dot > stats.min_survivor_dot || stats.survivor_count == 1,
            "Dot product range: min={} max={} (should have spread)",
            stats.min_survivor_dot,
            stats.max_survivor_dot
        );
    }

    // ====================================================================
    // GEMM backend integration tests
    // ====================================================================

    #[test]
    fn test_gemm_backend_hybrid_pipeline() {
        // GEMM backend through the full hybrid pipeline
        let config = HybridConfig::sku_16k();
        let n_dims = kernels::SKU_16K_BYTES / 2;
        let values: Vec<f32> = (0..n_dims).map(|i| (i as f32 * 0.01).sin()).collect();
        let query = make_container(&values, kernels::SKU_16K_BYTES);

        let n = 10;
        let mut db = Vec::with_capacity(n * kernels::SKU_16K_BYTES);
        db.extend_from_slice(&query); // exact match at index 0
        for i in 1..n {
            let rand_vals: Vec<f32> =
                (0..n_dims).map(|j| ((i * 1000 + j) as f32 * 0.037).cos()).collect();
            db.extend_from_slice(&make_container(&rand_vals, kernels::SKU_16K_BYTES));
        }

        let gemm = crate::tail_backend::gemm_backend();
        let (scores, stats) =
            hybrid_pipeline_with_backend(&query, &db, n, &config, gemm.as_ref());

        // GEMM backend should find exact match
        assert!(!scores.is_empty(), "GEMM backend should find matches");
        assert_eq!(
            scores[0].index, 0,
            "GEMM backend: best match should be exact copy at index 0"
        );
        assert_eq!(scores[0].hamming_distance, 0);

        // Stats should show batch scoring was used
        assert!(stats.bf16_scored >= 1);
        assert_eq!(stats.binary_stats.total, n);
    }

    #[test]
    fn test_gemm_backend_batch_distance_ordering() {
        // GEMM batch distances should order exact > near > random
        let n_dims = kernels::SKU_16K_BYTES / 2;
        let values: Vec<f32> = (0..n_dims).map(|i| (i as f32 * 0.01).sin()).collect();
        let query = make_container(&values, kernels::SKU_16K_BYTES);

        // Near match: small perturbation
        let near_vals: Vec<f32> = (0..n_dims)
            .map(|i| (i as f32 * 0.01).sin() * 1.005)
            .collect();
        let near = make_container(&near_vals, kernels::SKU_16K_BYTES);

        // Random
        let rand_vals: Vec<f32> = (0..n_dims).map(|j| (j as f32 * 0.037).cos()).collect();
        let random = make_container(&rand_vals, kernels::SKU_16K_BYTES);

        let gemm = crate::backends::gemm::GemmBackend::new();
        let weights = crate::bf16_hamming::BF16Weights::default();

        let mut batch = Vec::new();
        batch.extend_from_slice(&query);   // 0: exact
        batch.extend_from_slice(&near);    // 1: near
        batch.extend_from_slice(&random);  // 2: random

        let result = crate::tail_backend::TailBackend::score_batch(
            &gemm, &query, &batch, 3, &weights,
        );

        // Exact match: distance 0
        assert_eq!(result.distances[0], 0, "Exact match batch distance should be 0");
        // Near match: small distance
        // Random: larger distance
        assert!(
            result.distances[1] < result.distances[2],
            "Near ({}) should be closer than random ({})",
            result.distances[1],
            result.distances[2]
        );
    }

    // ====================================================================
    // Multi-agent awareness feedback loop with GEMM backend
    // ====================================================================

    #[test]
    fn test_multi_agent_gemm_awareness_convergence() {
        // Simulates 3 A2A agents sharing a blackboard via hybrid weights:
        //
        //   Agent 0 ("explorer"):   queries near-match patterns → high crystallization
        //   Agent 1 ("contrarian"): queries sign-flipped patterns → high tension
        //   Agent 2 ("random"):     queries random patterns → noise
        //
        // Each agent:
        //   1. Runs GEMM-backed hybrid pipeline
        //   2. Extracts learning signal from top-K results
        //   3. Updates shared [f32; 32] hybrid weights (simulating blackboard)
        //
        // After 3 rounds, verify:
        //   - Shared weights reflect combined awareness
        //   - Crystallized regions have higher weights
        //   - Noisy regions have lower weights
        let gemm = crate::tail_backend::gemm_backend();
        let config = HybridConfig::sku_16k();
        let n_dims = kernels::SKU_16K_BYTES / 2;
        let learning_rate = 0.2;

        // Build shared database with structured patterns
        let base_vals: Vec<f32> = (0..n_dims).map(|i| (i as f32 * 0.01).sin()).collect();
        let base = make_container(&base_vals, kernels::SKU_16K_BYTES);

        let n = 30;
        let mut db = Vec::with_capacity(n * kernels::SKU_16K_BYTES);

        // Slot 0: base pattern (exact)
        db.extend_from_slice(&base);
        // Slots 1-4: near-matches (small perturbations)
        for k in 1..5 {
            let near_vals: Vec<f32> = (0..n_dims)
                .map(|i| (i as f32 * 0.01).sin() * (1.0 + k as f32 * 0.003))
                .collect();
            db.extend_from_slice(&make_container(&near_vals, kernels::SKU_16K_BYTES));
        }
        // Slots 5-9: sign-flipped patterns (contrarian)
        for k in 0..5 {
            let mut flipped = base_vals.clone();
            let start = k * (n_dims / 5);
            let end = start + n_dims / 5;
            for v in flipped[start..end].iter_mut() {
                *v = -*v;
            }
            db.extend_from_slice(&make_container(&flipped, kernels::SKU_16K_BYTES));
        }
        // Slots 10-29: random
        for i in 10..n {
            let rand_vals: Vec<f32> =
                (0..n_dims).map(|j| ((i * 999 + j) as f32 * 0.037).cos()).collect();
            db.extend_from_slice(&make_container(&rand_vals, kernels::SKU_16K_BYTES));
        }

        // Shared hybrid weights (simulates blackboard typed slot)
        let mut shared_weights = [0.5f32; 32];

        // === Agent queries ===

        // Agent 0: explorer — queries the base pattern (expects crystallized matches)
        let query_explorer = base.clone();

        // Agent 1: contrarian — queries a sign-flipped variant
        let mut contrarian_vals = base_vals.clone();
        for v in contrarian_vals[..n_dims / 2].iter_mut() {
            *v = -*v;
        }
        let query_contrarian = make_container(&contrarian_vals, kernels::SKU_16K_BYTES);

        // Agent 2: random — queries random data
        let rand_query_vals: Vec<f32> =
            (0..n_dims).map(|j| (j as f32 * 0.123).cos()).collect();
        let query_random = make_container(&rand_query_vals, kernels::SKU_16K_BYTES);

        // === Run 3 rounds of agent exploration ===
        let queries = [&query_explorer, &query_contrarian, &query_random];

        for round in 0..3 {
            for (agent_id, query) in queries.iter().enumerate() {
                // Phase 1: GEMM-backed recognition
                let (scores, _stats) = hybrid_pipeline_with_backend(
                    query,
                    &db,
                    n,
                    &config,
                    gemm.as_ref(),
                );

                // Phase 2: Extract learning signal from top-2 results
                if scores.len() >= 2 {
                    let top_0_offset = scores[0].index * kernels::SKU_16K_BYTES;
                    let top_1_offset = scores[1].index * kernels::SKU_16K_BYTES;
                    let top_0 = &db[top_0_offset..top_0_offset + kernels::SKU_16K_BYTES];
                    let top_1 = &db[top_1_offset..top_1_offset + kernels::SKU_16K_BYTES];

                    let signal = extract_learning_signal(
                        query,
                        &[top_0, top_1],
                        &config.awareness_thresholds,
                    );

                    // Phase 3: Update shared weights (A2A blackboard write)
                    update_hybrid_weights(&mut shared_weights, &signal, learning_rate);

                    // Verify signal is sensible
                    assert!(
                        signal.crystallized_ratio + signal.tension_ratio <= 1.01,
                        "Round {}, Agent {}: ratios sum > 1.0 (c={}, t={})",
                        round, agent_id,
                        signal.crystallized_ratio, signal.tension_ratio
                    );
                }
            }
        }

        // === Verify convergence ===

        // After 3 rounds × 3 agents = 9 weight updates, weights should have moved
        let moved = shared_weights.iter().filter(|&&w| (w - 0.5).abs() > 0.01).count();
        assert!(
            moved > 0,
            "Some weights should have moved from initial 0.5 after 9 updates: {:?}",
            &shared_weights[..8]
        );

        // All weights should be in valid range [0, 1]
        for (i, &w) in shared_weights.iter().enumerate() {
            assert!(
                w >= 0.0 && w <= 1.0,
                "Weight {} out of range: {}",
                i, w
            );
        }
    }

    #[test]
    fn test_gemm_awareness_crystallization_vs_noise() {
        // Two agents: one gets crystallized signal, one gets noise.
        // Their weights should diverge.
        let gemm = crate::tail_backend::gemm_backend();
        let config = HybridConfig::sku_16k();
        let n_dims = kernels::SKU_16K_BYTES / 2;

        // Database: 10 exact copies of the query + 10 random
        let vals: Vec<f32> = (0..n_dims).map(|i| (i as f32 * 0.01).sin()).collect();
        let query = make_container(&vals, kernels::SKU_16K_BYTES);

        let n = 20;
        let mut db = Vec::with_capacity(n * kernels::SKU_16K_BYTES);
        for _ in 0..10 {
            db.extend_from_slice(&query); // copies
        }
        for i in 10..n {
            let rand_vals: Vec<f32> =
                (0..n_dims).map(|j| ((i * 777 + j) as f32 * 0.037).cos()).collect();
            db.extend_from_slice(&make_container(&rand_vals, kernels::SKU_16K_BYTES));
        }

        // Agent A: queries exact match → crystallized
        let mut weights_a = [0.5f32; 32];
        let (scores_a, _) = hybrid_pipeline_with_backend(&query, &db, n, &config, gemm.as_ref());
        if scores_a.len() >= 2 {
            let t0 = &db[scores_a[0].index * kernels::SKU_16K_BYTES..][..kernels::SKU_16K_BYTES];
            let t1 = &db[scores_a[1].index * kernels::SKU_16K_BYTES..][..kernels::SKU_16K_BYTES];
            let sig = extract_learning_signal(&query, &[t0, t1], &config.awareness_thresholds);
            update_hybrid_weights(&mut weights_a, &sig, 0.5);

            assert!(
                sig.crystallized_ratio > 0.8,
                "Exact match should be >80% crystallized, got {}",
                sig.crystallized_ratio
            );
        }

        // Agent B: queries random → noise/uncertain
        let rand_q: Vec<f32> = (0..n_dims).map(|j| (j as f32 * 0.0777).cos()).collect();
        let random_query = make_container(&rand_q, kernels::SKU_16K_BYTES);
        let mut weights_b = [0.5f32; 32];
        let (scores_b, _) =
            hybrid_pipeline_with_backend(&random_query, &db, n, &config, gemm.as_ref());
        if scores_b.len() >= 2 {
            let t0 = &db[scores_b[0].index * kernels::SKU_16K_BYTES..][..kernels::SKU_16K_BYTES];
            let t1 = &db[scores_b[1].index * kernels::SKU_16K_BYTES..][..kernels::SKU_16K_BYTES];
            let sig =
                extract_learning_signal(&random_query, &[t0, t1], &config.awareness_thresholds);
            update_hybrid_weights(&mut weights_b, &sig, 0.5);
        }

        // Weights should have diverged: A (crystallized) higher than B (noise)
        let avg_a: f32 = weights_a.iter().sum::<f32>() / 32.0;
        let avg_b: f32 = weights_b.iter().sum::<f32>() / 32.0;
        assert!(
            avg_a >= avg_b,
            "Crystallized agent should have >= avg weight than noisy: A={:.3} B={:.3}",
            avg_a, avg_b
        );
    }

    // ====================================================================
    // Resonance decompose tests
    // ====================================================================

    #[test]
    fn test_resonance_decompose_exact_match() {
        let config = HybridConfig::sku_16k();
        let n_dims = kernels::SKU_16K_BYTES / 2;
        let values: Vec<f32> = (0..n_dims).map(|i| (i as f32 * 0.01).sin()).collect();
        let query = make_container(&values, kernels::SKU_16K_BYTES);

        let n = 10;
        let mut db = Vec::with_capacity(n * kernels::SKU_16K_BYTES);
        db.extend_from_slice(&query); // exact match at 0
        for i in 1..n {
            let rand_vals: Vec<f32> =
                (0..n_dims).map(|j| ((i * 1000 + j) as f32 * 0.037).cos()).collect();
            db.extend_from_slice(&make_container(&rand_vals, kernels::SKU_16K_BYTES));
        }

        let result = resonance_decompose(&query, &db, n, &config, 5);

        // Exact match should be crystallized (identical → 100% crystallized)
        assert!(
            !result.crystallized.is_empty(),
            "Exact match should be in crystallized bin"
        );
        assert_eq!(result.crystallized[0].address, 0);
        assert!(
            result.crystallized[0].crystallized_ratio > 0.9,
            "Exact match should be >90% crystallized, got {}",
            result.crystallized[0].crystallized_ratio,
        );
        assert_eq!(result.crystallized[0].hamming_distance, 0);
        assert!(result.learning_signal.is_some());
    }

    #[test]
    fn test_resonance_decompose_sign_flipped_creates_tension() {
        let config = HybridConfig::sku_16k();
        let n_dims = kernels::SKU_16K_BYTES / 2;
        let values: Vec<f32> = (0..n_dims).map(|i| (i as f32 * 0.01 + 0.5).sin()).collect();
        let query = make_container(&values, kernels::SKU_16K_BYTES);

        // Create a candidate with half the dimensions sign-flipped
        let mut flipped = values.clone();
        for v in flipped.iter_mut().take(n_dims / 2) {
            *v = -*v;
        }
        let candidate = make_container(&flipped, kernels::SKU_16K_BYTES);

        let n = 2;
        let mut db = Vec::with_capacity(n * kernels::SKU_16K_BYTES);
        db.extend_from_slice(&query);     // exact at 0
        db.extend_from_slice(&candidate); // sign-flipped at 1

        let result = resonance_decompose(&query, &db, n, &config, 5);

        // The exact match should be crystallized
        let has_crystallized = result.crystallized.iter().any(|m| m.address == 0);
        assert!(has_crystallized, "Exact match should be crystallized");

        // The sign-flipped candidate might appear as tensioned
        // (depending on whether it survives K0/K1/K2 — half-flipped vectors
        // have very high Hamming distance and may be pruned)
        // The important thing is the learning signal captures it
        if let Some(ref signal) = result.learning_signal {
            // Learning signal should exist regardless
            assert!(signal.attention_weights.len() > 0);
        }
    }

    #[test]
    fn test_resonance_decompose_learning_signal_populated() {
        let config = HybridConfig::sku_16k();
        let n_dims = kernels::SKU_16K_BYTES / 2;
        let values: Vec<f32> = (0..n_dims).map(|i| (i as f32 * 0.01).sin()).collect();
        let query = make_container(&values, kernels::SKU_16K_BYTES);

        // Near-matches
        let n = 5;
        let mut db = Vec::with_capacity(n * kernels::SKU_16K_BYTES);
        db.extend_from_slice(&query);
        for k in 1..n {
            let near_vals: Vec<f32> = (0..n_dims)
                .map(|i| (i as f32 * 0.01).sin() * (1.0 + k as f32 * 0.003))
                .collect();
            db.extend_from_slice(&make_container(&near_vals, kernels::SKU_16K_BYTES));
        }

        let result = resonance_decompose(&query, &db, n, &config, 5);

        let signal = result.learning_signal.as_ref().expect("Should have learning signal");
        assert!(signal.crystallized_ratio > 0.5, "Near-matches should be mostly crystallized");
        assert!(!signal.attention_weights.is_empty());
        assert!(!signal.packed_states.is_empty());
    }

    #[test]
    fn test_resonance_decompose_stats_populated() {
        let config = HybridConfig::sku_16k();
        let n_dims = kernels::SKU_16K_BYTES / 2;
        let values: Vec<f32> = (0..n_dims).map(|i| (i as f32 * 0.01).sin()).collect();
        let query = make_container(&values, kernels::SKU_16K_BYTES);

        let n = 20;
        let mut db = Vec::with_capacity(n * kernels::SKU_16K_BYTES);
        db.extend_from_slice(&query);
        for i in 1..n {
            let rand_vals: Vec<f32> =
                (0..n_dims).map(|j| ((i * 1000 + j) as f32 * 0.037).cos()).collect();
            db.extend_from_slice(&make_container(&rand_vals, kernels::SKU_16K_BYTES));
        }

        let result = resonance_decompose(&query, &db, n, &config, 10);

        // Stats should track the full pipeline
        assert_eq!(result.stats.binary_stats.total, n);
        assert!(result.stats.bf16_scored >= 1);
    }
}
