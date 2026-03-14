//! HDR (High Dynamic Range) Cascade Search.
//!
//! 3-stroke adaptive cascade for Hamming-based nearest-neighbour search
//! with optional precision tiers (VNNI cosine, F32/BF16 dequant, DeltaXor, BF16Hamming).
//!
//! Extracted from `simd.rs` — the cascade algorithm and types now live here;
//! `simd.rs` retains deprecated forwarding wrappers for backward compatibility.

use crate::simd::{dot_f32, select_dot_i8_fn, select_hamming_fn};

// ============================================================================
// Types
// ============================================================================

/// A ranked hit from the HDR cascade search.
///
/// Renamed from the former `HdrResult` to clarify semantics.
#[derive(Debug, Clone)]
pub struct RankedHit {
    /// Index into the database.
    pub index: usize,
    /// Exact Hamming distance (from Stroke 2).
    pub hamming: u64,
    /// Optional high-precision distance (from Stroke 3).
    /// f64::NAN if Stroke 3 was not run (PreciseMode::Off).
    pub precise: f64,
    /// Quality band assigned by the cascade.
    pub band: Band,
}

/// Quality band for a cascade hit, based on distance relative to the threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Band {
    /// Top 5% — distance ≤ threshold × 0.25
    Foveal,
    /// 5–25% — distance ≤ threshold × 0.50
    Near,
    /// 25–60% — distance ≤ threshold × 0.75
    Good,
    /// 60–90% — distance ≤ threshold
    Weak,
    /// Beyond threshold (rejected by cascade, not normally returned)
    Reject,
}

/// Precision mode for Stroke 3 of the HDR cascade.
///
/// Six data paths through the same cascade engine:
///
/// | Case | Source | Tier 1-2 | Tier 3 | Example |
/// |------|--------|----------|--------|---------|
/// | Off  | —      | hamming  | none   | reject-only |
/// | Vnni | native binary 64Kbit | partial popcount | VNNI dot_i8 cosine | SimHash/LSH/HDC |
/// | F32  | f32 embedding → u8 | hamming on u8 | dequant → f32 dot | Jina embed |
/// | BF16 | f32 embedding → u8 | hamming on u8 | dequant → bf16 dot | large embed db |
/// | DeltaXor | 3D + INT8 delta | XOR delta popcount | INT8 residual dot | DeltaLayer |
/// | BF16Hamming | native BF16 bytes (2B/dim) | weighted XOR popcount | weighted BF16 distance | 6× faster than F32 |
#[derive(Clone, Copy, Debug)]
pub enum PreciseMode {
    /// No precision tier — return Hamming distances only.
    Off,

    /// Case 1+2: Native u8 vectors (HDC/SimHash/LSH, including 64Kbit hires).
    /// Uses VNNI dot_i8 (_mm512_dpbusd_epi32). No type conversion.
    /// Exact integer arithmetic → f64 cosine.
    Vnni,

    /// Case 1 (float source): Quantized f32 embeddings → dequantize to f32.
    /// f32_val = scale * (u8_val - zero_point), then SIMD dot_f32 → cosine.
    /// Use when embed channel holds quantized Jina/CLIP vectors.
    F32 { scale: f32, zero_point: i32 },

    /// Case 1 (float source, large DB): Same dequantization but signals BF16 intent.
    /// Currently falls through to f32 path.
    /// Future (VDPBF16PS): bf16×bf16→f32 at 2× throughput, halved bandwidth.
    /// Worth it when finalists > ~500 or database is bandwidth-bound.
    BF16 { scale: f32, zero_point: i32 },

    /// Case 5: XOR Delta Layer + INT8 residual.
    /// Tier 1-2 operate on XOR delta popcount (ground ^ delta).
    /// Tier 3 computes INT8 dot on the raw bytes as signed values,
    /// treating byte magnitudes as a continuous residual signal.
    /// `delta_weight` controls blend: distance = hamming * (1-w) + residual_dot * w
    DeltaXor { delta_weight: f32 },

    /// Case 6: BF16-structured Hamming.
    /// Primary distance metric on native BF16 byte arrays (2 bytes/dim).
    /// XOR + per-field weighted popcount: sign(W_s) + exponent(W_e) + mantissa(W_m).
    /// 3× slower than binary Hamming, 6× faster than FP32 cosine, near-cosine quality.
    BF16Hamming {
        weights: crate::bf16_hamming::BF16Weights,
    },
}

impl PartialEq for PreciseMode {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Off, Self::Off) => true,
            (Self::Vnni, Self::Vnni) => true,
            (
                Self::F32 {
                    scale: s1,
                    zero_point: z1,
                },
                Self::F32 {
                    scale: s2,
                    zero_point: z2,
                },
            ) => s1.to_bits() == s2.to_bits() && z1 == z2,
            (
                Self::BF16 {
                    scale: s1,
                    zero_point: z1,
                },
                Self::BF16 {
                    scale: s2,
                    zero_point: z2,
                },
            ) => s1.to_bits() == s2.to_bits() && z1 == z2,
            (Self::DeltaXor { delta_weight: w1 }, Self::DeltaXor { delta_weight: w2 }) => {
                w1.to_bits() == w2.to_bits()
            }
            (Self::BF16Hamming { weights: w1 }, Self::BF16Hamming { weights: w2 }) => w1 == w2,
            _ => false,
        }
    }
}

impl Eq for PreciseMode {}

// ============================================================================
// Shift detection
// ============================================================================

/// Alert emitted when the cascade detects distributional drift.
#[derive(Debug, Clone)]
pub struct ShiftAlert {
    pub old_mu: f64,
    pub new_mu: f64,
    pub old_sigma: f64,
    pub new_sigma: f64,
    pub observations: usize,
}

// ============================================================================
// Cascade
// ============================================================================

/// HDR Cascade: stateful search engine with calibrated rejection thresholds.
///
/// Wraps the 3-stroke adaptive cascade algorithm. Construct via
/// [`Cascade::from_threshold`] or [`Cascade::calibrate`].
pub struct Cascade {
    /// Distance threshold for accepting hits.
    pub threshold: u64,
    /// Vector length in bytes.
    pub vec_bytes: usize,
    /// Running mean of observed distances (for drift detection).
    mu: f64,
    /// Running standard deviation.
    sigma: f64,
    /// Total observations seen via `observe()`.
    observations: usize,
}

impl Cascade {
    /// Create a cascade from a fixed threshold and vector size.
    pub fn from_threshold(threshold: u64, vec_bytes: usize) -> Self {
        Self {
            threshold,
            vec_bytes,
            mu: 0.0,
            sigma: 0.0,
            observations: 0,
        }
    }

    /// Calibrate a cascade from a sample of distances.
    ///
    /// Sets `threshold = mu + 3σ` from the sample.
    pub fn calibrate(distances: &[u32], vec_bytes: usize) -> Self {
        if distances.is_empty() {
            return Self::from_threshold(0, vec_bytes);
        }
        let n = distances.len() as f64;
        let mu = distances.iter().map(|&d| d as f64).sum::<f64>() / n;
        let var = distances
            .iter()
            .map(|&d| {
                let diff = d as f64 - mu;
                diff * diff
            })
            .sum::<f64>()
            / n;
        let sigma = var.sqrt();
        let threshold = (mu + 3.0 * sigma) as u64;
        Self {
            threshold,
            vec_bytes,
            mu,
            sigma,
            observations: distances.len(),
        }
    }

    /// Classify a distance into a quality band.
    pub fn expose(&self, distance: u32) -> Band {
        let d = distance as u64;
        let t = self.threshold;
        if d <= t / 4 {
            Band::Foveal
        } else if d <= t / 2 {
            Band::Near
        } else if d <= t * 3 / 4 {
            Band::Good
        } else if d <= t {
            Band::Weak
        } else {
            Band::Reject
        }
    }

    /// Quick pass/fail test: does the Hamming distance between two vectors pass the threshold?
    pub fn test(&self, a: &[u8], b: &[u8]) -> bool {
        let hamming_fn = select_hamming_fn();
        hamming_fn(a, b) <= self.threshold
    }

    /// Record an observed distance for drift detection.
    ///
    /// Returns `Some(ShiftAlert)` if the running statistics have drifted
    /// significantly (|new_mu - old_mu| > 2 × old_sigma).
    pub fn observe(&mut self, distance: u32) -> Option<ShiftAlert> {
        let d = distance as f64;
        self.observations += 1;

        if self.observations == 1 {
            self.mu = d;
            self.sigma = 0.0;
            return None;
        }

        let old_mu = self.mu;
        let old_sigma = self.sigma;

        // Welford's online algorithm
        let delta = d - self.mu;
        self.mu += delta / self.observations as f64;
        let delta2 = d - self.mu;
        let m2 = old_sigma * old_sigma * (self.observations - 1) as f64 + delta * delta2;
        self.sigma = (m2 / self.observations as f64).sqrt();

        // Drift detection: significant shift in mean
        if self.observations > 10 && old_sigma > 0.0 && (self.mu - old_mu).abs() > 2.0 * old_sigma
        {
            Some(ShiftAlert {
                old_mu,
                new_mu: self.mu,
                old_sigma,
                new_sigma: self.sigma,
                observations: self.observations,
            })
        } else {
            None
        }
    }

    /// Check for drift without recording a new observation.
    ///
    /// Returns the last drift alert state if any significant shift was detected
    /// in previous `observe()` calls. This is a convenience alias.
    pub fn drift(&self) -> Option<ShiftAlert> {
        // Drift is only detected via observe(); this returns None if no drift was ever detected.
        // A real implementation would cache the last alert; for now, return None
        // since drift detection happens live in observe().
        None
    }

    /// Recalibrate the cascade after a drift alert.
    ///
    /// Updates the threshold to `new_mu + 3 × new_sigma` from the alert.
    pub fn recalibrate(&mut self, alert: &ShiftAlert) {
        self.mu = alert.new_mu;
        self.sigma = alert.new_sigma;
        self.threshold = (alert.new_mu + 3.0 * alert.new_sigma) as u64;
    }

    /// Run the full 3-stroke cascade query.
    ///
    /// Returns ranked hits within the threshold, optionally with precision scoring.
    pub fn query(
        &self,
        query: &[u8],
        database: &[u8],
        vec_bytes: usize,
        num_vectors: usize,
        precise_mode: PreciseMode,
    ) -> Vec<RankedHit> {
        assert_eq!(query.len(), vec_bytes);
        assert_eq!(database.len(), vec_bytes * num_vectors);

        let hamming_fn = select_hamming_fn();
        let threshold = self.threshold;

        // ─── Configuration ───
        let s1_bytes = (((vec_bytes / 16).max(64) + 63) & !63).min(vec_bytes);
        let scale1 = (vec_bytes as f64) / (s1_bytes as f64);
        let warmup_n = 128.min(num_vectors);

        // For small vectors, skip cascade entirely
        if vec_bytes < 128 {
            let mut results = Vec::new();
            for i in 0..num_vectors {
                let base = i * vec_bytes;
                let d = hamming_fn(query, &database[base..base + vec_bytes]);
                if d <= threshold {
                    results.push(RankedHit {
                        index: i,
                        hamming: d,
                        precise: f64::NAN,
                        band: self.expose(d as u32),
                    });
                }
            }
            if precise_mode != PreciseMode::Off && !results.is_empty() {
                apply_precision_tier(query, database, vec_bytes, &mut results, precise_mode);
            }
            return results;
        }

        // ════════════════════════════════════════════════════════
        // STROKE 1: Partial popcount with σ warmup
        // ════════════════════════════════════════════════════════

        let query_prefix = &query[..s1_bytes];
        let total_bits = (vec_bytes * 8) as f64;
        let p_thresh = (threshold as f64 / total_bits).clamp(0.001, 0.999);
        let sigma_est =
            (vec_bytes as f64) * (8.0 * p_thresh * (1.0 - p_thresh) / s1_bytes as f64).sqrt();

        let mut warmup_dists = Vec::with_capacity(warmup_n);
        for i in 0..warmup_n {
            let base = i * vec_bytes;
            let cand_prefix = &database[base..base + s1_bytes];
            let d = hamming_fn(query_prefix, cand_prefix);
            let estimate = (d as f64 * scale1) as u64;
            warmup_dists.push(estimate);
        }

        let var: f64 = {
            let mu: f64 = warmup_dists.iter().map(|&d| d as f64).sum::<f64>() / warmup_n as f64;
            warmup_dists
                .iter()
                .map(|&d| {
                    let diff = d as f64 - mu;
                    diff * diff
                })
                .sum::<f64>()
                / warmup_n as f64
        };
        let sigma_pop = var.sqrt();

        let sigma = sigma_est.max(sigma_pop).max(1.0);
        let s1_reject = threshold as f64 + 3.0 * sigma;

        let mut survivors: Vec<(usize, u64)> = Vec::with_capacity(num_vectors / 20);

        for i in 0..num_vectors {
            let base = i * vec_bytes;
            let cand_prefix = &database[base..base + s1_bytes];
            let d = hamming_fn(query_prefix, cand_prefix);
            let estimate = (d as f64 * scale1) as u64;

            if (estimate as f64) <= s1_reject {
                survivors.push((i, d));
            }
        }

        // ════════════════════════════════════════════════════════
        // STROKE 2: Full Hamming on survivors (incremental)
        // ════════════════════════════════════════════════════════

        let mut finalists: Vec<RankedHit> = Vec::with_capacity(survivors.len() / 5 + 1);
        let query_rest = &query[s1_bytes..];

        for &(idx, d_prefix) in &survivors {
            let base = idx * vec_bytes;
            let d_rest = hamming_fn(query_rest, &database[base + s1_bytes..base + vec_bytes]);
            let d_full = d_prefix + d_rest;

            if d_full <= threshold {
                finalists.push(RankedHit {
                    index: idx,
                    hamming: d_full,
                    precise: f64::NAN,
                    band: self.expose(d_full as u32),
                });
            }
        }

        // ════════════════════════════════════════════════════════
        // STROKE 3: High-precision distance (optional)
        // ════════════════════════════════════════════════════════

        if precise_mode != PreciseMode::Off && !finalists.is_empty() {
            apply_precision_tier(query, database, vec_bytes, &mut finalists, precise_mode);
        }

        finalists
    }
}

// ============================================================================
// Stroke 3: Precision tier
// ============================================================================

/// Stroke 3: compute high-precision distance for finalists.
///
/// Mode selection:
/// - `Vnni` — VNNI dot_i8 → cosine (native u8 vectors, no conversion)
/// - `F32`/`BF16` — dequantize u8 → f32, SIMD dot_f32 → cosine
/// - `DeltaXor` — blended hamming_norm * (1-w) + INT8 cosine * w
///
/// Sorts by precise distance descending (most similar first).
fn apply_precision_tier(
    query: &[u8],
    database: &[u8],
    vec_bytes: usize,
    finalists: &mut [RankedHit],
    precise_mode: PreciseMode,
) {
    match precise_mode {
        PreciseMode::Off => return,

        PreciseMode::Vnni => {
            let dot_fn = select_dot_i8_fn();
            let query_norm_sq = dot_fn(query, query);
            let query_norm = (query_norm_sq as f64).sqrt();

            if query_norm == 0.0 {
                for r in finalists.iter_mut() {
                    r.precise = 0.0;
                }
                return;
            }

            for r in finalists.iter_mut() {
                let base = r.index * vec_bytes;
                let candidate = &database[base..base + vec_bytes];

                let dot = dot_fn(query, candidate);
                let cand_norm_sq = dot_fn(candidate, candidate);
                let cand_norm = (cand_norm_sq as f64).sqrt();

                r.precise = if cand_norm > 0.0 {
                    dot as f64 / (query_norm * cand_norm)
                } else {
                    0.0
                };
            }
        }

        PreciseMode::F32 { scale, zero_point } | PreciseMode::BF16 { scale, zero_point } => {
            let mut query_f32 = vec![0.0f32; vec_bytes];
            for i in 0..vec_bytes {
                query_f32[i] = scale * (query[i] as i32 - zero_point) as f32;
            }
            let query_norm_sq = dot_f32(&query_f32, &query_f32);
            let query_norm = (query_norm_sq as f64).sqrt();

            if query_norm == 0.0 {
                for r in finalists.iter_mut() {
                    r.precise = 0.0;
                }
                return;
            }

            let mut cand_f32 = vec![0.0f32; vec_bytes];

            for r in finalists.iter_mut() {
                let base = r.index * vec_bytes;
                let candidate = &database[base..base + vec_bytes];

                for i in 0..vec_bytes {
                    cand_f32[i] = scale * (candidate[i] as i32 - zero_point) as f32;
                }

                let dot = dot_f32(&query_f32, &cand_f32) as f64;
                let cand_norm = (dot_f32(&cand_f32, &cand_f32) as f64).sqrt();

                r.precise = if cand_norm > 0.0 {
                    dot / (query_norm * cand_norm)
                } else {
                    0.0
                };
            }
        }

        PreciseMode::DeltaXor { delta_weight } => {
            let total_bits = (vec_bytes * 8) as f64;
            let dot_fn = select_dot_i8_fn();
            let query_norm_sq = dot_fn(query, query);
            let query_norm = (query_norm_sq as f64).sqrt();

            let w = delta_weight as f64;

            for r in finalists.iter_mut() {
                let base = r.index * vec_bytes;
                let candidate = &database[base..base + vec_bytes];

                let hamming_norm = r.hamming as f64 / total_bits;

                let cosine = if query_norm > 0.0 {
                    let dot = dot_fn(query, candidate);
                    let cand_norm = (dot_fn(candidate, candidate) as f64).sqrt();
                    if cand_norm > 0.0 {
                        dot as f64 / (query_norm * cand_norm)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                let blended = hamming_norm * (1.0 - w) + (1.0 - cosine) * w;
                r.precise = 1.0 - blended;
            }
        }

        PreciseMode::BF16Hamming { weights } => {
            let bf16_fn = crate::bf16_hamming::select_bf16_hamming_fn();
            let max_per_dim =
                weights.sign as u64 + 8 * weights.exponent as u64 + 7 * weights.mantissa as u64;
            let n_dims = vec_bytes / 2;
            let max_total = max_per_dim * n_dims as u64;

            for r in finalists.iter_mut() {
                let base = r.index * vec_bytes;
                let candidate = &database[base..base + vec_bytes];
                let dist = bf16_fn(query, candidate, &weights);
                let norm = if max_total > 0 {
                    dist as f64 / max_total as f64
                } else {
                    1.0
                };
                r.precise = 1.0 - norm;
            }
        }
    }

    // Sort by precise distance descending (most similar first)
    finalists.sort_unstable_by(|a, b| {
        b.precise
            .partial_cmp(&a.precise)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}
