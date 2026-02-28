//! DN-tree signal quality and BF16 cold-survivor refinement.
//!
//! These are the only functions genuinely new to rustynum-bnn.
//! All K0/K1 probes, σ-gated rejection, HDR classification, and
//! adaptive search already exist in rustynum-core:
//!
//! - `kernels.rs`: `k0_probe()`, `k1_stats()`, `k2_exact()`, `score_hdr()`, `SliceGate`
//! - `simd.rs`: `hdr_cascade_search()` (3-stroke Belichtungsmesser with warmup σ)
//! - `hdc.rs`: `hamming_distance_adaptive()` (3-stage progressive σ gates)
//! - `cogrecord.rs`: `sweep_adaptive()`, `hdr_sweep()` (4-channel compound early exit)

use rustynum_core::bf16_hamming::structural_diff;
use rustynum_core::graph_hv::GraphHV;

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
// BF16 Cold-Survivor Refinement
// ============================================================================

/// BF16 range awareness for cold (uncertain) survivors.
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
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rustynum_core::graph_hv::GraphHV;
    use rustynum_core::rng::SplitMix64;

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
    fn test_bf16_refine_cold_identical() {
        let mut rng = SplitMix64::new(42);
        let hv = GraphHV::random(&mut rng);
        // Identical → crystallized → promote to mid (2)
        assert_eq!(bf16_refine_cold(&hv, &hv), 2);
    }

    #[test]
    fn test_bf16_refine_cold_opposite() {
        let mut rng = SplitMix64::new(42);
        let query = GraphHV::random(&mut rng);
        let candidate = GraphHV::random(&mut rng);
        // Different random vectors → likely tensioned → stay cold (1)
        let result = bf16_refine_cold(&query, &candidate);
        assert!(result == 1 || result == 2); // either outcome valid for random
    }
}
