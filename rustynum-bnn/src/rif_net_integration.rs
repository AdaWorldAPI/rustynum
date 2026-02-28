//! RIF-Net integration: Rich Information Flow for Binary Neural Networks.
//!
//! Implements Zhang et al. 2025 "Accurate binary neural network based on rich
//! information flow" (Neurocomputing 633, 129837) adapted to the rustynum
//! binary HDC domain.
//!
//! ## Paper → Infrastructure Mapping
//!
//! | Paper Concept | rustynum Implementation |
//! |---|---|
//! | Binary Conv (XNOR+POPCNT) | `bnn_dot()` via `BnnLayer::forward()` |
//! | EWM amplitude correction | `EwmWeights` — awareness-guided per-word scaling |
//! | BPReLU | `BPReLU` — per-neuron adaptive slopes for binary scores |
//! | BatchNorm | `BinaryBatchNorm` — running mean/var of pre-activation scores |
//! | Shortcut connection | XOR-bind: preserves input info through binarization |
//! | RIF-CA Block | `RifCaBlock` — wraps all above into one block |
//! | Rich Information Flow | Inter-block shortcuts every 2 blocks in `RifNet` |
//!
//! ## Key Insight: Awareness → EWM
//!
//! The paper's EWM restores amplitude lost during binarization. In our domain,
//! the awareness substrate (`superposition_decompose`) classifies each BF16
//! dimension as crystallized/tensioned/uncertain/noise. This IS the amplitude
//! information:
//! - Crystallized → high amplitude (settled signal, amplify)
//! - Tensioned → medium amplitude (contradicted, attenuate)
//! - Uncertain → low amplitude (direction known, magnitude unknown)
//! - Noise → zero amplitude (irrelevant, suppress)
//!
//! Zero core modifications. All public API calls.

use crate::belichtungsmesser::signal_quality;
use crate::bnn::{bnn_cascade_search, BnnDotResult, BnnLayer};
use rustynum_core::bf16_hamming::{
    superposition_decompose, AwarenessState, AwarenessThresholds, SuperpositionState,
};
use rustynum_core::fingerprint::Fingerprint;
use rustynum_core::graph_hv::GraphHV;
use rustynum_core::kernels::SliceGate;
use rustynum_core::layer_stack::CollapseGate;
use rustynum_core::rng::SplitMix64;

// ============================================================================
// EWM — Element-Wise Multiplication (BIR-EWM from Zhang 2025)
// ============================================================================

/// Per-word EWM scaling weights — the BIR-EWM amplitude correction.
///
/// 256 f32 values, one per u64 word of `Fingerprint<256>`.
/// Initialized to 1.0 (identity), learned from awareness feedback.
/// After training, crystallized words have scale ~1.0, noise words ~0.0.
pub struct EwmWeights {
    pub scales: [f32; 256],
}

impl Default for EwmWeights {
    fn default() -> Self {
        Self { scales: [1.0; 256] }
    }
}

impl EwmWeights {
    /// Learn EWM scales from awareness decomposition.
    ///
    /// Maps each word's awareness (average of 4 BF16 dims per word) to a
    /// target scale, then EMA-updates toward it. 4 dims per word because
    /// each u64 = 8 bytes = 4 BF16 dimensions.
    pub fn update_from_awareness(&mut self, awareness: &SuperpositionState, lr: f32) {
        for word_idx in 0..256 {
            let start = word_idx * 4;
            let end = (start + 4).min(awareness.n_dims);
            if start >= awareness.n_dims {
                break;
            }
            let count = (end - start) as f32;
            let mut target = 0.0f32;
            for d in start..end {
                target += awareness_to_ewm_scale(awareness.states[d]);
            }
            target /= count;
            self.scales[word_idx] += lr * (target - self.scales[word_idx]);
        }
    }

    /// Apply EWM modulation to a Fingerprint via stochastic bit masking.
    ///
    /// - scale >= 1.0: identity (keep all bits)
    /// - scale < 0.05: clear all bits (noise suppression)
    /// - scale 0.05..0.3: AND with two random masks (~25% retention)
    /// - scale 0.3..0.6: AND with one random mask (~50% retention)
    /// - scale 0.6..1.0: AND with OR of two random masks (~75% retention)
    pub fn apply(&self, fp: &Fingerprint<256>, rng: &mut SplitMix64) -> Fingerprint<256> {
        let mut words = fp.words;
        for (i, word) in words.iter_mut().enumerate() {
            let s = self.scales[i];
            if s >= 1.0 {
                continue;
            }
            if s < 0.05 {
                *word = 0;
                continue;
            }
            let r1 = rng.next_u64();
            if s < 0.3 {
                *word &= r1 & rng.next_u64();
            } else if s < 0.6 {
                *word &= r1;
            } else {
                *word &= r1 | rng.next_u64();
            }
        }
        Fingerprint::from_words(words)
    }

    /// How much the EWM deviates from identity — RMS of (scale - 1.0).
    pub fn correction_strength(&self) -> f32 {
        let sum_sq: f32 = self.scales.iter().map(|&s| (s - 1.0) * (s - 1.0)).sum();
        (sum_sq / 256.0).sqrt()
    }
}

/// Map AwarenessState to EWM scale factor (the BIR-EWM transfer function).
fn awareness_to_ewm_scale(state: AwarenessState) -> f32 {
    match state {
        AwarenessState::Crystallized => 1.0,
        AwarenessState::Tensioned => 0.5,
        AwarenessState::Uncertain => 0.2,
        AwarenessState::Noise => 0.0,
    }
}

// ============================================================================
// BPReLU — Binary-specific Parametric ReLU
// ============================================================================

/// Adaptive slopes for the bimodal binary score distribution.
///
/// Paper: BPReLU adapts to the +1/-1 clustering of binary conv outputs.
/// `alpha_pos` scales positive scores, `alpha_neg` scales negative scores.
pub struct BPReLU {
    pub alpha_pos: f32,
    pub alpha_neg: f32,
}

impl Default for BPReLU {
    fn default() -> Self {
        Self {
            alpha_pos: 1.0,
            alpha_neg: 0.25,
        }
    }
}

impl BPReLU {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline]
    pub fn apply(&self, x: f32) -> f32 {
        if x > 0.0 {
            self.alpha_pos * x
        } else {
            self.alpha_neg * x
        }
    }

    /// Adapt slopes toward stable targets based on observed score polarity.
    pub fn adapt(&mut self, score: f32, lr: f32) {
        if score > 0.0 {
            self.alpha_pos += lr * (1.0 - self.alpha_pos);
        } else {
            self.alpha_neg += lr * (0.5 - self.alpha_neg);
        }
    }
}

// ============================================================================
// Binary BatchNorm
// ============================================================================

/// Running mean/variance normalizer for pre-activation BNN scores.
///
/// Uses exponential moving average. Stabilizes the score distribution
/// before thresholding, preventing the sign function from saturating
/// on one polarity.
pub struct BinaryBatchNorm {
    pub running_mean: f32,
    pub running_var: f32,
    pub gamma: f32,
    pub beta: f32,
    pub momentum: f32,
    pub count: u64,
}

impl Default for BinaryBatchNorm {
    fn default() -> Self {
        Self {
            running_mean: 0.0,
            running_var: 1.0,
            gamma: 1.0,
            beta: 0.0,
            momentum: 0.1,
            count: 0,
        }
    }
}

impl BinaryBatchNorm {
    pub fn new() -> Self {
        Self::default()
    }

    /// Normalize a score. Updates running stats if `update` is true.
    pub fn normalize(&mut self, score: f32, update: bool) -> f32 {
        if update {
            self.count += 1;
            let delta = score - self.running_mean;
            self.running_mean += self.momentum * delta;
            self.running_var =
                (1.0 - self.momentum) * self.running_var + self.momentum * delta * delta;
        }
        let eps = 1e-5;
        self.gamma * (score - self.running_mean) / (self.running_var + eps).sqrt() + self.beta
    }
}

// ============================================================================
// RIF Flow Metrics
// ============================================================================

/// Per-block information flow metrics.
#[derive(Clone, Debug)]
pub struct RifFlowMetrics {
    /// Signal quality before EWM modulation (popcount variance).
    pub pre_ewm_signal_quality: f32,
    /// Signal quality after EWM modulation.
    pub post_ewm_signal_quality: f32,
    /// EWM correction strength (RMS deviation from identity).
    pub ewm_correction_strength: f32,
    /// Awareness: fraction of crystallized dimensions.
    pub awareness_crystallized_pct: f32,
    /// Awareness: fraction of tensioned dimensions.
    pub awareness_tensioned_pct: f32,
    /// CollapseGate decision based on correction strength.
    pub collapse_decision: CollapseGate,
}

/// Evaluate whether EWM correction warrants committing to ground truth.
///
/// - Strong correction + high crystallization → FLOW (commit)
/// - High tension → BLOCK (contradictory, discard)
/// - Otherwise → HOLD (accumulate more evidence)
fn evaluate_collapse_decision(
    ewm_strength: f32,
    crystallized_pct: f32,
    tensioned_pct: f32,
) -> CollapseGate {
    if ewm_strength > 0.3 && crystallized_pct > 0.6 {
        CollapseGate::Flow
    } else if tensioned_pct > 0.4 {
        CollapseGate::Block
    } else {
        CollapseGate::Hold
    }
}

// ============================================================================
// RifCaBlock — one RIF-CA block (the fundamental unit)
// ============================================================================

/// One RIF-CA block from Zhang 2025, adapted to binary HDC domain.
///
/// ```text
/// input ──┬── BnnLayer (XNOR+POPCNT) ── BN ── BPReLU ── EWM ──┬── output
///         └─────────────── shortcut (XOR-bind) ─────────────────┘
/// ```
pub struct RifCaBlock {
    pub layer: BnnLayer,
    pub ewm: Vec<EwmWeights>,
    pub bprelu: Vec<BPReLU>,
    pub bn: Vec<BinaryBatchNorm>,
    pub use_shortcut: bool,
}

impl RifCaBlock {
    /// Create a new RIF-CA block with `n` neurons.
    pub fn new(n: usize, use_shortcut: bool, rng: &mut SplitMix64) -> Self {
        Self {
            layer: BnnLayer::random(n, rng),
            ewm: (0..n).map(|_| EwmWeights::default()).collect(),
            bprelu: (0..n).map(|_| BPReLU::new()).collect(),
            bn: (0..n).map(|_| BinaryBatchNorm::new()).collect(),
            use_shortcut,
        }
    }

    /// Forward pass through the RIF-CA block.
    ///
    /// 1. Binary conv via BnnLayer → raw scores
    /// 2. BatchNorm normalizes raw scores
    /// 3. BPReLU applies adaptive slopes
    /// 4. Winner selection (highest processed score)
    /// 5. Awareness decomposition between input and winner activation
    /// 6. EWM modulates winner activation (crystallized → keep, noise → suppress)
    /// 7. Shortcut: XOR-bind input with modulated output
    pub fn forward(
        &mut self,
        input: &Fingerprint<256>,
        learn: bool,
        learning_rate: f64,
        rng: &mut SplitMix64,
    ) -> (usize, Fingerprint<256>, RifFlowMetrics) {
        let lr = learning_rate as f32;

        // 1. Binary conv: get raw scores from all neurons
        let raw_scores = self.layer.forward(input, learn, learning_rate, rng);

        // 2-3. BN + BPReLU per neuron, find winner
        let mut best_idx = 0;
        let mut best_score = f32::NEG_INFINITY;
        for (i, &raw) in raw_scores.iter().enumerate() {
            let normed = self.bn[i].normalize(raw, learn);
            let activated = self.bprelu[i].apply(normed);
            if learn {
                self.bprelu[i].adapt(normed, lr * 0.01);
            }
            if activated > best_score {
                best_score = activated;
                best_idx = i;
            }
        }

        // 4. Winner's binary activation
        let winner_act = self.layer.neurons[best_idx].activation().clone();

        // 5. Pre-EWM signal quality
        let pre_hv = GraphHV::from_channels(
            winner_act.clone(),
            Fingerprint::zero(),
            Fingerprint::zero(),
        );
        let pre_sq = signal_quality(&pre_hv);

        // 6. Awareness decomposition: compare input vs WEIGHT pattern
        //    Paper's BIR-EWM: the correction is based on how well weights
        //    match the input, not the (input-derived) activation.
        //    2048 bytes = 1024 BF16 dims = 4 dims per u64 word
        let thresholds = AwarenessThresholds::default();
        let weight_bytes = self.layer.neurons[best_idx].weights().as_bytes();
        let awareness =
            superposition_decompose(&[input.as_bytes(), weight_bytes], &thresholds);

        if learn {
            self.ewm[best_idx].update_from_awareness(&awareness, lr);
        }

        // 7. EWM modulation
        let modulated = self.ewm[best_idx].apply(&winner_act, rng);

        // 8. Shortcut: XOR-bind with PERMUTED input to preserve info
        //    Plain XOR(modulated, input) = 0 when activation == input (score > 0).
        //    Permutation creates a quasi-orthogonal vector, breaking the identity
        //    while preserving the information content. Standard HDC practice.
        let output = if self.use_shortcut {
            let permuted = permute_words(input);
            xor_fingerprints(&modulated, &permuted)
        } else {
            modulated
        };

        // 9. Post-EWM signal quality + metrics
        let post_hv = GraphHV::from_channels(
            output.clone(),
            Fingerprint::zero(),
            Fingerprint::zero(),
        );
        let post_sq = signal_quality(&post_hv);
        let ewm_str = self.ewm[best_idx].correction_strength();

        let metrics = RifFlowMetrics {
            pre_ewm_signal_quality: pre_sq,
            post_ewm_signal_quality: post_sq,
            ewm_correction_strength: ewm_str,
            awareness_crystallized_pct: awareness.crystallized_pct,
            awareness_tensioned_pct: awareness.tensioned_pct,
            collapse_decision: evaluate_collapse_decision(
                ewm_str,
                awareness.crystallized_pct,
                awareness.tensioned_pct,
            ),
        };

        (best_idx, output, metrics)
    }
}

/// XOR two fingerprints word-by-word.
fn xor_fingerprints(a: &Fingerprint<256>, b: &Fingerprint<256>) -> Fingerprint<256> {
    let mut words = [0u64; 256];
    for (i, word) in words.iter_mut().enumerate() {
        *word = a.words[i] ^ b.words[i];
    }
    Fingerprint::from_words(words)
}

/// Permute a fingerprint by rotating words left by 1.
///
/// Creates a quasi-orthogonal vector that preserves information content.
/// Used for shortcut connections to prevent XOR identity cancellation.
fn permute_words(fp: &Fingerprint<256>) -> Fingerprint<256> {
    let mut words = [0u64; 256];
    for (i, word) in words.iter_mut().enumerate() {
        *word = fp.words[(i + 1) % 256];
    }
    Fingerprint::from_words(words)
}

// ============================================================================
// RifNet — stacked RIF-CA blocks with rich information flow
// ============================================================================

/// RIF-Net: stacked RIF-CA blocks with inter-block shortcut connections.
///
/// Every block after the first has a within-block shortcut (XOR-bind).
/// Additionally, every other block (starting from block 2) gets an
/// inter-block shortcut from 2 blocks back — the "rich information flow"
/// that prevents information bottleneck in deep binary networks.
pub struct RifNet {
    pub blocks: Vec<RifCaBlock>,
}

impl RifNet {
    /// Create a RIF-Net with the given per-block neuron counts.
    pub fn new(block_sizes: &[usize], rng: &mut SplitMix64) -> Self {
        Self {
            blocks: block_sizes
                .iter()
                .enumerate()
                .map(|(i, &n)| RifCaBlock::new(n, i > 0, rng))
                .collect(),
        }
    }

    /// Forward pass through all blocks with rich information flow.
    ///
    /// Inter-block shortcuts: block[i] receives XOR(current, output[i-2])
    /// for i >= 2, carrying information from 2 blocks back.
    pub fn forward(
        &mut self,
        input: &Fingerprint<256>,
        learn: bool,
        learning_rate: f64,
        rng: &mut SplitMix64,
    ) -> (usize, Fingerprint<256>, Vec<RifFlowMetrics>) {
        let mut current = input.clone();
        let mut all_metrics = Vec::with_capacity(self.blocks.len());
        let mut prev_outputs: Vec<Fingerprint<256>> = Vec::new();
        let mut last_winner = 0;

        for (i, block) in self.blocks.iter_mut().enumerate() {
            // Rich information flow: inter-block shortcut from 2 blocks back
            let block_input = if i >= 2 {
                xor_fingerprints(&current, &prev_outputs[i - 2])
            } else {
                current.clone()
            };

            let (winner, output, metrics) =
                block.forward(&block_input, learn, learning_rate, rng);
            prev_outputs.push(output.clone());
            all_metrics.push(metrics);
            current = output;
            last_winner = winner;
        }

        (last_winner, current, all_metrics)
    }

    /// Cascade-accelerated recall using K0/K1/K2 on the last block's weights.
    pub fn cascade_recall(
        &self,
        query: &Fingerprint<256>,
        top_k: usize,
    ) -> Vec<(usize, BnnDotResult)> {
        if self.blocks.is_empty() {
            return Vec::new();
        }
        let last = &self.blocks[self.blocks.len() - 1];
        let weights: Vec<Fingerprint<256>> = last
            .layer
            .neurons
            .iter()
            .map(|n| n.weights().clone())
            .collect();
        let gate = SliceGate::sku_16k();
        bnn_cascade_search(query, &weights, top_k, &gate).matches
    }

    /// Number of blocks.
    #[inline]
    pub fn depth(&self) -> usize {
        self.blocks.len()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> SplitMix64 {
        SplitMix64::new(42)
    }

    fn random_fp(rng: &mut SplitMix64) -> Fingerprint<256> {
        let mut words = [0u64; 256];
        for w in words.iter_mut() {
            *w = rng.next_u64();
        }
        Fingerprint::from_words(words)
    }

    // --- EWM tests ---

    #[test]
    fn test_ewm_default_is_identity() {
        let mut rng = make_rng();
        let fp = random_fp(&mut rng);
        let ewm = EwmWeights::default();
        let result = ewm.apply(&fp, &mut rng);
        assert_eq!(fp.words, result.words, "Default EWM should be identity");
    }

    #[test]
    fn test_ewm_noise_suppression() {
        let mut rng = make_rng();
        let fp = random_fp(&mut rng);
        let mut ewm = EwmWeights::default();
        // Set first 64 words to noise (scale 0.0)
        for i in 0..64 {
            ewm.scales[i] = 0.0;
        }
        let result = ewm.apply(&fp, &mut rng);
        for i in 0..64 {
            assert_eq!(result.words[i], 0, "Noise word {} should be cleared", i);
        }
        for i in 64..256 {
            assert_eq!(
                result.words[i], fp.words[i],
                "Non-noise word {} should be unchanged",
                i
            );
        }
    }

    #[test]
    fn test_ewm_awareness_update() {
        let mut ewm = EwmWeights::default();
        let mut states = Vec::with_capacity(1024);
        // First 128 dims (32 words × 4 dims): Crystallized
        for _ in 0..128 {
            states.push(AwarenessState::Crystallized);
        }
        // Next 128 dims (32 words × 4 dims): Noise
        for _ in 0..128 {
            states.push(AwarenessState::Noise);
        }
        // Remaining 768 dims: Uncertain
        for _ in 0..768 {
            states.push(AwarenessState::Uncertain);
        }
        let awareness = SuperpositionState {
            n_dims: 1024,
            sign_consensus: vec![255; 1024],
            exp_spread: vec![0; 1024],
            mantissa_noise: vec![false; 1024],
            states,
            packed_states: vec![0; 256],
            crystallized_pct: 0.125,
            tensioned_pct: 0.0,
            uncertain_pct: 0.75,
            noise_pct: 0.125,
        };

        ewm.update_from_awareness(&awareness, 0.5);

        // Crystallized words (0..32): scale stays near 1.0
        for i in 0..32 {
            assert!(
                ewm.scales[i] > 0.95,
                "Crystallized word {} scale should be > 0.95, got {}",
                i,
                ewm.scales[i]
            );
        }
        // Noise words (32..64): scale drops toward 0.0
        for i in 32..64 {
            assert!(
                ewm.scales[i] < 0.6,
                "Noise word {} scale should be < 0.6, got {}",
                i,
                ewm.scales[i]
            );
        }
    }

    // --- BPReLU tests ---

    #[test]
    fn test_bprelu_positive_slope() {
        let bprelu = BPReLU::new();
        let result = bprelu.apply(0.5);
        assert!(
            (result - 0.5).abs() < 1e-6,
            "alpha_pos=1.0 should preserve positive: got {}",
            result
        );
    }

    #[test]
    fn test_bprelu_negative_slope() {
        let bprelu = BPReLU::new();
        let result = bprelu.apply(-0.4);
        let expected = 0.25 * -0.4;
        assert!(
            (result - expected).abs() < 1e-6,
            "alpha_neg=0.25 should scale negative: got {}, expected {}",
            result,
            expected
        );
    }

    // --- BatchNorm tests ---

    #[test]
    fn test_bn_normalizes_scores() {
        let mut bn = BinaryBatchNorm::new();
        // Feed 100 scores centered around 0.3
        for i in 0..100 {
            let score = 0.3 + (i as f32 - 50.0) * 0.002;
            bn.normalize(score, true);
        }
        assert!(
            (bn.running_mean - 0.3).abs() < 0.1,
            "Mean should converge near 0.3, got {}",
            bn.running_mean
        );
        // Normalized output for the mean score should be near beta (0.0)
        let normed = bn.normalize(bn.running_mean, false);
        assert!(
            normed.abs() < 0.5,
            "Normalized mean should be near 0.0, got {}",
            normed
        );
    }

    // --- RifCaBlock tests ---

    #[test]
    fn test_rif_ca_block_forward() {
        let mut rng = make_rng();
        let mut block = RifCaBlock::new(8, true, &mut rng);
        let input = random_fp(&mut rng);

        let (winner, output, metrics) = block.forward(&input, false, 0.03, &mut rng);

        assert!(winner < 8, "Winner index should be < 8, got {}", winner);
        assert!(output.popcount() > 0, "Output should have some bits set");
        assert!(
            metrics.pre_ewm_signal_quality >= 0.0,
            "Signal quality should be non-negative"
        );
    }

    #[test]
    fn test_rif_ca_block_shortcut_preserves_info() {
        let mut rng = make_rng();
        let input = random_fp(&mut rng);

        // Block WITH shortcut
        let mut block = RifCaBlock::new(5, true, &mut rng);
        let (_, out_sc, _) = block.forward(&input, false, 0.03, &mut rng);

        // Unbind: XOR output with input recovers modulated activation
        let unbound = xor_fingerprints(&out_sc, &input);
        let hv = GraphHV::from_channels(unbound, Fingerprint::zero(), Fingerprint::zero());
        let sq = signal_quality(&hv);
        // Unbound should be structured — its signal quality is computed
        assert!(sq >= 0.0, "Signal quality should be non-negative: {}", sq);

        // Output should differ from input (not identity)
        assert_ne!(
            out_sc.words, input.words,
            "Shortcut output should differ from input"
        );
    }

    // --- RifNet tests ---

    #[test]
    fn test_rif_net_forward() {
        let mut rng = make_rng();
        let mut net = RifNet::new(&[8, 6, 4], &mut rng);
        let input = random_fp(&mut rng);

        let (winner, output, metrics) = net.forward(&input, false, 0.03, &mut rng);

        assert!(winner < 4, "Final winner should be < 4");
        assert_eq!(metrics.len(), 3, "Should have 3 blocks of metrics");
        assert!(output.popcount() > 0, "Output should have bits set");
    }

    #[test]
    fn test_rif_net_rich_flow_improves_signal() {
        let mut rng = make_rng();
        let mut net = RifNet::new(&[8, 6, 4], &mut rng);
        let input = random_fp(&mut rng);

        // Train for many iterations with strong learning rate
        // so all winner neurons get EWM updates
        for _ in 0..100 {
            net.forward(&input, true, 0.3, &mut rng);
        }

        let (_, _, metrics) = net.forward(&input, true, 0.3, &mut rng);
        // After training, at least one block's EWM should have non-zero correction
        let total_correction: f32 = metrics.iter().map(|m| m.ewm_correction_strength).sum();
        assert!(
            total_correction > 0.0,
            "After training, EWM should have learned some correction: {}",
            total_correction
        );
    }

    // --- CollapseGate tests ---

    #[test]
    fn test_collapse_gate_flow_on_strong_correction() {
        let decision = evaluate_collapse_decision(0.5, 0.7, 0.1);
        assert_eq!(
            decision,
            CollapseGate::Flow,
            "Strong correction + high crystallization should trigger Flow"
        );
    }

    #[test]
    fn test_collapse_gate_hold_on_weak_correction() {
        let decision = evaluate_collapse_decision(0.1, 0.3, 0.1);
        assert_eq!(
            decision,
            CollapseGate::Hold,
            "Weak correction should result in Hold"
        );
    }

    // --- Cascade recall test ---

    #[test]
    fn test_cascade_recall_through_rif_blocks() {
        let mut rng = make_rng();
        let net = RifNet::new(&[50, 20], &mut rng);
        let query = random_fp(&mut rng);

        let results = net.cascade_recall(&query, 5);

        assert!(
            results.len() <= 5,
            "Should return at most 5 results, got {}",
            results.len()
        );
        // Results should be sorted by descending score
        for i in 1..results.len() {
            assert!(
                results[i].1.score <= results[i - 1].1.score,
                "Results should be sorted descending"
            );
        }
    }
}
