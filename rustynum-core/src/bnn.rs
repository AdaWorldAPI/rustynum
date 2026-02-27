//! Binary Neural Network (BNN) inference primitives for HDC graph memory.
//!
//! Implements BNN convolution and inference using XNOR + popcount operations,
//! directly leveraging the existing 16,384-bit `Fingerprint<256>` containers
//! and `GraphHV` 3-channel hypervectors.
//!
//! ## RIF-Net Inspired Architecture
//!
//! Based on "Accurate binary neural network based on rich information flow"
//! (Zhang et al., Neurocomputing 2025):
//!
//! - Binary weights and activations (1-bit): stored as `Fingerprint<256>`
//! - XNOR + popcount = binary dot product (the Hamming-distance dual)
//! - Rich information flow via amplitude correction (Element-Wise Multiply)
//! - Shortcut connections + BN layers for gradient preservation
//!
//! ## Integration with GraphHV
//!
//! The 3-channel `GraphHV` serves as the binary "neuron state":
//! - Channel 0 (Node): binary activations from the forward pass
//! - Channel 1 (Edge): binary weight connections (synaptic pattern)
//! - Channel 2 (Plastic): running average via plastic bundling (learned state)
//!
//! This creates a graph-structured BNN where:
//! - Nodes are binary activation vectors (16,384 bits)
//! - Edges are binary weight vectors (XNOR similarity = synaptic strength)
//! - Plastic state enables continual one-shot learning without backprop
//!
//! ## Key Insight: Grey/White Matter Analogy
//!
//! | Brain Region | BNN Role | HDC Primitive |
//! |-------------|----------|---------------|
//! | Grey matter | Inference (XNOR+pop) | `bnn_dot()` on channel 0 |
//! | White matter | Connections (weights) | XOR-bind on channel 1 |
//! | Plasticity | Learning (bundling) | `bundle_into()` on channel 2 |

use crate::fingerprint::Fingerprint;
use crate::graph_hv::{bundle_into, GraphHV};
use crate::rng::SplitMix64;

/// Result of a binary convolution (XNOR + popcount).
#[derive(Clone, Copy, Debug)]
pub struct BnnDotResult {
    /// Raw popcount of XNOR(activation, weight) — number of matching bits.
    pub match_count: u32,
    /// Total bits compared.
    pub total_bits: u32,
    /// Normalized score in [-1.0, 1.0] (bipolar interpretation).
    /// +1.0 = perfect match, 0.0 = orthogonal, -1.0 = anti-match.
    pub score: f32,
}

/// Binary dot product via XNOR + popcount on a single channel.
///
/// This is the fundamental BNN inference primitive:
/// `dot(a, w) = 2 * popcount(XNOR(a, w)) - total_bits`
///
/// Equivalent to the Hamming-similarity dual: identical bits contribute +1,
/// differing bits contribute -1.
#[inline]
pub fn bnn_dot(activation: &Fingerprint<256>, weight: &Fingerprint<256>) -> BnnDotResult {
    let total_bits = Fingerprint::<256>::BITS as u32;
    // XNOR = NOT(XOR): count matching bits
    let mut match_count = 0u32;
    for i in 0..256 {
        // XNOR: bits that are the same in both vectors
        match_count += (!(activation.words[i] ^ weight.words[i])).count_ones();
    }
    let score = (2.0 * match_count as f32 / total_bits as f32) - 1.0;
    BnnDotResult {
        match_count,
        total_bits,
        score,
    }
}

/// Binary dot product across all 3 channels of a `GraphHV`.
///
/// Returns the sum of per-channel XNOR+popcount scores.
/// This computes the full 49,152-bit binary correlation.
pub fn bnn_dot_3ch(activation: &GraphHV, weight: &GraphHV) -> BnnDotResult {
    let total_bits = (Fingerprint::<256>::BITS * 3) as u32;
    let mut match_count = 0u32;
    for ch in 0..3 {
        for i in 0..256 {
            match_count +=
                (!(activation.channels[ch].words[i] ^ weight.channels[ch].words[i])).count_ones();
        }
    }
    let score = (2.0 * match_count as f32 / total_bits as f32) - 1.0;
    BnnDotResult {
        match_count,
        total_bits,
        score,
    }
}

/// A binary neuron in the graph BNN.
///
/// Combines inference (XNOR+popcount), connection topology (edge channel),
/// and plasticity (running average via bundle).
pub struct BnnNeuron {
    /// The neuron's state: 3-channel hypervector.
    /// - channels[0]: current binary activation
    /// - channels[1]: weight pattern (synaptic connections)
    /// - channels[2]: plastic running average (learned prototype)
    pub state: GraphHV,
    /// Bias (amplitude correction from RIF-Net BIR-EWM).
    pub bias: f32,
    /// Activation threshold for binary output.
    pub threshold: f32,
}

impl BnnNeuron {
    /// Create a new neuron with random weights.
    pub fn random(rng: &mut SplitMix64) -> Self {
        Self {
            state: GraphHV::random(rng),
            bias: 0.0,
            threshold: 0.0,
        }
    }

    /// Create from a pre-existing weight pattern.
    pub fn from_weights(weights: Fingerprint<256>, rng: &mut SplitMix64) -> Self {
        let mut state = GraphHV::zero();
        state.channels[1] = weights; // edge channel = weights
        // Initialize plastic channel with the weight pattern
        state.channels[2] = state.channels[1].clone();
        // Random initial activation
        let mut words = [0u64; 256];
        for w in words.iter_mut() {
            *w = rng.next_u64();
        }
        state.channels[0] = Fingerprint::from_words(words);
        Self {
            state,
            bias: 0.0,
            threshold: 0.0,
        }
    }

    /// Forward pass: compute binary activation from input.
    ///
    /// 1. XNOR+popcount between input and weight channel → raw score
    /// 2. Add bias (RIF-Net amplitude correction)
    /// 3. Binary threshold → update activation channel
    /// 4. Optionally update plastic channel via bundling
    ///
    /// Returns the pre-threshold score for downstream use.
    pub fn forward(
        &mut self,
        input: &Fingerprint<256>,
        learn: bool,
        learning_rate: f64,
        rng: &mut SplitMix64,
    ) -> f32 {
        // Grey matter: XNOR + popcount inference
        let dot = bnn_dot(input, &self.state.channels[1]);
        let pre_activation = dot.score + self.bias;

        // Update activation channel based on threshold
        if pre_activation > self.threshold {
            self.state.channels[0] = input.clone();
        } else {
            self.state.channels[0] = !input;
        }

        // Plasticity: bundle input into plastic channel (white matter learning)
        if learn {
            let input_hv = GraphHV::from_channels(
                input.clone(),
                Fingerprint::zero(),
                Fingerprint::zero(),
            );
            let new_plastic = bundle_into(&self.state, &input_hv, learning_rate, 1.0, rng);
            self.state.channels[2] = new_plastic.channels[2].clone();
        }

        pre_activation
    }

    /// Get the current binary activation.
    #[inline]
    pub fn activation(&self) -> &Fingerprint<256> {
        &self.state.channels[0]
    }

    /// Get the weight pattern.
    #[inline]
    pub fn weights(&self) -> &Fingerprint<256> {
        &self.state.channels[1]
    }

    /// Get the plastic (learned) prototype.
    #[inline]
    pub fn plastic(&self) -> &Fingerprint<256> {
        &self.state.channels[2]
    }
}

/// A binary layer: multiple neurons processing the same input in parallel.
///
/// Implements the BNN dense layer as a bank of XNOR+popcount operations.
/// Each neuron has independent weights, shared input.
pub struct BnnLayer {
    pub neurons: Vec<BnnNeuron>,
}

impl BnnLayer {
    /// Create a layer with `n` neurons, randomly initialized.
    pub fn random(n: usize, rng: &mut SplitMix64) -> Self {
        Self {
            neurons: (0..n).map(|_| BnnNeuron::random(rng)).collect(),
        }
    }

    /// Forward pass: compute all neurons' activations from input.
    ///
    /// Returns the pre-threshold scores for all neurons.
    pub fn forward(
        &mut self,
        input: &Fingerprint<256>,
        learn: bool,
        learning_rate: f64,
        rng: &mut SplitMix64,
    ) -> Vec<f32> {
        self.neurons
            .iter_mut()
            .map(|n| n.forward(input, learn, learning_rate, rng))
            .collect()
    }

    /// Find the neuron with the highest activation score (winner-take-all).
    pub fn winner(&self, input: &Fingerprint<256>) -> (usize, BnnDotResult) {
        let mut best_idx = 0;
        let mut best_result = bnn_dot(input, &self.neurons[0].state.channels[1]);
        for (i, neuron) in self.neurons.iter().enumerate().skip(1) {
            let result = bnn_dot(input, &neuron.state.channels[1]);
            if result.score > best_result.score {
                best_idx = i;
                best_result = result;
            }
        }
        (best_idx, best_result)
    }

    /// Number of neurons in this layer.
    #[inline]
    pub fn len(&self) -> usize {
        self.neurons.len()
    }

    /// Returns true if the layer has no neurons.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.neurons.is_empty()
    }
}

/// Batch XNOR+popcount: compute binary dot products between a query and
/// multiple weight vectors. Returns scores sorted by descending match.
///
/// This is the "grey matter" bulk inference: given one input activation,
/// find the top-K matching weight patterns from a bank of prototypes.
pub fn bnn_batch_dot(
    query: &Fingerprint<256>,
    weights: &[Fingerprint<256>],
    top_k: usize,
) -> Vec<(usize, BnnDotResult)> {
    let mut results: Vec<(usize, BnnDotResult)> = weights
        .iter()
        .enumerate()
        .map(|(i, w)| (i, bnn_dot(query, w)))
        .collect();

    results.sort_by(|a, b| {
        b.1.score
            .partial_cmp(&a.1.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(top_k);
    results
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
    fn test_bnn_dot_identical() {
        let mut rng = make_rng();
        let mut words = [0u64; 256];
        for w in words.iter_mut() {
            *w = rng.next_u64();
        }
        let fp = Fingerprint::from_words(words);
        let result = bnn_dot(&fp, &fp);
        assert_eq!(result.match_count, 16_384);
        assert!((result.score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bnn_dot_opposite() {
        let mut words = [0u64; 256];
        for w in words.iter_mut() {
            *w = 0xFFFF_FFFF_FFFF_FFFF;
        }
        let a = Fingerprint::from_words(words);
        let b = Fingerprint::<256>::zero();
        let result = bnn_dot(&a, &b);
        assert_eq!(result.match_count, 0);
        assert!((result.score - (-1.0)).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bnn_dot_random_near_zero() {
        let mut rng = make_rng();
        let mut words_a = [0u64; 256];
        let mut words_b = [0u64; 256];
        for i in 0..256 {
            words_a[i] = rng.next_u64();
            words_b[i] = rng.next_u64();
        }
        let a = Fingerprint::from_words(words_a);
        let b = Fingerprint::from_words(words_b);
        let result = bnn_dot(&a, &b);
        // Random vectors: ~50% match → score ≈ 0.0
        assert!(
            result.score.abs() < 0.05,
            "Expected ~0.0 score for random, got {:.4}",
            result.score
        );
    }

    #[test]
    fn test_bnn_dot_3ch() {
        let mut rng = make_rng();
        let a = GraphHV::random(&mut rng);
        let result = bnn_dot_3ch(&a, &a);
        assert_eq!(result.match_count, 49_152);
        assert!((result.score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bnn_neuron_forward() {
        let mut rng = make_rng();
        let mut neuron = BnnNeuron::random(&mut rng);

        let mut input_words = [0u64; 256];
        for w in input_words.iter_mut() {
            *w = rng.next_u64();
        }
        let input = Fingerprint::from_words(input_words);

        let score = neuron.forward(&input, false, 0.03, &mut rng);
        // Score should be in [-1, 1] + bias range
        assert!(score.abs() < 2.0, "Score out of range: {}", score);
    }

    #[test]
    fn test_bnn_neuron_plasticity() {
        let mut rng = make_rng();
        let mut neuron = BnnNeuron::random(&mut rng);

        let mut input_words = [0u64; 256];
        for w in input_words.iter_mut() {
            *w = rng.next_u64();
        }
        let input = Fingerprint::from_words(input_words);

        let initial_plastic = neuron.plastic().clone();

        // Multiple learning updates
        for _ in 0..50 {
            neuron.forward(&input, true, 0.1, &mut rng);
        }

        // Plastic channel should have changed
        assert_ne!(
            *neuron.plastic(),
            initial_plastic,
            "Plastic channel should change after learning"
        );
    }

    #[test]
    fn test_bnn_layer_winner() {
        let mut rng = make_rng();
        let layer = BnnLayer::random(10, &mut rng);

        let mut input_words = [0u64; 256];
        for w in input_words.iter_mut() {
            *w = rng.next_u64();
        }
        let input = Fingerprint::from_words(input_words);

        let (winner_idx, _) = layer.winner(&input);
        assert!(winner_idx < 10);
    }

    #[test]
    fn test_bnn_batch_dot_ordering() {
        let mut rng = make_rng();
        let mut query_words = [0u64; 256];
        for w in query_words.iter_mut() {
            *w = rng.next_u64();
        }
        let query = Fingerprint::from_words(query_words);

        let weights: Vec<_> = (0..20)
            .map(|_| {
                let mut w = [0u64; 256];
                for ww in w.iter_mut() {
                    *ww = rng.next_u64();
                }
                Fingerprint::from_words(w)
            })
            .collect();

        let results = bnn_batch_dot(&query, &weights, 5);
        assert_eq!(results.len(), 5);

        // Verify descending order
        for i in 1..results.len() {
            assert!(
                results[i].1.score <= results[i - 1].1.score,
                "Results not sorted: {} > {}",
                results[i].1.score,
                results[i - 1].1.score,
            );
        }
    }

    #[test]
    fn test_bnn_layer_forward() {
        let mut rng = make_rng();
        let mut layer = BnnLayer::random(5, &mut rng);
        assert_eq!(layer.len(), 5);

        let mut input_words = [0u64; 256];
        for w in input_words.iter_mut() {
            *w = rng.next_u64();
        }
        let input = Fingerprint::from_words(input_words);

        let scores = layer.forward(&input, false, 0.03, &mut rng);
        assert_eq!(scores.len(), 5);
        for &s in &scores {
            assert!(s.abs() < 2.0, "Score out of range: {}", s);
        }
    }

    #[test]
    fn test_batch_dot_finds_self() {
        let mut rng = make_rng();
        let mut target_words = [0u64; 256];
        for w in target_words.iter_mut() {
            *w = rng.next_u64();
        }
        let target = Fingerprint::from_words(target_words);

        let mut weights: Vec<_> = (0..50)
            .map(|_| {
                let mut w = [0u64; 256];
                for ww in w.iter_mut() {
                    *ww = rng.next_u64();
                }
                Fingerprint::from_words(w)
            })
            .collect();

        // Plant the exact match at index 25
        weights[25] = target.clone();

        let results = bnn_batch_dot(&target, &weights, 1);
        assert_eq!(results[0].0, 25);
        assert!((results[0].1.score - 1.0).abs() < f32::EPSILON);
    }
}
