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

use crate::cam_index::CamIndex;
use crate::fingerprint::Fingerprint;
use crate::graph_hv::{bundle_into, GraphHV};
use crate::kernels::{
    kernel_pipeline, EnergyConflict, KernelStage, PipelineStats, SKU_16K_WORDS, SliceGate,
};
use crate::rng::SplitMix64;

#[cfg(any(feature = "avx512", feature = "avx2"))]
use std::sync::OnceLock;

/// SIMD Hamming function pointer type.
#[cfg(any(feature = "avx512", feature = "avx2"))]
type HammingFn = fn(&[u8], &[u8]) -> u64;

/// Cached SIMD Hamming dispatch — resolved once at first call.
/// Falls back to scalar when SIMD features are not compiled in.
#[cfg(any(feature = "avx512", feature = "avx2"))]
fn hamming_simd() -> HammingFn {
    static FN: OnceLock<HammingFn> = OnceLock::new();
    *FN.get_or_init(crate::simd::select_hamming_fn)
}

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
///
/// Uses the SIMD Hamming kernel (AVX-512 VPOPCNTDQ / AVX2 Harley-Seal / scalar)
/// via `select_hamming_fn()` dispatch. The mathematical identity is:
/// `XNOR_popcount(a, b) = TOTAL_BITS - XOR_popcount(a, b)`
#[inline]
pub fn bnn_dot(activation: &Fingerprint<256>, weight: &Fingerprint<256>) -> BnnDotResult {
    let total_bits = Fingerprint::<256>::BITS as u32; // 16,384
    let xor_popcount = bnn_hamming_u32(activation.as_bytes(), weight.as_bytes());
    let match_count = total_bits - xor_popcount;
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
    let total_bits = (Fingerprint::<256>::BITS * 3) as u32; // 49,152
    let mut xor_total = 0u32;
    for ch in 0..3 {
        xor_total += bnn_hamming_u32(
            activation.channels[ch].as_bytes(),
            weight.channels[ch].as_bytes(),
        );
    }
    let match_count = total_bits - xor_total;
    let score = (2.0 * match_count as f32 / total_bits as f32) - 1.0;
    BnnDotResult {
        match_count,
        total_bits,
        score,
    }
}

/// Internal: XOR + popcount as u32 — dispatches to SIMD when available.
#[inline]
fn bnn_hamming_u32(a: &[u8], b: &[u8]) -> u32 {
    #[cfg(any(feature = "avx512", feature = "avx2"))]
    {
        hamming_simd()(a, b) as u32
    }
    #[cfg(not(any(feature = "avx512", feature = "avx2")))]
    {
        // Scalar fallback: XOR + popcount per byte pair
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x ^ y).count_ones())
            .sum()
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
            let input_hv =
                GraphHV::from_channels(input.clone(), Fingerprint::zero(), Fingerprint::zero());
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
// Phase 1: Wire BNN to K0/K1/K2 cascade (Task 1.1)
// ===========================================================================

/// Result of a cascade-accelerated BNN search.
#[derive(Clone, Debug)]
pub struct BnnCascadeResult {
    /// Matches found by the cascade pipeline.
    pub matches: Vec<(usize, BnnDotResult)>,
    /// Pipeline statistics (K0/K1/K2 rejection rates).
    pub stats: PipelineStats,
}

/// Cascade-accelerated BNN batch search using K0/K1/K2 pipeline.
///
/// Instead of computing full 16,384-bit XNOR+popcount for every weight vector,
/// uses the 3-kernel cascade to reject ~95% of candidates after touching
/// only 64-512 bits. For 10,000 weights, this means ~500 full computations
/// instead of 10,000.
///
/// The `gate` controls rejection aggressiveness:
/// - `SliceGate::sku_16k()` — default, very conservative (zero false negatives)
/// - `SliceGate::new(16384, 0.05, 0.15, 0.30, 0.90, 1.5)` — tighter, rejects more
///
/// Returns matches sorted by descending BNN score (most similar first),
/// truncated to `top_k`.
pub fn bnn_cascade_search(
    query: &Fingerprint<256>,
    weights: &[Fingerprint<256>],
    top_k: usize,
    gate: &SliceGate,
) -> BnnCascadeResult {
    if weights.is_empty() {
        return BnnCascadeResult {
            matches: Vec::new(),
            stats: PipelineStats::default(),
        };
    }

    // Pack weights into flat u64 database (zero-copy view of .words)
    let n_candidates = weights.len();
    let mut db_words = Vec::with_capacity(n_candidates * SKU_16K_WORDS);
    for w in weights {
        db_words.extend_from_slice(&w.words);
    }

    // Run cascade pipeline
    let (kernel_matches, stats) =
        kernel_pipeline(&query.words, &db_words, n_candidates, SKU_16K_WORDS, gate);

    // Convert KernelResult → BnnDotResult
    let total_bits = Fingerprint::<256>::BITS as u32;
    let mut results: Vec<(usize, BnnDotResult)> = kernel_matches
        .iter()
        .map(|kr| {
            let match_count = total_bits - kr.distance;
            let score = (2.0 * match_count as f32 / total_bits as f32) - 1.0;
            (
                kr.index,
                BnnDotResult {
                    match_count,
                    total_bits,
                    score,
                },
            )
        })
        .collect();

    // Sort by score descending
    results.sort_by(|a, b| {
        b.1.score
            .partial_cmp(&a.1.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(top_k);

    BnnCascadeResult {
        matches: results,
        stats,
    }
}

/// Extended result including energy/conflict decomposition from K2.
#[derive(Clone, Debug)]
pub struct BnnEnergyResult {
    /// Weight index.
    pub index: usize,
    /// Standard BNN dot result.
    pub dot: BnnDotResult,
    /// Energy/conflict decomposition from K2 exact.
    pub energy: EnergyConflict,
    /// Kernel stage that determined the outcome.
    pub stage: KernelStage,
}

/// Like `bnn_cascade_search` but also returns EnergyConflict decomposition.
///
/// Useful for awareness-guided learning: the energy/conflict split tells you
/// not just "how different" but "in what way" — shared information vs active
/// disagreement vs sparsity.
pub fn bnn_cascade_search_with_energy(
    query: &Fingerprint<256>,
    weights: &[Fingerprint<256>],
    top_k: usize,
    gate: &SliceGate,
) -> (Vec<BnnEnergyResult>, PipelineStats) {
    if weights.is_empty() {
        return (Vec::new(), PipelineStats::default());
    }

    let n_candidates = weights.len();
    let mut db_words = Vec::with_capacity(n_candidates * SKU_16K_WORDS);
    for w in weights {
        db_words.extend_from_slice(&w.words);
    }

    let (kernel_matches, stats) =
        kernel_pipeline(&query.words, &db_words, n_candidates, SKU_16K_WORDS, gate);

    let total_bits = Fingerprint::<256>::BITS as u32;
    let mut results: Vec<BnnEnergyResult> = kernel_matches
        .iter()
        .map(|kr| {
            let match_count = total_bits - kr.distance;
            let score = (2.0 * match_count as f32 / total_bits as f32) - 1.0;
            BnnEnergyResult {
                index: kr.index,
                dot: BnnDotResult {
                    match_count,
                    total_bits,
                    score,
                },
                energy: kr.energy,
                stage: kr.stage,
            }
        })
        .collect();

    results.sort_by(|a, b| {
        b.dot
            .score
            .partial_cmp(&a.dot.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    results.truncate(top_k);

    (results, stats)
}

// ===========================================================================
// Phase 1: Multi-layer BNN network (Task 1.2)
// ===========================================================================

/// A multi-layer binary neural network.
///
/// Stacks `BnnLayer`s for depth. Each layer's winner activation
/// (`Fingerprint<256>`) feeds into the next layer as input.
///
/// ## Architecture
///
/// ```text
/// Input (Fingerprint<256>)
///   → Layer 0 (N₀ neurons) → winner activation
///     → Layer 1 (N₁ neurons) → winner activation
///       → Layer 2 (N₂ neurons) → winner activation = output
/// ```
///
/// All layers operate on the same 16,384-bit binary vectors.
/// No dimension reduction — the sparsity is in the number of neurons per layer,
/// not the representation width.
pub struct BnnNetwork {
    pub layers: Vec<BnnLayer>,
}

impl BnnNetwork {
    /// Create a network with the given layer sizes.
    ///
    /// `layer_sizes[0]` = number of neurons in first layer, etc.
    pub fn new(layer_sizes: &[usize], rng: &mut SplitMix64) -> Self {
        Self {
            layers: layer_sizes
                .iter()
                .map(|&n| BnnLayer::random(n, rng))
                .collect(),
        }
    }

    /// Forward pass through all layers.
    ///
    /// Returns `(winner_index, BnnDotResult)` for the final layer.
    /// Each intermediate layer's winner activation becomes the next layer's input.
    ///
    /// If `learn` is true, plasticity updates occur at each layer.
    pub fn forward(
        &mut self,
        input: &Fingerprint<256>,
        learn: bool,
        learning_rate: f64,
        rng: &mut SplitMix64,
    ) -> (usize, BnnDotResult) {
        let mut current_input = input.clone();

        for layer in self.layers.iter_mut() {
            layer.forward(&current_input, learn, learning_rate, rng);
            let (winner_idx, _) = layer.winner(&current_input);
            // Winner's activation becomes next layer's input
            current_input = layer.neurons[winner_idx].activation().clone();
        }

        // Final layer's winner
        self.layers
            .last()
            .unwrap()
            .winner(&current_input)
    }

    /// Predict without learning. Returns winner info for each layer.
    pub fn predict(&self, input: &Fingerprint<256>) -> Vec<(usize, BnnDotResult)> {
        let mut current_input = input.clone();
        let mut layer_results = Vec::with_capacity(self.layers.len());

        for layer in &self.layers {
            let (winner_idx, result) = layer.winner(&current_input);
            layer_results.push((winner_idx, result));
            current_input = layer.neurons[winner_idx].activation().clone();
        }

        layer_results
    }

    /// Number of layers.
    #[inline]
    pub fn depth(&self) -> usize {
        self.layers.len()
    }
}

// ===========================================================================
// Phase 2: Wire BNN to HDR cascade (Task 2.1)
// ===========================================================================

/// HDR-cascade-accelerated BNN search using 3-stroke Belichtungsmesser.
///
/// For very large weight banks (100K+ vectors), the HDR cascade's statistical
/// warmup (Stroke 1) kills ~98% of candidates after sampling only 1/16th
/// of each vector. Stroke 2 does full Hamming on survivors. Stroke 3 is
/// skipped (PreciseMode::Off) since we already have exact Hamming from Stroke 2.
///
/// Returns matches sorted by ascending Hamming distance (closest first),
/// converted to BnnDotResult.
#[cfg(any(feature = "avx512", feature = "avx2"))]
pub fn bnn_hdr_search(
    query: &Fingerprint<256>,
    weights: &[Fingerprint<256>],
    threshold: u64,
    top_k: usize,
) -> Vec<(usize, BnnDotResult)> {
    use crate::simd::{hdr_cascade_search, PreciseMode};

    if weights.is_empty() {
        return Vec::new();
    }

    let vec_bytes = Fingerprint::<256>::BITS / 8; // 2048
    let query_bytes = query.as_bytes();

    // Pack weights into flat byte database
    let mut db_bytes = Vec::with_capacity(weights.len() * vec_bytes);
    for w in weights {
        db_bytes.extend_from_slice(w.as_bytes());
    }

    let hdr_results =
        hdr_cascade_search(query_bytes, &db_bytes, vec_bytes, weights.len(), threshold, PreciseMode::Off);

    let total_bits = Fingerprint::<256>::BITS as u32;
    let mut results: Vec<(usize, BnnDotResult)> = hdr_results
        .iter()
        .map(|hr| {
            let match_count = total_bits - hr.hamming as u32;
            let score = (2.0 * match_count as f32 / total_bits as f32) - 1.0;
            (
                hr.index,
                BnnDotResult {
                    match_count,
                    total_bits,
                    score,
                },
            )
        })
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
// Phase 2: Wire BNN to CamIndex (Task 2.2)
// ===========================================================================

impl BnnLayer {
    /// Build a CAM index over all neuron weight patterns.
    ///
    /// Indexes only the weight channel (channel 1) replicated across all 3
    /// GraphHV channels so that LSH projections produce consistent hashes
    /// when queried with `winner_cam()`.
    pub fn build_cam_index(&self, seed: u64) -> CamIndex {
        let mut cam = CamIndex::with_defaults(seed);
        for neuron in &self.neurons {
            let w = &neuron.state.channels[1]; // weight channel
            let hv = GraphHV::from_channels(w.clone(), w.clone(), w.clone());
            cam.insert(hv);
        }
        cam
    }

    /// Winner-take-all using CAM index: O(log N) instead of O(N).
    ///
    /// 1. Multi-probe LSH retrieves candidate indices (~O(log N))
    /// 2. Exact Hamming verification on shortlist
    /// 3. Returns best match from shortlist
    ///
    /// `cam` must have been built from this layer via `build_cam_index()`.
    pub fn winner_cam(
        &self,
        input: &Fingerprint<256>,
        cam: &CamIndex,
        shortlist_size: usize,
    ) -> Option<(usize, BnnDotResult)> {
        // Wrap input as GraphHV for CAM query.
        // Replicate input across all 3 channels so the LSH projection
        // (which samples bits from all channels) produces meaningful hashes.
        let query_hv = GraphHV::from_channels(
            input.clone(),
            input.clone(),
            input.clone(),
        );

        let hits = cam.query(&query_hv, shortlist_size);
        if hits.is_empty() {
            return None;
        }

        // Exact BNN dot on shortlist to find true winner
        let mut best_idx = hits[0].index;
        let mut best_result = bnn_dot(input, &self.neurons[hits[0].index].state.channels[1]);

        for hit in hits.iter().skip(1) {
            if hit.index < self.neurons.len() {
                let result = bnn_dot(input, &self.neurons[hit.index].state.channels[1]);
                if result.score > best_result.score {
                    best_idx = hit.index;
                    best_result = result;
                }
            }
        }

        Some((best_idx, best_result))
    }
}

// ===========================================================================
// Phase 3: Binary convolution (Task 3.2)
// ===========================================================================

/// 1D binary convolution: slide a kernel over a sequence of Fingerprints.
///
/// Each output position is `bnn_dot(input[i * stride], kernel)`.
/// Uses SIMD-dispatched Hamming via `select_hamming_fn()`.
///
/// Returns one `BnnDotResult` per output position.
pub fn bnn_conv1d(
    input: &[Fingerprint<256>],
    kernel: &Fingerprint<256>,
    stride: usize,
) -> Vec<BnnDotResult> {
    let stride = stride.max(1);
    (0..input.len())
        .step_by(stride)
        .map(|i| bnn_dot(&input[i], kernel))
        .collect()
}

/// 1D binary convolution over 3-channel GraphHV sequences.
///
/// Full 49,152-bit correlation per output position.
pub fn bnn_conv1d_3ch(
    input: &[GraphHV],
    kernel: &GraphHV,
    stride: usize,
) -> Vec<BnnDotResult> {
    let stride = stride.max(1);
    (0..input.len())
        .step_by(stride)
        .map(|i| bnn_dot_3ch(&input[i], kernel))
        .collect()
}

/// Cascade-accelerated 1D convolution: batch K0/K1/K2 over sequence positions.
///
/// For long sequences (1000+ positions), uses the kernel cascade to prune
/// positions that clearly don't match the kernel, avoiding full 16,384-bit
/// computation at uninteresting positions.
pub fn bnn_conv1d_cascade(
    input: &[Fingerprint<256>],
    kernel: &Fingerprint<256>,
    stride: usize,
    gate: &SliceGate,
) -> BnnCascadeResult {
    let stride = stride.max(1);
    let positions: Vec<Fingerprint<256>> = (0..input.len())
        .step_by(stride)
        .map(|i| input[i].clone())
        .collect();

    bnn_cascade_search(kernel, &positions, positions.len(), gate)
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
    fn test_bnn_dot_matches_scalar() {
        // Regression test: verify SIMD path produces identical results
        // to scalar XNOR+popcount reference implementation.
        let mut rng = make_rng();
        let mut words_a = [0u64; 256];
        let mut words_b = [0u64; 256];
        for i in 0..256 {
            words_a[i] = rng.next_u64();
            words_b[i] = rng.next_u64();
        }
        let a = Fingerprint::from_words(words_a);
        let b = Fingerprint::from_words(words_b);

        // Scalar reference: XOR + popcount word by word
        let scalar_xor_pop: u32 = words_a
            .iter()
            .zip(words_b.iter())
            .map(|(&x, &y)| (x ^ y).count_ones())
            .sum();
        let scalar_match = 16_384 - scalar_xor_pop;
        let scalar_score = (2.0 * scalar_match as f32 / 16_384.0) - 1.0;

        // SIMD-dispatched path
        let result = bnn_dot(&a, &b);
        assert_eq!(
            result.match_count, scalar_match,
            "match_count mismatch: SIMD={} scalar={}",
            result.match_count, scalar_match
        );
        assert!(
            (result.score - scalar_score).abs() < f32::EPSILON,
            "score mismatch: SIMD={} scalar={}",
            result.score,
            scalar_score
        );
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

    // ===================================================================
    // Phase 1 tests: Cascade search
    // ===================================================================

    #[test]
    fn test_cascade_search_finds_exact_match() {
        let mut rng = make_rng();
        let mut query_words = [0u64; 256];
        for w in query_words.iter_mut() {
            *w = rng.next_u64();
        }
        let query = Fingerprint::from_words(query_words);

        let mut weights: Vec<_> = (0..100)
            .map(|_| {
                let mut w = [0u64; 256];
                for ww in w.iter_mut() {
                    *ww = rng.next_u64();
                }
                Fingerprint::from_words(w)
            })
            .collect();

        // Plant exact match at index 42
        weights[42] = query.clone();

        let gate = SliceGate::sku_16k();
        let result = bnn_cascade_search(&query, &weights, 5, &gate);

        // Exact match must be found
        assert!(
            result.matches.iter().any(|(idx, dot)| *idx == 42 && (dot.score - 1.0).abs() < f32::EPSILON),
            "Exact match at index 42 not found. Matches: {:?}",
            result.matches.iter().map(|(i, d)| (*i, d.score)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_cascade_search_zero_false_negatives() {
        // The critical invariant: cascade must find everything that passes
        // the SliceGate thresholds. We plant known matches and verify they
        // appear in cascade output.
        let mut rng = make_rng();
        let mut query_words = [0u64; 256];
        for w in query_words.iter_mut() {
            *w = rng.next_u64();
        }
        let query = Fingerprint::from_words(query_words);

        let mut weights: Vec<_> = (0..200)
            .map(|_| {
                let mut w = [0u64; 256];
                for ww in w.iter_mut() {
                    *ww = rng.next_u64();
                }
                Fingerprint::from_words(w)
            })
            .collect();

        // Plant 3 exact/near matches
        weights[10] = query.clone(); // exact
        let mut near1 = query_words;
        near1[50] ^= 1;
        weights[50] = Fingerprint::from_words(near1); // 1 bit flip
        let mut near2 = query_words;
        near2[100] ^= 1;
        weights[150] = Fingerprint::from_words(near2); // 1 bit flip

        let gate = SliceGate::sku_16k();
        let cascade = bnn_cascade_search(&query, &weights, 200, &gate);

        // All planted matches must appear (they have distance 0 or 1,
        // well within any reasonable gate threshold)
        assert!(
            cascade.matches.iter().any(|(i, _)| *i == 10),
            "Exact match at 10 not found"
        );
        assert!(
            cascade.matches.iter().any(|(i, _)| *i == 50),
            "Near match at 50 not found"
        );
        assert!(
            cascade.matches.iter().any(|(i, _)| *i == 150),
            "Near match at 150 not found"
        );
    }

    #[test]
    fn test_cascade_search_with_energy() {
        let mut rng = make_rng();
        let mut query_words = [0u64; 256];
        for w in query_words.iter_mut() {
            *w = rng.next_u64();
        }
        let query = Fingerprint::from_words(query_words);

        let mut weights: Vec<_> = (0..50)
            .map(|_| {
                let mut w = [0u64; 256];
                for ww in w.iter_mut() {
                    *ww = rng.next_u64();
                }
                Fingerprint::from_words(w)
            })
            .collect();
        weights[25] = query.clone();

        let gate = SliceGate::sku_16k();
        let (results, stats) = bnn_cascade_search_with_energy(&query, &weights, 10, &gate);

        // Should find exact match with agreement == energy
        let exact = results.iter().find(|r| r.index == 25);
        assert!(exact.is_some(), "Exact match not found in energy results");
        let exact = exact.unwrap();
        assert_eq!(exact.energy.conflict, 0);
        assert_eq!(exact.energy.agreement, exact.energy.energy_a);
        assert_eq!(stats.total, 50);
    }

    #[test]
    fn test_cascade_search_empty() {
        let query = Fingerprint::<256>::zero();
        let gate = SliceGate::sku_16k();
        let result = bnn_cascade_search(&query, &[], 10, &gate);
        assert!(result.matches.is_empty());
    }

    // ===================================================================
    // Phase 1 tests: BnnNetwork
    // ===================================================================

    #[test]
    fn test_network_creation() {
        let mut rng = make_rng();
        let net = BnnNetwork::new(&[10, 5, 3], &mut rng);
        assert_eq!(net.depth(), 3);
        assert_eq!(net.layers[0].len(), 10);
        assert_eq!(net.layers[1].len(), 5);
        assert_eq!(net.layers[2].len(), 3);
    }

    #[test]
    fn test_network_forward() {
        let mut rng = make_rng();
        let mut net = BnnNetwork::new(&[8, 4], &mut rng);

        let mut input_words = [0u64; 256];
        for w in input_words.iter_mut() {
            *w = rng.next_u64();
        }
        let input = Fingerprint::from_words(input_words);

        let (winner_idx, result) = net.forward(&input, false, 0.03, &mut rng);
        assert!(winner_idx < 4); // final layer has 4 neurons
        assert!(result.score.abs() <= 1.0 + f32::EPSILON);
    }

    #[test]
    fn test_network_predict() {
        let mut rng = make_rng();
        let net = BnnNetwork::new(&[10, 5, 3], &mut rng);

        let mut input_words = [0u64; 256];
        for w in input_words.iter_mut() {
            *w = rng.next_u64();
        }
        let input = Fingerprint::from_words(input_words);

        let layer_results = net.predict(&input);
        assert_eq!(layer_results.len(), 3);
        assert!(layer_results[0].0 < 10);
        assert!(layer_results[1].0 < 5);
        assert!(layer_results[2].0 < 3);
    }

    #[test]
    fn test_network_learning_changes_state() {
        let mut rng = make_rng();
        let mut net = BnnNetwork::new(&[5, 3], &mut rng);

        let mut input_words = [0u64; 256];
        for w in input_words.iter_mut() {
            *w = rng.next_u64();
        }
        let input = Fingerprint::from_words(input_words);

        // Record state before learning
        let before = net.layers[0].neurons[0].plastic().clone();

        // Train repeatedly
        for _ in 0..50 {
            net.forward(&input, true, 0.1, &mut rng);
        }

        let after = net.layers[0].neurons[0].plastic().clone();
        assert_ne!(before, after, "Network should learn from repeated input");
    }

    // ===================================================================
    // Phase 2 tests: CamIndex wiring
    // ===================================================================

    #[test]
    fn test_build_cam_index() {
        let mut rng = make_rng();
        let layer = BnnLayer::random(50, &mut rng);
        let cam = layer.build_cam_index(42);
        assert_eq!(cam.len(), 50);
    }

    #[test]
    fn test_winner_cam_finds_match() {
        let mut rng = make_rng();
        let mut layer = BnnLayer::random(100, &mut rng);

        // Plant a known weight at neuron 42
        let mut target_words = [0u64; 256];
        for w in target_words.iter_mut() {
            *w = rng.next_u64();
        }
        let target = Fingerprint::from_words(target_words);
        layer.neurons[42] = BnnNeuron::from_weights(target.clone(), &mut rng);

        let cam = layer.build_cam_index(123);

        // Query with the exact weight should find neuron 42
        let result = layer.winner_cam(&target, &cam, 20);
        assert!(result.is_some());
        let (idx, dot) = result.unwrap();
        // The exact match should be the winner or very close
        // (CAM is approximate — may not always return the exact match in shortlist)
        assert!(
            dot.score > 0.5,
            "Expected high score for planted match, got {:.4} at idx {}",
            dot.score,
            idx
        );
    }

    // ===================================================================
    // Phase 3 tests: Binary convolution
    // ===================================================================

    #[test]
    fn test_conv1d_basic() {
        let mut rng = make_rng();
        let sequence: Vec<_> = (0..10)
            .map(|_| {
                let mut w = [0u64; 256];
                for ww in w.iter_mut() {
                    *ww = rng.next_u64();
                }
                Fingerprint::from_words(w)
            })
            .collect();

        let kernel = sequence[5].clone();
        let results = bnn_conv1d(&sequence, &kernel, 1);

        assert_eq!(results.len(), 10);
        // Position 5 should have perfect match
        assert!(
            (results[5].score - 1.0).abs() < f32::EPSILON,
            "Expected score 1.0 at position 5, got {:.4}",
            results[5].score
        );
    }

    #[test]
    fn test_conv1d_stride() {
        let mut rng = make_rng();
        let sequence: Vec<_> = (0..20)
            .map(|_| {
                let mut w = [0u64; 256];
                for ww in w.iter_mut() {
                    *ww = rng.next_u64();
                }
                Fingerprint::from_words(w)
            })
            .collect();

        let kernel = sequence[0].clone();

        // Stride 1: 20 outputs
        let r1 = bnn_conv1d(&sequence, &kernel, 1);
        assert_eq!(r1.len(), 20);

        // Stride 3: ceil(20/3) = 7 outputs
        let r3 = bnn_conv1d(&sequence, &kernel, 3);
        assert_eq!(r3.len(), 7);

        // Stride 5: ceil(20/5) = 4 outputs
        let r5 = bnn_conv1d(&sequence, &kernel, 5);
        assert_eq!(r5.len(), 4);
    }

    #[test]
    fn test_conv1d_3ch() {
        let mut rng = make_rng();
        let sequence: Vec<_> = (0..5).map(|_| GraphHV::random(&mut rng)).collect();
        let kernel = sequence[2].clone();

        let results = bnn_conv1d_3ch(&sequence, &kernel, 1);
        assert_eq!(results.len(), 5);
        assert!(
            (results[2].score - 1.0).abs() < f32::EPSILON,
            "Expected score 1.0 at position 2, got {:.4}",
            results[2].score
        );
    }

    #[test]
    fn test_conv1d_cascade() {
        let mut rng = make_rng();
        let sequence: Vec<_> = (0..100)
            .map(|_| {
                let mut w = [0u64; 256];
                for ww in w.iter_mut() {
                    *ww = rng.next_u64();
                }
                Fingerprint::from_words(w)
            })
            .collect();

        let kernel = sequence[50].clone();
        let gate = SliceGate::sku_16k();

        let result = bnn_conv1d_cascade(&sequence, &kernel, 1, &gate);

        // Should find position 50 (exact match)
        assert!(
            result.matches.iter().any(|(idx, dot)| *idx == 50 && (dot.score - 1.0).abs() < f32::EPSILON),
            "Exact match at position 50 not found"
        );
    }
}
