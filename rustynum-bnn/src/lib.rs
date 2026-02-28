//! # RustyNum BNN
//!
//! Binary Neural Network inference + K0/K1 Belichtungsmesser (Exposure Meter)
//! cascade for DN-tree traversal acceleration.
//!
//! BNN is the neural plasticity layer. It consumes `rustynum-core` types
//! (Fingerprint, GraphHV, kernels) without modification.
//!
//! ## Progressive Stichprobe (4 Tiers x sigma Gates)
//!
//! ```text
//! Tier  Bits   sigma-gate  Reject%  Cost   Measurement
//! ----  -----  ----------  -------  -----  -----------
//! K0    64     1 sigma     ~84%     ~1ns   Spot meter (1 word)
//! K1    512    2 sigma     ~97.5%   ~4ns   Zone meter (8 words)
//! BF16  512    3 sigma     ~99.7%   ~20ns  Range awareness (cold only)
//! Full  49152  exact       100%     ~48ns  Complete (leaves only)
//! ```

pub mod belichtungsmesser;
pub mod bnn;
pub mod rif_net_integration;

// Re-export BNN types (owned by this crate, not core)
pub use bnn::{
    bnn_batch_dot, bnn_cascade_search, bnn_cascade_search_with_energy, bnn_conv1d, bnn_conv1d_3ch,
    bnn_conv1d_cascade, bnn_dot, bnn_dot_3ch, BnnCascadeResult, BnnDotResult, BnnEnergyResult,
    BnnLayer, BnnNetwork, BnnNeuron,
};

#[cfg(any(feature = "avx512", feature = "avx2"))]
pub use bnn::bnn_hdr_search;

// Re-export Belichtungsmesser types (only genuinely new functions)
// All K0/K1 probes, σ-gated rejection, HDR classification already in rustynum-core.
pub use belichtungsmesser::{bf16_refine_cold, signal_quality};

// Re-export RIF-Net integration (Zhang et al. 2025 BIR-EWM + rich information flow)
pub use rif_net_integration::{
    BPReLU, BinaryBatchNorm, EwmWeights, RifCaBlock, RifFlowMetrics, RifNet,
};
