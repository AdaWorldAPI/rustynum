//! # RustyNum BNN
//!
//! K0/K1 Belichtungsmesser (Exposure Meter) cascade for DN-tree traversal
//! acceleration. Consumes `rustynum-core` infrastructure without modification.
//!
//! ## Progressive Stichprobe (4 Tiers × σ Gates)
//!
//! ```text
//! Tier  Bits   σ-gate  Reject%  Cost   Measurement
//! ────  ─────  ──────  ───────  ─────  ───────────
//! K0    64     1σ      ~84%     ~1ns   Spot meter (1 word)
//! K1    512    2σ      ~97.5%   ~4ns   Zone meter (8 words)
//! BF16  512    3σ      ~99.7%   ~20ns  Range awareness (cold only)
//! Full  49152  exact   100%     ~48ns  Complete (leaves only)
//! ```

pub mod belichtungsmesser;

// Re-export BNN types from rustynum-core (no duplication)
pub use rustynum_core::bnn::{
    bnn_batch_dot, bnn_cascade_search, bnn_cascade_search_with_energy, bnn_conv1d, bnn_conv1d_3ch,
    bnn_conv1d_cascade, bnn_dot, bnn_dot_3ch, BnnCascadeResult, BnnDotResult, BnnEnergyResult,
    BnnLayer, BnnNetwork, BnnNeuron,
};

pub use rustynum_core::bnn::bnn_hdr_search;

// Re-export Belichtungsmesser types
pub use belichtungsmesser::{
    bf16_refine_cold, classify_hdr, filter_children, hdr_beam_width, k0_probe_conflict,
    k1_stats_conflict, signal_quality, ChildScore, TraversalStats,
};
