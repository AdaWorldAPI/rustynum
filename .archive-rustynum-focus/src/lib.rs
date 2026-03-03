#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

//! # rustynum-focus
//!
//! Frozen snapshot of the CogRecordV3 + carrier + focus-gating model from
//! `rustynum-holo`, preserved before the holographic Gabor wavelet extension
//! was added.
//!
//! This crate contains:
//!
//! ## Phase-space operations (Operations 1-8)
//!
//! - **phase_bind/unbind** — reversible ADD/SUB mod 256
//! - **wasserstein_sorted** — O(N) Earth Mover's distance on sorted vectors
//! - **circular_distance** — wrap-around distance for unsorted phase vectors
//! - **phase_histogram_16** — 16-bin compact spatial address
//! - **phase_bundle_circular** — correct circular mean bundling (trig-based)
//! - **project_5d_to_phase / recover_5d_from_phase** — spatial coordinate encoding
//! - **sort/unsort** — write-time preparation for Wasserstein queries
//!
//! ## Carrier waveform operations (Operations 9-13)
//!
//! - **carrier_encode / carrier_decode** — frequency-domain concept encoding
//! - **carrier_bundle** — waveform addition (32 VPADDB)
//! - **carrier_distance_l1 / carrier_correlation** — waveform similarity
//! - **carrier_spectrum / spectral_distance** — frequency fingerprinting
//!
//! ## Focus-of-attention gating (Operations 14a-14d)
//!
//! - **focus_xor / focus_read / focus_add / focus_sub** — 3D gated ops
//! - **focus_hamming / focus_l1** — regional distance metrics
//! - **materialize_focus_mask** — expand 48-bit to 2048-byte mask
//! - **FocusRegistry / CompactDelta** — tracking and replication
//! - **concept_to_focus** — deterministic mask from concept ID
//!
//! ## Why archived?
//!
//! The live `rustynum-holo` crate was extended with holographic Gabor wavelet
//! binding (`holograph.rs`) that provides spatially-localized frequency encoding,
//! delta-cube relationship storage, and 3D spatial transforms. This archive
//! preserves the carrier + focus implementation for reference.

pub mod carrier;
pub mod cogrecord_v3;
pub mod focus;
pub mod phase;

pub use phase::{
    circular_distance_i8, generate_5d_basis, histogram_l1_distance, phase_bind_i8,
    phase_bind_i8_inplace, phase_bundle_approximate, phase_bundle_circular, phase_histogram_16,
    phase_inverse_i8, phase_unbind_i8, project_5d_to_phase, recover_5d_from_phase,
    sort_phase_vector, unsort_phase_vector, wasserstein_search_adaptive, wasserstein_sorted_i8,
};

pub use cogrecord_v3::{CogRecordV3, HybridDistances, HybridThresholds, CONTAINER_BYTES};

pub use carrier::{
    carrier_bundle, carrier_correlation, carrier_decode, carrier_distance_l1, carrier_encode,
    carrier_spectrum, spectral_distance, CarrierBasis, CarrierDistances, CarrierRecord,
    CarrierThresholds, CARRIER_AMPLITUDE, CARRIER_FREQUENCIES,
};

pub use focus::{
    concept_to_focus, focus_add, focus_add_materialized, focus_bind_binary, focus_bind_phase,
    focus_carrier_encode, focus_delta, focus_hamming, focus_l1, focus_read, focus_sub,
    focus_unbind_phase, focus_xor, focus_xor_auto, focus_xor_materialized, materialize_focus_mask,
    pack_focus, unpack_focus, CompactDelta, FocusDensity, FocusRegistry, FOCUS_DIM_X, FOCUS_DIM_Y,
    FOCUS_DIM_Z,
};
