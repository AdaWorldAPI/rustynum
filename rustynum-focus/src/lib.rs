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

pub mod phase;
pub mod cogrecord_v3;
pub mod carrier;
pub mod focus;

pub use phase::{
    phase_bind_i8, phase_bind_i8_inplace, phase_inverse_i8,
    phase_unbind_i8,
    wasserstein_sorted_i8, wasserstein_search_adaptive,
    circular_distance_i8,
    phase_histogram_16, histogram_l1_distance,
    phase_bundle_circular, phase_bundle_approximate,
    project_5d_to_phase, recover_5d_from_phase, generate_5d_basis,
    sort_phase_vector, unsort_phase_vector,
};

pub use cogrecord_v3::{
    CogRecordV3, HybridThresholds, HybridDistances, CONTAINER_BYTES,
};

pub use carrier::{
    CarrierBasis, CarrierRecord, CarrierThresholds, CarrierDistances,
    CARRIER_FREQUENCIES, CARRIER_AMPLITUDE,
    carrier_encode, carrier_decode, carrier_bundle,
    carrier_distance_l1, carrier_correlation,
    carrier_spectrum, spectral_distance,
};

pub use focus::{
    FocusDensity, FocusRegistry,
    pack_focus, unpack_focus,
    concept_to_focus, materialize_focus_mask,
    focus_xor, focus_read, focus_add, focus_sub,
    focus_xor_materialized, focus_add_materialized,
    focus_hamming, focus_l1,
    focus_bind_binary, focus_bind_phase, focus_unbind_phase,
    focus_carrier_encode,
    focus_delta, CompactDelta,
    focus_xor_auto,
    FOCUS_DIM_X, FOCUS_DIM_Y, FOCUS_DIM_Z,
};
