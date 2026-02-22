#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

//! # rustynum-carrier
//!
//! Frozen snapshot of the CogRecordV3 + carrier waveform model from `rustynum-holo`,
//! preserved before the focus-of-attention gating extension was added.
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
//! - **carrier_bundle** — waveform addition (32 VPADDB vs ~500 trig instructions)
//! - **carrier_distance_l1 / carrier_correlation** — waveform similarity
//! - **carrier_spectrum / spectral_distance** — frequency fingerprinting
//!
//! ## Structs
//!
//! - **CogRecordV3** — hybrid binary + phase containers (META, CAM, BTREE, EMBED)
//! - **CarrierRecord** — hybrid binary + carrier containers
//! - **CarrierBasis** — precomputed 16-frequency Chebyshev carrier basis (64KB)
//!
//! ## Why archived?
//!
//! The live `rustynum-holo` crate was extended with focus-of-attention gating
//! (`focus.rs`) that provides 3D spatial sub-selection within containers. This
//! archive preserves the pure carrier + phase implementation for reference.

pub mod phase;
pub mod cogrecord_v3;
pub mod carrier;

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
