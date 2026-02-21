//! # rustynum-holo
//!
//! Phase-space holographic operations for CogRecord v3.
//!
//! This crate implements the phase-space layer that sits on top of the
//! binary CogRecord engine in `rustynum-rs`. While binary containers
//! (XOR bind, Hamming distance) provide hash-table-like exact matching,
//! phase containers (ADD bind, Wasserstein/circular distance) provide
//! genuine spatial navigation.
//!
//! ## The Hybrid Architecture
//!
//! | Container | Mode   | Bind     | Distance    | Use                  |
//! |-----------|--------|----------|-------------|----------------------|
//! | META      | Binary | XOR      | Hamming     | Fast rejection       |
//! | CAM       | Phase  | ADD      | Wasserstein | Spatial navigation   |
//! | BTREE     | Binary | XOR      | Hamming     | Graph structure      |
//! | EMBED     | Phase  | ADD      | Circular    | Dense similarity     |
//!
//! ## Operations
//!
//! - `phase_bind_i8` / `phase_unbind_i8` — reversible phase-space binding
//! - `wasserstein_sorted_i8` — O(N) Earth Mover's distance on sorted vectors
//! - `circular_distance_i8` — wrap-around distance for unsorted phase vectors
//! - `phase_histogram_16` — 16-bin compact spatial address
//! - `phase_bundle_circular` — correct circular mean bundling
//! - `project_5d_to_phase` / `recover_5d_from_phase` — spatial coordinate encoding
//! - `sort_phase_vector` / `unsort_phase_vector` — write-time preparation
//!
//! ## Carrier Model (alternative encoding)
//!
//! - `carrier_encode` / `carrier_decode` — frequency-domain concept encoding
//! - `carrier_bundle` — waveform addition (32 VPADDB vs ~500 trig instructions)
//! - `carrier_distance_l1` / `carrier_correlation` — waveform similarity
//! - `carrier_spectrum` / `spectral_distance` — frequency fingerprinting
//! - `CarrierRecord` — hybrid binary + carrier containers

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
