//! # rustynum-archive-v3
//!
//! Frozen snapshot of the CogRecordV3 phase-space model from `rustynum-holo`,
//! preserved before the carrier waveform extension was added.
//!
//! This crate contains the original 8 phase-space operations and the hybrid
//! binary+phase CogRecordV3 struct:
//!
//! - **phase_bind/unbind** — reversible ADD/SUB mod 256
//! - **wasserstein_sorted** — O(N) Earth Mover's distance on sorted vectors
//! - **circular_distance** — wrap-around distance for unsorted phase vectors
//! - **phase_histogram_16** — 16-bin compact spatial address
//! - **phase_bundle_circular** — correct circular mean bundling (trig-based)
//! - **project_5d_to_phase / recover_5d_from_phase** — spatial coordinate encoding
//! - **sort/unsort** — write-time preparation for Wasserstein queries
//! - **CogRecordV3** — hybrid struct: META+BTREE binary, CAM phase-sorted, EMBED phase-unsorted
//!
//! ## Why archived?
//!
//! The live `rustynum-holo` crate was extended with a carrier model (`carrier.rs`)
//! that provides an alternative encoding for phase containers. This archive
//! preserves the pure random-phase implementation for reference.

pub mod phase;
pub mod cogrecord_v3;

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
