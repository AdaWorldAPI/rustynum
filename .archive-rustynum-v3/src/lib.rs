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

#![allow(clippy::needless_range_loop)]

pub mod cogrecord_v3;
pub mod phase;

pub use phase::{
    circular_distance_i8, generate_5d_basis, histogram_l1_distance, phase_bind_i8,
    phase_bind_i8_inplace, phase_bundle_approximate, phase_bundle_circular, phase_histogram_16,
    phase_inverse_i8, phase_unbind_i8, project_5d_to_phase, recover_5d_from_phase,
    sort_phase_vector, unsort_phase_vector, wasserstein_search_adaptive, wasserstein_sorted_i8,
};

pub use cogrecord_v3::{CogRecordV3, HybridDistances, HybridThresholds, CONTAINER_BYTES};
