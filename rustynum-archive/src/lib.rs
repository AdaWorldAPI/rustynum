//! # rustynum-archive
//!
//! Archive of the CogRecord v1 binary-only model, preserved before the
//! phase-space upgrade (CogRecordV3 in `rustynum-holo`).
//!
//! This crate contains frozen copies of the original model code:
//!
//! - **CogRecord** — 4 × 2048-byte binary containers, Hamming distance only
//! - **HDC primitives** — BIND (XOR), PERMUTE, BUNDLE (majority vote),
//!   DOT_I8 (VNNI), adaptive cascade search
//! - **VerbCodebook** — edge encoding/decoding, causality asymmetry
//! - **SimHash projection** — batch f32 → binary via tiled GEMM
//! - **3D XYZ binding matrix** — spectral analysis of HDC binding space
//!
//! ## Why archived?
//!
//! The v1 model uses binary containers exclusively (XOR bind, Hamming distance).
//! The v3 model (`rustynum-holo`) adds phase-space containers (ADD bind,
//! Wasserstein/circular distance) for genuine spatial navigation.
//!
//! This archive ensures the original model is always available for reference
//! and backward compatibility.

#![feature(portable_simd)]
#![allow(clippy::needless_range_loop)]

pub mod binding_matrix;
pub mod cogrecord;
pub mod graph;
pub mod hdc;
pub mod projection;

pub use binding_matrix::{
    binding_popcount_3d, find_discriminative_spots, find_holographic_sweet_spot,
};
pub use cogrecord::{sweep_cogrecords, CogRecord, SweepMode, SweepResult};
pub use cogrecord::{BTREE, CAM, COGRECORD_BYTES, CONTAINER_BITS, CONTAINER_BYTES, EMBED, META};
pub use graph::{decode_target_explicit, encode_edge_explicit, VerbCodebook};
pub use projection::{simhash_batch_project, simhash_project};
