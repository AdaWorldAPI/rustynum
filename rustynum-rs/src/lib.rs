// Numeric array operations use index loops and many-argument SIMD dispatch patterns.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

//! # RustyNum-Rs
//!
//! `rustynum-rs` is a numerical library for Rust, focusing on operations that can be vectorized using SIMD.
//! This crate provides efficient numerical arrays and operations, including basic arithmetic, dot products,
//! and transformations.

#![feature(portable_simd)]

/// Error type for rustynum operations that can fail on invalid input.
#[derive(Debug, Clone, PartialEq)]
pub enum NumError {
    /// Shape mismatch: data length doesn't match shape product
    ShapeMismatch {
        data_len: usize,
        shape_product: usize,
    },
    /// Dimension mismatch for matrix operations
    DimensionMismatch(String),
    /// Invalid parameter (e.g., step == 0 in arange)
    InvalidParameter(String),
    /// Axis out of bounds for the given array dimensionality
    AxisOutOfBounds { axis: usize, ndim: usize },
    /// Shapes are not broadcastable for the requested operation
    BroadcastError { lhs: Vec<usize>, rhs: Vec<usize> },
}

impl std::fmt::Display for NumError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NumError::ShapeMismatch {
                data_len,
                shape_product,
            } => {
                write!(
                    f,
                    "data length {} does not match shape product {}",
                    data_len, shape_product
                )
            }
            NumError::DimensionMismatch(msg) => write!(f, "dimension mismatch: {}", msg),
            NumError::InvalidParameter(msg) => write!(f, "invalid parameter: {}", msg),
            NumError::AxisOutOfBounds { axis, ndim } => {
                write!(f, "axis {} out of bounds for array with {} dimensions", axis, ndim)
            }
            NumError::BroadcastError { lhs, rhs } => {
                write!(f, "shapes not broadcastable: {:?} vs {:?}", lhs, rhs)
            }
        }
    }
}

impl std::error::Error for NumError {}

mod helpers;
pub mod num_array;
pub mod simd_ops;

pub mod traits;

pub use num_array::{binding_popcount_3d, find_discriminative_spots, find_holographic_sweet_spot};
pub use num_array::{decode_target_explicit, encode_edge_explicit, VerbCodebook};
pub use num_array::{simhash_batch_project, simhash_project};
pub use num_array::{
    sweep_cogrecords, CogRecord, SweepMode, SweepResult, COGRECORD_BYTES, CONTAINER_BITS,
    CONTAINER_BYTES, BTREE, CAM, EMBED, META,
};
pub use num_array::{ArrayView, ArrayViewMut};
pub use num_array::{NumArray, NumArrayF32, NumArrayF64, NumArrayI32, NumArrayI64, NumArrayU8};
pub use simd_ops::{BitwiseSimdOps, HammingSimdOps, SimdOps};
