mod array_struct;
pub mod binding_matrix;
pub mod bitwise;
pub mod cogrecord;
mod constructors;
pub mod graph;
pub mod hdc;
mod impl_clone_from;
pub mod linalg;
mod manipulation;
pub mod operations;
pub mod projection;
mod statistics;

pub use array_struct::{NumArray, NumArrayF32, NumArrayF64, NumArrayI32, NumArrayI64, NumArrayU8};
pub use binding_matrix::{binding_popcount_3d, find_holographic_sweet_spot, find_discriminative_spots};
pub use cogrecord::{CogRecord, SweepMode, SweepResult, sweep_cogrecords};
pub use graph::{VerbCodebook, encode_edge_explicit, decode_target_explicit};
pub use projection::{simhash_batch_project, simhash_project};
