#![allow(clippy::needless_range_loop)]

//! # rustynum-arrow
//!
//! Optional Arrow/Lance/DataFusion bridge for the rustynum ecosystem.
//!
//! This crate is **not** a dependency of any other rustynum crate.
//! Enable it only if you need Arrow interop, Lance dataset I/O,
//! or DataFusion cascade scanning.
//!
//! ## Features
//!
//! - `arrow` (default) — Zero-copy NumArray ↔ Arrow array conversions + DataFusion cascade scan
//! - `datafusion` — DataFusion 51 (implies `arrow`)
//! - `lance` — CogRecord ↔ Lance dataset read/write (implies `arrow`)
//!
//! ## Example
//!
//! ```rust,ignore
//! use rustynum_arrow::IntoArrow;
//! use rustynum_rs::NumArrayF32;
//!
//! let arr = NumArrayF32::new(vec![1.0, 2.0, 3.0]);
//! let arrow_arr = arr.into_arrow();
//! ```

#[cfg(feature = "arrow")]
pub mod arrow_bridge;

#[cfg(feature = "arrow")]
pub mod datafusion_bridge;

#[cfg(feature = "lance")]
pub mod lance_io;

#[cfg(feature = "arrow")]
pub mod fragment_index;

#[cfg(feature = "arrow")]
pub mod channel_index;

#[cfg(feature = "arrow")]
pub mod indexed_cascade;

#[cfg(feature = "arrow")]
pub mod horizontal_sweep;

// Re-exports for convenience
#[cfg(feature = "arrow")]
pub use arrow_bridge::{
    cogrecord_schema, cogrecords_to_record_batch, record_batch_to_cogrecords, FromArrow, IntoArrow,
};

#[cfg(feature = "arrow")]
pub use datafusion_bridge::{arrow_to_flat_bytes, cascade_scan_4ch, hamming_scan_column};

#[cfg(feature = "lance")]
pub use lance_io::{append_cogrecords, read_cogrecords, write_cogrecords};

#[cfg(feature = "arrow")]
pub use fragment_index::{FragmentIndex, FragmentMeta};

#[cfg(feature = "arrow")]
pub use channel_index::{ChannelIndex, ClusterMeta};

#[cfg(feature = "arrow")]
pub use indexed_cascade::{
    build_single_channel_index, indexed_cascade_search, learn, rebuild,
    single_channel_search, CascadeIndices, IndexedCascadeResult, IndexedCascadeStats,
    SingleChannelResult, SingleChannelStats,
};

#[cfg(feature = "arrow")]
pub use horizontal_sweep::{
    horizontal_sweep, horizontal_sweep_filtered, hybrid_cascade_sweep, HorizontalSweepConfig,
    HorizontalSweepResult, HorizontalSweepStats, HybridCascadeResult,
};
