//! Backend implementations for tail scoring.
//!
//! Each backend is an isolated module. Unsafe code and FFI live here,
//! nowhere else. The orchestration layer (hybrid.rs) only sees
//! `&dyn TailBackend`.

pub mod popcnt;

#[cfg(feature = "libxsmm")]
pub mod xsmm;
