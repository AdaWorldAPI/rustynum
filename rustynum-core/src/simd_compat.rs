//! Backward-compatibility shim. All types moved to simd_avx512.rs.
#[allow(deprecated)]
#[deprecated(since = "0.4.0", note = "renamed to simd_avx512")]
pub use crate::simd_avx512::*;
