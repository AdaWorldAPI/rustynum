//! AVX-512 backend: F32x16 = __m512, F64x8 = __m512d.
//!
//! One instruction per wide operation. Requires AVX-512F at minimum.
//!
//! **Every function in this module must be called from a context guarded by
//! `#[target_feature(enable = "avx512f")]` or `is_x86_feature_detected!("avx512f")`.**
//! The types themselves are thin wrappers around `__m512` / `__m512d`.
//!
//! This module re-exports from `simd_avx512.rs` to maintain a single source of
//! truth for the AVX-512 intrinsic wrappers. The types are the same; this module
//! exists so that `simd_types::avx512::F32x16` and `simd_types::avx2::F32x16`
//! are distinct types with the same API surface, enabling the dispatch pattern.

// Re-export the existing AVX-512 wrapper types.
// These are the SAME types as in simd_avx512.rs — no duplication.
pub use crate::simd_avx512::F32x16;
pub use crate::simd_avx512::F64x8;
pub use crate::simd_avx512::F32Mask16;
pub use crate::simd_avx512::U8x64;
pub use crate::simd_avx512::I32x16;
pub use crate::simd_avx512::I64x8;
pub use crate::simd_avx512::U32x16;
pub use crate::simd_avx512::U64x8;
