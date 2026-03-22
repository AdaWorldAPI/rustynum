//! Unified SIMD type abstraction — the wrapper IS the dispatch boundary.
//!
//! `F32x16` always means "16 f32 lanes" regardless of hardware. The backing
//! storage changes per target:
//!
//! | Target                  | F32x16 backing     | F64x8 backing      |
//! |------------------------|--------------------|--------------------|
//! | x86_64 + AVX-512       | `__m512`           | `__m512d`          |
//! | x86_64 + AVX2          | `[F32x8; 2]`       | `[F64x4; 2]`       |
//! | aarch64 (NEON)         | `[float32x4_t; 4]` | `[float64x2_t; 4]` |
//! | other (scalar fallback)| `[f32; 16]`        | `[f64; 8]`         |
//!
//! # Architecture
//!
//! Kernel code writes to ONE set of types:
//! ```rust,ignore
//! use rustynum_core::simd_types::F32x16;
//!
//! pub fn dot_f32(x: &[f32], y: &[f32]) -> f32 {
//!     let mut acc = F32x16::splat(0.0);
//!     for (cx, cy) in x.chunks_exact(16).zip(y.chunks_exact(16)) {
//!         acc = F32x16::from_slice(cx).mul_add(F32x16::from_slice(cy), acc);
//!     }
//!     acc.reduce_sum() // + handle tail
//! }
//! ```
//!
//! This compiles to optimal code on each target:
//! - AVX-512: one `vfmadd231ps zmm` per iteration
//! - AVX2: two `vfmadd231ps ymm` per iteration
//! - NEON: four `fmla v.4s` per iteration
//! - Scalar: 16 FMAs per iteration (LLVM may auto-vectorize)
//!
//! ## x86_64 runtime dispatch
//!
//! On x86_64 both AVX-512 and AVX2 backends exist. A `LazyLock<Tier>` detects
//! CPU features once at init. The `dispatch!` macro in `simd.rs` selects the
//! right backend at runtime. Kernel code stays identical — only the type changes.

// Backend modules — each defines the same API surface for F32x16, F64x8, etc.
#[cfg(target_arch = "x86_64")]
pub mod avx512;
#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "aarch64")]
pub mod neon;
pub mod scalar;

// Re-export the right backend per target.
// On x86_64 both avx512 and avx2 are available; the default export is avx2
// (safe on all x86_64 CPUs with AVX2+FMA). AVX-512 kernels are called via
// dispatch! with #[target_feature] guards.
//
// On aarch64, NEON is baseline — always available.
// On everything else, scalar fallback.

#[cfg(target_arch = "aarch64")]
pub use neon::{F32x16, F64x8};

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub use scalar::{F32x16, F64x8};

// On x86_64, we don't re-export a default F32x16 — callers use either
// avx512::F32x16 or avx2::F32x16 via the dispatch! macro, or use
// the "safe default" avx2 types for code that doesn't need dispatch.
#[cfg(target_arch = "x86_64")]
pub use avx2::{F32x16, F64x8};
