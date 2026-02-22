//! # RustyNum Core
//!
//! Shared SIMD primitives and zero-copy blackboard for the rustynum ecosystem.
//!
//! This crate provides:
//! - **Blackboard**: Zero-copy shared mutable memory arena with SIMD-aligned allocations.
//!   Eliminates serialization between rustynum-rs, rustyblas, and rustymkl — all crates
//!   operate directly on the same aligned memory.
//! - **SIMD primitives**: Portable SIMD helpers shared across all crates.
//! - **Parallel execution**: Thread pool utilities for data-parallel SIMD workloads.
//! - **CBLAS layout types**: Row-major / column-major layout abstractions.

#![cfg_attr(any(feature = "avx512", feature = "avx2"), feature(portable_simd))]
#![cfg_attr(feature = "avx512", feature(stdarch_x86_avx512))]
#![cfg_attr(feature = "avx512", feature(avx512_target_feature))]

pub mod blackboard;
pub mod compute;
pub mod fingerprint;
pub mod layout;
pub mod parallel;
pub mod rng;
pub mod bf16_hamming;

#[cfg(any(feature = "avx512", feature = "avx2"))]
pub mod prefilter;

// SIMD backend selection: AVX-512 or AVX2 (requires nightly for portable_simd)
#[cfg(feature = "avx512")]
pub mod simd;
#[cfg(all(feature = "avx2", not(feature = "avx512")))]
#[path = "simd_avx2.rs"]
pub mod simd;

// Intel MKL FFI bindings (only compiled when --features mkl is enabled)
#[cfg(feature = "mkl")]
pub mod mkl_ffi;

pub use blackboard::Blackboard;
pub use compute::{ComputeCaps, ComputeTier, Precision};
pub use fingerprint::{Fingerprint, Fingerprint2K, Fingerprint1K, Fingerprint64K};
pub use layout::{Layout, Transpose};
pub use parallel::parallel_for_chunks;
pub use rng::SplitMix64;
pub use bf16_hamming::{
    BF16Weights, BF16StructuralDiff,
    fp32_to_bf16_bytes, bf16_bytes_to_fp32,
    structural_diff, select_bf16_hamming_fn, bf16_hamming_scalar,
    JINA_WEIGHTS, TRAINING_WEIGHTS,
};
