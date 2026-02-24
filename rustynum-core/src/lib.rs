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

pub mod bf16_hamming;
pub mod blackboard;
pub mod compute;
pub mod delta;
pub mod jit_scan;
pub mod jitson;
pub mod fingerprint;
pub mod hybrid;
pub mod kernels;
pub mod backends;
pub mod tail_backend;
pub mod layer_stack;
pub mod layout;
pub mod parallel;
pub mod rng;

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

pub use bf16_hamming::{
    bf16_bytes_to_fp32, bf16_hamming_scalar, fp32_to_bf16_bytes, pack_awareness_states,
    select_bf16_hamming_fn, structural_diff, superposition_decompose, unpack_awareness_states,
    AwarenessState, AwarenessThresholds, BF16StructuralDiff, BF16Weights, SuperpositionState,
    JINA_WEIGHTS, TRAINING_WEIGHTS,
};
pub use blackboard::Blackboard;
pub use compute::{ComputeCaps, ComputeTier, Precision};
pub use jit_scan::{DefaultKernelRegistry, ScanConfig, ScanResult, SimdKernelRegistry};
pub use jitson::{
    BackendConfig, JitsonError, JitsonTemplate, PipelineStage, PrecompileQueue, from_json,
};
pub use delta::DeltaLayer;
pub use fingerprint::{Fingerprint, Fingerprint1K, Fingerprint2K, Fingerprint64K};
pub use layer_stack::{CollapseGate, LayerStack};
pub use layout::{Layout, Transpose};
pub use parallel::parallel_for_chunks;
pub use kernels::{
    kernel_pipeline, kernel_pipeline_bytes, full_sweep, bf16_tail_score,
    SliceGate, EnergyConflict, HdrScore, KernelResult, KernelStage,
    PipelineStats, BenchmarkTranscript,
    SKU_16K_BITS, SKU_16K_BYTES, SKU_16K_WORDS,
    SKU_64K_BITS, SKU_64K_BYTES, SKU_64K_WORDS,
};
pub use hybrid::{
    hybrid_pipeline, hybrid_pipeline_with_backend,
    extract_learning_signal, update_hybrid_weights,
    HybridScore, HybridConfig, HybridStats, LearningSignal,
};
pub use tail_backend::{TailBackend, TailScore, BatchTailScore, auto_detect as auto_detect_backend};
pub use rng::SplitMix64;

// BF16 3D Spatial Resonance — Crystal4K-aligned axis model
// Wires: SPO grammar + semantic kernel + Jina 1024-D into 3-axis BF16 space
pub mod spatial_resonance;
pub use spatial_resonance::{
    CrystalAxis, SpatialCrystal3D, SpatialDistances, SpatialAxis,
    SpatialAwareness, SpatialLearningSignal, SpatialMatch,
    spatial_awareness_decompose, extract_spatial_learning_signal, spatial_sweep,
};
