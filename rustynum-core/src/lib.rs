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

// All SIMD uses stable std::arch via simd_compat — no nightly required.

pub mod simd_compat;

pub mod backends;
pub mod bf16_hamming;
pub mod blackboard;
pub mod compute;
pub mod delta;
pub mod fingerprint;
pub mod hybrid;
pub mod jit_scan;
pub mod jitson;
pub mod kernels;
pub mod layer_stack;
pub mod layout;
pub mod parallel;
pub mod rng;
pub mod tail_backend;

#[cfg(any(feature = "avx512", feature = "avx2"))]
pub mod prefilter;

// SIMD backend selection: AVX-512 (stable via simd_compat) or AVX2 (still uses std::simd)
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
pub use delta::DeltaLayer;
pub use fingerprint::{Fingerprint, Fingerprint1K, Fingerprint2K, Fingerprint64K};
pub use hybrid::{
    extract_learning_signal, hybrid_pipeline, hybrid_pipeline_with_backend, resonance_decompose,
    update_hybrid_weights, HybridConfig, HybridScore, HybridStats, LearningSignal, ResonanceResult,
    ResonantMatch, Tier0Config, Tier0Mode, Tier0Stats,
};
pub use jit_scan::{DefaultKernelRegistry, ScanConfig, ScanResult, SimdKernelRegistry};
pub use jitson::{
    from_json, BackendConfig, JitsonError, JitsonTemplate, PipelineStage, PrecompileQueue,
};
pub use kernels::{
    bf16_tail_score, bytes_to_u64_words, full_sweep, k0_probe, k1_stats, k2_exact, kernel_pipeline,
    kernel_pipeline_bytes, score_hdr, BenchmarkTranscript, EnergyConflict, HdrScore, KernelResult,
    KernelStage, PipelineStats, SliceGate, SKU_16K_BITS, SKU_16K_BYTES, SKU_16K_WORDS,
    SKU_64K_BITS, SKU_64K_BYTES, SKU_64K_WORDS,
};
pub use layer_stack::{CollapseGate, LayerStack};
pub use layout::{Layout, Transpose};
pub use parallel::parallel_for_chunks;
pub use rng::SplitMix64;
pub use tail_backend::{
    auto_detect as auto_detect_backend, capabilities as backend_capabilities, gemm_backend,
    gemm_backend_with_scale, BatchTailScore, Capabilities, CompactTailScore, TailBackend,
    TailScore,
};

// BF16 3D Spatial Resonance — Crystal4K-aligned axis model
// Wires: SPO grammar + semantic kernel + Jina 1024-D into 3-axis BF16 space
pub mod spatial_resonance;
pub use spatial_resonance::{
    extract_spatial_learning_signal, spatial_awareness_decompose, spatial_sweep, CrystalAxis,
    SpatialAwareness, SpatialAxis, SpatialCrystal3D, SpatialDistances, SpatialLearningSignal,
    SpatialMatch,
};

// 3D Graph HDC: 16,384×3-bit hypervectors for plastic graph memory
// GraphHV = 3 × Fingerprint<256> (node/edge/plastic channels)
pub mod graph_hv;
pub use graph_hv::{
    bundle, bundle_into, decode_edge_source, encode_edge, GraphHV, GRAPH_HV_BITS, GRAPH_HV_BYTES,
    GRAPH_HV_CHANNELS,
};

// Content-Addressable Memory index: multi-probe LSH for O(log N) lookup
pub mod cam_index;
pub use cam_index::{CamConfig, CamHit, CamIndex};

// DN-tree: hierarchical plasticity tree with BTSP-gated bundling
pub mod dn_tree;
pub use dn_tree::{DNConfig, DNNode, DNTree, DNTreeStats, TraversalHit};

// BNN inference primitives live in rustynum-bnn crate (not core).
// BNN is purely additive neural plasticity — consumes core types without modifying them.
