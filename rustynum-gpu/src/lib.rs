//! # rustynum-gpu — GPU compute backend for rustynum
//!
//! Heterogeneous Belichtungsmesser: GPU Stroke 1 + CPU Strokes 2-3.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │                   NUC 185H (Meteor Lake)                 │
//! │                                                          │
//! │   CPU (AVX2 + VNNI)         iGPU (Xe-LPG, 128 XVEs)    │
//! │   ┌──────────────┐          ┌──────────────────────┐     │
//! │   │ Stroke 2:    │          │ Stroke 1:            │     │
//! │   │  incremental │  ←────  │  prefix XOR+popcount │     │
//! │   │  Hamming on  │ results │  ALL candidates      │     │
//! │   │  survivors   │          │  zero branching      │     │
//! │   │              │          │  128×8 = 1024 wide   │     │
//! │   │ Stroke 3:    │          └──────────────────────┘     │
//! │   │  VNNI cosine │                                       │
//! │   │  on finalists│                                       │
//! │   └──────────────┘                                       │
//! │                                                          │
//! │   ─────── 96 GB shared DDR5, zero copy ──────────       │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Why not GPU early-exit?
//!
//! GPUs hate divergent branching. The Belichtungsmesser's power comes from
//! early-exit: killing 90% of candidates after reading 20% of the vector.
//! On CPU, branch prediction makes this fast. On GPU, divergent threads
//! within a SIMD lane stall the entire wavefront.
//!
//! Solution: **2-stroke engine.** GPU does the uniform work (prefix popcount
//! on ALL candidates, no branching). CPU does the branchy work (σ threshold,
//! early-exit evaluation, tree traversal, CLAM refill).
//!
//! The GPU never decides. It just popcounts. The CPU never bulk-scans.
//! It just evaluates and routes. Each processor does what it's built for.
//!
//! ## Platform support via wgpu
//!
//! | Platform        | Backend    | GPU                    |
//! |-----------------|------------|------------------------|
//! | Linux native    | Vulkan/ANV | Intel Xe-LPG/Xe2       |
//! | WSL2            | DX12 (PV)  | Intel Xe-LPG via host  |
//! | Windows         | DX12       | Intel/NVIDIA/AMD       |
//! | macOS           | Metal      | Apple Silicon           |
//! | Snapdragon      | Vulkan     | Adreno                 |
//!
//! Same code. Same API. `wgpu` adapts.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use rustynum_gpu::hdr_cascade_search_gpu;
//! use rustynum_core::simd::PreciseMode;
//!
//! let results = hdr_cascade_search_gpu(
//!     &query, &database,
//!     1250,       // 10K-bit vectors
//!     100_000,    // candidate count
//!     4000,       // threshold
//!     PreciseMode::Off,
//! );
//! ```
//!
//! Falls back to CPU automatically when GPU unavailable or batch too small.

pub mod device;
pub mod dispatch;

// Re-export main entry points
pub use device::{gpu_available, gpu_capabilities, GpuCapabilities};
pub use dispatch::{hdr_cascade_search_gpu, plan_dispatch, DispatchStrategy};
