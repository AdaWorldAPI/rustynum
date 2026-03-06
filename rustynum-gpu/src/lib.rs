//! # rustynum-gpu — heterogeneous compute for rustynum
//!
//! Auto-dispatches HDC operations across CPU, GPU, and NPU.
//! Each backend is behind a feature flag. CPU is always available.
//!
//! ## Quick start
//!
//! ```toml
//! [dependencies]
//! rustynum-gpu = { version = "0.1", features = ["wgpu-backend"] }
//! ```
//!
//! ```rust,no_run
//! use rustynum_gpu::{hdr_cascade_search, list_backends};
//! use rustynum_core::simd::PreciseMode;
//!
//! // See what's available
//! for backend in list_backends() {
//!     println!("{:?}", backend);
//! }
//!
//! // Search — auto-dispatches to best backend
//! let results = hdr_cascade_search(
//!     &query, &database,
//!     1250,        // 10K-bit vectors
//!     100_000,     // candidates
//!     4000,        // threshold
//!     PreciseMode::Off,
//! );
//! ```
//!
//! ## Feature flags
//!
//! | Feature | Backend | GPU | Use case |
//! |---------|---------|-----|----------|
//! | *(none)* | CPU only | — | Servers, CI, no GPU needed |
//! | `wgpu-backend` | wgpu | Intel/NVIDIA/AMD/Apple/Qualcomm | Universal, no SDK install |
//! | `cuda` | cudarc | NVIDIA only | cuBLAS, tensor cores, max perf |
//! | `level-zero` | dlopen | Intel NPU + Xe | 75 TOPS on NUC 185H |
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │  rustynum-gpu dispatch                                        │
//! │                                                                │
//! │  hdr_cascade_search(query, db, threshold)                     │
//! │       │                                                        │
//! │       ├─ estimate(HammingPrefix, batch_size) for each backend │
//! │       ├─ pick highest throughput                               │
//! │       └─ fallback chain: best → next → CPU (always works)     │
//! │                                                                │
//! ├──────────────────────┬──────────────┬─────────────────────────┤
//! │  CPU (always)        │  wgpu (opt)  │  CUDA (opt)  │ ZE (opt)│
//! │                      │              │              │          │
//! │  AVX-512 VPOPCNTDQ   │  Vulkan      │  cuBLAS      │ NPU     │
//! │  AVX2 Harley-Seal    │  DX12        │  tensor core │ Xe GPU  │
//! │  AVX2 VNNI           │  Metal       │  __popc      │ dlopen  │
//! │  scalar POPCNT       │  WebGPU      │              │          │
//! │                      │              │              │          │
//! │  rustynum-core       │  stroke1.wgsl│  PTX kernel  │ SPIR-V  │
//! └──────────────────────┴──────────────┴──────────────┴──────────┘
//! ```
//!
//! ## Device matrix
//!
//! ```text
//!                     AVX-512   CUDA   iGPU(wgpu)  NPU(ze)   RAM
//! Laptop 11gen          ✓        ✓        ✓          ✗       64GB
//! NUC 185H              ✗        ✗        ✓          ✓       96GB
//! Cloud Sapphire        ✓        ✗        ✗          ✗       varies
//! MacBook M-series      ✗        ✗        ✓(Metal)   ✗       varies
//! Snapdragon X          ✗        ✗        ✓(Vulkan)  ✗       varies
//! ```

pub mod backends;
pub mod dispatch;
pub mod traits;

// Re-export the public API
pub use dispatch::{hamming_batch, hdr_cascade_search, list_backends, which_backend};
pub use traits::{BackendInfo, ComputeBackend, DeviceKind, ElementwiseOp, OpHint};
