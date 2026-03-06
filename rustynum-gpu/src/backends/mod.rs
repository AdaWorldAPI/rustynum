//! Compute backends — conditionally compiled via feature flags.
//!
//! ```toml
//! # CPU only (default, zero deps)
//! rustynum-gpu = "0.1"
//!
//! # CPU + universal GPU (Intel/NVIDIA/AMD/Apple/Qualcomm)
//! rustynum-gpu = { version = "0.1", features = ["wgpu-backend"] }
//!
//! # CPU + NVIDIA-optimized (cuBLAS, tensor cores)
//! rustynum-gpu = { version = "0.1", features = ["cuda"] }
//!
//! # CPU + Intel NPU/Xe (Level Zero, 75 TOPS on NUC 185H)
//! rustynum-gpu = { version = "0.1", features = ["level-zero"] }
//!
//! # Everything
//! rustynum-gpu = { version = "0.1", features = ["wgpu-backend", "cuda", "level-zero"] }
//! ```

// ─── Always available ───
pub mod cpu;

// ─── Feature-gated GPU backends ───

#[cfg(feature = "wgpu-backend")]
pub mod wgpu_backend;

#[cfg(feature = "cuda")]
pub mod cuda_backend;

#[cfg(feature = "level-zero")]
pub mod level_zero;
