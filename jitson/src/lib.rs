//! # jitson — JSON config → native code via Cranelift JIT
//!
//! `jitson` turns JSON/YAML configuration values into native function pointers.
//! Your config IS the code. Threshold comparisons become CMP immediates,
//! focus masks become VPANDQ bitmasks, branch weights become branch hints.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use jitson::{JitEngine, ScanKernel, ScanParams};
//!
//! let engine = JitEngine::new()?;
//!
//! let params = ScanParams {
//!     threshold: 500,
//!     top_k: 32,
//!     prefetch_ahead: 4,
//!     focus_mask: None,       // all dimensions awake
//! };
//!
//! let kernel = engine.compile_scan(params)?;
//! let candidates = unsafe { kernel.scan(query, field, field_len, record_size) };
//! ```
//!
//! ## Architecture
//!
//! ```text
//! JSON/YAML config
//!       │
//!       ▼
//! Parse (serde)  →  Intermediate params struct
//!       │
//!       ▼
//! Cranelift IR builder  →  CLIF IR
//!       │
//!       ▼
//! Cranelift codegen  →  native machine code
//!       │
//!       ▼
//! Function pointer  →  cached in kernel registry
//! ```
//!
//! Field values baked as immediates:
//! - `threshold: 500` → `CMP reg, 500` (not `LOAD + CMP`)
//! - `focus_mask: [47, 193]` → 8KB AND mask as immediate data
//! - `prefetch_ahead: 4` → `PREFETCHT0 [ptr + 4 * RECORD_SIZE]`
//! - `confidence: 0.7` → branch probability hint

pub mod detect;
pub mod engine;
pub mod ir;
pub mod scan_jit;

// Re-exports
pub use engine::JitEngine;
pub use ir::{JitError, ScanParams};
pub use scan_jit::ScanKernel;
