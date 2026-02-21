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

#![feature(portable_simd)]

pub mod blackboard;
pub mod layout;
pub mod parallel;
pub mod simd;

pub use blackboard::{Blackboard, BufferHandle};
pub use layout::{Layout, Transpose};
pub use parallel::parallel_for_chunks;
