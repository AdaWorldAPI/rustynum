//! CPU feature detection → Cranelift ISA flags.
//!
//! Maps runtime-detected CPU features to Cranelift's target ISA settings.
//! Reuses `rustynum_core::compute` for feature detection where possible.

use std::sync::Arc;

use cranelift_codegen::isa;
use cranelift_codegen::settings::{self, Configurable};
use target_lexicon::Triple;

use crate::ir::JitError;

/// Detected CPU capabilities relevant for JIT compilation.
#[derive(Debug, Clone)]
pub struct CpuCaps {
    pub has_avx2: bool,
    pub has_avx512f: bool,
    pub has_avx512bw: bool,
    pub has_avx512vl: bool,
    pub has_avx512vpopcntdq: bool,
    pub has_fma: bool,
    pub has_bmi2: bool,
}

impl CpuCaps {
    /// Detect CPU capabilities at runtime.
    #[cfg(target_arch = "x86_64")]
    pub fn detect() -> Self {
        Self {
            has_avx2: std::arch::is_x86_feature_detected!("avx2"),
            has_avx512f: std::arch::is_x86_feature_detected!("avx512f"),
            has_avx512bw: std::arch::is_x86_feature_detected!("avx512bw"),
            has_avx512vl: std::arch::is_x86_feature_detected!("avx512vl"),
            has_avx512vpopcntdq: std::arch::is_x86_feature_detected!("avx512vpopcntdq"),
            has_fma: std::arch::is_x86_feature_detected!("fma"),
            has_bmi2: std::arch::is_x86_feature_detected!("bmi2"),
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn detect() -> Self {
        Self {
            has_avx2: false,
            has_avx512f: false,
            has_avx512bw: false,
            has_avx512vl: false,
            has_avx512vpopcntdq: false,
            has_fma: false,
            has_bmi2: false,
        }
    }
}

/// Build a Cranelift target ISA from detected CPU capabilities.
pub fn build_isa(_caps: &CpuCaps) -> Result<Arc<dyn isa::TargetIsa>, JitError> {
    let mut flag_builder = settings::builder();
    flag_builder
        .set("opt_level", "speed")
        .map_err(|e| JitError::Codegen(e.to_string()))?;

    let isa_builder = isa::lookup(Triple::host()).map_err(|e| JitError::Codegen(e.to_string()))?;

    let flags = settings::Flags::new(flag_builder);
    isa_builder
        .finish(flags)
        .map_err(|e| JitError::Codegen(e.to_string()))
}
