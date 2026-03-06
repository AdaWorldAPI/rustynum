//! The `ComputeBackend` trait — common interface across CPU, wgpu, CUDA, and Level Zero.
//!
//! Every backend implements this trait. Dispatch logic picks the best one
//! based on operation type, batch size, and available memory.

use rustynum_core::simd::{HdrResult, PreciseMode};

/// Device type classification for dispatch decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    /// CPU (AVX-512, AVX2+VNNI, NEON, scalar)
    Cpu,
    /// Integrated GPU sharing system RAM (Xe-LPG, Apple, Adreno)
    IntegratedGpu,
    /// Discrete GPU with dedicated VRAM (3050Ti, Arc A770, etc.)
    DiscreteGpu,
    /// Neural Processing Unit (Intel AI Boost, Qualcomm Hexagon, Apple ANE)
    Npu,
}

/// Capabilities reported by a backend.
#[derive(Debug, Clone)]
pub struct BackendInfo {
    /// Human-readable name (e.g. "AVX-512 VPOPCNTDQ", "NVIDIA GeForce RTX 3050 Ti", "Intel Arc")
    pub name: String,
    /// What kind of device this is
    pub device_kind: DeviceKind,
    /// Available memory in bytes (system RAM for CPU, VRAM for discrete, shared for iGPU)
    pub available_memory: usize,
    /// Whether this device shares memory with CPU (zero-copy possible)
    pub unified_memory: bool,
    /// Rough throughput estimate for INT8 ops (TOPS, for dispatch heuristics)
    pub int8_tops: f32,
}

/// Operation hint for dispatch decisions.
#[derive(Debug, Clone, Copy)]
pub enum OpHint {
    /// XOR + popcount on bitpacked vectors
    HammingDistance,
    /// Prefix-only Hamming for Belichtungsmesser Stroke 1
    HammingPrefix,
    /// INT8 dot product (VNNI, tensor cores, NPU)
    DotI8,
    /// FP32 matrix multiply
    MatmulF32,
    /// FP32 elementwise (add, mul, etc.)
    ElementwiseF32,
    /// Reduction (sum, max, etc.)
    ReduceF32,
}

/// The trait every compute backend implements.
///
/// Backends are not required to support every operation.
/// Return `None` from optional methods to signal "not supported,
/// let dispatch try another backend."
pub trait ComputeBackend: Send + Sync {
    /// Backend identification and capabilities.
    fn info(&self) -> &BackendInfo;

    /// Can this backend handle the given operation at the given scale?
    ///
    /// Returns estimated throughput (higher = better) or 0.0 if unsupported.
    /// Dispatch uses this to pick the fastest available backend.
    fn estimate(&self, op: OpHint, batch_size: usize, element_bytes: usize) -> f64;

    // ─── HDC / Belichtungsmesser operations ───

    /// Batch Hamming distance: query against all rows in database.
    ///
    /// Default: returns None (unsupported). CPU backend always implements this.
    fn hamming_batch(
        &self,
        _query: &[u8],
        _database: &[u8],
        _num_rows: usize,
        _row_bytes: usize,
    ) -> Option<Vec<u64>> {
        None
    }

    /// HDR cascade search (3-stroke Belichtungsmesser).
    ///
    /// Default: returns None. CPU backend always implements this.
    /// GPU backends may implement heterogeneous dispatch (GPU Stroke 1 + CPU Strokes 2-3).
    fn hdr_cascade_search(
        &self,
        _query: &[u8],
        _database: &[u8],
        _vec_bytes: usize,
        _num_vectors: usize,
        _threshold: u64,
        _precise_mode: PreciseMode,
    ) -> Option<Vec<HdrResult>> {
        None
    }

    // ─── Dense linear algebra ───

    /// General matrix multiply: C = α·A·B + β·C
    ///
    /// A is m×k, B is k×n, C is m×n. Row-major.
    /// Default: returns None. CPU backend delegates to rustyblas.
    fn sgemm(
        &self,
        _a: &[f32],
        _b: &[f32],
        _c: &mut [f32],
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f32,
        _beta: f32,
    ) -> bool {
        false // not supported
    }

    /// INT8 matrix multiply: C_i32 = A_i8 · B_i8
    ///
    /// For VNNI, tensor cores, NPU. Returns false if unsupported.
    fn gemm_i8(
        &self,
        _a: &[i8],
        _b: &[i8],
        _c: &mut [i32],
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> bool {
        false
    }

    // ─── Elementwise ───

    /// Elementwise binary operation on f32 slices. Returns false if unsupported.
    fn elementwise_f32(
        &self,
        _op: ElementwiseOp,
        _a: &[f32],
        _b: &[f32],
        _out: &mut [f32],
    ) -> bool {
        false
    }
}

/// Elementwise binary operations.
#[derive(Debug, Clone, Copy)]
pub enum ElementwiseOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
}
