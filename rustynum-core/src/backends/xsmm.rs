//! LIBXSMM tail backend — JIT-compiled GEMM via FFI.
//!
//! Feature-gated: only compiled with `--features libxsmm`.
//!
//! ## FFI Surface
//!
//! Two dispatch paths, matching the maxsim-cpu + libCEED analysis:
//!
//! ```text
//! Path 1 — BLAS wrapper (SGEMM):
//!   libxsmm_sgemm(transa, transb, M, N, K, α, A, lda, B, ldb, β, C, ldc)
//!   Auto-JIT internally, BLAS-compatible signature.
//!   Used for batch BF16→f32 widened dot products.
//!
//! Path 2 — Explicit JIT dispatch (NEW — stolen from libCEED pattern):
//!   shape = libxsmm_create_gemm_shape(M, N, K, lda, ldb, ldc, types...)
//!   kernel = libxsmm_dispatch_gemm(shape, flags, prefetch_flags)
//!   kernel(&param)  // direct call, no dispatch overhead
//!   Used for fixed-shape BF16 GEMM when LIBXSMM supports BF16 natively.
//! ```
//!
//! ## Linking
//!
//! build.rs must link:
//! ```text
//! cargo:rustc-link-lib=static=xsmm
//! cargo:rustc-link-lib=static=xsmmext
//! cargo:rustc-link-lib=dl
//! cargo:rustc-link-lib=m
//! cargo:rustc-link-lib=pthread
//! ```
//!
//! ## Safety Contract
//!
//! All FFI calls are confined to this module. The `XsmmBackend` holds only
//! configuration and cached JIT kernel pointers (opaque function pointers).
//! No LIBXSMM heap allocations or mutable state leak through the `TailBackend`
//! trait boundary.
//!
//! ## Thread-Local Buffer Reuse (stolen from maxsim-cpu)
//!
//! The batch path uses thread-local buffers for BF16→f32 widening and GEMM
//! output, avoiding per-call allocations. The pattern:
//! ```text
//! thread_local! { static BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new()); }
//! BUF.with(|b| { b.borrow_mut().resize(needed, 0.0); ... })
//! ```
//! Buffers grow monotonically (high-water mark) and are never shrunk.

use crate::bf16_hamming::{self, BF16Weights};
use crate::tail_backend::{
    BatchTailScore, CompactTailScore, TailBackend, TailScore, compact_score_from_bytes,
};
use std::cell::RefCell;
use std::sync::Once;

// ============================================================================
// FFI bindings — LIBXSMM C API
// ============================================================================

/// LIBXSMM uses `int` for BLAS integers in LP64 mode.
type LibxsmmBlasint = libc::c_int;
/// Bitfield type for GEMM flags and prefetch strategy.
type LibxsmmBitfield = libc::c_uint;

/// LIBXSMM data type enum (from libxsmm_typedefs.h).
/// Only the types we actually use are listed.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub enum LibxsmmDatatype {
    F64 = 0,
    F32 = 1,
    BF16 = 2,
    F16 = 3,
    BF8 = 4,
    HF8 = 5,
    I32 = 6,
    I16 = 8,
    I8 = 12,
    U8 = 13,
}

/// GEMM shape descriptor — describes the kernel dimensions and data types.
/// Passed to `libxsmm_dispatch_gemm` to get a JIT-compiled function pointer.
///
/// From libxsmm_typedefs.h line 734.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct LibxsmmGemmShape {
    pub m: LibxsmmBlasint,
    pub n: LibxsmmBlasint,
    pub k: LibxsmmBlasint,
    pub lda: LibxsmmBlasint,
    pub ldb: LibxsmmBlasint,
    pub ldc: LibxsmmBlasint,
    pub a_in_type: LibxsmmDatatype,
    pub b_in_type: LibxsmmDatatype,
    pub out_type: LibxsmmDatatype,
    pub comp_type: LibxsmmDatatype,
}

/// Data carrier for GEMM operands. Only `primary` is used for basic GEMM.
/// From libxsmm_typedefs.h line 577.
#[repr(C)]
#[derive(Clone)]
pub struct LibxsmmMatrixArg {
    pub primary: *const libc::c_void,
    pub secondary: *const libc::c_void,
    pub tertiary: *const libc::c_void,
    pub quaternary: *const libc::c_void,
    pub quinary: *const libc::c_void,
    pub senary: *const libc::c_void,
}

impl LibxsmmMatrixArg {
    fn from_ptr(ptr: *const libc::c_void) -> Self {
        Self {
            primary: ptr,
            secondary: std::ptr::null(),
            tertiary: std::ptr::null(),
            quaternary: std::ptr::null(),
            quinary: std::ptr::null(),
            senary: std::ptr::null(),
        }
    }

    #[allow(dead_code)]
    fn null() -> Self {
        Self::from_ptr(std::ptr::null())
    }
}

/// Operator state for GEMM. Zeroed for basic GEMM.
/// From libxsmm_typedefs.h line 586.
#[repr(C)]
#[derive(Clone)]
pub struct LibxsmmMatrixOpArg {
    pub primary: *const libc::c_void,
    pub secondary: *const libc::c_void,
    pub tertiary: *const libc::c_void,
    pub quaternary: *const libc::c_void,
}

impl Default for LibxsmmMatrixOpArg {
    fn default() -> Self {
        Self {
            primary: std::ptr::null(),
            secondary: std::ptr::null(),
            tertiary: std::ptr::null(),
            quaternary: std::ptr::null(),
        }
    }
}

/// Call-site argument bundle for JIT kernels.
/// From libxsmm_typedefs.h line 716.
#[repr(C)]
#[derive(Clone)]
pub struct LibxsmmGemmParam {
    pub op: LibxsmmMatrixOpArg,
    pub a: LibxsmmMatrixArg,
    pub b: LibxsmmMatrixArg,
    pub c: LibxsmmMatrixArg,
}

/// JIT-compiled GEMM function pointer type.
/// The kernel takes a single `*const LibxsmmGemmParam` argument.
/// From libxsmm_typedefs.h line 787.
pub type LibxsmmGemmFunction = unsafe extern "C" fn(*const LibxsmmGemmParam);

/// GEMM flag constants (from libxsmm_typedefs.h line 448).
#[allow(dead_code)]
pub mod gemm_flags {
    use super::LibxsmmBitfield;
    /// C += A * B (default accumulate)
    pub const NONE: LibxsmmBitfield = 0;
    /// C = A * B (overwrite, beta=0)
    pub const BETA_0: LibxsmmBitfield = 4;
    /// A matrix in VNNI layout
    pub const VNNI_A: LibxsmmBitfield = 2048;
    /// B matrix in VNNI layout
    pub const VNNI_B: LibxsmmBitfield = 4096;
    /// Unsigned int A
    pub const A_UNSIGNED: LibxsmmBitfield = 256;
    /// Unsigned int B
    pub const B_UNSIGNED: LibxsmmBitfield = 512;
}

extern "C" {
    // Lifecycle
    fn libxsmm_init();
    #[allow(dead_code)]
    fn libxsmm_finalize();

    // Architecture detection
    fn libxsmm_get_target_archid() -> libc::c_int;

    // Shape constructor (fills struct fields, convenience only)
    fn libxsmm_create_gemm_shape(
        m: LibxsmmBlasint,
        n: LibxsmmBlasint,
        k: LibxsmmBlasint,
        lda: LibxsmmBlasint,
        ldb: LibxsmmBlasint,
        ldc: LibxsmmBlasint,
        a_in_type: LibxsmmDatatype,
        b_in_type: LibxsmmDatatype,
        out_type: LibxsmmDatatype,
        comp_type: LibxsmmDatatype,
    ) -> LibxsmmGemmShape;

    // JIT dispatch — returns a function pointer to generated machine code.
    // Returns null if the shape/type combination is unsupported.
    fn libxsmm_dispatch_gemm(
        gemm_shape: LibxsmmGemmShape,
        gemm_flags: LibxsmmBitfield,
        prefetch_flags: LibxsmmBitfield,
    ) -> Option<LibxsmmGemmFunction>;

    // BLAS-compatible SGEMM wrapper (auto-JIT internally)
    fn libxsmm_sgemm(
        transa: *const libc::c_char,
        transb: *const libc::c_char,
        m: *const LibxsmmBlasint,
        n: *const LibxsmmBlasint,
        k: *const LibxsmmBlasint,
        alpha: *const libc::c_float,
        a: *const libc::c_float,
        lda: *const LibxsmmBlasint,
        b: *const libc::c_float,
        ldb: *const LibxsmmBlasint,
        beta: *const libc::c_float,
        c: *mut libc::c_float,
        ldc: *const LibxsmmBlasint,
    );
}

// ============================================================================
// Architecture detection
// ============================================================================

/// LIBXSMM architecture IDs (from libxsmm_cpuid.h).
#[allow(dead_code)]
pub mod arch {
    pub const AVX2: i32 = 1006;
    pub const AVX512_SKX: i32 = 1101;
    pub const AVX512_CLX: i32 = 1102; // Cascade Lake — VNNI
    pub const AVX512_CPX: i32 = 1103; // Cooper Lake — BF16 (VDPBF16PS)
    pub const AVX512_SPR: i32 = 1104; // Sapphire Rapids — AMX
    pub const AVX512_GNR: i32 = 1105; // Granite Rapids
}

/// Get the LIBXSMM-detected target architecture ID.
/// Returns the arch constant (e.g., 1104 for SPR).
/// Must be called after `ensure_init()`.
pub fn target_archid() -> i32 {
    ensure_init();
    // SAFETY: libxsmm_init() has been called via ensure_init(). The function
    // returns an integer arch ID and has no other preconditions.
    unsafe { libxsmm_get_target_archid() }
}

// ============================================================================
// One-time init
// ============================================================================

static INIT: Once = Once::new();

fn ensure_init() {
    INIT.call_once(|| {
        // SAFETY: libxsmm_init() is safe to call from any thread and idempotent.
        // Once guards guarantee it runs exactly once across all threads.
        unsafe { libxsmm_init() };
    });
}

// ============================================================================
// JIT Kernel cache — dispatch once, call many times
// ============================================================================

/// A cached JIT-compiled GEMM kernel for a fixed (M, N, K, types) shape.
///
/// Dispatch cost is paid once at creation. The hot-path call is a single
/// indirect function call to JIT-generated machine code (no hash lookup).
///
/// Follows the libCEED pattern: override only the computation (163 lines),
/// delegate everything else to the reference implementation.
pub struct JitKernel {
    kernel: LibxsmmGemmFunction,
    shape: LibxsmmGemmShape,
}

impl JitKernel {
    /// Try to dispatch a JIT kernel for the given shape.
    /// Returns `None` if LIBXSMM can't generate code for this shape/type combo
    /// (e.g., BF16 on a CPU without VDPBF16PS and no AMX).
    pub fn try_dispatch(shape: LibxsmmGemmShape, flags: LibxsmmBitfield) -> Option<Self> {
        ensure_init();
        // SAFETY: libxsmm_init() has been called. shape is a valid LibxsmmGemmShape
        // struct (repr(C), matching the C API layout). Returns None if unsupported.
        let kernel = unsafe { libxsmm_dispatch_gemm(shape.clone(), flags, 0) }?;
        Some(Self { kernel, shape })
    }

    /// Dispatch an f32 GEMM kernel (C = A * B, beta=0).
    pub fn f32_gemm(m: i32, n: i32, k: i32) -> Option<Self> {
        // SAFETY: libxsmm_create_gemm_shape is a pure struct constructor with no
        // preconditions beyond valid integer arguments. It fills a stack-allocated
        // LibxsmmGemmShape struct.
        let shape = unsafe {
            libxsmm_create_gemm_shape(
                m,
                n,
                k,
                m,  // lda = M (column-major A)
                k,  // ldb = K
                m,  // ldc = M
                LibxsmmDatatype::F32,
                LibxsmmDatatype::F32,
                LibxsmmDatatype::F32,
                LibxsmmDatatype::F32,
            )
        };
        Self::try_dispatch(shape, gemm_flags::BETA_0)
    }

    /// Dispatch a BF16 GEMM kernel (BF16 inputs, f32 output, f32 accumulation).
    /// Requires CPX+ (VDPBF16PS) or SPR+ (AMX TDPBF16PS).
    pub fn bf16_gemm(m: i32, n: i32, k: i32) -> Option<Self> {
        // SAFETY: Same as f32_gemm -- pure struct constructor with no preconditions.
        // Uses LIBXSMM_DATATYPE_BF16 for A/B and F32 for C accumulation.
        let shape = unsafe {
            libxsmm_create_gemm_shape(
                m,
                n,
                k,
                m,
                k,
                m,
                LibxsmmDatatype::BF16,
                LibxsmmDatatype::BF16,
                LibxsmmDatatype::F32,
                LibxsmmDatatype::F32,
            )
        };
        Self::try_dispatch(shape, gemm_flags::BETA_0)
    }

    /// Call the JIT kernel with the given operands.
    ///
    /// # Safety
    /// - `a`, `b`, `c` must point to valid memory of the correct sizes
    ///   matching the shape used at dispatch time.
    /// - `c` must be writable.
    pub unsafe fn call(&self, a: *const libc::c_void, b: *const libc::c_void, c: *mut libc::c_void) {
        let param = LibxsmmGemmParam {
            op: LibxsmmMatrixOpArg::default(),
            a: LibxsmmMatrixArg::from_ptr(a),
            b: LibxsmmMatrixArg::from_ptr(b),
            c: LibxsmmMatrixArg::from_ptr(c as *const libc::c_void),
        };
        (self.kernel)(&param);
    }

    /// The shape this kernel was compiled for.
    pub fn shape(&self) -> &LibxsmmGemmShape {
        &self.shape
    }
}

// JitKernel holds a function pointer to mmap'd code. Safe across threads
// (the code is read-only after JIT compilation, protected by LIBXSMM's
// internal locking during dispatch).
unsafe impl Send for JitKernel {}
unsafe impl Sync for JitKernel {}

// ============================================================================
// Thread-local buffer reuse (stolen from maxsim-cpu)
// ============================================================================

thread_local! {
    /// Reusable buffer for BF16→f32 widened query.
    static QUERY_F32_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    /// Reusable buffer for BF16→f32 widened candidates.
    static CANDIDATES_F32_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
    /// Reusable buffer for GEMM output (similarity scores).
    static SIMILARITY_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
}

/// Widen BF16 bytes to f32 into a pre-allocated buffer.
/// Grows the buffer if needed (high-water mark, never shrinks).
fn widen_bf16_into(bytes: &[u8], buf: &mut Vec<f32>) {
    let n_dims = bytes.len() / 2;
    buf.resize(n_dims, 0.0);
    for (i, chunk) in bytes.chunks_exact(2).enumerate() {
        let bf16 = u16::from_le_bytes([chunk[0], chunk[1]]);
        buf[i] = f32::from_bits((bf16 as u32) << 16);
    }
}

/// Widen N candidate BF16 byte slices into a contiguous f32 buffer.
fn widen_candidates_into(
    candidate_slices: &[u8],
    n_candidates: usize,
    stride: usize,
    buf: &mut Vec<f32>,
) {
    let n_dims = stride / 2;
    buf.resize(n_candidates * n_dims, 0.0);
    for i in 0..n_candidates {
        let offset = i * stride;
        let base = i * n_dims;
        for (j, chunk) in candidate_slices[offset..offset + stride]
            .chunks_exact(2)
            .enumerate()
        {
            let bf16 = u16::from_le_bytes([chunk[0], chunk[1]]);
            buf[base + j] = f32::from_bits((bf16 as u32) << 16);
        }
    }
}

// ============================================================================
// Backend
// ============================================================================

/// LIBXSMM-backed tail scorer.
///
/// Single scoring falls back to POPCNT (LIBXSMM doesn't help for N=1).
/// Batch scoring uses SGEMM for fused GEMM + reduction.
///
/// Holds configuration + cached JIT kernel pointer. No LIBXSMM heap state.
pub struct XsmmBackend {
    /// Fallback for single-candidate scoring.
    popcnt_fn: bf16_hamming::BF16HammingFn,
    /// LIBXSMM architecture ID (for capability reporting).
    archid: i32,
}

unsafe impl Send for XsmmBackend {}
unsafe impl Sync for XsmmBackend {}

impl XsmmBackend {
    /// Try to create an XSMM backend.
    ///
    /// Returns `None` if LIBXSMM initialization fails.
    /// In practice this only fails if the shared library isn't found.
    pub fn try_new() -> Option<Self> {
        ensure_init();
        Some(Self {
            popcnt_fn: bf16_hamming::select_bf16_hamming_fn(),
            archid: unsafe { libxsmm_get_target_archid() },
        })
    }

    /// Returns true if this CPU supports native BF16 GEMM (CPX+ / SPR+).
    pub fn has_native_bf16(&self) -> bool {
        self.archid >= arch::AVX512_CPX
    }

    /// Returns true if this CPU supports AMX tiles (SPR+).
    pub fn has_amx(&self) -> bool {
        self.archid >= arch::AVX512_SPR
    }

    /// LIBXSMM-detected architecture ID.
    pub fn archid(&self) -> i32 {
        self.archid
    }

    /// Run SGEMM using thread-local buffers (no heap allocation in hot path).
    ///
    /// Computes: similarities[i] = dot(query_f32, candidate_f32[i])
    /// Uses the maxsim-cpu pattern: column-major BLAS with transposed A.
    fn sgemm_batch_similarities(
        &self,
        query_bytes: &[u8],
        candidate_slices: &[u8],
        n_candidates: usize,
    ) -> Vec<f32> {
        let stride = query_bytes.len();
        let n_dims = stride / 2;

        QUERY_F32_BUF.with(|qbuf| {
            let mut qbuf = qbuf.borrow_mut();
            widen_bf16_into(query_bytes, &mut qbuf);

            CANDIDATES_F32_BUF.with(|cbuf| {
                let mut cbuf = cbuf.borrow_mut();
                widen_candidates_into(candidate_slices, n_candidates, stride, &mut cbuf);

                SIMILARITY_BUF.with(|sbuf| {
                    let mut sbuf = sbuf.borrow_mut();
                    sbuf.resize(n_candidates, 0.0);

                    // SGEMM: C = Candidates^T × Query
                    // transa='T', transb='N', M=N_candidates, N=1, K=dim
                    unsafe {
                        let transa = b'T' as libc::c_char;
                        let transb = b'N' as libc::c_char;
                        let m = n_candidates as LibxsmmBlasint;
                        let n = 1 as LibxsmmBlasint;
                        let k = n_dims as LibxsmmBlasint;
                        let alpha = 1.0f32;
                        let beta = 0.0f32;
                        let lda = n_dims as LibxsmmBlasint;
                        let ldb = n_dims as LibxsmmBlasint;
                        let ldc = n_candidates as LibxsmmBlasint;

                        libxsmm_sgemm(
                            &transa,
                            &transb,
                            &m,
                            &n,
                            &k,
                            &alpha,
                            cbuf.as_ptr(),
                            &lda,
                            qbuf.as_ptr(),
                            &ldb,
                            &beta,
                            sbuf.as_mut_ptr(),
                            &ldc,
                        );
                    }

                    sbuf.clone()
                })
            })
        })
    }
}

impl TailBackend for XsmmBackend {
    fn name(&self) -> &'static str {
        "libxsmm"
    }

    /// Single candidate: use POPCNT (LIBXSMM overhead not worth it for N=1).
    fn score(
        &self,
        query_bytes: &[u8],
        candidate_bytes: &[u8],
        weights: &BF16Weights,
    ) -> TailScore {
        let bf16_distance = (self.popcnt_fn)(query_bytes, candidate_bytes, weights);
        let structural_diff = bf16_hamming::structural_diff(query_bytes, candidate_bytes);
        TailScore {
            bf16_distance,
            structural_diff,
        }
    }

    /// Batch scoring via LIBXSMM SGEMM with thread-local buffer reuse.
    ///
    /// 1. Widen BF16 bytes → f32 (into thread-local buffers, zero allocation)
    /// 2. SGEMM: Q × Candidates^T → similarity vector [N]
    /// 3. Compute exact BF16 structured distances (still needed for learning signal)
    ///
    /// The GEMM similarity is available for future pre-sort optimization.
    /// Currently we compute exact BF16 distances for all candidates because
    /// the learning signal needs per-dimension sign/exp/man decomposition.
    fn score_batch(
        &self,
        query_bytes: &[u8],
        candidate_slices: &[u8],
        n_candidates: usize,
        weights: &BF16Weights,
    ) -> BatchTailScore {
        let stride = query_bytes.len();

        // For small batches, fall back to per-candidate scoring
        if n_candidates < 8 {
            let mut distances = Vec::with_capacity(n_candidates);
            let mut diffs = Vec::with_capacity(n_candidates);
            for i in 0..n_candidates {
                let offset = i * stride;
                let cand = &candidate_slices[offset..offset + stride];
                let s = self.score(query_bytes, cand, weights);
                distances.push(s.bf16_distance);
                diffs.push(s.structural_diff);
            }
            return BatchTailScore { distances, diffs };
        }

        // Compute GEMM similarities (pre-sort hint, available for future use)
        let _similarities = self.sgemm_batch_similarities(query_bytes, candidate_slices, n_candidates);

        // Exact BF16 structured distances for learning signal
        let mut distances = Vec::with_capacity(n_candidates);
        let mut diffs = Vec::with_capacity(n_candidates);

        for i in 0..n_candidates {
            let offset = i * stride;
            let cand = &candidate_slices[offset..offset + stride];
            let d = (self.popcnt_fn)(query_bytes, cand, weights);
            let diff = bf16_hamming::structural_diff(query_bytes, cand);
            distances.push(d);
            diffs.push(diff);
        }

        BatchTailScore { distances, diffs }
    }

    fn supports_batch(&self) -> bool {
        true
    }

    /// Compact single scoring: distance + counters, no structural diff.
    fn score_compact(
        &self,
        query_bytes: &[u8],
        candidate_bytes: &[u8],
        weights: &BF16Weights,
    ) -> CompactTailScore {
        compact_score_from_bytes(query_bytes, candidate_bytes, weights)
    }

    /// Compact batch: distance + counters only, skips structural diff entirely.
    fn score_batch_compact(
        &self,
        query_bytes: &[u8],
        candidate_slices: &[u8],
        n_candidates: usize,
        weights: &BF16Weights,
    ) -> Vec<CompactTailScore> {
        let stride = query_bytes.len();
        let mut results = Vec::with_capacity(n_candidates);
        for i in 0..n_candidates {
            let offset = i * stride;
            let cand = &candidate_slices[offset..offset + stride];
            results.push(compact_score_from_bytes(query_bytes, cand, weights));
        }
        results
    }
}

impl Drop for XsmmBackend {
    fn drop(&mut self) {
        // libxsmm_finalize is optional and safe to call multiple times.
        // We don't call it here because other backends may still be alive.
        // It's called automatically at process exit by LIBXSMM's atexit handler.
    }
}
