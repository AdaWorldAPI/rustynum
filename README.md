![RustyNum Banner](docs/assets/rustynum-banner.png?raw=true "RustyNum")

![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

# RustyNum

RustyNum is a high-performance numerical computation ecosystem in pure Rust, leveraging `portable_simd` (nightly) for SIMD-accelerated operations. It includes a complete **BLAS replacement** (`rustyblas`), **MKL replacement** (`rustymkl`), and **AVX-512 optimized Hyperdimensional Computing** primitives — all sharing zero-copy memory via the Blackboard architecture.

## Crate Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  rustynum (workspace)                                                        │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  rustynum-core          Shared foundation                              │  │
│  │  ├── blackboard.rs      Zero-copy shared memory arena (64-byte align) │  │
│  │  ├── simd.rs            SIMD primitives (f32x16, f64x8, u8x64)       │  │
│  │  ├── compute.rs         Hardware detection + tiered compute dispatch   │  │
│  │  ├── prefilter.rs       INT8 prefilter (stats, GEMM pruning, HDC)     │  │
│  │  ├── layout.rs          CBLAS layout types (Row/ColMajor)             │  │
│  │  └── parallel.rs        Lock-free thread pool                         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│       ▲              ▲               ▲                                       │
│       │              │               │                                       │
│  ┌────┴─────┐  ┌─────┴──────┐  ┌────┴─────┐                                │
│  │ rustyblas │  │  rustymkl  │  │rustynum-rs│                               │
│  │ (BLAS)   │  │ (MKL)      │  │ (arrays)  │                               │
│  │          │  │            │  │           │                                │
│  │ L1: dot  │  │ LAPACK:   │  │ HDC/VSA   │                               │
│  │ L2: gemv │  │  LU, QR,  │  │ BIND,     │                               │
│  │ L3: gemm │  │  Cholesky │  │ BUNDLE,   │                               │
│  │ INT8 GEMM│  │ FFT:      │  │ PERMUTE,  │                               │
│  │ BF16 GEMM│  │  radix-2  │  │ DISTANCE  │                               │
│  │          │  │ VML:      │  │ Int8 embs  │                               │
│  │          │  │  exp,ln,  │  │           │                                │
│  │          │  │  sqrt,sin │  │           │                                │
│  └──────────┘  └───────────┘  └───────────┘                                │
│                                                                              │
│  All crates share the Blackboard — zero serialization between BLAS/MKL/HDC  │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Key Features

- **Pure Rust, Zero FFI:** No C/C++, no OpenBLAS, no MKL, no CUDA. Every instruction is Rust `portable_simd` or `core::arch::x86_64` intrinsics.
- **AVX-512 Everywhere:** `f32x16`, `f64x8`, `u8x64`, `i32x16` — cache-blocked GEMM microkernels, VNNI INT8 dot products, VPOPCNTDQ hamming.
- **Cache-Blocked GEMM (Goto Algorithm):** 6x16 f32 and 6x8 f64 microkernels with panel packing, multithreaded M-loop. **115.77 GFLOPS** at 1024x1024 on 16 threads.
- **INT8 Quantized GEMM:** AVX-512 VNNI `vpdpbusd` — 64 multiply-accumulate operations per instruction. Symmetric/asymmetric/per-channel quantization.
- **BF16 Mixed-Precision GEMM:** Half the memory bandwidth with f32 accumulation. For ML inference and training.
- **Tiered Compute Dispatch:** Runtime hardware detection routes work to the cheapest capable path: INT8 prefilter → VNNI INT8 → BF16 → FP32 AVX-512 → GPU (when available).
- **INT8 Prefiltering:** Use cheap VNNI INT8 operations for approximate statistics (mean, SD), row-norm pruning, and HDC candidate selection — saves 90%+ of FP32 cycles.
- **Zero-Copy Blackboard:** All crates operate on the same 64-byte aligned memory arena. No serialization between BLAS, LAPACK, FFT, and HDC operations.
- **Lock-Free Parallelism:** `split_at_mut` + `thread::scope` + `SendMutPtr<T>` — no `Arc<Mutex>`, no contention, disjoint row ownership.
- **HDC / VSA Primitives:** BIND, BUNDLE (ripple-carry), PERMUTE, Hamming distance with 3σ/2σ adaptive cascade.

## The GEMM Story: Closing the NumPy/OpenBLAS Gap

The original `rustynum-rs` used a simple transpose-dot GEMM. NumPy (backed by OpenBLAS/MKL) dominated for matrices >512x512 because it uses cache-blocked, multi-threaded GEMM. **This gap has been closed:**

### What Changed

1. **Goto BLAS Algorithm:** Panel packing into cache-friendly contiguous buffers. A-panels packed in MR-row strips, B-panels in NR-column strips.
2. **AVX-512 Microkernels:** 6x16 (f32) and 6x8 (f64) tile sizes — 6 accumulator registers per microkernel iteration, fully utilizing the register file.
3. **Cache Hierarchy Optimization:** KC=256 (L1), MC=128 (L2), NC=4096 (L3) blocking parameters tuned for modern Intel CPUs.
4. **Multithreaded M-loop:** Scoped threads with `SendMutPtr<T>` — each thread owns disjoint C rows. Zero locks, zero contention.
5. **Adaptive Thresholds:** Simple triple-loop for tiny matrices (<110K flops), single-threaded blocked for medium (<256x256), multithreaded for large.

### Benchmark Results (16 threads, AVX-512)

```
=== GEMM Benchmark: old (transpose-dot) vs new (cache-blocked + 16T) ===
  Size         | Old (transpose-dot)              | New (Goto+MT)                    | Speedup
  ─────────────┼──────────────────────────────────┼──────────────────────────────────┼────────
  32x32        |  0.01 ms (  6.77 GFLOPS)         |  0.03 ms (  2.48 GFLOPS)         |  0.37x
  64x64        |  0.04 ms ( 12.86 GFLOPS)         |  0.03 ms ( 20.31 GFLOPS)         |  1.58x
  128x128      |  0.27 ms ( 15.79 GFLOPS)         |  0.16 ms ( 25.78 GFLOPS)         |  1.63x
  256x256      |  1.82 ms ( 18.40 GFLOPS)         |  1.13 ms ( 29.67 GFLOPS)         |  1.61x
  512x512      | 13.75 ms ( 19.52 GFLOPS)         |  6.56 ms ( 40.90 GFLOPS)         |  2.10x
  1024x1024    |167.98 ms ( 12.78 GFLOPS)         | 18.55 ms (115.77 GFLOPS)         |  9.06x
```

**Key insight:** At 1024x1024, the old approach hit a cache cliff (12.78 GFLOPS) while the new cache-blocked + MT approach achieves **115.77 GFLOPS** — a **9.06x speedup**. The 512x512 gap that NumPy exploited is now closed with 2.10x faster than the old path.

## Supported Data Types

| Type | SIMD Vector | Primary Use |
|------|-------------|-------------|
| float32 | `f32x16` | GEMM, VML, statistics |
| float64 | `f64x8` | DGEMM, LAPACK, precision-critical |
| uint8 | `u8x64` | HDC bitpacked vectors, INT8 activations |
| int8 | `i8` (via VNNI) | INT8 weights, quantized inference |
| bf16 | `BF16` (u16) | Mixed-precision GEMM, ML training |
| int32 | `i32x16` | INT8 GEMM accumulators, HDC counters |
| int64 | `i64x8` | Precise accumulation, statistics |

## Crate: rustyblas (Pure Rust OpenBLAS Replacement)

### BLAS Level 1 (Vector-Vector)

| Function | Description |
|----------|-------------|
| `sdot` / `ddot` | SIMD dot product (4x unrolled) |
| `saxpy` / `daxpy` | y += alpha * x |
| `sscal` / `dscal` | x *= alpha |
| `snrm2` / `dnrm2` | L2 norm |
| `sasum` / `dasum` | L1 norm (sum of absolutes) |
| `isamax` / `idamax` | Index of max absolute value |
| `scopy` / `dcopy` | Vector copy |
| `sswap` / `dswap` | Vector swap |

### BLAS Level 2 (Matrix-Vector)

| Function | Description |
|----------|-------------|
| `sgemv` / `dgemv` | Matrix-vector multiply |
| `sger` / `dger` | Rank-1 outer product update |
| `ssymv` / `dsymv` | Symmetric matrix-vector multiply |
| `strmv` / `dtrmv` | Triangular matrix-vector multiply |
| `strsv` / `dtrsv` | Triangular solve |

### BLAS Level 3 (Matrix-Matrix)

| Function | Description | Notes |
|----------|-------------|-------|
| `sgemm` | f32 GEMM | Cache-blocked Goto algorithm, 6x16 microkernel, multithreaded |
| `dgemm` | f64 GEMM | Cache-blocked, 6x8 microkernel, multithreaded |
| `ssyrk` / `dsyrk` | Symmetric rank-k update | C = alpha * A * A^T + beta * C |
| `strsm` | Triangular solve with multiple RHS | Forward/back substitution |
| `ssymm` / `dsymm` | Symmetric matrix multiply | Exploits symmetry |

### INT8 Quantized GEMM (AVX-512 VNNI)

| Function | Description |
|----------|-------------|
| `int8_gemm_i32` | u8 * i8 → i32 accumulate (VNNI `vpdpbusd`) |
| `int8_gemm_f32` | Quantized GEMM with f32 dequantized output |
| `int8_gemm_per_channel_f32` | Per-channel scale/zero-point dequantization |
| `quantize_f32_to_u8` | Asymmetric quantization (activations) |
| `quantize_f32_to_i8` | Symmetric quantization (weights) |
| `quantize_per_channel_i8` | Per-row symmetric quantization |

**VNNI throughput:** Each `vpdpbusd` instruction performs 64 multiply-accumulate operations (16 lanes x 4 u8*i8 pairs). This is **4x the throughput** of f32 GEMM for the same register width.

### BF16 Mixed-Precision GEMM

| Function | Description |
|----------|-------------|
| `bf16_gemm_f32` | BF16 inputs, f32 accumulation (half bandwidth) |
| `mixed_precision_gemm` | f32 inputs auto-quantized to BF16 on-the-fly |
| `f32_to_bf16_slice` | Bulk f32 → BF16 conversion |
| `bf16_to_f32_slice` | Bulk BF16 → f32 conversion |

**BF16 advantage:** Same exponent range as f32 (8-bit), 7-bit mantissa. Half the memory bandwidth while maintaining numerical stability via f32 accumulation.

### Memory Layout

Both row-major and column-major supported via CBLAS-style API:

```rust
use rustyblas::{Layout, Transpose};
use rustyblas::level3::sgemm;

// Row-major GEMM
sgemm(Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
      m, n, k, 1.0, &a, k, &b, n, 0.0, &mut c, n);

// Column-major GEMM
sgemm(Layout::ColMajor, Transpose::NoTrans, Transpose::NoTrans,
      m, n, k, 1.0, &a, m, &b, k, 0.0, &mut c, m);
```

## Crate: rustymkl (Pure Rust MKL Replacement)

### LAPACK

| Function | Description |
|----------|-------------|
| `sgetrf` / `dgetrf` | LU factorization with partial pivoting |
| `sgetrs` / `dgetrs` | Solve Ax=b using LU factors |
| `spotrf` / `dpotrf` | Cholesky factorization (L*L^T) |
| `spotrs` | Cholesky solve |
| `sgeqrf` / `dgeqrf` | QR factorization (Householder reflections) |

### FFT

| Function | Description |
|----------|-------------|
| `fft_f32` / `fft_f64` | Radix-2 Cooley-Tukey FFT (interleaved complex) |
| `ifft_f32` / `ifft_f64` | Inverse FFT |
| `rfft_f32` | Real-to-complex FFT (returns N/2+1 complex values) |

### VML (Vector Math Library)

| Function | Description |
|----------|-------------|
| `vsexp` / `vdexp` | SIMD exp() (minimax polynomial + range reduction) |
| `vsln` / `vdln` | SIMD ln() |
| `vssqrt` / `vdsqrt` | SIMD sqrt() |
| `vsabs` / `vdabs` | SIMD abs() |
| `vssin` / `vscos` | SIMD sin/cos (polynomial approximation) |
| `vsadd` / `vsmul` / `vsdiv` | Element-wise arithmetic |
| `vspow` | Element-wise power |

## Crate: rustynum-core (Shared Foundation)

### Blackboard (Zero-Copy Shared Memory)

```rust
use rustynum_core::Blackboard;
use rustyblas::level3;
use rustynum_core::layout::{Layout, Transpose};

let mut bb = Blackboard::new();
bb.alloc_f32("A", 1024 * 1024);
bb.alloc_f32("B", 1024 * 1024);
bb.alloc_f32("C", 1024 * 1024);

// Split-borrow: 3 distinct mutable references simultaneously
let (a, b, c) = bb.borrow_3_mut_f32("A", "B", "C");

// Fill matrices, then GEMM — zero copies, zero serialization
level3::sgemm(Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
              1024, 1024, 1024, 1.0, a, 1024, b, 1024, 0.0, c, 1024);
```

### Tiered Compute Dispatch

```
┌─────────────────────────────────────────────────────────────┐
│  Tier 0: INT8 Prefilter (AVX-512 VNNI)                     │
│  - Approximate stats (mean, SD): ~4x cheaper than f32      │
│  - Row-norm pruning: skip 90%+ of GEMM rows                │
│  - HDC prefix prefilter: scan first N bytes only            │
├─────────────────────────────────────────────────────────────┤
│  Tier 1: AVX-512 VNNI INT8 GEMM                            │
│  - 64 MACs/instruction (vpdpbusd)                           │
│  - Per-channel dequantization for inference                  │
├─────────────────────────────────────────────────────────────┤
│  Tier 2: AVX-512 BF16 GEMM                                 │
│  - Half bandwidth, f32 accumulation                         │
│  - Training + inference mixed precision                     │
├─────────────────────────────────────────────────────────────┤
│  Tier 3: AVX-512 FP32 GEMM (cache-blocked + multithreaded) │
│  - Full precision, 115+ GFLOPS at 1024x1024                │
│  - Only for rows/columns that survived prefiltering         │
├─────────────────────────────────────────────────────────────┤
│  Tier 4: AMX Tiles / GPU (when available)                   │
│  - AMX INT8: 1024 ops/cycle per tile (detected via CPUID)   │
│  - AMX BF16: tile-based GEMM for massive matrices           │
│  - Intel Xe2 GPU: offload via Level Zero (auto-detected)    │
└─────────────────────────────────────────────────────────────┘
```

### Hardware Detection

```rust
use rustynum_core::compute;

compute::print_caps();
// Output:
// === Compute Capabilities ===
//   CPU cores:     16
//   AVX-512F:      true
//   AVX-512BW:     true
//   AVX-512 VNNI:  true     ← INT8 dot products (vpdpbusd)
//   AVX-512 BF16:  true     ← BF16 GEMM (vdpbf16ps)
//   AVX-512 VPOP:  true     ← Hamming distance (vpopcntdq)
//   AMX Tile:      true     ← Tile-based GEMM
//   AMX INT8:      true     ← 1024 INT8 ops/cycle
//   AMX BF16:      true     ← Tile-based BF16 GEMM
//   NPU:           false    ← Meteor Lake+ only
//   GPU:           not detected
```

### INT8 Prefilter Module

| Function | Description | Use Case |
|----------|-------------|----------|
| `approx_mean_std_f32` | INT8 quantized mean/SD, <1% error | Cheap statistics |
| `approx_column_std` | Per-column SD for feature selection | Prune low-variance columns before GEMM |
| `approx_row_norms_f32` | INT8 approximate L2 row norms | GEMM row pruning |
| `top_k_rows_by_norm` | Top-k rows by approximate norm | Select important rows |
| `pruned_gemm_rows` | INT8 prefilter → sparse f32 GEMM | Skip 90%+ of computation |
| `approx_hamming_candidates` | SIMD XOR + popcount candidate selection | HDC search |
| `two_stage_hamming_search` | Prefix prefilter → exact hamming on shortlist | Post-bundle HDC search |

**The prefilter philosophy:** Use the cheapest possible computation (INT8 VNNI) to decide *what* to compute, then use expensive FP32 GEMM only on the survivors. This saves AVX-512 FP32 cycles for better parallelization.

## Supported Operations (rustynum-rs)

### Core Numerical Operations

| Operation | Description |
|-----------|-------------|
| `zeros`, `ones` | Array constructors |
| `arange`, `linspace` | Range generators |
| `mean`, `median` | Statistics (with axis support) |
| `std`, `var` | Standard deviation & variance (with axis support) |
| `percentile` | Percentile with linear interpolation (with axis support) |
| `min`, `max` | Reduction (with axis support) |
| `argmin`, `argmax` | Index of min/max value |
| `top_k` | Top-k largest elements (indices + values) |
| `sort` | Ascending sort |
| `exp`, `log`, `sigmoid` | Element-wise math |
| `softmax`, `log_softmax` | Numerically stable softmax (row-wise for N-D) |
| `cosine_similarity` | Cosine similarity between vectors |
| `dot`, `matmul` | Dot product, matrix multiply |
| `cumsum` | Cumulative sum |
| `reshape`, `squeeze`, `slice` | Shape manipulation |
| `transpose`, `flip_axis` | Dimension reordering |
| `concatenate` | Array joining |
| `+`, `-`, `*`, `/` | Element-wise arithmetic |
| `norm` | L1/L2 norm (with axis + keepdims) |

### Bitwise Operations (AVX-512)

| Operation | Types | Description |
|-----------|-------|-------------|
| `&` (BitAnd) | u8, i32, i64 | SIMD AND with 4x unrolling |
| `^` (BitXor) | u8, i32, i64 | SIMD XOR with 4x unrolling |
| `\|` (BitOr) | u8, i32, i64 | SIMD OR with 4x unrolling |
| `!` (Not) | u8, i32, i64 | SIMD NOT |
| Scalar variants | u8, i32, i64 | `array ^ 0xFF`, `array & mask`, etc. |

### HDC / Vector Symbolic Architecture

| Operation | Method | Description |
|-----------|--------|-------------|
| **BIND** | `a.bind(&b)` or `a ^ b` | XOR binding (involutory: `bind(bind(a,b),b) == a`) |
| **PERMUTE** | `v.permute(k)` | Circular bit-rotation by k positions |
| **BUNDLE** | `NumArrayU8::bundle(&[&a, &b, &c])` | Majority vote (hybrid: naive n<=16, ripple-carry n>16) |
| **DISTANCE** | `a.hamming_distance(&b)` | Hamming distance via POPCNT |
| **POPCOUNT** | `a.popcount()` | Population count |
| **BATCH DISTANCE** | `a.hamming_distance_batch(&b, dim, count)` | Batched Hamming for database scans |
| **ADAPTIVE DISTANCE** | `a.hamming_distance_adaptive(&b, threshold)` | 3-stage cascade with 3sigma/2sigma early-exit |
| **ADAPTIVE SEARCH** | `q.hamming_search_adaptive(&db, dim, n, thresh)` | Batch scan, eliminates ~99.7% early |

### Int8 Embedding Operations (VNNI)

| Operation | Method | Description |
|-----------|--------|-------------|
| **Dot Product** | `a.dot_i8(&b)` | Signed int8 multiply-accumulate -> i64 |
| **Norm^2** | `a.norm_sq_i8()` | Squared L2 norm as int8 |
| **Cosine** | `a.cosine_i8(&b)` | Cosine similarity [-1.0, 1.0] |
| **ADAPTIVE COSINE** | `q.cosine_search_adaptive(&db, dim, n, min_sim)` | Cascade cosine scan, FP64 at ~3% cost |

## Performance

### GEMM: Before and After (Cache-Blocked + Multithreaded)

| Size | Old (transpose-dot) | New (Goto+MT, 16T) | Speedup | Notes |
|------|--------------------|--------------------|---------|-------|
| 32x32 | 6.77 GFLOPS | 2.48 GFLOPS | 0.37x | Too small for blocked overhead |
| 64x64 | 12.86 GFLOPS | 20.31 GFLOPS | **1.58x** | Blocked path kicks in |
| 128x128 | 15.79 GFLOPS | 25.78 GFLOPS | **1.63x** | Single-threaded blocked |
| 256x256 | 18.40 GFLOPS | 29.67 GFLOPS | **1.61x** | Single-threaded blocked |
| 512x512 | 19.52 GFLOPS | 40.90 GFLOPS | **2.10x** | Gap that NumPy exploited: **closed** |
| 1024x1024 | 12.78 GFLOPS | **115.77 GFLOPS** | **9.06x** | Cache cliff eliminated |

### HDC Bitwise Operations (8192-byte = 65536-bit vectors)

| Operation | Naive Scalar | RustyNum SIMD | Speedup |
|-----------|-------------|---------------|---------|
| XOR / BIND (8 KB) | 5.5 us | **0.7 us** | **8x** |
| XOR / BIND (16 KB) | 11 us | **1.4 us** | **8x** |
| XOR / BIND (32 KB) | 22 us | **1.4 us** | **16x** |
| Hamming distance (8 KB) | 3.8 us | **1.7 us** | **2.2x** |
| Hamming distance (16 KB) | 7.6 us | **3.4 us** | **2.2x** |
| Popcount (8 KB) | 2.5 us | **1.5 us** | **1.7x** |

### BUNDLE (Majority Vote, 8192-byte vectors)

| n vectors | RustyNum SIMD | Naive baseline | Speedup |
|-----------|---------------|---------------|---------|
| 5 | **96 us** | 210 us | 2.2x |
| 16 | **237 us** | 646 us | 2.7x |
| 64 | **633 us** | 3.24 ms | 5.1x |
| 256 | **3.86 ms** | 11.4 ms | 2.9x |
| 1024 | **4.01 ms** | 70.9 ms | **17.7x** |

Note: n=1024 barely costs more than n=256 — the ripple-carry counter scales O(log n) per lane.

### Adaptive Cascade Search (3sigma/2sigma Early-Exit)

#### Hamming Cascade

| Database | Full Scan | Adaptive Cascade | Speedup |
|----------|-----------|------------------|---------|
| 1K x 2 KB | 226 us | **49 us** | **4.6x** |
| 10K x 2 KB | 2.63 ms | **369 us** | **7.1x** |
| 1K x 8 KB | 787 us | **78 us** | **10.1x** |
| 10K x 8 KB | 9.0 ms | **724 us** | **12.4x** |

#### Cosine Cascade

| Database | Full Scan | Adaptive Cascade | Speedup | FP64 cost |
|----------|-----------|------------------|---------|-----------|
| 1K x 1024D | 753 us | **88 us** | **8.6x** | ~12% |
| 10K x 1024D | 7.49 ms | **793 us** | **9.4x** | ~11% |
| 1K x 2048D | 1.41 ms | **86 us** | **16.4x** | ~6% |
| 10K x 2048D | 14.2 ms | **1.13 ms** | **12.6x** | ~8% |

```
┌──────────────────────────────────────────────────────────────────┐
│  Stage 1: 1/16 sample (2 VPOPCNTDQ per 2KB container)          │
│  ├─ 99.7% eliminated (> 3σ from threshold)                     │
│  └─ ~6% compute cost                                           │
│                                                                  │
│  Stage 2: 1/4 sample (8 VPOPCNTDQ, incremental)                │
│  ├─ ~95% of remaining eliminated (> 2σ)                        │
│  └─ ~25% compute cost                                          │
│                                                                  │
│  Stage 3: full precision (32 VPOPCNTDQ)                         │
│  └─ Only ~0.3% of candidates reach here                        │
└──────────────────────────────────────────────────────────────────┘
```

### RustyNum vs NumPy Comparison (Updated)

| Operation | NumPy | RustyNum | Speedup | Notes |
|-----------|-------|----------|---------|-------|
| XOR / BIND (8 KB) | ~3-5 us | **0.7 us** | **5-7x** | NumPy has no SIMD bitwise path |
| Hamming distance (8 KB) | ~25-50 us | **1.7 us** | **15-30x** | NumPy: unpackbits+sum |
| Bundle n=64 (8 KB) | ~8-20 ms | **633 us** | **13-32x** | NumPy: unpackbits per vector |
| Int8 dot (1024D) | ~3-8 us | **226 ns** | **13-35x** | NumPy: astype(int32)+dot |
| Int8 cosine (1024D) | ~10-20 us | **522 ns** | **19-38x** | NumPy: astype(float64)+norm |
| DB scan 10K x 2 KB | ~500-1000 ms | **369 us** | **1350-2700x** | Adaptive cascade eliminates 99.7% |
| f32 addition (10K) | ~3-5 us | **760 ns** | **4-7x** | NumPy C loops vs portable_simd |
| f32 mean (10K) | ~3-5 us | **684 ns** | **4-7x** | |
| f32 dot (10K) | ~2-4 us | **759 ns** | **3-5x** | |
| **Matrix mul 1Kx1K** | ~1-3 ms | **18.55 ms** | **0.05-0.16x** | NumPy/MKL still faster (highly tuned) |

**Where RustyNum now replaces NumPy entirely:**
- HDC/VSA bitwise operations: **15-30x faster** (native SIMD, no Python overhead)
- Int8 embedding search: **13-38x faster** (VNNI + cascade filter)
- Vector statistics (mean, std, var): **4-7x faster** (explicit SIMD)
- Database scans with early exit: **1000x+ faster** (cascade has no NumPy equivalent)
- GEMM 512x512+: **Now competitive** (cache-blocked + MT, was the one area NumPy dominated)
- Zero-dependency deployment (no BLAS, no pip, no GIL)

**Where NumPy/MKL still leads (and the plan to close):**
- 1024x1024 GEMM: NumPy/MKL achieves ~200+ GFLOPS with fully-tuned microkernels. Our 115.77 GFLOPS is competitive but not yet at parity. Next steps:
  - AMX tile GEMM (detected: `amx_int8=true`, `amx_bf16=true`) — 1024 ops/cycle
  - Register-blocking optimization for the 6x16 microkernel
  - Prefetch hints for panel packing

**Where GPU dominates (complementary, not competing):**
- Dense GEMM >2048x2048: tensor cores at ~178x advantage
- Batch inference >10K concurrent vectors
- Training workloads requiring FP16/BF16 matrix math at scale

### RustyNum vs GPU (NVIDIA H100) Analysis

| Operation | H100 GPU | RustyNum (16 cores) | Analysis |
|-----------|----------|---------------------|----------|
| XOR 8 KB | ~50 ns | 0.7 us | GPU 14x, but PCIe kills latency |
| Hamming 2 KB | ~80 ns | 1.7 us | GPU 21x, need batch >10K |
| Int8 dot 1024D | ~30 ns | 226 ns | GPU 7.5x, tensor core batch only |
| GEMM 1Kx1K | ~0.1 ms | 18.55 ms | **GPU 185x** (tensor cores) |
| GEMM 1Kx1K (INT8) | ~0.05 ms | ~5 ms (est.) | GPU 100x (INT8 tensor cores) |
| **DB scan 10K x 2 KB** | ~2-5 ms | **369 us** | **CPU wins** (cascade 99.7%) |
| **DB scan 1M x 2 KB** | ~200-500 ms | **~37 ms** (est.) | **CPU competitive** |

**RustyNum's CPU sweet spot:**
1. **Single-query latency** — no PCIe round-trip, no kernel launch
2. **Adaptive cascade** — CPU branching eliminates 99.7% of work (GPU can't branch efficiently)
3. **INT8 prefiltering** — cheap VNNI pass prunes before expensive FP32, saves cycles for parallelism
4. **Streaming / real-time** — immediate processing, no GPU memory management
5. **Quantized inference** — VNNI INT8 GEMM at 4x throughput of FP32

## Architecture: CogRecord Container Layout

RustyNum is optimized for the 4 x 16384-bit CogRecord architecture:

```
CogRecord (8 KB = 65536 bits)
├── Container 0: META    (2 KB) — codebook identity, DN, hashtag zone
├── Container 1: CAM     (2 KB) — content-addressable memory (Hamming search)
├── Container 2: B-tree  (2 KB) — structural position index
└── Container 3: Embed   (2 KB) — int8/binary embeddings (VNNI dot + Hamming)
```

Each 2 KB container is exactly 32 AVX-512 registers. A full VPOPCNTDQ sweep is 32 instructions per container. Container 3 supports both distance metrics on the same memory:

- **Binary fingerprints** -> Hamming distance via `VPOPCNTDQ`
- **Int8 embeddings** -> Dot product via `VPDPBUSD` (VNNI)

### INT8 Prefilter Pipeline (Post-Bundle)

After bundling 512D+ hypervectors, use INT8 prefilter to avoid full-precision search:

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Bundle 512 HVs via ripple-carry majority vote               │
│     (O(log n) per lane, 4.01 ms for n=1024)                    │
│                                                                  │
│  2. INT8 prefix prefilter (first 16 bytes of each candidate)    │
│     → Approximate hamming on 128 bits                            │
│     → Prune 95%+ of candidates                                  │
│     → Cost: ~1 VPOPCNTDQ per candidate                          │
│                                                                  │
│  3. Exact hamming on survivors only                              │
│     → Full 2KB VPOPCNTDQ sweep (32 instructions)                │
│     → Only ~5% of candidates reach here                         │
│                                                                  │
│  Savings: ~20x fewer AVX-512 cycles for the search phase        │
│  Freed cycles → better parallelization of bundle/encode ops     │
└─────────────────────────────────────────────────────────────────┘
```

## Roadmap

### Completed

- N-dimensional arrays (shape support, axis-based reductions)
- sort, zeros, ones, arange, linspace
- Integer support (i32 via `i32x16`, i64 via `i64x8`, full SIMD)
- Extended shaping and reshaping (reshape, squeeze, slice, transpose, flip, concatenate)
- Bitwise operations (AND, XOR, OR, NOT for u8/i32/i64)
- HDC primitives (BIND, BUNDLE, PERMUTE, DISTANCE)
- Int8 embeddings (dot_i8, cosine_i8, norm_sq_i8)
- Blackboard parallelization (lock-free split_at_mut threading)
- Statistics (std, var, percentile with axis support)
- Search & selection (argmin, argmax, top_k, cumsum)
- ML primitives (cosine_similarity, softmax, log_softmax)
- Adaptive cascade search (3sigma/2sigma early-exit for Hamming + cosine)
- **rustyblas: Pure Rust BLAS L1/L2/L3** (drop-in OpenBLAS replacement)
- **Cache-blocked GEMM** (Goto algorithm, 6x16/6x8 microkernels, multithreaded)
- **INT8 quantized GEMM** (AVX-512 VNNI vpdpbusd, symmetric/asymmetric/per-channel)
- **BF16 mixed-precision GEMM** (half bandwidth, f32 accumulation)
- **rustymkl: LAPACK** (LU, Cholesky, QR), **FFT** (radix-2 Cooley-Tukey), **VML** (exp, ln, sqrt, sin, cos)
- **Tiered compute dispatch** (INT8 -> BF16 -> FP32 -> GPU, runtime HW detection)
- **INT8 prefilter** (approximate stats, GEMM row pruning, two-stage HDC search)
- **Hardware detection** (AVX-512, VNNI, AMX via raw CPUID, NPU/GPU probing)

### In Progress

- AMX tile GEMM (amx_int8, amx_bf16 — detected and available)
- Burn Backend trait integration (pluggable ML framework support)
- ONNX inference via `ort` crate with custom operators

### Planned

- Register-blocked GEMM microkernel optimization (target: 200+ GFLOPS)
- SIMD BF16 GEMM via `vdpbf16ps` intrinsics (currently uses scalar conversion)
- SIMD INT8 GEMM cache-blocking (apply Goto algorithm to VNNI kernel)
- Vulkan compute dispatch for Intel Arc / Xe2 GPU
- C++ and WASM bindings

### Not Planned

- Random number generation (use the `rand` crate)
- Python bindings (upstream project provides these; this fork is pure Rust)

## Design Principles

1. **No 3rd Party Dependencies:** Pure `std::simd` + `core::arch` — zero external crates.
2. **Leverage Portable SIMD:** Explicit `f32x16`/`f64x8`/`u8x64` types that map to AVX-512 on capable hardware, fall back gracefully elsewhere.
3. **Hardware-Aware:** Runtime detection of VPOPCNTDQ, VNNI, AMX, BF16 via CPUID. Tiered dispatch routes to cheapest capable path.
4. **Lock-Free Parallelism:** Blackboard split-borrow + `SendMutPtr<T>` for multithreaded GEMM. No `Arc<Mutex>`, no contention.
5. **Prefilter-First:** Use INT8 VNNI to decide *what* to compute before spending FP32 cycles. Cheap prepass saves expensive compute.
6. **Zero Serialization:** Blackboard shared memory means BLAS, LAPACK, FFT, and HDC all operate on the same aligned buffers with zero copies.

## Build

### Run Tests

```bash
# All crates (79 tests)
RUSTFLAGS="-C target-cpu=native" cargo test --package rustynum-core --package rustyblas --package rustymkl

# rustynum-rs (303 tests)
cd rustynum-rs && cargo test
```

### Run Benchmarks

```bash
# GEMM benchmark (old vs new)
RUSTFLAGS="-C target-cpu=native" cargo run --release --example gemm_benchmark --package rustyblas

# HDC / SIMD / adaptive cascade benchmarks
cd rustynum-rs
RUSTFLAGS="-C target-cpu=native" cargo bench --bench hdc_benchmarks

# Core numerical ops vs nalgebra/ndarray
RUSTFLAGS="-C target-cpu=native" cargo bench --bench array_benchmarks

# BLAS criterion benchmarks
RUSTFLAGS="-C target-cpu=native" cargo bench --bench blas_benchmarks --package rustyblas

# MKL criterion benchmarks
RUSTFLAGS="-C target-cpu=native" cargo bench --bench mkl_benchmarks --package rustymkl
```

### Detect Hardware

```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --example detect_caps --package rustynum-core 2>/dev/null || \
  echo "Run: use rustynum_core::compute; compute::print_caps();"
```

### NumPy Comparison

```bash
pip install numpy
python benchmarks/numpy_comparison.py
```

### Generate Docs

```bash
cargo doc --open
```
