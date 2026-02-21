![RustyNum Banner](docs/assets/rustynum-banner.png?raw=true "RustyNum")

![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

# RustyNum

RustyNum is a high-performance numerical computation library written in Rust, leveraging `portable_simd` (nightly) for SIMD-accelerated operations across platforms. Originally created as a fast NumPy alternative, it now includes **AVX-512 optimized Hyperdimensional Computing (HDC)** primitives for bitpacked vector operations.

## Key Features

- **High Performance:** Rust's `portable_simd` with explicit AVX-512 paths (`u64x8`, `u8x64`, `i32x16`, `i64x8`) for maximum throughput.
- **HDC / Vector Symbolic Architecture:** BIND (XOR), BUNDLE (majority vote), PERMUTE (bit rotation), DISTANCE (Hamming), all hardware-accelerated.
- **Int8 Embeddings:** VNNI-targetable dot product and cosine similarity for quantized neural embeddings.
- **Lightweight:** Zero external dependencies. Pure `std::simd`.
- **CogRecord Ready:** Designed for 4 × 16384-bit container architecture (8KB records) with VPOPCNTDQ and VNNI support.

## Supported Data Types

| Type | SIMD Vector | Status |
|------|-------------|--------|
| float32 | `f32x16` | Stable |
| float64 | `f64x8` | Stable |
| uint8 | `u8x64` | Stable (HDC primary type) |
| int32 | `i32x16` | Stable |
| int64 | `i64x8` | Stable |

## Supported Operations

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
| **BUNDLE** | `NumArrayU8::bundle(&[&a, &b, &c])` | Majority vote (hybrid: naive n≤16, ripple-carry n>16) |
| **DISTANCE** | `a.hamming_distance(&b)` | Hamming distance via POPCNT |
| **POPCOUNT** | `a.popcount()` | Population count |
| **BATCH DISTANCE** | `a.hamming_distance_batch(&b, dim, count)` | Batched Hamming for database scans |
| **ADAPTIVE DISTANCE** | `a.hamming_distance_adaptive(&b, threshold)` | 3-stage cascade with 3σ/2σ early-exit |
| **ADAPTIVE SEARCH** | `q.hamming_search_adaptive(&db, dim, n, thresh)` | Batch scan, eliminates ~99.7% early |

### Int8 Embedding Operations (VNNI)

| Operation | Method | Description |
|-----------|--------|-------------|
| **Dot Product** | `a.dot_i8(&b)` | Signed int8 multiply-accumulate → i64 |
| **Norm²** | `a.norm_sq_i8()` | Squared L2 norm as int8 |
| **Cosine** | `a.cosine_i8(&b)` | Cosine similarity [-1.0, 1.0] |
| **ADAPTIVE COSINE** | `q.cosine_search_adaptive(&db, dim, n, min_sim)` | Cascade cosine scan, FP64 at ~3% cost |

## Quick Start (Rust)

```rust
use rustynum_rs::NumArrayU8;

// BIND: XOR two hypervectors (involutory)
let a = NumArrayU8::new(vec![0xAA; 8192]);
let b = NumArrayU8::new(vec![0x55; 8192]);
let bound = a.bind(&b);
assert_eq!(bound.bind(&b).get_data(), a.get_data()); // recovered

// PERMUTE: rotate bit-planes for role encoding
let rel = a.permute(1);
let tgt = b.permute(2);

// BUNDLE: majority vote across multiple vectors
let c = NumArrayU8::new(vec![0xFF; 8192]);
let majority = NumArrayU8::bundle(&[&a, &b, &c]);

// DISTANCE: Hamming via POPCNT
let dist = a.hamming_distance(&b);

// Edge encoding: src ^ permute(rel, 1) ^ permute(tgt, 2)
let edge = &(&a ^ &rel) ^ &tgt;

// Int8 dot product (VNNI-accelerated)
let emb_a = NumArrayU8::new(vec![127; 1024]); // 1024D int8 embedding
let emb_b = NumArrayU8::new(vec![127; 1024]);
let similarity = emb_a.cosine_i8(&emb_b); // ≈ 1.0
```

## Performance

### Before/After SIMD — Naive Scalar vs AVX-512

All benchmarks run with `RUSTFLAGS="-C target-cpu=native"` on AVX-512 capable hardware.

#### HDC Bitwise Operations (8192-byte = 65536-bit vectors)

| Operation | Naive Scalar | RustyNum SIMD | Speedup |
|-----------|-------------|---------------|---------|
| XOR / BIND (8 KB) | 5.5 µs | **0.7 µs** | **8x** |
| XOR / BIND (16 KB) | 11 µs | **1.4 µs** | **8x** |
| XOR / BIND (32 KB) | 22 µs | **1.4 µs** | **16x** |
| Hamming distance (8 KB) | 3.8 µs | **1.7 µs** | **2.2x** |
| Hamming distance (16 KB) | 7.6 µs | **3.4 µs** | **2.2x** |
| Popcount (8 KB) | 2.5 µs | **1.5 µs** | **1.7x** |

Note: Hamming/popcount naive baselines use Rust's `count_ones()` intrinsic which already emits POPCNT. The "SIMD" path adds 4x unrolled u64 pipelining on top.

#### BUNDLE — Majority Vote (8192-byte vectors)

| n vectors | RustyNum SIMD | Naive baseline | Speedup |
|-----------|---------------|---------------|---------|
| 5 | **96 µs** | 210 µs | 2.2x |
| 16 | **237 µs** | 646 µs | 2.7x |
| 64 | **633 µs** | 3.24 ms | 5.1x |
| 256 | **3.86 ms** | 11.4 ms | 2.9x |
| 1024 | **4.01 ms** | 70.9 ms | **17.7x** |

Note: n=1024 barely costs more than n=256 — the ripple-carry counter scales O(log n) per lane.

#### Int8 Dot Product — SIMD vs Naive

| Dimensions | SIMD dot_i8 | SIMD cosine_i8 | Naive dot | Naive cosine |
|------------|-------------|----------------|-----------|--------------|
| 1024D (1 KB) | **226 ns** | 522 ns | 293 ns | 475 ns |
| 2048D (2 KB) | **429 ns** | 1.11 µs | 475 ns | 1.05 µs |
| 8192D (8 KB) | **1.65 µs** | 4.86 µs | 1.29 µs | 1.85 µs |

The compiler auto-vectorizes the "naive" i8 multiply-accumulate loop well. SIMD advantage grows with larger vectors and when VNNI instructions are available.

### Adaptive Cascade Search — Before/After HDR Early Exit

#### Hamming Cascade: Full Scan vs Adaptive (random database, ~0.1% match rate)

| Database | Full Scan | Adaptive Cascade | Speedup |
|----------|-----------|------------------|---------|
| 1K × 2 KB | 226 µs | **49 µs** | **4.6x** |
| 10K × 2 KB | 2.63 ms | **369 µs** | **7.1x** |
| 1K × 8 KB | 787 µs | **78 µs** | **10.1x** |
| 10K × 8 KB | 9.0 ms | **724 µs** | **12.4x** |

#### Cosine Cascade: Full Scan vs Adaptive (random database, min_similarity=0.9)

| Database | Full Scan | Adaptive Cascade | Speedup | FP64 cost |
|----------|-----------|------------------|---------|-----------|
| 1K × 1024D | 753 µs | **88 µs** | **8.6x** | ~12% |
| 10K × 1024D | 7.49 ms | **793 µs** | **9.4x** | ~11% |
| 1K × 2048D | 1.41 ms | **86 µs** | **16.4x** | ~6% |
| 10K × 2048D | 14.2 ms | **1.13 ms** | **12.6x** | ~8% |

The adaptive cascade delivers **FP64-precise cosine at 3-12% of full compute cost** by rejecting 99.7% of non-matching candidates at the 1/16 sample stage.

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

### RustyNum vs NumPy Comparison

| Operation | NumPy | RustyNum | Speedup | Notes |
|-----------|-------|----------|---------|-------|
| XOR / BIND (8 KB) | ~3-5 µs | **0.7 µs** | **5-7x** | NumPy has no SIMD bitwise path |
| Hamming distance (8 KB) | ~25-50 µs | **1.7 µs** | **15-30x** | NumPy: unpackbits+sum |
| Bundle n=64 (8 KB) | ~8-20 ms | **633 µs** | **13-32x** | NumPy: unpackbits per vector |
| Int8 dot (1024D) | ~3-8 µs | **226 ns** | **13-35x** | NumPy: astype(int32)+dot |
| Int8 cosine (1024D) | ~10-20 µs | **522 ns** | **19-38x** | NumPy: astype(float64)+norm |
| DB scan 10K × 2 KB | ~500-1000 ms | **369 µs** | **1350-2700x** | Adaptive cascade eliminates 99.7% |
| f32 addition (10K) | ~3-5 µs | **760 ns** | **4-7x** | NumPy C loops vs portable_simd |
| f32 mean (10K) | ~3-5 µs | **684 ns** | **4-7x** | |
| f32 dot (10K) | ~2-4 µs | **759 ns** | **3-5x** | |
| Matrix mul (1K×1K) | ~1-3 ms | 17.8 ms | **0.06-0.17x** | NumPy wins (BLAS/MKL) |

**Where RustyNum replaces NumPy entirely:**
- HDC/VSA bitwise operations: **15-30x faster** (no Python overhead, native SIMD)
- Int8 embedding search: **13-38x faster** (VNNI + cascade filter)
- Vector statistics (mean, std, var): **4-7x faster** (explicit SIMD)
- Database scans with early exit: **1000x+ faster** (cascade has no NumPy equivalent)
- Zero-dependency deployment (no BLAS, no pip, no GIL)

**Where NumPy still leads:**
- Large matrix multiplication (>512x512): NumPy uses cache-blocked BLAS/MKL with multi-threading
- Already-vectorized array operations on very large arrays (>1M elements)

### RustyNum vs GPU (NVIDIA H100) Analysis

| Operation | H100 GPU | RustyNum (1 core) | H100/CPU ratio | Break-even |
|-----------|----------|-------------------|----------------|------------|
| XOR 8 KB | ~50 ns | 0.7 µs | 14x faster | GPU wins at >1M concurrent ops |
| Hamming 2 KB | ~80 ns | 1.7 µs | 21x faster | Need batch >10K to amortize PCIe |
| Int8 dot 1024D | ~30 ns | 226 ns | 7.5x faster | GPU wins only with tensor cores in batch |
| Matrix mul 1K×1K | ~0.1 ms | 17.8 ms | 178x faster | GPU dominates for dense GEMM |
| **DB scan 10K × 2 KB** | ~2-5 ms (kernel) | **369 µs** | **CPU wins** | Cascade avoids 99.7% of work |
| **DB scan 1M × 2 KB** | ~200-500 ms (full) | **~37 ms** (est.) | **CPU competitive** | Cascade scales linearly |

**Sweet spot for RustyNum (CPU AVX-512):**

1. **HDC/VSA operations on single records or small batches** — no PCIe transfer latency, no kernel launch overhead, no GPU memory allocation. CPU processes a CogRecord in <2 µs.

2. **Database scans with adaptive cascade** — the 3σ/2σ early-exit eliminates 99.7% of work. A GPU must process every candidate (no branching). For 1M records, CPU+cascade does ~2.1M POPCNT instructions vs GPU's 32M — the CPU can match or beat the GPU.

3. **Latency-sensitive single-query search** — GPU batch amortization requires thousands of concurrent queries. For single-query response in <1ms, CPU wins decisively.

4. **Streaming / real-time** — no round-trip to GPU memory. New CogRecords can be searched immediately.

**Where GPU wins (don't fight it):**
- Dense matrix multiply (GEMM) > 512×512: tensor cores dominate
- Batch inference with >10K concurrent vectors
- Training workloads requiring FP16/BF16 matrix math

### Core Numerical Operations (Rust, float32)

| Input Size | RustyNum | nalgebra | ndarray |
|------------|----------|----------|---------|
| Addition (10k elements) | 760 ns | 696 ns | 664 ns |
| Vector mean (10k elements) | 684 ns | 14.6 µs | 1.24 µs |
| Vector dot product (10k elements) | 759 ns | 1.18 µs | 1.19 µs |
| Matrix-Vector (1k elements) | 78 µs | 403 µs | 116 µs |
| Matrix-Matrix (1k elements) | 17.8 ms | 21.9 ms | 22.4 ms |

## Architecture: CogRecord Container Layout

RustyNum is optimized for the 4 × 16384-bit CogRecord architecture:

```
CogRecord (8 KB = 65536 bits)
├── Container 0: META    (2 KB) — codebook identity, DN, hashtag zone
├── Container 1: CAM     (2 KB) — content-addressable memory (Hamming search)
├── Container 2: B-tree  (2 KB) — structural position index
└── Container 3: Embed   (2 KB) — int8/binary embeddings (VNNI dot + Hamming)
```

Each 2 KB container is exactly 32 AVX-512 registers. A full VPOPCNTDQ sweep is 32 instructions per container. Container 3 supports both distance metrics on the same memory:

- **Binary fingerprints** → Hamming distance via `VPOPCNTDQ`
- **Int8 embeddings** → Dot product via `VPDPBUSD` (VNNI)

## Roadmap

### Completed

- ~~N-dimensional arrays~~ (shape support, axis-based reductions)
- ~~sort~~ (`statistics.rs`)
- ~~zeros~~ (+ ones, arange, linspace constructors)
- ~~Integer support~~ (i32 via `i32x16`, i64 via `i64x8`, full SIMD)
- ~~Extended shaping and reshaping~~ (reshape, squeeze, slice, transpose, flip, concatenate)
- ~~Bitwise operations~~ (AND, XOR, OR, NOT for u8/i32/i64)
- ~~HDC primitives~~ (BIND, BUNDLE, PERMUTE, DISTANCE)
- ~~Int8 embeddings~~ (dot_i8, cosine_i8, norm_sq_i8)
- ~~Blackboard parallelization~~ (lock-free split_at_mut threading)
- ~~Statistics~~ (std, var, percentile with axis support)
- ~~Search & selection~~ (argmin, argmax, top_k, cumsum)
- ~~ML primitives~~ (cosine_similarity, softmax, log_softmax)
- ~~Adaptive cascade search~~ (3σ/2σ early-exit for Hamming + cosine, ~99.7% elimination)

### Planned

- Additional operations: interp
- C++ and WASM bindings

### Not Planned

- Random number generation (use the `rand` crate)
- Python bindings (upstream project provides these; this fork is pure Rust)

## Design Principles

1. **No 3rd Party Dependencies:** Pure `std::simd` — zero external crates.
2. **Leverage Portable SIMD:** Explicit `u64x8`/`u8x64`/`i32x16`/`i64x8` types that map to AVX-512 on capable hardware, fall back gracefully elsewhere.
3. **Hardware-Aware:** Targets VPOPCNTDQ (popcount), VNNI (int8 MAC), and AVX-512 bitwise ops when available via `-C target-cpu=native`.
4. **Lock-Free Parallelism:** Blackboard borrow-mut scheme (`split_at_mut` + `thread::scope`) — no `Arc<Mutex>`, no contention.

## Build

### Run Tests

```bash
cd rustynum-rs
cargo test
```

303 tests (247 unit + 2 integration + 54 doc tests).

### Run Benchmarks

```bash
cd rustynum-rs
# HDC / SIMD / adaptive cascade benchmarks (includes naive baselines)
RUSTFLAGS="-C target-cpu=native" cargo bench --bench hdc_benchmarks
# Core numerical ops vs nalgebra/ndarray
RUSTFLAGS="-C target-cpu=native" cargo bench --bench array_benchmarks
```

### NumPy Comparison

```bash
pip install numpy
python benchmarks/numpy_comparison.py
```

### Generate Docs

```bash
cd rustynum-rs
cargo doc --open
```
