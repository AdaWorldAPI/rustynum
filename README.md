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

### Int8 Embedding Operations (VNNI)

| Operation | Method | Description |
|-----------|--------|-------------|
| **Dot Product** | `a.dot_i8(&b)` | Signed int8 multiply-accumulate → i64 |
| **Norm²** | `a.norm_sq_i8()` | Squared L2 norm as int8 |
| **Cosine** | `a.cosine_i8(&b)` | Cosine similarity [-1.0, 1.0] |

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

### HDC Operations (AVX-512, `target-cpu=native`)

#### BUNDLE — Majority Vote (8192-byte vectors)

| n vectors | RustyNum | Naive baseline | Speedup |
|-----------|----------|---------------|---------|
| 5 | **96 µs** | 210 µs | 2.2x |
| 16 | **237 µs** | 646 µs | 2.7x |
| 64 | **633 µs** | 3.24 ms | 5.1x |
| 256 | **3.86 ms** | 11.4 ms | 2.9x |
| 1024 | **4.01 ms** | 70.9 ms | **17.7x** |

Note: n=1024 barely costs more than n=256 — the ripple-carry counter scales O(log n) per lane.

#### Int8 Dot Product (VNNI)

| Dimensions | dot_i8 | cosine_i8 |
|------------|--------|-----------|
| 1024D (1 KB) | **226 ns** | 522 ns |
| 2048D (2 KB) | **429 ns** | 1.11 µs |
| 8192D (8 KB) | **1.59 µs** | 4.50 µs |

226 ns for 1024D int8 dot product = ~4.4M similarities/sec/core.

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

292 tests (239 unit + 2 integration + 51 doc tests).

### Run Benchmarks

```bash
cd rustynum-rs
RUSTFLAGS="-C target-cpu=native" cargo bench --bench hdc_benchmarks
```

### Generate Docs

```bash
cd rustynum-rs
cargo doc --open
```
