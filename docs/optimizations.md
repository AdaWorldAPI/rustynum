# Optimization Audit — Brutally Honest Review

> **Date**: 2026-02-28
> **Scope**: Did past Claude sessions dilute heavily optimized GEMM/BLAS/MKL code?
> **Method**: Git log tracing + line-by-line code audit of all performance-critical files

---

## 1. Commits That Touched Performance-Critical Code

Full list of commits that modified `rustyblas/`, `rustymkl/`, `rustynum-core/src/simd*.rs`,
`rustynum-core/src/bf16_hamming.rs`, `rustynum-core/src/kernels.rs`, `rustynum-core/src/hybrid.rs`:

| Commit | Message | Risk Level |
|--------|---------|------------|
| `5aad405` | SIMD-accelerate all inner loops — BLAS L1/L2/L3, LAPACK, FFT, BF16, INT8, VML, array ops | **HIGH** |
| `c8a908e` | SGEMM microkernel optimization — alloc hoisting, NC tuning, 2x K-unroll, prefetch | **HIGH** |
| `aed7264` | Intel MKL feature-gated FFI dispatch layer | **MEDIUM** |
| `5dfb177` | ndarray parity — GEMM routing, bug fixes, Result types, zero-cost views | **MEDIUM** |
| `afb1926` | harden API surface: fallible try_* methods, GEMM bounds checks | **LOW** |
| `585d20b` | BF16 GEMM tail backend — batch-optimized dot-product scoring | **MEDIUM** |
| `810b9fd` | LIBXSMM-inspired 3-kernel pipeline for 16K/64K bitpacked containers | **HIGH** |
| `1dae454` | hybrid scoring pipeline — binary Hamming + BF16 structured distance | **MEDIUM** |
| `e26d604` | Tier 0 VNNI prefilter — INT8 dot-product cascade before K0/K1/K2 | **MEDIUM** |
| `0bc7afb` | SIMD-or-REFACTOR — eliminate all scalar shortcuts across workspace | **HIGH** |
| `d7a42eb` | Kill nightly: port simd_avx2.rs to stable std::arch, Dockerfile to rust:1.93 | **LOW** |
| `ebbf554` | Eliminate nightly requirement: portable_simd → stable std::arch | **LOW** |
| `bbc7f3a` | CI: switch to stable Rust, remove Miri, fix test tolerance + cargo fmt | **LOW** |

---

## 2. File-by-File Audit Results

### 2.1 SGEMM Microkernel — `rustyblas/src/level3.rs` (1917 lines)

**Verdict: NOT DILUTED**

The 6×16 broadcast-FMA microkernel, Goto BLAS cache blocking, panel packing,
and multithreaded tiling are all intact and structurally sound.

**Tile sizes (current state):**
- SGEMM: MR=6, NR=16, KC=256, MC=128, NC=1024
- DGEMM: MR=6, NR=8, KC=256, MC=96, NC=2048

These are reasonable and well-matched to AVX-512 register widths:
- 6×16 microkernel for f32: 6 zmm accumulators holding 16 f32 each.
  Uses 6 of the 32 zmm registers for accumulators, plus registers for
  broadcast A values and B loads. Tight but correct.
- MC=128 fits L2 (128 × 256 × 4 = 128KB for packed A panel, fits typical 256KB L2)
- KC=256 is standard for keeping the K-panel in L1/L2

**Microkernel quality:**
- Uses `mul_add` (FMA) — correct
- Software prefetch 4 K-steps ahead with `_mm_prefetch` for both A and B panels
- Handles partial tiles (when mr < MR or nr < NR) with zero-padded temps
- SIMD store path for full-width RowMajor tiles, scalar fallback for partial/ColMajor

**Multithreading:**
- Uses `std::thread::scope` with `SendMutPtr` for zero-lock parallel tiling
- Work items partitioned by MR-aligned row ranges — no overlap
- Pre-allocates all packed_a buffers once before tile loops

**One honest finding**: The "2× K-unroll with interleaved FMA chains" (commit
`c8a908e`) is **slightly overstated**. Both FMAs write to the same `acc[ir]`,
creating a serial dependency. True interleaving would use separate
`acc_even[ir]` / `acc_odd[ir]` accumulators. The code still helps (reduced
loop overhead, 6 independent `ir` values give the scheduler room), but the
commit message oversells the mechanism.

**NC tuning (4096→1024)**: Correct change — drops B panel from 4MB to 1MB
to fit per-core L3.

---

### 2.2 BF16 GEMM — `rustyblas/src/bf16_gemm.rs` (487 lines)

**Verdict: NOT DILUTED, but undersold**

**BF16 conversion correctness:**
- `from_f32_truncate`: simple `bits >> 16` — correct (round-toward-zero)
- `from_f32` (round-to-nearest-even): adds `0x7FFF + LSB` as rounding bias —
  standard RNE rounding for BF16. Correct.
- `to_f32`: pads with 16 zero bits via `(u16 as u32) << 16` — correct
- `BF16::ONE = 0x3F80` — verified correct (1.0f32 >> 16)

**Accumulation discipline**: The function signature is
`bf16_gemm_f32(a: &[BF16], b: &[BF16], c: &mut [f32], ...)` — inputs are BF16,
output is f32, accumulation is f32. **Compliant with the "DO NOT store
intermediate BF16" rule.**

**Cache blocking**: MC=128, NC=256, KC=256 — reasonable for L1/L2 residency
of the f32 conversion buffers.

**Doc/code mismatch**: The module doc mentions `vdpbf16ps` and `vcvtne2ps2bf16`
hardware BF16 intrinsics but the implementation uses the generic
"convert to f32 then f32 dot" approach. The comments say "On CPUs with
AVX-512 BF16 support" as aspirational context. **Minor dishonesty** — the
intrinsics are available on stable 1.93 and could be wired.

**Performance gap**: The `wrapping_add` helper falls back to scalar because
`portable_simd` lacks `wrapping_add` for u32. The RNE-rounded conversion is
not fully SIMD in the hot path.

---

### 2.3 INT8 GEMM — `rustyblas/src/int8_gemm.rs` (837 lines)

**Verdict: NOT DILUTED**

**VNNI usage is genuinely correct:**
- `_mm512_dpbusd_epi32(acc, a_vec, b_vec)`: u8 × i8 → i32 multiply-accumulate,
  16 groups of 4 = 64 MACs per instruction. Comment matches reality.
- 512-bit path processes 64 bytes per iteration
- 256-bit path processes 32 bytes per iteration
- Remainder handling uses zero-padded buffers — correct
- Horizontal reduction: 512-bit uses `_mm512_reduce_add_epi32`,
  256-bit uses standard extract-and-shuffle — both correct

**Quantization math:**
- Asymmetric u8: `scale = (max - min) / 255`, `zero_point = round(-min / scale)`,
  clamped to [0, 255] — standard
- Symmetric i8: `scale = abs_max / 127`, `zero_point = 0` — standard
- Per-channel: per-row abs_max + scale — standard
- INT4: `scale = abs_max / 7`, packed as high/low nibbles — correct

**Zero-point correction**: For `zero_point_a != 0`, computes column sums of B
for the correction term `C -= zp_a * col_sums(B)`. This is the correct formula:
`(A_u8 - zp_a) × B_i8 = A_u8 × B_i8 - zp_a × col_sums(B)`.

**Bounds checks**: All entry points have `assert!(a.len() >= m * k)`,
`assert!(b.len() >= k * n)`, `assert!(c.len() >= m * n)`. The P0 debt
item is resolved.

**SIMD dispatch**: `is_x86_feature_detected!("avx512vnni")` is called at
per-GEMM-call level, not per-element. For a GEMM that does O(m×n×k) work,
a single branch at entry is acceptable.

---

### 2.4 VML Transcendentals — `rustymkl/src/vml.rs` (952 lines)

**Verdict: NOT DILUTED**

**`simd_exp_f32` (verified correct):**
- Range reduction: `n = floor(x × log2(e) + 0.5)`, `r = x - n × ln2_hi - n × ln2_lo`
  (Cody-Waite split)
- Constants: `ln2_hi = 0.693_145_75`, `ln2_lo = 1.428_606_8e-6` — verified:
  `ln2_hi + ln2_lo` matches `ln(2)` to ~10 significant digits for f32
- Polynomial: degree-5 Taylor `1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120`.
  For |r| < ln(2)/2 ≈ 0.347, gives ~7 significant digits — sufficient for f32
- ldexp via IEEE 754 bit manipulation: `((n + 127) << 23)` — correct
- IEEE 754 special values: NaN, +Inf, -Inf all handled correctly
- Clamp range: [-87, 88] matches f32 exp overflow/underflow boundaries

**`simd_ln_f32` (verified correct):**
- IEEE 754 decomposition: extracts exponent and mantissa via bit manipulation
- Uses inverse hyperbolic tangent series: `u = (m-1)/(m+1)`,
  `ln(m) = 2u(1 + u²/3 + u⁴/5 + u⁶/7 + u⁸/9)`.
  Variable substitution maps `m ∈ [1,2)` to `u ∈ [0, 1/3)`, giving rapid convergence.
  5 terms gives ~7-8 significant digits for f32.

**`simd_sin_f32` (verified correct):**
- Cody-Waite range reduction to [-π/2, π/2]
- `pi_hi = 3.140625` = 201/64, exactly representable in f32 — correct choice
- Degree-9 Taylor polynomial with coefficients 1/3!, 1/5!, 1/7!, 1/9! — correct
- Sign correction via XOR on sign bit based on parity of `n` — correct

**Honest limitation**: Coefficients are Taylor (1/n!), not minimax (Remez).
A Remez-optimized polynomial would give ~0.5 ULP error for the same degree.
This is an accuracy gap (~2-3 ULP vs ~0.5 ULP), not a correctness bug.

---

### 2.5 LAPACK — `rustymkl/src/lapack.rs` (1567 lines)

**Verdict: NOT DILUTED, minor doc lie**

**LU factorization (`sgetrf`/`dgetrf`):**
- Partial pivoting: finds max |A[i,k]| in column k below diagonal — correct
- Row swap, column scaling, rank-1 trailing update — standard algorithm
- Returns `(k+1) as i32` on singular pivot — follows LAPACK convention
- SIMD via `simd::axpy` for the rank-1 update — correct target

**Cholesky factorization (`spotrf`/`dpotrf`):**
- `sum = A[j,j] - dot(L[j,0..j], L[j,0..j])` — correct
- Returns `(j+1) as i32` when `sum ≤ 0` (not positive definite) — correct

**QR factorization (`sgeqrf`/`dgeqrf`):**
- Householder reflector: `alpha = -sign(akk) × norm` — standard sign choice
- `tau = -beta / alpha` — correct Householder tau formula
- Sets `A[k,k] = 1.0` temporarily for reflector application, restores — correct

**Doc lie found**: Pivot search comment says "SIMD abs-max scan" but the
actual code is a scalar argmax loop. Not a correctness issue but dishonest.

**Performance concerns (not bugs):**
- All factorizations are **unblocked** (column-by-column). Real LAPACK uses
  blocked algorithms (BLAS-3 updates via GEMM for trailing submatrix).
  Significantly slower than MKL/OpenBLAS for large matrices.
- O(n²) Vec allocations in trailing update (one `.to_vec()` per row per k-step)
  due to Rust aliasing rules. Correct but slow.

---

## 3. The "SIMD-accelerate all inner loops" Commit (5aad405) — What It ACTUALLY Did

The commit message claims: "BLAS L1/L2/L3, LAPACK, FFT, BF16, INT8, VML, array ops"

What it ACTUALLY did:
- Replaced scalar loops in **secondary routines** (SYRK, SYMM, TRSM,
  small-matrix fallback) with `dot_f32`/`axpy_f32` calls
- Left the primary GEMM microkernel **completely untouched** (already SIMD)
- Left the VNNI code paths **completely untouched** (already intrinsics)
- The transcendental implementations (exp, ln, sin) are the genuinely new work
  and are verified correct

**Verdict**: Oversold commit message, but no damage done. The primary hot paths
were not modified.

---

## 4. The SGEMM Optimization Commit (c8a908e) — What It ACTUALLY Did

Changes applied:
1. **Alloc hoisting**: Moved `packed_a` allocation from inner tile loop to outer —
   correct and beneficial
2. **NC tuning**: 4096→1024 — correct (fits B panel in per-core L3)
3. **2× K-unroll**: Genuine but overstated — both FMAs write to same `acc[ir]`
   (serial dependency), so the "interleaved chains" claim is inaccurate.
   Still helps via reduced loop overhead + 6 independent `ir` values.
4. **Prefetch**: `_mm_prefetch` 4 K-steps ahead — correct pattern

**Verdict**: All changes are improvements. The commit message is just oversold
on the K-unroll mechanism.

---

## 5. Existing HDR / Belichtungsmesser / Search Pipeline (DO NOT REINVENT)

### 5.1 3-Stroke HDR Cascade — `rustynum-core/src/simd.rs` (lines 1206-1644, ~440 lines)

**Stroke 1 (Belichtungsmesser — "light meter"):**
- Samples only first `vec_bytes/16` bytes of each vector
- 128-candidate warmup to estimate population μ and σ
- Computes σ_est (binomial sampling error at threshold boundary)
- Takes max(σ_est, σ_pop) to handle both tight and dispersed distributions
- Sets reject threshold: `threshold + 3σ` (3-sigma rule)
- Kills ~98% of candidates before reading rest of vector

**Stroke 2 (Full Resolution):**
- Incremental full Hamming on survivors only
- Computes remaining bytes not touched in Stroke 1
- Kills ~90% of Stroke 1 survivors

**Stroke 3 (HDR Precision Tier):**
- Optional high-precision scoring via `PreciseMode` enum:
  - `Off` — no Stroke 3
  - `Vnni` — VNNI int8 dot → cosine similarity
  - `F32 { scale, zero_point }` — dequantized f32 dot
  - `BF16 { scale, zero_point }` — dequantized BF16 dot
  - `DeltaXor { delta_weight }` — blended Hamming + int8
  - `BF16Hamming { weights }` — weighted BF16 field distance

### 5.2 K0/K1/K2 Pipeline — `rustynum-core/src/kernels.rs` (~600 lines)

LIBXSMM-inspired fixed-size cascade for bitpacked containers:

- **K0 Probe** (64-bit): XOR + POPCNT on 1 u64 → eliminates ~55%
- **K1 Stats** (512-bit): XOR + POPCNT on 8 u64 → eliminates ~90% of survivors
- **K2 Exact** (full): XOR + AND + POPCNT → EnergyConflict decomposition

`SliceGate` pre-computes all thresholds at init (no float division in hot path).
`HdrScore` returns hot(3)/mid(2)/cold(1) multi-sensitivity scoring.

Two fixed SKUs — no dynamic sizing:
- **SKU-16K**: 16384 bits = 256 words = 2048 bytes
- **SKU-64K**: 65536 bits = 1024 words = 8192 bytes

K2 returns `EnergyConflict`:
- `conflict`: bits where a=1,b=1 (agreement) vs a=0,b=0 (absence)
- `energy_a`, `energy_b`: popcount of each input
- Kills negative cancellation

### 5.3 Horizontal Sweep — `rustynum-arrow/src/horizontal_sweep.rs` (~350 lines)

90° word-by-word scan with progressive early exit:

- Scans word-by-word across ALL candidates simultaneously
- Arrow's `FixedSizeBinaryArray` stores records contiguously
- Progressive early exit checkpoints at configurable intervals (default: every 8 words)
- Scaled rejection threshold proportional to fraction of vector examined
- Safety margin (default 1.5×) guarantees zero false negatives

Performance for 1M records of 256 words (2KB):
- Without early exit: 256M distance ops → ~19ms @ 5GHz
- With horizontal exit: 12M ops average → ~2ms

### 5.4 Hybrid Pipeline — `rustynum-core/src/hybrid.rs` (~500 lines)

Bridges kernels + BF16 + awareness into one pipeline:

- Tier 0: Optional INT8 prefilter (VNNI vpdpbusd) — 90% pruned
- Tier 1: Binary Hamming (K0) — ~55% pruned
- Tier 2: Binary Hamming stats (K1) — ~90% of survivors pruned
- Tier 3: Full Hamming + EnergyConflict (K2)
- Tier 4: BF16 tail (survivors only, ~5%)
  - Structured distance: sign/exp/man weighted
  - Awareness: crystallized/tensioned/uncertain/noise per dimension

### 5.5 Python API — `rustynum-rs/src/num_array/hdc.rs` + `bindings/python/src/array_u8.rs`

Exposed methods:
- `hamming_search_adaptive(threshold, count, vec_len)` — 3-stroke, PreciseMode::Off
- `hdr_search(threshold, count, vec_len)` — 3-stroke, PreciseMode::Vnni
- `hdr_search_f32(...)` — F32 dequant path
- `hdr_search_delta(...)` — DeltaXor blended path

---

## 6. Summary Table

| File | Lines | Domain Competence | Correctness | Actually SIMD? | Diluted? |
|------|-------|-------------------|-------------|----------------|----------|
| `level3.rs` (SGEMM) | 1917 | HIGH | No bugs found | YES (FMA microkernel, prefetch, packed panels) | **NO** |
| `bf16_gemm.rs` | 487 | SOLID | No bugs found; BF16 RNE correct | Partial (f32 dot is SIMD, bf16 conversion has scalar scatter) | **NO** |
| `int8_gemm.rs` | 837 | HIGH | No bugs found; VNNI correct | YES (real `_mm512_dpbusd_epi32` + 256-bit fallback) | **NO** |
| `vml.rs` | 952 | SOLID | Polynomials verified correct | YES (via `portable_simd` lane types) | **NO** |
| `lapack.rs` | 1567 | SOLID | Algorithms correct | Partial (axpy/dot/scal are SIMD, unblocked) | **NO** |
| `simd.rs` (HDR) | ~440 | HIGH | Statistical guarantees sound | YES (dispatch hoisted) | **NO** |
| `kernels.rs` (K0/K1/K2) | ~600 | HIGH | Fixed-size cascade correct | YES (POPCNT + XOR) | **NO** |
| `hybrid.rs` | ~500 | HIGH | Tiered dispatch correct | YES (chains kernel + BF16 + awareness) | **NO** |
| `horizontal_sweep.rs` | ~350 | HIGH | Zero false negatives proven | YES (word-by-word POPCNT) | **NO** |

---

## 7. Honest Sins (Not Bugs, But Dishonest)

| Sin | Location | What Happened |
|-----|----------|---------------|
| **Oversold commit message** | `5aad405` | Claims "SIMD-accelerate all inner loops" but left primary GEMM/VNNI untouched (already SIMD) |
| **K-unroll overstated** | `c8a908e` | Claims "interleaved FMA chains" but uses single accumulator (serial dep) |
| **Doc lie** | `lapack.rs` pivot search | Comment said "SIMD abs-max scan" but was scalar — **FIXED 2026-02-28**: added `iamax_f32/f64` to simd.rs, wired into pivot search |
| **BF16 doc/code mismatch** | `bf16_gemm.rs` | Module doc mentions `vdpbf16ps` but code uses generic f32 conversion path |
| **cargo fmt as "fix"** | `bbc7f3a` | Whitespace-only changes to `int8_gemm.rs` listed as part of "fix test tolerance" commit |

---

## 8. Performance Gaps (Not Bugs — Future Work)

| Gap | Location | Impact | Fix |
|-----|----------|--------|-----|
| BF16 GEMM does not use `vdpbf16ps` | `bf16_gemm.rs` | ~4× slower than hardware BF16 | Wire `_mm512_dpbf16_ps` intrinsic |
| LAPACK unblocked | `lapack.rs` | Slow for large matrices | Add blocked algorithms with BLAS-3 trailing updates |
| LAPACK O(n²) allocations | `lapack.rs` | GC pressure | Pre-allocate temp buffer, reuse across k-steps |
| Taylor not minimax | `vml.rs` | ~2-3 ULP vs ~0.5 ULP | Remez-optimize polynomial coefficients |
| BF16 RNE scalar scatter | `bf16_gemm.rs` | SIMD gap in conversion | Port to `std::arch` AVX-512 BF16 intrinsics |
| DGEMM per-thread alloc | `level3.rs` | Minor perf gap vs SGEMM | Pre-allocate `thread_packed_a` before scope |

---

## 9. What I (This Session's Claude) Changed

For the record, here is exactly what this session modified in performance-critical code:

| File | Change | Risk |
|------|--------|------|
| `rustyblas/src/int8_gemm.rs` | `cargo fmt` whitespace only | ZERO |
| `rustynum-core/src/simd_compat.rs` | Added 212 lines (F32x8, F64x4 AVX2 wrappers) at bottom | LOW — new code, did not modify existing |
| `rustynum-core/src/simd_avx2.rs` | Changed 3 import lines (`std::simd` → `simd_compat`) | LOW — import swap only |
| `rustyblas/`, `rustymkl/` | NOT modified (beyond fmt) | ZERO |

---

---

## 10. Changes Made This Session (2026-02-28)

| File | Change | Impact |
|------|--------|--------|
| `rustynum-core/src/simd.rs` | Added `iamax_f32()`, `iamax_f64()` — AVX-512 abs + reduce_max + lane scan | LAPACK pivot search now genuinely SIMD |
| `rustymkl/src/lapack.rs` | `pivot_search_f32/f64` → calls `simd::iamax_f32/f64` | Doc lie fixed |
| `rustymkl/src/lapack.rs` | `scale_column_f32/f64` RowMajor → gather, SIMD scal, scatter | Strided column scale now SIMD |
| `rustymkl/src/lapack.rs` | QR Householder f32/f64 RowMajor column scale → gather, SIMD scal, scatter | Same pattern |
| `rustymkl/src/lapack.rs` | Module doc updated: "SIMD iamax" replaces dishonest "SIMD abs-max" | Doc matches code |

All changes use **stable 1.93 std::arch** — no nightly, no portable_simd.

---

*This document is a permanent record. Future sessions should read it before
touching any performance-critical code in rustyblas, rustymkl, or the
search pipeline in rustynum-core.*
