# AVX-512 Intrinsic Audit: Nightly → Stable 1.93.1

> **Date**: 2026-02-27
> **Auditor**: Claude (prompted by Jan Hübener)
> **Method**: Compiled all intrinsics on both nightly and stable 1.93.1
> **Verdict**: Zero blockers. All AVX-512 intrinsics compile on stable.

---

## Context

rustynum targets AVX-512 exclusively (no AVX2 fallback in hot paths).
Prior assumption was that several AVX-512 intrinsic families were
nightly-only or partially stabilized. This audit proves that wrong.

---

## Tested Intrinsic Families

### Core Compute (16K bitpack hot path)

| Intrinsic | Feature Gate | Stable 1.93.1 | Notes |
|-----------|-------------|---------------|-------|
| `_mm512_loadu_si512` | avx512f | Yes | |
| `_mm512_xor_si512` | avx512f | Yes | |
| `_mm512_and_si512` | avx512f | Yes | |
| `_mm512_or_si512` | avx512f | Yes | |
| `_mm512_popcnt_epi64` | avx512vpopcntdq | Yes | K1/K2 Hamming |
| `_mm512_add_epi64` | avx512f | Yes | |
| `_mm512_ternarylogic_epi64` | avx512f | Yes | 3-input bitwise |
| `_mm512_reduce_add_epi64` | avx512f | Yes | Was assumed missing — it's not |
| `_mm512_reduce_or_epi64` | avx512f | Yes | |

### BF16

| Intrinsic | Feature Gate | Stable 1.93.1 | Notes |
|-----------|-------------|---------------|-------|
| `_mm512_dpbf16_ps` | avx512bf16 | Yes | BF16 dot product → f32 accumulate |
| `_mm512_cvtne2ps_pbh` | avx512bf16 | Yes | 2×f32 → packed bf16 |
| `_mm512_cvtpbh_ps` | avx512bf16 | Yes | bf16 → f32 unpack |

### VNNI (INT8 prefilter)

| Intrinsic | Feature Gate | Stable 1.93.1 | Notes |
|-----------|-------------|---------------|-------|
| `_mm512_dpbusd_epi32` | avx512vnni | Yes | u8×i8 dot → i32 |
| `_mm512_dpbusds_epi32` | avx512vnni | Yes | Saturating variant |

### Mask Registers

| Intrinsic | Feature Gate | Stable 1.93.1 | Notes |
|-----------|-------------|---------------|-------|
| `_kand_mask16` | avx512f | Yes | Was listed as "partial" — fully stable |
| `_knot_mask16` | avx512f | Yes | |
| `_kor_mask16` | avx512f | Yes | |
| `_kxor_mask16` | avx512f | Yes | |
| `_kand_mask8` | avx512dq | Yes | |
| `_knot_mask8` | avx512dq | Yes | |
| `_mm512_movepi8_mask` | avx512bw | Yes | Was listed as "some missing" — stable |
| `_mm512_movm_epi8` | avx512bw | Yes | |

### Masked Operations

| Intrinsic | Feature Gate | Stable 1.93.1 | Notes |
|-----------|-------------|---------------|-------|
| `_mm512_mask_blend_epi64` | avx512f | Yes | |
| `_mm512_maskz_loadu_epi64` | avx512f | Yes | |
| `_mm512_cmpeq_epi64_mask` | avx512f | Yes | |

### Horizontal Reduces

| Intrinsic | Feature Gate | Stable 1.93.1 | Notes |
|-----------|-------------|---------------|-------|
| `_mm512_reduce_add_epi64` | avx512f | Yes | Was assumed nightly-only — compiles fine |
| `_mm512_reduce_or_epi64` | avx512f | Yes | |
| `_mm512_reduce_add_ps` | avx512f | Yes | |
| `_mm512_reduce_max_epi64` | avx512f | Yes | |

---

## What About std::simd (portable SIMD)?

std::simd remains nightly-only (`#![feature(portable_simd)]`).
It is irrelevant for rustynum because:

1. We target AVX-512 exclusively — no portability benefit
2. No `Simd<bf16, N>` type exists even on nightly
3. `std::arch` intrinsics compile to identical machine code
4. std::simd adds abstraction overhead with zero performance gain for single-ISA targets

**Decision:** std::simd is not on the roadmap. std::arch is the permanent API.

---

## Corrected Assumptions

| Prior Claim | Reality |
|-------------|---------|
| `_mm512_reduce_add_epi64` missing on stable | Compiles on 1.93.1 |
| Mask arithmetic (`_kand_mask16` etc.) partial | All stable |
| `_mm512_movepi8_mask` some missing | Stable |
| `_mm512_movm_epi8` some missing | Stable |
| `_mm512_cvtpbh_ps` (bf16→f32) needs manual shift | Intrinsic is stable |
| Nightly needed for AVX-512 BF16 intrinsics | All stable since ≤1.93 |
| Nightly needed for AMX testing | AMX unstable even on nightly (#126622) |
| std::simd adds value for AVX-512 targets | Zero benefit, single-ISA |

---

## AMX Status

AMX is unstable on BOTH stable AND nightly. It requires
`#![feature(x86_amx_intrinsics)]` gated behind
[rust-lang/rust#126622](https://github.com/rust-lang/rust/issues/126622).
`_ldtilecfg`, `_tile_dpbssd`, `_tile_dpbf16ps`, `_tile_loadd` — none of
these exist in scope without the feature flag, even on nightly 1.95.

| ISA | Stable 1.93.1 | Nightly 1.95 |
|-----|---------------|--------------|
| AVX-512 (F, BW, DQ, VPOPCNTDQ, BITALG) | Yes | Yes |
| AVX-512 BF16 (dpbf16_ps, cvtpbh_ps, cvtne2ps_pbh) | Yes | Yes |
| AVX-512 mask ops (_kand_mask16, movepi8_mask, etc.) | Yes | Yes |
| AVX-512 reduces (reduce_add_epi64, etc.) | Yes | Yes |
| AMX (_tile_dpbssd, _tile_dpbf16ps, _tile_loadd) | **No** — unstable | **No** — unstable |

This validates removing AMX from the tiered dispatch — it's literally not
usable in Rust yet. BF16 via AVX-512 intrinsics (which is what we actually
need) is 100% stable. The only BF16 path that doesn't work is AMX BF16
(`_tile_dpbf16ps`), the matrix-tile version.

---

## Toolchain Policy (updated)

```toml
rust-version   = "1.93"          # MSRV for all crates
edition        = "2021"          # core crates; 2024 acceptable for leaf
toolchain      = "stable"        # production, CI, release
nightly use    = none            # AMX is unstable even on nightly — no reason to use nightly
portable_simd  = never           # single-ISA target, no benefit
AMX            = blocked         # rust-lang/rust#126622, not usable in any toolchain
```

No nightly. No feature flags. Stable 1.93 only. Period.
