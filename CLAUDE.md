# CLAUDE.md — Rustynum

> **Updated**: 2026-03-22
> **CI Status**: BROKEN — see details below
> **Branch**: main

---

## What This Is

"The Muscle." SIMD numerical substrate. AVX-512 Hamming, BF16 GEMM, BNN,
organic plasticity, CLAM clustering, Arrow integration, DataFusion UDFs.

## ⚠ READ BEFORE WRITING CODE

### 1. CI IS BROKEN

CI is BROKEN as of 2026-03-22.
Root cause: missing `unsafe` block in `rustynum-core/src/compute.rs:64` (call to `__cpuid_count`).
Rust CI and Python bindings both failing.
**FIX CI BEFORE ADDING NEW CODE.**

### 2. THREE-TIER WORKSPACE

```
Tier 1 (default): rustynum-rs, rustynum-core, rustynum-bnn, rustynum-clam,
                   rustynum-cam, rustynum-arrow, rustynum-accel
  → cargo build / cargo test operates on this tier only

Tier 2 (qualia):  qualia-xor (Nib4 vs BERT comparison)
  → cargo test -p qualia-xor

Tier 3 (holo):    rustynum-holo, rustynum-oracle, rustynum-carrier, rustynum-focus
  → cargo test -p rustynum-holo

Frozen:           .archive-rustynum-v1, .archive-rustynum-v3, etc.
  → DO NOT MODIFY. Path dep resolution only.
```

**Note (2026-03-22):** Some workspace crates mentioned in docs don't exist as directories: rustynum-cam, rustynum-accel, rustynum-carrier, rustynum-focus.

### 3. ladybug-rs DEPENDS ON THIS

ladybug-rs has path deps on: `rustynum-rs`, `rustynum-core`, `rustynum-bnn`,
`rustynum-arrow`, `rustynum-holo`, `rustynum-clam`.
Breaking changes here break ladybug-rs. Check compatibility.

### 4. DEPRECATED API MIGRATION (PRs 91-92)

The deprecated API migration is COMPLETE as of 2026-03-22: 34 deprecated + 38 try_* replacements.
Old: `softmax()` (panics on bad input)
New: `try_softmax()` (returns Result)
Callers in ladybug-rs may not be updated yet.

### 5. DO NOT MODIFY FROM ladybug-rs SESSIONS

Prompt 00_SESSION_A_META.md says: "You have read access to rustynum.
Do NOT modify rustynum — a separate session owns that."
ladybug-rs sessions import types, never change them.

## Build

```bash
cargo test                    # Tier 1 only (default members)
cargo test --workspace        # All tiers
cargo test -p rustynum-core   # Single crate
```

## Role in Four-Repo Architecture

```
rustynum     = The Muscle    ← THIS REPO (SIMD substrate)
ladybug-rs   = The Brain     (BindSpace, server)
staunen      = The Bet       (6 instructions, no GPU)
lance-graph  = The Face      (query surface)
```

## Key Crates

```
rustynum-core     SIMD dispatch: AVX-512 → AVX2 → scalar. Hamming, popcount.
rustynum-bnn      Binary Neural Network: CausalTrajectory, BPReLU, pentary.
rustynum-clam     CLAM clustering: bipolar splits, codebook training.
rustynum-cam      Content-Addressable Memory: 48-bit fingerprint, scent index.
rustynum-arrow    Arrow/DataFusion integration: ScalarUDF wrappers for SIMD kernels.
rustynum-holo     Holographic: wave substrate, experimental.
rustynum-rs       Legacy NumArray API (being deprecated in favor of -core).
```

## THE SIMD LAW — MANDATORY FOR ALL SESSIONS

**CI enforces this. `scripts/simd-police.sh` runs on every PR. Violations = red CI.**

### The Rules

```
1. STABLE ONLY. std::arch intrinsics (stable since Rust 1.72+).
   No #![feature(...)]. No nightly. No unconditional std::simd / portable_simd.
   (simd_isa.rs has opt-in portable_simd behind cfg(feature) — that's fine.)
2. ONE BINARY WORKS EVERYWHERE. Silent runtime fallback, not compile-time cfg.
3. Every AVX-512 intrinsic (__m512, _mm512_*) MUST be behind:
   - #[target_feature(enable = "avx512f,...")]  on the function
   - is_x86_feature_detected!("avx512f")       at the call site
4. Every AVX-512 code path MUST have an AVX2 fallback.
5. Every AVX2 code path MUST have a scalar fallback.
6. simd.rs already does this correctly. COPY THE PATTERN. DO NOT INVENT YOUR OWN.
```

### The Pattern (from simd.rs — THE correct implementation)

```rust
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") {
            return unsafe { hamming_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { hamming_avx2(a, b) };
        }
    }
    hamming_scalar(a, b)  // ALWAYS reachable. Works on ARM, WASM, anything.
}

#[target_feature(enable = "avx512f,avx512vpopcntdq")]
unsafe fn hamming_avx512(...) { /* __m512 lives here ONLY */ }

#[target_feature(enable = "avx2")]
unsafe fn hamming_avx2(...) { /* __m256 lives here ONLY */ }

fn hamming_scalar(...) { /* no intrinsics, works everywhere */ }
```

### Known Violations (pre-existing, must be fixed)

```
BROKEN: simd_avx512.rs (renamed from simd_compat.rs in PR #99) — ~1550 lines
        of wrapper types (F32x16, etc.) whose methods call AVX-512 intrinsics
        unconditionally. E.g. F32x16::splat() calls _mm512_set1_ps without
        #[target_feature] guard → SIGILL if called on non-AVX-512 CPU.
        AVX2 types (F32x8, F64x4) defined but never wired as fallbacks.

FIXED:  rustyblas/src/level3.rs — imports F32x16 as F32Simd, but call path
        IS guarded by is_x86_feature_detected!("avx512f") at line ~131.
        Non-AVX-512 CPUs fall back to sgemm_simple → simd::dot_f32 (dispatched).
        Old comment "adapts at compile time via feature flags" is misleading
        but the runtime detection is correct. SGEMM_NR=16 in simd.rs is a
        tile-size constant, not an ISA issue.

FIXED:  fingerprint.rs — previously used #[cfg(feature)] gates, now delegates
        to crate::simd::hamming_distance() which has proper runtime detection.

ROOT CAUSE: Session 01BasHJkqcrCm171oam423Wu created simd_avx512.rs
            (originally simd_compat.rs, renamed PR #99) with wrapper types
            that call AVX-512 intrinsics without #[target_feature] guards.
            The Isa trait (PR #101) is the path to fixing this.
```

### Why This Matters

```
Without fallback: SIGILL on GitHub Actions, every laptop, macOS, ARM, WASM.
With fallback:    Runs everywhere. Fast where AVX-512 exists. Correct where it doesn't.
simd.rs proves:   The pattern works. ~2400 lines of tiered dispatch, zero SIGILLs.
```

## What NOT To Do

```
× Don't modify from a ladybug-rs Claude Code session
× Don't add panicking public APIs (use try_* pattern from PRs 91-92)
× Don't break Tier 1 default build (ladybug-rs depends on it)
× Don't add GPU deps (this is CPU SIMD only)
× Don't remove deprecated functions yet (ladybug-rs may still call them)
× Don't use __m512 without #[target_feature] + runtime detection + AVX2 fallback
× Don't use nightly features or unconditional std::simd (opt-in cfg(feature) OK)
× Don't write SIMD code without checking simd.rs for the correct pattern first
```

## Session Documents

```
.claude/prompts/01-21         Various session prompts (numbered)
IMPROVEMENT_ROADMAP.md        Outstanding improvements
COMPARISON_RUSTYNUM_VS_NDARRAY.md  Performance comparison
SIMD_INTEGRATION_ANALYSIS.md  SIMD tier analysis
```
