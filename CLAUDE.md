# CLAUDE.md — Rustynum

> **Updated**: 2026-03-12
> **CI Status**: Rust CI FAILING, Python bindings FAILING
> **Branch**: main

---

## What This Is

"The Muscle." SIMD numerical substrate. AVX-512 Hamming, BF16 GEMM, BNN,
organic plasticity, CLAM clustering, Arrow integration, DataFusion UDFs.

## ⚠ READ BEFORE WRITING CODE

### 1. CI IS BROKEN

Rust CI and Python bindings both failing as of 2026-03-03 (sha d46f0d20).
Likely cause: PRs 91/92 deprecated panicking APIs → broke downstream callers.
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

### 3. ladybug-rs DEPENDS ON THIS

ladybug-rs has path deps on: `rustynum-rs`, `rustynum-core`, `rustynum-bnn`,
`rustynum-arrow`, `rustynum-holo`, `rustynum-clam`.
Breaking changes here break ladybug-rs. Check compatibility.

### 4. DEPRECATED API MIGRATION (PRs 91-92)

28 panicking public functions deprecated, `try_*` versions added.
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

## What NOT To Do

```
× Don't modify from a ladybug-rs Claude Code session
× Don't add panicking public APIs (use try_* pattern from PRs 91-92)
× Don't break Tier 1 default build (ladybug-rs depends on it)
× Don't add GPU deps (this is CPU SIMD only)
× Don't remove deprecated functions yet (ladybug-rs may still call them)
```

## Session Documents

```
.claude/prompts/01-21         Various session prompts (numbered)
IMPROVEMENT_ROADMAP.md        Outstanding improvements
COMPARISON_RUSTYNUM_VS_NDARRAY.md  Performance comparison
SIMD_INTEGRATION_ANALYSIS.md  SIMD tier analysis
```
