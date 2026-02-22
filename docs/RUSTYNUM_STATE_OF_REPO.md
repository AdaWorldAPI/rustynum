# rustynum State-of-Repo Review — Post PR #26

**Date**: 2026-02-22  
**Commit**: 49b8ed92 (main, after PRs #23–26 merged)  
**Scope**: Full repository, brutally honest

---

## The Numbers

| Metric | Value |
|---|---|
| Total .rs lines | 48,348 |
| Active crate lines (core + clam + oracle) | 14,006 |
| Byte-identical duplicated lines | 5,343 |
| Workspace crates | 13 |
| Crates with meaningful recent work | 3 |
| Total `#[test]` functions | 1,031 |
| Tests in active crates | 245 |
| `unsafe` blocks in active crates | 14 |
| `debug_assert_eq!` guarding SIMD loads | 7 |
| TODO/FIXME in active crates | 1 |

---

## Structural Assessment

### The Good: Core → CLAM → Oracle Is Clean

The three active crates form a clear dependency chain:

```
rustynum-core (2,977 lines)
  ├── Blackboard (zero-copy shared memory)
  ├── Fingerprint<N> (const-generic binary vectors)
  ├── SplitMix64 (consolidated PRNG)
  ├── SIMD (AVX-512/AVX2 Hamming, dot, axpy)
  └── Prefilter (INT8 approximate screening)

rustynum-clam (2,203 lines) → depends on core
  ├── ClamTree (divisive hierarchical clustering)
  ├── Search (ρ-NN, DFS Sieve k-NN)
  └── Compress (unitary + recursive panCAKES)

rustynum-oracle (8,826 lines) → depends on core (NOT clam)
  ├── Sweep (holographic vector algebra, 3 bases)
  ├── Organic WAL (incremental learning)
  ├── Ghost Discovery (template generation)
  ├── Recognize (LSH projection + recognition)
  ├── NARS (reverse causality, Granger, contradiction)
  └── AiWar Ghost (entity management)
```

Each crate has a single responsibility. No circular dependencies. Public APIs are exported through `lib.rs` with explicit `pub use` statements. The PRNG consolidation (PR #25) and WAL encapsulation are genuinely good — they eliminated real defect classes.

### The Bad: 5,343 Lines of Byte-Identical Duplication

Four files are copy-pasted across multiple crates with **identical checksums**:

| File | Copies | Lines × Copies | Duplicated Lines |
|---|---|---|---|
| carrier.rs | 3 (carrier, holo, focus) | 1,121 × 3 | 2,242 |
| focus.rs | 2 (focus, holo) | 1,330 × 2 | 1,330 |
| phase.rs | 3 (carrier, holo, archive-v3) | 682 × 3 | 1,364 |
| cogrecord_v3.rs | 2 (carrier, archive-v3) | 407 × 2 | 407 |
| **Total** | | | **5,343** |

This is 11% of the entire repo. Any bug fix to carrier.rs must be applied in 3 places or the crates silently diverge. This is a maintenance timebomb.

**Fix**: Move shared code to `rustynum-core` or a new `rustynum-common` crate. The duplicated crates (holo, carrier, focus) should depend on it. One afternoon of work, eliminates a permanent maintenance hazard.

### The Ugly: 10 Crates That Don't Ship

Of 13 workspace crates, only 3 have received meaningful work in the PR #23–26 cycle. The others sit in the workspace consuming compile time:

| Crate | Lines | Status |
|---|---|---|
| rustynum-rs | 7,577 | Foundation array type. Stable but rarely touched. |
| rustynum-holo | 7,292 | ~3,500 lines are duplicates of carrier/focus/phase. holograph.rs (3,223 lines) is unique. |
| rustynum-focus | 3,621 | ~2,450 lines are duplicates. Remaining is unique focus logic. |
| rustynum-carrier | 2,272 | ~1,500 lines are duplicates. |
| rustynum-archive | 1,483 | Legacy. Contains original cogrecord + hdc + graph + projection. |
| rustynum-archive-v3 | 1,129 | ~1,000 lines are duplicates. |
| rustyblas | 3,164 | BLAS L1-L3. Has benchmarks. Independent of the holographic stack. |
| rustymkl | 1,378 | MKL FFT/LAPACK/VML bindings. Independent. |
| bindings/python | ~800 | PyO3 bindings for rustynum-rs array. |

Not a crisis — Cargo handles unused crates fine. But it inflates the mental model. New contributors see 13 crates and think the system is 13-crate complex. It's actually 3-crate complex.

---

## Open Debt Items

### 🔴 Critical: N5 — `debug_assert_eq!` in SIMD Paths

`rustynum_core::simd::hamming_distance()`, `hamming_batch()`, `dot_f32()`, `dot_f64()`, `axpy_f32()`, `axpy_f64()` — all 7 functions use `debug_assert_eq!` for length validation. In release builds, these asserts are elided.

**Current mitigation**: The CLAM wrapper (`HammingSIMD::distance()`) adds a hard `assert_eq!` before calling `hamming_distance()`. So the CLAM → core path is safe.

**Unmitigated paths**: Any future direct caller of `rustynum_core::simd::hamming_distance()` that doesn't add its own assert — including callers in holo, carrier, etc. — would silently pass mismatched lengths to AVX-512 intrinsics that load 64-byte chunks via `_mm512_loadu_si512`. With mismatched lengths, the shorter buffer gets indexed with the longer's chunk count → **reads past allocation** → undefined behavior.

**Why this is still red**: The `unsafe` function `hamming_vpopcntdq` trusts its caller to have validated lengths. The public `hamming_distance` is the validation point. Using `debug_assert_eq!` at that point means the validation disappears in the binary that actually ships. The cost of `assert_eq!` is one comparison per call — negligible compared to the SIMD work.

**Fix**: `s/debug_assert_eq!/assert_eq!/` in all 7 functions. 5-minute change.

### 🟡 Architecture: simd.rs / simd_avx2.rs Duplication

`simd.rs` (594 lines, AVX-512) and `simd_avx2.rs` (429 lines, AVX2) export the same 13 public functions. They're compiled via mutually exclusive feature gates:

```rust
#[cfg(feature = "avx512")]
pub mod simd;
#[cfg(all(feature = "avx2", not(feature = "avx512")))]
#[path = "simd_avx2.rs"]
pub mod simd;
```

The scalar fallback logic, hamming_batch unrolling, and hamming_top_k partial-sort are identical between them. Only the intrinsic inner loops differ. This is ~300 lines of structural duplication.

Not urgent — the feature gates prevent both from compiling simultaneously — but any new SIMD function must be added in two places.

### 🟡 Build: CLAM's Silent Feature Dependency

`rustynum-clam/Cargo.toml` depends on `rustynum-core` without `default-features = false`:

```toml
rustynum-core = { path = "../rustynum-core" }
```

This silently inherits `default = ["avx512"]`, so CLAM always gets the simd module. But CLAM directly calls `rustynum_core::simd::hamming_distance` in `HammingSIMD::distance()` without any `#[cfg]` guard. If someone builds CLAM with `--no-default-features`, the simd module doesn't exist and compilation fails.

Compare with oracle, which does it correctly:
```toml
rustynum-core = { path = "../rustynum-core", default-features = false }
avx512 = ["rustynum-core/avx512"]
```

**Fix**: Either add `default-features = false` + explicit feature forwarding to CLAM's Cargo.toml (like oracle does), or add `#[cfg]` guards around the SIMD calls. 10-minute fix.

### 🟡 Dead Code: `Fingerprint::from_words()` and `from_word_slice()`

PR #25 added these constructors to enable the `hamming_64k` → `Fingerprint64K` delegation. PR #26 reverted that delegation (16KB copy regression). Now both constructors have zero callers outside tests.

Not harmful — dead code in a library is fine until someone assumes it's maintained. Flag for removal if they don't gain callers in the next sprint.

### 🟢 Remaining from Debt Ledger

| Item | Status | Notes |
|---|---|---|
| N2 (recognize.rs ≠ Fingerprint) | 🟡 Partial | `hamming_64k` was reverted to inline loop. Projection still uses `Vec<u64>` not `Fingerprint<1024>`. |
| N6 (3 Hamming type signatures) | 🟡 Partial | &[u8] / [u64; N] / &[u64] paths still exist. hamming_64k revert means no type unification. |
| N9 (Flat confidence threshold) | 🟢 Deferred | `reverse_trace()` uses 0.35 flat, CRP integration planned for later. |
| N11 (Clone-per-hop) | 🟢 Latent | Fine at current usage. Flag if reverse_trace enters hot loop. |

---

## What's Actually Good

The active codebase (14K lines) is well-structured research code. Specifics worth noting:

**CLAM tree.rs** (898 lines): Clean implementation of the CHESS divisive hierarchical clustering algorithm. `hamming_inline()` is hand-unrolled 4× and processes 32 bytes per iteration — this is likely faster than the rustynum_core scalar fallback and comparable to AVX2 for small inputs. The delta_plus/delta_minus triangle inequality primitives are correct.

**compress.rs** (656 lines): Full panCAKES implementation with both unitary and recursive compression modes. The bottom-up cost comparison correctly chooses min(unitary, recursive) per cluster. `hamming_from_query()` avoids full decompression for distance computation — this is the real optimization, not the code path but the algorithmic skip.

**search.rs** (480 lines): DFS Sieve k-NN with proper min-heap/max-heap dual structure. The pruning logic correctly uses delta_plus/delta_minus bounds. `rho_nn` implements exact CHESS ρ-NN with triangle inequality pruning.

**nars.rs** (617 lines): Clean algebra. Binary unbind is exact (XOR self-inverse). Unsigned uses `rem_euclid` correctly. Signed acknowledges saturation approximation. The Granger convention is now consistent (fixed in PR #26). Tests verify noise floor detection — wrong-role unbinding gives normalized distance ~0.5, correctly triggering the confidence threshold.

**SplitMix64** (118 lines): Passes BigCrush, single u64 state, `next_gaussian()` via Box-Muller. Replaced 5 independent PRNGs with different statistical properties. Real improvement.

**organic.rs** (2,061 lines): The WAL design — private fields with `template()`, `template_norm()`, `update_template()` accessors — prevents the stale-norm bug that was possible with direct field access. `update_template()` atomically recomputes the norm whenever the template changes.

---

## Priority Actions

### Immediate (today)

1. **N5**: `debug_assert_eq!` → `assert_eq!` in 7 SIMD functions. 5 minutes. Prevents potential UB via mismatched-length buffer overread. The one change that matters for correctness.

### This week

2. **CLAM feature gates**: Add `default-features = false` + feature forwarding to `rustynum-clam/Cargo.toml`. 10 minutes. Prevents compilation failure on non-default feature sets.

3. **Decide on dead Fingerprint constructors**: Either find a use for `from_words`/`from_word_slice` or remove them. 5 minutes either way.

### Next sprint

4. **Deduplicate carrier/focus/phase/cogrecord**: Extract shared code to a common crate. Eliminates 5,343 duplicated lines. One afternoon.

5. **CLAM completeness** (from debt ledger): BFS Sieve, improved child pruning, recursive decompression chain walk, CompressedSearch adapter. 4-5 days.

### Eventually

6. **SIMD module unification**: Extract common scalar/batch/top-k logic from simd.rs and simd_avx2.rs into a shared core, keeping only the intrinsic inner loops in the feature-gated files. Eliminates ~300 lines of structural duplication.

7. **Workspace cleanup**: Consider whether holo/carrier/focus/archive-v3 should be active workspace members or moved to an `archive/` directory outside the workspace.
