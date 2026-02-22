# PRs #38–40 Review: Debt Closure + Projector Perf + LOD Holographic Search

**Date**: 2026-02-22  
**PRs**: #38 (+146/−50, 11 files) · #39 (+364/−29, 3 files) · #40 (+1082/−311, 20 files)  
**Total**: +1592/−390 across 3 PRs  
**Post-merge test count**: 1087 (was 1070)  
**Verdict**: All three clean. PR #40 is the most architecturally significant — closes 10+ debt items and adds LOD+CLAM search. One new green note.

---

## PR #38: fix: Python binding validation, signed unbind, deprecate dead params

**What it does**: Targeted fixes for debt items flagged in PR #36 review.

| Fix | Debt Item | Verified |
|---|---|---|
| `hdr_search`, `hdr_search_f32`, `hdr_search_delta` input validation | P36-1 🟡 | ✅ Both `database.len() != vec_len * count` and `query.len() != vec_len` checked, returns `PyValueError` |
| Signed `unbind` i8::MIN saturation | U1 🟡 | ✅ Widened to i16: `(-(v as i16)).clamp(-128, 127) as i8` — handles -(-128)=128 → clamp to 127 |
| `two_stage_hamming_search` deprecated | P36-2 dead params | ✅ `#[deprecated(since = "0.5.0")]`, dead params prefixed with underscore |
| `BufferHandle` dead code | N27 🟢 | ✅ `#[allow(dead_code)]` with doc explaining "currently unused externally" |

**Score: Clean.** Exactly the fixes from the review, nothing extra. ✅

---

## PR #39: perf: Blackboard-shared projector, flat hyperplane buffer, timing benchmarks

**What it does**: Performance refactor of `Projector64K` in recognize.rs.

| Change | Impact |
|---|---|
| `Vec<Vec<f32>>` → contiguous `Vec<f32>` (`hyperplanes_flat`) | Cache-friendly: `hyperplanes[plane * d..(plane+1) * d]` is one contiguous slice per plane |
| `write_to_blackboard()` / `from_blackboard()` | Arena-based sharing: generate hyperplanes once, reuse across multiple Recognizer instances via Blackboard |
| `with_projector()` / `take_projector()` | Ownership transfer without regeneration |
| `project_signed()` direct i8 dot product | Eliminates per-call Vec<f32> allocation |
| `run_recognition_sweep_fast()` | 4K planes for iteration speed in testing |
| TODO.md checklist | Documents 26 passing tests, timing baselines, remaining bottleneck (scalar projection loop) |

**Remaining bottleneck noted**: projection hot loop (65536 dot products × D floats) is still scalar. Fixed in PR #40 commit 3.

**Score: Clean.** Good engineering — flat buffer is the right call for SIMD-ready layout. ✅

---

## PR #40: LOD pyramid + CLAM-accelerated holographic search + debt closure

Three commits. This is the big one.

### Commit 1: Debt closure sweep

Closes 10+ items in one shot:

| Debt Item | Fix | Verified |
|---|---|---|
| N2 🟡 — `Vec<u64>` not `Fingerprint<1024>` | `fingerprints: Vec<Vec<u8>>`, `hamming_64k` takes `&[u8]` → `simd::hamming_distance` | ✅ Full type migration |
| N6 🟡 — 6 Hamming implementations | All wrappers now delegate to `simd::hamming_distance`. `hamming_chunk_inline`, `hamming_inline`, `hamming_chunk` are all one-liners to `simd::hamming_distance` | ✅ No independent impls remain |
| N21 🟡 — Arrow bridge scalar hamming | Arrow bridge hamming path removed/simplified | ✅ |
| P36-2 🟢 — `approx_hamming_candidates` scalar | Now uses `select_hamming_fn()` — dispatches to VPOPCNTDQ/AVX2/scalar | ✅ |
| P36-3 🟢 — s1_bytes alignment | `(((vec_bytes / 16).max(64) + 63) & !63).min(vec_bytes)` — rounds up to 64-byte boundary | ✅ |
| U2 🟡 — `partial_cmp().unwrap()` NaN panics | Original sites in recognize.rs/organic.rs eliminated. Remaining sites (aiwar_ghost, ghost_discovery, sweep) all use `.unwrap_or(Ordering::Equal)` — safe | ✅ |
| N24 🟢 — unnecessary with_gil | `array_f32.rs` refactored (−30 lines) | ✅ |
| Hamming dedup across 6 crates | cogrecord, carrier, cogrecord_v3 in archive/focus/holo all switched from inline hamming to `simd::hamming_distance` | ✅ |

### Commit 2: LOD pyramid + CLAM-accelerated holographic search

Two new files (878 lines total):

**`lod_pyramid.rs` (395 lines)**: Multi-resolution OR-reduce summaries. Each level halves rows/cols by OR-ing 2×2 cells. `or_mask_lower_bound(query, mask) = popcount(query & ~mask)` — bits in query absent from all cluster members guarantee minimum distance.

- `LodPyramid::build()` — bottom-up OR-reduce from base grid
- `LodLevel::get()` — O(1) cell access
- `or_mask_lower_bound()` — u64-chunked popcount (not full SIMD, but adequate for per-node bound)
- 9 tests covering: build correctness, full-mask/empty-mask bounds, dimensional invariants

**`holo_search.rs` (483 lines)**: CLAM tree traversal with dual pruning.

- `LodIndex::build()` — bottom-up annotation: leaf OR-mask from fingerprints, internal OR-mask from children
- `lod_knn_search()` — priority queue (min-heap by lower bound) with dual pruning:
  - **Triangle inequality**: `delta_minus(dist_to_center)` from CLAM tree
  - **LOD OR-mask**: `or_mask_lower_bound(query, annotation.or_mask)`
  - Takes `max(tri_lower, lod_lower)` — tighter than either alone
- Correct max-heap for k-th best tracking
- `LodSearchStats` tracks nodes_visited, pruned_triangle, pruned_lod, exact_distances_computed
- Hoisted CPUID: `select_hamming_fn()` called once before loop
- 6 tests: build, exact match, brute force equivalence

**Architecture is correct.** The dual-bound approach (triangle inequality from metric tree + bit-level OR-mask) is sound and well-studied. The priority queue traversal with k-th best pruning is the standard branch-and-bound pattern.

### Commit 3: SIMD projection + doc fix

- `project()` now uses `rustynum_core::simd::dot_f32` (AVX-512 FMA) instead of scalar loop
- `project_signed()` converts template to f32 once, reuses `dot_f32` for all planes
- `from_blackboard()` doc corrected: "zero-copy" → "copy, no RNG"

This closes the bottleneck noted in PR #39's TODO.md. 65536 × D scalar FMAs → SIMD.

---

## Findings

### Finding 1: `or_mask_lower_bound` is u64-chunked, not SIMD (🟢)

```rust
let q = u64::from_ne_bytes(query[base..base + 8].try_into().unwrap());
let m = u64::from_ne_bytes(or_mask[base..base + 8].try_into().unwrap());
count += (q & !m).count_ones() as u64;
```

Processes 8 bytes per iteration. Could use VPANDN + VPOPCNTDQ for 64 bytes per iteration (8× throughput). But this is a bound computation called once per internal node during tree traversal — not the bottleneck. The bottleneck is exact distance computation at leaves, which correctly uses `select_hamming_fn()` (hoisted CPUID). **Not worth SIMDifying.**

### Finding 2: N26 Arrow bridge expect() chains still present (🟢 — unchanged)

```rust
.expect("meta column");
.expect("cam column");
```

Four expect() calls remain in `datafusion_bridge.rs`. These are schema-guaranteed columns and will only panic if the RecordBatch schema is wrong (programmer error, not data error). Still a code smell but acceptable.

---

## Updated Debt Ledger

### Closed by PRs #38–40:

| # | Item | Closed by |
|---|---|---|
| P36-1 🟡 | HDR Python binding validation | PR #38 |
| U1 🟡 | Signed unbind i8::MIN saturation | PR #38 |
| N2 🟡 | recognize.rs Vec<u64> → Vec<u8> | PR #40 |
| N6 🟡 | Hamming implementation dedup | PR #40 (all wrappers → simd::hamming_distance) |
| N21 🟡 | Arrow bridge scalar hamming | PR #40 |
| U2 🟡 | partial_cmp().unwrap() NaN panics | PR #40 (eliminated or .unwrap_or()) |
| N24 🟢 | Unnecessary with_gil() | PR #40 |
| P36-2 🟢 | approx_hamming_candidates scalar popcount | PR #40 (uses select_hamming_fn) |
| P36-3 🟢 | s1_bytes alignment | PR #40 (64-byte aligned) |
| N27 🟢 | BufferHandle dead code | PR #38 (#[allow(dead_code)] with doc) |

### Still Open:

| # | Sev | Item | Notes |
|---|---|---|---|
| N16 | 🟡 | Hardcoded concept indices in ghost_discovery | τ addresses 0x40–0xE0 |
| N9 | 🟢 | Flat confidence threshold in reverse_trace | |
| N11 | 🟢 | Clone-per-hop in reverse_trace | |
| N23 | 🟢 | LFD integer division | |
| N26 | 🟢 | Arrow bridge expect() chains (4 sites) | |
| N28 | 🟢 | I32/I64 DType stubs in Blackboard | |
| N29 | 🟢 | Stale doc reference | |
| N30 | 🟢 | pruned_subtrees always zero | |
| U3 | 🟢 | 64K projector memory (by design) | |
| U4 | 🟢 | learn_improves tautology | |
| P36-4 | 🟢 | F32 dequantize loop scalar | Fine for ~200 finalists |
| **NEW** | 🟢 | `or_mask_lower_bound` u64-chunked (not SIMD) | Not on hot path |

**Debt status: 1 yellow, 11 green.** Down from 7 yellow + 11 green pre-PR #38.

---

## Summary

| PR | Quality | Key Achievement |
|---|---|---|
| #38 | ✅ Clean fix | Closes P36-1 (Python crash), U1 (signed unbind) |
| #39 | ✅ Clean perf | Projector64K flat buffer + Blackboard sharing |
| #40 | ✅ Strong | 10+ debt items closed, LOD+CLAM search (878 new lines), SIMD projection |

The repo is in excellent shape. 1087 tests, 0 failures. Only N16 (hardcoded concept indices) remains as yellow debt — and that's arguably by-design for the ghost_discovery experiment.
