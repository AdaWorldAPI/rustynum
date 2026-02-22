# PR #27 Full Workspace Review — Brutally Honest

> **Scope**: All 5 crates: rustynum-core, rustynum-oracle, rustynum-clam, rustynum (Python bindings), rustynum-arrow
> **Date**: 2026-02-22
> **Post-merge**: PRs #23–#26 all on main
> **Methodology**: Line-by-line read of every `.rs` file, cross-referenced agent claims against source

---

## Executive Summary

The codebase is structurally sound for single-threaded research use. But there are
**3 correctness bugs that produce wrong results**, **2 soundness holes** in unsafe code,
**2 PyO3 panics** that crash the Python interpreter, and a pervasive pattern of
`.unwrap()`/`assert!()` where `Result<>` is appropriate. The CLAM and oracle crates
are the strongest; the Python bindings and Arrow bridge are the weakest.

**Verdict**: Not production-ready. Fine for research prototyping, but any path to
deployment needs the items below fixed first.

---

## N13–N30: New Debt Items

### Correctness Bugs

| # | Severity | Crate | File:Line | Issue |
|---|----------|-------|-----------|-------|
| **N13** | 🔴 | rustynum | `bindings/python/src/array_f32.rs:96` | **f32 `mean_axis(None)` calls `self.inner.mean()` instead of `self.inner.mean_axis(None)`.** The f64 version (array_f64.rs:94) correctly calls `mean_axis(None)`. So `arr_f32.mean_axis(None)` and `arr_f64.mean_axis(None)` return different shapes — one is a scalar, the other an array. API contract violation. |
| **N14** | 🔴 | rustynum-core | `prefilter.rs:197` | **`pruned_gemm_rows` percentile threshold is inverted.** `thresh_idx = (m * threshold_percentile) as usize` on ascending-sorted norms. With `threshold_percentile=0.9`, `thresh_idx=90` picks the 90th smallest value (10th percentile). Rows "above threshold" then keeps ~10% of rows. The comment says "0.9 = keep top 10%" — so the *behavior* is correct, but the *parameter name* `threshold_percentile` misleads callers who'd read 0.9 as "90th percentile = keep top 90%". Rename to `prune_fraction` or invert the math. |
| **N15** | 🟡 | rustynum-arrow | `datafusion_bridge.rs:69-70` | **`cascade_scan_4ch` assumes all 4 columns have identical row count and value_length.** Only `meta_col.len()` and `meta_col.value_length()` are read. If `cam_col` has fewer rows, the slice `&cam_flat[offset..offset+vec_len]` panics or reads garbage. Add `assert_eq!` for all 4 column dimensions at function entry. |
| **N16** | 🟡 | rustynum-oracle | `ghost_discovery.rs:364,379-385` | **Hardcoded concept indices are fragile.** `(16..24).collect()` for "sens domain" and `vec![1, 33, 19, 6, 38]` for navigation scenario. If anyone reorders `CONCEPTS[]`, these become silently wrong. Build indices from name lookups instead. |

### Soundness / Safety

| # | Severity | Crate | File:Line | Issue |
|---|----------|-------|-----------|-------|
| **N17** | 🔴 | rustynum-core | `blackboard.rs:257,286` | **`borrow_2_mut_f32` / `borrow_3_mut_f32` take `&self` but return `&mut` slices.** The runtime `assert_ne!` prevents aliasing the *same* buffer, and different buffers are different heap allocations, so disjointness holds. **But**: nothing stops two threads from calling `borrow_2_mut_f32` concurrently on the same Blackboard (it's `Send` per line 382, so ownership can transfer). If thread A holds `&mut` to buffer "X" and thread B calls `borrow_2_mut_f32("X", "Y")`, thread B gets a second `&mut` to "X" — two `&mut` to the same memory = instant UB. The `&self` receiver means Rust's borrow checker doesn't prevent this. **Fix**: either (a) take `&mut self` (prevents concurrent access), (b) remove `Send`, or (c) add internal `Mutex`/`RwLock` per buffer. |
| **N18** | 🟡 | rustynum | `bindings/python/src/functions.rs:29-32` | **`assert!()` in PyO3 function panics → crashes Python interpreter.** `matmul_f32` and `matmul_f64` use `assert!` to validate input shapes. Panics in PyO3 kill the process. Replace with `return Err(PyTypeError::new_err(...))`. |
| **N19** | 🟡 | rustynum-clam | `compress.rs:341-344` | **`decompress_point()` indexes `data[center_idx * vec_len..]` without bounds check.** If `encoding_centers[point_idx]` is corrupt or out of range, this panics with an unhelpful message. Add `assert!(center_idx < data.len() / vec_len)`. |
| **N20** | 🟡 | rustynum-clam | `search.rs:203` | **`knn_repeated_rho` divides by `root.cardinality as u64` which panics if cardinality=0.** The `if rho == 0` guard on line 204 runs *after* the division. Add early return for empty tree. |

### Performance

| # | Severity | Crate | File:Line | Issue |
|---|----------|-------|-----------|-------|
| **N21** | 🟡 | rustynum-arrow | `datafusion_bridge.rs:119-136` | **`hamming_slice` is scalar u64 popcount, not SIMD.** Comment says "processes 8 bytes at a time" but that's just a u64 loop. For 2048-byte containers, this misses 4–8x from AVX-512 VPOPCNTDQ. Should delegate to `rustynum_core::simd::hamming_distance`. |
| **N22** | 🟡 | rustynum-clam | `compress.rs:373-399` | **DistanceCache uses `Vec<(usize, u64)>` with linear scan.** For large trees this is O(n) per lookup. Replace with `HashMap<usize, u64>`. |
| **N23** | 🟢 | rustynum-clam | `tree.rs:469-472` | **LFD `half_radius = radius / 2` uses integer division.** For odd Hamming radii, this rounds down, undercounting points in the inner ball and underestimating LFD. Use `d * 2 <= radius` instead. |
| **N24** | 🟢 | rustynum | `bindings/python/src/functions.rs` | **Unnecessary `Python::with_gil()` wrappers.** ~10 functions (zeros, ones, full, dot) call `with_gil` but never touch Python objects. Adds overhead for no reason. |

### API Design / Naming

| # | Severity | Crate | File:Line | Issue |
|---|----------|-------|-----------|-------|
| **N25** | 🟡 | rustynum-core | `prefilter.rs:196` | **`partial_cmp().unwrap()` in sort panics on NaN.** `approx_row_norms_f32` can produce NaN if input contains Infinity. Use `.unwrap_or(Ordering::Equal)`. |
| **N26** | 🟡 | rustynum-arrow | `arrow_bridge.rs:97-118` | **9 `.expect()` calls hide real errors.** `cogrecords_to_record_batch` and `record_batch_to_cogrecords` panic on malformed input instead of returning `Result`. No way for callers to handle gracefully. |
| **N27** | 🟢 | rustynum-core | `blackboard.rs:44-45,161-175` | **`BufferHandle` is dead code.** Allocated and stored but never returned or used by any public API. Remove or expose. |
| **N28** | 🟢 | rustynum | `bindings/python/src/lib.rs:3-296` | **~300 lines of commented-out I32/I64 PyO3 stubs.** Dead code. Delete or gate behind a feature flag. |
| **N29** | 🟢 | rustynum-arrow | `datafusion_bridge.rs:14` | **Doc references `hamming_search_adaptive` which doesn't exist.** Copy-paste from another codebase. |
| **N30** | 🟢 | rustynum-clam | `compress.rs:305` | **`pruned_subtrees: 0` is always zero.** The `// TODO: count pruned subtrees` has not been implemented, so the stat is a lie. |

---

## Previously-Reported Items: Status Update

| # | Status | Notes |
|---|--------|-------|
| N1 | ✅ CLOSED | PR #25 consolidated PRNGs |
| N3 | ✅ CLOSED | PR #25 added accessors |
| N7 | ✅ CLOSED | PR #26 fixed Granger sign convention |
| N8 | ✅ CLOSED | PR #26 renamed Contradiction → SimilarPair |
| N10 | ✅ CLOSED | PR #26 renamed hamming_i8 → symbol_distance |
| N12 | ✅ CLOSED | PR #26 reverted 16KB stack copy regression |
| N2 | 🟡 OPEN | recognize.rs still uses ad-hoc `&[u64]`, not `Fingerprint<1024>` |
| N5 | 🟡 OPEN | `debug_assert_eq!` in HammingSIMD still not upgraded |
| N6 | 🟡 OPEN | Three Hamming distance type signatures still divergent |
| N9 | 🟢 OPEN | Flat confidence threshold in reverse_trace |
| N11 | 🟢 OPEN | Clone-per-hop in reverse_trace |

---

## Priority Fix Order

### Immediate (< 1 hour, prevents wrong results / UB)

1. **N17** — Blackboard `Send` + `&self` → `&mut` soundness hole. Either remove `Send` or take `&mut self`.
2. **N13** — f32 `mean_axis(None)` wrong dispatch. One-line fix: `self.inner.mean()` → `self.inner.mean_axis(None)`.
3. **N18** — PyO3 `assert!()` → `PyErr`. Replace 2 asserts with `return Err(...)`.
4. **N15** — `cascade_scan_4ch` column dimension validation. Add 6 `assert_eq!` lines.
5. **N20** — `knn_repeated_rho` div-by-zero guard. Add `if root.cardinality == 0 { return empty }`.

### Short-term (< 1 day, improves correctness)

6. **N14** — Rename `threshold_percentile` to `prune_fraction` and add doc example.
7. **N25** — NaN-safe sort in `pruned_gemm_rows`.
8. **N16** — Replace hardcoded concept indices with name lookups.
9. **N19** — Bounds check in `decompress_point`.
10. **N5** — Upgrade `debug_assert_eq!` → `assert_eq!` in HammingSIMD (still open from PR #25).

### Medium-term (next sprint)

11. **N21** — Wire `hamming_slice` to `rustynum_core::simd::hamming_distance`.
12. **N22** — Replace linear DistanceCache with HashMap.
13. **N26** — Convert Arrow bridge functions to return `Result`.
14. **N6** — Unify Hamming distance type signatures across crates.

### Cleanup (when convenient)

15. **N27** — Remove dead BufferHandle.
16. **N28** — Remove commented-out I32/I64 stubs.
17. **N24** — Remove unnecessary `with_gil()` wrappers.
18. **N29, N30** — Fix stale doc reference; implement pruned_subtrees counter.

---

## What's Actually Good

Credit where due — these are solid:

- **CLAM tree build + search** — Algorithms match papers exactly. LFD, ρ-NN, DFS sieve, repeated-ρ all correct.
- **panCAKES compression framework** — Unitary + recursive modes, mixed-mode tree, compressive distance primitive. Architecture is right.
- **Fingerprint<N> const-generic type** — Clean API, correct XOR/popcount, similarity/distance methods.
- **SplitMix64 consolidation** — 5 copies → 1 canonical impl. Clean.
- **NARS reverse causality** — Granger signal now correct (PR #26). Symbol distance properly named. Causal traces work.
- **AVX-512 VPOPCNTDQ Hamming** — Properly gated behind `target_feature`, safe intrinsics, correct 4x ILP unrolling.
- **Blackboard split-borrow API** — The *concept* is sound (UnsafeCell + runtime name checks + disjoint heaps). Just needs the `Send` issue fixed.

---

*N13–N30 extend the debt ledger. Items N7, N8, N10, N12 are now closed by PR #26.*
