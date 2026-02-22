# rustynum Consolidated Open Debt — Post PR #27

**Date**: 2026-02-22 | **Main**: fabf722e | **Sources**: Debt Ledger N1–N12, PR #27 Review N13–N30, State-of-Repo

---

## Closed (8 items)

N1 (PRNG), N3 (WAL), N7 (Granger docs), N8 (SimilarPair rename), N10 (symbol_distance rename), N12 (hamming_64k revert) — all via PRs #25–26.

---

## Open Items by Priority

### 🔴 Correctness / Soundness (fix before any deployment)

| # | Crate | Issue | Fix |
|---|---|---|---|
| **N17** | core | **Blackboard `Send` + `&self` → dual `&mut`**. `borrow_2_mut_f32` takes `&self` but returns `&mut` slices. With `unsafe impl Send`, two threads can get `&mut` to same buffer = UB. | Remove `Send`, or take `&mut self`, or add per-buffer lock |
| **N13** | python | **f32 `mean_axis(None)` dispatches to `mean()` not `mean_axis(None)`**. Returns scalar instead of array. f64 version is correct. | `self.inner.mean()` → `self.inner.mean_axis(None)` |
| **N5** | core | **7 SIMD functions use `debug_assert_eq!` for length validation**. Elided in release. AVX-512 path loads 64-byte chunks — mismatched lengths → reads past allocation. CLAM's wrapper adds its own hard assert, so current clam→core path is safe. | `debug_assert_eq!` → `assert_eq!` in 7 functions |

### 🟡 Bugs / Safety Issues

| # | Crate | Issue | Fix |
|---|---|---|---|
| **N18** | python | **`assert!()` in PyO3 matmul → crashes Python interpreter**. 2 sites. | Replace with `return Err(PyTypeError::new_err(...))` |
| **N20** | clam | **`knn_repeated_rho` divides by cardinality before zero check**. `root.radius / root.cardinality as u64` on line 203, `if rho == 0` guard on line 204 runs after. | Add `if root.cardinality == 0 { return empty }` before division |
| **N15** | arrow | **`cascade_scan_4ch` assumes 4 columns have identical dimensions**. Only reads meta_col dimensions. | Add `assert_eq!` for all 4 column lengths at entry |
| **N14** | core | **`pruned_gemm_rows` parameter name misleads**. `threshold_percentile=0.9` actually prunes 90%, keeps 10%. Behavior matches comment but parameter name is backwards. | Rename to `prune_fraction` |
| **N19** | clam | **`decompress_point()` no bounds check on center_idx**. Corrupt index → unhelpful panic. | Add `assert!(center_idx < data.len() / vec_len)` |
| **N25** | core | **`partial_cmp().unwrap()` panics on NaN in prefilter sort**. | `.unwrap_or(Ordering::Equal)` |
| **N16** | oracle | **Hardcoded concept indices in ghost_discovery**. `(16..24)` and `vec![1,33,19,6,38]` break silently if CONCEPTS reordered. | Build indices from name lookups |

### 🟡 Architecture / Performance

| # | Crate | Issue | Fix |
|---|---|---|---|
| **N2** | oracle | recognize.rs still uses `Vec<u64>`, not `Fingerprint<1024>` | Refactor projection path |
| **N6** | cross | Three Hamming type signatures: `&[u8]` / `[u64; N]` / `&[u64]` | Unify via trait or adapter |
| **N21** | arrow | `hamming_slice` is scalar, not SIMD | Delegate to `rustynum_core::simd::hamming_distance` |
| **N22** | clam | DistanceCache uses `Vec<(usize,u64)>` linear scan | Replace with `HashMap<usize,u64>` |
| **simd** | core | simd.rs (594 lines) / simd_avx2.rs (429 lines) duplicate ~300 lines of batch/top-k/scalar logic | Extract shared logic |
| **dup** | workspace | 5,343 lines byte-identical across carrier/holo/focus/archive-v3 | Extract to common crate |
| **dead FP** | core | `Fingerprint::from_words()` and `from_word_slice()` have zero callers outside tests after PR #26 revert | Remove or find use |

### 🟢 Cleanup

| # | Crate | Issue |
|---|---|---|
| **N9** | oracle | Flat 0.35 confidence threshold, not CRP-calibrated |
| **N11** | oracle | Clone-per-hop in reverse_trace (latent perf) |
| **N23** | clam | LFD `half_radius = radius / 2` integer rounding |
| **N24** | python | Unnecessary `with_gil()` in ~10 functions |
| **N26** | arrow | 9 `.expect()` calls instead of `Result` return |
| **N27** | core | `BufferHandle` dead code |
| **N28** | python | ~300 lines commented-out I32/I64 stubs |
| **N29** | arrow | Doc references nonexistent `hamming_search_adaptive` |
| **N30** | clam | `pruned_subtrees: 0` always zero (TODO not implemented) |

---

## Summary

| Category | Count | Items |
|---|---|---|
| 🔴 Correctness/Soundness | 3 | N17, N13, N5 |
| 🟡 Bugs/Safety | 7 | N18, N20, N15, N14, N19, N25, N16 |
| 🟡 Architecture/Perf | 7 | N2, N6, N21, N22, simd dup, workspace dup, dead FP |
| 🟢 Cleanup | 9 | N9, N11, N23–N30 |
| ✅ Closed | 8 | N1, N3, N7, N8, N10, N12 |
| **Total open** | **26** | |

The three red items (N17, N13, N5) are each under 10 minutes to fix. N17 is the most serious — actual UB possible under concurrent access.
