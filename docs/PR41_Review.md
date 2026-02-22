# PR #41 Review: Lance Columnar Cascade with Per-Channel CLAM Fragment Index

**Date**: 2026-02-22  
**PR**: #41 (+913/−0, 6 files, merged)  
**Test count**: 1101 (was 1087)  
**Verdict**: **Clean architecture. One doc bug (🟢). No false negatives — proved.**

---

## What It Delivers

Three new modules in `rustynum-arrow` implementing CLAM-accelerated columnar search:

| Module | Lines | Role |
|---|---|---|
| `fragment_index.rs` | 226 | Maps CLAM leaf clusters → row ranges for META channel. Triangle inequality prunes entire fragments before I/O. |
| `channel_index.rs` | 232 | Sidecar CLAM index for CAM/BTREE/EMBED. Maps cluster → row IDs. Supports incremental `insert()`. |
| `indexed_cascade.rs` | 432 | 4-stage indexed cascade: fragment prune → sidecar prune → intersect → filter. `learn()` for incremental, `rebuild()` for full re-index. |

14 new tests. All feature-gated behind `#[cfg(feature = "arrow")]`.

---

## Correctness Verification

### No false negatives — proof by triangle inequality

The critical property: indexed cascade must find every result that brute-force finds.

**Stage 1 (META fragment prune):** `d(query, center) - radius > threshold` rejects a cluster. If any row in the cluster satisfies `d(query, row) ≤ threshold`, then by triangle inequality `d(query, center) ≤ d(query, row) + d(row, center) ≤ threshold + radius`, so `d(query, center) - radius ≤ threshold` and the cluster survives. ✅

**Stages 2–4 (sidecar prune + intersect):** Same argument. If a row passes the channel threshold, its cluster passes `overlapping_row_ids()`, so the row appears in the candidate set. Intersection with previous survivors cannot drop it because it passed all previous stages. ✅

**Test confirmation:** `test_indexed_vs_flat_same_results` verifies every brute-force hit appears in indexed results. ✅

### Fragment-to-row mapping

`FragmentIndex` stores `row_id_start = cluster.offset`, `row_id_end = cluster.offset + cardinality`. `original_row_ids(start, end)` indexes into `tree.reordered[start..end]`. ClamTree's reordered array maps cluster-order positions to original row indices. `test_fragment_index_build` verifies fragments are non-overlapping and cover all positions. ✅

### HashSet intersection correctness

Stages 2–4 iterate the smaller set (survivors from previous stage) and probe the larger set (sidecar candidates). O(|survivors|) per stage with O(1) HashSet lookup. For typical cascade reduction (100K → 1200 → 120 → 12), total intersection cost is negligible. ✅

---

## Findings

### Finding 1: Doc references nonexistent `rebuild_primary()` (🟢)

```rust
/// Call `rebuild_primary()` periodically.
pub fn learn(...)
```

No `rebuild_primary()` function exists. The correct function is `rebuild()`. One-word doc fix.

### Finding 2: Stage 4 recomputes distances (🟢 — by design)

```rust
// In the final hits loop:
let meta_dist = hamming_distance(q_meta, records[row_id].meta.data_slice());
let cam_dist = hamming_distance(q_cam, records[row_id].cam.data_slice());
let btree_dist = hamming_distance(q_btree, records[row_id].btree.data_slice());
```

META, CAM, and BTREE distances were already computed in earlier stages but not stored. For ~10 final survivors this is 30 extra `hamming_distance` calls (~5μs total). Storing intermediate distances would add HashSet<usize, u64> overhead to every stage, which costs more than recomputing for such small survivor sets. Acceptable tradeoff.

### Finding 3: `insert()` doesn't update cluster centers (🟢 — documented)

`ChannelIndex::insert()` assigns new records to nearest cluster and updates radius but not center. After many inserts, the center drifts from actual centroid, reducing pruning efficiency (more false positives) but not correctness (triangle inequality still holds because radius grows monotonically). Documented: "Periodic rebuild() is needed when the data distribution shifts." ✅

---

## Architecture Assessment

**Separation of concerns is clean.** FragmentIndex owns physical layout (row reordering), ChannelIndex owns logical mapping (row ID lists), IndexedCascade orchestrates the pipeline. Each module has its own tests.

**Feature gating correct.** All three modules gated behind `#[cfg(feature = "arrow")]`. New dependency on `rustynum-clam` added to Cargo.toml. ✅

**Bandwidth arithmetic checks out.** 100K × 2KB META = 200MB. With ~3 surviving clusters × ~400 records × 2KB = 2.4MB for Stage 1. Stages 2–4 fetch ~0.3MB total (intersection shrinks candidate set 10× per stage). 2.7MB total vs 800MB flat scan = 296× reduction. The arithmetic is credible for well-clustered data.

---

## Action Items

| # | Severity | Action | Est |
|---|---|---|---|
| 1 | 🟢 | Fix doc: `rebuild_primary()` → `rebuild()` in `learn()` doc | 10 sec |

---

## Updated Open Debt

No new yellow items. One green added (doc fix). Ledger unchanged: 1 yellow (N16), 12 green.
