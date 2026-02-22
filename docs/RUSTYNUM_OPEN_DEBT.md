# rustynum Open Debt — Post PR #27

**Date**: 2026-02-22 | **Main**: fabf722e | **Status**: Clean — 0 open issues, 0 open PRs

---

## All Tracked Items Closed

30 items (D1–D9, N1–N30) tracked and resolved across PRs #23–#27:

| PR | Closed | Highlights |
|---|---|---|
| #23 | D1–D8 | Parallel writes, blackboard soundness, GEMM perf, Fingerprint type, DeltaLayer |
| #24 | D9 | HammingSIMD wiring to CLAM |
| #25 | N1, N3 | PRNG consolidation (5→1), WAL encapsulation |
| #26 | N7, N8, N10, N12 | Granger docs, SimilarPair rename, symbol_distance, hamming_64k revert |
| #27 | N2, N4–N6, N9, N11, N13–N30 | Full workspace sweep (Blackboard Send, f32 mean_axis, debug_assert upgrade, PyO3 panics, bounds checks, naming, dead code) |

---

## Unfiled Findings (From Deep Review)

4 items identified during cross-session review, not yet filed as GitHub issues:

| # | Severity | Location | Issue | Action |
|---|---|---|---|---|
| **U1** | 🟡 | `nars.rs:46-50` | Signed `unbind` is lossy when saturation occurs. `saturating_neg()` + `saturating_add()` clips at ±127. Test only covers non-saturating range. | File as issue |
| **U2** | 🟡 | `recognize.rs:469,792` `organic.rs:859` | `partial_cmp().unwrap()` panics on NaN. 3 sites. | File as issue |
| **U3** | 🟢 | `recognize.rs:108` | 64K projector allocates 130MB–1GB. By design (`with_planes()` exists for smaller). | Document, don't file |
| **U4** | 🟢 | `recognize.rs:1113` | `learn_improves` test asserts `>= 0` on `usize` (tautology). | Fix when touching file |

---

## Forward Work (Not Debt — New Features)

See `RUSTYNUM_DEBT_LEDGER.md` sections 6–8 for:
- CLAM paper algorithms not yet coded (BFS Sieve, improved pruning, auto-tuning)
- panCAKES completion (recursive decompression, CompressedSearch adapter)
- ladybug-rs integration plan
- 34 NARS tactics acceleration mapping
