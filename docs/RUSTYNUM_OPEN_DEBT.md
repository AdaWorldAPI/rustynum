# rustynum Open Debt — Post PR #36

**Date**: 2026-02-22 | **Main**: post PR #36 merge | **Status**: 0 open issues, 0 open PRs

---

## All Original Tracked Items Closed

30 items (D1–D9, N1–N30) resolved across PRs #23–#36:

| PR Range | Closed | Highlights |
|---|---|---|
| #23–#24 | D1–D9 | Parallel writes, blackboard soundness, GEMM perf, Fingerprint, DeltaLayer, HammingSIMD |
| #25–#26 | N1, N3, N7, N8, N10, N12 | PRNG consolidation, WAL encapsulation, Granger docs, naming |
| #27 | N2, N4–N6, N9, N11, N13–N30 | Full workspace sweep |
| #29 | N13–N15, N17–N20, N22 | 8 correctness/soundness fixes |
| #35–#36 | CPUID hoisting, batch stroke, PreciseMode | HDR cascade architecture |

---

## Open Items

### Yellow (should file)

| # | Location | Issue |
|---|---|---|
| U1 | `nars.rs:46-50` | Signed `unbind` lossy when saturation occurs. Test only covers non-saturating range. |
| U2 | `recognize.rs:469,792` `organic.rs:859` | `partial_cmp().unwrap()` panics on NaN. 3 sites. |
| **P36-1** | `bindings/python/src/array_u8.rs` | `hdr_search`, `hdr_search_f32`, `hdr_search_delta` — no input validation. Rust assert_eq panic → Python crash. Same N18 pattern. |
| N2 | `rustynum-oracle/src/recognize.rs` | `Vec<u64>` not `Fingerprint<1024>` |
| N6 | Multiple | 6 Hamming implementations (including `approx_hamming_candidates` scalar) |
| N16 | `ghost_discovery.rs` | Hardcoded concept indices |
| N21 | Arrow bridge | `hamming_slice` still scalar |

### Green (fix when touching file)

| # | Location | Issue |
|---|---|---|
| N9 | reverse_trace | Flat confidence threshold (not CRP-calibrated) |
| N11 | reverse_trace | Clone-per-hop (16KB × depth) |
| N23 | LFD | Integer division rounding |
| N24 | Python bindings | Unnecessary with_gil() |
| N26 | Arrow bridge | expect() chains |
| N27–N30 | Various | Dead code, stale docs, pruned_subtrees always zero |
| U3 | recognize.rs:108 | 64K projector 130MB–1GB memory (by design) |
| U4 | recognize.rs:1113 | learn_improves asserts >= 0 on usize (tautology) |
| P36-2 | prefilter.rs | `approx_hamming_candidates` scalar popcount (test-only) |
| P36-3 | simd.rs | s1_bytes not 64-byte aligned for non-power-of-2 |
| P36-4 | simd.rs | F32 dequantize loop scalar (fine for ~200 finalists) |

---

## Forward Work (Not Debt)

See `RUSTYNUM_DEBT_LEDGER.md` sections 6–8 for CLAM completeness, panCAKES, ladybug integration, 34 NARS tactics.
