# rustynum Open Debt — Post PR #40

**Date**: 2026-02-22 | **Main**: post PR #40 merge | **Tests**: 1087, 0 failures

---

## All Original Tracked Items Closed

30 items (D1–D9, N1–N30) + 4 unfiled findings (U1–U4) + 4 PR #36 findings (P36-1 through P36-4) resolved across PRs #23–#40. Only N16 remains yellow.

---

## Open Items

### Yellow (1)

| # | Location | Issue |
|---|---|---|
| N16 | `ghost_discovery.rs` | Hardcoded concept τ addresses (0x40–0xE0). Arguably by-design for the 52-concept topology experiment. |

### Green (11 — fix when touching file)

| # | Location | Issue |
|---|---|---|
| N9 | nars.rs reverse_trace | Flat confidence threshold (not CRP-calibrated) |
| N11 | nars.rs reverse_trace | Clone-per-hop (16KB × depth) |
| N23 | clam/tree.rs | LFD integer division rounding |
| N26 | datafusion_bridge.rs | 4 expect() chains on schema columns |
| N28 | blackboard.rs | I32/I64 DType stubs (allocated but no typed getter) |
| N29 | Various | Stale doc reference |
| N30 | compress.rs | `pruned_subtrees` always zero |
| U3 | recognize.rs:108 | 64K projector 130MB–1GB memory (by design) |
| U4 | recognize.rs:1113 | `learn_improves` asserts >= 0 on usize (tautology) |
| P36-4 | simd.rs | F32 dequantize loop scalar (fine for ~200 finalists) |
| NEW | lod_pyramid.rs | `or_mask_lower_bound` u64-chunked, not SIMD (not on hot path) |

---

## Closed Since Last Update (PRs #38–40)

P36-1 (Python validation), U1 (signed unbind), N2 (Vec<u64>→Vec<u8>), N6 (hamming dedup), N21 (Arrow scalar), U2 (partial_cmp NaN), N24 (with_gil), P36-2 (approx_hamming SIMD), P36-3 (s1_bytes alignment), N27 (dead code).
