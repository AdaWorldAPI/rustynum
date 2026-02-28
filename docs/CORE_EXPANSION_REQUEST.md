# Core Expansion Request: Move Belichtungsmesser into rustynum-core

**Date**: 2026-02-28
**Author**: Claude (session `claude/implement-neural-network-research-owTKn`)

---

## Problem

`bnn.rs` lives in `rustynum-core`. The Belichtungsmesser cascade functions
(K0/K1 probes, TraversalStats, signal_quality, HDR classification, BF16
cold-refinement) live in `rustynum-bnn/src/belichtungsmesser.rs`.

Any downstream consumer (main binary, ladybug-rs, etc.) that uses BNN from
core also needs the Belichtungsmesser for traversal acceleration. They must
add `rustynum-bnn` as a separate dependency. This is wrong — all compute
should be in one crate (core) since all crates compile into ONE binary.

## What Should Move to rustynum-core

Move `belichtungsmesser.rs` (496 lines, 11 tests) into `rustynum-core/src/`:

| Function | What It Does | Uses From Core |
|----------|-------------|----------------|
| `k0_probe_conflict(a: u64, b: u64) -> u32` | 1 XOR+POPCNT spot meter | nothing (pure arithmetic) |
| `k1_stats_conflict(a: &[u64], b: &[u64]) -> u32` | 8 XOR+POPCNT zone meter | nothing (pure arithmetic) |
| `TraversalStats` (Welford) | Auto-adjusting sigma thresholds | nothing (pure math) |
| `signal_quality(summary: &GraphHV) -> f32` | Per-word popcount variance | `GraphHV` |
| `classify_hdr(k1: u32, stats: &TraversalStats) -> u8` | HDR hot/mid/cold/dark | `TraversalStats` |
| `bf16_refine_cold(query: &GraphHV, candidate: &GraphHV) -> u8` | BF16 structural_diff on cold | `GraphHV`, `bf16_hamming::structural_diff` |
| `filter_children(...)` | Full K0->K1->HDR->BF16 cascade | `GraphHV`, `bf16_hamming` |
| `hdr_beam_width(scores, base) -> usize` | HDR-aware beam width | `ChildScore` |
| `ChildScore` | Cascade result struct | nothing |

All dependencies are already in `rustynum-core`. Zero new deps needed.

## After Move

1. `rustynum-core/src/lib.rs` adds `pub mod belichtungsmesser;` and re-exports
2. `rustynum-bnn` becomes a thin re-export crate (or can be removed entirely)
3. Downstream consumers get everything from `rustynum-core` — one dep, one binary

## What rustynum-bnn Would Then Contain

After moving Belichtungsmesser to core, `rustynum-bnn` would contain only
re-exports. It could either:

- **A)** Be removed entirely (all BNN + Belichtungsmesser in core)
- **B)** Stay as a convenience facade that re-exports BNN + Belichtungsmesser
  from core (zero code, just `pub use`)

Option A is cleaner. Option B exists if external consumers want a
single-purpose crate name.

## No Code Duplication

The Belichtungsmesser functions are NEW code (not copies of anything in core).
They consume core types (`GraphHV`, `structural_diff`) without duplicating them.
Moving them to core is a location change, not a duplication risk.
