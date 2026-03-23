# Rustynum Retrofit Plan: ndarray HPC Features → Rustynum

> **Created**: 2026-03-22
> **Context**: ndarray fork has 55 HPC modules (880 tests) that rustynum lacks.
> This plan backports the Zeckendorf/golden-ratio and search features so
> rustynum remains valuable as a standalone SIMD substrate.

---

## Why Retrofit (Not Just Abandon)

1. **ladybug-rs depends on 6 rustynum crates** — migration takes time
2. **rustynum has battle-tested SIMD dispatch** (simd.rs, 2400 LOC, stable intrinsics)
3. **93K LOC of working code** — BNN, CLAM, Arrow bridge, organic plasticity
4. **Python bindings** via PyO3 — ndarray doesn't have this
5. **rustyblas GEMM beats ndarray** at 1024x1024 (111 vs 62 GFLOPS, 1.8x)

ndarray is the foundation; rustynum is the specialized muscle. Both survive.

---

## What ndarray Has That Rustynum Is Missing

### Tier A: Zeckendorf / Golden-Ratio Core (PRIORITY)

| ndarray module | LOC | What it does | Why rustynum needs it |
|---|---|---|---|
| `bgz17_bridge` | 399 | Golden-step octave averaging → Base17 (34-byte) patterns | Foundation for fenestration distance; bridges to lance-graph container format |
| `spo_bundle` | 1514 | Golden-shift (φ²) cyclic permutation for SPO bundling | 8Kbit + 16Kbit levels; pure binary; the golden-ratio primitive |
| `palette_distance` | ~500 | Pre-computed k×k L1 distance matrix, O(1) lookups | Makes CLAM search O(1) per comparison instead of O(d) |
| `layered_distance` | ~400 | Container → palette index extraction + TruthGate filter | O(1) distance via palette; powers parallel_search |
| `parallel_search` | 612 | HHTL + CLAM dual-path with TruthGate merge | The search strategy rustynum-clam completely lacks |
| `merkle_tree` | 521 | 8Kbit 3-level Merkle (typed Staunen) | Zeckendorf extension ready; 2× compression proxy |
| `vsa` | 727 | 10K-dim binary VSA (bind=XOR, bundle=majority, permute=shift) | Fibonacci-VSA bridge connecting to holo/carrier.rs frequencies |

### Tier B: Cognitive Pipeline (SECONDARY)

| ndarray module | LOC | What it does | Status in rustynum |
|---|---|---|---|
| `crystal_encoder` | 883 | NSM codebook → distillation → encoding | Missing entirely |
| `deepnsm` | 845 | 65 semantic primes → 40K derived concepts | Missing entirely |
| `tekamolo` | 502 | Sentence → TEKAMOLO slot decomposition | Missing entirely |
| `surround_metadata` | 1283 | Givens rotation 7-component bundling (100% recovery) | Missing entirely |
| `compression_curves` | 1733 | Adaptive cascade rate modeling | Missing entirely |
| `bnn_cross_plane` | 1631 | 3-channel GraphHV cross-plane BNN | Missing entirely |
| `bnn_causal_trajectory` | 2116 | Causal routing across plane boundaries | Partially in rustynum-bnn |
| `udf_kernels` | 789 | DataFusion UDFs (hamming, NARS, sigma) | Partially in rustynum-arrow |
| `bf16_truth` | 680 | BF16 awareness classification, PackedQualia | Missing entirely |

---

## Retrofit Architecture

### Phase 1: `rustynum-bgz17` (NEW CRATE) — Golden Foundation

**What**: Port `bgz17_bridge` + `palette_distance` + `layered_distance` into a new crate.

**Why first**: Everything else (parallel_search, CLAM improvements) depends on Base17 patterns.

```
rustynum-bgz17/
├── Cargo.toml          # depends on rustynum-core (Fingerprint, Hamming)
└── src/
    ├── lib.rs
    ├── base17.rs       # ← from ndarray bgz17_bridge.rs
    │                   #   Base17, SpoBase17, PaletteEdge
    │                   #   golden_step_fold(), spo_from_planes()
    ├── palette.rs      # ← from ndarray palette_distance.rs
    │                   #   Palette, DistanceMatrix, SpoDistanceMatrices
    │                   #   build_matrix(), lookup()
    └── layered.rs      # ← from ndarray layered_distance.rs
                        #   read_palette_edge(), read_truth(), TruthGate
                        #   container_spo_distance()
```

**Adaptation needed**:
- ndarray uses `Fingerprint<256>` (const generic) → rustynum uses `[u64; 256]` raw
- ndarray TruthGate reads from `[u64; 256]` container → same in rustynum
- No ndarray dependency — self-contained, uses rustynum-core SIMD only

**Sanity gate**: `cargo test -p rustynum-bgz17` passes with ≥15 tests covering:
- Round-trip: fingerprint → Base17 → palette_index → distance
- SpoDistanceMatrices symmetry and diagonal==0
- TruthGate filtering at various thresholds

### Phase 2: Extend `rustynum-clam` — Parallel Search

**What**: Add parallel_search (HHTL + CLAM dual-path) to rustynum-clam.

**Why second**: CLAM tree already exists; parallel_search adds the missing search *strategy*.

```
rustynum-clam/src/
├── ... (existing: tree.rs, search.rs, compress.rs, etc.)
├── parallel_search.rs  # ← from ndarray parallel_search.rs
│                       #   PaletteScope, SearchResult, hhtl_search(), clam_search()
│                       #   merge_results(), dual_path_search()
└── lib.rs              # add: pub mod parallel_search;
```

**Adaptation needed**:
- Import `rustynum_bgz17::{PaletteEdge, SpoDistanceMatrices, TruthGate}` instead of ndarray hpc modules
- CLAM tree integration: wire `clam_search()` to existing `rustynum-clam::tree::ClamTree`
- HHTL uses palette indices from bgz17 crate

**Sanity gate**: `cargo test -p rustynum-clam` passes including new parallel_search tests

### Phase 3: Extend `rustynum-core` — SPO Bundle + Merkle

**What**: Add `spo_bundle` and `merkle_tree` to rustynum-core.

**Why third**: SPO bundling is a primitive (golden-shift permutation); Merkle tree is the Zeckendorf extension point.

```
rustynum-core/src/
├── ... (existing: fingerprint.rs, plane.rs, seal.rs, etc.)
├── spo_bundle.rs       # ← from ndarray spo_bundle.rs
│                       #   golden_shift(), cyclic_shift(), bundle_spo()
│                       #   Level A (8Kbit) + Level B (16Kbit)
├── merkle_tree.rs      # ← from ndarray merkle_tree.rs
│                       #   MerkleTree, StaunenType, build(), xor_diff()
│                       #   3-level: root(48b) → 8 branches → 64 leaves
└── lib.rs              # add: pub mod spo_bundle; pub mod merkle_tree;
```

**Adaptation needed**:
- `spo_bundle` uses `[u64; N]` const generics — matches rustynum's existing fingerprint format
- `merkle_tree` uses `seal::MerkleRoot` — rustynum-core already has `seal.rs`
- Hamming → delegate to `rustynum_core::simd::hamming_distance()`

**Sanity gate**: `cargo test -p rustynum-core` passes including spo_bundle + merkle tests

### Phase 4: `rustynum-vsa` (NEW CRATE) — Fibonacci-VSA Bridge

**What**: Port VSA module and bridge it to rustynum-holo's carrier frequencies.

**Why last**: This bridges two existing systems (holo carrier.rs + VSA) via Fibonacci spacing.

```
rustynum-vsa/
├── Cargo.toml          # depends on rustynum-core
└── src/
    ├── lib.rs
    ├── vector.rs       # ← from ndarray vsa.rs
    │                   #   VsaVector (10K-dim), VsaAccumulator
    │                   #   bind(), bundle(), clean(), permute()
    └── fibonacci_bridge.rs  # NEW — connects VSA to holo carrier frequencies
                             #   Fibonacci-spaced carrier ↔ VSA dimension mapping
                             #   Phase-to-binary projection for hybrid search
```

**Adaptation needed**:
- VsaVector is self-contained (pure `[u64; 157]`) — trivial port
- Fibonacci bridge is new code connecting holo carrier.rs (16 Fibonacci-spaced frequencies) to VSA dims
- Uses rustynum-core SIMD for popcount in similarity

**Sanity gate**: `cargo test -p rustynum-vsa` with ≥10 tests

---

## Dependency Graph After Retrofit

```
                 rustynum-core
                 ├── spo_bundle     (NEW in Phase 3)
                 ├── merkle_tree    (NEW in Phase 3)
                 └── ...existing...
                    ↑
         ┌─────────┼──────────┐
         │         │          │
   rustynum-bgz17  │   rustynum-vsa
   (NEW Phase 1)   │   (NEW Phase 4)
         ↑         │
         │         │
   rustynum-clam ──┘
   (EXTENDED Phase 2)
         ↑
         │
   ladybug-rs
```

---

## What NOT To Retrofit

These ndarray modules should NOT be ported — they're ndarray-specific or redundant:

| Module | Why skip |
|---|---|
| `arrow_bridge` | rustynum-arrow already has this |
| `blas_level{1,2,3}` | rustyblas already covers this (and is faster) |
| `fft`, `lapack`, `statistics` | Already in rustynum-core or rustyblas |
| `activations`, `vml` | Already in rustynum-rs |
| `hdc` | Overlaps with rustynum-rs HDC exports |
| `blackboard` | rustynum-core already has blackboard.rs |
| `fingerprint`, `plane`, `node`, `seal` | Already in rustynum-core |
| `cascade` | Already in rustynum-core (hdr.rs) |
| `nars` | Already in rustynum-bnn |
| `qualia`, `qualia_gate` | Already in rustynum-core |

---

## Execution Order & Time Estimates

| Phase | Crate | Modules | Depends On | Risk |
|---|---|---|---|---|
| **0** | rustynum-core | Fix `unsafe` in compute.rs | Nothing | Done (1 line) |
| **1** | rustynum-bgz17 | base17 + palette + layered | Phase 0 | Low — self-contained port |
| **2** | rustynum-clam | parallel_search | Phase 1 | Medium — needs CLAM tree wiring |
| **3** | rustynum-core | spo_bundle + merkle_tree | Phase 0 | Low — self-contained |
| **4** | rustynum-vsa | vsa + fibonacci_bridge | Phase 0 | Medium — bridge is new code |

Phases 1 and 3 are independent and can run in parallel.
Phase 2 depends on Phase 1.
Phase 4 is independent of everything.

---

## Zeckendorf Extension Point (Future)

After Phase 3, the Merkle tree is ready for Zeckendorf encoding:

1. **Replace Blake3 leaf hashes** with Zeckendorf-encoded branch summaries
2. **CLZ coarse distance** (1 CPU instruction) replaces full popcount for pre-filtering
3. **Hierarchical truncation** — truncate Zeckendorf bits at any point with known precision loss
4. **Non-consecutivity constraint** acts as error-correction (guaranteed ~55% sparsity)

This is documented in `/home/user/ndarray/.claude/FIBONACCI_MERKLE_FINDINGS.md` and validated by Fibonacci-VSA benchmarks (946× speedup for coarse distance, 100% classification accuracy).

The Merkle tree + VSA crate together form the foundation for this extension.

---

## Sanity Checklist (Run After Each Phase)

```bash
# Phase 0 (already done)
cargo check -p rustynum-core

# Phase 1
cargo test -p rustynum-bgz17

# Phase 2
cargo test -p rustynum-clam

# Phase 3
cargo test -p rustynum-core

# Phase 4
cargo test -p rustynum-vsa

# Full regression
cargo test --workspace
cd ../ladybug-rs && cargo check  # must still compile
```
