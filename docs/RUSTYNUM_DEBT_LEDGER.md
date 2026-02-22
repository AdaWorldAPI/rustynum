# rustynum Technical Debt & Knowledge Transfer Ledger

> **Scope**: PR #23 + PR #24 merged analysis, CLAM paper knowledge transfer, ladybug-rs acceleration needs  
> **Date**: 2026-02-22  
> **Repos**: AdaWorldAPI/rustynum, AdaWorldAPI/ladybug-rs, AdaWorldAPI/clam (fork)

---

## 1. Pre-PR #23 Debt (Inherited State)

These issues existed on `main` before PR #23 was created:

| # | Issue | Location | Severity | Description |
|---|---|---|---|---|
| D1 | **Arc\<Mutex\> parallel writes** | `rustynum-rs/src/simd_ops/mod.rs` | 🔴 Soundness | 4 `matrix_multiply` impls used `Arc<Mutex<&mut [T]>>` for parallel output writes. Semantically unsound (borrowed slice under Mutex), performance poison (lock contention per row). |
| D2 | **Blackboard raw pointer aliasing** | `rustynum-core/src/blackboard.rs` | 🔴 Soundness | `borrow_3_mut` created multiple `&mut` from `&self` via raw `*mut u8` without interior mutability wrapper — UB under stacked borrows. |
| D3 | **GEMM heap allocation in hot loop** | `rustyblas/src/level3.rs` | 🟡 Performance | `vec![0.0; NR]` allocated on heap every K iteration in sgemm/dgemm microkernels. |
| D4 | **No FMA guarantee** | `rustyblas/src/level3.rs` | 🟡 Performance | `acc += a * b` relies on compiler to fuse to FMA; `mul_add()` guarantees it. |
| D5 | **Scalar GEMM store path** | `rustyblas/src/level3.rs` | 🟡 Performance | RowMajor full-width output fell through to scalar `to_array()` + per-element `layout.index()` loop. |
| D6 | **Dead imports** | Multiple files | 🟢 Hygiene | Unused `parallel_for_chunks`, `Arc`, `Mutex`, `PhantomData`, type aliases scattered across crate. |
| D7 | **No Fingerprint type** | `rustynum-core` | 🟡 Architecture | Binary vectors represented as raw `[u8]` or `[u64]` without a unifying const-generic type. |
| D8 | **No XOR delta layer** | `rustynum-holo` | 🟡 Architecture | Holographic storage lacked borrow-safe mutation — no way to have immutable ground truth with mutable overlay without `RefCell`. |
| D9 | **rustynum-clam: only scalar Hamming** | `rustynum-clam/src/tree.rs` | 🟡 Performance | `hamming_inline()` relied on compiler auto-vectorization. No connection to rustynum-core's AVX-512 VPOPCNTDQ path. |

---

## 2. What PR #23 Fixed

PR #23 (`claude/unsigned-holograph-ghosts-NasOb`, merged 2026-02-22T12:54, +3,075/-845, 17 files):

| Pre-existing Debt | Fix | Quality |
|---|---|---|
| **D1** Arc\<Mutex\> parallel writes | New `parallel_into_slices()` using `std::thread::scope` + `split_at_mut`. Applied to all 4 type impls (u8, f32, f64, i32). New `parallel_reduce_sum()` for reductions. | ✅ Excellent — zero synchronization, provably non-aliasing, idiomatic |
| **D2** Blackboard raw pointer aliasing | Wrapped `BufferMeta` in `UnsafeCell`. All `unsafe` blocks have safety comments. Runtime assertions prevent same-buffer aliasing. | ✅ Sound — miri passed all 9 tests |
| **D3** GEMM heap in hot loop | `vec![0.0; NR]` → `[0.0; NR]` stack array in both sgemm and dgemm | ✅ Clean |
| **D4** No FMA guarantee | `acc += a * b` → `a.mul_add(b, acc)` with `StdFloat` import | ✅ Clean |
| **D5** Scalar GEMM store | RowMajor full-width `F32Simd::copy_to_slice` path with scalar fallback for partial/ColMajor | ✅ Clean |
| **D6** Dead imports | Removed `parallel_for_chunks`, `Arc`, `Mutex`, `SimdFloat→StdFloat`, `PhantomData`, unused type aliases | ✅ Clean |
| **D7** No Fingerprint type | New `Fingerprint<N>` in rustynum-core: const-generic `[u64; N]`, XOR/AND/OR/NOT ops, Hamming distance, byte serialization. Type aliases: `Fingerprint2K`, `Fingerprint1K`, `Fingerprint64K` | ✅ Excellent |
| **D8** No XOR delta layer | New `DeltaLayer<N>` + `LayerStack<N>` in rustynum-holo. `read()`, `write()`, `xor_patch()`, `collapse()`. Algebraically sound: `effective = ground ^ delta`, no interior mutability needed. | ✅ Excellent |

---

## 3. What PR #23 Introduced (New Items)

### 3.1 New Debt

| # | Issue | Location | Severity | Description |
|---|---|---|---|---|
| N1 | **5 separate PRNG impls** | search.rs, compress.rs, tree.rs (splitmix64) + recognize.rs, ghost_discovery.rs (SimpleRng/xorshift) | 🟡 Maintenance | Five independent PRNG copies. The xorshift in recognize.rs and ghost_discovery.rs is statistically weaker than splitmix64. |
| N2 | **recognize.rs ≠ Fingerprint\<1024\>** | `rustynum-oracle/src/recognize.rs` | 🟡 Architecture | 64K-bit LSH projection uses `Vec<u64>` of 1024 words instead of the new `Fingerprint<1024>` added in the same PR. Manual popcount loops that won't benefit from future Fingerprint SIMD. |
| N3 | **pub(crate) field escalation** | `rustynum-oracle/src/organic.rs` | 🟢 Encapsulation | `known_templates` and `template_norms` promoted from private to `pub(crate)` so recognize.rs can read them. Should be accessor methods. |
| N4 | **Concept ontology rename** | `rustynum-oracle/src/ghost_discovery.rs` | 🟢 API break | 52 concepts changed from Ada-specific (ada.hybrid, rel.devotion) to signal-processing (motor.servo, ctrl.pid). Intentional, but breaks reproducibility of prior ghost discovery runs. |

### 3.2 New Capabilities (Value-Add)

| Module | What | Lines | Impact |
|---|---|---|---|
| `Fingerprint<N>` | Const-generic binary vector with XOR group algebra | 309 | Foundation for all holographic ops |
| `DeltaLayer<N>` | Borrow-safe XOR overlay on immutable ground truth | 410 | Eliminates RefCell/UnsafeCell for holographic mutation |
| `LayerStack<N>` | Multi-layer composition with per-layer collapse | (in delta_layer) | Version control for binary vectors |
| `Recognizer` / `Projector64K` | 64K-bit LSH + Gram-Schmidt readout, novelty detection | 1,332 | Recognition as projection — write phase IS class score |
| `parallel_into_slices` | Zero-sync parallel write via split_at_mut | ~50 | Replaces all Arc\<Mutex\> patterns crate-wide |
| `parallel_reduce_sum` | Zero-sync parallel reduction | ~25 | Clean reduction pattern |
| GEMM microkernel optimization | Stack arrays + FMA + SIMD store | ~50 | 3 performance fixes in BLAS hot path |

---

## 4. What PR #24 Fixed / Added

PR #24 (`claude/review-rustynum-8yz8o`, merged same day, +135/-1, 4 files):

| Item | What | Quality |
|---|---|---|
| **D9** rustynum-clam scalar Hamming | Added `HammingSIMD` struct delegating to `rustynum_core::simd::hamming_distance`. Runtime dispatches to VPOPCNTDQ when available. | ✅ Clean — additive, no existing code modified |
| `hamming_batch_simd()` | Pass-through to `rustynum_core::simd::hamming_batch` — 4x ILP unrolled | ✅ Thin wrapper, no duplication |
| `hamming_top_k_simd()` | Pass-through to `rustynum_core::simd::hamming_top_k` — partial sort O(n) | ✅ Thin wrapper, no duplication |
| 6 new tests | SIMD-matches-scalar verification, various sizes, batch, top-k ordering | ✅ Good coverage |

**PR #24 did NOT introduce new debt.** But it added a 4th copy of `splitmix64` (in tree.rs tests), worsening N1.

---

## 5. Open Debt (Current State)

| # | Item | Severity | Fix Estimate | Recommendation |
|---|---|---|---|---|
| **N1** | ~~5× PRNG copies~~ | ✅ CLOSED | — | Consolidated into `rustynum_core::rng::SplitMix64` by PR #25. All 5 sites replaced. |
| **N2** | recognize.rs ≠ Fingerprint\<1024\> | 🟡 | 2 hours | Refactor `Projector64K.project()` → `Fingerprint<1024>`, use `Fingerprint::hamming_distance()` |
| **N3** | ~~pub(crate) field escalation~~ | ✅ CLOSED | — | Fields now private with `template()`, `template_norm()`, `update_template()` accessors. PR #25. |
| **N5** | `debug_assert_eq!` in HammingSIMD | 🟡 Safety | 5 min | `HammingSIMD::distance()` uses `debug_assert_eq!` for length check — elided in release builds. The AVX-512 path loads 64-byte chunks via `_mm512_loadu_si512`; mismatched buffer lengths could read past allocation. **Upgrade to `assert_eq!`** (one comparison per call, negligible cost). |
| **N6** | Three Hamming distance type signatures | 🟡 Architecture | 2 hours | `rustynum_core::simd` operates on `&[u8]`, `Fingerprint::hamming_distance()` on `[u64; N]`, `recognize.rs::hamming_64k()` on `&[u64]`. PR #25 partially addressed — `hamming_64k` delegates to `Fingerprint64K` for 1024-word case. SIMD acceleration of Fingerprint deferred. |
| **N7** | Granger sign convention contradiction | 🔴 Correctness | 5 min | `granger_signal()` doc says G > 0 = A predicts B. Code + test prove G < 0 = A predicts B. `granger_scan()` has correct convention but prefaced with "Wait — let me clarify" debug thinking. Fix: swap G>0/G<0 descriptions in `granger_signal()`, remove debug text from `granger_scan()`. (PR #25) |
| **N8** | `Contradiction` struct overpromises | 🟢 Naming | 10 min | Checks structural similarity only, not truth-value conflict. Should be `SimilarPair` until NARS truth values are integrated. (PR #25) |
| **N9** | Flat confidence threshold in reverse_trace | 🟢 Improvement | 30 min | Uses 0.35 flat threshold, not CRP-calibrated. Wire to `ClusterDistribution.p95` when available. (PR #25) |
| **N10** | `hamming_i8` misleading name | 🟡 Naming | 10 min | Symbol-level distance (`a[i] != b[i]` on i8), not bit-level Hamming (XOR+popcount). Every other `hamming_*` in the codebase is bit-level. Rename to `symbol_distance_i8` or `disagreement_count`. (PR #25) |
| **N11** | Clone-per-hop in `reverse_trace()` | 🟢 Latent perf | 30 min | 16KB clone per hop × depth. Fine for research. Rewrite to two pre-allocated buffers + `copy_from_slice` if it enters a batch hot loop. (PR #25) |
| **N12** | `hamming_64k` 16KB stack copy regression | 🟡 Performance | 15 min | PR #25 delegation creates two `Fingerprint64K` via `from_word_slice` (2×8KB memcpy onto stack) to run the same `(a^b).count_ones()` loop the inline code did. Fix: add `Fingerprint::hamming_distance_slices(a: &[u64], b: &[u64]) → u32` that works on borrows, or revert to inline loop. (PR #25) |

---

## 6. Paper Knowledge Successfully Transferred to Code

The 4 CLAM papers (CHESS, CHAODA, CAKES, panCAKES) were reviewed, converted to clean markdown, and their algorithms mapped against rustynum-clam. Here's what made it into working code vs what remains theoretical:

### 6.1 Successfully Implemented in rustynum-clam

| Paper Concept | Code | Lines | Paper Validation |
|---|---|---|---|
| **CLAM divisive hierarchical clustering** (CHESS Alg 1) | `ClamTree::build()` — bipolar split, √n seeds, geometric median poles | 250+ | ✅ Matches paper algorithm exactly |
| **Local Fractal Dimension** (CHESS Eq 2) | `Lfd::compute()` + `lfd_percentiles()` + `lfd_by_depth()` | 80+ | ✅ Parity with upstream. *Exceeds* upstream with per-depth statistics. |
| **ρ-NN search** (CHESS Alg 2) | `rho_nn()` with triangle inequality pruning | 100+ | ✅ Exact match with paper algorithm |
| **Repeated ρ-NN k-NN** (CAKES Alg 4) | `knn_repeated_rho()` | 100+ | ✅ Exact match |
| **DFS Sieve k-NN** (CAKES Alg 6) | `knn_dfs_sieve()` with min-heap Q + max-heap H | 100+ | ✅ Exact match |
| **δ⁺/δ⁻ pruning** (CAKES Fig 1) | `Cluster::delta_plus()`, `delta_minus()` | 20 | ✅ Triangle inequality bounds |
| **Depth-first reordering** (CAKES §2.1.3) | `ClamTree.order` permutation array | integrated | ✅ O(n) memory |
| **XOR-diff encoding** (panCAKES unitary) | `XorDiffEncoding::encode()/decode()` | 100+ | ✅ Matches paper unitary mode |
| **Recursive compression** (panCAKES Alg 2) | `CompressionMode::Recursive`, bottom-up cost comparison in `compress()`, recursive `assign_encodings()` | 200+ | ✅ Full framework — chooses min(unitary, recursive) per cluster |
| **Mixed-mode compressed tree** (panCAKES) | `CompressedTree` with `cluster_modes: Vec<CompressionMode>` | integrated | ✅ Tracks which clusters use unitary vs recursive |
| **Compressed Hamming distance** (panCAKES §II-C) | `hamming_from_query()` on `XorDiffEncoding` | 50+ | ✅ Avoids full decompression for distance computation |
| **Compressive distance primitive** (panCAKES) | `hamming_to_compressed()` on `CompressedTree` | 30+ | 🟡 Distance primitive only — used in test, but no search function (rho_nn, knn_dfs_sieve) actually calls it yet |
| **VPOPCNTDQ-accelerated Hamming** (PR #24) | `HammingSIMD` + batch + top-k | 130+ | ✅ Links rustynum-core SIMD to CLAM search |

### 6.2 Paper Knowledge Documented But Not Yet Coded

| Concept | Paper | Why It Matters | Gap Analysis Reference |
|---|---|---|---|
| **BFS Sieve** (CAKES Alg 5) | CAKES | Often 2nd-fastest algorithm. Uses QuickSelect for level-wise pruning. | Phase 1a in todo |
| **Improved child pruning** (CAKES Supp §1.1) | CAKES | Law-of-cosines projection saves ~20% distance computations | Phase 1b |
| **Auto-tuning** (CAKES §2.3) | CAKES | Sample-then-select across 3 algorithms | Phase 1c |
| **Approximate k-NN** | CAKES | Early termination when recall tolerance allows | Phase 1d |
| **Recursive decompression chain walk** | panCAKES | `decompress_point()` does single-hop (center + XOR diff). For recursively-encoded points, need to walk ancestor chain: leaf → parent center → grandparent center → ... → root. Currently ~50% done. | Phase 2b (reduced scope) |
| **Search-on-compressed wrapper** | panCAKES | `hamming_to_compressed()` exists as distance primitive but no search function (rho_nn, knn_dfs_sieve) plugs it in as the distance oracle. Need a CompressedSearch adapter. | Phase 2c (new) |
| **Graph induction** | CHAODA | Overlapping cluster detection → edge graph | Phase 5 (not prioritized) |
| **6 anomaly scorers** | CHAODA | Cluster cardinality, component cardinality, graph neighborhood, etc. | Phase 5 |
| **Transition/subsumed edges** (PR #21) | CHAODA | Python-only, never ported to Rust anywhere | Phase 5 |

### 6.3 Key Theoretical Insight: Why This Matters for Holographic Vectors

The papers prove that CAKES search complexity is O(log N_r̂ + k · 2^LFD) where LFD is Local Fractal Dimension. For 10K-bit holographic vectors:

- Holographic codes from the same domain share structure → **low LFD** (manifold hypothesis)
- Low LFD means **near-constant query time** as database grows
- panCAKES compression ratio is proportional to within-cluster similarity → **high compression** for structured embeddings
- CAKES Table 2 (Fashion-MNIST) proves this empirically: 3,000 QPS constant across 512× data augmentation while HNSW recall drops from 0.954 → 0.361

This is the mathematical proof that ladybug-rs's hierarchical scent index *can* scale, but only if it uses geometry-aware partitioning (CLAM bipolar split) rather than hash-like XOR-fold bucketing.

---

## 7. Cross-Reference: CLAM Hardening Plan vs rustynum-clam Status

The ladybug-rs `CLAM_HARDENING.md` proposes 8 integrations. Here's what rustynum-clam already provides:

| CLAM_HARDENING Section | Ladybug Needs | rustynum-clam Has | Gap |
|---|---|---|---|
| §1: Replace scent hierarchy with CLAM tree | `Tree::par_new_minimal()` + `KnnBranch` | `ClamTree::build()` + `knn_dfs_sieve()` | ✅ Ready. rustynum-clam's tree IS the CLAM tree. Ladybug just needs to call it on `Fingerprint` data. |
| §2: LFD estimation | `cluster.lfd()` | `Lfd::compute()` + `lfd_percentiles()` + `lfd_by_depth()` | ✅ Ready. rustynum-clam even exceeds — has per-depth LFD stats. |
| §3: d_min/d_max triangle inequality | `d_min = d - r`, `d_max = d + r` | `delta_plus()`, `delta_minus()` on `Cluster` | ✅ Ready. Exact same formulas. |
| §4: panCAKES compression | `CompressedFingerprint` | `XorDiffEncoding` + `CompressedTree` + `CompressionMode::{Unitary,Recursive}` + `hamming_to_compressed()` | 🟡 Framework complete (unitary + recursive + mixed-mode tree). Gaps: recursive decompression walks single-hop only (no ancestor chain), `hamming_to_compressed()` is a distance primitive not yet wired into search functions. |
| §5: CHAODA anomaly | Anomaly scoring on CLAM tree | ❌ Not implemented | 🔴 Missing. Not in upstream Rust either. |
| §6: HDR-stacked CRP distributions | `ClusterDistribution` with μ, σ, percentiles, INT4 histogram | ❌ Not in rustynum-clam | 🟡 **This is ladybug-unique** — CLAM doesn't have it. rustynum-clam provides the tree; ladybug adds the distribution statistics per cluster. |
| §7: DistanceValue trait | Generic distance metric | `Distance` trait in tree.rs | ✅ Ready. Same concept, different name. |
| §8: Causal certificates | Cohen's d from CRP → Granger → Pearl | ❌ Not in rustynum-clam | 🟡 **This is ladybug-unique** — built on top of CRP (§6), which itself sits on top of the CLAM tree. |

**Key takeaway**: rustynum-clam is the **engine layer** (tree + search + compression). Ladybug builds the **intelligence layer** on top (CRP distributions, Mexican hat calibration, causal certificates). These are complementary, not competing.

---

## 8. 34 NARS Tactics: Which Need CLAM/SIMD Acceleration

Of the 34 tactics, I identified which operations are **compute-bound** and would benefit from rustynum-clam's SIMD Hamming, CLAM tree search, or panCAKES compression — vs which are pure logic/protocol that don't need acceleration.

### 8.1 Tactics That NEED Acceleration (Hot Path)

| # | Tactic | Bottleneck Operation | Acceleration Path |
|---|---|---|---|
| **#4** | **Reverse Causality Reasoning** | ABBA retrieval: `outcome ⊗ CAUSES = candidate_cause` then nearest-neighbor search. **Each hop in the causal chain is a k-NN query.** For depth-N trace, that's N × k-NN. | **CAKES DFS Sieve** — each hop is O(log N + k·2^LFD) instead of O(N). For a depth-5 trace in a 1M-entry store, this is ~5ms vs ~500ms brute-force. |
| **#5** | **Thought Chain Pruning** | `hamming_distance(&Fingerprint::bundle(...))` over chain — comparing each step to accumulated bundle. O(chain_length × 16K-bit Hamming). | **hamming_batch_simd** — batch distance from bundle to all chain steps in one SIMD pass. 4x ILP unrolling. |
| **#11** | **Contradiction Detection** | O(n²) pairwise Hamming + truth-value comparison across beliefs. For 1000 beliefs: 500K distance computations. | **CLAM tree**: build tree over beliefs, then find pairs where structural_sim > 0.7 ∧ truth_conflict > threshold. CAKES ρ-NN with radius = 0.3 × TOTAL_BITS finds all pairs in O(n·log n) instead of O(n²). |
| **#12** | **Temporal Context Augmentation** | Granger signal: `d(A_t, B_{t+τ}) - d(B_t, B_{t+τ})` for each lag τ ∈ [1..max_lag]. Two k-NN queries per lag per pair. | **hamming_batch_simd** for the pairwise distances. If comparing N series over L lags: N²×L Hamming computations → SIMD batch. |
| **#15** | **Latent Space Introspection** (CRP) | Computing `ClusterDistribution` requires O(cluster_size) Hamming distances from center to each member. For a 10K-member cluster: 10K × 16K-bit distances. | **hamming_batch_simd** — single batch call, center as query, all members as database. This IS the CRP construction path. |
| **#20** | **Thought Cascade Filtering** | Run 7 CAKES algorithms, take best. Each is a full k-NN query. | **CAKES auto-tuning** (Phase 1c) — sample to pick the fastest algorithm, then run only that one. Reduces 7 queries to 1 + sampling cost. |
| **#25** | **Hyperdimensional Pattern Matching** | This IS rustynum-clam. The 65M comparisons/sec claim comes from SIMD Hamming. | **Already accelerated** via PR #24's HammingSIMD. Benchmark suite needed. |
| **#26** | **Cascading Uncertainty Reduction** | HDR cascade at 4 resolutions. Each level filters candidates → next level does finer distance. | **CLAM tree** replaces the heuristic cascade with provable d_min/d_max bounds. CRP (§6 in CLAM_HARDENING) calibrates which level to use per cluster. |
| **#27** | **Multi-Perspective Compression** | `weighted_bundle` is O(n_perspectives × 16K bits). panCAKES can compress the perspectives. | **panCAKES XOR-diff** — store each perspective as delta from bundle center. Compressed bundle takes O(n × avg_diff_bits) instead of O(n × 16K). |
| **#30** | **Shadow Parallel Processing** | Background pre-computation: `top_k` for 10 neighbors, then `top_k` for 5 follow-ups each → 60 k-NN queries. | **CAKES DFS Sieve + rayon** — parallel tree search. Each query is O(log N + k·2^LFD). 60 queries at ~1ms each = 60ms total, can be pipelined. |

### 8.2 Reverse Causality (#4): The Direction Problem

This is the tactic you specifically called out. The core challenge:

**Forward**: `A ⊗ CAUSES ⊗ B` — bind A with CAUSES verb, find B. This is a standard k-NN: bind query, search.

**Reverse**: `B ⊗ CAUSES ⊗ ? = A` — given outcome B, find what caused it. This requires unbinding: `B ⊗ CAUSES = candidate_A`, then searching for the nearest real entity to `candidate_A`.

The asymmetry: **forward is O(1) bind + O(log N) search. Reverse is O(1) unbind + O(log N) search + O(?) validation.**

The validation is where CLAM helps:

1. **Without CLAM**: Unbind gives you a noisy candidate. You search brute-force for the nearest real entity. If the candidate is wrong (noisy unbinding), you get the wrong cause. No way to know confidence.

2. **With CLAM**: Unbind gives you a noisy candidate. CAKES DFS Sieve finds the k-nearest real entities. The CRP distribution of the cluster containing the best match tells you the **z-score** of the match — i.e., is this a confident recovery or a noise-floor hit? If the distance to the nearest entity is within σ of the cluster center, it's a confident causal attribution. If it's beyond p95, the causal chain is broken.

3. **With Granger signal**: For temporal causal chains (A happened, then B happened), the Granger signal `G(A→B,τ) = d(A_t, B_{t+τ}) - d(B_t, B_{t+τ})` gives you **directional** information. If G > 0, A predicts B. If G < 0, B predicts A. The CRP confidence interval tells you whether this is statistically significant.

**Acceleration need**: Each step in the reverse trace is a k-NN query. A depth-5 reverse trace through a 1M-entry BindSpace is 5 × O(log 1M) with CAKES = ~5ms, vs 5 × O(1M) brute-force = ~500ms. The 100× speedup makes real-time causal reasoning feasible.

### 8.3 Tactics That DON'T Need Acceleration (Logic/Protocol)

| # | Tactic | Why No Acceleration Needed |
|---|---|---|
| #1 | Recursive Thought Expansion | O(max_depth) iterations, each is one rung transform. Bound by depth cap (7), not data size. |
| #3 | Structured Multi-Agent Debate | O(n_agents) iterations. Bottleneck is bundle + revision, not search. |
| #6 | Thought Randomization | O(16K bits) noise injection. Single fingerprint operation. |
| #7 | Adversarial Self-Critique | 5 challenge types, each is O(1) truth-value computation. |
| #8 | Conditional Abstraction Scaling | O(1) entropy check → level selection. |
| #9 | Iterative Roleplay Synthesis | O(n_personas) bind + search. Small n. |
| #10 | Meta-Cognition Prompting | O(history_length) statistics. Window capped at 100. |
| #13 | Convergent/Divergent Thinking | O(rounds) oscillation. Small rounds count. |
| #14 | Multimodal Chain-of-Thought | Encoding is the bottleneck (Jina API), not search. |
| #16-19, 21-24, 28-29, 31-34 | Various | Either map to existing operations, are thin wrappers on bind/bundle, or are protocol/logic with no data-scale dependency. |

### 8.4 Acceleration Priority for ladybug-rs

Based on compute intensity × frequency of use:

| Priority | Tactic | Operation | rustynum-clam Enabler | Est. Impact |
|---|---|---|---|---|
| 🔴 P0 | #4 Reverse Causality | Chain of k-NN queries | CAKES DFS Sieve | ~100× theoretical (brute→tree), unbenchmarked on binary vectors |
| 🔴 P0 | #15 CRP Construction | Batch Hamming per cluster | `hamming_batch_simd` | 8× (scalar→AVX-512) |
| 🔴 P0 | #25 Pattern Matching | Core SIMD Hamming | `HammingSIMD` (PR #24) | Already done |
| 🟡 P1 | #11 Contradiction Detection | Pairwise similarity | CAKES ρ-NN | O(n²)→O(n·log n) |
| 🟡 P1 | #26 Cascading Uncertainty | HDR cascade calibration | CLAM tree d_min/d_max | Provable bounds vs heuristic |
| 🟡 P1 | #20 Cascade Filtering | Multi-algorithm k-NN | CAKES auto-tune | 7×→1× queries |
| 🟢 P2 | #5 Chain Pruning | Batch distance from bundle | `hamming_batch_simd` | 4× ILP |
| 🟢 P2 | #12 Temporal Context | Lag-distance computation | `hamming_batch_simd` | 4× ILP |
| 🟢 P2 | #27 Multi-Perspective Compression | Delta storage | panCAKES XOR-diff | 5-70× storage |
| 🟢 P2 | #30 Shadow Processing | Parallel precompute | CAKES + rayon | ~60ms for 60 queries |

---

## 9. Consolidated Action Items

### Immediate (debt cleanup, <1 day)

1. **Upgrade `debug_assert_eq!` → `assert_eq!` in `HammingSIMD::distance()`** — AVX-512 safety. One comparison per call, negligible cost. Also check upstream `rustynum_core::simd::hamming_distance` (N5).
2. **Fix Granger sign convention** — swap G>0/G<0 in `granger_signal()` doc, remove "Wait — let me clarify" from `granger_scan()` doc (N7). 5-minute fix.
3. ~~Extract `rustynum_core::rng::SplitMix64`~~ — ✅ Done in PR #25 (N1 closed).
4. ~~Refactor recognize.rs → Fingerprint\<1024\>~~ — ✅ Partially done in PR #25 (`hamming_64k` delegates for 1024-word case).
5. ~~Add OrganicWAL accessor methods~~ — ✅ Done in PR #25 (N3 closed).

### Next sprint (CLAM completeness, 2-3 days)

4. **BFS Sieve k-NN** (CAKES Alg 5) — ~150 lines in search.rs
5. **Improved child pruning** — law-of-cosines projection in rho_nn(), ~30 lines
6. **Auto-tuning** — sample-then-select, ~50 lines

### Following sprint (panCAKES completion, 2-3 days)

7. **Recursive decompression chain walk** — extend `decompress_point()` to follow ancestor chain for recursively-encoded points, ~80 lines
8. **CompressedSearch adapter** — wire `hamming_to_compressed()` into DFS sieve as swappable distance oracle, ~100 lines
9. **Compression benchmarks** — measure ratio + query speed for unitary vs recursive vs uncompressed on Ada's actual 10K-bit vectors, ~50 lines

### Ladybug integration (after rustynum-clam is complete)

10. **Wire `ClamTree::build()` as alternative to ScentIndex** — ladybug `src/core/clam_index.rs`
11. **Add CRP construction using `hamming_batch_simd`** — ladybug `src/search/distribution.rs`
12. **Wire CRP to Mexican hat** — replace hardcoded thresholds with measured percentiles
13. **Implement reverse causal trace using CAKES DFS Sieve** — ladybug `src/search/causal.rs::reverse_trace()`

---

## Appendix: File Inventory

### Documents Created This Session
| File | Location(s) | Content |
|---|---|---|
| `CAKES_paper.md` | ada-docs/research/clam/, rustynum/docs/ | Full CAKES paper conversion (36KB) |
| `panCAKES_paper.md` | ada-docs/research/clam/, rustynum/docs/ | Full panCAKES paper conversion (17KB) |
| `CLAM_Full_Gap_Analysis.md` | ada-docs/research/clam/, rustynum/docs/ | 4-paper gap analysis |
| `PR23_debt_analysis.md` | ada-docs/research/rustynum/, rustynum/docs/ | PR #23 technical debt review |
| This document | ada-docs/research/rustynum/, rustynum/docs/ | Merged ledger |

### Code Changes (PR #23 + #24 combined)
| Crate | Files Changed | Added | Removed | Net |
|---|---|---|---|---|
| rustynum-core | 3 | fingerprint.rs (309), blackboard.rs fixes | — | +426 |
| rustynum-holo | 2 | delta_layer.rs (410) | — | +413 |
| rustynum-oracle | 5 | recognize.rs (1332), ghost_discovery rewrite | — | +1685 |
| rustyblas | 1 | GEMM microkernel fixes | — | +25 |
| rustynum-rs | 2 | parallel_into_slices, cleanup | Arc/Mutex | +5 |
| rustynum-clam | 3 | HammingSIMD + batch + top-k | — | +134 |
| root | 1 | rust-toolchain.toml | — | +2 |
| **Total** | **17+4** | **+3,210** | **-846** | **+2,364** |
