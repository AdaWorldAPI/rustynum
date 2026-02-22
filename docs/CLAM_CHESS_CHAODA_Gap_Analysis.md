# CLAM / CHESS / CHAODA → rustynum Implementation Gap Analysis

> **Date**: 2026-02-22  
> **Papers**: CHESS (arXiv:1908.08551v2), CHAODA (arXiv:2103.11774v2)  
> **Upstream**: [URI-ABD/clam](https://github.com/URI-ABD/clam) (Rust, `master`)  
> **Downstream**: [AdaWorldAPI/rustynum](https://github.com/AdaWorldAPI/rustynum) (`rustynum-clam` crate)  
> **Cross-ref**: URI-ABD/clam PR #21 — "feat: split edges into transition edges and subsumed edges"

---

## 1. Paper-to-Code Concept Map

### 1.1 CHESS (arXiv:1908.08551v2) — Entropy-Scaling Search

| Paper Concept | Section | URI-ABD/clam (Rust) | rustynum-clam | Status |
|---|---|---|---|---|
| **Divisive hierarchical clustering** (Alg 1) | §2.2.1 | `tree::partition` — bipolar split, √n seeds, geometric median, max-distance poles | `tree.rs` — same algorithm, u64 distance, `HammingDistance` trait impl | ✅ Parity |
| **Local Fractal Dimension** (Eq 2) | §2.2 | `Cluster.lfd` — computed during construction | `tree.rs` — `Lfd { lfd: f64 }`, computed per-cluster | ✅ Parity |
| **ρ-NN search** (Alg 2) | §2.2.2 | `cakes::exact::rnn_chess` | `search.rs` — `rho_nn()` with triangle-inequality pruning | ✅ Parity |
| **k-NN via repeated ρ-NN** | §3.5 | `cakes::exact::knn_rrnn` | `search.rs` — `knn_repeated_rho()` | ✅ Parity |
| **DFS Sieve** (CAKES Alg 6) | CAKES paper | `cakes::exact::knn_dfs` + `knn_bfs` + `knn_branch` | `search.rs` — `knn_dfs_sieve()` | ✅ Parity |
| **Approximate kNN** | CAKES | `cakes::approximate::knn_dfs` | ❌ Not in rustynum-clam | 🔴 Missing |
| **Data compression** | §2.3 | Not in Rust crate (was Python-only) | `compress.rs` — panCAKES XOR-diff encoding | ✅ **Exceeds** upstream |
| **Asymptotic complexity** O(log₂k + \|B\|·scaling) | §3.1, Eq 3 | Demonstrated empirically | Claimed in doc; benchmarks needed | ⚠️ Untested |
| **Cosine distance** support | §3.2 | `distances` crate — SIMD cosine | ❌ Only `HammingDistance` impl | 🔴 Missing |
| **Euclidean distance** support | §3.2 | `distances` crate — SIMD L2 | ❌ Only `HammingDistance` impl | 🔴 Missing |
| **Levenshtein/edit distance** | §2.1.2 | `distances::strings::needleman_wunsch` | ❌ Not implemented | 🔴 Missing |

### 1.2 CHAODA (arXiv:2103.11774v2) — Anomaly Detection

| Paper Concept | Section | URI-ABD/clam (Rust) | rustynum-clam | Status |
|---|---|---|---|---|
| **Graph Induction** — overlapping clusters → edges | §2.3 | ❌ Marked as "WIP" in lib.rs (`chaoda` feature gate, commented out) | ❌ Not implemented | 🔴 Missing in BOTH |
| **Cluster Selection** (Alg 4) — meta-ML models | §2.6 | ❌ WIP | ❌ Not implemented | 🔴 Missing |
| **Relative Cluster Cardinality** | §2.4.1 | ❌ WIP | ❌ Not implemented | 🔴 Missing |
| **Relative Component Cardinality** | §2.4.2 | ❌ WIP | ❌ Not implemented | 🔴 Missing |
| **Graph Neighborhood Size** (Alg 2) | §2.4.3 | ❌ WIP | ❌ Not implemented | 🔴 Missing |
| **Child-Parent Cardinality Ratio** | §2.4.4 | ❌ WIP | ❌ Not implemented | 🔴 Missing |
| **Stationary Probabilities** (Alg 3) | §2.4.5 | ❌ WIP | ❌ Not implemented | 🔴 Missing |
| **Relative Vertex Degree** | §2.4.6 | ❌ WIP | ❌ Not implemented | 🔴 Missing |
| **Meta-ML training** — linear regression + decision tree | §2.5 | ❌ WIP | ❌ Not implemented | 🔴 Missing |
| **Ensemble scoring** — Gaussian normalization + mean aggregation | §2.7 | ❌ WIP | ❌ Not implemented | 🔴 Missing |
| **Transition vs Subsumed edges** | PR #21 | ❌ Was in Python `pyclam/manifold.py`, never ported to Rust | ❌ Not implemented | 🔴 Missing |

### 1.3 panCAKES (arXiv:2409.12161) — Compression

| Paper Concept | URI-ABD/clam | rustynum-clam | Status |
|---|---|---|---|
| **Hierarchical XOR-diff encoding** | Not in Rust | `compress.rs` — full implementation (656 lines) | ✅ **Unique to rustynum** |
| **Min-cost tree pruning** | Not in Rust | `compress.rs` — unitary vs recursive cost | ✅ **Unique to rustynum** |
| **Decompression for search** | Not in Rust | `compress.rs` — reconstruct + Hamming | ✅ **Unique to rustynum** |

---

## 2. URI-ABD/clam PR #21 Analysis

### 2.1 What PR #21 Did

PR #21 ("feat: split edges into transition edges and subsumed edges") by nishaq503 was merged into the **Python** `pyclam` implementation. It introduced:

1. **Edge type split**: `Edge` namedtuple changed from `(neighbor, distance, transition_probability)` to `(neighbor, distance, probability)`. New `CacheEdge` type: `(source, neighbor, distance, probability)`.

2. **Transition vs Subsumed clusters**: A cluster is "subsumed" if its center lies within another cluster's radius. Transition clusters are not subsumed by any other.

3. **Separate edge dictionaries**: `Graph.transition_edges` and `Graph.subsumed_edges` — allows CHAODA algorithms to operate on different topologies.

4. **Removed `absorbable` flag** from cluster cache — replaced by the subsumed concept.

5. **Refined candidate neighbor propagation**: Changed from "keep optimal clusters" to "keep candidates from parent" + all children at the same depth.

### 2.2 What Was Never Ported to Rust

The Python `pyclam` directory no longer exists in the current `master` branch. The Rust crate (`abd-clam`) has:

- ✅ Tree construction (partition, LFD, cluster properties)
- ✅ CAKES search algorithms (exact + approximate k-NN, ρ-NN)
- ✅ Serialization via serde
- ❌ **No Graph type at all** — no overlapping-volume edge detection
- ❌ **No transition/subsumed distinction** from PR #21
- ❌ **No CHAODA algorithms** — module is commented out with "WIP"
- ❌ **No meta-ML cluster selection**
- ❌ **No ensemble scoring**

### 2.3 What rustynum-clam Inherited vs What It Didn't

rustynum-clam was built from the *research papers*, not from the Python codebase. It implements the tree + search + compression pipeline but has **zero graph-induction or anomaly-detection code**.

---

## 3. AVX-512 / VNNI / VPOPCNTDQ Optimization Status

### 3.1 What rustynum Already Has

| Feature | File | Implementation | AVX-512 Optimized? |
|---|---|---|---|
| **Hamming distance** (XOR+POPCNT) | `rustynum-clam/tree.rs` + `rustynum-rs/bitwise.rs` | 4× u64 unrolled XOR+POPCNT | ⚠️ **Scalar POPCNT** — relies on compiler auto-vectorization for VPOPCNTDQ |
| **f32 dot product** | `rustynum-core/simd.rs` | `f32x16` (std::simd portable) | ✅ Uses 512-bit `f32x16` via `std::simd` |
| **f64 dot product** | `rustynum-core/simd.rs` | `f64x8` (std::simd portable) | ✅ Uses 512-bit `f64x8` |
| **INT8 dot product** | `rustyblas/int8_gemm.rs` | Accumulate as i32, 4× unrolled | ⚠️ Targets VNNI but doesn't use `_mm512_dpbusd_epi32` intrinsics directly |
| **BF16 GEMM** | `rustyblas/bf16_gemm.rs` | Conversion via `f32x16` | ⚠️ Software BF16 conversion, not using `_mm512_dpbf16_ps` |
| **CPU capability detection** | `rustynum-core/compute.rs` | `CpuCaps` struct | ✅ Detects avx512f, avx512bw, avx512vnni, avx512_bf16, avx512_vpopcntdq |
| **AVX2 fallback** | `rustynum-core/simd_avx2.rs` | Full parallel impl with f32x8/f64x4 | ✅ Complete |

### 3.2 What's NOT Optimized with AVX-512 Intrinsics

| Missing Optimization | Impact | How to Fix |
|---|---|---|
| **VPOPCNTDQ** for Hamming | Current: scalar `u64::count_ones()` in 4× loop. Compiler *may* emit VPOPCNTDQ on `-C target-cpu=native` but no guarantee. | Use `core::arch::x86_64::_mm512_popcnt_epi64` intrinsics directly, gated on `avx512_vpopcntdq` detection |
| **VNNI `vpdpbusd`** for INT8 dot | Current: manual i8×i8→i32 accumulation. Missing the fused multiply-add instruction that does 4×i8 dot in one cycle. | Use `_mm512_dpbusd_epi32(acc, a, b)` — 4 byte-pairs multiplied and accumulated per element, 16 elements = 64 byte-pairs per clock |
| **BF16 `dpbf16ps`** for BF16 GEMM | Current: convert bf16→f32, multiply, accumulate. Missing fused BF16 dot-product. | Use `_mm512_dpbf16_ps(acc, a, b)` for 2× throughput vs f32 |
| **Hamming distance in CLAM tree** | `hamming_inline()` in `tree.rs` duplicates the pattern from `hdc.rs` — no shared VPOPCNTDQ primitive | Extract to shared `rustynum-core::simd::hamming_u512()` with intrinsic path |
| **Euclidean distance for CLAM** | Not implemented at all in rustynum-clam | Use `rustynum-core::simd::dot_f32` for squared L2 |
| **Cosine distance for CLAM** | Not implemented at all in rustynum-clam | Normalize + dot product using existing f32x16 primitives |

### 3.3 Compiler Auto-Vectorization vs Explicit Intrinsics

The current approach relies on Rust's `std::simd` portable SIMD plus compiler auto-vectorization for the popcount path. This is fragile:

```text
PROBLEM:
  u64::count_ones() → compiler may emit:
    - POPCNT instruction (scalar, 1 per clock)
    - VPOPCNTDQ (AVX-512, 8 u64 per clock) ← only if -C target-cpu=znver4 or sapphirerapids
    - Software popcount (fallback)

SOLUTION:
  #[cfg(target_feature = "avx512vpopcntdq")]
  unsafe fn hamming_512(a: &[u8; 64], b: &[u8; 64]) -> u64 {
      let va = _mm512_loadu_si512(a.as_ptr() as *const i32);
      let vb = _mm512_loadu_si512(b.as_ptr() as *const i32);
      let xor = _mm512_xor_si512(va, vb);
      let pop = _mm512_popcnt_epi64(xor);           // 8 × u64 popcounts
      _mm512_reduce_add_epi64(pop) as u64
  }
```

---

## 4. CLAM Upstream (URI-ABD/clam) Architecture

### 4.1 Crate Structure

```
URI-ABD/clam/
├── crates/
│   ├── abd-clam/          # Core: Tree, Cluster, CAKES search
│   │   ├── src/
│   │   │   ├── tree/
│   │   │   │   ├── cluster/     # Cluster struct (depth, center, cardinality, radius, lfd)
│   │   │   │   └── partition/   # PartitionStrategy (bipolar split, branching factor, SRF)
│   │   │   ├── cakes/
│   │   │   │   ├── exact/       # rnn_chess, knn_dfs, knn_bfs, knn_branch, knn_rrnn, knn_linear
│   │   │   │   ├── approximate/ # knn_dfs (approximate variant)
│   │   │   │   └── selection/   # Algorithm selection strategies
│   │   │   ├── musals/          # Multiple Sequence Alignment (feature-gated)
│   │   │   └── lib.rs           # chaoda, codec, mbed modules: ALL WIP/COMMENTED OUT
│   │   └── tests/
│   ├── distances/         # SIMD distance functions
│   │   ├── src/
│   │   │   ├── simd/      # Portable SIMD: F32x16, F64x8, etc (NOT std::simd, custom types)
│   │   │   ├── vectors/   # L1, L2, cosine, correlations
│   │   │   └── strings/   # Needleman-Wunsch, edit distances
│   │   └── benches/
│   ├── symagen/           # Synthetic data generation for testing
│   └── shell/             # CLI: cakes build/search, mbed, musals
└── pypi/distances/        # Python bindings via pyo3
```

### 4.2 Key Differences: URI-ABD/clam vs rustynum-clam

| Aspect | URI-ABD/clam | rustynum-clam |
|---|---|---|
| **SIMD approach** | Custom portable SIMD types (F32x16, F64x8) — NO `std::simd`, no intrinsics | `std::simd` portable SIMD + compiler auto-vec |
| **Distance abstraction** | `Fn(&I, &I) -> T` closure-based | `Distance` trait with associated `Point` type |
| **Tree type** | Generic `Tree<T, A>` with annotation type param | `ClamTree<D: Distance>` with `Cluster` struct |
| **Partition strategy** | Configurable: MaxSplit, BranchingFactor, SpanReductionFactor | Fixed binary split (paper Algorithm 1) |
| **Parallelism** | Rayon-based `par_partition` | None (single-threaded) |
| **Graph induction** | ❌ WIP | ❌ Not attempted |
| **CHAODA** | ❌ WIP | ❌ Not attempted |
| **Compression** | ❌ Not present | ✅ panCAKES XOR-diff |
| **Distance functions** | L1, L2, cosine, Hamming, Needleman-Wunsch, sets, correlations | Hamming only |
| **Serialization** | serde + databuf | None |

---

## 5. What's Still Missing: Research → Implementation Gaps

### 5.1 Critical Gaps (CHAODA pipeline — absent everywhere)

These components exist ONLY in the Python `pyclam` (archived) and in the paper:

1. **Graph Induction** (§2.3): Given a set of selected clusters, build G=(V,E) where edges connect overlapping clusters (d(c₁, c₂) ≤ r₁ + r₂). This is the foundation for all CHAODA algorithms.

2. **Cluster Selection** (§2.6, Alg 4): Meta-ML models predict which clusters from the tree would build a graph with high ROC AUC. Requires training phase with labeled datasets.

3. **Six Anomaly Algorithms** (§2.4):
   - Relative Cluster Cardinality — O(|V|)
   - Relative Component Cardinality — O(|E|+|V|)
   - Graph Neighborhood Size — O(|E|·|V|) — BFS with eccentricity-scaled depth
   - Child-Parent Cardinality Ratio — O(|V|) — memoized during tree build
   - Stationary Probabilities — O(|V|^2.37) — transition matrix convergence
   - Relative Vertex Degree — O(|V|)

4. **Gaussian Score Normalization** (§7.8, Alg 5): `score = ½(1 + erf((s-μ)/(σ√2)))`

5. **Ensemble Aggregation** (§2.7): Mean of normalized scores from all (distance × algorithm × meta-ML model) combinations.

### 5.2 PR #21-Specific Gaps (Transition/Subsumed Edge Split)

PR #21 introduced a refined graph topology:

- **Subsumed clusters**: center of cluster A lies within radius of cluster B → A is subsumed by B
- **Transition edges**: connect non-subsumed clusters
- **Subsumed edges**: connect subsumed clusters to their absorbers
- **Separate edge dictionaries**: `transition_edges` and `subsumed_edges` enable different scoring strategies

This was implemented in Python only and never ported to Rust in either upstream or rustynum.

### 5.3 Distance Function Gaps in rustynum-clam

Only `HammingDistance` is implemented. Missing:

| Distance | Paper Usage | SIMD Opportunity |
|---|---|---|
| **Euclidean (L2)** | CHESS §3.2 (APOGEE) + CHAODA training | `f32x16` squared diff + horizontal sum |
| **Manhattan (L1)** | CHAODA Table 2 | `f32x16` abs diff + horizontal sum |
| **Cosine** | CHESS §3.2 (APOGEE) | Dot product / (norm × norm) using `f32x16` |
| **Hamming on bitpacked** | Already have | Add VPOPCNTDQ intrinsic path |
| **Levenshtein/Edit** | CHESS §2.1.2 (GreenGenes) | SIMD-parallelized NW in `distances` crate |
| **Jaccard (sets)** | CHAODA §5 (future) | Intersection/union via SIMD bit ops |

### 5.4 Performance/Optimization Gaps

| Gap | Current State | Fix |
|---|---|---|
| **No parallelism** in rustynum-clam | Single-threaded tree construction | Add rayon + `split_at_mut` pattern (matches upstream `par_partition`) |
| **Duplicate Hamming** | `tree.rs::hamming_inline()` duplicates `hdc.rs` | Extract to `rustynum-core` shared primitive |
| **No benchmarks** | No `cargo bench` for rustynum-clam | Add criterion benches matching URI-ABD patterns |
| **No serialization** | Trees lost between sessions | Add serde for ClamTree + SearchConfig |
| **No streaming** | Must rebuild tree for new data | CHESS §4: O(log|V|) insert via tree-search with zero radius |

---

## 6. Recommendations: Implementation Priority

### Phase 1: Foundation (Complete CLAM)
1. Add `EuclideanDistance` and `CosineDistance` implementations to `rustynum-clam`
2. Add VPOPCNTDQ intrinsic path for `hamming_inline`
3. Add `rayon` parallel tree construction
4. Add serde serialization for `ClamTree`

### Phase 2: Graph Layer (CHAODA Foundation)
5. Implement `ClamGraph` type with overlapping-volume edge detection
6. Port PR #21's transition/subsumed edge split
7. Implement connected components (BFS/DFS)
8. Add `child_parent_ratios` memoization during tree construction

### Phase 3: CHAODA Algorithms
9. Implement all six anomaly scoring algorithms
10. Implement Gaussian normalization
11. Implement ensemble mean aggregation
12. Build meta-ML training pipeline (can use external regression library)

### Phase 4: AVX-512 Deep Optimization
13. Replace auto-vec popcount with `_mm512_popcnt_epi64` intrinsics
14. Add VNNI `_mm512_dpbusd_epi32` for INT8 dot product
15. Add BF16 fused dot with `_mm512_dpbf16_ps`
16. Benchmark against URI-ABD/clam's `distances` crate

---

## 7. Paper Markdown Conversions

> The full CHESS and CHAODA papers have been converted from the uploaded PDFs. See companion files:
> - `CHESS_Paper.md` — CHESS: Clustered Hierarchical Entropy-Scaling Search (arXiv:1908.08551v2)  
> - `CHAODA_Paper.md` — CHAODA: Clustered Hierarchical Anomaly and Outlier Detection Algorithms (arXiv:2103.11774v2)
>
> *(To be posted as Part 2 per user request)*

---

## Appendix A: Key Equations Reference

### A.1 Local Fractal Dimension (CHESS Eq 2, CHAODA Eq 1)
```
LFD(q, r) = log₂( |B_X(q, r)| / |B_X(q, r/2)| )
```

### A.2 Entropy-Scaling Search Complexity (CHESS Eq 3)
```
O( log₂(k) + |B_D(q,r)| · ((r + 2r̂_c) / r)^d )
```
where k = leaf clusters, r̂_c = mean leaf cluster radius, d = fractal dimension.

### A.3 CHAODA Anomaly Score Normalization (Alg 5)
```
normalized(p) = ½ · (1 + erf((score(p) - μ) / (σ · √2)))
```

### A.4 CHAODA Child-Parent Ratio EMA (§2.5)
```
ema_{i+1} = α · R_{i+1} + (1 - α) · ema_i,   α = 2/11
```

### A.5 Stationary Probability (§2.4.5)
```
M ← transition matrix (inversely proportional to inter-center distance)
Repeat: M ← M² until convergence
score(c) = -Σ(row corresponding to c in converged M)
```

---

## Appendix B: Dataset Summary (CHAODA Benchmarks)

| Dataset | n | dim | Outliers | % |
|---|---|---|---|---|
| annthyroid | 7,200 | 6 | 534 | 7.42 |
| arrhythmia | 452 | 274 | 66 | 15 |
| breastw | 683 | 9 | 239 | 35 |
| cardio | 1,831 | 21 | 176 | 9.6 |
| cover | 286,048 | 10 | 2,747 | 0.9 |
| glass | 214 | 9 | 9 | 4.2 |
| http | 567,479 | 4 | 2,211 | 0.4 |
| ionosphere | 351 | 33 | 126 | 36 |
| lympho | 148 | 18 | 6 | 4.1 |
| mammography | 11,183 | 6 | 260 | 2.32 |
| mnist | 7,603 | 100 | 700 | 9.2 |
| musk | 3,062 | 166 | 97 | 3.2 |
| optdigits | 5,216 | 64 | 150 | 3 |
| pendigits | 6,870 | 16 | 156 | 2.27 |
| pima | 768 | 8 | 268 | 35 |
| satellite | 6,435 | 36 | 2,036 | 32 |
| satimage-2 | 5,803 | 36 | 71 | 1.2 |
| shuttle | 59,097 | 9 | 3,511 | 7 |
| smtp | 95,156 | 3 | 30 | 0.03 |
| thyroid | 3,772 | 6 | 93 | 2.5 |
| vertebral | 240 | 6 | 30 | 12.5 |
| vowels | 1,456 | 12 | 50 | 3.4 |
| wbc | 278 | 30 | 21 | 5.6 |
| wine | 129 | 13 | 10 | 7.7 |
| **APOGEE2** | **528,319** | **8,575** | N/A | N/A |

---

*End of gap analysis. Awaiting Part 2 documents for continued conversion.*
