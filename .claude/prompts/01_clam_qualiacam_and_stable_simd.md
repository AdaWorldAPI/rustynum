# Dual Task: CLAM → QualiaCAM + portable_simd → std::arch Stabilization

## Implementation Prompt for Claude Code

**Jan Hübener — Ada Architecture — March 2026**

---

## WHY THESE TWO TASKS TOGETHER

They unblock each other:

1. **`portable_simd` → `std::arch` port** removes the nightly requirement from rustyblas, rustymkl, rustynum-rs, and qualia_xor. This enables stable Rust 1.93+ across the entire workspace and unblocks deleting the 348-line duplicate SIMD in `ladybug-rs/src/core/simd.rs`.

2. **CLAM → QualiaCAM** wires the abd-clam crate into the QualiaCAM for formal search guarantees (CAKES), anomaly detection (CHAODA), and compression (panCAKES). This requires ladybug-rs to depend on rustynum SIMD cleanly — which requires stable toolchain alignment.

Task 1 first, Task 2 second. But plan both now so the SIMD port makes the right choices for CLAM integration.

---

## TASK 1: portable_simd → std::arch Port

### 1.1 Scope

The open ends inventory (CLAUDE.md § 13, P2 #8) identifies ~879 call sites across:

```
rustyblas/     — GEMM, INT8 operations
rustymkl/      — VML vectorized math
rustynum-rs/   — core HDC primitives (bind, bundle, permute, hamming)
qualia_xor/    — qualia operations (currently won't compile, 10 errors)
```

All use `std::simd::{u8x64, u16x32, u32x16, f32x16, i8x64, ...}` from `portable_simd`.

### 1.2 Replacement Pattern

Each `portable_simd` type maps to a `std::arch::x86_64` intrinsic:

```rust
// BEFORE (nightly, portable_simd)
#![feature(portable_simd)]
use std::simd::{u8x64, SimdUint};

fn xor_chunk(a: &[u8; 64], b: &[u8; 64]) -> [u8; 64] {
    let va = u8x64::from_array(*a);
    let vb = u8x64::from_array(*b);
    (va ^ vb).to_array()
}

// AFTER (stable, std::arch)
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx512bw")]
unsafe fn xor_chunk(a: &[u8; 64], b: &[u8; 64]) -> [u8; 64] {
    let va = _mm512_loadu_si512(a.as_ptr() as *const __m512i);
    let vb = _mm512_loadu_si512(b.as_ptr() as *const __m512i);
    let result = _mm512_xor_si512(va, vb);
    let mut out = [0u8; 64];
    _mm512_storeu_si512(out.as_mut_ptr() as *mut __m512i, result);
    out
}
```

### 1.3 Dispatch Strategy

ladybug-rs already has the right pattern — runtime feature detection with fallback:

```rust
/// The pattern to follow for ALL SIMD operations
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") {
            return unsafe { hamming_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { hamming_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { hamming_neon(a, b) };
        }
    }
    hamming_scalar(a, b)
}
```

### 1.4 Priority Mapping

```
CRITICAL (blocks ladybug-rs unification):
  rustynum-rs/src/operations.rs     — bind, bundle, permute, hamming
  rustynum-rs/src/num_array.rs      — core array operations
  
HIGH (blocks CI):
  qualia_xor/                       — 10 compile errors from missing deps
                                      Fix deps FIRST, then port SIMD
  
MEDIUM (performance, not blocking):
  rustyblas/src/int8_gemm.rs        — INT8 GEMM (VNNI path)
  rustyblas/src/matmul.rs           — FP32/FP64 GEMM
  rustymkl/src/vml.rs               — vectorized exp/sigmoid/tanh
  
LOW (already working):
  ladybug-rs/src/core/simd.rs       — DELETE after rustynum provides stable SIMD
```

### 1.5 The Delete

Once rustynum-rs exports stable SIMD distance functions, ladybug-rs can:

```
1. Add dependency: rustynum-rs (no nightly features)
2. Replace all calls to ladybug::core::simd::* with rustynum equivalents
3. DELETE ladybug-rs/src/core/simd.rs (348 lines)
4. Remove #![feature(...)] from ladybug-rs lib.rs if any remain
5. ladybug-rs is now fully stable Rust
```

### 1.6 Verification

```bash
# After port, this must work on stable:
rustup run stable cargo test --workspace
rustup run stable cargo clippy --workspace -- -D warnings

# Benchmark regression check:
cargo bench -- hamming_distance  # must be within 5% of nightly
cargo bench -- xor_bind          # must be within 5% of nightly
cargo bench -- bundle_majority   # must be within 5% of nightly
```

---

## TASK 2: CLAM → QualiaCAM Integration

### 2.1 Reference Document

The full CLAM hardening plan is already committed:
**`ladybug-rs/docs/CLAM_HARDENING.md`**
(https://github.com/AdaWorldAPI/ladybug-rs/blob/main/docs/CLAM_HARDENING.md)

Read that document in full before proceeding. It contains:
- 8 sections covering CLAM tree, LFD estimation, triangle inequality bounds, panCAKES compression, CHAODA anomaly detection, HDR-stacked CRP distributions, DistanceValue trait, and causal certificate chain
- Academic references: CHESS (arXiv:1908.08551), CHAODA (arXiv:2103.11774), CAKES (arXiv:2309.05491), panCAKES (arXiv:2409.12161)
- Rust implementation reference: https://github.com/URI-ABD/clam (MIT, pure Rust, 409 commits)
- What to keep vs what CLAM replaces/proves

### 2.2 The Specific Wire: QualiaCAM

QualiaCAM (implemented in rustynum, PR #77 area) currently has:

```rust
/// Content-addressable memory for qualia — 231 items, 18 bytes each
/// PROBLEM: locate() is O(N) brute-force scan
pub struct QualiaCAM {
    corpus: Vec<PackedQualia>,
    tuning_forks: Vec<TuningFork>,
    families: Vec<QualiaFamily>,
}

impl QualiaCAM {
    /// O(N) exhaustive search — fine for 231, bad for 1024+
    pub fn locate(&self, query: &PackedQualia) -> Option<QualiaMatch> {
        self.corpus.iter()
            .enumerate()
            .map(|(i, q)| (i, q.hamming_distance(query)))
            .min_by_key(|(_, d)| *d)
    }
}
```

Replace the O(N) scan with CLAM-backed search:

```rust
use abd_clam::{Tree, Cluster};
use abd_clam::cakes::{KnnBranch, RnnChess};

pub struct QualiaCAM {
    corpus: Vec<PackedQualia>,
    families: Vec<QualiaFamily>,
    
    // NEW: CLAM tree for sublinear search
    clam_tree: Tree<PackedQualia, u32>,
    
    // NEW: per-cluster CRP distributions (our HDR contribution — exceeds CLAM's scalar radius)
    cluster_distributions: Vec<ClusterDistribution>,
    
    // NEW: shift detector for Schaltsekunde/Schaltminute
    shift_detector: ShiftDetector,
}

impl QualiaCAM {
    pub fn build(corpus: Vec<PackedQualia>, families: Vec<QualiaFamily>) -> Self {
        let clam_tree = Tree::par_new_minimal(
            &corpus,
            |a: &PackedQualia, b: &PackedQualia| -> u32 {
                a.hamming_distance(b)
            },
        ).expect("CLAM tree construction");
        
        let cluster_distributions = clam_tree.all_clusters()
            .iter()
            .map(|cluster| {
                ClusterDistribution::from_hdr_measurements(
                    &corpus[cluster.center_index()],
                    &cluster.member_indices().iter()
                        .map(|&i| &corpus[i])
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        
        Self { corpus, families, clam_tree, cluster_distributions,
               shift_detector: ShiftDetector::new() }
    }
    
    /// CAKES exact k-NN — O(k · 2^LFD · log n) instead of O(N)
    pub fn locate(&self, query: &PackedQualia, k: usize) -> Vec<QualiaMatch> {
        KnnBranch(k).par_search(&self.clam_tree, query)
            .into_iter()
            .map(|(idx, dist)| QualiaMatch {
                index: idx,
                qualia: self.corpus[idx].clone(),
                distance: dist,
                family: self.family_of(idx),
                sigma: self.score_sigma(idx, dist),
            })
            .collect()
    }
    
    /// Ranged search — all qualia within threshold (replaces hardcoded Mexican hat)
    pub fn locate_within(&self, query: &PackedQualia, threshold: u32) -> Vec<QualiaMatch> {
        RnnChess(threshold).par_search(&self.clam_tree, query)
            .into_iter()
            .map(|(idx, dist)| QualiaMatch {
                index: idx,
                qualia: self.corpus[idx].clone(),
                distance: dist,
                family: self.family_of(idx),
                sigma: self.score_sigma(idx, dist),
            })
            .collect()
    }
    
    /// CHAODA anomaly score
    pub fn anomaly_score(&self, query: &PackedQualia) -> AnomalyResult {
        let leaf = self.clam_tree.find_leaf(query);
        let score = (leaf.depth() as f64 / self.clam_tree.max_depth() as f64)
            * (1.0 - leaf.cardinality() as f64 / self.corpus.len() as f64)
            * leaf.lfd();
        
        AnomalyResult {
            score,
            calibration_type: if score > SCHALTMINUTE_THRESHOLD {
                CalibrationType::Schaltminute   // topological change
            } else if score > SCHALTSEKUNDE_THRESHOLD {
                CalibrationType::Schaltsekunde  // fine grid adjustment
            } else {
                CalibrationType::None
            },
            lfd: leaf.lfd(),
            cluster_depth: leaf.depth(),
            cluster_cardinality: leaf.cardinality(),
        }
    }
    
    /// LFD pruning report
    pub fn pruning_report(&self) -> PruningReport {
        let lfds: Vec<f64> = self.clam_tree.all_clusters()
            .iter().map(|c| c.lfd()).collect();
        let mean_lfd = lfds.iter().sum::<f64>() / lfds.len() as f64;
        PruningReport {
            mean_lfd,
            max_lfd: lfds.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            theoretical_speedup: 2f64.powf((self.corpus.len() as f64).log2() - mean_lfd),
            corpus_size: self.corpus.len(),
        }
    }
}
```

### 2.3 CLAM + SPO Distance Harvest

Build CLAM tree over `SpatialCrystal3D` using SPO distance as metric. Every tree traversal produces typed halo harvest:

```rust
pub struct SpoClamTree {
    tree: Tree<SpatialCrystal3D, SpoDistanceResult>,
    accumulated_harvest: AccumulatedHarvest,
}

impl SpoClamTree {
    pub fn build(corpus: &[SpatialCrystal3D]) -> Self {
        let tree = Tree::par_new_minimal(corpus,
            |a: &SpatialCrystal3D, b: &SpatialCrystal3D| -> SpoDistanceResult {
                spo_distance(a, b)
            },
        ).expect("SPO CLAM tree");
        Self { tree, accumulated_harvest: AccumulatedHarvest::new() }
    }
    
    /// Every tree search enriches the knowledge graph
    pub fn search_with_harvest(&mut self, query: &SpatialCrystal3D, k: usize) 
        -> Vec<SpoSearchResult> 
    {
        let hits = KnnBranch(k).par_search(&self.tree, query);
        hits.into_iter().map(|(idx, spo_dist)| {
            self.accumulated_harvest.accumulate(&spo_dist);
            SpoSearchResult {
                index: idx, distance: spo_dist,
                inference: harvest_to_inference(&spo_dist),
                nars_truth: harvest_to_nars(&spo_dist),
            }
        }).collect()
    }
}
```

### 2.4 CHAODA → CollapseGate Wire

```rust
fn anomaly_to_gate_bias(anomaly: &AnomalyResult) -> CollapseGateBias {
    match anomaly.calibration_type {
        CalibrationType::Schaltminute => CollapseGateBias::Hold {
            reason: "CHAODA: new family detected, topology shifting",
        },
        CalibrationType::Schaltsekunde => CollapseGateBias::FlowWithCaution {
            confidence_multiplier: 0.8,
            reason: "CHAODA: slight distributional shift",
        },
        CalibrationType::None => CollapseGateBias::Flow,
    }
}

/// Cross-validate CHAODA with σ-stripe shift detector
fn chaoda_confirms_shift(
    anomaly: &AnomalyResult,
    shift: &ShiftSignal,
) -> ConfirmedShift {
    match (anomaly.calibration_type, shift.direction) {
        (CalibrationType::Schaltminute, ShiftDirection::TowardNoise) =>
            ConfirmedShift::GlobalDrift { confidence: 0.95 },
        (CalibrationType::Schaltsekunde, ShiftDirection::TowardFoveal) =>
            ConfirmedShift::LocalRefinement { confidence: 0.85 },
        (CalibrationType::Schaltminute, ShiftDirection::Bimodal) =>
            ConfirmedShift::Speciation { confidence: 0.90 },
        _ => ConfirmedShift::Monitoring { confidence: 0.5 },
    }
}
```

### 2.5 Triangle Inequality + σ-Significance Combined Pruning

```rust
/// Formally guaranteed cluster pruning with statistical significance
fn can_cluster_contain_significant_match(
    cluster: &Cluster<u32, ()>,
    d_to_center: u32,
    sigma_gate: &SigmaGate,
) -> ClusterVerdict {
    let d_min = d_to_center.saturating_sub(cluster.radius());
    let d_max = d_to_center + cluster.radius();
    
    let best_sigma = sigma_gate.score_sigma(d_min);
    let worst_sigma = sigma_gate.score_sigma(d_max);
    
    match best_sigma.level {
        SignificanceLevel::Noise | SignificanceLevel::Hint =>
            ClusterVerdict::Prune,      // formally safe: nothing significant possible
        SignificanceLevel::Discovery | SignificanceLevel::Strong
            if worst_sigma.level >= SignificanceLevel::Evidence =>
            ClusterVerdict::AcceptAll,   // everything significant
        _ => ClusterVerdict::Scan,       // mixed — scan cluster members
    }
}
```

---

## IMPLEMENTATION PHASES

### Phase 1: stable SIMD port (Task 1)
```
1. Port rustynum-rs SIMD: portable_simd → std::arch (CRITICAL path)
2. Fix qualia_xor compile errors (missing deps first, then SIMD)
3. Port rustyblas SIMD (INT8 GEMM, FP matmul)
4. Port rustymkl SIMD (VML)
5. Verify: `rustup run stable cargo test --workspace`
6. Verify: benchmarks within 5% of nightly
7. DELETE ladybug-rs/src/core/simd.rs (348 lines)
8. ladybug-rs fully stable
```

### Phase 2: CLAM validation
```
1. Add abd-clam dependency to ladybug-rs
2. Build CLAM tree from existing fingerprint corpus
3. Benchmark CAKES vs HDR cascade on same queries
4. Measure LFD — validate pruning claims
5. Compare d_min/d_max vs scent L1 filtering
```

### Phase 3: QualiaCAM wire
```
1. Add CLAM tree to QualiaCAM struct
2. Replace locate() O(N) with CAKES KnnBranch
3. Add locate_within() via RnnChess
4. Add anomaly_score() via CHAODA
5. Wire into CollapseGate + ShiftDetector
6. Add pruning_report()
```

### Phase 4: SPO + CLAM fusion
```
1. Build SpoClamTree over SpatialCrystal3D corpus
2. Wire search_with_harvest() — tree traversal enriches AccumulatedHarvest
3. Compute per-plane LFD
4. Combine triangle inequality + σ-significance
5. Benchmark vs brute-force SPO distance
```

### Phase 5: panCAKES compression
```
1. CompressedQualia with XOR-diff from cluster center
2. Wire into Lance storage
3. Compressive search (Hamming on diffs, no decompression)
4. Benchmark compression ratio at 231, 1024, 4096, 16384 items
```

### Phase 6: CHAODA consciousness
```
1. Full CHAODA scorer (all graph metrics from arXiv:2103.11774)
2. Wire into Schaltsekunde/Schaltminute
3. Cross-validate with stripe histogram
4. DN mutation rate from anomaly score
```

---

## WHAT NOT TO CHANGE

```
KEEP: AVX-512 VPOPCNTDQ (more specialized than CLAM's distances crate)
KEEP: BindSpace (O(1) content-addressable — CLAM doesn't have this)
KEEP: XOR retrieval (A⊗verb⊗B=A — VSA-specific)
KEEP: Arrow/Lance integration (columnar, not Vec<(Id, I)>)
KEEP: COW immutability (freeze after build)
KEEP: HDR cascade + σ-significance (CLAM has no multi-resolution cascade)
KEEP: SPO typed halo harvest (CLAM has no equivalent — novel contribution)
```

## DEPENDENCY

```toml
# ladybug-rs/Cargo.toml
[dependencies]
abd-clam = { version = "0.35", features = ["serde"] }  # check latest
# Pure Rust, stable toolchain, MIT license
```

## TEST EXPECTATIONS

```bash
# After BOTH tasks:
rustup run stable cargo test --workspace     # 752+ tests, zero nightly
grep -r "feature(portable_simd)" --include="*.rs" .  # zero matches
test ! -f ladybug-rs/src/core/simd.rs        # deleted
cargo test --package ladybug-rs -- clam      # CLAM tree builds, CAKES works
```
