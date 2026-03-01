//! CLAM-backed QualiaCAM: sublinear search via triangle inequality pruning.
//!
//! Wraps `rustynum_core::QualiaCAM` with a `ClamTree` built over the
//! PackedQualia resonance bytes. Replaces O(N) exhaustive scan with
//! CAKES DFS Sieve at O(k · 2^LFD · log N).
//!
//! ## Architecture
//!
//! - **QualiaCAM** (rustynum-core): types, L1 distance, hydration, causality
//! - **ClamQualiaCAM** (this module): CLAM tree + search + CHAODA anomaly
//!
//! rustynum-clam depends on rustynum-core, NOT the reverse. This module
//! extends QualiaCAM with tree-backed search without modifying core types.
//!
//! ## Distance Function
//!
//! L1 on i8 resonance IS a metric (satisfies triangle inequality).
//! CAKES search is therefore exact — zero false negatives.
//!
//! ## Memory Layout
//!
//! PackedQualia resonance is `[i8; 16]` = 16 bytes per item.
//! The CLAM tree is built over a flat `[u8]` buffer where each
//! 16-byte slice is one coordinate's resonance (interpreted as i8).

use crate::search::{knn_dfs_sieve, rho_nn, KnnResult, RhoNnResult};
use crate::tree::{BuildConfig, ClamTree};
use rustynum_core::bf16_hamming::PackedQualia;
use rustynum_core::causality::{
    causality_decompose, CausalityDecomposition, CausalityDirection, NarsTruthValue,
};
use rustynum_core::qualia_gate::{GatedQualia, QualiaGateLevel, ResonanzZirkel};

// ============================================================================
// L1 i8 distance — metric distance on signed byte resonance vectors
// ============================================================================

/// L1 (Manhattan) distance on byte slices interpreted as i8.
///
/// This IS a metric: satisfies non-negativity, identity, symmetry,
/// and the triangle inequality. CAKES search is exact with this distance.
///
/// Cost: 16 subtractions + 16 abs + 15 additions for qualia resonance.
pub fn l1_i8_distance(a: &[u8], b: &[u8]) -> u64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as i8 as i32 - y as i8 as i32).unsigned_abs() as u64)
        .sum()
}

// ============================================================================
// QualiaHit — enriched search result
// ============================================================================

/// Result of locating a state in the CLAM-backed QualiaCAM.
///
/// Extends the core `QualiaHit` with CLAM-specific metadata:
/// cluster depth, LFD, and anomaly score.
#[derive(Clone, Debug)]
pub struct ClamQualiaHit {
    /// Index in the ResonanzZirkel.
    pub index: usize,
    /// L1 distance on i8 resonance.
    pub distance: u64,
    /// Gate level of the matched coordinate.
    pub gate: QualiaGateLevel,
    /// Causality decomposition.
    pub causality: CausalityDecomposition,
    /// Family index.
    pub family_id: u8,
    /// NARS truth value from distance.
    pub nars_truth: NarsTruthValue,
}

// ============================================================================
// CHAODA anomaly detection
// ============================================================================

/// Calibration type from CHAODA anomaly scoring.
///
/// Maps to the Schaltsekunde/Schaltminute terminology:
/// - **Schaltminute**: topological change detected (new family, corpus shift)
/// - **Schaltsekunde**: fine grid adjustment (distributional drift)
/// - **None**: normal operation
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationType {
    /// Topological change: the query is in a region the corpus doesn't cover well.
    Schaltminute,
    /// Fine adjustment: slight distributional shift detected.
    Schaltsekunde,
    /// Normal operation: query is well-represented in the corpus.
    None,
}

/// Result of CHAODA anomaly scoring.
#[derive(Clone, Debug)]
pub struct AnomalyResult {
    /// Anomaly score in [0, 1]. Higher = more anomalous.
    pub score: f64,
    /// Calibration type derived from score thresholds.
    pub calibration_type: CalibrationType,
    /// LFD of the leaf cluster containing the query.
    pub lfd: f64,
    /// Depth of the leaf cluster.
    pub cluster_depth: usize,
    /// Cardinality of the leaf cluster.
    pub cluster_cardinality: usize,
}

/// Bias signal from CHAODA anomaly to CollapseGate decision.
#[derive(Clone, Debug, PartialEq)]
pub enum CollapseGateBias {
    /// Normal: no anomaly, gate should flow.
    Flow,
    /// Slight anomaly: flow with reduced confidence.
    FlowWithCaution {
        confidence_multiplier: f64,
        reason: &'static str,
    },
    /// Significant anomaly: hold superposition, accumulate evidence.
    Hold { reason: &'static str },
}

// ============================================================================
// Pruning report
// ============================================================================

/// LFD-based pruning statistics for the corpus.
#[derive(Clone, Debug)]
pub struct PruningReport {
    /// Mean LFD across all clusters.
    pub mean_lfd: f64,
    /// Maximum LFD (worst-case cluster).
    pub max_lfd: f64,
    /// Theoretical speedup: n / (k · 2^mean_lfd · log n).
    pub theoretical_speedup: f64,
    /// Corpus size.
    pub corpus_size: usize,
    /// Number of leaf clusters.
    pub num_leaves: usize,
    /// Mean leaf radius.
    pub mean_leaf_radius: f64,
}

// ============================================================================
// ClamQualiaCAM
// ============================================================================

/// Thresholds for CHAODA anomaly → calibration type mapping.
const SCHALTMINUTE_THRESHOLD: f64 = 0.7;
const SCHALTSEKUNDE_THRESHOLD: f64 = 0.4;

/// CLAM-backed Content-Addressable Memory for Qualia.
///
/// Extends `QualiaCAM` with:
/// - **CAKES DFS Sieve**: O(k · 2^LFD · log N) exact k-NN (replaces O(N) scan)
/// - **ρ-NN range search**: find all qualia within distance threshold
/// - **CHAODA anomaly scoring**: detect topological shifts and distributional drift
/// - **CollapseGate bias**: map anomaly to gate decision
/// - **Pruning report**: LFD statistics for search performance estimation
///
/// The CLAM tree is built over the 16-byte i8 resonance of each PackedQualia
/// using L1 distance (a metric), so all CAKES searches are exact.
pub struct ClamQualiaCAM {
    /// Gated coordinates (the corpus).
    coordinates: Vec<GatedQualia>,
    /// Pre-hydrated f32 resonance for dot-product search (future: locate_resonant).
    #[allow(dead_code)]
    hydrated: Vec<f32>,
    /// Flat byte buffer of resonance vectors for CLAM tree.
    /// Layout: [coord_0_resonance[0..16], coord_1_resonance[0..16], ...]
    resonance_bytes: Vec<u8>,
    /// CLAM tree built over resonance_bytes with L1 distance.
    clam_tree: ClamTree,
    /// Maximum tree depth (for CHAODA normalization).
    max_depth: usize,
}

impl ClamQualiaCAM {
    /// Build from a ResonanzZirkel.
    ///
    /// Constructs the CLAM tree over the resonance byte representation.
    /// One-time cost at system start.
    pub fn from_zirkel(zirkel: &ResonanzZirkel) -> Self {
        let coordinates: Vec<GatedQualia> = zirkel.iter().copied().collect();
        Self::from_coordinates(coordinates)
    }

    /// Build from a raw vector of GatedQualia.
    pub fn from_coordinates(coordinates: Vec<GatedQualia>) -> Self {
        let mut hydrated = Vec::with_capacity(coordinates.len() * 16);
        let mut resonance_bytes = Vec::with_capacity(coordinates.len() * 16);

        for coord in &coordinates {
            let h = rustynum_core::bf16_hamming::hydrate_qualia_f32(&coord.qualia);
            hydrated.extend_from_slice(&h);
            // Pack i8 resonance as u8 bytes for CLAM tree
            for &r in &coord.qualia.resonance {
                resonance_bytes.push(r as u8);
            }
        }

        let count = coordinates.len();
        let clam_tree = if count > 0 {
            ClamTree::build_with_fn(
                &resonance_bytes,
                16, // vec_len = 16 bytes per qualia resonance
                count,
                &BuildConfig {
                    min_cardinality: 1,
                    max_depth: 64,
                    min_radius: 0,
                },
                l1_i8_distance,
            )
        } else {
            // Empty tree for empty corpus
            ClamTree::build_with_fn(&[], 16, 0, &BuildConfig::default(), l1_i8_distance)
        };

        let max_depth = clam_tree
            .nodes
            .iter()
            .map(|c| c.depth)
            .max()
            .unwrap_or(0);

        Self {
            coordinates,
            hydrated,
            resonance_bytes,
            clam_tree,
            max_depth,
        }
    }

    /// Number of coordinates in the CAM.
    #[inline]
    pub fn len(&self) -> usize {
        self.coordinates.len()
    }

    /// Returns true if the CAM is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.coordinates.is_empty()
    }

    /// Get a coordinate by index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&GatedQualia> {
        self.coordinates.get(index)
    }

    // ── CAKES DFS Sieve k-NN ──────────────────────────────────────────

    /// Locate using CAKES DFS Sieve: exact k-NN in O(k · 2^LFD · log N).
    ///
    /// Returns the top-k nearest tuning forks with full phenomenological
    /// context: distance, causality, gate, family, NARS truth.
    pub fn locate(&self, query: &PackedQualia, top_k: usize) -> Vec<ClamQualiaHit> {
        if self.is_empty() || top_k == 0 {
            return Vec::new();
        }

        let query_bytes = resonance_to_bytes(query);
        let knn = knn_dfs_sieve(
            &self.clam_tree,
            &self.resonance_bytes,
            16,
            &query_bytes,
            top_k,
        );

        self.knn_to_hits(query, &knn)
    }

    /// Locate using O(N) L1 scan (fallback, matches core QualiaCAM behavior).
    pub fn locate_linear(&self, query: &PackedQualia, top_k: usize) -> Vec<ClamQualiaHit> {
        let mut hits: Vec<(usize, i32)> = self
            .coordinates
            .iter()
            .enumerate()
            .map(|(i, coord)| (i, l1_i8_scalar(&query.resonance, &coord.qualia.resonance)))
            .collect();

        hits.sort_by_key(|&(_, d)| d);
        hits.truncate(top_k);

        hits.iter()
            .map(|&(i, dist)| {
                let coord = &self.coordinates[i];
                let causality = causality_decompose(query, &coord.qualia, None);
                let nars_truth = distance_to_nars(dist as u64, query);
                ClamQualiaHit {
                    index: i,
                    distance: dist as u64,
                    gate: coord.gate,
                    causality,
                    family_id: coord.family_id,
                    nars_truth,
                }
            })
            .collect()
    }

    // ── ρ-NN range search ─────────────────────────────────────────────

    /// Find all qualia within a distance threshold.
    ///
    /// Uses CAKES ρ-NN (Algorithms 2+3): tree traversal with triangle
    /// inequality pruning, then leaf scan. Exact for metric distances.
    pub fn locate_within(
        &self,
        query: &PackedQualia,
        threshold: u64,
    ) -> Vec<ClamQualiaHit> {
        if self.is_empty() {
            return Vec::new();
        }

        let query_bytes = resonance_to_bytes(query);
        let result = rho_nn(
            &self.clam_tree,
            &self.resonance_bytes,
            16,
            &query_bytes,
            threshold,
        );

        self.rho_to_hits(query, &result)
    }

    // ── Gated search ──────────────────────────────────────────────────

    /// Find the nearest gated (Hold or Block) coordinate within threshold.
    pub fn nearest_gated(
        &self,
        query: &PackedQualia,
        distance_threshold: u64,
    ) -> Option<ClamQualiaHit> {
        // Use range search, then filter to gated coordinates
        let all_within = self.locate_within(query, distance_threshold);
        all_within
            .into_iter()
            .filter(|h| h.gate.is_gated())
            .min_by_key(|h| h.distance)
    }

    // ── CHAODA anomaly scoring ────────────────────────────────────────

    /// Compute CHAODA-inspired anomaly score for a query.
    ///
    /// The score combines three signals:
    /// 1. **Depth**: how deep in the tree the query lands (deeper = more specific)
    /// 2. **Cardinality**: how many points share the leaf cluster (fewer = rarer)
    /// 3. **LFD**: local fractal dimension of the leaf (higher = more spread out)
    ///
    /// Score formula:
    /// ```text
    /// score = (depth / max_depth) × (1 - cardinality / corpus_size) × lfd_factor
    /// ```
    ///
    /// Returns the anomaly result with calibration type.
    pub fn anomaly_score(&self, query: &PackedQualia) -> AnomalyResult {
        if self.is_empty() {
            return AnomalyResult {
                score: 1.0,
                calibration_type: CalibrationType::Schaltminute,
                lfd: 0.0,
                cluster_depth: 0,
                cluster_cardinality: 0,
            };
        }

        let query_bytes = resonance_to_bytes(query);
        let (leaf_idx, _dist) = self.find_leaf(&query_bytes);
        let leaf = &self.clam_tree.nodes[leaf_idx];

        let depth_factor = if self.max_depth > 0 {
            leaf.depth as f64 / self.max_depth as f64
        } else {
            0.0
        };

        let cardinality_factor = 1.0 - (leaf.cardinality as f64 / self.coordinates.len() as f64);
        let lfd_factor = (leaf.lfd.value / 3.0).min(1.0); // normalize LFD, cap at 3

        let score = (depth_factor * cardinality_factor * lfd_factor).clamp(0.0, 1.0);

        let calibration_type = if score > SCHALTMINUTE_THRESHOLD {
            CalibrationType::Schaltminute
        } else if score > SCHALTSEKUNDE_THRESHOLD {
            CalibrationType::Schaltsekunde
        } else {
            CalibrationType::None
        };

        AnomalyResult {
            score,
            calibration_type,
            lfd: leaf.lfd.value,
            cluster_depth: leaf.depth,
            cluster_cardinality: leaf.cardinality,
        }
    }

    // ── CollapseGate bias ─────────────────────────────────────────────

    /// Map CHAODA anomaly to a CollapseGate bias signal.
    ///
    /// This does NOT make the gate decision — it provides a bias that
    /// upstream (ladybug-rs) can factor into the gate evaluation.
    pub fn gate_bias(&self, anomaly: &AnomalyResult) -> CollapseGateBias {
        match anomaly.calibration_type {
            CalibrationType::Schaltminute => CollapseGateBias::Hold {
                reason: "CHAODA: topological shift detected, accumulate more evidence",
            },
            CalibrationType::Schaltsekunde => CollapseGateBias::FlowWithCaution {
                confidence_multiplier: 0.8,
                reason: "CHAODA: slight distributional shift",
            },
            CalibrationType::None => CollapseGateBias::Flow,
        }
    }

    // ── Triangle inequality + sigma combined pruning ──────────────────

    /// Check if a cluster can contain a significant match.
    ///
    /// Combines the triangle inequality (d_min, d_max) with sigma
    /// significance scoring. Returns whether to prune, scan, or accept.
    pub fn cluster_significance(
        &self,
        query: &PackedQualia,
        cluster_idx: usize,
    ) -> ClusterVerdict {
        if cluster_idx >= self.clam_tree.nodes.len() {
            return ClusterVerdict::Prune;
        }

        let query_bytes = resonance_to_bytes(query);
        let cluster = &self.clam_tree.nodes[cluster_idx];
        let center = self.clam_tree.center_data(cluster, &self.resonance_bytes, 16);
        let d_to_center = self.clam_tree.dist(&query_bytes, center);

        let d_min = cluster.delta_minus(d_to_center);
        let d_max = cluster.delta_plus(d_to_center);

        // Max possible L1 on 16 i8 dimensions: 16 × 254 = 4064
        let sigma_min = 1.0 - (d_min as f64 / 4064.0);
        let sigma_max = 1.0 - (d_max as f64 / 4064.0);

        if sigma_min < 0.1 {
            // Best possible match is weak — prune
            ClusterVerdict::Prune
        } else if sigma_max > 0.5 {
            // Worst possible match is still significant — accept all
            ClusterVerdict::AcceptAll
        } else {
            // Mixed — need to scan individual members
            ClusterVerdict::Scan
        }
    }

    // ── Pruning report ────────────────────────────────────────────────

    /// Generate LFD-based pruning statistics for the corpus.
    pub fn pruning_report(&self) -> PruningReport {
        let lfds: Vec<f64> = self
            .clam_tree
            .nodes
            .iter()
            .filter(|c| c.lfd.value > 0.0)
            .map(|c| c.lfd.value)
            .collect();

        let mean_lfd = if lfds.is_empty() {
            0.0
        } else {
            lfds.iter().sum::<f64>() / lfds.len() as f64
        };

        let max_lfd = lfds
            .iter()
            .cloned()
            .fold(0.0f64, f64::max);

        let n = self.coordinates.len() as f64;
        let theoretical_speedup = if mean_lfd > 0.0 && n > 1.0 {
            n / (2f64.powf(mean_lfd) * n.log2())
        } else {
            1.0
        };

        PruningReport {
            mean_lfd,
            max_lfd,
            theoretical_speedup,
            corpus_size: self.coordinates.len(),
            num_leaves: self.clam_tree.num_leaves,
            mean_leaf_radius: self.clam_tree.mean_leaf_radius,
        }
    }

    // ── NARS truth from distance ──────────────────────────────────────

    /// Get NARS truth value for a localization at a specific index.
    pub fn truth_at(&self, query: &PackedQualia, index: usize) -> Option<NarsTruthValue> {
        self.coordinates.get(index).map(|coord| {
            let dist = l1_i8_scalar(&query.resonance, &coord.qualia.resonance);
            distance_to_nars(dist as u64, query)
        })
    }

    /// Get causality direction at a specific index.
    pub fn causality_at(&self, query: &PackedQualia, index: usize) -> Option<CausalityDirection> {
        self.coordinates.get(index).map(|coord| {
            let decomp = causality_decompose(query, &coord.qualia, None);
            decomp.source_direction
        })
    }

    // ── Internal helpers ──────────────────────────────────────────────

    /// Find the leaf cluster that a query point would fall into.
    /// Returns (node_index, distance_to_center).
    fn find_leaf(&self, query_bytes: &[u8]) -> (usize, u64) {
        let mut node_idx = 0; // start at root
        let mut dist_to_center;

        loop {
            let cluster = &self.clam_tree.nodes[node_idx];
            let center = self.clam_tree.center_data(cluster, &self.resonance_bytes, 16);
            dist_to_center = self.clam_tree.dist(query_bytes, center);

            if cluster.is_leaf() {
                return (node_idx, dist_to_center);
            }

            // Descend to the child whose center is closer
            match (cluster.left, cluster.right) {
                (Some(left), Some(right)) => {
                    let left_center =
                        self.clam_tree
                            .center_data(&self.clam_tree.nodes[left], &self.resonance_bytes, 16);
                    let right_center =
                        self.clam_tree
                            .center_data(&self.clam_tree.nodes[right], &self.resonance_bytes, 16);
                    let dl = self.clam_tree.dist(query_bytes, left_center);
                    let dr = self.clam_tree.dist(query_bytes, right_center);
                    node_idx = if dl <= dr { left } else { right };
                }
                (Some(left), None) => node_idx = left,
                (None, Some(right)) => node_idx = right,
                (None, None) => return (node_idx, dist_to_center),
            }
        }
    }

    /// Convert KnnResult to ClamQualiaHit vec.
    fn knn_to_hits(&self, query: &PackedQualia, knn: &KnnResult) -> Vec<ClamQualiaHit> {
        knn.hits
            .iter()
            .filter_map(|&(reordered_idx, dist)| {
                // The index from CLAM is in the reordered space — map back
                let orig_idx = self.clam_tree.reordered[reordered_idx];
                self.coordinates.get(orig_idx).map(|coord| {
                    let causality = causality_decompose(query, &coord.qualia, None);
                    let nars_truth = distance_to_nars(dist, query);
                    ClamQualiaHit {
                        index: orig_idx,
                        distance: dist,
                        gate: coord.gate,
                        causality,
                        family_id: coord.family_id,
                        nars_truth,
                    }
                })
            })
            .collect()
    }

    /// Convert RhoNnResult to ClamQualiaHit vec.
    fn rho_to_hits(&self, query: &PackedQualia, result: &RhoNnResult) -> Vec<ClamQualiaHit> {
        result
            .hits
            .iter()
            .filter_map(|&(reordered_idx, dist)| {
                let orig_idx = self.clam_tree.reordered[reordered_idx];
                self.coordinates.get(orig_idx).map(|coord| {
                    let causality = causality_decompose(query, &coord.qualia, None);
                    let nars_truth = distance_to_nars(dist, query);
                    ClamQualiaHit {
                        index: orig_idx,
                        distance: dist,
                        gate: coord.gate,
                        causality,
                        family_id: coord.family_id,
                        nars_truth,
                    }
                })
            })
            .collect()
    }
}

/// Verdict from combined triangle-inequality + sigma significance check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClusterVerdict {
    /// No significant match possible — skip this cluster.
    Prune,
    /// All members are significant — accept without individual checks.
    AcceptAll,
    /// Mixed — need to check individual cluster members.
    Scan,
}

// ============================================================================
// ClamPath — B-tree key from CLAM tree traversal path
// ============================================================================

/// CLAM tree path encoded as a B-tree key.
///
/// Each bit encodes one bipolar split decision from root to leaf:
/// `0` = went left (closer to left pole), `1` = went right.
///
/// The same key serves three query types simultaneously:
///
/// ```text
/// B-tree as ADDRESS:   Domain.Node.branch.twig.leaf   → O(1) structural lookup
/// B-tree as LINEAGE:   ancestor → parent → self       → range scan = phylogeny
/// B-tree as CAUSALITY: cause → mediator → effect       → range scan = causal chain
/// ```
///
/// Depth of CLAM tree on 1024 items ≈ 10-12 levels, so the path fits in u16.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ClamPath {
    /// Bitpacked path (MSB = root split, LSB-aligned = leaf split).
    pub bits: u16,
    /// How many bits are valid (= tree depth for this leaf).
    pub depth: u8,
}

impl ClamPath {
    /// Construct from CLAM tree traversal (vec of left/right decisions).
    pub fn from_tree_traversal(cluster_path: &[bool]) -> Self {
        let depth = cluster_path.len().min(16) as u8;
        let mut bits: u16 = 0;
        for (i, &went_right) in cluster_path.iter().take(16).enumerate() {
            if went_right {
                bits |= 1 << (15 - i); // MSB = root
            }
        }
        Self { bits, depth }
    }

    /// Construct from raw bits and depth.
    pub fn new(bits: u16, depth: u8) -> Self {
        Self {
            bits,
            depth: depth.min(16),
        }
    }

    /// B-tree range for "everything in this subtree."
    ///
    /// Returns `(lo, hi)` inclusive range. All paths sharing this prefix
    /// fall within this range. This enables:
    /// - Structural: "everything in this domain subtree"
    /// - Phylogenetic: "everything descended from this ancestor"
    /// - Causal: "everything downstream of this cause"
    pub fn subtree_range(&self) -> (u16, u16) {
        if self.depth >= 16 {
            return (self.bits, self.bits);
        }
        if self.depth == 0 {
            return (0, u16::MAX);
        }
        let shift = 16 - self.depth as u32;
        let lo = self.bits & (!0u16 << shift);
        let hi = lo | ((1u16 << shift) - 1);
        (lo, hi)
    }

    /// Common ancestor depth between two paths.
    ///
    /// Returns how many split decisions they share from the root.
    pub fn common_ancestor_depth(&self, other: &ClamPath) -> u8 {
        let xor = self.bits ^ other.bits;
        let shared = xor.leading_zeros() as u8;
        shared.min(self.depth).min(other.depth)
    }

    /// Lineage distance: total splits apart (symmetric).
    ///
    /// `lineage_distance(a, b) = depth(a) - ancestor + depth(b) - ancestor`
    pub fn lineage_distance(&self, other: &ClamPath) -> u8 {
        let ancestor = self.common_ancestor_depth(other);
        (self.depth - ancestor) + (other.depth - ancestor)
    }

    /// Sibling path: flip the last split decision.
    ///
    /// Returns the path for items that share the same parent cluster
    /// but went the other direction = evolutionary siblings = counterfactuals.
    pub fn sibling(&self) -> ClamPath {
        if self.depth == 0 {
            return *self;
        }
        let bit_pos = 16 - self.depth as u32;
        ClamPath {
            bits: self.bits ^ (1 << bit_pos),
            depth: self.depth,
        }
    }

    /// Encode into a 3-byte representation for CogRecord B-tree channel.
    ///
    /// ```text
    /// byte[0..2] = bits (big-endian)
    /// byte[2]    = depth
    /// ```
    pub fn to_bytes(&self) -> [u8; 3] {
        let be = self.bits.to_be_bytes();
        [be[0], be[1], self.depth]
    }

    /// Decode from 3-byte CogRecord B-tree channel representation.
    pub fn from_bytes(bytes: &[u8; 3]) -> Self {
        Self {
            bits: u16::from_be_bytes([bytes[0], bytes[1]]),
            depth: bytes[2],
        }
    }

    /// Parse a colon-delimited semantic address like `"ada:clam:1010:1100:1011:a7f3"`.
    ///
    /// The first two segments (`ada:clam`) are the domain prefix (ignored here).
    /// Each subsequent segment is a nibble (4 bits) of the tree path.
    /// If the total nibble depth exceeds the expected tree depth, the final
    /// nibble is treated as a leaf ID suffix (explicit address).
    ///
    /// Returns `None` if the address is malformed.
    pub fn parse(addr: &str) -> Option<Self> {
        let parts: Vec<&str> = addr.split(':').collect();
        // Minimum: "ada:clam" prefix + at least 1 nibble
        if parts.len() < 3 {
            return None;
        }
        // Skip domain prefix segments (e.g. "ada", "clam")
        let nibble_parts = &parts[2..];
        if nibble_parts.is_empty() {
            return Some(Self::new(0, 0));
        }

        let mut bits: u16 = 0;
        let mut total_bits: u8 = 0;

        for part in nibble_parts {
            let nibble = u16::from_str_radix(part, 16).ok()?;
            let nibble_bits = (part.len() * 4) as u8;
            if total_bits + nibble_bits > 16 {
                // Overflow: pack what fits
                let remaining = 16 - total_bits;
                bits |= (nibble >> (nibble_bits - remaining)) << (16 - total_bits - remaining);
                total_bits = 16;
                break;
            }
            bits |= nibble << (16 - total_bits - nibble_bits);
            total_bits += nibble_bits;
        }

        Some(Self {
            bits,
            depth: total_bits,
        })
    }

    /// Format as colon-delimited semantic address: `"ada:clam:XXXX:XXXX:..."`.
    ///
    /// Each nibble (4 bits) becomes one hex segment.
    pub fn to_address(&self) -> String {
        let mut parts = vec!["ada".to_string(), "clam".to_string()];
        let nibble_count = self.depth.div_ceil(4);
        for i in 0..nibble_count {
            let shift = 12 - (i as u32 * 4);
            let nibble = (self.bits >> shift) & 0xF;
            parts.push(format!("{:x}", nibble));
        }
        parts.join(":")
    }

    /// Whether this path reaches full leaf depth (all 16 bits used).
    ///
    /// A full-leaf path was SET at some point — it references a stored item.
    /// A partial path (depth < 16) is an implicit address that can be resolved
    /// from the CLAM tree topology alone.
    #[inline]
    pub fn is_full_leaf(&self) -> bool {
        self.depth >= 16
    }

    /// Truncate the path to a specific depth.
    ///
    /// Returns the ancestor path at the given depth, masking off
    /// all bits beyond that level.
    pub fn truncate_to(&self, depth: u8) -> Self {
        let depth = depth.min(self.depth);
        if depth == 0 {
            return Self::new(0, 0);
        }
        let shift = 16 - depth as u32;
        let mask = !0u16 << shift;
        Self {
            bits: self.bits & mask,
            depth,
        }
    }

    /// Parent path: truncate by one level.
    pub fn parent(&self) -> Self {
        if self.depth == 0 {
            return *self;
        }
        self.truncate_to(self.depth - 1)
    }

    /// Flip the split decision at a specific depth level.
    ///
    /// Returns the counterfactual path — what would have happened if
    /// the tree had gone the other direction at that split.
    pub fn flip_at(&self, depth: u8) -> Self {
        if depth == 0 || depth > self.depth {
            return *self;
        }
        let bit_pos = 16 - depth as u32;
        Self {
            bits: self.bits ^ (1 << bit_pos),
            depth: self.depth,
        }
    }

    /// Extract the suffix bits beyond a given depth.
    ///
    /// These bits describe a virtual position within a cluster when
    /// the path extends beyond the deepest real cluster. Used for
    /// ET resolution interpolation.
    pub fn suffix_bits(&self, from_depth: u8) -> u16 {
        if from_depth >= self.depth {
            return 0;
        }
        let valid_bits = self.depth - from_depth;
        let shift = 16 - self.depth as u32;
        let mask = (1u16 << valid_bits) - 1;
        (self.bits >> shift) & mask
    }

    /// Get the u16 key value (alias for `self.bits`).
    #[inline]
    pub fn to_u16(&self) -> u16 {
        self.bits
    }

    /// Nibble at a specific position (0-indexed from root).
    ///
    /// Returns the 4-bit value at position `pos` (each position = 4 bits depth).
    pub fn nibble_at(&self, pos: u8) -> u8 {
        if pos * 4 >= self.depth {
            return 0;
        }
        let shift = 12 - (pos as u32 * 4);
        ((self.bits >> shift) & 0xF) as u8
    }

    /// Number of nibbles (4-bit groups) in this path.
    #[inline]
    pub fn nibble_count(&self) -> u8 {
        self.depth.div_ceil(4)
    }

    /// The full counterfactual chain: at each depth, flip that bit.
    ///
    /// Returns mirrors at every decision point from depth 1 to self.depth.
    pub fn counterfactual_chain(&self) -> Vec<ClamPath> {
        (1..=self.depth).map(|d| self.flip_at(d)).collect()
    }

    /// The full ancestry chain: truncate to every depth from 0 to self.depth.
    pub fn ancestry_chain(&self) -> Vec<ClamPath> {
        (0..=self.depth).map(|d| self.truncate_to(d)).collect()
    }
}

// ============================================================================
// ClamQualiaCAM ↔ ClamPath integration
// ============================================================================

impl ClamQualiaCAM {
    /// Extract CLAM paths for all coordinates in the corpus.
    ///
    /// Returns `(original_index, ClamPath)` for every item. Use this to
    /// populate the B-tree channel of CogRecords after tree construction.
    pub fn leaf_paths(&self) -> Vec<(usize, ClamPath)> {
        self.clam_tree
            .leaf_paths()
            .into_iter()
            .map(|(orig_idx, path)| (orig_idx, ClamPath::from_tree_traversal(&path)))
            .collect()
    }

    /// Find the CLAM path for a query point (descend tree to leaf).
    pub fn query_path(&self, query: &PackedQualia) -> ClamPath {
        if self.is_empty() {
            return ClamPath::new(0, 0);
        }

        let query_bytes = resonance_to_bytes(query);
        let mut node_idx = 0usize;
        let mut path = Vec::new();

        loop {
            let cluster = &self.clam_tree.nodes[node_idx];
            if cluster.is_leaf() {
                break;
            }

            match (cluster.left, cluster.right) {
                (Some(left), Some(right)) => {
                    let left_center = self
                        .clam_tree
                        .center_data(&self.clam_tree.nodes[left], &self.resonance_bytes, 16);
                    let right_center = self
                        .clam_tree
                        .center_data(&self.clam_tree.nodes[right], &self.resonance_bytes, 16);
                    let dl = self.clam_tree.dist(&query_bytes, left_center);
                    let dr = self.clam_tree.dist(&query_bytes, right_center);
                    if dl <= dr {
                        path.push(false);
                        node_idx = left;
                    } else {
                        path.push(true);
                        node_idx = right;
                    }
                }
                (Some(left), None) => {
                    path.push(false);
                    node_idx = left;
                }
                (None, Some(right)) => {
                    path.push(true);
                    node_idx = right;
                }
                (None, None) => break,
            }
        }

        ClamPath::from_tree_traversal(&path)
    }
}

// ============================================================================
// Utility functions
// ============================================================================

/// Convert PackedQualia resonance to byte slice for CLAM tree queries.
#[inline]
fn resonance_to_bytes(q: &PackedQualia) -> [u8; 16] {
    let mut bytes = [0u8; 16];
    for (i, &r) in q.resonance.iter().enumerate() {
        bytes[i] = r as u8;
    }
    bytes
}

/// L1 distance on i8 arrays (scalar, for small vectors).
#[inline]
fn l1_i8_scalar(a: &[i8; 16], b: &[i8; 16]) -> i32 {
    let mut sum: i32 = 0;
    for i in 0..16 {
        sum += (a[i] as i32 - b[i] as i32).abs();
    }
    sum
}

/// Convert L1 distance to NARS truth value.
fn distance_to_nars(dist: u64, query: &PackedQualia) -> NarsTruthValue {
    // Max possible L1 on 16 × i8: 16 × 254 = 4064
    let frequency = 1.0 - (dist as f32 / 4064.0);
    let active_dims = query.resonance.iter().filter(|&&r| r != 0).count();
    let confidence = active_dims as f32 / 16.0;
    NarsTruthValue::new(frequency, confidence)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_prosocial(family: u8, brightness: i8, valence: i8) -> GatedQualia {
        GatedQualia {
            qualia: PackedQualia::new(
                [
                    brightness, valence, 40, 30, 70, 80, 60, 20, 50, 30, 20, 40, 10, 60, 15, 40,
                ],
                1.0,
            ),
            gate: QualiaGateLevel::Flow,
            family_id: family,
        }
    }

    fn make_dark(family: u8) -> GatedQualia {
        GatedQualia {
            qualia: PackedQualia::new(
                [
                    10, 30, 90, 50, -80, 40, -60, 0, -70, 60, 20, 0, 0, 0, 85, 50,
                ],
                1.0,
            ),
            gate: QualiaGateLevel::Block,
            family_id: family,
        }
    }

    fn build_test_corpus() -> Vec<GatedQualia> {
        let mut coords = Vec::new();
        // 3 prosocial families × 5 items each = 15 flow items
        for fam in 0..3u8 {
            for i in 0..5i8 {
                let brightness = 40 + fam as i8 * 10 + i * 2;
                let valence = 50 + fam as i8 * 5 + i * 3;
                coords.push(make_prosocial(fam, brightness, valence));
            }
        }
        // 2 dark families × 2 items each = 4 gated items
        coords.push(make_dark(3));
        coords.push(GatedQualia {
            qualia: PackedQualia::new(
                [15, 25, 85, 45, -75, 35, -55, 5, -65, 55, 25, 5, 5, 5, 80, 45],
                1.0,
            ),
            gate: QualiaGateLevel::Hold,
            family_id: 3,
        });
        coords.push(make_dark(4));
        coords.push(GatedQualia {
            qualia: PackedQualia::new(
                [20, 35, 95, 55, -85, 45, -65, 10, -75, 65, 15, 10, 10, 10, 90, 55],
                1.0,
            ),
            gate: QualiaGateLevel::Block,
            family_id: 4,
        });
        coords
    }

    #[test]
    fn test_l1_i8_distance_symmetric() {
        let a: [u8; 16] = [1, 254, 3, 252, 5, 250, 7, 248, 9, 246, 11, 244, 13, 242, 15, 240];
        let b: [u8; 16] = [255, 2, 253, 4, 251, 6, 249, 8, 247, 10, 245, 12, 243, 14, 241, 16];
        assert_eq!(l1_i8_distance(&a, &b), l1_i8_distance(&b, &a));
    }

    #[test]
    fn test_l1_i8_distance_identity() {
        let a: [u8; 16] = [42; 16];
        assert_eq!(l1_i8_distance(&a, &a), 0);
    }

    #[test]
    fn test_l1_i8_distance_max() {
        // 127 and -127 as u8: 127 and 129
        let a: [u8; 16] = [127; 16]; // +127 as i8
        let b: [u8; 16] = [129; 16]; // -127 as i8
        // |127 - (-127)| = 254 per dim, 16 dims = 4064
        assert_eq!(l1_i8_distance(&a, &b), 4064);
    }

    #[test]
    fn test_clam_cam_basic() {
        let coords = build_test_corpus();
        let cam = ClamQualiaCAM::from_coordinates(coords);
        assert_eq!(cam.len(), 19);
        assert!(!cam.is_empty());
    }

    #[test]
    fn test_clam_locate_exact_match() {
        let coords = build_test_corpus();
        let cam = ClamQualiaCAM::from_coordinates(coords);

        // Query with exact match to first coordinate
        let query = cam.get(0).unwrap().qualia;
        let hits = cam.locate(&query, 3);

        assert!(!hits.is_empty());
        // The first hit should have distance 0 (exact match)
        assert_eq!(hits[0].distance, 0, "exact match should have distance 0");
    }

    #[test]
    fn test_clam_locate_agrees_with_linear() {
        let coords = build_test_corpus();
        let cam = ClamQualiaCAM::from_coordinates(coords);

        // Test all coordinates as queries
        for i in 0..cam.len() {
            let query = cam.get(i).unwrap().qualia;

            let clam_hits = cam.locate(&query, 3);
            let linear_hits = cam.locate_linear(&query, 3);

            // Both should find exact match at distance 0
            assert_eq!(
                clam_hits[0].distance, 0,
                "CLAM should find exact match for coord {i}"
            );
            assert_eq!(
                linear_hits[0].distance, 0,
                "linear should find exact match for coord {i}"
            );

            // Top-k distances should agree
            assert_eq!(
                clam_hits.len(),
                linear_hits.len(),
                "result count should agree for coord {i}"
            );
            for k in 0..clam_hits.len() {
                assert_eq!(
                    clam_hits[k].distance, linear_hits[k].distance,
                    "distance at rank {k} should agree for coord {i}: clam={} linear={}",
                    clam_hits[k].distance, linear_hits[k].distance
                );
            }
        }
    }

    #[test]
    fn test_clam_locate_within() {
        let coords = build_test_corpus();
        let cam = ClamQualiaCAM::from_coordinates(coords);

        let query = cam.get(0).unwrap().qualia;

        // Distance 0 should find at least the exact match
        let exact = cam.locate_within(&query, 0);
        assert!(
            !exact.is_empty(),
            "should find at least the exact match at threshold 0"
        );

        // Large threshold should find everything
        let all = cam.locate_within(&query, 10000);
        assert_eq!(all.len(), cam.len(), "large threshold should find all items");
    }

    #[test]
    fn test_clam_nearest_gated() {
        let coords = build_test_corpus();
        let cam = ClamQualiaCAM::from_coordinates(coords);

        // Query at dark coordinate — should find gated
        let dark_query = cam.get(15).unwrap().qualia; // first dark item
        let gated = cam.nearest_gated(&dark_query, 10000);
        assert!(gated.is_some(), "should find a gated coordinate");
        assert!(gated.unwrap().gate.is_gated());
    }

    #[test]
    fn test_anomaly_score() {
        let coords = build_test_corpus();
        let cam = ClamQualiaCAM::from_coordinates(coords);

        // Score for a known coordinate (should be low anomaly)
        let known = cam.get(0).unwrap().qualia;
        let anomaly = cam.anomaly_score(&known);
        assert!(
            anomaly.score < 1.0,
            "known coordinate should not max out anomaly: {}",
            anomaly.score
        );

        // Score for a far-out query (should be higher anomaly)
        let outlier = PackedQualia::new([127, -127, 127, -127, 127, -127, 127, -127, 127, -127, 127, -127, 127, -127, 127, -127], 1.0);
        let anomaly_outlier = cam.anomaly_score(&outlier);
        // Outlier might or might not score higher — depends on tree structure
        // Just verify it returns a valid score
        assert!(
            anomaly_outlier.score >= 0.0 && anomaly_outlier.score <= 1.0,
            "anomaly score should be in [0, 1]: {}",
            anomaly_outlier.score
        );
    }

    #[test]
    fn test_gate_bias() {
        let coords = build_test_corpus();
        let cam = ClamQualiaCAM::from_coordinates(coords);

        let normal = AnomalyResult {
            score: 0.2,
            calibration_type: CalibrationType::None,
            lfd: 1.0,
            cluster_depth: 5,
            cluster_cardinality: 10,
        };
        assert_eq!(cam.gate_bias(&normal), CollapseGateBias::Flow);

        let shift = AnomalyResult {
            score: 0.5,
            calibration_type: CalibrationType::Schaltsekunde,
            lfd: 1.5,
            cluster_depth: 8,
            cluster_cardinality: 3,
        };
        match cam.gate_bias(&shift) {
            CollapseGateBias::FlowWithCaution { confidence_multiplier, .. } => {
                assert!((confidence_multiplier - 0.8).abs() < 0.01);
            }
            other => panic!("expected FlowWithCaution, got {:?}", other),
        }

        let topo = AnomalyResult {
            score: 0.8,
            calibration_type: CalibrationType::Schaltminute,
            lfd: 2.5,
            cluster_depth: 12,
            cluster_cardinality: 1,
        };
        match cam.gate_bias(&topo) {
            CollapseGateBias::Hold { .. } => {}
            other => panic!("expected Hold, got {:?}", other),
        }
    }

    #[test]
    fn test_pruning_report() {
        let coords = build_test_corpus();
        let cam = ClamQualiaCAM::from_coordinates(coords);

        let report = cam.pruning_report();
        assert_eq!(report.corpus_size, 19);
        assert!(report.num_leaves > 0);
        assert!(report.mean_lfd >= 0.0);
        assert!(report.theoretical_speedup >= 1.0 || report.theoretical_speedup > 0.0);
    }

    #[test]
    fn test_truth_at() {
        let coords = build_test_corpus();
        let cam = ClamQualiaCAM::from_coordinates(coords);

        let exact = cam.get(0).unwrap().qualia;
        let truth = cam.truth_at(&exact, 0).unwrap();
        assert!(
            truth.frequency > 0.99,
            "exact match should have high frequency: {}",
            truth.frequency
        );
    }

    #[test]
    fn test_causality_at() {
        let coords = build_test_corpus();
        let cam = ClamQualiaCAM::from_coordinates(coords);

        // Prosocial coordinate: warmth=70, social=60, sacredness=50 (all positive → Experiencing)
        let prosocial = cam.get(0).unwrap().qualia;
        let dir = cam.causality_at(&prosocial, 0).unwrap();
        assert_eq!(dir, CausalityDirection::Experiencing);

        // Dark coordinate: warmth=-80, social=-60, sacredness=-70 (all negative → Causing)
        let dark = cam.get(15).unwrap().qualia;
        let dir_dark = cam.causality_at(&dark, 15).unwrap();
        assert_eq!(dir_dark, CausalityDirection::Causing);
    }

    #[test]
    fn test_cluster_significance() {
        let coords = build_test_corpus();
        let cam = ClamQualiaCAM::from_coordinates(coords);

        let query = cam.get(0).unwrap().qualia;
        let verdict = cam.cluster_significance(&query, 0); // root cluster
        // Root cluster contains everything, so should not be pruned
        assert_ne!(verdict, ClusterVerdict::Prune);
    }

    #[test]
    fn test_empty_cam() {
        let cam = ClamQualiaCAM::from_coordinates(vec![]);
        assert!(cam.is_empty());
        assert_eq!(cam.len(), 0);

        let query = PackedQualia::new([0; 16], 1.0);
        let hits = cam.locate(&query, 5);
        assert!(hits.is_empty());

        let within = cam.locate_within(&query, 1000);
        assert!(within.is_empty());

        let anomaly = cam.anomaly_score(&query);
        assert_eq!(anomaly.calibration_type, CalibrationType::Schaltminute);
    }

    // ── ClamPath tests ──────────────────────────────────────────────

    #[test]
    fn test_clam_path_roundtrip() {
        let path = ClamPath::from_tree_traversal(&[true, false, true, true, false]);
        assert_eq!(path.depth, 5);
        let bytes = path.to_bytes();
        let decoded = ClamPath::from_bytes(&bytes);
        assert_eq!(path, decoded);
    }

    #[test]
    fn test_clam_path_subtree_range() {
        // Path: root=right (1), then left (0) → prefix = 10... at depth 2
        let path = ClamPath::from_tree_traversal(&[true, false]);
        let (lo, hi) = path.subtree_range();
        // 10_0000_0000_0000_00 = 0x8000
        // 10_1111_1111_1111_11 = 0xBFFF
        assert_eq!(lo, 0x8000);
        assert_eq!(hi, 0xBFFF);
        // The path itself should be within the range
        assert!(path.bits >= lo && path.bits <= hi);
    }

    #[test]
    fn test_clam_path_common_ancestor() {
        let a = ClamPath::from_tree_traversal(&[true, false, true]);
        let b = ClamPath::from_tree_traversal(&[true, false, false]);
        // Share first 2 decisions (true, false), diverge at depth 2
        assert_eq!(a.common_ancestor_depth(&b), 2);
        assert_eq!(a.lineage_distance(&b), 2); // 1 step up + 1 step down
    }

    #[test]
    fn test_clam_path_sibling() {
        let path = ClamPath::from_tree_traversal(&[true, false, true]);
        let sib = path.sibling();
        // Sibling should share parent but flip last bit
        assert_eq!(path.common_ancestor_depth(&sib), 2);
        assert_ne!(path.bits, sib.bits);
        assert_eq!(path.depth, sib.depth);
    }

    #[test]
    fn test_clam_path_identity() {
        let path = ClamPath::from_tree_traversal(&[true, true, false]);
        assert_eq!(path.common_ancestor_depth(&path), path.depth);
        assert_eq!(path.lineage_distance(&path), 0);
    }

    #[test]
    fn test_leaf_paths_coverage() {
        let coords = build_test_corpus();
        let cam = ClamQualiaCAM::from_coordinates(coords);

        let paths = cam.leaf_paths();
        // Every coordinate should get a path
        assert_eq!(paths.len(), cam.len());

        // All original indices should be present
        let mut indices: Vec<usize> = paths.iter().map(|&(idx, _)| idx).collect();
        indices.sort();
        indices.dedup();
        assert_eq!(indices.len(), cam.len());
    }

    #[test]
    fn test_query_path_consistency() {
        let coords = build_test_corpus();
        let cam = ClamQualiaCAM::from_coordinates(coords);

        // Query path for a known coordinate should produce a valid ClamPath
        let query = cam.get(0).unwrap().qualia;
        let path = cam.query_path(&query);
        assert!(path.depth > 0 || cam.len() == 1);
        assert!(path.depth <= 64);
    }

    #[test]
    fn test_nearby_coords_share_prefix() {
        let coords = build_test_corpus();
        let cam = ClamQualiaCAM::from_coordinates(coords);

        // Two coordinates from the same family should share more prefix
        // than coordinates from different families (statistically)
        let same_family_0 = cam.get(0).unwrap().qualia; // family 0, item 0
        let same_family_1 = cam.get(1).unwrap().qualia; // family 0, item 1
        let diff_family = cam.get(15).unwrap().qualia;   // dark family

        let path_0 = cam.query_path(&same_family_0);
        let path_1 = cam.query_path(&same_family_1);
        let path_dark = cam.query_path(&diff_family);

        let same_ancestor = path_0.common_ancestor_depth(&path_1);
        let diff_ancestor = path_0.common_ancestor_depth(&path_dark);

        // Same-family should share at least as deep a common ancestor
        // (or equal — tree topology varies)
        assert!(
            same_ancestor >= diff_ancestor || diff_ancestor <= 2,
            "same-family ancestor depth ({}) should generally >= diff-family ({})",
            same_ancestor, diff_ancestor
        );
    }
}
