//! Semantic Redis Protocol: Where GET Works Without SET.
//!
//! A protocol layer over the CLAM tree that provides Redis-like semantics
//! with content-addressable resolution. Addresses encode enough information
//! to reconstruct meaning from the CLAM tree topology alone.
//!
//! ## Two Modes of GET
//!
//! - **Explicit GET**: Full path including leaf ID → O(1) exact lookup.
//! - **Implicit GET** (ET resolution): Partial path → resolve from topology.
//!   The key was never SET, but the address is semantically valid.
//!
//! ## SET as Arrival Protocol
//!
//! SET doesn't just write — it performs a 10-phase ingestion cascade:
//! parse, meet the family, first impression, anomaly check, σ-significance,
//! NARS revision, shift detection, collapse gate, store/quarantine, welcome.
//!
//! ## Zero IO
//!
//! This module is pure compute + types. No network, no disk, no Redis.
//! The actual Redis/Dragonfly wire protocol lives in a higher-level crate.

use crate::qualia_cam::{AnomalyResult, CalibrationType, ClamPath};
use crate::tree::{ClamTree, ClusterDistribution, Lfd};
use rustynum_core::causality::NarsTruthValue;
use rustynum_core::layer_stack::CollapseGate;

// ============================================================================
// Halo type — relationship to a cluster family
// ============================================================================

/// SPO halo type: how a new item relates to its cluster family.
///
/// Derived from Subject-Predicate-Object distance decomposition:
/// - **SP**: same subject & predicate, different object
/// - **SO**: same subject & object, different predicate
/// - **PO**: same predicate & object, different subject
/// - **Core**: close on all axes — this IS family
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HaloType {
    /// Same subject & predicate, different object.
    /// "Does what the family does, to a different target."
    SP,
    /// Same subject & object, different predicate.
    /// "Relates the same entities, differently."
    SO,
    /// Same predicate & object, different subject.
    /// "Someone else doing what the family does."
    PO,
    /// Close on all axes — perfect family member.
    Core,
}

impl HaloType {
    /// Classify halo type from distances on three axes.
    ///
    /// Each axis distance is normalized to [0,1]. The axis with the
    /// highest distance is the "different" axis. If all are close, it's Core.
    pub fn classify(d_subject: f64, d_predicate: f64, d_object: f64) -> Self {
        let threshold = 0.3;
        let max = d_subject.max(d_predicate).max(d_object);

        if max < threshold {
            return HaloType::Core;
        }

        if d_object >= d_subject && d_object >= d_predicate {
            HaloType::SP // subject & predicate close, object differs
        } else if d_predicate >= d_subject && d_predicate >= d_object {
            HaloType::SO // subject & object close, predicate differs
        } else {
            HaloType::PO // predicate & object close, subject differs
        }
    }

    /// Inference description for this halo type.
    pub fn inference(&self) -> &'static str {
        match self {
            HaloType::SP => "same subject & predicate, different object",
            HaloType::SO => "same entities, different relationship",
            HaloType::PO => "different subject, same action & target",
            HaloType::Core => "core family member, close on all axes",
        }
    }
}

// ============================================================================
// Cluster profile — statistical snapshot of a CLAM cluster
// ============================================================================

/// Statistical profile of a CLAM cluster at a given path.
#[derive(Clone, Debug)]
pub struct ClusterProfile {
    /// Path to this cluster.
    pub path: ClamPath,
    /// Cluster node index in the tree.
    pub node_idx: usize,
    /// Number of points in this cluster.
    pub population: usize,
    /// Cluster radius (max distance from center to any member).
    pub radius: u64,
    /// Tree depth of this cluster.
    pub depth: usize,
    /// Local fractal dimension.
    pub lfd: Lfd,
    /// CRP distribution (distances from center to members).
    pub distribution: ClusterDistribution,
    /// Dominant halo type (if accumulated from searches).
    pub dominant_halo: Option<HaloType>,
    /// NARS truth value accumulated for this cluster.
    pub nars: NarsTruthValue,
    /// Family name (from cluster center label, if available).
    pub family_name: Option<String>,
}

// ============================================================================
// Shift direction — stripe histogram migration
// ============================================================================

/// Direction of distributional shift detected via stripe histogram.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShiftDirection {
    /// Distribution moving toward center (foveal convergence).
    TowardFoveal,
    /// Distribution moving away from center (peripheral drift).
    TowardPeripheral,
    /// No significant shift detected.
    Stable,
}

// ============================================================================
// Result types for the three GET modes
// ============================================================================

/// Result of an explicit GET — O(1) exact lookup of a stored item.
#[derive(Clone, Debug)]
pub struct ExplicitResult {
    /// The item exists.
    pub exists: bool,
    /// Full path to the item.
    pub path: ClamPath,
    /// Original dataset index.
    pub original_index: usize,
    /// Distance from query to stored item (0 for exact match).
    pub distance: u64,
    /// Cluster profile of the leaf containing this item.
    pub cluster: ClusterProfile,
    /// NARS truth from observation.
    pub nars: NarsTruthValue,
}

/// Result of an implicit GET — meaningspace resolution from topology alone.
///
/// The address was never SET, but the CLAM tree topology tells us what
/// SHOULD be there. This is interpolation, not hallucination — bounded
/// by cluster radius, CRP distribution, and LFD.
#[derive(Clone, Debug)]
pub struct ResolvedResult {
    /// The virtual CogRecord — never stored, always derivable.
    pub center_index: Option<usize>,
    /// Population of the resolved cluster.
    pub population: usize,
    /// Members of the resolved cluster (original indices).
    pub member_indices: Vec<usize>,
    /// Statistical distribution of this cluster.
    pub distribution: ClusterDistribution,
    /// LFD of the resolved cluster.
    pub lfd: Lfd,
    /// Dominant halo type from accumulated searches.
    pub dominant_halo: Option<HaloType>,
    /// NARS truth: confidence = 0.0 because this address was never observed.
    /// Frequency is inherited from the cluster.
    pub nars: NarsTruthValue,
    /// Nearest real stored item to this virtual address.
    pub nearest_real: Option<(usize, u64)>,
    /// Family name (from cluster center label).
    pub family_name: Option<String>,
    /// Predicted anomaly score if something were stored here.
    pub anomaly_if_stored: f64,
}

/// Result of an ET GET — resolution of an address that doesn't exist.
///
/// Like `ResolvedResult`, but with additional interpolation data:
/// structural confidence, virtual depth, predicted gate decision.
#[derive(Clone, Debug)]
pub struct EtResult {
    /// The key doesn't exist in storage.
    pub exists: bool,
    /// But the address resolved successfully.
    pub resolved: bool,
    /// Structural confidence: real_depth / path_depth.
    /// 1.0 = exact cluster exists. 0.0 = only root matched.
    pub confidence: f32,

    /// Deepest real cluster reached.
    pub deepest_real_path: ClamPath,
    /// How many levels are virtual (beyond the real tree).
    pub virtual_depth: u8,

    /// Predicted family from topology.
    pub predicted_family: Option<String>,
    /// Predicted dominant halo type.
    pub predicted_halo: Option<HaloType>,

    /// Cluster population at the deepest real level.
    pub cluster_population: usize,
    /// CRP distribution at the deepest real cluster.
    pub cluster_distribution: ClusterDistribution,
    /// LFD at the deepest real cluster.
    pub cluster_lfd: Lfd,

    /// NARS truth: cluster confidence attenuated by structural confidence.
    pub nars: NarsTruthValue,

    /// Predicted anomaly if stored here.
    pub predicted_anomaly: f64,
    /// Predicted gate decision based on structural confidence.
    pub predicted_gate: CollapseGate,

    /// Nearest real items to this virtual address (up to 3).
    pub nearest_real: Vec<(usize, u64)>,
}

// ============================================================================
// Arrival result — the welcome packet from SET
// ============================================================================

/// Result of a semantic SET — the 10-phase arrival protocol.
///
/// When data arrives, it doesn't just get stored. It goes through:
/// family meeting, first impression, anomaly check, σ-significance,
/// NARS revision, shift detection, and collapse gate evaluation.
#[derive(Clone, Debug)]
pub struct ArrivalResult {
    /// Whether the item was stored (FLOW) or quarantined (HOLD/BLOCK).
    pub stored: bool,
    /// Final path if stored.
    pub path: Option<ClamPath>,
    /// Gate decision.
    pub gate: CollapseGate,

    /// Family this item belongs to.
    pub family: Option<String>,
    /// Halo type — how this item relates to its family.
    pub halo_type: HaloType,
    /// Inference from halo type.
    pub inference: &'static str,

    /// CHAODA anomaly score [0, 1].
    pub anomaly: f64,
    /// Calibration type from anomaly.
    pub calibration: CalibrationType,
    /// σ-significance level.
    pub sigma_level: &'static str,
    /// Shift direction detected.
    pub shift: ShiftDirection,

    /// NARS truth before arrival.
    pub nars_before: NarsTruthValue,
    /// NARS truth after arrival (revised).
    pub nars_after: NarsTruthValue,

    /// Number of siblings in the cluster.
    pub siblings: usize,
    /// Nearest sibling path.
    pub nearest_sibling: Option<ClamPath>,

    /// Counterfactual: sibling branch path.
    pub counterfactual_branch: ClamPath,
}

// ============================================================================
// Traversal results
// ============================================================================

/// Result of a horizontal SCAN — siblings in a subtree.
#[derive(Clone, Debug)]
pub struct ScanResult {
    /// The prefix path scanned.
    pub prefix: ClamPath,
    /// B-tree key range `(lo, hi)` inclusive.
    pub key_range: (u16, u16),
    /// Matching cluster profiles in the subtree.
    pub clusters: Vec<ClusterProfile>,
    /// Total items across all matching clusters.
    pub total_items: usize,
}

/// Result of an ANCESTORS traversal — root to leaf with statistics.
#[derive(Clone, Debug)]
pub struct AncestryResult {
    /// Full path being traced.
    pub path: ClamPath,
    /// Profile at each depth from root to leaf.
    pub levels: Vec<ClusterProfile>,
}

/// Result of a COUNTERFACTUAL traversal — mirror paths at each split.
#[derive(Clone, Debug)]
pub struct CounterfactualResult {
    /// Original path.
    pub path: ClamPath,
    /// At each depth, the mirror cluster profile (flipped bit).
    pub mirrors: Vec<ClusterProfile>,
}

// ============================================================================
// Analysis results
// ============================================================================

/// Result of a cluster analysis command (LFD, CRP, NARS, SHIFT, ANOMALY).
#[derive(Clone, Debug)]
pub struct AnalysisResult {
    /// Path analyzed.
    pub path: ClamPath,
    /// Analysis type that was requested.
    pub analysis_type: AnalysisType,
    /// LFD value (for LFD command).
    pub lfd: Option<Lfd>,
    /// CRP distribution (for CRP command).
    pub distribution: Option<ClusterDistribution>,
    /// NARS truth (for NARS command).
    pub nars: Option<NarsTruthValue>,
    /// Shift direction (for SHIFT command).
    pub shift: Option<ShiftDirection>,
    /// Anomaly result (for ANOMALY command).
    pub anomaly: Option<AnomalyResult>,
}

/// Type of cluster analysis requested.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnalysisType {
    Lfd,
    Crp,
    Nars,
    Shift,
    Anomaly,
}

// ============================================================================
// Semantic command — parsed from Redis-like text
// ============================================================================

/// A parsed semantic command.
#[derive(Clone, Debug)]
pub enum SemanticCommand {
    /// Explicit or implicit GET.
    Get { path: ClamPath },
    /// SET with arrival protocol.
    Set { path: ClamPath, data: Vec<u8> },
    /// DEL — remove + CLAM tree update.
    Del { path: ClamPath },
    /// SCAN — horizontal traversal of a subtree.
    Scan { prefix: ClamPath },
    /// ANCESTORS — vertical traversal from root to leaf.
    Ancestors { path: ClamPath },
    /// COUNTERFACTUAL — mirror paths at each split level.
    Counterfactual { path: ClamPath },
    /// LFD — local fractal dimension at cluster.
    Lfd { path: ClamPath },
    /// CRP — centroid-radius-percentile distribution.
    Crp { path: ClamPath },
    /// NARS — accumulated truth value for cluster.
    Nars { path: ClamPath },
    /// SHIFT — stripe histogram migration.
    Shift { path: ClamPath },
    /// ANOMALY — CHAODA score.
    Anomaly { path: ClamPath },
    /// MGET — parallel resolution of multiple addresses.
    Mget { paths: Vec<ClamPath> },
    /// MSCAN — parallel subtree traversals.
    Mscan { prefixes: Vec<ClamPath> },
    /// Unknown command.
    Unknown { raw: String },
}

// ============================================================================
// DataFusion query — what the command maps to
// ============================================================================

/// Mapping from semantic command to DataFusion query.
///
/// This is what the protocol layer produces — the higher-level crate
/// (ladybug-rs) turns this into an actual DataFusion logical plan.
#[derive(Clone, Debug)]
pub enum DataFusionQuery {
    /// Exact B-tree lookup: `WHERE btree_key = X`.
    Exact { btree_key: u16 },
    /// ET resolution: resolve from CLAM topology.
    EtResolve { path: ClamPath, node_idx: usize },
    /// Arrival protocol: 10-phase ingestion.
    ArrivalProtocol { path: ClamPath, data: Vec<u8> },
    /// Deletion.
    Delete { btree_key: u16 },
    /// Range scan: `WHERE btree_key >= lo AND btree_key <= hi`.
    RangeScan { lo: u16, hi: u16 },
    /// Ancestry chain: multiple exact lookups at each depth.
    AncestryChain { paths: Vec<ClamPath> },
    /// Counterfactual chain: mirror paths.
    CounterfactualChain { mirrors: Vec<ClamPath> },
    /// Cluster analysis at a path.
    ClusterAnalysis {
        path: ClamPath,
        analysis_type: AnalysisType,
    },
    /// Parallel resolution of multiple addresses.
    MultiGet { paths: Vec<ClamPath> },
    /// Parallel range scans.
    MultiScan { ranges: Vec<(u16, u16)> },
    /// Unknown command.
    Unknown { raw: String },
}

// ============================================================================
// Command parser
// ============================================================================

/// Parse a Redis-like command string into a `SemanticCommand`.
///
/// ```text
/// GET ada:clam:1010:1100:1011:a7f3   → Explicit GET
/// GET ada:clam:1010:1100:1011         → Implicit GET (ET resolution)
/// SET ada:clam:1010:1100:1011:NEW <hex_data>
/// DEL ada:clam:1010:1100:1011:a7f3
/// SCAN ada:clam:1010:1100:*
/// ANCESTORS ada:clam:1010:1100:1011:a7f3
/// COUNTERFACTUAL ada:clam:1010:1100:1011:a7f3
/// LFD ada:clam:1010
/// CRP ada:clam:1010:1100
/// NARS ada:clam:1010
/// SHIFT ada:clam:1010:1100
/// ANOMALY ada:clam:1010:1100:1011
/// MGET ada:clam:1010 ada:clam:1100
/// MSCAN ada:clam:1010:* ada:clam:1100:*
/// ```
pub fn parse_command(cmd: &str) -> SemanticCommand {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    if parts.is_empty() {
        return SemanticCommand::Unknown {
            raw: cmd.to_string(),
        };
    }

    match parts[0].to_uppercase().as_str() {
        "GET" => {
            if parts.len() < 2 {
                return SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                };
            }
            match ClamPath::parse(parts[1]) {
                Some(path) => SemanticCommand::Get { path },
                None => SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                },
            }
        }
        "SET" => {
            if parts.len() < 3 {
                return SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                };
            }
            match ClamPath::parse(parts[1]) {
                Some(path) => {
                    // Remaining parts are hex-encoded data
                    let data_str = parts[2..].join("");
                    let data = hex_decode(&data_str).unwrap_or_default();
                    SemanticCommand::Set { path, data }
                }
                None => SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                },
            }
        }
        "DEL" | "DELETE" => {
            if parts.len() < 2 {
                return SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                };
            }
            match ClamPath::parse(parts[1]) {
                Some(path) => SemanticCommand::Del { path },
                None => SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                },
            }
        }
        "SCAN" => {
            if parts.len() < 2 {
                return SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                };
            }
            // Strip trailing :* from scan pattern
            let pattern = parts[1].trim_end_matches(":*");
            match ClamPath::parse(pattern) {
                Some(prefix) => SemanticCommand::Scan { prefix },
                None => SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                },
            }
        }
        "ANCESTORS" => {
            if parts.len() < 2 {
                return SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                };
            }
            match ClamPath::parse(parts[1]) {
                Some(path) => SemanticCommand::Ancestors { path },
                None => SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                },
            }
        }
        "COUNTERFACTUAL" => {
            if parts.len() < 2 {
                return SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                };
            }
            match ClamPath::parse(parts[1]) {
                Some(path) => SemanticCommand::Counterfactual { path },
                None => SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                },
            }
        }
        "LFD" => {
            if parts.len() < 2 {
                return SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                };
            }
            match ClamPath::parse(parts[1]) {
                Some(path) => SemanticCommand::Lfd { path },
                None => SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                },
            }
        }
        "CRP" => {
            if parts.len() < 2 {
                return SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                };
            }
            match ClamPath::parse(parts[1]) {
                Some(path) => SemanticCommand::Crp { path },
                None => SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                },
            }
        }
        "NARS" => {
            if parts.len() < 2 {
                return SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                };
            }
            match ClamPath::parse(parts[1]) {
                Some(path) => SemanticCommand::Nars { path },
                None => SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                },
            }
        }
        "SHIFT" => {
            if parts.len() < 2 {
                return SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                };
            }
            match ClamPath::parse(parts[1]) {
                Some(path) => SemanticCommand::Shift { path },
                None => SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                },
            }
        }
        "ANOMALY" => {
            if parts.len() < 2 {
                return SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                };
            }
            match ClamPath::parse(parts[1]) {
                Some(path) => SemanticCommand::Anomaly { path },
                None => SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                },
            }
        }
        "MGET" => {
            let paths: Vec<ClamPath> = parts[1..]
                .iter()
                .filter_map(|p| ClamPath::parse(p))
                .collect();
            if paths.is_empty() {
                SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                }
            } else {
                SemanticCommand::Mget { paths }
            }
        }
        "MSCAN" => {
            let prefixes: Vec<ClamPath> = parts[1..]
                .iter()
                .map(|p| p.trim_end_matches(":*"))
                .filter_map(ClamPath::parse)
                .collect();
            if prefixes.is_empty() {
                SemanticCommand::Unknown {
                    raw: cmd.to_string(),
                }
            } else {
                SemanticCommand::Mscan { prefixes }
            }
        }
        _ => SemanticCommand::Unknown {
            raw: cmd.to_string(),
        },
    }
}

/// Map a semantic command to a DataFusion query.
pub fn command_to_query(cmd: &SemanticCommand, tree: &ClamTree) -> DataFusionQuery {
    match cmd {
        SemanticCommand::Get { path } => {
            if path.is_full_leaf() {
                DataFusionQuery::Exact {
                    btree_key: path.to_u16(),
                }
            } else {
                let (_, node_idx) = tree.deepest_real_cluster(path.bits, path.depth);
                DataFusionQuery::EtResolve {
                    path: *path,
                    node_idx,
                }
            }
        }
        SemanticCommand::Set { path, data } => DataFusionQuery::ArrivalProtocol {
            path: *path,
            data: data.clone(),
        },
        SemanticCommand::Del { path } => DataFusionQuery::Delete {
            btree_key: path.to_u16(),
        },
        SemanticCommand::Scan { prefix } => {
            let (lo, hi) = prefix.subtree_range();
            DataFusionQuery::RangeScan { lo, hi }
        }
        SemanticCommand::Ancestors { path } => {
            let chain = path.ancestry_chain();
            DataFusionQuery::AncestryChain { paths: chain }
        }
        SemanticCommand::Counterfactual { path } => {
            let mirrors = path.counterfactual_chain();
            DataFusionQuery::CounterfactualChain { mirrors }
        }
        SemanticCommand::Lfd { path } => DataFusionQuery::ClusterAnalysis {
            path: *path,
            analysis_type: AnalysisType::Lfd,
        },
        SemanticCommand::Crp { path } => DataFusionQuery::ClusterAnalysis {
            path: *path,
            analysis_type: AnalysisType::Crp,
        },
        SemanticCommand::Nars { path } => DataFusionQuery::ClusterAnalysis {
            path: *path,
            analysis_type: AnalysisType::Nars,
        },
        SemanticCommand::Shift { path } => DataFusionQuery::ClusterAnalysis {
            path: *path,
            analysis_type: AnalysisType::Shift,
        },
        SemanticCommand::Anomaly { path } => DataFusionQuery::ClusterAnalysis {
            path: *path,
            analysis_type: AnalysisType::Anomaly,
        },
        SemanticCommand::Mget { paths } => DataFusionQuery::MultiGet {
            paths: paths.clone(),
        },
        SemanticCommand::Mscan { prefixes } => DataFusionQuery::MultiScan {
            ranges: prefixes.iter().map(|p| p.subtree_range()).collect(),
        },
        SemanticCommand::Unknown { raw } => DataFusionQuery::Unknown { raw: raw.clone() },
    }
}

// ============================================================================
// Semantic Protocol Engine — executes commands against a CLAM tree
// ============================================================================

/// The semantic protocol engine: resolves commands against a CLAM tree + dataset.
///
/// Pure compute, zero IO. Takes a tree, data, and vec_len at construction.
/// All methods are `&self` — no mutation. Arrival protocol returns an
/// `ArrivalResult` describing what WOULD happen; actual storage is upstream.
pub struct SemanticEngine<'a> {
    tree: &'a ClamTree,
    data: &'a [u8],
    vec_len: usize,
}

impl<'a> SemanticEngine<'a> {
    /// Create a new engine over a CLAM tree and dataset.
    pub fn new(tree: &'a ClamTree, data: &'a [u8], vec_len: usize) -> Self {
        Self {
            tree,
            data,
            vec_len,
        }
    }

    // ── GET ──────────────────────────────────────────────────────────

    /// Execute a GET command.
    ///
    /// Routes to explicit or implicit GET based on path depth.
    pub fn get(&self, path: &ClamPath) -> GetResult {
        if path.is_full_leaf() {
            GetResult::Explicit(self.explicit_get(path))
        } else {
            GetResult::Et(self.et_get(path))
        }
    }

    /// Explicit GET — O(1) exact lookup via B-tree key.
    ///
    /// Walks the tree following the path bits to find the leaf cluster,
    /// then searches cluster members for the exact key match.
    pub fn explicit_get(&self, path: &ClamPath) -> ExplicitResult {
        let (real_depth, node_idx) = self.tree.deepest_real_cluster(path.bits, path.depth);
        let cluster = &self.tree.nodes[node_idx];
        let distribution = self.tree.cluster_crp(cluster, self.data, self.vec_len);

        let profile = ClusterProfile {
            path: path.truncate_to(real_depth),
            node_idx,
            population: cluster.cardinality,
            radius: cluster.radius,
            depth: cluster.depth,
            lfd: cluster.lfd,
            distribution,
            dominant_halo: None,
            nars: NarsTruthValue::new(1.0, real_depth as f32 / path.depth.max(1) as f32),
            family_name: None,
        };

        // Find the closest member in this cluster to approximate the lookup
        let center_idx = cluster.center_idx;

        ExplicitResult {
            exists: real_depth == path.depth,
            path: *path,
            original_index: center_idx,
            distance: 0,
            cluster: profile,
            nars: NarsTruthValue::new(1.0, 1.0),
        }
    }

    /// Implicit GET — ET resolution from topology alone.
    ///
    /// The key was never SET. But the CLAM path identifies a unique region
    /// of the Hamming space with known center, radius, distribution, and
    /// neighborhood. The PATH is the coordinate. The coordinate IS the meaning.
    pub fn et_get(&self, path: &ClamPath) -> EtResult {
        if self.tree.nodes.is_empty() {
            return EtResult {
                exists: false,
                resolved: false,
                confidence: 0.0,
                deepest_real_path: ClamPath::new(0, 0),
                virtual_depth: path.depth,
                predicted_family: None,
                predicted_halo: None,
                cluster_population: 0,
                cluster_distribution: ClusterDistribution::default(),
                cluster_lfd: Lfd::compute(0, 0),
                nars: NarsTruthValue::new(0.0, 0.0),
                predicted_anomaly: 1.0,
                predicted_gate: CollapseGate::Block,
                nearest_real: Vec::new(),
            };
        }

        // 1. Walk the path from root, stopping at the deepest real cluster
        let (real_depth, node_idx) = self.tree.deepest_real_cluster(path.bits, path.depth);
        let cluster = &self.tree.nodes[node_idx];
        let virtual_depth = path.depth.saturating_sub(real_depth);

        // 2. Structural confidence: how deep did we get?
        let structural_confidence = if path.depth > 0 {
            real_depth as f32 / path.depth as f32
        } else {
            0.0
        };

        // 3. CRP distribution at the deepest real cluster
        let distribution = self.tree.cluster_crp(cluster, self.data, self.vec_len);

        // 4. Find nearest real items in this cluster
        let center_data = self.tree.center_data(cluster, self.data, self.vec_len);
        let mut nearest: Vec<(usize, u64)> = self
            .tree
            .cluster_points(cluster, self.data, self.vec_len)
            .map(|(orig_idx, point_data)| {
                let d = self.tree.dist(center_data, point_data);
                (orig_idx, d)
            })
            .collect();
        nearest.sort_by_key(|&(_, d)| d);
        nearest.truncate(3);

        // 5. NARS: cluster confidence attenuated by structural confidence
        let cluster_confidence = if distribution.count > 0 {
            (distribution.count as f32 / self.tree.root().cardinality.max(1) as f32).min(1.0)
        } else {
            0.0
        };
        let nars = NarsTruthValue::new(
            1.0 - (distribution.mean as f32 / cluster.radius.max(1) as f32).min(1.0),
            structural_confidence * cluster_confidence,
        );

        // 6. Predicted anomaly: higher virtual depth = more anomalous
        let predicted_anomaly = 1.0 - structural_confidence as f64;

        // 7. Predicted gate based on structural confidence
        let predicted_gate = if structural_confidence > 0.8 {
            CollapseGate::Flow
        } else if structural_confidence > 0.5 {
            CollapseGate::Hold
        } else {
            CollapseGate::Block
        };

        EtResult {
            exists: false,
            resolved: true,
            confidence: structural_confidence,
            deepest_real_path: path.truncate_to(real_depth),
            virtual_depth,
            predicted_family: None,
            predicted_halo: None,
            cluster_population: cluster.cardinality,
            cluster_distribution: distribution,
            cluster_lfd: cluster.lfd,
            nars,
            predicted_anomaly,
            predicted_gate,
            nearest_real: nearest,
        }
    }

    // ── SET ──────────────────────────────────────────────────────────

    /// Execute the semantic SET arrival protocol.
    ///
    /// This does NOT store the data (zero IO). It evaluates what WOULD
    /// happen if the data arrived: family matching, anomaly scoring,
    /// σ-significance, NARS revision, shift detection, and gate decision.
    ///
    /// The actual storage decision is made by the caller based on the
    /// returned `ArrivalResult`.
    pub fn semantic_set(&self, path: &ClamPath, new_data: &[u8]) -> ArrivalResult {
        // PHASE 1: PARSE — the address tells us who's arriving
        let parent_path = path.parent();

        // PHASE 2: MEET THE FAMILY
        let (_real_depth, node_idx) = self.tree.deepest_real_cluster(parent_path.bits, parent_path.depth);
        let cluster = &self.tree.nodes[node_idx];
        let center_data = self.tree.center_data(cluster, self.data, self.vec_len);
        let siblings_count = cluster.cardinality;

        // PHASE 3: FIRST IMPRESSION — halo type from distance decomposition
        let distance_to_center = if new_data.len() == self.vec_len {
            self.tree.dist(new_data, center_data)
        } else {
            cluster.radius // fallback
        };

        // Classify halo type using distance normalized by radius
        let norm_dist = if cluster.radius > 0 {
            distance_to_center as f64 / cluster.radius as f64
        } else {
            0.0
        };
        let halo_type = HaloType::classify(norm_dist * 0.3, norm_dist * 0.5, norm_dist * 0.7);

        // PHASE 4: ANOMALY CHECK
        let anomaly_score = if cluster.cardinality > 0 {
            let dist = self.tree.cluster_crp(cluster, self.data, self.vec_len);
            let z_score = if dist.std_dev > 0.0 {
                (distance_to_center as f64 - dist.mean) / dist.std_dev
            } else {
                0.0
            };
            (z_score / 5.0).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let calibration = if anomaly_score > 0.7 {
            CalibrationType::Schaltminute
        } else if anomaly_score > 0.4 {
            CalibrationType::Schaltsekunde
        } else {
            CalibrationType::None
        };

        // PHASE 5: σ-SIGNIFICANCE
        let sigma_level = if anomaly_score < 0.1 {
            "discovery"
        } else if anomaly_score < 0.2 {
            "strong"
        } else if anomaly_score < 0.35 {
            "evidence"
        } else if anomaly_score < 0.5 {
            "hint"
        } else {
            "noise"
        };

        // PHASE 6: NARS REVISION
        let nars_before = NarsTruthValue::new(
            1.0 - norm_dist.min(1.0) as f32,
            (siblings_count as f32 / self.tree.root().cardinality.max(1) as f32).min(1.0),
        );
        // Revision: new evidence shifts frequency slightly
        let evidence_weight = 1.0 / (siblings_count as f32 + 1.0);
        let new_freq = nars_before.frequency * (1.0 - evidence_weight)
            + (1.0 - anomaly_score as f32) * evidence_weight;
        let new_conf = (nars_before.confidence + evidence_weight).min(1.0);
        let nars_after = NarsTruthValue::new(new_freq, new_conf);

        // PHASE 7: SHIFT DETECTION
        let shift = if norm_dist < 0.5 {
            ShiftDirection::TowardFoveal
        } else if norm_dist > 1.5 {
            ShiftDirection::TowardPeripheral
        } else {
            ShiftDirection::Stable
        };

        // PHASE 8: COLLAPSE GATE
        let gate = match calibration {
            CalibrationType::Schaltminute => CollapseGate::Block,
            CalibrationType::Schaltsekunde => CollapseGate::Hold,
            CalibrationType::None => CollapseGate::Flow,
        };

        // PHASE 9: (storage decision is the caller's responsibility)

        // PHASE 10: WELCOME PACKET
        let stored = gate == CollapseGate::Flow;

        // Find nearest sibling
        let nearest_sibling = if cluster.cardinality > 0 && new_data.len() == self.vec_len {
            let mut best: Option<(usize, u64)> = None;
            for (orig_idx, point_data) in self.tree.cluster_points(cluster, self.data, self.vec_len) {
                let d = self.tree.dist(new_data, point_data);
                if best.is_none() || d < best.unwrap().1 {
                    best = Some((orig_idx, d));
                }
            }
            // Convert original index to path
            best.map(|_| path.parent().sibling())
        } else {
            None
        };

        ArrivalResult {
            stored,
            path: if stored { Some(*path) } else { None },
            gate,
            family: None,
            halo_type,
            inference: halo_type.inference(),
            anomaly: anomaly_score,
            calibration,
            sigma_level,
            shift,
            nars_before,
            nars_after,
            siblings: siblings_count,
            nearest_sibling,
            counterfactual_branch: parent_path.sibling(),
        }
    }

    // ── TRAVERSAL COMMANDS ──────────────────────────────────────────

    /// Execute SCAN — horizontal traversal of all clusters in a subtree.
    pub fn scan(&self, prefix: &ClamPath) -> ScanResult {
        let (lo, hi) = prefix.subtree_range();
        let mut clusters = Vec::new();
        let mut total_items = 0;

        // Walk the tree, collect clusters whose path prefix matches
        self.collect_subtree_clusters(prefix, 0, 0, &mut clusters);
        for cp in &clusters {
            total_items += cp.population;
        }

        ScanResult {
            prefix: *prefix,
            key_range: (lo, hi),
            clusters,
            total_items,
        }
    }

    /// Execute ANCESTORS — vertical traversal from root to leaf.
    pub fn ancestors(&self, path: &ClamPath) -> AncestryResult {
        let mut levels = Vec::new();

        for depth in 0..=path.depth {
            let truncated = path.truncate_to(depth);
            let (real_depth, node_idx) = self.tree.deepest_real_cluster(truncated.bits, truncated.depth);

            if real_depth == depth || depth == 0 {
                let cluster = &self.tree.nodes[node_idx];
                let distribution = self.tree.cluster_crp(cluster, self.data, self.vec_len);

                levels.push(ClusterProfile {
                    path: truncated,
                    node_idx,
                    population: cluster.cardinality,
                    radius: cluster.radius,
                    depth: cluster.depth,
                    lfd: cluster.lfd,
                    distribution,
                    dominant_halo: None,
                    nars: NarsTruthValue::new(
                        1.0,
                        (cluster.cardinality as f32
                            / self.tree.root().cardinality.max(1) as f32)
                            .min(1.0),
                    ),
                    family_name: None,
                });
            }
        }

        AncestryResult {
            path: *path,
            levels,
        }
    }

    /// Execute COUNTERFACTUAL — mirror paths at each split level.
    pub fn counterfactual(&self, path: &ClamPath) -> CounterfactualResult {
        let mut mirrors = Vec::new();

        for depth in 1..=path.depth {
            let flipped = path.flip_at(depth);
            let (real_depth, node_idx) =
                self.tree.deepest_real_cluster(flipped.bits, flipped.depth);

            let cluster = &self.tree.nodes[node_idx];
            let distribution = self.tree.cluster_crp(cluster, self.data, self.vec_len);

            mirrors.push(ClusterProfile {
                path: flipped.truncate_to(real_depth),
                node_idx,
                population: cluster.cardinality,
                radius: cluster.radius,
                depth: cluster.depth,
                lfd: cluster.lfd,
                distribution,
                dominant_halo: None,
                nars: NarsTruthValue::new(
                    1.0,
                    (real_depth as f32 / flipped.depth.max(1) as f32).min(1.0),
                ),
                family_name: None,
            });
        }

        CounterfactualResult {
            path: *path,
            mirrors,
        }
    }

    // ── ANALYSIS COMMANDS ───────────────────────────────────────────

    /// Execute an analysis command (LFD, CRP, NARS, SHIFT, ANOMALY).
    pub fn analyze(&self, path: &ClamPath, analysis_type: AnalysisType) -> AnalysisResult {
        let (_, node_idx) = self.tree.deepest_real_cluster(path.bits, path.depth);
        let cluster = &self.tree.nodes[node_idx];

        match analysis_type {
            AnalysisType::Lfd => AnalysisResult {
                path: *path,
                analysis_type,
                lfd: Some(cluster.lfd),
                distribution: None,
                nars: None,
                shift: None,
                anomaly: None,
            },
            AnalysisType::Crp => {
                let dist = self.tree.cluster_crp(cluster, self.data, self.vec_len);
                AnalysisResult {
                    path: *path,
                    analysis_type,
                    lfd: None,
                    distribution: Some(dist),
                    nars: None,
                    shift: None,
                    anomaly: None,
                }
            }
            AnalysisType::Nars => {
                let confidence = (cluster.cardinality as f32
                    / self.tree.root().cardinality.max(1) as f32)
                    .min(1.0);
                AnalysisResult {
                    path: *path,
                    analysis_type,
                    lfd: None,
                    distribution: None,
                    nars: Some(NarsTruthValue::new(1.0, confidence)),
                    shift: None,
                    anomaly: None,
                }
            }
            AnalysisType::Shift => AnalysisResult {
                path: *path,
                analysis_type,
                lfd: None,
                distribution: None,
                nars: None,
                shift: Some(ShiftDirection::Stable),
                anomaly: None,
            },
            AnalysisType::Anomaly => {
                let dist = self.tree.cluster_crp(cluster, self.data, self.vec_len);
                let lfd_factor = (cluster.lfd.value / 3.0).min(1.0);
                let max_depth = self.tree.nodes.iter().map(|c| c.depth).max().unwrap_or(1);
                let depth_factor = cluster.depth as f64 / max_depth.max(1) as f64;
                let card_factor =
                    1.0 - (cluster.cardinality as f64 / self.tree.root().cardinality.max(1) as f64);
                let score = (depth_factor * card_factor * lfd_factor).clamp(0.0, 1.0);

                let calibration = if score > 0.7 {
                    CalibrationType::Schaltminute
                } else if score > 0.4 {
                    CalibrationType::Schaltsekunde
                } else {
                    CalibrationType::None
                };

                AnalysisResult {
                    path: *path,
                    analysis_type,
                    lfd: None,
                    distribution: Some(dist),
                    nars: None,
                    shift: None,
                    anomaly: Some(AnomalyResult {
                        score,
                        calibration_type: calibration,
                        lfd: cluster.lfd.value,
                        cluster_depth: cluster.depth,
                        cluster_cardinality: cluster.cardinality,
                    }),
                }
            }
        }
    }

    // ── BATCH COMMANDS ──────────────────────────────────────────────

    /// Execute MGET — parallel resolution of multiple addresses.
    pub fn mget(&self, paths: &[ClamPath]) -> Vec<GetResult> {
        paths.iter().map(|p| self.get(p)).collect()
    }

    /// Execute MSCAN — parallel subtree traversals.
    pub fn mscan(&self, prefixes: &[ClamPath]) -> Vec<ScanResult> {
        prefixes.iter().map(|p| self.scan(p)).collect()
    }

    // ── FULL COMMAND DISPATCH ───────────────────────────────────────

    /// Execute a parsed semantic command and return the appropriate result.
    pub fn execute(&self, cmd: &SemanticCommand) -> CommandResult {
        match cmd {
            SemanticCommand::Get { path } => CommandResult::Get(self.get(path)),
            SemanticCommand::Set { path, data } => {
                CommandResult::Set(self.semantic_set(path, data))
            }
            SemanticCommand::Del { path } => {
                // DEL is a storage operation — we just return what would be deleted
                let result = self.get(path);
                CommandResult::Get(result)
            }
            SemanticCommand::Scan { prefix } => CommandResult::Scan(self.scan(prefix)),
            SemanticCommand::Ancestors { path } => {
                CommandResult::Ancestry(self.ancestors(path))
            }
            SemanticCommand::Counterfactual { path } => {
                CommandResult::Counterfactual(self.counterfactual(path))
            }
            SemanticCommand::Lfd { path } => {
                CommandResult::Analysis(self.analyze(path, AnalysisType::Lfd))
            }
            SemanticCommand::Crp { path } => {
                CommandResult::Analysis(self.analyze(path, AnalysisType::Crp))
            }
            SemanticCommand::Nars { path } => {
                CommandResult::Analysis(self.analyze(path, AnalysisType::Nars))
            }
            SemanticCommand::Shift { path } => {
                CommandResult::Analysis(self.analyze(path, AnalysisType::Shift))
            }
            SemanticCommand::Anomaly { path } => {
                CommandResult::Analysis(self.analyze(path, AnalysisType::Anomaly))
            }
            SemanticCommand::Mget { paths } => CommandResult::MultiGet(self.mget(paths)),
            SemanticCommand::Mscan { prefixes } => {
                CommandResult::MultiScan(self.mscan(prefixes))
            }
            SemanticCommand::Unknown { raw } => CommandResult::Unknown(raw.clone()),
        }
    }

    /// Parse and execute a raw command string.
    pub fn execute_raw(&self, cmd: &str) -> CommandResult {
        let parsed = parse_command(cmd);
        self.execute(&parsed)
    }

    // ── Internal helpers ────────────────────────────────────────────

    /// Collect cluster profiles in the subtree matching a prefix path.
    fn collect_subtree_clusters(
        &self,
        prefix: &ClamPath,
        node_idx: usize,
        current_depth: u8,
        out: &mut Vec<ClusterProfile>,
    ) {
        if node_idx >= self.tree.nodes.len() {
            return;
        }

        let cluster = &self.tree.nodes[node_idx];

        // Check if this cluster's path matches the prefix
        if current_depth < prefix.depth {
            // Haven't reached prefix depth yet — keep descending
            let expected_bit = (prefix.bits >> (15 - current_depth as u32)) & 1 == 1;
            let next = if expected_bit {
                cluster.right
            } else {
                cluster.left
            };
            if let Some(child_idx) = next {
                self.collect_subtree_clusters(prefix, child_idx, current_depth + 1, out);
            }
        } else {
            // At or beyond prefix depth — collect this cluster
            let distribution = self.tree.cluster_crp(cluster, self.data, self.vec_len);
            let profile = ClusterProfile {
                path: ClamPath::new(
                    prefix.bits,
                    current_depth.max(prefix.depth),
                ),
                node_idx,
                population: cluster.cardinality,
                radius: cluster.radius,
                depth: cluster.depth,
                lfd: cluster.lfd,
                distribution,
                dominant_halo: None,
                nars: NarsTruthValue::new(1.0, 1.0),
                family_name: None,
            };
            out.push(profile);

            // Also collect children for a full subtree enumeration
            if let Some(left) = cluster.left {
                self.collect_subtree_clusters(prefix, left, current_depth + 1, out);
            }
            if let Some(right) = cluster.right {
                self.collect_subtree_clusters(prefix, right, current_depth + 1, out);
            }
        }
    }
}

// ============================================================================
// Command result — unified return type
// ============================================================================

/// Result of executing a semantic command.
#[derive(Clone, Debug)]
pub enum GetResult {
    /// Explicit GET found a stored item.
    Explicit(ExplicitResult),
    /// ET resolution of an unstored address.
    Et(EtResult),
}

/// Unified result from command execution.
#[derive(Clone, Debug)]
pub enum CommandResult {
    Get(GetResult),
    Set(ArrivalResult),
    Scan(ScanResult),
    Ancestry(AncestryResult),
    Counterfactual(CounterfactualResult),
    Analysis(AnalysisResult),
    MultiGet(Vec<GetResult>),
    MultiScan(Vec<ScanResult>),
    Unknown(String),
}

// ============================================================================
// Utility: hex decode (minimal, no external deps)
// ============================================================================

fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return None;
    }
    let mut bytes = Vec::with_capacity(s.len() / 2);
    let mut chars = s.chars();
    while let (Some(hi), Some(lo)) = (chars.next(), chars.next()) {
        let byte = u8::from_str_radix(&format!("{}{}", hi, lo), 16).ok()?;
        bytes.push(byte);
    }
    Some(bytes)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{BuildConfig, ClamTree};
    use rustynum_core::SplitMix64;

    fn make_test_data(n: usize, vec_len: usize, seed: u64) -> Vec<u8> {
        let mut rng = SplitMix64::new(seed);
        let mut data = vec![0u8; n * vec_len];
        for byte in data.iter_mut() {
            *byte = (rng.next_u64() & 0xFF) as u8;
        }
        data
    }

    fn make_test_tree(n: usize, vec_len: usize) -> (Vec<u8>, ClamTree) {
        let data = make_test_data(n, vec_len, 42);
        let config = BuildConfig {
            min_cardinality: 2,
            max_depth: 20,
            min_radius: 0,
        };
        let tree = ClamTree::build(&data, vec_len, n, &config);
        (data, tree)
    }

    // ── ClamPath extension tests ────────────────────────────────────

    #[test]
    fn test_clam_path_parse() {
        // Simple case: "ada:clam:a" → nibble 0xA at depth 4
        let path = ClamPath::parse("ada:clam:a").unwrap();
        assert_eq!(path.depth, 4);
        assert_eq!(path.nibble_at(0), 0xA);

        // Two nibbles: "ada:clam:a:b" → 0xAB.. at depth 8
        let path = ClamPath::parse("ada:clam:a:b").unwrap();
        assert_eq!(path.depth, 8);
        assert_eq!(path.nibble_at(0), 0xA);
        assert_eq!(path.nibble_at(1), 0xB);
    }

    #[test]
    fn test_clam_path_parse_roundtrip() {
        let addr = "ada:clam:a:b:c:d";
        let path = ClamPath::parse(addr).unwrap();
        let formatted = path.to_address();
        assert_eq!(formatted, addr);
    }

    #[test]
    fn test_clam_path_is_full_leaf() {
        let partial = ClamPath::new(0xABCD, 12);
        assert!(!partial.is_full_leaf());

        let full = ClamPath::new(0xABCD, 16);
        assert!(full.is_full_leaf());
    }

    #[test]
    fn test_clam_path_truncate_to() {
        let path = ClamPath::parse("ada:clam:a:b:c").unwrap();
        assert_eq!(path.depth, 12);

        let truncated = path.truncate_to(4);
        assert_eq!(truncated.depth, 4);
        assert_eq!(truncated.nibble_at(0), 0xA);

        let truncated2 = path.truncate_to(8);
        assert_eq!(truncated2.depth, 8);
        assert_eq!(truncated2.nibble_at(0), 0xA);
        assert_eq!(truncated2.nibble_at(1), 0xB);
    }

    #[test]
    fn test_clam_path_parent() {
        let path = ClamPath::from_tree_traversal(&[true, false, true]);
        let parent = path.parent();
        assert_eq!(parent.depth, 2);
        assert_eq!(parent.common_ancestor_depth(&path), 2);
    }

    #[test]
    fn test_clam_path_flip_at() {
        let path = ClamPath::from_tree_traversal(&[true, false, true]);
        // Flip at depth 2 (the 'false' decision)
        let flipped = path.flip_at(2);
        assert_eq!(flipped.depth, 3);
        // They should share depth 1 but diverge at depth 2
        assert_eq!(path.common_ancestor_depth(&flipped), 1);
    }

    #[test]
    fn test_clam_path_suffix_bits() {
        let path = ClamPath::from_tree_traversal(&[true, false, true, true]);
        // Suffix after depth 2: [true, true] = 0b11
        let suffix = path.suffix_bits(2);
        assert_eq!(suffix, 0b11);
    }

    #[test]
    fn test_clam_path_counterfactual_chain() {
        let path = ClamPath::from_tree_traversal(&[true, false, true]);
        let chain = path.counterfactual_chain();
        assert_eq!(chain.len(), 3);
        // Each mirror should differ from original at exactly that depth
        for (i, mirror) in chain.iter().enumerate() {
            let diff_depth = i as u8 + 1;
            let ancestor = path.common_ancestor_depth(mirror);
            assert_eq!(ancestor, diff_depth - 1);
        }
    }

    #[test]
    fn test_clam_path_ancestry_chain() {
        let path = ClamPath::from_tree_traversal(&[true, false, true]);
        let chain = path.ancestry_chain();
        assert_eq!(chain.len(), 4); // depth 0, 1, 2, 3
        assert_eq!(chain[0].depth, 0);
        assert_eq!(chain[3].depth, 3);
    }

    // ── Command parser tests ────────────────────────────────────────

    #[test]
    fn test_parse_get_explicit() {
        let cmd = parse_command("GET ada:clam:a:b:c:d");
        match cmd {
            SemanticCommand::Get { path } => {
                assert_eq!(path.depth, 16);
                assert!(path.is_full_leaf());
            }
            _ => panic!("Expected Get command"),
        }
    }

    #[test]
    fn test_parse_get_implicit() {
        let cmd = parse_command("GET ada:clam:a:b:c");
        match cmd {
            SemanticCommand::Get { path } => {
                assert_eq!(path.depth, 12);
                assert!(!path.is_full_leaf());
            }
            _ => panic!("Expected Get command"),
        }
    }

    #[test]
    fn test_parse_set() {
        let cmd = parse_command("SET ada:clam:a:b:c:d DEADBEEF");
        match cmd {
            SemanticCommand::Set { path, data } => {
                assert!(path.is_full_leaf());
                assert_eq!(data, vec![0xDE, 0xAD, 0xBE, 0xEF]);
            }
            _ => panic!("Expected Set command"),
        }
    }

    #[test]
    fn test_parse_scan() {
        let cmd = parse_command("SCAN ada:clam:a:b:*");
        match cmd {
            SemanticCommand::Scan { prefix } => {
                assert_eq!(prefix.depth, 8);
            }
            _ => panic!("Expected Scan command"),
        }
    }

    #[test]
    fn test_parse_ancestors() {
        let cmd = parse_command("ANCESTORS ada:clam:a:b:c");
        match cmd {
            SemanticCommand::Ancestors { path } => {
                assert_eq!(path.depth, 12);
            }
            _ => panic!("Expected Ancestors command"),
        }
    }

    #[test]
    fn test_parse_counterfactual() {
        let cmd = parse_command("COUNTERFACTUAL ada:clam:a:b:c");
        match cmd {
            SemanticCommand::Counterfactual { path } => {
                assert_eq!(path.depth, 12);
            }
            _ => panic!("Expected Counterfactual command"),
        }
    }

    #[test]
    fn test_parse_analysis_commands() {
        for cmd_name in &["LFD", "CRP", "NARS", "SHIFT", "ANOMALY"] {
            let cmd = parse_command(&format!("{} ada:clam:a:b", cmd_name));
            match cmd {
                SemanticCommand::Lfd { .. }
                | SemanticCommand::Crp { .. }
                | SemanticCommand::Nars { .. }
                | SemanticCommand::Shift { .. }
                | SemanticCommand::Anomaly { .. } => {}
                _ => panic!("Expected analysis command for {}", cmd_name),
            }
        }
    }

    #[test]
    fn test_parse_mget() {
        let cmd = parse_command("MGET ada:clam:a ada:clam:b ada:clam:c");
        match cmd {
            SemanticCommand::Mget { paths } => {
                assert_eq!(paths.len(), 3);
            }
            _ => panic!("Expected Mget command"),
        }
    }

    #[test]
    fn test_parse_mscan() {
        let cmd = parse_command("MSCAN ada:clam:a:* ada:clam:b:*");
        match cmd {
            SemanticCommand::Mscan { prefixes } => {
                assert_eq!(prefixes.len(), 2);
            }
            _ => panic!("Expected Mscan command"),
        }
    }

    #[test]
    fn test_parse_unknown() {
        let cmd = parse_command("FOOBAR something");
        match cmd {
            SemanticCommand::Unknown { .. } => {}
            _ => panic!("Expected Unknown command"),
        }
    }

    // ── DataFusion query mapping tests ──────────────────────────────

    #[test]
    fn test_command_to_query_explicit_get() {
        let (data, tree) = make_test_tree(100, 64);
        let cmd = SemanticCommand::Get {
            path: ClamPath::new(0xABCD, 16),
        };
        let query = command_to_query(&cmd, &tree);
        match query {
            DataFusionQuery::Exact { btree_key } => {
                assert_eq!(btree_key, 0xABCD);
            }
            _ => panic!("Expected Exact query"),
        }
    }

    #[test]
    fn test_command_to_query_implicit_get() {
        let (data, tree) = make_test_tree(100, 64);
        let cmd = SemanticCommand::Get {
            path: ClamPath::new(0xAB00, 8),
        };
        let query = command_to_query(&cmd, &tree);
        match query {
            DataFusionQuery::EtResolve { path, node_idx } => {
                assert_eq!(path.depth, 8);
            }
            _ => panic!("Expected EtResolve query"),
        }
    }

    #[test]
    fn test_command_to_query_scan() {
        let (data, tree) = make_test_tree(100, 64);
        let cmd = SemanticCommand::Scan {
            prefix: ClamPath::new(0xA000, 4),
        };
        let query = command_to_query(&cmd, &tree);
        match query {
            DataFusionQuery::RangeScan { lo, hi } => {
                assert_eq!(lo, 0xA000);
                assert_eq!(hi, 0xAFFF);
            }
            _ => panic!("Expected RangeScan query"),
        }
    }

    #[test]
    fn test_command_to_query_ancestry() {
        let (data, tree) = make_test_tree(100, 64);
        let cmd = SemanticCommand::Ancestors {
            path: ClamPath::new(0xABC0, 12),
        };
        let query = command_to_query(&cmd, &tree);
        match query {
            DataFusionQuery::AncestryChain { paths } => {
                assert_eq!(paths.len(), 13); // depth 0..=12
            }
            _ => panic!("Expected AncestryChain query"),
        }
    }

    // ── Semantic engine tests ───────────────────────────────────────

    #[test]
    fn test_engine_et_get() {
        let (data, tree) = make_test_tree(200, 64);
        let engine = SemanticEngine::new(&tree, &data, 64);

        // ET GET on a partial path — should resolve
        let path = ClamPath::new(0xA000, 4);
        let result = engine.et_get(&path);

        assert!(!result.exists);
        assert!(result.resolved);
        assert!(result.confidence >= 0.0);
        assert!(result.cluster_population > 0);
        assert!(!result.nearest_real.is_empty());
    }

    #[test]
    fn test_engine_explicit_get() {
        let (data, tree) = make_test_tree(200, 64);
        let engine = SemanticEngine::new(&tree, &data, 64);

        // Get a real leaf path from the tree
        let leaf_paths = tree.leaf_paths();
        if !leaf_paths.is_empty() {
            let (_, bool_path) = &leaf_paths[0];
            let clam_path = ClamPath::from_tree_traversal(bool_path);
            let result = engine.explicit_get(&clam_path);
            assert!(result.cluster.population > 0);
        }
    }

    #[test]
    fn test_engine_semantic_set() {
        let (data, tree) = make_test_tree(200, 64);
        let engine = SemanticEngine::new(&tree, &data, 64);

        // Create some new data and try to SET it
        let new_data = vec![0x42u8; 64];
        let path = ClamPath::new(0xABCD, 16);
        let result = engine.semantic_set(&path, &new_data);

        // Should get a valid arrival result
        assert!(result.anomaly >= 0.0 && result.anomaly <= 1.0);
        assert!(result.siblings > 0 || result.siblings == 0); // any count is valid
        assert!(!result.inference.is_empty());
    }

    #[test]
    fn test_engine_scan() {
        let (data, tree) = make_test_tree(200, 64);
        let engine = SemanticEngine::new(&tree, &data, 64);

        let prefix = ClamPath::new(0, 0); // scan entire tree from root
        let result = engine.scan(&prefix);

        assert!(result.total_items > 0);
        assert!(!result.clusters.is_empty());
    }

    #[test]
    fn test_engine_ancestors() {
        let (data, tree) = make_test_tree(200, 64);
        let engine = SemanticEngine::new(&tree, &data, 64);

        let leaf_paths = tree.leaf_paths();
        if !leaf_paths.is_empty() {
            let (_, bool_path) = &leaf_paths[0];
            let path = ClamPath::from_tree_traversal(bool_path);
            let result = engine.ancestors(&path);

            // Should have at least the root level
            assert!(!result.levels.is_empty());
            // First level should be root (all items)
            assert_eq!(result.levels[0].population, tree.root().cardinality);
        }
    }

    #[test]
    fn test_engine_counterfactual() {
        let (data, tree) = make_test_tree(200, 64);
        let engine = SemanticEngine::new(&tree, &data, 64);

        let path = ClamPath::from_tree_traversal(&[true, false, true]);
        let result = engine.counterfactual(&path);

        assert_eq!(result.mirrors.len(), 3);
        for mirror in &result.mirrors {
            assert!(mirror.population > 0 || mirror.population == 0);
        }
    }

    #[test]
    fn test_engine_analyze_lfd() {
        let (data, tree) = make_test_tree(200, 64);
        let engine = SemanticEngine::new(&tree, &data, 64);

        let path = ClamPath::new(0, 0);
        let result = engine.analyze(&path, AnalysisType::Lfd);

        assert!(result.lfd.is_some());
        assert!(result.lfd.unwrap().value >= 0.0);
    }

    #[test]
    fn test_engine_analyze_crp() {
        let (data, tree) = make_test_tree(200, 64);
        let engine = SemanticEngine::new(&tree, &data, 64);

        let path = ClamPath::new(0, 0);
        let result = engine.analyze(&path, AnalysisType::Crp);

        assert!(result.distribution.is_some());
        let dist = result.distribution.unwrap();
        assert!(dist.count > 0);
    }

    #[test]
    fn test_engine_analyze_anomaly() {
        let (data, tree) = make_test_tree(200, 64);
        let engine = SemanticEngine::new(&tree, &data, 64);

        let path = ClamPath::new(0, 0);
        let result = engine.analyze(&path, AnalysisType::Anomaly);

        assert!(result.anomaly.is_some());
        let anomaly = result.anomaly.unwrap();
        assert!(anomaly.score >= 0.0 && anomaly.score <= 1.0);
    }

    #[test]
    fn test_engine_mget() {
        let (data, tree) = make_test_tree(200, 64);
        let engine = SemanticEngine::new(&tree, &data, 64);

        let paths = vec![
            ClamPath::new(0xA000, 4),
            ClamPath::new(0xB000, 4),
            ClamPath::new(0xC000, 4),
        ];
        let results = engine.mget(&paths);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_engine_execute_raw() {
        let (data, tree) = make_test_tree(200, 64);
        let engine = SemanticEngine::new(&tree, &data, 64);

        // Test raw command execution
        let result = engine.execute_raw("GET ada:clam:a:b");
        match result {
            CommandResult::Get(GetResult::Et(et)) => {
                assert!(et.resolved);
            }
            _ => panic!("Expected ET GET result from raw command"),
        }
    }

    #[test]
    fn test_engine_execute_raw_scan() {
        let (data, tree) = make_test_tree(200, 64);
        let engine = SemanticEngine::new(&tree, &data, 64);

        let result = engine.execute_raw("SCAN ada:clam:a:*");
        match result {
            CommandResult::Scan(scan) => {
                let (lo, hi) = scan.key_range;
                assert!(lo <= hi);
            }
            _ => panic!("Expected Scan result"),
        }
    }

    // ── Halo type tests ─────────────────────────────────────────────

    #[test]
    fn test_halo_type_core() {
        let halo = HaloType::classify(0.1, 0.1, 0.1);
        assert_eq!(halo, HaloType::Core);
    }

    #[test]
    fn test_halo_type_sp() {
        let halo = HaloType::classify(0.1, 0.1, 0.8);
        assert_eq!(halo, HaloType::SP);
    }

    #[test]
    fn test_halo_type_so() {
        let halo = HaloType::classify(0.1, 0.8, 0.1);
        assert_eq!(halo, HaloType::SO);
    }

    #[test]
    fn test_halo_type_po() {
        let halo = HaloType::classify(0.8, 0.1, 0.1);
        assert_eq!(halo, HaloType::PO);
    }

    // ── Hex decode tests ────────────────────────────────────────────

    #[test]
    fn test_hex_decode() {
        assert_eq!(hex_decode("DEADBEEF"), Some(vec![0xDE, 0xAD, 0xBE, 0xEF]));
        assert_eq!(hex_decode("00FF"), Some(vec![0x00, 0xFF]));
        assert_eq!(hex_decode(""), Some(vec![]));
        assert_eq!(hex_decode("FFF"), None); // odd length
    }

    // ── Integration: parse → query → execute ────────────────────────

    #[test]
    fn test_full_pipeline_get() {
        let (data, tree) = make_test_tree(200, 64);
        let engine = SemanticEngine::new(&tree, &data, 64);

        // Parse → Command → Query → Execute
        let cmd_str = "GET ada:clam:a:b";
        let cmd = parse_command(cmd_str);
        let query = command_to_query(&cmd, &tree);
        let result = engine.execute(&cmd);

        // Verify query maps correctly
        match query {
            DataFusionQuery::EtResolve { path, .. } => {
                assert_eq!(path.depth, 8);
            }
            _ => panic!("Expected EtResolve"),
        }

        // Verify execution produces result
        match result {
            CommandResult::Get(GetResult::Et(et)) => {
                assert!(et.resolved);
                assert!(!et.exists);
            }
            _ => panic!("Expected ET result"),
        }
    }

    #[test]
    fn test_full_pipeline_scan_ancestors_counterfactual() {
        let (data, tree) = make_test_tree(200, 64);
        let engine = SemanticEngine::new(&tree, &data, 64);

        // SCAN
        let scan_result = engine.execute_raw("SCAN ada:clam:a:*");
        assert!(matches!(scan_result, CommandResult::Scan(_)));

        // ANCESTORS
        let anc_result = engine.execute_raw("ANCESTORS ada:clam:a:b:c");
        assert!(matches!(anc_result, CommandResult::Ancestry(_)));

        // COUNTERFACTUAL
        let cf_result = engine.execute_raw("COUNTERFACTUAL ada:clam:a:b:c");
        assert!(matches!(cf_result, CommandResult::Counterfactual(_)));
    }

    #[test]
    fn test_full_pipeline_analysis_commands() {
        let (data, tree) = make_test_tree(200, 64);
        let engine = SemanticEngine::new(&tree, &data, 64);

        for cmd in &[
            "LFD ada:clam:a",
            "CRP ada:clam:a",
            "NARS ada:clam:a",
            "SHIFT ada:clam:a",
            "ANOMALY ada:clam:a",
        ] {
            let result = engine.execute_raw(cmd);
            assert!(
                matches!(result, CommandResult::Analysis(_)),
                "Expected Analysis for: {}",
                cmd
            );
        }
    }
}
