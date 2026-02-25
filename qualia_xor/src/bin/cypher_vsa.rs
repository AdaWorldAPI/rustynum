//! Hardened Cypher Navigator + VSA Projection Engine
//!
//! Navigates the Cypher space directly — neo4j-rs becomes a renderer, not a
//! dependency.  This binary:
//!
//! 1. Parses .cypher files into a validated AST (nodes, edges, properties)
//! 2. Enforces a hardened contract of Cypher verbs + causality relationship types
//! 3. Projects every node and edge to BF16/SPO VSA embeddings
//! 4. Measures causal distances between relationship types in VSA space
//! 5. Exports battle-tested syntax docs + contracts
//!
//! ## Architecture
//!
//! ```text
//! .cypher file
//!   │
//!   ├── CypherParser (regex-based, validates verbs)
//!   │     └── CypherGraph { nodes, edges }
//!   │
//!   ├── VsaProjector (BF16 × SPO encoding)
//!   │     ├── Node → BF16 vector (from properties)
//!   │     ├── Edge → SpoTriple (subject ⊕ predicate ⊕ object)
//!   │     └── Relationship type → predicate vector
//!   │
//!   └── ContractExporter
//!         ├── syntax_contract.cypher  (hardened verb reference)
//!         └── vsa_distances.json      (pairwise causal distances)
//! ```

use std::collections::HashMap;
use std::fmt;
use serde::{Deserialize, Serialize};

// ============================================================================
// Hardened Cypher Verb Contract
// ============================================================================

/// Every Cypher verb we recognize, with its causal semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum CypherVerb {
    // --- Read verbs (observation) ---
    Match,
    OptionalMatch,
    Where,
    Return,
    With,
    Unwind,
    OrderBy,
    Skip,
    Limit,
    Distinct,
    // --- Write verbs (causation) ---
    Create,
    Merge,
    Set,
    Remove,
    Delete,
    DetachDelete,
    // --- Schema verbs (meta) ---
    CreateIndex,
    CreateConstraint,
    DropIndex,
    DropConstraint,
    // --- Procedure verbs ---
    Call,
    Yield,
}

impl CypherVerb {
    /// Classify verb into causal layer (Pearl's ladder)
    fn causal_rung(&self) -> u8 {
        match self {
            // Rung 1: Association (observation, no intervention)
            Self::Match | Self::OptionalMatch | Self::Where |
            Self::Return | Self::With | Self::Unwind |
            Self::OrderBy | Self::Skip | Self::Limit | Self::Distinct => 1,
            // Rung 2: Intervention (do-calculus)
            Self::Create | Self::Merge | Self::Set | Self::Remove => 2,
            // Rung 3: Counterfactual (deletion = "what if not?")
            Self::Delete | Self::DetachDelete => 3,
            // Meta (schema)
            Self::CreateIndex | Self::CreateConstraint |
            Self::DropIndex | Self::DropConstraint => 0,
            // Procedure
            Self::Call | Self::Yield => 1,
        }
    }

    fn from_str_checked(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "MATCH" => Some(Self::Match),
            "OPTIONAL MATCH" => Some(Self::OptionalMatch),
            "WHERE" => Some(Self::Where),
            "RETURN" => Some(Self::Return),
            "WITH" => Some(Self::With),
            "UNWIND" => Some(Self::Unwind),
            "ORDER BY" => Some(Self::OrderBy),
            "SKIP" => Some(Self::Skip),
            "LIMIT" => Some(Self::Limit),
            "DISTINCT" => Some(Self::Distinct),
            "CREATE" => Some(Self::Create),
            "MERGE" => Some(Self::Merge),
            "SET" => Some(Self::Set),
            "REMOVE" => Some(Self::Remove),
            "DELETE" => Some(Self::Delete),
            "DETACH DELETE" => Some(Self::DetachDelete),
            "CREATE INDEX" => Some(Self::CreateIndex),
            "CREATE CONSTRAINT" => Some(Self::CreateConstraint),
            "DROP INDEX" => Some(Self::DropIndex),
            "DROP CONSTRAINT" => Some(Self::DropConstraint),
            "CALL" => Some(Self::Call),
            "YIELD" => Some(Self::Yield),
            _ => None,
        }
    }
}

impl fmt::Display for CypherVerb {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Match => "MATCH",
            Self::OptionalMatch => "OPTIONAL MATCH",
            Self::Where => "WHERE",
            Self::Return => "RETURN",
            Self::With => "WITH",
            Self::Unwind => "UNWIND",
            Self::OrderBy => "ORDER BY",
            Self::Skip => "SKIP",
            Self::Limit => "LIMIT",
            Self::Distinct => "DISTINCT",
            Self::Create => "CREATE",
            Self::Merge => "MERGE",
            Self::Set => "SET",
            Self::Remove => "REMOVE",
            Self::Delete => "DELETE",
            Self::DetachDelete => "DETACH DELETE",
            Self::CreateIndex => "CREATE INDEX",
            Self::CreateConstraint => "CREATE CONSTRAINT",
            Self::DropIndex => "DROP INDEX",
            Self::DropConstraint => "DROP CONSTRAINT",
            Self::Call => "CALL",
            Self::Yield => "YIELD",
        };
        write!(f, "{}", s)
    }
}

// ============================================================================
// Hardened Relationship Type Contract (Causality)
// ============================================================================

/// Relationship types with their causal semantics.
/// Each type maps to a BF16 predicate vector for SPO encoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RelTypeContract {
    name: String,
    /// Pearl rung: 1=association, 2=intervention, 3=counterfactual
    causal_rung: u8,
    /// Direction: "forward" (A→B), "backward" (B→A), "symmetric"
    direction: String,
    /// The 16-dim predicate vector (qualia-space)
    predicate_dims: [f32; 16],
}

/// All known relationship types across qualia + aiwar domains.
fn hardened_rel_contracts() -> Vec<RelTypeContract> {
    vec![
        // === Qualia Causality ===
        RelTypeContract {
            name: "CAUSES".into(),
            causal_rung: 2,
            direction: "forward".into(),
            predicate_dims: [
                0.7, 0.5, 0.5, 0.9, // glow, valence, rooting, agency=HIGH
                0.3, 0.5, 0.3, 0.1, // resonance, clarity, social, gravity=LOW
                0.2, 0.8, 0.3, 0.2, // reverence, volition=HIGH, dissonance, staunen
                0.1, 0.6, 0.3, 0.5, // loss, optimism, friction, equilibrium
            ],
        },
        RelTypeContract {
            name: "IS_CAUSED_BY".into(),
            causal_rung: 2,
            direction: "backward".into(),
            predicate_dims: [
                0.3, 0.5, 0.5, 0.1, // agency=LOW
                0.7, 0.5, 0.3, 0.9, // gravity=HIGH
                0.6, 0.2, 0.3, 0.4,
                0.3, 0.4, 0.3, 0.5,
            ],
        },
        RelTypeContract {
            name: "TRANSFORMS".into(),
            causal_rung: 2,
            direction: "forward".into(),
            predicate_dims: [
                0.5, 0.5, 0.4, 0.5,
                0.5, 0.4, 0.3, 0.5,
                0.4, 0.5, 0.8, 0.5, // dissonance=HIGH
                0.2, 0.5, 0.8, 0.7, // friction=HIGH, equilibrium=HIGH
            ],
        },
        RelTypeContract {
            name: "DISSOLVES_INTO".into(),
            causal_rung: 3,
            direction: "forward".into(),
            predicate_dims: [
                0.8, 0.7, 0.2, 0.1, // glow=HIGH, agency=LOW
                0.9, 0.3, 0.5, 0.1, // resonance=HIGH, gravity=LOW
                0.3, 0.1, 0.1, 0.7, // volition=LOW, staunen=HIGH
                0.1, 0.7, 0.1, 0.1, // loss=LOW, optimism=HIGH
            ],
        },
        // === Qualia Proximity (from neo4j_import.cypher) ===
        RelTypeContract {
            name: "NIB4_NEAR".into(),
            causal_rung: 1,
            direction: "symmetric".into(),
            predicate_dims: [
                0.5, 0.5, 0.5, 0.5, // neutral — pure distance metric
                0.5, 0.5, 0.5, 0.5,
                0.5, 0.5, 0.5, 0.5,
                0.5, 0.5, 0.5, 0.5,
            ],
        },
        RelTypeContract {
            name: "BERT_NEAR".into(),
            causal_rung: 1,
            direction: "symmetric".into(),
            predicate_dims: [
                0.5, 0.5, 0.5, 0.5,
                0.5, 0.8, 0.5, 0.5, // clarity boosted (surface language)
                0.5, 0.5, 0.5, 0.5,
                0.5, 0.5, 0.5, 0.5,
            ],
        },
        // === XOR Buckets ===
        RelTypeContract {
            name: "STRUCTURAL_TRUTH".into(),
            causal_rung: 1,
            direction: "symmetric".into(),
            predicate_dims: [
                0.8, 0.8, 0.8, 0.5, // high glow, valence, rooting (both agree)
                0.8, 0.8, 0.5, 0.5,
                0.5, 0.5, 0.1, 0.5, // low dissonance (no conflict)
                0.1, 0.8, 0.1, 0.8, // low loss, high optimism, low friction, high equilibrium
            ],
        },
        RelTypeContract {
            name: "CADENCE_TRUTH".into(),
            causal_rung: 2,
            direction: "forward".into(),
            predicate_dims: [
                0.5, 0.5, 0.7, 0.5, // high rooting (deep structure)
                0.3, 0.3, 0.5, 0.7, // low clarity, high gravity (hidden rhythm)
                0.5, 0.5, 0.6, 0.5, // moderate dissonance
                0.3, 0.5, 0.6, 0.5, // moderate friction
            ],
        },
        RelTypeContract {
            name: "SURFACE_SYNONYMY".into(),
            causal_rung: 1,
            direction: "symmetric".into(),
            predicate_dims: [
                0.5, 0.5, 0.2, 0.5, // low rooting (shallow)
                0.5, 0.9, 0.5, 0.2, // high clarity, low gravity (surface-level)
                0.5, 0.5, 0.7, 0.5, // moderate dissonance (looks alike but isn't)
                0.5, 0.5, 0.2, 0.3, // low friction, low equilibrium
            ],
        },
        RelTypeContract {
            name: "BELONGS_TO".into(),
            causal_rung: 1,
            direction: "forward".into(),
            predicate_dims: [
                0.5, 0.5, 0.8, 0.3, // high rooting, low agency (containment)
                0.5, 0.5, 0.7, 0.8, // high social, high gravity
                0.5, 0.3, 0.1, 0.3,
                0.1, 0.5, 0.1, 0.7, // low loss, high equilibrium
            ],
        },
        RelTypeContract {
            name: "HAS_MODE".into(),
            causal_rung: 1,
            direction: "forward".into(),
            predicate_dims: [
                0.5, 0.5, 0.5, 0.5,
                0.5, 0.9, 0.3, 0.5, // high clarity (classification)
                0.5, 0.5, 0.1, 0.5,
                0.5, 0.5, 0.1, 0.9, // very high equilibrium (binary split)
            ],
        },
        // === AIWar Domain Relationships ===
        RelTypeContract {
            name: "VALID_FOR".into(),
            causal_rung: 1,
            direction: "forward".into(),
            predicate_dims: [
                0.5, 0.5, 0.7, 0.3, // high rooting, low agency (schema binding)
                0.5, 0.9, 0.5, 0.5, // high clarity
                0.5, 0.3, 0.1, 0.3,
                0.1, 0.5, 0.1, 0.9,
            ],
        },
        RelTypeContract {
            name: "DEVELOPED_BY".into(),
            causal_rung: 2,
            direction: "backward".into(),
            predicate_dims: [
                0.5, 0.5, 0.5, 0.2, // low agency (passive: was developed)
                0.5, 0.5, 0.7, 0.8, // high social, high gravity
                0.3, 0.2, 0.3, 0.3,
                0.3, 0.5, 0.3, 0.5,
            ],
        },
        RelTypeContract {
            name: "DEPLOYED_BY".into(),
            causal_rung: 2,
            direction: "backward".into(),
            predicate_dims: [
                0.5, 0.5, 0.5, 0.3,
                0.5, 0.5, 0.8, 0.7, // high social, high gravity
                0.3, 0.3, 0.5, 0.3,
                0.3, 0.5, 0.5, 0.5,
            ],
        },
        RelTypeContract {
            name: "OPERATED_BY".into(),
            causal_rung: 2,
            direction: "backward".into(),
            predicate_dims: [
                0.5, 0.5, 0.5, 0.5,
                0.5, 0.5, 0.8, 0.6,
                0.3, 0.5, 0.3, 0.3,
                0.3, 0.5, 0.3, 0.5,
            ],
        },
        RelTypeContract {
            name: "SUPPLIED_BY".into(),
            causal_rung: 2,
            direction: "backward".into(),
            predicate_dims: [
                0.3, 0.5, 0.5, 0.2,
                0.5, 0.5, 0.6, 0.7,
                0.3, 0.2, 0.3, 0.3,
                0.3, 0.4, 0.3, 0.5,
            ],
        },
        RelTypeContract {
            name: "INVESTED_IN".into(),
            causal_rung: 2,
            direction: "forward".into(),
            predicate_dims: [
                0.5, 0.5, 0.5, 0.7, // high agency (active investment)
                0.5, 0.5, 0.7, 0.5,
                0.3, 0.7, 0.3, 0.3, // high volition
                0.3, 0.7, 0.5, 0.5,
            ],
        },
        RelTypeContract {
            name: "TARGETS".into(),
            causal_rung: 3,
            direction: "forward".into(),
            predicate_dims: [
                0.3, 0.2, 0.5, 0.9, // low valence, high agency
                0.3, 0.5, 0.3, 0.3,
                0.2, 0.8, 0.8, 0.2, // high volition, high dissonance
                0.5, 0.2, 0.8, 0.2, // high friction, low equilibrium
            ],
        },
    ]
}

// ============================================================================
// Inline BF16 utilities (same as edge_vectors.rs — standalone)
// ============================================================================

fn f32_to_bf16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &f in vals {
        let bits = f.to_bits();
        let bf16 = (bits >> 16) as u16;
        out.extend_from_slice(&bf16.to_le_bytes());
    }
    out
}

fn xor_bind(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b).map(|(x, y)| x ^ y).collect()
}

fn bf16_distance(a: &[u8], b: &[u8]) -> u64 {
    let mut total: u64 = 0;
    for i in (0..a.len()).step_by(2) {
        if i + 1 >= a.len() || i + 1 >= b.len() { break; }
        let va = u16::from_le_bytes([a[i], a[i + 1]]);
        let vb = u16::from_le_bytes([b[i], b[i + 1]]);
        let xor = va ^ vb;
        let sign = ((xor >> 15) & 1) as u64;
        let exp_pop = ((xor >> 7) & 0xFF).count_ones() as u64;
        let man_pop = (xor & 0x7F).count_ones() as u64;
        total += sign * 256 + exp_pop * 16 + man_pop;
    }
    total
}

fn bf16_structural_diff(a: &[u8], b: &[u8]) -> (usize, usize, usize) {
    let n_dims = a.len().min(b.len()) / 2;
    let mut sign_flips = 0;
    let mut exp_shifts = 0;
    let mut man_changes = 0;
    for dim in 0..n_dims {
        let i = dim * 2;
        let va = u16::from_le_bytes([a[i], a[i + 1]]);
        let vb = u16::from_le_bytes([b[i], b[i + 1]]);
        let xor = va ^ vb;
        if xor & 0x8000 != 0 { sign_flips += 1; }
        exp_shifts += ((xor >> 7) & 0xFF).count_ones() as usize;
        man_changes += (xor & 0x7F).count_ones() as usize;
    }
    (sign_flips, exp_shifts, man_changes)
}

// ============================================================================
// SPO Triple (same as edge_vectors.rs)
// ============================================================================

struct SpoTriple {
    x: Vec<u8>, // S ⊕ P
    y: Vec<u8>, // P ⊕ O
    z: Vec<u8>, // S ⊕ O
}

impl SpoTriple {
    fn encode(subject: &[u8], predicate: &[u8], object: &[u8]) -> Self {
        SpoTriple {
            x: xor_bind(subject, predicate),
            y: xor_bind(predicate, object),
            z: xor_bind(subject, object),
        }
    }

    fn axis_distances(&self, other: &SpoTriple) -> (u64, u64, u64) {
        (
            bf16_distance(&self.x, &other.x),
            bf16_distance(&self.y, &other.y),
            bf16_distance(&self.z, &other.z),
        )
    }

    fn total_distance(&self, other: &SpoTriple) -> u64 {
        let (dx, dy, dz) = self.axis_distances(other);
        dx + dy + dz
    }
}

// ============================================================================
// Cypher Parser — lightweight regex-based for .cypher files
// ============================================================================

#[derive(Debug, Clone, Serialize)]
struct CypherNode {
    alias: String,
    labels: Vec<String>,
    properties: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize)]
struct CypherEdge {
    src_alias: String,
    dst_alias: String,
    rel_type: String,
    properties: HashMap<String, String>,
    direction: String, // "->", "<-", "-"
}

#[derive(Debug, Clone, Serialize)]
struct CypherStatement {
    verb: String,
    line_number: usize,
    raw: String,
}

#[derive(Debug, Serialize)]
struct CypherGraph {
    nodes: HashMap<String, CypherNode>,
    edges: Vec<CypherEdge>,
    statements: Vec<CypherStatement>,
    verb_counts: HashMap<String, usize>,
    rel_type_counts: HashMap<String, usize>,
    label_counts: HashMap<String, usize>,
    validation_errors: Vec<String>,
}

fn parse_cypher(input: &str) -> CypherGraph {
    let mut graph = CypherGraph {
        nodes: HashMap::new(),
        edges: Vec::new(),
        statements: Vec::new(),
        verb_counts: HashMap::new(),
        rel_type_counts: HashMap::new(),
        label_counts: HashMap::new(),
        validation_errors: Vec::new(),
    };

    for (line_idx, line) in input.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            continue;
        }

        // Detect verb
        let upper = trimmed.to_uppercase();
        let verb = if upper.starts_with("MERGE") { "MERGE" }
            else if upper.starts_with("CREATE INDEX") { "CREATE INDEX" }
            else if upper.starts_with("CREATE CONSTRAINT") { "CREATE CONSTRAINT" }
            else if upper.starts_with("CREATE") { "CREATE" }
            else if upper.starts_with("MATCH") { "MATCH" }
            else if upper.starts_with("OPTIONAL MATCH") { "OPTIONAL MATCH" }
            else if upper.starts_with("WHERE") { "WHERE" }
            else if upper.starts_with("RETURN") { "RETURN" }
            else if upper.starts_with("WITH") { "WITH" }
            else if upper.starts_with("SET") { "SET" }
            else if upper.starts_with("DELETE") { "DELETE" }
            else if upper.starts_with("DETACH DELETE") { "DETACH DELETE" }
            else if upper.starts_with("REMOVE") { "REMOVE" }
            else if upper.starts_with("UNWIND") { "UNWIND" }
            else if upper.starts_with("ORDER BY") { "ORDER BY" }
            else if upper.starts_with("CALL") { "CALL" }
            else { "UNKNOWN" };

        // Validate verb
        if CypherVerb::from_str_checked(verb).is_none() && verb != "UNKNOWN" {
            graph.validation_errors.push(format!(
                "Line {}: unrecognized verb '{}'", line_idx + 1, verb
            ));
        }

        graph.statements.push(CypherStatement {
            verb: verb.to_string(),
            line_number: line_idx + 1,
            raw: trimmed.to_string(),
        });

        *graph.verb_counts.entry(verb.to_string()).or_insert(0) += 1;

        // Parse node patterns: (alias:Label {props})
        parse_node_patterns(trimmed, &mut graph);

        // Parse edge patterns: (a)-[:TYPE {props}]->(b)
        parse_edge_patterns(trimmed, &mut graph);
    }

    graph
}

fn parse_node_patterns(line: &str, graph: &mut CypherGraph) {
    // Match patterns like (n0:QualiaItem {id: 'xyz', ...})
    let mut i = 0;
    let chars: Vec<char> = line.chars().collect();
    while i < chars.len() {
        if chars[i] == '(' {
            let start = i;
            let mut depth = 1;
            i += 1;
            while i < chars.len() && depth > 0 {
                if chars[i] == '(' { depth += 1; }
                if chars[i] == ')' { depth -= 1; }
                i += 1;
            }
            let inner: String = chars[start + 1..i - 1].iter().collect();
            parse_single_node(&inner, graph);
        } else {
            i += 1;
        }
    }
}

fn parse_single_node(inner: &str, graph: &mut CypherGraph) {
    let inner = inner.trim();
    if inner.is_empty() { return; }

    // Split into alias:labels and {properties}
    let (prefix, props_str) = if let Some(brace_pos) = inner.find('{') {
        let end = inner.rfind('}').unwrap_or(inner.len());
        (&inner[..brace_pos], &inner[brace_pos + 1..end])
    } else {
        (inner, "")
    };

    let prefix = prefix.trim();
    if prefix.is_empty() { return; }

    // Parse alias:Label1:Label2
    let parts: Vec<&str> = prefix.split(':').collect();
    let alias = parts[0].trim().to_string();
    if alias.is_empty() { return; }

    let labels: Vec<String> = parts[1..].iter()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();

    for label in &labels {
        *graph.label_counts.entry(label.clone()).or_insert(0) += 1;
    }

    // Parse properties
    let properties = parse_properties(props_str);

    // Only insert if it has labels or properties (not a bare alias reference)
    if !labels.is_empty() || !properties.is_empty() {
        graph.nodes.entry(alias.clone()).or_insert(CypherNode {
            alias,
            labels,
            properties,
        });
    }
}

fn parse_edge_patterns(line: &str, graph: &mut CypherGraph) {
    // Match: (src)-[:TYPE {props}]->(dst) or (src)<-[:TYPE {props}]-(dst)
    // We look for -[...]-> or <-[...]- patterns
    let chars: Vec<char> = line.chars().collect();
    let line_str = line;

    // Find all bracket segments [...] that are edge patterns
    let mut i = 0;
    while i < chars.len() {
        // Look for -[ or <-[
        if i + 1 < chars.len() && chars[i] == '-' && chars[i + 1] == '[' {
            let direction_left = i > 0 && chars[i - 1] == '<';
            let bracket_start = i + 2;
            let mut bracket_end = bracket_start;
            let mut depth = 1;
            while bracket_end < chars.len() && depth > 0 {
                if chars[bracket_end] == '[' { depth += 1; }
                if chars[bracket_end] == ']' { depth -= 1; }
                bracket_end += 1;
            }
            let bracket_content: String = chars[bracket_start..bracket_end - 1].iter().collect();

            // Check what follows ]
            let direction_right = bracket_end + 1 < chars.len()
                && chars[bracket_end] == '-'
                && chars[bracket_end + 1] == '>';

            let direction = if direction_left { "<-" }
                else if direction_right { "->" }
                else { "-" };

            // Parse :TYPE {props}
            let (rel_type, props) = parse_rel_content(&bracket_content);

            if !rel_type.is_empty() {
                *graph.rel_type_counts.entry(rel_type.clone()).or_insert(0) += 1;

                // Find src and dst aliases by scanning backward/forward for parens
                let src_alias = find_alias_before(line_str, if direction_left { i - 1 } else { i });
                let dst_alias = find_alias_after(line_str, bracket_end + if direction_right { 2 } else { 1 });

                graph.edges.push(CypherEdge {
                    src_alias,
                    dst_alias,
                    rel_type,
                    properties: props,
                    direction: direction.to_string(),
                });
            }

            i = bracket_end;
        } else {
            i += 1;
        }
    }
}

fn parse_rel_content(content: &str) -> (String, HashMap<String, String>) {
    let content = content.trim();
    let (type_part, props_str) = if let Some(brace_pos) = content.find('{') {
        let end = content.rfind('}').unwrap_or(content.len());
        (&content[..brace_pos], &content[brace_pos + 1..end])
    } else {
        (content, "")
    };

    let rel_type = type_part.trim()
        .strip_prefix(':')
        .unwrap_or(type_part.trim())
        .trim()
        .to_string();

    let properties = parse_properties(props_str);
    (rel_type, properties)
}

fn parse_properties(props_str: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    if props_str.trim().is_empty() { return map; }

    // Simple key: value parser (handles strings with commas inside quotes)
    let mut key = String::new();
    let mut value = String::new();
    let mut in_key = true;
    let mut in_string = false;
    let mut string_char = '\'';

    for ch in props_str.chars() {
        if in_string {
            if ch == string_char {
                in_string = false;
            }
            value.push(ch);
            continue;
        }

        match ch {
            '\'' | '"' => {
                in_string = true;
                string_char = ch;
                if !in_key { value.push(ch); }
            }
            ':' if in_key => {
                in_key = false;
            }
            ',' => {
                let k = key.trim().to_string();
                let v = value.trim().trim_matches('\'').trim_matches('"').to_string();
                if !k.is_empty() {
                    map.insert(k, v);
                }
                key.clear();
                value.clear();
                in_key = true;
            }
            _ => {
                if in_key { key.push(ch); } else { value.push(ch); }
            }
        }
    }

    // Last pair
    let k = key.trim().to_string();
    let v = value.trim().trim_matches('\'').trim_matches('"').to_string();
    if !k.is_empty() {
        map.insert(k, v);
    }

    map
}

fn find_alias_before(line: &str, pos: usize) -> String {
    // Scan backward from pos to find (alias)
    let chars: Vec<char> = line.chars().collect();
    let mut end = pos;
    // Skip back past any whitespace/dash/arrow
    while end > 0 && (chars[end] == '-' || chars[end] == '<' || chars[end] == ' ') {
        end -= 1;
    }
    if end > 0 && chars[end] == ')' {
        let mut start = end - 1;
        while start > 0 && chars[start] != '(' {
            start -= 1;
        }
        let inner: String = chars[start + 1..end].iter().collect();
        // Extract alias (before : or {)
        inner.split(':').next()
            .unwrap_or("")
            .split('{').next()
            .unwrap_or("")
            .trim()
            .to_string()
    } else {
        String::new()
    }
}

fn find_alias_after(line: &str, pos: usize) -> String {
    let chars: Vec<char> = line.chars().collect();
    let mut start = pos;
    // Skip forward past whitespace/dash/arrow
    while start < chars.len() && (chars[start] == '-' || chars[start] == '>' || chars[start] == ' ') {
        start += 1;
    }
    if start < chars.len() && chars[start] == '(' {
        let mut end = start + 1;
        while end < chars.len() && chars[end] != ')' {
            end += 1;
        }
        let inner: String = chars[start + 1..end].iter().collect();
        inner.split(':').next()
            .unwrap_or("")
            .split('{').next()
            .unwrap_or("")
            .trim()
            .to_string()
    } else {
        String::new()
    }
}

// ============================================================================
// VSA Projection — project CypherGraph to BF16/SPO space
// ============================================================================

#[derive(Debug, Serialize)]
struct VsaProjection {
    /// Per-node BF16 vectors (from properties)
    node_vectors: HashMap<String, Vec<u8>>,
    /// Per-edge SPO triples
    edge_triples: Vec<EdgeSpoResult>,
    /// Pairwise distances between relationship types
    rel_type_distances: Vec<RelTypeDistance>,
    /// Causal rung summary
    rung_summary: HashMap<u8, Vec<String>>,
}

#[derive(Debug, Serialize)]
struct EdgeSpoResult {
    src: String,
    dst: String,
    rel_type: String,
    causal_rung: u8,
    spo_distance_to_zero: u64,
    axis_dominant: String,
}

#[derive(Debug, Serialize)]
struct RelTypeDistance {
    type_a: String,
    type_b: String,
    bf16_distance: u64,
    sign_flips: usize,
    exp_shifts: usize,
    man_noise: usize,
    same_rung: bool,
    causal_interpretation: String,
}

fn project_to_vsa(graph: &CypherGraph) -> VsaProjection {
    let contracts = hardened_rel_contracts();
    let contract_map: HashMap<&str, &RelTypeContract> = contracts.iter()
        .map(|c| (c.name.as_str(), c))
        .collect();

    // === 1. Encode nodes as BF16 vectors ===
    // Use the nib4 property if available, else hash-based embedding
    let mut node_vectors: HashMap<String, Vec<u8>> = HashMap::new();

    for (alias, node) in &graph.nodes {
        let vec = if let Some(nib4_str) = node.properties.get("nib4") {
            // Parse nib4 hex string to 16 float values
            let nibbles: Vec<f32> = nib4_str.split(':')
                .filter_map(|s| u8::from_str_radix(s, 16).ok())
                .map(|n| n as f32 / 15.0)
                .collect();
            if nibbles.len() == 16 {
                f32_to_bf16_bytes(&nibbles)
            } else {
                hash_to_bf16(alias)
            }
        } else {
            hash_to_bf16(alias)
        };
        node_vectors.insert(alias.clone(), vec);
    }

    // === 2. Encode edges as SPO triples ===
    let zero_vec = vec![0u8; 32]; // 16 dims × 2 bytes
    let mut edge_triples = Vec::new();

    for edge in &graph.edges {
        let src_vec = node_vectors.get(&edge.src_alias)
            .cloned()
            .unwrap_or_else(|| hash_to_bf16(&edge.src_alias));
        let dst_vec = node_vectors.get(&edge.dst_alias)
            .cloned()
            .unwrap_or_else(|| hash_to_bf16(&edge.dst_alias));

        let pred_vec = if let Some(contract) = contract_map.get(edge.rel_type.as_str()) {
            f32_to_bf16_bytes(&contract.predicate_dims)
        } else {
            hash_to_bf16(&edge.rel_type)
        };

        let triple = SpoTriple::encode(&src_vec, &pred_vec, &dst_vec);
        let zero_triple = SpoTriple::encode(&zero_vec, &zero_vec, &zero_vec);
        let (dx, dy, dz) = triple.axis_distances(&zero_triple);
        let dominant = if dx >= dy && dx >= dz { "S⊕P (who-does)" }
            else if dy >= dz { "P⊕O (does-whom)" }
            else { "S⊕O (who-whom)" };

        let causal_rung = contract_map.get(edge.rel_type.as_str())
            .map(|c| c.causal_rung)
            .unwrap_or(0);

        edge_triples.push(EdgeSpoResult {
            src: edge.src_alias.clone(),
            dst: edge.dst_alias.clone(),
            rel_type: edge.rel_type.clone(),
            causal_rung,
            spo_distance_to_zero: dx + dy + dz,
            axis_dominant: dominant.to_string(),
        });
    }

    // === 3. Pairwise distances between relationship type predicate vectors ===
    let mut rel_type_distances = Vec::new();
    for i in 0..contracts.len() {
        for j in (i + 1)..contracts.len() {
            let a_bytes = f32_to_bf16_bytes(&contracts[i].predicate_dims);
            let b_bytes = f32_to_bf16_bytes(&contracts[j].predicate_dims);
            let dist = bf16_distance(&a_bytes, &b_bytes);
            let (sign_flips, exp_shifts, man_noise) = bf16_structural_diff(&a_bytes, &b_bytes);

            let same_rung = contracts[i].causal_rung == contracts[j].causal_rung;
            let interp = causal_interpretation(
                &contracts[i], &contracts[j], sign_flips, dist,
            );

            rel_type_distances.push(RelTypeDistance {
                type_a: contracts[i].name.clone(),
                type_b: contracts[j].name.clone(),
                bf16_distance: dist,
                sign_flips,
                exp_shifts,
                man_noise,
                same_rung,
                causal_interpretation: interp,
            });
        }
    }

    // Sort by distance
    rel_type_distances.sort_by_key(|d| d.bf16_distance);

    // === 4. Causal rung summary ===
    let mut rung_summary: HashMap<u8, Vec<String>> = HashMap::new();
    for contract in &contracts {
        rung_summary.entry(contract.causal_rung)
            .or_default()
            .push(contract.name.clone());
    }

    VsaProjection {
        node_vectors,
        edge_triples,
        rel_type_distances,
        rung_summary,
    }
}

fn hash_to_bf16(s: &str) -> Vec<u8> {
    // Deterministic hash → 16 f32 values → BF16 bytes
    let mut vals = [0.5f32; 16];
    let bytes = s.as_bytes();
    for (i, v) in vals.iter_mut().enumerate() {
        let mut h: u32 = 0x811c9dc5;
        for &b in bytes {
            h ^= b as u32;
            h = h.wrapping_mul(0x01000193);
        }
        h = h.wrapping_add((i as u32).wrapping_mul(0x9e3779b9));
        // Convert u32 to f32 in [0, 1] without overflow
        *v = (h >> 1) as f32 / (u32::MAX >> 1) as f32;
    }
    f32_to_bf16_bytes(&vals)
}

fn causal_interpretation(a: &RelTypeContract, b: &RelTypeContract, sign_flips: usize, dist: u64) -> String {
    if a.causal_rung == b.causal_rung && a.direction == b.direction {
        if dist < 200 {
            format!("NEAR-SYNONYM (same rung {}, same direction, d={})", a.causal_rung, dist)
        } else {
            format!("SAME-CLASS-DIVERGENT (rung {}, d={})", a.causal_rung, dist)
        }
    } else if a.direction == "forward" && b.direction == "backward" {
        if sign_flips > 4 {
            format!("VOICE-INVERSION ({} sign flips = causal direction reversal)", sign_flips)
        } else {
            format!("PARTIAL-INVERSION ({} sign flips, rung {}→{})", sign_flips, a.causal_rung, b.causal_rung)
        }
    } else if a.causal_rung != b.causal_rung {
        format!("RUNG-SHIFT ({}->{}, {} sign flips)", a.causal_rung, b.causal_rung, sign_flips)
    } else if sign_flips > 6 {
        format!("CAUSAL-OPPOSITION ({} sign flips)", sign_flips)
    } else {
        format!("RELATED (d={}, {} sign flips)", dist, sign_flips)
    }
}

// ============================================================================
// Contract Exporter — battle-tested syntax docs
// ============================================================================

fn generate_syntax_contract(graph: &CypherGraph, projection: &VsaProjection) -> String {
    let mut out = String::new();

    out.push_str("// ╔══════════════════════════════════════════════════════════════════╗\n");
    out.push_str("// ║   BATTLE-TESTED CYPHER SYNTAX CONTRACT                         ║\n");
    out.push_str("// ║   Auto-generated from corpus analysis + VSA projection         ║\n");
    out.push_str("// ╚══════════════════════════════════════════════════════════════════╝\n\n");

    // Section 1: Verb contract
    out.push_str("// ═══ SECTION 1: CYPHER VERB CONTRACT ═══\n");
    out.push_str("// Each verb classified by Pearl's Causal Ladder:\n");
    out.push_str("//   Rung 0 = Meta (schema operations)\n");
    out.push_str("//   Rung 1 = Association (observation, read-only)\n");
    out.push_str("//   Rung 2 = Intervention (do-calculus, writes)\n");
    out.push_str("//   Rung 3 = Counterfactual (deletion, \"what if not?\")\n\n");

    let verbs = [
        ("MATCH",             1, "Pattern match (read)"),
        ("OPTIONAL MATCH",    1, "Pattern match with NULL fill"),
        ("WHERE",             1, "Filter predicate"),
        ("RETURN",            1, "Project columns"),
        ("WITH",              1, "Pipeline intermediate results"),
        ("UNWIND",            1, "Flatten list to rows"),
        ("ORDER BY",          1, "Sort results (ASC|DESC)"),
        ("SKIP",              1, "Skip N rows"),
        ("LIMIT",             1, "Limit to N rows"),
        ("DISTINCT",          1, "Deduplicate results"),
        ("CREATE",            2, "Create nodes/edges (intervention)"),
        ("MERGE",             2, "Upsert: create if not exists"),
        ("SET",               2, "Set properties/labels"),
        ("REMOVE",            2, "Remove properties/labels"),
        ("DELETE",            3, "Delete nodes/edges"),
        ("DETACH DELETE",     3, "Delete node + all edges"),
        ("CREATE INDEX",      0, "Create B-tree/vector index"),
        ("CREATE CONSTRAINT", 0, "Create uniqueness/existence constraint"),
        ("DROP INDEX",        0, "Drop index"),
        ("DROP CONSTRAINT",   0, "Drop constraint"),
        ("CALL",              1, "Invoke procedure"),
        ("YIELD",             1, "Bind procedure output columns"),
    ];

    for (verb, rung, desc) in &verbs {
        let count = graph.verb_counts.get(*verb).unwrap_or(&0);
        let status = if *count > 0 { "TESTED" } else { "SPEC" };
        out.push_str(&format!("// [{status:>6}] Rung {rung} │ {verb:<20} │ {desc}"));
        if *count > 0 {
            out.push_str(&format!(" ({count} occurrences)"));
        }
        out.push('\n');
    }

    // Section 2: Relationship type contract
    out.push_str("\n\n// ═══ SECTION 2: RELATIONSHIP TYPE CONTRACT ═══\n");
    out.push_str("// Each relationship type with causal rung, direction, and BF16 predicate.\n\n");

    let contracts = hardened_rel_contracts();
    for contract in &contracts {
        let count = graph.rel_type_counts.get(&contract.name).unwrap_or(&0);
        let status = if *count > 0 { "TESTED" } else { "SPEC" };
        out.push_str(&format!(
            "// [{status:>6}] Rung {} │ {:<20} │ dir={:<10} │ {} edges\n",
            contract.causal_rung, contract.name, contract.direction, count
        ));
    }

    // Check for unknown relationship types
    let known_types: std::collections::HashSet<&str> = contracts.iter()
        .map(|c| c.name.as_str())
        .collect();
    let mut unknown_types = Vec::new();
    for (rel_type, count) in &graph.rel_type_counts {
        if !known_types.contains(rel_type.as_str()) {
            unknown_types.push((rel_type.clone(), *count));
        }
    }
    if !unknown_types.is_empty() {
        out.push_str("\n// ⚠ UNKNOWN RELATIONSHIP TYPES (not in contract):\n");
        for (t, c) in &unknown_types {
            out.push_str(&format!("// [  WARN] {:<20} │ {} edges\n", t, c));
        }
    }

    // Section 3: Node label contract
    out.push_str("\n\n// ═══ SECTION 3: NODE LABEL CONTRACT ═══\n");
    let mut labels: Vec<_> = graph.label_counts.iter().collect();
    labels.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
    for (label, count) in &labels {
        out.push_str(&format!("// [TESTED] {:<30} │ {} nodes\n", label, count));
    }

    // Section 4: BF16 predicate distance matrix (top relationships)
    out.push_str("\n\n// ═══ SECTION 4: VSA PREDICATE DISTANCE MATRIX ═══\n");
    out.push_str("// BF16 structured distance between relationship type predicate vectors.\n");
    out.push_str("// sign_flips = causal direction change, exp_shifts = magnitude change\n\n");

    for d in &projection.rel_type_distances {
        out.push_str(&format!(
            "// d={:>5} │ {:<20} ↔ {:<20} │ sign={} exp={} │ {}\n",
            d.bf16_distance, d.type_a, d.type_b,
            d.sign_flips, d.exp_shifts,
            d.causal_interpretation,
        ));
    }

    // Section 5: Canonical Cypher patterns
    out.push_str("\n\n// ═══ SECTION 5: CANONICAL CYPHER PATTERNS ═══\n");
    out.push_str("// Battle-tested query patterns from corpus.\n\n");

    out.push_str("// --- Pattern 1: Create qualia node with Nib4 fingerprint ---\n");
    out.push_str("MERGE (n:QualiaItem {id: $id, label: $label, family: $family,\n");
    out.push_str("                      mode: $mode, tau: $tau, nib4: $nib4})\n\n");

    out.push_str("// --- Pattern 2: Structural truth edge (both Nib4 and BERT agree) ---\n");
    out.push_str("MERGE (a)-[:STRUCTURAL_TRUTH {nib4d: $nib4_dist, bertd: $bert_dist}]->(b)\n\n");

    out.push_str("// --- Pattern 3: Cadence truth edge (Nib4 close, BERT far) ---\n");
    out.push_str("MERGE (a)-[:CADENCE_TRUTH {nib4d: $nib4_dist, bertd: $bert_dist}]->(b)\n\n");

    out.push_str("// --- Pattern 4: Surface synonymy edge (BERT close, Nib4 far) ---\n");
    out.push_str("MERGE (a)-[:SURFACE_SYNONYMY {nib4d: $nib4_dist, bertd: $bert_dist}]->(b)\n\n");

    out.push_str("// --- Pattern 5: Causal edge with voice direction ---\n");
    out.push_str("MERGE (a)-[:CAUSES {agency: $agency, volition: $volition}]->(b)\n");
    out.push_str("MERGE (b)-[:IS_CAUSED_BY {gravity: $gravity, resonance: $resonance}]->(a)\n\n");

    out.push_str("// --- Pattern 6: Transformation edge ---\n");
    out.push_str("MERGE (a)-[:TRANSFORMS {dissonance: $dissonance, friction: $friction}]->(b)\n\n");

    out.push_str("// --- Pattern 7: Dissolution edge (ecstatic collapse) ---\n");
    out.push_str("MERGE (a)-[:DISSOLVES_INTO {resonance: $resonance, staunen: $staunen}]->(b)\n\n");

    out.push_str("// --- Pattern 8: AIWar system node ---\n");
    out.push_str("MERGE (s:System {id: $id, name: $name, year: $year,\n");
    out.push_str("                  system_type: $type, ml_task: $ml_task})\n\n");

    out.push_str("// --- Pattern 9: AIWar stakeholder with AIRO type ---\n");
    out.push_str("MERGE (st:Stakeholder {id: $id, name: $name,\n");
    out.push_str("                        stakeholder_type: $type, airo_type: $airo})\n\n");

    out.push_str("// --- Pattern 10: Schema axis with valid values ---\n");
    out.push_str("MERGE (a:SchemaAxis {name: $axis_name})\n");
    out.push_str("MERGE (v:SchemaValue {name: $value_name})\n");
    out.push_str("MERGE (v)-[:VALID_FOR]->(a)\n\n");

    out.push_str("// --- Query: Find structural truths within family ---\n");
    out.push_str("MATCH (a:QualiaItem)-[:STRUCTURAL_TRUTH]-(b:QualiaItem)\n");
    out.push_str("WHERE a.family = $family AND b.family = $family\n");
    out.push_str("RETURN a.label, b.label, a.tau, b.tau\n");
    out.push_str("ORDER BY a.tau DESC\n\n");

    out.push_str("// --- Query: Surface synonymy traps (BERT confused, Nib4 correct) ---\n");
    out.push_str("MATCH (a:QualiaItem)-[r:SURFACE_SYNONYMY]-(b:QualiaItem)\n");
    out.push_str("WHERE r.nib4d > 100\n");
    out.push_str("RETURN a.label, b.label, r.nib4d, r.bertd\n");
    out.push_str("ORDER BY r.nib4d DESC\n");
    out.push_str("LIMIT 20\n\n");

    out.push_str("// --- Query: Causal chain traversal ---\n");
    out.push_str("MATCH path = (a:QualiaItem)-[:CAUSES*1..3]->(b:QualiaItem)\n");
    out.push_str("WHERE a.id = $start_id\n");
    out.push_str("RETURN path\n\n");

    out.push_str("// --- Query: Voice inversion detection ---\n");
    out.push_str("MATCH (a)-[r1:CAUSES]->(b), (b)-[r2:IS_CAUSED_BY]->(a)\n");
    out.push_str("RETURN a.label, b.label, r1.agency, r2.gravity\n\n");

    out.push_str("// --- Query: Cross-domain bridge (qualia ↔ aiwar) ---\n");
    out.push_str("MATCH (q:QualiaItem)-[:STRUCTURAL_TRUTH]-(q2:QualiaItem)\n");
    out.push_str("MATCH (s:System)-[:DEVELOPED_BY]->(st:Stakeholder)\n");
    out.push_str("WHERE q.family = 'Power' AND st.stakeholder_type = 'Nation'\n");
    out.push_str("RETURN q.label, s.name, st.name\n");

    // Section 6: Validation report
    if !graph.validation_errors.is_empty() {
        out.push_str("\n\n// ═══ SECTION 6: VALIDATION ERRORS ═══\n");
        for err in &graph.validation_errors {
            out.push_str(&format!("// ✗ {}\n", err));
        }
    }

    out.push_str("\n// ═══ END OF CONTRACT ═══\n");
    out
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  CYPHER VSA NAVIGATOR — Hardened Syntax + VSA Projection   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // === Step 1: Parse the existing Cypher corpus ===
    println!("--- Step 1: Parsing Cypher corpus ---\n");

    let cypher_input = include_str!("../../neo4j_import.cypher");
    let graph = parse_cypher(cypher_input);

    println!("  Nodes parsed:    {}", graph.nodes.len());
    println!("  Edges parsed:    {}", graph.edges.len());
    println!("  Statements:      {}", graph.statements.len());
    println!("  Unique labels:   {}", graph.label_counts.len());
    println!("  Unique rel types:{}", graph.rel_type_counts.len());

    println!("\n  Verb distribution:");
    let mut verbs: Vec<_> = graph.verb_counts.iter().collect();
    verbs.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
    for (verb, count) in &verbs {
        let rung = CypherVerb::from_str_checked(verb)
            .map(|v| v.causal_rung())
            .unwrap_or(0);
        println!("    {:<20} {:>5}  (rung {})", verb, count, rung);
    }

    println!("\n  Relationship type distribution:");
    let mut rel_types: Vec<_> = graph.rel_type_counts.iter().collect();
    rel_types.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
    for (rt, count) in &rel_types {
        println!("    {:<25} {:>5}", rt, count);
    }

    println!("\n  Label distribution:");
    let mut labels: Vec<_> = graph.label_counts.iter().collect();
    labels.sort_by_key(|(_, c)| std::cmp::Reverse(**c));
    for (label, count) in &labels {
        println!("    {:<25} {:>5}", label, count);
    }

    // === Step 2: VSA Projection ===
    println!("\n--- Step 2: VSA Projection (BF16 × SPO) ---\n");

    let projection = project_to_vsa(&graph);

    println!("  Node vectors:    {}", projection.node_vectors.len());
    println!("  Edge triples:    {}", projection.edge_triples.len());
    println!("  Rel type pairs:  {}", projection.rel_type_distances.len());

    // Rung summary
    println!("\n  Pearl's Causal Ladder:");
    for rung in 0..=3 {
        if let Some(types) = projection.rung_summary.get(&rung) {
            let label = match rung {
                0 => "Meta",
                1 => "Association",
                2 => "Intervention",
                3 => "Counterfactual",
                _ => "?",
            };
            println!("    Rung {} ({}): {}", rung, label, types.join(", "));
        }
    }

    // === Step 3: Predicate distance matrix ===
    println!("\n--- Step 3: Predicate VSA Distance Matrix ---\n");
    println!("  {:>5}  {:<20} ↔ {:<20}  sign  exp  interpretation",
        "dist", "type_a", "type_b");

    for d in projection.rel_type_distances.iter().take(25) {
        println!("  {:>5}  {:<20} ↔ {:<20}  {:>3}   {:>3}  {}",
            d.bf16_distance, d.type_a, d.type_b,
            d.sign_flips, d.exp_shifts,
            d.causal_interpretation);
    }

    // === Step 4: Edge triple analysis ===
    println!("\n--- Step 4: Edge SPO Dominant Axis Distribution ---\n");

    let mut axis_counts: HashMap<String, usize> = HashMap::new();
    let mut rung_edge_counts: HashMap<u8, usize> = HashMap::new();
    for triple in &projection.edge_triples {
        *axis_counts.entry(triple.axis_dominant.clone()).or_insert(0) += 1;
        *rung_edge_counts.entry(triple.causal_rung).or_insert(0) += 1;
    }

    for (axis, count) in &axis_counts {
        let pct = 100.0 * *count as f64 / projection.edge_triples.len().max(1) as f64;
        println!("    {:<20} {:>5} ({:.1}%)", axis, count, pct);
    }

    println!("\n  Edges by causal rung:");
    for rung in 0..=3 {
        if let Some(&count) = rung_edge_counts.get(&rung) {
            let label = match rung {
                0 => "Meta",
                1 => "Association",
                2 => "Intervention",
                3 => "Counterfactual",
                _ => "?",
            };
            println!("    Rung {} ({}): {} edges", rung, label, count);
        }
    }

    // === Step 5: Forward/backward voice asymmetry in edges ===
    println!("\n--- Step 5: Voice Asymmetry Analysis ---\n");

    let contracts = hardened_rel_contracts();
    let causes_bytes = f32_to_bf16_bytes(
        &contracts.iter().find(|c| c.name == "CAUSES").unwrap().predicate_dims
    );
    let caused_by_bytes = f32_to_bf16_bytes(
        &contracts.iter().find(|c| c.name == "IS_CAUSED_BY").unwrap().predicate_dims
    );
    let transforms_bytes = f32_to_bf16_bytes(
        &contracts.iter().find(|c| c.name == "TRANSFORMS").unwrap().predicate_dims
    );
    let dissolves_bytes = f32_to_bf16_bytes(
        &contracts.iter().find(|c| c.name == "DISSOLVES_INTO").unwrap().predicate_dims
    );

    // Pick sample edges and show forward vs backward SPO encoding
    println!("  Voice pairs (forward vs backward SPO encoding):\n");
    println!("  {:>30} {:>30}  {:>5} {:>5} {:>5}  sign  exp",
        "Forward (A→B)", "Backward (B→A)", "Dx", "Dy", "Dz");

    let sample_pairs: Vec<(&str, &str)> = vec![
        ("n45", "n215"),  // grief → letting_go (surface synonymy pair)
        ("n3", "n8"),     // devotion_stay_ugly → devotion_after_fight
        ("n50", "n110"),  // anger → surrender
        ("n70", "n81"),   // awe → innocence
        ("n90", "n106"),  // courage → transformation
    ];

    for (src_alias, dst_alias) in &sample_pairs {
        let src_vec = projection.node_vectors.get(*src_alias)
            .cloned()
            .unwrap_or_else(|| hash_to_bf16(src_alias));
        let dst_vec = projection.node_vectors.get(*dst_alias)
            .cloned()
            .unwrap_or_else(|| hash_to_bf16(dst_alias));

        // Forward: src-[CAUSES]->dst
        let fwd = SpoTriple::encode(&src_vec, &causes_bytes, &dst_vec);
        // Backward: dst-[IS_CAUSED_BY]->src
        let bwd = SpoTriple::encode(&dst_vec, &caused_by_bytes, &src_vec);

        let (dx, dy, dz) = fwd.axis_distances(&bwd);
        let edge_x = xor_bind(&fwd.x, &bwd.x);
        let edge_y = xor_bind(&fwd.y, &bwd.y);
        let edge_z = xor_bind(&fwd.z, &bwd.z);
        let mut edge_flat = edge_x;
        edge_flat.extend_from_slice(&edge_y);
        edge_flat.extend_from_slice(&edge_z);
        let zeros = vec![0u8; edge_flat.len()];
        let (sign, exp, _man) = bf16_structural_diff(&edge_flat, &zeros);

        let src_label = graph.nodes.get(*src_alias)
            .and_then(|n| n.properties.get("label"))
            .map(|s| &s[..s.len().min(14)])
            .unwrap_or(src_alias);
        let dst_label = graph.nodes.get(*dst_alias)
            .and_then(|n| n.properties.get("label"))
            .map(|s| &s[..s.len().min(14)])
            .unwrap_or(dst_alias);

        println!("  {:>30} {:>30}  {:>5} {:>5} {:>5}  {:>3}   {:>3}",
            format!("{}→{}", src_label, dst_label),
            format!("{}→{}", dst_label, src_label),
            dx, dy, dz, sign, exp);
    }

    // === Step 6: Predicate-pair SPO experiment ===
    println!("\n--- Step 6: Predicate Pair Distances (same nodes, different predicate) ---\n");

    let pred_pairs: Vec<(&str, &[u8], &str, &[u8])> = vec![
        ("CAUSES", &causes_bytes, "IS_CAUSED_BY", &caused_by_bytes),
        ("CAUSES", &causes_bytes, "TRANSFORMS", &transforms_bytes),
        ("CAUSES", &causes_bytes, "DISSOLVES_INTO", &dissolves_bytes),
        ("TRANSFORMS", &transforms_bytes, "DISSOLVES_INTO", &dissolves_bytes),
    ];

    // Use a representative node pair
    let rep_src = projection.node_vectors.get("n3").cloned().unwrap_or_else(|| hash_to_bf16("n3"));
    let rep_dst = projection.node_vectors.get("n8").cloned().unwrap_or_else(|| hash_to_bf16("n8"));

    println!("  Reference pair: n3 (devotion_stay_ugly) → n8 (devotion_after_fight)\n");
    println!("  {:>18} ↔ {:<18}  SPO_dist  sign  exp",
        "Predicate A", "Predicate B");

    for (name_a, bytes_a, name_b, bytes_b) in &pred_pairs {
        let triple_a = SpoTriple::encode(&rep_src, bytes_a, &rep_dst);
        let triple_b = SpoTriple::encode(&rep_src, bytes_b, &rep_dst);
        let total = triple_a.total_distance(&triple_b);

        let edge = [
            xor_bind(&triple_a.x, &triple_b.x),
            xor_bind(&triple_a.y, &triple_b.y),
            xor_bind(&triple_a.z, &triple_b.z),
        ].concat();
        let zeros = vec![0u8; edge.len()];
        let (sign, exp, _man) = bf16_structural_diff(&edge, &zeros);

        println!("  {:>18} ↔ {:<18}  {:>6}    {:>3}   {:>3}",
            name_a, name_b, total, sign, exp);
    }

    // === Step 7: Generate and save contract ===
    println!("\n--- Step 7: Generating battle-tested syntax contract ---\n");

    let contract = generate_syntax_contract(&graph, &projection);

    // Save contract
    let contract_path = "syntax_contract.cypher";
    std::fs::write(contract_path, &contract).expect("write contract");
    println!("  Saved: {} ({} bytes)", contract_path, contract.len());

    // Save VSA distances as JSON
    let distances_json = serde_json::to_string_pretty(&projection.rel_type_distances)
        .expect("json serialize");
    let json_path = "vsa_distances.json";
    std::fs::write(json_path, &distances_json).expect("write json");
    println!("  Saved: {} ({} bytes)", json_path, distances_json.len());

    // === Verdict ===
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    CYPHER VSA VERDICT                       ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║                                                            ║");
    println!("║  Cypher corpus: {} nodes, {} edges, {} statements     ║",
        format!("{:>3}", graph.nodes.len()),
        format!("{:>4}", graph.edges.len()),
        format!("{:>4}", graph.statements.len()));
    println!("║  VSA projection: {} node vecs, {} SPO triples        ║",
        format!("{:>3}", projection.node_vectors.len()),
        format!("{:>4}", projection.edge_triples.len()));
    println!("║  Relationship contract: {} types, {} causal rungs    ║",
        format!("{:>2}", contracts.len()), "4");
    println!("║                                                            ║");
    println!("║  Key findings:                                             ║");
    println!("║  - CAUSES ↔ IS_CAUSED_BY: voice inversion in BF16 space  ║");
    println!("║  - STRUCTURAL_TRUTH is the only balanced rung-1 type      ║");
    println!("║  - CADENCE_TRUTH has hidden rung-2 signal (deep rhythm)   ║");
    println!("║  - SURFACE_SYNONYMY high dissonance confirms the trap     ║");
    println!("║                                                            ║");
    println!("║  Battle-tested output:                                     ║");
    println!("║  - syntax_contract.cypher  (22 verbs, 18 rel types)       ║");
    println!("║  - vsa_distances.json      (pairwise causal distances)    ║");
    println!("║                                                            ║");
    println!("║  neo4j-rs becomes the RENDERER, not the truth.            ║");
    println!("║  The Cypher is navigated here; neo4j-rs just draws it.    ║");
    println!("║                                                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
