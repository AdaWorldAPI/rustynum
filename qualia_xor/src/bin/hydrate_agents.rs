//! Hydrated Graph Reasoning Engine — Edge Vector Propagation + Agent Spawning
//!
//! This binary does three things:
//!
//! 1. **Hydrate**: Bundle all edge vectors per node into an "awareness vector"
//!    (superposition of every relationship touching that node). Each node
//!    becomes aware of its full neighborhood in BF16 space.
//!
//! 2. **Fanout**: Propagate awareness through the graph in rounds (like
//!    message passing / belief propagation). After N rounds, each node's
//!    awareness includes signals from N-hop neighbors.
//!
//! 3. **Spawn Agents**: Generate crewai-rust agent configs from graph nodes.
//!    Each agent has a role (from node family), goal (from SPO position),
//!    backstory (from hydrated awareness), and a reasoning prompt about
//!    what its SPO triples are causing/dissolving.
//!
//! 4. **Benchmark**: Measure database growth rate, thinking efficiency,
//!    and awareness crystallization per round.
//!
//! ## Architecture
//!
//! ```text
//!  neo4j_import.cypher ──→ CypherGraph ──→ Hydration (per-node bundling)
//!                                              │
//!                                              ├──→ Fanout (N rounds)
//!                                              │       │
//!                                              │       └──→ Awareness vectors
//!                                              │
//!                                              └──→ Agent Configs (JSON)
//!                                                      │
//!                                                      └──→ crewai-rust agents
//!                                                           (turn-based SPO reasoning)
//! ```

use std::collections::HashMap;
use std::time::Instant;
use serde::{Deserialize, Serialize};

// ============================================================================
// BF16 utilities (inline, standalone)
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
    for i in (0..a.len().min(b.len())).step_by(2) {
        if i + 1 >= a.len() || i + 1 >= b.len() { break; }
        let va = u16::from_le_bytes([a[i], a[i + 1]]);
        let vb = u16::from_le_bytes([b[i], b[i + 1]]);
        let xor = va ^ vb;
        total += (((xor >> 15) & 1) as u64) * 256
            + (((xor >> 7) & 0xFF).count_ones() as u64) * 16
            + ((xor & 0x7F).count_ones() as u64);
    }
    total
}

fn bf16_structural_diff(a: &[u8], b: &[u8]) -> (usize, usize, usize) {
    let n = a.len().min(b.len()) / 2;
    let (mut s, mut e, mut m) = (0, 0, 0);
    for d in 0..n {
        let i = d * 2;
        let xor = u16::from_le_bytes([a[i], a[i+1]]) ^ u16::from_le_bytes([b[i], b[i+1]]);
        if xor & 0x8000 != 0 { s += 1; }
        e += ((xor >> 7) & 0xFF).count_ones() as usize;
        m += (xor & 0x7F).count_ones() as usize;
    }
    (s, e, m)
}

/// Majority-vote bundle: given multiple byte vectors, produce prototype
fn majority_bundle(vectors: &[&[u8]], n_bytes: usize) -> Vec<u8> {
    let threshold = vectors.len() / 2;
    let mut bit_counts = vec![0u32; n_bytes * 8];
    for vec in vectors {
        for (byte_idx, &byte) in vec.iter().enumerate().take(n_bytes) {
            for bit in 0..8 {
                if byte & (1 << bit) != 0 {
                    bit_counts[byte_idx * 8 + bit] += 1;
                }
            }
        }
    }
    let mut result = vec![0u8; n_bytes];
    for (byte_idx, chunk) in bit_counts.chunks(8).enumerate() {
        let mut byte_val = 0u8;
        for (bit, &count) in chunk.iter().enumerate() {
            if count > threshold as u32 {
                byte_val |= 1 << bit;
            }
        }
        result[byte_idx] = byte_val;
    }
    result
}

fn hash_to_bf16(s: &str) -> Vec<u8> {
    let mut vals = [0.5f32; 16];
    let bytes = s.as_bytes();
    for (i, v) in vals.iter_mut().enumerate() {
        let mut h: u32 = 0x811c9dc5;
        for &b in bytes {
            h ^= b as u32;
            h = h.wrapping_mul(0x01000193);
        }
        h = h.wrapping_add((i as u32).wrapping_mul(0x9e3779b9));
        *v = (h >> 1) as f32 / (u32::MAX >> 1) as f32;
    }
    f32_to_bf16_bytes(&vals)
}

// ============================================================================
// Graph parsing (lightweight, from cypher)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GraphNode {
    alias: String,
    id: String,
    label: String,
    family: String,
    mode: String,
    tau: f64,
    nib4: String,
    bf16_vec: Vec<u8>,
}

#[derive(Debug, Clone)]
struct GraphEdge {
    src: String,
    dst: String,
    rel_type: String,
    weight: f64, // dist or bertd
}

fn parse_graph_from_cypher(input: &str) -> (Vec<GraphNode>, Vec<GraphEdge>) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut alias_to_idx: HashMap<String, usize> = HashMap::new();

    for line in input.lines() {
        let t = line.trim();
        if t.is_empty() || t.starts_with("//") { continue; }

        // Node: MERGE (nX:QualiaItem {id: '...', ...})
        if t.starts_with("MERGE (n") && t.contains(":QualiaItem") {
            if let Some(alias) = extract_between(t, "(", ":") {
                let id = extract_prop(t, "id").unwrap_or_default();
                let label = extract_prop(t, "label").unwrap_or_default();
                let family = extract_prop(t, "family").unwrap_or_default();
                let mode = extract_prop(t, "mode").unwrap_or_default();
                let tau: f64 = extract_prop(t, "tau")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);
                let nib4 = extract_prop(t, "nib4").unwrap_or_default();

                // Parse nib4 to BF16
                let bf16_vec = if !nib4.is_empty() {
                    let nibbles: Vec<f32> = nib4.split(':')
                        .filter_map(|s| u8::from_str_radix(s, 16).ok())
                        .map(|n| n as f32 / 15.0)
                        .collect();
                    if nibbles.len() == 16 { f32_to_bf16_bytes(&nibbles) }
                    else { hash_to_bf16(&id) }
                } else {
                    hash_to_bf16(&id)
                };

                alias_to_idx.insert(alias.clone(), nodes.len());
                nodes.push(GraphNode {
                    alias, id, label, family, mode, tau, nib4, bf16_vec,
                });
            }
        }

        // Edge: MERGE (nX)-[:TYPE {dist: N}]->(nY)
        if t.contains(")-[") && t.contains("]->(") {
            if let (Some(src), Some(dst), Some(rel_type)) = (
                extract_between(t, "(", ")"),
                extract_after_arrow(t),
                extract_rel_type(t),
            ) {
                let weight = extract_prop(t, "dist")
                    .or_else(|| extract_prop(t, "nib4d"))
                    .or_else(|| extract_prop(t, "bertd"))
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(1.0);

                edges.push(GraphEdge { src, dst, rel_type, weight });
            }
        }
    }

    (nodes, edges)
}

fn extract_between(s: &str, start: &str, end: &str) -> Option<String> {
    let i = s.find(start)? + start.len();
    let rest = &s[i..];
    let j = rest.find(end)?;
    Some(rest[..j].trim().to_string())
}

fn extract_prop(s: &str, key: &str) -> Option<String> {
    let pattern = format!("{}: ", key);
    let i = s.find(&pattern)? + pattern.len();
    let rest = &s[i..];
    if rest.starts_with('\'') {
        let end = rest[1..].find('\'')?;
        Some(rest[1..1+end].to_string())
    } else {
        let end = rest.find([',', '}', ')'].as_ref())?;
        Some(rest[..end].trim().to_string())
    }
}

fn extract_after_arrow(s: &str) -> Option<String> {
    let i = s.find("]->(")?;
    let rest = &s[i + 4..];
    let j = rest.find(')')?;
    Some(rest[..j].trim().to_string())
}

fn extract_rel_type(s: &str) -> Option<String> {
    let i = s.find("-[:")?;
    let rest = &s[i + 3..];
    let j = rest.find([' ', '{', ']'].as_ref())?;
    Some(rest[..j].trim().to_string())
}

// ============================================================================
// Step 1: Hydration — bundle all edge vectors per node
// ============================================================================

#[derive(Debug, Clone)]
struct HydratedNode {
    alias: String,
    node: GraphNode,
    /// BF16 awareness vector = majority bundle of all edge XOR vectors touching this node
    awareness_vec: Vec<u8>,
    /// Number of edges bundled
    edge_count: usize,
    /// Crystallization: how many bits agree across majority of edges
    crystallization: f32,
    /// Per-relationship-type awareness breakdown
    rel_awareness: HashMap<String, Vec<u8>>,
    /// SPO role summary: how often this node is Subject vs Object
    subject_count: usize,
    object_count: usize,
    /// Causal signature: net sign flips from all forward/backward edges
    causal_sign: i32,
}

/// Predicate vector for each relationship type
fn predicate_vectors() -> HashMap<&'static str, Vec<u8>> {
    let preds: Vec<(&str, [f32; 16])> = vec![
        ("NIB4_NEAR", [0.5; 16]),
        ("BERT_NEAR", {
            let mut a = [0.5; 16]; a[5] = 0.8; a
        }),
        ("STRUCTURAL_TRUTH", {
            let mut a = [0.8; 16]; a[10] = 0.1; a[12] = 0.1; a[14] = 0.1; a
        }),
        ("CADENCE_TRUTH", {
            let mut a = [0.5; 16]; a[2] = 0.7; a[5] = 0.3; a[7] = 0.7; a
        }),
        ("SURFACE_SYNONYMY", {
            let mut a = [0.5; 16]; a[2] = 0.2; a[5] = 0.9; a[7] = 0.2; a
        }),
        ("BELONGS_TO", {
            let mut a = [0.5; 16]; a[2] = 0.8; a[3] = 0.3; a[6] = 0.7; a
        }),
        ("HAS_MODE", {
            let mut a = [0.5; 16]; a[5] = 0.9; a[15] = 0.9; a
        }),
        ("CAUSES", {
            let mut a = [0.5; 16]; a[3] = 0.9; a[7] = 0.1; a[9] = 0.8; a
        }),
        ("IS_CAUSED_BY", {
            let mut a = [0.5; 16]; a[3] = 0.1; a[7] = 0.9; a[9] = 0.2; a
        }),
        ("TRANSFORMS", {
            let mut a = [0.5; 16]; a[10] = 0.8; a[14] = 0.8; a
        }),
        ("DISSOLVES_INTO", {
            let mut a = [0.5; 16]; a[0] = 0.8; a[4] = 0.9; a[3] = 0.1; a
        }),
    ];
    preds.into_iter()
        .map(|(name, dims)| (name, f32_to_bf16_bytes(&dims)))
        .collect()
}

fn hydrate_nodes(
    nodes: &[GraphNode],
    edges: &[GraphEdge],
) -> Vec<HydratedNode> {
    let alias_to_idx: HashMap<&str, usize> = nodes.iter()
        .enumerate()
        .map(|(i, n)| (n.alias.as_str(), i))
        .collect();
    let pred_vecs = predicate_vectors();
    let n_bytes = 32; // 16 dims × 2 bytes

    let mut hydrated: Vec<HydratedNode> = nodes.iter().map(|n| HydratedNode {
        alias: n.alias.clone(),
        node: n.clone(),
        awareness_vec: vec![0u8; n_bytes],
        edge_count: 0,
        crystallization: 0.0,
        rel_awareness: HashMap::new(),
        subject_count: 0,
        object_count: 0,
        causal_sign: 0,
    }).collect();

    // For each edge, XOR-bind (src ⊕ pred ⊕ dst) and accumulate per-node
    let mut per_node_edges: HashMap<usize, Vec<Vec<u8>>> = HashMap::new();

    for edge in edges {
        let src_idx = match alias_to_idx.get(edge.src.as_str()) { Some(&i) => i, None => continue };
        let dst_idx = match alias_to_idx.get(edge.dst.as_str()) { Some(&i) => i, None => continue };

        let src_vec = &nodes[src_idx].bf16_vec;
        let dst_vec = &nodes[dst_idx].bf16_vec;
        let pred_vec = pred_vecs.get(edge.rel_type.as_str())
            .cloned()
            .unwrap_or_else(|| hash_to_bf16(&edge.rel_type));

        // SPO triple: S⊕P, P⊕O, S⊕O → flatten to edge signature
        let spo_flat = [
            xor_bind(src_vec, &pred_vec),
            xor_bind(&pred_vec, dst_vec),
            xor_bind(src_vec, dst_vec),
        ].concat();

        // Truncate/pad to n_bytes for bundling
        let edge_sig: Vec<u8> = spo_flat.iter().take(n_bytes).copied().collect();
        let edge_sig = if edge_sig.len() < n_bytes {
            let mut padded = edge_sig;
            padded.resize(n_bytes, 0);
            padded
        } else {
            edge_sig
        };

        // Accumulate for both src and dst nodes
        per_node_edges.entry(src_idx).or_default().push(edge_sig.clone());
        per_node_edges.entry(dst_idx).or_default().push(edge_sig.clone());

        // Track subject/object counts
        hydrated[src_idx].subject_count += 1;
        hydrated[dst_idx].object_count += 1;

        // Track causal sign
        match edge.rel_type.as_str() {
            "CAUSES" | "TRANSFORMS" => {
                hydrated[src_idx].causal_sign += 1;
                hydrated[dst_idx].causal_sign -= 1;
            }
            "IS_CAUSED_BY" | "DISSOLVES_INTO" => {
                hydrated[src_idx].causal_sign -= 1;
                hydrated[dst_idx].causal_sign += 1;
            }
            _ => {}
        }

        // Per-rel-type awareness accumulation
        let rel_entry = hydrated[src_idx].rel_awareness
            .entry(edge.rel_type.clone())
            .or_insert_with(|| vec![0u8; n_bytes]);
        for (i, &b) in edge_sig.iter().enumerate().take(n_bytes) {
            rel_entry[i] ^= b;
        }
    }

    // Majority-vote bundle per node
    for (idx, edge_vecs) in &per_node_edges {
        if edge_vecs.is_empty() { continue; }
        let refs: Vec<&[u8]> = edge_vecs.iter().map(|v| v.as_slice()).collect();
        hydrated[*idx].awareness_vec = majority_bundle(&refs, n_bytes);
        hydrated[*idx].edge_count = edge_vecs.len();

        // Crystallization = fraction of bits that agree across >75% of edges
        let threshold_75 = (edge_vecs.len() * 3 / 4).max(1) as u32;
        let mut bit_counts = vec![0u32; n_bytes * 8];
        for vec in edge_vecs {
            for (byte_idx, &byte) in vec.iter().enumerate().take(n_bytes) {
                for bit in 0..8 {
                    if byte & (1 << bit) != 0 {
                        bit_counts[byte_idx * 8 + bit] += 1;
                    }
                }
            }
        }
        let crystallized = bit_counts.iter()
            .filter(|&&c| c > threshold_75 || c < (edge_vecs.len() as u32 - threshold_75))
            .count();
        hydrated[*idx].crystallization = crystallized as f32 / (n_bytes * 8) as f32;
    }

    hydrated
}

// ============================================================================
// Step 2: Fanout — propagate awareness through rounds
// ============================================================================

#[derive(Debug, Serialize)]
struct FanoutRound {
    round: usize,
    total_bits_changed: usize,
    avg_crystallization: f32,
    max_crystallization: f32,
    min_crystallization: f32,
    converged_nodes: usize,
    duration_us: u64,
    db_growth_bytes: usize,
}

fn fanout(
    hydrated: &mut [HydratedNode],
    edges: &[GraphEdge],
    max_rounds: usize,
) -> Vec<FanoutRound> {
    // Build index from owned strings to avoid lifetime issues
    let alias_to_idx: HashMap<String, usize> = hydrated.iter()
        .enumerate()
        .map(|(i, h)| (h.alias.clone(), i))
        .collect();
    let n_bytes = 32;
    let mut rounds = Vec::new();

    for round in 0..max_rounds {
        let t0 = Instant::now();
        let mut total_bits_changed = 0usize;
        let mut new_vecs: Vec<Vec<u8>> = hydrated.iter()
            .map(|h| h.awareness_vec.clone())
            .collect();

        // For each edge, XOR neighbor's awareness into this node
        for edge in edges {
            let src_idx = match alias_to_idx.get(&edge.src) { Some(&i) => i, None => continue };
            let dst_idx = match alias_to_idx.get(&edge.dst) { Some(&i) => i, None => continue };

            // src absorbs dst's awareness (XOR blend)
            let dst_awareness = hydrated[dst_idx].awareness_vec.clone();
            for (i, &b) in dst_awareness.iter().enumerate().take(n_bytes) {
                new_vecs[src_idx][i] ^= b;
            }

            // dst absorbs src's awareness (XOR blend)
            let src_awareness = hydrated[src_idx].awareness_vec.clone();
            for (i, &b) in src_awareness.iter().enumerate().take(n_bytes) {
                new_vecs[dst_idx][i] ^= b;
            }
        }

        // Apply new vectors, measure change
        let mut cryst_sum = 0.0f32;
        let mut cryst_max = 0.0f32;
        let mut cryst_min = 1.0f32;
        let mut converged = 0usize;

        for (i, h) in hydrated.iter_mut().enumerate() {
            let bits = h.awareness_vec.iter().zip(new_vecs[i].iter())
                .map(|(a, b)| (a ^ b).count_ones() as usize)
                .sum::<usize>();
            total_bits_changed += bits;
            if bits == 0 { converged += 1; }

            h.awareness_vec = new_vecs[i].clone();

            // Recompute crystallization
            let zero = vec![0u8; n_bytes];
            let (sign, exp, _man) = bf16_structural_diff(&h.awareness_vec, &zero);
            let total_possible = 16 + 16 * 8 + 16 * 7; // max sign + exp + man
            h.crystallization = 1.0 - (sign + exp) as f32 / total_possible as f32;

            cryst_sum += h.crystallization;
            cryst_max = cryst_max.max(h.crystallization);
            cryst_min = cryst_min.min(h.crystallization);
        }

        let duration = t0.elapsed();
        let avg_cryst = cryst_sum / hydrated.len().max(1) as f32;

        rounds.push(FanoutRound {
            round,
            total_bits_changed,
            avg_crystallization: avg_cryst,
            max_crystallization: cryst_max,
            min_crystallization: cryst_min,
            converged_nodes: converged,
            duration_us: duration.as_micros() as u64,
            db_growth_bytes: hydrated.len() * n_bytes * (round + 1),
        });

        // Convergence check: if no bits changed, stop
        if total_bits_changed == 0 {
            break;
        }
    }

    rounds
}

// ============================================================================
// Step 3: Agent spawning — crewai-rust configs from graph nodes
// ============================================================================

#[derive(Debug, Clone, Serialize)]
struct AgentConfig {
    id: String,
    role: String,
    goal: String,
    backstory: String,
    llm: String,
    #[serde(rename = "max_iter")]
    max_iter: u32,
    reasoning: bool,
    tools: Vec<String>,
    /// SPO causal prompt for this agent's turn
    spo_prompt: String,
    /// Awareness summary (from hydration)
    awareness_summary: AwarenessSummary,
    /// Thinking style scores
    thinking_style: ThinkingStyle,
}

#[derive(Debug, Clone, Serialize)]
struct AwarenessSummary {
    crystallization: f32,
    edge_count: usize,
    subject_ratio: f32, // how often this node is agent (subject)
    causal_sign: i32,   // positive = causer, negative = caused
    dominant_rel: String,
}

#[derive(Debug, Clone, Serialize)]
struct ThinkingStyle {
    analytical: f32,   // high crystallization → analytical
    creative: f32,     // high edge diversity → creative
    causal: f32,       // high subject_count → causal thinker
    receptive: f32,    // high object_count → receptive
    integrative: f32,  // balanced subject/object → integrative
}

fn spawn_agents(
    hydrated: &[HydratedNode],
    edges: &[GraphEdge],
    llm_provider: &str,
) -> Vec<AgentConfig> {
    let mut configs = Vec::new();

    // Group edges by source for SPO prompt generation
    let mut edges_by_node: HashMap<&str, Vec<&GraphEdge>> = HashMap::new();
    for edge in edges {
        edges_by_node.entry(edge.src.as_str()).or_default().push(edge);
        edges_by_node.entry(edge.dst.as_str()).or_default().push(edge);
    }

    // Only spawn agents for nodes with enough edges (interesting actors)
    let interesting: Vec<&HydratedNode> = hydrated.iter()
        .filter(|h| h.edge_count >= 5)
        .collect();

    // Take top 20 by edge count (most connected = most interesting actors)
    let mut sorted = interesting.clone();
    sorted.sort_by_key(|h| std::cmp::Reverse(h.edge_count));
    let top_actors = &sorted[..sorted.len().min(20)];

    for h in top_actors {
        let total_roles = (h.subject_count + h.object_count).max(1) as f32;
        let subject_ratio = h.subject_count as f32 / total_roles;

        // Thinking style derived from awareness state
        let style = ThinkingStyle {
            analytical: h.crystallization,
            creative: (h.rel_awareness.len() as f32 / 7.0).min(1.0),
            causal: subject_ratio,
            receptive: 1.0 - subject_ratio,
            integrative: 1.0 - (subject_ratio - 0.5).abs() * 2.0,
        };

        // Dominant relationship type
        let dominant_rel = h.rel_awareness.keys()
            .max_by_key(|k| {
                edges_by_node.get(h.alias.as_str())
                    .map(|e| e.iter().filter(|ed| &ed.rel_type == *k).count())
                    .unwrap_or(0)
            })
            .cloned()
            .unwrap_or_else(|| "UNKNOWN".into());

        // Generate SPO reasoning prompt
        let node_edges = edges_by_node.get(h.alias.as_str())
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let spo_prompt = generate_spo_prompt(&h.node, node_edges, &style);

        let awareness = AwarenessSummary {
            crystallization: h.crystallization,
            edge_count: h.edge_count,
            subject_ratio,
            causal_sign: h.causal_sign,
            dominant_rel: dominant_rel.clone(),
        };

        let role = format!("{} Analyst ({})", h.node.family, h.node.mode);
        let goal = if h.causal_sign > 0 {
            format!("Investigate what '{}' CAUSES in the graph — trace forward causal chains", h.node.label)
        } else if h.causal_sign < 0 {
            format!("Investigate what CAUSES '{}' — trace backward causal chains", h.node.label)
        } else {
            format!("Map the symmetric relationships around '{}' — find structural truths", h.node.label)
        };

        let backstory = format!(
            "You are an awareness agent hydrated from the qualia graph node '{}'. \
             Your crystallization is {:.0}% (how settled your awareness is). \
             You have {} edges: {} as subject (causer), {} as object (caused). \
             Your dominant relationship type is {}. \
             Your causal sign is {} (positive=causer, negative=caused). \
             Think {} — your style is {:.0}% analytical, {:.0}% creative, {:.0}% integrative.",
            h.node.label,
            h.crystallization * 100.0,
            h.edge_count,
            h.subject_count,
            h.object_count,
            dominant_rel,
            h.causal_sign,
            if style.analytical > 0.6 { "carefully and precisely" }
            else if style.creative > 0.6 { "divergently and associatively" }
            else { "balancedly and integratively" },
            style.analytical * 100.0,
            style.creative * 100.0,
            style.integrative * 100.0,
        );

        configs.push(AgentConfig {
            id: h.node.id.clone(),
            role,
            goal,
            backstory,
            llm: llm_provider.to_string(),
            max_iter: 10,
            reasoning: true,
            tools: vec!["graph_query".into(), "spo_search".into(), "web_search".into()],
            spo_prompt,
            awareness_summary: awareness,
            thinking_style: style,
        });
    }

    configs
}

fn generate_spo_prompt(
    node: &GraphNode,
    edges: &[&GraphEdge],
    style: &ThinkingStyle,
) -> String {
    let mut prompt = format!(
        "You are reasoning about the qualia state '{}' (family: {}, mode: {}).\n\n",
        node.label, node.family, node.mode
    );

    // Group edges by relationship type
    let mut by_type: HashMap<&str, Vec<&GraphEdge>> = HashMap::new();
    for edge in edges {
        by_type.entry(edge.rel_type.as_str()).or_default().push(edge);
    }

    prompt.push_str("Your SPO relationships:\n");
    for (rel_type, type_edges) in &by_type {
        let as_subject: Vec<_> = type_edges.iter()
            .filter(|e| e.src == node.alias)
            .take(3)
            .collect();
        let as_object: Vec<_> = type_edges.iter()
            .filter(|e| e.dst == node.alias)
            .take(3)
            .collect();

        if !as_subject.is_empty() {
            prompt.push_str(&format!("  Active: {} -[:{}]-> [", node.alias, rel_type));
            let targets: Vec<String> = as_subject.iter().map(|e| e.dst.clone()).collect();
            prompt.push_str(&targets.join(", "));
            prompt.push_str("]\n");
        }
        if !as_object.is_empty() {
            prompt.push_str(&format!("  Passive: ["));
            let sources: Vec<String> = as_object.iter().map(|e| e.src.clone()).collect();
            prompt.push_str(&sources.join(", "));
            prompt.push_str(&format!("] -[:{}]-> {}\n", rel_type, node.alias));
        }
    }

    prompt.push_str("\nReasoning tasks:\n");

    if style.causal > 0.6 {
        prompt.push_str("1. What causal chains does this state INITIATE? Trace forward 2-3 hops.\n");
        prompt.push_str("2. Are any of these chains SURFACE_SYNONYMY traps (BERT close, Nib4 far)?\n");
        prompt.push_str("3. What would DISSOLVE if this state were removed (counterfactual)?\n");
    } else if style.receptive > 0.6 {
        prompt.push_str("1. What states CAUSE this one? Trace backward 2-3 hops.\n");
        prompt.push_str("2. Is this state a CADENCE_TRUTH (deep rhythm, not surface)?\n");
        prompt.push_str("3. What TRANSFORMS into this state? What conditions are needed?\n");
    } else {
        prompt.push_str("1. Map the STRUCTURAL_TRUTH cluster around this state.\n");
        prompt.push_str("2. Find symmetric pairs (same family, different mode RGB/CMYK).\n");
        prompt.push_str("3. Which neighbors are surface synonyms vs deep structural matches?\n");
    }

    prompt.push_str("\nUse web_search to find real-world examples that illustrate these qualia transitions.\n");
    prompt.push_str("Report any new entities or relationships discovered for graph enrichment.\n");

    prompt
}

// ============================================================================
// Step 4: Crew config — orchestration for turn-based reasoning
// ============================================================================

#[derive(Debug, Serialize)]
struct CrewConfig {
    name: String,
    process: String,
    agents: Vec<AgentConfig>,
    tasks: Vec<TaskConfig>,
    benchmark: BenchmarkReport,
}

#[derive(Debug, Serialize)]
struct TaskConfig {
    name: String,
    description: String,
    expected_output: String,
    agent_role: String,
    context_from: Vec<String>,
}

#[derive(Debug, Serialize)]
struct BenchmarkReport {
    total_nodes: usize,
    total_edges: usize,
    hydration_time_us: u64,
    fanout_rounds: Vec<FanoutRound>,
    agents_spawned: usize,
    db_size_bytes: usize,
    thinking_efficiency: Vec<ThinkingEfficiency>,
}

#[derive(Debug, Clone, Serialize)]
struct ThinkingEfficiency {
    agent_id: String,
    style_name: String,
    crystallization: f32,
    edge_coverage: f32,  // what fraction of graph this agent "sees"
    reasoning_depth: u32, // max hops in causal chain
}

fn generate_crew(
    agents: &[AgentConfig],
    hydrated: &[HydratedNode],
    _edges: &[GraphEdge],
    fanout_data: &[FanoutRound],
    hydration_time: u64,
    total_nodes: usize,
    total_edges: usize,
) -> CrewConfig {
    // Generate tasks — each agent gets one turn, chained sequentially
    let mut tasks = Vec::new();
    let mut prev_task_name: Option<String> = None;

    for (i, agent) in agents.iter().enumerate() {
        let task_name = format!("turn_{:02}_{}", i, agent.id.replace(' ', "_"));
        let context = prev_task_name.iter().cloned().collect();

        tasks.push(TaskConfig {
            name: task_name.clone(),
            description: agent.spo_prompt.clone(),
            expected_output: format!(
                "A JSON report with: (1) causal chain analysis, \
                 (2) new entities/relationships discovered, \
                 (3) confidence scores for each finding"
            ),
            agent_role: agent.role.clone(),
            context_from: context,
        });

        prev_task_name = Some(task_name);
    }

    // Benchmark thinking efficiency
    let thinking_efficiency: Vec<ThinkingEfficiency> = agents.iter().map(|a| {
        let style_name = if a.thinking_style.analytical > 0.6 { "analytical" }
            else if a.thinking_style.creative > 0.6 { "creative" }
            else { "integrative" };

        ThinkingEfficiency {
            agent_id: a.id.clone(),
            style_name: style_name.into(),
            crystallization: a.awareness_summary.crystallization,
            edge_coverage: a.awareness_summary.edge_count as f32 / total_edges.max(1) as f32,
            reasoning_depth: if a.thinking_style.causal > 0.5 { 3 } else { 2 },
        }
    }).collect();

    let db_size = hydrated.len() * 32 // awareness vectors
        + total_edges * 96             // SPO triples (3 × 32 bytes)
        + agents.len() * 512;          // agent configs (approx)

    CrewConfig {
        name: "Qualia Graph Reasoning Crew".into(),
        process: "sequential".into(),
        agents: agents.to_vec(),
        tasks,
        benchmark: BenchmarkReport {
            total_nodes,
            total_edges,
            hydration_time_us: hydration_time,
            fanout_rounds: fanout_data.to_vec(),
            agents_spawned: agents.len(),
            db_size_bytes: db_size,
            thinking_efficiency,
        },
    }
}

// Derive Clone + Serialize for FanoutRound
impl Clone for FanoutRound {
    fn clone(&self) -> Self {
        FanoutRound {
            round: self.round,
            total_bits_changed: self.total_bits_changed,
            avg_crystallization: self.avg_crystallization,
            max_crystallization: self.max_crystallization,
            min_crystallization: self.min_crystallization,
            converged_nodes: self.converged_nodes,
            duration_us: self.duration_us,
            db_growth_bytes: self.db_growth_bytes,
        }
    }
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  HYDRATED GRAPH REASONING — Edge Propagation + Agents      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let cypher = include_str!("../../neo4j_import.cypher");

    // === Step 1: Parse ===
    println!("--- Step 1: Parse graph ---\n");
    let (nodes, edges) = parse_graph_from_cypher(cypher);
    println!("  Nodes: {}, Edges: {}", nodes.len(), edges.len());

    // === Step 2: Hydrate ===
    println!("\n--- Step 2: Hydrate nodes (bundle edge vectors) ---\n");
    let t0 = Instant::now();
    let mut hydrated = hydrate_nodes(&nodes, &edges);
    let hydration_us = t0.elapsed().as_micros() as u64;
    println!("  Hydration time: {}μs", hydration_us);

    // Show top 10 most crystallized nodes
    let mut by_cryst: Vec<&HydratedNode> = hydrated.iter().collect();
    by_cryst.sort_by(|a, b| b.crystallization.partial_cmp(&a.crystallization).unwrap());
    println!("\n  Top 10 most crystallized (settled awareness):");
    println!("  {:>5}  {:>40}  {:>5}  {:>5}  {:>4}  {:>6}",
        "cryst", "label", "edges", "S/O", "sign", "family");
    for h in by_cryst.iter().take(10) {
        println!("  {:>4.0}%  {:>40}  {:>5}  {}/{}  {:>4}  {:>6}",
            h.crystallization * 100.0,
            &h.node.label[..h.node.label.len().min(40)],
            h.edge_count,
            h.subject_count, h.object_count,
            h.causal_sign,
            h.node.family);
    }

    // Bottom 10 (most tensioned / uncertain)
    println!("\n  Bottom 10 (most tensioned / uncertain):");
    for h in by_cryst.iter().rev().take(10) {
        println!("  {:>4.0}%  {:>40}  {:>5}  {}/{}  {:>4}  {:>6}",
            h.crystallization * 100.0,
            &h.node.label[..h.node.label.len().min(40)],
            h.edge_count,
            h.subject_count, h.object_count,
            h.causal_sign,
            h.node.family);
    }

    // === Step 3: Fanout ===
    println!("\n--- Step 3: Fanout propagation (awareness spreading) ---\n");
    let fanout_rounds = fanout(&mut hydrated, &edges, 8);

    println!("  {:>5}  {:>12}  {:>8}  {:>8}  {:>8}  {:>6}  {:>8}",
        "Round", "Bits changed", "Avg crys", "Max crys", "Min crys", "Conv'd", "Time μs");
    for r in &fanout_rounds {
        println!("  {:>5}  {:>12}  {:>7.1}%  {:>7.1}%  {:>7.1}%  {:>6}  {:>8}",
            r.round, r.total_bits_changed,
            r.avg_crystallization * 100.0,
            r.max_crystallization * 100.0,
            r.min_crystallization * 100.0,
            r.converged_nodes,
            r.duration_us);
    }

    // Database growth
    println!("\n  Database growth per round:");
    for r in &fanout_rounds {
        let kb = r.db_growth_bytes as f64 / 1024.0;
        let bar_len = (kb / 10.0).min(50.0) as usize;
        println!("    R{}: {:>6.1} KB  {}", r.round, kb, "#".repeat(bar_len));
    }

    // === Step 4: Spawn agents ===
    println!("\n--- Step 4: Spawn crewai-rust agents ---\n");
    let agents = spawn_agents(&hydrated, &edges, "xai/grok-3");
    println!("  Agents spawned: {} (top actors by edge count)\n", agents.len());

    println!("  {:>30}  {:>6}  {:>5}  {:>5}  {:>5}  {:>5}  style",
        "Role", "cryst%", "edges", "S/O", "sign", "depth");
    for a in &agents {
        let style = if a.thinking_style.analytical > 0.6 { "analytical" }
            else if a.thinking_style.creative > 0.6 { "creative" }
            else { "integrative" };
        let depth = if a.thinking_style.causal > 0.5 { 3 } else { 2 };
        println!("  {:>30}  {:>5.0}%  {:>5}  {:.1}/{:.1}  {:>4}  {:>5}  {}",
            &a.role[..a.role.len().min(30)],
            a.awareness_summary.crystallization * 100.0,
            a.awareness_summary.edge_count,
            a.thinking_style.causal,
            a.thinking_style.receptive,
            a.awareness_summary.causal_sign,
            depth,
            style);
    }

    // === Step 5: Generate crew config ===
    println!("\n--- Step 5: Generate crew config ---\n");
    let crew = generate_crew(
        &agents, &hydrated, &edges, &fanout_rounds,
        hydration_us, nodes.len(), edges.len(),
    );

    let crew_json = serde_json::to_string_pretty(&crew).expect("serialize crew");
    let crew_path = "crew_config.json";
    std::fs::write(crew_path, &crew_json).expect("write crew config");
    println!("  Saved: {} ({} bytes)", crew_path, crew_json.len());

    // === Step 6: Benchmark report ===
    println!("\n--- Step 6: Benchmark ---\n");
    let bench = &crew.benchmark;

    println!("  Graph:        {} nodes, {} edges", bench.total_nodes, bench.total_edges);
    println!("  Hydration:    {}μs ({:.1} μs/node)",
        bench.hydration_time_us,
        bench.hydration_time_us as f64 / bench.total_nodes.max(1) as f64);
    println!("  Fanout:       {} rounds to converge", bench.fanout_rounds.len());
    println!("  DB size:      {:.1} KB ({:.1} bytes/node)",
        bench.db_size_bytes as f64 / 1024.0,
        bench.db_size_bytes as f64 / bench.total_nodes.max(1) as f64);
    println!("  Agents:       {} spawned", bench.agents_spawned);

    println!("\n  Thinking style distribution:");
    let mut style_counts: HashMap<&str, usize> = HashMap::new();
    for te in &bench.thinking_efficiency {
        *style_counts.entry(te.style_name.as_str()).or_insert(0) += 1;
    }
    for (style, count) in &style_counts {
        println!("    {:<15} {} agents", style, count);
    }

    let avg_coverage: f32 = bench.thinking_efficiency.iter()
        .map(|t| t.edge_coverage)
        .sum::<f32>() / bench.thinking_efficiency.len().max(1) as f32;
    println!("  Avg coverage: {:.1}% of graph per agent", avg_coverage * 100.0);

    // === Verdict ===
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║              HYDRATED REASONING VERDICT                     ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║                                                            ║");
    println!("║  Hydration: every node now carries a BF16 awareness        ║");
    println!("║  vector = majority bundle of ALL edge SPO signatures.      ║");
    println!("║                                                            ║");
    println!("║  Fanout: awareness propagated through {} rounds.            ║",
        fanout_rounds.len());
    println!("║  Each round XOR-blends neighbor awareness.                 ║");
    println!("║                                                            ║");
    println!("║  Agents: {} crewai-rust agents spawned, each with:         ║",
        agents.len());
    println!("║  - Hydrated awareness (crystallization %)                  ║");
    println!("║  - Thinking style (analytical/creative/integrative)        ║");
    println!("║  - SPO reasoning prompt (causal chain analysis)            ║");
    println!("║  - Turn-based sequential execution                         ║");
    println!("║                                                            ║");
    println!("║  Output: crew_config.json ready for crewai-rust kickoff.   ║");
    println!("║  LLM backend: xai/grok-3 (configurable).                  ║");
    println!("║                                                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
