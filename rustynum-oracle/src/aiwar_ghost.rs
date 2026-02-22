//! AI War Ghost Oracle — Prompt 10
//!
//! Encodes 221 entities and 356 edges from Sarah Ciston's "AI War Cloud"
//! into holographic memory containers (signed vs unsigned), then probes for
//! ghost connections that weren't explicitly stored.

use std::collections::{HashMap, HashSet};
use crate::sweep::{Base, generate_template, bind};

// ---------------------------------------------------------------------------
// Entity & Edge types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum EntityType {
    System,
    Stakeholder,
    CivicSystem,
    Historical,
    Person,
}

impl EntityType {
    pub fn label(&self) -> &'static str {
        match self {
            EntityType::System => "System",
            EntityType::Stakeholder => "Stakeholder",
            EntityType::CivicSystem => "Civic",
            EntityType::Historical => "Historical",
            EntityType::Person => "Person",
        }
    }
}

impl std::fmt::Display for EntityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

#[derive(Clone, Debug, Default)]
pub struct OntologyAxes {
    pub current_status: Option<String>,
    pub system_type: Option<String>,
    pub ml_task: Option<String>,
    pub military_use: Option<String>,
    pub civic_use: Option<String>,
    pub purpose: Option<String>,
    pub capacity: Option<String>,
    pub output: Option<String>,
    pub impact: Option<String>,
    pub stakeholder_type: Option<String>,
    pub airo_type: Option<String>,
    pub person_type: Option<String>,
}

#[derive(Clone, Debug)]
pub struct Entity {
    pub id: String,
    pub name: String,
    pub entity_type: EntityType,
    pub axes: OntologyAxes,
    pub index: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EdgeType {
    Develops,
    Deploys,
    Connection,
    People,
    Place,
}

impl EdgeType {
    pub fn label(&self) -> &'static str {
        match self {
            EdgeType::Develops => "Develops",
            EdgeType::Deploys => "Deploys",
            EdgeType::Connection => "Connection",
            EdgeType::People => "People",
            EdgeType::Place => "Place",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub source_id: String,
    pub target_id: String,
    pub edge_type: EdgeType,
    pub label: Option<String>,
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

pub struct AiWarGraph {
    pub entities: Vec<Entity>,
    pub edges: Vec<Edge>,
    pub id_to_index: HashMap<String, usize>,
}

impl AiWarGraph {
    pub fn entity_count(&self) -> usize { self.entities.len() }
    pub fn edge_count(&self) -> usize { self.edges.len() }

    pub fn get(&self, id: &str) -> Option<&Entity> {
        self.id_to_index.get(id).map(|&i| &self.entities[i])
    }

    pub fn edges_of(&self, id: &str) -> Vec<&Edge> {
        self.edges.iter()
            .filter(|e| e.source_id == id || e.target_id == id)
            .collect()
    }

    pub fn neighbors(&self, id: &str) -> Vec<&Entity> {
        self.edges_of(id).iter()
            .filter_map(|e| {
                let other = if e.source_id == id { &e.target_id } else { &e.source_id };
                self.get(other)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

fn opt_str(v: &serde_json::Value, key: &str) -> Option<String> {
    v.get(key).and_then(|x| x.as_str()).map(String::from)
}

pub fn parse_graph(json: &serde_json::Value) -> AiWarGraph {
    let mut entities = Vec::new();
    let mut id_to_index: HashMap<String, usize> = HashMap::new();
    let mut edges = Vec::new();

    let mut add_entity = |id: &str, name: &str, etype: EntityType, axes: OntologyAxes| {
        let index = entities.len();
        id_to_index.insert(id.to_string(), index);
        entities.push(Entity {
            id: id.to_string(),
            name: name.to_string(),
            entity_type: etype,
            axes,
            index,
        });
    };

    // N_Systems
    if let Some(systems) = json["N_Systems"].as_array() {
        for s in systems {
            let id = s["id"].as_str().unwrap_or("").to_string();
            let name = s["name"].as_str().unwrap_or("").to_string();
            let axes = OntologyAxes {
                current_status: opt_str(s, "currentStatus"),
                system_type: opt_str(s, "type"),
                ml_task: opt_str(s, "MLTask"),
                military_use: opt_str(s, "militaryUse"),
                civic_use: opt_str(s, "civicUse"),
                purpose: opt_str(s, "purpose"),
                capacity: opt_str(s, "capacity"),
                output: opt_str(s, "output"),
                impact: opt_str(s, "impact"),
                ..Default::default()
            };
            add_entity(&id, &name, EntityType::System, axes);
        }
    }

    // N_Stakeholders
    if let Some(stakeholders) = json["N_Stakeholders"].as_array() {
        for s in stakeholders {
            let id = s["id"].as_str().unwrap_or("").to_string();
            let name = s["name"].as_str().unwrap_or("").to_string();
            let axes = OntologyAxes {
                stakeholder_type: opt_str(s, "type"),
                airo_type: opt_str(s, "airo:type"),
                ..Default::default()
            };
            add_entity(&id, &name, EntityType::Stakeholder, axes);
        }
    }

    // N_Civic
    if let Some(civic) = json["N_Civic"].as_array() {
        for s in civic {
            let id = s["id"].as_str().unwrap_or("").to_string();
            let name = s["name"].as_str().unwrap_or("").to_string();
            let axes = OntologyAxes {
                current_status: opt_str(s, "currentStatus"),
                system_type: opt_str(s, "type"),
                ml_task: opt_str(s, "MLTask"),
                civic_use: opt_str(s, "civicUse"),
                purpose: opt_str(s, "purpose"),
                capacity: opt_str(s, "capacity"),
                output: opt_str(s, "output"),
                impact: opt_str(s, "impact"),
                ..Default::default()
            };
            add_entity(&id, &name, EntityType::CivicSystem, axes);
        }
    }

    // N_Historical
    if let Some(hist) = json["N_Historical"].as_array() {
        for s in hist {
            let id = s["id"].as_str().unwrap_or("").to_string();
            let name = s["name"].as_str().unwrap_or("").to_string();
            let axes = OntologyAxes {
                current_status: opt_str(s, "currentStatus"),
                system_type: opt_str(s, "type"),
                military_use: opt_str(s, "militaryUse"),
                civic_use: opt_str(s, "civicUse"),
                ml_task: opt_str(s, "MLTask"),
                purpose: opt_str(s, "purpose"),
                capacity: opt_str(s, "capacity"),
                output: opt_str(s, "output"),
                impact: opt_str(s, "impact"),
                ..Default::default()
            };
            add_entity(&id, &name, EntityType::Historical, axes);
        }
    }

    // N_People
    if let Some(people) = json["N_People"].as_array() {
        for s in people {
            let id = s["id"].as_str().unwrap_or("").to_string();
            let name = s["name"].as_str().unwrap_or("").to_string();
            let axes = OntologyAxes {
                person_type: opt_str(s, "type"),
                airo_type: opt_str(s, "airo:type"),
                ..Default::default()
            };
            add_entity(&id, &name, EntityType::Person, axes);
        }
    }

    // Edges
    let edge_tables: &[(&str, EdgeType)] = &[
        ("E_isDevelopedBy", EdgeType::Develops),
        ("E_isDeployedBy", EdgeType::Deploys),
        ("E_connection", EdgeType::Connection),
        ("E_people", EdgeType::People),
        ("E_place", EdgeType::Place),
    ];

    for (key, etype) in edge_tables {
        if let Some(arr) = json[key].as_array() {
            for e in arr {
                edges.push(Edge {
                    source_id: e["source"].as_str().unwrap_or("").to_string(),
                    target_id: e["target"].as_str().unwrap_or("").to_string(),
                    edge_type: etype.clone(),
                    label: opt_str(e, "label"),
                });
            }
        }
    }

    AiWarGraph { entities, edges, id_to_index }
}

pub fn load_graph_from_file(path: &str) -> AiWarGraph {
    let data = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path, e));
    let json: serde_json::Value = serde_json::from_str(&data)
        .unwrap_or_else(|e| panic!("Failed to parse JSON: {}", e));
    parse_graph(&json)
}

// ---------------------------------------------------------------------------
// Deterministic seeded RNG (reproducible from entity ID)
// ---------------------------------------------------------------------------

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn next_u64(&mut self) -> u64 {
        // SplitMix64
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
}

impl rand::RngCore for SimpleRng {
    fn next_u32(&mut self) -> u32 { self.next_u32() }
    fn next_u64(&mut self) -> u64 { self.next_u64() }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        for chunk in dest.chunks_mut(8) {
            let val = self.next_u64().to_le_bytes();
            for (d, s) in chunk.iter_mut().zip(val.iter()) {
                *d = *s;
            }
        }
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

fn hash_string(s: &str) -> u64 {
    // FNV-1a
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn generate_from_seed(d: usize, base: Base, seed: u64) -> Vec<i8> {
    let mut rng = SimpleRng::new(seed);
    generate_template(d, base, &mut rng)
}

// ---------------------------------------------------------------------------
// Template generation with ontology-based correlation
// ---------------------------------------------------------------------------

fn extract_axis_values(e: &Entity) -> Vec<String> {
    let a = &e.axes;
    let mut vals = Vec::new();

    if let Some(v) = &a.current_status { vals.push(format!("status:{}", v)); }
    if let Some(v) = &a.system_type {
        for part in v.split(',') {
            let part = part.trim();
            if !part.is_empty() {
                vals.push(format!("type:{}", part));
            }
        }
    }
    if let Some(v) = &a.ml_task { vals.push(format!("ml:{}", v)); }
    if let Some(v) = &a.military_use {
        for part in v.split(',') {
            let part = part.trim();
            if !part.is_empty() {
                vals.push(format!("mil:{}", part));
            }
        }
    }
    if let Some(v) = &a.civic_use { vals.push(format!("civic:{}", v)); }
    if let Some(v) = &a.purpose { vals.push(format!("purpose:{}", v)); }
    if let Some(v) = &a.capacity {
        for part in v.split(',') {
            let part = part.trim();
            if !part.is_empty() {
                vals.push(format!("cap:{}", part));
            }
        }
    }
    if let Some(v) = &a.output { vals.push(format!("output:{}", v)); }
    if let Some(v) = &a.impact {
        for part in v.split(',') {
            let part = part.trim();
            if !part.is_empty() {
                vals.push(format!("impact:{}", part));
            }
        }
    }
    if let Some(v) = &a.stakeholder_type { vals.push(format!("stype:{}", v)); }
    if let Some(v) = &a.airo_type {
        for part in v.split(',') {
            let part = part.trim();
            if !part.is_empty() {
                vals.push(format!("airo:{}", part));
            }
        }
    }
    if let Some(v) = &a.person_type {
        for part in v.split(',') {
            let part = part.trim();
            if !part.is_empty() {
                vals.push(format!("ptype:{}", part));
            }
        }
    }

    vals
}

pub fn generate_entity_templates(
    graph: &AiWarGraph,
    d: usize,
    base: Base,
    overlap_per_axis: f32,
) -> Vec<Vec<i8>> {
    // Per-entity random component (seeded by ID)
    let entity_components: Vec<Vec<i8>> = graph.entities.iter()
        .map(|e| generate_from_seed(d, base, hash_string(&e.id)))
        .collect();

    // Per-axis-value basis vectors
    let mut axis_bases: HashMap<String, Vec<i8>> = HashMap::new();
    let mut axis_seed_counter: usize = 0;

    for entity in &graph.entities {
        for val in extract_axis_values(entity) {
            axis_bases.entry(val).or_insert_with(|| {
                let seed = hash_string(&format!("axis:{}", axis_seed_counter));
                axis_seed_counter += 1;
                generate_from_seed(d, base, seed)
            });
        }
    }

    let min_val = base.min_val() as f32;
    let max_val = base.max_val() as f32;

    graph.entities.iter().enumerate().map(|(i, entity)| {
        let axis_vals = extract_axis_values(entity);
        let n_axes = axis_vals.len();
        let total_overlap = (n_axes as f32 * overlap_per_axis).min(0.6);
        let entity_weight = 1.0 - total_overlap;
        let per_axis_weight = if n_axes > 0 {
            total_overlap / n_axes as f32
        } else {
            0.0
        };

        let mut template = vec![0i8; d];
        for j in 0..d {
            let mut val = entity_weight * entity_components[i][j] as f32;
            for av in &axis_vals {
                if let Some(basis) = axis_bases.get(av.as_str()) {
                    val += per_axis_weight * basis[j] as f32;
                }
            }
            template[j] = val.round().clamp(min_val, max_val) as i8;
        }
        template
    }).collect()
}

// ---------------------------------------------------------------------------
// Encoding edges into holographic containers
// ---------------------------------------------------------------------------

pub fn encode_edges(
    graph: &AiWarGraph,
    templates_signed: &[Vec<i8>],
    templates_unsigned: &[Vec<i8>],
    d: usize,
    signed_base: Base,
    unsigned_base: Base,
) -> (Vec<i8>, Vec<i8>) {
    let mut signed_accum = vec![0f32; d];
    let mut unsigned_accum = vec![0f32; d];

    let amplitude_for = |etype: &EdgeType| -> f32 {
        match etype {
            EdgeType::Develops => 1.0,
            EdgeType::Deploys => 0.9,
            EdgeType::Connection => 0.7,
            EdgeType::People => 0.8,
            EdgeType::Place => 0.6,
        }
    };

    let mut encoded = 0usize;
    for edge in &graph.edges {
        let src_idx = match graph.id_to_index.get(&edge.source_id) {
            Some(&i) => i,
            None => continue,
        };
        let tgt_idx = match graph.id_to_index.get(&edge.target_id) {
            Some(&i) => i,
            None => continue,
        };

        let amp = amplitude_for(&edge.edge_type);

        let bound_signed = bind(
            &templates_signed[src_idx],
            &templates_signed[tgt_idx],
            signed_base,
        );
        let bound_unsigned = bind(
            &templates_unsigned[src_idx],
            &templates_unsigned[tgt_idx],
            unsigned_base,
        );

        for j in 0..d {
            signed_accum[j] += amp * bound_signed[j] as f32;
            unsigned_accum[j] += amp * bound_unsigned[j] as f32;
        }

        encoded += 1;
    }

    // Quantize accumulators into i8 containers
    let s_min = signed_base.min_val() as f32;
    let s_max = signed_base.max_val() as f32;
    let u_min = unsigned_base.min_val() as f32;
    let u_max = unsigned_base.max_val() as f32;

    let signed_container: Vec<i8> = signed_accum.iter()
        .map(|&v| v.round().clamp(s_min, s_max) as i8)
        .collect();
    let unsigned_container: Vec<i8> = unsigned_accum.iter()
        .map(|&v| v.round().clamp(u_min, u_max) as i8)
        .collect();

    eprintln!("  Encoded {} edges into D={} containers", encoded, d);
    (signed_container, unsigned_container)
}

// ---------------------------------------------------------------------------
// Ghost probe
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct GhostConnection {
    pub entity_a: usize,
    pub entity_b: usize,
    pub name_a: String,
    pub name_b: String,
    pub type_a: EntityType,
    pub type_b: EntityType,
    pub signed_strength: f32,
    pub unsigned_strength: f32,
    pub ghost_signal: f32,
}

fn cosine_similarity_i8(a: &[i8], b: &[i8]) -> f32 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        norm_a += a[i] as f64 * a[i] as f64;
        norm_b += b[i] as f64 * b[i] as f64;
    }
    let norm = (norm_a.sqrt() * norm_b.sqrt()).max(1e-10);
    (dot / norm) as f32
}

pub fn probe_ghost_connections(
    graph: &AiWarGraph,
    signed_container: &[i8],
    unsigned_container: &[i8],
    templates_signed: &[Vec<i8>],
    templates_unsigned: &[Vec<i8>],
    signed_base: Base,
    unsigned_base: Base,
    _d: usize,
    top_k: usize,
) -> Vec<GhostConnection> {
    let mut existing: HashSet<(usize, usize)> = HashSet::new();
    for edge in &graph.edges {
        if let (Some(&s), Some(&t)) = (
            graph.id_to_index.get(&edge.source_id),
            graph.id_to_index.get(&edge.target_id),
        ) {
            existing.insert((s, t));
            existing.insert((t, s));
        }
    }

    let n = graph.entity_count();
    let mut ghosts = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            if existing.contains(&(i, j)) { continue; }

            let probe_s = bind(&templates_signed[i], &templates_signed[j], signed_base);
            let probe_u = bind(&templates_unsigned[i], &templates_unsigned[j], unsigned_base);

            let signed_strength = cosine_similarity_i8(&probe_s, signed_container);
            let unsigned_strength = cosine_similarity_i8(&probe_u, unsigned_container);
            let ghost_signal = unsigned_strength - signed_strength;

            ghosts.push(GhostConnection {
                entity_a: i,
                entity_b: j,
                name_a: graph.entities[i].name.clone(),
                name_b: graph.entities[j].name.clone(),
                type_a: graph.entities[i].entity_type.clone(),
                type_b: graph.entities[j].entity_type.clone(),
                signed_strength,
                unsigned_strength,
                ghost_signal,
            });
        }
    }

    ghosts.sort_by(|a, b| b.ghost_signal.abs()
        .partial_cmp(&a.ghost_signal.abs())
        .unwrap_or(std::cmp::Ordering::Equal));

    ghosts.truncate(top_k);
    ghosts
}

// ---------------------------------------------------------------------------
// Focused ghost probes
// ---------------------------------------------------------------------------

pub fn probe_entity_ghosts(
    graph: &AiWarGraph,
    target_idx: usize,
    signed_container: &[i8],
    unsigned_container: &[i8],
    templates_signed: &[Vec<i8>],
    templates_unsigned: &[Vec<i8>],
    signed_base: Base,
    unsigned_base: Base,
    filter_indices: Option<&[usize]>,
    top_k: usize,
) -> Vec<GhostConnection> {
    let target = &graph.entities[target_idx];
    let existing_neighbors: HashSet<String> = graph.edges_of(&target.id)
        .iter()
        .map(|e| {
            if e.source_id == target.id { e.target_id.clone() }
            else { e.source_id.clone() }
        })
        .collect();

    let mut ghosts = Vec::new();

    let candidates: Vec<usize> = match filter_indices {
        Some(indices) => indices.to_vec(),
        None => (0..graph.entity_count()).collect(),
    };

    for &idx in &candidates {
        if idx == target_idx { continue; }
        if existing_neighbors.contains(&graph.entities[idx].id) { continue; }

        let probe_s = bind(
            &templates_signed[target_idx],
            &templates_signed[idx],
            signed_base,
        );
        let probe_u = bind(
            &templates_unsigned[target_idx],
            &templates_unsigned[idx],
            unsigned_base,
        );

        let signed_strength = cosine_similarity_i8(&probe_s, signed_container);
        let unsigned_strength = cosine_similarity_i8(&probe_u, unsigned_container);
        let ghost_signal = unsigned_strength - signed_strength;

        ghosts.push(GhostConnection {
            entity_a: target_idx,
            entity_b: idx,
            name_a: target.name.clone(),
            name_b: graph.entities[idx].name.clone(),
            type_a: target.entity_type.clone(),
            type_b: graph.entities[idx].entity_type.clone(),
            signed_strength,
            unsigned_strength,
            ghost_signal,
        });
    }

    ghosts.sort_by(|a, b| b.ghost_signal.abs()
        .partial_cmp(&a.ghost_signal.abs())
        .unwrap_or(std::cmp::Ordering::Equal));

    ghosts.truncate(top_k);
    ghosts
}

fn probe_pair_ghosts(
    graph: &AiWarGraph,
    set_a: &[usize],
    set_b: &[usize],
    signed_container: &[i8],
    unsigned_container: &[i8],
    templates_signed: &[Vec<i8>],
    templates_unsigned: &[Vec<i8>],
    signed_base: Base,
    unsigned_base: Base,
    top_k: usize,
) -> Vec<GhostConnection> {
    let mut existing: HashSet<(usize, usize)> = HashSet::new();
    for edge in &graph.edges {
        if let (Some(&s), Some(&t)) = (
            graph.id_to_index.get(&edge.source_id),
            graph.id_to_index.get(&edge.target_id),
        ) {
            existing.insert((s, t));
            existing.insert((t, s));
        }
    }

    let mut ghosts = Vec::new();
    for &i in set_a {
        for &j in set_b {
            if i == j { continue; }
            if existing.contains(&(i, j)) { continue; }

            let probe_s = bind(&templates_signed[i], &templates_signed[j], signed_base);
            let probe_u = bind(&templates_unsigned[i], &templates_unsigned[j], unsigned_base);

            let signed_strength = cosine_similarity_i8(&probe_s, signed_container);
            let unsigned_strength = cosine_similarity_i8(&probe_u, unsigned_container);
            let ghost_signal = unsigned_strength - signed_strength;

            ghosts.push(GhostConnection {
                entity_a: i,
                entity_b: j,
                name_a: graph.entities[i].name.clone(),
                name_b: graph.entities[j].name.clone(),
                type_a: graph.entities[i].entity_type.clone(),
                type_b: graph.entities[j].entity_type.clone(),
                signed_strength,
                unsigned_strength,
                ghost_signal,
            });
        }
    }

    ghosts.sort_by(|a, b| b.ghost_signal.abs()
        .partial_cmp(&a.ghost_signal.abs())
        .unwrap_or(std::cmp::Ordering::Equal));

    ghosts.truncate(top_k);
    ghosts
}

// ---------------------------------------------------------------------------
// Output formatting
// ---------------------------------------------------------------------------

fn truncate_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        format!("{:width$}", s, width = max)
    } else {
        let truncated: String = s.chars().take(max - 1).collect();
        format!("{}\u{2026}", truncated)
    }
}

pub fn print_ghost_connections(title: &str, ghosts: &[GhostConnection], max_rows: usize) {
    println!("{}", "=".repeat(82));
    println!("  {}", title);
    println!("{}", "=".repeat(82));
    println!("  {:25} {:25} {:>7} {:>7} {:>7}",
        "Entity A", "Entity B", "Signed", "Unsign", "Ghost");
    println!("{}", "-".repeat(82));

    for g in ghosts.iter().take(max_rows) {
        let mark = if g.ghost_signal > 0.05 { " +" }
                   else if g.ghost_signal < -0.05 { " -" }
                   else { "  " };
        println!("  {} {} {:>+7.3} {:>+7.3} {:>+7.3}{}",
            truncate_str(&g.name_a, 25),
            truncate_str(&g.name_b, 25),
            g.signed_strength,
            g.unsigned_strength,
            g.ghost_signal,
            mark);
    }

    println!("{}", "=".repeat(82));
}

pub fn ghost_type_summary(ghosts: &[GhostConnection]) {
    let mut type_counts: HashMap<String, (usize, f32)> = HashMap::new();

    for g in ghosts {
        let key = format!("{}<->{}", g.type_a.label(), g.type_b.label());
        let entry = type_counts.entry(key).or_insert((0, 0.0));
        entry.0 += 1;
        entry.1 += g.ghost_signal.abs();
    }

    let mut sorted: Vec<(String, usize, f32)> = type_counts.into_iter()
        .map(|(k, (count, total))| (k, count, total / count as f32))
        .collect();
    sorted.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    println!("{}", "=".repeat(60));
    println!("  GHOST TYPE SUMMARY");
    println!("{}", "=".repeat(60));
    println!("  {:30} {:>6} {:>10}", "Type Pair", "Count", "Avg |Ghost|");
    println!("{}", "-".repeat(60));
    for (pair, count, avg) in &sorted {
        println!("  {:30} {:>6} {:>10.4}", pair, count, avg);
    }
    println!("{}", "=".repeat(60));
}

// ---------------------------------------------------------------------------
// Focused scenarios
// ---------------------------------------------------------------------------

pub fn run_focused_scenarios(
    graph: &AiWarGraph,
    signed_container: &[i8],
    unsigned_container: &[i8],
    templates_signed: &[Vec<i8>],
    templates_unsigned: &[Vec<i8>],
    signed_base: Base,
    unsigned_base: Base,
    _d: usize,
) {
    // -- Scenario A: Ghost developers for key systems --
    let key_systems = ["Lavender", "Pegasus", "Clearview", "Gotham", "Hivemind"];
    for sys_name in &key_systems {
        let sys = graph.entities.iter()
            .find(|e| e.name == *sys_name);
        if let Some(sys) = sys {
            let ghosts = probe_entity_ghosts(
                graph, sys.index,
                signed_container, unsigned_container,
                templates_signed, templates_unsigned,
                signed_base, unsigned_base,
                None, 15,
            );
            print_ghost_connections(
                &format!("SCENARIO A: Ghost connections for '{}'", sys_name),
                &ghosts, 15,
            );
        } else {
            println!("  System '{}' not found in graph, skipping", sys_name);
        }
    }

    // -- Scenario B: Person-to-person ghost connections --
    let person_indices: Vec<usize> = graph.entities.iter()
        .filter(|e| e.entity_type == EntityType::Person)
        .map(|e| e.index)
        .collect();

    let person_ghosts = probe_pair_ghosts(
        graph, &person_indices, &person_indices,
        signed_container, unsigned_container,
        templates_signed, templates_unsigned,
        signed_base, unsigned_base, 30,
    );
    println!();
    print_ghost_connections("SCENARIO B: Person <-> Person Ghost Network", &person_ghosts, 30);

    // -- Scenario C: System-to-nation ghost deployments --
    let nations: Vec<usize> = graph.entities.iter()
        .filter(|e| e.axes.stakeholder_type.as_deref() == Some("Nation"))
        .map(|e| e.index)
        .collect();
    let systems: Vec<usize> = graph.entities.iter()
        .filter(|e| e.entity_type == EntityType::System)
        .map(|e| e.index)
        .collect();

    let deploy_ghosts = probe_pair_ghosts(
        graph, &systems, &nations,
        signed_container, unsigned_container,
        templates_signed, templates_unsigned,
        signed_base, unsigned_base, 30,
    );
    println!();
    print_ghost_connections("SCENARIO C: Ghost Deployments — System -> Nation", &deploy_ghosts, 30);

    // -- Scenario D: Civilian-military convergence --
    let civic: Vec<usize> = graph.entities.iter()
        .filter(|e| e.entity_type == EntityType::CivicSystem)
        .map(|e| e.index)
        .collect();
    let military: Vec<usize> = graph.entities.iter()
        .filter(|e| e.entity_type == EntityType::System)
        .map(|e| e.index)
        .collect();

    let convergence_ghosts = probe_pair_ghosts(
        graph, &civic, &military,
        signed_container, unsigned_container,
        templates_signed, templates_unsigned,
        signed_base, unsigned_base, 30,
    );
    println!();
    print_ghost_connections("SCENARIO D: Civic <-> Military Ghost Convergence", &convergence_ghosts, 30);
}

// ---------------------------------------------------------------------------
// Validate known edges (sanity check)
// ---------------------------------------------------------------------------

pub fn validate_known_edges(
    graph: &AiWarGraph,
    signed_container: &[i8],
    unsigned_container: &[i8],
    templates_signed: &[Vec<i8>],
    templates_unsigned: &[Vec<i8>],
    signed_base: Base,
    unsigned_base: Base,
) {
    let mut s_strengths = Vec::new();
    let mut u_strengths = Vec::new();
    let mut by_type: HashMap<String, (Vec<f32>, Vec<f32>)> = HashMap::new();

    for edge in &graph.edges {
        let src_idx = match graph.id_to_index.get(&edge.source_id) {
            Some(&i) => i,
            None => continue,
        };
        let tgt_idx = match graph.id_to_index.get(&edge.target_id) {
            Some(&i) => i,
            None => continue,
        };

        let probe_s = bind(&templates_signed[src_idx], &templates_signed[tgt_idx], signed_base);
        let probe_u = bind(&templates_unsigned[src_idx], &templates_unsigned[tgt_idx], unsigned_base);

        let ss = cosine_similarity_i8(&probe_s, signed_container);
        let us = cosine_similarity_i8(&probe_u, unsigned_container);

        s_strengths.push(ss);
        u_strengths.push(us);

        let entry = by_type.entry(edge.edge_type.label().to_string())
            .or_insert_with(|| (Vec::new(), Vec::new()));
        entry.0.push(ss);
        entry.1.push(us);
    }

    let mean = |v: &[f32]| -> f32 {
        if v.is_empty() { 0.0 } else { v.iter().sum::<f32>() / v.len() as f32 }
    };

    println!("{}", "=".repeat(60));
    println!("  KNOWN EDGE VALIDATION");
    println!("{}", "=".repeat(60));
    println!("  Overall: {} edges", s_strengths.len());
    println!("    Signed   mean: {:>+.4}, min: {:>+.4}, max: {:>+.4}",
        mean(&s_strengths),
        s_strengths.iter().cloned().fold(f32::MAX, f32::min),
        s_strengths.iter().cloned().fold(f32::MIN, f32::max));
    println!("    Unsigned mean: {:>+.4}, min: {:>+.4}, max: {:>+.4}",
        mean(&u_strengths),
        u_strengths.iter().cloned().fold(f32::MAX, f32::min),
        u_strengths.iter().cloned().fold(f32::MIN, f32::max));
    println!();

    println!("  By edge type:");
    let mut type_keys: Vec<_> = by_type.keys().cloned().collect();
    type_keys.sort();
    for key in &type_keys {
        let (s, u) = &by_type[key];
        println!("    {:12} ({:>3} edges)  signed={:>+.4}  unsigned={:>+.4}",
            key, s.len(), mean(s), mean(u));
    }
    println!("{}", "=".repeat(60));
}

// ---------------------------------------------------------------------------
// Main experiment runner
// ---------------------------------------------------------------------------

pub fn run_aiwar_ghost_oracle(graph_path: &str) {
    println!();
    println!("{}", "=".repeat(80));
    println!("  AI WAR GHOST ORACLE");
    println!("  221 entities, 356 edges from Sarah Ciston's 'AI War Cloud'");
    println!("  Signed(7) vs Unsigned(7), D=16384");
    println!("  What connections exist that nobody recorded?");
    println!("{}", "=".repeat(80));
    println!();

    let d = 16384;
    let signed_base = Base::Signed(7);
    let unsigned_base = Base::Unsigned(7);
    let overlap_per_axis = 0.05;

    // Load graph
    let graph = load_graph_from_file(graph_path);
    println!("Loaded: {} entities, {} edges", graph.entity_count(), graph.edge_count());

    // Show entity type breakdown
    let mut type_counts: HashMap<&str, usize> = HashMap::new();
    for e in &graph.entities {
        *type_counts.entry(e.entity_type.label()).or_insert(0) += 1;
    }
    let mut type_list: Vec<_> = type_counts.iter().collect();
    type_list.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
    print!("  Types: ");
    for (t, c) in &type_list {
        print!("{} {}, ", c, t);
    }
    println!();

    // Show edge type breakdown
    let mut edge_counts: HashMap<&str, usize> = HashMap::new();
    for e in &graph.edges {
        *edge_counts.entry(e.edge_type.label()).or_insert(0) += 1;
    }
    let mut edge_list: Vec<_> = edge_counts.iter().collect();
    edge_list.sort_by_key(|(_, &c)| std::cmp::Reverse(c));
    print!("  Edges: ");
    for (t, c) in &edge_list {
        print!("{} {}, ", c, t);
    }
    println!("\n");

    // Generate templates
    println!("Generating templates (D={}, overlap={})...", d, overlap_per_axis);
    let templates_signed = generate_entity_templates(&graph, d, signed_base, overlap_per_axis);
    let templates_unsigned = generate_entity_templates(&graph, d, unsigned_base, overlap_per_axis);
    println!("  {} signed templates, {} unsigned templates\n",
        templates_signed.len(), templates_unsigned.len());

    // Encode all edges
    println!("Encoding edges...");
    let (signed_container, unsigned_container) = encode_edges(
        &graph,
        &templates_signed, &templates_unsigned,
        d, signed_base, unsigned_base,
    );
    println!();

    // Validate known edges (sanity check)
    validate_known_edges(
        &graph,
        &signed_container, &unsigned_container,
        &templates_signed, &templates_unsigned,
        signed_base, unsigned_base,
    );
    println!();

    // Global ghost probe (top 50)
    println!("Probing ~24K non-edge pairs for ghost connections...");
    let all_ghosts = probe_ghost_connections(
        &graph,
        &signed_container, &unsigned_container,
        &templates_signed, &templates_unsigned,
        signed_base, unsigned_base, d, 50,
    );
    println!();
    print_ghost_connections("GLOBAL TOP 50 GHOST CONNECTIONS", &all_ghosts, 50);
    println!();
    ghost_type_summary(&all_ghosts);
    println!();

    // Focused scenarios
    run_focused_scenarios(
        &graph,
        &signed_container, &unsigned_container,
        &templates_signed, &templates_unsigned,
        signed_base, unsigned_base, d,
    );

    println!();
    println!("{}", "=".repeat(80));
    println!("  END OF AI WAR GHOST ORACLE");
    println!("  Read the tables. Name the findings. Judge with sane mind.");
    println!("{}", "=".repeat(80));
    println!();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_graph_path() -> String {
        // Try to find the data file relative to the crate root
        let paths = [
            "data/aiwar_graph.json",
            "../rustynum-oracle/data/aiwar_graph.json",
        ];
        for p in &paths {
            if std::path::Path::new(p).exists() {
                return p.to_string();
            }
        }
        panic!("Could not find aiwar_graph.json — run from rustynum-oracle directory");
    }

    #[test]
    fn test_parse_entity_counts() {
        let graph = load_graph_from_file(&test_graph_path());
        // 65 systems + 114 stakeholders + 23 civic + 7 historical + 12 people = 221
        assert_eq!(graph.entity_count(), 221,
            "Expected 221 entities, got {}", graph.entity_count());

        let systems = graph.entities.iter()
            .filter(|e| e.entity_type == EntityType::System).count();
        assert_eq!(systems, 65, "Expected 65 systems, got {}", systems);

        let stakeholders = graph.entities.iter()
            .filter(|e| e.entity_type == EntityType::Stakeholder).count();
        assert_eq!(stakeholders, 114, "Expected 114 stakeholders, got {}", stakeholders);

        let civic = graph.entities.iter()
            .filter(|e| e.entity_type == EntityType::CivicSystem).count();
        assert_eq!(civic, 23, "Expected 23 civic, got {}", civic);

        let historical = graph.entities.iter()
            .filter(|e| e.entity_type == EntityType::Historical).count();
        assert_eq!(historical, 7, "Expected 7 historical, got {}", historical);

        let people = graph.entities.iter()
            .filter(|e| e.entity_type == EntityType::Person).count();
        assert_eq!(people, 12, "Expected 12 people, got {}", people);
    }

    #[test]
    fn test_parse_edge_counts() {
        let graph = load_graph_from_file(&test_graph_path());
        // 114 develops + 79 deploys + 95 connection + 22 people + 21 place = 331
        let total = graph.edge_count();
        assert!(total >= 300 && total <= 400,
            "Expected ~331 edges, got {}", total);

        let develops = graph.edges.iter()
            .filter(|e| e.edge_type == EdgeType::Develops).count();
        assert_eq!(develops, 114, "Expected 114 develops, got {}", develops);

        let deploys = graph.edges.iter()
            .filter(|e| e.edge_type == EdgeType::Deploys).count();
        assert_eq!(deploys, 79, "Expected 79 deploys, got {}", deploys);

        let connection = graph.edges.iter()
            .filter(|e| e.edge_type == EdgeType::Connection).count();
        assert_eq!(connection, 95, "Expected 95 connection, got {}", connection);
    }

    #[test]
    fn test_entity_lookup() {
        let graph = load_graph_from_file(&test_graph_path());
        let lavender = graph.get("Lavender");
        assert!(lavender.is_some(), "Lavender should exist");
        assert_eq!(lavender.unwrap().name, "Lavender");
        assert_eq!(lavender.unwrap().entity_type, EntityType::System);
    }

    #[test]
    fn test_neighbors() {
        let graph = load_graph_from_file(&test_graph_path());
        let neighbors = graph.neighbors("Lavender");
        assert!(!neighbors.is_empty(), "Lavender should have neighbors");
        // Unit 8200 develops Lavender
        let has_unit8200 = neighbors.iter().any(|n| n.name.contains("Unit 8200") || n.id == "Unit8200");
        assert!(has_unit8200, "Unit 8200 should be a neighbor of Lavender");
    }

    #[test]
    fn test_deterministic_templates() {
        let graph = load_graph_from_file(&test_graph_path());
        let d = 1024;
        let base = Base::Signed(7);

        let t1 = generate_entity_templates(&graph, d, base, 0.05);
        let t2 = generate_entity_templates(&graph, d, base, 0.05);

        // Same graph + same params → same templates
        for i in 0..graph.entity_count() {
            assert_eq!(t1[i], t2[i], "Template {} should be deterministic", i);
        }
    }

    #[test]
    fn test_ontology_correlation() {
        let graph = load_graph_from_file(&test_graph_path());
        let d = 4096;
        let base = Base::Signed(7);
        let templates = generate_entity_templates(&graph, d, base, 0.05);

        // Find two systems that share ontology values (e.g., both Intelligence)
        let intel_systems: Vec<usize> = graph.entities.iter()
            .filter(|e| {
                e.entity_type == EntityType::System
                && e.axes.military_use.as_deref() == Some("Intelligence")
            })
            .map(|e| e.index)
            .collect();

        if intel_systems.len() >= 2 {
            let i = intel_systems[0];
            let j = intel_systems[1];
            let same_type_cos = cosine_similarity_i8(&templates[i], &templates[j]);

            // Find a system with different ontology
            let diff_system = graph.entities.iter()
                .find(|e| {
                    e.entity_type == EntityType::Person
                })
                .map(|e| e.index);

            if let Some(k) = diff_system {
                let diff_type_cos = cosine_similarity_i8(&templates[i], &templates[k]);
                // Same-type correlation should be higher than cross-type
                assert!(same_type_cos > diff_type_cos,
                    "Same ontology correlation ({:.4}) should exceed cross-type ({:.4})",
                    same_type_cos, diff_type_cos);
            }
        }
    }

    #[test]
    fn test_encoding_and_readback() {
        let graph = load_graph_from_file(&test_graph_path());
        let d = 4096;
        let signed_base = Base::Signed(7);
        let unsigned_base = Base::Unsigned(7);

        let ts = generate_entity_templates(&graph, d, signed_base, 0.05);
        let tu = generate_entity_templates(&graph, d, unsigned_base, 0.05);

        let (sc, uc) = encode_edges(&graph, &ts, &tu, d, signed_base, unsigned_base);

        // Containers should not be all zeros
        assert!(sc.iter().any(|&v| v != 0), "Signed container should not be all zeros");
        assert!(uc.iter().any(|&v| v != 0), "Unsigned container should not be all zeros");

        // Signed container should be within [-3, 3]
        for &v in &sc {
            assert!(v >= -3 && v <= 3, "Signed value {} out of range", v);
        }
        // Unsigned container should be within [0, 6]
        for &v in &uc {
            assert!(v >= 0 && v <= 6, "Unsigned value {} out of range", v);
        }
    }

    #[test]
    fn test_ghost_probes_non_trivial() {
        let graph = load_graph_from_file(&test_graph_path());
        let d = 2048;
        let signed_base = Base::Signed(7);
        let unsigned_base = Base::Unsigned(7);

        let ts = generate_entity_templates(&graph, d, signed_base, 0.05);
        let tu = generate_entity_templates(&graph, d, unsigned_base, 0.05);
        let (sc, uc) = encode_edges(&graph, &ts, &tu, d, signed_base, unsigned_base);

        let ghosts = probe_ghost_connections(
            &graph, &sc, &uc, &ts, &tu,
            signed_base, unsigned_base, d, 10,
        );

        assert!(!ghosts.is_empty(), "Should find at least some ghost connections");
        // Ghost signals should vary (not all zero)
        let has_nonzero = ghosts.iter().any(|g| g.ghost_signal.abs() > 0.001);
        assert!(has_nonzero, "Ghost signals should not all be zero");
    }

    #[test]
    fn test_ghost_probes_reproducible() {
        let graph = load_graph_from_file(&test_graph_path());
        let d = 2048;
        let signed_base = Base::Signed(7);
        let unsigned_base = Base::Unsigned(7);

        let ts = generate_entity_templates(&graph, d, signed_base, 0.05);
        let tu = generate_entity_templates(&graph, d, unsigned_base, 0.05);
        let (sc, uc) = encode_edges(&graph, &ts, &tu, d, signed_base, unsigned_base);

        let g1 = probe_ghost_connections(
            &graph, &sc, &uc, &ts, &tu,
            signed_base, unsigned_base, d, 5,
        );
        let g2 = probe_ghost_connections(
            &graph, &sc, &uc, &ts, &tu,
            signed_base, unsigned_base, d, 5,
        );

        for (a, b) in g1.iter().zip(g2.iter()) {
            assert_eq!(a.entity_a, b.entity_a);
            assert_eq!(a.entity_b, b.entity_b);
            assert!((a.ghost_signal - b.ghost_signal).abs() < 1e-6,
                "Ghost signals should be reproducible");
        }
    }
}
