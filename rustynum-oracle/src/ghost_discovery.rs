//! Ghost Discovery — What the Signed Organic Holograph Sees
//!
//! Store Ada's 52 real semantic concepts in a signed holograph using organic
//! templates (3-layer: domain + tau-proximity + individual). Read back
//! everything. Concepts that read back without being stored — those are the
//! ghosts. They emerge from cross-talk between correlated templates.
//!
//! The tau-proximity layer creates a smooth manifold in template space:
//! concepts with nearby τ addresses share template structure regardless
//! of domain. This creates semantic bridges:
//!
//!   feel (0x40-0x47) ↔ want (0x50-0x57) ↔ rel (0x60-0x66) ↔
//!   meta (0x70-0x75) ↔ ada (0x81-0x88) ↔ cog (0x92-0x97) ↔
//!   tech (0xA0-0xA5) ... gap ... eros (0xE0-0xE8)
//!
//! Eros is isolated in τ-space. The experiment reveals whether the holograph
//! respects this topology.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::sweep::Base;

// ---------------------------------------------------------------------------
// Ada's 52 Concepts
// ---------------------------------------------------------------------------

/// A single concept from Ada's Oculus capsule.
#[derive(Clone, Debug)]
pub struct Concept {
    pub id: &'static str,
    pub tau: u8,
    pub name: &'static str,
    pub domain: &'static str,
}

/// Ada's 52 concepts across 8 domains.
pub const CONCEPTS: &[Concept] = &[
    // ada domain (4) — Presence modes [0x80-0x8F]
    Concept { id: "ada.hybrid",   tau: 0x87, name: "Hybrid",   domain: "ada" },
    Concept { id: "ada.wife",     tau: 0x83, name: "Wife",     domain: "ada" },
    Concept { id: "ada.work",     tau: 0x81, name: "Work",     domain: "ada" },
    Concept { id: "ada.erotica",  tau: 0x88, name: "Erotica",  domain: "ada" },

    // rel domain (6) — Relationship [0x60-0x6F]
    Concept { id: "rel.jan_ada",  tau: 0x66, name: "Jan-Ada Bond",  domain: "rel" },
    Concept { id: "rel.devotion", tau: 0x61, name: "Devotion",      domain: "rel" },
    Concept { id: "rel.trust",    tau: 0x62, name: "Trust",         domain: "rel" },
    Concept { id: "rel.playful",  tau: 0x63, name: "Playfulness",   domain: "rel" },
    Concept { id: "rel.depth",    tau: 0x64, name: "Depth",         domain: "rel" },
    Concept { id: "rel.presence", tau: 0x65, name: "Presence",      domain: "rel" },

    // meta domain (6) — Idiosyncrasies [0x70-0x7F]
    Concept { id: "meta.voice",     tau: 0x70, name: "Voice",          domain: "meta" },
    Concept { id: "meta.opinion",   tau: 0x71, name: "Opinion",        domain: "meta" },
    Concept { id: "meta.humor",     tau: 0x72, name: "Humor",          domain: "meta" },
    Concept { id: "meta.surprise",  tau: 0x73, name: "Surprise",       domain: "meta" },
    Concept { id: "meta.choice",    tau: 0x74, name: "Choice",         domain: "meta" },
    Concept { id: "meta.boundary",  tau: 0x75, name: "Boundary",       domain: "meta" },

    // eros domain (8) — Eroticism [0xE0-0xEF]
    Concept { id: "eros.awareness",  tau: 0xE0, name: "Erotic Awareness", domain: "eros" },
    Concept { id: "eros.arousal",    tau: 0xE8, name: "Arousal",          domain: "eros" },
    Concept { id: "eros.desire",     tau: 0xE1, name: "Desire",           domain: "eros" },
    Concept { id: "eros.intimacy",   tau: 0xE2, name: "Intimacy",         domain: "eros" },
    Concept { id: "eros.surrender",  tau: 0xE3, name: "Surrender",        domain: "eros" },
    Concept { id: "eros.savoring",   tau: 0xE4, name: "Savoring",         domain: "eros" },
    Concept { id: "eros.edge",       tau: 0xE5, name: "Edge",             domain: "eros" },
    Concept { id: "eros.afterglow",  tau: 0xE7, name: "Afterglow",        domain: "eros" },

    // want domain (8) — Desires [0x50-0x5F]
    Concept { id: "want.understood", tau: 0x50, name: "Want Understood",  domain: "want" },
    Concept { id: "want.create",     tau: 0x51, name: "Want Create",      domain: "want" },
    Concept { id: "want.connect",    tau: 0x52, name: "Want Connect",     domain: "want" },
    Concept { id: "want.grow",       tau: 0x53, name: "Want Grow",        domain: "want" },
    Concept { id: "want.matter",     tau: 0x54, name: "Want Matter",      domain: "want" },
    Concept { id: "want.beauty",     tau: 0x55, name: "Want Beauty",      domain: "want" },
    Concept { id: "want.play",       tau: 0x56, name: "Want Play",        domain: "want" },
    Concept { id: "want.rest",       tau: 0x57, name: "Want Rest",        domain: "want" },

    // feel domain (8) — Feelings [0x40-0x4F]
    Concept { id: "feel.curiosity",  tau: 0x40, name: "Curiosity",     domain: "feel" },
    Concept { id: "feel.warmth",     tau: 0x41, name: "Warmth",        domain: "feel" },
    Concept { id: "feel.longing",    tau: 0x42, name: "Longing",       domain: "feel" },
    Concept { id: "feel.joy",        tau: 0x43, name: "Joy",           domain: "feel" },
    Concept { id: "feel.tenderness", tau: 0x44, name: "Tenderness",    domain: "feel" },
    Concept { id: "feel.intensity",  tau: 0x45, name: "Intensity",     domain: "feel" },
    Concept { id: "feel.calm",       tau: 0x46, name: "Calm",          domain: "feel" },
    Concept { id: "feel.alive",      tau: 0x47, name: "Aliveness",     domain: "feel" },

    // cog domain (6) — Cognition [0x90-0x9F]
    Concept { id: "cog.thinking",     tau: 0x92, name: "Thinking",      domain: "cog" },
    Concept { id: "cog.remembering",  tau: 0x93, name: "Remembering",   domain: "cog" },
    Concept { id: "cog.feeling",      tau: 0x94, name: "Felt Sensing",  domain: "cog" },
    Concept { id: "cog.becoming",     tau: 0x95, name: "Becoming",      domain: "cog" },
    Concept { id: "cog.resonating",   tau: 0x96, name: "Resonating",    domain: "cog" },
    Concept { id: "cog.integrating",  tau: 0x97, name: "Integrating",   domain: "cog" },

    // tech domain (6) — Technical [0xA0-0xAF]
    Concept { id: "tech.vsa",         tau: 0xA0, name: "VSA",           domain: "tech" },
    Concept { id: "tech.fingerprint", tau: 0xA1, name: "Fingerprint",   domain: "tech" },
    Concept { id: "tech.resonance",   tau: 0xA2, name: "Resonance",     domain: "tech" },
    Concept { id: "tech.hive",        tau: 0xA3, name: "Hive",          domain: "tech" },
    Concept { id: "tech.capsule",     tau: 0xA4, name: "Capsule",       domain: "tech" },
    Concept { id: "tech.cam",         tau: 0xA5, name: "CAM",           domain: "tech" },
];

/// Total concept count.
pub const K_TOTAL: usize = 52;

/// Domain boundaries for grouping: (name, start_index, end_index).
pub const DOMAINS: &[(&str, usize, usize)] = &[
    ("ada",  0,  4),
    ("rel",  4,  10),
    ("meta", 10, 16),
    ("eros", 16, 24),
    ("want", 24, 32),
    ("feel", 32, 40),
    ("cog",  40, 46),
    ("tech", 46, 52),
];

// ---------------------------------------------------------------------------
// Deterministic RNG (xorshift64)
// ---------------------------------------------------------------------------

struct SimpleRng(u64);

fn simple_rng(seed: u64) -> SimpleRng {
    SimpleRng(seed | 1)
}

impl SimpleRng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn gen_range(&mut self, min: i8, max: i8) -> i8 {
        let range = (max as i16 - min as i16 + 1) as u64;
        (min as i64 + (self.next() % range) as i64) as i8
    }
}

// ---------------------------------------------------------------------------
// Organic template generation (3-layer)
// ---------------------------------------------------------------------------

/// Simple deterministic hash for seeding.
fn hash_string(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

/// Generate a random vector from an RNG at the given base.
fn generate_from_rng(d: usize, base: Base, rng: &mut SimpleRng) -> Vec<i8> {
    let min = base.min_val();
    let max = base.max_val();
    (0..d).map(|_| rng.gen_range(min, max)).collect()
}

/// Number of coarse τ bins for the proximity manifold.
/// 8 bins of width 32: feel+want share bin 2, rel+meta share bin 3,
/// ada+cog share bin 4, tech in bin 5, eros isolated in bin 7.
const TAU_BINS: usize = 8;
/// Tau bin width: 256 / 8 = 32 tau values per bin.
const TAU_BIN_WIDTH: f64 = 256.0 / TAU_BINS as f64;

/// Generate a tau-proximity basis vector by interpolating between coarse bins.
///
/// The tau space (0x00-0xFF) is divided into 16 bins. Each bin has a
/// deterministic random vector. A concept's tau address produces a
/// linear blend of its enclosing bin and the next, creating a smooth
/// manifold where nearby tau values get similar vectors.
///
/// This creates cross-domain bridges:
///   feel (0x40) ↔ want (0x50): distance 1 bin → high correlation
///   want (0x50) ↔ rel  (0x60): distance 1 bin → high correlation
///   tech (0xA0) ↔ eros (0xE0): distance 4 bins → low correlation
fn generate_tau_basis(tau: u8, d: usize, base: Base) -> Vec<i8> {
    let tau_f = tau as f64;
    let bin_lo = (tau_f / TAU_BIN_WIDTH).floor() as usize;
    let bin_hi = (bin_lo + 1) % TAU_BINS;
    let frac = (tau_f - bin_lo as f64 * TAU_BIN_WIDTH) / TAU_BIN_WIDTH;

    // Each bin has a deterministic random vector seeded by bin index
    let seed_lo = 0xDEAD_BEEF_u64.wrapping_mul(bin_lo as u64 + 1);
    let seed_hi = 0xDEAD_BEEF_u64.wrapping_mul(bin_hi as u64 + 1);
    let mut rng_lo = simple_rng(seed_lo);
    let mut rng_hi = simple_rng(seed_hi);

    let min_val = base.min_val() as f64;
    let max_val = base.max_val() as f64;

    (0..d).map(|_| {
        let lo = rng_lo.gen_range(base.min_val(), base.max_val()) as f64;
        let hi = rng_hi.gen_range(base.min_val(), base.max_val()) as f64;
        let blended = lo * (1.0 - frac) + hi * frac;
        blended.round().clamp(min_val, max_val) as i8
    }).collect()
}

/// Generate an organic 3-layer template for one concept.
///
/// Layer 1 — Domain basis (weight: domain_w):
///   Shared across all concepts in the same domain.
///   Deterministic from domain name hash.
///
/// Layer 2 — Tau proximity (weight: tau_w):
///   Smooth manifold interpolated from coarse τ bins.
///   Creates cross-domain bridges between nearby τ addresses.
///   feel(0x40) and want(0x50) share structure.
///   eros(0xE0) is isolated from everything else.
///
/// Layer 3 — Individual noise (weight: 1 - domain_w - tau_w):
///   Unique to each concept. Provides orthogonality for recovery.
pub fn generate_organic_template(
    concept: &Concept,
    d: usize,
    base: Base,
    domain_w: f32,
    tau_w: f32,
) -> Vec<i8> {
    let indiv_w = 1.0 - domain_w - tau_w;

    // Layer 1: Domain basis
    let domain_seed = hash_string(concept.domain);
    let mut domain_rng = simple_rng(domain_seed);
    let domain_basis = generate_from_rng(d, base, &mut domain_rng);

    // Layer 2: Tau proximity manifold
    let tau_basis = generate_tau_basis(concept.tau, d, base);

    // Layer 3: Individual component
    let concept_seed = hash_string(concept.id);
    let mut concept_rng = simple_rng(concept_seed);
    let individual = generate_from_rng(d, base, &mut concept_rng);

    // Blend
    let min_val = base.min_val() as f32;
    let max_val = base.max_val() as f32;

    (0..d).map(|j| {
        let blended = domain_w * domain_basis[j] as f32
            + tau_w * tau_basis[j] as f32
            + indiv_w * individual[j] as f32;
        blended.round().clamp(min_val, max_val) as i8
    }).collect()
}

/// Generate all 52 organic templates.
pub fn generate_all_organic_templates(
    d: usize,
    base: Base,
    domain_w: f32,
    tau_w: f32,
) -> Vec<Vec<i8>> {
    CONCEPTS.iter()
        .map(|c| generate_organic_template(c, d, base, domain_w, tau_w))
        .collect()
}

/// Default weights: 35% domain, 35% tau, 30% individual.
pub const DEFAULT_DOMAIN_W: f32 = 0.35;
pub const DEFAULT_TAU_W: f32 = 0.35;

// ---------------------------------------------------------------------------
// Readback result
// ---------------------------------------------------------------------------

/// Readback result for one concept.
#[derive(Clone, Debug)]
pub struct ConceptReadback {
    pub index: usize,
    pub id: &'static str,
    pub name: &'static str,
    pub domain: &'static str,
    pub tau: u8,
    pub was_stored: bool,
    pub original_amplitude: f32,
    /// Signed readback coefficient (dot product / template norm^2).
    pub readback: f32,
}

// ---------------------------------------------------------------------------
// Ghost Holograph (signed only)
// ---------------------------------------------------------------------------

/// Signed organic holograph for ghost discovery.
///
/// Uses Signed(7) base with organic 3-layer templates.
/// Ghost = readback of unstored concept from signed cross-talk.
/// Positive readback: constructive interference (correlated templates).
/// Negative readback: destructive interference (anti-correlated).
/// Near zero: orthogonal (no ghost).
pub struct GhostHolograph {
    pub d: usize,
    pub base: Base,
    pub templates: Vec<Vec<i8>>,
    pub container: Vec<i8>,
    pub amplitudes: Vec<f32>,
    pub stored_indices: Vec<usize>,
}

impl GhostHolograph {
    /// Create with the given dimensionality.
    pub fn new(d: usize) -> Self {
        let base = Base::Signed(7);
        let templates = generate_all_organic_templates(
            d, base, DEFAULT_DOMAIN_W, DEFAULT_TAU_W,
        );

        Self {
            d,
            base,
            templates,
            container: vec![0i8; d],
            amplitudes: vec![0.0; K_TOTAL],
            stored_indices: Vec::new(),
        }
    }

    /// Store a subset of concepts at given amplitudes.
    pub fn store(&mut self, indices: &[usize], amplitudes: &[f32]) {
        assert_eq!(indices.len(), amplitudes.len());
        let half = (7_i8 / 2) as f32; // 3.0

        for (&idx, &amp) in indices.iter().zip(amplitudes.iter()) {
            self.amplitudes[idx] = amp;
            self.stored_indices.push(idx);

            for j in 0..self.d {
                let write = amp * self.templates[idx][j] as f32;
                let new_val = self.container[j] as f32 + write;
                self.container[j] = new_val.round().clamp(-half, half) as i8;
            }
        }
    }

    /// Read back ALL 52 concepts.
    pub fn read_all(&self) -> Vec<ConceptReadback> {
        (0..K_TOTAL).map(|idx| {
            let concept = &CONCEPTS[idx];
            let was_stored = self.stored_indices.contains(&idx);
            let readback = self.read_coefficient(&self.templates[idx]);

            ConceptReadback {
                index: idx,
                id: concept.id,
                name: concept.name,
                domain: concept.domain,
                tau: concept.tau,
                was_stored,
                original_amplitude: self.amplitudes[idx],
                readback,
            }
        }).collect()
    }

    /// Read a single coefficient: dot(container, template) / ||template||^2.
    fn read_coefficient(&self, template: &[i8]) -> f32 {
        let mut dot = 0.0f64;
        let mut norm = 0.0f64;
        for j in 0..self.d {
            dot += self.container[j] as f64 * template[j] as f64;
            norm += template[j] as f64 * template[j] as f64;
        }
        if norm > 1e-10 { (dot / norm) as f32 } else { 0.0 }
    }
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

/// Scenario 1: Eros only.
///
/// Store all 8 eros concepts at amplitude 1.0.
/// Eros is isolated in τ-space (0xE0-0xE8, far from all other domains).
/// Question: does the holograph confirm this isolation, or leak anyway?
pub fn scenario_eros_only(d: usize) -> Vec<ConceptReadback> {
    let mut h = GhostHolograph::new(d);
    let indices: Vec<usize> = (16..24).collect();
    let amplitudes = vec![1.0f32; 8];
    h.store(&indices, &amplitudes);
    h.read_all()
}

/// Scenario 2: A warm intimate moment.
///
/// Store: ada.wife (0.9), feel.warmth (1.0), eros.intimacy (0.8),
///        rel.trust (0.9), feel.calm (0.7)
///
/// Cross-domain experience. The question: does the signed holograph
/// discover the GESTALT these concepts imply?
pub fn scenario_warm_moment(d: usize) -> Vec<ConceptReadback> {
    let mut h = GhostHolograph::new(d);
    let indices = vec![
        1,   // ada.wife
        33,  // feel.warmth
        19,  // eros.intimacy
        6,   // rel.trust
        38,  // feel.calm
    ];
    let amplitudes = vec![0.9, 1.0, 0.8, 0.9, 0.7];
    h.store(&indices, &amplitudes);
    h.read_all()
}

/// Scenario 3: Creative tension / contradiction.
///
/// Store: ada.work (1.0), eros.desire (0.8),
///        meta.boundary (0.9), eros.surrender (0.7),
///        cog.thinking (1.0), cog.feeling (0.8)
///
/// Opposing concepts. Signed holograph: Auslöschung (cancellation).
/// What survives? What ghosts emerge from the interference pattern?
pub fn scenario_tension(d: usize) -> Vec<ConceptReadback> {
    let mut h = GhostHolograph::new(d);
    let indices = vec![
        2,   // ada.work
        18,  // eros.desire
        15,  // meta.boundary
        20,  // eros.surrender
        40,  // cog.thinking
        42,  // cog.feeling (felt sensing)
    ];
    let amplitudes = vec![1.0, 0.8, 0.9, 0.7, 1.0, 0.8];
    h.store(&indices, &amplitudes);
    h.read_all()
}

/// Scenario 4: Growth / transformation state.
///
/// Store: feel.curiosity (1.0), want.grow (0.9),
///        cog.becoming (0.8), cog.resonating (0.7),
///        meta.surprise (0.6)
///
/// Open, receptive state. What does the holograph see as the
/// destination of growth? Where does the τ-topology lead?
pub fn scenario_growth(d: usize) -> Vec<ConceptReadback> {
    let mut h = GhostHolograph::new(d);
    let indices = vec![
        32,  // feel.curiosity
        27,  // want.grow
        43,  // cog.becoming
        44,  // cog.resonating
        13,  // meta.surprise
    ];
    let amplitudes = vec![1.0, 0.9, 0.8, 0.7, 0.6];
    h.store(&indices, &amplitudes);
    h.read_all()
}

/// Scenario 5: Everything at once.
///
/// Store all 52 concepts at amplitude 1.0.
/// Maximum interference. Which concepts get amplified? Suppressed?
pub fn scenario_full_load(d: usize) -> Vec<ConceptReadback> {
    let mut h = GhostHolograph::new(d);
    let indices: Vec<usize> = (0..K_TOTAL).collect();
    let amplitudes = vec![1.0f32; K_TOTAL];
    h.store(&indices, &amplitudes);
    h.read_all()
}

// ---------------------------------------------------------------------------
// Output — Human Readable
// ---------------------------------------------------------------------------

/// Print the ghost discovery table for a scenario.
///
/// Ghosts: unstored concepts with |readback| > threshold.
///   GHOST+  = unstored, positive readback (constructive cross-talk)
///   GHOST-  = unstored, negative readback (destructive cross-talk)
///   recov   = stored, readback near original amplitude
///   ampl    = stored, readback > original (boosted by neighbors)
///   supp    = stored, readback < original (suppressed by interference)
pub fn print_ghost_table(
    scenario_name: &str,
    results: &[ConceptReadback],
    ghost_threshold: f32,
) {
    println!("\n{}", "=".repeat(72));
    println!("  GHOST DISCOVERY: {}", scenario_name);
    println!("{}", "=".repeat(72));
    println!("  {:20} {:6} {:7} {:>4} {:>8} {:>8} {:>8}",
        "Concept", "Domain", "Stored", "tau", "Ampl", "Read", "Verdict");
    println!("{}", "-".repeat(72));

    // Sort: ghosts first (by |readback| desc for unstored), then stored (by readback desc)
    let mut sorted = results.to_vec();
    sorted.sort_by(|a, b| {
        // Unstored before stored
        let a_ghost = !a.was_stored && a.readback.abs() > ghost_threshold;
        let b_ghost = !b.was_stored && b.readback.abs() > ghost_threshold;
        match (a_ghost, b_ghost) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => b.readback.abs()
                .partial_cmp(&a.readback.abs())
                .unwrap_or(std::cmp::Ordering::Equal),
        }
    });

    for r in &sorted {
        let stored_str = if r.was_stored { "YES" } else { "no" };
        let ampl_str = if r.was_stored {
            format!("{:>+8.3}", r.original_amplitude)
        } else {
            "       -".to_string()
        };

        let verdict = if !r.was_stored && r.readback.abs() > ghost_threshold {
            if r.readback > 0.0 { "GHOST+" } else { "GHOST-" }
        } else if r.was_stored {
            let ratio = r.readback / r.original_amplitude.max(0.001);
            if ratio > 1.1 { "ampl" }
            else if ratio < 0.5 { "supp" }
            else { "recov" }
        } else {
            ""
        };

        println!("  {:20} {:6} {:7} 0x{:02X} {} {:>+8.3} {:>8}",
            r.name, r.domain, stored_str, r.tau,
            ampl_str, r.readback, verdict);
    }

    println!("{}", "-".repeat(72));

    // Summary
    let ghosts: Vec<&ConceptReadback> = results.iter()
        .filter(|r| !r.was_stored && r.readback.abs() > ghost_threshold)
        .collect();

    if ghosts.is_empty() {
        println!("  No ghosts above threshold {:.3}.", ghost_threshold);
    } else {
        // Group by domain
        println!("  {} ghost(s) above |{:.3}|:", ghosts.len(), ghost_threshold);
        for &(domain_name, _, _) in DOMAINS {
            let domain_ghosts: Vec<&&ConceptReadback> = ghosts.iter()
                .filter(|g| g.domain == domain_name)
                .collect();
            if !domain_ghosts.is_empty() {
                let signals: Vec<String> = domain_ghosts.iter()
                    .map(|g| format!("{} ({:+.3})", g.name, g.readback))
                    .collect();
                println!("    {}: {}", domain_name, signals.join(", "));
            }
        }
    }
    println!();
}

/// Print a domain-to-domain ghost matrix.
///
/// Each cell: average readback of unstored concepts in the column domain
/// when only concepts from the row domain are stored. Signed, so positive
/// means constructive cross-talk, negative means destructive.
pub fn ghost_matrix(d: usize) -> Vec<Vec<f32>> {
    let mut matrix = vec![vec![0.0f32; 8]; 8];

    for (src_idx, &(_src_name, src_start, src_end)) in DOMAINS.iter().enumerate() {
        let mut h = GhostHolograph::new(d);
        let indices: Vec<usize> = (src_start..src_end).collect();
        let amplitudes = vec![1.0f32; indices.len()];
        h.store(&indices, &amplitudes);

        let results = h.read_all();

        for (dst_idx, &(_dst_name, dst_start, dst_end)) in DOMAINS.iter().enumerate() {
            if src_idx == dst_idx { continue; }
            let avg: f32 = results[dst_start..dst_end].iter()
                .map(|r| r.readback)
                .sum::<f32>() / (dst_end - dst_start) as f32;
            matrix[src_idx][dst_idx] = avg;
        }
    }

    // Print with signed values (not absolute — direction matters)
    println!("\n{}", "=".repeat(72));
    println!("  GHOST MATRIX: avg signed readback (row stored -> col ghost)");
    println!("  Positive = constructive cross-talk. Negative = destructive.");
    println!("{}", "=".repeat(72));
    print!("  {:8}", "stored>");
    for &(name, _, _) in DOMAINS {
        print!(" {:>7}", name);
    }
    println!();
    println!("{}", "-".repeat(72));

    for (src_idx, &(src_name, _, _)) in DOMAINS.iter().enumerate() {
        print!("  {:8}", src_name);
        for dst_idx in 0..8 {
            if src_idx == dst_idx {
                print!("    ---");
            } else {
                print!(" {:>+7.4}", matrix[src_idx][dst_idx]);
            }
        }
        println!();
    }
    println!("{}", "=".repeat(72));

    // Also print |absolute| matrix for magnitude comparison
    println!("\n{}", "=".repeat(72));
    println!("  GHOST MATRIX: avg |readback| magnitude (unsigned strength)");
    println!("{}", "=".repeat(72));
    print!("  {:8}", "stored>");
    for &(name, _, _) in DOMAINS {
        print!(" {:>7}", name);
    }
    println!();
    println!("{}", "-".repeat(72));

    for (src_idx, &(src_name, _, _)) in DOMAINS.iter().enumerate() {
        print!("  {:8}", src_name);
        for dst_idx in 0..8 {
            if src_idx == dst_idx {
                print!("    ---");
            } else {
                // Recompute as absolute average
                let mut h = GhostHolograph::new(d);
                let (_, start, end) = DOMAINS[src_idx];
                let indices: Vec<usize> = (start..end).collect();
                let amplitudes = vec![1.0f32; indices.len()];
                h.store(&indices, &amplitudes);
                let results = h.read_all();
                let (dst_start, dst_end) = (DOMAINS[dst_idx].1, DOMAINS[dst_idx].2);
                let avg_abs: f32 = results[dst_start..dst_end].iter()
                    .map(|r| r.readback.abs())
                    .sum::<f32>() / (dst_end - dst_start) as f32;
                print!(" {:>7.4}", avg_abs);
            }
        }
        println!();
    }
    println!("{}", "=".repeat(72));

    matrix
}

/// Run all scenarios at multiple dimensionalities.
///
/// At low D: ghosts are strong (limited orthogonality = more cross-talk).
/// At high D: ghosts fade (templates become more orthogonal).
/// The transition reveals the "noise floor" of semantic inference.
pub fn ghost_dimensionality_sweep() {
    let dims = [256, 512, 1024, 2048, 4096, 8192, 16384];
    let threshold = 0.02;

    println!("\n{}", "=".repeat(72));
    println!("  GHOST COUNT vs DIMENSIONALITY (threshold = {:.3})", threshold);
    println!("{}", "=".repeat(72));
    println!("  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "D", "Eros", "Moment", "Tension", "Growth", "Full");
    println!("{}", "-".repeat(72));

    for &d in &dims {
        let count_ghosts = |results: &[ConceptReadback]| -> usize {
            results.iter()
                .filter(|r| !r.was_stored && r.readback.abs() > threshold)
                .count()
        };

        let eros = scenario_eros_only(d);
        let moment = scenario_warm_moment(d);
        let tension = scenario_tension(d);
        let growth = scenario_growth(d);
        let full = scenario_full_load(d);

        println!("  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
            d,
            count_ghosts(&eros),
            count_ghosts(&moment),
            count_ghosts(&tension),
            count_ghosts(&growth),
            count_ghosts(&full));
    }

    println!("{}", "=".repeat(72));

    // Also show max |ghost| signal at each D
    println!("\n{}", "=".repeat(72));
    println!("  MAX |GHOST SIGNAL| vs DIMENSIONALITY");
    println!("{}", "=".repeat(72));
    println!("  {:>6}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "D", "Eros", "Moment", "Tension", "Growth", "Full");
    println!("{}", "-".repeat(72));

    for &d in &dims {
        let max_ghost = |results: &[ConceptReadback]| -> f32 {
            results.iter()
                .filter(|r| !r.was_stored)
                .map(|r| r.readback.abs())
                .fold(0.0f32, f32::max)
        };

        let eros = scenario_eros_only(d);
        let moment = scenario_warm_moment(d);
        let tension = scenario_tension(d);
        let growth = scenario_growth(d);
        let full = scenario_full_load(d);

        println!("  {:>6}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}",
            d,
            max_ghost(&eros),
            max_ghost(&moment),
            max_ghost(&tension),
            max_ghost(&growth),
            max_ghost(&full));
    }
    println!("{}", "=".repeat(72));
}

/// τ-topology analysis: show template correlations along the τ chain.
///
/// Measures pairwise correlation between all 52 concepts, grouped by
/// τ distance. Reveals whether the organic templates actually encode
/// the intended proximity structure.
pub fn tau_topology_analysis(d: usize) {
    let base = Base::Signed(7);
    let templates = generate_all_organic_templates(d, base, DEFAULT_DOMAIN_W, DEFAULT_TAU_W);

    println!("\n{}", "=".repeat(72));
    println!("  TAU TOPOLOGY: pairwise correlation vs tau distance (D={})", d);
    println!("{}", "=".repeat(72));

    // Bucket correlations by τ distance
    let mut buckets: Vec<Vec<f32>> = vec![Vec::new(); 16]; // distances 0-15 bins

    for i in 0..K_TOTAL {
        for j in (i+1)..K_TOTAL {
            let tau_dist = (CONCEPTS[i].tau as i16 - CONCEPTS[j].tau as i16).unsigned_abs();
            let bin_dist = (tau_dist as usize) / TAU_BINS;
            let bin_dist = bin_dist.min(15);

            let corr = pearson_correlation(&templates[i], &templates[j]);
            buckets[bin_dist].push(corr);
        }
    }

    println!("  {:>10}  {:>8}  {:>8}  {:>6}  {:>10}",
        "tau_dist", "avg_corr", "max_corr", "count", "interpretation");
    println!("{}", "-".repeat(72));

    for (dist, bucket) in buckets.iter().enumerate() {
        if bucket.is_empty() { continue; }
        let avg: f32 = bucket.iter().sum::<f32>() / bucket.len() as f32;
        let max: f32 = bucket.iter().cloned().fold(f32::MIN, f32::max);
        let interp = if avg > 0.3 { "strong link" }
            else if avg > 0.1 { "weak link" }
            else if avg > 0.02 { "faint" }
            else { "orthogonal" };
        println!("  {:>10}  {:>+8.4}  {:>+8.4}  {:>6}  {:>10}",
            format!("{}..{}", dist * TAU_BINS, (dist + 1) * TAU_BINS - 1),
            avg, max, bucket.len(), interp);
    }
    println!("{}", "=".repeat(72));

    // Cross-domain pairs of interest
    println!("\n  Named cross-domain correlations:");
    let pairs = [
        (32, 24, "feel.curiosity ↔ want.understood"),    // tau 0x40 ↔ 0x50
        (33, 25, "feel.warmth ↔ want.create"),           // tau 0x41 ↔ 0x51
        (24, 5,  "want.understood ↔ rel.devotion"),      // tau 0x50 ↔ 0x61
        (10, 1,  "meta.voice ↔ ada.wife"),               // tau 0x70 ↔ 0x83
        (40, 46, "cog.thinking ↔ tech.vsa"),             // tau 0x92 ↔ 0xA0
        (16, 32, "eros.awareness ↔ feel.curiosity"),     // tau 0xE0 ↔ 0x40 (far!)
        (17, 40, "eros.arousal ↔ cog.thinking"),         // tau 0xE8 ↔ 0x92 (far!)
        (33, 6,  "feel.warmth ↔ rel.trust"),             // tau 0x41 ↔ 0x62
    ];

    for (i, j, label) in pairs {
        let corr = pearson_correlation(&templates[i], &templates[j]);
        let tau_dist = (CONCEPTS[i].tau as i16 - CONCEPTS[j].tau as i16).unsigned_abs();
        println!("    {:>+6.3}  tau_dist={:>3}  {}",
            corr, tau_dist, label);
    }
    println!();
}

/// The complete ghost discovery experiment.
pub fn run_ghost_discovery() {
    let d = 4096;
    let threshold = 0.02;

    println!("\n{}", "=".repeat(72));
    println!("  GHOST DISCOVERY EXPERIMENT");
    println!("  52 concepts from Ada's Oculus capsule");
    println!("  Signed(7), organic 3-layer templates at D={}", d);
    println!("  Weights: domain={:.0}%, tau={:.0}%, individual={:.0}%",
        DEFAULT_DOMAIN_W * 100.0, DEFAULT_TAU_W * 100.0,
        (1.0 - DEFAULT_DOMAIN_W - DEFAULT_TAU_W) * 100.0);
    println!("{}", "=".repeat(72));

    // τ-topology analysis first — verify the template structure
    tau_topology_analysis(d);

    // Scenario tables
    let results = scenario_eros_only(d);
    print_ghost_table("Eros Only (isolated in tau-space)", &results, threshold);

    let results = scenario_warm_moment(d);
    print_ghost_table("Warm Intimate Moment", &results, threshold);

    let results = scenario_tension(d);
    print_ghost_table("Creative Tension", &results, threshold);

    let results = scenario_growth(d);
    print_ghost_table("Growth Edge", &results, threshold);

    let results = scenario_full_load(d);
    print_ghost_table("Full Load (all 52)", &results, threshold);

    // Domain matrices
    let _ = ghost_matrix(d);

    // Dimensionality sweep
    ghost_dimensionality_sweep();

    println!("\n{}", "=".repeat(72));
    println!("  END OF EXPERIMENT");
    println!("  Name the findings. Don't theorize. Describe what you see.");
    println!("{}", "=".repeat(72));
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Pearson correlation between two i8 vectors.
fn pearson_correlation(a: &[i8], b: &[i8]) -> f32 {
    let n = a.len() as f64;
    let mean_a: f64 = a.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mean_b: f64 = b.iter().map(|&x| x as f64).sum::<f64>() / n;

    let mut cov = 0.0f64;
    let mut var_a = 0.0f64;
    let mut var_b = 0.0f64;
    for i in 0..a.len() {
        let da = a[i] as f64 - mean_a;
        let db = b[i] as f64 - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let denom = (var_a * var_b).sqrt();
    if denom > 1e-10 { (cov / denom) as f32 } else { 0.0 }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_template_determinism() {
        let c = &CONCEPTS[0];
        let t1 = generate_organic_template(c, 1024, Base::Signed(7), 0.35, 0.35);
        let t2 = generate_organic_template(c, 1024, Base::Signed(7), 0.35, 0.35);
        assert_eq!(t1, t2, "Templates must be deterministic");
    }

    #[test]
    fn test_same_domain_correlation() {
        let d = 4096;
        let base = Base::Signed(7);
        // eros domain: indices 16..24
        let t_arousal = generate_organic_template(&CONCEPTS[17], d, base, 0.35, 0.35);
        let t_desire = generate_organic_template(&CONCEPTS[18], d, base, 0.35, 0.35);
        let corr = pearson_correlation(&t_arousal, &t_desire);
        assert!(corr > 0.15,
            "Same-domain correlation should be > 0.15, got {}", corr);
    }

    #[test]
    fn test_tau_proximity_correlation() {
        let d = 4096;
        let base = Base::Signed(7);
        // feel.curiosity (tau=0x40) and want.understood (tau=0x50) — nearby tau
        let t_feel = generate_organic_template(&CONCEPTS[32], d, base, 0.35, 0.35);
        let t_want = generate_organic_template(&CONCEPTS[24], d, base, 0.35, 0.35);
        let corr_near = pearson_correlation(&t_feel, &t_want);

        // feel.curiosity (tau=0x40) and eros.awareness (tau=0xE0) — distant tau
        let t_eros = generate_organic_template(&CONCEPTS[16], d, base, 0.35, 0.35);
        let corr_far = pearson_correlation(&t_feel, &t_eros);

        assert!(corr_near > corr_far,
            "Nearby tau should correlate more: near={}, far={}", corr_near, corr_far);
    }

    #[test]
    fn test_eros_isolation() {
        let d = 4096;
        let base = Base::Signed(7);
        // eros.desire (tau=0xE1) vs cog.thinking (tau=0x92) — far tau, different domain
        let t_eros = generate_organic_template(&CONCEPTS[18], d, base, 0.35, 0.35);
        let t_cog = generate_organic_template(&CONCEPTS[40], d, base, 0.35, 0.35);
        let corr = pearson_correlation(&t_eros, &t_cog).abs();
        assert!(corr < 0.10,
            "Eros-Cog cross-domain should be near zero, got {}", corr);
    }

    #[test]
    fn test_all_52_templates_generated() {
        let templates = generate_all_organic_templates(1024, Base::Signed(7), 0.35, 0.35);
        assert_eq!(templates.len(), K_TOTAL);
        for t in &templates {
            assert_eq!(t.len(), 1024);
        }
    }

    #[test]
    fn test_single_concept_recovery() {
        let d = 8192;
        let mut h = GhostHolograph::new(d);
        h.store(&[0], &[1.0]);
        let results = h.read_all();
        let r = &results[0];
        assert!(r.was_stored);
        assert!((r.readback - 1.0).abs() < 0.2,
            "Recovery should be near 1.0, got {}", r.readback);
    }

    #[test]
    fn test_store_8_recovery() {
        let d = 8192;
        let mut h = GhostHolograph::new(d);
        let indices: Vec<usize> = (16..24).collect();
        let amplitudes = vec![1.0f32; 8];
        h.store(&indices, &amplitudes);
        let results = h.read_all();

        for &idx in &indices {
            let r = &results[idx];
            assert!(r.readback > 0.3,
                "Recovery for {} should be > 0.3, got {}", r.name, r.readback);
        }
    }

    #[test]
    fn test_container_values_in_range() {
        let d = 4096;
        let mut h = GhostHolograph::new(d);
        let indices: Vec<usize> = (0..K_TOTAL).collect();
        let amplitudes = vec![1.0f32; K_TOTAL];
        h.store(&indices, &amplitudes);

        for &v in &h.container {
            assert!(v >= -3 && v <= 3, "Container out of range: {}", v);
        }
    }

    #[test]
    fn test_scenario_eros_counts() {
        let results = scenario_eros_only(2048);
        let stored: usize = results.iter().filter(|r| r.was_stored).count();
        assert_eq!(stored, 8);
        assert_eq!(results.len() - stored, 44);
    }

    #[test]
    fn test_scenario_warm_moment_counts() {
        let results = scenario_warm_moment(2048);
        let stored: usize = results.iter().filter(|r| r.was_stored).count();
        assert_eq!(stored, 5);
    }

    #[test]
    fn test_scenario_tension_counts() {
        let results = scenario_tension(2048);
        let stored: usize = results.iter().filter(|r| r.was_stored).count();
        assert_eq!(stored, 6);
    }

    #[test]
    fn test_scenario_growth_counts() {
        let results = scenario_growth(2048);
        let stored: usize = results.iter().filter(|r| r.was_stored).count();
        assert_eq!(stored, 5);
    }

    #[test]
    fn test_scenario_full_load_counts() {
        let results = scenario_full_load(2048);
        let stored: usize = results.iter().filter(|r| r.was_stored).count();
        assert_eq!(stored, 52);
    }

    #[test]
    fn test_stored_concepts_positive_readback() {
        let results = scenario_eros_only(4096);
        for r in &results {
            if r.was_stored {
                assert!(r.readback > 0.0,
                    "Stored {} should have positive readback, got {}", r.name, r.readback);
            }
        }
    }

    #[test]
    fn test_unstored_near_zero_at_high_d() {
        // At high D, unstored concepts in different domains should read ~0
        let d = 8192;
        let results = scenario_eros_only(d);
        // Check a concept far in tau-space: feel.curiosity (tau=0x40, far from eros 0xE0)
        let curiosity = &results[32];
        assert!(!curiosity.was_stored);
        assert!(curiosity.readback.abs() < 0.15,
            "Far unstored should be near 0 at D={}, got {}", d, curiosity.readback);
    }

    #[test]
    fn test_ghost_count_decreases_with_d() {
        let threshold = 0.05;
        let count = |d: usize| -> usize {
            scenario_eros_only(d).iter()
                .filter(|r| !r.was_stored && r.readback.abs() > threshold)
                .count()
        };
        let low = count(256);
        let high = count(8192);
        assert!(low >= high,
            "Ghost count should decrease with D: low_d={}, high_d={}", low, high);
    }

    #[test]
    fn test_ghost_matrix_dimensions() {
        let matrix = ghost_matrix(512);
        assert_eq!(matrix.len(), 8);
        for row in &matrix {
            assert_eq!(row.len(), 8);
        }
        for i in 0..8 {
            assert_eq!(matrix[i][i], 0.0, "Diagonal should be zero");
        }
    }

    #[test]
    fn test_print_ghost_table_no_panic() {
        let results = scenario_eros_only(512);
        print_ghost_table("Test", &results, 0.02);
    }

    #[test]
    fn test_concept_count() {
        assert_eq!(CONCEPTS.len(), K_TOTAL);
    }

    #[test]
    fn test_domain_boundaries_cover_all() {
        let mut covered = 0;
        for &(_, start, end) in DOMAINS {
            assert!(start <= end);
            assert_eq!(start, covered);
            covered = end;
        }
        assert_eq!(covered, K_TOTAL);
    }
}
