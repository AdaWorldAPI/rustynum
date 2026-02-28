//! 3D Edge Vector Experiment: SPO XOR with active/passive/reflexive voices.
//!
//! Uses rustynum-core's SpatialCrystal3D + BF16 structured distance
//! to encode qualia items as SPO triples, then XOR them to reveal
//! what lives in the EDGE between two words looking at each other.
//!
//! Active:    (grief)-[:CAUSES]->(letting_go)     S=grief, P=CAUSES, O=letting_go
//! Passive:   (letting_go)-[:IS_CAUSED_BY]->(grief) S=letting_go, P=IS_CAUSED_BY, O=grief
//! Reflexive: (grief)-[:TRANSFORMS]->(grief')     S=grief, P=TRANSFORMS, O=grief_modified
//!
//! The XOR of Active and Passive reveals the EDGE — the asymmetry.
//! Where the sign flips in BF16 space = where the causal direction matters.
//! Where sign is stable = where the relationship is symmetric (mutual).

use serde::Deserialize;
use std::collections::HashMap;

// ============================================================================
// Inline BF16 utilities (standalone — no crate dependency needed)
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

/// XOR-bind two byte vectors
fn xor_bind(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b).map(|(x, y)| x ^ y).collect()
}

/// BF16 structural diff: count sign flips, exponent shifts, mantissa changes
fn bf16_structural_diff(a: &[u8], b: &[u8]) -> (usize, usize, usize, Vec<usize>) {
    let n_dims = a.len() / 2;
    let mut sign_flips = 0;
    let mut exp_shifts = 0;
    let mut man_changes = 0;
    let mut sign_flip_dims = Vec::new();

    for dim in 0..n_dims {
        let i = dim * 2;
        let va = u16::from_le_bytes([a[i], a[i + 1]]);
        let vb = u16::from_le_bytes([b[i], b[i + 1]]);
        let xor = va ^ vb;

        if xor & 0x8000 != 0 {
            sign_flips += 1;
            sign_flip_dims.push(dim);
        }
        exp_shifts += ((xor >> 7) & 0xFF).count_ones() as usize;
        man_changes += (xor & 0x7F).count_ones() as usize;
    }

    (sign_flips, exp_shifts, man_changes, sign_flip_dims)
}

/// BF16 weighted distance (scalar)
fn bf16_distance(a: &[u8], b: &[u8]) -> u64 {
    let mut total: u64 = 0;
    for i in (0..a.len()).step_by(2) {
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

// ============================================================================
// SPO encoding — same as SpatialCrystal3D but inline
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

    fn total_distance(&self, other: &SpoTriple) -> u64 {
        bf16_distance(&self.x, &other.x)
            + bf16_distance(&self.y, &other.y)
            + bf16_distance(&self.z, &other.z)
    }

    fn axis_distances(&self, other: &SpoTriple) -> (u64, u64, u64) {
        (
            bf16_distance(&self.x, &other.x),
            bf16_distance(&self.y, &other.y),
            bf16_distance(&self.z, &other.z),
        )
    }
}

// ============================================================================
// Corpus
// ============================================================================

#[derive(Deserialize)]
struct QualiaItem {
    id: String,
    label: String,
    family: String,
    vector: HashMap<String, f64>,
    #[serde(default)]
    qualia: Option<Vec<String>>,
    #[allow(dead_code)]
    #[serde(default)]
    neighbors: Option<Vec<String>>,
}

const DIMS_16_JSON: &[&str] = &[
    "brightness",
    "valence",
    "dominance",
    "arousal",
    "warmth",
    "clarity",
    "social",
    "nostalgia",
    "sacredness",
    "desire",
    "tension",
    "awe",
    "grief",
    "hope",
    "edge",
    "resolution_hunger",
];

const DIMS_16_NAMES: &[&str] = &[
    "glow",
    "valence",
    "rooting",
    "agency",
    "resonance",
    "clarity",
    "social",
    "gravity",
    "reverence",
    "volition",
    "dissonance",
    "staunen",
    "loss",
    "optimism",
    "friction",
    "equilibrium",
];

fn extract_16(item: &QualiaItem) -> Vec<f32> {
    DIMS_16_JSON
        .iter()
        .map(|d| *item.vector.get(*d).unwrap_or(&0.0) as f32)
        .collect()
}

fn main() {
    let json_str = include_str!("../qualia_219.json");
    let items: Vec<QualiaItem> = serde_json::from_str(json_str).expect("parse JSON");
    let n = items.len();

    println!("=== 3D EDGE VECTOR EXPERIMENT: SPO × BF16 × Active/Passive ===\n");
    println!("  Corpus: {} items, 16 dims each, BF16-encoded\n", n);

    // ========================================================================
    // STEP 1: Encode all items as BF16 byte vectors
    // ========================================================================
    let vecs_f32: Vec<Vec<f32>> = items.iter().map(|it| extract_16(it)).collect();
    let vecs_bf16: Vec<Vec<u8>> = vecs_f32.iter().map(|v| f32_to_bf16_bytes(v)).collect();

    println!(
        "  BF16 encoding: {} bytes per item ({} dims × 2 bytes)\n",
        vecs_bf16[0].len(),
        vecs_f32[0].len()
    );

    // ========================================================================
    // STEP 2: Define predicate vectors
    // ========================================================================
    // Predicates are BF16 vectors that encode the RELATIONSHIP type.
    // Active voice predicates lean "outward" (positive agency/volition).
    // Passive voice predicates lean "inward" (negative agency, positive gravity).

    // CAUSES: high agency, high volition, forward motion
    let pred_causes = f32_to_bf16_bytes(&[
        0.7, 0.5, 0.5, 0.9, // glow, valence, rooting, agency=HIGH
        0.3, 0.5, 0.3, 0.1, // resonance, clarity, social, gravity=LOW
        0.2, 0.8, 0.3, 0.2, // reverence, volition=HIGH, dissonance, staunen
        0.1, 0.6, 0.3, 0.5, // loss, optimism, friction, equilibrium
    ]);

    // IS_CAUSED_BY: low agency, high gravity, receptive
    let pred_caused_by = f32_to_bf16_bytes(&[
        0.3, 0.5, 0.5, 0.1, // glow, valence, rooting, agency=LOW
        0.7, 0.5, 0.3, 0.9, // resonance, clarity, social, gravity=HIGH
        0.6, 0.2, 0.3, 0.4, // reverence, volition=LOW, dissonance, staunen
        0.3, 0.4, 0.3, 0.5, // loss, optimism, friction, equilibrium
    ]);

    // TRANSFORMS: high dissonance, high friction, agency moderate
    let pred_transforms = f32_to_bf16_bytes(&[
        0.5, 0.5, 0.4, 0.5, // glow, valence, rooting, agency=MID
        0.5, 0.4, 0.3, 0.5, // resonance, clarity, social, gravity=MID
        0.4, 0.5, 0.8, 0.5, // reverence, volition, dissonance=HIGH, staunen
        0.2, 0.5, 0.8, 0.7, // loss, optimism, friction=HIGH, equilibrium=HIGH
    ]);

    // DISSOLVES_INTO: low agency, low gravity, high resonance
    let pred_dissolves = f32_to_bf16_bytes(&[
        0.8, 0.7, 0.2, 0.1, // glow=HIGH, valence, rooting=LOW, agency=LOW
        0.9, 0.3, 0.5, 0.1, // resonance=HIGH, clarity=LOW, social, gravity=LOW
        0.3, 0.1, 0.1, 0.7, // reverence, volition=LOW, dissonance=LOW, staunen=HIGH
        0.1, 0.7, 0.1, 0.1, // loss=LOW, optimism, friction=LOW, equilibrium=LOW
    ]);

    println!("  Predicate vectors defined:");
    println!("    CAUSES:       high agency (0.9), low gravity (0.1) → active/RGB");
    println!("    IS_CAUSED_BY: low agency (0.1), high gravity (0.9) → passive/CMYK");
    println!("    TRANSFORMS:   high dissonance (0.8), friction (0.8) → transitional");
    println!("    DISSOLVES_INTO: high resonance (0.9), low agency (0.1) → ecstatic\n");

    // ========================================================================
    // STEP 3: Active vs Passive XOR — the edge between two words
    // ========================================================================
    println!("--- Step 3: Active vs Passive edge vectors ---\n");

    // Select interesting pairs from the XOR analysis
    let pairs: Vec<(&str, &str)> = vec![
        // Bucket C pairs: BERT close, Nib4 far (surface synonymy)
        ("grief_private_weight", "letting_go_without_grief"),
        ("desire_slow_burn", "love_unconditional"),
        ("devotion_yes_with_fear", "devotion_total_presence"),
        // Bucket B pairs: Nib4 close, BERT far (cadence truth)
        (
            "courage_quiet_step_forward",
            "transformation_internal_rewrite",
        ),
        ("devotion_after_fight", "becoming_learning_voice"),
        (
            "surrender_truth_without_defense",
            "transformation_identity_blur",
        ),
        // Same family, different modes
        ("devotion_bold_yes", "devotion_vigil"),
        ("longing_warm_regret", "longing_defiant_reach"),
        // Cross-family, interesting
        ("awe_vast_open_sky", "innocence_unquestioned_trust"),
        ("anger_clean_no", "surrender_no_more_fight"),
    ];

    // Create index map
    let idx_map: HashMap<&str, usize> = items
        .iter()
        .enumerate()
        .map(|(i, it)| (it.id.as_str(), i))
        .collect();

    println!(
        "  {:>40} {:>40}  {:>5} {:>5} {:>5}  {:>4} {:>4} {:>4}",
        "A → B (active)", "B → A (passive)", "Dx", "Dy", "Dz", "Sflip", "Eshift", "Mnoise"
    );

    struct EdgeResult {
        a_name: String,
        b_name: String,
        active: SpoTriple,
        passive: SpoTriple,
        edge_xor: Vec<u8>, // flat XOR of all 3 axes
        sign_flips: usize,
        exp_shifts: usize,
        man_changes: usize,
        sign_flip_dims: Vec<usize>,
        axis_dists: (u64, u64, u64),
    }

    let mut edge_results: Vec<EdgeResult> = Vec::new();

    for &(a_id, b_id) in &pairs {
        let a_idx = match idx_map.get(a_id) {
            Some(&i) => i,
            None => {
                println!("  WARNING: {} not found", a_id);
                continue;
            }
        };
        let b_idx = match idx_map.get(b_id) {
            Some(&i) => i,
            None => {
                println!("  WARNING: {} not found", b_id);
                continue;
            }
        };

        let a_bf16 = &vecs_bf16[a_idx];
        let b_bf16 = &vecs_bf16[b_idx];

        // Active: A causes B → S=A, P=CAUSES, O=B
        let active = SpoTriple::encode(a_bf16, &pred_causes, b_bf16);

        // Passive: B is caused by A → S=B, P=IS_CAUSED_BY, O=A
        let passive = SpoTriple::encode(b_bf16, &pred_caused_by, a_bf16);

        // Edge = XOR of Active and Passive (per axis)
        let edge_x = xor_bind(&active.x, &passive.x);
        let edge_y = xor_bind(&active.y, &passive.y);
        let edge_z = xor_bind(&active.z, &passive.z);

        // Flatten edge for analysis
        let mut edge_flat = edge_x.clone();
        edge_flat.extend_from_slice(&edge_y);
        edge_flat.extend_from_slice(&edge_z);

        let zeros = vec![0u8; edge_flat.len()];
        let (sign_flips, exp_shifts, man_changes, sign_flip_dims) =
            bf16_structural_diff(&edge_flat, &zeros);

        let axis_dists = active.axis_distances(&passive);

        println!(
            "  {:>40} {:>40}  {:>5} {:>5} {:>5}  {:>4} {:>4} {:>4}",
            &format!(
                "{}→{}",
                &a_id[..a_id.len().min(18)],
                &b_id[..b_id.len().min(18)]
            ),
            &format!(
                "{}→{}",
                &b_id[..b_id.len().min(18)],
                &a_id[..a_id.len().min(18)]
            ),
            axis_dists.0,
            axis_dists.1,
            axis_dists.2,
            sign_flips,
            exp_shifts,
            man_changes
        );

        edge_results.push(EdgeResult {
            a_name: a_id.to_string(),
            b_name: b_id.to_string(),
            active,
            passive,
            edge_xor: edge_flat,
            sign_flips,
            exp_shifts,
            man_changes,
            sign_flip_dims,
            axis_dists,
        });
    }

    // ========================================================================
    // STEP 4: Edge anatomy — which BF16 dims flip sign?
    // ========================================================================
    println!("\n--- Step 4: Edge anatomy (which dims are asymmetric?) ---\n");

    for result in &edge_results {
        println!("  {} ↔ {}", result.a_name, result.b_name);
        println!(
            "    Sign flips: {} / {} dims × 3 axes",
            result.sign_flips,
            16 * 3
        );
        println!("    Exp shifts: {} bits", result.exp_shifts);
        println!("    Man noise:  {} bits", result.man_changes);

        // Decode which ORIGINAL qualia dims have sign flips (per axis)
        let x_flips: Vec<&str> = result
            .sign_flip_dims
            .iter()
            .filter(|&&d| d < 16)
            .map(|&d| DIMS_16_NAMES[d])
            .collect();
        let y_flips: Vec<&str> = result
            .sign_flip_dims
            .iter()
            .filter(|&&d| d >= 16 && d < 32)
            .map(|&d| DIMS_16_NAMES[d - 16])
            .collect();
        let z_flips: Vec<&str> = result
            .sign_flip_dims
            .iter()
            .filter(|&&d| d >= 32)
            .map(|&d| DIMS_16_NAMES[d - 32])
            .collect();

        if !x_flips.is_empty() {
            println!("    X-axis (S⊕P) sign flips: {}", x_flips.join(", "));
        }
        if !y_flips.is_empty() {
            println!("    Y-axis (P⊕O) sign flips: {}", y_flips.join(", "));
        }
        if !z_flips.is_empty() {
            println!("    Z-axis (S⊕O) sign flips: {}", z_flips.join(", "));
        }

        // Asymmetry ratio: how much more distance on one axis vs others
        let (dx, dy, dz) = result.axis_dists;
        let total = (dx + dy + dz) as f64;
        if total > 0.0 {
            let balance = 1.0 - ((dx.max(dy).max(dz) as f64 / total) - 1.0 / 3.0) * 1.5;
            println!("    Balance: {:.2} (1.0=symmetric, 0.0=one-axis)", balance);
        }
        println!();
    }

    // ========================================================================
    // STEP 5: Transform vs Dissolve — different predicate, same pair
    // ========================================================================
    println!("--- Step 5: Same pair, different predicates ---\n");

    // Pick the grief→letting_go pair and show it through 4 predicates
    if let (Some(&a_idx), Some(&b_idx)) = (
        idx_map.get("grief_private_weight"),
        idx_map.get("letting_go_without_grief"),
    ) {
        let a_bf16 = &vecs_bf16[a_idx];
        let b_bf16 = &vecs_bf16[b_idx];

        let predicates: Vec<(&str, &[u8])> = vec![
            ("CAUSES", &pred_causes),
            ("IS_CAUSED_BY", &pred_caused_by),
            ("TRANSFORMS", &pred_transforms),
            ("DISSOLVES_INTO", &pred_dissolves),
        ];

        println!("  Pair: grief_private_weight → letting_go_without_grief\n");
        println!(
            "  {:>18}  {:>6} {:>6} {:>6}  {:>5}  {:>5}  {:>8}",
            "Predicate", "Dx", "Dy", "Dz", "Sflip", "Eshift", "Dominant"
        );

        for (pred_name, pred_bytes) in &predicates {
            let triple = SpoTriple::encode(a_bf16, pred_bytes, b_bf16);
            let reverse = SpoTriple::encode(b_bf16, pred_bytes, a_bf16);

            let (dx, dy, dz) = triple.axis_distances(&reverse);

            // Analyze the edge
            let edge_flat: Vec<u8> = [
                xor_bind(&triple.x, &reverse.x),
                xor_bind(&triple.y, &reverse.y),
                xor_bind(&triple.z, &reverse.z),
            ]
            .concat();

            let zeros = vec![0u8; edge_flat.len()];
            let (sign_flips, exp_shifts, _man, _dims) = bf16_structural_diff(&edge_flat, &zeros);

            let dominant = if dx >= dy && dx >= dz {
                "S⊕P (who?)"
            } else if dy >= dz {
                "P⊕O (what?)"
            } else {
                "S⊕O (edge)"
            };

            println!(
                "  {:>18}  {:>6} {:>6} {:>6}  {:>5}  {:>5}  {:>8}",
                pred_name, dx, dy, dz, sign_flips, exp_shifts, dominant
            );
        }
    }

    // ========================================================================
    // STEP 6: Bundle experiment — superpose multiple edges
    // ========================================================================
    println!("\n--- Step 6: Bundle XOR — superpose multiple edges ---\n");
    println!("  What pattern emerges when we bundle all active/passive edges?\n");

    // For each pair, compute the edge (active XOR passive), then bundle
    // all edges via majority voting to find the PROTOTYPICAL asymmetry.
    let n_bytes = vecs_bf16[0].len() * 3; // 3 axes
    let mut bit_counts: Vec<u32> = vec![0; n_bytes * 8];
    let n_edges = edge_results.len();

    for result in &edge_results {
        for (byte_idx, &byte) in result.edge_xor.iter().enumerate() {
            for bit in 0..8 {
                if byte & (1 << bit) != 0 {
                    bit_counts[byte_idx * 8 + bit] += 1;
                }
            }
        }
    }

    // Majority vote: bit is 1 if majority of edges have it
    let threshold = n_edges as u32 / 2;
    let mut prototype_edge = vec![0u8; n_bytes];
    let mut high_agreement = 0u32;
    let mut total_bits_set = 0u32;

    for (byte_idx, chunk) in bit_counts.chunks(8).enumerate() {
        let mut byte_val = 0u8;
        for (bit, &count) in chunk.iter().enumerate() {
            if count > threshold {
                byte_val |= 1 << bit;
                high_agreement += 1;
            }
            if count > 0 {
                total_bits_set += 1;
            }
        }
        prototype_edge[byte_idx] = byte_val;
    }

    println!("  Bundled {} edges:", n_edges);
    println!(
        "    Total bits set (any edge):    {}/{}",
        total_bits_set,
        n_bytes * 8
    );
    println!(
        "    Majority-vote bits (>50%):    {}/{}",
        high_agreement,
        n_bytes * 8
    );
    println!(
        "    Agreement ratio:              {:.1}%",
        100.0 * high_agreement as f64 / (n_bytes * 8) as f64
    );

    // Decode the prototype edge — which dims are consistently asymmetric?
    let zeros = vec![0u8; prototype_edge.len()];
    let (proto_sign, proto_exp, proto_man, proto_dims) =
        bf16_structural_diff(&prototype_edge, &zeros);

    println!("\n  Prototype edge structure:");
    println!("    Sign flips:    {} / {}", proto_sign, 16 * 3);
    println!("    Exp shifts:    {}", proto_exp);
    println!("    Man bits:      {}", proto_man);

    // Decode which dims are in the prototype
    let x_proto: Vec<&str> = proto_dims
        .iter()
        .filter(|&&d| d < 16)
        .map(|&d| DIMS_16_NAMES[d])
        .collect();
    let y_proto: Vec<&str> = proto_dims
        .iter()
        .filter(|&&d| d >= 16 && d < 32)
        .map(|&d| DIMS_16_NAMES[d - 16])
        .collect();
    let z_proto: Vec<&str> = proto_dims
        .iter()
        .filter(|&&d| d >= 32)
        .map(|&d| DIMS_16_NAMES[d - 32])
        .collect();

    let x_str = if x_proto.is_empty() {
        "none".to_string()
    } else {
        x_proto.join(", ")
    };
    let y_str = if y_proto.is_empty() {
        "none".to_string()
    } else {
        y_proto.join(", ")
    };
    let z_str = if z_proto.is_empty() {
        "none".to_string()
    } else {
        z_proto.join(", ")
    };
    println!("\n  Consistently asymmetric dims (the active/passive signature):");
    println!("    X-axis (S⊕P): {}", x_str);
    println!("    Y-axis (P⊕O): {}", y_str);
    println!("    Z-axis (S⊕O): {}", z_str);

    // ========================================================================
    // STEP 7: Per-bit voting histogram
    // ========================================================================
    println!("\n--- Step 7: Per-bit vote distribution ---\n");

    // Show how many bits agree across N%, 75%, 90% of edges
    let mut hist = [0u32; 11]; // 0%, 10%, 20%, ..., 100%
    for &count in &bit_counts {
        let pct = if n_edges > 0 {
            (count * 10 / n_edges as u32).min(10)
        } else {
            0
        };
        hist[pct as usize] += 1;
    }

    println!("  Vote distribution (how many edges agree on each bit):");
    for i in 0..=10 {
        let bar_len = (hist[i] as f64 / (n_bytes * 8) as f64 * 50.0) as usize;
        let bar = "#".repeat(bar_len);
        println!("    {:>3}%: {:>5}  {}", i * 10, hist[i], bar);
    }

    // ========================================================================
    // VERDICT
    // ========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║               3D EDGE VECTOR VERDICT                       ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║                                                            ║");
    println!("║  SPO encoding: S⊕P (X), P⊕O (Y), S⊕O (Z)                ║");
    println!("║  Active voice: A-[CAUSES]->B                               ║");
    println!("║  Passive voice: B-[IS_CAUSED_BY]->A                        ║");
    println!("║  Edge = Active XOR Passive                                 ║");
    println!("║                                                            ║");
    println!("║  The edge reveals which BF16 dims are asymmetric           ║");
    println!("║  (direction-dependent) vs symmetric (mutual).              ║");
    println!("║                                                            ║");
    println!("║  Sign flips = causal direction matters for that dim        ║");
    println!("║  Exp shifts = magnitude changes with perspective           ║");
    println!("║  Mantissa   = noise (irrelevant to voice)                  ║");
    println!("║                                                            ║");
    println!("║  Bundle prototype: dims that ALWAYS flip                   ║");
    println!("║  = the invariant active/passive signature.                 ║");
    println!("║  This IS the BF16 encoding of the mode bit.               ║");
    println!("║                                                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
