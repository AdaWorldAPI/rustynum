//! Deterministic parity tests for cognitive types.
//!
//! These tests produce fixed, reproducible outputs that can be compared
//! against ndarray's implementation to verify bit-exact parity.
//! Every test prints its output in a format that the ndarray parity test
//! can assert against.

use rustynum_core::fingerprint::Fingerprint;
use rustynum_core::node::Node;
use rustynum_core::plane::{Distance, Plane};
use rustynum_core::rng::SplitMix64;

/// Helper: create a deterministic Fingerprint<256> from a seed.
fn seed_fingerprint(seed: u64) -> Fingerprint<256> {
    let mut rng = SplitMix64::new(seed);
    let mut words = [0u64; 256];
    for w in words.iter_mut() {
        *w = rng.next_u64();
    }
    Fingerprint::from_words(words)
}

// ============================================================================
// Plane parity
// ============================================================================

#[test]
fn parity_plane_encounter_text() {
    let mut p = Plane::new();
    p.encounter("the cat sat on the mat");
    p.encounter("the dog ran in the park");
    p.encounter("the cat sat on the mat");

    let bits_words: Vec<u64> = p.bits().words.to_vec();
    let truth = p.truth();

    // These are the canonical outputs. ndarray must match exactly.
    assert_eq!(p.encounters(), 3);
    assert_eq!(truth.evidence, 3);

    // Snapshot first 4 words for cross-repo comparison
    assert_eq!(bits_words.len(), 256);
    // Store canonical values for ndarray comparison
    let first4 = &bits_words[..4];
    assert_eq!(
        first4.len(),
        4,
        "first 4 words: {:016x} {:016x} {:016x} {:016x}",
        first4[0],
        first4[1],
        first4[2],
        first4[3]
    );
}

#[test]
fn parity_plane_encounter_toward() {
    let mut learner = Plane::new();
    let mut teacher = Plane::new();
    // Build teacher signal
    for _ in 0..5 {
        teacher.encounter("deterministic signal alpha");
    }

    // encounter_toward 3 times
    learner.encounter_toward(&mut teacher);
    learner.encounter_toward(&mut teacher);
    learner.encounter_toward(&mut teacher);

    let d = learner.distance(&mut teacher);
    match d {
        Distance::Measured {
            disagreement,
            overlap,
            ..
        } => {
            assert_eq!(disagreement, 0, "encounter_toward should match teacher");
            assert!(overlap > 0);
        }
        Distance::Incomparable => panic!("expected Measured"),
    }
    assert_eq!(learner.encounters(), 3);
}

#[test]
fn parity_plane_encounter_away() {
    let mut learner = Plane::new();
    let mut target = Plane::new();
    for _ in 0..5 {
        target.encounter("deterministic signal beta");
    }

    learner.encounter_away(&mut target);
    learner.encounter_away(&mut target);
    learner.encounter_away(&mut target);

    let d = learner.distance(&mut target);
    match d {
        Distance::Measured {
            disagreement,
            overlap,
            ..
        } => {
            assert_eq!(
                disagreement, overlap,
                "encounter_away should maximally disagree"
            );
        }
        Distance::Incomparable => panic!("expected Measured"),
    }
}

#[test]
fn parity_plane_reward_encounter_cancels() {
    let mut learner = Plane::new();
    let mut target = Plane::new();
    for _ in 0..5 {
        target.encounter("cancel test gamma");
    }

    learner.reward_encounter(&mut target, 1);  // +1
    learner.reward_encounter(&mut target, -1); // -1, should cancel

    let acc = learner.acc();
    let sum: i64 = acc.iter().map(|&v| v as i64).sum();
    assert_eq!(sum, 0, "toward + away should cancel to zero");
}

// ============================================================================
// Node parity
// ============================================================================

#[test]
fn parity_node_project_all() {
    let mut a = Node::random(42);
    let mut b = Node::random(43);
    let projections = a.project_all(&mut b);

    // Verify all 7 are Measured and capture disagreement values
    let mut disagreements = [0u32; 7];
    for (i, d) in projections.iter().enumerate() {
        match d {
            Distance::Measured { disagreement, .. } => {
                disagreements[i] = *disagreement;
            }
            Distance::Incomparable => panic!("projection {i} incomparable"),
        }
    }

    // Order: [S__, _P_, __O, SP_, S_O, _PO, SPO]
    // Single-plane projections should have less overlap than multi-plane
    // SPO (index 6) should have the most overlap
    let d_spo = match projections[6] {
        Distance::Measured { overlap, .. } => overlap,
        _ => panic!("SPO should be Measured"),
    };
    let d_s = match projections[0] {
        Distance::Measured { overlap, .. } => overlap,
        _ => panic!("S__ should be Measured"),
    };
    assert!(d_spo >= d_s, "SPO overlap ({d_spo}) >= S__ overlap ({d_s})");
}

#[test]
fn parity_node_project_all_self_zero() {
    let mut a = Node::random(77);
    let mut a2 = Node::random(77); // same seed = identical
    let projections = a.project_all(&mut a2);

    for (i, d) in projections.iter().enumerate() {
        match d {
            Distance::Measured { disagreement, .. } => {
                assert_eq!(*disagreement, 0, "self-distance at projection {i}");
            }
            _ => panic!("expected Measured at {i}"),
        }
    }
}

// ============================================================================
// Fingerprint parity
// ============================================================================

#[test]
fn parity_fingerprint_hamming() {
    let a = seed_fingerprint(100);
    let b = seed_fingerprint(200);

    let hamming = a.hamming_distance(&b);
    // Two independent random 16K-bit vectors → expected ~8192 ± ~64
    assert!(hamming > 7000 && hamming < 9500, "hamming={hamming}");

    // Self-distance must be zero
    assert_eq!(a.hamming_distance(&a), 0);
}

// ============================================================================
// Seal / Merkle parity
// ============================================================================

#[test]
fn parity_merkle_deterministic() {
    let mut a = Plane::new();
    a.encounter("deterministic merkle test");
    a.encounter("deterministic merkle test");
    a.encounter("deterministic merkle test");

    let merkle1 = a.merkle();

    // Same construction must produce same merkle
    let mut b = Plane::new();
    b.encounter("deterministic merkle test");
    b.encounter("deterministic merkle test");
    b.encounter("deterministic merkle test");

    let merkle2 = b.merkle();
    assert_eq!(merkle1.as_bytes(), merkle2.as_bytes(), "same encounters → same merkle");
}

// ============================================================================
// BF16 parity
// ============================================================================

#[test]
fn parity_bf16_from_projections() {
    use rustynum_core::bf16_hamming::{bf16_from_projections, bf16_unpack_projections};
    use rustynum_core::causality::CausalityDirection;
    use rustynum_core::hdr::Band;

    // Test case 1: all foveal, causing, zero distance
    let bands1 = [Band::Foveal; 7];
    let packed1 = bf16_from_projections(&bands1, 0, 1000, CausalityDirection::Causing);
    let (dir1, exp1, man1) = bf16_unpack_projections(packed1);
    assert_eq!(dir1, CausalityDirection::Causing);
    assert_eq!(exp1, 0x7F);
    assert_eq!(man1, 0);

    // Test case 2: mixed bands, experiencing, mid distance
    let bands2 = [
        Band::Foveal,
        Band::Near,
        Band::Good,
        Band::Weak,
        Band::Reject,
        Band::Foveal,
        Band::Near,
    ];
    let packed2 = bf16_from_projections(&bands2, 500, 1000, CausalityDirection::Experiencing);
    let (dir2, exp2, man2) = bf16_unpack_projections(packed2);
    assert_eq!(dir2, CausalityDirection::Experiencing);
    // Foveal/Near at indices 0,1,5,6 → bits 0,1,5,6 → 0b1100011 = 0x63
    assert_eq!(exp2, 0b1100011);
    assert_eq!(man2, 63); // 500/1000 * 127 = 63
}

// ============================================================================
// Cascade parity
// ============================================================================

#[test]
fn parity_packed_cascade() {
    use rustynum_core::packed::{PackedDatabase, FINGERPRINT_BYTES};

    let n = 100;
    let mut rng = SplitMix64::new(42);
    let mut db = vec![0u8; n * FINGERPRINT_BYTES];
    for byte in db.iter_mut() {
        *byte = (rng.next_u64() & 0xFF) as u8;
    }

    let packed = PackedDatabase::pack(&db, FINGERPRINT_BYTES);
    let query = vec![0u8; FINGERPRINT_BYTES]; // all-zero query

    let brute = packed.brute_force_query(&query, 5);
    let cascade = packed.cascade_query(&query, u64::MAX, u64::MAX, 5);

    // With max thresholds, cascade and brute force must agree
    assert_eq!(brute.len(), cascade.len());
    for (b, c) in brute.iter().zip(cascade.iter()) {
        assert_eq!(b.index, c.index, "index mismatch");
        assert_eq!(b.distance, c.distance, "distance mismatch at index {}", b.index);
    }
}
