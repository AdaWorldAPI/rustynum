use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rustynum_core::fingerprint::Fingerprint;
use rustynum_core::node::{Node, SPO, SP_, S__};
use rustynum_core::plane::{distance_slices, Distance, Plane};
use rustynum_core::rng::SplitMix64;

// ============================================================================
// Helpers
// ============================================================================

/// Build a Plane with N encounters of deterministic random data.
/// Cache is pre-warmed via bits()/alpha() so benchmarks measure pure distance.
fn seeded_plane(seed: u64, encounters: u32) -> Plane {
    let mut rng = SplitMix64::new(seed);
    let mut plane = Plane::new();
    for _ in 0..encounters {
        let mut words = [0u64; 256];
        for w in words.iter_mut() {
            *w = rng.next_u64();
        }
        let fp = Fingerprint::<256>::from_words(words);
        plane.encounter_bits(&fp);
    }
    // Pre-warm cache so benchmark measures pure distance, not lazy refresh.
    let _ = plane.bits();
    let _ = plane.alpha();
    plane
}

fn random_fingerprint(seed: u64) -> Fingerprint<256> {
    let mut rng = SplitMix64::new(seed);
    let mut words = [0u64; 256];
    for w in words.iter_mut() {
        *w = rng.next_u64();
    }
    Fingerprint::<256>::from_words(words)
}

// ============================================================================
// Plane.distance() — the 4-cycle bullet
// ============================================================================

fn bench_plane_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Plane Distance");

    // ---- Single Plane distance (the 4-cycle claim) ----
    let mut a = seeded_plane(42, 5);
    let mut b = seeded_plane(43, 5);

    group.bench_function("plane_distance_single", |bencher| {
        bencher.iter(|| black_box(a.distance(&mut b)))
    });

    // ---- Raw distance_slices (skips ensure_cache overhead) ----
    // This isolates the SIMD kernel: XOR + AND + popcount on 2048 bytes.
    let a_bits = a.bits().as_bytes().to_vec();
    let a_alpha = a.alpha().as_bytes().to_vec();
    let b_bits = b.bits().as_bytes().to_vec();
    let b_alpha = b.alpha().as_bytes().to_vec();

    group.bench_function("distance_slices_2048B", |bencher| {
        bencher.iter(|| {
            black_box(distance_slices(
                black_box(&a_bits),
                black_box(&a_alpha),
                black_box(&b_bits),
                black_box(&b_alpha),
            ))
        })
    });

    // ---- Scalar baseline for comparison ----
    group.bench_function("distance_scalar_baseline_2048B", |bencher| {
        bencher.iter(|| {
            let len = a_bits.len();
            let mut disagreement = 0u32;
            let mut overlap = 0u32;
            let mut penalty = 0u32;
            for i in 0..len {
                let xor = a_bits[i] ^ b_bits[i];
                let shared = a_alpha[i] & b_alpha[i];
                let masked = xor & shared;
                disagreement += masked.count_ones();
                overlap += shared.count_ones();
                penalty += (!a_alpha[i]).count_ones();
            }
            black_box((disagreement, overlap, penalty))
        })
    });

    // ---- Fingerprint Hamming distance (raw XOR+popcount path) ----
    let fp_a = random_fingerprint(100);
    let fp_b = random_fingerprint(200);

    group.bench_function("fingerprint_hamming_2048B", |bencher| {
        bencher.iter(|| black_box(fp_a.hamming_distance(&fp_b)))
    });

    group.bench_function("fingerprint_popcount_2048B", |bencher| {
        bencher.iter(|| black_box(fp_a.popcount()))
    });

    group.finish();
}

// ============================================================================
// Node.distance() — 4/8/12 cycle scaling
// ============================================================================

fn bench_node_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Node Distance");

    let mut a = Node::random(42);
    let mut b = Node::random(43);

    // Pre-warm all plane caches via public API
    let _ = a.s.bits();
    let _ = a.s.alpha();
    let _ = a.p.bits();
    let _ = a.p.alpha();
    let _ = a.o.bits();
    let _ = a.o.alpha();
    let _ = b.s.bits();
    let _ = b.s.alpha();
    let _ = b.p.bits();
    let _ = b.p.alpha();
    let _ = b.o.bits();
    let _ = b.o.alpha();

    // S__ = 1 plane = ~4 cycles
    group.bench_function("node_S__  (1 plane)", |bencher| {
        bencher.iter(|| black_box(a.distance(&mut b, S__)))
    });

    // SP_ = 2 planes = ~8 cycles
    group.bench_function("node_SP_  (2 planes)", |bencher| {
        bencher.iter(|| black_box(a.distance(&mut b, SP_)))
    });

    // SPO = 3 planes = ~12 cycles
    group.bench_function("node_SPO  (3 planes)", |bencher| {
        bencher.iter(|| black_box(a.distance(&mut b, SPO)))
    });

    group.finish();
}

// ============================================================================
// Scaling: how does distance_slices scale with data size?
// ============================================================================

fn bench_distance_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("Distance Scaling");

    for &n_words in &[16, 64, 128, 256, 512, 1024] {
        let n_bytes = n_words * 8;
        let mut rng = SplitMix64::new(n_words as u64);
        let a_bits: Vec<u8> = (0..n_bytes).map(|_| rng.next_u64() as u8).collect();
        let a_alpha: Vec<u8> = (0..n_bytes).map(|_| rng.next_u64() as u8).collect();
        let b_bits: Vec<u8> = (0..n_bytes).map(|_| rng.next_u64() as u8).collect();
        let b_alpha: Vec<u8> = (0..n_bytes).map(|_| rng.next_u64() as u8).collect();

        group.bench_with_input(
            BenchmarkId::new("distance_slices", format!("{}B", n_bytes)),
            &n_bytes,
            |bencher, _| {
                bencher.iter(|| {
                    black_box(distance_slices(
                        black_box(&a_bits),
                        black_box(&a_alpha),
                        black_box(&b_bits),
                        black_box(&b_alpha),
                    ))
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// Comparison: Plane distance vs raw Hamming on same data size (2048B)
// ============================================================================

fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Plane vs Raw Hamming (2048B)");

    let fp_a = random_fingerprint(300);
    let fp_b = random_fingerprint(400);

    // Raw Hamming: 1 XOR + 1 popcount per chunk
    group.bench_function("hamming_distance", |bencher| {
        bencher.iter(|| black_box(fp_a.hamming_distance(&fp_b)))
    });

    // Plane distance: XOR + AND + 3x popcount (alpha-aware)
    let mut a = seeded_plane(42, 5);
    let mut b = seeded_plane(43, 5);

    group.bench_function("plane_distance_alpha_aware", |bencher| {
        bencher.iter(|| black_box(a.distance(&mut b)))
    });

    group.finish();
}

// ============================================================================
// Batch: how fast can we scan N nodes? (linear scan, nearest neighbor)
// ============================================================================

fn bench_batch_node_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("Batch Node Distance");
    group.sample_size(20);

    for &count in &[10, 100, 1000] {
        let mut query = Node::random(0);
        let _ = query.s.bits();
        let _ = query.p.bits();
        let _ = query.o.bits();

        let mut db: Vec<Node> = (1..=count as u64)
            .map(|seed| {
                let mut n = Node::random(seed);
                let _ = n.s.bits();
                let _ = n.p.bits();
                let _ = n.o.bits();
                n
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::new("SPO_scan", count),
            &count,
            |bencher, &_| {
                bencher.iter(|| {
                    let mut best = u32::MAX;
                    for node in db.iter_mut() {
                        if let Distance::Measured { disagreement, .. } = query.distance(node, SPO) {
                            if disagreement < best {
                                best = disagreement;
                            }
                        }
                    }
                    black_box(best)
                })
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_plane_distance,
    bench_node_distance,
    bench_distance_scaling,
    bench_comparison,
    bench_batch_node_distance,
);
criterion_main!(benches);
