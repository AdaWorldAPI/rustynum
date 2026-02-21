use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rustynum_rs::NumArrayU8;

/// All vector sizes we benchmark: 8K, 16K, 32K, 64K bytes
const VEC_SIZES: &[usize] = &[8192, 16384, 32768, 65536];

fn create_random_vector(seed: u64, len: usize) -> Vec<u8> {
    // Simple LCG for reproducible pseudo-random data
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (state >> 33) as u8
        })
        .collect()
}

fn bench_bind(c: &mut Criterion) {
    let mut group = c.benchmark_group("HDC Bind (XOR)");

    for &vec_len in VEC_SIZES {
        let a = NumArrayU8::new(create_random_vector(42, vec_len));
        let b = NumArrayU8::new(create_random_vector(123, vec_len));

        group.throughput(Throughput::Bytes(vec_len as u64));
        group.bench_with_input(
            BenchmarkId::new("xor", vec_len),
            &vec_len,
            |bencher, &_| bencher.iter(|| black_box(&a) ^ black_box(&b)),
        );
    }

    group.finish();
}

fn bench_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("HDC Distance (Hamming)");

    for &vec_len in VEC_SIZES {
        let a = NumArrayU8::new(create_random_vector(42, vec_len));
        let b = NumArrayU8::new(create_random_vector(123, vec_len));

        group.throughput(Throughput::Bytes(vec_len as u64));
        group.bench_with_input(
            BenchmarkId::new("hamming", vec_len),
            &vec_len,
            |bencher, &_| bencher.iter(|| black_box(a.hamming_distance(&b))),
        );
    }

    group.finish();
}

fn bench_permute(c: &mut Criterion) {
    let mut group = c.benchmark_group("HDC Permute");

    for &vec_len in VEC_SIZES {
        let v = NumArrayU8::new(create_random_vector(42, vec_len));

        group.throughput(Throughput::Bytes(vec_len as u64));
        group.bench_with_input(
            BenchmarkId::new("k=1", vec_len),
            &vec_len,
            |bencher, &_| bencher.iter(|| black_box(&v).permute(black_box(1))),
        );
    }

    group.finish();
}

fn bench_bundle(c: &mut Criterion) {
    let mut group = c.benchmark_group("HDC Bundle (Majority Vote)");

    for &vec_len in VEC_SIZES {
        for &count in &[5, 16, 64, 256, 1024] {
            let vectors: Vec<NumArrayU8> = (0..count)
                .map(|i| NumArrayU8::new(create_random_vector(i as u64, vec_len)))
                .collect();
            let vec_refs: Vec<&NumArrayU8> = vectors.iter().collect();

            group.throughput(Throughput::Bytes((vec_len * count) as u64));

            group.bench_with_input(
                BenchmarkId::new(format!("ripple_{}", vec_len), count),
                &count,
                |bencher, &_| bencher.iter(|| NumArrayU8::bundle(black_box(&vec_refs))),
            );

            // Naive baseline only for 8192 bytes to keep benchmark time reasonable
            if vec_len == 8192 {
                group.bench_with_input(
                    BenchmarkId::new("naive_8192", count),
                    &count,
                    |bencher, &_| {
                        bencher.iter(|| {
                            let len = vec_len;
                            let n = vec_refs.len();
                            let threshold = n / 2;
                            let mut out = vec![0u8; len];
                            for byte_idx in 0..len {
                                let mut result_byte = 0u8;
                                for bit in 0..8u8 {
                                    let mut count = 0u32;
                                    for v in vec_refs.iter() {
                                        count +=
                                            ((v.get_data()[byte_idx] >> bit) & 1) as u32;
                                    }
                                    if count as usize > threshold {
                                        result_byte |= 1 << bit;
                                    }
                                }
                                out[byte_idx] = result_byte;
                            }
                            black_box(out)
                        })
                    },
                );
            }
        }
    }

    group.finish();
}

fn bench_edge_encode_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("HDC Edge Encode/Decode");

    for &vec_len in &[8192usize, 65536] {
        let src = NumArrayU8::new(create_random_vector(1, vec_len));
        let rel = NumArrayU8::new(create_random_vector(2, vec_len));
        let tgt = NumArrayU8::new(create_random_vector(3, vec_len));

        group.throughput(Throughput::Bytes(vec_len as u64 * 3));

        group.bench_with_input(
            BenchmarkId::new("encode", vec_len),
            &vec_len,
            |bencher, &_| {
                bencher.iter(|| {
                    let perm_rel = black_box(&rel).permute(1);
                    let perm_tgt = black_box(&tgt).permute(2);
                    let edge = &(black_box(&src) ^ &perm_rel) ^ &perm_tgt;
                    black_box(edge)
                })
            },
        );

        let perm_rel = rel.permute(1);
        let perm_tgt = tgt.permute(2);
        let edge = &(&src ^ &perm_rel) ^ &perm_tgt;
        let total_bits = vec_len * 8;

        group.bench_with_input(
            BenchmarkId::new("decode_target", vec_len),
            &vec_len,
            |bencher, &_| {
                bencher.iter(|| {
                    let recovered_perm =
                        &(black_box(&edge) ^ black_box(&src)) ^ black_box(&perm_rel);
                    let recovered = recovered_perm.permute(total_bits - 2);
                    black_box(recovered)
                })
            },
        );
    }

    group.finish();
}

fn bench_batch_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("HDC Batch Distance");

    for &vec_len in &[8192usize, 65536] {
        for &count in &[10, 100, 1000] {
            let a_data: Vec<u8> = (0..vec_len * count)
                .map(|i| ((i * 37 + 13) % 256) as u8)
                .collect();
            let b_data: Vec<u8> = (0..vec_len * count)
                .map(|i| ((i * 71 + 42) % 256) as u8)
                .collect();
            let a = NumArrayU8::new(a_data);
            let b = NumArrayU8::new(b_data);

            group.throughput(Throughput::Bytes((vec_len * count * 2) as u64));

            group.bench_with_input(
                BenchmarkId::new(format!("batch_{}", vec_len), count),
                &count,
                |bencher, &count| {
                    bencher.iter(|| black_box(a.hamming_distance_batch(&b, vec_len, count)))
                },
            );
        }
    }

    group.finish();
}

fn bench_dot_i8(c: &mut Criterion) {
    let mut group = c.benchmark_group("HDC Int8 Dot Product (VNNI)");

    // CogRecord Container 3 sizes: 1024D (1KB), 2048D (2KB full container)
    for &dim in &[1024usize, 2048, 8192] {
        let a = NumArrayU8::new(create_random_vector(42, dim));
        let b = NumArrayU8::new(create_random_vector(123, dim));

        group.throughput(Throughput::Bytes((dim * 2) as u64));

        group.bench_with_input(
            BenchmarkId::new("dot_i8", dim),
            &dim,
            |bencher, &_| bencher.iter(|| black_box(a.dot_i8(&b))),
        );

        group.bench_with_input(
            BenchmarkId::new("cosine_i8", dim),
            &dim,
            |bencher, &_| bencher.iter(|| black_box(a.cosine_i8(&b))),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_bind,
    bench_distance,
    bench_permute,
    bench_bundle,
    bench_edge_encode_decode,
    bench_batch_distance,
    bench_dot_i8,
);
criterion_main!(benches);
