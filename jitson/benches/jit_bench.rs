use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use jitson::{JitEngine, ScanParams};

fn bench_compile_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("JIT Compile");

    group.bench_function("scan_kernel_compile", |b| {
        b.iter(|| {
            let mut engine = JitEngine::new().unwrap();
            let params = ScanParams {
                threshold: black_box(500),
                top_k: 32,
                prefetch_ahead: 4,
                focus_mask: None,
                record_size: 1024,
            };
            engine.compile_scan(params).unwrap();
        })
    });

    group.finish();
}

fn bench_scan_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("JIT Scan");

    let mut engine = JitEngine::new().unwrap();
    let params = ScanParams {
        threshold: 500,
        top_k: 32,
        prefetch_ahead: 4,
        focus_mask: None,
        record_size: 8, // 8 bytes per record for the POC (single u64)
    };
    let kernel = engine.compile_scan(params).unwrap();

    for num_records in [100, 1000, 10_000] {
        // Create test data: random bytes
        let query = vec![0xAA_u8; 8];
        let field: Vec<u8> = (0..num_records * 8).map(|i| (i % 256) as u8).collect();
        let mut candidates = vec![0u64; 32];

        group.bench_with_input(
            BenchmarkId::new("jit_scan", num_records),
            &num_records,
            |b, _| {
                b.iter(|| unsafe {
                    kernel.scan(
                        query.as_ptr(),
                        field.as_ptr(),
                        num_records as u64,
                        8,
                        candidates.as_mut_ptr(),
                    )
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_compile_scan, bench_scan_execution);
criterion_main!(benches);
