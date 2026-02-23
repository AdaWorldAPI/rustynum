//! End-to-end test: compile a scan kernel and verify it finds correct candidates.

use jitson::{JitEngine, JitEngineBuilder, ScanParams};

#[test]
fn test_scan_finds_exact_match() {
    let mut engine = JitEngine::new().unwrap();

    let params = ScanParams {
        threshold: 10, // low threshold — only very close matches
        top_k: 8,
        prefetch_ahead: 4,
        focus_mask: None,
        record_size: 8, // single u64 per record for the POC
    };

    let kernel = engine.compile_scan(params).unwrap();

    // Query: all 0xAA bytes
    let query: [u8; 8] = [0xAA; 8];

    // Field: 10 records of 8 bytes each
    // Record 0: exact match (0xAA * 8) → hamming distance = 0
    // Record 1: one bit different → hamming distance = 1
    // Record 2-9: very different → high hamming distance
    let mut field = vec![0u8; 80]; // 10 records * 8 bytes

    // Record 0: exact match
    field[0..8].copy_from_slice(&[0xAA; 8]);

    // Record 1: one byte different (0xAB = 0xAA ^ 0x01 → 1 bit)
    field[8..16].copy_from_slice(&[0xAB, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA]);

    // Records 2-9: all zeros (very different from 0xAA)
    // 0xAA = 10101010, popcount of 0xAA ^ 0x00 per byte = 4
    // For 8 bytes: 4*8 = 32 bits → distance = 32

    let mut candidates = vec![0u64; 8];
    let num = unsafe {
        kernel.scan(
            query.as_ptr(),
            field.as_ptr(),
            10,
            8,
            candidates.as_mut_ptr(),
        )
    };

    // Should find record 0 (dist=0) and record 1 (dist=1), both < threshold=10
    // NOTE: the POC kernel only XOR+popcnt's the first 8 bytes (one u64),
    // so record 0 has dist=0, record 1 has dist=1, records 2-9 have dist=32
    assert!(num >= 2, "expected at least 2 candidates, got {num}");
    assert_eq!(candidates[0], 0, "first candidate should be record 0");
    assert_eq!(candidates[1], 1, "second candidate should be record 1");
}

#[test]
fn test_scan_respects_top_k() {
    let mut engine = JitEngine::new().unwrap();

    let params = ScanParams {
        threshold: 64, // high threshold — matches everything
        top_k: 3,      // but only return 3
        prefetch_ahead: 0,
        focus_mask: None,
        record_size: 8,
    };

    let kernel = engine.compile_scan(params).unwrap();

    let query = [0u8; 8];
    let field = vec![0u8; 80]; // 10 records, all zeros → dist=0 for each
    let mut candidates = vec![0u64; 10];

    let num = unsafe {
        kernel.scan(
            query.as_ptr(),
            field.as_ptr(),
            10,
            8,
            candidates.as_mut_ptr(),
        )
    };

    assert_eq!(num, 3, "should stop at top_k=3, got {num}");
}

#[test]
fn test_kernel_caching() {
    let mut engine = JitEngine::new().unwrap();

    let params = ScanParams::default();

    // Compile twice with same params
    let _k1 = engine.compile_scan(params.clone()).unwrap();
    let _k2 = engine.compile_scan(params).unwrap();

    // Should only compile once (cache hit)
    assert_eq!(engine.cached_count(), 1);
}

#[test]
fn test_compile_latency() {
    // Measure engine creation vs kernel compilation separately
    let start_engine = std::time::Instant::now();
    let mut engine = JitEngine::new().unwrap();
    let engine_time = start_engine.elapsed();

    let params = ScanParams::default();

    let start_compile = std::time::Instant::now();
    let _kernel = engine.compile_scan(params.clone()).unwrap();
    let compile_time = start_compile.elapsed();

    // Second compile should be cache hit
    let start_cached = std::time::Instant::now();
    let _kernel2 = engine.compile_scan(params).unwrap();
    let cache_time = start_cached.elapsed();

    let total = engine_time + compile_time;
    eprintln!("JIT engine creation: {:?}", engine_time);
    eprintln!("JIT kernel compile:  {:?}", compile_time);
    eprintln!("JIT cache hit:       {:?}", cache_time);
    eprintln!("JIT total (cold):    {:?}", total);

    assert!(
        total.as_millis() < 100,
        "JIT compilation took too long: {total:?}"
    );
}

// ── Hybrid scan tests: JIT loop calling external distance function ──

/// A simple Hamming distance function that JIT code will CALL.
/// This simulates what `rustynum_core::hamming_distance` does.
///
/// extern "C" ABI: `fn(a: *const u8, b: *const u8, len: u64) -> u64`
unsafe extern "C" fn test_hamming_distance(a: *const u8, b: *const u8, len: u64) -> u64 {
    let mut dist = 0u64;
    for i in 0..len as usize {
        let byte_a = *a.add(i);
        let byte_b = *b.add(i);
        dist += (byte_a ^ byte_b).count_ones() as u64;
    }
    dist
}

#[test]
fn test_hybrid_scan_calls_external_fn() {
    // Register our test distance function with the JIT engine
    let mut engine = unsafe {
        JitEngineBuilder::new()
            .register_fn(
                "hamming_distance",
                test_hamming_distance as *const u8,
            )
            .build()
            .unwrap()
    };

    let params = ScanParams {
        threshold: 10,
        top_k: 8,
        prefetch_ahead: 0,
        focus_mask: None,
        record_size: 16, // 16 bytes per record — the hybrid fn handles arbitrary sizes
    };

    let kernel = engine
        .compile_hybrid_scan(params, "hamming_distance")
        .unwrap();

    // Query: all 0xFF bytes (16 bytes)
    let query = [0xFFu8; 16];

    // Field: 5 records of 16 bytes each
    let mut field = vec![0u8; 80]; // 5 records * 16 bytes

    // Record 0: exact match → dist=0
    field[0..16].copy_from_slice(&[0xFF; 16]);

    // Record 1: 2 bits different → dist=2
    field[16..32].copy_from_slice(&[0xFE, 0xFE, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                     0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);

    // Record 2: all zeros → dist = 128 (each byte 0xFF^0x00 = 8 bits * 16 bytes)
    // field[32..48] already zeros

    // Record 3: close match → dist=4
    field[48..64].copy_from_slice(&[0xF0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                     0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);

    // Record 4: all zeros → dist=128
    // field[64..80] already zeros

    let mut candidates = vec![0u64; 8];
    let num = unsafe {
        kernel.scan(
            query.as_ptr(),
            field.as_ptr(),
            5,
            16,
            candidates.as_mut_ptr(),
        )
    };

    // Should find record 0 (dist=0), record 1 (dist=2), record 3 (dist=4)
    // All are < threshold=10. Records 2 and 4 have dist=128, which is >= threshold.
    assert_eq!(num, 3, "expected 3 candidates, got {num}");
    assert_eq!(candidates[0], 0, "first candidate should be record 0");
    assert_eq!(candidates[1], 1, "second candidate should be record 1");
    assert_eq!(candidates[2], 3, "third candidate should be record 3");
}

#[test]
fn test_hybrid_scan_with_different_record_sizes() {
    // Verify that record_size is properly baked as an immediate
    let mut engine = unsafe {
        JitEngineBuilder::new()
            .register_fn(
                "hamming_distance",
                test_hamming_distance as *const u8,
            )
            .build()
            .unwrap()
    };

    // 32-byte records
    let params = ScanParams {
        threshold: 20,
        top_k: 4,
        prefetch_ahead: 0,
        focus_mask: None,
        record_size: 32,
    };

    let kernel = engine
        .compile_hybrid_scan(params, "hamming_distance")
        .unwrap();

    let query = vec![0xAAu8; 32];
    let mut field = vec![0u8; 128]; // 4 records * 32 bytes

    // Record 0: exact match
    field[0..32].copy_from_slice(&vec![0xAA; 32]);

    // Record 1: 5 bits different
    field[32..64].copy_from_slice(&vec![0xAA; 32]);
    field[32] = 0xA0; // flip 2 bits
    field[33] = 0xA8; // flip 1 bit
    field[34] = 0xAE; // flip 2 bits

    // Records 2-3: all zeros → dist = 128 (way above threshold)

    let mut candidates = vec![0u64; 4];
    let num = unsafe {
        kernel.scan(
            query.as_ptr(),
            field.as_ptr(),
            4,
            32,
            candidates.as_mut_ptr(),
        )
    };

    assert_eq!(num, 2, "expected 2 candidates, got {num}");
    assert_eq!(candidates[0], 0);
    assert_eq!(candidates[1], 1);
}
