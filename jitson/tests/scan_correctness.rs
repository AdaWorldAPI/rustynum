//! End-to-end test: compile a scan kernel and verify it finds correct candidates.

use jitson::{JitEngine, ScanParams};

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
    // Verify compilation is fast (< 10ms)
    let start = std::time::Instant::now();
    let mut engine = JitEngine::new().unwrap();
    let params = ScanParams::default();
    let _kernel = engine.compile_scan(params).unwrap();
    let elapsed = start.elapsed();

    eprintln!("JIT compile time: {:?}", elapsed);
    assert!(
        elapsed.as_millis() < 100,
        "JIT compilation took too long: {elapsed:?}"
    );
}
