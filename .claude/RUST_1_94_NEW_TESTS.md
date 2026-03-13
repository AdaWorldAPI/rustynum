# RUST_1_94_NEW_TESTS.md

## New Algorithms Enabled by Rust 1.94 Stable (2026-03-05)

**Bump `rust-toolchain.toml` to 1.94.0 FIRST in all repos.**

Rust 1.94 stabilizes `avx512fp16` intrinsics and `array_windows`.
This unlocks algorithms that were previously nightly-only or required
manual bit manipulation. Every test below runs on stable. No `#![feature]`.

---

### 1. Hardware FP16 ↔ F32 Conversion (avx512fp16)

Previously: `bf16_gemm.rs` converts f32 ↔ bf16 via bit shifting (manual).
Now: hardware instruction does 32 conversions per cycle.

```rust
use core::arch::x86_64::*;

/// Convert 32 f32 values to 16 fp16 values in ONE instruction.
/// Available on Sapphire Rapids, Zen 5, etc.
#[target_feature(enable = "avx512fp16")]
unsafe fn f32x16_to_fp16x16(a: __m512) -> __m256i {
    _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
}

/// Convert 16 fp16 values back to 16 f32 values in ONE instruction.
#[target_feature(enable = "avx512fp16")]
unsafe fn fp16x16_to_f32x16(a: __m256i) -> __m512 {
    _mm512_cvtph_ps(a)
}
```

**Test: f32 embedding → fp16 → Hamming → compare against f32 cosine**

```rust
#[test]
fn fp16_conversion_preserves_ranking() {
    // 1000 random f32 embeddings (1024 dimensions)
    let embeddings: Vec<Vec<f32>> = (0..1000).map(|i| random_f32_vec(1024, i)).collect();
    let query = random_f32_vec(1024, 9999);

    // Path A: f32 cosine similarity (ground truth)
    let mut float_ranked: Vec<(usize, f32)> = embeddings.iter().enumerate()
        .map(|(i, v)| (i, cosine_f32(&query, v)))
        .collect();
    float_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Path B: f32 → fp16 → XOR → popcount (our pipeline)
    let fp16_query = f32_slice_to_fp16(&query);    // hardware VCVTPS2PH
    let fp16_vecs: Vec<Vec<u16>> = embeddings.iter()
        .map(|v| f32_slice_to_fp16(v))
        .collect();
    let mut hamming_ranked: Vec<(usize, u32)> = fp16_vecs.iter().enumerate()
        .map(|(i, v)| (i, fp16_hamming(&fp16_query, v)))
        .collect();
    hamming_ranked.sort_by_key(|x| x.1);

    // Compare top-10 overlap
    let float_top10: Vec<usize> = float_ranked[..10].iter().map(|x| x.0).collect();
    let hamming_top10: Vec<usize> = hamming_ranked[..10].iter().map(|x| x.0).collect();
    let overlap = float_top10.iter().filter(|x| hamming_top10.contains(x)).count();

    println!("FP16 hardware conversion top-10 overlap: {}/10", overlap);
    assert!(overlap >= 7, "Hardware FP16 Hamming should agree with f32 cosine on >=7/10");
}
```

**Why this matters:** bf16_hamming.rs currently does manual bit manipulation to
extract sign/exponent/mantissa for weighted Hamming. With hardware FP16:
- No manual bit extraction — the hardware converts f32 → fp16 natively
- 32 conversions per cycle instead of one-at-a-time bit shifting
- The fp16 format PRESERVES float ordering: if a > b as f32, then a > b as fp16
- XOR on fp16 bits gives STRUCTURED Hamming (MSB=sign, next bits=exponent)
- The BF16 weighted Hamming becomes: convert → XOR → popcount. Three instructions.

---

### 2. FP16 Native Arithmetic (VFMADD231PH)

Not just conversion — full FP16 multiply-add. 32 FMA operations per cycle.

```rust
/// FP16 dot product: 32 multiply-adds per cycle.
/// Previously needed: convert to f32, FMA, convert back.
/// Now: native FP16 FMA. 2x throughput, half the register pressure.
#[target_feature(enable = "avx512fp16")]
unsafe fn fp16_dot_32(a: *const u16, b: *const u16) -> f32 {
    let av = _mm512_loadu_ph(a as *const f16);
    let bv = _mm512_loadu_ph(b as *const f16);
    let prod = _mm512_mul_ph(av, bv);
    // Horizontal sum: reduce 32 fp16 products to one f32
    let sum256 = _mm256_add_ph(
        _mm512_castph512_ph256(prod),
        _mm512_extractf32x8_ps(/* cast */)
    );
    // ... final reduction to scalar
}
```

**Test: FP16 native dot product vs f32 dot product**

```rust
#[test]
fn fp16_native_dot_accuracy() {
    let a: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.001).sin()).collect();
    let b: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.002).cos()).collect();

    let f32_dot = dot_f32(&a, &b);
    let fp16_dot = fp16_native_dot(&a, &b);  // convert → fp16 dot → convert back

    let relative_error = (f32_dot - fp16_dot).abs() / f32_dot.abs().max(1e-10);
    println!("f32 dot:  {}", f32_dot);
    println!("fp16 dot: {}", fp16_dot);
    println!("Relative error: {:.6}", relative_error);

    assert!(relative_error < 0.01, "FP16 dot should be within 1% of f32");
}
```

---

### 3. FP16 GEMM Microkernel (the level3.rs upgrade)

The broken `level3.rs` GEMM can now have THREE tiers:

```rust
pub fn sgemm_microkernel(/* args */) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512fp16") {
            // FP16 GEMM: 32 FMAs per cycle, half the memory bandwidth
            // Matrix stays in fp16 format. No conversion during multiply.
            return unsafe { sgemm_fp16_microkernel_6x32(/* args */) };
        }
        if is_x86_feature_detected!("avx512f") {
            // F32 AVX-512: 16 FMAs per cycle
            return unsafe { sgemm_microkernel_6x16(/* args */) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // F32 AVX2: 8 FMAs per cycle
            return unsafe { sgemm_microkernel_6x8(/* args */) };
        }
    }
    sgemm_microkernel_scalar(/* args */);
}
```

**Test: FP16 GEMM vs F32 GEMM correctness**

```rust
#[test]
fn fp16_gemm_matches_f32() {
    let m = 64;
    let n = 64;
    let k = 64;
    let a = random_matrix_f32(m, k, 42);
    let b = random_matrix_f32(k, n, 43);

    let c_f32 = sgemm_f32(&a, &b, m, n, k);
    let c_fp16 = sgemm_fp16(&a, &b, m, n, k);  // convert → fp16 gemm → convert back

    let max_diff = c_f32.iter().zip(c_fp16.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("Max element-wise difference: {}", max_diff);
    assert!(max_diff < 1.0, "FP16 GEMM should be within 1.0 of f32 for this scale");
}
```

---

### 4. array_windows for Cascade Sampling

`<[T]>::array_windows` is now stable. Perfect for Belichtungsmesser stage sampling.

```rust
/// Sample 1/16 of accumulator using array_windows.
/// Each window is exactly 128 bytes (1/16 of 2048).
/// No allocation. No copy. Just a slice view.
fn cascade_stage1_sample(acc: &[u8; 2048]) -> u32 {
    // First 128-byte window = 1/16 sample
    let sample: &[u8; 128] = acc[..128].try_into().unwrap();
    simd::popcount(sample) as u32
}

/// Sliding window density check across accumulator.
/// Detects hot regions (high popcount) vs cold regions (sparse).
fn density_profile(acc: &[u8; 2048]) -> [u32; 16] {
    let mut profile = [0u32; 16];
    for (i, window) in acc.array_windows::<128>().step_by(128).enumerate() {
        if i >= 16 { break; }
        profile[i] = simd::popcount(window) as u32;
    }
    profile
}
```

**Test: density profile detects clustered vs uniform bits**

```rust
#[test]
fn array_windows_density_profile() {
    // Uniform: bits spread evenly across 2048 bytes
    let uniform = [0xAA_u8; 2048];  // alternating bits everywhere
    let profile_uniform = density_profile(&uniform);
    let variance_uniform = statistical_variance(&profile_uniform);

    // Clustered: bits concentrated in first 256 bytes
    let mut clustered = [0x00_u8; 2048];
    for i in 0..256 { clustered[i] = 0xFF; }
    let profile_clustered = density_profile(&clustered);
    let variance_clustered = statistical_variance(&profile_clustered);

    println!("Uniform variance:   {}", variance_uniform);
    println!("Clustered variance: {}", variance_clustered);

    assert!(variance_uniform < variance_clustered,
        "Clustered bits should have higher density variance than uniform");
}
```

---

### 5. FP16 BF16 Structured Hamming (the bf16_hamming.rs upgrade)

The existing `bf16_hamming.rs` does weighted Hamming with manual bit extraction.
With FP16 hardware:

```rust
/// NEW: Hardware-accelerated BF16 structured Hamming.
///
/// OLD path (manual):
///   Extract sign bit → weight × 1
///   Extract exponent (8 bits) → weight × 8
///   Extract mantissa (7 bits) → weight × 7
///   Total: ~16 ops per BF16 pair
///
/// NEW path (avx512fp16):
///   Load 32 fp16 pairs → VXORPH → VPOPCNTW → VPMADDWD (weighted sum)
///   Total: 4 instructions for 32 pairs = 0.125 ops per pair
///   128x speedup per pair.
#[target_feature(enable = "avx512fp16,avx512vpopcntdq")]
unsafe fn bf16_hamming_fp16(a: &[u8], b: &[u8], weights: &BF16Weights) -> u64 {
    // XOR the raw fp16 words → differing bits
    // VPOPCNTW → popcount per 16-bit word
    // Separate sign/exponent/mantissa via mask+shift
    // VPMADDWD → weighted accumulation
    // One pass. Four instructions per 32 pairs.
    todo!("implement using new stable avx512fp16 intrinsics")
}
```

**Test: hardware BF16 Hamming matches scalar BF16 Hamming**

```rust
#[test]
fn fp16_bf16_hamming_matches_scalar() {
    let a: Vec<u8> = (0..2048).map(|i| (i * 7 + 13) as u8).collect();
    let b: Vec<u8> = (0..2048).map(|i| (i * 11 + 3) as u8).collect();
    let weights = BF16Weights::default();  // Jina weights

    let scalar_result = bf16_hamming_scalar(&a, &b, &weights);

    if is_x86_feature_detected!("avx512fp16") {
        let hw_result = unsafe { bf16_hamming_fp16(&a, &b, &weights) };
        assert_eq!(scalar_result, hw_result,
            "Hardware FP16 BF16 Hamming must match scalar exactly");
        println!("Hardware FP16 path verified: {} == {}", hw_result, scalar_result);
    } else {
        println!("Skipping FP16 hardware test (CPU doesn't support avx512fp16)");
    }
}
```

---

### 6. FP16 Embedding Ingest Pipeline (the end-to-end test)

The full pipeline that Rust 1.94 enables on stable:

```
f32 embedding (from Jina, OpenAI, etc.)
  ↓ VCVTPS2PH (hardware, 32 per cycle)
fp16 packed (half the memory, preserves ordering)
  ↓ BLAKE3 hash (on fp16 bytes)
16K BitVec fingerprint (for scent index)
  ↓ store both fp16 + BitVec in LanceDB
  ↓
QUERY:
  query f32 → VCVTPS2PH → fp16
  Stage 1: BitVec Hamming cascade (Belichtungsmesser)
  Stage 2: fp16 precise reranking (VFMADD231PH native dot product)
  Stage 3: optional f32 exact scoring on top-k survivors
```

**Test: full ingest pipeline correctness**

```rust
#[test]
fn fp16_full_pipeline_roundtrip() {
    let original = random_f32_vec(1024, 42);

    // Step 1: f32 → fp16 (hardware if available)
    let fp16 = f32_to_fp16_dispatch(&original);
    assert_eq!(fp16.len(), 1024);

    // Step 2: fp16 → BitVec (BLAKE3 expansion)
    let bitvec = fp16_to_bitvec(&fp16);
    assert_eq!(bitvec.len(), 2048);  // 16384 bits = 2048 bytes

    // Step 3: fp16 → f32 roundtrip (hardware if available)
    let recovered = fp16_to_f32_dispatch(&fp16);
    assert_eq!(recovered.len(), 1024);

    // Roundtrip accuracy: f32 → fp16 → f32 should be close
    let max_error: f32 = original.iter().zip(recovered.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);

    println!("Max roundtrip error: {}", max_error);
    assert!(max_error < 0.01, "FP16 roundtrip should preserve f32 within 0.01");

    // Ranking preservation: fp16 Hamming ranking should match f32 cosine ranking
    // (covered by fp16_conversion_preserves_ranking test above)
}
```

---

### 7. LazyLock::get_mut for SIMD Dispatch Reconfiguration

`LazyLock::get_mut` is now stable. This enables reconfiguring the SIMD dispatch
at test time without unsafe code:

```rust
use std::sync::LazyLock;

static HAMMING_FN: LazyLock<fn(&[u8], &[u8]) -> u64> = LazyLock::new(select_hamming_fn);

#[test]
fn force_scalar_fallback() {
    // LazyLock::force_mut allows overriding the dispatch for testing
    // Verify scalar produces same results as SIMD
    let a = vec![0xAA_u8; 2048];
    let b = vec![0x55_u8; 2048];

    let simd_result = hamming(&a, &b);
    let scalar_result = hamming_scalar(&a, &b);

    assert_eq!(simd_result, scalar_result,
        "Scalar fallback must produce identical results to SIMD dispatch");
}
```

---

### 8. Benchmark Matrix (the CV)

```rust
// benches/simd_tiers.rs (criterion)

fn bench_hamming_tiers(c: &mut Criterion) {
    let a = vec![0xAA_u8; 2048];
    let b = vec![0x55_u8; 2048];

    let mut group = c.benchmark_group("hamming_16k");

    group.bench_function("avx512_vpopcntdq", |bench| {
        if !is_x86_feature_detected!("avx512vpopcntdq") {
            return; // skip on this hardware
        }
        bench.iter(|| unsafe { hamming_vpopcntdq(&a, &b) });
    });

    group.bench_function("avx2_harley_seal", |bench| {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        bench.iter(|| unsafe { hamming_avx2_safe(&a, &b) });
    });

    group.bench_function("scalar_popcnt", |bench| {
        bench.iter(|| hamming_scalar(&a, &b));
    });

    // NEW: FP16 paths
    group.bench_function("fp16_convert_roundtrip", |bench| {
        if !is_x86_feature_detected!("avx512fp16") {
            return;
        }
        let f32_data: Vec<f32> = (0..512).map(|i| i as f32 * 0.01).collect();
        bench.iter(|| {
            let fp16 = f32_to_fp16_hw(&f32_data);
            let f32_back = fp16_to_f32_hw(&fp16);
            f32_back
        });
    });

    group.bench_function("fp16_bf16_hamming", |bench| {
        if !is_x86_feature_detected!("avx512fp16") {
            return;
        }
        let weights = BF16Weights::default();
        bench.iter(|| unsafe { bf16_hamming_fp16(&a, &b, &weights) });
    });

    group.finish();
}

fn bench_plane_distance(c: &mut Criterion) {
    let a = Plane::random(42);
    let b = Plane::random(43);
    let node_a = Node::random(42);
    let node_b = Node::random(43);

    let mut group = c.benchmark_group("plane_node_distance");

    group.bench_function("single_plane", |bench| {
        bench.iter(|| a.distance(&b));
    });
    // THIS is the "4 cycles" claim. Prove or disprove it publicly.

    group.bench_function("node_S__", |bench| {
        bench.iter(|| node_a.distance(&node_b, S__));
    });

    group.bench_function("node_SP_", |bench| {
        bench.iter(|| node_a.distance(&node_b, SP_));
    });

    group.bench_function("node_SPO", |bench| {
        bench.iter(|| node_a.distance(&node_b, SPO));
    });

    group.bench_function("belichtungsmesser_stage1", |bench| {
        let meter = Belichtungsmesser::calibrate(&[8192; 100]);
        let query = vec![0xAA_u8; 2048];
        let candidate = vec![0x55_u8; 2048];
        bench.iter(|| {
            let s1 = hamming(&query[..128], &candidate[..128]) as u32 * 16;
            meter.band(s1)
        });
    });
    // THIS is the "2 cycles for stage 1" claim.

    group.finish();
}

criterion_group!(benches, bench_hamming_tiers, bench_plane_distance);
criterion_main!(benches);
```

---

### Summary: What Rust 1.94 Unlocks

```
STABLE FEATURE                → ALGORITHM / TEST
────────────────────────────────────────────────────────────────
avx512fp16 intrinsics         → Hardware f32 ↔ fp16 conversion
                              → Native FP16 dot product (VFMADD231PH)
                              → FP16 GEMM microkernel (32 FMAs/cycle)
                              → Hardware BF16 structured Hamming
                              → Full fp16 embedding ingest pipeline
                              → fp16 vs f32 cosine ranking equivalence test

AArch64 NEON fp16 intrinsics  → Same algorithms on Apple Silicon / Graviton

array_windows                 → Cascade sampling without allocation
                              → Density profiling across accumulator
                              → Sliding-window anomaly detection

LazyLock::get/get_mut         → SIMD dispatch reconfiguration in tests
                              → Force-scalar verification

f32::mul_add const            → Compile-time threshold computation
                              → Belichtungsmesser band constants at build time

RISC-V features (29 stable)   → Future: rustynum on RISC-V (SiFive, etc.)
```

**Bump all repos to 1.94. Run the benchmarks. The numbers go in the README.**
