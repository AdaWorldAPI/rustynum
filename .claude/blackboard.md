# Rustynum — Blackboard

> Shared state surface for all agents. Read before starting, update after completing work.

## Global Goal
Replace simd.rs bloat with simd_clean.rs dispatch.
Verify rustynum is the CLEAN REFERENCE that ndarray ports FROM.
After this: rustynum's kernels are bit-exact verified, benchmarked, documented.
The ndarray CC session copies FROM here. Not the other way around.

### Environment
- rust_version: 1.94-stable (rust-toolchain.toml)
- simd_tiers: AVX-512 (primary), AVX2 (fallback), scalar (baseline)
- test_command: `cargo test --workspace --exclude rustynum-rs --exclude rustynum-arrow --exclude rustynum`
- bench_command: `RUSTFLAGS="-C target-cpu=native" cargo bench -p rustynum-core`

---

## Dispatch Entries (32 functions)

### BLAS Level 1 (12 functions)
| Function | AVX-512 | AVX2 | Scalar | Tested |
|----------|---------|------|--------|--------|
| dot_f32 | ? | ? | ? | ? |
| dot_f64 | ? | ? | ? | ? |
| axpy_f32 | ? | ? | ? | ? |
| axpy_f64 | ? | ? | ? | ? |
| scal_f32 | ? | ? | ? | ? |
| scal_f64 | ? | ? | ? | ? |
| asum_f32 | ? | ? | ? | ? |
| asum_f64 | ? | ? | ? | ? |
| nrm2_f32 | ? | ? | ? | ? |
| nrm2_f64 | ? | ? | ? | ? |
| iamax_f32 | ? | ? | ? | ? |
| iamax_f64 | ? | ? | ? | ? |

### Element-wise f32 (8 functions)
| Function | AVX-512 | AVX2 | Scalar | Tested |
|----------|---------|------|--------|--------|
| add_f32_scalar | ? | ? | ? | ? |
| sub_f32_scalar | ? | ? | ? | ? |
| mul_f32_scalar | ? | ? | ? | ? |
| div_f32_scalar | ? | ? | ? | ? |
| add_f32_vec | ? | ? | ? | ? |
| sub_f32_vec | ? | ? | ? | ? |
| mul_f32_vec | ? | ? | ? | ? |
| div_f32_vec | ? | ? | ? | ? |

### Element-wise f64 (8 functions)
| Function | AVX-512 | AVX2 | Scalar | Tested |
|----------|---------|------|--------|--------|
| add_f64_scalar | ? | ? | ? | ? |
| sub_f64_scalar | ? | ? | ? | ? |
| mul_f64_scalar | ? | ? | ? | ? |
| div_f64_scalar | ? | ? | ? | ? |
| add_f64_vec | ? | ? | ? | ? |
| sub_f64_vec | ? | ? | ? | ? |
| mul_f64_vec | ? | ? | ? | ? |
| div_f64_vec | ? | ? | ? | ? |

### Binary / HDC (4 functions)
| Function | AVX-512 | AVX2 | Scalar | Tested |
|----------|---------|------|--------|--------|
| hamming_distance | ? | ? | ? | ? |
| popcount | ? | ? | ? | ? |
| dot_i8 | ? | ? | ? | ? |
| hamming_batch | ? | ? | ? | ? |

---

## Kernel Status
<!-- savant-architect writes here -->

## Binary Kernel Status
<!-- vector-synthesis writes here -->

## Gap Analysis
<!-- l3-strategist writes here -->

## API Surface
<!-- product-engineer writes here -->

## QA Audit Log
<!-- sentinel-qa writes here -->

## Benchmark Results
<!-- sentinel-qa writes here after benchmarking -->
| Kernel | Pre-PR#102 | Post-PR#102 | After simd_clean.rs | Target |
|--------|-----------|-------------|--------------------:|--------|
| sdot 1M f32 | ? | -24% | ? | ≥ pre-PR#102 |
| saxpy 1M f32 | ? | -22% | ? | ≥ pre-PR#102 |
| sgemm 512×512 | 13.3ms | 13.45ms | ? | ≤ 13.5ms |
| hamming 2KB | ? | ? | ? | < 5μs (AVX-512) |

---

## Three-Repo Stack (context)
```
rustynum (THIS REPO) → clean reference, verified kernels
ndarray (fork)       → ports FROM rustynum, becomes the product
lance-graph          → graph algebra, uses ndarray for compute
rs-graph-llm (fork)  → orchestration, uses lance-graph
```

Rustynum's job: be correct, be fast, be benchmarked.
ndarray's job: be rustynum but with better containers.
Rustynum retires AFTER ndarray has every kernel ported and verified.
