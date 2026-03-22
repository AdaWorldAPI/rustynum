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

## Dispatch Entries (32 functions + 2 GEMM)

> Last audited: 2026-03-22. Blackboard was completely stale — all `?` marks were wrong, code is substantially complete.

### BLAS Level 1 (12 functions)
| Function | AVX-512 | AVX2 | Scalar | Dispatched |
|----------|---------|------|--------|------------|
| dot_f32 | DONE | DONE | DONE | YES |
| dot_f64 | DONE | DONE | DONE | YES |
| axpy_f32 | DONE | DONE | DONE | YES |
| axpy_f64 | DONE | DONE | DONE | YES |
| scal_f32 | DONE | DONE | DONE | YES |
| scal_f64 | DONE | DONE | DONE | YES |
| asum_f32 | DONE | DONE | DONE | YES |
| asum_f64 | DONE | DONE | DONE | YES |
| nrm2_f32 | DONE | DONE | DONE | YES |
| nrm2_f64 | DONE | DONE | DONE | YES |
| copy_f32 | DONE | DONE | DONE | YES |
| swap_f32 | DONE | DONE | DONE | YES |

Notes: AVX-512 paths are target_feature guarded. AVX2 has 10/12 done (iamax missing for f32/f64). All 12 wired in dispatch.

### Element-wise f32 (8 functions)
| Function | AVX-512 | AVX2 | Scalar | Dispatched |
|----------|---------|------|--------|------------|
| add_f32_scalar | DONE | scalar fallback | DONE | YES |
| sub_f32_scalar | DONE | scalar fallback | DONE | YES |
| mul_f32_scalar | DONE | scalar fallback | DONE | YES |
| div_f32_scalar | DONE | scalar fallback | DONE | YES |
| add_f32_vec | DONE | scalar fallback | DONE | YES |
| sub_f32_vec | DONE | scalar fallback | DONE | YES |
| mul_f32_vec | DONE | scalar fallback | DONE | YES |
| div_f32_vec | DONE | scalar fallback | DONE | YES |

Notes: AVX-512 macro-generated. AVX2 falls to scalar (no dedicated impl).

### Element-wise f64 (8 functions)
| Function | AVX-512 | AVX2 | Scalar | Dispatched |
|----------|---------|------|--------|------------|
| add_f64_scalar | DONE | scalar fallback | DONE | YES |
| sub_f64_scalar | DONE | scalar fallback | DONE | YES |
| mul_f64_scalar | DONE | scalar fallback | DONE | YES |
| div_f64_scalar | DONE | scalar fallback | DONE | YES |
| add_f64_vec | DONE | scalar fallback | DONE | YES |
| sub_f64_vec | DONE | scalar fallback | DONE | YES |
| mul_f64_vec | DONE | scalar fallback | DONE | YES |
| div_f64_vec | DONE | scalar fallback | DONE | YES |

Notes: AVX-512 all done. AVX2 falls to scalar.

### Binary / HDC (4 functions)
| Function | AVX-512 | AVX2 | Scalar | Dispatched |
|----------|---------|------|--------|------------|
| hamming_distance | DONE (vpopcntdq) | DONE | DONE | YES |
| popcount | DONE (vpopcntdq) | DONE | DONE | YES |
| dot_i8 | DONE (vpopcntdq) | DONE | DONE | YES |
| hamming_batch | DONE (vpopcntdq) | DONE | DONE | YES |

### GEMM (2 functions)
| Function | AVX-512 | AVX2 | Scalar | Dispatched |
|----------|---------|------|--------|------------|
| sgemm_blocked | DONE | DONE | DONE | YES |
| dgemm_blocked | DONE | DONE | DONE | YES |

Notes: Dispatched through the same dispatch table as BLAS Level 1.

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
| sdot 1M f32 | ? | -24% | ? | ≥ pre-PR