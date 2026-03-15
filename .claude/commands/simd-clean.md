---
name: simd-clean
description: >
  Replace the bloated simd.rs (2435 lines, 107 detections) with
  simd_clean.rs (229 lines, 32 dispatch entries, 1 detection).
  Verify all tests pass. Benchmark against pre-replacement baseline.
  This is Session G — the foundation everything else builds on.
allowed-tools: Read, Edit, Write, Bash, Glob, Grep, Task
---

# MISSION: Replace simd.rs with simd_clean.rs

## PRIME DIRECTIVE
Replace `rustynum-core/src/simd.rs` with `.claude/simd_clean.rs`.
Ensure every dispatch! entry has a working kernel in ALL THREE tiers.
Verify all 1543+ tests pass. Benchmark key functions. Fix regressions.
Do NOT stop until cargo test --workspace passes clean.

## SCOPE LOCK
- WRITE ONLY to files inside `rustynum-core/src/`
- READ `.claude/simd_clean.rs` as the template
- READ `simd_avx512.rs`, `simd_avx2.rs`, `scalar_fns.rs` for existing kernels
- NEVER modify `.claude/` files. They are the spec.

## STEP 1: Audit existing kernels (Agent: l3-strategist)

Before replacing anything, check that every dispatch! entry has a kernel.

```bash
# Extract the 32 function names from simd_clean.rs
grep "^dispatch!" .claude/simd_clean.rs | sed 's/dispatch!(\([a-z_0-9]*\).*/\1/'

# For each, check all three tiers:
# simd_avx512.rs: unsafe fn $name or pub unsafe fn $name
# simd_avx2.rs: pub fn $name
# scalar_fns.rs: pub fn $name (may have _scalar suffix that needs aliasing)
```

KNOWN GAPS from prior analysis:
- `scalar_fns.rs` has `_scalar` suffix on names → need plain-name aliases
- `simd_avx512.rs` has wrapper TYPES (F32x16) but may lack standalone BLAS functions
  → the BLAS functions (dot_f32, axpy_f32, etc.) for AVX-512 may be INLINE in the
  OLD simd.rs, not in simd_avx512.rs. Extract them.

## STEP 2: Fix scalar_fns.rs naming (Agent: savant-architect)

Add plain-name pub functions that delegate to the _scalar versions:

```rust
// At end of scalar_fns.rs:
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 { dot_f32_scalar(a, b) }
pub fn dot_f64(a: &[f64], b: &[f64]) -> f64 { dot_f64_scalar(a, b) }
pub fn axpy_f32(alpha: f32, x: &[f32], y: &mut [f32]) { axpy_f32_scalar(alpha, x, y) }
// ... for ALL 32 dispatch entries
```

## STEP 3: Extract AVX-512 BLAS functions (Agent: savant-architect)

The OLD simd.rs has `dot_f32_avx512`, `axpy_f32_avx512`, etc. as standalone functions.
simd_avx512.rs has the wrapper types (F32x16) but may lack these top-level functions.

For each dispatch! entry, ensure simd_avx512.rs exports a function with the PLAIN name:

```rust
// In simd_avx512.rs:
#[target_feature(enable = "avx512f")]
pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    // Use F32x16 wrapper types internally
    let mut acc0 = F32x16::splat(0.0);
    let mut acc1 = F32x16::splat(0.0);
    let mut acc2 = F32x16::splat(0.0);
    let mut acc3 = F32x16::splat(0.0);
    // 4-accumulator unrolled loop
    // ...
    (acc0 + acc1 + acc2 + acc3).reduce_sum()
}
```

For hamming_distance: use VPOPCNTDQ (U8x64 type):
```rust
#[target_feature(enable = "avx512f,avx512bw,avx512vpopcntdq")]
pub unsafe fn hamming_distance(a: &[u8], b: &[u8]) -> u64 {
    // U8x64::from_slice → xor → popcount_epi64 → sum
}
```

## STEP 4: Replace simd.rs (Agent: savant-architect)

```bash
cp rustynum-core/src/simd.rs rustynum-core/src/simd_old_backup.rs
cp .claude/simd_clean.rs rustynum-core/src/simd.rs
```

## STEP 5: Wire module in lib.rs (Agent: product-engineer)

Ensure lib.rs exports:
```rust
pub mod simd;          // the new dispatch (was already exported)
pub mod simd_avx512;   // AVX-512 kernels
pub mod simd_avx2;     // AVX2 kernels
pub mod scalar_fns;    // scalar fallbacks (ADD THIS if missing)
```

## STEP 6: Compile and fix (Agent: savant-architect)

```bash
cargo check -p rustynum-core 2>&1 | head -50
# Fix every error. Common issues:
# - Missing function in a tier → add it
# - Signature mismatch → match the dispatch! signature exactly
# - Missing import → add use statement
```

## STEP 7: Test (Agent: sentinel-qa)

```bash
cargo test --workspace --exclude rustynum-rs --exclude rustynum-arrow --exclude rustynum 2>&1
# Must pass 1543+ tests
# If any fail: fix and re-run. Do NOT skip.
```

## STEP 8: Benchmark (Agent: sentinel-qa)

```bash
RUSTFLAGS="-C target-cpu=native" cargo bench -p rustynum-core 2>&1 | tee bench_after.txt
```

Key metrics to verify:
```
sdot 1M f32:      must match or beat pre-PR#102 (NOT post-PR#102)
saxpy 1M f32:     must match or beat pre-PR#102
hamming 2KB:      < 5μs on AVX-512
sgemm 512×512:    must match baseline (13.3ms or better)
```

If sdot/saxpy show the PR#102 regression (22-24% slower): the dispatch!
macro eliminates the per-function unsafe wrapper that caused it.
Verify it's FIXED.

## STEP 9: Commit (Agent: product-engineer)

```bash
git add rustynum-core/src/simd.rs rustynum-core/src/simd_avx512.rs \
        rustynum-core/src/simd_avx2.rs rustynum-core/src/scalar_fns.rs \
        rustynum-core/src/lib.rs
git commit -m "refactor: simd.rs 2435→229 lines — dispatch! macro + LazyLock<Tier>

Replace 107 is_x86_feature_detected! calls with ONE LazyLock detection.
32 dispatch! entries. Each function = one line in simd.rs.
AVX-512 → AVX2 → scalar fallback chain.

Fixes PR#102 sdot/saxpy regression (per-function unsafe removed).
All 1543+ tests passing."
```

## COMPLETION CRITERIA
- [ ] simd.rs is 229 lines (was 2435)
- [ ] Every dispatch! entry resolves in all three tiers
- [ ] `cargo test --workspace` passes (excluding legacy crates)
- [ ] `cargo clippy --workspace -- -D warnings` clean
- [ ] sdot/saxpy regression from PR#102 is fixed
- [ ] Benchmark numbers captured in blackboard
