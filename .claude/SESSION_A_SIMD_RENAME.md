# SESSION_A_SIMD_RENAME.md

## Rename SIMD files + create hdr.rs (rustynum only)

**Repo:** rustynum (WRITE)
**Scope:** file renames, re-export shims, extract hdr.rs. Nothing else.
**Stop when:** `cargo test --workspace` passes, `cargo clippy --workspace` clean.

---

## CONTEXT YOU NEED

The filenames are backwards from what they sound like:

```
simd_compat.rs  = AVX-512 PRIMARY (F32x16, F64x8, VPOPCNTDQ, the fast path)
simd.rs         = AVX2 FALLBACK + dispatcher (the compatibility layer)
```

This confuses every session. Fix: rename so filenames match their purpose.

---

## STEP 1: Rename simd_compat.rs → simd_avx512.rs

```bash
cd rustynum-core/src
git mv simd_compat.rs simd_avx512.rs
```

Inside `simd_avx512.rs`: change nothing. Just the filename.

## STEP 2: Create re-export shim

Create NEW `rustynum-core/src/simd_compat.rs` with ONLY:

```rust
//! Backward-compatibility shim. All types moved to simd_avx512.rs.
#[allow(deprecated)]
#[deprecated(since = "0.4.0", note = "renamed to simd_avx512")]
pub use crate::simd_avx512::*;
```

## STEP 3: Update lib.rs

In `rustynum-core/src/lib.rs`, add the new module:

```rust
pub mod simd_avx512;      // AVX-512 primary (was simd_compat)

// Backward compat shim — remove in next major version
#[allow(deprecated)]
pub mod simd_compat;
```

Keep existing `pub mod simd;` unchanged.

## STEP 4: Migrate imports (sed)

Replace all direct `simd_compat` imports with `simd_avx512`:

```bash
find . -name "*.rs" -not -path "*/target/*" -not -name "simd_compat.rs" \
  -exec grep -l "simd_compat" {} \; | while read f; do
  sed -i 's/crate::simd_compat/crate::simd_avx512/g' "$f"
  sed -i 's/rustynum_core::simd_compat/rustynum_core::simd_avx512/g' "$f"
done
```

Also update comments that mention `simd_compat`:

```bash
find . -name "*.rs" -not -path "*/target/*" -not -name "simd_compat.rs" \
  -exec grep -l "simd_compat" {} \; | while read f; do
  sed -i 's/simd_compat/simd_avx512/g' "$f"
done
```

**Files that will be touched** (verify each compiled after):

```
rustyblas/src/level3.rs
rustyblas/src/bf16_gemm.rs
rustyblas/src/int8_gemm.rs
rustyblas/src/lib.rs
rustyblas/examples/gemm_benchmark.rs
rustymkl/src/fft.rs
rustymkl/src/vml.rs
rustymkl/src/lib.rs
rustynum-core/src/prefilter.rs
rustynum-core/src/simd_avx2.rs (if it exists, or simd.rs)
rustynum-core/src/bf16_hamming.rs
rustynum-core/src/simd.rs
rustynum-core/src/lib.rs
rustynum-rs/src/simd_ops/mod.rs
rustynum-rs/src/num_array/hdc.rs
rustynum-rs/src/num_array/array_struct.rs
rustynum-rs/src/num_array/bitwise.rs
rustynum-rs/src/lib.rs
```

## STEP 5: Create hdr.rs (move from simd.rs)

Create `rustynum-core/src/hdr.rs`. Move these items FROM `simd.rs`:

```
MOVE TO hdr.rs:
  struct HdrResult           → rename to RankedHit, add band: Band field
  enum PreciseMode           → keep as-is
  fn hdr_cascade_search()    → becomes Cascade::query() method
  fn apply_precision_tier()  → becomes Cascade::apply_precision() method

ADD NEW TO hdr.rs:
  pub struct Cascade { mu, sigma, bands, ... }
  pub enum Band { Foveal, Near, Good, Weak, Reject }
  pub struct ShiftAlert { old_mu, new_mu, old_sigma, new_sigma, observations }
  pub fn Cascade::calibrate(&[u32]) → Self
  pub fn Cascade::expose(u32) → Band
  pub fn Cascade::test(&[u8], &[u8]) → bool
  pub fn Cascade::observe(u32) → Option<ShiftAlert>
  pub fn Cascade::drift() → Option<ShiftAlert>   (alias for checking observe state)
  pub fn Cascade::recalibrate(&ShiftAlert)
```

Leave deprecated wrappers in `simd.rs`:

```rust
#[deprecated(since = "0.4.0", note = "use hdr::Cascade::query()")]
pub fn hdr_cascade_search(
    query: &[u8], database: &[u8], vec_bytes: usize,
    num_vectors: usize, threshold: u64, precise_mode: PreciseMode,
) -> Vec<hdr::RankedHit> {
    let cascade = hdr::Cascade::from_threshold(threshold, vec_bytes);
    cascade.query(query, database, vec_bytes, num_vectors, precise_mode)
}
```

## STEP 6: Update hdc.rs wrappers

In `rustynum-rs/src/num_array/hdc.rs`:

```rust
// Keep public API unchanged. Just delegate to new location:
pub fn hamming_search_adaptive(...) -> Vec<hdr::RankedHit> {
    let cascade = hdr::Cascade::from_threshold(threshold, vec_bytes);
    cascade.query(query, database, vec_bytes, num_vectors, PreciseMode::Off)
}
```

## STEP 7: Add to lib.rs

```rust
pub mod hdr;  // HDR cascade search
```

## STEP 8: Verify

```bash
RUSTFLAGS="-C target-cpu=native" cargo test --workspace
cargo clippy --workspace -- -D warnings
```

ALL existing tests must pass. Zero new tests in this session. This is a rename, not a feature.

---

## NOT IN SCOPE

```
× Don't extract simd_avx2.rs from simd.rs yet (Part 1 does this)
× Don't add ReservoirSample (Session C does this)
× Don't add integer hot path (Session C does this)
× Don't touch lance-graph (Session B does this)
× Don't add new tests (renames don't need new tests)
× Don't refactor simd.rs dispatcher logic (just move hdr stuff out)
```
