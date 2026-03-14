# SESSION_F_AVX2_GAPS.md

## Six-line dispatch. Compile-time ISA selection via target_feature.

**Repo:** rustynum (WRITE)
**Scope:** fix simd.rs to use `target_feature` gates, fix defaults, fill two gaps
**Stop when:** `cargo test --workspace` passes with NO flags

---

## THE FIX

The codebase already has three complete implementations:
- `simd_avx512.rs` — AVX-512 (F32x16, VPOPCNTDQ, etc.)
- `simd_avx2.rs` — AVX2 (F32x8, vpshufb nibble LUT, etc.)
- `scalar_fns.rs` — scalar fallback (plain loops)

The ONLY problem was `simd.rs` using `cfg(feature = "avx512")` (Cargo feature)
instead of `cfg(target_feature = "avx512f")` (CPU feature).

Cargo features are set in Cargo.toml. `target_feature` is set by the compiler
based on `-C target-cpu=native` or `-C target-feature=+avx512f`. The compiler
knows what the CPU has. Let it pick.

---

## STEP 1: simd.rs becomes six lines of re-exports

Delete the entire body of simd.rs. Replace with:

```rust
//! SIMD dispatch: compile-time ISA selection via target_feature.
//!
//! Build with:
//!   cargo build                        → scalar (CI, ARM, WASM)
//!   cargo build -C target-cpu=native   → best for this CPU
//!   cargo build -C target-feature=+avx2,+fma → explicit AVX2

#[cfg(target_feature = "avx512f")]
pub use crate::simd_avx512::*;

#[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
pub use crate::simd_avx2::*;

#[cfg(not(any(target_feature = "avx2", target_feature = "avx512f")))]
pub use crate::scalar_fns::*;

// ─── Functions with RUNTIME dispatch (these need all tiers at once) ───
// Hamming, popcount, dot_i8 use OnceLock fn pointers because they're
// called from hot loops where the caller selects the function ONCE
// and calls it millions of times. For these, keep runtime dispatch.

// Re-export the select_*_fn() functions that return fn pointers.
// These do is_x86_feature_detected! internally and cache the result.
// They are NOT affected by target_feature — they always compile all paths.

mod hamming_dispatch;
pub use hamming_dispatch::{
    hamming_distance, popcount, dot_i8,
    select_hamming_fn, select_dot_i8_fn,
    hamming_batch, hamming_top_k,
};
```

---

## STEP 2: Move hamming/popcount/dot_i8 dispatch to hamming_dispatch.rs

These functions are SPECIAL — they use runtime dispatch because callers
do `let f = select_hamming_fn()` and call `f` millions of times in a loop.
The `select_*_fn()` pattern requires runtime detection.

Create `rustynum-core/src/hamming_dispatch.rs` containing the existing
hamming_distance, popcount, dot_i8, hamming_batch, hamming_top_k,
select_hamming_fn, and select_dot_i8_fn functions UNCHANGED from
the current simd.rs. Cut and paste. Don't rewrite.

These functions already have correct three-tier dispatch:
  VPOPCNTDQ → AVX2 vpshufb → scalar count_ones

---

## STEP 3: Ensure scalar_fns.rs has EVERY function that simd.rs exports

The scalar module must be a COMPLETE replacement. Check:

```bash
# Every pub fn in simd_avx512.rs that simd.rs re-exports:
grep "^pub fn\|^    pub fn" rustynum-core/src/simd_avx512.rs

# Every pub fn in simd_avx2.rs:
grep "^pub fn" rustynum-core/src/simd_avx2.rs

# Every pub fn currently in scalar_fns.rs:
grep "^pub fn" rustynum-core/src/scalar_fns.rs
```

Any function in simd_avx512 or simd_avx2 that doesn't have a scalar
equivalent → add it to scalar_fns.rs. Scalar implementations are
simple loops — one or two lines each.

CRITICAL: the function SIGNATURES must be identical across all three
modules. Same name, same argument types, same return type. The re-export
in simd.rs only works if the names match.

Functions that exist in simd_avx512 but NOT simd_avx2:
- iamax_f32, iamax_f64 → add to simd_avx2.rs OR scalar fallback
- Element-wise ops (add/sub/mul/div_f32/f64_scalar/vec) → add to simd_avx2.rs OR scalar
- hdr_cascade_search → add scalar version to scalar_fns.rs

For functions without AVX2 versions: the scalar fallback is fine.
LLVM auto-vectorizes scalar loops to AVX2 when built with `-C target-cpu=native`.
You don't need hand-written AVX2 for element-wise add.

---

## STEP 4: Fix lib.rs

```rust
// BEFORE:
#[cfg(feature = "avx512")] pub mod simd;
#[cfg(all(feature = "avx2", not(feature = "avx512")))]
#[path = "simd_avx2.rs"] pub mod simd;

// AFTER:
#[cfg(target_arch = "x86_64")] pub mod simd_avx512;
#[cfg(target_arch = "x86_64")] pub mod simd_avx2;
pub mod scalar_fns;
pub mod simd;            // re-exports from the right module
pub mod simd_isa;
mod hamming_dispatch;    // runtime dispatch for hamming/popcount/dot_i8
```

---

## STEP 5: Fix ALL Cargo.tomls

```toml
# EVERY crate:
default = []
# Remove avx512 = [] and avx2 = [] features entirely.
# Remove features = ["avx512"] from dependency declarations.
# ISA is selected by the COMPILER, not by Cargo features.
```

Keep empty `avx512 = []` and `avx2 = []` ONLY if archive crates reference them.

---

## STEP 6: Fix consumer files that import simd_compat directly

```bash
grep -rn "simd_compat::" --include="*.rs" | grep -v target/
```

Replace `use rustynum_core::simd_compat::F32x16` with
`use rustynum_core::simd::F32x16` (which re-exports from the right module).

The types F32x16, F64x8, U8x64, etc. exist in simd_avx512.rs.
On AVX2/scalar builds, these types DON'T EXIST. Consumer files that
use them directly (level3.rs, bf16_gemm.rs, int8_gemm.rs, fft.rs, vml.rs)
must be gated with `#[cfg(target_feature = "avx512f")]` or rewritten
to use the Isa trait from simd_isa.rs.

For now: gate with `#[cfg(target_feature = "avx512f")]` and provide
scalar fallback functions. The Isa trait migration is a future session.

---

## STEP 7: Fill the two AVX2 gaps

### dot_i8: VNNI → scalar, no AVX2

The dot_i8 dispatch in hamming_dispatch.rs goes VNNI → scalar.
Add an AVX2 tier using VPMADDUBSW + VPMADDWD with the same
XOR-0x80 bias correction. Read the VNNI version first:

```bash
grep -A 60 "unsafe fn dot_i8_vnni" rustynum-core/src/hamming_dispatch.rs
```

Same algorithm, __m256i instead of __m512i, 32 bytes per iteration.

### bf16_hamming: AVX-512 → scalar, no AVX2

Read the AVX-512 version, write the AVX2 version with __m256i.
Same bit manipulation, half the lanes.

---

## STEP 8: CI workflow

```yaml
# .github/workflows/rust.yml
# Default build: no target-cpu flag → scalar, works everywhere
cargo test --workspace

# Optional: AVX2 build on x86 runners
# RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test --workspace

# The point: default cargo test NEVER needs AVX-512.
# AVX-512 builds are for production deployment on known hardware.
```

Add `rustup component add rustfmt` before the fmt check step.

---

## VERIFICATION

```bash
# Scalar (default, CI):
cargo test --workspace

# AVX2 (most x86 hardware):
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test --workspace

# AVX-512 (this server):
RUSTFLAGS="-C target-cpu=native" cargo test --workspace

# All three must pass.
```

---

## NOT IN SCOPE

```
× Don't add runtime dispatch for BLAS-1 or element-wise ops (compile-time is correct)
× Don't write a CommandSet struct or dispatch macro
× Don't rewrite simd_avx512.rs or simd_avx2.rs internals
× Don't touch hdr.rs
× Don't add GPU backends
```
