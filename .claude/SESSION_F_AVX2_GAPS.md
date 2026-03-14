# SESSION_F_AVX2_GAPS.md

## One binary. Runtime dispatch. No feature flags for ISA selection.

**Repo:** rustynum (WRITE — rustynum-core + rustyblas + rustymkl + rustynum-rs)
**Prereq:** Session E (simd_isa.rs exists with Isa trait + dispatch())
**Scope:** eliminate compile-time ISA selection, make every entry point runtime-dispatched
**Stop when:** `cargo test --workspace` passes with NO feature flags, NO RUSTFLAGS

---

## THE REAL PROBLEM

The codebase currently uses COMPILE-TIME feature gates for ISA selection:

```toml
# rustynum-core/Cargo.toml:
default = ["avx512"]

# lib.rs:
#[cfg(feature = "avx512")] pub mod simd;
#[cfg(all(feature = "avx2", not(feature = "avx512")))]
#[path = "simd_avx2.rs"] pub mod simd;
```

This means: `cargo build` compiles with `avx512` feature ON. Every function
using `F32x16` (which IS `__m512` inside) is compiled into the binary.
On ANY machine without AVX-512 — most laptops, all CI runners, all ARM —
calling these functions is SIGILL.

CI runs `cargo test --workspace` with no flags on ubuntu-latest and macos-latest.
Neither guarantees AVX-512. macOS-latest is ARM. This is broken.

The fix is NOT switching the default from `avx512` to `avx2`.
The fix is removing the choice entirely. One binary. Runtime dispatch.
The CPU decides, not Cargo.toml.

---

## WORKFLOW: READ BEFORE WRITING

```
1. READ the files you're about to change. Fully. Not grep. READ.
2. READ the files that IMPORT what you're changing.
3. PLAN the change in a comment or thinking block.
4. WRITE the code ONCE, correctly.
5. THEN compile to verify.

DO NOT:
× Compile first to "see what's there"
× Write stub code and iterate on compiler errors
× Use the compiler as a code explorer
× Fix errors one at a time in a loop

The compiler is a VERIFIER, not a NAVIGATOR.
```

---

## STEP 0: Understand the existing CORRECT pattern

`simd.rs` already has runtime dispatch for hamming/popcount. READ IT:

```bash
# The CORRECT pattern (already exists):
grep -A 15 "pub fn hamming_distance" rustynum-core/src/simd.rs
grep -A 15 "pub fn popcount" rustynum-core/src/simd.rs
grep -A 15 "pub fn select_hamming_fn" rustynum-core/src/simd.rs
```

These use `is_x86_feature_detected!` at runtime to pick AVX-512 vs AVX2 vs scalar.
This is the pattern. Replicate it everywhere.

Also READ the Isa trait from Session E:
```bash
cat rustynum-core/src/simd_isa.rs
```

The trait provides `dispatch()` for generic kernels. Use it for GEMM.

---

## STEP 1: Remove compile-time ISA feature gates

### 1a. rustynum-core/Cargo.toml

```toml
# BEFORE:
[features]
default = ["avx512"]
avx512 = []
avx2 = []
portable_simd = []

# AFTER:
[features]
default = []
portable_simd = []
# avx512 and avx2 features: REMOVED. ISA is runtime-detected.
```

### 1b. rustynum-core/src/lib.rs

```rust
# BEFORE:
#[cfg(feature = "avx512")] pub mod simd;
#[cfg(all(feature = "avx2", not(feature = "avx512")))]
#[path = "simd_avx2.rs"] pub mod simd;

# AFTER:
#[cfg(target_arch = "x86_64")] pub mod simd_avx512;  // AVX-512 impls, always compiled on x86
#[cfg(target_arch = "x86_64")] pub mod simd_avx2;    // AVX2 impls, always compiled on x86
pub mod simd;      // runtime dispatch: picks best available at runtime
pub mod simd_isa;  // Isa trait: generic kernels + portable_simd bridge
```

NOTE: `simd_avx512.rs` and `simd_avx2.rs` use `#[target_feature(enable = "...")]`
on every function. They compile on any x86_64 regardless of CPU.
The `unsafe` + `#[target_feature]` is what makes this work — the compiler
generates the AVX-512 instructions but they're only CALLED after runtime detection.

### 1c. Remove feature gates INSIDE simd.rs

The dispatch functions in `simd.rs` may be behind `#[cfg(feature = "avx512")]`.
Remove those gates. The functions should be unconditional.
Their INTERNAL dispatch uses `is_x86_feature_detected!` — that stays.

```bash
# Find all feature gates in simd.rs:
grep -n "cfg.*feature.*avx" rustynum-core/src/simd.rs
```

Remove every `#[cfg(feature = "avx512")]` and `#[cfg(feature = "avx2")]`.
Replace with `#[cfg(target_arch = "x86_64")]` where the code uses x86 intrinsics.

### 1d. Same for all other Cargo.tomls

```bash
# Find all avx512/avx2 feature references:
grep -rn "avx512\|avx2" */Cargo.toml
grep -rn "avx512\|avx2" Cargo.toml
```

Remove `features = ["avx512"]` from dependency declarations.
Remove `default = ["avx512"]` from every crate.
The `avx512` and `avx2` features cease to exist.

### 1e. Remove all `#[cfg(feature = "avx512")]` / `#[cfg(feature = "avx2")]` in ALL .rs files

```bash
grep -rn 'cfg.*feature.*avx' --include="*.rs" | grep -v target/
```

Replace each with the appropriate `#[cfg(target_arch = "x86_64")]`
or remove entirely if the function is behind a runtime `is_x86_feature_detected!`.

---

## STEP 2: Make simd.rs the universal dispatcher

After Step 1, `simd.rs` is always compiled. It must provide every function
that consumers need, with runtime dispatch inside each one.

READ what `simd.rs` currently exports:
```bash
grep "pub fn" rustynum-core/src/simd.rs
```

For each function, verify it has the three-tier dispatch:
```rust
pub fn some_operation(args) -> result {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { simd_avx512::some_operation(args) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_avx2::some_operation(args) };
        }
    }
    scalar_fallback(args)
}
```

Functions that ALREADY have this pattern: hamming_distance, popcount.
Functions that are MISSING AVX2: dot_i8 (VNNI → scalar, no AVX2).
Functions that may be MISSING entirely in simd_avx2.rs: check by reading both files.

### Fill the dot_i8 AVX2 gap

READ the existing VNNI implementation:
```bash
grep -A 60 "unsafe fn dot_i8_vnni" rustynum-core/src/simd.rs
```

The AVX2 version uses VPMADDUBSW + VPMADDWD with the same XOR-0x80 bias
correction pattern. Write it using the VNNI version as template but with
__m256i (32 bytes per iteration instead of 64).

Update dot_i8() and select_dot_i8_fn() to include the AVX2 tier.

### Fill the bf16_hamming AVX2 gap

READ:
```bash
grep -A 80 "unsafe fn bf16_hamming_avx512" rustynum-core/src/bf16_hamming.rs
grep -A 15 "fn select_bf16_hamming_fn" rustynum-core/src/bf16_hamming.rs
```

Same algorithm, __m256i instead of __m512i. Half the lanes per iteration.
Update select_bf16_hamming_fn() to include AVX2.

---

## STEP 3: Consumers that use wrapper types directly

These files import `simd_compat::{F32x16, f32x16, u8x64, ...}` which ARE
`__m512` types. After Step 1, these types still exist in `simd_avx512.rs`
but they can only be used inside `#[target_feature]` functions.

The fix for each file depends on what it does:

### 3a. rustyblas/src/level3.rs — GEMM microkernels

```rust
// Currently:
use rustynum_core::simd_compat::{F32x16 as F32Simd, F64x8 as F64Simd};
```

This is the biggest change. The GEMM microkernel must be generic over ISA.
Use the Isa trait from Session E:

```rust
use rustynum_core::simd_isa::{self, Isa};

fn sgemm_microkernel<I: Isa>(/* ... */) {
    let mut acc = [I::f32_zero(); 6];
    for p in 0..k {
        let b = I::f32_load(b_panel[p * I::F32_LANES..].as_ptr());
        // ... same algorithm, I:: instead of F32Simd::
    }
}

pub fn sgemm(/* args */) {
    simd_isa::dispatch(
        || sgemm_blocked::<simd_isa::Avx512>(/* args */),
        || sgemm_blocked::<simd_isa::Avx2>(/* args */),
        || sgemm_blocked::<simd_isa::Scalar>(/* args */),
    );
}
```

NR (number of register-width columns) becomes `I::F32_LANES`.
Panel packing and cache blocking must use the runtime NR.

READ the FULL level3.rs before rewriting. Understand the panel packing,
the macrokernel dispatch, the MR/NR constants, the threading model.
Then rewrite ONCE. Do not iterate on compiler errors.

### 3b. rustyblas/src/bf16_gemm.rs — BF16 conversion + GEMM

The conversion functions (`f32_to_bf16_slice`, `bf16_to_f32_slice`) use F32x16.

Simplest correct fix: dispatch at each entry point.
```rust
pub fn f32_to_bf16_slice(src: &[f32], dst: &mut [BF16]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        return unsafe { f32_to_bf16_slice_avx512(src, dst) };
    }
    f32_to_bf16_slice_scalar(src, dst)
    // Scalar is one line per element: (v.to_bits() >> 16) as u16
    // LLVM auto-vectorizes this to AVX2 on capable hardware.
}
```

The BF16 GEMM itself can dispatch to the scalar GEMM with bf16→f32 conversion
at the edges. No need for a separate AVX2 BF16 GEMM kernel.

### 3c. rustyblas/src/int8_gemm.rs — quantize functions

`simd_abs_max()`, `quantize_f32_to_u8()`, `quantize_f32_to_i8()` use f32x16.

Replace with dispatch at entry:
```rust
fn simd_abs_max(data: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        return unsafe { simd_abs_max_avx512(data) };
    }
    // Scalar: plain loop. LLVM auto-vectorizes.
    data.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
}
```

The quantize loops: same pattern. Move the f32x16 code into an
`unsafe fn quantize_avx512(...)` with `#[target_feature(enable = "avx512f")]`,
add a scalar fallback, dispatch at entry.

Note: int8_gemm.rs already has `int8_gemm_vnni_512` AND `int8_gemm_vnni_256`.
READ the existing dispatch in `int8_gemm_i32()` — it may already be correct.

### 3d. rustynum-rs/src/num_array/array_struct.rs, simd_ops/mod.rs

```bash
grep -n "simd_compat" rustynum-rs/src/num_array/array_struct.rs
grep -n "simd_compat" rustynum-rs/src/simd_ops/mod.rs
```

These import the wrapper types for NumArray operations (sum, dot, axpy, etc.).
CHECK: do these operations go through `simd.rs` functions (which are dispatched)?
Or do they use the types directly in inline loops?

If they go through `simd.rs` → already safe after Step 2.
If they use types directly → need the same dispatch-at-entry treatment.

### 3e. rustynum-rs/src/num_array/bitwise.rs, hdc.rs

```bash
grep -n "simd_compat\|HammingSimdOps" rustynum-rs/src/num_array/bitwise.rs
```

Bitwise ops likely go through `HammingSimdOps` → `simd::hamming_distance()`.
If so, already safe. VERIFY by reading the trait impl.

### 3f. rustymkl/src/fft.rs, rustymkl/src/vml.rs

```bash
grep -n "simd_compat" rustymkl/src/fft.rs
grep -n "simd_compat" rustymkl/src/vml.rs
```

Same treatment: dispatch at entry or route through simd.rs.

### 3g. rustynum-core/src/prefilter.rs

```bash
sed -n '50,60p' rustynum-core/src/prefilter.rs
```

Uses f32x16. Same fix: dispatch at entry or scalar fallback.

---

## STEP 4: Update all `simd_compat::` imports

After Step 1, `simd_compat.rs` is a deprecated re-export shim pointing to
`simd_avx512.rs`. Any file still importing from `simd_compat::` needs updating.

```bash
grep -rn "simd_compat::" --include="*.rs" | grep -v "target/"
```

For files that need runtime dispatch: replace with `use crate::simd;` or
`use rustynum_core::simd;` and call the dispatched functions.

For files that use the Isa trait: replace with `use crate::simd_isa::Isa;`.

For files that are INSIDE `#[target_feature(enable = "avx512f")]` functions:
they can keep using `simd_avx512::F32x16` directly — that's correct, the
target_feature guard makes it safe.

---

## VERIFICATION

```bash
# THE definitive test. No flags. No RUSTFLAGS. No features.
# If this passes, every code path is safe on any x86_64.
cargo test --workspace

# Also verify that AVX-512 path still works when available:
RUSTFLAGS="-C target-cpu=native" cargo test --workspace

# Clippy clean:
cargo clippy --workspace -- -D warnings

# If cross toolchain available, verify ARM compiles:
cargo check --target aarch64-unknown-linux-gnu
```

---

## NOT IN SCOPE

```
× Don't rewrite simd_avx512.rs internals (it SHOULD use __m512 — it's the AVX-512 impl)
× Don't rewrite simd_avx2.rs internals (already optimized, don't assume algorithms)
× Don't touch hdr.rs (Sessions C/D)
× Don't touch Plane/Node/Mask
× Don't add new SIMD algorithms — only add dispatch + scalar fallbacks
× Don't add the portable_simd nightly path (Session E already did the trait)
× Don't rename files or move code between files (just add dispatch wrappers)
```
