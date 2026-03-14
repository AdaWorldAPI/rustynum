# SESSION_E_ISA_TRAIT.md

## Isa trait: bridge our stable SIMD types to portable_simd

**Repo:** rustynum (WRITE)
**Prereq:** Session A completed (simd_avx512.rs exists)
**Scope:** one new file (~80 lines), rewire level3.rs. Nothing else.
**Stop when:** `cargo test --workspace` passes, level3.rs uses `<I: Isa>`.

---

## CONTEXT: THE SYNTAX GAP

Stable Rust and nightly portable_simd are TWO COMPLETELY DIFFERENT LANGUAGES
for the same operations:

```
OPERATION          STABLE (core::arch)                    NIGHTLY (std::simd)
──────────────────────────────────────────────────────────────────────────────
Add 16 i32s        _mm512_add_epi32(a, b)                 a + b
Masked add         _mm512_mask_add_epi32(c, k, a, b)      k.select(a + b, c)
Reduce sum         _mm512_reduce_add_epi32(a)              a.reduce_sum()
FMA                _mm512_fmadd_ps(a, b, c)                a.mul_add(b, c)
Splat              _mm512_set1_ps(v)                       f32x16::splat(v)
XOR                _mm512_xor_si512(a, b)                  a ^ b
Popcount           _mm512_popcnt_epi64(a)                  (no direct equivalent)
Types              __m512i, __m512, __m512d                 Simd<i32,16>, f32x16
```

Stable is Intel's function-call vocabulary. Nightly is Rust operators and methods.
These are NOT similar. You cannot alias one to the other with a `type X = Y`.

OUR `simd_avx512.rs` (1500 lines) IS THE TRANSLATION LAYER. It wraps every
Intel intrinsic function call behind Rust operators and methods:

```rust
// Inside simd_avx512.rs:
impl Add for F32x16 {
    fn add(self, other: Self) -> Self {
        Self(unsafe { _mm512_add_ps(self.0, other.0) })  // Intel inside
    }
}
impl F32x16 {
    fn splat(v: f32) -> Self { Self(unsafe { _mm512_set1_ps(v) }) }
    fn reduce_sum(self) -> f32 { unsafe { _mm512_reduce_add_ps(self.0) } }
    fn mul_add(self, b: Self, c: Self) -> Self {
        Self(unsafe { _mm512_fmadd_ps(self.0, b.0, c.0) })
    }
}
```

AFTER this wrapping, our types and nightly portable_simd look identical:

```
OUR simd_avx512.rs:                 NIGHTLY std::simd:
let c = a + b;                      let c = a + b;               // same
let d = F32x16::splat(1.0);        let d = f32x16::splat(1.0);  // same
let e = a.mul_add(b, c);           let e = a.mul_add(b, c);     // same
let f = a.reduce_sum();            let f = a.reduce_sum();      // same
```

THAT is why the Isa trait is thin (~80 lines of one-liner forwards).
Not because the APIs happen to be similar. Because simd_avx512.rs was
PURPOSE-BUILT as the stable translation of portable_simd's operator syntax
into Intel's function-call syntax. The 1500 lines ARE the translation.
The trait is just the last mile — bridging type names (F32x16 vs f32x16).

`portable_simd` handles ALL architectures (x86, ARM, WASM, RISC-V) on nightly.
Our `simd_avx512.rs` + `simd_avx2.rs` handle x86 on stable.
The trait lets you switch with a feature flag. No separate `simd_arm.rs` needed —
`portable_simd` IS the ARM path.

---

## STEP 1: Read the ACTUAL implementations before writing anything

The AVX2 implementations in `simd.rs` are already optimized. Don't reinvent them.
Don't rename them to match paper names. Don't assume you know what algorithm they use.
READ THE CODE:

```bash
# AVX2 hamming (vpshufb nibble LUT + vpsadbw, blocks of 8 for u8 saturation):
sed -n '595,665p' rustynum-core/src/simd.rs

# AVX2 popcount (same nibble LUT pattern, standalone):
sed -n '802,900p' rustynum-core/src/simd.rs

# Scalar popcount (4x unrolled u64::count_ones):
sed -n '870,900p' rustynum-core/src/simd.rs

# dot_i8 dispatch (VNNI → scalar, NO AVX2 path — this is a known gap):
grep -n "fn dot_i8" rustynum-core/src/simd.rs

# AVX-512 hamming (VPOPCNTDQ, the fast path):
sed -n '665,730p' rustynum-core/src/simd.rs

# Our stable wrapper types (the portable_simd translation layer):
grep "pub fn" rustynum-core/src/simd_avx512.rs | head -30
```

The Avx2 Isa impl FORWARDS to these existing functions.
Do NOT rewrite them. Do NOT rename them. Just call them.

---

## STEP 2: Create simd_isa.rs (~80 lines)

```rust
//! ISA trait: bridge between our stable types and std::simd portable_simd.
//!
//! Our simd_avx512.rs IS the stable portable_simd. Method names match.
//! This trait lets kernels compile against either backend via feature flag.
//!
//! Default (stable): simd_avx512.rs types. Production. Ships everywhere.
//! Optional (nightly --features portable_simd): std::simd. All architectures.

pub trait Isa: Copy + 'static {
    type F32: Copy + std::ops::Add<Output = Self::F32>
                   + std::ops::Sub<Output = Self::F32>
                   + std::ops::Mul<Output = Self::F32>;
    type F64: Copy + std::ops::Add<Output = Self::F64>
                   + std::ops::Sub<Output = Self::F64>
                   + std::ops::Mul<Output = Self::F64>;
    type U8:  Copy;

    const F32_LANES: usize;
    const LABEL: &'static str;

    // Only methods that DIFFER between our types and std::simd.
    // Operators (+, -, *) work via trait bounds above.
    fn f32_splat(val: f32) -> Self::F32;
    fn f32_zero() -> Self::F32;
    fn f32_load(ptr: *const f32) -> Self::F32;
    fn f32_store(val: Self::F32, ptr: *mut f32);
    fn f32_fmadd(a: Self::F32, b: Self::F32, c: Self::F32) -> Self::F32;
    fn f32_reduce_sum(a: Self::F32) -> f32;
    fn u8_xor(a: Self::U8, b: Self::U8) -> Self::U8;
    fn u8_popcnt(a: Self::U8) -> u32;
}

// ═══════════════════════════════════════════════════════
// STABLE (default): forward to our simd_avx512.rs
// ═══════════════════════════════════════════════════════

#[cfg(not(feature = "portable_simd"))]
mod impls {
    use super::Isa;
    use crate::simd_avx512;

    #[derive(Clone, Copy)]
    pub struct Native;

    impl Isa for Native {
        type F32 = simd_avx512::F32x16;
        type F64 = simd_avx512::F64x8;
        type U8  = simd_avx512::U8x64;
        const F32_LANES: usize = 16;
        const LABEL: &'static str = "stable";

        // 1:1 forwards. Same method names on both sides.
        #[inline(always)] fn f32_splat(v: f32) -> Self::F32 { simd_avx512::F32x16::splat(v) }
        #[inline(always)] fn f32_zero() -> Self::F32 { simd_avx512::F32x16::splat(0.0) }
        #[inline(always)] fn f32_load(p: *const f32) -> Self::F32 {
            simd_avx512::F32x16::from_slice(unsafe { std::slice::from_raw_parts(p, 16) })
        }
        #[inline(always)] fn f32_store(v: Self::F32, p: *mut f32) {
            v.copy_to_slice(unsafe { std::slice::from_raw_parts_mut(p, 16) })
        }
        #[inline(always)] fn f32_fmadd(a: Self::F32, b: Self::F32, c: Self::F32) -> Self::F32 { a.mul_add(b, c) }
        #[inline(always)] fn f32_reduce_sum(a: Self::F32) -> f32 { a.reduce_sum() }
        #[inline(always)] fn u8_xor(a: Self::U8, b: Self::U8) -> Self::U8 { a ^ b }
        #[inline(always)] fn u8_popcnt(a: Self::U8) -> u32 {
            // CHECK: verify actual method name in simd_avx512.rs
            // May be .popcnt() or need to go through simd::popcount
            todo!("verify method name, forward to existing impl")
        }
    }
}

// ═══════════════════════════════════════════════════════
// NIGHTLY (opt-in): forward to std::simd
// ═══════════════════════════════════════════════════════

#[cfg(feature = "portable_simd")]
mod impls {
    use super::Isa;
    use std::simd::*;
    use std::simd::prelude::*;

    #[derive(Clone, Copy)]
    pub struct Native;

    impl Isa for Native {
        type F32 = f32x16;  // std::simd type, handles x86 + ARM + WASM + RISC-V
        type F64 = f64x8;
        type U8  = u8x64;
        const F32_LANES: usize = 16;
        const LABEL: &'static str = "portable_simd";

        // Same method names — portable_simd mirrors our API (or we mirror it).
        #[inline(always)] fn f32_splat(v: f32) -> Self::F32 { f32x16::splat(v) }
        #[inline(always)] fn f32_zero() -> Self::F32 { f32x16::splat(0.0) }
        #[inline(always)] fn f32_load(p: *const f32) -> Self::F32 {
            f32x16::from_slice(unsafe { std::slice::from_raw_parts(p, 16) })
        }
        #[inline(always)] fn f32_store(v: Self::F32, p: *mut f32) {
            v.copy_to_slice(unsafe { std::slice::from_raw_parts_mut(p, 16) })
        }
        #[inline(always)] fn f32_fmadd(a: Self::F32, b: Self::F32, c: Self::F32) -> Self::F32 {
            // std::simd: check if mul_add exists on Simd<f32, 16>
            // May need: SimdFloat::mul_add(a, b, c)
            a.mul_add(b, c)
        }
        #[inline(always)] fn f32_reduce_sum(a: Self::F32) -> f32 { a.reduce_sum() }
        #[inline(always)] fn u8_xor(a: Self::U8, b: Self::U8) -> Self::U8 { a ^ b }
        #[inline(always)] fn u8_popcnt(a: Self::U8) -> u32 {
            // std::simd doesn't have direct popcnt on u8x64.
            // Convert to bytes, use our simd::popcount which handles all arches.
            let bytes: [u8; 64] = a.to_array();
            crate::simd::popcount(&bytes) as u32
        }
    }
}

pub use impls::Native;

/// Runtime dispatch at kernel entry.
#[inline]
pub fn dispatch<R>(
    f_avx512: impl FnOnce() -> R,
    f_avx2: impl FnOnce() -> R,
    f_scalar: impl FnOnce() -> R,
) -> R {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") { return f_avx512(); }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") { return f_avx2(); }
    }
    f_scalar()
}
```

---

## STEP 3: Add to lib.rs + Cargo.toml

```rust
pub mod simd_isa;
```

```toml
[features]
portable_simd = []
```

---

## STEP 4: Rewire level3.rs

```rust
// BEFORE:
use rustynum_core::simd_avx512::{F32x16 as F32Simd};
fn sgemm_microkernel_6x16(/* ... */) {
    let b = F32Simd::load(ptr);  // SIGILL on non-AVX-512
}

// AFTER:
use rustynum_core::simd_isa::{self, Isa, Native};
fn sgemm_microkernel<I: Isa>(/* ... */) {
    let b = I::f32_load(ptr);  // compiles for any ISA
}
pub fn sgemm(/* args */) {
    simd_isa::dispatch(
        || sgemm_blocked::<Native>(/* args */),  // AVX-512 on this machine
        || sgemm_blocked::<Native>(/* args */),  // or AVX2, or scalar — Native resolves it
        || sgemm_blocked::<Native>(/* args */),
    );
}
```

---

## STEP 5: Same for bf16_gemm.rs, int8_gemm.rs

Replace `simd_avx512::F32x16` with `<I: Isa>` or `Native`.

---

## STEP 6: Verify

```bash
RUSTFLAGS="-C target-cpu=native" cargo test --workspace
RUSTFLAGS="-C target-cpu=native" cargo run --release --example gemm_benchmark -p rustyblas
```

---

## NOT IN SCOPE

```
× Don't create simd_arm.rs (portable_simd handles ARM on nightly,
  scalar + LLVM auto-vectorization handles ARM on stable)
× Don't add GPU backends
× Don't touch hdr.rs
× Don't rewrite simd_avx512.rs internals
× Don't add FP16/AMX to the trait
```
