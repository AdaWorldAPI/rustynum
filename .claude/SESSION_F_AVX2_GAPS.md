# SESSION_F_AVX2_GAPS.md

## Fill every AVX2 fallback gap. Eliminate all SIGILL paths.

**Repo:** rustynum (WRITE — rustynum-core + rustyblas + rustymkl)
**Prereq:** Session A (simd_avx512.rs renamed), Session E (Isa trait exists)
**Scope:** add AVX2 fallbacks where missing, add dispatch at every entry point
**Stop when:** `cargo test --workspace` passes WITHOUT `-C target-cpu=native`

That last point is critical. If you compile with default target (no AVX-512 assumed)
and ALL tests pass, there are no SIGILL paths left.

---

## THE COMPLETE SIGILL MAP

Every file that uses AVX-512 types without runtime dispatch. Read EACH before fixing.

```
FILE                              PROBLEM                         FIX
──────────────────────────────────────────────────────────────────────────────
rustynum-core/src/simd.rs
  dot_i8()                        VNNI → scalar, NO AVX2          Add AVX2 tier
  select_dot_i8_fn()              same gap                        Add AVX2 fn ptr

rustyblas/src/level3.rs
  sgemm_microkernel_6x16()        Uses F32x16 (=__m512) directly  Write 6x8 AVX2 kernel
  dgemm_microkernel_6x8()         Uses F64x8 (=__m512d) directly  Write 6x4 AVX2 kernel
  sgemm_macrokernel()             Calls 6x16 without dispatch     Add dispatch at entry
  SGEMM_NR = 16                   Hardcoded for AVX-512 width     Runtime NR selection

rustyblas/src/bf16_gemm.rs
  f32_to_bf16_slice()             Uses F32Simd (=F32x16) directly Scalar fallback
  f32_to_bf16_rounded()           Same                            Same
  bf16_to_f32_slice()             Same                            Same
  bf16_gemm_f32()                 Uses F32Simd in hot loop        Dispatch or Isa trait

rustyblas/src/int8_gemm.rs
  simd_abs_max()                  Uses f32x16 directly            AVX2 f32x8 fallback
  quantize_f32_to_u8()            Uses f32x16 directly            Same
  quantize_f32_to_i8()            Uses f32x16 directly            Same
  quantize_per_channel_i8()       Uses f32x16 directly            Same

rustynum-core/src/bf16_hamming.rs
  bf16_hamming_avx512()           AVX-512 → scalar, NO AVX2       Add AVX2 tier

rustynum-core/src/prefilter.rs
  (line 53)                       Uses f32x16 inside function     Guard or fallback

rustynum-rs/src/simd_ops/mod.rs   Imports f32x16, f64x8 etc      Re-route via Isa or dispatch
rustynum-rs/src/num_array/
  array_struct.rs                 Imports f32x16, f64x8 etc      Same
  bitwise.rs                      Uses u8x64 for hamming/popcount Goes through simd.rs (OK?)
  hdc.rs                          Imports u64x8                   Check if guarded

rustymkl/src/fft.rs               Imports f32x16, f64x8           Guard or fallback
rustymkl/src/vml.rs               Imports multiple 512 types      Guard or fallback
```

---

## FIX ORDER (dependencies matter)

### FIX 1: dot_i8 AVX2 tier in simd.rs

The simplest gap. dot_i8 jumps from VNNI to scalar. Add AVX2 middle tier.

READ the existing VNNI implementation first:
```bash
sed -n '914,1000p' rustynum-core/src/simd.rs
```

The AVX2 path uses VPMADDUBSW (unsigned × signed → i16 pairs) then
VPMADDWD (adjacent pair i16 → i32 sums). These are AVX2 instructions
available since Haswell (2013). Every modern x86 has them.

```rust
// Add to simd.rs, between dot_i8_vnni and dot_i8_scalar:

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dot_i8_avx2(a: &[u8], b: &[u8]) -> i64 {
    use core::arch::x86_64::*;
    
    let len = a.len();
    let chunks = len / 32;  // 256-bit = 32 bytes per __m256i
    
    let bias = _mm256_set1_epi8(0x80u8 as i8);  // signed → unsigned shift
    let ones_16 = _mm256_set1_epi16(1);           // for PMADDWD reduction
    let mut acc = _mm256_setzero_si256();
    let mut b_sum = _mm256_setzero_si256();
    
    for i in 0..chunks {
        let base = i * 32;
        let av = _mm256_loadu_si256(a[base..].as_ptr() as *const __m256i);
        let bv = _mm256_loadu_si256(b[base..].as_ptr() as *const __m256i);
        
        // Convert a from signed to unsigned-with-bias
        let av_u = _mm256_xor_si256(av, bias);
        
        // VPMADDUBSW: pairs of (u8 × i8) → i16, then adjacent pairs summed
        let prod = _mm256_maddubs_epi16(av_u, bv);
        
        // VPMADDWD: pairs of i16 → i32
        let widened = _mm256_madd_epi16(prod, ones_16);
        
        // Accumulate i32 lanes
        acc = _mm256_add_epi32(acc, widened);
        
        // Accumulate sum(b) for bias correction
        let b_prod = _mm256_maddubs_epi16(_mm256_set1_epi8(1), bv);
        let b_wide = _mm256_madd_epi16(b_prod, ones_16);
        b_sum = _mm256_add_epi32(b_sum, b_wide);
    }
    
    // Horizontal sum of 8 × i32 → i64
    let mut acc_vals = [0i32; 8];
    _mm256_storeu_si256(acc_vals.as_mut_ptr() as *mut __m256i, acc);
    let total_biased: i64 = acc_vals.iter().map(|&v| v as i64).sum();
    
    let mut bsum_vals = [0i32; 8];
    _mm256_storeu_si256(bsum_vals.as_mut_ptr() as *mut __m256i, b_sum);
    let total_b: i64 = bsum_vals.iter().map(|&v| v as i64).sum();
    
    let mut result = total_biased - 128 * total_b;
    
    // Scalar tail
    for i in (chunks * 32)..len {
        result += (a[i] as i8 as i64) * (b[i] as i8 as i64);
    }
    
    result
}
```

UPDATE the dispatch:

```rust
pub fn dot_i8(a: &[u8], b: &[u8]) -> i64 {
    assert_eq!(a.len(), b.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512f") {
            return unsafe { dot_i8_vnni(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_i8_avx2(a, b) };  // NEW
        }
    }
    dot_i8_scalar(a, b)
}

pub fn select_dot_i8_fn() -> fn(&[u8], &[u8]) -> i64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512f") {
            return |a, b| unsafe { dot_i8_vnni(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            return |a, b| unsafe { dot_i8_avx2(a, b) };  // NEW
        }
    }
    dot_i8_scalar
}
```

TEST: verify AVX2 matches VNNI matches scalar on same data.

---

### FIX 2: bf16_hamming AVX2 tier

READ the existing implementations:
```bash
sed -n '107,160p' rustynum-core/src/bf16_hamming.rs   # scalar
sed -n '158,244p' rustynum-core/src/bf16_hamming.rs   # avx512
sed -n '245,260p' rustynum-core/src/bf16_hamming.rs   # dispatch
```

The scalar does per-BF16-pair XOR + weighted field extraction.
The AVX-512 does bulk XOR + parallel field extraction.
AVX2 version: same algorithm as AVX-512 but on __m256i (16 BF16 pairs per iteration
instead of 32). The bit manipulation (mask, shift, popcount) maps directly.

Update `select_bf16_hamming_fn()` to include AVX2 tier.

---

### FIX 3: level3.rs GEMM dispatch

This is the biggest fix. The GEMM microkernel is hardcoded to 6×16 (AVX-512 width).

**Option A (use Isa trait from Session E):**
```rust
use rustynum_core::simd_isa::{self, Isa, Native};

fn sgemm_microkernel<I: Isa>(/* ... */) {
    // Generic kernel, I::F32_LANES determines NR
}

// Entry:
simd_isa::dispatch(
    || sgemm_blocked::<simd_isa::Avx512>(/* */),
    || sgemm_blocked::<simd_isa::Avx2>(/* */),
    || sgemm_blocked::<simd_isa::Scalar>(/* */),
);
```

**Option B (separate kernels, less refactoring):**
```rust
// Keep existing 6x16 kernel, add #[target_feature]:
#[target_feature(enable = "avx512f")]
unsafe fn sgemm_microkernel_6x16(/* ... */) { /* existing code */ }

// New 6x8 kernel using F32x8 (AVX2):
#[target_feature(enable = "avx2,fma")]
unsafe fn sgemm_microkernel_6x8(/* ... */) {
    // Same algorithm, half the lanes.
    // F32x8 instead of F32x16. NR=8 instead of NR=16.
}

// Scalar fallback:
fn sgemm_microkernel_scalar(/* ... */) { /* plain loops */ }

// Dispatch at sgemm() entry:
pub fn sgemm(/* args */) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { sgemm_blocked_avx512(/* args */) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return sgemm_blocked_avx2(/* args */);
        }
    }
    sgemm_blocked_scalar(/* args */);
}
```

NR must be runtime-selected:
```rust
fn sgemm_nr() -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") { return 16; }
        if is_x86_feature_detected!("avx2") { return 8; }
    }
    4  // scalar
}
```

Same for DGEMM: existing 6×8 (F64x8 = __m512d) becomes 6×4 on AVX2 (F64x4 = __m256d).

**Choose Option A or B based on how much the existing kernel code can be made generic.
Read level3.rs lines 490-600 (the sgemm microkernel) before deciding.**

---

### FIX 4: bf16_gemm.rs conversion + GEMM

The f32→bf16 conversion is trivial without SIMD — just bit-shift:
```rust
fn f32_to_bf16_scalar(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16  // truncate mantissa, that's it
}
```

Add scalar fallback paths for `f32_to_bf16_slice()`, `bf16_to_f32_slice()`,
and the GEMM itself. The GEMM can use the same scalar/AVX2 f32 GEMM
with bf16→f32 conversion at the edges.

---

### FIX 5: int8_gemm.rs quantize functions

`simd_abs_max()`, `quantize_f32_to_u8()`, `quantize_f32_to_i8()` use f32x16 directly.

Simplest fix: use the EXISTING `simd.rs` functions where possible:
```rust
fn simd_abs_max(data: &[f32]) -> f32 {
    // This is just asum / max reduction — simd.rs may already have it
    // Check: does simd.rs have iamax_f32 or asum_f32?
    // If yes, delegate. If no, write AVX2 + scalar inline.
}
```

For quantize loops: replace `f32x16::from_slice()` with a dispatch:
```rust
fn quantize_f32_to_u8(data: &[f32]) -> (Vec<u8>, QuantParams) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx512f") {
        return unsafe { quantize_f32_to_u8_avx512(data) };
    }
    quantize_f32_to_u8_scalar(data)  // plain loop, LLVM auto-vectorizes to AVX2
}
```

---

### FIX 6: rustynum-rs imports (array_struct, bitwise, simd_ops, hdc)

These files import `simd_compat::{f32x16, u8x64, ...}` which ARE __m512 types.

CHECK each usage:
```bash
grep -n "f32x16\|u8x64\|i32x16\|i64x8\|u64x8" rustynum-rs/src/num_array/*.rs | grep -v test
```

Most of these are in the impl blocks for NumArray operations. They need either:
- Dispatch wrapper (check if the op goes through simd.rs already)
- Or replacement with Isa trait

The bitwise ops (hamming_distance, popcount) already go through `simd.rs` via
the `HammingSimdOps` trait → `simd::hamming_distance()` → dispatched. These may be SAFE.
VERIFY by checking if `HammingSimdOps` calls simd_compat directly or via simd.rs.

---

### FIX 7: rustymkl fft.rs + vml.rs

```bash
grep -n "f32x16\|f64x8" rustymkl/src/fft.rs
grep -n "f32x16\|f64x8\|u8x64" rustymkl/src/vml.rs
```

These need the same treatment: dispatch at entry point, scalar fallback,
or Isa trait parameterization.

---

## VERIFICATION (THE REAL TEST)

The definitive test that ALL SIGILL paths are eliminated:

```bash
# Compile WITHOUT AVX-512 target features.
# Default target = whatever the CI runner has (usually just SSE4.2).
# If this passes, no SIGILL possible.

cargo test --workspace
# NOT: RUSTFLAGS="-C target-cpu=native" cargo test --workspace
# The point is to compile for generic x86_64, not for this specific CPU.
```

If any test calls an AVX-512 intrinsic without runtime dispatch,
it will SIGILL on a non-AVX-512 runner. Tests passing = safe.

Additionally:

```bash
# Cross-compile for aarch64 to verify ARM compatibility:
# (requires cross or cargo-zigbuild)
cargo check --target aarch64-unknown-linux-gnu

# If this compiles, no x86-specific code is in the default path.
```

---

## NOT IN SCOPE

```
× Don't touch simd_avx512.rs internals (it's the implementation, it SHOULD use __m512)
× Don't touch simd.rs dispatch logic for hamming/popcount (already correct)
× Don't add new SIMD algorithms (just add fallback tiers for existing ones)
× Don't touch hdr.rs (separate session)
× Don't touch Plane/Node/Mask (separate prompt)
× Don't rewrite existing working kernels (just add dispatch + fallback alongside)
```
