# CC SESSION REDIRECT — READ THIS FIRST

## Continue Stages 2-5 and 7-10 as planned. Change of plans for Stage 1 and Stage 6.

---

## STAGE 1 — CHANGE OF PLANS: Use simd_clean.rs, Not the Current simd.rs

The current `simd.rs` (2435 lines, 107 `is_x86_feature_detected!` calls) is being replaced.
Do NOT build backend traits on top of the current dispatch. It's dead code walking.

**Read this file FIRST:** `.claude/simd_clean.rs` (on main, 234 lines)

It uses:
- `LazyLock<Tier>` (Rust 1.94) — ONE detection at first call, cached forever
- `dispatch!` macro — ONE line per function, generates the match on tier
- Three tiers: `Avx512 → Avx2 → Scalar`
- `#[inline(always)]` on every dispatch function

```rust
// This is the ENTIRE dispatch for a function:
dispatch!(dot_f32(a: &[f32], b: &[f32]) -> f32);
```

The macro expands to:
```rust
#[inline(always)]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    match *TIER {
        Tier::Avx512 => crate::simd_avx512::dot_f32(a, b),
        Tier::Avx2   => crate::simd_avx2::dot_f32(a, b),
        Tier::Scalar => crate::scalar_fns::dot_f32(a, b),
    }
}
```

**Your backend trait should be THIS dispatch, not a new abstraction layer.**
The `dispatch!` macro IS the backend trait — it routes to the right implementation
based on detected hardware. Adding another trait on top adds indirection for zero benefit.

**Action for Stage 1:**
1. Replace `rustynum-core/src/simd.rs` with `.claude/simd_clean.rs`
2. Ensure `simd_avx512.rs` has `#[target_feature(enable = "avx512f")]` on every fn
3. Ensure `simd_avx2.rs` has `#[target_feature(enable = "avx2,fma")]` on every fn
4. Fill gaps in `scalar_fns.rs` (element-wise ops missing)
5. Wire `pub mod scalar_fns;` in `lib.rs`
6. Remove unnecessary `unsafe` blocks (Rust 1.94 safe intrinsics)
7. Run `cargo test --workspace` — must pass 1543+ tests
8. Benchmark sdot — must match or beat pre-PR#102 numbers (PR#102 introduced a 24% regression from the per-function unsafe wrapper pattern)

**Key: the kernels take `&[T]` slices. That interface doesn't change. Everything above calls `simd::dot_f32(&slice_a, &slice_b)`. The dispatch is invisible to callers.**

---

## STAGE 6 — CHANGE OF PLANS: ndarray IS the Product, Not a Migration Target

The original plan treats Stage 6 as "port NumArray to ArrayBase." That's too narrow.

**The new goal: make ndarray the better rustynum.** Not ndarray + extension traits on the side. ndarray with our SIMD as its engine. The user writes standard ndarray code and gets our speed.

**How this works:**

```rust
// The user writes standard ndarray:
use ndarray::array;
let a = array![1.0f32, 2.0, 3.0, 4.0];
let b = array![5.0f32, 6.0, 7.0, 8.0];
let dot = a.dot(&b);  // ← this calls OUR kernel

// Behind the scenes:
// ndarray's .dot() → our backend → simd::dot_f32(a.as_slice(), b.as_slice())
// → LazyLock tier → AVX-512 or AVX2 or scalar
// 
// The user NEVER imports rustynum_core. They import ndarray.
// ndarray is fast because our kernels are its backend.
```

**The backend integration pattern:**

ndarray supports BLAS backends via the `ndarray-linalg` / `blas-src` pattern.
We become a BLAS backend. When ndarray does matmul, it calls our `sgemm`.
When it does dot, it calls our `dot_f32`. Feature gate selects us:

```toml
[dependencies]
ndarray = { version = "0.16", features = ["rustynum-blas"] }
# OR
ndarray = { version = "0.16" }
rustynum-blas-src = { version = "0.1" }  # activates our BLAS as the backend
```

**For operations ndarray doesn't have (our unique value):**

```rust
// These are extension traits because ndarray has NO concept of them:
use rustynum_ndarray::HammingOps;      // hamming_distance, popcount
use rustynum_ndarray::Bf16Ops;         // bf16 conversion, bf16 hamming
use rustynum_ndarray::HdcOps;          // fingerprint, bundle, bind
use rustynum_ndarray::CascadeOps;      // cascade search on arrays

let binary_a = array![0u8; 2048];
let binary_b = array![0u8; 2048];
let dist = binary_a.hamming_distance(&binary_b);  // → VPOPCNTDQ
```

**Action for Stage 6:**
1. Create `rustynum-blas-src` crate that registers our kernels as BLAS provider
2. Create `rustynum-ndarray` crate with extension traits for operations ndarray lacks (hamming, bf16, hdc, cascade)
3. The user's import is `ndarray` with our feature flag. ONE product, not two.
4. Port ladybug-rs from `NumArray<T>` to `ndarray::Array<T, D>` with our extensions
5. Retire `rustynum-rs` (the old NumArray container)

**What stays the same:**
- `rustynum-core` (kernels on `&[T]`) — untouched, this is the engine
- `rustyblas` (GEMM, BLAS Level 2-3) — untouched, becomes the BLAS backend
- `rustymkl` (FFT, VML) — untouched
- Plane, Node, Seal, Fingerprint — untouched, these are `&[u8]`/`&[i8]`, not arrays
- All binary/cognitive types — untouched, they sit ABOVE ndarray

---

## STAGES 2-5 — CONTINUE AS PLANNED

These are the kernel stages. They operate on `&[T]` slices. Nothing tonight changes them.

```
Stage 2 (BLAS Level 1): dot, axpy, scal, nrm2, asum, iamax
  → these ARE the dispatch! entries in simd_clean.rs
  → ensure simd_avx2.rs has all of them (some were added in PR#102)
  → ensure scalar_fns.rs has all of them (currently incomplete)
  
Stage 3 (BLAS Level 2): gemv, trsv, ger
  → rustyblas already has these
  → add runtime dispatch (same pattern as simd_clean.rs)
  
Stage 4 (BLAS Level 3): sgemm, dgemm, bf16_gemm, int8_gemm
  → rustyblas/level3.rs already works on AVX-512
  → the blocked GEMM needs AVX2 microkernel (6×8 instead of 6×16)
  → SGEMM_NR must be runtime-selected based on tier, NOT hardcoded 16
  
Stage 5 (MKL + LAPACK): FFT, VML, FFI bindings
  → rustymkl already has these
  → continue as planned
```

**One critical fix for Stage 4:**

```rust
// CURRENT (broken): SGEMM_NR hardcoded to 16 (AVX-512 width)
pub const SGEMM_NR: usize = 16;

// FIXED: runtime NR based on tier
pub fn sgemm_nr() -> usize {
    match *TIER {
        Tier::Avx512 => 16,  // 16 floats per zmm register
        Tier::Avx2   => 8,   // 8 floats per ymm register
        Tier::Scalar => 4,   // 4-wide for cache line utilization
    }
}
```

This unblocks the AVX2 blocked GEMM path that currently falls through to `sgemm_simple`.

---

## STAGES 7-10 — CONTINUE AS PLANNED

```
Stage 7 (HDC + Binary):
  → Plane, Node, Fingerprint, BNN, CLAM, CAM
  → these operate on &[u8], &[i8], [u64; N]
  → they DON'T use ndarray. They're above it.
  → continue as planned

Stage 8 (CogRecord + Graph):
  → see .claude/INTEGRATION_SESSIONS.md for Sessions H, I, K, L
  → encounter_toward, encounter_away, project_all, bf16_from_projections
  → message passing, credit assignment
  → continue as planned, use the session prompts for specifics

Stage 9 (QA sweep):
  → after all stages
  → unsafe audit: remove unnecessary unsafe (Rust 1.94 safe intrinsics)
  → clippy: workspace-wide -D warnings
  → tests: restore the 3 excluded crates to CI

Stage 10 (Docs + publish):
  → after QA
  → README: update benchmarks (the 138 GFLOPS claim is real, don't change it)
  → CLAUDE.md: update with new file structure
```

---

## KEY FILES TO READ

Before continuing, read these `.claude/` files on main for context:

```
MUST READ:
  .claude/simd_clean.rs                    — the 234-line replacement for simd.rs
  .claude/SESSION_M_NDARRAY_MIGRATION.md   — full ndarray migration plan
  .claude/INVENTORY_MAP.md                 — what exists, what's missing, exact file paths
  .claude/L1_CACHE_BOUNDARY.md             — 64KB cliff, don't violate

OPTIONAL (architectural context):
  .claude/ARCHITECTURE_INDEX.md            — master index of everything
  .claude/BF16_SEMIRING_EPIPHANIES.md      — 5:2 semiring split
  .claude/SESSION_NARRATIVE.md             — what went wrong in PR#102, don't repeat
```

---

## SUMMARY

```
STAGE    STATUS          CHANGE
1        CHANGE          Use simd_clean.rs dispatch! macro. Not new abstraction.
2        CONTINUE        BLAS Level 1 kernels on &[T]
3        CONTINUE        BLAS Level 2 kernels on &[T]
4        CONTINUE        BLAS Level 3 + fix SGEMM_NR runtime selection
5        CONTINUE        MKL + LAPACK
6        CHANGE          ndarray IS the product. Our SIMD IS its engine. Not side-by-side.
7        CONTINUE        HDC + Binary (operates on &[u8], not arrays)
8        CONTINUE        CogRecord + Graph (see INTEGRATION_SESSIONS.md)
9        CONTINUE        QA sweep
10       CONTINUE        Docs + publish
```

Two stages change. Eight continue. The kernel work (Stages 2-5, 7-8) is unaffected.
The dispatch (Stage 1) gets cleaner. The container (Stage 6) gets better.
