# SESSION_M_NDARRAY_MIGRATION.md

## Mission: Replace NumArray with ndarray, Keep Kernels, Bridge with Extension Traits

**Prereqs:** Session G (simd_clean.rs) should be done first but is NOT blocking.
The kernels take `&[T]`. ndarray gives `as_slice() → &[T]`. Orthogonal.
**Effort:** 1-2 weeks
**Branch:** claude/ndarray-migration

---

## THE PRINCIPLE

RustyNum's value was NEVER the array container. It's the kernels.
Keep the kernels. Use the best container. ndarray has 64M downloads
and 10+ years of iteration. We can't out-engineer that.

---

## STEP 1: Create rustynum-ndarray crate (1-2 days)

New crate in the workspace. Thin extension traits bridging ndarray → rustynum-core.

```toml
# rustynum-ndarray/Cargo.toml
[package]
name = "rustynum-ndarray"
version = "0.1.0"

[dependencies]
ndarray = "0.16"
rustynum-core = { path = "../rustynum-core" }
```

```rust
// rustynum-ndarray/src/lib.rs
use ndarray::{ArrayBase, Data, Ix1, Ix2};
use rustynum_core::simd;

/// Extension trait: SIMD-accelerated operations on ndarray vectors.
/// Dispatches to rustynum-core kernels via as_slice() → &[T].
pub trait SimdArrayOps {
    fn simd_dot(&self, other: &Self) -> f32;
    fn simd_nrm2(&self) -> f32;
    fn simd_asum(&self) -> f32;
    fn simd_axpy(&mut self, alpha: f32, x: &Self);
    fn simd_scal(&mut self, alpha: f32);
}

impl<S: Data<Elem = f32>> SimdArrayOps for ArrayBase<S, Ix1> {
    fn simd_dot(&self, other: &Self) -> f32 {
        match (self.as_slice(), other.as_slice()) {
            (Some(a), Some(b)) => simd::dot_f32(a, b),
            _ => {
                // Non-contiguous fallback: use ndarray's own dot
                // This handles views, slices with stride, transposes
                self.dot(other)
            }
        }
    }
    // ... similar for nrm2, asum, axpy, scal
}

/// Extension trait: Hamming distance on binary arrays.
pub trait HammingOps {
    fn hamming_distance(&self, other: &Self) -> u64;
    fn popcount(&self) -> u64;
}

impl<S: Data<Elem = u8>> HammingOps for ArrayBase<S, Ix1> {
    fn hamming_distance(&self, other: &Self) -> u64 {
        match (self.as_slice(), other.as_slice()) {
            (Some(a), Some(b)) => simd::hamming_distance(a, b),
            _ => {
                // Fallback: copy to contiguous, then SIMD
                let a_owned = self.to_owned();
                let b_owned = other.to_owned();
                simd::hamming_distance(a_owned.as_slice().unwrap(), b_owned.as_slice().unwrap())
            }
        }
    }
}

/// Extension trait: GEMM on 2D arrays.
pub trait GemmOps {
    fn simd_gemm(&self, other: &Self) -> ndarray::Array2<f32>;
}

impl<S: Data<Elem = f32>> GemmOps for ArrayBase<S, Ix2> {
    fn simd_gemm(&self, other: &Self) -> ndarray::Array2<f32> {
        let (m, k) = self.dim();
        let (_, n) = other.dim();
        let mut c = ndarray::Array2::<f32>::zeros((m, n));
        
        match (self.as_slice(), other.as_slice(), c.as_slice_mut()) {
            (Some(a), Some(b), Some(c_slice)) => {
                // Contiguous: use our GEMM kernel
                rustyblas::level3::sgemm(
                    m, n, k,
                    1.0, a, k,    // A is m×k row-major
                    b, n,          // B is k×n row-major
                    0.0, c_slice, n // C is m×n row-major
                );
            }
            _ => {
                // Non-contiguous: use ndarray's dot
                c = self.dot(other);
            }
        }
        c
    }
}
```

### What ndarray gives us FOR FREE

```
THINGS WE DELETE FROM RUSTYNUM-RS:            NDARRAY EQUIVALENT:
NumArray<T, Ops> struct                    →  ndarray::Array<T, D>
shape checking, dim validation             →  compile-time Ix1, Ix2, IxDyn
stride arithmetic                          →  ndarray handles internally
view implementation                        →  ArrayView, s![] macro
Display, Debug, PartialEq                  →  derived by ndarray
Serde serialization                        →  ndarray-serde
broadcast                                  →  ndarray broadcast (we can't do this at all)
transpose                                  →  .t() zero-cost view
reshape                                    →  .into_shape()
iteration                                  →  .iter(), .axis_iter(), Zip
```

---

## STEP 2: Verify rustynum-core is UNTOUCHED (0 days)

The whole point: rustynum-core kernels take `&[T]` slices.
ndarray's `as_slice()` returns `Option<&[T]>`.
The bridge is the `match` in Step 1. rustynum-core doesn't change.

```
FILES THAT DON'T CHANGE:
  rustynum-core/src/simd.rs          — dispatch on &[T], untouched
  rustynum-core/src/simd_avx512.rs   — AVX-512 kernels on &[T], untouched
  rustynum-core/src/simd_avx2.rs     — AVX2 kernels on &[T], untouched
  rustynum-core/src/scalar_fns.rs    — scalar fallbacks on &[T], untouched
  rustynum-core/src/plane.rs         — Plane uses Fingerprint<256>, not arrays
  rustynum-core/src/node.rs          — Node uses Plane, not arrays
  rustynum-core/src/seal.rs          — Seal uses blake3, not arrays
  rustynum-core/src/hdr.rs           — Cascade uses &[u8], not arrays
  rustynum-core/src/bf16_hamming.rs  — BF16 uses &[u8], not arrays
  rustynum-core/src/fingerprint.rs   — Fingerprint is [u64; N], not arrays
  rustynum-core/src/blackboard.rs    — Blackboard is raw arena, not arrays
  
  ALL of these operate on &[T], &[u8], &[i8], or [u64; N].
  NONE of them know or care about array containers.
  ndarray migration does NOT touch rustynum-core.
```

---

## STEP 3: Migrate ladybug-rs consumers (3-5 days)

This is the biggest cost. Every place that uses `NumArray<f32, SimdOps32>`
changes to `ndarray::Array1<f32>` (or Array2, ArrayDyn as appropriate).

```rust
// BEFORE (rustynum-rs NumArray):
use rustynum_rs::NumArray;
let a = NumArray::<f32, _>::from_vec(vec![1.0, 2.0, 3.0]);
let b = NumArray::<f32, _>::from_vec(vec![4.0, 5.0, 6.0]);
let dot = a.dot(&b);  // uses SimdOps trait

// AFTER (ndarray + extension):
use ndarray::array;
use rustynum_ndarray::SimdArrayOps;
let a = array![1.0f32, 2.0, 3.0];
let b = array![4.0f32, 5.0, 6.0];
let dot = a.simd_dot(&b);  // dispatches to simd::dot_f32 via as_slice()
```

### Search pattern for migration

```bash
# Find all NumArray usage in ladybug-rs
grep -rn "NumArray\|rustynum_rs\|SimdOps" --include="*.rs" ladybug-rs/
# Find all rustynum-rs imports
grep -rn "use rustynum_rs" --include="*.rs" ladybug-rs/
```

### Types that change

```
BEFORE:                          AFTER:
NumArray<f32, SimdOps32>     →  ndarray::Array1<f32>
NumArray<f64, SimdOps64>     →  ndarray::Array1<f64>
NumArray<i32, SimdOpsI32>    →  ndarray::Array1<i32>
2D operations (manual shape) →  ndarray::Array2<f32>
Dynamic shape                →  ndarray::ArrayD<f32>
```

### Types that DON'T change

```
UNCHANGED (these are &[u8]/&[i8], not arrays):
  Plane               — stays Plane (i8 accumulator, not an array)
  Node                — stays Node (3 × Plane)
  Fingerprint<256>    — stays Fingerprint (u64 array, not ndarray)
  BitVec              — stays BitVec (u64 array in lance-graph)
  PackedQualia        — stays PackedQualia (custom struct)
  BF16 types          — stays &[u8] (byte-level operations)
```

---

## STEP 4: Retire rustynum-rs (1 day)

After ladybug-rs migration is complete and all tests pass:

```
1. Remove rustynum-rs from default-members in workspace Cargo.toml
2. Move to archive/ directory (don't delete yet)
3. Remove from CI test exclusion list
4. Update CLAUDE.md to reflect the change
```

---

## STEP 5: Alignment consideration (non-blocking)

AVX-512 wants 64-byte alignment. ndarray uses standard Vec<T> (8/16-byte).

```
OPTION A (recommended): Don't worry about it.
  Unaligned AVX-512 loads (_mm512_loadu_ps) work fine.
  Penalty: ~1-3% on modern CPUs. Negligible.
  rustynum-core already uses unaligned loads throughout.

OPTION B (hot paths only): Copy to aligned buffer.
  For GEMM on large matrices: copy to Blackboard-allocated aligned memory.
  The copy cost is amortized over the O(N³) GEMM computation.
  
  Blackboard.alloc_aligned(size) → 64-byte aligned &mut [f32]
  Copy ndarray slice → aligned buffer → GEMM → copy result back
  
  Only for GEMM. Everything else: unaligned is fine.
```

---

## STEP 6: What rustynum-ndarray enables

```
NEW CAPABILITY:                               HOW:
Zero-cost transpose                        →  ndarray .t() (view, no copy)
Broadcasting (add scalar to matrix)        →  ndarray broadcast
Slicing with strides (every Nth element)   →  ndarray s![..;N]
2D/3D/ND operations                        →  ndarray Ix2, Ix3, IxDyn
ndarray-stats (mean, std, quantile)        →  extension crate, free
ndarray-rand (random arrays)               →  extension crate, free
ndarray-linalg (SVD, eigenvalue)           →  extension crate, free
Serde serialization of arrays              →  ndarray-serde, free
Display/Debug for arrays                   →  ndarray derives, free
PyO3 numpy interop                         →  numpy crate bridge to ndarray
```

---

## CONNECTION TO TONIGHT'S ARCHITECTURE

```
NDARRAY MIGRATION IS ORTHOGONAL TO:
  simd_clean.rs dispatch (kernels take &[T], unchanged)
  Plane/Node/Seal (custom types, not arrays)
  graph-flow orchestration (operates on Context, not arrays)
  LanceDB truth cache (BF16 columns, not arrays)
  GPU tensor core NARS (BF16 matrices, separate pipeline)
  Cascade search (operates on &[u8], not arrays)
  BF16 truth assembly (bit packing, not arrays)
  
NDARRAY HELPS WITH:
  GEMM performance comparison (ndarray-linalg vs our rustyblas)
  Statistics for cascade calibration (ndarray-stats: mean, std)
  Random initialization for tests (ndarray-rand)
  Serialization for Lance columns (ndarray → Arrow → Lance)
  Python bindings (ndarray → numpy via PyO3)
```

---

## SESSION ORDER

```
Session G:  simd_clean.rs refactor (dispatch layer)     ← DO FIRST
Session M:  ndarray migration (container layer)          ← DO SECOND
Session H:  Plane evolution (encounter_toward/away)      ← INDEPENDENT
Session I:  BF16 truth assembly                          ← NEEDS H
Session J:  PackedDatabase                               ← INDEPENDENT

G and M are sequential (G simplifies what M bridges to).
H, I, J are independent of both G and M.
All can proceed after G is merged.
```
