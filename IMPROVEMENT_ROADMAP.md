# RustyNum: Road to ndarray-Level Maturity

## Complete Improvement & Refactoring Guide

**Date:** February 25, 2026
**Scope:** All rustynum crates (rustynum-core, rustynum-rs, rustyblas, rustymkl)
**Reference:** ndarray v0.17.2 (the gold standard for Rust N-dimensional arrays)

---

## Table of Contents

1. [Critical: Type System & Memory Safety](#1-critical-type-system--memory-safety)
2. [Critical: Error Handling](#2-critical-error-handling)
3. [Critical: Unsafe Code Audit](#3-critical-unsafe-code-audit)
4. [High: Missing Core Abstractions](#4-high-missing-core-abstractions)
5. [High: API Design & Ergonomics](#5-high-api-design--ergonomics)
6. [High: Robustness & Edge Cases](#6-high-robustness--edge-cases)
7. [Medium: Code Duplication & Maintainability](#7-medium-code-duplication--maintainability)
8. [Medium: Performance Gaps](#8-medium-performance-gaps)
9. [Medium: Documentation](#9-medium-documentation)
10. [Medium: Testing & CI](#10-medium-testing--ci)
11. [Low: Ecosystem & Publishing](#11-low-ecosystem--publishing)
12. [ndarray Best Practices to Adopt (with code examples)](#12-ndarray-best-practices-to-adopt)

---

## 1. Critical: Type System & Memory Safety

### 1.1 No Compile-Time Dimension Safety

**Current state:** `NumArray<T, Ops>` uses `Vec<usize>` for shape — all dimension checks happen at runtime.

**ndarray approach:** `ArrayBase<S, D>` where `D: Dimension` — compile-time dimension tracking via `Ix1`, `Ix2`, ..., `IxDyn`.

```rust
// ndarray: compiler catches dimension errors
// File: ndarray/src/dimension/dim.rs
pub type Ix1 = Dim<[Ix; 1]>;
pub type Ix2 = Dim<[Ix; 2]>;
// ...
pub type IxDyn = Dim<IxDynImpl>;

// You can't accidentally pass a 2D array where 3D is expected
fn process_matrix(arr: &Array2<f64>) { ... }  // Compile-time enforced
```

**Recommendation:** Add a `Dim` type parameter to `NumArray`:
```rust
// Proposed rustynum improvement
pub struct NumArray<T, D: Dimension, Ops> {
    data: Vec<T>,
    dim: D,           // Compile-time dimension
    strides: D,       // Same type as dim
    _ops: PhantomData<Ops>,
}
pub type Vector<T> = NumArray<T, Ix1, ...>;
pub type Matrix<T> = NumArray<T, Ix2, ...>;
```

**Files to change:**
- `rustynum-rs/src/num_array/array_struct.rs:55-63` — struct definition
- All files in `rustynum-rs/src/num_array/` — every method signature

### 1.2 No Ownership/View Distinction

**Current state:** `NumArray` always owns its data (`Vec<T>`). Every slice/subarray operation copies.

**ndarray approach:** `ArrayBase<S, D>` separates storage strategy from array logic:

```rust
// ndarray/src/data_traits.rs:28-47
// Storage traits: RawData → Data → DataOwned → DataMut
pub unsafe trait RawData: Sized {
    type Elem;
}
pub unsafe trait Data: RawData {
    fn into_base_iter(/* ... */);
}

// This allows:
type Array<T, D>     = ArrayBase<OwnedRepr<T>, D>;     // Owns data
type ArrayView<T, D> = ArrayBase<ViewRepr<&T>, D>;      // Borrows data
type ArcArray<T, D>  = ArrayBase<OwnedArcRepr<T>, D>;   // Shared ownership
type CowArray<T, D>  = ArrayBase<CowRepr<T>, D>;        // Copy-on-write
```

**Recommendation:** Introduce a `Storage` trait:
```rust
pub trait Storage {
    type Elem;
    fn as_ptr(&self) -> *const Self::Elem;
    fn len(&self) -> usize;
}
pub struct Owned<T>(Vec<T>);
pub struct View<'a, T>(&'a [T]);
pub struct ViewMut<'a, T>(&'a mut [T]);

pub struct NumArray<S: Storage, D: Dimension, Ops> { ... }
```

### 1.3 Strides Are Not Properly Used

**Current state:** `rustynum-rs/src/num_array/array_struct.rs:61` — strides exist but most operations ignore them and assume contiguous row-major layout.

**ndarray approach:** Every operation respects strides. Transpose just swaps strides (zero-cost, 0.19 ns). Slicing adjusts strides and pointer offset without copying.

```rust
// ndarray: transpose is zero-cost
// ndarray/src/impl_methods.rs
pub fn t(&self) -> ArrayView<'_, A, D> {
    // Just reverses the axes (strides) — no data movement
    let mut d = self.dim.clone();
    let mut s = self.strides.clone();
    d.slice_mut().reverse();
    s.slice_mut().reverse();
    // Returns a view with swapped strides
}
```

**rustynum today:**
```rust
// rustynum-rs/src/num_array/manipulation.rs — transpose COPIES data
pub fn transpose(&self) -> Self {
    let mut result = vec![T::default(); self.data.len()];
    // ... copies every element ...
    NumArray::new_with_shape(result, new_shape)
}
```

**Recommendation:** Make all operations stride-aware. Transpose, slice, reshape should return views (no copy).

---

## 2. Critical: Error Handling

### 2.1 No Error Type for Array Operations

**Current state:** RustyNum has **zero** error types in rustynum-rs or rustyblas. The only error types exist in jitson (`ParseError`, `ValidationError`).

**Stats:**
- **189 `unwrap()` calls** across the codebase
- **18 `expect()` calls**
- **19 `panic!()` calls**
- **0 `Result<>` returns** from public array/BLAS functions

**ndarray approach:** Proper `ShapeError` with categorized `ErrorKind`:

```rust
// ndarray/src/error.rs:15-57
pub struct ShapeError { repr: ErrorKind }

#[non_exhaustive]
pub enum ErrorKind {
    IncompatibleShape = 1,
    IncompatibleLayout,
    RangeLimited,
    OutOfBounds,
    Unsupported,
    Overflow,
}

// ndarray returns Result from fallible operations:
pub fn into_shape_with_order<E>(self, shape: E) -> Result<...> { ... }
pub fn broadcast<E>(&self, dim: E) -> Option<ArrayView<...>> { ... }
```

**Recommendation:** Create a comprehensive error system:

```rust
// Proposed: rustynum-rs/src/error.rs
#[non_exhaustive]
pub enum ArrayError {
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    DimensionMismatch { expected: usize, got: usize },
    OutOfBounds { index: Vec<usize>, shape: Vec<usize> },
    EmptyArray,
    SingularMatrix,
    InvalidAxis { axis: usize, ndim: usize },
    InvalidStride,
    Overflow,
    NumericalError(String),  // NaN, Inf, convergence failure
}

impl std::error::Error for ArrayError {}
impl std::fmt::Display for ArrayError { ... }

pub type ArrayResult<T> = Result<T, ArrayError>;
```

**Files to change (all `panic!`/`unwrap` → `Result`):**
- `rustynum-rs/src/num_array/array_struct.rs` — 34 public functions
- `rustynum-rs/src/num_array/operations.rs:1` — already has a TODO for this!
- `rustynum-rs/src/num_array/manipulation.rs` — reshape, slice, concatenate
- `rustynum-rs/src/num_array/statistics.rs` — mean, std, percentile
- `rustynum-rs/src/num_array/linalg.rs` — matrix_multiply
- `rustyblas/src/level1.rs` — all 16 functions
- `rustyblas/src/level2.rs` — all GEMV functions
- `rustyblas/src/level3.rs` — all GEMM functions
- `rustymkl/src/lapack.rs` — all factorizations (info return code)
- `rustymkl/src/fft.rs` — non-power-of-2 handling

### 2.2 LAPACK Missing Info Return Codes

**Current state:** `rustymkl/src/lapack.rs` — LAPACK functions don't return `info` error codes. Standard LAPACK returns `info < 0` for invalid arguments and `info > 0` for singularity.

**Recommendation:** Return `Result<(), LapackError>` with:
```rust
pub enum LapackError {
    InvalidArgument { position: i32 },
    SingularMatrix { position: i32 },
    NotPositiveDefinite { position: i32 },
    ConvergenceFailure,
}
```

---

## 3. Critical: Unsafe Code Audit

**Total: 146 `unsafe` blocks across the codebase.**

### 3.1 Blackboard Raw Pointer Casting (rustynum-core/src/blackboard.rs)

```rust
// Line 191: Raw pointer → slice without bounds checking
Some(unsafe { std::slice::from_raw_parts(meta.ptr as *const f32, meta.len_elements) })
// Line 202: Mutable version
Some(unsafe { std::slice::from_raw_parts_mut(meta.ptr as *mut f32, meta.len_elements) })
```

**Issues:**
1. No alignment verification before casting `*mut u8` → `*const f32`
2. `len_elements` could be wrong if DType doesn't match
3. No lifetime enforcement — returned slice could outlive the Blackboard
4. Missing `// SAFETY:` comments on every unsafe block

**ndarray approach:** Uses `NonNull<T>` (never-null pointer) and wraps unsafe in well-tested abstractions:
```rust
// ndarray/src/data_repr.rs — safe wrapper around raw allocation
pub struct OwnedRepr<A> {
    ptr: NonNull<A>,
    len: usize,
    capacity: usize,
}
```

**Recommendation:**
1. Add `// SAFETY:` comments documenting invariants on ALL 146 unsafe blocks
2. Replace raw `*mut u8` with `NonNull<u8>` in Blackboard
3. Add alignment checks: `assert!(meta.ptr as usize % std::mem::align_of::<f32>() == 0)`
4. Use lifetimes to tie borrowed slices to Blackboard lifetime

### 3.2 SendMutPtr in GEMM (rustyblas/src/level3.rs:28-45)

```rust
// Line 28-45: Wraps *mut T to send across threads
pub(crate) struct SendMutPtr<T> {
    ptr: *mut T,
    len: usize,
}
unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}

impl<T> SendMutPtr<T> {
    unsafe fn as_mut_slice(&self) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.ptr, self.len)
    }
}
```

**Issues:**
1. No `// SAFETY:` comment explaining why Send/Sync is safe
2. The invariant (disjoint row ownership) is not enforced by the type system
3. Could be made safer with a scoped-thread approach that returns when done

**Recommendation:** Add safety documentation and consider using `rayon`'s parallel iterators for the M-loop instead of manual `SendMutPtr`. ndarray uses rayon for its parallel operations.

### 3.3 SIMD Intrinsics (rustynum-core/src/simd.rs, backends/)

All `core::arch::x86_64` intrinsics are unsafe. These are mostly justified but need:
1. Feature detection guards (`is_x86_feature_detected!("avx512f")`)
2. `// SAFETY:` comments
3. Fallback paths when CPU doesn't support the feature

---

## 4. High: Missing Core Abstractions

### 4.1 No Broadcasting

**Current state:** RustyNum has no broadcasting. Shape mismatch → panic.

**ndarray approach:** Full NumPy-style broadcasting:
```rust
// ndarray/src/dimension/broadcast.rs
pub trait DimMax<Other: Dimension> {
    type Output: Dimension;
}

// ndarray/src/impl_methods.rs
pub fn broadcast<E>(&self, dim: E) -> Option<ArrayView<'_, A, E>> { ... }
```

**Broadcasting rules (ndarray follows NumPy):**
1. Dimensions are compared from trailing edge
2. Dimensions are compatible when equal, or one of them is 1
3. Missing dimensions are treated as 1

**Recommendation:** Implement broadcasting for all binary operations (add, sub, mul, div). This is the single most impactful usability improvement.

### 4.2 No Iterators

**Current state:** No custom iterator types. Users must use `.get_data()` and iterate the raw slice.

**ndarray approach:** Rich iterator ecosystem:
```rust
// ndarray/src/iterators/mod.rs
pub struct Iter<'a, A, D> { ... }           // General iterator
pub struct IterMut<'a, A, D> { ... }        // Mutable iterator
pub struct IndexedIter<'a, A, D> { ... }    // (index, &element)
pub struct Lanes<'a, A, D> { ... }          // Iterate over lanes (rows/columns)
pub struct LanesMut<'a, A, D> { ... }
pub struct AxisIter<'a, A, D> { ... }       // Iterate along an axis
pub struct Windows<'a, A, D> { ... }        // Sliding windows

// Usage:
for row in array.rows() { ... }
for (idx, val) in array.indexed_iter() { ... }
for window in array.windows([3, 3]) { ... }
```

**Recommendation:** Implement at minimum: `Iter`, `IterMut`, `IndexedIter`, `Rows`/`Columns`, `AxisIter`.

### 4.3 No Zip/azip! Combinator

**ndarray's `Zip`** is one of its most powerful features — lock-step iteration over multiple arrays:

```rust
// ndarray/src/zip/mod.rs
// Efficient multi-array traversal with automatic layout optimization
use ndarray::Zip;

Zip::from(&mut output)
    .and(&input1)
    .and(&input2)
    .for_each(|o, &a, &b| { *o = a + b; });

// Or with the azip! macro:
azip!((o in &mut output, &a in &input1, &b in &input2) {
    *o = a + b;
});
```

**Recommendation:** Implement a `Zip` combinator for efficient multi-array operations.

### 4.4 No Slicing DSL

**ndarray's `s![]` macro** provides a powerful slicing DSL:
```rust
// ndarray/src/slice.rs
let view = array.slice(s![1..3, .., ..;-1]);  // Negative step = reverse
let view = array.slice(s![ndarray::NewAxis, ..]);  // Add new axis
```

**Current rustynum:** `.slice()` takes a flat range, no multi-dimensional slicing syntax.

---

## 5. High: API Design & Ergonomics

### 5.1 Trait Bound Explosion

**Current state:** Every method on `NumArray` requires 10+ trait bounds:

```rust
// rustynum-rs/src/num_array/operations.rs:12-29
impl<T, Ops> Add<T> for NumArray<T, Ops>
where
    T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T>
        + Div<Output = T> + Sum<T> + NumOps + Copy + PartialOrd
        + FromU32 + FromUsize + ExpLog + Neg<Output = T> + Default + Debug,
    Ops: SimdOps<T>,
```

**ndarray approach:** Uses a single supertrait:
```rust
// ndarray/src/lib.rs — clean, minimal bounds
impl<A, S, D> ArrayBase<S, D>
where
    S: Data<Elem = A>,
    D: Dimension,
{
    pub fn sum(&self) -> A where A: Clone + Add<Output = A> + num_traits::Zero { ... }
}
```

**Recommendation:** Create a single supertrait `NumElement` that bundles all required traits:
```rust
pub trait NumElement: Clone + Copy + Debug + Default + PartialOrd
    + Add<Output=Self> + Sub<Output=Self> + Mul<Output=Self> + Div<Output=Self>
    + Sum + FromU32 + FromUsize + ExpLog + Neg<Output=Self> + 'static {}

impl NumElement for f32 {}
impl NumElement for f64 {}
```

### 5.2 Missing Standard Trait Implementations

**Current state:** `NumArray` is missing many standard Rust traits:

| Trait | Implemented? | ndarray? | Notes |
|-------|-------------|----------|-------|
| `Display` | No | Yes | Pretty-print arrays |
| `Debug` | No (derived) | Yes | Custom debug formatting |
| `PartialEq` | No | Yes | Array equality |
| `Index` | No | Yes | `array[idx]` syntax |
| `IndexMut` | No | Yes | `array[idx] = val` |
| `IntoIterator` | No | Yes | `for x in array { }` |
| `FromIterator` | No | Yes | `array.collect()` |
| `Serialize/Deserialize` | No | Yes (optional) | serde support |
| `Send` | Implicit | Explicit | Thread safety |
| `Sync` | Implicit | Explicit | Thread safety |
| `AsRef<[T]>` | No | Yes | Borrow as slice |
| `Neg` | No | Yes | Unary negation |

**ndarray examples:**
```rust
// ndarray/src/arraytraits.rs — Display implementation
impl<A: fmt::Display, S, D> fmt::Display for ArrayBase<S, D> { ... }

// ndarray/src/arraytraits.rs — Index implementation
impl<S, D, I> Index<I> for ArrayBase<S, D>
where D: Dimension, I: NdIndex<D>, S: Data {
    type Output = S::Elem;
    fn index(&self, index: I) -> &S::Elem { ... }
}
```

### 5.3 Inconsistent Constructor Patterns

**Current:**
```rust
NumArrayF32::new(vec![1.0, 2.0, 3.0])                    // 1D from vec
NumArrayF32::new_with_shape(vec![1.0, 2.0], vec![1, 2])   // 2D
NumArrayF32::zeros(vec![3, 3])                             // Zeros
NumArrayF32::ones(vec![3, 3])                              // Ones
```

**ndarray (more ergonomic):**
```rust
// ndarray/src/constructors.rs
Array1::from_vec(vec![1.0, 2.0, 3.0])
Array2::zeros((3, 3))                           // Tuple, not vec
Array2::ones((3, 3))
Array::from_shape_vec((2, 3), vec![...])         // Fallible: returns Result
array![[1., 2.], [3., 4.]]                      // Macro for literals
Array::from_elem((3, 3), 0.0)                   // Fill with value
Array::from_shape_fn((3, 3), |(i, j)| ...)      // From function
Array::linspace(0., 1., 100)                    // Linspace
Array::eye(3)                                    // Identity matrix
```

**Recommendation:** Add `from_shape_vec()` returning `Result`, add `array![]` macro, add `eye()`, `from_elem()`, `from_shape_fn()`.

---

## 6. High: Robustness & Edge Cases

### 6.1 No Empty Array Handling

**Current state:** Many functions panic or produce incorrect results on empty arrays.

**ndarray approach:** Explicitly handles empty arrays in every operation:
```rust
// ndarray/src/numeric_util.rs
pub fn unrolled_fold<A, I, F>(mut xs: &[A], init: I, f: F) -> A
where A: Clone, F: Fn(A, A) -> A, I: Fn() -> A {
    // Safe: handles xs.len() == 0
    let mut acc = init();
    // ...
}
```

**Recommendation:** Audit every function for empty-array behavior. Add tests:
```rust
#[test]
fn mean_empty_array() {
    let arr = NumArrayF32::new(vec![]);
    assert!(arr.mean_checked().is_err());  // Should return Err, not panic/NaN
}
```

### 6.2 No NaN/Inf Handling

**Current state:** No special handling for NaN or Inf in statistical functions, comparisons, or GEMM.

**ndarray approach:** Uses `num_traits::Float` for NaN-aware operations and documents NaN behavior.

**Recommendation:**
- Document NaN behavior for every function
- Add `nan_mean()`, `nan_std()` variants that skip NaN values
- Add NaN checks in LAPACK functions (singular matrix detection)

### 6.3 No Overflow Protection

**Current state:** Shape calculations use `usize` multiplication without overflow checks.

```rust
// rustynum-rs/src/num_array/array_struct.rs — no overflow check
let total_size: usize = shape.iter().product();  // Can overflow!
```

**ndarray approach:**
```rust
// ndarray/src/dimension/mod.rs:79-109
/// Returns the `size` of the `dim`, checking that the product of non-zero axis
/// lengths does not exceed `isize::MAX`.
pub fn size_of_shape_checked<D: Dimension>(dim: &D) -> Result<usize, ShapeError> {
    let mut size = 1usize;
    for &d in dim.slice() {
        if d != 0 {
            size = size.checked_mul(d)
                .ok_or_else(|| from_kind(ErrorKind::Overflow))?;
        }
    }
    if size > isize::MAX as usize {
        Err(from_kind(ErrorKind::Overflow))
    } else {
        Ok(size)
    }
}
```

**Recommendation:** Use `checked_mul` for all shape/stride calculations.

### 6.4 CONFIRMED BUGS in rustyblas

The following bugs were found during deep audit:

#### Bug 1: `strmv`/`strsv` Transpose Fallback (rustyblas/src/level2.rs:760-778)

The transpose (Trans) case in `strmv` and `strsv` falls back to a generic loop that **incorrectly accesses the triangular matrix**. The row/column index logic is swapped in the fallback path, producing wrong results for transposed triangular solves.

```rust
// BUG: Current code in strmv for Trans case
// The inner loop index should iterate over the OTHER dimension
// when accessing a transposed triangular matrix, but the bounds
// are computed as if accessing the non-transposed layout.
```

**Impact:** Silent wrong results for `strmv(..., Transpose::Trans, ...)` and `strsv(..., Transpose::Trans, ...)`.

**Fix:** Swap the loop bounds for the Trans case, or implement it as "apply non-transposed to the opposite triangle layout."

#### Bug 2: `vsexp(+Inf)` and `vsln(0)` Return Wrong Values (rustymkl/src/vml.rs)

```rust
// BUG: vsexp clamps input to [-88, 88] before computing polynomial
// vsexp(+Inf) returns exp(88) ≈ 1.65e38 instead of +Inf
// vsexp(-Inf) returns exp(-88) ≈ 6.07e-39 instead of 0.0

// BUG: vsln clamps output to [-88, 88] range
// vsln(0.0) returns approximately -88.0 instead of -Inf
// vsln(-1.0) returns a finite number instead of NaN
```

**Impact:** Any code relying on IEEE 754 special values gets wrong results. This silently corrupts gradient computations in ML workloads where Inf/NaN signals are important.

**Fix:** Add pre-/post-processing checks for special values:
```rust
fn vsexp(input: &[f32], output: &mut [f32]) {
    for (x, y) in input.iter().zip(output.iter_mut()) {
        if x.is_nan() { *y = f32::NAN; continue; }
        if *x == f32::INFINITY { *y = f32::INFINITY; continue; }
        if *x == f32::NEG_INFINITY { *y = 0.0; continue; }
        // ... polynomial approximation for normal range
    }
}
```

#### Bug 3: False Documentation — `strmm`/`dtrmm` (rustyblas/src/lib.rs:16-18)

`strmm` and `dtrmm` (triangular matrix-matrix multiply) are **listed in the public API documentation** but are **NOT actually implemented**. Calling them will fail at compile time (missing symbol), but any documentation that references them is misleading.

**Fix:** Either implement them or remove from documentation. They are important for LAPACK routines (needed by `spotrf`/`dpotrf`).

#### Bug 4: Missing f64 BLAS Level 2/3 Functions

The following f64 variants are documented or expected but missing:
- `dtrmv` — triangular matrix-vector multiply (f64)
- `dtrsv` — triangular solve (f64)
- `dtrsm` — triangular matrix-matrix solve (f64)
- `dpotrs` — Cholesky solve (f64) in rustymkl

Only f32 versions (`strmv`, `strsv`, `strsm`, `spotrs`) exist.

### 6.5 Accuracy Issues in VML f64 Functions

**Problem:** `simd_exp_f64` and `simd_ln_f64` in `rustymkl/src/vml.rs` use degree-7 polynomial approximations that only achieve ~8 digits of accuracy. IEEE 754 `f64` has ~15.9 significant digits.

```
// Current f64 exp accuracy: ~8 digits (1e-8 relative error)
// Required for f64: ~15 digits (1e-15 relative error)
// Fix: Use degree-13 minimax polynomial or Cody-Waite range reduction
```

**Impact:** Accumulates errors in iterative algorithms (optimization, ODE solvers, matrix exponentials).

**Recommendation:** Increase polynomial degree to 13 for `exp` and use proper range reduction. Or use Remez-optimized coefficients. Reference: Cephes library double-precision implementations.

### 6.6 rustynum-core: NaN-Unsafe Sort and Missing Bounds Checks

1. **NaN-unsafe sort** in `rustynum-core/src/kernels.rs`: Uses standard comparison for sorting that does not handle NaN values. NaN comparisons return false for all orderings, causing undefined sort order and potential infinite loops in quicksort variants.

2. **Missing bounds checks in batch scoring** (`rustynum-core/src/delta.rs`): Batch scoring functions don't validate that input array lengths match, leading to potential out-of-bounds access.

3. **Dead computation in SGEMM** (`rustynum-core/src/simd.rs`): There's a dead SGEMM computation path that is compiled but never reached, wasting binary size.

4. **16 duplicated SIMD functions**: `rustynum-core/src/simd.rs` contains 16 pairs of nearly identical f32/f64 functions that could be unified via macros.

### 6.7 rustynum-rs: 156 Panic Sites and API Issues

1. **156 panic/unwrap/expect sites** across rustynum-rs — none return `Result`. This means any shape mismatch, empty array, or invalid input crashes the program.

2. **Integer types bypass SIMD entirely** — `NumArray<i32, _>` and `NumArray<i64, _>` have SIMD ops defined but many code paths fall through to scalar loops without clear indication.

3. **`get_data()` returns `&Vec<T>` instead of `&[T]`** — exposes implementation detail. Should return `&[T]` (slice) to allow future storage changes (e.g., mmap, arena allocation).

```rust
// Current (leaky):
pub fn get_data(&self) -> &Vec<T> { &self.data }

// Should be:
pub fn get_data(&self) -> &[T] { &self.data }
```

### 6.8 No Bounds Checking on BLAS Functions

**Current state:** `rustyblas/src/level3.rs` — GEMM functions don't validate that `lda >= max(1, k)` for row-major or that buffer sizes match dimensions. Passing wrong dimensions = silent memory corruption.

**Recommendation:** Add dimension validation at the start of every BLAS function:
```rust
pub fn sgemm(layout: Layout, transa: Transpose, transb: Transpose,
             m: usize, n: usize, k: usize,
             alpha: f32, a: &[f32], lda: usize,
             b: &[f32], ldb: usize,
             beta: f32, c: &mut [f32], ldc: usize) -> Result<(), BlasError> {
    // Validate dimensions
    let (a_rows, a_cols) = match (layout, transa) { ... };
    if a.len() < a_rows * lda { return Err(BlasError::InvalidDimension); }
    if b.len() < b_rows * ldb { return Err(BlasError::InvalidDimension); }
    if c.len() < m * ldc { return Err(BlasError::InvalidDimension); }
    // ...
}
```

---

## 7. Medium: Code Duplication & Maintainability

### 7.1 Massive f32/f64 Duplication in SIMD

**Current state:** `rustynum-core/src/simd.rs` (2,123 lines) has nearly identical code for `f32x16` and `f64x8` operations. Same for `rustyblas/src/level1.rs` (`sdot`/`ddot`, `saxpy`/`daxpy`, etc.).

**ndarray approach:** Uses `num_traits` and generic programming:
```rust
// ndarray uses matrixmultiply crate which is generic over GemmKernel trait
// ndarray/src/linalg/impl_linalg.rs
pub trait LinalgScalar: 'static + Copy + num_traits::Zero + num_traits::One
    + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> {}
impl LinalgScalar for f32 {}
impl LinalgScalar for f64 {}
```

**Recommendation:** Use macros or const generics to generate f32/f64 variants:
```rust
macro_rules! impl_blas_level1 {
    ($ty:ty, $simd:ty, $prefix:ident) => {
        pub fn ${prefix}dot(n: usize, x: &[$ty], incx: usize, y: &[$ty], incy: usize) -> $ty {
            // Single implementation parameterized by type
        }
    }
}
impl_blas_level1!(f32, f32x16, s);
impl_blas_level1!(f64, f64x8, d);
```

### 7.2 Trait Bound Repetition

**Current state:** The same 10+ trait bounds are copy-pasted on every `impl` block:

```rust
// This exact block appears 20+ times across operations.rs, statistics.rs, etc.
where T: Clone + Mul<Output = T> + Add<Output = T> + Sub<Output = T>
    + Div<Output = T> + Sum<T> + NumOps + Copy + PartialOrd
    + FromU32 + FromUsize + ExpLog + Neg<Output = T> + Default + Debug,
```

**Recommendation:** Use a supertrait (see 5.1) and a type alias:
```rust
pub trait Float: NumElement + num_traits::Float {}
impl Float for f32 {}
impl Float for f64 {}
```

### 7.3 Archive Crate Proliferation

**Current state:** The workspace has 4 "archive" crates (`rustynum-archive`, `rustynum-archive-v3`, `rustynum-carrier`, `rustynum-focus`) that are frozen snapshots of old designs.

**Recommendation:** Move archives to a separate `archives/` directory excluded from the workspace, or use git tags instead. This reduces build times and confusion.

---

## 8. Medium: Performance Gaps

### 8.1 Vector Addition 1.5x Slower Than ndarray

**Benchmark result:** ndarray addition at 10K = 1,634 ns, rustynum = 2,414 ns (1.48x slower).

**Root cause:** `rustynum-rs/src/num_array/operations.rs:33-38` allocates a new `Vec` and calls `Ops::add_scalar`, while ndarray uses in-place operations with optimized memory layout:

```rust
// rustynum: allocates result vec
fn add(self, rhs: T) -> Self::Output {
    let data = self.get_data();
    let mut result_data = vec![T::default(); data.len()];  // ALLOCATION
    Ops::add_scalar(data, rhs, &mut result_data);
    Self::new(result_data)  // ANOTHER ALLOCATION (new shape vec)
}
```

**ndarray approach:**
```rust
// ndarray uses Zip for in-place operations, avoiding extra allocations
// ndarray/src/impl_ops.rs — uses clone-then-mutate pattern
impl<A, S, D> Add<&ArrayBase<S2, D>> for &ArrayBase<S, D> {
    fn add(self, rhs: &ArrayBase<S2, D>) -> Array<A, D> {
        let mut result = self.to_owned();
        result += rhs;  // In-place add
        result
    }
}
```

**Recommendation:**
1. Add `add_assign` / `AddAssign` for in-place operations
2. Avoid allocating a new shape `Vec` on every operation result
3. Consider using `SmallVec` for shape (most arrays are < 6D)

### 8.2 GEMM: Complete Tiered Strategy (Tiny → Small → Medium → Large)

#### THE CRITICAL BUG: NumArray Doesn't Use rustyblas

**Root cause of 236x slowdown at 100×100:** `rustynum-rs/src/num_array/linalg.rs` uses its **OWN transpose-dot implementation**, NOT `rustyblas::sgemm`. The `SimdOps::matrix_multiply` in `rustynum-rs/src/simd_ops/mod.rs:387-415` transposes B, then does row-by-row dot products — completely bypassing the Goto BLAS in rustyblas.

```rust
// CURRENT (rustynum-rs/src/simd_ops/mod.rs:387-415) — NOT using rustyblas!
fn matrix_multiply(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    let mut b_transposed = vec![0.0f32; n * k];  // FULL TRANSPOSE COPY
    Self::transpose(b, &mut b_transposed, k, n);
    // ... row-by-row dot products (no cache blocking, no microkernel)
}
```

**FIX #1 (highest impact, lowest effort):** Route to `rustyblas::level3::sgemm`:
```rust
fn matrix_multiply(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    rustyblas::sgemm(Layout::RowMajor, Transpose::NoTrans, Transpose::NoTrans,
        m, n, k, 1.0, a, k, b, n, 0.0, c, n);
}
```

This single change closes the 236x gap immediately.

#### Tiered GEMM Strategy (Do NOT Use One Algorithm For All Sizes)

```
┌──────────────────────────────────────────────────────────────────────┐
│ Matrix Size    │ Algorithm               │ Threading  │ Target       │
├──────────────────────────────────────────────────────────────────────┤
│ Tiny (≤8×8)   │ Direct register µkernel  │ None       │ ~50 ns       │
│ Small (≤48)   │ B-gather + SIMD dot      │ None       │ ~5 µs        │
│ Medium (≤512) │ Blocked Goto BLAS        │ Adaptive   │ ~500 µs      │
│ Large (>512)  │ Blocked Goto BLAS        │ Full       │ DON'T TOUCH  │
└──────────────────────────────────────────────────────────────────────┘
```

##### Tier 1: Tiny (M,N,K ≤ 8) — Direct Register Microkernel, No Packing

Current: Falls into `sgemm_simple()` which allocates a `b_cols` vec. For 4×4, allocation dominates.

**Recommendation:** Add a direct 4×4/8×8 microkernel that loads from source arrays with no allocation, no packing. Just register-level broadcast-FMA and direct C store. Threshold: `m * n * k < 512`.

##### Tier 2: Small (max dim ≤ 48) — Current `sgemm_simple()` Is Fine

Current threshold `m * n * k < 110,000` is **too high** — includes ~48³ matrices that would benefit from blocking. **Lower to 50,000** (~37³). DO NOT add threading here — thread spawn overhead (~5 µs) exceeds total compute.

##### Tier 3: Medium (48 < max dim ≤ 512) — Blocked + ADAPTIVE Threading

**This is where the biggest improvement opportunity exists.**

Current problems:
- Parallel threshold `m * n > 65,536` (256²) is too conservative
- No adaptive thread count — either 1 thread or ALL cores
- Thread spawn overhead is significant relative to medium-matrix compute

**Recommendation — adaptive thread count based on total FLOPs:**
```rust
fn gemm_thread_count(m: usize, n: usize, k: usize) -> usize {
    let flops = m * n * k * 2;
    if flops < 200_000       { 1 }                          // ~50×50: single thread
    else if flops < 2_000_000  { 2.min(available_threads()) } // ~100×100: 2 threads
    else if flops < 20_000_000 { 4.min(available_threads()) } // ~200×200: 4 threads
    else                       { available_threads() }        // Full parallelism
}
```

**Why:** At 128×128 with 16 threads, each thread gets 8 rows = 131K FLOPs. Thread sync overhead (~2-5 µs) becomes significant vs ~50 µs total compute. 2-4 threads keeps per-thread work meaningful.

**Also:** For medium K < KC (256), skip packing — access A/B with strides directly, as ndarray's `matrixmultiply` does by passing both row and column strides to the microkernel.

##### Tier 4: Large (>512) — DO NOT CHANGE

Already beats ndarray by 1.8x at 1024×1024. See Section 8.7 for what must be preserved.

### 8.3 Microkernel Tile Sizing: Perfect Tiles vs. Tailing vs. AVX-512 Bandwidth

#### Current Tile Architecture

```
SGEMM: MR=6, NR=16 (f32x16 = one zmm register width)
  → 6 accumulators × 16 lanes = 96 FMA ops/K-step
  → 2x K-unrolled → 192 FMA ops per 2 K-steps

DGEMM: MR=6, NR=8 (f64x8 = one zmm register width)
  → 6 accumulators × 8 lanes = 48 FMA ops/K-step
  → NOT K-unrolled (should be — see recommendation below)
```

#### Perfect Tiles (M % MR == 0 AND N % NR == 0) — ALREADY OPTIMAL, DO NOT TOUCH

When both dimensions align:
- **B-load:** Direct `F32Simd::from_slice()` — single 512-bit load, no branching
- **FMA loop:** All 6 accumulators active, full pipeline utilization
- **C-store:** Direct `copy_to_slice()` — single 512-bit store
- **Prefetch:** T0 hint 4 ahead keeps L1 hot

Matrix sizes that are "perfect" (zero tailing overhead):
- **M:** 6, 12, 18, 24, ..., 126, 128(MC), 252, 256, 384, 512, 768, 1024
- **N:** 16, 32, 48, ..., 1024(NC)
- **K:** 256(KC), 512, 768, 1024

Always benchmark with these sizes first to measure peak, then +1 sizes (129×129) to measure tail overhead.

#### N Tailing (nr < NR=16) — NEEDS IMPROVEMENT

**Current approach** (`level3.rs:521-538`): Zero-pad on stack **per K-step**:
```rust
// CURRENT: Stack alloc + copy per K-step (256 times per microkernel!)
let mut tmp = [0.0f32; SGEMM_NR];
tmp[..nr].copy_from_slice(&packed_b[b_base0..b_base0 + nr]);
let b_vec0 = F32Simd::from_slice(&tmp);
```

**Better — AVX-512 mask registers (eliminates per-K-step overhead):**
```rust
let mask: u16 = (1u16 << nr) - 1;  // nr=10 → 0b0000001111111111
unsafe {
    // Single masked load: valid lanes loaded, rest zeroed — ONE instruction
    let b_vec0 = _mm512_maskz_loadu_ps(__mmask16::from(mask), packed_b[b_base0..].as_ptr());
}
```

**Also apply mask to C-store** (eliminates scalar fallback for N tails):
```rust
// CURRENT: to_array() → scalar loop for nr < 16
// PROPOSED: Single masked store
_mm512_mask_storeu_ps(c[base..].as_mut_ptr(), mask, sum);
```

Estimated 10-20% speedup for non-aligned N dimensions.

#### M Tailing (mr < MR=6) — ACCEPTABLE AS-IS

K-loop iterates `ir in 0..mr.min(MR)` — fewer FMA ops, unused accumulators stay zero. Cost is just wasted register capacity. **DO NOT add separate 4×16 or 2×16 microkernels** — instruction cache and code size cost outweighs the small register saving.

#### AVX-512 Bandwidth: Burst Without Tail Choke

The 512-bit FMA pipeline needs continuous feeding for peak throughput:

```
AVX-512 FMA (Intel SPR): 2 FMA/cycle, 4-cycle latency
  → Need 2 × 4 = 8 independent FMA chains to hide latency
  → MR=6: provides 6 chains → ~75% FMA utilization (good enough)
  → MR=8: would give 8 chains → 100% (but tighter register budget)
```

| MR | Accumulators | With 2x Unroll | FMA Chains | Utilization | Verdict |
|----|-------------|----------------|------------|-------------|---------|
| 4  | 4 zmm       | 8 zmm          | 4          | 50%         | Too small |
| 6  | 6 zmm       | 12 zmm         | 6          | 75%         | **KEEP** |
| 8  | 8 zmm       | 16 zmm         | 8          | 100%        | Consider future |
| 12 | 12 zmm      | 24 zmm         | 12         | Over-sub    | Spill risk |

**Current MR=6 is a good balance.** Memory bandwidth (not FMA) is the bottleneck for large GEMM. MR=6 leaves room for prefetch hints. **DO NOT increase beyond 8 without benchmarking 1024×1024.**

**NR=16 = one full zmm for f32 = exactly one 64-byte cache line.** This is already optimal. Smaller wastes SIMD lanes. Larger needs multi-register accumulation per row with no benefit.

**DGEMM improvement:** Add 2x K-unrolling (currently absent). SGEMM has it and benefits from latency hiding — DGEMM should too.

#### Packing Buffer Alignment for Burst

packed_a: MR=6 × 4 bytes = 24 bytes — straddles cache line boundaries. **Improvement:** Pad to MR_PADDED=8 in packing (32 bytes = half cache line), compute only 6 rows. Eliminates split-line loads, 33% more buffer but cleaner prefetch:

```rust
const SGEMM_MR_PADDED: usize = 8;  // Aligned to 32 bytes
// Pack: fill first 6 with data, pad 2 with zeros per K-step
```

### 8.4 Software Prefetch: Why Current Parameters Work (DO NOT CHANGE)

**Current** (`level3.rs:510-518`): T0 prefetch 4 K-steps ahead for both A and B panels.

**Why 4 works:**
- Each K-step: NR=16 f32 = 64 bytes (1 cache line) from B, MR=6 f32 = 24 bytes from A
- L1 latency: ~4-5 cycles. FMA per K-step: ~12 cycles (6 rows × 2 cycles/FMA)
- Distance: 4 × 12 = ~48 cycles ahead → comfortably covers L1 latency

**DO NOT change prefetch distance** unless targeting a different µarch (AMD Zen 4 may prefer 6).

**Missing improvement:** Add T1 (L2) prefetch for next C output tile — one macrokernel iteration ahead.

### 8.5 Zero-Cost Slicing & Transpose: ndarray Takeaways

#### Transpose: ndarray = 0.19 ns (Zero-Cost) vs rustynum = O(n) Copy

**ndarray** (`ndarray/src/impl_methods.rs`): Transpose just reverses strides. No data movement:
```rust
pub fn t(&self) -> ArrayView<'_, A, D> {
    // Reverse dimension order and stride order — THREE integer swaps
    d.slice_mut().reverse();
    s.slice_mut().reverse();
    // Returns view pointing to SAME data. Cost: ~0.2 ns.
}
```

**rustynum** (`rustynum-rs/src/num_array/manipulation.rs`): Transpose COPIES every element:
```rust
pub fn transpose(&self) -> Self {
    let mut result = vec![T::default(); self.data.len()];  // ALLOCATE
    // ... copy every element with index swap ...            // O(n) COPY
    NumArray::new_with_shape(result, new_shape)
}
```

**Performance:** ndarray transpose = 0.19 ns (constant). rustynum 1000×1000 transpose = ~4 ms (copies 1M elements). That's **~20,000,000x slower**.

**Fix requires:** Views + stride-based addressing (Sections 1.2, 1.3). Once strides are respected, transpose becomes:
```rust
pub fn t(&self) -> ArrayView<'_, T> {
    ArrayView { data: &self.data, dim: self.dim.reversed(), strides: self.strides.reversed() }
}
```

#### Slicing: ndarray = O(1) Pointer Arithmetic vs rustynum = O(n) Copy

**ndarray's core slice operation** (`ndarray/src/dimension/mod.rs:449-495`):
```rust
pub fn do_slice(dim: &mut usize, stride: &mut usize, slice: Slice) -> isize {
    let (start, end, step) = to_abs_slice(*dim, slice);
    let offset = stride_offset(start, *stride);        // ptr += start * stride
    *dim = ceil_div(end - start, step.unsigned_abs());  // new length
    *stride = (*stride as isize * step) as usize;       // new stride (step=-1 reverses)
    offset
}
// THREE integer operations. Zero allocation. Zero copy.
```

**Key insight:** Negative strides enable reversed iteration with zero copy. `s![..;-1]` just negates the stride and adjusts the base pointer.

**ndarray's contiguity detection** (for choosing fast vs strided path):
```rust
// ndarray/src/dimension/mod.rs:687-711
fn is_layout_c(dim: &D, strides: &D) -> bool {
    let mut expected = 1;
    for (&d, &s) in dim.iter().rev().zip(strides.iter().rev()) {
        if d > 1 && s != expected { return false; }
        expected *= d;
    }
    true
}
```

When contiguous → flat slice iteration (vectorizable). When strided → nested stride-aware loops. This is how ndarray's Zip picks optimal iteration for both sliced and unsliced arrays.

**Recommendation for rustynum:** After implementing views (Section 1.2), add:
1. `is_contiguous()` check on every operation entry point
2. Fast path: operate on raw `&[T]` slice (current SIMD code works here unchanged)
3. Strided path: iterate with stride-aware indexing
4. This preserves rustynum's SIMD performance for contiguous arrays while adding zero-cost slicing

#### ndarray's s![] Macro (Future Goal)

```rust
let row = matrix.slice(s![2, ..]);       // Zero-copy row view
let block = matrix.slice(s![1..3, 2..5]); // Zero-copy submatrix
let rev = matrix.slice(s![..;-1, ..]);    // Reversed rows, zero-copy
```

The macro tracks input/output dimensions at compile time via `SliceInfo<T, Din, Dout>`.

### 8.6 Median Insight That Could Inspire Standard Deviation

**ndarray's numeric utilities** (`ndarray/src/numeric_util.rs`) use an **8x unrolled fold** for reductions:

```rust
// 8 independent accumulators for instruction-level parallelism
let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
    (init(), init(), init(), init(), init(), init(), init(), init());
while xs.len() >= 8 {
    p0 = f(p0, xs[0].clone());
    p1 = f(p1, xs[1].clone());
    // ... 8x unrolled
    xs = &xs[8..];
}
// Merge: ((p0+p1)+(p2+p3)) + ((p4+p5)+(p6+p7))
```

**Why this matters for std dev:** Standard deviation requires `Σ(x - mean)²`. The naive two-pass approach (compute mean, then compute variance) requires two full passes over the data. But with 8x unrolled accumulators, you can:

1. **One-pass Welford's algorithm** with SIMD: maintain running mean + M2 (sum of squared deviations) using the numerically stable recurrence, but across 8 parallel lanes
2. **Two-pass with fused loops**: Use ndarray's unrolled pattern — first pass computes mean with 8 accumulators, second pass computes `Σ(x-mean)²` with 8 accumulators. Each pass is maximally ILP-friendly.

**The key takeaway:** ndarray achieves near-SIMD performance for reductions WITHOUT explicit SIMD — just by exploiting ILP through manual unrolling. This is robust, portable, and should be adopted for any reduction where rustynum doesn't already have a SIMD kernel.

For rustynum, the median sort (`kernels.rs`) should also be improved:
- Use `sort_unstable_by` with `total_cmp()` (not `partial_cmp`) to handle NaN safely
- For large arrays, consider `nth_element` / introselect (O(n) average) instead of full sort (O(n log n))
- ndarray-stats uses `kth_by` from the `ndarray-stats` crate for this

### 8.7 DO NOT CHANGE: Rustynum's Winning Approaches

These components already outperform ndarray. Changing them risks regression.

#### 8.7.1 The 6×16 SGEMM Microkernel with 2x K-Unrolling — KEEP

**File:** `rustyblas/src/level3.rs:477-593`

What makes it fast: 2x K-unrolling hides FMA latency, software prefetch (T0, 4 ahead) keeps L1 hot, `splat()` → `vbroadcastss` (single cycle), direct `from_slice()` → aligned `vmovups` (single cycle).

**DO NOT:** Replace with generic microkernel, reduce unrolling, or change MR/NR without benchmarking 1024×1024 first.

#### 8.7.2 B Panel Sharing Across Threads — KEEP

**File:** `rustyblas/src/level3.rs:294-346`

packed_b packed once, shared via `&packed_b` (immutable reference) to all threads. Each thread packs its own packed_a. No false sharing, no lock contention, no redundant work.

#### 8.7.3 SendMutPtr Thread Partitioning — KEEP (But Add Safety Docs)

**File:** `rustyblas/src/level3.rs:28-45`

Each thread gets exclusive C rows via `split_at_mut()`. Faster than rayon's work-stealing for GEMM (perfectly balanced, zero sync after spawn).

**DO NOT** replace with rayon for GEMM inner loop. **DO** add `// SAFETY:` comments.

#### 8.7.4 Cache Blocking MC=128, KC=256, NC=1024 — KEEP FOR LARGE

Tuned for L1=32KB, L2=256KB, L3=2MB/core:
- A panel: 128 × 256 × 4 = 128 KB (fits L2)
- B panel: 256 × 1024 × 4 = 1 MB (fits L3)
- C tile: 128 × 16 × 4 = 8 KB (fits L1)

**DO NOT change for large matrices.** DO consider separate smaller blocking for medium (see 8.2 Tier 3).

### 8.8 Rustynum-Unique Strengths ndarray CANNOT Match

These are competitive advantages — preserve and extend.

#### 8.8.1 Cascading Dispatch and Early Exit — PRESERVE AND EXTEND

Rustynum's tiered compute dispatch (blackboard → SIMD → BLAS) with early exit on threshold is fundamentally superior to ndarray's flat "always compute everything" model.

**Where:** `rustynum-core/src/kernels.rs` (cascading scoring), `rustynum-core/src/delta.rs` (early exit batch scoring)

**Don't flatten into ndarray-style always-complete.** Instead make robust: add bounds checking per cascade level, return `Result`, document thresholds, test cascade boundary transitions.

#### 8.8.2 BF16 Top-5% GEMM — PRESERVE AND OPTIMIZE

**File:** `rustyblas/src/bf16_gemm.rs`

ndarray has zero BF16 support. Rustynum's BF16 GEMM with top-K extraction is unique for ML inference. Keep intact, but apply mask-register tailing improvements from Section 8.3.

#### 8.8.3 INT8 Quantized GEMM — PRESERVE

**File:** `rustyblas/src/int8_gemm.rs`

ndarray has no integer GEMM. Critical for inference. Keep as-is, consider adding asymmetric quantization (per-channel zero points) and INT4 packing.

#### 8.8.4 Pure Rust LAPACK/FFT/VML — PRESERVE (Fix Bugs)

ndarray depends on external C LAPACK. Rustynum's pure Rust = `cargo install` just works, no C toolchain. Don't replace with C bindings. Fix bugs (Section 6.4/6.5) and bring accuracy to production grade.

#### 8.8.5 Blackboard Zero-Copy Memory Arena — PRESERVE

**File:** `rustynum-core/src/blackboard.rs`

ndarray has no shared memory arena. Fix safety issues (Section 3.1) but keep the architecture.

### 8.9 The Best Possible Mix: Combining SIMD Innovation with ndarray Maturity

```
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 4: API Ergonomics (STEAL from ndarray)                         │
│   Display, Index, s![], broadcasting, From/Into, error types,        │
│   property tests, #![warn(missing_docs)]                             │
├──────────────────────────────────────────────────────────────────────┤
│ LAYER 3: Memory Safety & Views (STEAL from ndarray)                  │
│   Storage traits, Dimension types, zero-cost slicing, layout         │
│   detection, checked_mul, contiguity-based fast path selection       │
├──────────────────────────────────────────────────────────────────────┤
│ LAYER 2: Dispatch & Orchestration (KEEP rustynum)                    │
│   Cascading dispatch, early exit, tiered threading, adaptive thread  │
│   count, Blackboard arena, BF16/INT8 top-K paths                    │
├──────────────────────────────────────────────────────────────────────┤
│ LAYER 1: Compute Kernels (KEEP rustynum, targeted improvements)      │
│   AVX-512 6×16 µkernel, pure Rust BLAS/LAPACK/FFT/VML, prefetch    │
│   IMPROVE: mask registers for tailing, DGEMM 2x unroll, tiny-matrix │
│   direct kernels, medium-matrix adaptive threading                   │
└──────────────────────────────────────────────────────────────────────┘
```

**Principle:** ndarray is the role model for Layers 3-4 (safe, ergonomic, correct). Rustynum is already superior at Layers 1-2 (fast, specialized, innovative). Bring Layer 3-4 maturity WITHOUT touching Layer 1-2 performance.

### 8.10 No SIMD Fallback Path

**Current state:** If AVX-512 is not available, many SIMD functions produce incorrect results or crash. The `avx512` feature is default-on with no runtime check.

**ndarray approach:** `matrixmultiply` has AVX2, SSE2, and scalar fallbacks with runtime detection.

**Recommendation:** Add `#[cfg]` fallback for every SIMD path:
```rust
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_feature = "avx512f")]
    { dot_f32_avx512(a, b) }
    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    { dot_f32_avx2(a, b) }
    #[cfg(not(target_feature = "avx2"))]
    { dot_f32_scalar(a, b) }
}
```

---

## 9. Medium: Documentation

### 9.1 Massive Documentation Gaps

**Stats:**
- **~280 public functions** across core crates with **no doc comments**
- Worst offenders:
  - `rustyblas/src/level1.rs` — **16/16** functions undocumented
  - `rustynum-core/src/simd.rs` — **31/34** functions undocumented
  - `rustynum-core/src/delta.rs` — **11/11** functions undocumented
  - `rustynum-core/src/mkl_ffi.rs` — **60/62** functions undocumented
  - `rustynum-core/src/fingerprint.rs` — **7/9** functions undocumented
  - `rustynum-core/src/kernels.rs` — **8/18** functions undocumented
  - `rustynum-core/src/layer_stack.rs` — **5/14** functions undocumented

**ndarray approach:** `#![warn(missing_docs)]` in lib.rs — compiler warns on any undocumented public item. Every public function has a doc comment with examples:
```rust
// ndarray/src/impl_methods.rs
/// Return the shape of the array in its "pattern" form,
/// an integer in the one-dimensional case, tuple in the n-dimensional cases
/// and so on.
pub fn dim(&self) -> D::Pattern { ... }
```

**Recommendation:**
1. Add `#![warn(missing_docs)]` to all lib.rs files
2. Document every public function with `///` + `# Examples` + `# Panics` sections
3. Add `#![doc(test(attr(deny(warnings))))]` to run doc-tests

### 9.2 No Module-Level Documentation

**Current state:** Most modules lack `//!` documentation.

**ndarray approach:** Every module has a `//!` header explaining its purpose:
```rust
// ndarray/src/data_traits.rs:9
//! The data (inner representation) traits for ndarray
```

### 9.3 No docs.rs Configuration

**ndarray has:**
```toml
# ndarray/Cargo.toml:124-127
[package.metadata.docs.rs]
features = ["approx", "serde", "rayon"]
rustdoc-args = ["--cfg", "docsrs"]
```

---

## 10. Medium: Testing & CI

### 10.1 Testing Coverage Gaps

**Current stats:**
- **1,285 `#[test]` annotations** total (good count)
- **Only 2 dedicated test files** (both in rustynum-rs)
- **No property-based testing** (QuickCheck, proptest)
- **No fuzz testing**
- **No numeric accuracy tests** (epsilon comparisons vs reference implementations)

**ndarray approach:**
```rust
// ndarray uses quickcheck for property testing
// ndarray/tests/array.rs
quickcheck! {
    fn test_slice_identity(arr: Array2<f64>) -> bool {
        arr.slice(s![.., ..]) == arr.view()
    }
}

// Numeric tests with approx
// ndarray/crates/numeric-tests/tests/accuracy.rs
assert_abs_diff_eq!(result, expected, epsilon = 1e-6);
```

**Recommendation:**
1. Add `proptest` or `quickcheck` for property testing
2. Add numeric accuracy tests comparing against known-good implementations
3. Add edge-case tests for: empty arrays, single-element, very large, NaN, Inf, denormals
4. Add LAPACK accuracy tests comparing against reference (LAPACK test suite)
5. Add FFT accuracy tests comparing against `rustfft` or DFT definition

### 10.2 No CI Configuration Visible

**Current state:** No `.github/workflows/` visible in the repo.

**ndarray approach:**
```yaml
# ndarray/.github/workflows/ci.yaml
# Runs on: ubuntu, macos, windows
# Tests: stable, beta, nightly, MSRV
# Checks: clippy, rustfmt, doc-tests, miri (for UB detection)
```

**Recommendation:** Add CI with:
- `cargo test` (all packages)
- `cargo clippy -- -D warnings`
- `cargo fmt -- --check`
- `cargo miri test` (detect UB in unsafe code)
- `cargo doc --no-deps` (verify documentation builds)
- Cross-platform: Linux, macOS, Windows
- Feature matrix: default, avx2, avx512

### 10.3 No Miri Testing

**ndarray uses Miri** to detect undefined behavior in unsafe code.

**Recommendation:** Add `cargo +nightly miri test` to CI for rustynum-core, rustyblas, and rustymkl to catch UB in unsafe blocks.

---

## 11. Low: Ecosystem & Publishing

### 11.1 Not Published to crates.io

**Recommendation:** Prepare for publishing:
1. Add `license`, `repository`, `description`, `keywords`, `categories` to all Cargo.toml
2. Add `include` to avoid publishing build artifacts
3. Set up `cargo-release` for version management
4. Add CHANGELOG.md

### 11.2 Nightly-Only Requirement

**Current state:** Requires nightly Rust for `portable_simd`.

**Recommendation:** Add a `stable` feature that uses explicit SIMD intrinsics (`core::arch::x86_64`) or scalar fallbacks instead of `portable_simd`, allowing the crate to compile on stable Rust.

### 11.3 No `no_std` Support

**ndarray supports `no_std`** with `default-features = false`.

**Recommendation:** Make rustynum-core `no_std` compatible for embedded use.

---

## 12. ndarray Best Practices to Adopt

### 12.1 Pattern: Data Representation Traits (Steal This)

**Where:** `ndarray/src/data_traits.rs:28-180`

This is ndarray's most important design pattern. It separates "what the array can do" from "how data is stored":

```rust
// The trait hierarchy:
//   RawData          -- pointer may not be safe to deref
//   ├── Data         -- safe to read elements
//   │   ├── DataOwned -- can be created from Vec
//   │   └── DataMut  -- can mutate elements
//   └── RawDataMut   -- can get mutable pointer (but not safe to deref)

// This enables ONE impl block to work for owned, viewed, and shared arrays:
impl<A, S: Data<Elem=A>, D: Dimension> ArrayBase<S, D> {
    pub fn sum(&self) -> A { ... }  // Works for Array, ArrayView, ArcArray
}
```

**Why adopt:** Eliminates the need to duplicate methods for owned vs borrowed arrays. Currently rustynum has no views at all.

### 12.2 Pattern: Dimension Trait (Steal This)

**Where:** `ndarray/src/dimension/dimension_trait.rs`

```rust
pub trait Dimension: Clone + Eq + Debug + Send + Sync + Default + IndexMut<usize> + Mul<usize> {
    const NDIM: Option<usize>;
    type Pattern;
    type Smaller: Dimension;
    type Larger: Dimension;

    fn ndim(&self) -> usize;
    fn into_pattern(self) -> Self::Pattern;
    fn zeros(ndim: usize) -> Self;
    fn size(&self) -> usize;
    fn default_strides(&self) -> Self;
    fn fortran_strides(&self) -> Self;
    fn _fastest_varying_stride_order(&self) -> Self;
    // ... 20+ methods
}
```

**Why adopt:** Compile-time dimension safety. `Ix2` can't be confused with `Ix3`. The `Pattern` associated type provides ergonomic destructuring: `let (rows, cols) = array.dim()`.

### 12.3 Pattern: Layout Detection for Performance (Steal This)

**Where:** `ndarray/src/layout/mod.rs`

```rust
// ndarray detects memory layout to choose optimal iteration order
pub(crate) fn is_standard_layout<D: Dimension>(dim: &D, strides: &D) -> bool {
    // Check if array is C-contiguous (row-major)
}

pub(crate) fn is_layout_f<D: Dimension>(dim: &D, strides: &D) -> bool {
    // Check if array is Fortran-contiguous (column-major)
}

// Used in operations to pick fast path:
if self.is_standard_layout() {
    // Fast: iterate as flat slice
    self.as_slice().unwrap().iter().sum()
} else {
    // Slow: respect strides
    self.iter().sum()
}
```

**Why adopt:** Many operations can be 10x faster on contiguous arrays. Detecting layout allows choosing the optimal code path.

### 12.4 Pattern: Zero-Cost Slicing (Steal This)

**Where:** `ndarray/src/slice.rs`, `ndarray/src/impl_methods.rs`

```rust
// ndarray/src/impl_methods.rs — slice returns a view (no copy)
pub fn slice<I>(&self, info: I) -> ArrayView<'_, A, I::OutDim>
where I: SliceArg<D> {
    // Adjusts pointer offset and strides — NO DATA COPY
    let (out_dim, out_strides) = info.out_ndim();
    // Returns a view pointing into self's data
}

// The s![] macro makes this ergonomic:
let row = matrix.slice(s![2, ..]);       // Zero-copy row view
let block = matrix.slice(s![1..3, 2..5]); // Zero-copy submatrix
```

**Why adopt:** Slicing is one of the most common operations. Zero-copy makes it O(1) instead of O(n).

### 12.5 Pattern: Checked Shape Construction (Steal This)

**Where:** `ndarray/src/dimension/mod.rs:79-109`

```rust
// ndarray validates ALL shape operations
pub fn size_of_shape_checked<D: Dimension>(dim: &D) -> Result<usize, ShapeError> {
    let mut size = 1usize;
    for &d in dim.slice() {
        if d != 0 {
            size = size.checked_mul(d)
                .ok_or_else(|| from_kind(ErrorKind::Overflow))?;
        }
    }
    if size > isize::MAX as usize {
        Err(from_kind(ErrorKind::Overflow))
    } else {
        Ok(size)
    }
}

// Also validates that strides don't alias:
pub(crate) fn dim_stride_overlap<D: Dimension>(dim: &D, strides: &D) -> bool { ... }
```

**Why adopt:** Prevents integer overflow → memory corruption bugs.

### 12.6 Pattern: Efficient Fold/Reduce (Steal This)

**Where:** `ndarray/src/numeric_util.rs`

```rust
// ndarray's unrolled fold — 8x unrolled for ILP
pub fn unrolled_fold<A, I, F>(mut xs: &[A], init: I, f: F) -> A
where A: Clone, F: Fn(A, A) -> A, I: Fn() -> A {
    let mut acc = init();
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (init(), init(), init(), init(), init(), init(), init(), init());
    while xs.len() >= 8 {
        p0 = f(p0, xs[0].clone());
        p1 = f(p1, xs[1].clone());
        // ... 8x unrolled
        xs = &xs[8..];
    }
    // Reduce remaining + merge accumulators
}
```

**Why adopt:** 8x loop unrolling exploits instruction-level parallelism. This is why ndarray's sum is fast even without explicit SIMD.

### 12.7 Pattern: Private Trait Sealing (Steal This)

**Where:** `ndarray/src/private.rs`

```rust
// Prevent downstream crates from implementing internal traits
mod private {
    pub trait Sealed {}
}
macro_rules! private_decl {
    () => { fn __private__(&self) -> private::Sealed; }
}
```

**Why adopt:** Allows adding methods to public traits without breaking semver. Essential for library stability.

### 12.8 Pattern: Comprehensive Test Infrastructure (Steal This)

**Where:** `ndarray/tests/`, `ndarray/crates/numeric-tests/`

```rust
// ndarray/crates/numeric-tests/tests/accuracy.rs
// Tests numeric accuracy against known results
#[test]
fn accurate_eye_mul() {
    let eye = Array2::eye(128);
    let a = Array2::from_shape_fn((128, 128), |(i, j)| (i * 131 + j * 37) as f64);
    let result = eye.dot(&a);
    assert_abs_diff_eq!(result, a, epsilon = 1e-12);
}

// Property tests with quickcheck
quickcheck! {
    fn dot_commutative(a: Vec<f64>, b: Vec<f64>) -> bool {
        let n = a.len().min(b.len());
        let a = ArrayView1::from(&a[..n]);
        let b = ArrayView1::from(&b[..n]);
        (a.dot(&b) - b.dot(&a)).abs() < 1e-10
    }
}
```

### 12.9 Pattern: Workspace Organization (Steal This)

**ndarray workspace:**
```
ndarray/
├── src/                      # Main crate
├── ndarray-rand/             # Random array generation
├── crates/
│   ├── ndarray-gen/          # Test data generation
│   ├── numeric-tests/        # Accuracy tests
│   ├── serialization-tests/  # Serde tests
│   ├── blas-tests/           # BLAS integration tests
│   └── blas-mock-tests/      # Mock BLAS for testing
├── benches/                  # Benchmarks
├── examples/                 # Usage examples
├── tests/                    # Integration tests
└── scripts/                  # CI scripts
```

**Recommendation:** Reorganize rustynum workspace:
- Move archives to `archives/` (exclude from workspace)
- Add `crates/tests/` for integration tests
- Add `crates/benchmarks/` for cross-crate benchmarks
- Add `examples/` with runnable examples

---

## Priority-Ordered Implementation Roadmap

### Phase 1: Safety & Correctness (Weeks 1-4)
1. **Fix confirmed bugs:**
   - `strmv`/`strsv` transpose fallback (wrong results for Trans case)
   - `vsexp`/`vsln` special-value handling (IEEE 754 compliance)
   - Remove false `strmm`/`dtrmm` documentation or implement them
   - Fix NaN-unsafe sort in `rustynum-core/src/kernels.rs`
2. **Add `ArrayError` type** and convert all 156 panic/unwrap/expect sites to `Result` returns
3. **Add `// SAFETY:` comments** to all 146 unsafe blocks
4. **Add bounds checking** to BLAS functions and batch scoring in delta.rs
5. **Add overflow checks** to shape calculations (`checked_mul`)
6. **Add empty array handling** to all functions
7. **Fix f64 VML accuracy** — increase polynomial degree for `simd_exp_f64`/`simd_ln_f64`
8. **Implement missing f64 variants:** `dtrmv`, `dtrsv`, `dtrsm`, `dpotrs`
9. **Change `get_data()` to return `&[T]`** instead of `&Vec<T>`
10. **Run `cargo miri test`** and fix all UB

### Phase 2: Core Abstractions (Weeks 5-10)
1. **Implement views** (`ArrayView`, `ArrayViewMut`) — zero-copy slicing
2. **Make operations stride-aware** — zero-cost transpose
3. **Implement broadcasting** for binary operations
4. **Add basic iterators** (`Iter`, `IterMut`, `IndexedIter`)
5. **Implement standard traits** (Display, PartialEq, Index, IntoIterator)

### Phase 3: API Maturity (Weeks 11-16)
1. **Consolidate trait bounds** into supertraits
2. **Add dimension type parameter** (or at least typed aliases: `Vector<T>`, `Matrix<T>`)
3. **Add ergonomic constructors** (from_shape_vec, array!, eye, from_shape_fn)
4. **Add slicing DSL** (s![] macro or similar)
5. **Reduce f32/f64 code duplication** with macros

### Phase 4: Polish & Publish (Weeks 17-20)
1. **Add `#![warn(missing_docs)]`** and document all public items
2. **Add CI pipeline** (clippy, fmt, test, miri, cross-platform)
3. **Add property-based tests** and numeric accuracy tests
4. **Add serde support** (optional feature)
5. **Prepare crates.io metadata** and publish
6. **Add stable Rust support** via feature flag

---

## Appendix: File-Level Issue Counts

| File | Undocumented Fns | Unsafe Blocks | unwrap() | panic!() |
|------|-----------------|---------------|----------|----------|
| rustynum-core/src/simd.rs | 31/34 | ~20 | ~15 | 0 |
| rustynum-core/src/blackboard.rs | 1/19 | 12 | 2 | 0 |
| rustynum-core/src/mkl_ffi.rs | 60/62 | 0 | 0 | 0 |
| rustynum-core/src/kernels.rs | 8/18 | ~5 | ~8 | 0 |
| rustynum-core/src/delta.rs | 11/11 | 0 | 0 | 0 |
| rustyblas/src/level1.rs | 16/16 | ~6 | ~4 | 0 |
| rustyblas/src/level3.rs | 0/16 | 13 | 0 | 2 |
| rustyblas/src/int8_gemm.rs | 0/10 | ~8 | 0 | 0 |
| rustymkl/src/lapack.rs | 0/12 | ~10 | ~5 | 3 |
| rustymkl/src/fft.rs | 0/6 | 4 | ~3 | 1 |
| rustymkl/src/vml.rs | 0/14 | ~8 | 0 | 0 |
| rustynum-rs/src/num_array/array_struct.rs | 2/34 | 0 | ~20 | 3 |
| rustynum-rs/src/num_array/operations.rs | 0/20 | 0 | ~5 | 2 |
| rustynum-rs/src/num_array/statistics.rs | 1/11 | 0 | ~8 | 2 |
| rustynum-rs/src/num_array/manipulation.rs | 0/12 | 0 | ~10 | 4 |
