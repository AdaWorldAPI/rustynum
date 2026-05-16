# rustynum Parity Plan: Catching Up with ndarray

> **Target**: Make `rustynum-rs::NumArray` a viable drop-in alternative to `ndarray::Array`
> for dense tensor workloads, with SIMD-first architecture as the differentiator.

---

## Current State Gap Analysis

### What ndarray Has That rustynum Lacks

| Category | ndarray | rustynum-rs | Gap |
|----------|---------|-------------|-----|
| **Type system** | `ArrayBase<S, D>` generic over storage + dimension | `NumArray<T, Ops>` generic over element + SIMD ops | No storage polymorphism, no dimension type safety |
| **Dimensionality** | Compile-time `Ix1..Ix6`, runtime `IxDyn` | Runtime `Vec<usize>` only | No compile-time shape safety |
| **Views** | `ArrayView`/`ArrayViewMut` with full op support | `ArrayView`/`ArrayViewMut` (basic: get, transpose, slice) | Views can't do arithmetic, no broadcasting |
| **Slicing** | `s![]` macro, `Slice`, `SliceArg`, negative indices, step | `view().slice(ranges)` basic ranges only | No macro, no step, no newaxis |
| **Broadcasting** | Full NumPy-style broadcasting (any compatible shapes) | `try_{add,sub,mul,div}_broadcast()` — only `[m,n] op [m,1]` | Very limited broadcasting |
| **Axis operations** | `sum_axis`, `mean_axis`, `map_axis`, `fold_axis` | `mean_axis`, `sum_axis` (basic) | Missing map/fold along axis |
| **Iteration** | `Zip` (up to 9 producers), `Lanes`, `AxisIter`, `Windows`, `Chunks` | No Zip, no Lanes, no AxisIter | Major gap |
| **Parallel** | `par_azip!`, `par_map_inplace`, rayon integration | None | No parallel iteration |
| **Stacking/splitting** | `stack`, `concatenate`, `split_at` | `try_concatenate` (axis=0 only) | Limited |
| **Assignment ops** | `+=`, `-=`, `*=`, `/=`, `assign`, `zip_mut_with` | None | No in-place ops |
| **Linear algebra** | `dot` (general), matmul via BLAS, `norm`, `solve` (via ndarray-linalg) | `try_matrix_multiply`, `dot` (1D only) | Weak linalg |
| **Data ownership** | `OwnedRepr`, `ViewRepr`, `RawViewRepr`, `CowRepr`, `ArcRepr` | `Vec<T>` only (always owned) | No zero-copy sharing |
| **Constructors** | `zeros`, `ones`, `eye`, `linspace`, `logspace`, `geomspace`, `from_fn`, `from_shape_fn` | `zeros`, `ones`, `linspace`, `arange`, `full` | Missing eye, logspace, from_fn |
| **Serialization** | serde support | None | Missing |
| **Approx comparison** | `assert_abs_diff_eq!` via `approx` crate | None | Missing |
| **Standard traits** | `Clone`, `Debug`, `Display`, `PartialEq`, all iter traits | `Clone`, `Debug`, basic | Incomplete trait coverage |

---

## Phased Implementation Plan

### Phase 1: Foundation — Storage & Dimension Type System (4 weeks)

The core architectural gap. Without this, everything else is workarounds.

#### 1.1 Compile-Time Dimension Types

```rust
pub trait Dimension: Clone + Debug + PartialEq {
    const NDIM: Option<usize>; // None = dynamic
    fn ndim(&self) -> usize;
    fn as_slice(&self) -> &[usize];
    fn as_slice_mut(&mut self) -> &mut [usize];
    fn zeros(ndim: usize) -> Self;
}

pub struct Ix1([usize; 1]);
pub struct Ix2([usize; 2]);
pub struct Ix3([usize; 3]);
pub struct IxDyn(Vec<usize>);
```

**Why first**: Every subsequent API (slicing, broadcasting, iteration) benefits from compile-time dimension tracking.

#### 1.2 Storage Trait

```rust
pub trait Storage<T> {
    fn as_slice(&self) -> &[T];
    fn len(&self) -> usize;
}

pub trait StorageMut<T>: Storage<T> {
    fn as_slice_mut(&mut self) -> &mut [T];
}

pub struct OwnedStorage<T>(Vec<T>);
pub struct ViewStorage<'a, T>(&'a [T]);
pub struct ViewMutStorage<'a, T>(&'a mut [T]);
```

**Why**: Enables views to participate in arithmetic without copying.

#### 1.3 Unified Array Type

```rust
pub struct NdArray<T, S: Storage<T>, D: Dimension> {
    storage: S,
    shape: D,
    strides: D,  // signed strides for reversed axes
    offset: usize,
    _phantom: PhantomData<T>,
}

pub type Array<T, D> = NdArray<T, OwnedStorage<T>, D>;
pub type ArrayView<'a, T, D> = NdArray<T, ViewStorage<'a, T>, D>;
pub type ArrayViewMut<'a, T, D> = NdArray<T, ViewMutStorage<'a, T>, D>;
```

**Migration**: Keep existing `NumArray<T, Ops>` as a type alias for `Array<T, IxDyn>` with SIMD ops as extension traits.

---

### Phase 2: Slicing & Broadcasting (3 weeks)

#### 2.1 Full Broadcasting

Implement NumPy-style broadcasting rules:
- Shapes aligned from trailing dimension
- Dimensions of 1 broadcast to any size
- Missing dimensions treated as 1

```rust
impl<T, S, D> NdArray<T, S, D> {
    pub fn broadcast<D2: Dimension>(&self, shape: D2) -> Option<ArrayView<T, D2>>;
}
```

SIMD-accelerated: broadcast loops should use `f32x16` strided load patterns.

#### 2.2 Advanced Slicing

```rust
// s![] macro equivalent
macro_rules! slice {
    ($($args:tt)*) => { ... };
}

pub enum SliceOrIndex {
    Slice { start: isize, end: Option<isize>, step: isize },
    Index(isize),
    NewAxis,
}
```

Support: negative indices, step, newaxis, `..*`, `..`.

#### 2.3 In-Place Operations

```rust
impl<T, D> Array<T, D> {
    pub fn add_assign(&mut self, rhs: &impl AsArray<T, D>);
    pub fn sub_assign(&mut self, rhs: &impl AsArray<T, D>);
    pub fn mul_assign(&mut self, rhs: &impl AsArray<T, D>);
    pub fn div_assign(&mut self, rhs: &impl AsArray<T, D>);
    pub fn map_inplace<F: FnMut(&mut T)>(&mut self, f: F);
}
```

---

### Phase 3: Iteration & Parallelism (3 weeks)

#### 3.1 Zip

```rust
pub struct Zip<Parts, D: Dimension> { ... }

impl Zip {
    pub fn from(a: impl IntoNdProducer) -> Self;
    pub fn and(self, b: impl IntoNdProducer) -> Self;
    pub fn for_each<F>(self, f: F);        // sequential
    pub fn par_for_each<F>(self, f: F);    // rayon parallel
    pub fn map_collect<R, F>(self, f: F) -> Array<R, D>;
}
```

#### 3.2 Axis Iterators

```rust
impl<T, S, D> NdArray<T, S, D> {
    pub fn lanes(&self, axis: Axis) -> Lanes<T, D::Smaller>;
    pub fn axis_iter(&self, axis: Axis) -> AxisIter<T, D::Smaller>;
    pub fn axis_chunks_iter(&self, axis: Axis, chunk: usize) -> AxisChunksIter<T, D>;
    pub fn rows(&self) -> Lanes<T, D::Smaller>;
    pub fn columns(&self) -> Lanes<T, D::Smaller>;
    pub fn windows(&self, window_size: D) -> Windows<T, D>;
}
```

#### 3.3 Rayon Integration

```rust
// Feature-gated: #[cfg(feature = "rayon")]
impl<T: Send + Sync, D: Dimension> NdArray<T, OwnedStorage<T>, D> {
    pub fn par_map_inplace<F: Fn(&mut T) + Sync>(&mut self, f: F);
    pub fn par_mapv<F: Fn(T) -> T + Sync>(&self, f: F) -> Self;
}

// par_azip! macro
macro_rules! par_azip {
    ($($tt:tt)*) => { ... };
}
```

SIMD note: parallel chunks should be ≥64 elements to amortize SIMD setup.

---

### Phase 4: Linear Algebra (2 weeks)

Already have rustyblas — wire it properly.

#### 4.1 General Dot Product

```rust
impl<T, D1, D2> Dot<NdArray<T, S2, D2>> for NdArray<T, S1, D1> {
    type Output = ...;
    fn dot(&self, rhs: &NdArray<T, S2, D2>) -> Self::Output;
}
```

Rules (matching ndarray):
- 1D · 1D → scalar (inner product)
- 2D · 1D → 1D (matrix-vector via `rustyblas::level2::sgemv`)
- 2D · 2D → 2D (matmul via `rustyblas::level3::sgemm`)
- ND · 1D → (N-1)D (sum over last axis of lhs with rhs)

#### 4.2 Norms

```rust
impl<T: Float, D> NdArray<T, S, D> {
    pub fn norm_l1(&self) -> T;      // via simd::asum_f32
    pub fn norm_l2(&self) -> T;      // via simd::nrm2_f32
    pub fn norm_max(&self) -> T;     // via simd::amax (add to simd.rs)
}
```

#### 4.3 Batched GEMM

```rust
pub fn batched_matmul<T>(a: &Array<T, Ix3>, b: &Array<T, Ix3>) -> Array<T, Ix3>;
```

Wire to `rustyblas::level3::sgemm` with batch loop + rayon parallel.

---

### Phase 5: Ecosystem Parity (2 weeks)

#### 5.1 Serde Support

```rust
#[cfg(feature = "serde")]
impl<T: Serialize, D: Dimension> Serialize for NdArray<T, OwnedStorage<T>, D> { ... }
```

#### 5.2 Standard Constructors

```rust
pub fn eye<T>(n: usize) -> Array<T, Ix2>;
pub fn logspace<T>(start: T, end: T, n: usize, base: T) -> Array<T, Ix1>;
pub fn geomspace<T>(start: T, end: T, n: usize) -> Array<T, Ix1>;
pub fn from_shape_fn<D, F>(shape: D, f: F) -> Array<T, D>;
pub fn diag<T>(v: &Array<T, Ix1>) -> Array<T, Ix2>;
```

#### 5.3 Display & Approx

```rust
impl<T: Display, D> Display for NdArray<T, S, D> { ... }  // NumPy-style formatting

#[cfg(feature = "approx")]
impl<T: ApproxEq, D> AbsDiffEq for Array<T, D> { ... }
```

#### 5.4 Stacking & Splitting

```rust
pub fn stack<T, D>(axis: Axis, arrays: &[ArrayView<T, D>]) -> Array<T, D::Larger>;
pub fn concatenate<T, D>(axis: Axis, arrays: &[ArrayView<T, D>]) -> Array<T, D>;
impl<T, D> Array<T, D> {
    pub fn split_at(self, axis: Axis, index: usize) -> (Self, Self);
}
```

---

### Phase 6: SIMD Differentiation — Where rustynum WINS (2 weeks)

These are features ndarray does NOT have — rustynum's unfair advantage.

#### 6.1 SIMD-Aware Memory Layout

```rust
impl<T, D> Array<T, D> {
    pub fn ensure_aligned(&mut self);  // Reallocate to 64-byte boundary
    pub fn is_simd_aligned(&self) -> bool;
}
```

Blackboard integration: allocations from Blackboard are always 64-byte aligned.

#### 6.2 Fused Kernel Operations

```rust
impl Array<f32, D> {
    pub fn fused_mul_add(&self, a: &Self, b: &Self) -> Self;  // FMA
    pub fn axpy(&mut self, alpha: f32, x: &Self);  // y += alpha*x
    pub fn scal(&mut self, alpha: f32);  // x *= alpha
}
```

These call directly into `rustynum_core::simd::axpy_f32`, etc.

#### 6.3 Quantized Operations

```rust
impl Array<f32, D> {
    pub fn quantize_to_i8(&self) -> (Array<i8, D>, f32, i8);  // (quantized, scale, zero_point)
    pub fn quantize_to_bf16(&self) -> Array<BF16, D>;
    pub fn matmul_i8(&self, rhs: &Array<i8, D>) -> Array<f32, D>;  // via int8_gemm
    pub fn matmul_bf16(&self, rhs: &Array<BF16, D>) -> Array<f32, D>;  // via bf16_gemm
}
```

#### 6.4 HDC / Binary Operations

```rust
impl Array<u8, D> {
    pub fn hamming_distance(&self, other: &Self) -> u64;  // via simd
    pub fn hamming_batch(&self, queries: &Self) -> Array<u64, Ix1>;
    pub fn popcount(&self) -> u64;
}
```

---

## Implementation Priority & Timeline

```
Week 1-4:   Phase 1 (Foundation) — blocks everything else
Week 5-7:   Phase 2 (Slicing & Broadcasting) — usability leap
Week 8-10:  Phase 3 (Iteration & Parallelism) — performance + ergonomics
Week 11-12: Phase 4 (Linear Algebra) — mostly wiring existing rustyblas
Week 13-14: Phase 5 (Ecosystem) — adoption enablers
Week 15-16: Phase 6 (SIMD Differentiation) — competitive moat
```

## Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Keep `NumArray<T, Ops>` | Yes, as alias | Backward compat; existing users don't break |
| New type lives where? | `rustynum-rs/src/ndarray/` | Same crate, new module |
| Dimension type system | Compile-time fixed + IxDyn | Matches ndarray, catches shape bugs at compile time |
| SIMD dispatch | Extension traits on new array type | `Array<f32, D>` gets SIMD methods; generic `Array<T, D>` stays general |
| Storage alignment | `OwnedStorage` uses 64-byte aligned alloc | SIMD perf without runtime checks |
| Rayon | Feature-gated | Optional dep, matches ndarray approach |
| Feature flag | `ndarray-compat` | When enabled, exports ndarray-compatible API names |

## Success Criteria

- [ ] Can run ndarray's basic test suite against rustynum equivalents
- [ ] `Array<f32, Ix2>` matmul is ≥1.5x faster than ndarray (thanks to AVX-512 GEMM)
- [ ] Broadcasting matches NumPy rules completely
- [ ] `Zip::from(a).and(b).for_each(|a, b| ...)` works
- [ ] serde round-trip works
- [ ] burn-rustynum can switch from `RustyNumTensor` to new `Array` type seamlessly
