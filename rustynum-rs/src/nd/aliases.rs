//! Type aliases providing ndarray-compatible naming conventions.
//!
//! This module defines convenience type aliases that mirror the naming used by
//! the `ndarray` crate. Users familiar with ndarray can use `Array2<f64>`,
//! `ArrayView1<f32>`, etc., and get the corresponding `NdArray` instantiation.
//!
//! # Naming convention
//!
//! | Alias pattern       | Storage              | Meaning                        |
//! |---------------------|----------------------|--------------------------------|
//! | `Array<T, D>`       | `OwnedStorage<T>`   | Owned N-dimensional array      |
//! | `ArrayView<'a,T,D>` | `ViewStorage<'a,T>` | Immutable borrowed view        |
//! | `ArrayViewMut<'a,T,D>` | `ViewMutStorage<'a,T>` | Mutable borrowed view   |
//!
//! Numeric suffixes (e.g. `Array2`) fix the dimensionality at compile time.
//! The `D` suffix (e.g. `ArrayD`) uses `IxDyn` for dynamic dimensionality.

use super::array::NdArray;
use super::dimension::*;
use super::storage::*;

// ---------------------------------------------------------------------------
// Standard index type
// ---------------------------------------------------------------------------

/// A single axis index, aliased to `usize` for consistency with ndarray.
pub type Ix = usize;

// ---------------------------------------------------------------------------
// Owned array aliases
// ---------------------------------------------------------------------------

/// An owned N-dimensional array with compile-time or dynamic dimensionality `D`.
pub type Array<T, D> = NdArray<T, OwnedStorage<T>, D>;

/// Owned 1-dimensional array (vector).
pub type Array1<T> = Array<T, Ix1>;

/// Owned 2-dimensional array (matrix).
pub type Array2<T> = Array<T, Ix2>;

/// Owned 3-dimensional array.
pub type Array3<T> = Array<T, Ix3>;

/// Owned 4-dimensional array.
pub type Array4<T> = Array<T, Ix4>;

/// Owned 5-dimensional array.
pub type Array5<T> = Array<T, Ix5>;

/// Owned 6-dimensional array.
pub type Array6<T> = Array<T, Ix6>;

/// Owned array with dynamic dimensionality.
pub type ArrayD<T> = Array<T, IxDyn>;

// ---------------------------------------------------------------------------
// Immutable view aliases
// ---------------------------------------------------------------------------

/// An immutable view into an N-dimensional array.
pub type ArrayView<'a, T, D> = NdArray<T, ViewStorage<'a, T>, D>;

/// Immutable 1-dimensional view.
pub type ArrayView1<'a, T> = ArrayView<'a, T, Ix1>;

/// Immutable 2-dimensional view.
pub type ArrayView2<'a, T> = ArrayView<'a, T, Ix2>;

/// Immutable 3-dimensional view.
pub type ArrayView3<'a, T> = ArrayView<'a, T, Ix3>;

/// Immutable view with dynamic dimensionality.
pub type ArrayViewD<'a, T> = ArrayView<'a, T, IxDyn>;

// ---------------------------------------------------------------------------
// Mutable view aliases
// ---------------------------------------------------------------------------

/// A mutable view into an N-dimensional array.
pub type ArrayViewMut<'a, T, D> = NdArray<T, ViewMutStorage<'a, T>, D>;

/// Mutable 1-dimensional view.
pub type ArrayViewMut1<'a, T> = ArrayViewMut<'a, T, Ix1>;

/// Mutable 2-dimensional view.
pub type ArrayViewMut2<'a, T> = ArrayViewMut<'a, T, Ix2>;

/// Mutable 3-dimensional view.
pub type ArrayViewMut3<'a, T> = ArrayViewMut<'a, T, Ix3>;

/// Mutable view with dynamic dimensionality.
pub type ArrayViewMutD<'a, T> = ArrayViewMut<'a, T, IxDyn>;
