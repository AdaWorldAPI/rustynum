//! Core N-dimensional array type and constructors.
//!
//! `NdArray<T, S, D>` is the central array type, generic over element type `T`,
//! storage backend `S`, and dimensionality `D`. Constructors live on the owned
//! variant (`OwnedStorage`); views are produced by slicing operations elsewhere.

use std::fmt;
use std::marker::PhantomData;

use super::dimension::{Dimension, IntoDimension};
use super::order::{compute_c_strides, is_c_contiguous, offset_from_indices, shape_size};
use super::storage::{OwnedStorage, Storage, StorageMut};

// ---------------------------------------------------------------------------
// ShapeError
// ---------------------------------------------------------------------------

/// Error returned when array construction fails due to shape/data mismatch.
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeError {
    /// The number of elements in the data does not match the shape product.
    LengthMismatch { expected: usize, got: usize },
    /// The requested shape is incompatible with the operation.
    IncompatibleShape,
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeError::LengthMismatch { expected, got } => {
                write!(
                    f,
                    "shape requires {} elements but {} were provided",
                    expected, got
                )
            }
            ShapeError::IncompatibleShape => {
                write!(f, "incompatible shape for the requested operation")
            }
        }
    }
}

impl std::error::Error for ShapeError {}

// ---------------------------------------------------------------------------
// NdArray
// ---------------------------------------------------------------------------

/// N-dimensional array with pluggable storage and compile-time or dynamic
/// dimensionality.
///
/// # Type parameters
///
/// - `T` — element type (e.g. `f32`, `f64`, `i32`)
/// - `S` — storage backend (owned, immutable view, mutable view)
/// - `D` — dimension descriptor (fixed `Ix1`..`Ix6` or dynamic `IxDyn`)
pub struct NdArray<T, S, D>
where
    S: Storage<T>,
    D: Dimension,
{
    /// Underlying data buffer.
    storage: S,
    /// Shape stored as a dimension type.
    dim: D,
    /// Signed strides in units of elements (not bytes).
    strides: Vec<isize>,
    /// Offset into the storage buffer (element index of the first element).
    offset: usize,
    /// Marker to tie the element type into the struct without storing it.
    _phantom: PhantomData<T>,
}

// Send + Sync are auto-derived when T, S, D satisfy them. Explicit impls are
// not required because `PhantomData<T>` already propagates the bounds and all
// other fields are `Send + Sync` when `T` is.

// ---------------------------------------------------------------------------
// Constructors (owned storage only)
// ---------------------------------------------------------------------------

impl<T, D> NdArray<T, OwnedStorage<T>, D>
where
    D: Dimension,
{
    /// Create an array filled with the default value for `T`.
    ///
    /// ```ignore
    /// let a = NdArray::<f64, _, Ix2>::zeros([3, 4]);
    /// assert_eq!(a.shape(), &[3, 4]);
    /// assert_eq!(a.len(), 12);
    /// ```
    pub fn zeros(shape: impl IntoDimension<Dim = D>) -> Self
    where
        T: Default + Clone,
    {
        let dim = shape.into_dimension();
        let size = dim.size();
        let strides = compute_c_strides(dim.as_slice());
        let storage = OwnedStorage::zeros(size);
        Self {
            storage,
            dim,
            strides,
            offset: 0,
            _phantom: PhantomData,
        }
    }

    /// Create an array filled with ones.
    ///
    /// Requires `T: From<u8>` so we can produce the value `1` generically.
    pub fn ones(shape: impl IntoDimension<Dim = D>) -> Self
    where
        T: Clone + From<u8>,
    {
        Self::from_elem(shape, T::from(1u8))
    }

    /// Create an array where every element is `elem`.
    pub fn from_elem(shape: impl IntoDimension<Dim = D>, elem: T) -> Self
    where
        T: Clone,
    {
        let dim = shape.into_dimension();
        let size = dim.size();
        let strides = compute_c_strides(dim.as_slice());
        let storage = OwnedStorage::new(vec![elem; size]);
        Self {
            storage,
            dim,
            strides,
            offset: 0,
            _phantom: PhantomData,
        }
    }

    /// Create an array from an existing `Vec<T>`.
    ///
    /// Returns `Err(ShapeError::LengthMismatch)` if `data.len()` does not
    /// equal the product of the shape dimensions.
    pub fn from_vec(shape: impl IntoDimension<Dim = D>, data: Vec<T>) -> Result<Self, ShapeError> {
        let dim = shape.into_dimension();
        let expected = dim.size();
        if data.len() != expected {
            return Err(ShapeError::LengthMismatch {
                expected,
                got: data.len(),
            });
        }
        let strides = compute_c_strides(dim.as_slice());
        Ok(Self {
            storage: OwnedStorage::new(data),
            dim,
            strides,
            offset: 0,
            _phantom: PhantomData,
        })
    }

    /// Create an array by calling `f` for each multi-dimensional index.
    ///
    /// Indices are generated in row-major (C) order. The closure receives the
    /// index as a `&[usize]` slice whose length equals `ndim`.
    pub fn from_shape_fn<F>(shape: impl IntoDimension<Dim = D>, mut f: F) -> Self
    where
        F: FnMut(&[usize]) -> T,
    {
        let dim = shape.into_dimension();
        let shape_slice = dim.as_slice();
        let ndim = shape_slice.len();
        let size = shape_size(shape_slice);
        let strides = compute_c_strides(shape_slice);

        let mut data = Vec::with_capacity(size);
        let mut index = vec![0usize; ndim];

        for _ in 0..size {
            data.push(f(&index));
            // Increment the multi-index in row-major order (rightmost first).
            Self::increment_index(&mut index, shape_slice);
        }

        Self {
            storage: OwnedStorage::new(data),
            dim,
            strides,
            offset: 0,
            _phantom: PhantomData,
        }
    }

    /// Consume the array and return its data as a `Vec<T>`.
    ///
    /// If the array is contiguous with zero offset, this is a zero-copy
    /// operation. Otherwise the elements are collected into a new vector in
    /// row-major order.
    pub fn into_vec(self) -> Vec<T>
    where
        T: Clone,
    {
        if self.offset == 0 && self.is_contiguous() {
            Vec::from(self.storage)
        } else {
            // Non-contiguous or offset view: collect elements.
            self.iter_elements().cloned().collect()
        }
    }

    // Increment a multi-index in row-major (C) order.
    fn increment_index(index: &mut [usize], shape: &[usize]) {
        for i in (0..index.len()).rev() {
            index[i] += 1;
            if index[i] < shape[i] {
                return;
            }
            index[i] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Read-only accessors (any storage)
// ---------------------------------------------------------------------------

impl<T, S, D> NdArray<T, S, D>
where
    S: Storage<T>,
    D: Dimension,
{
    /// Returns the shape as a slice of dimension sizes.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.dim.as_slice()
    }

    /// Returns the number of dimensions (axes).
    #[inline]
    pub fn ndim(&self) -> usize {
        self.dim.ndim()
    }

    /// Returns the total number of elements in the array.
    #[inline]
    pub fn len(&self) -> usize {
        self.dim.size()
    }

    /// Returns `true` if the array contains zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the strides (in element counts, signed) for each axis.
    #[inline]
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Returns `true` if the data is laid out contiguously in row-major (C)
    /// order with no gaps or repeated elements.
    #[inline]
    pub fn is_contiguous(&self) -> bool {
        is_c_contiguous(self.dim.as_slice(), &self.strides)
    }

    /// Returns a slice over the underlying data if the array is contiguous
    /// and starts at the beginning of the storage buffer.
    ///
    /// Returns `None` for non-contiguous arrays or arrays with a non-zero
    /// offset into the storage.
    pub fn as_slice(&self) -> Option<&[T]> {
        if self.is_contiguous() && self.offset == 0 {
            let slice = self.storage.as_slice();
            Some(&slice[..self.len()])
        } else {
            None
        }
    }

    /// Bounds-checked element access by multi-dimensional index.
    ///
    /// Returns `None` if the index is out of bounds or has the wrong number
    /// of dimensions.
    pub fn get(&self, index: &[usize]) -> Option<&T> {
        if !self.check_bounds(index) {
            return None;
        }
        let flat = offset_from_indices(index, &self.strides) + self.offset;
        self.storage.as_slice().get(flat)
    }

    /// Create an owned copy of this array, regardless of storage type.
    pub fn to_owned(&self) -> NdArray<T, OwnedStorage<T>, D>
    where
        T: Clone,
    {
        let data: Vec<T> = self.iter_elements().cloned().collect();
        let strides = compute_c_strides(self.dim.as_slice());
        NdArray {
            storage: OwnedStorage::new(data),
            dim: self.dim.clone(),
            strides,
            offset: 0,
            _phantom: PhantomData,
        }
    }

    // -- internal helpers --------------------------------------------------

    /// Validate that `index` has the right number of dimensions and each
    /// component is within bounds.
    fn check_bounds(&self, index: &[usize]) -> bool {
        let shape = self.dim.as_slice();
        if index.len() != shape.len() {
            return false;
        }
        index.iter().zip(shape.iter()).all(|(&i, &s)| i < s)
    }

    /// Iterate over all elements in logical (row-major) order.
    ///
    /// This works for both contiguous and non-contiguous arrays by walking
    /// multi-indices.
    fn iter_elements(&self) -> ElementIter<'_, T, S, D> {
        ElementIter {
            array: self,
            index: vec![0usize; self.ndim()],
            remaining: self.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Mutable accessors (StorageMut only)
// ---------------------------------------------------------------------------

impl<T, S, D> NdArray<T, S, D>
where
    S: StorageMut<T>,
    D: Dimension,
{
    /// Returns a mutable slice over the underlying data if the array is
    /// contiguous and starts at the beginning of the storage buffer.
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        let len = self.len();
        if self.is_contiguous() && self.offset == 0 {
            let slice = self.storage.as_slice_mut();
            Some(&mut slice[..len])
        } else {
            None
        }
    }

    /// Bounds-checked mutable element access by multi-dimensional index.
    pub fn get_mut(&mut self, index: &[usize]) -> Option<&mut T> {
        if !self.check_bounds(index) {
            return None;
        }
        let flat = offset_from_indices(index, &self.strides) + self.offset;
        self.storage.as_slice_mut().get_mut(flat)
    }
}

// ---------------------------------------------------------------------------
// Clone
// ---------------------------------------------------------------------------

impl<T, S, D> Clone for NdArray<T, S, D>
where
    T: Clone,
    S: Storage<T> + Clone,
    D: Dimension + Clone,
{
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            dim: self.dim.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            _phantom: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Debug
// ---------------------------------------------------------------------------

impl<T, S, D> fmt::Debug for NdArray<T, S, D>
where
    T: fmt::Debug,
    S: Storage<T>,
    D: Dimension + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NdArray")
            .field("shape", &self.dim)
            .field("strides", &self.strides)
            .field("offset", &self.offset)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ElementIter — internal iterator over logical elements
// ---------------------------------------------------------------------------

/// Internal iterator that walks elements in row-major order via multi-index
/// arithmetic. Not public; used by `to_owned()` and `into_vec()`.
struct ElementIter<'a, T, S, D>
where
    S: Storage<T>,
    D: Dimension,
{
    array: &'a NdArray<T, S, D>,
    index: Vec<usize>,
    remaining: usize,
}

impl<'a, T, S, D> Iterator for ElementIter<'a, T, S, D>
where
    S: Storage<T>,
    D: Dimension,
{
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;

        let flat =
            offset_from_indices(&self.index, self.array.strides()) + self.array.offset;
        let elem = &self.array.storage.as_slice()[flat];

        // Advance multi-index (row-major).
        let shape = self.array.shape();
        for i in (0..self.index.len()).rev() {
            self.index[i] += 1;
            if self.index[i] < shape[i] {
                break;
            }
            self.index[i] = 0;
        }

        Some(elem)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl<T, S, D> ExactSizeIterator for ElementIter<'_, T, S, D>
where
    S: Storage<T>,
    D: Dimension,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nd::dimension::{Ix1, Ix2, Ix3, IxDyn};
    use crate::nd::storage::OwnedStorage;

    // -- ShapeError --------------------------------------------------------

    #[test]
    fn shape_error_display_length_mismatch() {
        let e = ShapeError::LengthMismatch {
            expected: 12,
            got: 10,
        };
        let msg = format!("{}", e);
        assert!(msg.contains("12"));
        assert!(msg.contains("10"));
    }

    #[test]
    fn shape_error_display_incompatible() {
        let e = ShapeError::IncompatibleShape;
        let msg = format!("{}", e);
        assert!(msg.contains("incompatible"));
    }

    // -- zeros / ones / from_elem -----------------------------------------

    #[test]
    fn zeros_creates_correct_shape_and_values() {
        let a = NdArray::<f64, OwnedStorage<f64>, Ix2>::zeros([3, 4]);
        assert_eq!(a.shape(), &[3, 4]);
        assert_eq!(a.ndim(), 2);
        assert_eq!(a.len(), 12);
        assert!(!a.is_empty());
        assert!(a.is_contiguous());
        let s = a.as_slice().unwrap();
        assert!(s.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn ones_creates_all_ones() {
        let a = NdArray::<f64, OwnedStorage<f64>, Ix1>::ones([5]);
        let s = a.as_slice().unwrap();
        assert!(s.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn from_elem_fills_correctly() {
        let a = NdArray::<i32, OwnedStorage<i32>, Ix2>::from_elem([2, 3], 42);
        assert_eq!(a.len(), 6);
        let s = a.as_slice().unwrap();
        assert!(s.iter().all(|&x| x == 42));
    }

    // -- from_vec ---------------------------------------------------------

    #[test]
    fn from_vec_ok() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = NdArray::<f64, OwnedStorage<f64>, Ix2>::from_vec([2, 3], data).unwrap();
        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.get(&[0, 0]), Some(&1.0));
        assert_eq!(a.get(&[1, 2]), Some(&6.0));
    }

    #[test]
    fn from_vec_length_mismatch() {
        let data = vec![1, 2, 3];
        let result = NdArray::<i32, OwnedStorage<i32>, Ix2>::from_vec([2, 3], data);
        assert!(result.is_err());
        match result.unwrap_err() {
            ShapeError::LengthMismatch { expected, got } => {
                assert_eq!(expected, 6);
                assert_eq!(got, 3);
            }
            _ => panic!("wrong error variant"),
        }
    }

    // -- from_shape_fn ----------------------------------------------------

    #[test]
    fn from_shape_fn_generates_indices() {
        // Fill with row * 10 + col
        let a = NdArray::<usize, OwnedStorage<usize>, Ix2>::from_shape_fn(
            [3, 4],
            |idx| idx[0] * 10 + idx[1],
        );
        assert_eq!(a.get(&[0, 0]), Some(&0));
        assert_eq!(a.get(&[0, 3]), Some(&3));
        assert_eq!(a.get(&[2, 3]), Some(&23));
    }

    // -- element access ---------------------------------------------------

    #[test]
    fn get_out_of_bounds_returns_none() {
        let a = NdArray::<f64, OwnedStorage<f64>, Ix2>::zeros([2, 3]);
        assert!(a.get(&[2, 0]).is_none()); // row out of bounds
        assert!(a.get(&[0, 3]).is_none()); // col out of bounds
        assert!(a.get(&[0]).is_none());    // wrong ndim
    }

    #[test]
    fn get_mut_modifies_element() {
        let mut a = NdArray::<i32, OwnedStorage<i32>, Ix2>::from_elem([2, 2], 0);
        *a.get_mut(&[1, 0]).unwrap() = 99;
        assert_eq!(a.get(&[1, 0]), Some(&99));
        assert_eq!(a.get(&[0, 0]), Some(&0));
    }

    // -- strides / contiguity ---------------------------------------------

    #[test]
    fn strides_are_row_major() {
        let a = NdArray::<f64, OwnedStorage<f64>, Ix3>::zeros([2, 3, 4]);
        // Row-major strides for [2,3,4]: [12, 4, 1]
        assert_eq!(a.strides(), &[12, 4, 1]);
    }

    // -- into_vec / to_owned ----------------------------------------------

    #[test]
    fn into_vec_roundtrip() {
        let data = vec![10, 20, 30, 40];
        let a = NdArray::<i32, OwnedStorage<i32>, Ix1>::from_vec([4], data.clone()).unwrap();
        assert_eq!(a.into_vec(), data);
    }

    #[test]
    fn to_owned_copies_data() {
        let data = vec![1.0, 2.0, 3.0];
        let a = NdArray::<f64, OwnedStorage<f64>, Ix1>::from_vec([3], data).unwrap();
        let b = a.to_owned();
        assert_eq!(b.shape(), a.shape());
        assert_eq!(b.as_slice().unwrap(), a.as_slice().unwrap());
    }

    // -- clone ------------------------------------------------------------

    #[test]
    fn clone_produces_independent_copy() {
        let mut a = NdArray::<i32, OwnedStorage<i32>, Ix1>::from_vec([3], vec![1, 2, 3]).unwrap();
        let b = a.clone();
        *a.get_mut(&[0]).unwrap() = 99;
        assert_eq!(a.get(&[0]), Some(&99));
        assert_eq!(b.get(&[0]), Some(&1)); // clone is independent
    }

    // -- empty / scalar-ish cases -----------------------------------------

    #[test]
    fn empty_array() {
        let a = NdArray::<f64, OwnedStorage<f64>, Ix1>::zeros([0]);
        assert!(a.is_empty());
        assert_eq!(a.len(), 0);
        assert!(a.as_slice().unwrap().is_empty());
    }

    #[test]
    fn scalar_like_1x1() {
        let a = NdArray::<f64, OwnedStorage<f64>, Ix2>::from_elem([1, 1], 3.14);
        assert_eq!(a.len(), 1);
        assert_eq!(a.get(&[0, 0]), Some(&3.14));
    }

    // -- IxDyn ------------------------------------------------------------

    #[test]
    fn dynamic_dimension_works() {
        let a = NdArray::<f64, OwnedStorage<f64>, IxDyn>::zeros(vec![2, 3, 4]);
        assert_eq!(a.ndim(), 3);
        assert_eq!(a.shape(), &[2, 3, 4]);
        assert_eq!(a.len(), 24);
    }
}
