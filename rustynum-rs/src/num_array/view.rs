//! Zero-cost array views with stride-based memory access.
//!
//! `ArrayView` and `ArrayViewMut` provide O(1) transpose, slice, and reshape
//! operations by manipulating shape/strides/offset metadata without copying data.
//! This matches ndarray's core design: views are cheap pointer+metadata wrappers.
//!
//! # Design
//!
//! A view consists of:
//! - A borrowed slice of the underlying data
//! - `shape`: dimensions of the view
//! - `strides`: element-count steps per dimension (can be negative for reversed axes)
//! - `offset`: starting element index into the data slice
//!
//! All view operations (transpose, slice, reshape) return new views in O(1)
//! by only modifying metadata. No data is copied or allocated.

use std::fmt::Debug;

/// Immutable view into array data with stride-based access.
///
/// Created via `NumArray::view()`. Supports O(1) transpose, slice, and reshape.
///
/// # Example
/// ```
/// use rustynum_rs::NumArrayF32;
///
/// let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
/// let v = a.view();
/// let vt = v.t();  // O(1) transpose — no copy
/// assert_eq!(vt.shape(), &[3, 2]);
/// assert_eq!(vt.get(&[0, 1]), 4.0);
/// ```
pub struct ArrayView<'a, T> {
    data: &'a [T],
    shape: Vec<usize>,
    strides: Vec<isize>, // signed: negative = reversed axis
    offset: usize,
}

/// Mutable view into array data with stride-based access.
///
/// Created via `NumArray::view_mut()`. Supports O(1) transpose, slice, and reshape.
pub struct ArrayViewMut<'a, T> {
    data: &'a mut [T],
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
}

// ============================================================================
// ArrayView — immutable
// ============================================================================

impl<'a, T: Copy + Debug> ArrayView<'a, T> {
    /// Create a new view from raw parts.
    pub(crate) fn new(
        data: &'a [T],
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: usize,
    ) -> Self {
        Self {
            data,
            shape,
            strides,
            offset,
        }
    }

    /// Shape of the view.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Strides of the view (signed, in element counts).
    #[inline]
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements in the view.
    #[inline]
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Whether the view is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether the view is C-contiguous (row-major, no gaps).
    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }
        let mut expected_stride = 1isize;
        for i in (0..self.shape.len()).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride *= self.shape[i] as isize;
        }
        true
    }

    /// Access a single element by multi-dimensional index.
    ///
    /// # Panics
    /// Panics if the index is out of bounds.
    #[inline]
    pub fn get(&self, index: &[usize]) -> T {
        debug_assert_eq!(index.len(), self.shape.len());
        let flat = self.flat_index(index);
        self.data[flat]
    }

    /// Transpose — O(1), no data copy. Swaps shape and strides.
    ///
    /// For 2D: equivalent to matrix transpose.
    /// For nD: reverses all axes.
    pub fn t(&self) -> ArrayView<'a, T> {
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.reverse();
        strides.reverse();
        ArrayView {
            data: self.data,
            shape,
            strides,
            offset: self.offset,
        }
    }

    /// Swap two axes — O(1).
    pub fn swap_axes(&self, a: usize, b: usize) -> ArrayView<'a, T> {
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.swap(a, b);
        strides.swap(a, b);
        ArrayView {
            data: self.data,
            shape,
            strides,
            offset: self.offset,
        }
    }

    /// Slice along one axis — O(1). Returns a narrower view.
    ///
    /// `start..end` selects elements along `axis` in the half-open range.
    pub fn slice_axis(&self, axis: usize, start: usize, end: usize) -> ArrayView<'a, T> {
        assert!(axis < self.ndim(), "axis out of bounds");
        assert!(end <= self.shape[axis], "slice end out of bounds");
        assert!(start <= end, "slice start must be <= end");

        let mut shape = self.shape.clone();
        shape[axis] = end - start;

        // Advance offset by start * stride along this axis
        let new_offset = (self.offset as isize + start as isize * self.strides[axis]) as usize;

        ArrayView {
            data: self.data,
            shape,
            strides: self.strides.clone(),
            offset: new_offset,
        }
    }

    /// Reverse an axis — O(1). Negates the stride and adjusts offset.
    pub fn flip_axis(&self, axis: usize) -> ArrayView<'a, T> {
        assert!(axis < self.ndim(), "axis out of bounds");
        let mut strides = self.strides.clone();
        let n = self.shape[axis];
        if n > 0 {
            let new_offset = (self.offset as isize + (n as isize - 1) * strides[axis]) as usize;
            strides[axis] = -strides[axis];
            ArrayView {
                data: self.data,
                shape: self.shape.clone(),
                strides,
                offset: new_offset,
            }
        } else {
            self.reborrow()
        }
    }

    /// Reshape — O(1) if contiguous, otherwise None.
    ///
    /// Returns `None` if the view is not contiguous (would require data copy).
    pub fn reshape(&self, new_shape: &[usize]) -> Option<ArrayView<'a, T>> {
        let new_len: usize = new_shape.iter().product();
        if new_len != self.len() {
            return None;
        }
        if !self.is_contiguous() {
            return None;
        }
        let strides = compute_c_strides(new_shape);
        Some(ArrayView {
            data: self.data,
            shape: new_shape.to_vec(),
            strides,
            offset: self.offset,
        })
    }

    /// Collect view into a contiguous Vec (materializes the view).
    pub fn to_vec(&self) -> Vec<T> {
        if self.is_contiguous() && self.offset == 0 && self.len() == self.data.len() {
            return self.data.to_vec();
        }
        let mut result = Vec::with_capacity(self.len());
        self.collect_recursive(&mut result, 0, self.offset);
        result
    }

    /// Iterate over elements in logical (row-major) order.
    pub fn iter(&self) -> ViewIter<'a, T> {
        ViewIter::new(self)
    }

    /// Create a reborrowed view (same lifetime, same data).
    fn reborrow(&self) -> ArrayView<'a, T> {
        ArrayView {
            data: self.data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
        }
    }

    #[inline]
    fn flat_index(&self, index: &[usize]) -> usize {
        let mut flat = self.offset as isize;
        for (i, &idx) in index.iter().enumerate() {
            debug_assert!(idx < self.shape[i], "index out of bounds");
            flat += idx as isize * self.strides[i];
        }
        flat as usize
    }

    fn collect_recursive(&self, result: &mut Vec<T>, dim: usize, current_offset: usize) {
        if dim == self.shape.len() {
            result.push(self.data[current_offset]);
            return;
        }
        for i in 0..self.shape[dim] {
            let next = (current_offset as isize + i as isize * self.strides[dim]) as usize;
            self.collect_recursive(result, dim + 1, next);
        }
    }
}

// ============================================================================
// ArrayViewMut — mutable
// ============================================================================

impl<'a, T: Copy + Debug> ArrayViewMut<'a, T> {
    /// Create a new mutable view from raw parts.
    pub(crate) fn new(
        data: &'a mut [T],
        shape: Vec<usize>,
        strides: Vec<isize>,
        offset: usize,
    ) -> Self {
        Self {
            data,
            shape,
            strides,
            offset,
        }
    }

    /// Shape of the view.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Strides of the view.
    #[inline]
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// Access a single element by multi-dimensional index.
    #[inline]
    pub fn get(&self, index: &[usize]) -> T {
        debug_assert_eq!(index.len(), self.shape.len());
        let flat = self.flat_index(index);
        self.data[flat]
    }

    /// Set a single element by multi-dimensional index.
    #[inline]
    pub fn set(&mut self, index: &[usize], value: T) {
        debug_assert_eq!(index.len(), self.shape.len());
        let flat = self.flat_index(index);
        self.data[flat] = value;
    }

    /// Downgrade to immutable view.
    pub fn as_view(&self) -> ArrayView<'_, T> {
        ArrayView {
            data: self.data,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
        }
    }

    /// Fill all elements in the view with a value.
    pub fn fill(&mut self, value: T) {
        let len = self.shape.iter().product::<usize>();
        let shape = self.shape.clone();
        let strides = self.strides.clone();
        let offset = self.offset;
        self.fill_recursive(value, 0, offset, &shape, &strides, len);
    }

    #[inline]
    fn flat_index(&self, index: &[usize]) -> usize {
        let mut flat = self.offset as isize;
        for (i, &idx) in index.iter().enumerate() {
            debug_assert!(idx < self.shape[i], "index out of bounds");
            flat += idx as isize * self.strides[i];
        }
        flat as usize
    }

    fn fill_recursive(
        &mut self,
        value: T,
        dim: usize,
        current_offset: usize,
        shape: &[usize],
        strides: &[isize],
        _len: usize,
    ) {
        if dim == shape.len() {
            self.data[current_offset] = value;
            return;
        }
        for i in 0..shape[dim] {
            let next = (current_offset as isize + i as isize * strides[dim]) as usize;
            self.fill_recursive(value, dim + 1, next, shape, strides, _len);
        }
    }
}

// ============================================================================
// Iterator
// ============================================================================

/// Row-major iterator over view elements.
pub struct ViewIter<'a, T> {
    data: &'a [T],
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
    indices: Vec<usize>,
    done: bool,
}

impl<'a, T: Copy + Debug> ViewIter<'a, T> {
    fn new(view: &ArrayView<'a, T>) -> Self {
        let done = view.is_empty();
        ViewIter {
            data: view.data,
            shape: view.shape.clone(),
            strides: view.strides.clone(),
            offset: view.offset,
            indices: vec![0; view.shape.len()],
            done,
        }
    }

    #[inline]
    fn flat_index(&self) -> usize {
        let mut flat = self.offset as isize;
        for (i, &idx) in self.indices.iter().enumerate() {
            flat += idx as isize * self.strides[i];
        }
        flat as usize
    }
}

impl<'a, T: Copy + Debug> Iterator for ViewIter<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        if self.done {
            return None;
        }
        let val = self.data[self.flat_index()];

        // Increment indices in row-major order (last dimension fastest)
        let ndim = self.shape.len();
        let mut carry = true;
        for d in (0..ndim).rev() {
            if carry {
                self.indices[d] += 1;
                if self.indices[d] < self.shape[d] {
                    carry = false;
                } else {
                    self.indices[d] = 0;
                }
            }
        }
        if carry {
            self.done = true;
        }

        Some(val)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done {
            return (0, Some(0));
        }
        let total: usize = self.shape.iter().product();
        // Approximate remaining from current position
        let mut consumed = 0usize;
        let mut multiplier = 1usize;
        for d in (0..self.shape.len()).rev() {
            consumed += self.indices[d] * multiplier;
            multiplier *= self.shape[d];
        }
        let remaining = total.saturating_sub(consumed);
        (remaining, Some(remaining))
    }
}

impl<'a, T: Copy + Debug> ExactSizeIterator for ViewIter<'a, T> {}

// ============================================================================
// Helpers
// ============================================================================

/// Compute C-contiguous (row-major) strides for a shape.
pub(crate) fn compute_c_strides(shape: &[usize]) -> Vec<isize> {
    let ndim = shape.len();
    let mut strides = vec![1isize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as isize;
    }
    strides
}

// ============================================================================
// Integration: view() and view_mut() on NumArray
// ============================================================================

use super::NumArray;

impl<T: Copy + Debug, Ops> NumArray<T, Ops>
where
    T: Debug,
{
    /// Create an immutable view of this array. O(1), no data copy.
    ///
    /// The view supports O(1) transpose, slice, and reshape operations.
    pub fn view(&self) -> ArrayView<'_, T> {
        let strides = compute_c_strides(&self.shape);
        ArrayView::new(&self.data, self.shape.clone(), strides, 0)
    }

    /// Create a mutable view of this array. O(1), no data copy.
    pub fn view_mut(&mut self) -> ArrayViewMut<'_, T> {
        let strides = compute_c_strides(&self.shape);
        let shape = self.shape.clone();
        ArrayViewMut::new(&mut self.data, shape, strides, 0)
    }

    /// O(1) transpose via view — returns the transposed data as a new owned array.
    ///
    /// This is faster than the existing `transpose()` for read-only access patterns
    /// because it avoids the intermediate copy when the result is only iterated.
    pub fn t_view(&self) -> ArrayView<'_, T> {
        self.view().t()
    }
}

// ============================================================================
// Display
// ============================================================================

impl<T: Copy + Debug> std::fmt::Debug for ArrayView<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ArrayView(shape={:?}, strides={:?})",
            self.shape, self.strides
        )
    }
}

impl<T: Copy + Debug> std::fmt::Debug for ArrayViewMut<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ArrayViewMut(shape={:?}, strides={:?})",
            self.shape, self.strides
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use crate::NumArrayF32;

    #[test]
    fn test_view_basic_access() {
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let v = a.view();
        assert_eq!(v.shape(), &[2, 3]);
        assert_eq!(v.get(&[0, 0]), 1.0);
        assert_eq!(v.get(&[0, 2]), 3.0);
        assert_eq!(v.get(&[1, 0]), 4.0);
        assert_eq!(v.get(&[1, 2]), 6.0);
    }

    #[test]
    fn test_view_transpose_o1() {
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let vt = a.view().t();
        assert_eq!(vt.shape(), &[3, 2]);
        // Transposed: column 0 of original becomes row 0
        assert_eq!(vt.get(&[0, 0]), 1.0);
        assert_eq!(vt.get(&[0, 1]), 4.0);
        assert_eq!(vt.get(&[1, 0]), 2.0);
        assert_eq!(vt.get(&[1, 1]), 5.0);
        assert_eq!(vt.get(&[2, 0]), 3.0);
        assert_eq!(vt.get(&[2, 1]), 6.0);
    }

    #[test]
    fn test_view_slice_axis() {
        let a = NumArrayF32::new_with_shape(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
        );
        // Slice rows 1..3 (second and third row)
        let v = a.view().slice_axis(0, 1, 3);
        assert_eq!(v.shape(), &[2, 3]);
        assert_eq!(v.get(&[0, 0]), 4.0); // was row 1
        assert_eq!(v.get(&[1, 2]), 9.0); // was row 2, col 2

        // Slice columns 0..2
        let v2 = a.view().slice_axis(1, 0, 2);
        assert_eq!(v2.shape(), &[3, 2]);
        assert_eq!(v2.get(&[0, 0]), 1.0);
        assert_eq!(v2.get(&[0, 1]), 2.0);
        assert_eq!(v2.get(&[2, 0]), 7.0);
        assert_eq!(v2.get(&[2, 1]), 8.0);
    }

    #[test]
    fn test_view_flip_axis() {
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let v = a.view().flip_axis(1); // reverse columns
        assert_eq!(v.get(&[0, 0]), 3.0);
        assert_eq!(v.get(&[0, 1]), 2.0);
        assert_eq!(v.get(&[0, 2]), 1.0);
        assert_eq!(v.get(&[1, 0]), 6.0);
    }

    #[test]
    fn test_view_reshape() {
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let v = a.view().reshape(&[3, 2]).unwrap();
        assert_eq!(v.shape(), &[3, 2]);
        assert_eq!(v.get(&[0, 0]), 1.0);
        assert_eq!(v.get(&[0, 1]), 2.0);
        assert_eq!(v.get(&[1, 0]), 3.0);
    }

    #[test]
    fn test_view_reshape_non_contiguous_fails() {
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let vt = a.view().t(); // transposed = non-contiguous
        assert!(vt.reshape(&[6]).is_none());
    }

    #[test]
    fn test_view_to_vec() {
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let vt = a.view().t();
        let collected = vt.to_vec();
        // Transposed row-major: [1, 4, 2, 5, 3, 6]
        assert_eq!(collected, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_view_iter() {
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let vt = a.view().t();
        let collected: Vec<f32> = vt.iter().collect();
        assert_eq!(collected, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_view_is_contiguous() {
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert!(a.view().is_contiguous());
        assert!(!a.view().t().is_contiguous()); // transposed = non-contiguous
    }

    #[test]
    fn test_view_mut_set() {
        let mut a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        {
            let mut v = a.view_mut();
            v.set(&[0, 1], 99.0);
            v.set(&[1, 0], 88.0);
        }
        assert_eq!(a.get_data(), &[1.0, 99.0, 88.0, 4.0]);
    }

    #[test]
    fn test_view_chained_ops() {
        // Chain: view → slice → transpose → flip → all O(1)
        let a = NumArrayF32::new_with_shape((1..=12).map(|i| i as f32).collect(), vec![3, 4]);
        let v = a
            .view()
            .slice_axis(0, 0, 2) // rows 0,1 → 2×4
            .t() // → 4×2
            .flip_axis(0); // reverse rows → 4×2

        assert_eq!(v.shape(), &[4, 2]);
        // Original rows 0,1: [1,2,3,4], [5,6,7,8]
        // After slice: [[1,2,3,4],[5,6,7,8]]
        // After transpose: [[1,5],[2,6],[3,7],[4,8]]
        // After flip axis 0: [[4,8],[3,7],[2,6],[1,5]]
        assert_eq!(v.get(&[0, 0]), 4.0);
        assert_eq!(v.get(&[0, 1]), 8.0);
        assert_eq!(v.get(&[3, 0]), 1.0);
        assert_eq!(v.get(&[3, 1]), 5.0);
    }

    #[test]
    fn test_t_view_shortcut() {
        let a = NumArrayF32::new_with_shape(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let vt = a.t_view();
        assert_eq!(vt.shape(), &[2, 2]);
        assert_eq!(vt.get(&[0, 1]), 3.0);
    }

    #[test]
    fn test_view_1d() {
        let a = NumArrayF32::new(vec![1.0, 2.0, 3.0, 4.0]);
        let v = a.view();
        assert_eq!(v.shape(), &[4]);
        assert_eq!(v.get(&[2]), 3.0);
        assert!(v.is_contiguous());
    }

    #[test]
    fn test_view_3d() {
        let a = NumArrayF32::new_with_shape((1..=24).map(|i| i as f32).collect(), vec![2, 3, 4]);
        let v = a.view();
        assert_eq!(v.get(&[0, 0, 0]), 1.0);
        assert_eq!(v.get(&[0, 0, 3]), 4.0);
        assert_eq!(v.get(&[0, 1, 0]), 5.0);
        assert_eq!(v.get(&[1, 0, 0]), 13.0);
        assert_eq!(v.get(&[1, 2, 3]), 24.0);

        // Transpose 3D reverses all axes: [2,3,4] → [4,3,2]
        let vt = v.t();
        assert_eq!(vt.shape(), &[4, 3, 2]);
        assert_eq!(vt.get(&[0, 0, 0]), 1.0);
        assert_eq!(vt.get(&[0, 0, 1]), 13.0);
    }
}
