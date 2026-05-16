/// Memory layout order and stride utilities for N-dimensional arrays.

/// Memory layout order for an N-dimensional array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// C order (row-major) -- last index varies fastest.
    RowMajor,
    /// Fortran order (column-major) -- first index varies fastest.
    ColumnMajor,
}

impl Order {
    /// Alias for `Order::RowMajor` (C order).
    pub const C: Order = Order::RowMajor;
    /// Alias for `Order::ColumnMajor` (Fortran order).
    pub const F: Order = Order::ColumnMajor;

    /// Returns `true` if this is row-major (C) order.
    pub fn is_row_major(&self) -> bool {
        matches!(self, Order::RowMajor)
    }

    /// Returns `true` if this is column-major (Fortran) order.
    pub fn is_column_major(&self) -> bool {
        matches!(self, Order::ColumnMajor)
    }
}

/// Layout property flags indicating whether data is contiguous in C or F order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayoutFlags {
    /// Data is contiguous in C (row-major) order.
    pub c_contiguous: bool,
    /// Data is contiguous in Fortran (column-major) order.
    pub f_contiguous: bool,
}

impl LayoutFlags {
    /// Compute layout flags from a shape and strides.
    pub fn from_shape_strides(shape: &[usize], strides: &[isize]) -> Self {
        LayoutFlags {
            c_contiguous: is_c_contiguous(shape, strides),
            f_contiguous: is_f_contiguous(shape, strides),
        }
    }

    /// Returns `true` if the data is contiguous in either C or F order.
    pub fn is_contiguous(&self) -> bool {
        self.c_contiguous || self.f_contiguous
    }

    /// Returns `true` if C-contiguous.
    pub fn is_c_contiguous(&self) -> bool {
        self.c_contiguous
    }

    /// Returns `true` if F-contiguous.
    pub fn is_f_contiguous(&self) -> bool {
        self.f_contiguous
    }
}

/// Compute C-order (row-major) strides for the given shape.
///
/// `strides[ndim-1] = 1`, `strides[i] = strides[i+1] * shape[i+1]`.
/// Returns an empty vec for a 0-dimensional shape.
pub fn compute_c_strides(shape: &[usize]) -> Vec<isize> {
    let ndim = shape.len();
    if ndim == 0 {
        return Vec::new();
    }
    let mut strides = vec![1isize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as isize;
    }
    strides
}

/// Compute F-order (column-major) strides for the given shape.
///
/// `strides[0] = 1`, `strides[i] = strides[i-1] * shape[i-1]`.
/// Returns an empty vec for a 0-dimensional shape.
pub fn compute_f_strides(shape: &[usize]) -> Vec<isize> {
    let ndim = shape.len();
    if ndim == 0 {
        return Vec::new();
    }
    let mut strides = vec![1isize; ndim];
    for i in 1..ndim {
        strides[i] = strides[i - 1] * shape[i - 1] as isize;
    }
    strides
}

/// Check whether the given shape and strides describe a C-contiguous layout.
///
/// Size-1 dimensions can have any stride without breaking contiguity.
pub fn is_c_contiguous(shape: &[usize], strides: &[isize]) -> bool {
    let ndim = shape.len();
    if ndim == 0 {
        return true;
    }
    if ndim != strides.len() {
        return false;
    }
    let expected = compute_c_strides(shape);
    for i in 0..ndim {
        // Size-1 dimensions can have any stride.
        if shape[i] == 1 {
            continue;
        }
        if strides[i] != expected[i] {
            return false;
        }
    }
    true
}

/// Check whether the given shape and strides describe an F-contiguous layout.
///
/// Size-1 dimensions can have any stride without breaking contiguity.
pub fn is_f_contiguous(shape: &[usize], strides: &[isize]) -> bool {
    let ndim = shape.len();
    if ndim == 0 {
        return true;
    }
    if ndim != strides.len() {
        return false;
    }
    let expected = compute_f_strides(shape);
    for i in 0..ndim {
        if shape[i] == 1 {
            continue;
        }
        if strides[i] != expected[i] {
            return false;
        }
    }
    true
}

/// Compute the product of all dimensions (total number of elements).
pub fn shape_size(shape: &[usize]) -> usize {
    shape.iter().copied().product()
}

/// Compute the flat-buffer offset from multi-dimensional indices and strides.
///
/// # Panics
///
/// Panics if `indices` and `strides` have different lengths.
pub fn offset_from_indices(indices: &[usize], strides: &[isize]) -> usize {
    assert_eq!(
        indices.len(),
        strides.len(),
        "indices and strides must have the same length"
    );
    let mut offset: isize = 0;
    for (&idx, &stride) in indices.iter().zip(strides.iter()) {
        offset += idx as isize * stride;
    }
    offset as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Order tests ---

    #[test]
    fn test_order_aliases() {
        assert_eq!(Order::C, Order::RowMajor);
        assert_eq!(Order::F, Order::ColumnMajor);
    }

    #[test]
    fn test_order_predicates() {
        assert!(Order::RowMajor.is_row_major());
        assert!(!Order::RowMajor.is_column_major());
        assert!(Order::ColumnMajor.is_column_major());
        assert!(!Order::ColumnMajor.is_row_major());
    }

    // --- C stride computation ---

    #[test]
    fn test_c_strides_1d() {
        assert_eq!(compute_c_strides(&[5]), vec![1]);
    }

    #[test]
    fn test_c_strides_2d() {
        assert_eq!(compute_c_strides(&[3, 4]), vec![4, 1]);
    }

    #[test]
    fn test_c_strides_3d() {
        assert_eq!(compute_c_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn test_c_strides_empty() {
        let empty: Vec<isize> = Vec::new();
        assert_eq!(compute_c_strides(&[]), empty);
    }

    // --- F stride computation ---

    #[test]
    fn test_f_strides_1d() {
        assert_eq!(compute_f_strides(&[5]), vec![1]);
    }

    #[test]
    fn test_f_strides_2d() {
        assert_eq!(compute_f_strides(&[3, 4]), vec![1, 3]);
    }

    #[test]
    fn test_f_strides_3d() {
        assert_eq!(compute_f_strides(&[2, 3, 4]), vec![1, 2, 6]);
    }

    #[test]
    fn test_f_strides_empty() {
        let empty: Vec<isize> = Vec::new();
        assert_eq!(compute_f_strides(&[]), empty);
    }

    // --- Contiguity checks ---

    #[test]
    fn test_c_contiguous_positive() {
        let shape = [2, 3, 4];
        let strides = compute_c_strides(&shape);
        assert!(is_c_contiguous(&shape, &strides));
    }

    #[test]
    fn test_f_contiguous_positive() {
        let shape = [2, 3, 4];
        let strides = compute_f_strides(&shape);
        assert!(is_f_contiguous(&shape, &strides));
    }

    #[test]
    fn test_c_contiguous_negative() {
        let shape = [2, 3, 4];
        let strides = compute_f_strides(&shape); // F strides are not C contiguous
        assert!(!is_c_contiguous(&shape, &strides));
    }

    #[test]
    fn test_f_contiguous_negative() {
        let shape = [2, 3, 4];
        let strides = compute_c_strides(&shape);
        assert!(!is_f_contiguous(&shape, &strides));
    }

    #[test]
    fn test_contiguous_size_one_dims() {
        // Shape with size-1 dims: any stride is fine for those axes.
        let shape = [1, 3, 1, 4];
        let c_strides = compute_c_strides(&shape);
        assert!(is_c_contiguous(&shape, &c_strides));

        // Modify the stride for size-1 dim -- should still be contiguous.
        let custom = vec![999, 4, 999, 1];
        assert!(is_c_contiguous(&shape, &custom));
    }

    #[test]
    fn test_1d_is_both_c_and_f() {
        let shape = [5];
        let strides = vec![1isize];
        assert!(is_c_contiguous(&shape, &strides));
        assert!(is_f_contiguous(&shape, &strides));
    }

    #[test]
    fn test_scalar_is_both_c_and_f() {
        assert!(is_c_contiguous(&[], &[]));
        assert!(is_f_contiguous(&[], &[]));
    }

    #[test]
    fn test_mismatched_lengths() {
        assert!(!is_c_contiguous(&[2, 3], &[1]));
        assert!(!is_f_contiguous(&[2, 3], &[1]));
    }

    // --- LayoutFlags ---

    #[test]
    fn test_layout_flags_c() {
        let shape = [2, 3];
        let strides = compute_c_strides(&shape);
        let flags = LayoutFlags::from_shape_strides(&shape, &strides);
        assert!(flags.is_c_contiguous());
        assert!(!flags.is_f_contiguous());
        assert!(flags.is_contiguous());
    }

    #[test]
    fn test_layout_flags_f() {
        let shape = [2, 3];
        let strides = compute_f_strides(&shape);
        let flags = LayoutFlags::from_shape_strides(&shape, &strides);
        assert!(!flags.is_c_contiguous());
        assert!(flags.is_f_contiguous());
        assert!(flags.is_contiguous());
    }

    // --- shape_size ---

    #[test]
    fn test_shape_size() {
        assert_eq!(shape_size(&[2, 3, 4]), 24);
        assert_eq!(shape_size(&[1]), 1);
        assert_eq!(shape_size(&[]), 1); // product of empty = 1
    }

    // --- offset_from_indices ---

    #[test]
    fn test_offset_c_order() {
        let shape = [2, 3, 4];
        let strides = compute_c_strides(&shape);
        // Element [1, 2, 3] in C-order: 1*12 + 2*4 + 3*1 = 23
        assert_eq!(offset_from_indices(&[1, 2, 3], &strides), 23);
        // Element [0, 0, 0]
        assert_eq!(offset_from_indices(&[0, 0, 0], &strides), 0);
    }

    #[test]
    fn test_offset_f_order() {
        let shape = [2, 3, 4];
        let strides = compute_f_strides(&shape);
        // Element [1, 2, 3] in F-order: 1*1 + 2*2 + 3*6 = 23
        assert_eq!(offset_from_indices(&[1, 2, 3], &strides), 23);
        assert_eq!(offset_from_indices(&[0, 0, 0], &strides), 0);
    }

    #[test]
    fn test_offset_2d() {
        let strides = compute_c_strides(&[3, 4]);
        // [2, 3] -> 2*4 + 3 = 11
        assert_eq!(offset_from_indices(&[2, 3], &strides), 11);
    }

    #[test]
    #[should_panic(expected = "indices and strides must have the same length")]
    fn test_offset_mismatched_lengths() {
        offset_from_indices(&[1, 2], &[1]);
    }
}
