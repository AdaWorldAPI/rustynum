//! Compile-time and runtime dimension types for N-dimensional arrays.

use std::fmt;
use std::ops::{Index, IndexMut};

// ---------------------------------------------------------------------------
// Axis
// ---------------------------------------------------------------------------

/// A newtype representing an array axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Axis(pub usize);

impl Axis {
    /// Return the index of this axis.
    #[inline]
    pub fn index(&self) -> usize {
        self.0
    }
}

impl fmt::Display for Axis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Axis({})", self.0)
    }
}

// ---------------------------------------------------------------------------
// Dimension trait
// ---------------------------------------------------------------------------

/// The core trait implemented by all dimension types.
pub trait Dimension: Clone + fmt::Debug + PartialEq + Eq + Send + Sync {
    /// `Some(n)` for fixed-size dimensions, `None` for dynamic.
    const NDIM: Option<usize>;

    /// Number of axes (dimensions).
    fn ndim(&self) -> usize;

    /// Total number of elements (product of all dimension sizes).
    fn size(&self) -> usize {
        self.as_slice().iter().product()
    }

    /// View the dimension sizes as a slice.
    fn as_slice(&self) -> &[usize];

    /// View the dimension sizes as a mutable slice.
    fn as_slice_mut(&mut self) -> &mut [usize];

    /// Create a dimension of the given rank with all zeros.
    fn zeros(ndim: usize) -> Self;

    /// Compute default (C-order / row-major) strides.
    fn default_strides(&self) -> Self {
        let shape = self.as_slice();
        let ndim = shape.len();
        let mut strides = Self::zeros(ndim);
        {
            let s = strides.as_slice_mut();
            if ndim > 0 {
                s[ndim - 1] = 1;
                for i in (0..ndim - 1).rev() {
                    s[i] = s[i + 1] * shape[i + 1];
                }
            }
        }
        strides
    }

    /// Compute Fortran-order (column-major) strides.
    fn fortran_strides(&self) -> Self {
        let shape = self.as_slice();
        let ndim = shape.len();
        let mut strides = Self::zeros(ndim);
        {
            let s = strides.as_slice_mut();
            if ndim > 0 {
                s[0] = 1;
                for i in 1..ndim {
                    s[i] = s[i - 1] * shape[i - 1];
                }
            }
        }
        strides
    }
}

// ---------------------------------------------------------------------------
// IntoDimension trait
// ---------------------------------------------------------------------------

/// Conversion trait for types that can be turned into a `Dimension`.
pub trait IntoDimension {
    type Dim: Dimension;
    fn into_dimension(self) -> Self::Dim;
}

// ---------------------------------------------------------------------------
// Fixed-size dimension types: Ix1 through Ix6
// ---------------------------------------------------------------------------

macro_rules! impl_fixed_dimension {
    ($name:ident, $n:expr) => {
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub struct $name(pub [usize; $n]);

        impl $name {
            /// Create from a raw array.
            #[inline]
            pub fn new(dims: [usize; $n]) -> Self {
                Self(dims)
            }
        }

        impl Dimension for $name {
            const NDIM: Option<usize> = Some($n);

            #[inline]
            fn ndim(&self) -> usize {
                $n
            }

            #[inline]
            fn as_slice(&self) -> &[usize] {
                &self.0
            }

            #[inline]
            fn as_slice_mut(&mut self) -> &mut [usize] {
                &mut self.0
            }

            #[inline]
            fn zeros(_ndim: usize) -> Self {
                Self([0; $n])
            }
        }

        impl From<[usize; $n]> for $name {
            #[inline]
            fn from(arr: [usize; $n]) -> Self {
                Self(arr)
            }
        }

        impl Index<usize> for $name {
            type Output = usize;
            #[inline]
            fn index(&self, idx: usize) -> &usize {
                &self.0[idx]
            }
        }

        impl IndexMut<usize> for $name {
            #[inline]
            fn index_mut(&mut self, idx: usize) -> &mut usize {
                &mut self.0[idx]
            }
        }

        impl IntoDimension for [usize; $n] {
            type Dim = $name;
            #[inline]
            fn into_dimension(self) -> $name {
                $name(self)
            }
        }
    };
}

impl_fixed_dimension!(Ix1, 1);
impl_fixed_dimension!(Ix2, 2);
impl_fixed_dimension!(Ix3, 3);
impl_fixed_dimension!(Ix4, 4);
impl_fixed_dimension!(Ix5, 5);
impl_fixed_dimension!(Ix6, 6);

// ---------------------------------------------------------------------------
// Tuple conversions for fixed-size dimensions
// ---------------------------------------------------------------------------

impl From<(usize,)> for Ix1 {
    #[inline]
    fn from(t: (usize,)) -> Self {
        Self([t.0])
    }
}

impl IntoDimension for (usize,) {
    type Dim = Ix1;
    #[inline]
    fn into_dimension(self) -> Ix1 {
        Ix1([self.0])
    }
}

impl From<(usize, usize)> for Ix2 {
    #[inline]
    fn from(t: (usize, usize)) -> Self {
        Self([t.0, t.1])
    }
}

impl IntoDimension for (usize, usize) {
    type Dim = Ix2;
    #[inline]
    fn into_dimension(self) -> Ix2 {
        Ix2([self.0, self.1])
    }
}

impl From<(usize, usize, usize)> for Ix3 {
    #[inline]
    fn from(t: (usize, usize, usize)) -> Self {
        Self([t.0, t.1, t.2])
    }
}

impl IntoDimension for (usize, usize, usize) {
    type Dim = Ix3;
    #[inline]
    fn into_dimension(self) -> Ix3 {
        Ix3([self.0, self.1, self.2])
    }
}

impl From<(usize, usize, usize, usize)> for Ix4 {
    #[inline]
    fn from(t: (usize, usize, usize, usize)) -> Self {
        Self([t.0, t.1, t.2, t.3])
    }
}

impl IntoDimension for (usize, usize, usize, usize) {
    type Dim = Ix4;
    #[inline]
    fn into_dimension(self) -> Ix4 {
        Ix4([self.0, self.1, self.2, self.3])
    }
}

impl From<(usize, usize, usize, usize, usize)> for Ix5 {
    #[inline]
    fn from(t: (usize, usize, usize, usize, usize)) -> Self {
        Self([t.0, t.1, t.2, t.3, t.4])
    }
}

impl IntoDimension for (usize, usize, usize, usize, usize) {
    type Dim = Ix5;
    #[inline]
    fn into_dimension(self) -> Ix5 {
        Ix5([self.0, self.1, self.2, self.3, self.4])
    }
}

impl From<(usize, usize, usize, usize, usize, usize)> for Ix6 {
    #[inline]
    fn from(t: (usize, usize, usize, usize, usize, usize)) -> Self {
        Self([t.0, t.1, t.2, t.3, t.4, t.5])
    }
}

impl IntoDimension for (usize, usize, usize, usize, usize, usize) {
    type Dim = Ix6;
    #[inline]
    fn into_dimension(self) -> Ix6 {
        Ix6([self.0, self.1, self.2, self.3, self.4, self.5])
    }
}

// ---------------------------------------------------------------------------
// IxDyn — dynamic dimension
// ---------------------------------------------------------------------------

/// A dynamic-rank dimension type backed by `Vec<usize>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IxDyn(pub Vec<usize>);

impl IxDyn {
    /// Create from a slice.
    #[inline]
    pub fn new(dims: &[usize]) -> Self {
        Self(dims.to_vec())
    }
}

impl Dimension for IxDyn {
    const NDIM: Option<usize> = None;

    #[inline]
    fn ndim(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn as_slice(&self) -> &[usize] {
        &self.0
    }

    #[inline]
    fn as_slice_mut(&mut self) -> &mut [usize] {
        &mut self.0
    }

    #[inline]
    fn zeros(ndim: usize) -> Self {
        Self(vec![0; ndim])
    }
}

impl From<Vec<usize>> for IxDyn {
    #[inline]
    fn from(v: Vec<usize>) -> Self {
        Self(v)
    }
}

impl From<&[usize]> for IxDyn {
    #[inline]
    fn from(s: &[usize]) -> Self {
        Self(s.to_vec())
    }
}

impl Index<usize> for IxDyn {
    type Output = usize;
    #[inline]
    fn index(&self, idx: usize) -> &usize {
        &self.0[idx]
    }
}

impl IndexMut<usize> for IxDyn {
    #[inline]
    fn index_mut(&mut self, idx: usize) -> &mut usize {
        &mut self.0[idx]
    }
}

impl IntoDimension for Vec<usize> {
    type Dim = IxDyn;
    #[inline]
    fn into_dimension(self) -> IxDyn {
        IxDyn(self)
    }
}

impl IntoDimension for &[usize] {
    type Dim = IxDyn;
    #[inline]
    fn into_dimension(self) -> IxDyn {
        IxDyn(self.to_vec())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_axis() {
        let a = Axis(2);
        assert_eq!(a.index(), 2);
        assert_eq!(format!("{}", a), "Axis(2)");
    }

    #[test]
    fn test_ix1() {
        let d = Ix1::from([5]);
        assert_eq!(d.ndim(), 1);
        assert_eq!(d.size(), 5);
        assert_eq!(d.as_slice(), &[5]);
        assert_eq!(d[0], 5);
    }

    #[test]
    fn test_ix2_from_tuple() {
        let d = Ix2::from((3, 4));
        assert_eq!(d.ndim(), 2);
        assert_eq!(d.size(), 12);
        assert_eq!(d.as_slice(), &[3, 4]);
    }

    #[test]
    fn test_ix3() {
        let d = Ix3::from([2, 3, 4]);
        assert_eq!(d.ndim(), 3);
        assert_eq!(d.size(), 24);
    }

    #[test]
    fn test_default_strides_2d() {
        let shape = Ix2::from([3, 4]);
        let strides = shape.default_strides();
        // C-order: row-major, last dim stride=1
        assert_eq!(strides.as_slice(), &[4, 1]);
    }

    #[test]
    fn test_default_strides_3d() {
        let shape = Ix3::from([2, 3, 4]);
        let strides = shape.default_strides();
        // C-order strides: [12, 4, 1]
        assert_eq!(strides.as_slice(), &[12, 4, 1]);
    }

    #[test]
    fn test_fortran_strides_2d() {
        let shape = Ix2::from([3, 4]);
        let strides = shape.fortran_strides();
        // F-order: column-major, first dim stride=1
        assert_eq!(strides.as_slice(), &[1, 3]);
    }

    #[test]
    fn test_fortran_strides_3d() {
        let shape = Ix3::from([2, 3, 4]);
        let strides = shape.fortran_strides();
        // F-order strides: [1, 2, 6]
        assert_eq!(strides.as_slice(), &[1, 2, 6]);
    }

    #[test]
    fn test_ixdyn() {
        let d = IxDyn::from(vec![2, 3, 4]);
        assert_eq!(d.ndim(), 3);
        assert_eq!(d.size(), 24);
        assert_eq!(d[0], 2);
        assert_eq!(d[1], 3);
        assert_eq!(d[2], 4);
    }

    #[test]
    fn test_ixdyn_from_slice() {
        let s: &[usize] = &[5, 6];
        let d = IxDyn::from(s);
        assert_eq!(d.ndim(), 2);
        assert_eq!(d.size(), 30);
    }

    #[test]
    fn test_ixdyn_strides() {
        let d = IxDyn::from(vec![2, 3, 4]);
        let strides = d.default_strides();
        assert_eq!(strides.as_slice(), &[12, 4, 1]);

        let fstrides = d.fortran_strides();
        assert_eq!(fstrides.as_slice(), &[1, 2, 6]);
    }

    #[test]
    fn test_into_dimension_array() {
        let d: Ix3 = [2, 3, 4].into_dimension();
        assert_eq!(d.size(), 24);
    }

    #[test]
    fn test_into_dimension_tuple() {
        let d: Ix2 = (3usize, 4usize).into_dimension();
        assert_eq!(d.size(), 12);
    }

    #[test]
    fn test_into_dimension_vec() {
        let d: IxDyn = vec![2, 3].into_dimension();
        assert_eq!(d.size(), 6);
    }

    #[test]
    fn test_index_mut() {
        let mut d = Ix2::from([3, 4]);
        d[0] = 5;
        assert_eq!(d[0], 5);
        assert_eq!(d.size(), 20);
    }

    #[test]
    fn test_zeros() {
        let d = Ix3::zeros(3);
        assert_eq!(d.as_slice(), &[0, 0, 0]);
        assert_eq!(d.size(), 0);
    }

    #[test]
    fn test_ixdyn_zeros() {
        let d = IxDyn::zeros(4);
        assert_eq!(d.ndim(), 4);
        assert_eq!(d.as_slice(), &[0, 0, 0, 0]);
    }

    #[test]
    fn test_ndim_const() {
        assert_eq!(Ix1::NDIM, Some(1));
        assert_eq!(Ix2::NDIM, Some(2));
        assert_eq!(Ix3::NDIM, Some(3));
        assert_eq!(Ix4::NDIM, Some(4));
        assert_eq!(Ix5::NDIM, Some(5));
        assert_eq!(Ix6::NDIM, Some(6));
        assert_eq!(IxDyn::NDIM, None);
    }

    #[test]
    fn test_empty_dimension() {
        let d = IxDyn::from(vec![]);
        assert_eq!(d.ndim(), 0);
        // Product of empty slice is 1 (identity for multiplication)
        assert_eq!(d.size(), 1);
    }

    #[test]
    fn test_default_strides_1d() {
        let shape = Ix1::from([5]);
        let strides = shape.default_strides();
        assert_eq!(strides.as_slice(), &[1]);
    }

    #[test]
    fn test_ix6_from_tuple() {
        let d = Ix6::from((1, 2, 3, 4, 5, 6));
        assert_eq!(d.ndim(), 6);
        assert_eq!(d.size(), 720);
    }

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Ix1>();
        assert_send_sync::<Ix2>();
        assert_send_sync::<Ix3>();
        assert_send_sync::<IxDyn>();
        assert_send_sync::<Axis>();
    }
}
