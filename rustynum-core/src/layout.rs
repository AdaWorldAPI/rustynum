//! CBLAS-style layout and transpose enumerations.
//!
//! Both row-major and column-major layouts are supported throughout the
//! rustynum ecosystem. This matches the CBLAS API convention.

/// Memory layout for matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Layout {
    /// Row-major (C-style): elements in a row are contiguous.
    RowMajor = 101,
    /// Column-major (Fortran-style): elements in a column are contiguous.
    ColMajor = 102,
}

impl Default for Layout {
    fn default() -> Self {
        Self::RowMajor
    }
}

/// Transpose operation for matrices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Transpose {
    /// No transpose.
    NoTrans = 111,
    /// Transpose.
    Trans = 112,
    /// Conjugate transpose (for complex types).
    ConjTrans = 113,
}

impl Default for Transpose {
    fn default() -> Self {
        Self::NoTrans
    }
}

impl Layout {
    /// Leading dimension stride for an M x N matrix.
    #[inline(always)]
    pub fn leading_dim(self, rows: usize, cols: usize) -> usize {
        match self {
            Layout::RowMajor => cols,
            Layout::ColMajor => rows,
        }
    }

    /// Linear index into a flat array for element (i, j) of an M x N matrix.
    #[inline(always)]
    pub fn index(self, i: usize, j: usize, ld: usize) -> usize {
        match self {
            Layout::RowMajor => i * ld + j,
            Layout::ColMajor => j * ld + i,
        }
    }
}

/// BLAS triangle specifier (upper/lower).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Uplo {
    Upper = 121,
    Lower = 122,
}

impl Default for Uplo {
    fn default() -> Self {
        Self::Upper
    }
}

/// BLAS side specifier (left/right multiplication).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Side {
    Left = 141,
    Right = 142,
}

impl Default for Side {
    fn default() -> Self {
        Self::Left
    }
}

/// BLAS diagonal specifier (unit/non-unit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Diag {
    NonUnit = 131,
    Unit = 132,
}

impl Default for Diag {
    fn default() -> Self {
        Self::NonUnit
    }
}
