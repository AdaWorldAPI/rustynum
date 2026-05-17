//! Storage abstractions for N-dimensional arrays.
//!
//! Provides owned data, immutable views, and mutable views with a unified trait interface.

use std::fmt::Debug;

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

/// Minimal raw access to the underlying data buffer.
pub trait RawStorage<T> {
    /// Returns a raw pointer to the start of the data.
    fn as_ptr(&self) -> *const T;

    /// Returns the number of elements in the storage.
    fn len(&self) -> usize;

    /// Returns `true` if the storage contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Safe immutable access to contiguous storage.
pub trait Storage<T>: RawStorage<T> {
    /// Returns the data as an immutable slice.
    fn as_slice(&self) -> &[T];
}

/// Safe mutable access to contiguous storage.
pub trait StorageMut<T>: Storage<T> {
    /// Returns the data as a mutable slice.
    fn as_slice_mut(&mut self) -> &mut [T];
}

// ---------------------------------------------------------------------------
// OwnedStorage
// ---------------------------------------------------------------------------

/// Vec-backed owned storage.
///
// TODO: Provide 64-byte aligned allocation (e.g. via a custom allocator) for
// SIMD-friendly access patterns. For now this uses the default global allocator.
#[derive(Clone)]
pub struct OwnedStorage<T> {
    data: Vec<T>,
}

impl<T> OwnedStorage<T> {
    /// Create storage from an existing `Vec<T>`.
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    /// Create storage with a given capacity but no elements.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            data: Vec::with_capacity(cap),
        }
    }

    /// Create storage of `len` elements initialized to the default value.
    pub fn zeros(len: usize) -> Self
    where
        T: Default + Clone,
    {
        Self {
            data: vec![T::default(); len],
        }
    }
}

impl<T: Debug> Debug for OwnedStorage<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OwnedStorage")
            .field("data", &self.data)
            .finish()
    }
}

impl<T> RawStorage<T> for OwnedStorage<T> {
    fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

impl<T> Storage<T> for OwnedStorage<T> {
    fn as_slice(&self) -> &[T] {
        &self.data
    }
}

impl<T> StorageMut<T> for OwnedStorage<T> {
    fn as_slice_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T> From<Vec<T>> for OwnedStorage<T> {
    fn from(data: Vec<T>) -> Self {
        Self::new(data)
    }
}

impl<T> From<OwnedStorage<T>> for Vec<T> {
    fn from(storage: OwnedStorage<T>) -> Self {
        storage.data
    }
}

// ---------------------------------------------------------------------------
// ViewStorage
// ---------------------------------------------------------------------------

/// Borrowed immutable view into a contiguous data buffer.
#[derive(Clone, Copy)]
pub struct ViewStorage<'a, T> {
    data: &'a [T],
}

impl<'a, T> ViewStorage<'a, T> {
    /// Create an immutable view over a slice.
    pub fn new(data: &'a [T]) -> Self {
        Self { data }
    }
}

impl<T: Debug> Debug for ViewStorage<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ViewStorage")
            .field("data", &self.data)
            .finish()
    }
}

impl<T> RawStorage<T> for ViewStorage<'_, T> {
    fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

impl<T> Storage<T> for ViewStorage<'_, T> {
    fn as_slice(&self) -> &[T] {
        self.data
    }
}

// ---------------------------------------------------------------------------
// ViewMutStorage
// ---------------------------------------------------------------------------

/// Borrowed mutable view into a contiguous data buffer.
pub struct ViewMutStorage<'a, T> {
    data: &'a mut [T],
}

impl<'a, T> ViewMutStorage<'a, T> {
    /// Create a mutable view over a slice.
    pub fn new(data: &'a mut [T]) -> Self {
        Self { data }
    }
}

impl<T: Debug> Debug for ViewMutStorage<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ViewMutStorage")
            .field("data", &self.data)
            .finish()
    }
}

impl<T> RawStorage<T> for ViewMutStorage<'_, T> {
    fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

impl<T> Storage<T> for ViewMutStorage<'_, T> {
    fn as_slice(&self) -> &[T] {
        self.data
    }
}

impl<T> StorageMut<T> for ViewMutStorage<'_, T> {
    fn as_slice_mut(&mut self) -> &mut [T] {
        self.data
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn owned_storage_basic() {
        let s = OwnedStorage::new(vec![1, 2, 3]);
        assert_eq!(s.len(), 3);
        assert!(!s.is_empty());
        assert_eq!(s.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn owned_storage_zeros() {
        let s: OwnedStorage<f64> = OwnedStorage::zeros(4);
        assert_eq!(s.as_slice(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn owned_storage_with_capacity() {
        let s: OwnedStorage<i32> = OwnedStorage::with_capacity(10);
        assert_eq!(s.len(), 0);
        assert!(s.is_empty());
    }

    #[test]
    fn owned_storage_mut() {
        let mut s = OwnedStorage::new(vec![1, 2, 3]);
        s.as_slice_mut()[0] = 42;
        assert_eq!(s.as_slice(), &[42, 2, 3]);
    }

    #[test]
    fn owned_storage_from_into_vec() {
        let v = vec![10, 20, 30];
        let s: OwnedStorage<i32> = v.into();
        assert_eq!(s.len(), 3);
        let v2: Vec<i32> = s.into();
        assert_eq!(v2, vec![10, 20, 30]);
    }

    #[test]
    fn owned_storage_clone() {
        let s = OwnedStorage::new(vec![1, 2, 3]);
        let s2 = s.clone();
        assert_eq!(s.as_slice(), s2.as_slice());
    }

    #[test]
    fn view_storage_basic() {
        let data = vec![5, 6, 7, 8];
        let v = ViewStorage::new(&data);
        assert_eq!(v.len(), 4);
        assert_eq!(v.as_slice(), &[5, 6, 7, 8]);
    }

    #[test]
    fn view_storage_copy() {
        let data = vec![1, 2, 3];
        let v1 = ViewStorage::new(&data);
        let v2 = v1; // Copy
        assert_eq!(v1.as_slice(), v2.as_slice());
    }

    #[test]
    fn view_mut_storage_basic() {
        let mut data = vec![1, 2, 3];
        let mut v = ViewMutStorage::new(&mut data);
        v.as_slice_mut()[1] = 99;
        assert_eq!(v.as_slice(), &[1, 99, 3]);
    }

    #[test]
    fn view_storage_empty() {
        let empty: &[u8] = &[];
        let v = ViewStorage::new(empty);
        assert!(v.is_empty());
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn raw_storage_as_ptr() {
        let data = vec![10u32, 20, 30];
        let s = OwnedStorage::new(data);
        let ptr = s.as_ptr();
        // Safety: we know the pointer is valid for the lifetime of s
        assert_eq!(unsafe { *ptr }, 10);
        assert_eq!(unsafe { *ptr.add(1) }, 20);
    }
}
