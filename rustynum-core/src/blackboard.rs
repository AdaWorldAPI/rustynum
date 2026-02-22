//! Zero-copy blackboard: a shared mutable memory arena with SIMD-aligned allocations.
//!
//! The blackboard eliminates serialization between crates. All three crates (rustynum-rs,
//! rustyblas, rustymkl) can allocate buffers from the same arena and operate on them
//! directly — no copies, no marshalling.
//!
//! # Design
//!
//! - 64-byte aligned allocations (AVX-512 cache-line aligned)
//! - Named buffers for clarity (`"A"`, `"B"`, `"C"` for GEMM operands)
//! - Split-borrow API: multiple buffers can be mutably borrowed simultaneously
//!   as long as they don't alias (like struct field borrows)
//! - Sound interior mutability via `UnsafeCell` + disjoint-buffer invariant
//!
//! # Example
//!
//! ```
//! use rustynum_core::Blackboard;
//!
//! let mut bb = Blackboard::new();
//!
//! // Allocate GEMM operands
//! let a = bb.alloc_f32("A", 1024 * 1024);
//! let b = bb.alloc_f32("B", 1024 * 1024);
//! let c = bb.alloc_f32("C", 1024 * 1024);
//!
//! // Get non-overlapping mutable slices — no borrow conflicts
//! let (a_slice, b_slice, c_slice) = bb.borrow_3_mut_f32("A", "B", "C");
//!
//! // Fill A and B, compute into C — all zero-copy
//! a_slice.fill(1.0);
//! b_slice.fill(2.0);
//! // ... rustyblas::sgemm operates directly on these slices
//! ```

use std::alloc;
use std::cell::UnsafeCell;
use std::collections::HashMap;

/// Alignment for all blackboard allocations (AVX-512 = 64 bytes).
const ALIGNMENT: usize = 64;

/// Opaque handle to a buffer in the blackboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BufferHandle(u32);

/// Type tag for buffer element types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    U8,
}

impl Default for DType {
    fn default() -> Self {
        Self::F32
    }
}

impl DType {
    fn element_size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::U8 => 1,
        }
    }
}

/// Metadata for a single buffer allocation.
struct BufferMeta {
    ptr: *mut u8,
    len_elements: usize,
    dtype: DType,
    layout: alloc::Layout,
}

impl Drop for BufferMeta {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.layout.size() > 0 {
            unsafe { alloc::dealloc(self.ptr, self.layout) };
        }
    }
}

/// Zero-copy shared memory arena with SIMD-aligned allocations.
///
/// The blackboard owns all buffer memory. Crates borrow slices directly
/// from the arena — no serialization, no copies.
///
/// # Safety model
///
/// Each named buffer is a separate heap allocation at a distinct address.
/// `UnsafeCell` wrapping opts into interior mutability, making the
/// `&self` → `&mut [T]` pattern sound under Rust's stacked borrows model.
/// Runtime assertions prevent borrowing the same buffer twice.
pub struct Blackboard {
    buffers: HashMap<String, UnsafeCell<BufferMeta>>,
    next_handle: u32,
    handles: HashMap<String, BufferHandle>,
}

impl Blackboard {
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            next_handle: 0,
            handles: HashMap::new(),
        }
    }

    /// Allocate a 64-byte-aligned f32 buffer with `len` elements.
    /// Returns a handle for later retrieval.
    pub fn alloc_f32(&mut self, name: &str, len: usize) -> BufferHandle {
        self.alloc_typed(name, len, DType::F32)
    }

    /// Allocate a 64-byte-aligned f64 buffer with `len` elements.
    pub fn alloc_f64(&mut self, name: &str, len: usize) -> BufferHandle {
        self.alloc_typed(name, len, DType::F64)
    }

    /// Allocate a 64-byte-aligned i32 buffer with `len` elements.
    pub fn alloc_i32(&mut self, name: &str, len: usize) -> BufferHandle {
        self.alloc_typed(name, len, DType::I32)
    }

    /// Allocate a 64-byte-aligned i64 buffer with `len` elements.
    pub fn alloc_i64(&mut self, name: &str, len: usize) -> BufferHandle {
        self.alloc_typed(name, len, DType::I64)
    }

    /// Allocate a 64-byte-aligned u8 buffer with `len` elements.
    pub fn alloc_u8(&mut self, name: &str, len: usize) -> BufferHandle {
        self.alloc_typed(name, len, DType::U8)
    }

    fn alloc_typed(&mut self, name: &str, len: usize, dtype: DType) -> BufferHandle {
        // Deallocate existing buffer with the same name if present
        self.buffers.remove(name);

        let byte_len = len * dtype.element_size();
        let layout = alloc::Layout::from_size_align(byte_len.max(1), ALIGNMENT)
            .expect("Invalid layout");

        let ptr = if byte_len == 0 {
            std::ptr::null_mut()
        } else {
            let p = unsafe { alloc::alloc_zeroed(layout) };
            if p.is_null() {
                alloc::handle_alloc_error(layout);
            }
            p
        };

        let handle = BufferHandle(self.next_handle);
        self.next_handle += 1;

        self.handles.insert(name.to_string(), handle);
        self.buffers.insert(
            name.to_string(),
            UnsafeCell::new(BufferMeta {
                ptr,
                len_elements: len,
                dtype,
                layout,
            }),
        );

        handle
    }

    /// Get an immutable f32 slice for the named buffer.
    pub fn get_f32(&self, name: &str) -> &[f32] {
        let cell = self.buffers.get(name).expect("Buffer not found");
        // Safety: We only create a shared reference to read metadata and data.
        let meta = unsafe { &*cell.get() };
        assert_eq!(meta.dtype, DType::F32, "Buffer is not f32");
        if meta.len_elements == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(meta.ptr as *const f32, meta.len_elements) }
    }

    /// Get a mutable f32 slice for the named buffer.
    pub fn get_f32_mut(&mut self, name: &str) -> &mut [f32] {
        let cell = self.buffers.get(name).expect("Buffer not found");
        // Safety: &mut self guarantees exclusive access to the entire Blackboard.
        let meta = unsafe { &*cell.get() };
        assert_eq!(meta.dtype, DType::F32, "Buffer is not f32");
        if meta.len_elements == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(meta.ptr as *mut f32, meta.len_elements) }
    }

    /// Get an immutable f64 slice for the named buffer.
    pub fn get_f64(&self, name: &str) -> &[f64] {
        let cell = self.buffers.get(name).expect("Buffer not found");
        let meta = unsafe { &*cell.get() };
        assert_eq!(meta.dtype, DType::F64, "Buffer is not f64");
        if meta.len_elements == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(meta.ptr as *const f64, meta.len_elements) }
    }

    /// Get a mutable f64 slice for the named buffer.
    pub fn get_f64_mut(&mut self, name: &str) -> &mut [f64] {
        let cell = self.buffers.get(name).expect("Buffer not found");
        let meta = unsafe { &*cell.get() };
        assert_eq!(meta.dtype, DType::F64, "Buffer is not f64");
        if meta.len_elements == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(meta.ptr as *mut f64, meta.len_elements) }
    }

    /// Get an immutable u8 slice for the named buffer.
    pub fn get_u8(&self, name: &str) -> &[u8] {
        let cell = self.buffers.get(name).expect("Buffer not found");
        let meta = unsafe { &*cell.get() };
        assert_eq!(meta.dtype, DType::U8, "Buffer is not u8");
        if meta.len_elements == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(meta.ptr as *const u8, meta.len_elements) }
    }

    /// Get a mutable u8 slice for the named buffer.
    pub fn get_u8_mut(&mut self, name: &str) -> &mut [u8] {
        let cell = self.buffers.get(name).expect("Buffer not found");
        let meta = unsafe { &*cell.get() };
        assert_eq!(meta.dtype, DType::U8, "Buffer is not u8");
        if meta.len_elements == 0 {
            return &mut [];
        }
        unsafe { std::slice::from_raw_parts_mut(meta.ptr as *mut u8, meta.len_elements) }
    }

    /// Split-borrow: get 2 non-overlapping mutable f32 slices simultaneously.
    ///
    /// # Safety invariant
    ///
    /// Sound because: (1) `UnsafeCell` opts into interior mutability under stacked
    /// borrows, (2) the `assert_ne!` prevents aliasing the same buffer, and
    /// (3) different named buffers are different heap allocations that cannot overlap.
    ///
    /// # Panics
    ///
    /// Panics if names are the same or buffers don't exist.
    pub fn borrow_2_mut_f32<'a>(&'a mut self, a: &str, b: &str) -> (&'a mut [f32], &'a mut [f32]) {
        assert_ne!(a, b, "Cannot borrow the same buffer twice mutably");
        let ca = self.buffers.get(a).expect("Buffer A not found");
        let cb = self.buffers.get(b).expect("Buffer B not found");
        // Safety: UnsafeCell allows &self → &mut for interior data.
        // The assert_ne! above guarantees these are different buffers,
        // which are independent heap allocations (disjoint memory).
        unsafe {
            let ma = &*ca.get();
            let mb = &*cb.get();
            assert_eq!(ma.dtype, DType::F32);
            assert_eq!(mb.dtype, DType::F32);
            (
                std::slice::from_raw_parts_mut(ma.ptr as *mut f32, ma.len_elements),
                std::slice::from_raw_parts_mut(mb.ptr as *mut f32, mb.len_elements),
            )
        }
    }

    /// Split-borrow: get 3 non-overlapping mutable f32 slices simultaneously.
    /// This is the key pattern for GEMM: A, B, C all mutable at once.
    ///
    /// # Safety invariant
    ///
    /// Same as `borrow_2_mut_f32` — `UnsafeCell` + distinct names + separate allocations.
    ///
    /// # Panics
    ///
    /// Panics if any names are the same or buffers don't exist.
    pub fn borrow_3_mut_f32<'a>(
        &'a mut self,
        a: &str,
        b: &str,
        c: &str,
    ) -> (&'a mut [f32], &'a mut [f32], &'a mut [f32]) {
        assert!(a != b && b != c && a != c, "Buffer names must be distinct");
        let ca = self.buffers.get(a).expect("Buffer A not found");
        let cb = self.buffers.get(b).expect("Buffer B not found");
        let cc = self.buffers.get(c).expect("Buffer C not found");
        unsafe {
            let ma = &*ca.get();
            let mb = &*cb.get();
            let mc = &*cc.get();
            assert_eq!(ma.dtype, DType::F32);
            assert_eq!(mb.dtype, DType::F32);
            assert_eq!(mc.dtype, DType::F32);
            (
                std::slice::from_raw_parts_mut(ma.ptr as *mut f32, ma.len_elements),
                std::slice::from_raw_parts_mut(mb.ptr as *mut f32, mb.len_elements),
                std::slice::from_raw_parts_mut(mc.ptr as *mut f32, mc.len_elements),
            )
        }
    }

    /// Split-borrow: get 3 non-overlapping mutable f64 slices simultaneously.
    ///
    /// # Panics
    ///
    /// Panics if any names are the same or buffers don't exist.
    pub fn borrow_3_mut_f64<'a>(
        &'a mut self,
        a: &str,
        b: &str,
        c: &str,
    ) -> (&'a mut [f64], &'a mut [f64], &'a mut [f64]) {
        assert!(a != b && b != c && a != c, "Buffer names must be distinct");
        let ca = self.buffers.get(a).expect("Buffer A not found");
        let cb = self.buffers.get(b).expect("Buffer B not found");
        let cc = self.buffers.get(c).expect("Buffer C not found");
        unsafe {
            let ma = &*ca.get();
            let mb = &*cb.get();
            let mc = &*cc.get();
            assert_eq!(ma.dtype, DType::F64);
            assert_eq!(mb.dtype, DType::F64);
            assert_eq!(mc.dtype, DType::F64);
            (
                std::slice::from_raw_parts_mut(ma.ptr as *mut f64, ma.len_elements),
                std::slice::from_raw_parts_mut(mb.ptr as *mut f64, mb.len_elements),
                std::slice::from_raw_parts_mut(mc.ptr as *mut f64, mc.len_elements),
            )
        }
    }

    /// Get the raw pointer and length for a named buffer (for FFI or advanced usage).
    pub fn raw_ptr(&self, name: &str) -> (*mut u8, usize, DType) {
        let cell = self.buffers.get(name).expect("Buffer not found");
        let meta = unsafe { &*cell.get() };
        (meta.ptr, meta.len_elements, meta.dtype)
    }

    /// Returns the number of elements in a named buffer.
    pub fn len(&self, name: &str) -> usize {
        let cell = self.buffers.get(name).expect("Buffer not found");
        let meta = unsafe { &*cell.get() };
        meta.len_elements
    }

    /// Check if a named buffer exists.
    pub fn contains(&self, name: &str) -> bool {
        self.buffers.contains_key(name)
    }

    /// Remove and deallocate a named buffer.
    pub fn free(&mut self, name: &str) {
        self.buffers.remove(name);
        self.handles.remove(name);
    }

    /// List all buffer names.
    pub fn buffer_names(&self) -> Vec<&str> {
        self.buffers.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for Blackboard {
    fn default() -> Self {
        Self::new()
    }
}

// Note: Blackboard is intentionally !Send and !Sync due to UnsafeCell.
// The split-borrow API (&mut self → multiple &mut slices) is sound for
// single-threaded use. For multi-threaded access, wrap in Mutex<Blackboard>.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_and_access_f32() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("test", 1024);
        let slice = bb.get_f32_mut("test");
        assert_eq!(slice.len(), 1024);
        // Should be zero-initialized
        assert!(slice.iter().all(|&x| x == 0.0));
        slice[0] = 42.0;
        assert_eq!(bb.get_f32("test")[0], 42.0);
    }

    #[test]
    fn test_alloc_and_access_f64() {
        let mut bb = Blackboard::new();
        bb.alloc_f64("test", 512);
        let slice = bb.get_f64_mut("test");
        assert_eq!(slice.len(), 512);
        slice[0] = 99.0;
        assert_eq!(bb.get_f64("test")[0], 99.0);
    }

    #[test]
    fn test_alignment() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("aligned", 256);
        let (ptr, _, _) = bb.raw_ptr("aligned");
        assert_eq!(ptr as usize % ALIGNMENT, 0, "Buffer not 64-byte aligned");
    }

    #[test]
    fn test_split_borrow_3() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("A", 16);
        bb.alloc_f32("B", 16);
        bb.alloc_f32("C", 16);

        let (a, b, c) = bb.borrow_3_mut_f32("A", "B", "C");
        a.fill(1.0);
        b.fill(2.0);
        c.fill(0.0);

        // Verify independent
        assert!(a.iter().all(|&x| x == 1.0));
        assert!(b.iter().all(|&x| x == 2.0));
        assert!(c.iter().all(|&x| x == 0.0));
    }

    #[test]
    #[should_panic(expected = "Buffer names must be distinct")]
    fn test_split_borrow_same_name_panics() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("A", 16);
        bb.borrow_3_mut_f32("A", "A", "B");
    }

    #[test]
    fn test_realloc_overwrites() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("buf", 8);
        bb.get_f32_mut("buf")[0] = 42.0;
        // Re-allocate with different size
        bb.alloc_f32("buf", 16);
        assert_eq!(bb.len("buf"), 16);
        // Should be zero-initialized again
        assert_eq!(bb.get_f32("buf")[0], 0.0);
    }

    #[test]
    fn test_free() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("temp", 64);
        assert!(bb.contains("temp"));
        bb.free("temp");
        assert!(!bb.contains("temp"));
    }

    #[test]
    fn test_split_borrow_2() {
        let mut bb = Blackboard::new();
        bb.alloc_f32("X", 8);
        bb.alloc_f32("Y", 8);

        let (x, y) = bb.borrow_2_mut_f32("X", "Y");
        x.fill(3.0);
        y.fill(7.0);

        assert!(x.iter().all(|&v| v == 3.0));
        assert!(y.iter().all(|&v| v == 7.0));
    }

    #[test]
    fn test_split_borrow_f64() {
        let mut bb = Blackboard::new();
        bb.alloc_f64("A", 8);
        bb.alloc_f64("B", 8);
        bb.alloc_f64("C", 8);

        let (a, b, c) = bb.borrow_3_mut_f64("A", "B", "C");
        a.fill(1.0);
        b.fill(2.0);
        c.fill(3.0);

        assert!(a.iter().all(|&v| v == 1.0));
        assert!(b.iter().all(|&v| v == 2.0));
        assert!(c.iter().all(|&v| v == 3.0));
    }
}
