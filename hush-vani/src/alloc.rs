//! 64-byte aligned f32 buffer. Vec<f32> only guarantees 16 bytes on Windows, which makes
//! half of every 32-byte AVX2 load straddle a cache line (~30% slower, measured).

use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::ops::{Deref, DerefMut};

pub const ALIGN: usize = 64;

pub struct AlignedVec {
    ptr: *mut f32,
    len: usize,
}

unsafe impl Send for AlignedVec {}
unsafe impl Sync for AlignedVec {}

impl AlignedVec {
    pub fn zeros(len: usize) -> Self {
        assert!(len > 0);
        let layout = Layout::from_size_align(len * 4, ALIGN).unwrap();
        let ptr = unsafe { alloc_zeroed(layout) as *mut f32 };
        assert!(!ptr.is_null(), "allocation failed");
        AlignedVec { ptr, len }
    }

    pub fn from_slice(s: &[f32]) -> Self {
        let mut v = Self::zeros(s.len());
        v.copy_from_slice(s);
        v
    }
}

impl Drop for AlignedVec {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr as *mut u8, Layout::from_size_align(self.len * 4, ALIGN).unwrap()) }
    }
}

impl Deref for AlignedVec {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl DerefMut for AlignedVec {
    fn deref_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

#[inline(always)]
pub fn is_aligned32(p: *const f32) -> bool {
    (p as usize) % 32 == 0
}
