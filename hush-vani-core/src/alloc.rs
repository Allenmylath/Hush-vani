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

/// 64-byte aligned `u16` buffer, holding IEEE binary16 (f16) bit patterns.
///
/// The hot weight matrices live here: halving the bytes streamed per FMA is a real speedup
/// for the GRU's recurrent matvec, which is bandwidth-bound rather than issue-bound.
/// `_mm256_cvtph_ps` (F16C) converts 8 lanes to f32 in one instruction on the way in.
pub struct AlignedU16 {
    ptr: *mut u16,
    len: usize,
}

unsafe impl Send for AlignedU16 {}
unsafe impl Sync for AlignedU16 {}

impl AlignedU16 {
    pub fn zeros(len: usize) -> Self {
        assert!(len > 0);
        let layout = Layout::from_size_align(len * 2, ALIGN).unwrap();
        let ptr = unsafe { alloc_zeroed(layout) as *mut u16 };
        assert!(!ptr.is_null(), "allocation failed");
        AlignedU16 { ptr, len }
    }

    pub fn from_slice(s: &[u16]) -> Self {
        let mut v = Self::zeros(s.len());
        v.copy_from_slice(s);
        v
    }
}

impl Drop for AlignedU16 {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr as *mut u8, Layout::from_size_align(self.len * 2, ALIGN).unwrap()) }
    }
}

impl Deref for AlignedU16 {
    type Target = [u16];
    fn deref(&self) -> &[u16] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl DerefMut for AlignedU16 {
    fn deref_mut(&mut self) -> &mut [u16] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

#[inline(always)]
pub fn is_aligned16_u16(p: *const u16) -> bool {
    (p as usize) % 16 == 0
}
