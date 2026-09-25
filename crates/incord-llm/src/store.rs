//! Weight storage: an owned buffer, or a read-only view into a model file.
//!
//! `load_model` memory-maps the `.incord` file and hands out views into it
//! instead of copying every weight onto the heap. Mapped pages are backed by
//! the file, so under memory pressure the OS drops them and reads them back on
//! the next touch instead of running out of memory. The same mechanism later
//! serves streaming: a weight is a range of the file, wherever it lives.
//!
//! [`Buf<T>`] derefs to `&[T]`, so compute code reads it like a `Vec<T>`.

use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;

use half::f16;

/// Element types that can be read straight from little-endian file bytes.
///
/// # Safety
/// Every bit pattern of `size_of::<Self>()` bytes must be a valid value, with
/// no padding (plain integers and `f16`).
pub unsafe trait Plain: Copy + Send + Sync + 'static {
    /// Decode one element from exactly `size_of::<Self>()` little-endian bytes.
    fn from_le(bytes: &[u8]) -> Self;
}

unsafe impl Plain for u8 {
    fn from_le(b: &[u8]) -> Self {
        b[0]
    }
}

unsafe impl Plain for i8 {
    fn from_le(b: &[u8]) -> Self {
        b[0] as i8
    }
}

unsafe impl Plain for f16 {
    fn from_le(b: &[u8]) -> Self {
        f16::from_le_bytes([b[0], b[1]])
    }
}

/// Bytes shared by every view into one file: a memory map, or an in-memory
/// copy of the file (tests).
pub type Backing = Arc<dyn Deref<Target = [u8]> + Send + Sync>;

enum Keep<T> {
    Owned(Vec<T>),
    Shared(Backing),
}

/// A read-only `[T]`, owned or viewed from a [`Backing`].
pub struct Buf<T: Plain> {
    ptr: *const T,
    len: usize,
    keep: Keep<T>,
    _t: PhantomData<T>,
}

// SAFETY: the data behind `ptr` is immutable and kept alive by `keep` (an
// owned Vec or a shared, immutable backing), both of which are Send + Sync.
unsafe impl<T: Plain> Send for Buf<T> {}
unsafe impl<T: Plain> Sync for Buf<T> {}

impl<T: Plain> Buf<T> {
    /// `len` elements of `T` starting at byte `offset` of `backing`, viewed in
    /// place when the host is little-endian and the range is aligned for `T`,
    /// otherwise decoded into an owned copy. `None` if the range is out of
    /// bounds.
    pub fn from_backing(backing: &Backing, offset: usize, len: usize) -> Option<Self> {
        let size = std::mem::size_of::<T>();
        let end = len.checked_mul(size)?.checked_add(offset)?;
        let bytes = backing.get(offset..end)?;
        let ptr = bytes.as_ptr().cast::<T>();
        if cfg!(target_endian = "little") && ptr.is_aligned() {
            return Some(Self { ptr, len, keep: Keep::Shared(Arc::clone(backing)), _t: PhantomData });
        }
        Some(bytes.chunks_exact(size).map(T::from_le).collect::<Vec<T>>().into())
    }

    /// True when the data is a view into a backing rather than a heap copy.
    pub fn is_view(&self) -> bool {
        matches!(self.keep, Keep::Shared(_))
    }

    /// Size of the data in bytes.
    pub fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

impl<T: Plain> From<Vec<T>> for Buf<T> {
    fn from(v: Vec<T>) -> Self {
        // A Vec's heap buffer does not move when the Vec itself moves.
        Self { ptr: v.as_ptr(), len: v.len(), keep: Keep::Owned(v), _t: PhantomData }
    }
}

impl<T: Plain> FromIterator<T> for Buf<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        iter.into_iter().collect::<Vec<T>>().into()
    }
}

impl<T: Plain> Deref for Buf<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &[T] {
        // SAFETY: `ptr` points to `len` initialized, aligned `T`s that `keep`
        // keeps alive and nothing mutates (checked in `from_backing` / `from`).
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T: Plain> Clone for Buf<T> {
    /// A view clones the handle (no copy); an owned buffer copies its data.
    fn clone(&self) -> Self {
        match &self.keep {
            Keep::Owned(v) => v.clone().into(),
            Keep::Shared(b) => Self { ptr: self.ptr, len: self.len, keep: Keep::Shared(Arc::clone(b)), _t: PhantomData },
        }
    }
}

impl<T: Plain + std::fmt::Debug> std::fmt::Debug for Buf<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buf").field("len", &self.len).field("view", &self.is_view()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backing(bytes: Vec<u8>) -> Backing {
        Arc::new(bytes)
    }

    #[test]
    fn owned_reads_like_the_vec() {
        let b: Buf<f16> = vec![f16::from_f32(1.5), f16::from_f32(-2.0)].into();
        assert_eq!(b.len(), 2);
        assert_eq!(b[1].to_f32(), -2.0);
        assert!(!b.is_view());
        assert_eq!(b.byte_len(), 4);
    }

    #[test]
    fn view_decodes_little_endian_f16_at_any_offset() {
        let vals = [1.0f32, -0.5, 65504.0, 0.0];
        let mut bytes = vec![0xAAu8; 3];
        for v in vals {
            bytes.extend(f16::from_f32(v).to_le_bytes());
        }
        let bk = backing(bytes);
        // Offset 3 may be misaligned for f16: then it must copy, never misread.
        let b = Buf::<f16>::from_backing(&bk, 3, 4).unwrap();
        assert_eq!(b.iter().map(|x| x.to_f32()).collect::<Vec<_>>(), vals);
        // One of offsets 2 and 3 is 2-byte aligned; that one is a view on
        // little-endian hosts.
        let even = if (bk.as_ptr() as usize + 3) % 2 == 0 { 3 } else { 2 };
        let v = Buf::<f16>::from_backing(&bk, even, 1).unwrap();
        assert_eq!(v.is_view(), cfg!(target_endian = "little"));
    }

    #[test]
    fn bytes_are_always_viewed() {
        let bk = backing(vec![1, 2, 0xFF, 4]);
        let b = Buf::<i8>::from_backing(&bk, 1, 3).unwrap();
        assert_eq!(&*b, &[2i8, -1, 4]);
        assert!(b.is_view());
        let c = b.clone();
        assert!(c.is_view());
        assert_eq!(c.as_ptr(), b.as_ptr(), "cloning a view must not copy");
    }

    #[test]
    fn out_of_bounds_is_rejected() {
        let bk = backing(vec![0; 8]);
        assert!(Buf::<u8>::from_backing(&bk, 4, 5).is_none());
        assert!(Buf::<f16>::from_backing(&bk, 0, 5).is_none());
        assert!(Buf::<u8>::from_backing(&bk, usize::MAX, 2).is_none());
        assert!(Buf::<u8>::from_backing(&bk, 8, 0).is_some());
    }

    #[test]
    fn view_outlives_the_callers_handle() {
        let bk = backing(vec![7, 8, 9]);
        let b = Buf::<u8>::from_backing(&bk, 0, 3).unwrap();
        drop(bk);
        assert_eq!(&*b, &[7, 8, 9]);
    }
}
