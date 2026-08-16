use crate::CpuBackendType;
use num_traits::Float;
use std::fmt::Debug;

mod sealed {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Supported floating point type for K-Means data.
///
/// This trait is sealed and currently implemented for `f32` and `f64`.
pub trait Primitive: sealed::Sealed + Float + Debug + Send + Sync + 'static {
    type DefaultInferenceBackend: CpuBackendType<Self>;

    /// Converts a `usize` to this primitive type.
    ///
    /// This method is preferred over `as` casting or generic `From` traits to ensure
    /// consistent behavior across backends and explicit handling of potential precision loss
    /// (though for k-means counts/indices, values are expected to fit).
    fn from_usize(n: usize) -> Self;
}

#[cfg(feature = "wide")]
impl Primitive for f32 {
    type DefaultInferenceBackend = crate::CPUSimdAdaptive;

    #[inline(always)]
    fn from_usize(n: usize) -> Self {
        n as f32
    }
}

#[cfg(not(feature = "wide"))]
impl Primitive for f32 {
    type DefaultInferenceBackend = crate::CPUScalar;

    #[inline(always)]
    fn from_usize(n: usize) -> Self {
        n as f32
    }
}

#[cfg(feature = "wide")]
impl Primitive for f64 {
    type DefaultInferenceBackend = crate::CPUSimdAdaptive;

    #[inline(always)]
    fn from_usize(n: usize) -> Self {
        n as f64
    }
}

#[cfg(not(feature = "wide"))]
impl Primitive for f64 {
    type DefaultInferenceBackend = crate::CPUScalar;

    #[inline(always)]
    fn from_usize(n: usize) -> Self {
        n as f64
    }
}
