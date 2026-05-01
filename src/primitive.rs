use crate::CpuBackendType;
use num_traits::Float;
use std::fmt::Debug;

pub trait Primitive: Float + Debug + Send + Sync + 'static {
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
    type DefaultInferenceBackend = crate::CPUSimd;

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
    type DefaultInferenceBackend = crate::CPUSimd;

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
