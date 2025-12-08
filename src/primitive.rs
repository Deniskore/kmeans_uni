use num_traits::Float;
use std::fmt::Debug;

pub trait Primitive: Float + Debug + Send + Sync + 'static {
    /// Converts a `usize` to this primitive type.
    ///
    /// This method is preferred over `as` casting or generic `From` traits to ensure
    /// consistent behavior across backends and explicit handling of potential precision loss
    /// (though for k-means counts/indices, values are expected to fit).
    fn from_usize(n: usize) -> Self;
}

impl Primitive for f32 {
    #[inline(always)]
    fn from_usize(n: usize) -> Self {
        n as f32
    }
}

impl Primitive for f64 {
    #[inline(always)]
    fn from_usize(n: usize) -> Self {
        n as f64
    }
}
