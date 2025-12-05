use num_traits::Float;
use std::fmt::Debug;

pub trait Primitive: Float + Debug + Send + Sync + 'static {}
impl<T: Float + Debug + Send + Sync + 'static> Primitive for T {}
