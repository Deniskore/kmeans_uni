#![forbid(unsafe_code)]

mod backend;
mod error;
mod kmeans_core;
mod kmeans_core_common;
mod kmeans_core_scalar;
pub mod kmeans_cpu;
mod kmeans_mini_batch;
mod point_source;
mod primitive;

use backend::CoreBackend;
use error::Error as KMeansError;
pub use error::{Error, Result};
use kmeans_core_common::{calculate_chunk_size, dot_product, squared_euclidean};
use kmeans_cpu::run as run_cpu;
pub use point_source::{PointSource, SlicePointSource};
pub use primitive::Primitive;
use std::marker::PhantomData;

#[cfg(feature = "wide")]
mod kmeans_core_simd;
#[cfg(feature = "wide")]
pub struct CPUSimd;
#[cfg(feature = "wide")]
impl BackendType for CPUSimd {}
#[cfg(feature = "wide")]
impl CpuBackendType<f32> for CPUSimd {
    type Core = kmeans_core_simd::SimdBackend;
}

#[cfg(feature = "wide")]
impl CpuBackendType<f64> for CPUSimd {
    type Core = kmeans_core_simd::SimdBackend;
}

pub struct CPUScalar;

pub trait BackendType: Send + Sync {}
impl BackendType for CPUScalar {}

pub trait CpuBackendType<F: Primitive>: BackendType {
    type Core: backend::CoreBackend<F>;
}

impl<F: Primitive> CpuBackendType<F> for CPUScalar {
    type Core = kmeans_core_scalar::ScalarBackend;
}

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
pub mod wasm;

#[derive(Clone, Copy)]
pub struct Euclidean;

#[derive(Clone, Copy)]
pub struct DotProduct;

/// Metric used by the trained model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MetricType {
    Euclidean,
    DotProduct,
}

pub trait AlgorithmType<F: Primitive>: Copy + Send + Sync {
    type Metric: backend::DistanceMetric<F>;
    fn metric() -> MetricType;
}

impl<F: Primitive> AlgorithmType<F> for Euclidean {
    type Metric = backend::Euclidean;
    fn metric() -> MetricType {
        MetricType::Euclidean
    }
}

impl<F: Primitive> AlgorithmType<F> for DotProduct {
    type Metric = backend::DotProduct;
    fn metric() -> MetricType {
        MetricType::DotProduct
    }
}

pub use kmeans_core::{InitializationStrategy, KMeansPlusPlus};

/// A trained K-Means model.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct KMeans<F: Primitive> {
    /// The cluster centers (centroids).
    centroids: Vec<F>,
    /// The number of features (columns) in the data.
    ncols: usize,
    /// The number of clusters (k).
    k: usize,
    /// The inertia (sum of squared distances to nearest centroid) of the model.
    inertia: F,
    /// Metric used during training (controls prediction/transform behavior).
    metric: MetricType,
}

#[cfg(feature = "serde")]
impl<'de, F> serde::Deserialize<'de> for KMeans<F>
where
    F: Primitive + serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Parts<F> {
            centroids: Vec<F>,
            ncols: usize,
            k: usize,
            inertia: F,
            metric: MetricType,
        }

        let parts = Parts::deserialize(deserializer)?;
        KMeans::new(
            parts.centroids,
            parts.ncols,
            parts.k,
            parts.inertia,
            parts.metric,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl<F: Primitive> std::fmt::Display for KMeans<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "KMeans(k={}, ncols={}, metric={:?}, inertia={:?})",
            self.k, self.ncols, self.metric, self.inertia
        )
    }
}

impl<F: Primitive> KMeans<F> {
    /// Build a model from parts, validating centroid shape against `k` and `ncols`.
    ///
    /// # Examples
    /// ```
    /// use kmeans_uni::{KMeans, MetricType};
    ///
    /// let centroids = vec![0.0f32, 0.0, 1.0, 1.0];
    /// let model = KMeans::new(centroids, 2, 2, 0.0, MetricType::Euclidean).unwrap();
    /// assert_eq!(model.k(), 2);
    /// ```
    pub fn new(
        centroids: Vec<F>,
        ncols: usize,
        k: usize,
        inertia: F,
        metric: MetricType,
    ) -> Result<Self> {
        validate_centroid_shape(&centroids, ncols, k)?;
        Ok(Self {
            centroids,
            ncols,
            k,
            inertia,
            metric,
        })
    }

    /// Centroids as a borrowed slice (row-major).
    pub fn centroids(&self) -> &[F] {
        &self.centroids
    }

    /// Consume the model and return owned centroids.
    pub fn into_centroids(self) -> Vec<F> {
        self.centroids
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn inertia(&self) -> F {
        self.inertia
    }

    pub fn metric(&self) -> MetricType {
        self.metric
    }

    fn validate_model_shape(&self) -> Result<()> {
        validate_centroid_shape(&self.centroids, self.ncols, self.k)
    }

    /// Fit with the scalar CPU backend and Euclidean metric using builder defaults.
    ///
    /// # Examples
    /// ```
    /// use kmeans_uni::KMeans;
    ///
    /// let data = [1.0f32, 1.0, 5.0, 5.0];
    /// let model = KMeans::fit_default_scalar(&data, 2, 2).unwrap();
    /// let labels = model.predict(&data).unwrap();
    /// assert_eq!(labels.len(), 2);
    /// ```
    pub fn fit_default_scalar(points: impl AsRef<[F]>, ncols: usize, k: usize) -> Result<Self> {
        KMeansBuilder::new(k)
            .cpu_scalar()
            .euclidean()
            .build()
            .fit(points, ncols)
    }

    /// Predict the closest cluster for each point in the input data.
    pub fn predict(&self, points: impl AsRef<[F]>) -> Result<Vec<usize>> {
        self.predict_with_backend::<CPUScalar>(points)
    }

    /// Predict using a sequential execution path (no Rayon), useful when thread-pinning or deterministic scheduling is required.
    ///
    /// # Examples
    /// ```
    /// use kmeans_uni::KMeans;
    ///
    /// let data = [1.0f32, 1.0, 5.0, 5.0];
    /// let model = KMeans::fit_default_scalar(&data, 2, 2).unwrap();
    /// let labels = model.predict_sequential(&data).unwrap();
    /// assert_eq!(labels.len(), 2);
    /// ```
    pub fn predict_sequential(&self, points: impl AsRef<[F]>) -> Result<Vec<usize>> {
        self.predict_with_backend_sequential::<CPUScalar>(points)
    }

    pub fn predict_with_backend<B: CpuBackendType<F>>(
        &self,
        points: impl AsRef<[F]>,
    ) -> Result<Vec<usize>> {
        self.predict_with_backend_mode::<B>(points, true)
    }

    /// Predict with an explicit sequential mode, bypassing Rayon even on native builds.
    pub fn predict_with_backend_sequential<B: CpuBackendType<F>>(
        &self,
        points: impl AsRef<[F]>,
    ) -> Result<Vec<usize>> {
        self.predict_with_backend_mode::<B>(points, false)
    }

    fn predict_with_backend_mode<B: CpuBackendType<F>>(
        &self,
        points: impl AsRef<[F]>,
        parallel: bool,
    ) -> Result<Vec<usize>> {
        self.validate_model_shape()?;
        let points = points.as_ref();
        validate_inputs(points, self.ncols, self.k)?;

        let npoints = points.len() / self.ncols;
        let mut labels = vec![0usize; npoints];

        let chunk_size = calculate_chunk_size::<F>(self.ncols);
        let prepared_centroids = B::Core::prepare_centroids(&self.centroids, self.ncols, self.k);
        #[cfg(feature = "wasm")]
        let _ = parallel;

        #[cfg(not(feature = "wasm"))]
        if parallel {
            use rayon::prelude::*;
            labels
                .par_chunks_mut(chunk_size)
                .enumerate()
                .for_each(|(chunk_idx, chunk_labels)| {
                    let start = chunk_idx * chunk_size;
                    let end = start + chunk_labels.len();
                    let chunk_points = &points[start * self.ncols..end * self.ncols];

                    match self.metric {
                        MetricType::Euclidean => B::Core::find_nearest_centroids_euc(
                            chunk_points,
                            self.ncols,
                            &prepared_centroids,
                            self.k,
                            chunk_labels,
                            None,
                        ),
                        MetricType::DotProduct => B::Core::find_nearest_centroids_dot_product(
                            chunk_points,
                            self.ncols,
                            &prepared_centroids,
                            self.k,
                            chunk_labels,
                            None,
                        ),
                    }
                });

            return Ok(labels);
        }

        // Fallback to sequential execution (used for explicit sequential requests and wasm builds).
        {
            for (chunk_idx, chunk_labels) in labels.chunks_mut(chunk_size).enumerate() {
                let start = chunk_idx * chunk_size;
                let end = start + chunk_labels.len();
                let chunk_points = &points[start * self.ncols..end * self.ncols];

                match self.metric {
                    MetricType::Euclidean => B::Core::find_nearest_centroids_euc(
                        chunk_points,
                        self.ncols,
                        &prepared_centroids,
                        self.k,
                        chunk_labels,
                        None,
                    ),
                    MetricType::DotProduct => B::Core::find_nearest_centroids_dot_product(
                        chunk_points,
                        self.ncols,
                        &prepared_centroids,
                        self.k,
                        chunk_labels,
                        None,
                    ),
                }
            }
        }

        Ok(labels)
    }

    /// Predict the closest cluster for each point in the input source.
    pub fn predict_from_source<S: PointSource<F>>(&self, source: &S) -> Result<Vec<usize>> {
        self.predict_from_source_with_backend::<CPUScalar, _>(source)
    }

    /// Predict the closest cluster for each point in the input source using a specific backend.
    pub fn predict_from_source_with_backend<B: CpuBackendType<F>, S: PointSource<F>>(
        &self,
        source: &S,
    ) -> Result<Vec<usize>> {
        self.validate_model_shape()?;
        validate_source_inputs(source, self.k)?;
        let npoints = source.num_points();
        let mut labels = vec![0usize; npoints];

        // Use backend-prepared centroids for cache-friendly access (e.g., SIMD packing)
        let prepared_centroids = B::Core::prepare_centroids(&self.centroids, self.ncols, self.k);

        // Chunk size scales with dimensionality
        let chunk_size = calculate_chunk_size::<F>(self.ncols);
        let mut buffer = vec![F::zero(); chunk_size * self.ncols];

        let mut processed = 0;
        while processed < npoints {
            let current_chunk_size = std::cmp::min(chunk_size, npoints - processed);
            let chunk_buffer = &mut buffer[..current_chunk_size * self.ncols];
            source.read_batch(processed, current_chunk_size, chunk_buffer);

            match self.metric {
                MetricType::Euclidean => B::Core::find_nearest_centroids_euc(
                    chunk_buffer,
                    self.ncols,
                    &prepared_centroids,
                    self.k,
                    &mut labels[processed..processed + current_chunk_size],
                    None,
                ),
                MetricType::DotProduct => B::Core::find_nearest_centroids_dot_product(
                    chunk_buffer,
                    self.ncols,
                    &prepared_centroids,
                    self.k,
                    &mut labels[processed..processed + current_chunk_size],
                    None,
                ),
            }

            processed += current_chunk_size;
        }

        Ok(labels)
    }

    /// Transform the input data to per-centroid scores.
    /// For `MetricType::Euclidean` this returns squared Euclidean distances,
    /// for `MetricType::DotProduct` it returns raw dot-product similarities.
    /// Returns a flat vector of shape (n_points * k).
    pub fn transform(&self, points: impl AsRef<[F]>) -> Result<Vec<F>> {
        self.validate_model_shape()?;
        let points = points.as_ref();
        validate_inputs(points, self.ncols, self.k)?;

        let npoints = points.len() / self.ncols;
        let mut distances = vec![F::zero(); npoints * self.k];

        let chunk_size = calculate_chunk_size::<F>(self.ncols);

        #[cfg(not(feature = "wasm"))]
        {
            use rayon::prelude::*;
            distances
                .par_chunks_mut(chunk_size * self.k)
                .enumerate()
                .for_each(|(chunk_idx, chunk_dists)| {
                    let start_point_idx = chunk_idx * chunk_size;
                    let num_points_in_chunk = chunk_dists.len() / self.k;

                    for i in 0..num_points_in_chunk {
                        let point_idx = start_point_idx + i;
                        let point = &points[point_idx * self.ncols..(point_idx + 1) * self.ncols];

                        match self.metric {
                            MetricType::Euclidean => {
                                for c in 0..self.k {
                                    let centroid =
                                        &self.centroids[c * self.ncols..(c + 1) * self.ncols];
                                    let dist = squared_euclidean(point, centroid);
                                    chunk_dists[i * self.k + c] = dist;
                                }
                            }
                            MetricType::DotProduct => {
                                for c in 0..self.k {
                                    let centroid =
                                        &self.centroids[c * self.ncols..(c + 1) * self.ncols];
                                    let dot = dot_product(point, centroid);
                                    chunk_dists[i * self.k + c] = dot;
                                }
                            }
                        }
                    }
                });
        }

        #[cfg(feature = "wasm")]
        {
            for (chunk_idx, chunk_dists) in distances.chunks_mut(chunk_size * self.k).enumerate() {
                let start_point_idx = chunk_idx * chunk_size;
                let num_points_in_chunk = chunk_dists.len() / self.k;

                for i in 0..num_points_in_chunk {
                    let point_idx = start_point_idx + i;
                    let point = &points[point_idx * self.ncols..(point_idx + 1) * self.ncols];

                    match self.metric {
                        MetricType::Euclidean => {
                            for c in 0..self.k {
                                let centroid =
                                    &self.centroids[c * self.ncols..(c + 1) * self.ncols];
                                let dist = squared_euclidean(point, centroid);
                                chunk_dists[i * self.k + c] = dist;
                            }
                        }
                        MetricType::DotProduct => {
                            for c in 0..self.k {
                                let centroid =
                                    &self.centroids[c * self.ncols..(c + 1) * self.ncols];
                                let dot = dot_product(point, centroid);
                                chunk_dists[i * self.k + c] = dot;
                            }
                        }
                    }
                }
            }
        }

        Ok(distances)
    }
}

#[cfg(feature = "wide")]
impl KMeans<f32> {
    /// Fit with the SIMD CPU backend (requires `wide` feature) and Euclidean metric.
    ///
    /// # Examples
    /// ```
    /// # #[cfg(feature = "wide")] {
    /// use kmeans_uni::KMeans;
    ///
    /// let data = [0.0f32, 0.0, 2.0, 2.0];
    /// let model = KMeans::<f32>::fit_default_simd(&data, 2, 2).unwrap();
    /// assert_eq!(model.k(), 2);
    /// # }
    /// ```
    pub fn fit_default_simd(points: impl AsRef<[f32]>, ncols: usize, k: usize) -> Result<Self> {
        KMeansBuilder::new(k)
            .cpu_simd()
            .euclidean()
            .build()
            .fit(points, ncols)
    }
}

#[cfg(feature = "wide")]
impl KMeans<f64> {
    /// Fit with the SIMD CPU backend (requires `wide` feature) and Euclidean metric.
    pub fn fit_default_simd(points: impl AsRef<[f64]>, ncols: usize, k: usize) -> Result<Self> {
        KMeansBuilder::new(k)
            .cpu_simd()
            .euclidean()
            .build()
            .fit(points, ncols)
    }
}

pub struct KMeansConfig<
    F: Primitive,
    B: BackendType,
    A: AlgorithmType<F>,
    const PARALLEL: bool,
    I: InitializationStrategy = KMeansPlusPlus,
> {
    _marker: PhantomData<(F, B, A, I)>,
    k: usize,
    iterations: usize,
    attempts: usize,
    tolerance: F,
    seed: Option<u64>,
    mini_batch_rel_tolerance: f64,
    mini_batch_min_iterations: usize,
    mini_batch_patience: usize,
}

pub struct BackendNotSet;
pub struct AlgorithmNotSet;

pub struct KMeansBuilder<
    F: Primitive,
    B = BackendNotSet,
    A = AlgorithmNotSet,
    I: InitializationStrategy = KMeansPlusPlus,
> {
    _marker: PhantomData<(F, B, A, I)>,
    k: usize,
    iterations: usize,
    attempts: usize,
    tolerance: F,
    seed: Option<u64>,
    mini_batch_rel_tolerance: f64,
    mini_batch_min_iterations: usize,
    mini_batch_patience: usize,
}

impl<F: Primitive> KMeansBuilder<F> {
    #[inline]
    pub fn new(k: usize) -> KMeansBuilder<F, BackendNotSet, AlgorithmNotSet, KMeansPlusPlus> {
        KMeansBuilder {
            _marker: PhantomData,
            k,
            iterations: 100,
            attempts: 1,
            tolerance: F::from(1e-4).unwrap_or(F::zero()),
            seed: None,
            mini_batch_rel_tolerance: kmeans_mini_batch::DEFAULT_MINI_BATCH_REL_TOL,
            mini_batch_min_iterations: kmeans_mini_batch::DEFAULT_MINI_BATCH_MIN_ITERATIONS,
            mini_batch_patience: kmeans_mini_batch::DEFAULT_MINI_BATCH_PATIENCE,
        }
    }
}

impl<F: Primitive> Default for KMeansBuilder<F> {
    fn default() -> Self {
        Self::new(8)
    }
}

impl<F: Primitive, I: InitializationStrategy> KMeansBuilder<F, BackendNotSet, AlgorithmNotSet, I> {
    /// Build with the common scalar backend and Euclidean metric without needing to set type-state flags.
    ///
    /// # Examples
    /// ```
    /// use kmeans_uni::KMeansBuilder;
    ///
    /// let data = [0.0f32, 0.0, 1.0, 1.0];
    /// let model = KMeansBuilder::new(2)
    ///     .build_default()
    ///     .fit(&data, 2)
    ///     .unwrap();
    /// assert_eq!(model.k(), 2);
    /// ```
    #[inline]
    pub fn build_default(self) -> KMeansConfig<F, CPUScalar, Euclidean, false, I> {
        self.cpu_scalar().euclidean().build()
    }
}

impl<F: Primitive, B, A, I: InitializationStrategy> KMeansBuilder<F, B, A, I> {
    #[inline]
    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    #[inline]
    pub fn attempts(mut self, attempts: usize) -> Self {
        self.attempts = attempts;
        self
    }

    #[inline]
    pub fn tolerance(mut self, tolerance: F) -> Self {
        self.tolerance = tolerance;
        self
    }

    #[inline]
    pub fn mini_batch_rel_tolerance(mut self, rel_tolerance: f64) -> Self {
        self.mini_batch_rel_tolerance = rel_tolerance.max(0.0);
        self
    }

    #[inline]
    pub fn mini_batch_min_iterations(mut self, min_iterations: usize) -> Self {
        self.mini_batch_min_iterations = min_iterations;
        self
    }

    #[inline]
    pub fn mini_batch_patience(mut self, patience: usize) -> Self {
        self.mini_batch_patience = patience;
        self
    }

    #[inline]
    pub fn backend<NB: BackendType>(self) -> KMeansBuilder<F, NB, A, I> {
        KMeansBuilder {
            _marker: PhantomData,
            k: self.k,
            iterations: self.iterations,
            attempts: self.attempts,
            tolerance: self.tolerance,
            seed: self.seed,
            mini_batch_rel_tolerance: self.mini_batch_rel_tolerance,
            mini_batch_min_iterations: self.mini_batch_min_iterations,
            mini_batch_patience: self.mini_batch_patience,
        }
    }

    #[inline]
    pub fn algorithm<NA: AlgorithmType<F>>(self) -> KMeansBuilder<F, B, NA, I> {
        KMeansBuilder {
            _marker: PhantomData,
            k: self.k,
            iterations: self.iterations,
            attempts: self.attempts,
            tolerance: self.tolerance,
            seed: self.seed,
            mini_batch_rel_tolerance: self.mini_batch_rel_tolerance,
            mini_batch_min_iterations: self.mini_batch_min_iterations,
            mini_batch_patience: self.mini_batch_patience,
        }
    }

    #[inline]
    pub fn init_plus_plus(self) -> KMeansBuilder<F, B, A, KMeansPlusPlus> {
        KMeansBuilder {
            _marker: PhantomData,
            k: self.k,
            iterations: self.iterations,
            attempts: self.attempts,
            tolerance: self.tolerance,
            seed: self.seed,
            mini_batch_rel_tolerance: self.mini_batch_rel_tolerance,
            mini_batch_min_iterations: self.mini_batch_min_iterations,
            mini_batch_patience: self.mini_batch_patience,
        }
    }

    #[inline]
    pub fn cpu_scalar(self) -> KMeansBuilder<F, CPUScalar, A, I> {
        self.backend::<CPUScalar>()
    }

    #[cfg(feature = "wide")]
    #[inline]
    pub fn cpu_simd(self) -> KMeansBuilder<F, CPUSimd, A, I> {
        self.backend::<CPUSimd>()
    }

    #[inline]
    pub fn euclidean(self) -> KMeansBuilder<F, B, Euclidean, I> {
        self.algorithm::<Euclidean>()
    }

    #[inline]
    pub fn dot_product(self) -> KMeansBuilder<F, B, DotProduct, I> {
        self.algorithm::<DotProduct>()
    }

    #[inline]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

impl<F: Primitive, B: BackendType, A: AlgorithmType<F>, I: InitializationStrategy>
    KMeansBuilder<F, B, A, I>
{
    #[inline]
    pub fn build(self) -> KMeansConfig<F, B, A, false, I> {
        KMeansConfig {
            _marker: PhantomData,
            k: self.k,
            iterations: self.iterations,
            attempts: self.attempts,
            tolerance: self.tolerance,
            seed: self.seed,
            mini_batch_rel_tolerance: self.mini_batch_rel_tolerance,
            mini_batch_min_iterations: self.mini_batch_min_iterations,
            mini_batch_patience: self.mini_batch_patience,
        }
    }

    #[inline]
    pub fn sequential(self) -> KMeansConfig<F, B, A, false, I> {
        KMeansConfig {
            _marker: PhantomData,
            k: self.k,
            iterations: self.iterations,
            attempts: self.attempts,
            tolerance: self.tolerance,
            seed: self.seed,
            mini_batch_rel_tolerance: self.mini_batch_rel_tolerance,
            mini_batch_min_iterations: self.mini_batch_min_iterations,
            mini_batch_patience: self.mini_batch_patience,
        }
    }

    #[inline]
    pub fn parallel(self) -> KMeansConfig<F, B, A, true, I> {
        KMeansConfig {
            _marker: PhantomData,
            k: self.k,
            iterations: self.iterations,
            attempts: self.attempts,
            tolerance: self.tolerance,
            seed: self.seed,
            mini_batch_rel_tolerance: self.mini_batch_rel_tolerance,
            mini_batch_min_iterations: self.mini_batch_min_iterations,
            mini_batch_patience: self.mini_batch_patience,
        }
    }
}

impl<F: Primitive, B, A, I> KMeansConfig<F, B, A, false, I>
where
    B: CpuBackendType<F>,
    A: AlgorithmType<F>,
    I: InitializationStrategy,
{
    pub fn fit(&self, points: impl AsRef<[F]>, ncols: usize) -> Result<KMeans<F>> {
        let points = points.as_ref();
        validate_inputs(points, ncols, self.k)?;
        validate_attempts(self.attempts)?;
        let source = SlicePointSource::new(points, ncols)?;
        self.fit_from_source(&source)
    }

    pub fn fit_from_source<S: PointSource<F>>(&self, source: &S) -> Result<KMeans<F>> {
        validate_source_inputs(source, self.k)?;
        validate_attempts(self.attempts)?;
        let mut best_model = None;
        let mut min_inertia = F::infinity();

        for i in 0..self.attempts {
            let seed = self.seed.map(|s| s.wrapping_add(i as u64));
            let (centroids, inertia) =
                run_cpu::<F, B::Core, A::Metric, _, I, kmeans_core::Sequential>(
                    source,
                    self.k,
                    self.iterations,
                    self.tolerance,
                    seed,
                )?;

            if inertia < min_inertia {
                min_inertia = inertia;
                best_model = Some(KMeans::new(
                    centroids,
                    source.num_columns(),
                    self.k,
                    inertia,
                    A::metric(),
                )?);
            }
        }

        best_model.ok_or(KMeansError::InvalidInput(
            "Failed to produce a model".into(),
        ))
    }

    pub fn fit_mini_batch_from_source<S: PointSource<F>>(
        &self,
        source: &S,
        batch_size: usize,
    ) -> Result<KMeans<F>> {
        validate_source_inputs(source, self.k)?;
        validate_attempts(self.attempts)?;
        let mut best_model = None;
        let mut min_inertia = F::infinity();

        for i in 0..self.attempts {
            let seed = self.seed.map(|s| s.wrapping_add(i as u64));
            let (centroids, inertia) =
                kmeans_mini_batch::run::<F, B::Core, A::Metric, _, I, kmeans_core::Sequential>(
                    source,
                    self.k,
                    self.iterations,
                    batch_size,
                    seed,
                    self.mini_batch_rel_tolerance,
                    self.mini_batch_min_iterations,
                    self.mini_batch_patience,
                )?;

            if inertia < min_inertia {
                min_inertia = inertia;
                best_model = Some(KMeans::new(
                    centroids,
                    source.num_columns(),
                    self.k,
                    inertia,
                    A::metric(),
                )?);
            }
        }

        best_model.ok_or(KMeansError::InvalidInput(
            "Failed to produce a model".into(),
        ))
    }
}

impl<F: Primitive, B, A, I> KMeansConfig<F, B, A, true, I>
where
    B: CpuBackendType<F>,
    A: AlgorithmType<F>,
    I: InitializationStrategy,
{
    pub fn fit(&self, points: impl AsRef<[F]>, ncols: usize) -> Result<KMeans<F>> {
        let points = points.as_ref();
        validate_inputs(points, ncols, self.k)?;
        validate_attempts(self.attempts)?;
        let source = SlicePointSource::new(points, ncols)?;
        self.fit_from_source(&source)
    }

    pub fn fit_from_source<S: PointSource<F>>(&self, source: &S) -> Result<KMeans<F>> {
        validate_source_inputs(source, self.k)?;
        validate_attempts(self.attempts)?;
        let mut best_model = None;
        let mut min_inertia = F::infinity();

        for i in 0..self.attempts {
            let seed = self.seed.map(|s| s.wrapping_add(i as u64));
            let (centroids, inertia) = run_cpu::<F, B::Core, A::Metric, _, I, kmeans_core::Parallel>(
                source,
                self.k,
                self.iterations,
                self.tolerance,
                seed,
            )?;

            if inertia < min_inertia {
                min_inertia = inertia;
                best_model = Some(KMeans::new(
                    centroids,
                    source.num_columns(),
                    self.k,
                    inertia,
                    A::metric(),
                )?);
            }
        }

        best_model.ok_or(KMeansError::InvalidInput(
            "Failed to produce a model".into(),
        ))
    }

    pub fn fit_mini_batch_from_source<S: PointSource<F>>(
        &self,
        source: &S,
        batch_size: usize,
    ) -> Result<KMeans<F>> {
        validate_source_inputs(source, self.k)?;
        validate_attempts(self.attempts)?;
        let mut best_model = None;
        let mut min_inertia = F::infinity();

        for i in 0..self.attempts {
            let seed = self.seed.map(|s| s.wrapping_add(i as u64));
            let (centroids, inertia) =
                kmeans_mini_batch::run::<F, B::Core, A::Metric, _, I, kmeans_core::Parallel>(
                    source,
                    self.k,
                    self.iterations,
                    batch_size,
                    seed,
                    self.mini_batch_rel_tolerance,
                    self.mini_batch_min_iterations,
                    self.mini_batch_patience,
                )?;

            if inertia < min_inertia {
                min_inertia = inertia;
                best_model = Some(KMeans::new(
                    centroids,
                    source.num_columns(),
                    self.k,
                    inertia,
                    A::metric(),
                )?);
            }
        }

        best_model.ok_or(KMeansError::InvalidInput(
            "Failed to produce a model".into(),
        ))
    }
}

fn validate_centroid_shape<F: Primitive>(centroids: &[F], ncols: usize, k: usize) -> Result<()> {
    if ncols == 0 {
        return Err(KMeansError::InvalidInput(
            "number of columns must be greater than zero".into(),
        ));
    }
    if k == 0 {
        return Err(KMeansError::InvalidInput(
            "number of centroids must be greater than zero".into(),
        ));
    }
    if centroids.len() != ncols.saturating_mul(k) {
        return Err(KMeansError::InvalidInput(
            "centroids length must equal k * ncols".into(),
        ));
    }
    Ok(())
}

#[inline]
fn validate_inputs<F: Primitive>(points: &[F], ncols: usize, k: usize) -> Result<()> {
    if ncols == 0 {
        return Err(KMeansError::InvalidInput(
            "number of columns must be greater than zero".into(),
        ));
    }
    if k == 0 {
        return Err(KMeansError::InvalidInput(
            "number of centroids must be greater than zero".into(),
        ));
    }
    if !points.len().is_multiple_of(ncols) {
        return Err(KMeansError::InvalidInput(
            "points length must be divisible by ncols".into(),
        ));
    }
    if points.is_empty() {
        return Err(KMeansError::InvalidInput(
            "points must contain at least one row".into(),
        ));
    }
    Ok(())
}

fn validate_attempts(attempts: usize) -> Result<()> {
    if attempts == 0 {
        return Err(KMeansError::InvalidInput(
            "number of attempts must be greater than zero".into(),
        ));
    }
    Ok(())
}

fn validate_source_inputs<F: Primitive, S: PointSource<F>>(source: &S, k: usize) -> Result<()> {
    if source.num_columns() == 0 {
        return Err(KMeansError::InvalidInput(
            "number of columns must be greater than zero".into(),
        ));
    }
    if k == 0 {
        return Err(KMeansError::InvalidInput(
            "number of centroids must be greater than zero".into(),
        ));
    }
    if source.num_points() == 0 {
        return Err(KMeansError::InvalidInput(
            "point source must contain at least one point".into(),
        ));
    }
    Ok(())
}
