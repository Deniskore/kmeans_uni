use divan::black_box;
use kmeans_uni::{
    CPUScalar, CpuBackendType, KMeansBuilder, PointSource, Primitive, SlicePointSource,
};
#[cfg(feature = "wide")]
use kmeans_uni::{CPUSimd256, CPUSimdAdaptive};
use linfa::DatasetBase;
use linfa::traits::{Fit, FitWith};
use linfa_clustering::{IncrKMeansError, KMeans as LinfaKMeans};
use linfa_nn::distance::L2Dist;
use ndarray::{ArrayView2, s};
use rand08::distributions::uniform::SampleUniform;
use rand08::{Rng, SeedableRng, rngs::StdRng};
use std::sync::OnceLock;

static DATA_F32: OnceLock<Vec<f32>> = OnceLock::new();
static DATA_F64: OnceLock<Vec<f64>> = OnceLock::new();
static SINGLE_THREAD_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

static UNI_MODEL_SCALAR_F32: OnceLock<kmeans_uni::KMeans<f32>> = OnceLock::new();
static UNI_MODEL_SCALAR_F64: OnceLock<kmeans_uni::KMeans<f64>> = OnceLock::new();
#[cfg(feature = "wide")]
static UNI_MODEL_SIMD_F32: OnceLock<kmeans_uni::KMeans<f32>> = OnceLock::new();
#[cfg(feature = "wide")]
static UNI_MODEL_SIMD_F64: OnceLock<kmeans_uni::KMeans<f64>> = OnceLock::new();
static LINFA_MODEL_F32: OnceLock<LinfaKMeans<f32, L2Dist>> = OnceLock::new();
static LINFA_MODEL_F64: OnceLock<LinfaKMeans<f64, L2Dist>> = OnceLock::new();

pub(crate) const N_COLS: usize = 16;
pub(crate) const K: usize = 16;
// 16,384 points: ~1 MiB for f32 and ~2 MiB for f64 at 16 columns.
pub(crate) const N_POINTS: usize = 16384;
pub(crate) const BATCH: usize = 256;
pub(crate) const EPOCHS: usize = 10;
pub(crate) const SEED: u64 = 42;

#[derive(Clone, Copy)]
pub(crate) struct Shape {
    pub npoints: usize,
    pub ncols: usize,
    pub k: usize,
}

impl Shape {
    pub const fn new(npoints: usize, ncols: usize, k: usize) -> Self {
        Self { npoints, ncols, k }
    }
}

pub(crate) struct CopyOnlySource<F> {
    data: Vec<F>,
    ncols: usize,
}

impl<F> CopyOnlySource<F> {
    pub(crate) fn new(data: Vec<F>, ncols: usize) -> Self {
        Self { data, ncols }
    }
}

impl<F: Primitive> PointSource<F> for CopyOnlySource<F> {
    fn num_points(&self) -> usize {
        self.data.len() / self.ncols
    }

    fn num_columns(&self) -> usize {
        self.ncols
    }

    fn supports_view_batch(&self) -> bool {
        false
    }

    fn view_batch(&self, _start: usize, _count: usize) -> kmeans_uni::Result<&[F]> {
        unreachable!("copy-only source does not expose contiguous batch views")
    }

    fn read_batch(&self, start: usize, count: usize, dst: &mut [F]) {
        let len = count * self.ncols;
        let src_start = start * self.ncols;
        let src_end = src_start + len;
        dst[..len].copy_from_slice(&self.data[src_start..src_end]);
    }
}

pub(crate) fn single_thread_pool() -> &'static rayon::ThreadPool {
    SINGLE_THREAD_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("single-thread pool")
    })
}

fn get_data_generic<F: Primitive + SampleUniform>(
    storage: &'static OnceLock<Vec<F>>,
) -> &'static [F] {
    storage.get_or_init(|| {
        let mut rng = StdRng::seed_from_u64(SEED);
        let mut data = Vec::with_capacity(N_POINTS * N_COLS);
        let min = F::from(-100.0).unwrap();
        let max = F::from(100.0).unwrap();
        for _ in 0..(N_POINTS * N_COLS) {
            data.push(rng.gen_range(min..max));
        }
        data
    })
}

pub(crate) fn get_data_f32() -> &'static [f32] {
    get_data_generic(&DATA_F32)
}

pub(crate) fn get_data_f64() -> &'static [f64] {
    get_data_generic(&DATA_F64)
}

pub(crate) fn shape_data_f32(shape: Shape) -> Vec<f32> {
    let seed =
        SEED ^ ((shape.npoints as u64) << 32) ^ ((shape.ncols as u64) << 16) ^ shape.k as u64;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(shape.npoints * shape.ncols);
    for _ in 0..(shape.npoints * shape.ncols) {
        data.push(rng.gen_range(-100.0f32..100.0f32));
    }
    data
}

pub(crate) fn shape_data_f64(shape: Shape) -> Vec<f64> {
    let seed =
        SEED ^ ((shape.npoints as u64) << 32) ^ ((shape.ncols as u64) << 16) ^ shape.k as u64;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(shape.npoints * shape.ncols);
    for _ in 0..(shape.npoints * shape.ncols) {
        data.push(rng.gen_range(-100.0f64..100.0f64));
    }
    data
}

pub(crate) fn run_uni_full<F, B>(data: &[F], parallel: bool) -> kmeans_uni::KMeans<F>
where
    F: Primitive,
    B: CpuBackendType<F>,
{
    run_uni_full_shape::<F, B>(data, Shape::new(data.len() / N_COLS, N_COLS, K), parallel)
}

pub(crate) fn run_uni_full_shape<F, B>(
    data: &[F],
    shape: Shape,
    parallel: bool,
) -> kmeans_uni::KMeans<F>
where
    F: Primitive,
    B: CpuBackendType<F>,
{
    let base = KMeansBuilder::<F>::new(shape.k)
        .iterations(10)
        .backend::<B>()
        .euclidean()
        .seed(SEED);

    let source = SlicePointSource::new(data, shape.ncols).expect("slice source");
    if parallel {
        base.parallel()
            .fit_from_source(&source)
            .expect("kmeans fit")
    } else {
        base.build().fit_from_source(&source).expect("kmeans fit")
    }
}

#[cfg(feature = "wide")]
pub(crate) fn run_uni_full_shape_adaptive<F>(
    data: &[F],
    shape: Shape,
    parallel: bool,
) -> kmeans_uni::KMeans<F>
where
    F: Primitive,
    CPUSimdAdaptive: CpuBackendType<F>,
{
    run_uni_full_shape::<F, CPUSimdAdaptive>(data, shape, parallel)
}

pub(crate) fn run_uni_init_from_source<F, B, S>(source: &S, parallel: bool) -> kmeans_uni::KMeans<F>
where
    F: Primitive,
    B: CpuBackendType<F>,
    S: PointSource<F>,
{
    let base = KMeansBuilder::<F>::new(K)
        .iterations(0)
        .backend::<B>()
        .euclidean()
        .seed(SEED);

    if parallel {
        base.parallel()
            .fit_from_source(source)
            .expect("kmeans init")
    } else {
        base.build().fit_from_source(source).expect("kmeans init")
    }
}

pub(crate) fn run_uni_minibatch<F, B>(
    data: &[F],
    iterations: usize,
    parallel: bool,
) -> kmeans_uni::KMeans<F>
where
    F: Primitive,
    B: CpuBackendType<F>,
{
    let base = KMeansBuilder::<F>::new(K)
        .iterations(iterations)
        .backend::<B>()
        .euclidean()
        .seed(SEED)
        .mini_batch_rel_tolerance(0.0)
        .mini_batch_patience(0);

    let source = SlicePointSource::new(data, N_COLS).expect("slice source");
    if parallel {
        base.parallel()
            .fit_mini_batch_from_source(&source, BATCH)
            .expect("kmeans mini-batch fit")
    } else {
        base.build()
            .fit_mini_batch_from_source(&source, BATCH)
            .expect("kmeans mini-batch fit")
    }
}

pub(crate) fn get_uni_model_scalar_f32() -> (&'static kmeans_uni::KMeans<f32>, &'static [f32]) {
    let data = get_data_f32();
    let model = UNI_MODEL_SCALAR_F32.get_or_init(|| run_uni_full::<f32, CPUScalar>(data, false));
    (model, data)
}

pub(crate) fn get_uni_model_scalar_f64() -> (&'static kmeans_uni::KMeans<f64>, &'static [f64]) {
    let data = get_data_f64();
    let model = UNI_MODEL_SCALAR_F64.get_or_init(|| run_uni_full::<f64, CPUScalar>(data, false));
    (model, data)
}

#[cfg(feature = "wide")]
pub(crate) fn get_uni_model_simd_f32() -> (&'static kmeans_uni::KMeans<f32>, &'static [f32]) {
    let data = get_data_f32();
    let model = UNI_MODEL_SIMD_F32.get_or_init(|| run_uni_full::<f32, CPUSimd256>(data, false));
    (model, data)
}

#[cfg(feature = "wide")]
pub(crate) fn get_uni_model_simd_f64() -> (&'static kmeans_uni::KMeans<f64>, &'static [f64]) {
    let data = get_data_f64();
    let model = UNI_MODEL_SIMD_F64.get_or_init(|| run_uni_full::<f64, CPUSimd256>(data, false));
    (model, data)
}

pub(crate) fn linfa_dataset_f32<'a>(
    data: &'a [f32],
    shape: Shape,
) -> DatasetBase<ArrayView2<'a, f32>, ()> {
    let npoints = data.len() / shape.ncols;
    let array = ArrayView2::from_shape((npoints, shape.ncols), data).expect("array view");
    DatasetBase::new(array, ())
}

pub(crate) fn linfa_dataset_f64<'a>(
    data: &'a [f64],
    shape: Shape,
) -> DatasetBase<ArrayView2<'a, f64>, ()> {
    let npoints = data.len() / shape.ncols;
    let array = ArrayView2::from_shape((npoints, shape.ncols), data).expect("array view");
    DatasetBase::new(array, ())
}

pub(crate) fn get_linfa_model_f32() -> (
    &'static LinfaKMeans<f32, L2Dist>,
    DatasetBase<ArrayView2<'static, f32>, ()>,
) {
    let data = get_data_f32();
    let dataset = linfa_dataset_f32(data, Shape::new(data.len() / N_COLS, N_COLS, K));

    let model = LINFA_MODEL_F32.get_or_init(|| {
        let rng = StdRng::seed_from_u64(SEED);
        LinfaKMeans::params_with_rng(K, rng)
            .n_runs(1)
            .max_n_iterations(10)
            .tolerance(1e-4)
            .fit(&dataset)
            .expect("linfa model")
    });

    (model, dataset)
}

pub(crate) fn get_linfa_model_f64() -> (
    &'static LinfaKMeans<f64, L2Dist>,
    DatasetBase<ArrayView2<'static, f64>, ()>,
) {
    let data = get_data_f64();
    let dataset = linfa_dataset_f64(data, Shape::new(data.len() / N_COLS, N_COLS, K));

    let model = LINFA_MODEL_F64.get_or_init(|| {
        let rng = StdRng::seed_from_u64(SEED);
        LinfaKMeans::params_with_rng(K, rng)
            .n_runs(1)
            .max_n_iterations(10)
            .tolerance(1e-4)
            .fit(&dataset)
            .expect("linfa model")
    });

    (model, dataset)
}

pub(crate) fn run_linfa_full_f32(data: &[f32], parallel: bool) -> LinfaKMeans<f32, L2Dist> {
    run_linfa_full_shape_f32(data, Shape::new(data.len() / N_COLS, N_COLS, K), parallel)
}

pub(crate) fn run_linfa_full_shape_f32(
    data: &[f32],
    shape: Shape,
    parallel: bool,
) -> LinfaKMeans<f32, L2Dist> {
    let dataset = linfa_dataset_f32(data, shape);
    let params = LinfaKMeans::params_with_rng(shape.k, StdRng::seed_from_u64(SEED))
        .n_runs(1)
        .max_n_iterations(10)
        .tolerance(1e-4);

    if parallel {
        params.fit(&dataset).unwrap()
    } else {
        single_thread_pool().install(|| params.fit(&dataset).unwrap())
    }
}

pub(crate) fn run_linfa_full_f64(data: &[f64], parallel: bool) -> LinfaKMeans<f64, L2Dist> {
    run_linfa_full_shape_f64(data, Shape::new(data.len() / N_COLS, N_COLS, K), parallel)
}

pub(crate) fn run_linfa_full_shape_f64(
    data: &[f64],
    shape: Shape,
    parallel: bool,
) -> LinfaKMeans<f64, L2Dist> {
    let dataset = linfa_dataset_f64(data, shape);
    let params = LinfaKMeans::params_with_rng(shape.k, StdRng::seed_from_u64(SEED))
        .n_runs(1)
        .max_n_iterations(10)
        .tolerance(1e-4);

    if parallel {
        params.fit(&dataset).unwrap()
    } else {
        single_thread_pool().install(|| params.fit(&dataset).unwrap())
    }
}

pub(crate) fn run_linfa_minibatch_f32(data: &[f32]) -> LinfaKMeans<f32, L2Dist> {
    let npoints = data.len() / N_COLS;
    let array = ArrayView2::from_shape((npoints, N_COLS), data).unwrap();

    let clf = LinfaKMeans::params_with_rng(K, StdRng::seed_from_u64(SEED))
        .n_runs(1)
        .tolerance(1e-4);

    // Linfa exposes mini-batch through fit_with over caller-supplied batches. This is a
    // product-level reference, not an exact match for our internally sampled mini-batches.
    let mut model = None;
    for _ in 0..EPOCHS {
        for start in (0..npoints).step_by(BATCH) {
            let end = (start + BATCH).min(npoints);
            let batch = DatasetBase::new(array.slice(s![start..end, ..]), ());
            model = match clf.fit_with(model, black_box(&batch)) {
                Ok(m) => Some(m),
                Err(IncrKMeansError::NotConverged(m)) => Some(m),
                Err(e) => panic!("Linfa error: {:?}", e),
            };
        }
    }
    model.unwrap()
}

pub(crate) fn run_linfa_minibatch_f64(data: &[f64]) -> LinfaKMeans<f64, L2Dist> {
    let npoints = data.len() / N_COLS;
    let array = ArrayView2::from_shape((npoints, N_COLS), data).unwrap();

    let clf = LinfaKMeans::params_with_rng(K, StdRng::seed_from_u64(SEED))
        .n_runs(1)
        .tolerance(1e-4);

    // Linfa exposes mini-batch through fit_with over caller-supplied batches. This is a
    // product-level reference, not an exact match for our internally sampled mini-batches.
    let mut model = None;
    for _ in 0..EPOCHS {
        for start in (0..npoints).step_by(BATCH) {
            let end = (start + BATCH).min(npoints);
            let batch = DatasetBase::new(array.slice(s![start..end, ..]), ());
            model = match clf.fit_with(model, black_box(&batch)) {
                Ok(m) => Some(m),
                Err(IncrKMeansError::NotConverged(m)) => Some(m),
                Err(e) => panic!("Linfa error: {:?}", e),
            };
        }
    }
    model.unwrap()
}

pub(crate) fn warm_up() {
    let _ = get_data_f32();
    let _ = get_data_f64();
}
