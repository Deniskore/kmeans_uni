use divan::{Bencher, black_box};
#[cfg(feature = "wide")]
use kmeans_uni::CPUSimd;
use kmeans_uni::{CPUScalar, CpuBackendType, KMeansBuilder, Primitive, SlicePointSource};
use linfa::DatasetBase;
use linfa::prelude::Predict;
use linfa::traits::{Fit, FitWith};
use linfa_clustering::{IncrKMeansError, KMeans as LinfaKMeans};
use linfa_nn::distance::L2Dist;
use ndarray::{Array1, ArrayView2};
use rand08::distributions::uniform::SampleUniform;
use rand08::{Rng, SeedableRng, rngs::StdRng};
use std::sync::OnceLock;

static DATA_F32: OnceLock<Vec<f32>> = OnceLock::new();
static DATA_F64: OnceLock<Vec<f64>> = OnceLock::new();

static UNI_MODEL_SCALAR_F32: OnceLock<kmeans_uni::KMeans<f32>> = OnceLock::new();
static UNI_MODEL_SCALAR_F64: OnceLock<kmeans_uni::KMeans<f64>> = OnceLock::new();
#[cfg(feature = "wide")]
static UNI_MODEL_SIMD_F32: OnceLock<kmeans_uni::KMeans<f32>> = OnceLock::new();
#[cfg(feature = "wide")]
static UNI_MODEL_SIMD_F64: OnceLock<kmeans_uni::KMeans<f64>> = OnceLock::new();
static LINFA_MODEL_F32: OnceLock<LinfaKMeans<f32, L2Dist>> = OnceLock::new();
static LINFA_MODEL_F64: OnceLock<LinfaKMeans<f64, L2Dist>> = OnceLock::new();

const N_COLS: usize = 16;
const K: usize = 16;
// 256KB buffer / 4 bytes per float = 64k floats
// 64k floats / 16 cols = 4096 points
const N_POINTS: usize = 65536;
const BATCH: usize = 256;
const EPOCHS: usize = 10;
const SEED: u64 = 42;

fn get_data_generic<F: Primitive + SampleUniform>(
    storage: &'static OnceLock<Vec<F>>,
) -> &'static [F] {
    storage.get_or_init(|| {
        let mut rng = StdRng::seed_from_u64(42);
        let mut data = Vec::with_capacity(N_POINTS * N_COLS);
        let min = F::from(-100.0).unwrap();
        let max = F::from(100.0).unwrap();
        for _ in 0..(N_POINTS * N_COLS) {
            data.push(rng.gen_range(min..max));
        }
        data
    })
}

fn get_data_f32() -> &'static [f32] {
    get_data_generic(&DATA_F32)
}

fn get_data_f64() -> &'static [f64] {
    get_data_generic(&DATA_F64)
}

fn run_uni_full<F, B>(data: &[F], parallel: bool) -> kmeans_uni::KMeans<F>
where
    F: Primitive,
    B: CpuBackendType<F>,
{
    let base = KMeansBuilder::new(K)
        .iterations(10)
        .backend::<B>()
        .euclidean()
        .seed(SEED);

    if parallel {
        base.parallel()
            .fit_from_source(&SlicePointSource::new(data, N_COLS).expect("slice source"))
            .expect("kmeans fit")
    } else {
        base.build()
            .fit_from_source(&SlicePointSource::new(data, N_COLS).expect("slice source"))
            .expect("kmeans fit")
    }
}

fn run_uni_minibatch<F, B>(data: &[F], iterations: usize, parallel: bool) -> kmeans_uni::KMeans<F>
where
    F: Primitive,
    B: CpuBackendType<F>,
{
    let base = KMeansBuilder::new(K)
        .iterations(iterations)
        .backend::<B>()
        .euclidean()
        .seed(SEED)
        .mini_batch_rel_tolerance(0.0)
        .mini_batch_patience(0);

    if parallel {
        base.parallel()
            .fit_mini_batch_from_source(
                &SlicePointSource::new(data, N_COLS).expect("slice source"),
                BATCH,
            )
            .expect("kmeans mini-batch fit")
    } else {
        base.build()
            .fit_mini_batch_from_source(
                &SlicePointSource::new(data, N_COLS).expect("slice source"),
                BATCH,
            )
            .expect("kmeans mini-batch fit")
    }
}

fn get_uni_model_scalar_f32() -> (&'static kmeans_uni::KMeans<f32>, &'static [f32]) {
    let data = get_data_f32();
    let model = UNI_MODEL_SCALAR_F32.get_or_init(|| run_uni_full::<f32, CPUScalar>(data, false));
    (model, data)
}

fn get_uni_model_scalar_f64() -> (&'static kmeans_uni::KMeans<f64>, &'static [f64]) {
    let data = get_data_f64();
    let model = UNI_MODEL_SCALAR_F64.get_or_init(|| run_uni_full::<f64, CPUScalar>(data, false));
    (model, data)
}

#[cfg(feature = "wide")]
fn get_uni_model_simd_f32() -> (&'static kmeans_uni::KMeans<f32>, &'static [f32]) {
    let data = get_data_f32();
    let model = UNI_MODEL_SIMD_F32.get_or_init(|| run_uni_full::<f32, CPUSimd>(data, false));
    (model, data)
}

#[cfg(feature = "wide")]
fn get_uni_model_simd_f64() -> (&'static kmeans_uni::KMeans<f64>, &'static [f64]) {
    let data = get_data_f64();
    let model = UNI_MODEL_SIMD_F64.get_or_init(|| run_uni_full::<f64, CPUSimd>(data, false));
    (model, data)
}

fn get_linfa_model_f32() -> (
    &'static LinfaKMeans<f32, L2Dist>,
    DatasetBase<ArrayView2<'static, f32>, ()>,
) {
    let data = get_data_f32();
    let npoints = data.len() / N_COLS;
    let array = ArrayView2::from_shape((npoints, N_COLS), data).expect("array view");
    let dataset = DatasetBase::new(array, ());

    let model = LINFA_MODEL_F32.get_or_init(|| {
        let rng = StdRng::seed_from_u64(SEED);
        LinfaKMeans::params_with_rng(K, rng)
            .n_runs(1)
            .max_n_iterations(10)
            .tolerance(1e-4)
            .fit(&dataset)
            .expect("linfa model")
    });

    (model, DatasetBase::new(array, ()))
}

fn get_linfa_model_f64() -> (
    &'static LinfaKMeans<f64, L2Dist>,
    DatasetBase<ArrayView2<'static, f64>, ()>,
) {
    let data = get_data_f64();
    let npoints = data.len() / N_COLS;
    let array = ArrayView2::from_shape((npoints, N_COLS), data).expect("array view");
    let dataset = DatasetBase::new(array, ());

    let model = LINFA_MODEL_F64.get_or_init(|| {
        let rng = StdRng::seed_from_u64(SEED);
        LinfaKMeans::params_with_rng(K, rng)
            .n_runs(1)
            .max_n_iterations(10)
            .tolerance(1e-4)
            .fit(&dataset)
            .expect("linfa model")
    });

    (model, DatasetBase::new(array, ()))
}

fn run_linfa_full_f32(data: &[f32], parallel: bool) -> LinfaKMeans<f32, L2Dist> {
    let npoints = data.len() / N_COLS;
    let array = ArrayView2::from_shape((npoints, N_COLS), data).unwrap();
    let dataset = DatasetBase::new(array, ());

    let params = LinfaKMeans::params_with_rng(K, StdRng::seed_from_u64(SEED))
        .n_runs(1)
        .max_n_iterations(10)
        .tolerance(1e-4);

    if parallel {
        params.fit(&dataset).unwrap()
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| params.fit(&dataset).unwrap())
    }
}

fn run_linfa_full_f64(data: &[f64], parallel: bool) -> LinfaKMeans<f64, L2Dist> {
    let npoints = data.len() / N_COLS;
    let array = ArrayView2::from_shape((npoints, N_COLS), data).unwrap();
    let dataset = DatasetBase::new(array, ());

    let params = LinfaKMeans::params_with_rng(K, StdRng::seed_from_u64(SEED))
        .n_runs(1)
        .max_n_iterations(10)
        .tolerance(1e-4);

    if parallel {
        params.fit(&dataset).unwrap()
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| params.fit(&dataset).unwrap())
    }
}

fn run_linfa_minibatch_f32(data: &[f32]) -> LinfaKMeans<f32, L2Dist> {
    let npoints = data.len() / N_COLS;
    let array = ArrayView2::from_shape((npoints, N_COLS), data).unwrap();
    let targets = Array1::<f32>::zeros(npoints);
    let dataset = DatasetBase::new(array, targets);

    let clf = LinfaKMeans::params_with_rng(K, StdRng::seed_from_u64(SEED))
        .n_runs(1)
        .tolerance(1e-4);

    let mut model = None;
    for _ in 0..EPOCHS {
        for batch in dataset.sample_chunks(BATCH) {
            model = match clf.fit_with(model, black_box(&batch)) {
                Ok(m) => Some(m),
                Err(IncrKMeansError::NotConverged(m)) => Some(m),
                Err(e) => panic!("Linfa error: {:?}", e),
            };
        }
    }
    model.unwrap()
}

fn run_linfa_minibatch_f64(data: &[f64]) -> LinfaKMeans<f64, L2Dist> {
    let npoints = data.len() / N_COLS;
    let array = ArrayView2::from_shape((npoints, N_COLS), data).unwrap();
    let targets = Array1::<f64>::zeros(npoints);
    let dataset = DatasetBase::new(array, targets);

    let clf = LinfaKMeans::params_with_rng(K, StdRng::seed_from_u64(SEED))
        .n_runs(1)
        .tolerance(1e-4);

    let mut model = None;
    for _ in 0..EPOCHS {
        for batch in dataset.sample_chunks(BATCH) {
            model = match clf.fit_with(model, black_box(&batch)) {
                Ok(m) => Some(m),
                Err(IncrKMeansError::NotConverged(m)) => Some(m),
                Err(e) => panic!("Linfa error: {:?}", e),
            };
        }
    }
    model.unwrap()
}

fn main() {
    let _ = get_data_f32();
    let _ = get_data_f64();
    // Ensure models are trained before benchmarking predict to keep timings to inference only.
    let _ = get_uni_model_scalar_f32();
    let _ = get_uni_model_scalar_f64();
    #[cfg(feature = "wide")]
    let _ = get_uni_model_simd_f32();
    #[cfg(feature = "wide")]
    let _ = get_uni_model_simd_f64();
    let _ = get_linfa_model_f32();
    let _ = get_linfa_model_f64();
    divan::main();
}

#[divan::bench]
fn f32_uni_minibatch_scalar_seq(bencher: Bencher) {
    let data = get_data_f32();
    let iterations = (N_POINTS / BATCH) * EPOCHS;
    bencher.bench_local(|| run_uni_minibatch::<f32, CPUScalar>(black_box(data), iterations, false));
}

#[divan::bench]
fn f32_uni_minibatch_scalar_par(bencher: Bencher) {
    let data = get_data_f32();
    let iterations = (N_POINTS / BATCH) * EPOCHS;
    bencher.bench_local(|| run_uni_minibatch::<f32, CPUScalar>(black_box(data), iterations, true));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_minibatch_simd_seq(bencher: Bencher) {
    let data = get_data_f32();
    let iterations = (N_POINTS / BATCH) * EPOCHS;
    bencher.bench_local(|| run_uni_minibatch::<f32, CPUSimd>(black_box(data), iterations, false));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_minibatch_simd_par(bencher: Bencher) {
    let data = get_data_f32();
    let iterations = (N_POINTS / BATCH) * EPOCHS;
    bencher.bench_local(|| run_uni_minibatch::<f32, CPUSimd>(black_box(data), iterations, true));
}

#[divan::bench]
fn f32_uni_scalar_seq(bencher: Bencher) {
    let data = get_data_f32();
    bencher.bench_local(|| run_uni_full::<f32, CPUScalar>(black_box(data), false));
}

#[divan::bench]
fn f32_uni_scalar_par(bencher: Bencher) {
    let data = get_data_f32();
    bencher.bench_local(|| run_uni_full::<f32, CPUScalar>(black_box(data), true));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_simd_seq(bencher: Bencher) {
    let data = get_data_f32();
    bencher.bench_local(|| run_uni_full::<f32, CPUSimd>(black_box(data), false));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_simd_par(bencher: Bencher) {
    let data = get_data_f32();
    bencher.bench_local(|| run_uni_full::<f32, CPUSimd>(black_box(data), true));
}

#[divan::bench]
fn f32_uni_predict_scalar(bencher: Bencher) {
    let (model, data) = get_uni_model_scalar_f32();
    bencher.bench_local(|| black_box(model).predict(black_box(data)).unwrap());
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_predict_simd(bencher: Bencher) {
    let (model, data) = get_uni_model_simd_f32();
    bencher.bench_local(|| black_box(model).predict(black_box(data)).unwrap());
}

#[divan::bench]
fn f32_linfa_predict(bencher: Bencher) {
    let (model, dataset) = get_linfa_model_f32();
    bencher.bench_local(|| black_box(model).predict(black_box(&dataset)));
}

#[divan::bench]
fn f32_linfa_scalar_par(bencher: Bencher) {
    let data = get_data_f32();
    bencher.bench_local(|| run_linfa_full_f32(black_box(data), true));
}

#[divan::bench]
fn f32_linfa_scalar_seq(bencher: Bencher) {
    let data = get_data_f32();
    bencher.bench_local(|| run_linfa_full_f32(black_box(data), false));
}

#[divan::bench]
fn f32_linfa_minibatch_scalar_par(bencher: Bencher) {
    let data = get_data_f32();
    bencher.bench_local(|| run_linfa_minibatch_f32(black_box(data)));
}

#[divan::bench]
fn f64_uni_minibatch_scalar_seq(bencher: Bencher) {
    let data = get_data_f64();
    let iterations = (N_POINTS / BATCH) * EPOCHS;
    bencher.bench_local(|| run_uni_minibatch::<f64, CPUScalar>(black_box(data), iterations, false));
}

#[divan::bench]
fn f64_uni_minibatch_scalar_par(bencher: Bencher) {
    let data = get_data_f64();
    let iterations = (N_POINTS / BATCH) * EPOCHS;
    bencher.bench_local(|| run_uni_minibatch::<f64, CPUScalar>(black_box(data), iterations, true));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_minibatch_simd_seq(bencher: Bencher) {
    let data = get_data_f64();
    let iterations = (N_POINTS / BATCH) * EPOCHS;
    bencher.bench_local(|| run_uni_minibatch::<f64, CPUSimd>(black_box(data), iterations, false));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_minibatch_simd_par(bencher: Bencher) {
    let data = get_data_f64();
    let iterations = (N_POINTS / BATCH) * EPOCHS;
    bencher.bench_local(|| run_uni_minibatch::<f64, CPUSimd>(black_box(data), iterations, true));
}

#[divan::bench]
fn f64_uni_scalar_seq(bencher: Bencher) {
    let data = get_data_f64();
    bencher.bench_local(|| run_uni_full::<f64, CPUScalar>(black_box(data), false));
}

#[divan::bench]
fn f64_uni_scalar_par(bencher: Bencher) {
    let data = get_data_f64();
    bencher.bench_local(|| run_uni_full::<f64, CPUScalar>(black_box(data), true));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_simd_seq(bencher: Bencher) {
    let data = get_data_f64();
    bencher.bench_local(|| run_uni_full::<f64, CPUSimd>(black_box(data), false));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_simd_par(bencher: Bencher) {
    let data = get_data_f64();
    bencher.bench_local(|| run_uni_full::<f64, CPUSimd>(black_box(data), true));
}

#[divan::bench]
fn f64_uni_predict_scalar(bencher: Bencher) {
    let (model, data) = get_uni_model_scalar_f64();
    bencher.bench_local(|| black_box(model).predict(black_box(data)).unwrap());
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_predict_simd(bencher: Bencher) {
    let (model, data) = get_uni_model_simd_f64();
    bencher.bench_local(|| black_box(model).predict(black_box(data)).unwrap());
}

#[divan::bench]
fn f64_linfa_predict(bencher: Bencher) {
    let (model, dataset) = get_linfa_model_f64();
    bencher.bench_local(|| black_box(model).predict(black_box(&dataset)));
}

#[divan::bench]
fn f64_linfa_scalar_par(bencher: Bencher) {
    let data = get_data_f64();
    bencher.bench_local(|| run_linfa_full_f64(black_box(data), true));
}

#[divan::bench]
fn f64_linfa_scalar_seq(bencher: Bencher) {
    let data = get_data_f64();
    bencher.bench_local(|| run_linfa_full_f64(black_box(data), false));
}

#[divan::bench]
fn f64_linfa_minibatch_scalar_par(bencher: Bencher) {
    let data = get_data_f64();
    bencher.bench_local(|| run_linfa_minibatch_f64(black_box(data)));
}
