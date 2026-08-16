#[path = "common/mod.rs"]
mod common_tests;

use kmeans_uni::{CPUScalar, CpuBackendType, KMeansBuilder, Primitive, SlicePointSource};
#[cfg(feature = "wide")]
use kmeans_uni::{CPUSimd128, CPUSimd256, CPUSimd512, CPUSimdAdaptive};

use linfa::DatasetBase;
use linfa::traits::{Fit, FitWith};
use linfa_clustering::{IncrKMeansError, KMeans as LinfaKMeans};
use ndarray::{Array1, Array2, s};
use rand08::distributions::uniform::SampleUniform;
use rand08::{Rng, SeedableRng, rngs::StdRng};

const N_COLS: usize = 3;
const K: usize = 4;
const TARGET_BYTES: usize = 256 * 1024;
const RANDOM_POINTS: usize = 2000;

const CLUSTERED_ITERATIONS: usize = 500;
const RANDOM_ITERATIONS: usize = 200;
const CLUSTERED_CENTER_MAX_DIST: f64 = 5e-3;
const RANDOM_COST_MAX_PERCENT_DIFF: f64 = 2.0;
const MINIBATCH_COST_MAX_PERCENT_DIFF: f64 = 5.0;

#[test]
fn kmeans_f32_scalar_matches_linfa_on_clustered_data() {
    clustered_parity::<f32, CPUScalar>(1234);
}

#[cfg(feature = "wide")]
#[test]
fn kmeans_f32_simd128_matches_linfa_on_clustered_data() {
    clustered_parity::<f32, CPUSimd128>(4567);
}

#[cfg(feature = "wide")]
#[test]
fn kmeans_f32_adaptive_simd_matches_linfa_on_clustered_data() {
    clustered_parity::<f32, CPUSimdAdaptive>(4567);
}

#[cfg(feature = "wide")]
#[test]
fn kmeans_f32_simd_matches_linfa_on_clustered_data() {
    clustered_parity::<f32, CPUSimd256>(4567);
}

#[cfg(feature = "wide")]
#[test]
fn kmeans_f32_simd512_matches_linfa_on_clustered_data() {
    clustered_parity::<f32, CPUSimd512>(4567);
}

#[test]
fn kmeans_f64_scalar_matches_linfa_on_clustered_data() {
    clustered_parity::<f64, CPUScalar>(8901);
}

#[cfg(feature = "wide")]
#[test]
fn kmeans_f64_simd128_matches_linfa_on_clustered_data() {
    clustered_parity::<f64, CPUSimd128>(1357);
}

#[cfg(feature = "wide")]
#[test]
fn kmeans_f64_adaptive_simd_matches_linfa_on_clustered_data() {
    clustered_parity::<f64, CPUSimdAdaptive>(1357);
}

#[cfg(feature = "wide")]
#[test]
fn kmeans_f64_simd_matches_linfa_on_clustered_data() {
    clustered_parity::<f64, CPUSimd256>(1357);
}

#[cfg(feature = "wide")]
#[test]
fn kmeans_f64_simd512_matches_linfa_on_clustered_data() {
    clustered_parity::<f64, CPUSimd512>(1357);
}

#[test]
fn kmeans_f32_scalar_cost_matches_linfa_on_random_data() {
    random_cost_parity::<f32, CPUScalar>();
}

#[test]
fn kmeans_f64_scalar_cost_matches_linfa_on_random_data() {
    random_cost_parity::<f64, CPUScalar>();
}

#[test]
fn minibatch_f32_scalar_cost_matches_linfa() {
    minibatch_parity::<f32, CPUScalar>();
}

#[test]
fn minibatch_f64_scalar_cost_matches_linfa() {
    minibatch_parity::<f64, CPUScalar>();
}

fn clustered_parity<F, B>(seed: u64)
where
    F: Primitive + SampleUniform,
    B: CpuBackendType<F>,
{
    let data = random_clustered_data::<F>();

    let sog_centroids = fit_cpu_backend::<F, B>(&data, CLUSTERED_ITERATIONS, seed);
    let linfa_centroids = fit_linfa_kmeans(&data, CLUSTERED_ITERATIONS, seed);

    common_tests::assert_centroids_match_by_sorting(
        &sog_centroids,
        &linfa_centroids,
        N_COLS,
        CLUSTERED_CENTER_MAX_DIST,
    );
}

fn random_cost_parity<F, B>()
where
    F: Primitive + SampleUniform,
    B: CpuBackendType<F>,
{
    let data = random_uniform_data::<F>(0x0BAD_5EED);

    let sog_centroids = fit_cpu_backend::<F, B>(&data, RANDOM_ITERATIONS, 0xCAFEBABE);
    let linfa_centroids = fit_linfa_kmeans(&data, RANDOM_ITERATIONS, 0xCAFEBABE);

    let sog_cost = common_tests::inertia_f64(&data, &sog_centroids, N_COLS);
    let linfa_cost = common_tests::inertia_f64(&data, &linfa_centroids, N_COLS);

    let diff_percent = ((sog_cost - linfa_cost).abs() / linfa_cost) * 100.0;

    assert!(
        diff_percent < RANDOM_COST_MAX_PERCENT_DIFF,
        "Costs diverged! Sog: {}, Linfa: {}, Diff: {:.2}%",
        sog_cost,
        linfa_cost,
        diff_percent
    );
}

fn minibatch_parity<F, B>()
where
    F: Primitive + SampleUniform,
    B: CpuBackendType<F>,
{
    let data = random_uniform_data::<F>(0x0BAD_5EED);
    let batch_size = 256;
    let iterations = 10;

    let sog_centroids = fit_sog_minibatch::<F, B>(&data, iterations, batch_size, 0xCAFEBABE);
    let linfa_centroids = fit_linfa_minibatch(&data, iterations, batch_size, 0xCAFEBABE);

    let sog_cost = common_tests::inertia_f64(&data, &sog_centroids, N_COLS);
    let linfa_cost = common_tests::inertia_f64(&data, &linfa_centroids, N_COLS);

    let diff_percent = ((sog_cost - linfa_cost).abs() / linfa_cost) * 100.0;

    assert!(
        diff_percent < MINIBATCH_COST_MAX_PERCENT_DIFF,
        "MiniBatch costs diverged! Sog: {}, Linfa: {}, Diff: {:.2}%",
        sog_cost,
        linfa_cost,
        diff_percent
    );
}

fn fit_sog_minibatch<F, B>(data: &[F], epochs: usize, batch_size: usize, seed: u64) -> Vec<F>
where
    F: Primitive,
    B: CpuBackendType<F>,
{
    let n_points = data.len() / N_COLS;
    let batches_per_epoch = n_points.div_ceil(batch_size);
    let total_steps = epochs * batches_per_epoch;

    let model = KMeansBuilder::new(K)
        .iterations(total_steps)
        .backend::<B>()
        .euclidean()
        .init_plus_plus()
        .seed(seed)
        .build()
        .fit_mini_batch_from_source(
            &SlicePointSource::new(data, N_COLS).expect("slice source"),
            batch_size,
        )
        .expect("kmeans mini-batch fit");
    model.into_centroids()
}

fn fit_linfa_minibatch<F: Primitive>(
    data: &[F],
    iterations: usize,
    batch_size: usize,
    seed: u64,
) -> Vec<F> {
    let npoints = data.len() / N_COLS;
    let data_f64 = to_f64_vec(data);
    let array = Array2::from_shape_vec((npoints, N_COLS), data_f64).expect("array shape match");
    // Dummy targets for chunking
    let targets = Array1::<f32>::zeros(npoints);

    let rng = StdRng::seed_from_u64(seed);
    let clf = LinfaKMeans::params_with_rng(K, rng).tolerance(1e-4);

    let mut model = None;
    for _ in 0..iterations {
        for start in (0..npoints).step_by(batch_size) {
            let end = (start + batch_size).min(npoints);
            let batch = DatasetBase::new(
                array.slice(s![start..end, ..]).to_owned(),
                targets.slice(s![start..end]).to_owned(),
            );
            model = match clf.fit_with(model, &batch) {
                Ok(m) => Some(m),
                Err(IncrKMeansError::NotConverged(m)) => Some(m),
                Err(e) => panic!("Linfa error: {:?}", e),
            };
        }
    }
    model
        .unwrap()
        .centroids()
        .mapv(|x| F::from(x).unwrap())
        .into_raw_vec_and_offset()
        .0
}

fn to_f64_vec<F: Primitive>(values: &[F]) -> Vec<f64> {
    values.iter().map(|v| v.to_f64().unwrap()).collect()
}

fn fit_cpu_backend<F, B>(data: &[F], iterations: usize, seed: u64) -> Vec<F>
where
    F: Primitive,
    B: CpuBackendType<F>,
{
    let source = SlicePointSource::<F>::new(data, N_COLS).expect("slice source");
    let model = KMeansBuilder::<F>::new(K)
        .iterations(iterations)
        .backend::<B>()
        .euclidean()
        .init_plus_plus()
        .seed(seed)
        .build()
        .fit_from_source(&source)
        .expect("kmeans fit");
    model.into_centroids()
}

fn fit_linfa_kmeans<F: Primitive>(data: &[F], iterations: usize, seed: u64) -> Vec<F> {
    let npoints = data.len() / N_COLS;
    let data_f64 = to_f64_vec(data);
    let array = Array2::from_shape_vec((npoints, N_COLS), data_f64).expect("array shape match");
    let dataset = DatasetBase::new(array, ());

    let rng = StdRng::seed_from_u64(seed);
    LinfaKMeans::params_with_rng(K, rng)
        .max_n_iterations(iterations as u64)
        .tolerance(1e-5)
        .fit(&dataset)
        .expect("linfa failed")
        .centroids()
        .mapv(|x| F::from(x).unwrap())
        .into_raw_vec_and_offset()
        .0
}

fn random_clustered_data<F>() -> Vec<F>
where
    F: Primitive + SampleUniform,
{
    const CENTERS: [[f64; N_COLS]; K] = [
        [0.0, 0.0, 0.0],
        [250.0, 0.0, 0.0],
        [0.0, 250.0, 0.0],
        [0.0, 0.0, 250.0],
    ];

    let points_per_cluster = points_per_cluster_for::<F>();
    let mut rng = StdRng::seed_from_u64(0x1234_5678);
    let mut data = Vec::with_capacity(points_per_cluster * K * N_COLS);

    for center in CENTERS {
        for _ in 0..points_per_cluster {
            for &val in center.iter() {
                let noise: f64 = rng.gen_range(-10.0..10.0);
                data.push(F::from(val + noise).unwrap());
            }
        }
    }
    data
}

fn points_per_cluster_for<F>() -> usize {
    let bytes_per_point = N_COLS * std::mem::size_of::<F>();
    let total_points = TARGET_BYTES / bytes_per_point;
    (total_points / K) & !1
}

fn random_uniform_data<F>(seed: u64) -> Vec<F>
where
    F: Primitive + SampleUniform,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Vec::with_capacity(RANDOM_POINTS * N_COLS);
    for _ in 0..(RANDOM_POINTS * N_COLS) {
        let value: F = rng.gen_range(F::from(-500.0).unwrap()..F::from(500.0).unwrap());
        data.push(value);
    }
    data
}
