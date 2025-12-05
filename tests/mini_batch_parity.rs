#[path = "common/mod.rs"]
mod common_tests;

use kmeans_uni::{KMeansBuilder, SlicePointSource};
use linfa::DatasetBase;
use linfa::traits::FitWith;
use linfa_clustering::{IncrKMeansError, KMeans as LinfaKMeans};
use ndarray::{Array1, ArrayView2, s};
use rand08::{Rng, SeedableRng, rngs::StdRng};

const _TARGET_BYTES: usize = 10 * 1024 * 1024;
const N_COLS: usize = 6;
const _FLOAT_BYTES: usize = std::mem::size_of::<f32>();
const _TOTAL_FLOATS: usize = _TARGET_BYTES / _FLOAT_BYTES;
const N_POINTS: usize = 2_000;
const K: usize = 16;
const FULL_ITERATIONS: usize = 10;
const MINI_ITERATIONS: usize = 30;
const MINI_BATCH_SIZE: usize = {
    let size = N_POINTS / 4;
    if size == 0 { 1 } else { size }
};
const MINI_EPOCHS: usize = 3;
const MINIBATCH_COST_MAX_PERCENT_DIFF: f32 = 5.0;
const MINIBATCH_CENTER_MAX_DIST: f32 = 0.1;
const MIN_INERTIA_EPS: f32 = 1e-6;

#[test]
fn mini_batch_matches_full_on_large_buffer() {
    let data = build_dataset();

    let full_centroids = best_full(&data, K, FULL_ITERATIONS, 0xCAFEBABE);
    let mini_centroids = best_mini(&data, K, MINI_ITERATIONS, MINI_BATCH_SIZE, 0xCAFEBABE);
    let linfa_centroids = best_linfa_mini(&data, K, MINI_BATCH_SIZE, MINI_EPOCHS, 0xCAFEBABE);

    let full_cost = common_tests::inertia_f64(&data, &full_centroids, N_COLS);
    let mini_cost = common_tests::inertia_f64(&data, &mini_centroids, N_COLS);
    let linfa_cost = common_tests::inertia_f64(&data, &linfa_centroids, N_COLS);
    let percent_diff = if full_cost.abs() < MIN_INERTIA_EPS as f64 {
        0.0
    } else {
        ((mini_cost - full_cost).abs() / full_cost.abs()) * 100.0
    };
    let linfa_percent_diff = if full_cost.abs() < MIN_INERTIA_EPS as f64 {
        0.0
    } else {
        ((linfa_cost - full_cost).abs() / full_cost.abs()) * 100.0
    };
    eprintln!("Full cost={full_cost:.3}, Mini cost={mini_cost:.3}, diff={percent_diff:.2}%");
    eprintln!("Linfa mini cost={linfa_cost:.3}, diff={linfa_percent_diff:.2}%");
    assert!(
        percent_diff <= MINIBATCH_COST_MAX_PERCENT_DIFF as f64,
        "mini-batch deviated too far from full run ({percent_diff:.2}% > {MINIBATCH_COST_MAX_PERCENT_DIFF}%)"
    );
    assert!(
        linfa_percent_diff <= MINIBATCH_COST_MAX_PERCENT_DIFF as f64,
        "linfa mini-batch deviated too far from full run ({linfa_percent_diff:.2}% > {MINIBATCH_COST_MAX_PERCENT_DIFF}%)"
    );

    common_tests::assert_centroids_match_by_sorting(
        &full_centroids,
        &mini_centroids,
        N_COLS,
        MINIBATCH_CENTER_MAX_DIST as f64,
    );
    common_tests::assert_centroids_match_by_sorting(
        &full_centroids,
        &linfa_centroids,
        N_COLS,
        MINIBATCH_CENTER_MAX_DIST as f64,
    );
}

fn best_full(points: &[f32], k: usize, iterations: usize, seed: u64) -> Vec<f32> {
    let source = SlicePointSource::new(points, N_COLS).unwrap();
    let model = KMeansBuilder::<f32>::new(k)
        .iterations(iterations)
        .cpu_scalar()
        .euclidean()
        .init_plus_plus()
        .seed(seed)
        .build()
        .fit_from_source(&source)
        .unwrap();
    model.into_centroids()
}

fn best_mini(
    points: &[f32],
    k: usize,
    iterations: usize,
    batch_size: usize,
    seed: u64,
) -> Vec<f32> {
    let source = SlicePointSource::new(points, N_COLS).unwrap();
    let model = KMeansBuilder::<f32>::new(k)
        .iterations(iterations)
        .cpu_scalar()
        .euclidean()
        .init_plus_plus()
        .seed(seed)
        .attempts(5)
        .build()
        .fit_mini_batch_from_source(&source, batch_size)
        .unwrap();
    model.into_centroids()
}

fn best_linfa_mini(
    points: &[f32],
    k: usize,
    batch_size: usize,
    epochs: usize,
    seed: u64,
) -> Vec<f32> {
    let npoints = points.len() / N_COLS;
    let array = ArrayView2::from_shape((npoints, N_COLS), points).unwrap();
    let targets = Array1::<f32>::zeros(npoints);

    let mut model = None;
    let rng = StdRng::seed_from_u64(seed);
    for _ in 0..epochs {
        for start in (0..npoints).step_by(batch_size) {
            let end = (start + batch_size).min(npoints);
            let batch = DatasetBase::new(
                array.slice(s![start..end, ..]).to_owned(),
                targets.slice(s![start..end]).to_owned(),
            );
            let clf = LinfaKMeans::params_with_rng(k, rng.clone())
                .n_runs(1)
                .tolerance(1e-4);
            model = match clf.fit_with(model, &batch) {
                Ok(m) => Some(m),
                Err(IncrKMeansError::NotConverged(m)) => Some(m),
                Err(e) => panic!("Linfa error: {:?}", e),
            };
        }
    }
    let model = model.expect("linfa mini-batch model");
    model.centroids().iter().cloned().collect()
}

fn build_dataset() -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(0x0BAD_5EED);
    let mut data = Vec::with_capacity(N_POINTS * N_COLS);
    for idx in 0..N_POINTS {
        let cluster = (idx % K) as f32;
        for comp in 0..N_COLS {
            let base = cluster * 5.0 + comp as f32;
            let noise: f32 = rng.gen_range(-0.02..0.02);
            data.push(base + noise);
        }
    }
    data
}
