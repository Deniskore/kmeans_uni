#![cfg(feature = "wide")]

#[path = "common/mod.rs"]
mod common_tests;

use kmeans_uni::{KMeansBuilder, SlicePointSource};
use rand08::{Rng, SeedableRng, rngs::StdRng};

const N_COLS: usize = 6;
const N_POINTS: usize = 2_000;
const K: usize = 16;
const FULL_ITERATIONS: usize = 10;
const MINI_ITERATIONS: usize = 50;
const MINI_BATCH_SIZE: usize = 1000;
const MINIBATCH_COST_MAX_PERCENT_DIFF: f32 = 5.0;
const MIN_INERTIA_EPS: f32 = 1e-6;
const MINIBATCH_CENTER_MAX_DIST: f32 = 0.1;

#[test]
fn mini_batch_simd_parity() {
    let data = build_dataset();
    let source = SlicePointSource::<f32>::new(&data, N_COLS).expect("slice source");

    let full_centroids = best_full(&source);
    let mini_centroids = best_mini(&source);
    let mini_centroids_par = best_mini_par(&source);

    let full_cost = common_tests::inertia_f64(&data, &full_centroids, N_COLS);
    let mini_cost = common_tests::inertia_f64(&data, &mini_centroids, N_COLS);
    let mini_cost_par = common_tests::inertia_f64(&data, &mini_centroids_par, N_COLS);
    let percent_diff = if full_cost.abs() < MIN_INERTIA_EPS as f64 {
        0.0
    } else {
        ((mini_cost - full_cost).abs() / full_cost.abs()) * 100.0
    };
    let percent_diff_par = if full_cost.abs() < MIN_INERTIA_EPS as f64 {
        0.0
    } else {
        ((mini_cost_par - full_cost).abs() / full_cost.abs()) * 100.0
    };
    let mini_parity = if mini_cost.abs() < MIN_INERTIA_EPS as f64 {
        0.0
    } else {
        ((mini_cost_par - mini_cost).abs() / mini_cost.abs()) * 100.0
    };
    eprintln!("Full cost={full_cost:.3}, Mini cost={mini_cost:.3}, diff={percent_diff:.2}%");
    eprintln!(
        "Mini cost par={mini_cost_par:.3}, diff vs full={percent_diff_par:.2}%, vs mini={mini_parity:.2}%"
    );
    assert!(
        percent_diff <= MINIBATCH_COST_MAX_PERCENT_DIFF as f64,
        "mini-batch (SIMD) deviated too far from full run ({percent_diff:.2}% > {MINIBATCH_COST_MAX_PERCENT_DIFF}%)"
    );
    assert!(
        percent_diff_par <= MINIBATCH_COST_MAX_PERCENT_DIFF as f64,
        "mini-batch parallel (SIMD) deviated too far from full run ({percent_diff_par:.2}% > {MINIBATCH_COST_MAX_PERCENT_DIFF}%)"
    );
    assert!(
        mini_parity <= MINIBATCH_COST_MAX_PERCENT_DIFF as f64,
        "mini-batch parallel (SIMD) deviated too far from sequential ({mini_parity:.2}% > {MINIBATCH_COST_MAX_PERCENT_DIFF}%)"
    );
    common_tests::assert_centroids_match_by_sorting(
        &full_centroids,
        &mini_centroids,
        N_COLS,
        MINIBATCH_CENTER_MAX_DIST as f64,
    );
    common_tests::assert_centroids_match_by_sorting(
        &full_centroids,
        &mini_centroids_par,
        N_COLS,
        MINIBATCH_CENTER_MAX_DIST as f64,
    );
}

fn best_full(source: &SlicePointSource<'_, f32>) -> Vec<f32> {
    let model = KMeansBuilder::<f32>::new(K)
        .iterations(FULL_ITERATIONS)
        .cpu_simd()
        .euclidean()
        .init_plus_plus()
        .seed(0xCAFEBABE)
        .build()
        .fit_from_source(source)
        .expect("full kmeans");
    model.into_centroids()
}

fn best_mini(source: &SlicePointSource<'_, f32>) -> Vec<f32> {
    let model = KMeansBuilder::<f32>::new(K)
        .iterations(MINI_ITERATIONS)
        .cpu_simd()
        .euclidean()
        .init_plus_plus()
        .seed(0xCAFEBABE)
        .build()
        .fit_mini_batch_from_source(source, MINI_BATCH_SIZE.max(1))
        .expect("mini batch kmeans");
    model.into_centroids()
}

fn best_mini_par(source: &SlicePointSource<'_, f32>) -> Vec<f32> {
    let model = KMeansBuilder::<f32>::new(K)
        .iterations(MINI_ITERATIONS)
        .cpu_simd()
        .euclidean()
        .init_plus_plus()
        .seed(0xCAFEBABE)
        .parallel()
        .fit_mini_batch_from_source(source, MINI_BATCH_SIZE.max(1))
        .expect("mini batch kmeans parallel");
    model.into_centroids()
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
