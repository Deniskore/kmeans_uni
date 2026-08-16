#![cfg(not(feature = "wasm"))]

use kmeans_uni::{CPUScalar, KMeans, MetricType};

#[test]
fn balanced_parallel_prediction_matches_sequential_at_thresholds() {
    const THREADS: usize = 4;
    const N_COLS: usize = 3;
    const K: usize = 17;
    const MIN_POINTS_PER_TASK: usize = 128;

    let centroids = deterministic_values(K * N_COLS, 0.25);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(THREADS)
        .build()
        .expect("test thread pool");

    pool.install(|| {
        for metric in [MetricType::Euclidean, MetricType::DotProduct] {
            let model = KMeans::new(centroids.clone(), N_COLS, K, 0.0, metric).unwrap();

            for npoints in [
                THREADS * MIN_POINTS_PER_TASK - 1,
                THREADS * MIN_POINTS_PER_TASK,
                THREADS * MIN_POINTS_PER_TASK + 1,
            ] {
                let points = deterministic_values(npoints * N_COLS, 0.125);
                let expected = model
                    .predict_with_backend_sequential::<CPUScalar>(&points)
                    .unwrap();
                let actual = model.predict_with_backend::<CPUScalar>(&points).unwrap();

                assert_eq!(actual, expected, "metric={metric:?}, npoints={npoints}");
            }
        }
    });
}

fn deterministic_values(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let base = ((i * 37 + 11) % 101) as f32 - 50.0;
            base * scale + (i % 7) as f32 * 0.001
        })
        .collect()
}
