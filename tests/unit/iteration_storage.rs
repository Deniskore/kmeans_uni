use super::*;
use crate::backend::Euclidean;
use crate::kmeans_core_scalar::ScalarBackend;
use crate::point_source::SlicePointSource;

fn fixture() -> (SlicePointSource<'static, f32>, Vec<f32>) {
    static POINTS: [f32; 16] = [
        0.0, 0.0, 0.2, 0.1, 1.0, 1.0, 1.2, 0.9, 5.0, 5.0, 5.1, 4.9, 9.0, 9.0, 9.2, 8.8,
    ];
    let source = SlicePointSource::new(&POINTS, 2).unwrap();
    let centroids = ScalarBackend::prepare_centroids(&[0.0, 0.0, 5.0, 5.0], 2, 2);
    (source, centroids)
}

#[test]
fn sequential_iteration_storage_is_reused() {
    let (source, centroids) = fixture();
    let mut scratch = IterationScratch::new(2, 2);

    let first = Sequential::compute_stats_full::<f32, ScalarBackend, Euclidean, _>(
        &source,
        2,
        2,
        &centroids,
        4,
        &mut scratch,
    )
    .unwrap();
    let sums_ptr = scratch.sums.as_ptr();
    let counts_ptr = scratch.counts.as_ptr();
    let first_sums = scratch.sums.clone();
    let first_counts = scratch.counts.clone();

    let second = Sequential::compute_stats_full::<f32, ScalarBackend, Euclidean, _>(
        &source,
        2,
        2,
        &centroids,
        4,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(scratch.sums.as_ptr(), sums_ptr);
    assert_eq!(scratch.counts.as_ptr(), counts_ptr);
    assert_eq!(scratch.sums, first_sums);
    assert_eq!(scratch.counts, first_counts);
    assert_eq!(second, first);

    let indices = [0, 2, 4, 6];
    Sequential::compute_stats_indexed::<f32, ScalarBackend, Euclidean, _>(
        &source,
        2,
        2,
        &centroids,
        &indices,
        4,
        &mut scratch,
    )
    .unwrap();
    let indexed_ptr = scratch.indexed_buffer.as_ptr();
    Sequential::compute_stats_indexed::<f32, ScalarBackend, Euclidean, _>(
        &source,
        2,
        2,
        &centroids,
        &indices,
        4,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(scratch.indexed_buffer.as_ptr(), indexed_ptr);
}

#[test]
fn empty_cluster_storage_is_allocated_lazily_and_reused() {
    let (source, mut centroids) = fixture();
    let sums = [0.0, 0.0, 40.0, 40.0];
    let counts = [0, 8];
    let mut scratch = IterationScratch::new(2, 2);

    ScalarBackend::update_centroids_and_get_max_shift(
        &mut centroids,
        &sums,
        &counts,
        2,
        &source,
        &mut rand::rng(),
        scratch.empty_cluster_buffer(),
    );
    let buffer_ptr = scratch.empty_cluster_buffer.as_ptr();
    assert_eq!(scratch.empty_cluster_buffer.len(), 2);

    ScalarBackend::update_centroids_and_get_max_shift(
        &mut centroids,
        &sums,
        &counts,
        2,
        &source,
        &mut rand::rng(),
        scratch.empty_cluster_buffer(),
    );
    assert_eq!(scratch.empty_cluster_buffer.as_ptr(), buffer_ptr);
}

#[cfg(not(feature = "wasm"))]
#[test]
fn parallel_iteration_storage_is_reused() {
    let (source, centroids) = fixture();
    let mut scratch = IterationScratch::new(2, 2);

    let first = Parallel::compute_stats_full::<f32, ScalarBackend, Euclidean, _>(
        &source,
        2,
        2,
        &centroids,
        2,
        &mut scratch,
    )
    .unwrap();
    let worker_ptrs: Vec<_> = scratch
        .parallel_workers
        .iter()
        .map(|worker| (worker.sums.as_ptr(), worker.counts.as_ptr()))
        .collect();
    let first_sums = scratch.sums.clone();
    let first_counts = scratch.counts.clone();

    let second = Parallel::compute_stats_full::<f32, ScalarBackend, Euclidean, _>(
        &source,
        2,
        2,
        &centroids,
        2,
        &mut scratch,
    )
    .unwrap();
    let second_ptrs: Vec<_> = scratch
        .parallel_workers
        .iter()
        .map(|worker| (worker.sums.as_ptr(), worker.counts.as_ptr()))
        .collect();
    assert_eq!(second_ptrs, worker_ptrs);
    assert_eq!(scratch.sums, first_sums);
    assert_eq!(scratch.counts, first_counts);
    assert_eq!(second, first);

    let indices = [0, 2, 3, 5, 7];
    let mut sequential_scratch = IterationScratch::new(2, 2);
    let expected = Sequential::compute_stats_indexed::<f32, ScalarBackend, Euclidean, _>(
        &source,
        2,
        2,
        &centroids,
        &indices,
        2,
        &mut sequential_scratch,
    )
    .unwrap();

    let actual = Parallel::compute_stats_indexed::<f32, ScalarBackend, Euclidean, _>(
        &source,
        2,
        2,
        &centroids,
        &indices,
        2,
        &mut scratch,
    )
    .unwrap();
    let indexed_ptrs: Vec<_> = scratch
        .parallel_workers
        .iter()
        .filter(|worker| !worker.indexed_buffer.is_empty())
        .map(|worker| worker.indexed_buffer.as_ptr())
        .collect();

    assert_eq!(scratch.counts, sequential_scratch.counts);
    for (&actual_sum, &expected_sum) in scratch.sums.iter().zip(&sequential_scratch.sums) {
        let sum_tolerance = 1e-6 * expected_sum.abs().max(1.0);
        assert!((actual_sum - expected_sum).abs() <= sum_tolerance);
    }
    let inertia_tolerance = 1e-6 * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= inertia_tolerance,
        "parallel inertia {actual} differs from sequential inertia {expected}"
    );

    Parallel::compute_stats_indexed::<f32, ScalarBackend, Euclidean, _>(
        &source,
        2,
        2,
        &centroids,
        &indices,
        2,
        &mut scratch,
    )
    .unwrap();
    let reused_indexed_ptrs: Vec<_> = scratch
        .parallel_workers
        .iter()
        .filter(|worker| !worker.indexed_buffer.is_empty())
        .map(|worker| worker.indexed_buffer.as_ptr())
        .collect();
    assert_eq!(reused_indexed_ptrs, indexed_ptrs);

    let empty_inertia = Parallel::compute_stats_indexed::<f32, ScalarBackend, Euclidean, _>(
        &source,
        2,
        2,
        &centroids,
        &[],
        2,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(empty_inertia, 0.0);
    assert!(scratch.sums.iter().all(|&sum| sum == 0.0));
    assert!(scratch.counts.iter().all(|&count| count == 0));
}
