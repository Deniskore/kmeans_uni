#![cfg(feature = "wide")]

use kmeans_uni::{CPUScalar, CPUSimd, KMeans, KMeansBuilder, MetricType};

#[test]
fn test_predict_simd() {
    let points = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let k = 2;
    let model = KMeansBuilder::<f32>::new(k)
        .iterations(10)
        .cpu_simd()
        .euclidean()
        .build()
        .fit(&points, 2)
        .unwrap();

    let labels_scalar = model.predict_with_backend::<CPUScalar>(&points).unwrap();
    let labels_simd = model.predict_with_backend::<CPUSimd>(&points).unwrap();

    assert_eq!(labels_scalar, labels_simd);
}

#[test]
fn simd_predict_and_transform_match_scalar_for_varied_shapes() {
    for &(ncols, k, npoints) in &[(3, 3, 17), (6, 8, 19), (9, 9, 23), (17, 16, 11)] {
        let points = deterministic_values(npoints * ncols, 0.125);
        let centroids = deterministic_values(k * ncols, 0.25);

        assert_backend_outputs_match(&points, &centroids, ncols, k, MetricType::Euclidean);
        assert_backend_outputs_match(&points, &centroids, ncols, k, MetricType::DotProduct);
    }
}

fn assert_backend_outputs_match(
    points: &[f32],
    centroids: &[f32],
    ncols: usize,
    k: usize,
    metric: MetricType,
) {
    let model = KMeans::new(centroids.to_vec(), ncols, k, 0.0, metric).unwrap();

    let labels_scalar = model
        .predict_with_backend_sequential::<CPUScalar>(points)
        .unwrap();
    let labels_simd = model
        .predict_with_backend_sequential::<CPUSimd>(points)
        .unwrap();
    assert_eq!(labels_scalar, labels_simd);

    let scores_scalar = model
        .transform_with_backend_sequential::<CPUScalar>(points)
        .unwrap();
    let scores_simd = model
        .transform_with_backend_sequential::<CPUSimd>(points)
        .unwrap();

    assert_eq!(scores_scalar.len(), scores_simd.len());
    for (idx, (&scalar, &simd)) in scores_scalar.iter().zip(scores_simd.iter()).enumerate() {
        let diff = (scalar - simd).abs();
        assert!(
            diff <= 1e-3,
            "score diverged at {idx}: scalar={scalar}, simd={simd}, diff={diff}"
        );
    }
}

fn deterministic_values(len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let base = ((i * 37 + 11) % 101) as f32 - 50.0;
            base * scale + (i % 7) as f32 * 0.001
        })
        .collect()
}
