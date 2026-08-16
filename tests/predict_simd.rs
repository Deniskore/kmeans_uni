#![cfg(feature = "wide")]

use kmeans_uni::{
    CPUScalar, CPUSimd128, CPUSimd256, CPUSimd512, CPUSimdAdaptive, CpuBackendType, KMeans,
    KMeansBuilder, MetricType,
};

#[test]
fn predict_simd_matches_scalar() {
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
    let labels_simd = model.predict(&points).unwrap();

    assert_eq!(labels_scalar, labels_simd);
}

#[test]
fn dot_product_prediction_ignores_padded_centroid_lanes() {
    const K: usize = 17;
    const EXPECTED_LABEL: usize = 15;

    let mut centroids = vec![0.0_f32; K];
    centroids[EXPECTED_LABEL] = 100.0;
    centroids[K - 1] = 1.0;
    let model = KMeans::new(centroids, 1, K, 0.0, MetricType::DotProduct).unwrap();
    let points = [1.0_f32];

    let labels_scalar = model
        .predict_with_backend_sequential::<CPUScalar>(points)
        .unwrap();
    assert_eq!(labels_scalar, [EXPECTED_LABEL]);

    assert_specific_backend_outputs_match::<CPUSimd128>(&model, &points);
    assert_specific_backend_outputs_match::<CPUSimd256>(&model, &points);
    assert_specific_backend_outputs_match::<CPUSimd512>(&model, &points);
    assert_specific_backend_outputs_match::<CPUSimdAdaptive>(&model, &points);
}

#[test]
fn f64_dot_product_prediction_ignores_padded_centroid_lanes() {
    const K: usize = 17;
    const EXPECTED_LABEL: usize = 15;

    let mut centroids = vec![0.0_f64; K];
    centroids[EXPECTED_LABEL] = 100.0;
    centroids[K - 1] = 1.0;
    let model = KMeans::new(centroids, 1, K, 0.0, MetricType::DotProduct).unwrap();
    let points = [1.0_f64];

    let labels_scalar = model
        .predict_with_backend_sequential::<CPUScalar>(points)
        .unwrap();
    assert_eq!(labels_scalar, [EXPECTED_LABEL]);

    assert_specific_backend_outputs_match_f64::<CPUSimd128>(&model, &points);
    assert_specific_backend_outputs_match_f64::<CPUSimd256>(&model, &points);
    assert_specific_backend_outputs_match_f64::<CPUSimd512>(&model, &points);
    assert_specific_backend_outputs_match_f64::<CPUSimdAdaptive>(&model, &points);
}

#[test]
fn simd_predict_and_transform_match_scalar_for_varied_shapes() {
    for &(ncols, k, npoints) in &[
        (3, 3, 17),
        (6, 8, 19),
        (9, 15, 23),
        (17, 16, 11),
        (7, 17, 13),
        (5, 31, 9),
        (4, 32, 10),
        (3, 33, 12),
    ] {
        let points = deterministic_values(npoints * ncols, 0.125);
        let centroids = deterministic_values(k * ncols, 0.25);

        assert_backend_outputs_match(&points, &centroids, ncols, k, MetricType::Euclidean);
        assert_backend_outputs_match(&points, &centroids, ncols, k, MetricType::DotProduct);

        let points_f64 = deterministic_values_f64(npoints * ncols, 0.125);
        let centroids_f64 = deterministic_values_f64(k * ncols, 0.25);
        assert_backend_outputs_match_f64(
            &points_f64,
            &centroids_f64,
            ncols,
            k,
            MetricType::Euclidean,
        );
        assert_backend_outputs_match_f64(
            &points_f64,
            &centroids_f64,
            ncols,
            k,
            MetricType::DotProduct,
        );
    }
}

#[test]
fn simd_ties_choose_the_first_real_centroid() {
    const N_COLS: usize = 3;
    const K: usize = 17;
    const N_POINTS: usize = 5;

    for metric in [MetricType::Euclidean, MetricType::DotProduct] {
        let model = KMeans::new(vec![0.0_f32; K * N_COLS], N_COLS, K, 0.0, metric).unwrap();
        let points = vec![0.0_f32; N_POINTS * N_COLS];
        let scalar = model
            .predict_with_backend_sequential::<CPUScalar>(&points)
            .unwrap();
        assert_eq!(scalar, vec![0; N_POINTS]);
        assert_specific_backend_outputs_match::<CPUSimd128>(&model, &points);
        assert_specific_backend_outputs_match::<CPUSimd256>(&model, &points);
        assert_specific_backend_outputs_match::<CPUSimd512>(&model, &points);
        assert_specific_backend_outputs_match::<CPUSimdAdaptive>(&model, &points);

        let model = KMeans::new(vec![0.0_f64; K * N_COLS], N_COLS, K, 0.0, metric).unwrap();
        let points = vec![0.0_f64; N_POINTS * N_COLS];
        let scalar = model
            .predict_with_backend_sequential::<CPUScalar>(&points)
            .unwrap();
        assert_eq!(scalar, vec![0; N_POINTS]);
        assert_specific_backend_outputs_match_f64::<CPUSimd128>(&model, &points);
        assert_specific_backend_outputs_match_f64::<CPUSimd256>(&model, &points);
        assert_specific_backend_outputs_match_f64::<CPUSimd512>(&model, &points);
        assert_specific_backend_outputs_match_f64::<CPUSimdAdaptive>(&model, &points);
    }
}

#[test]
fn simd_cross_lane_ties_choose_the_earliest_centroid() {
    const N_COLS: usize = 3;
    const K: usize = 17;
    const N_POINTS: usize = 5;
    const TIED_INDICES: [usize; 4] = [1, 4, 8, 16];

    for metric in [MetricType::Euclidean, MetricType::DotProduct] {
        let (base_value, tied_value) = match metric {
            MetricType::Euclidean => (100.0, 1.0),
            MetricType::DotProduct => (-100.0, 5.0),
        };
        let mut centroids = vec![base_value; K * N_COLS];
        for index in TIED_INDICES {
            centroids[index * N_COLS..(index + 1) * N_COLS].fill(tied_value);
        }
        let points = match metric {
            MetricType::Euclidean => vec![0.0_f32; N_POINTS * N_COLS],
            MetricType::DotProduct => vec![1.0_f32; N_POINTS * N_COLS],
        };
        let model = KMeans::new(centroids, N_COLS, K, 0.0, metric).unwrap();
        let scalar = model
            .predict_with_backend_sequential::<CPUScalar>(&points)
            .unwrap();
        assert_eq!(scalar, vec![TIED_INDICES[0]; N_POINTS]);
        assert_specific_backend_outputs_match::<CPUSimd128>(&model, &points);
        assert_specific_backend_outputs_match::<CPUSimd256>(&model, &points);
        assert_specific_backend_outputs_match::<CPUSimd512>(&model, &points);
        assert_specific_backend_outputs_match::<CPUSimdAdaptive>(&model, &points);

        let centroids: Vec<f64> = model
            .centroids()
            .iter()
            .map(|&value| value as f64)
            .collect();
        let points: Vec<f64> = points.iter().map(|&value| value as f64).collect();
        let model = KMeans::new(centroids, N_COLS, K, 0.0, metric).unwrap();
        let scalar = model
            .predict_with_backend_sequential::<CPUScalar>(&points)
            .unwrap();
        assert_eq!(scalar, vec![TIED_INDICES[0]; N_POINTS]);
        assert_specific_backend_outputs_match_f64::<CPUSimd128>(&model, &points);
        assert_specific_backend_outputs_match_f64::<CPUSimd256>(&model, &points);
        assert_specific_backend_outputs_match_f64::<CPUSimd512>(&model, &points);
        assert_specific_backend_outputs_match_f64::<CPUSimdAdaptive>(&model, &points);
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

    assert_specific_backend_outputs_match::<CPUSimd128>(&model, points);
    assert_specific_backend_outputs_match::<CPUSimd256>(&model, points);
    assert_specific_backend_outputs_match::<CPUSimd512>(&model, points);
    assert_specific_backend_outputs_match::<CPUSimdAdaptive>(&model, points);
}

fn assert_specific_backend_outputs_match<B: CpuBackendType<f32>>(
    model: &KMeans<f32>,
    points: &[f32],
) {
    let labels_scalar = model
        .predict_with_backend_sequential::<CPUScalar>(points)
        .unwrap();
    let labels_simd = model.predict_with_backend_sequential::<B>(points).unwrap();
    assert_eq!(labels_scalar, labels_simd);

    let scores_scalar = model
        .transform_with_backend_sequential::<CPUScalar>(points)
        .unwrap();
    let scores_simd = model
        .transform_with_backend_sequential::<B>(points)
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

fn assert_backend_outputs_match_f64(
    points: &[f64],
    centroids: &[f64],
    ncols: usize,
    k: usize,
    metric: MetricType,
) {
    let model = KMeans::new(centroids.to_vec(), ncols, k, 0.0, metric).unwrap();

    assert_specific_backend_outputs_match_f64::<CPUSimd128>(&model, points);
    assert_specific_backend_outputs_match_f64::<CPUSimd256>(&model, points);
    assert_specific_backend_outputs_match_f64::<CPUSimd512>(&model, points);
    assert_specific_backend_outputs_match_f64::<CPUSimdAdaptive>(&model, points);
}

fn assert_specific_backend_outputs_match_f64<B: CpuBackendType<f64>>(
    model: &KMeans<f64>,
    points: &[f64],
) {
    let labels_scalar = model
        .predict_with_backend_sequential::<CPUScalar>(points)
        .unwrap();
    let labels_simd = model.predict_with_backend_sequential::<B>(points).unwrap();
    assert_eq!(labels_scalar, labels_simd);

    let scores_scalar = model
        .transform_with_backend_sequential::<CPUScalar>(points)
        .unwrap();
    let scores_simd = model
        .transform_with_backend_sequential::<B>(points)
        .unwrap();

    assert_eq!(scores_scalar.len(), scores_simd.len());
    for (idx, (&scalar, &simd)) in scores_scalar.iter().zip(scores_simd.iter()).enumerate() {
        let diff = (scalar - simd).abs();
        assert!(
            diff <= 1e-10,
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

fn deterministic_values_f64(len: usize, scale: f64) -> Vec<f64> {
    (0..len)
        .map(|i| {
            let base = ((i * 37 + 11) % 101) as f64 - 50.0;
            base * scale + (i % 7) as f64 * 0.001
        })
        .collect()
}
