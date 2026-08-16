use super::*;

#[test]
fn adaptive_and_explicit_backends_share_the_same_prepared_cache() {
    let model = KMeans::new(
        (0..16 * 3).map(|i| i as f32).collect(),
        3,
        16,
        0.0,
        MetricType::Euclidean,
    )
    .unwrap();

    let scalar = model.prepared_centroids_for_backend::<CPUScalar>();
    assert!(std::ptr::eq(scalar, model.centroids()));
    assert!(model.prepared_centroid_cache.simd128.get().is_none());
    assert!(model.prepared_centroid_cache.simd256.get().is_none());
    assert!(model.prepared_centroid_cache.simd512.get().is_none());

    let adaptive = model.prepared_centroids_for_backend::<CPUSimdAdaptive>();
    let explicit = match <<CPUSimdAdaptive as CpuBackendType<f32>>::Core as CoreBackend<f32>>::prepared_centroid_layout(model.k()) {
        PreparedCentroidLayout::Simd128 => {
            model.prepared_centroids_for_backend::<CPUSimd128>()
        }
        PreparedCentroidLayout::Simd256 => {
            model.prepared_centroids_for_backend::<CPUSimd256>()
        }
        PreparedCentroidLayout::Simd512 => {
            model.prepared_centroids_for_backend::<CPUSimd512>()
        }
        PreparedCentroidLayout::Identity => {
            unreachable!("adaptive SIMD uses a packed layout")
        }
    };

    assert!(std::ptr::eq(adaptive, explicit));
    let initialized = [
        model.prepared_centroid_cache.simd128.get().is_some(),
        model.prepared_centroid_cache.simd256.get().is_some(),
        model.prepared_centroid_cache.simd512.get().is_some(),
    ]
    .into_iter()
    .filter(|initialized| *initialized)
    .count();
    assert_eq!(initialized, 1);
}
