#[cfg(feature = "wide")]
use kmeans_uni::CPUSimd;
use kmeans_uni::{
    AlgorithmNotSet, BackendNotSet, CPUScalar, DotProduct, Euclidean, KMeansBuilder, KMeansConfig,
    KMeansPlusPlus, SlicePointSource,
};

#[test]
fn test_builder_types() {
    let _: KMeansBuilder<f32, CPUScalar, Euclidean, KMeansPlusPlus> = KMeansBuilder::<f32>::new(8)
        .cpu_scalar()
        .euclidean()
        .init_plus_plus();
    let _: KMeansBuilder<f32, CPUScalar, DotProduct, KMeansPlusPlus> = KMeansBuilder::<f32>::new(8)
        .cpu_scalar()
        .dot_product()
        .init_plus_plus();
}

#[test]
fn test_f64_builder_types() {
    let _: KMeansBuilder<f64, CPUScalar, Euclidean, KMeansPlusPlus> = KMeansBuilder::<f64>::new(8)
        .cpu_scalar()
        .euclidean()
        .init_plus_plus();
    let _: KMeansBuilder<f64, CPUScalar, DotProduct, KMeansPlusPlus> = KMeansBuilder::<f64>::new(8)
        .cpu_scalar()
        .dot_product()
        .init_plus_plus();
}

#[cfg(feature = "wide")]
#[test]
fn test_simd_builder_types() {
    let _: KMeansBuilder<f32, CPUSimd, Euclidean, KMeansPlusPlus> = KMeansBuilder::<f32>::new(8)
        .cpu_simd()
        .euclidean()
        .init_plus_plus();
    let _: KMeansBuilder<f32, CPUSimd, DotProduct, KMeansPlusPlus> = KMeansBuilder::<f32>::new(8)
        .cpu_simd()
        .dot_product()
        .init_plus_plus();
}

#[cfg(feature = "wide")]
#[test]
fn test_simd_f64_builder_types() {
    let _: KMeansBuilder<f64, CPUSimd, Euclidean, KMeansPlusPlus> = KMeansBuilder::<f64>::new(8)
        .cpu_simd()
        .euclidean()
        .init_plus_plus();
    let _: KMeansBuilder<f64, CPUSimd, DotProduct, KMeansPlusPlus> = KMeansBuilder::<f64>::new(8)
        .cpu_simd()
        .dot_product()
        .init_plus_plus();
}

#[test]
fn test_kmeans_config_types() {
    let _: KMeansConfig<f32, CPUScalar, Euclidean, false, KMeansPlusPlus> =
        KMeansBuilder::<f32>::new(8)
            .cpu_scalar()
            .euclidean()
            .init_plus_plus()
            .build();
}

#[test]
fn test_f64_kmeans_config_types() {
    let _: KMeansConfig<f64, CPUScalar, Euclidean, false, KMeansPlusPlus> =
        KMeansBuilder::<f64>::new(8)
            .cpu_scalar()
            .euclidean()
            .init_plus_plus()
            .build();
}

#[cfg(feature = "wide")]
#[test]
fn test_simd_f64_kmeans_config_types() {
    let _: KMeansConfig<f64, CPUSimd, Euclidean, false, KMeansPlusPlus> =
        KMeansBuilder::<f64>::new(8)
            .cpu_simd()
            .euclidean()
            .init_plus_plus()
            .build();
}

#[test]
fn test_default_builder() {
    let builder = KMeansBuilder::<f32>::new(8);
    let _: KMeansBuilder<f32, BackendNotSet, AlgorithmNotSet, KMeansPlusPlus> = builder;
}

#[test]
fn test_builder_methods_chaining() {
    let builder = KMeansBuilder::<f32>::new(8)
        .cpu_scalar()
        .euclidean()
        .init_plus_plus()
        .seed(12345);
    let _: KMeansBuilder<f32, CPUScalar, Euclidean, KMeansPlusPlus> = builder;
}

#[test]
fn test_build_method() {
    let config = KMeansBuilder::<f32>::new(8)
        .cpu_scalar()
        .euclidean()
        .init_plus_plus()
        .build();
    let _: KMeansConfig<f32, CPUScalar, Euclidean, false, KMeansPlusPlus> = config;
}

#[test]
fn test_builder_type_state_progression() {
    let builder = KMeansBuilder::<f32>::new(8);
    let with_backend: KMeansBuilder<f32, CPUScalar, AlgorithmNotSet, KMeansPlusPlus> =
        builder.backend::<CPUScalar>();
    let with_algorithm: KMeansBuilder<f32, CPUScalar, Euclidean, KMeansPlusPlus> =
        with_backend.algorithm::<Euclidean>();
    let _sequential: KMeansConfig<f32, CPUScalar, Euclidean, false, KMeansPlusPlus> =
        with_algorithm.build();

    let parallel_builder: KMeansBuilder<f32, CPUScalar, Euclidean, KMeansPlusPlus> =
        KMeansBuilder::<f32>::new(8)
            .backend::<CPUScalar>()
            .algorithm::<Euclidean>();
    let _parallel: KMeansConfig<f32, CPUScalar, Euclidean, true, KMeansPlusPlus> =
        parallel_builder.parallel();
}

#[test]
fn test_mini_batch_api_compiles() {
    // Need at least k points for k clusters
    let points = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let source = SlicePointSource::<f32>::new(&points, 1).unwrap();
    let _ = KMeansBuilder::<f32>::new(8)
        .cpu_scalar()
        .euclidean()
        .build()
        .fit_mini_batch_from_source(&source, 1)
        .unwrap();
}
