use kmeans_uni::{Error, KMeans, KMeansBuilder, MetricType, SlicePointSource};

#[test]
fn dot_product_prediction_uses_trained_metric() {
    // Craft a model where Euclidean would choose cluster 1 but dot-product prefers cluster 0.
    let model = KMeans::new(
        vec![2.0f32, 0.0, -0.5, 0.0],
        2,
        2,
        0.0,
        MetricType::DotProduct,
    )
    .unwrap();

    let point = [0.4f32, 0.0];
    let labels = model.predict(point).unwrap();
    assert_eq!(labels, vec![0]);

    // Transform should also reflect the dot-product scores.
    let distances = model.transform([0.4f32, 0.0]).unwrap();
    assert!(distances[0] > distances[1]);

    let source = SlicePointSource::new(&point, 2).unwrap();
    let labels_from_source = model.predict_from_source(&source).unwrap();
    assert_eq!(labels_from_source, vec![0]);
}

#[test]
fn fit_rejects_empty_inputs() {
    let result = KMeansBuilder::<f32>::new(2)
        .cpu_scalar()
        .euclidean()
        .build()
        .fit([], 1);
    assert!(matches!(result, Err(Error::InvalidInput(_))));

    let empty = [];
    let source = SlicePointSource::<f32>::new(&empty, 1).unwrap();
    let result = KMeansBuilder::<f32>::new(2)
        .cpu_scalar()
        .euclidean()
        .build()
        .fit_from_source(&source);
    assert!(matches!(result, Err(Error::InvalidInput(_))));
}
