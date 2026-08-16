use kmeans_uni::{Error, KMeans, KMeansBuilder, MetricType, SlicePointSource};

#[test]
fn k_greater_than_n_points_fails() {
    let data = vec![1.0f32; 10]; // 10 points
    let ncols = 1;
    let k = 11; // k > n

    let result = KMeansBuilder::new(k).build_default().fit(&data, ncols);

    assert!(result.is_err());
    match result {
        Err(Error::InvalidInput(msg)) => {
            assert!(msg.contains("cannot be greater than number of points"));
        }
        _ => panic!("Expected InvalidInput error"),
    }
}

#[test]
fn k_equals_n_points_succeeds() {
    let data = vec![1.0f32, 2.0, 3.0]; // 3 points
    let ncols = 1;
    let k = 3; // k == n

    let result = KMeansBuilder::new(k).build_default().fit(&data, ncols);

    if let Err(e) = result {
        panic!("Validation failed for k=n: {:?}", e);
    }
}

#[test]
fn basic_fit_validation() {
    let data = vec![0.0f32; 100];
    let ncols = 2;
    let k = 5;

    let result = KMeansBuilder::new(k).build_default().fit(&data, ncols);

    assert!(result.is_ok());
}

#[test]
fn fit_rejects_zero_or_malformed_configuration() {
    let zero_columns = KMeansBuilder::<f32>::new(1).build_default().fit([1.0], 0);
    assert!(matches!(zero_columns, Err(Error::InvalidInput(_))));

    let zero_clusters = KMeansBuilder::<f32>::new(0).build_default().fit([1.0], 1);
    assert!(matches!(zero_clusters, Err(Error::InvalidInput(_))));

    let partial_row = KMeansBuilder::<f32>::new(1)
        .build_default()
        .fit([1.0, 2.0, 3.0], 2);
    assert!(matches!(partial_row, Err(Error::InvalidInput(_))));

    let zero_attempts = KMeansBuilder::<f32>::new(1)
        .attempts(0)
        .build_default()
        .fit([1.0], 1);
    assert!(matches!(zero_attempts, Err(Error::InvalidInput(_))));
}

#[test]
fn model_constructor_rejects_invalid_shapes_without_panicking() {
    for result in [
        KMeans::new(vec![], 0, 1, 0.0_f32, MetricType::Euclidean),
        KMeans::new(vec![], 1, 0, 0.0_f32, MetricType::Euclidean),
        KMeans::new(vec![1.0], 2, 1, 0.0_f32, MetricType::Euclidean),
        KMeans::new(vec![], usize::MAX, 2, 0.0_f32, MetricType::Euclidean),
    ] {
        assert!(matches!(result, Err(Error::InvalidInput(_))));
    }
}

#[test]
fn prediction_and_transform_reject_invalid_input_shapes() {
    let model = KMeans::new(
        vec![0.0_f32, 0.0, 1.0, 1.0],
        2,
        2,
        0.0,
        MetricType::Euclidean,
    )
    .unwrap();

    assert!(matches!(model.predict([]), Err(Error::InvalidInput(_))));
    assert!(matches!(model.transform([]), Err(Error::InvalidInput(_))));
    assert!(matches!(model.predict([1.0]), Err(Error::InvalidInput(_))));
    assert!(matches!(
        model.transform([1.0]),
        Err(Error::InvalidInput(_))
    ));

    let wrong_columns = SlicePointSource::new(&[1.0_f32, 2.0, 3.0], 3).unwrap();
    assert!(matches!(
        model.predict_from_source(&wrong_columns),
        Err(Error::DimensionMismatch(_))
    ));

    let empty = SlicePointSource::<f32>::new(&[], 2).unwrap();
    assert!(matches!(
        model.predict_from_source(&empty),
        Err(Error::InvalidInput(_))
    ));
}
