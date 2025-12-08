use kmeans_uni::{Error, KMeansBuilder};

#[test]
fn test_k_greater_than_n_points_fails() {
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
fn test_k_equals_n_points_succeeds() {
    let data = vec![1.0f32, 2.0, 3.0]; // 3 points
    let ncols = 1;
    let k = 3; // k == n

    let result = KMeansBuilder::new(k).build_default().fit(&data, ncols);

    if let Err(e) = result {
        panic!("Validation failed for k=n: {:?}", e);
    }
}

#[test]
fn test_basic_fit_validation() {
    let data = vec![0.0f32; 100];
    let ncols = 2;
    let k = 5;

    let result = KMeansBuilder::new(k).build_default().fit(&data, ncols);

    assert!(result.is_ok());
}
