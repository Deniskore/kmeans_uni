#![cfg(feature = "wide")]

use kmeans_uni::{CPUSimd, KMeansBuilder};

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

    let labels_scalar = model.predict(&points).unwrap();
    let labels_simd = model.predict_with_backend::<CPUSimd>(&points).unwrap();

    assert_eq!(labels_scalar, labels_simd);
}
