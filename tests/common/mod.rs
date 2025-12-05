use kmeans_uni::Primitive;

#[inline]
pub fn inertia_f64<F: Primitive>(points: &[F], centroids: &[F], ncols: usize) -> f64 {
    points
        .chunks_exact(ncols)
        .map(|point| {
            centroids
                .chunks_exact(ncols)
                .map(|centroid| distance_sq_f64(point, centroid))
                .fold(f64::INFINITY, f64::min)
        })
        .sum()
}

#[inline]
pub fn distance_sq_f64<F: Primitive>(a: &[F], b: &[F]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let dx = x.to_f64().unwrap() - y.to_f64().unwrap();
            dx * dx
        })
        .sum()
}

#[inline]
pub fn assert_centroids_match_by_sorting<F: Primitive>(
    expected: &[F],
    actual: &[F],
    ncols: usize,
    tolerance: f64,
) {
    let mut remaining = reshape_centroids_to_f64(actual, ncols);
    let expected = reshape_centroids_to_f64(expected, ncols);

    assert_eq!(expected.len(), remaining.len(), "Centroid counts differ");

    for left in expected {
        let (idx, diff) = remaining
            .iter()
            .enumerate()
            .map(|(idx, right)| (idx, distance_sq_f64(&left, right).sqrt()))
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .expect("must have centroid match");

        assert!(
            diff <= tolerance,
            "Centroids diverged: {left:?} vs {:?} (dist={diff})",
            remaining[idx]
        );
        remaining.remove(idx);
    }
}

#[inline]
pub fn reshape_centroids_to_f64<F: Primitive>(values: &[F], ncols: usize) -> Vec<Vec<f64>> {
    values
        .chunks(ncols)
        .map(|chunk| chunk.iter().map(|v| v.to_f64().unwrap()).collect())
        .collect()
}
