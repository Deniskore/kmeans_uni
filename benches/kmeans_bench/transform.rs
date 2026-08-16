use crate::common::{get_uni_model_scalar_f32, get_uni_model_scalar_f64};
use divan::{Bencher, black_box};
use kmeans_uni::CPUScalar;
#[cfg(feature = "wide")]
use kmeans_uni::{CPUSimd128, CPUSimd256, CPUSimd512, CPUSimdAdaptive};

#[cfg(feature = "wide")]
const SMALL_TRANSFORM_NCOLS: usize = 8;
#[cfg(feature = "wide")]
const SMALL_TRANSFORM_K: usize = 16;

#[cfg(feature = "wide")]
fn small_transform_fixture(npoints: usize) -> (kmeans_uni::KMeans<f32>, Vec<f32>) {
    let centroids = (0..SMALL_TRANSFORM_K * SMALL_TRANSFORM_NCOLS)
        .map(|i| (i % 31) as f32 * 0.125 - 2.0)
        .collect();
    let points = (0..npoints * SMALL_TRANSFORM_NCOLS)
        .map(|i| (i % 37) as f32 * 0.0625 - 1.0)
        .collect();
    let model = kmeans_uni::KMeans::new(
        centroids,
        SMALL_TRANSFORM_NCOLS,
        SMALL_TRANSFORM_K,
        0.0,
        kmeans_uni::MetricType::Euclidean,
    )
    .expect("small transform model");
    (model, points)
}

// No Linfa counterpart is listed here: this crate's transform returns full
// per-centroid scores (n_points * k), while Linfa's transform returns only each
// point's closest-centroid distance.

#[divan::bench]
fn f32_uni_transform(bencher: Bencher) {
    let (model, data) = get_uni_model_scalar_f32();
    bencher.bench_local(|| black_box(model).transform(black_box(data)).unwrap());
}

#[divan::bench]
fn f32_uni_transform_scalar(bencher: Bencher) {
    let (model, data) = get_uni_model_scalar_f32();
    bencher.bench_local(|| {
        black_box(model)
            .transform_with_backend::<CPUScalar>(black_box(data))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_transform_simd128(bencher: Bencher) {
    let (model, data) = get_uni_model_scalar_f32();
    bencher.bench_local(|| {
        black_box(model)
            .transform_with_backend::<CPUSimd128>(black_box(data))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_transform_simd_adaptive(bencher: Bencher) {
    let (model, data) = get_uni_model_scalar_f32();
    bencher.bench_local(|| {
        black_box(model)
            .transform_with_backend::<CPUSimdAdaptive>(black_box(data))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench(args = [1, 32, 128, 512, 4096, 8192, 16384])]
fn f32_uni_transform_simd_adaptive_small(bencher: Bencher, npoints: usize) {
    let (model, points) = small_transform_fixture(npoints);
    model
        .transform_with_backend::<CPUSimdAdaptive>(points.as_slice())
        .expect("warm prepared-centroid cache");
    bencher.bench_local(|| {
        black_box(&model)
            .transform_with_backend::<CPUSimdAdaptive>(black_box(points.as_slice()))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench(args = [1, 32, 128, 512, 4096, 8192, 16384])]
fn f32_uni_transform_simd_adaptive_small_sequential(bencher: Bencher, npoints: usize) {
    let (model, points) = small_transform_fixture(npoints);
    model
        .transform_with_backend_sequential::<CPUSimdAdaptive>(points.as_slice())
        .expect("warm prepared-centroid cache");
    bencher.bench_local(|| {
        black_box(&model)
            .transform_with_backend_sequential::<CPUSimdAdaptive>(black_box(points.as_slice()))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_transform_simd256(bencher: Bencher) {
    let (model, data) = get_uni_model_scalar_f32();
    bencher.bench_local(|| {
        black_box(model)
            .transform_with_backend::<CPUSimd256>(black_box(data))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_transform_simd512(bencher: Bencher) {
    let (model, data) = get_uni_model_scalar_f32();
    bencher.bench_local(|| {
        black_box(model)
            .transform_with_backend::<CPUSimd512>(black_box(data))
            .unwrap()
    });
}

#[divan::bench]
fn f64_uni_transform(bencher: Bencher) {
    let (model, data) = get_uni_model_scalar_f64();
    bencher.bench_local(|| black_box(model).transform(black_box(data)).unwrap());
}
