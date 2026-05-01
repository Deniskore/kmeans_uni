use crate::common::{get_uni_model_scalar_f32, get_uni_model_scalar_f64};
use divan::{Bencher, black_box};
use kmeans_uni::CPUScalar;
#[cfg(feature = "wide")]
use kmeans_uni::CPUSimd;

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
fn f32_uni_transform_simd(bencher: Bencher) {
    let (model, data) = get_uni_model_scalar_f32();
    bencher.bench_local(|| {
        black_box(model)
            .transform_with_backend::<CPUSimd>(black_box(data))
            .unwrap()
    });
}

#[divan::bench]
fn f64_uni_transform(bencher: Bencher) {
    let (model, data) = get_uni_model_scalar_f64();
    bencher.bench_local(|| black_box(model).transform(black_box(data)).unwrap());
}
