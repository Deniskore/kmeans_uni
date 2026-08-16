use crate::common::{
    CopyOnlySource, N_COLS, get_linfa_model_f32, get_linfa_model_f64, get_uni_model_scalar_f32,
    get_uni_model_scalar_f64,
};
#[cfg(feature = "wide")]
use crate::common::{get_uni_model_simd_f32, get_uni_model_simd_f64};
use divan::{Bencher, black_box};
use kmeans_uni::{CPUScalar, SlicePointSource};
#[cfg(feature = "wide")]
use kmeans_uni::{CPUSimd128, CPUSimd256, CPUSimd512, CPUSimdAdaptive};
use linfa::prelude::Predict;

const SMALL_PREDICT_NCOLS: usize = 8;
const SMALL_PREDICT_K: usize = 16;

fn small_predict_fixture(npoints: usize) -> (kmeans_uni::KMeans<f32>, Vec<f32>) {
    let centroids = (0..SMALL_PREDICT_K * SMALL_PREDICT_NCOLS)
        .map(|i| (i % 31) as f32 * 0.125 - 2.0)
        .collect();
    let points = (0..npoints * SMALL_PREDICT_NCOLS)
        .map(|i| (i % 37) as f32 * 0.0625 - 1.0)
        .collect();
    let model = kmeans_uni::KMeans::new(
        centroids,
        SMALL_PREDICT_NCOLS,
        SMALL_PREDICT_K,
        0.0,
        kmeans_uni::MetricType::Euclidean,
    )
    .expect("small prediction model");
    (model, points)
}

// Linfa predict benches cover ndarray-backed model prediction. The PointSource
// benches below are crate-specific paths, so adding Linfa names there would
// measure conversion/glue rather than the same unit of work.

#[divan::bench]
fn f32_uni_predict_scalar(bencher: Bencher) {
    let (model, data) = get_uni_model_scalar_f32();
    bencher.bench_local(|| {
        black_box(model)
            .predict_with_backend::<CPUScalar>(black_box(data))
            .unwrap()
    });
}

#[divan::bench(args = [1, 32, 128, 512, 4096, 8192])]
fn f32_uni_predict_scalar_small(bencher: Bencher, npoints: usize) {
    let (model, points) = small_predict_fixture(npoints);
    bencher.bench_local(|| {
        black_box(&model)
            .predict_with_backend::<CPUScalar>(black_box(points.as_slice()))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_predict_simd128(bencher: Bencher) {
    let (model, data) = get_uni_model_simd_f32();
    bencher.bench_local(|| {
        black_box(model)
            .predict_with_backend::<CPUSimd128>(black_box(data))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_predict_simd_adaptive(bencher: Bencher) {
    let (model, data) = get_uni_model_simd_f32();
    bencher.bench_local(|| {
        black_box(model)
            .predict_with_backend::<CPUSimdAdaptive>(black_box(data))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench(args = [1, 32, 128, 512, 4096, 8192])]
fn f32_uni_predict_simd_adaptive_small(bencher: Bencher, npoints: usize) {
    let (model, points) = small_predict_fixture(npoints);
    model
        .predict_with_backend::<CPUSimdAdaptive>(points.as_slice())
        .expect("warm prepared-centroid cache");
    bencher.bench_local(|| {
        black_box(&model)
            .predict_with_backend::<CPUSimdAdaptive>(black_box(points.as_slice()))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench(args = [1, 32, 128, 512, 4096, 8192])]
fn f32_uni_predict_simd_adaptive_small_sequential(bencher: Bencher, npoints: usize) {
    let (model, points) = small_predict_fixture(npoints);
    model
        .predict_with_backend_sequential::<CPUSimdAdaptive>(points.as_slice())
        .expect("warm prepared-centroid cache");
    bencher.bench_local(|| {
        black_box(&model)
            .predict_with_backend_sequential::<CPUSimdAdaptive>(black_box(points.as_slice()))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench(args = [1, 32, 128, 512, 4096, 8192])]
fn f32_uni_predict_from_source_simd_adaptive_small(bencher: Bencher, npoints: usize) {
    let (model, points) = small_predict_fixture(npoints);
    let source = SlicePointSource::new(points.as_slice(), SMALL_PREDICT_NCOLS).unwrap();
    model
        .predict_from_source_with_backend::<CPUSimdAdaptive, _>(&source)
        .expect("warm prepared-centroid cache");
    bencher.bench_local(|| {
        black_box(&model)
            .predict_from_source_with_backend::<CPUSimdAdaptive, _>(black_box(&source))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_predict_simd_adaptive_cold(bencher: Bencher) {
    let (model, points) = small_predict_fixture(32);
    bencher
        .with_inputs(|| model.clone())
        .bench_local_values(|model| {
            model
                .predict_with_backend::<CPUSimdAdaptive>(black_box(points.as_slice()))
                .unwrap()
        });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_predict_simd256(bencher: Bencher) {
    let (model, data) = get_uni_model_simd_f32();
    bencher.bench_local(|| {
        black_box(model)
            .predict_with_backend::<CPUSimd256>(black_box(data))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_predict_simd512(bencher: Bencher) {
    let (model, data) = get_uni_model_simd_f32();
    bencher.bench_local(|| {
        black_box(model)
            .predict_with_backend::<CPUSimd512>(black_box(data))
            .unwrap()
    });
}

#[divan::bench]
fn f32_linfa_predict(bencher: Bencher) {
    let (model, dataset) = get_linfa_model_f32();
    bencher.bench_local(|| black_box(model).predict(black_box(&dataset)));
}

#[divan::bench]
fn f32_uni_predict_from_source_default_slice(bencher: Bencher) {
    let (model, data) = get_uni_model_scalar_f32();
    let source = SlicePointSource::new(data, N_COLS).expect("slice source");
    bencher.bench_local(|| {
        black_box(model)
            .predict_from_source(black_box(&source))
            .unwrap()
    });
}

#[divan::bench]
fn f32_uni_predict_from_source_scalar_slice(bencher: Bencher) {
    let (model, data) = get_uni_model_scalar_f32();
    let source = SlicePointSource::new(data, N_COLS).expect("slice source");
    bencher.bench_local(|| {
        black_box(model)
            .predict_from_source_with_backend::<CPUScalar, _>(black_box(&source))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_predict_from_source_simd256_slice(bencher: Bencher) {
    let (model, data) = get_uni_model_scalar_f32();
    let source = SlicePointSource::new(data, N_COLS).expect("slice source");
    bencher.bench_local(|| {
        black_box(model)
            .predict_from_source_with_backend::<CPUSimd256, _>(black_box(&source))
            .unwrap()
    });
}

#[divan::bench]
fn f32_uni_predict_from_source_default_copy_only(bencher: Bencher) {
    let (model, data) = get_uni_model_scalar_f32();
    let source = CopyOnlySource::new(data.to_vec(), N_COLS);
    bencher.bench_local(|| {
        black_box(model)
            .predict_from_source(black_box(&source))
            .unwrap()
    });
}

#[divan::bench]
fn f32_uni_predict_from_source_scalar_copy_only(bencher: Bencher) {
    let (model, data) = get_uni_model_scalar_f32();
    let source = CopyOnlySource::new(data.to_vec(), N_COLS);
    bencher.bench_local(|| {
        black_box(model)
            .predict_from_source_with_backend::<CPUScalar, _>(black_box(&source))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_predict_from_source_simd256_copy_only(bencher: Bencher) {
    let (model, data) = get_uni_model_scalar_f32();
    let source = CopyOnlySource::new(data.to_vec(), N_COLS);
    bencher.bench_local(|| {
        black_box(model)
            .predict_from_source_with_backend::<CPUSimd256, _>(black_box(&source))
            .unwrap()
    });
}

#[divan::bench]
fn f64_uni_predict_scalar(bencher: Bencher) {
    let (model, data) = get_uni_model_scalar_f64();
    bencher.bench_local(|| {
        black_box(model)
            .predict_with_backend::<CPUScalar>(black_box(data))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_predict_simd128(bencher: Bencher) {
    let (model, data) = get_uni_model_simd_f64();
    bencher.bench_local(|| {
        black_box(model)
            .predict_with_backend::<CPUSimd128>(black_box(data))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_predict_simd_adaptive(bencher: Bencher) {
    let (model, data) = get_uni_model_simd_f64();
    bencher.bench_local(|| {
        black_box(model)
            .predict_with_backend::<CPUSimdAdaptive>(black_box(data))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_predict_simd256(bencher: Bencher) {
    let (model, data) = get_uni_model_simd_f64();
    bencher.bench_local(|| {
        black_box(model)
            .predict_with_backend::<CPUSimd256>(black_box(data))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_predict_simd512(bencher: Bencher) {
    let (model, data) = get_uni_model_simd_f64();
    bencher.bench_local(|| {
        black_box(model)
            .predict_with_backend::<CPUSimd512>(black_box(data))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_predict_simd128_sequential(bencher: Bencher) {
    let (model, data) = get_uni_model_simd_f32();
    bencher.bench_local(|| {
        black_box(model)
            .predict_with_backend_sequential::<CPUSimd128>(black_box(data))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_predict_simd256_sequential(bencher: Bencher) {
    let (model, data) = get_uni_model_simd_f32();
    bencher.bench_local(|| {
        black_box(model)
            .predict_with_backend_sequential::<CPUSimd256>(black_box(data))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_predict_simd512_sequential(bencher: Bencher) {
    let (model, data) = get_uni_model_simd_f32();
    bencher.bench_local(|| {
        black_box(model)
            .predict_with_backend_sequential::<CPUSimd512>(black_box(data))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_predict_simd128_sequential(bencher: Bencher) {
    let (model, data) = get_uni_model_simd_f64();
    bencher.bench_local(|| {
        black_box(model)
            .predict_with_backend_sequential::<CPUSimd128>(black_box(data))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_predict_simd256_sequential(bencher: Bencher) {
    let (model, data) = get_uni_model_simd_f64();
    bencher.bench_local(|| {
        black_box(model)
            .predict_with_backend_sequential::<CPUSimd256>(black_box(data))
            .unwrap()
    });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_predict_simd512_sequential(bencher: Bencher) {
    let (model, data) = get_uni_model_simd_f64();
    bencher.bench_local(|| {
        black_box(model)
            .predict_with_backend_sequential::<CPUSimd512>(black_box(data))
            .unwrap()
    });
}

#[divan::bench]
fn f64_linfa_predict(bencher: Bencher) {
    let (model, dataset) = get_linfa_model_f64();
    bencher.bench_local(|| black_box(model).predict(black_box(&dataset)));
}
