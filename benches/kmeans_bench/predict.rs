use crate::common::{
    CopyOnlySource, N_COLS, get_linfa_model_f32, get_linfa_model_f64, get_uni_model_scalar_f32,
    get_uni_model_scalar_f64,
};
#[cfg(feature = "wide")]
use crate::common::{get_uni_model_simd_f32, get_uni_model_simd_f64};
use divan::{Bencher, black_box};
#[cfg(feature = "wide")]
use kmeans_uni::CPUSimd;
use kmeans_uni::{CPUScalar, SlicePointSource};
use linfa::prelude::Predict;

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

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_predict_simd(bencher: Bencher) {
    let (model, data) = get_uni_model_simd_f32();
    bencher.bench_local(|| {
        black_box(model)
            .predict_with_backend::<CPUSimd>(black_box(data))
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
fn f32_uni_predict_from_source_simd_slice(bencher: Bencher) {
    let (model, data) = get_uni_model_scalar_f32();
    let source = SlicePointSource::new(data, N_COLS).expect("slice source");
    bencher.bench_local(|| {
        black_box(model)
            .predict_from_source_with_backend::<CPUSimd, _>(black_box(&source))
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
fn f32_uni_predict_from_source_simd_copy_only(bencher: Bencher) {
    let (model, data) = get_uni_model_scalar_f32();
    let source = CopyOnlySource::new(data.to_vec(), N_COLS);
    bencher.bench_local(|| {
        black_box(model)
            .predict_from_source_with_backend::<CPUSimd, _>(black_box(&source))
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
fn f64_uni_predict_simd(bencher: Bencher) {
    let (model, data) = get_uni_model_simd_f64();
    bencher.bench_local(|| {
        black_box(model)
            .predict_with_backend::<CPUSimd>(black_box(data))
            .unwrap()
    });
}

#[divan::bench]
fn f64_linfa_predict(bencher: Bencher) {
    let (model, dataset) = get_linfa_model_f64();
    bencher.bench_local(|| black_box(model).predict(black_box(&dataset)));
}
