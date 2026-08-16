use crate::common::{
    BATCH, EPOCHS, N_POINTS, get_data_f32, get_data_f64, run_linfa_minibatch_f32,
    run_linfa_minibatch_f64, run_uni_minibatch, single_thread_pool,
};
use divan::{Bencher, black_box};
use kmeans_uni::CPUScalar;
#[cfg(feature = "wide")]
use kmeans_uni::{CPUSimd256, CPUSimdAdaptive};

#[divan::bench]
fn f32_uni_minibatch_scalar_seq(bencher: Bencher) {
    let data = get_data_f32();
    let iterations = (N_POINTS / BATCH) * EPOCHS;
    bencher.bench_local(|| run_uni_minibatch::<f32, CPUScalar>(black_box(data), iterations, false));
}

#[divan::bench]
fn f32_uni_minibatch_scalar_par(bencher: Bencher) {
    let data = get_data_f32();
    let iterations = (N_POINTS / BATCH) * EPOCHS;
    bencher.bench_local(|| run_uni_minibatch::<f32, CPUScalar>(black_box(data), iterations, true));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_minibatch_simd256_seq(bencher: Bencher) {
    let data = get_data_f32();
    let iterations = (N_POINTS / BATCH) * EPOCHS;
    bencher
        .bench_local(|| run_uni_minibatch::<f32, CPUSimd256>(black_box(data), iterations, false));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_minibatch_simd256_par(bencher: Bencher) {
    let data = get_data_f32();
    let iterations = (N_POINTS / BATCH) * EPOCHS;
    bencher.bench_local(|| run_uni_minibatch::<f32, CPUSimd256>(black_box(data), iterations, true));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_minibatch_simd_adaptive_seq(bencher: Bencher) {
    let data = get_data_f32();
    let iterations = (N_POINTS / BATCH) * EPOCHS;
    bencher.bench_local(|| {
        run_uni_minibatch::<f32, CPUSimdAdaptive>(black_box(data), iterations, false)
    });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_minibatch_simd_adaptive_par(bencher: Bencher) {
    let data = get_data_f32();
    let iterations = (N_POINTS / BATCH) * EPOCHS;
    bencher.bench_local(|| {
        run_uni_minibatch::<f32, CPUSimdAdaptive>(black_box(data), iterations, true)
    });
}

#[divan::bench]
fn f32_linfa_minibatch_seq(bencher: Bencher) {
    let data = get_data_f32();
    bencher
        .bench_local(|| single_thread_pool().install(|| run_linfa_minibatch_f32(black_box(data))));
}

#[divan::bench]
fn f32_linfa_minibatch_par(bencher: Bencher) {
    let data = get_data_f32();
    bencher.bench_local(|| run_linfa_minibatch_f32(black_box(data)));
}

#[divan::bench]
fn f64_uni_minibatch_scalar_seq(bencher: Bencher) {
    let data = get_data_f64();
    let iterations = (N_POINTS / BATCH) * EPOCHS;
    bencher.bench_local(|| run_uni_minibatch::<f64, CPUScalar>(black_box(data), iterations, false));
}

#[divan::bench]
fn f64_uni_minibatch_scalar_par(bencher: Bencher) {
    let data = get_data_f64();
    let iterations = (N_POINTS / BATCH) * EPOCHS;
    bencher.bench_local(|| run_uni_minibatch::<f64, CPUScalar>(black_box(data), iterations, true));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_minibatch_simd256_seq(bencher: Bencher) {
    let data = get_data_f64();
    let iterations = (N_POINTS / BATCH) * EPOCHS;
    bencher
        .bench_local(|| run_uni_minibatch::<f64, CPUSimd256>(black_box(data), iterations, false));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_minibatch_simd256_par(bencher: Bencher) {
    let data = get_data_f64();
    let iterations = (N_POINTS / BATCH) * EPOCHS;
    bencher.bench_local(|| run_uni_minibatch::<f64, CPUSimd256>(black_box(data), iterations, true));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_minibatch_simd_adaptive_seq(bencher: Bencher) {
    let data = get_data_f64();
    let iterations = (N_POINTS / BATCH) * EPOCHS;
    bencher.bench_local(|| {
        run_uni_minibatch::<f64, CPUSimdAdaptive>(black_box(data), iterations, false)
    });
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_minibatch_simd_adaptive_par(bencher: Bencher) {
    let data = get_data_f64();
    let iterations = (N_POINTS / BATCH) * EPOCHS;
    bencher.bench_local(|| {
        run_uni_minibatch::<f64, CPUSimdAdaptive>(black_box(data), iterations, true)
    });
}

#[divan::bench]
fn f64_linfa_minibatch_seq(bencher: Bencher) {
    let data = get_data_f64();
    bencher
        .bench_local(|| single_thread_pool().install(|| run_linfa_minibatch_f64(black_box(data))));
}

#[divan::bench]
fn f64_linfa_minibatch_par(bencher: Bencher) {
    let data = get_data_f64();
    bencher.bench_local(|| run_linfa_minibatch_f64(black_box(data)));
}
