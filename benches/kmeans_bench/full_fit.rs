use crate::common::{
    get_data_f32, get_data_f64, run_linfa_full_f32, run_linfa_full_f64, run_uni_full,
};
use divan::{Bencher, black_box};
use kmeans_uni::CPUScalar;
#[cfg(feature = "wide")]
use kmeans_uni::CPUSimd;

#[divan::bench]
fn f32_uni_scalar_seq(bencher: Bencher) {
    let data = get_data_f32();
    bencher.bench_local(|| run_uni_full::<f32, CPUScalar>(black_box(data), false));
}

#[divan::bench]
fn f32_uni_scalar_par(bencher: Bencher) {
    let data = get_data_f32();
    bencher.bench_local(|| run_uni_full::<f32, CPUScalar>(black_box(data), true));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_simd_seq(bencher: Bencher) {
    let data = get_data_f32();
    bencher.bench_local(|| run_uni_full::<f32, CPUSimd>(black_box(data), false));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_simd_par(bencher: Bencher) {
    let data = get_data_f32();
    bencher.bench_local(|| run_uni_full::<f32, CPUSimd>(black_box(data), true));
}

#[divan::bench]
fn f32_linfa_fit_seq(bencher: Bencher) {
    let data = get_data_f32();
    bencher.bench_local(|| run_linfa_full_f32(black_box(data), false));
}

#[divan::bench]
fn f32_linfa_fit_par(bencher: Bencher) {
    let data = get_data_f32();
    bencher.bench_local(|| run_linfa_full_f32(black_box(data), true));
}

#[divan::bench]
fn f64_uni_scalar_seq(bencher: Bencher) {
    let data = get_data_f64();
    bencher.bench_local(|| run_uni_full::<f64, CPUScalar>(black_box(data), false));
}

#[divan::bench]
fn f64_uni_scalar_par(bencher: Bencher) {
    let data = get_data_f64();
    bencher.bench_local(|| run_uni_full::<f64, CPUScalar>(black_box(data), true));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_simd_seq(bencher: Bencher) {
    let data = get_data_f64();
    bencher.bench_local(|| run_uni_full::<f64, CPUSimd>(black_box(data), false));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_simd_par(bencher: Bencher) {
    let data = get_data_f64();
    bencher.bench_local(|| run_uni_full::<f64, CPUSimd>(black_box(data), true));
}

#[divan::bench]
fn f64_linfa_fit_seq(bencher: Bencher) {
    let data = get_data_f64();
    bencher.bench_local(|| run_linfa_full_f64(black_box(data), false));
}

#[divan::bench]
fn f64_linfa_fit_par(bencher: Bencher) {
    let data = get_data_f64();
    bencher.bench_local(|| run_linfa_full_f64(black_box(data), true));
}
