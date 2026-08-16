use crate::common::{
    get_data_f32, get_data_f64, run_linfa_full_f32, run_linfa_full_f64, run_uni_full,
};
use divan::{Bencher, black_box};
use kmeans_uni::CPUScalar;
#[cfg(feature = "wide")]
use kmeans_uni::{CPUSimd128, CPUSimd256, CPUSimd512, CPUSimdAdaptive};

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
fn f32_uni_simd128_seq(bencher: Bencher) {
    let data = get_data_f32();
    bencher.bench_local(|| run_uni_full::<f32, CPUSimd128>(black_box(data), false));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_simd_adaptive_seq(bencher: Bencher) {
    let data = get_data_f32();
    bencher.bench_local(|| run_uni_full::<f32, CPUSimdAdaptive>(black_box(data), false));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_simd_adaptive_par(bencher: Bencher) {
    let data = get_data_f32();
    bencher.bench_local(|| run_uni_full::<f32, CPUSimdAdaptive>(black_box(data), true));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_simd256_seq(bencher: Bencher) {
    let data = get_data_f32();
    bencher.bench_local(|| run_uni_full::<f32, CPUSimd256>(black_box(data), false));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_simd256_par(bencher: Bencher) {
    let data = get_data_f32();
    bencher.bench_local(|| run_uni_full::<f32, CPUSimd256>(black_box(data), true));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_simd512_seq(bencher: Bencher) {
    let data = get_data_f32();
    bencher.bench_local(|| run_uni_full::<f32, CPUSimd512>(black_box(data), false));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_simd512_par(bencher: Bencher) {
    let data = get_data_f32();
    bencher.bench_local(|| run_uni_full::<f32, CPUSimd512>(black_box(data), true));
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
fn f64_uni_simd128_seq(bencher: Bencher) {
    let data = get_data_f64();
    bencher.bench_local(|| run_uni_full::<f64, CPUSimd128>(black_box(data), false));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_simd_adaptive_seq(bencher: Bencher) {
    let data = get_data_f64();
    bencher.bench_local(|| run_uni_full::<f64, CPUSimdAdaptive>(black_box(data), false));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_simd_adaptive_par(bencher: Bencher) {
    let data = get_data_f64();
    bencher.bench_local(|| run_uni_full::<f64, CPUSimdAdaptive>(black_box(data), true));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_simd256_seq(bencher: Bencher) {
    let data = get_data_f64();
    bencher.bench_local(|| run_uni_full::<f64, CPUSimd256>(black_box(data), false));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_simd256_par(bencher: Bencher) {
    let data = get_data_f64();
    bencher.bench_local(|| run_uni_full::<f64, CPUSimd256>(black_box(data), true));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_simd512_seq(bencher: Bencher) {
    let data = get_data_f64();
    bencher.bench_local(|| run_uni_full::<f64, CPUSimd512>(black_box(data), false));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f64_uni_simd512_par(bencher: Bencher) {
    let data = get_data_f64();
    bencher.bench_local(|| run_uni_full::<f64, CPUSimd512>(black_box(data), true));
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
