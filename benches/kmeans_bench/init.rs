use crate::common::{CopyOnlySource, N_COLS, get_data_f32, run_uni_init_from_source};
use divan::{Bencher, black_box};
#[cfg(feature = "wide")]
use kmeans_uni::CPUSimd;
use kmeans_uni::{CPUScalar, SlicePointSource};

// No Linfa counterpart is listed here: Linfa does not expose a public
// init-only KMeans++ operation. A one-iteration Linfa fit would include
// assignment/update work and would not measure the same unit.

#[divan::bench]
fn f32_uni_init_scalar_seq(bencher: Bencher) {
    let data = get_data_f32();
    let source = SlicePointSource::new(data, N_COLS).expect("slice source");
    bencher
        .bench_local(|| run_uni_init_from_source::<f32, CPUScalar, _>(black_box(&source), false));
}

#[divan::bench]
fn f32_uni_init_scalar_copy_only_seq(bencher: Bencher) {
    let data = get_data_f32();
    let source = CopyOnlySource::new(data.to_vec(), N_COLS);
    bencher
        .bench_local(|| run_uni_init_from_source::<f32, CPUScalar, _>(black_box(&source), false));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_init_simd_seq(bencher: Bencher) {
    let data = get_data_f32();
    let source = SlicePointSource::new(data, N_COLS).expect("slice source");
    bencher.bench_local(|| run_uni_init_from_source::<f32, CPUSimd, _>(black_box(&source), false));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_init_simd_par(bencher: Bencher) {
    let data = get_data_f32();
    let source = SlicePointSource::new(data, N_COLS).expect("slice source");
    bencher.bench_local(|| run_uni_init_from_source::<f32, CPUSimd, _>(black_box(&source), true));
}

#[cfg(feature = "wide")]
#[divan::bench]
fn f32_uni_init_simd_copy_only_par(bencher: Bencher) {
    let data = get_data_f32();
    let source = CopyOnlySource::new(data.to_vec(), N_COLS);
    bencher.bench_local(|| run_uni_init_from_source::<f32, CPUSimd, _>(black_box(&source), true));
}
