use crate::common::{
    Shape, linfa_dataset_f32, linfa_dataset_f64, run_linfa_full_shape_f32,
    run_linfa_full_shape_f64, run_uni_full_shape, shape_data_f32, shape_data_f64,
};
use divan::{Bencher, black_box};
use kmeans_uni::CPUScalar;
#[cfg(feature = "wide")]
use kmeans_uni::CPUSimd;
use linfa::prelude::Predict;

// Shape matrix benchmarks are intentionally split into two stories:
// Uni scalar vs Uni SIMD is a controlled backend comparison inside this crate.
// Linfa is an external product-level reference on the same data shape/config, not a
// proof of identical algorithm kernels.

macro_rules! shape_fit_set {
    ($uni_scalar:ident, $uni_simd:ident, $linfa:ident, $shape:expr) => {
        #[divan::bench]
        fn $uni_scalar(bencher: Bencher) {
            let shape = $shape;
            let data = shape_data_f32(shape);
            bencher.bench_local(|| {
                run_uni_full_shape::<f32, CPUScalar>(black_box(data.as_slice()), shape, false)
            });
        }

        #[cfg(feature = "wide")]
        #[divan::bench]
        fn $uni_simd(bencher: Bencher) {
            let shape = $shape;
            let data = shape_data_f32(shape);
            bencher.bench_local(|| {
                run_uni_full_shape::<f32, CPUSimd>(black_box(data.as_slice()), shape, false)
            });
        }

        #[divan::bench]
        fn $linfa(bencher: Bencher) {
            let shape = $shape;
            let data = shape_data_f32(shape);
            bencher
                .bench_local(|| run_linfa_full_shape_f32(black_box(data.as_slice()), shape, false));
        }
    };
}

macro_rules! shape_fit_set_f64 {
    ($uni_scalar:ident, $uni_simd:ident, $linfa:ident, $shape:expr) => {
        #[divan::bench]
        fn $uni_scalar(bencher: Bencher) {
            let shape = $shape;
            let data = shape_data_f64(shape);
            bencher.bench_local(|| {
                run_uni_full_shape::<f64, CPUScalar>(black_box(data.as_slice()), shape, false)
            });
        }

        #[cfg(feature = "wide")]
        #[divan::bench]
        fn $uni_simd(bencher: Bencher) {
            let shape = $shape;
            let data = shape_data_f64(shape);
            bencher.bench_local(|| {
                run_uni_full_shape::<f64, CPUSimd>(black_box(data.as_slice()), shape, false)
            });
        }

        #[divan::bench]
        fn $linfa(bencher: Bencher) {
            let shape = $shape;
            let data = shape_data_f64(shape);
            bencher
                .bench_local(|| run_linfa_full_shape_f64(black_box(data.as_slice()), shape, false));
        }
    };
}

macro_rules! shape_predict_set {
    ($uni_scalar:ident, $uni_simd:ident, $linfa:ident, $shape:expr) => {
        #[divan::bench]
        fn $uni_scalar(bencher: Bencher) {
            let shape = $shape;
            let data = shape_data_f32(shape);
            let model = run_uni_full_shape::<f32, CPUScalar>(data.as_slice(), shape, false);
            bencher.bench_local(|| {
                black_box(&model)
                    .predict_with_backend::<CPUScalar>(black_box(data.as_slice()))
                    .unwrap()
            });
        }

        #[cfg(feature = "wide")]
        #[divan::bench]
        fn $uni_simd(bencher: Bencher) {
            let shape = $shape;
            let data = shape_data_f32(shape);
            let model = run_uni_full_shape::<f32, CPUSimd>(data.as_slice(), shape, false);
            bencher.bench_local(|| {
                black_box(&model)
                    .predict_with_backend::<CPUSimd>(black_box(data.as_slice()))
                    .unwrap()
            });
        }

        #[divan::bench]
        fn $linfa(bencher: Bencher) {
            let shape = $shape;
            let data = shape_data_f32(shape);
            let model = run_linfa_full_shape_f32(data.as_slice(), shape, false);
            let dataset = linfa_dataset_f32(data.as_slice(), shape);
            bencher.bench_local(|| black_box(&model).predict(black_box(&dataset)));
        }
    };
}

macro_rules! shape_predict_set_f64 {
    ($uni_scalar:ident, $uni_simd:ident, $linfa:ident, $shape:expr) => {
        #[divan::bench]
        fn $uni_scalar(bencher: Bencher) {
            let shape = $shape;
            let data = shape_data_f64(shape);
            let model = run_uni_full_shape::<f64, CPUScalar>(data.as_slice(), shape, false);
            bencher.bench_local(|| {
                black_box(&model)
                    .predict_with_backend::<CPUScalar>(black_box(data.as_slice()))
                    .unwrap()
            });
        }

        #[cfg(feature = "wide")]
        #[divan::bench]
        fn $uni_simd(bencher: Bencher) {
            let shape = $shape;
            let data = shape_data_f64(shape);
            let model = run_uni_full_shape::<f64, CPUSimd>(data.as_slice(), shape, false);
            bencher.bench_local(|| {
                black_box(&model)
                    .predict_with_backend::<CPUSimd>(black_box(data.as_slice()))
                    .unwrap()
            });
        }

        #[divan::bench]
        fn $linfa(bencher: Bencher) {
            let shape = $shape;
            let data = shape_data_f64(shape);
            let model = run_linfa_full_shape_f64(data.as_slice(), shape, false);
            let dataset = linfa_dataset_f64(data.as_slice(), shape);
            bencher.bench_local(|| black_box(&model).predict(black_box(&dataset)));
        }
    };
}

shape_fit_set!(
    shape_fit_f32_cols8_k8_uni_scalar_seq,
    shape_fit_f32_cols8_k8_uni_simd_seq,
    shape_fit_f32_cols8_k8_linfa_seq,
    Shape::new(8192, 8, 8)
);

shape_fit_set!(
    shape_fit_f32_cols15_k16_uni_scalar_seq,
    shape_fit_f32_cols15_k16_uni_simd_seq,
    shape_fit_f32_cols15_k16_linfa_seq,
    Shape::new(8192, 15, 16)
);

shape_fit_set!(
    shape_fit_f32_cols16_k17_uni_scalar_seq,
    shape_fit_f32_cols16_k17_uni_simd_seq,
    shape_fit_f32_cols16_k17_linfa_seq,
    Shape::new(8192, 16, 17)
);

shape_fit_set!(
    shape_fit_f32_cols64_k32_uni_scalar_seq,
    shape_fit_f32_cols64_k32_uni_simd_seq,
    shape_fit_f32_cols64_k32_linfa_seq,
    Shape::new(4096, 64, 32)
);

shape_fit_set_f64!(
    shape_fit_f64_cols16_k16_uni_scalar_seq,
    shape_fit_f64_cols16_k16_uni_simd_seq,
    shape_fit_f64_cols16_k16_linfa_seq,
    Shape::new(4096, 16, 16)
);

shape_fit_set_f64!(
    shape_fit_f64_cols17_k17_uni_scalar_seq,
    shape_fit_f64_cols17_k17_uni_simd_seq,
    shape_fit_f64_cols17_k17_linfa_seq,
    Shape::new(4096, 17, 17)
);

shape_predict_set!(
    shape_predict_f32_cols8_k8_uni_scalar,
    shape_predict_f32_cols8_k8_uni_simd,
    shape_predict_f32_cols8_k8_linfa,
    Shape::new(8192, 8, 8)
);

shape_predict_set!(
    shape_predict_f32_cols15_k16_uni_scalar,
    shape_predict_f32_cols15_k16_uni_simd,
    shape_predict_f32_cols15_k16_linfa,
    Shape::new(8192, 15, 16)
);

shape_predict_set!(
    shape_predict_f32_cols16_k17_uni_scalar,
    shape_predict_f32_cols16_k17_uni_simd,
    shape_predict_f32_cols16_k17_linfa,
    Shape::new(8192, 16, 17)
);

shape_predict_set!(
    shape_predict_f32_cols64_k32_uni_scalar,
    shape_predict_f32_cols64_k32_uni_simd,
    shape_predict_f32_cols64_k32_linfa,
    Shape::new(4096, 64, 32)
);

shape_predict_set_f64!(
    shape_predict_f64_cols16_k16_uni_scalar,
    shape_predict_f64_cols16_k16_uni_simd,
    shape_predict_f64_cols16_k16_linfa,
    Shape::new(4096, 16, 16)
);

shape_predict_set_f64!(
    shape_predict_f64_cols17_k17_uni_scalar,
    shape_predict_f64_cols17_k17_uni_simd,
    shape_predict_f64_cols17_k17_linfa,
    Shape::new(4096, 17, 17)
);
