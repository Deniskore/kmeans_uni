#[path = "kmeans_bench/common.rs"]
mod common;
#[path = "kmeans_bench/full_fit.rs"]
mod full_fit;
#[path = "kmeans_bench/init.rs"]
mod init;
#[path = "kmeans_bench/minibatch.rs"]
mod minibatch;
#[path = "kmeans_bench/predict.rs"]
mod predict;
#[path = "kmeans_bench/shape_matrix.rs"]
mod shape_matrix;
#[path = "kmeans_bench/transform.rs"]
mod transform;

fn main() {
    common::warm_up();
    divan::main();
}
