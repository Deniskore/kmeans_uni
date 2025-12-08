![Build Status](https://github.com/Deniskore/kmeans_uni/actions/workflows/ci.yml/badge.svg)
[![Crates.io](https://img.shields.io/crates/v/kmeans_uni.svg)](https://crates.io/crates/kmeans_uni)
[![API reference](https://docs.rs/kmeans_uni/badge.svg)](https://docs.rs/kmeans_uni)
[![License](https://img.shields.io/crates/l/kmeans_uni.svg)](https://crates.io/crates/kmeans_uni)

# kmeans_uni

Fast, safe K-Means++ for CPU-only workloads with optional SIMD acceleration. Supports Euclidean distance and dot-product scoring, provides both classic Lloyd iterations and a mini-batch variant, and includes parity tests against `linfa-clustering` to guard correctness. Benchmarks show significantly faster training and prediction than `linfa` on the same CPU. The crate builds on stable Rust.

## Key Features

- 100% safe Rust (`#![forbid(unsafe_code)]`) with a small dependency set.
- Optimized for speed, beats `linfa-clustering` in AArch64/x86_64 benches for training and prediction.
- Optional SIMD acceleration (`wide` feature) and WebAssembly support (see [`WASM.md`](./WASM.md)).
- Ergonomic builder API

## Quickstart

```rust
use kmeans_uni::KMeansBuilder;

const N_COLS: usize = 2;
let data: Vec<f32> = vec![
    1.0, 1.0,
    1.2, 0.9,
    -1.0, -1.1,
    -1.2, -0.8,
];

match KMeansBuilder::new(2)
    .iterations(100)
    .cpu_simd() // requires default "wide" feature
    .euclidean()
    .build()
    .fit(&data, N_COLS)
{
    Ok(model) => match model.predict(&data) {
        Ok(labels) => println!("labels: {labels:?}"),
        Err(err) => eprintln!("prediction failed: {err}"),
    },
    Err(err) => eprintln!("training failed: {err}"),
}
```

Mini-batch training for large datasets:

```rust
use kmeans_uni::KMeansBuilder;

match KMeansBuilder::new(8)
    .iterations(50) // iterations = number of batches
    .cpu_scalar()
    .euclidean()
    .mini_batch_rel_tolerance(0.0)
    .mini_batch_patience(0)
    .build()
    .fit_mini_batch_from_source(
        &kmeans_uni::SlicePointSource::new(&data, N_COLS).unwrap(),
        256, // batch size
    )
{
    Ok(model) => println!("centroids: {:?}", model.centroids),
    Err(err) => eprintln!("mini-batch training failed: {err}"),
}
```

## Crate features

- `wide` (default): enables SIMD CPU backend for K-Means (`f32` and `f64`) via the `wide` crate.
- As `std::simd` stabilizes, you can expect to squeeze more performance from
  portable SIMD without depending on `wide`.
- `serde`: derive `Serialize`/`Deserialize` on public types.
- `wasm`: build with a wasm-friendly configuration (sequential execution, no Rayon). See [`WASM.md`](./WASM.md) for a browser demo and build steps.
- Build without defaults (`--no-default-features`) to force scalar-only code paths.

## License

Licensed under either of:

- MIT license
- Apache License, Version 2.0

at your option.
