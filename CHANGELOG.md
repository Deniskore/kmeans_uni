# Changelog

## [0.2.0] - 2026-05-01

### Added

- Added backend-selectable prediction and transform APIs for explicit scalar or SIMD inference.
- Added support for `PointSource` implementations that only provide `read_batch`, allowing data sources to stream batches without implementing `view_batch`.
- Added broader tests for copy-only sources and scalar/SIMD prediction and transform parity.
- Split benchmarks into focused modules for initialization, full fitting, mini-batch fitting, prediction, transform, and shape coverage.

### Changed

- `Primitive` is now sealed to the supported `f32`/`f64` types and includes a default inference backend associated type used by prediction and transform.
- Hid the internal `kmeans_cpu` module from the public API; supported entry points remain `KMeans`, `KMeansBuilder`, and `KMeansConfig`.

### Maintenance

- Updated dependencies.

### Performance

- Improved inference and training hot paths with reusable centroid preparation, chunk-level batching, and fused assignment/accumulation work.
- Default prediction and transform now use the fastest available inference backend for the enabled feature set.
- Refined SIMD paths for centroid assignment, transform, centroid updates, and K-Means++ initialization.
- Benchmarks show full fitting with roughly 80% lower median runtime versus `linfa-clustering` in the headline parallel cases, with some wider SIMD-backed `f32` shapes above 90% lower.
- Mini-batch fitting is substantially faster in both scalar and SIMD modes, with SIMD-backed median runtimes roughly 55-70% lower than the comparable `linfa-clustering` runs.
- Prediction benefits from the prepared-centroid cache and default SIMD inference, with headline median runtimes roughly 40-60% lower than `linfa-clustering`.
- Transform now uses the optimized backend path too: the SIMD `f32` transform benchmark is roughly 40% lower median runtime than the scalar path.
- Copy-only point sources now stay close to slice-backed prediction performance, generally within about 10% in the benchmarked cases, while avoiding unnecessary failed view probes.
