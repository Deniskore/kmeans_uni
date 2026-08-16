# Changelog

## [0.2.0]

### Added

- Added `CPUSimdAdaptive` plus explicit `CPUSimd128`, `CPUSimd256`, and `CPUSimd512` logical-width backends on all targets supported by the `wide` feature.
- Added `.cpu_simd128()`, `.cpu_simd256()`, and `.cpu_simd512()` builder selection, matching `fit_default_simd128/256/512` helpers, and reusable prepared-centroid caching for every SIMD width.
- Added explicit-width and adaptive shape benchmarks for fitting and prediction.
- Added backend-selectable prediction and transform APIs for explicit scalar or SIMD inference.
- Added support for `PointSource` implementations that only provide `read_batch`, allowing data sources to stream batches without implementing `view_batch`.
- Added broader tests for copy-only sources and scalar/SIMD prediction and transform parity.
- Split benchmarks into focused modules for initialization, full fitting, mini-batch fitting, prediction, transform, and shape coverage.

### Changed

- `.cpu_simd()` and default inference now use the conservative shape-adaptive backend. It maps directly to the 128-bit core on AArch64/WebAssembly and avoids width choices that regressed another operation in the benchmark matrix.
- `CPUSimd256` and `CPUSimd512` use their stated logical widths on every target; `wide` composes them from narrower vectors when the target lacks matching native registers.
- `Primitive` is now sealed to the supported `f32`/`f64` types and includes a default inference backend associated type used by prediction and transform.
- Hid the internal `kmeans_cpu` module from the public API; supported entry points remain `KMeans`, `KMeansBuilder`, and `KMeansConfig`.

### Removed

- Removed the public `CPUSimd` backend type. Code that names a backend must now use `CPUSimdAdaptive`, `CPUSimd128`, `CPUSimd256`, or `CPUSimd512` explicitly.

### Maintenance

- Updated dependencies and migrated SIMD mask selection to the non-deprecated `wide` 1.6 API.

### Performance

- Reused full-fit and mini-batch iteration scratch storage, including sequential buffers and parallel worker accumulators, instead of reallocating it on every iteration.
- Reused empty-cluster sampling buffers during centroid updates and assigned contiguous chunk ranges to parallel fitting workers.
- Balanced native prediction and transform chunks across available Rayon workers, with benchmark-tuned sequential cutoffs for smaller workloads.
- Shared prepared-centroid storage between adaptive and matching explicit SIMD backends; scalar inference now borrows the model centroids directly.
- Improved inference and training hot paths with reusable centroid preparation, chunk-level batching, and fused assignment/accumulation work.
- Default prediction and transform now use the fastest available inference backend for the enabled feature set.
- Refined SIMD paths for centroid assignment, transform, centroid updates, and K-Means++ initialization.
- Benchmarks show full fitting with roughly 80% lower median runtime versus `linfa-clustering` in the headline parallel cases, with some wider SIMD-backed `f32` shapes above 90% lower.
- Mini-batch fitting is substantially faster in both scalar and SIMD modes, with SIMD-backed median runtimes roughly 55-70% lower than the comparable `linfa-clustering` runs.
- Prediction benefits from the prepared-centroid cache and default SIMD inference, with headline median runtimes roughly 40-60% lower than `linfa-clustering`.
- Transform now uses the optimized backend path too: the SIMD `f32` transform benchmark is roughly 40% lower median runtime than the scalar path.
- Copy-only point sources now stay close to slice-backed prediction performance, generally within about 10% in the benchmarked cases, while avoiding unnecessary failed view probes.
