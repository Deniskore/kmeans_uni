use rand::Rng;
#[cfg(not(feature = "wasm"))]
use rayon::prelude::*;

use crate::Primitive;
use crate::backend::{CoreBackend, DistanceMetric};
use crate::kmeans_core_common::calculate_chunk_size;
use crate::point_source::PointSource;

use crate::error::{Error, Result};

pub trait InitializationStrategy:
    Clone + Copy + std::fmt::Debug + Default + Send + Sync + 'static
{
    fn initialize<
        F: Primitive,
        C: CoreBackend<F>,
        M: DistanceMetric<F>,
        S: PointSource<F>,
        E: ExecutionStrategy,
        R: Rng,
    >(
        source: &S,
        k: usize,
        rng: &mut R,
    ) -> Result<Vec<F>>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct KMeansPlusPlus;

impl InitializationStrategy for KMeansPlusPlus {
    fn initialize<
        F: Primitive,
        C: CoreBackend<F>,
        M: DistanceMetric<F>,
        S: PointSource<F>,
        E: ExecutionStrategy,
        R: Rng,
    >(
        source: &S,
        k: usize,
        rng: &mut R,
    ) -> Result<Vec<F>> {
        init_plus_plus_generic::<F, C, M, S, E, R>(source, k, rng)
    }
}

pub trait ExecutionStrategy: Send + Sync + 'static {
    fn initialize<
        F: Primitive,
        C: CoreBackend<F>,
        M: DistanceMetric<F>,
        S: PointSource<F>,
        I: InitializationStrategy,
        R: rand::Rng,
    >(
        source: &S,
        k: usize,
        rng: &mut R,
    ) -> Result<Vec<F>>;

    fn compute_stats_full<
        F: Primitive,
        C: CoreBackend<F>,
        M: DistanceMetric<F>,
        S: PointSource<F>,
    >(
        source: &S,
        ncols: usize,
        k: usize,
        prepared_centroids: &[F],
        par_chunk: usize,
    ) -> Result<(Vec<F>, Vec<usize>, f64)>;

    fn compute_stats_indexed<
        F: Primitive,
        C: CoreBackend<F>,
        M: DistanceMetric<F>,
        S: PointSource<F>,
    >(
        source: &S,
        ncols: usize,
        k: usize,
        prepared_centroids: &[F],
        indices: &[usize],
        par_chunk: usize,
    ) -> Result<(Vec<F>, Vec<usize>, f64)>;

    fn update_min_dists<F: Primitive, C: CoreBackend<F>, M: DistanceMetric<F>, S: PointSource<F>>(
        source: &S,
        ncols: usize,
        newest_centroid: &[F],
        min_dists: &mut [F],
        par_chunk: usize,
    ) -> Result<Vec<F>>;
}

pub struct Sequential;
#[cfg(not(feature = "wasm"))]
pub struct Parallel;
#[cfg(feature = "wasm")]
pub type Parallel = Sequential;

impl ExecutionStrategy for Sequential {
    fn initialize<
        F: Primitive,
        C: CoreBackend<F>,
        M: DistanceMetric<F>,
        S: PointSource<F>,
        I: InitializationStrategy,
        R: rand::Rng,
    >(
        source: &S,
        k: usize,
        rng: &mut R,
    ) -> Result<Vec<F>> {
        I::initialize::<F, C, M, S, Self, R>(source, k, rng)
    }

    fn compute_stats_full<
        F: Primitive,
        C: CoreBackend<F>,
        M: DistanceMetric<F>,
        S: PointSource<F>,
    >(
        source: &S,
        ncols: usize,
        k: usize,
        prepared_centroids: &[F],
        _par_chunk: usize,
    ) -> Result<(Vec<F>, Vec<usize>, f64)> {
        let mut sums = vec![F::zero(); k * ncols];
        let mut counts = vec![0usize; k];
        let mut inertia = 0.0_f64;

        let npoints = source.num_points();
        if npoints == 0 {
            return Ok((sums, counts, inertia));
        }

        let chunk_size = calculate_chunk_size::<F>(ncols);
        let mut buffer = vec![F::zero(); chunk_size * ncols];
        let mut distances = vec![F::zero(); chunk_size];
        let mut labels = vec![0usize; chunk_size];

        let mut processed = 0;
        while processed < npoints {
            let batch_size = (npoints - processed).min(chunk_size);
            let slice = &mut buffer[..batch_size * ncols];
            let batch_labels = &mut labels[..batch_size];
            let batch_dists = &mut distances[..batch_size];

            source.read_batch(processed, batch_size, slice);

            M::find_nearest::<C>(
                slice,
                ncols,
                prepared_centroids,
                k,
                batch_labels,
                Some(batch_dists),
            );

            for i in 0..batch_size {
                let best = batch_labels[i];
                let dist = batch_dists[i];
                let point = &slice[i * ncols..(i + 1) * ncols];

                counts[best] += 1;
                C::accumulate_point_slice(point, ncols, &mut sums, best);
                inertia += dist.to_f64().ok_or(Error::ConversionFailure)?;
            }

            processed += batch_size;
        }

        Ok((sums, counts, inertia))
    }

    fn compute_stats_indexed<
        F: Primitive,
        C: CoreBackend<F>,
        M: DistanceMetric<F>,
        S: PointSource<F>,
    >(
        source: &S,
        ncols: usize,
        k: usize,
        prepared_centroids: &[F],
        indices: &[usize],
        _par_chunk: usize,
    ) -> Result<(Vec<F>, Vec<usize>, f64)> {
        let mut sums = vec![F::zero(); k * ncols];
        let mut counts = vec![0usize; k];
        let mut inertia = 0.0_f64;

        let npoints = indices.len();
        if npoints == 0 {
            return Ok((sums, counts, inertia));
        }

        let chunk_size = calculate_chunk_size::<F>(ncols);
        let mut buffer = vec![F::zero(); chunk_size * ncols];
        let mut distances = vec![F::zero(); chunk_size];
        let mut labels = vec![0usize; chunk_size];

        let mut processed = 0;
        while processed < npoints {
            let batch_size = (npoints - processed).min(chunk_size);
            let slice = &mut buffer[..batch_size * ncols];
            let batch_labels = &mut labels[..batch_size];
            let batch_dists = &mut distances[..batch_size];

            let mut current_batch_idx = 0;
            while current_batch_idx < batch_size {
                let global_idx_pos = processed + current_batch_idx;
                let start_idx = indices[global_idx_pos];
                let mut run_end = global_idx_pos + 1;
                while run_end < npoints
                    && run_end < processed + batch_size
                    && indices[run_end] == indices[run_end - 1] + 1
                {
                    run_end += 1;
                }
                let run_len = run_end - global_idx_pos;

                let run_slice =
                    &mut slice[current_batch_idx * ncols..(current_batch_idx + run_len) * ncols];
                source.read_batch(start_idx, run_len, run_slice);

                current_batch_idx += run_len;
            }

            M::find_nearest_with_dists::<C>(
                slice,
                ncols,
                prepared_centroids,
                k,
                batch_labels,
                batch_dists,
            );

            for i in 0..batch_size {
                let best = batch_labels[i];
                let dist = batch_dists[i];
                let point = &slice[i * ncols..(i + 1) * ncols];

                counts[best] += 1;
                C::accumulate_point_slice(point, ncols, &mut sums, best);
                inertia += dist.to_f64().ok_or(Error::ConversionFailure)?;
            }

            processed += batch_size;
        }

        Ok((sums, counts, inertia))
    }

    fn update_min_dists<
        F: Primitive,
        C: CoreBackend<F>,
        M: DistanceMetric<F>,
        S: PointSource<F>,
    >(
        source: &S,
        ncols: usize,
        newest_centroid: &[F],
        min_dists: &mut [F],
        _par_chunk: usize,
    ) -> Result<Vec<F>> {
        let npoints = source.num_points();
        let chunk_size = calculate_chunk_size::<F>(ncols);
        let num_chunks = npoints.div_ceil(chunk_size);
        let mut chunk_sums = vec![F::zero(); num_chunks];

        let mut point_batch = vec![F::zero(); chunk_size * ncols];

        let mut offset = 0;
        let mut chunk_idx = 0;
        while offset < npoints {
            let count = (npoints - offset).min(chunk_size);
            let current_points = &mut point_batch[..count * ncols];
            source.read_batch(offset, count, current_points);

            let chunk_sum = M::calculate_and_update_min_distance_sum::<C>(
                current_points,
                ncols,
                newest_centroid,
                &mut min_dists[offset..offset + count],
            );
            chunk_sums[chunk_idx] = chunk_sum;

            offset += count;
            chunk_idx += 1;
        }
        Ok(chunk_sums)
    }
}

#[cfg(not(feature = "wasm"))]
impl ExecutionStrategy for Parallel {
    fn initialize<
        F: Primitive,
        C: CoreBackend<F>,
        M: DistanceMetric<F>,
        S: PointSource<F>,
        I: InitializationStrategy,
        R: rand::Rng,
    >(
        source: &S,
        k: usize,
        rng: &mut R,
    ) -> Result<Vec<F>> {
        I::initialize::<F, C, M, S, Self, R>(source, k, rng)
    }

    fn compute_stats_full<
        F: Primitive,
        C: CoreBackend<F>,
        M: DistanceMetric<F>,
        S: PointSource<F>,
    >(
        source: &S,
        ncols: usize,
        k: usize,
        prepared_centroids: &[F],
        par_chunk: usize,
    ) -> Result<(Vec<F>, Vec<usize>, f64)> {
        let npoints = source.num_points();
        if npoints == 0 {
            return Ok((vec![F::zero(); k * ncols], vec![0usize; k], 0.0_f64));
        }

        let par_chunk = par_chunk.min(npoints).max(1);
        let num_chunks = npoints.div_ceil(par_chunk);

        let res = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * par_chunk;
                let end = (start + par_chunk).min(npoints);
                let len = end - start;

                let mut local_sums = vec![F::zero(); k * ncols];
                let mut local_counts = vec![0usize; k];
                let mut local_inertia = 0.0_f64;

                let mut buffer = vec![F::zero(); len * ncols];
                let mut labels = vec![0usize; len];
                let mut distances = vec![F::zero(); len];

                source.read_batch(start, len, &mut buffer);

                M::find_nearest::<C>(
                    &buffer,
                    ncols,
                    prepared_centroids,
                    k,
                    &mut labels,
                    Some(&mut distances),
                );

                for i in 0..len {
                    let best = labels[i];
                    let dist = distances[i];
                    let point = &buffer[i * ncols..(i + 1) * ncols];

                    local_counts[best] += 1;
                    C::accumulate_point_slice(point, ncols, &mut local_sums, best);
                    local_inertia += dist.to_f64().ok_or(Error::ConversionFailure)?;
                }

                Ok((local_sums, local_counts, local_inertia))
            })
            .reduce(
                || Ok((vec![F::zero(); k * ncols], vec![0usize; k], 0.0_f64)),
                |a, b| {
                    let (mut sums_a, mut counts_a, inertia_a) = a?;
                    let (sums_b, counts_b, inertia_b) = b?;
                    for i in 0..sums_a.len() {
                        sums_a[i] = sums_a[i] + sums_b[i];
                    }
                    for i in 0..counts_a.len() {
                        counts_a[i] += counts_b[i];
                    }
                    Ok((sums_a, counts_a, inertia_a + inertia_b))
                },
            )?;

        Ok(res)
    }

    fn compute_stats_indexed<
        F: Primitive,
        C: CoreBackend<F>,
        M: DistanceMetric<F>,
        S: PointSource<F>,
    >(
        source: &S,
        ncols: usize,
        k: usize,
        prepared_centroids: &[F],
        indices: &[usize],
        par_chunk: usize,
    ) -> Result<(Vec<F>, Vec<usize>, f64)> {
        let npoints = indices.len();
        if npoints == 0 {
            return Ok((vec![F::zero(); k * ncols], vec![0usize; k], 0.0_f64));
        }

        let par_chunk = par_chunk.min(npoints).max(1);
        let num_chunks = npoints.div_ceil(par_chunk);

        let res = (0..num_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * par_chunk;
                let end = (start + par_chunk).min(npoints);
                let len = end - start;

                let mut local_sums = vec![F::zero(); k * ncols];
                let mut local_counts = vec![0usize; k];
                let mut local_inertia = 0.0_f64;

                let mut buffer = vec![F::zero(); len * ncols];
                let mut labels = vec![0usize; len];
                let mut distances = vec![F::zero(); len];

                let chunk_indices = &indices[start..end];
                let mut current_idx = 0;
                while current_idx < len {
                    let start_idx = chunk_indices[current_idx];
                    let mut run_end = current_idx + 1;
                    while run_end < len && chunk_indices[run_end] == chunk_indices[run_end - 1] + 1
                    {
                        run_end += 1;
                    }
                    let run_len = run_end - current_idx;
                    let slice = &mut buffer[current_idx * ncols..(current_idx + run_len) * ncols];
                    source.read_batch(start_idx, run_len, slice);
                    current_idx = run_end;
                }

                M::find_nearest_with_dists::<C>(
                    &buffer,
                    ncols,
                    prepared_centroids,
                    k,
                    &mut labels,
                    &mut distances,
                );

                for i in 0..len {
                    let best = labels[i];
                    let dist = distances[i];
                    let point = &buffer[i * ncols..(i + 1) * ncols];

                    local_counts[best] += 1;
                    C::accumulate_point_slice(point, ncols, &mut local_sums, best);
                    local_inertia += dist.to_f64().ok_or(Error::ConversionFailure)?;
                }

                Ok((local_sums, local_counts, local_inertia))
            })
            .reduce(
                || Ok((vec![F::zero(); k * ncols], vec![0usize; k], 0.0_f64)),
                |a, b| {
                    let (mut sums_a, mut counts_a, inertia_a) = a?;
                    let (sums_b, counts_b, inertia_b) = b?;
                    for i in 0..sums_a.len() {
                        sums_a[i] = sums_a[i] + sums_b[i];
                    }
                    for i in 0..counts_a.len() {
                        counts_a[i] += counts_b[i];
                    }
                    Ok((sums_a, counts_a, inertia_a + inertia_b))
                },
            )?;

        Ok(res)
    }

    fn update_min_dists<
        F: Primitive,
        C: CoreBackend<F>,
        M: DistanceMetric<F>,
        S: PointSource<F>,
    >(
        source: &S,
        ncols: usize,
        newest_centroid: &[F],
        min_dists: &mut [F],
        par_chunk: usize,
    ) -> Result<Vec<F>> {
        let npoints = source.num_points();
        if npoints == 0 {
            return Ok(Vec::new());
        }
        let chunk_size = par_chunk.clamp(1, npoints);
        let num_chunks = npoints.div_ceil(chunk_size);
        let mut chunk_sums = vec![F::zero(); num_chunks];

        chunk_sums
            .par_iter_mut()
            .zip(min_dists.par_chunks_mut(chunk_size).enumerate())
            .for_each_init(
                || Vec::new(),
                |points, (sum_slot, (chunk_idx, min_chunk))| {
                    let start = chunk_idx * chunk_size;
                    let len = min_chunk.len();

                    points.resize(len * ncols, F::zero());
                    source.read_batch(start, len, points);

                    let chunk_sum = M::calculate_and_update_min_distance_sum::<C>(
                        points,
                        ncols,
                        newest_centroid,
                        min_chunk,
                    );
                    *sum_slot = chunk_sum;
                },
            );

        Ok(chunk_sums)
    }
}

fn pick_weighted_index<F: Primitive>(
    target: F,
    chunk_sums: &[F],
    chunk_size: usize,
    min_dists: &[F],
    npoints: usize,
) -> usize {
    let mut prefix = F::zero();
    let mut chunk_idx = chunk_sums.len().saturating_sub(1);

    for (idx, &chunk_sum) in chunk_sums.iter().enumerate() {
        let next = prefix + chunk_sum;
        if next >= target {
            chunk_idx = idx;
            break;
        }
        prefix = next;
    }

    let chunk_start = chunk_idx * chunk_size;
    let chunk_end = (chunk_start + chunk_size).min(npoints);
    let mut running = F::zero();
    let mut selected = chunk_end.saturating_sub(1);

    for (offset, &d) in min_dists[chunk_start..chunk_end].iter().enumerate() {
        running = running + d;
        if prefix + running >= target {
            selected = chunk_start + offset;
            break;
        }
    }

    selected
}

pub(crate) fn init_plus_plus_generic<
    F: Primitive,
    C: crate::backend::CoreBackend<F>,
    M: crate::backend::DistanceMetric<F>,
    S: PointSource<F>,
    E: ExecutionStrategy,
    R: Rng,
>(
    source: &S,
    k: usize,
    rng: &mut R,
) -> Result<Vec<F>> {
    let ncols = source.num_columns();
    let npoints = source.num_points();
    if npoints == 0 || k == 0 {
        return Ok(vec![F::zero(); k * ncols]);
    }
    let chunk_size = calculate_chunk_size::<F>(ncols);

    let mut centroids = Vec::with_capacity(k * ncols);
    let mut min_dists = vec![F::infinity(); npoints];

    // 1. Choose first centroid uniformly at random
    let first_idx = rng.random_range(0..npoints);
    let mut point_buf = vec![F::zero(); ncols];
    source.read_batch(first_idx, 1, &mut point_buf);
    centroids.extend_from_slice(&point_buf);

    for _ in 1..k {
        let newest_centroid = &centroids[(centroids.len() / ncols - 1) * ncols..];

        let chunk_sums = E::update_min_dists::<F, C, M, S>(
            source,
            ncols,
            newest_centroid,
            &mut min_dists,
            chunk_size,
        )?;

        let sum_sq_dist = chunk_sums.iter().copied().fold(F::zero(), |a, b| a + b);

        // Sample next centroid
        if sum_sq_dist <= F::zero() {
            let next_idx = rng.random_range(0..npoints);
            source.read_batch(next_idx, 1, &mut point_buf);
            centroids.extend_from_slice(&point_buf);
        } else {
            let target =
                F::from(rng.random::<f32>()).ok_or(Error::ConversionFailure)? * sum_sq_dist;
            let selected_idx =
                pick_weighted_index(target, &chunk_sums, chunk_size, &min_dists, npoints);
            source.read_batch(selected_idx, 1, &mut point_buf);
            centroids.extend_from_slice(&point_buf);
        }
    }

    Ok(centroids)
}
