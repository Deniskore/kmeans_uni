use rand::{Rng, RngExt};
#[cfg(not(feature = "wasm"))]
use rayon::prelude::*;

use crate::Primitive;
use crate::backend::{CoreBackend, DistanceMetric};
use crate::kmeans_core_common::calculate_chunk_size;
use crate::point_source::{PointSource, view_or_copy_batch};

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

pub struct IterationScratch<F> {
    sums: Vec<F>,
    counts: Vec<usize>,
    fallback_buffer: Option<Vec<F>>,
    indexed_buffer: Vec<F>,
    empty_cluster_buffer: Vec<F>,
    #[cfg(not(feature = "wasm"))]
    parallel_workers: Vec<ParallelIterationScratch<F>>,
}

#[cfg(not(feature = "wasm"))]
struct ParallelIterationScratch<F> {
    sums: Vec<F>,
    counts: Vec<usize>,
    inertia: f64,
    fallback_buffer: Option<Vec<F>>,
    indexed_buffer: Vec<F>,
}

impl<F: Primitive> IterationScratch<F> {
    pub(crate) fn new(k: usize, ncols: usize) -> Self {
        Self {
            sums: vec![F::zero(); k * ncols],
            counts: vec![0; k],
            fallback_buffer: None,
            indexed_buffer: Vec::new(),
            empty_cluster_buffer: Vec::new(),
            #[cfg(not(feature = "wasm"))]
            parallel_workers: Vec::new(),
        }
    }

    #[inline]
    fn reset_totals(&mut self) {
        self.sums.fill(F::zero());
        self.counts.fill(0);
    }

    #[inline]
    pub(crate) fn sums(&self) -> &[F] {
        &self.sums
    }

    #[inline]
    pub(crate) fn counts(&self) -> &[usize] {
        &self.counts
    }

    #[inline]
    pub(crate) fn centroid_update_parts(&mut self) -> (&[F], &[usize], &mut Vec<F>) {
        (&self.sums, &self.counts, &mut self.empty_cluster_buffer)
    }

    #[inline]
    pub(crate) fn empty_cluster_buffer(&mut self) -> &mut Vec<F> {
        &mut self.empty_cluster_buffer
    }

    #[cfg(not(feature = "wasm"))]
    fn prepare_parallel_workers(&mut self, worker_count: usize, indexed_buffer_len: usize) {
        while self.parallel_workers.len() < worker_count {
            self.parallel_workers.push(ParallelIterationScratch {
                sums: vec![F::zero(); self.sums.len()],
                counts: vec![0; self.counts.len()],
                inertia: 0.0,
                fallback_buffer: None,
                indexed_buffer: vec![F::zero(); indexed_buffer_len],
            });
        }

        for worker in &mut self.parallel_workers[..worker_count] {
            worker.sums.fill(F::zero());
            worker.counts.fill(0);
            worker.inertia = 0.0;
            if worker.indexed_buffer.len() < indexed_buffer_len {
                worker.indexed_buffer.resize(indexed_buffer_len, F::zero());
            }
        }
    }

    #[cfg(not(feature = "wasm"))]
    fn merge_parallel_workers(&mut self, worker_count: usize) -> f64 {
        self.reset_totals();
        let mut inertia = 0.0;
        for worker in &self.parallel_workers[..worker_count] {
            for (sum, &local_sum) in self.sums.iter_mut().zip(&worker.sums) {
                *sum = *sum + local_sum;
            }
            for (count, &local_count) in self.counts.iter_mut().zip(&worker.counts) {
                *count += local_count;
            }
            inertia += worker.inertia;
        }
        inertia
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
        scratch: &mut IterationScratch<F>,
    ) -> Result<f64>;

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
        scratch: &mut IterationScratch<F>,
    ) -> Result<f64>;

    fn update_min_dists<F: Primitive, C: CoreBackend<F>, M: DistanceMetric<F>, S: PointSource<F>>(
        source: &S,
        ncols: usize,
        newest_centroid: &[F],
        min_dists: &mut [F],
        par_chunk: usize,
        chunk_sums: &mut [F],
    ) -> Result<()>;
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
        scratch: &mut IterationScratch<F>,
    ) -> Result<f64> {
        scratch.reset_totals();
        let mut inertia = 0.0_f64;

        let npoints = source.num_points();
        if npoints == 0 {
            return Ok(0.0);
        }

        let chunk_size = calculate_chunk_size::<F>(ncols);

        let mut processed = 0;
        while processed < npoints {
            let batch_size = (npoints - processed).min(chunk_size);
            let points = view_or_copy_batch(
                source,
                &mut scratch.fallback_buffer,
                processed,
                batch_size,
                chunk_size,
                ncols,
            )?;
            let mut batch_inertia = F::zero();
            M::assign_and_accumulate::<C>(
                points,
                ncols,
                prepared_centroids,
                k,
                &mut scratch.sums,
                &mut scratch.counts,
                &mut batch_inertia,
            );
            inertia += batch_inertia.to_f64().ok_or(Error::ConversionFailure)?;

            processed += batch_size;
        }

        Ok(inertia)
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
        scratch: &mut IterationScratch<F>,
    ) -> Result<f64> {
        scratch.reset_totals();
        let mut inertia = 0.0_f64;

        let npoints = indices.len();
        if npoints == 0 {
            return Ok(0.0);
        }

        let chunk_size = calculate_chunk_size::<F>(ncols);
        let buffer_len = chunk_size * ncols;
        if scratch.indexed_buffer.len() < buffer_len {
            scratch.indexed_buffer.resize(buffer_len, F::zero());
        }

        let mut processed = 0;
        while processed < npoints {
            let batch_size = (npoints - processed).min(chunk_size);
            let slice = &mut scratch.indexed_buffer[..batch_size * ncols];

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

            let mut batch_inertia = F::zero();
            M::assign_and_accumulate::<C>(
                slice,
                ncols,
                prepared_centroids,
                k,
                &mut scratch.sums,
                &mut scratch.counts,
                &mut batch_inertia,
            );
            inertia += batch_inertia.to_f64().ok_or(Error::ConversionFailure)?;

            processed += batch_size;
        }

        Ok(inertia)
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
        chunk_sums: &mut [F],
    ) -> Result<()> {
        let npoints = source.num_points();
        let chunk_size = calculate_chunk_size::<F>(ncols);
        let num_chunks = npoints.div_ceil(chunk_size);
        debug_assert_eq!(chunk_sums.len(), num_chunks);
        let mut fallback_buffer = None;

        let mut offset = 0;
        let mut chunk_idx = 0;
        while offset < npoints {
            let count = (npoints - offset).min(chunk_size);
            let current_points = view_or_copy_batch(
                source,
                &mut fallback_buffer,
                offset,
                count,
                chunk_size,
                ncols,
            )?;

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
        Ok(())
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
        scratch: &mut IterationScratch<F>,
    ) -> Result<f64> {
        let npoints = source.num_points();
        if npoints == 0 {
            scratch.reset_totals();
            return Ok(0.0);
        }

        let par_chunk = par_chunk.min(npoints).max(1);
        let num_chunks = npoints.div_ceil(par_chunk);
        let worker_count = rayon::current_num_threads().min(num_chunks).max(1);
        scratch.prepare_parallel_workers(worker_count, 0);

        scratch.parallel_workers[..worker_count]
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(worker_idx, worker)| -> Result<()> {
                let first_chunk = worker_idx * num_chunks / worker_count;
                let end_chunk = (worker_idx + 1) * num_chunks / worker_count;
                for chunk_idx in first_chunk..end_chunk {
                    let start = chunk_idx * par_chunk;
                    let end = (start + par_chunk).min(npoints);
                    let len = end - start;

                    let points = view_or_copy_batch(
                        source,
                        &mut worker.fallback_buffer,
                        start,
                        len,
                        par_chunk,
                        ncols,
                    )?;

                    let mut chunk_inertia = F::zero();
                    M::assign_and_accumulate::<C>(
                        points,
                        ncols,
                        prepared_centroids,
                        k,
                        &mut worker.sums,
                        &mut worker.counts,
                        &mut chunk_inertia,
                    );
                    worker.inertia += chunk_inertia.to_f64().ok_or(Error::ConversionFailure)?;
                }
                Ok(())
            })?;

        Ok(scratch.merge_parallel_workers(worker_count))
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
        scratch: &mut IterationScratch<F>,
    ) -> Result<f64> {
        let npoints = indices.len();
        if npoints == 0 {
            scratch.reset_totals();
            return Ok(0.0);
        }

        let par_chunk = par_chunk.min(npoints).max(1);
        let num_chunks = npoints.div_ceil(par_chunk);
        let worker_count = rayon::current_num_threads().min(num_chunks).max(1);
        scratch.prepare_parallel_workers(worker_count, par_chunk * ncols);

        scratch.parallel_workers[..worker_count]
            .par_iter_mut()
            .enumerate()
            .try_for_each(|(worker_idx, worker)| -> Result<()> {
                let first_chunk = worker_idx * num_chunks / worker_count;
                let end_chunk = (worker_idx + 1) * num_chunks / worker_count;
                for chunk_idx in first_chunk..end_chunk {
                    let start = chunk_idx * par_chunk;
                    let end = (start + par_chunk).min(npoints);
                    let len = end - start;
                    let chunk_buffer = &mut worker.indexed_buffer[..len * ncols];

                    let chunk_indices = &indices[start..end];
                    let mut current_idx = 0;
                    while current_idx < len {
                        let start_idx = chunk_indices[current_idx];
                        let mut run_end = current_idx + 1;
                        while run_end < len
                            && chunk_indices[run_end] == chunk_indices[run_end - 1] + 1
                        {
                            run_end += 1;
                        }
                        let run_len = run_end - current_idx;
                        let slice =
                            &mut chunk_buffer[current_idx * ncols..(current_idx + run_len) * ncols];
                        source.read_batch(start_idx, run_len, slice);
                        current_idx = run_end;
                    }

                    let mut chunk_inertia = F::zero();
                    M::assign_and_accumulate::<C>(
                        chunk_buffer,
                        ncols,
                        prepared_centroids,
                        k,
                        &mut worker.sums,
                        &mut worker.counts,
                        &mut chunk_inertia,
                    );
                    worker.inertia += chunk_inertia.to_f64().ok_or(Error::ConversionFailure)?;
                }
                Ok(())
            })?;

        Ok(scratch.merge_parallel_workers(worker_count))
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
        chunk_sums: &mut [F],
    ) -> Result<()> {
        let npoints = source.num_points();
        if npoints == 0 {
            return Ok(());
        }
        let chunk_size = par_chunk.clamp(1, npoints);
        let num_chunks = npoints.div_ceil(chunk_size);
        debug_assert_eq!(chunk_sums.len(), num_chunks);

        chunk_sums
            .par_iter_mut()
            .zip(min_dists.par_chunks_mut(chunk_size).enumerate())
            .try_for_each_init(
                || None::<Vec<F>>,
                |fallback_buffer, (sum_slot, (chunk_idx, min_chunk))| -> Result<()> {
                    let start = chunk_idx * chunk_size;
                    let len = min_chunk.len();
                    let points =
                        view_or_copy_batch(source, fallback_buffer, start, len, chunk_size, ncols)?;

                    let chunk_sum = M::calculate_and_update_min_distance_sum::<C>(
                        points,
                        ncols,
                        newest_centroid,
                        min_chunk,
                    );
                    *sum_slot = chunk_sum;
                    Ok(())
                },
            )?;

        Ok(())
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
    let num_chunks = npoints.div_ceil(chunk_size);

    let mut centroids = Vec::with_capacity(k * ncols);
    let mut min_dists = vec![F::infinity(); npoints];
    let mut chunk_sums = vec![F::zero(); num_chunks];

    // 1. Choose first centroid uniformly at random
    let first_idx = rng.random_range(0..npoints);
    let mut point_buf = vec![F::zero(); ncols];
    source.read_batch(first_idx, 1, &mut point_buf);
    centroids.extend_from_slice(&point_buf);

    for _ in 1..k {
        let newest_centroid = &centroids[(centroids.len() / ncols - 1) * ncols..];

        E::update_min_dists::<F, C, M, S>(
            source,
            ncols,
            newest_centroid,
            &mut min_dists,
            chunk_size,
            &mut chunk_sums,
        )?;

        let sum_sq_dist = chunk_sums.iter().copied().fold(F::zero(), |a, b| a + b);

        // Sample next centroid
        if sum_sq_dist <= F::zero() {
            let next_idx = rng.random_range(0..npoints);
            source.read_batch(next_idx, 1, &mut point_buf);
            centroids.extend_from_slice(&point_buf);
        } else {
            let sample = if std::mem::size_of::<F>() == std::mem::size_of::<f64>() {
                F::from(rng.random::<f64>()).ok_or(Error::ConversionFailure)?
            } else {
                F::from(rng.random::<f32>()).ok_or(Error::ConversionFailure)?
            };
            let target = sample * sum_sq_dist;
            let selected_idx =
                pick_weighted_index(target, &chunk_sums, chunk_size, &min_dists, npoints);
            source.read_batch(selected_idx, 1, &mut point_buf);
            centroids.extend_from_slice(&point_buf);
        }
    }

    Ok(centroids)
}

#[cfg(test)]
#[path = "../tests/unit/iteration_storage.rs"]
mod tests;
