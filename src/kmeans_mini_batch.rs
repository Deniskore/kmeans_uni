use crate::backend::{CoreBackend, DistanceMetric};
use crate::error::{Error as KMeansError, Result};
use crate::{
    Primitive,
    kmeans_core::{ExecutionStrategy, InitializationStrategy, IterationScratch},
    kmeans_core_common::calculate_chunk_size,
    point_source::PointSource,
};
use rand::rngs::SmallRng;
use rand::seq::index::sample;
use rand::{RngExt, SeedableRng};

pub(crate) const DEFAULT_MINI_BATCH_REL_TOL: f64 = 1e-4;
pub(crate) const DEFAULT_MINI_BATCH_MIN_ITERATIONS: usize = 5;
pub(crate) const DEFAULT_MINI_BATCH_PATIENCE: usize = 3;

#[allow(clippy::too_many_arguments)]
pub(crate) fn run<
    F: Primitive,
    C: CoreBackend<F>,
    M: DistanceMetric<F>,
    S: PointSource<F>,
    I: InitializationStrategy,
    E: ExecutionStrategy,
>(
    source: &S,
    k: usize,
    iterations: usize,
    batch_size: usize,
    seed: Option<u64>,
    inertia_tol_rel: f64,
    min_iterations: usize,
    patience: usize,
) -> Result<(Vec<F>, F)> {
    let npoints = source.num_points();
    if npoints == 0 {
        return Err(KMeansError::InvalidInput(
            "point source must contain at least one point".into(),
        ));
    }
    let ncols = source.num_columns();
    let batch = batch_size.clamp(1, npoints);
    let par_chunk = calculate_chunk_size::<F>(ncols);

    let mut rng = match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_rng(&mut rand::rng()),
    };
    let init_size = batch.max(k * 10).min(npoints);
    let mut init_buffer = vec![F::zero(); init_size * ncols];
    let mut init_indices = sample(&mut rng, npoints, init_size).into_vec();
    init_indices.sort_unstable();

    let mut read_pos = 0;
    while read_pos < init_indices.len() {
        let start_idx = init_indices[read_pos];
        let mut run_end = read_pos + 1;
        while run_end < init_indices.len() && init_indices[run_end] == init_indices[run_end - 1] + 1
        {
            run_end += 1;
        }
        let run_len = run_end - read_pos;
        let slice = &mut init_buffer[read_pos * ncols..(read_pos + run_len) * ncols];
        source.read_batch(start_idx, run_len, slice);
        read_pos = run_end;
    }

    let init_source = crate::point_source::SlicePointSource::new(&init_buffer, ncols)?;
    let centroids = E::initialize::<F, C, M, _, I, _>(&init_source, k, &mut rng)?;
    let mut prepared_centroids = C::prepare_centroids(&centroids, ncols, k);

    let mut total_counts = vec![0usize; k];
    let mut total_sums = vec![F::zero(); k * ncols];
    let mut prev_batch_inertia = f64::INFINITY;
    let mut no_improvement_streak = 0;
    let mut iteration_scratch = IterationScratch::new(k, ncols);

    for iter in 0..iterations.max(1) {
        let mut indices = sample(&mut rng, npoints, batch).into_vec();
        indices.sort_unstable();

        let batch_inertia = E::compute_stats_indexed::<F, C, M, S>(
            source,
            ncols,
            k,
            &prepared_centroids,
            &indices,
            par_chunk,
            &mut iteration_scratch,
        )?;

        // Merge batch statistics into running totals
        for (c, (&count, batch_sums)) in iteration_scratch
            .counts()
            .iter()
            .zip(iteration_scratch.sums().chunks_exact(ncols))
            .enumerate()
        {
            if count > 0 {
                let base = c * ncols;
                total_counts[c] += count;
                for (total_sum, &batch_sum) in
                    total_sums[base..base + ncols].iter_mut().zip(batch_sums)
                {
                    *total_sum = *total_sum + batch_sum;
                }
            }
        }

        if patience > 0 && iter >= min_iterations {
            // Relative change in mini-batch inertia
            let rel_change =
                (prev_batch_inertia - batch_inertia).abs() / prev_batch_inertia.max(1e-12);

            if rel_change < inertia_tol_rel {
                no_improvement_streak += 1;
                if no_improvement_streak >= patience {
                    break; // Converged: inertia stable across batches
                }
            } else {
                no_improvement_streak = 0; // Reset on significant change
            }
        }
        // Update for next iteration's comparison
        prev_batch_inertia = batch_inertia;

        for c in 0..k {
            if total_counts[c] == 0 {
                let random_idx = rng.random_range(0..npoints);
                // Load point directly into sum vector (centroid will be this point)
                source.read_batch(random_idx, 1, &mut total_sums[c * ncols..(c + 1) * ncols]);
                total_counts[c] = 1; // Mark as updated
            }
        }

        C::update_centroids(
            &mut prepared_centroids,
            &total_sums,
            &total_counts,
            ncols,
            source,
            &mut rng,
            iteration_scratch.empty_cluster_buffer(),
        );
    }

    // Recompute inertia across the full dataset with the final centroids for an accurate score
    let full_inertia = E::compute_stats_full::<F, C, M, S>(
        source,
        ncols,
        k,
        &prepared_centroids,
        par_chunk,
        &mut iteration_scratch,
    )?;
    let centroids = C::finalize_centroids(&prepared_centroids, ncols, k);
    let inertia = F::from(full_inertia).ok_or(KMeansError::ConversionFailure)?;

    Ok((centroids, inertia))
}
