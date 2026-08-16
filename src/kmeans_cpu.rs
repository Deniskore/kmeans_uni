use crate::backend::{CoreBackend, DistanceMetric};
use crate::error::Result;
use crate::{
    Primitive,
    kmeans_core::{ExecutionStrategy, InitializationStrategy, IterationScratch},
    kmeans_core_common::calculate_chunk_size,
    point_source::PointSource,
};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[inline(always)]
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
    tolerance: F,
    seed: Option<u64>,
) -> Result<(Vec<F>, F)> {
    let ncols = source.num_columns();
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_rng(&mut rand::rng()),
    };

    let centroids = E::initialize::<F, C, M, _, I, _>(source, k, &mut rng)?;
    let mut prepared_centroids = C::prepare_centroids(&centroids, ncols, k);

    let par_chunk_size = calculate_chunk_size::<F>(ncols);
    let mut inertia = F::zero();
    let mut iteration_scratch = IterationScratch::new(k, ncols);

    for _i in 0..iterations {
        let iter_inertia = E::compute_stats_full::<F, C, M, S>(
            source,
            ncols,
            k,
            &prepared_centroids,
            par_chunk_size,
            &mut iteration_scratch,
        )?;

        inertia = F::from(iter_inertia).ok_or(crate::error::Error::ConversionFailure)?;
        let (sums, counts, empty_cluster_buffer) = iteration_scratch.centroid_update_parts();
        let max_shift = C::update_centroids_and_get_max_shift(
            &mut prepared_centroids,
            sums,
            counts,
            ncols,
            source,
            &mut rng,
            empty_cluster_buffer,
        );

        if max_shift < tolerance {
            break;
        }
    }

    let centroids = C::finalize_centroids(&prepared_centroids, ncols, k);
    Ok((centroids, inertia))
}
