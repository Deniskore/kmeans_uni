use crate::backend::{CoreBackend, DistanceMetric};
use crate::error::Result;
use crate::{
    Primitive,
    kmeans_core::{ExecutionStrategy, InitializationStrategy},
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
    let _npoints = source.num_points();
    let ncols = source.num_columns();
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_rng(&mut rand::rng()),
    };

    let centroids = E::initialize::<F, C, M, _, I, _>(source, k, &mut rng)?;
    let mut prepared_centroids = C::prepare_centroids(&centroids, ncols, k);

    let par_chunk_size = calculate_chunk_size::<F>(ncols);
    let mut inertia = F::zero();

    for _i in 0..iterations {
        let (sums, counts, iter_inertia) = E::compute_stats_full::<F, C, M, S>(
            source,
            ncols,
            k,
            &prepared_centroids,
            par_chunk_size,
        )?;

        inertia = F::from(iter_inertia).unwrap_or(F::zero());

        let old_centroids = C::finalize_centroids(&prepared_centroids, ncols, k);

        C::update_centroids(
            &mut prepared_centroids,
            &sums,
            &counts,
            ncols,
            source,
            &mut rng,
        );

        let new_centroids = C::finalize_centroids(&prepared_centroids, ncols, k);

        // Check convergence
        let mut max_shift = F::zero();
        for (old, new) in old_centroids.iter().zip(new_centroids.iter()) {
            let diff = (*old - *new).abs();
            if diff > max_shift {
                max_shift = diff;
            }
        }

        if max_shift < tolerance {
            break;
        }
    }

    let centroids = C::finalize_centroids(&prepared_centroids, ncols, k);
    Ok((centroids, inertia))
}
