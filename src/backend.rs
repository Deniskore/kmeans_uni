use crate::Primitive;
use crate::point_source::PointSource;
use rand::Rng;

pub trait CoreBackend<F: Primitive> {
    fn accumulate_point_slice(point: &[F], ncols: usize, sums: &mut [F], label: usize);
    fn prepare_centroids(centroids: &[F], ncols: usize, k: usize) -> Vec<F>;
    fn finalize_centroids(packed: &[F], ncols: usize, k: usize) -> Vec<F>;

    /// Finds the nearest centroid for a batch of points.
    ///
    /// # Arguments
    /// * `points` - A slice of points, flattened (length = num_points * ncols).
    /// * `ncols` - Number of columns (dimensions) per point.
    /// * `packed_centroids` - Centroids prepared by `prepare_centroids`.
    /// * `k` - Number of centroids.
    /// * `out_indices` - Output slice for nearest centroid indices (length = num_points).
    /// * `out_distances` - Optional output slice for squared distances/dot products (length = num_points).
    fn find_nearest_centroids_euc(
        points: &[F],
        ncols: usize,
        packed_centroids: &[F],
        k: usize,
        out_indices: &mut [usize],
        out_distances: Option<&mut [F]>,
    );

    fn find_nearest_centroids_euc_with_dists(
        points: &[F],
        ncols: usize,
        packed_centroids: &[F],
        k: usize,
        out_indices: &mut [usize],
        out_distances: &mut [F],
    );

    fn find_nearest_centroids_dot_product(
        points: &[F],
        ncols: usize,
        packed_centroids: &[F],
        k: usize,
        out_indices: &mut [usize],
        out_distances: Option<&mut [F]>,
    );

    fn find_nearest_centroids_dot_product_with_dists(
        points: &[F],
        ncols: usize,
        packed_centroids: &[F],
        k: usize,
        out_indices: &mut [usize],
        out_distances: &mut [F],
    );

    fn update_centroids<R: Rng, S: PointSource<F>>(
        packed_centroids: &mut [F],
        sums: &[F],
        counts: &[usize],
        ncols: usize,
        source: &S,
        rng: &mut R,
    );

    /// Calculates the Euclidean distance from each point to the given centroid and updates the minimum distance found so far.
    fn calculate_and_update_min_distance_euc(
        points: &[F],
        ncols: usize,
        centroid: &[F],
        min_dists: &mut [F],
    );

    /// Same as `calculate_and_update_min_distance_euc` but also returns the sum of updated min distances.
    /// Default implementation performs a second pass to sum; backends can override with fused accumulation.
    fn calculate_and_update_min_distance_euc_sum(
        points: &[F],
        ncols: usize,
        centroid: &[F],
        min_dists: &mut [F],
    ) -> F {
        Self::calculate_and_update_min_distance_euc(points, ncols, centroid, min_dists);
        min_dists.iter().copied().fold(F::zero(), |a, b| a + b)
    }

    /// Calculates the Dot Product distance from each point to the given centroid and updates the minimum distance found so far.
    fn calculate_and_update_min_distance_dot(
        points: &[F],
        ncols: usize,
        centroid: &[F],
        min_dists: &mut [F],
    );

    /// Dot-product variant that returns the sum of updated min distances.
    fn calculate_and_update_min_distance_dot_sum(
        points: &[F],
        ncols: usize,
        centroid: &[F],
        min_dists: &mut [F],
    ) -> F {
        Self::calculate_and_update_min_distance_dot(points, ncols, centroid, min_dists);
        min_dists.iter().copied().fold(F::zero(), |a, b| a + b)
    }
}

pub trait DistanceMetric<F: Primitive>: Copy + Send + Sync {
    fn find_nearest<C: CoreBackend<F>>(
        points: &[F],
        ncols: usize,
        packed_centroids: &[F],
        k: usize,
        out_indices: &mut [usize],
        out_distances: Option<&mut [F]>,
    );

    fn find_nearest_with_dists<C: CoreBackend<F>>(
        points: &[F],
        ncols: usize,
        packed_centroids: &[F],
        k: usize,
        out_indices: &mut [usize],
        out_distances: &mut [F],
    );

    fn calculate_and_update_min_distance<C: CoreBackend<F>>(
        points: &[F],
        ncols: usize,
        centroid: &[F],
        min_dists: &mut [F],
    );

    fn calculate_and_update_min_distance_sum<C: CoreBackend<F>>(
        points: &[F],
        ncols: usize,
        centroid: &[F],
        min_dists: &mut [F],
    ) -> F;
}

#[derive(Clone, Copy, Debug)]
pub struct Euclidean;

impl<F: Primitive> DistanceMetric<F> for Euclidean {
    #[inline(always)]
    fn find_nearest<C: CoreBackend<F>>(
        points: &[F],
        ncols: usize,
        packed_centroids: &[F],
        k: usize,
        out_indices: &mut [usize],
        out_distances: Option<&mut [F]>,
    ) {
        C::find_nearest_centroids_euc(
            points,
            ncols,
            packed_centroids,
            k,
            out_indices,
            out_distances,
        )
    }

    #[inline(always)]
    fn find_nearest_with_dists<C: CoreBackend<F>>(
        points: &[F],
        ncols: usize,
        packed_centroids: &[F],
        k: usize,
        out_indices: &mut [usize],
        out_distances: &mut [F],
    ) {
        C::find_nearest_centroids_euc_with_dists(
            points,
            ncols,
            packed_centroids,
            k,
            out_indices,
            out_distances,
        )
    }

    #[inline(always)]
    fn calculate_and_update_min_distance<C: CoreBackend<F>>(
        points: &[F],
        ncols: usize,
        centroid: &[F],
        min_dists: &mut [F],
    ) {
        C::calculate_and_update_min_distance_euc(points, ncols, centroid, min_dists)
    }

    #[inline(always)]
    fn calculate_and_update_min_distance_sum<C: CoreBackend<F>>(
        points: &[F],
        ncols: usize,
        centroid: &[F],
        min_dists: &mut [F],
    ) -> F {
        C::calculate_and_update_min_distance_euc_sum(points, ncols, centroid, min_dists)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DotProduct;

impl<F: Primitive> DistanceMetric<F> for DotProduct {
    #[inline(always)]
    fn find_nearest<C: CoreBackend<F>>(
        points: &[F],
        ncols: usize,
        packed_centroids: &[F],
        k: usize,
        out_indices: &mut [usize],
        out_distances: Option<&mut [F]>,
    ) {
        C::find_nearest_centroids_dot_product(
            points,
            ncols,
            packed_centroids,
            k,
            out_indices,
            out_distances,
        )
    }

    #[inline(always)]
    fn find_nearest_with_dists<C: CoreBackend<F>>(
        points: &[F],
        ncols: usize,
        packed_centroids: &[F],
        k: usize,
        out_indices: &mut [usize],
        out_distances: &mut [F],
    ) {
        C::find_nearest_centroids_dot_product_with_dists(
            points,
            ncols,
            packed_centroids,
            k,
            out_indices,
            out_distances,
        )
    }

    #[inline(always)]
    fn calculate_and_update_min_distance<C: CoreBackend<F>>(
        points: &[F],
        ncols: usize,
        centroid: &[F],
        min_dists: &mut [F],
    ) {
        C::calculate_and_update_min_distance_dot(points, ncols, centroid, min_dists)
    }

    #[inline(always)]
    fn calculate_and_update_min_distance_sum<C: CoreBackend<F>>(
        points: &[F],
        ncols: usize,
        centroid: &[F],
        min_dists: &mut [F],
    ) -> F {
        C::calculate_and_update_min_distance_dot_sum(points, ncols, centroid, min_dists)
    }
}
