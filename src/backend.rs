use crate::Primitive;
use crate::kmeans_core_common::calculate_chunk_size;
use crate::point_source::PointSource;
use rand::Rng;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreparedCentroidLayout {
    Identity,
    #[cfg(feature = "wide")]
    Simd128,
    #[cfg(feature = "wide")]
    Simd256,
    #[cfg(feature = "wide")]
    Simd512,
}

pub trait CoreBackend<F: Primitive> {
    fn prepared_centroid_layout(k: usize) -> PreparedCentroidLayout;
    fn accumulate_point_slice(point: &[F], ncols: usize, sums: &mut [F], label: usize);
    fn prepare_centroids(centroids: &[F], ncols: usize, k: usize) -> Vec<F>;
    fn finalize_centroids(packed: &[F], ncols: usize, k: usize) -> Vec<F>;

    fn transform_points_per_chunk(ncols: usize, _k: usize) -> usize {
        calculate_chunk_size::<F>(ncols)
    }

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

    /// Assign points to nearest centroids (Euclidean), accumulating cluster sums/counts and total score.
    fn assign_and_accumulate_euc(
        points: &[F],
        ncols: usize,
        packed_centroids: &[F],
        k: usize,
        sums: &mut [F],
        counts: &mut [usize],
        total: &mut F,
    ) {
        let npoints = points.len() / ncols;
        if npoints == 0 {
            return;
        }

        let mut labels = vec![0usize; npoints];
        let mut distances = vec![F::zero(); npoints];
        Self::find_nearest_centroids_euc(
            points,
            ncols,
            packed_centroids,
            k,
            &mut labels,
            Some(&mut distances),
        );

        for i in 0..npoints {
            let label = labels[i];
            counts[label] += 1;
            let point = &points[i * ncols..(i + 1) * ncols];
            Self::accumulate_point_slice(point, ncols, sums, label);
            *total = *total + distances[i];
        }
    }

    /// Assign points to nearest centroids (dot-product), accumulating cluster sums/counts and total score.
    fn assign_and_accumulate_dot(
        points: &[F],
        ncols: usize,
        packed_centroids: &[F],
        k: usize,
        sums: &mut [F],
        counts: &mut [usize],
        total: &mut F,
    ) {
        let npoints = points.len() / ncols;
        if npoints == 0 {
            return;
        }

        let mut labels = vec![0usize; npoints];
        let mut distances = vec![F::zero(); npoints];
        Self::find_nearest_centroids_dot_product(
            points,
            ncols,
            packed_centroids,
            k,
            &mut labels,
            Some(&mut distances),
        );

        for i in 0..npoints {
            let label = labels[i];
            counts[label] += 1;
            let point = &points[i * ncols..(i + 1) * ncols];
            Self::accumulate_point_slice(point, ncols, sums, label);
            *total = *total + distances[i];
        }
    }

    /// Computes squared Euclidean distances from each point to each centroid.
    ///
    /// Writes a flat matrix of shape `(npoints * k)` into `out_scores`,
    /// where `out_scores[i * k + c]` is the score for point `i` and centroid `c`.
    fn transform_euc(
        points: &[F],
        ncols: usize,
        packed_centroids: &[F],
        k: usize,
        out_scores: &mut [F],
    ) {
        let npoints = points.len() / ncols;
        if npoints == 0 {
            return;
        }

        let centroids = Self::finalize_centroids(packed_centroids, ncols, k);
        for (point_idx, point) in points.chunks_exact(ncols).enumerate() {
            let out_row = &mut out_scores[point_idx * k..(point_idx + 1) * k];
            for (c_idx, centroid) in centroids.chunks_exact(ncols).enumerate() {
                let mut dist = F::zero();
                for (p, c) in point.iter().zip(centroid) {
                    let dv = *p - *c;
                    dist = dist + dv * dv;
                }
                out_row[c_idx] = dist;
            }
        }
    }

    /// Computes dot-product similarities from each point to each centroid.
    ///
    /// Writes a flat matrix of shape `(npoints * k)` into `out_scores`,
    /// where `out_scores[i * k + c]` is the score for point `i` and centroid `c`.
    fn transform_dot(
        points: &[F],
        ncols: usize,
        packed_centroids: &[F],
        k: usize,
        out_scores: &mut [F],
    ) {
        let npoints = points.len() / ncols;
        if npoints == 0 {
            return;
        }

        let centroids = Self::finalize_centroids(packed_centroids, ncols, k);
        for (point_idx, point) in points.chunks_exact(ncols).enumerate() {
            let out_row = &mut out_scores[point_idx * k..(point_idx + 1) * k];
            for (c_idx, centroid) in centroids.chunks_exact(ncols).enumerate() {
                let mut dot = F::zero();
                for (p, c) in point.iter().zip(centroid) {
                    dot = dot + *p * *c;
                }
                out_row[c_idx] = dot;
            }
        }
    }

    fn update_centroids<R: Rng, S: PointSource<F>>(
        packed_centroids: &mut [F],
        sums: &[F],
        counts: &[usize],
        ncols: usize,
        source: &S,
        rng: &mut R,
        empty_cluster_buffer: &mut Vec<F>,
    );

    /// Updates centroids and returns the maximum absolute centroid shift.
    ///
    /// Default implementation computes shift by finalizing before/after updates.
    /// Backends can override to avoid extra conversions/layout transforms.
    fn update_centroids_and_get_max_shift<R: Rng, S: PointSource<F>>(
        packed_centroids: &mut [F],
        sums: &[F],
        counts: &[usize],
        ncols: usize,
        source: &S,
        rng: &mut R,
        empty_cluster_buffer: &mut Vec<F>,
    ) -> F {
        let k = counts.len();
        let old_centroids = Self::finalize_centroids(packed_centroids, ncols, k);
        Self::update_centroids(
            packed_centroids,
            sums,
            counts,
            ncols,
            source,
            rng,
            empty_cluster_buffer,
        );
        let new_centroids = Self::finalize_centroids(packed_centroids, ncols, k);

        let mut max_shift = F::zero();
        for (old, new) in old_centroids.iter().zip(new_centroids.iter()) {
            let diff = (*old - *new).abs();
            if diff > max_shift {
                max_shift = diff;
            }
        }
        max_shift
    }

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

    fn assign_and_accumulate<C: CoreBackend<F>>(
        points: &[F],
        ncols: usize,
        packed_centroids: &[F],
        k: usize,
        sums: &mut [F],
        counts: &mut [usize],
        total: &mut F,
    );
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

    #[inline(always)]
    fn assign_and_accumulate<C: CoreBackend<F>>(
        points: &[F],
        ncols: usize,
        packed_centroids: &[F],
        k: usize,
        sums: &mut [F],
        counts: &mut [usize],
        total: &mut F,
    ) {
        C::assign_and_accumulate_euc(points, ncols, packed_centroids, k, sums, counts, total)
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

    #[inline(always)]
    fn assign_and_accumulate<C: CoreBackend<F>>(
        points: &[F],
        ncols: usize,
        packed_centroids: &[F],
        k: usize,
        sums: &mut [F],
        counts: &mut [usize],
        total: &mut F,
    ) {
        C::assign_and_accumulate_dot(points, ncols, packed_centroids, k, sums, counts, total)
    }
}
