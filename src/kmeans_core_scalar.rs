use rand::{Rng, RngExt};

use crate::Primitive;
use crate::backend::{CoreBackend, PreparedCentroidLayout};
use crate::kmeans_core_common::{
    calculate_and_update_min_distance_generic, calculate_and_update_min_distance_generic_sum,
    calculate_transform_chunk_size, find_nearest_centroids_generic,
};
use crate::point_source::PointSource;

pub struct ScalarBackend;

impl<F: Primitive> CoreBackend<F> for ScalarBackend {
    #[inline(always)]
    fn prepared_centroid_layout(_k: usize) -> PreparedCentroidLayout {
        PreparedCentroidLayout::Identity
    }

    fn accumulate_point_slice(point: &[F], ncols: usize, sums: &mut [F], label: usize) {
        let cluster_sums = &mut sums[label * ncols..(label + 1) * ncols];
        for (sum, val) in cluster_sums.iter_mut().zip(point) {
            *sum = *sum + *val;
        }
    }

    fn prepare_centroids(centroids: &[F], _ncols: usize, _k: usize) -> Vec<F> {
        centroids.to_vec()
    }

    fn finalize_centroids(packed: &[F], _ncols: usize, _k: usize) -> Vec<F> {
        packed.to_vec()
    }

    #[inline(always)]
    fn transform_points_per_chunk(ncols: usize, k: usize) -> usize {
        calculate_transform_chunk_size::<F>(ncols, k)
    }

    fn update_centroids<R: Rng, S: PointSource<F>>(
        centroids: &mut [F],
        sums: &[F],
        counts: &[usize],
        ncols: usize,
        source: &S,
        rng: &mut R,
        _empty_cluster_buffer: &mut Vec<F>,
    ) {
        for c in 0..counts.len() {
            let base = c * ncols;
            if counts[c] > 0 {
                let count_f = F::from_usize(counts[c]);
                let inv_count = F::one() / count_f;
                for j in 0..ncols {
                    centroids[base + j] = sums[base + j] * inv_count;
                }
            } else if source.num_points() > 0 {
                let idx = rng.random_range(0..source.num_points());
                let centroid = &mut centroids[c * ncols..(c + 1) * ncols];
                source.read_batch(idx, 1, centroid);
            }
        }
    }

    #[inline(always)]
    fn update_centroids_and_get_max_shift<R: Rng, S: PointSource<F>>(
        centroids: &mut [F],
        sums: &[F],
        counts: &[usize],
        ncols: usize,
        source: &S,
        rng: &mut R,
        empty_cluster_buffer: &mut Vec<F>,
    ) -> F {
        let mut max_shift = F::zero();
        let npoints = source.num_points();
        if npoints > 0 && counts.contains(&0) && empty_cluster_buffer.len() < ncols {
            empty_cluster_buffer.resize(ncols, F::zero());
        }

        for (c, &count) in counts.iter().enumerate() {
            let base = c * ncols;
            if count > 0 {
                let count_f = F::from_usize(count);
                let inv_count = F::one() / count_f;
                for j in 0..ncols {
                    let idx = base + j;
                    let new_value = sums[idx] * inv_count;
                    let diff = (centroids[idx] - new_value).abs();
                    if diff > max_shift {
                        max_shift = diff;
                    }
                    centroids[idx] = new_value;
                }
            } else if npoints > 0 {
                let rand_idx = rng.random_range(0..npoints);
                let sample = &mut empty_cluster_buffer[..ncols];
                source.read_batch(rand_idx, 1, sample);
                for (j, &new_value) in sample.iter().enumerate() {
                    let idx = base + j;
                    let diff = (centroids[idx] - new_value).abs();
                    if diff > max_shift {
                        max_shift = diff;
                    }
                    centroids[idx] = new_value;
                }
            }
        }

        max_shift
    }

    #[inline(always)]
    fn assign_and_accumulate_euc(
        points: &[F],
        ncols: usize,
        centroids: &[F],
        k: usize,
        sums: &mut [F],
        counts: &mut [usize],
        total: &mut F,
    ) {
        const POINT_BATCH: usize = 4;
        for points_batch in points.chunks(ncols * POINT_BATCH) {
            let batch_size = points_batch.len() / ncols;
            let mut best_indices = [0usize; POINT_BATCH];
            let mut best_dists = [F::infinity(); POINT_BATCH];

            for (c_idx, centroid) in centroids.chunks_exact(ncols).take(k).enumerate() {
                for (i, point) in points_batch.chunks_exact(ncols).enumerate() {
                    let mut d = F::zero();
                    for (p, c) in point.iter().zip(centroid) {
                        let dv = *p - *c;
                        d = d + dv * dv;
                    }

                    if d < best_dists[i] {
                        best_dists[i] = d;
                        best_indices[i] = c_idx;
                    }
                }
            }

            for i in 0..batch_size {
                let best = best_indices[i];
                counts[best] += 1;
                let point = &points_batch[i * ncols..(i + 1) * ncols];
                Self::accumulate_point_slice(point, ncols, sums, best);
                *total = *total + best_dists[i];
            }
        }
    }

    #[inline(always)]
    fn assign_and_accumulate_dot(
        points: &[F],
        ncols: usize,
        centroids: &[F],
        k: usize,
        sums: &mut [F],
        counts: &mut [usize],
        total: &mut F,
    ) {
        const POINT_BATCH: usize = 4;
        for points_batch in points.chunks(ncols * POINT_BATCH) {
            let batch_size = points_batch.len() / ncols;
            let mut best_indices = [0usize; POINT_BATCH];
            let mut best_dots = [F::neg_infinity(); POINT_BATCH];

            for (c_idx, centroid) in centroids.chunks_exact(ncols).take(k).enumerate() {
                for (i, point) in points_batch.chunks_exact(ncols).enumerate() {
                    let mut dot = F::zero();
                    for (p, c) in point.iter().zip(centroid) {
                        dot = dot + *p * *c;
                    }

                    if dot > best_dots[i] {
                        best_dots[i] = dot;
                        best_indices[i] = c_idx;
                    }
                }
            }

            for i in 0..batch_size {
                let best = best_indices[i];
                counts[best] += 1;
                let point = &points_batch[i * ncols..(i + 1) * ncols];
                Self::accumulate_point_slice(point, ncols, sums, best);
                *total = *total + best_dots[i];
            }
        }
    }

    #[inline(always)]
    fn transform_euc(points: &[F], ncols: usize, centroids: &[F], k: usize, out_scores: &mut [F]) {
        const POINT_BATCH: usize = 4;
        let mut point_base = 0usize;

        for points_batch in points.chunks(ncols * POINT_BATCH) {
            let batch_size = points_batch.len() / ncols;
            let out_batch = &mut out_scores[point_base * k..(point_base + batch_size) * k];

            for i in 0..batch_size {
                let point = &points_batch[i * ncols..(i + 1) * ncols];
                let out_row = &mut out_batch[i * k..(i + 1) * k];
                for (c_idx, centroid) in centroids.chunks_exact(ncols).take(k).enumerate() {
                    let mut dist = F::zero();
                    for (p, c) in point.iter().zip(centroid) {
                        let dv = *p - *c;
                        dist = dist + dv * dv;
                    }
                    out_row[c_idx] = dist;
                }
            }

            point_base += batch_size;
        }
    }

    #[inline(always)]
    fn transform_dot(points: &[F], ncols: usize, centroids: &[F], k: usize, out_scores: &mut [F]) {
        const POINT_BATCH: usize = 4;
        let mut point_base = 0usize;

        for points_batch in points.chunks(ncols * POINT_BATCH) {
            let batch_size = points_batch.len() / ncols;
            let out_batch = &mut out_scores[point_base * k..(point_base + batch_size) * k];

            for (c_idx, centroid) in centroids.chunks_exact(ncols).take(k).enumerate() {
                for i in 0..batch_size {
                    let point = &points_batch[i * ncols..(i + 1) * ncols];
                    let mut dot = F::zero();
                    for (p, c) in point.iter().zip(centroid) {
                        dot = dot + *p * *c;
                    }
                    out_batch[i * k + c_idx] = dot;
                }
            }

            point_base += batch_size;
        }
    }

    #[inline(always)]
    fn find_nearest_centroids_euc(
        points: &[F],
        ncols: usize,
        centroids: &[F],
        k: usize,
        out_indices: &mut [usize],
        out_distances: Option<&mut [F]>,
    ) {
        find_nearest_centroids_generic(
            points,
            ncols,
            k,
            out_indices,
            out_distances,
            |points_batch, batch_size, indices_batch, dists_batch| {
                const POINT_BATCH: usize = 4;
                let mut best_indices = [0usize; POINT_BATCH];
                let mut best_dists = [F::infinity(); POINT_BATCH];

                for (c_idx, centroid) in centroids.chunks_exact(ncols).enumerate() {
                    for (i, point) in points_batch.chunks_exact(ncols).enumerate() {
                        let mut d = F::zero();
                        for (p, c) in point.iter().zip(centroid) {
                            let dv = *p - *c;
                            d = d + dv * dv;
                        }

                        if d < best_dists[i] {
                            best_dists[i] = d;
                            best_indices[i] = c_idx;
                        }
                    }
                }

                indices_batch[..batch_size].copy_from_slice(&best_indices[..batch_size]);
                if let Some(dists) = dists_batch {
                    dists[..batch_size].copy_from_slice(&best_dists[..batch_size]);
                }
            },
        )
    }

    #[inline(always)]
    fn find_nearest_centroids_dot_product(
        points: &[F],
        ncols: usize,
        centroids: &[F],
        k: usize,
        out_indices: &mut [usize],
        out_distances: Option<&mut [F]>,
    ) {
        find_nearest_centroids_generic(
            points,
            ncols,
            k,
            out_indices,
            out_distances,
            |points_batch, batch_size, indices_batch, dists_batch| {
                const POINT_BATCH: usize = 4;
                let mut best_indices = [0usize; POINT_BATCH];
                let mut best_dots = [F::neg_infinity(); POINT_BATCH];

                for (c_idx, centroid) in centroids.chunks_exact(ncols).enumerate() {
                    for (i, point) in points_batch.chunks_exact(ncols).enumerate() {
                        let mut dot = F::zero();
                        for (p, c) in point.iter().zip(centroid) {
                            dot = dot + *p * *c;
                        }

                        if dot > best_dots[i] {
                            best_dots[i] = dot;
                            best_indices[i] = c_idx;
                        }
                    }
                }

                indices_batch[..batch_size].copy_from_slice(&best_indices[..batch_size]);
                if let Some(dists) = dists_batch {
                    dists[..batch_size].copy_from_slice(&best_dots[..batch_size]);
                }
            },
        )
    }

    #[inline(always)]
    fn find_nearest_centroids_euc_with_dists(
        points: &[F],
        ncols: usize,
        centroids: &[F],
        k: usize,
        out_indices: &mut [usize],
        out_distances: &mut [F],
    ) {
        Self::find_nearest_centroids_euc(
            points,
            ncols,
            centroids,
            k,
            out_indices,
            Some(out_distances),
        )
    }

    #[inline(always)]
    fn find_nearest_centroids_dot_product_with_dists(
        points: &[F],
        ncols: usize,
        centroids: &[F],
        k: usize,
        out_indices: &mut [usize],
        out_distances: &mut [F],
    ) {
        Self::find_nearest_centroids_dot_product(
            points,
            ncols,
            centroids,
            k,
            out_indices,
            Some(out_distances),
        )
    }

    #[inline(always)]
    fn calculate_and_update_min_distance_euc(
        points: &[F],
        ncols: usize,
        centroid: &[F],
        min_dists: &mut [F],
    ) {
        calculate_and_update_min_distance_generic(points, ncols, min_dists, |point, min_dist| {
            let mut d = F::zero();
            for (p, c) in point.iter().zip(centroid) {
                let dv = *p - *c;
                d = d + dv * dv;
            }
            if d < *min_dist {
                *min_dist = d;
            }
        })
    }

    #[inline(always)]
    fn calculate_and_update_min_distance_euc_sum(
        points: &[F],
        ncols: usize,
        centroid: &[F],
        min_dists: &mut [F],
    ) -> F {
        calculate_and_update_min_distance_generic_sum(
            points,
            ncols,
            min_dists,
            |point, min_dist| {
                let mut d = F::zero();
                for (p, c) in point.iter().zip(centroid) {
                    let dv = *p - *c;
                    d = d + dv * dv;
                }
                if d < *min_dist {
                    *min_dist = d;
                }
            },
        )
    }

    #[inline(always)]
    fn calculate_and_update_min_distance_dot(
        points: &[F],
        ncols: usize,
        centroid: &[F],
        min_dists: &mut [F],
    ) {
        // Fallback to Euclidean for initialization stability
        Self::calculate_and_update_min_distance_euc(points, ncols, centroid, min_dists)
    }

    #[inline(always)]
    fn calculate_and_update_min_distance_dot_sum(
        points: &[F],
        ncols: usize,
        centroid: &[F],
        min_dists: &mut [F],
    ) -> F {
        Self::calculate_and_update_min_distance_euc_sum(points, ncols, centroid, min_dists)
    }
}
