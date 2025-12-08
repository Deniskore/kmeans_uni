use crate::backend::CoreBackend;
use crate::kmeans_core_common::{
    calculate_and_update_min_distance_generic, calculate_and_update_min_distance_generic_sum,
    find_nearest_centroids_generic,
};
use crate::point_source::PointSource;
use crate::primitive::Primitive;

pub struct SimdBackend;

macro_rules! impl_simd_backend {
    ($scalar:ty, $vec:ty, $idx:ty, $width:expr, $module:ident, $idx_scalar:ty) => {
        mod $module {
            use super::*;
            use bytemuck::cast;
            use rand::Rng;
            use wide::CmpLt;

            type SimdVec = $vec;
            type SimdIdx = $idx;
            const SIMD_WIDTH: usize = $width;

            impl CoreBackend<$scalar> for super::SimdBackend {
                #[inline(always)]
                fn accumulate_point_slice(
                    point: &[$scalar],
                    ncols: usize,
                    sums: &mut [$scalar],
                    label: usize,
                ) {
                    accumulate_point_slice_impl(point, ncols, sums, label);
                }

                #[inline(always)]
                fn prepare_centroids(
                    centroids: &[$scalar],
                    ncols: usize,
                    k: usize,
                ) -> Vec<$scalar> {
                    prepare_centroids_impl(centroids, ncols, k)
                }

                #[inline(always)]
                fn finalize_centroids(packed: &[$scalar], ncols: usize, k: usize) -> Vec<$scalar> {
                    finalize_centroids_impl(packed, ncols, k)
                }

                #[inline(always)]
                fn update_centroids<R: Rng, S: PointSource<$scalar>>(
                    packed_centroids: &mut [$scalar],
                    sums: &[$scalar],
                    counts: &[usize],
                    ncols: usize,
                    source: &S,
                    rng: &mut R,
                ) {
                    update_centroids_impl(packed_centroids, sums, counts, source, ncols, rng);
                }

                #[inline(always)]
                fn find_nearest_centroids_euc(
                    points: &[$scalar],
                    ncols: usize,
                    packed_centroids: &[$scalar],
                    k: usize,
                    out_indices: &mut [usize],
                    out_distances: Option<&mut [$scalar]>,
                ) {
                    let num_chunks = k.div_ceil(SIMD_WIDTH);

                    let mut current_indices_arr: [$idx_scalar; SIMD_WIDTH] = [0; SIMD_WIDTH];
                    for (i, slot) in current_indices_arr.iter_mut().enumerate() {
                        *slot = i as $idx_scalar;
                    }
                    let base_indices = SimdIdx::from(current_indices_arr);
                    let increment = SimdIdx::splat(SIMD_WIDTH as $idx_scalar);

                    find_nearest_centroids_generic(
                        points,
                        ncols,
                        k,
                        out_indices,
                        out_distances,
                        |points_batch, batch_size, indices_batch, mut dists_batch| {
                            const POINT_BATCH: usize = 4;
                            // Initialize best distances and indices for the current batch of points
                            let mut best_dists = [SimdVec::splat(<$scalar>::INFINITY); POINT_BATCH];
                            let mut best_indices = [SimdIdx::ZERO; POINT_BATCH];

                            let mut current_indices = base_indices;

                            for chunk_data in packed_centroids
                                .chunks_exact(ncols * SIMD_WIDTH)
                                .take(num_chunks)
                            {
                                // Accumulate sums for each point in the batch
                                let mut sums = [SimdVec::ZERO; POINT_BATCH];

                                for (d, centroid_chunk) in
                                    chunk_data.chunks_exact(SIMD_WIDTH).enumerate().take(ncols)
                                {
                                    let c_vec = load_chunk(centroid_chunk, 0);

                                    for i in 0..batch_size {
                                        let p_val = SimdVec::splat(points_batch[i * ncols + d]);
                                        let diff = p_val - c_vec;
                                        sums[i] += diff * diff;
                                    }
                                }

                                for i in 0..batch_size {
                                    let mask = sums[i].simd_lt(best_dists[i]);
                                    best_dists[i] = mask.blend(sums[i], best_dists[i]);

                                    let mask_bits: SimdIdx = cast(mask);
                                    best_indices[i] =
                                        mask_bits.blend(current_indices, best_indices[i]);
                                }

                                current_indices += increment;
                            }

                            // Reduce results for each point
                            for i in 0..batch_size {
                                let dists = best_dists[i].to_array();
                                let indices = best_indices[i].to_array();

                                let mut best_idx = 0;
                                let mut best_dist = <$scalar>::INFINITY;

                                for (&dist, &idx_val) in dists.iter().zip(indices.iter()) {
                                    let idx = idx_val as usize;
                                    if idx < k && dist < best_dist {
                                        best_dist = dist;
                                        best_idx = idx;
                                    }
                                }

                                indices_batch[i] = best_idx;
                                if let Some(out_dists) = dists_batch.as_deref_mut() {
                                    out_dists[i] = best_dist;
                                }
                            }
                        },
                    )
                }

                #[inline(always)]
                fn find_nearest_centroids_euc_with_dists(
                    points: &[$scalar],
                    ncols: usize,
                    packed_centroids: &[$scalar],
                    k: usize,
                    out_indices: &mut [usize],
                    out_distances: &mut [$scalar],
                ) {
                    Self::find_nearest_centroids_euc(
                        points,
                        ncols,
                        packed_centroids,
                        k,
                        out_indices,
                        Some(out_distances),
                    )
                }

                #[inline(always)]
                fn find_nearest_centroids_dot_product(
                    points: &[$scalar],
                    ncols: usize,
                    packed_centroids: &[$scalar],
                    k: usize,
                    out_indices: &mut [usize],
                    out_distances: Option<&mut [$scalar]>,
                ) {
                    let num_chunks = k.div_ceil(SIMD_WIDTH);

                    let mut current_indices_arr: [$idx_scalar; SIMD_WIDTH] = [0; SIMD_WIDTH];
                    for (i, slot) in current_indices_arr.iter_mut().enumerate() {
                        *slot = i as $idx_scalar;
                    }
                    let base_indices = SimdIdx::from(current_indices_arr);
                    let increment = SimdIdx::splat(SIMD_WIDTH as $idx_scalar);

                    find_nearest_centroids_generic(
                        points,
                        ncols,
                        k,
                        out_indices,
                        out_distances,
                        |points_batch, batch_size, indices_batch, mut dists_batch| {
                            const POINT_BATCH: usize = 4;
                            let mut best_dots =
                                [SimdVec::splat(<$scalar>::NEG_INFINITY); POINT_BATCH];
                            let mut best_indices = [SimdIdx::ZERO; POINT_BATCH];

                            let mut current_indices = base_indices;

                            for chunk_data in packed_centroids
                                .chunks_exact(ncols * SIMD_WIDTH)
                                .take(num_chunks)
                            {
                                let mut sums = [SimdVec::ZERO; POINT_BATCH];

                                for (d, centroid_chunk) in
                                    chunk_data.chunks_exact(SIMD_WIDTH).enumerate().take(ncols)
                                {
                                    let c_vec = load_chunk(centroid_chunk, 0);

                                    for i in 0..batch_size {
                                        let p_val = SimdVec::splat(points_batch[i * ncols + d]);
                                        sums[i] += p_val * c_vec;
                                    }
                                }

                                for i in 0..batch_size {
                                    let mask = best_dots[i].simd_lt(sums[i]);
                                    best_dots[i] = mask.blend(sums[i], best_dots[i]);

                                    let mask_bits: SimdIdx = cast(mask);
                                    best_indices[i] =
                                        mask_bits.blend(current_indices, best_indices[i]);
                                }

                                current_indices += increment;
                            }

                            for i in 0..batch_size {
                                let dots = best_dots[i].to_array();
                                let indices = best_indices[i].to_array();

                                let mut best_idx = 0;
                                let mut best_dot = <$scalar>::NEG_INFINITY;

                                for (&dot, &idx_val) in dots.iter().zip(indices.iter()) {
                                    let idx = idx_val as usize;
                                    if idx < k && dot > best_dot {
                                        best_dot = dot;
                                        best_idx = idx;
                                    }
                                }

                                indices_batch[i] = best_idx;
                                if let Some(out_dists) = dists_batch.as_deref_mut() {
                                    out_dists[i] = best_dot;
                                }
                            }
                        },
                    )
                }

                #[inline(always)]
                fn find_nearest_centroids_dot_product_with_dists(
                    points: &[$scalar],
                    ncols: usize,
                    packed_centroids: &[$scalar],
                    k: usize,
                    out_indices: &mut [usize],
                    out_distances: &mut [$scalar],
                ) {
                    Self::find_nearest_centroids_dot_product(
                        points,
                        ncols,
                        packed_centroids,
                        k,
                        out_indices,
                        Some(out_distances),
                    )
                }

                #[inline(always)]
                fn calculate_and_update_min_distance_euc(
                    points: &[$scalar],
                    ncols: usize,
                    centroid: &[$scalar],
                    min_dists: &mut [$scalar],
                ) {
                    calculate_and_update_min_distance_generic(
                        points,
                        ncols,
                        min_dists,
                        |point, min_dist| {
                            let mut sum_vec = SimdVec::ZERO;
                            let mut j = 0;

                            while j + SIMD_WIDTH <= ncols {
                                let p_vec = load_chunk(point, j);
                                let c_vec = load_chunk(centroid, j);
                                let diff = p_vec - c_vec;
                                sum_vec += diff * diff;
                                j += SIMD_WIDTH;
                            }

                            let mut d = sum_vec.reduce_add();

                            while j < ncols {
                                let diff = point[j] - centroid[j];
                                d += diff * diff;
                                j += 1;
                            }

                            if d < *min_dist {
                                *min_dist = d;
                            }
                        },
                    )
                }

                #[inline(always)]
                fn calculate_and_update_min_distance_euc_sum(
                    points: &[$scalar],
                    ncols: usize,
                    centroid: &[$scalar],
                    min_dists: &mut [$scalar],
                ) -> $scalar {
                    calculate_and_update_min_distance_generic_sum(
                        points,
                        ncols,
                        min_dists,
                        |point, min_dist| {
                            let mut sum_vec = SimdVec::ZERO;
                            let mut j = 0;

                            while j + SIMD_WIDTH <= ncols {
                                let p_vec = load_chunk(point, j);
                                let c_vec = load_chunk(centroid, j);
                                let diff = p_vec - c_vec;
                                sum_vec += diff * diff;
                                j += SIMD_WIDTH;
                            }

                            let mut d = sum_vec.reduce_add();

                            while j < ncols {
                                let diff = point[j] - centroid[j];
                                d += diff * diff;
                                j += 1;
                            }

                            if d < *min_dist {
                                *min_dist = d;
                            }
                        },
                    )
                }

                fn calculate_and_update_min_distance_dot(
                    points: &[$scalar],
                    ncols: usize,
                    centroid: &[$scalar],
                    min_dists: &mut [$scalar],
                ) {
                    // Fallback to Euclidean for initialization stability
                    Self::calculate_and_update_min_distance_euc(points, ncols, centroid, min_dists)
                }

                #[inline(always)]
                fn calculate_and_update_min_distance_dot_sum(
                    points: &[$scalar],
                    ncols: usize,
                    centroid: &[$scalar],
                    min_dists: &mut [$scalar],
                ) -> $scalar {
                    Self::calculate_and_update_min_distance_euc_sum(
                        points, ncols, centroid, min_dists,
                    )
                }
            }

            #[inline(always)]
            fn load_chunk(slice: &[$scalar], offset: usize) -> SimdVec {
                debug_assert!(offset + SIMD_WIDTH <= slice.len());
                let mut arr: [$scalar; SIMD_WIDTH] = [0.0; SIMD_WIDTH];
                arr.copy_from_slice(&slice[offset..offset + SIMD_WIDTH]);
                SimdVec::from(arr)
            }

            #[inline(always)]
            fn store_chunk(slice: &mut [$scalar], offset: usize, value: SimdVec) {
                debug_assert!(offset + SIMD_WIDTH <= slice.len());
                let chunk = &mut slice[offset..offset + SIMD_WIDTH];
                let arr = value.to_array();
                chunk.copy_from_slice(&arr);
            }

            #[inline(always)]
            fn accumulate_point_slice_impl(
                point: &[$scalar],
                ncols: usize,
                sums: &mut [$scalar],
                label: usize,
            ) {
                let cluster_sums = &mut sums[label * ncols..(label + 1) * ncols];
                let mut j = 0;

                while j + SIMD_WIDTH <= ncols {
                    let point_chunk = load_chunk(point, j);
                    let sum_chunk = load_chunk(cluster_sums, j);
                    let updated = sum_chunk + point_chunk;
                    store_chunk(cluster_sums, j, updated);
                    j += SIMD_WIDTH;
                }

                while j < ncols {
                    cluster_sums[j] += point[j];
                    j += 1;
                }
            }

            #[inline(always)]
            fn prepare_centroids_impl(
                centroids: &[$scalar],
                ncols: usize,
                k: usize,
            ) -> Vec<$scalar> {
                let num_chunks = k.div_ceil(SIMD_WIDTH);
                let mut packed = vec![<$scalar>::INFINITY; num_chunks * SIMD_WIDTH * ncols];

                for i in 0..k {
                    let chunk_idx = i / SIMD_WIDTH;
                    let lane_idx = i % SIMD_WIDTH;
                    let centroid = &centroids[i * ncols..(i + 1) * ncols];
                    let chunk_start = chunk_idx * (SIMD_WIDTH * ncols);
                    let chunk_end = chunk_start + SIMD_WIDTH * ncols;
                    let chunk = &mut packed[chunk_start..chunk_end];
                    for (d, &value) in centroid.iter().enumerate() {
                        chunk[d * SIMD_WIDTH + lane_idx] = value;
                    }
                }
                packed
            }

            #[inline(always)]
            fn finalize_centroids_impl(packed: &[$scalar], ncols: usize, k: usize) -> Vec<$scalar> {
                let mut standard = vec![0.0; k * ncols];

                for i in 0..k {
                    let chunk_idx = i / SIMD_WIDTH;
                    let lane_idx = i % SIMD_WIDTH;
                    let chunk_offset = chunk_idx * ncols * SIMD_WIDTH;
                    let dst = &mut standard[i * ncols..][..ncols];

                    // Copy each dimension directly
                    for (d, dst_val) in dst.iter_mut().enumerate() {
                        *dst_val = packed[chunk_offset + d * SIMD_WIDTH + lane_idx];
                    }
                }
                standard
            }

            #[inline(always)]
            fn update_centroids_impl<S: PointSource<$scalar>>(
                packed_centroids: &mut [$scalar],
                sums: &[$scalar],
                counts: &[usize],
                source: &S,
                ncols: usize,
                rng: &mut impl Rng,
            ) {
                let num_chunks = counts.len().div_ceil(SIMD_WIDTH);

                // Single buffer reused for all zero samples (avoids many allocations)
                let mut zero_samples: Vec<$scalar> = vec![0.0; SIMD_WIDTH * ncols];
                let mut zero_indices = [0; SIMD_WIDTH];

                for chunk_idx in 0..num_chunks {
                    let chunk_offset = chunk_idx * ncols * SIMD_WIDTH;

                    // Precompute inverse counts and sample zero points once per chunk
                    let mut inv_counts: [$scalar; SIMD_WIDTH] = [0.0; SIMD_WIDTH];
                    let mut active_mask = [false; SIMD_WIDTH];

                    for lane in 0..SIMD_WIDTH {
                        let k_idx = chunk_idx * SIMD_WIDTH + lane;
                        if k_idx >= counts.len() {
                            break;
                        }

                        if counts[k_idx] > 0 {
                            inv_counts[lane] = 1.0 / <$scalar>::from_usize(counts[k_idx]);
                            active_mask[lane] = true;
                        } else if source.num_points() > 0 {
                            zero_indices[lane] = rng.random_range(0..source.num_points());
                            active_mask[lane] = false;
                        }
                    }

                    // Load zero samples once per chunk (better locality)
                    for lane in 0..SIMD_WIDTH {
                        let k_idx = chunk_idx * SIMD_WIDTH + lane;
                        if k_idx >= counts.len() {
                            break;
                        }

                        if counts[k_idx] == 0 && source.num_points() > 0 {
                            source.read_batch(
                                zero_indices[lane],
                                1,
                                &mut zero_samples[lane * ncols..],
                            );
                        }
                    }

                    // Process dimensions sequentially for each cluster (cache-friendly)
                    let inv_vec = SimdVec::from(inv_counts);

                    for d in 0..ncols {
                        let mut sum_lane: [$scalar; SIMD_WIDTH] = [0.0; SIMD_WIDTH];

                        // Gather sums for this dimension across all lanes
                        for lane in 0..SIMD_WIDTH {
                            let k_idx = chunk_idx * SIMD_WIDTH + lane;
                            if k_idx >= counts.len() {
                                break;
                            }

                            if active_mask[lane] {
                                sum_lane[lane] = sums[k_idx * ncols + d];
                            }
                        }

                        let updated = inv_vec * SimdVec::from(sum_lane);
                        let arr = updated.to_array();

                        // Scatter results back
                        for lane in 0..SIMD_WIDTH {
                            let k_idx = chunk_idx * SIMD_WIDTH + lane;
                            if k_idx >= counts.len() {
                                break;
                            }

                            let dest_idx = chunk_offset + d * SIMD_WIDTH + lane;
                            if active_mask[lane] {
                                packed_centroids[dest_idx] = arr[lane];
                            } else if counts[k_idx] == 0 && source.num_points() > 0 {
                                packed_centroids[dest_idx] = zero_samples[lane * ncols + d];
                            }
                        }
                    }
                }
            }
        }
    };
}

cfg_if::cfg_if! {
    if #[cfg(any(target_arch = "aarch64", target_arch = "wasm32"))] {
        impl_simd_backend!(f32, wide::f32x4, wide::u32x4, 4, simd_f32, u32);
        impl_simd_backend!(f64, wide::f64x2, wide::u64x2, 2, simd_f64, u64);
    } else {
        impl_simd_backend!(f32, wide::f32x8, wide::u32x8, 8, simd_f32, u32);
        impl_simd_backend!(f64, wide::f64x4, wide::u64x4, 4, simd_f64, u64);
    }
}
