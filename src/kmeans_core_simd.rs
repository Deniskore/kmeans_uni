use crate::backend::{CoreBackend, PreparedCentroidLayout};
use crate::kmeans_core_common::{
    calculate_and_update_min_distance_generic, calculate_transform_chunk_size,
    find_nearest_centroids_generic,
};
use crate::point_source::PointSource;
use crate::primitive::Primitive;
use rand::Rng;

pub struct SimdBackend128;
pub struct SimdBackend256;
pub struct SimdBackend512;
pub struct SimdBackendAdaptive;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AdaptiveWidth {
    W128,
    W256,
    W512,
}

#[inline(always)]
fn prepared_layout(width: AdaptiveWidth) -> PreparedCentroidLayout {
    match width {
        AdaptiveWidth::W128 => PreparedCentroidLayout::Simd128,
        AdaptiveWidth::W256 => PreparedCentroidLayout::Simd256,
        AdaptiveWidth::W512 => PreparedCentroidLayout::Simd512,
    }
}

#[inline(always)]
fn default_width() -> AdaptiveWidth {
    if cfg!(any(target_arch = "aarch64", target_arch = "wasm32")) {
        AdaptiveWidth::W128
    } else {
        AdaptiveWidth::W256
    }
}

#[inline(always)]
fn adaptive_width_f32(k: usize) -> AdaptiveWidth {
    if cfg!(any(target_arch = "aarch64", target_arch = "wasm32")) {
        return AdaptiveWidth::W128;
    }

    if cfg!(target_feature = "avx") {
        if cfg!(target_feature = "avx512f") && k >= 32 && k.is_multiple_of(16) {
            return AdaptiveWidth::W512;
        }
        return AdaptiveWidth::W256;
    }

    if k >= 32 && k.is_multiple_of(16) {
        return AdaptiveWidth::W512;
    }

    let padded_128 = k.div_ceil(4) * 4;
    let padded_256 = k.div_ceil(8) * 8;
    if padded_128 < padded_256 {
        AdaptiveWidth::W128
    } else {
        AdaptiveWidth::W256
    }
}

#[inline(always)]
fn adaptive_width_f64(_k: usize) -> AdaptiveWidth {
    if cfg!(any(target_arch = "aarch64", target_arch = "wasm32")) {
        return AdaptiveWidth::W128;
    }

    AdaptiveWidth::W256
}

macro_rules! dispatch_width {
    ($width:expr, $scalar:ty, $method:ident($($arg:expr),* $(,)?)) => {
        match $width {
            AdaptiveWidth::W128 =>
                <SimdBackend128 as CoreBackend<$scalar>>::$method($($arg),*),
            AdaptiveWidth::W256 =>
                <SimdBackend256 as CoreBackend<$scalar>>::$method($($arg),*),
            AdaptiveWidth::W512 =>
                <SimdBackend512 as CoreBackend<$scalar>>::$method($($arg),*),
        }
    };
}

macro_rules! adaptive_assign_helpers {
    ($module:ident, $scalar:ty) => {
        mod $module {
            use super::*;

            macro_rules! assign_helper {
                ($name:ident, $backend:ty, $method:ident) => {
                    #[inline(never)]
                    pub(super) fn $name(
                        points: &[$scalar],
                        ncols: usize,
                        packed_centroids: &[$scalar],
                        k: usize,
                        sums: &mut [$scalar],
                        counts: &mut [usize],
                        total: &mut $scalar,
                    ) {
                        <$backend as CoreBackend<$scalar>>::$method(
                            points,
                            ncols,
                            packed_centroids,
                            k,
                            sums,
                            counts,
                            total,
                        )
                    }
                };
            }

            assign_helper!(euc128, SimdBackend128, assign_and_accumulate_euc);
            assign_helper!(euc256, SimdBackend256, assign_and_accumulate_euc);
            assign_helper!(euc512, SimdBackend512, assign_and_accumulate_euc);
            assign_helper!(dot128, SimdBackend128, assign_and_accumulate_dot);
            assign_helper!(dot256, SimdBackend256, assign_and_accumulate_dot);
            assign_helper!(dot512, SimdBackend512, assign_and_accumulate_dot);
        }
    };
}

adaptive_assign_helpers!(adaptive_assign_f32, f32);
adaptive_assign_helpers!(adaptive_assign_f64, f64);

macro_rules! impl_simd_backend {
    ($backend:ident, $scalar:ty, $vec:ty, $idx:ty, $width:expr, $module:ident, $idx_scalar:ty) => {
        mod $module {
            use super::*;
            use bytemuck::cast;
            use rand::{Rng, RngExt};

            type SimdVec = $vec;
            type SimdIdx = $idx;
            const SIMD_WIDTH: usize = $width;

            impl CoreBackend<$scalar> for super::$backend {
                #[inline(always)]
                fn prepared_centroid_layout(_k: usize) -> PreparedCentroidLayout {
                    match SIMD_WIDTH * std::mem::size_of::<$scalar>() {
                        16 => PreparedCentroidLayout::Simd128,
                        32 => PreparedCentroidLayout::Simd256,
                        64 => PreparedCentroidLayout::Simd512,
                        _ => unreachable!("unsupported SIMD centroid layout"),
                    }
                }

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
                fn transform_points_per_chunk(ncols: usize, k: usize) -> usize {
                    calculate_transform_chunk_size::<$scalar>(ncols, k)
                }

                #[inline(always)]
                fn update_centroids<R: Rng, S: PointSource<$scalar>>(
                    packed_centroids: &mut [$scalar],
                    sums: &[$scalar],
                    counts: &[usize],
                    ncols: usize,
                    source: &S,
                    rng: &mut R,
                    empty_cluster_buffer: &mut Vec<$scalar>,
                ) {
                    let _ = update_centroids_impl::<false, _>(
                        packed_centroids,
                        sums,
                        counts,
                        source,
                        ncols,
                        rng,
                        empty_cluster_buffer,
                    );
                }

                #[inline(always)]
                fn update_centroids_and_get_max_shift<R: Rng, S: PointSource<$scalar>>(
                    packed_centroids: &mut [$scalar],
                    sums: &[$scalar],
                    counts: &[usize],
                    ncols: usize,
                    source: &S,
                    rng: &mut R,
                    empty_cluster_buffer: &mut Vec<$scalar>,
                ) -> $scalar {
                    update_centroids_impl::<true, _>(
                        packed_centroids,
                        sums,
                        counts,
                        source,
                        ncols,
                        rng,
                        empty_cluster_buffer,
                    )
                }

                #[inline(always)]
                fn assign_and_accumulate_euc(
                    points: &[$scalar],
                    ncols: usize,
                    packed_centroids: &[$scalar],
                    k: usize,
                    sums: &mut [$scalar],
                    counts: &mut [usize],
                    total: &mut $scalar,
                ) {
                    const POINT_BATCH: usize = 64;
                    let num_chunks = k.div_ceil(SIMD_WIDTH);

                    let mut current_indices_arr: [$idx_scalar; SIMD_WIDTH] = [0; SIMD_WIDTH];
                    for (i, slot) in current_indices_arr.iter_mut().enumerate() {
                        *slot = i as $idx_scalar;
                    }
                    let base_indices = SimdIdx::from(current_indices_arr);
                    let increment = SimdIdx::splat(SIMD_WIDTH as $idx_scalar);

                    for points_batch in points.chunks(ncols * POINT_BATCH) {
                        let batch_size = points_batch.len() / ncols;
                        let mut best_dists = [SimdVec::splat(<$scalar>::INFINITY); POINT_BATCH];
                        let mut best_indices = [SimdIdx::ZERO; POINT_BATCH];

                        let mut current_indices = base_indices;
                        for chunk_data in packed_centroids
                            .chunks_exact(ncols * SIMD_WIDTH)
                            .take(num_chunks)
                        {
                            let mut lane_sums = [SimdVec::ZERO; POINT_BATCH];

                            for (d, centroid_chunk) in
                                chunk_data.chunks_exact(SIMD_WIDTH).enumerate().take(ncols)
                            {
                                let c_vec = load_chunk(centroid_chunk, 0);
                                for i in 0..batch_size {
                                    let p_val = SimdVec::splat(points_batch[i * ncols + d]);
                                    let diff = p_val - c_vec;
                                    lane_sums[i] += diff * diff;
                                }
                            }

                            for i in 0..batch_size {
                                let mask = lane_sums[i].simd_lt(best_dists[i]);
                                best_dists[i] = mask.select(lane_sums[i], best_dists[i]);
                                let mask_bits: SimdIdx = cast(mask);
                                best_indices[i] =
                                    mask_bits.select(current_indices, best_indices[i]);
                            }

                            current_indices += increment;
                        }

                        for i in 0..batch_size {
                            let dists = best_dists[i].to_array();
                            let indices = best_indices[i].to_array();
                            let mut best_dist = <$scalar>::INFINITY;

                            for dist in dists {
                                if dist < best_dist {
                                    best_dist = dist;
                                }
                            }
                            let mut tied_lanes = best_dists[i]
                                .simd_eq(SimdVec::splat(best_dist))
                                .to_bitmask();
                            let mut best_idx = 0usize;
                            if tied_lanes != 0 {
                                best_idx = usize::MAX;
                                while tied_lanes != 0 {
                                    let lane = tied_lanes.trailing_zeros() as usize;
                                    best_idx = best_idx.min(indices[lane] as usize);
                                    tied_lanes &= tied_lanes - 1;
                                }
                            }

                            counts[best_idx] += 1;
                            let point = &points_batch[i * ncols..(i + 1) * ncols];
                            accumulate_point_slice_impl(point, ncols, sums, best_idx);
                            *total += best_dist;
                        }
                    }
                }

                #[inline(always)]
                fn assign_and_accumulate_dot(
                    points: &[$scalar],
                    ncols: usize,
                    packed_centroids: &[$scalar],
                    k: usize,
                    sums: &mut [$scalar],
                    counts: &mut [usize],
                    total: &mut $scalar,
                ) {
                    const POINT_BATCH: usize = 64;
                    let num_chunks = k.div_ceil(SIMD_WIDTH);
                    let all_lanes_valid = k == num_chunks * SIMD_WIDTH;

                    let mut current_indices_arr: [$idx_scalar; SIMD_WIDTH] = [0; SIMD_WIDTH];
                    for (i, slot) in current_indices_arr.iter_mut().enumerate() {
                        *slot = i as $idx_scalar;
                    }
                    let base_indices = SimdIdx::from(current_indices_arr);
                    let increment = SimdIdx::splat(SIMD_WIDTH as $idx_scalar);

                    for points_batch in points.chunks(ncols * POINT_BATCH) {
                        let batch_size = points_batch.len() / ncols;
                        let mut best_dots = [SimdVec::splat(<$scalar>::NEG_INFINITY); POINT_BATCH];
                        let mut best_indices = [SimdIdx::ZERO; POINT_BATCH];

                        let mut current_indices = base_indices;
                        for (chunk_idx, chunk_data) in packed_centroids
                            .chunks_exact(ncols * SIMD_WIDTH)
                            .take(num_chunks)
                            .enumerate()
                        {
                            let mut lane_sums = [SimdVec::ZERO; POINT_BATCH];

                            for (d, centroid_chunk) in
                                chunk_data.chunks_exact(SIMD_WIDTH).enumerate().take(ncols)
                            {
                                let c_vec = load_chunk(centroid_chunk, 0);
                                for i in 0..batch_size {
                                    let p_val = SimdVec::splat(points_batch[i * ncols + d]);
                                    lane_sums[i] += p_val * c_vec;
                                }
                            }

                            if !all_lanes_valid && chunk_idx + 1 == num_chunks {
                                let valid_lanes = k - chunk_idx * SIMD_WIDTH;
                                for sums in &mut lane_sums[..batch_size] {
                                    mask_invalid_dot_lanes(sums, valid_lanes);
                                }
                            }

                            for i in 0..batch_size {
                                let mask = best_dots[i].simd_lt(lane_sums[i]);
                                best_dots[i] = mask.select(lane_sums[i], best_dots[i]);
                                let mask_bits: SimdIdx = cast(mask);
                                best_indices[i] =
                                    mask_bits.select(current_indices, best_indices[i]);
                            }

                            current_indices += increment;
                        }

                        for i in 0..batch_size {
                            let dots = best_dots[i].to_array();
                            let indices = best_indices[i].to_array();
                            let mut best_dot = <$scalar>::NEG_INFINITY;

                            for dot in dots {
                                if dot > best_dot {
                                    best_dot = dot;
                                }
                            }
                            let mut tied_lanes =
                                best_dots[i].simd_eq(SimdVec::splat(best_dot)).to_bitmask();
                            let mut best_idx = 0usize;
                            if tied_lanes != 0 {
                                best_idx = usize::MAX;
                                while tied_lanes != 0 {
                                    let lane = tied_lanes.trailing_zeros() as usize;
                                    best_idx = best_idx.min(indices[lane] as usize);
                                    tied_lanes &= tied_lanes - 1;
                                }
                            }

                            counts[best_idx] += 1;
                            let point = &points_batch[i * ncols..(i + 1) * ncols];
                            accumulate_point_slice_impl(point, ncols, sums, best_idx);
                            *total += best_dot;
                        }
                    }
                }

                #[inline(always)]
                fn transform_euc(
                    points: &[$scalar],
                    ncols: usize,
                    packed_centroids: &[$scalar],
                    k: usize,
                    out_scores: &mut [$scalar],
                ) {
                    let num_chunks = k.div_ceil(SIMD_WIDTH);
                    let full_chunks = k / SIMD_WIDTH;

                    for (point_idx, point) in points.chunks_exact(ncols).enumerate() {
                        let out_row = &mut out_scores[point_idx * k..(point_idx + 1) * k];

                        for (chunk_idx, chunk_data) in packed_centroids
                            .chunks_exact(ncols * SIMD_WIDTH)
                            .take(full_chunks)
                            .enumerate()
                        {
                            let mut sums = SimdVec::ZERO;

                            for (d, centroid_chunk) in
                                chunk_data.chunks_exact(SIMD_WIDTH).enumerate().take(ncols)
                            {
                                let c_vec = load_chunk(centroid_chunk, 0);
                                let p_val = SimdVec::splat(point[d]);
                                let diff = p_val - c_vec;
                                sums += diff * diff;
                            }

                            let arr = sums.to_array();
                            let out_base = chunk_idx * SIMD_WIDTH;
                            out_row[out_base..out_base + SIMD_WIDTH].copy_from_slice(&arr);
                        }

                        if full_chunks < num_chunks {
                            let chunk_idx = full_chunks;
                            let chunk_data = &packed_centroids[chunk_idx * ncols * SIMD_WIDTH
                                ..(chunk_idx + 1) * ncols * SIMD_WIDTH];
                            let mut sums = SimdVec::ZERO;

                            for (d, centroid_chunk) in
                                chunk_data.chunks_exact(SIMD_WIDTH).enumerate().take(ncols)
                            {
                                let c_vec = load_chunk(centroid_chunk, 0);
                                let p_val = SimdVec::splat(point[d]);
                                let diff = p_val - c_vec;
                                sums += diff * diff;
                            }

                            let arr = sums.to_array();
                            let out_base = chunk_idx * SIMD_WIDTH;
                            for (lane, &value) in arr.iter().enumerate().take(k - out_base) {
                                let out_idx = out_base + lane;
                                out_row[out_idx] = value;
                            }
                        }
                    }
                }

                #[inline(always)]
                fn transform_dot(
                    points: &[$scalar],
                    ncols: usize,
                    packed_centroids: &[$scalar],
                    k: usize,
                    out_scores: &mut [$scalar],
                ) {
                    let num_chunks = k.div_ceil(SIMD_WIDTH);
                    let full_chunks = k / SIMD_WIDTH;

                    for (point_idx, point) in points.chunks_exact(ncols).enumerate() {
                        let out_row = &mut out_scores[point_idx * k..(point_idx + 1) * k];

                        for (chunk_idx, chunk_data) in packed_centroids
                            .chunks_exact(ncols * SIMD_WIDTH)
                            .take(full_chunks)
                            .enumerate()
                        {
                            let mut sums = SimdVec::ZERO;

                            for (d, centroid_chunk) in
                                chunk_data.chunks_exact(SIMD_WIDTH).enumerate().take(ncols)
                            {
                                let c_vec = load_chunk(centroid_chunk, 0);
                                let p_val = SimdVec::splat(point[d]);
                                sums += p_val * c_vec;
                            }

                            let arr = sums.to_array();
                            let out_base = chunk_idx * SIMD_WIDTH;
                            out_row[out_base..out_base + SIMD_WIDTH].copy_from_slice(&arr);
                        }

                        if full_chunks < num_chunks {
                            let chunk_idx = full_chunks;
                            let chunk_data = &packed_centroids[chunk_idx * ncols * SIMD_WIDTH
                                ..(chunk_idx + 1) * ncols * SIMD_WIDTH];
                            let mut sums = SimdVec::ZERO;

                            for (d, centroid_chunk) in
                                chunk_data.chunks_exact(SIMD_WIDTH).enumerate().take(ncols)
                            {
                                let c_vec = load_chunk(centroid_chunk, 0);
                                let p_val = SimdVec::splat(point[d]);
                                sums += p_val * c_vec;
                            }

                            let arr = sums.to_array();
                            let out_base = chunk_idx * SIMD_WIDTH;
                            for (lane, &value) in arr.iter().enumerate().take(k - out_base) {
                                let out_idx = out_base + lane;
                                out_row[out_idx] = value;
                            }
                        }
                    }
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
                                    best_dists[i] = mask.select(sums[i], best_dists[i]);

                                    let mask_bits: SimdIdx = cast(mask);
                                    best_indices[i] =
                                        mask_bits.select(current_indices, best_indices[i]);
                                }

                                current_indices += increment;
                            }

                            // Reduce results for each point
                            for i in 0..batch_size {
                                let dists = best_dists[i].to_array();
                                let indices = best_indices[i].to_array();
                                let mut best_dist = <$scalar>::INFINITY;

                                for dist in dists {
                                    if dist < best_dist {
                                        best_dist = dist;
                                    }
                                }
                                let mut tied_lanes = best_dists[i]
                                    .simd_eq(SimdVec::splat(best_dist))
                                    .to_bitmask();
                                let mut best_idx = 0usize;
                                if tied_lanes != 0 {
                                    best_idx = usize::MAX;
                                    while tied_lanes != 0 {
                                        let lane = tied_lanes.trailing_zeros() as usize;
                                        best_idx = best_idx.min(indices[lane] as usize);
                                        tied_lanes &= tied_lanes - 1;
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
                    let all_lanes_valid = k == num_chunks * SIMD_WIDTH;

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

                            for (chunk_idx, chunk_data) in packed_centroids
                                .chunks_exact(ncols * SIMD_WIDTH)
                                .take(num_chunks)
                                .enumerate()
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

                                if !all_lanes_valid && chunk_idx + 1 == num_chunks {
                                    let valid_lanes = k - chunk_idx * SIMD_WIDTH;
                                    for sums in &mut sums[..batch_size] {
                                        mask_invalid_dot_lanes(sums, valid_lanes);
                                    }
                                }

                                for i in 0..batch_size {
                                    let mask = best_dots[i].simd_lt(sums[i]);
                                    best_dots[i] = mask.select(sums[i], best_dots[i]);

                                    let mask_bits: SimdIdx = cast(mask);
                                    best_indices[i] =
                                        mask_bits.select(current_indices, best_indices[i]);
                                }

                                current_indices += increment;
                            }

                            for i in 0..batch_size {
                                let dots = best_dots[i].to_array();
                                let indices = best_indices[i].to_array();
                                let mut best_dot = <$scalar>::NEG_INFINITY;

                                for dot in dots {
                                    if dot > best_dot {
                                        best_dot = dot;
                                    }
                                }
                                let mut tied_lanes =
                                    best_dots[i].simd_eq(SimdVec::splat(best_dot)).to_bitmask();
                                let mut best_idx = 0usize;
                                if tied_lanes != 0 {
                                    best_idx = usize::MAX;
                                    while tied_lanes != 0 {
                                        let lane = tied_lanes.trailing_zeros() as usize;
                                        best_idx = best_idx.min(indices[lane] as usize);
                                        tied_lanes &= tied_lanes - 1;
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
                    const POINT_BATCH: usize = 4;

                    let mut total = 0.0;
                    let mut processed = 0;

                    for points_batch in points.chunks(ncols * POINT_BATCH) {
                        let batch_size = points_batch.len() / ncols;
                        let mut sum_vecs = [SimdVec::ZERO; POINT_BATCH];
                        let mut j = 0;

                        while j + SIMD_WIDTH <= ncols {
                            let c_vec = load_chunk(centroid, j);
                            for i in 0..batch_size {
                                let point = &points_batch[i * ncols..(i + 1) * ncols];
                                let p_vec = load_chunk(point, j);
                                let diff = p_vec - c_vec;
                                sum_vecs[i] += diff * diff;
                            }
                            j += SIMD_WIDTH;
                        }

                        for i in 0..batch_size {
                            let point = &points_batch[i * ncols..(i + 1) * ncols];
                            let mut d = sum_vecs[i].reduce_add();
                            let mut tail = j;
                            while tail < ncols {
                                let diff = point[tail] - centroid[tail];
                                d += diff * diff;
                                tail += 1;
                            }

                            let min_dist = &mut min_dists[processed + i];
                            if d < *min_dist {
                                *min_dist = d;
                            }
                            total += *min_dist;
                        }

                        processed += batch_size;
                    }

                    total
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
            fn mask_invalid_dot_lanes(value: &mut SimdVec, valid_lanes: usize) {
                debug_assert!(valid_lanes < SIMD_WIDTH);
                let mut lanes = value.to_array();
                lanes[valid_lanes..].fill(<$scalar>::NEG_INFINITY);
                *value = SimdVec::from(lanes);
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
            fn update_centroids_impl<const TRACK_SHIFT: bool, S: PointSource<$scalar>>(
                packed_centroids: &mut [$scalar],
                sums: &[$scalar],
                counts: &[usize],
                source: &S,
                ncols: usize,
                rng: &mut impl Rng,
                empty_cluster_buffer: &mut Vec<$scalar>,
            ) -> $scalar {
                let num_chunks = counts.len().div_ceil(SIMD_WIDTH);
                let npoints = source.num_points();
                let mut max_shift: $scalar = 0.0;

                let required_buffer_len = SIMD_WIDTH * ncols;
                if npoints > 0
                    && counts.contains(&0)
                    && empty_cluster_buffer.len() < required_buffer_len
                {
                    empty_cluster_buffer.resize(required_buffer_len, 0.0);
                }
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
                        } else if npoints > 0 {
                            zero_indices[lane] = rng.random_range(0..npoints);
                            active_mask[lane] = false;
                        }
                    }

                    // Load zero samples once per chunk (better locality)
                    for lane in 0..SIMD_WIDTH {
                        let k_idx = chunk_idx * SIMD_WIDTH + lane;
                        if k_idx >= counts.len() {
                            break;
                        }

                        if counts[k_idx] == 0 && npoints > 0 {
                            source.read_batch(
                                zero_indices[lane],
                                1,
                                &mut empty_cluster_buffer[lane * ncols..],
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
                            let new_value = if active_mask[lane] {
                                arr[lane]
                            } else if counts[k_idx] == 0 && npoints > 0 {
                                empty_cluster_buffer[lane * ncols + d]
                            } else {
                                packed_centroids[dest_idx]
                            };

                            if TRACK_SHIFT {
                                let diff = (packed_centroids[dest_idx] - new_value).abs();
                                if diff > max_shift {
                                    max_shift = diff;
                                }
                            }
                            packed_centroids[dest_idx] = new_value;
                        }
                    }
                }
                max_shift
            }
        }
    };
}

impl_simd_backend!(
    SimdBackend128,
    f32,
    wide::f32x4,
    wide::u32x4,
    4,
    simd128_f32,
    u32
);
impl_simd_backend!(
    SimdBackend128,
    f64,
    wide::f64x2,
    wide::u64x2,
    2,
    simd128_f64,
    u64
);

impl_simd_backend!(
    SimdBackend256,
    f32,
    wide::f32x8,
    wide::u32x8,
    8,
    simd256_f32,
    u32
);
impl_simd_backend!(
    SimdBackend256,
    f64,
    wide::f64x4,
    wide::u64x4,
    4,
    simd256_f64,
    u64
);

impl_simd_backend!(
    SimdBackend512,
    f32,
    wide::f32x16,
    wide::u32x16,
    16,
    simd512_f32,
    u32
);

impl_simd_backend!(
    SimdBackend512,
    f64,
    wide::f64x8,
    wide::u64x8,
    8,
    simd512_f64,
    u64
);

macro_rules! impl_adaptive_backend {
    ($scalar:ty, $select_width:ident, $assign_helpers:ident) => {
        impl CoreBackend<$scalar> for SimdBackendAdaptive {
            #[inline(always)]
            fn prepared_centroid_layout(k: usize) -> PreparedCentroidLayout {
                prepared_layout($select_width(k))
            }

            #[inline(always)]
            fn accumulate_point_slice(
                point: &[$scalar],
                ncols: usize,
                sums: &mut [$scalar],
                label: usize,
            ) {
                dispatch_width!(
                    default_width(),
                    $scalar,
                    accumulate_point_slice(point, ncols, sums, label)
                )
            }

            #[inline(always)]
            fn prepare_centroids(centroids: &[$scalar], ncols: usize, k: usize) -> Vec<$scalar> {
                dispatch_width!(
                    $select_width(k),
                    $scalar,
                    prepare_centroids(centroids, ncols, k)
                )
            }

            #[inline(always)]
            fn finalize_centroids(packed: &[$scalar], ncols: usize, k: usize) -> Vec<$scalar> {
                dispatch_width!(
                    $select_width(k),
                    $scalar,
                    finalize_centroids(packed, ncols, k)
                )
            }

            #[inline(always)]
            fn transform_points_per_chunk(ncols: usize, k: usize) -> usize {
                calculate_transform_chunk_size::<$scalar>(ncols, k)
            }

            #[inline(always)]
            fn update_centroids<R: Rng, S: PointSource<$scalar>>(
                packed_centroids: &mut [$scalar],
                sums: &[$scalar],
                counts: &[usize],
                ncols: usize,
                source: &S,
                rng: &mut R,
                empty_cluster_buffer: &mut Vec<$scalar>,
            ) {
                dispatch_width!(
                    $select_width(counts.len()),
                    $scalar,
                    update_centroids(
                        packed_centroids,
                        sums,
                        counts,
                        ncols,
                        source,
                        rng,
                        empty_cluster_buffer
                    )
                )
            }

            #[inline(always)]
            fn update_centroids_and_get_max_shift<R: Rng, S: PointSource<$scalar>>(
                packed_centroids: &mut [$scalar],
                sums: &[$scalar],
                counts: &[usize],
                ncols: usize,
                source: &S,
                rng: &mut R,
                empty_cluster_buffer: &mut Vec<$scalar>,
            ) -> $scalar {
                dispatch_width!(
                    $select_width(counts.len()),
                    $scalar,
                    update_centroids_and_get_max_shift(
                        packed_centroids,
                        sums,
                        counts,
                        ncols,
                        source,
                        rng,
                        empty_cluster_buffer
                    )
                )
            }

            #[inline(always)]
            fn assign_and_accumulate_euc(
                points: &[$scalar],
                ncols: usize,
                packed_centroids: &[$scalar],
                k: usize,
                sums: &mut [$scalar],
                counts: &mut [usize],
                total: &mut $scalar,
            ) {
                match $select_width(k) {
                    AdaptiveWidth::W128 => $assign_helpers::euc128(
                        points,
                        ncols,
                        packed_centroids,
                        k,
                        sums,
                        counts,
                        total,
                    ),
                    AdaptiveWidth::W256 => $assign_helpers::euc256(
                        points,
                        ncols,
                        packed_centroids,
                        k,
                        sums,
                        counts,
                        total,
                    ),
                    AdaptiveWidth::W512 => $assign_helpers::euc512(
                        points,
                        ncols,
                        packed_centroids,
                        k,
                        sums,
                        counts,
                        total,
                    ),
                }
            }

            #[inline(always)]
            fn assign_and_accumulate_dot(
                points: &[$scalar],
                ncols: usize,
                packed_centroids: &[$scalar],
                k: usize,
                sums: &mut [$scalar],
                counts: &mut [usize],
                total: &mut $scalar,
            ) {
                match $select_width(k) {
                    AdaptiveWidth::W128 => $assign_helpers::dot128(
                        points,
                        ncols,
                        packed_centroids,
                        k,
                        sums,
                        counts,
                        total,
                    ),
                    AdaptiveWidth::W256 => $assign_helpers::dot256(
                        points,
                        ncols,
                        packed_centroids,
                        k,
                        sums,
                        counts,
                        total,
                    ),
                    AdaptiveWidth::W512 => $assign_helpers::dot512(
                        points,
                        ncols,
                        packed_centroids,
                        k,
                        sums,
                        counts,
                        total,
                    ),
                }
            }

            #[inline(always)]
            fn transform_euc(
                points: &[$scalar],
                ncols: usize,
                packed_centroids: &[$scalar],
                k: usize,
                out_scores: &mut [$scalar],
            ) {
                dispatch_width!(
                    $select_width(k),
                    $scalar,
                    transform_euc(points, ncols, packed_centroids, k, out_scores)
                )
            }

            #[inline(always)]
            fn transform_dot(
                points: &[$scalar],
                ncols: usize,
                packed_centroids: &[$scalar],
                k: usize,
                out_scores: &mut [$scalar],
            ) {
                dispatch_width!(
                    $select_width(k),
                    $scalar,
                    transform_dot(points, ncols, packed_centroids, k, out_scores)
                )
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
                dispatch_width!(
                    $select_width(k),
                    $scalar,
                    find_nearest_centroids_euc(
                        points,
                        ncols,
                        packed_centroids,
                        k,
                        out_indices,
                        out_distances
                    )
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
                dispatch_width!(
                    $select_width(k),
                    $scalar,
                    find_nearest_centroids_euc_with_dists(
                        points,
                        ncols,
                        packed_centroids,
                        k,
                        out_indices,
                        out_distances
                    )
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
                dispatch_width!(
                    $select_width(k),
                    $scalar,
                    find_nearest_centroids_dot_product(
                        points,
                        ncols,
                        packed_centroids,
                        k,
                        out_indices,
                        out_distances
                    )
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
                dispatch_width!(
                    $select_width(k),
                    $scalar,
                    find_nearest_centroids_dot_product_with_dists(
                        points,
                        ncols,
                        packed_centroids,
                        k,
                        out_indices,
                        out_distances
                    )
                )
            }

            #[inline(always)]
            fn calculate_and_update_min_distance_euc(
                points: &[$scalar],
                ncols: usize,
                centroid: &[$scalar],
                min_dists: &mut [$scalar],
            ) {
                dispatch_width!(
                    default_width(),
                    $scalar,
                    calculate_and_update_min_distance_euc(points, ncols, centroid, min_dists)
                )
            }

            #[inline(always)]
            fn calculate_and_update_min_distance_euc_sum(
                points: &[$scalar],
                ncols: usize,
                centroid: &[$scalar],
                min_dists: &mut [$scalar],
            ) -> $scalar {
                dispatch_width!(
                    default_width(),
                    $scalar,
                    calculate_and_update_min_distance_euc_sum(points, ncols, centroid, min_dists)
                )
            }

            #[inline(always)]
            fn calculate_and_update_min_distance_dot(
                points: &[$scalar],
                ncols: usize,
                centroid: &[$scalar],
                min_dists: &mut [$scalar],
            ) {
                dispatch_width!(
                    default_width(),
                    $scalar,
                    calculate_and_update_min_distance_dot(points, ncols, centroid, min_dists)
                )
            }

            #[inline(always)]
            fn calculate_and_update_min_distance_dot_sum(
                points: &[$scalar],
                ncols: usize,
                centroid: &[$scalar],
                min_dists: &mut [$scalar],
            ) -> $scalar {
                dispatch_width!(
                    default_width(),
                    $scalar,
                    calculate_and_update_min_distance_dot_sum(points, ncols, centroid, min_dists)
                )
            }
        }
    };
}

impl_adaptive_backend!(f32, adaptive_width_f32, adaptive_assign_f32);
impl_adaptive_backend!(f64, adaptive_width_f64, adaptive_assign_f64);

#[cfg(test)]
#[path = "../tests/unit/simd_core.rs"]
mod tests;
