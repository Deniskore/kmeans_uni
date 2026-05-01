use crate::Primitive;

#[inline(always)]
pub fn find_nearest_centroids_generic<F, CoreFn>(
    points: &[F],
    ncols: usize,
    _k: usize,
    out_indices: &mut [usize],
    out_distances: Option<&mut [F]>,
    mut core_fn: CoreFn,
) where
    F: Primitive,
    CoreFn: FnMut(&[F], usize, &mut [usize], Option<&mut [F]>),
{
    const POINT_BATCH: usize = 4;

    // Prepare iterators
    let points_chunks = points.chunks(ncols * POINT_BATCH);
    let indices_chunks = out_indices.chunks_mut(POINT_BATCH);

    if let Some(dists) = out_distances {
        let dists_chunks = dists.chunks_mut(POINT_BATCH);

        for ((points_batch, indices_batch), dists_batch) in
            points_chunks.zip(indices_chunks).zip(dists_chunks)
        {
            let batch_size = points_batch.len() / ncols;
            core_fn(points_batch, batch_size, indices_batch, Some(dists_batch));
        }
    } else {
        for (points_batch, indices_batch) in points_chunks.zip(indices_chunks) {
            let batch_size = points_batch.len() / ncols;
            core_fn(points_batch, batch_size, indices_batch, None);
        }
    }
}

#[inline(always)]
pub fn calculate_and_update_min_distance_generic<F, CoreFn>(
    points: &[F],
    ncols: usize,
    min_dists: &mut [F],
    mut core_fn: CoreFn,
) where
    F: Primitive,
    CoreFn: FnMut(&[F], &mut F),
{
    for (point, min_dist) in points.chunks_exact(ncols).zip(min_dists.iter_mut()) {
        core_fn(point, min_dist);
    }
}

#[inline(always)]
pub fn calculate_and_update_min_distance_generic_sum<F, CoreFn>(
    points: &[F],
    ncols: usize,
    min_dists: &mut [F],
    mut core_fn: CoreFn,
) -> F
where
    F: Primitive,
    CoreFn: FnMut(&[F], &mut F),
{
    let mut sum = F::zero();
    for (point, min_dist) in points.chunks_exact(ncols).zip(min_dists.iter_mut()) {
        core_fn(point, min_dist);
        sum = sum + *min_dist;
    }
    sum
}

#[inline(always)]
pub fn calculate_chunk_size<F: Primitive>(ncols: usize) -> usize {
    let bytes_per_point = ncols.saturating_mul(std::mem::size_of::<F>()).max(4);
    let target_bytes = 64 * 1024;
    let points_per_chunk = (target_bytes / bytes_per_point.max(1)).max(1);
    points_per_chunk.clamp(128, 262_144)
}

#[inline(always)]
pub fn calculate_transform_chunk_size<F: Primitive>(ncols: usize, k: usize) -> usize {
    let elem_size = std::mem::size_of::<F>().max(4);
    let bytes_per_point = ncols.saturating_mul(elem_size);
    let bytes_per_output_row = k.saturating_mul(elem_size);
    let working_set = bytes_per_point
        .saturating_add(bytes_per_output_row)
        .max(elem_size);
    let target_bytes = 128 * 1024;
    let points_per_chunk = (target_bytes / working_set.max(1)).max(1);
    points_per_chunk.clamp(128, 262_144)
}
