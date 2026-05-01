use crate::Primitive;
use crate::error::{Error, Result};

/// Point source with a zero-copy view for contiguous, row-major batches.
///
/// Note: call sites assume reads are infallible. Implementors should validate
/// bounds and either return an error from `view_batch` or override
/// `read_batch` to surface failures instead of silently dropping data.
pub trait PointSource<F: Primitive>: Send + Sync {
    fn num_points(&self) -> usize;
    fn num_columns(&self) -> usize;

    /// Returns whether this source can provide contiguous batch views via `view_batch`.
    ///
    /// Copy-only sources can override this to `false` to let callers skip a failed
    /// `view_batch` probe on every chunk and fall back directly to `read_batch`.
    fn supports_view_batch(&self) -> bool {
        true
    }

    /// Return a contiguous slice for `count` points starting at `start`.
    fn view_batch(&self, start: usize, count: usize) -> Result<&[F]>;

    /// Copy a batch into `dst`. The default calls `view_batch` and ignores errors,
    /// so callers must ensure `start`/`count` are in range and `dst` is large enough.
    fn read_batch(&self, start: usize, count: usize, dst: &mut [F]) {
        if let Ok(view) = self.view_batch(start, count) {
            let len = count * self.num_columns();
            if dst.len() >= len {
                dst[..len].copy_from_slice(view);
            }
        }
    }

    fn read_point(&self, index: usize, dst: &mut [F]) {
        self.read_batch(index, 1, dst);
    }
}

#[inline]
pub(crate) fn view_or_copy_batch<'a, F: Primitive, S: PointSource<F>>(
    source: &'a S,
    fallback_buffer: &'a mut Option<Vec<F>>,
    start: usize,
    count: usize,
    chunk_capacity: usize,
    ncols: usize,
) -> Result<&'a [F]> {
    if source.supports_view_batch()
        && let Ok(view) = source.view_batch(start, count)
    {
        return Ok(view);
    }

    let buffer = fallback_buffer
        .get_or_insert_with(|| vec![F::zero(); chunk_capacity.saturating_mul(ncols)]);
    let batch = &mut buffer[..count * ncols];
    source.read_batch(start, count, batch);
    Ok(batch)
}

/// Slice-backed point source with zero-copy views.
pub struct SlicePointSource<'a, F: Primitive> {
    points: &'a [F],
    ncols: usize,
}

impl<'a, F: Primitive> SlicePointSource<'a, F> {
    pub fn new(points: &'a [F], ncols: usize) -> Result<Self> {
        if ncols == 0 {
            return Err(Error::InvalidInput(
                "slice sources require at least one column".into(),
            ));
        }
        if !points.len().is_multiple_of(ncols) {
            return Err(Error::InvalidInput(
                "points length must be divisible by the column count".into(),
            ));
        }
        Ok(Self { points, ncols })
    }
}

impl<'a, F: Primitive> PointSource<F> for SlicePointSource<'a, F> {
    fn num_points(&self) -> usize {
        self.points.len() / self.ncols
    }

    fn num_columns(&self) -> usize {
        self.ncols
    }

    fn view_batch(&self, start: usize, count: usize) -> Result<&[F]> {
        if count == 0 {
            return Ok(&[]);
        }
        let end = start
            .checked_add(count)
            .ok_or_else(|| Error::InvalidInput("batch range overflow".into()))?;
        let npoints = self.num_points();
        if end > npoints {
            return Err(Error::InvalidInput(
                "batch range exceeds available points".into(),
            ));
        }
        let len = count * self.ncols;
        let src_start = start * self.ncols;
        let src_end = src_start + len;
        Ok(&self.points[src_start..src_end])
    }
}
