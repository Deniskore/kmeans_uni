use bytemuck::cast_slice;
use kmeans_uni::{KMeansBuilder, PointSource};
use linfa::DatasetBase;
use linfa::traits::Fit;
use linfa_clustering::KMeans as LinfaKMeans;
use memmap2::Mmap;
use ndarray::Array2;
use rand08::{Rng, SeedableRng, rngs::StdRng};
use std::io::Write;

const MMAP_FILE_SIZE_BYTES: usize = 1024 * 1024; // 1MB
const N_COLS: usize = 8;
const K: usize = 4;
const N_POINTS: usize = MMAP_FILE_SIZE_BYTES / (std::mem::size_of::<f32>() * N_COLS);
const ITERATIONS: usize = 50;
const SEED: u64 = 0xBEEFu64;

#[test]
fn sync_memmap_matches_linfa() {
    let data = make_data();
    let (_temp, mmap) = write_and_mmap(&data);
    let source = MmapPointSource::new(mmap, N_COLS).expect("mmap source");

    let ours = fit_kmeans_from_source(&source, ITERATIONS, SEED);
    let linfa = fit_linfa(&data, ITERATIONS, SEED);

    assert_costs_close(&data, &ours, &linfa, 5.0);
}

fn make_data() -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut data = Vec::with_capacity(N_POINTS * N_COLS);
    for _ in 0..(N_POINTS * N_COLS) {
        data.push(rng.gen_range(-100.0f32..100.0f32));
    }
    data
}

fn write_and_mmap(data: &[f32]) -> (tempfile::NamedTempFile, Mmap) {
    let mut file = tempfile::NamedTempFile::new().expect("tempfile");
    file.write_all(cast_slice(data)).expect("write data");
    let mmap = unsafe {
        memmap2::MmapOptions::new()
            .map(file.as_file())
            .expect("mmap")
    };
    (file, mmap)
}

struct MmapPointSource {
    mmap: Mmap,
    ncols: usize,
}

impl MmapPointSource {
    fn new(mmap: Mmap, ncols: usize) -> Result<Self, String> {
        if ncols == 0 {
            return Err("ncols must be > 0".into());
        }
        if !mmap
            .len()
            .is_multiple_of(std::mem::size_of::<f32>() * ncols)
        {
            return Err("mmap length not divisible by ncols*sizeof(f32)".into());
        }
        Ok(Self { mmap, ncols })
    }
}

impl PointSource<f32> for MmapPointSource {
    fn num_points(&self) -> usize {
        self.mmap.len() / (std::mem::size_of::<f32>() * self.ncols)
    }

    fn num_columns(&self) -> usize {
        self.ncols
    }

    fn view_batch(&self, start: usize, count: usize) -> kmeans_uni::Result<&[f32]> {
        if count == 0 {
            return Ok(&[]);
        }
        let end = start
            .checked_add(count)
            .ok_or_else(|| kmeans_uni::Error::InvalidInput("batch range overflow".into()))?;
        let npoints = self.num_points();
        if end > npoints {
            return Err(kmeans_uni::Error::InvalidInput(
                "batch range exceeds available points".into(),
            ));
        }
        let floats: &[f32] = bytemuck::try_cast_slice(&self.mmap).map_err(|_| {
            kmeans_uni::Error::InvalidInput("mmap alignment invalid for f32".into())
        })?;
        let len = count * self.ncols;
        let start_idx = start * self.ncols;
        let end_idx = start_idx + len;
        Ok(&floats[start_idx..end_idx])
    }
}

fn fit_kmeans_from_source<S: PointSource<f32>>(
    source: &S,
    iterations: usize,
    seed: u64,
) -> Vec<f32> {
    KMeansBuilder::new(K)
        .iterations(iterations)
        .cpu_scalar()
        .euclidean()
        .seed(seed)
        .build()
        .fit_from_source(source)
        .expect("kmeans fit")
        .into_centroids()
}

fn fit_linfa(data: &[f32], iterations: usize, seed: u64) -> Vec<f32> {
    let npoints = data.len() / N_COLS;
    let data_f64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
    let array = Array2::from_shape_vec((npoints, N_COLS), data_f64).expect("array shape");
    let dataset = DatasetBase::new(array, ());
    LinfaKMeans::params_with_rng(K, StdRng::seed_from_u64(seed))
        .max_n_iterations(iterations as u64)
        .tolerance(1e-5)
        .fit(&dataset)
        .expect("linfa fit")
        .centroids()
        .mapv(|x| x as f32)
        .into_raw_vec_and_offset()
        .0
}

fn assert_costs_close(data: &[f32], ours: &[f32], linfa: &[f32], tolerance_percent: f32) {
    let ours_cost = inertia(data, ours);
    let linfa_cost = inertia(data, linfa);
    let diff_percent = ((ours_cost - linfa_cost).abs() / linfa_cost) * 100.0;
    assert!(
        diff_percent < tolerance_percent,
        "inertia diff too large: ours {:.4}, linfa {:.4}, diff {:.2}%",
        ours_cost,
        linfa_cost,
        diff_percent
    );
}

fn inertia(points: &[f32], centroids: &[f32]) -> f32 {
    let mut total = 0.0;
    for point in points.chunks_exact(N_COLS) {
        let mut best = f32::INFINITY;
        for centroid in centroids.chunks_exact(N_COLS) {
            let mut dist = 0.0;
            for (a, b) in point.iter().zip(centroid.iter()) {
                let d = a - b;
                dist += d * d;
            }
            if dist < best {
                best = dist;
            }
        }
        total += best;
    }
    total
}
