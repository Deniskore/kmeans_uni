use kmeans_uni::{KMeans, KMeansBuilder, MetricType, PointSource, SlicePointSource};

const N_COLS: usize = 2;
const K: usize = 3;
const FULL_ITERATIONS: usize = 20;
const MINI_BATCH_ITERATIONS: usize = 24;
const MINI_BATCH_SIZE: usize = 4;
const SEED: u64 = 0xC0FF_EE11;
const TOLERANCE: f64 = 1e-6;

struct CopyOnlySource {
    data: Vec<f32>,
    ncols: usize,
}

impl CopyOnlySource {
    fn new(data: Vec<f32>, ncols: usize) -> Self {
        Self { data, ncols }
    }
}

impl PointSource<f32> for CopyOnlySource {
    fn num_points(&self) -> usize {
        self.data.len() / self.ncols
    }

    fn num_columns(&self) -> usize {
        self.ncols
    }

    fn supports_view_batch(&self) -> bool {
        false
    }

    fn view_batch(&self, _start: usize, _count: usize) -> kmeans_uni::Result<&[f32]> {
        unreachable!("copy-only source does not expose contiguous batch views")
    }

    fn read_batch(&self, start: usize, count: usize, dst: &mut [f32]) {
        let len = count * self.ncols;
        let src_start = start * self.ncols;
        let src_end = src_start + len;
        dst[..len].copy_from_slice(&self.data[src_start..src_end]);
    }
}

#[test]
fn predict_from_copy_only_source_matches_slice_source() {
    let data = build_dataset();
    let copy_source = CopyOnlySource::new(data.clone(), N_COLS);
    let slice_source = SlicePointSource::new(&data, N_COLS).expect("slice source");
    let model = KMeans::new(
        vec![0.0f32, 0.0, 10.0, 10.0, -10.0, 10.0],
        N_COLS,
        K,
        0.0,
        MetricType::Euclidean,
    )
    .expect("valid model");

    let expected = model
        .predict_from_source(&slice_source)
        .expect("slice predict succeeds");
    let actual = model
        .predict_from_source(&copy_source)
        .expect("copy-only predict succeeds");

    assert_eq!(expected, actual);
}

#[test]
fn fit_from_copy_only_source_matches_slice_source() {
    let data = build_dataset();
    let copy_source = CopyOnlySource::new(data.clone(), N_COLS);
    let slice_source = SlicePointSource::new(&data, N_COLS).expect("slice source");

    let expected = fit_full_sequential(&slice_source);
    let actual = fit_full_sequential(&copy_source);

    assert_models_close(&expected, &actual);
}

#[test]
fn fit_from_copy_only_source_parallel_matches_slice_source() {
    let data = build_dataset();
    let copy_source = CopyOnlySource::new(data.clone(), N_COLS);
    let slice_source = SlicePointSource::new(&data, N_COLS).expect("slice source");

    let expected = fit_full_parallel(&slice_source);
    let actual = fit_full_parallel(&copy_source);

    assert_models_close(&expected, &actual);
}

#[test]
fn fit_mini_batch_from_copy_only_source_matches_slice_source() {
    let data = build_dataset();
    let copy_source = CopyOnlySource::new(data.clone(), N_COLS);
    let slice_source = SlicePointSource::new(&data, N_COLS).expect("slice source");

    let expected = fit_mini_batch_sequential(&slice_source);
    let actual = fit_mini_batch_sequential(&copy_source);

    assert_models_close(&expected, &actual);
}

#[test]
fn fit_mini_batch_from_copy_only_source_parallel_matches_slice_source() {
    let data = build_dataset();
    let copy_source = CopyOnlySource::new(data.clone(), N_COLS);
    let slice_source = SlicePointSource::new(&data, N_COLS).expect("slice source");

    let expected = fit_mini_batch_parallel(&slice_source);
    let actual = fit_mini_batch_parallel(&copy_source);

    assert_models_close(&expected, &actual);
}

fn fit_full_sequential<S: PointSource<f32>>(source: &S) -> KMeans<f32> {
    KMeansBuilder::<f32>::new(K)
        .iterations(FULL_ITERATIONS)
        .attempts(1)
        .cpu_scalar()
        .euclidean()
        .init_plus_plus()
        .seed(SEED)
        .build()
        .fit_from_source(source)
        .expect("full sequential fit succeeds")
}

fn fit_full_parallel<S: PointSource<f32>>(source: &S) -> KMeans<f32> {
    KMeansBuilder::<f32>::new(K)
        .iterations(FULL_ITERATIONS)
        .attempts(1)
        .cpu_scalar()
        .euclidean()
        .init_plus_plus()
        .seed(SEED)
        .parallel()
        .fit_from_source(source)
        .expect("full parallel fit succeeds")
}

fn fit_mini_batch_sequential<S: PointSource<f32>>(source: &S) -> KMeans<f32> {
    KMeansBuilder::<f32>::new(K)
        .iterations(MINI_BATCH_ITERATIONS)
        .attempts(1)
        .cpu_scalar()
        .euclidean()
        .init_plus_plus()
        .seed(SEED)
        .mini_batch_rel_tolerance(0.0)
        .mini_batch_patience(0)
        .build()
        .fit_mini_batch_from_source(source, MINI_BATCH_SIZE)
        .expect("mini-batch sequential fit succeeds")
}

fn fit_mini_batch_parallel<S: PointSource<f32>>(source: &S) -> KMeans<f32> {
    KMeansBuilder::<f32>::new(K)
        .iterations(MINI_BATCH_ITERATIONS)
        .attempts(1)
        .cpu_scalar()
        .euclidean()
        .init_plus_plus()
        .seed(SEED)
        .mini_batch_rel_tolerance(0.0)
        .mini_batch_patience(0)
        .parallel()
        .fit_mini_batch_from_source(source, MINI_BATCH_SIZE)
        .expect("mini-batch parallel fit succeeds")
}

fn assert_models_close(expected: &KMeans<f32>, actual: &KMeans<f32>) {
    assert_centroids_match_by_sorting(expected.centroids(), actual.centroids(), N_COLS, TOLERANCE);

    let inertia_diff = (expected.inertia() - actual.inertia()).abs() as f64;
    assert!(
        inertia_diff <= TOLERANCE,
        "inertia diverged: expected {}, actual {}",
        expected.inertia(),
        actual.inertia()
    );
}

fn build_dataset() -> Vec<f32> {
    vec![
        -10.1, 10.0, -9.9, 10.2, -10.2, 9.8, -9.8, 10.1, 0.0, 0.1, 0.2, -0.1, -0.2, 0.0, 0.1, 0.2,
        10.0, 10.0, 10.2, 9.8, 9.8, 10.1, 10.1, 10.2,
    ]
}

fn assert_centroids_match_by_sorting(
    expected: &[f32],
    actual: &[f32],
    ncols: usize,
    tolerance: f64,
) {
    let mut remaining: Vec<Vec<f64>> = actual
        .chunks(ncols)
        .map(|chunk| chunk.iter().map(|&v| v as f64).collect())
        .collect();
    let expected: Vec<Vec<f64>> = expected
        .chunks(ncols)
        .map(|chunk| chunk.iter().map(|&v| v as f64).collect())
        .collect();

    assert_eq!(expected.len(), remaining.len(), "centroid counts differ");

    for left in expected {
        let (idx, diff) = remaining
            .iter()
            .enumerate()
            .map(|(idx, right)| (idx, distance_sq(left.as_slice(), right.as_slice()).sqrt()))
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("distances are finite"))
            .expect("must have centroid match");

        assert!(
            diff <= tolerance,
            "centroids diverged: {left:?} vs {:?} (dist={diff})",
            remaining[idx]
        );
        remaining.remove(idx);
    }
}

fn distance_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let dx = x - y;
            dx * dx
        })
        .sum()
}
