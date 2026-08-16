use super::*;

fn assert_dot_assignment_ignores_padding<B: CoreBackend<f32>>() {
    const K: usize = 17;
    const EXPECTED_LABEL: usize = 15;

    let points = [1.0_f32];
    let mut centroids = vec![0.0_f32; K];
    centroids[EXPECTED_LABEL] = 100.0;
    centroids[K - 1] = 1.0;
    let packed_centroids = B::prepare_centroids(&centroids, 1, K);
    let mut sums = vec![0.0; K];
    let mut counts = vec![0; K];
    let mut total = 0.0;

    B::assign_and_accumulate_dot(
        &points,
        1,
        &packed_centroids,
        K,
        &mut sums,
        &mut counts,
        &mut total,
    );

    assert_eq!(counts[EXPECTED_LABEL], 1);
    assert_eq!(counts.iter().sum::<usize>(), 1);
    assert_eq!(sums[EXPECTED_LABEL], points[0]);
    assert_eq!(total, 100.0);
}

fn assert_dot_assignment_ignores_padding_f64<B: CoreBackend<f64>>() {
    const K: usize = 17;
    const EXPECTED_LABEL: usize = 15;

    let points = [1.0_f64];
    let mut centroids = vec![0.0_f64; K];
    centroids[EXPECTED_LABEL] = 100.0;
    centroids[K - 1] = 1.0;
    let packed_centroids = B::prepare_centroids(&centroids, 1, K);
    let mut sums = vec![0.0; K];
    let mut counts = vec![0; K];
    let mut total = 0.0;

    B::assign_and_accumulate_dot(
        &points,
        1,
        &packed_centroids,
        K,
        &mut sums,
        &mut counts,
        &mut total,
    );

    assert_eq!(counts[EXPECTED_LABEL], 1);
    assert_eq!(counts.iter().sum::<usize>(), 1);
    assert_eq!(sums[EXPECTED_LABEL], points[0]);
    assert_eq!(total, 100.0);
}

#[test]
fn dot_assignment_ignores_padded_centroid_lanes() {
    assert_dot_assignment_ignores_padding::<SimdBackend128>();
    assert_dot_assignment_ignores_padding::<SimdBackend256>();
    assert_dot_assignment_ignores_padding::<SimdBackend512>();
    assert_dot_assignment_ignores_padding::<SimdBackendAdaptive>();

    assert_dot_assignment_ignores_padding_f64::<SimdBackend128>();
    assert_dot_assignment_ignores_padding_f64::<SimdBackend256>();
    assert_dot_assignment_ignores_padding_f64::<SimdBackend512>();
    assert_dot_assignment_ignores_padding_f64::<SimdBackendAdaptive>();
}

fn assert_cross_lane_assignment_ties_choose_earliest_f32<B: CoreBackend<f32>>() {
    const K: usize = 17;
    const TIED_INDICES: [usize; 4] = [1, 4, 8, 16];
    let points = [1.0_f32];

    let mut centroids = vec![100.0_f32; K];
    for index in TIED_INDICES {
        centroids[index] = 2.0;
    }
    let packed = B::prepare_centroids(&centroids, 1, K);
    let mut sums = vec![0.0; K];
    let mut counts = vec![0; K];
    let mut total = 0.0;
    B::assign_and_accumulate_euc(&points, 1, &packed, K, &mut sums, &mut counts, &mut total);
    assert_eq!(counts[TIED_INDICES[0]], 1);
    assert_eq!(counts.iter().sum::<usize>(), 1);

    let mut centroids = vec![-100.0_f32; K];
    for index in TIED_INDICES {
        centroids[index] = 5.0;
    }
    let packed = B::prepare_centroids(&centroids, 1, K);
    counts.fill(0);
    sums.fill(0.0);
    total = 0.0;
    B::assign_and_accumulate_dot(&points, 1, &packed, K, &mut sums, &mut counts, &mut total);
    assert_eq!(counts[TIED_INDICES[0]], 1);
    assert_eq!(counts.iter().sum::<usize>(), 1);
}

fn assert_cross_lane_assignment_ties_choose_earliest_f64<B: CoreBackend<f64>>() {
    const K: usize = 17;
    const TIED_INDICES: [usize; 4] = [1, 4, 8, 16];
    let points = [1.0_f64];

    let mut centroids = vec![100.0_f64; K];
    for index in TIED_INDICES {
        centroids[index] = 2.0;
    }
    let packed = B::prepare_centroids(&centroids, 1, K);
    let mut sums = vec![0.0; K];
    let mut counts = vec![0; K];
    let mut total = 0.0;
    B::assign_and_accumulate_euc(&points, 1, &packed, K, &mut sums, &mut counts, &mut total);
    assert_eq!(counts[TIED_INDICES[0]], 1);
    assert_eq!(counts.iter().sum::<usize>(), 1);

    let mut centroids = vec![-100.0_f64; K];
    for index in TIED_INDICES {
        centroids[index] = 5.0;
    }
    let packed = B::prepare_centroids(&centroids, 1, K);
    counts.fill(0);
    sums.fill(0.0);
    total = 0.0;
    B::assign_and_accumulate_dot(&points, 1, &packed, K, &mut sums, &mut counts, &mut total);
    assert_eq!(counts[TIED_INDICES[0]], 1);
    assert_eq!(counts.iter().sum::<usize>(), 1);
}

#[test]
fn cross_lane_assignment_ties_choose_the_earliest_centroid() {
    assert_cross_lane_assignment_ties_choose_earliest_f32::<SimdBackend128>();
    assert_cross_lane_assignment_ties_choose_earliest_f32::<SimdBackend256>();
    assert_cross_lane_assignment_ties_choose_earliest_f32::<SimdBackend512>();
    assert_cross_lane_assignment_ties_choose_earliest_f32::<SimdBackendAdaptive>();
    assert_cross_lane_assignment_ties_choose_earliest_f64::<SimdBackend128>();
    assert_cross_lane_assignment_ties_choose_earliest_f64::<SimdBackend256>();
    assert_cross_lane_assignment_ties_choose_earliest_f64::<SimdBackend512>();
    assert_cross_lane_assignment_ties_choose_earliest_f64::<SimdBackendAdaptive>();
}

#[test]
fn native_128_targets_never_select_composite_widths() {
    if cfg!(any(target_arch = "aarch64", target_arch = "wasm32")) {
        for k in 1..=128 {
            assert_eq!(adaptive_width_f32(k), AdaptiveWidth::W128);
            assert_eq!(adaptive_width_f64(k), AdaptiveWidth::W128);
        }
    }
}

#[test]
fn x86_like_policy_covers_padding_and_wide_cases() {
    if !cfg!(any(target_arch = "aarch64", target_arch = "wasm32")) {
        assert_eq!(adaptive_width_f32(8), AdaptiveWidth::W256);
        if cfg!(target_feature = "avx") {
            assert_eq!(adaptive_width_f32(17), AdaptiveWidth::W256);
            assert_eq!(
                adaptive_width_f32(32),
                if cfg!(target_feature = "avx512f") {
                    AdaptiveWidth::W512
                } else {
                    AdaptiveWidth::W256
                }
            );
        } else {
            assert_eq!(adaptive_width_f32(17), AdaptiveWidth::W128);
            assert_eq!(adaptive_width_f32(32), AdaptiveWidth::W512);
        }
        assert_eq!(adaptive_width_f64(2), AdaptiveWidth::W256);
        assert_eq!(adaptive_width_f64(16), AdaptiveWidth::W256);
        assert_eq!(adaptive_width_f64(17), AdaptiveWidth::W256);
    }
}
