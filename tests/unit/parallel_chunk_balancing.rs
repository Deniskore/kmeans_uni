use super::*;

#[test]
fn parallel_chunk_balancing_has_a_real_sequential_cutoff() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("test thread pool");

    pool.install(|| {
        assert_eq!(
            balanced_parallel_chunk_size(511, usize::MAX, MIN_POINTS_PER_PARALLEL_TASK),
            None
        );
        assert_eq!(
            balanced_parallel_chunk_size(512, usize::MAX, MIN_POINTS_PER_PARALLEL_TASK),
            Some(128)
        );
        assert_eq!(
            balanced_parallel_chunk_size(513, usize::MAX, MIN_POINTS_PER_PARALLEL_TASK),
            Some(129)
        );
        assert_eq!(
            balanced_parallel_chunk_size(1023, usize::MAX, MIN_TRANSFORM_POINTS_PER_PARALLEL_TASK,),
            None
        );
        assert_eq!(
            balanced_parallel_chunk_size(1024, usize::MAX, MIN_TRANSFORM_POINTS_PER_PARALLEL_TASK,),
            Some(256)
        );
    });
}
