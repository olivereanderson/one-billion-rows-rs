//! This module contains solutions aiming to take advantage of simd instructions. It uses the (currently) nightly only std::simd API.
use super::*;
#[cfg(any(
    not(feature = "vbmi"),
    not(all(
        target_arch = "x86_64",
        target_feature = "avx512vbmi2",
        target_feature = "avx512vbmi"
    ))
))]
mod generic_impl;
#[cfg(all(
    feature = "vbmi",
    all(
        target_arch = "x86_64",
        target_feature = "avx512vbmi2",
        target_feature = "avx512vbmi"
    )
))]
mod vbmi_impl;

#[cfg(any(
    not(feature = "vbmi"),
    not(all(
        target_arch = "x86_64",
        target_feature = "avx512vbmi2",
        target_feature = "avx512vbmi"
    ))
))]
use generic_impl::collect_stats_simd;
#[cfg(all(
    feature = "vbmi",
    all(
        target_arch = "x86_64",
        target_feature = "avx512vbmi2",
        target_feature = "avx512vbmi"
    )
))]
use vbmi_impl::collect_stats_simd;

/// Solve the challenge with a single thread using vectorized code.
#[inline(always)]
pub fn solve_challenge<W: Write>(input: OneBillionRowsChallengeRows<'_>, writer: &mut W) {
    let hasher = SelectedBuildHasher::default();
    let mut stats_per_station: HashMap<Box<[u8]>, StationStats, SelectedBuildHasher> =
        HashMap::with_capacity_and_hasher(MAX_STATION_NAMES, hasher);
    let collector = global_alloc_collecting_closure(&mut stats_per_station);
    collect_stats_simd(input, collector);
    write_summary(stats_per_station.into_iter(), writer);
}

/// Solve the challenge with up to the given number of threads using vectorized code.
#[inline(always)]
pub fn solve_challenge_with_threads<W: Write>(
    input: OneBillionRowsChallengeRows<'_>,
    writer: &mut W,
    num_threads: usize,
) {
    if num_threads < 2 {
        return solve_challenge(input, writer);
    }
    let joined_maps = std::thread::scope(|s| {
        let mut handles: Vec<_> = input
            .chunks(num_threads)
            .map(|chunk| {
                s.spawn(move || {
                    let hasher = SelectedBuildHasher::default();
                    let mut stats_per_station: HashMap<
                        Box<[u8]>,
                        StationStats,
                        SelectedBuildHasher,
                    > = HashMap::with_capacity_and_hasher(MAX_STATION_NAMES, hasher);
                    let collector = global_alloc_collecting_closure(&mut stats_per_station);
                    collect_stats_simd(chunk, collector);
                    stats_per_station
                })
            })
            .collect();
        let mut main_collector = handles.pop().unwrap().join().unwrap();
        for handle in handles {
            let collector = handle.join().unwrap();
            join_entries(&mut main_collector, collector);
        }
        main_collector
    });
    write_summary(joined_maps.into_iter(), writer);
}
