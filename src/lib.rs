#![cfg_attr(
    feature = "nightly",
    feature(portable_simd, stdarch_x86_avx512, array_chunks)
)]
mod input;
pub mod scalar;
#[cfg(feature = "simd")]
pub mod simd;
mod temperature;
use std::{
    borrow::Borrow,
    collections::HashMap,
    hash::{BuildHasher, BuildHasherDefault, Hash},
    io::Write,
};

#[cfg(feature = "gxhash")]
use gxhash::GxHasher as SelectedHasher;
pub use input::OneBillionRowsChallengeRows;
#[cfg(not(feature = "gxhash"))]
use rustc_hash::FxHasher as SelectedHasher;
pub use temperature::StationStats;
use temperature::{FinalizedStationStats, UpscaledTempValue};
type SelectedBuildHasher = BuildHasherDefault<SelectedHasher>;
const MAX_STATION_NAMES: usize = 10_000;

#[inline(always)]
fn global_alloc_collecting_closure<'map, 'closure, S>(
    stats_per_station: &'map mut HashMap<Box<[u8]>, StationStats, S>,
) -> impl FnMut(&[u8], UpscaledTempValue) + 'closure
where
    S: BuildHasher,
    'map: 'closure,
{
    let collector = move |station_name: &[u8], upscaled_temp_value: UpscaledTempValue| {
        debug_assert!(std::str::from_utf8(station_name).is_ok());
        debug_assert!(!station_name.is_empty());
        debug_assert!(station_name.len() <= 100);
        if let Some(stats) = stats_per_station.get_mut(station_name) {
            stats.update(upscaled_temp_value);
        } else {
            let boxed_station_name = station_name.to_vec().into_boxed_slice();
            temperature::update_stationstats_map(
                stats_per_station,
                boxed_station_name,
                upscaled_temp_value,
            );
        }
    };
    collector
}

#[inline(always)]
fn join_entries<K: Hash + Eq, S: BuildHasher>(
    main_map: &mut HashMap<K, StationStats, S>,
    other: HashMap<K, StationStats, S>,
) {
    for (station, stats) in other {
        if let Some(entry) = main_map.get_mut(&station) {
            entry.join(stats);
        } else {
            let _ = main_map.insert(station, stats);
        }
    }
}

#[inline(always)]
fn write_summary<T: Borrow<[u8]>, W: Write>(
    stations_with_stats: impl Iterator<Item = (T, StationStats)>,
    mut writer: &mut W,
) {
    let mut summary: Vec<(T, FinalizedStationStats)> = stations_with_stats
        .map(|(name, stats)| (name, stats.finalize()))
        .collect();
    summary.sort_unstable_by(|(name1, _), (name2, _)| name1.borrow().cmp(name2.borrow()));
    let total_number_of_stations = summary.len();
    if total_number_of_stations == 0 {
        return;
    };
    let _ = writer.write(&[b'{']);
    let (name_first_station, stats_first_station) = summary.first().unwrap();
    let _ = write!(
        &mut writer,
        "{}={}/{}/{}",
        unsafe { std::str::from_utf8_unchecked(name_first_station.borrow()) },
        stats_first_station.min,
        stats_first_station.avg,
        stats_first_station.max
    );

    for (station_name, stats) in summary.into_iter().skip(1) {
        let _ = write!(
            &mut writer,
            ", {}={}/{}/{}",
            unsafe { std::str::from_utf8_unchecked(station_name.borrow()) },
            stats.min,
            stats.avg,
            stats.max
        );
    }

    let _ = writer.write(&[b'}', b'\n']);
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! make_test_single_thread {
        ($num_rows:literal) => {
            paste::item! {
                #[test]
                fn [< scalar_single_thread_ $num_rows _rows >] () {
                    let input = std::fs::read(format!("measurements_{}.txt", stringify!($num_rows))).unwrap();
                    let input = unsafe {OneBillionRowsChallengeRows::new(&input)};
                    let expected = std::fs::read(format!("expected_output_{}.txt", stringify!($num_rows))).unwrap();
                    let mut result = Vec::new();
                    scalar::solve_challenge(input, &mut result);
                    assert_eq!(
                        String::from_utf8(expected).unwrap(),
                        String::from_utf8(result).unwrap()
                    );
                    }
                }
            paste::item! {
                #[cfg(feature = "arena-allocation")]
                #[test]
                fn [< scalar_single_thread_ arena_allocation_ $num_rows _rows >] () {
                    let input = std::fs::read(format!("measurements_{}.txt", stringify!($num_rows))).unwrap();
                    let input = unsafe {OneBillionRowsChallengeRows::new(&input)};
                    let expected = std::fs::read(format!("expected_output_{}.txt", stringify!($num_rows))).unwrap();
                    let mut result = Vec::new();
                    scalar::solve_challenge_with_arena(input, &mut result);
                    assert_eq!(
                        String::from_utf8(expected).unwrap(),
                        String::from_utf8(result).unwrap()
                    );
                    }
                }
            paste::item! {
                #[cfg(feature = "simd")]
                #[test]
                fn [< simd_single_thread_ $num_rows _rows >] () {
                    let input = std::fs::read(format!("measurements_{}.txt", stringify!($num_rows))).unwrap();
                    let input = unsafe {OneBillionRowsChallengeRows::new(&input)};
                    let expected = std::fs::read(format!("expected_output_{}.txt", stringify!($num_rows))).unwrap();
                    let mut result = Vec::new();
                    simd::solve_challenge(input, &mut result);
                    assert_eq!(
                        String::from_utf8(expected).unwrap(),
                        String::from_utf8(result).unwrap()
                    );
                    }
                }
            };
        }

    macro_rules! make_test_multi_thread {
        ($num_rows:literal, $num_threads:literal) => {
            paste::item! {
                #[test]
                fn [< scalar_ $num_threads _threads_ $num_rows _rows >] () {
                    let input = std::fs::read(format!("measurements_{}.txt", stringify!($num_rows))).unwrap();
                    let input = unsafe {OneBillionRowsChallengeRows::new(&input)};
                    let expected = std::fs::read(format!("expected_output_{}.txt", stringify!($num_rows))).unwrap();
                    let mut result = Vec::new();
                    scalar::solve_challenge_with_threads(input, &mut result, $num_threads);
                    assert_eq!(
                        String::from_utf8(expected).unwrap(),
                        String::from_utf8(result).unwrap()
                    );
                    }
                }
            paste::item! {
                #[cfg(feature = "simd")]
                #[test]
                fn [< simd_ $num_threads _threads_ $num_rows _rows >] () {
                    let input = std::fs::read(format!("measurements_{}.txt", stringify!($num_rows))).unwrap();
                    let input = unsafe {OneBillionRowsChallengeRows::new(&input)};
                    let expected = std::fs::read(format!("expected_output_{}.txt", stringify!($num_rows))).unwrap();
                    let mut result = Vec::new();
                    simd::solve_challenge_with_threads(input, &mut result, $num_threads);
                    assert_eq!(
                        String::from_utf8(expected).unwrap(),
                        String::from_utf8(result).unwrap()
                    );
                    }
                }
            };
        }
    make_test_single_thread!(20);
    make_test_single_thread!(1000);
    make_test_single_thread!(10_000);
    make_test_single_thread!(100_000);

    make_test_multi_thread!(20, 5);
    make_test_multi_thread!(1000, 3);
    make_test_multi_thread!(10_000, 9);
    make_test_multi_thread!(100_000, 16);
}
