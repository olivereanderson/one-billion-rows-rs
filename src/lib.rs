#![cfg_attr(feature = "nightly", feature(portable_simd))]
mod input;
mod temperature;
use std::{
    borrow::Borrow,
    collections::HashMap,
    hash::{BuildHasher, BuildHasherDefault, Hash},
    io::Write,
};

pub use input::OneBillionRowsChallengeRows;
// use rustc_hash::FxHasher;
pub use temperature::StationStats;
use temperature::{FinalizedStationStats, UpscaledTempValue};
// type SelectedBuildHasher = BuildHasherDefault<FxHasher>;
type SelectedBuildHasher = BuildHasherDefault<gxhash::GxHasher>;
const MAX_STATION_NAMES: usize = 10_000;

pub mod scalar {
    use super::*;
    /// Solve the challenge with a single thread and scalar style code.
    #[inline(always)]
    pub fn solve_challenge<W: Write>(input: OneBillionRowsChallengeRows<'_>, writer: &mut W) {
        let hasher = SelectedBuildHasher::default();
        let mut stats_per_station: HashMap<Box<[u8]>, StationStats, SelectedBuildHasher> =
            HashMap::with_capacity_and_hasher(MAX_STATION_NAMES, hasher);
        let collector = global_alloc_collecting_closure(&mut stats_per_station);
        collect_stats_scalar(input, collector);
        write_summary(stats_per_station.into_iter(), writer);
    }

    /// Solve the challenge with up to the given number of threads using scalar style code.
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
                        collect_stats_scalar(chunk, collector);
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

    #[inline(always)]
    pub(crate) fn collect_stats_scalar<Collector: FnMut(&[u8], UpscaledTempValue)>(
        OneBillionRowsChallengeRows(input): OneBillionRowsChallengeRows<'_>,
        mut collector: Collector,
    ) {
        const ASCII_DIGIT_ZERO: u8 = 48;
        let mut next_station_from = 0;
        // process the last station first. To avoid index out of bounds problems in the main loop.
        let Some(position_of_semicol_relative_to_end) =
            input.iter().rev().position(|byte| *byte == b';')
        else {
            return;
        };
        // Correct for the fact that we count from 0.
        let num_bytes_after_last_station_name = position_of_semicol_relative_to_end + 1;
        let bytes_to_process = input.len();
        // This is the index of the ";" after the last station name.
        let last_semicol_idx = bytes_to_process - num_bytes_after_last_station_name;
        let last_station_name_starts = input[..last_semicol_idx]
            .iter()
            .rev()
            .position(|byte| *byte == b'\n')
            .map(|position| last_semicol_idx - position)
            .unwrap_or(0);

        let last_station_name = &input[last_station_name_starts..last_semicol_idx];
        let temp_value_from = last_semicol_idx + 1;
        let last_byte_is_newline = usize::from(input[bytes_to_process - 1] == b'\n');
        let temp_value_to = bytes_to_process - last_byte_is_newline;
        let last_temp_value_abs_value_times_ten: i16 = input[temp_value_from..temp_value_to]
            .iter()
            .rev()
            .filter(|byte| (**byte != b'-') & (**byte != b'.'))
            .map(|digit_byte| i16::from(*digit_byte) - (ASCII_DIGIT_ZERO as i16))
            .zip([1, 10, 100])
            .map(|(digit, multiplier)| digit * multiplier)
            .sum();
        let last_temp_is_negative = input[temp_value_from] == b'-';
        let last_temp_value_times_ten = (i16::from(last_temp_is_negative)
            * (-last_temp_value_abs_value_times_ten))
            + ((i16::from(!last_temp_is_negative)) * last_temp_value_abs_value_times_ten);
        collector(
            last_station_name,
            UpscaledTempValue(last_temp_value_times_ten),
        );

        while let Some(idx) = input
            .get(next_station_from..last_station_name_starts)
            .and_then(|slice| slice.iter().position(|byte| *byte == b';'))
        {
            let temp_value_from: usize = next_station_from + idx + 1;
            let is_negative =
                (input[temp_value_from] == b'-') || (input[temp_value_from + 1] == b'-');
            let is_abs_val_gt_ten = (input[temp_value_from + 5] == b'\n')
                || ((input[temp_value_from + 4] == b'\n') & (!is_negative));
            let x = |idx| {
                let val: u8 = input[temp_value_from + idx];
                val.wrapping_sub(ASCII_DIGIT_ZERO) as i16
            };

            let abs_value = {
                x(0) * 100 * (i16::from((!is_negative) & is_abs_val_gt_ten))
                    + x(0) * 10 * (i16::from((!is_abs_val_gt_ten) & (!is_negative)))
                    + x(1) * 100 * (i16::from(is_abs_val_gt_ten & (is_negative)))
                    + x(1) * 10 * (i16::from(is_abs_val_gt_ten & (!is_negative)))
                    + x(1) * 10 * (i16::from((!is_abs_val_gt_ten) & is_negative))
                    + x(2) * 10 * (i16::from(is_abs_val_gt_ten & is_negative))
                    + x(2) * (i16::from((!is_abs_val_gt_ten) & (!is_negative)))
                    + x(3) * (i16::from(is_abs_val_gt_ten & (!is_negative)))
                    + x(3) * (i16::from((!is_abs_val_gt_ten) & is_negative))
                    + x(4) * (i16::from(is_abs_val_gt_ten & is_negative))
            };
            let temperature_value_times_ten =
                -i16::from(is_negative) * abs_value + i16::from(!is_negative) * abs_value;

            let temperature_value_string_length =
                3 + usize::from(is_negative) + usize::from(is_abs_val_gt_ten);

            let station_name = &input[next_station_from..(next_station_from + idx)];
            collector(station_name, UpscaledTempValue(temperature_value_times_ten));
            // TODO: Why + 1?
            next_station_from = temp_value_from + temperature_value_string_length + 1;
        }
    }
}

#[cfg(feature = "simd")]
pub mod simd {
    //! This module contains solutions aiming to take advantage of simd instructions. It uses the (currently) nightly only std::simd API.
    use super::*;
    use std::simd::{
        cmp::SimdPartialEq,
        num::{SimdInt, SimdUint},
        Mask, Simd,
    };
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
    #[inline(always)]
    fn collect_stats_simd(
        OneBillionRowsChallengeRows(input): OneBillionRowsChallengeRows<'_>,
        mut collector: impl FnMut(&[u8], UpscaledTempValue),
    ) {
        const BUFFER_SIZE: usize = 192;
        let mut buffer_start = 0;
        let mut buffer_end = BUFFER_SIZE;
        let bytes_to_process = input.len();
        while buffer_end <= bytes_to_process {
            let buffer: [u8; BUFFER_SIZE] = unsafe {
                (&input[buffer_start..buffer_end])
                    .try_into()
                    .unwrap_unchecked()
            };
            let BufferExtracts {
                number_of_records_found,
                post_decimal_digit_per_record,
                lowest_integral_digit_per_record,
                maybe_highest_digit_per_record,
                maybe_first_char_in_temperature_per_record,
                newline_positions,
            } = {
                let mut post_decimal_digit_per_record = [0u8; 32];
                let mut lowest_integral_digit_per_record = [0u8; 32];
                let mut maybe_highest_digit_per_record = [0u8; 32];
                let mut maybe_first_char_in_temperature_per_record = [0u8; 32];
                let newline_splat: Simd<u8, 64> = Simd::splat(b'\n');
                // Note that there are min 6 bytes per record hence there can be at most 32 newlines
                // found in a 192 byte buffer (which is the maximum buffer size).
                let mut newline_positions = [0u8; 32];
                let mut insert_at_position: usize = 0;
                // Locate and insert the positions of newlines within the first 64 bytes using SIMD instructions.
                {
                    let chunk: Simd<u8, 64> = Simd::from_slice(&buffer[..64]);
                    let newline_bitset = chunk.simd_eq(newline_splat).to_bitmask();
                    newline_bitset.for_each_set_bit(|bit_position| {
                        // SAFETY: The index is always in bounds because we happen to know that there are at most 32 newlines entries
                        // in our buffer.
                        unsafe {
                            // TODO: Can we avoud the assignment in the loop by using trailing zeroes instead?
                            *newline_positions.get_unchecked_mut(insert_at_position) =
                                bit_position as u8;
                            *post_decimal_digit_per_record.get_unchecked_mut(insert_at_position) =
                                *buffer.get_unchecked(bit_position.unchecked_sub(1));
                            *lowest_integral_digit_per_record
                                .get_unchecked_mut(insert_at_position) =
                                *buffer.get_unchecked(bit_position.unchecked_sub(3));
                            *maybe_highest_digit_per_record.get_unchecked_mut(insert_at_position) =
                                *buffer.get_unchecked(bit_position.unchecked_sub(4));
                            *maybe_first_char_in_temperature_per_record
                                .get_unchecked_mut(insert_at_position) =
                                *buffer.get_unchecked(bit_position.unchecked_sub(5));
                            insert_at_position += 1;
                        }
                    });
                }
                // Now process the next 64 bytes of the buffer. The code is quite similar to the previous block,
                // but there are some small changes. We might look into moving these blocks into one or more dedicated functions in the future :)
                {
                    let chunk: Simd<u8, 64> = Simd::from_slice(&buffer[64..128]);
                    let newline_bitset = chunk.simd_eq(newline_splat).to_bitmask();
                    newline_bitset.for_each_set_bit(|bit_position| {
                        let newline_position = unsafe { 64_usize.unchecked_add(bit_position) };

                        // SAFETY: The index is always in bounds becase we happen to know that there are at most 32 newline entries
                        // in our buffer.
                        unsafe {
                            *newline_positions.get_unchecked_mut(insert_at_position) =
                                newline_position as u8;
                            *post_decimal_digit_per_record.get_unchecked_mut(insert_at_position) =
                                *buffer.get_unchecked(newline_position.unchecked_sub(1));
                            *lowest_integral_digit_per_record
                                .get_unchecked_mut(insert_at_position) =
                                *buffer.get_unchecked(newline_position.unchecked_sub(3));
                            *maybe_highest_digit_per_record.get_unchecked_mut(insert_at_position) =
                                *buffer.get_unchecked(newline_position.unchecked_sub(4));
                            *maybe_first_char_in_temperature_per_record
                                .get_unchecked_mut(insert_at_position) =
                                *buffer.get_unchecked(newline_position.unchecked_sub(5));
                        }
                        // TODO: Consider an unchecked add here, but the compiler should see that this cannot overflow.
                        insert_at_position += 1;
                    });
                }
                // We process the last 64 bytes of the 192 byte buffer. The code is very similar to the previous block and we might consider extracting the logic
                // into its own function.
                {
                    let chunk: Simd<u8, 64> = Simd::from_slice(&buffer[128..]);
                    let newline_bitset = chunk.simd_eq(newline_splat).to_bitmask();
                    newline_bitset.for_each_set_bit(|bit_position| {
                        let newline_position = unsafe { 128_usize.unchecked_add(bit_position) };
                        // SAFETY: The index is always in bounds becase we happen to know that there are at most 32 newline entries
                        // in our buffer.
                        unsafe {
                            *newline_positions.get_unchecked_mut(insert_at_position) =
                                newline_position as u8;
                            *post_decimal_digit_per_record.get_unchecked_mut(insert_at_position) =
                                *buffer.get_unchecked(newline_position.unchecked_sub(1));
                            *lowest_integral_digit_per_record
                                .get_unchecked_mut(insert_at_position) =
                                *buffer.get_unchecked(newline_position.unchecked_sub(3));
                            *maybe_highest_digit_per_record.get_unchecked_mut(insert_at_position) =
                                *buffer.get_unchecked(newline_position.unchecked_sub(4));
                            *maybe_first_char_in_temperature_per_record
                                .get_unchecked_mut(insert_at_position) =
                                *buffer.get_unchecked(newline_position.unchecked_sub(5));
                        }
                        // TODO: Consider an unchecked add here, but the compiler should see that this cannot overflow.
                        insert_at_position += 1;
                    });
                }
                // Use a binding here to simplify debuging if needed

                BufferExtracts {
                    number_of_records_found: unsafe {
                        u8::try_from(insert_at_position).unwrap_unchecked()
                    },
                    post_decimal_digit_per_record,
                    lowest_integral_digit_per_record,
                    maybe_highest_digit_per_record,
                    maybe_first_char_in_temperature_per_record,
                    newline_positions,
                }
            };
            // We can update buffer_start and buffer_end already as we can now easily compute where the last record within our buffer ended
            buffer_start = unsafe {
                buffer_start
                    .unchecked_add(
                        *newline_positions
                            .get_unchecked(number_of_records_found.unchecked_sub(1) as usize)
                            as usize,
                    )
                    .unchecked_add(1)
            };
            buffer_end = unsafe { buffer_start.unchecked_add(BUFFER_SIZE) };
            // We will need to subtract this from all parsed digits as the first ascii digit starts at 48.
            const ASCII_DIGIT_ZERO: u8 = 48;
            let post_decimal_digits = Simd::<u8, 32>::from_array(post_decimal_digit_per_record)
                .saturating_sub(Simd::splat(ASCII_DIGIT_ZERO));
            let lowest_integral_digits =
                Simd::<u8, 32>::from_array(lowest_integral_digit_per_record)
                    .saturating_sub(Simd::splat(ASCII_DIGIT_ZERO));
            let maybe_highest_digits = Simd::<u8, 32>::from_array(maybe_highest_digit_per_record);
            let maybe_first_char_in_temperatures =
                Simd::<u8, 32>::from_array(maybe_first_char_in_temperature_per_record);

            let minus_splat: Simd<u8, 32> = Simd::splat(b'-');
            let minus_mask_maybe_highest_digits = maybe_highest_digits.simd_eq(minus_splat);
            let semicol_mask_maybe_highest_digits = maybe_highest_digits.simd_eq(Simd::splat(b';'));
            let is_digit_mask_maybe_highest_digits =
                !(minus_mask_maybe_highest_digits | semicol_mask_maybe_highest_digits);
            let minus_mask = minus_mask_maybe_highest_digits
                | (minus_splat.simd_eq(maybe_first_char_in_temperatures));

            // Make sure not to interpret the number to be negative in the technically possible scenario
            // where the station name ends with a "-"
            let minus_mask = minus_mask & (!semicol_mask_maybe_highest_digits);

            let temp_value_string_lengths: Simd<i8, 32> =
                Simd::splat(3) - is_digit_mask_maybe_highest_digits.to_int() - minus_mask.to_int();
            let temp_value_string_lengths: Simd<u8, 32> = temp_value_string_lengths.cast::<u8>();

            // TODO: Can we use an "unsafe" sub instead of a saturating one here? It should be OK.
            let highest_digits = is_digit_mask_maybe_highest_digits
                .select(maybe_highest_digits, Simd::splat(ASCII_DIGIT_ZERO))
                .saturating_sub(Simd::splat(ASCII_DIGIT_ZERO));
            let sign: Simd<i8, 32> = minus_mask.select(Simd::splat(-1), Simd::splat(1));
            let temp_values_times_ten = sign.cast::<i16>()
                * (post_decimal_digits.cast::<i16>()
                    + (Simd::splat(10) * lowest_integral_digits.cast::<i16>())
                    + (Simd::splat(100) * highest_digits.cast::<i16>()));

            let newline_positions = Simd::from_array(newline_positions);
            // The entries contain the positions in the buffer containing the last byte of each station name.
            // Note that this only applies to the `number_of_newlines` first entries after that we don't care.
            let station_name_ends: Simd<u8, 32> =
                newline_positions.saturating_sub(temp_value_string_lengths + Simd::splat(2));
            // Compute station name lengths
            let station_name_lengths = {
                let almost_correct = station_name_ends
                    .saturating_sub(newline_positions.rotate_elements_right::<1>());
                // TODO: Try mutating the first element directly. It might be faster.
                Mask::<_, 32>::from_bitmask(1)
                    .select(station_name_ends + Simd::splat(1), almost_correct)
            };
            let relevant_entries_mask =
                !Mask::<i8, 32>::from_bitmask(u64::MAX << (number_of_records_found));

            let station_name_ends: [u8; 32] = station_name_ends.to_array();
            let station_name_lengths: [u8; 32] = station_name_lengths.to_array();
            let temp_values_times_ten: [i16; 32] = temp_values_times_ten.to_array();

            let lookup_metadata = |idx: usize| {
                let temperature_value = unsafe { *temp_values_times_ten.get_unchecked(idx) };
                let station_name_ends_at = *unsafe { station_name_ends.get_unchecked(idx) };
                let station_name_length = *unsafe { station_name_lengths.get_unchecked(idx) };
                let station_name_starts_from = unsafe {
                    station_name_ends_at.unchecked_sub(station_name_length.unchecked_sub(1))
                };
                (
                    temperature_value,
                    station_name_starts_from as usize,
                    station_name_length as usize,
                )
            };

            relevant_entries_mask.to_bitmask().for_each_set_bit(|idx| {
                let (temp_value, name_starts_from, name_length) = lookup_metadata(idx);
                let station_name = &buffer[name_starts_from..][..name_length];
                collector(station_name, UpscaledTempValue(temp_value));
            });
        }
        // Not enough bytes left to do another iteration. We fallback to the scalar version for the remaining bytes.
        if buffer_start < bytes_to_process {
            scalar::collect_stats_scalar(
                OneBillionRowsChallengeRows(&input[buffer_start..]),
                collector,
            )
        }
    }

    #[derive(Debug)]
    struct BufferExtracts {
        /// The number of records within the given buffer. This number is guaranteed
        /// to be <= 32. All entries in the arrays in this struct are only valid up to (but
        /// not including) this index.
        number_of_records_found: u8,
        /// The digit after the `.` in the temperature per record in the buffer.
        post_decimal_digit_per_record: [u8; 32],
        /// The digit directly preceding the `.` in the temperature per record in the buffer.
        lowest_integral_digit_per_record: [u8; 32],
        /// The byte two steps to the left of the `.` in the temperature per record in the buffer.
        /// This may be a digit, a minus symbol, or a semicolon.
        maybe_highest_digit_per_record: [u8; 32],
        /// The first byte where a temperature value can start from per record in the buffer.
        maybe_first_char_in_temperature_per_record: [u8; 32],
        /// The positions of newlines within the given buffer
        newline_positions: [u8; 32],
    }

    /// Use this for internal iteration over the bits set in a u64.
    ///
    /// We could alternatively have used external iterators provided
    /// by the bitvec crate.
    trait ForEachSetBitExt: Sized {
        fn for_each_set_bit<F: FnMut(usize)>(self, callback: F);
    }

    impl ForEachSetBitExt for u64 {
        #[inline(always)]
        fn for_each_set_bit<F: FnMut(usize)>(self, mut callback: F) {
            let mut bitmask = self;
            while bitmask != 0 {
                let idx = bitmask.trailing_zeros() as usize;
                callback(idx);
                let least_significant_bit = bitmask & bitmask.wrapping_neg();
                bitmask ^= least_significant_bit;
            }
        }
    }
}

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
