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
        let is_negative = (input[temp_value_from] == b'-') || (input[temp_value_from + 1] == b'-');
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
