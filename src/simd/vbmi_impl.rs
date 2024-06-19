//! This module contains solutions aiming to take advantage of simd instructions. It uses the (currently) nightly only std::simd API together with vbmi-based intrinsics.
use super::*;
use std::{
    arch::x86_64::{
        __m256i, __m512i, _mm256_permutex2var_epi8, _mm256_permutexvar_epi8,
        _mm512_permutex2var_epi8,
    },
    simd::{
        cmp::SimdPartialEq,
        num::{SimdInt, SimdUint},
        Mask, Simd,
    },
};

#[inline(always)]
pub(crate) fn collect_stats_simd(
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

        let (
            buffer,
            BufferExtracts {
                number_of_records_found,
                post_decimal_digit_per_record,
                lowest_integral_digit_per_record,
                maybe_highest_digit_per_record,
                maybe_first_char_in_temperature_per_record,
                newline_positions,
                last_newline_position,
            },
        ) = preprocess_buffer(buffer);
        // We can update buffer_start and buffer_end already as we can now easily compute where the last record within our buffer ended
        buffer_start = unsafe {
            buffer_start
                .unchecked_add(last_newline_position as usize)
                .unchecked_add(1)
        };
        buffer_end = unsafe { buffer_start.unchecked_add(BUFFER_SIZE) };
        // We will need to subtract 48 from all parsed digits as the first ascii digit starts at 48.
        // In the case where we know we have a digit we may then instead apply & (u8::MAX >> 4) since
        // 48 = 2**5 + 2**4, hence when we apply & with this nibble mask we obtain the decimal digit.
        const NIBBLE_MASK: u8 = u8::MAX >> 4;
        let nibble_mask_splat = Simd::splat(NIBBLE_MASK);
        let post_decimal_digits = post_decimal_digit_per_record & nibble_mask_splat;
        let lowest_integral_digits = lowest_integral_digit_per_record & nibble_mask_splat;
        let maybe_highest_digits = maybe_highest_digit_per_record;
        let maybe_first_char_in_temperatures = maybe_first_char_in_temperature_per_record;

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

        // TODO: Use cfg here
        let highest_digits_swizzle_indices =
            (maybe_highest_digits & nibble_mask_splat).cast::<u16>();
        const LOOKUP_TABLE: [u16; 32] = const {
            const ONE_HUNDRED: u16 = 100;
            [
                0,
                1 * ONE_HUNDRED,
                2 * ONE_HUNDRED,
                3 * ONE_HUNDRED,
                4 * ONE_HUNDRED,
                5 * ONE_HUNDRED,
                6 * ONE_HUNDRED,
                7 * ONE_HUNDRED,
                8 * ONE_HUNDRED,
                9 * ONE_HUNDRED,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        };
        let highest_digits = {
            let intinsic: std::arch::x86_64::__m512i = highest_digits_swizzle_indices.into();
            let output = unsafe {
                std::arch::x86_64::_mm512_permutexvar_epi16(
                    intinsic,
                    Simd::from_array(LOOKUP_TABLE).into(),
                )
            };
            Simd::<i16, 32>::from(output)
        };

        let lowest_integral_digits: Simd<u8, 32> = {
            const SMALL_LUT: [u8; 32] = const {
                const TEN: u8 = 10;
                [
                    0,
                    TEN,
                    TEN * 2,
                    TEN * 3,
                    TEN * 4,
                    TEN * 5,
                    TEN * 6,
                    TEN * 7,
                    TEN * 8,
                    TEN * 9,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            };
            let indices: __m256i = lowest_integral_digits.into();
            unsafe { _mm256_permutexvar_epi8(indices, Simd::from_array(SMALL_LUT).into()) }.into()
        };
        let abs_values = post_decimal_digits.cast::<i16>()
            + lowest_integral_digits.cast::<i16>()
            + (highest_digits);

        // This negates the values based on the minus mask. The trick for scalar values is described
        // at https://graphics.stanford.edu/~seander/bithacks.html#ConditionalNegate
        let temp_values_times_ten = (abs_values ^ (minus_mask.to_int().cast::<i16>()))
            + minus_mask.cast().select(Simd::splat(1), Simd::splat(0));

        // The entries contain the positions in the buffer containing the last byte of each station name.
        // Note that this only applies to the `number_of_newlines` first entries after that we don't care.
        let station_name_ends: Simd<u8, 32> =
            newline_positions - (temp_value_string_lengths + Simd::splat(2));
        const STATION_START_SWIZZLE_INDICES: [u8; 32] = const {
            [
                0, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
            ]
        };
        debug_assert!(STATION_START_SWIZZLE_INDICES[1..]
            .array_chunks::<2>()
            .all(|chunk| (chunk[1] - chunk[0]) == 1));

        let station_name_starts: Simd<u8, 32> = {
            let a: __m256i = Simd::<u8, 32>::splat(0).into();
            let indices: __m256i = Simd::from_array(STATION_START_SWIZZLE_INDICES).into();
            let b: __m256i = (newline_positions + Simd::splat(1)).into();
            unsafe { _mm256_permutex2var_epi8(a, indices, b) }.into()
        };

        const STATION_NAME_SWIZZLE_INDICES: [u8; 64] = const {
            [
                0, 64, 1, 65, 2, 66, 3, 67, 4, 68, 5, 69, 6, 70, 7, 71, 8, 72, 9, 73, 10, 74, 11,
                75, 12, 76, 13, 77, 14, 78, 15, 79, 16, 80, 17, 81, 18, 82, 19, 83, 20, 84, 21, 85,
                22, 86, 23, 87, 24, 88, 25, 89, 26, 90, 27, 91, 28, 92, 29, 93, 30, 94, 31, 95,
            ]
        };
        debug_assert!(STATION_NAME_SWIZZLE_INDICES
            .array_chunks::<2>()
            .all(|chunk| (chunk[1] - chunk[0]) == 64));

        let station_name_indices: Simd<u8, 64> = {
            let a: __m512i = station_name_starts.resize::<64>(0).into();
            let b: __m512i = (station_name_ends + Simd::splat(1)).resize::<64>(0).into();
            let indices: __m512i = Simd::from_array(STATION_NAME_SWIZZLE_INDICES).into();
            unsafe { _mm512_permutex2var_epi8(a, indices, b) }.into()
        };

        let station_name_indices = station_name_indices.to_array();
        let temp_values_times_ten = temp_values_times_ten.to_array();
        for i in 0..number_of_records_found {
            let station_name_idx = i << 1;
            let (name_start, name_end) = unsafe {
                (
                    *station_name_indices.get_unchecked(station_name_idx as usize),
                    *station_name_indices.get_unchecked(station_name_idx.unchecked_add(1) as usize),
                )
            };
            let temp_value_times_ten = unsafe { *temp_values_times_ten.get_unchecked(i as usize) };
            let station_name =
                unsafe { buffer.get_unchecked((name_start as usize)..(name_end as usize)) };
            collector(station_name, UpscaledTempValue(temp_value_times_ten));
        }
    }
    // Not enough bytes left to do another iteration. We fallback to the scalar version for the remaining bytes.
    if buffer_start < bytes_to_process {
        scalar::collect_stats_scalar(
            OneBillionRowsChallengeRows(&input[buffer_start..]),
            collector,
        )
    }
}

#[inline(always)]
fn preprocess_buffer(buffer: [u8; 192]) -> ([u8; 192], BufferExtracts) {
    use std::{
        arch::x86_64::{
            __m512i, __mmask64, _mm512_maskz_compress_epi8, _mm512_maskz_permutex2var_epi8,
            _mm512_maskz_permutexvar_epi8,
        },
        simd::cmp::SimdOrd,
    };
    // Note that there are min 6 bytes per record hence there can be at most 32 newlines
    // found in a 192 byte buffer (which is the maximum buffer size).
    // We split the buffer into three 64 byte chunks and extract information from each of them.
    // We merge the extracted information at the end before returning.
    let mut post_decimal_digit_per_record: Simd<u8, 64>;
    let mut lowest_integral_digit_per_record: Simd<u8, 64>;
    let mut maybe_highest_digit_per_record: Simd<u8, 64>;
    let mut maybe_first_char_in_temperature_per_record: Simd<u8, 64>;
    let newline_splat: Simd<u8, 64> = Simd::splat(b'\n');
    let mut newline_positions: Simd<u8, 64>;
    let newlines_in_first_chunk: u8;
    let newlines_in_second_chunk: u8;
    let newlines_in_third_chunk: u8;
    let mask_up_to = |i: u8| u64::MAX >> (unsafe { 64u8.unchecked_sub(i) });
    // Locate and insert the positions of newlines within the first 64 bytes using SIMD instructions.
    let first_chunk = {
        let chunk: Simd<u8, 64> = Simd::from_slice(&buffer[..64]);
        let newline_mask = chunk.simd_eq(newline_splat).to_bitmask();
        newlines_in_first_chunk =
            unsafe { newline_mask.count_ones().try_into().unwrap_unchecked() };
        let first_chunk_mask = mask_up_to(newlines_in_first_chunk);
        let indices: Simd<u8, 64> = Simd::from_array(std::array::from_fn(|i| i as u8));
        newline_positions = unsafe {
            let mask: __mmask64 = newline_mask.into();
            let a: __m512i = indices.into();
            _mm512_maskz_compress_epi8(mask, a).into()
        };
        post_decimal_digit_per_record = unsafe {
            let a: __m512i = chunk.into();
            let indices: __m512i = (newline_positions - Simd::splat(1)).into();
            _mm512_maskz_permutexvar_epi8(first_chunk_mask, indices, a).into()
        };
        lowest_integral_digit_per_record = unsafe {
            let a: __m512i = chunk.into();
            let indices: __m512i = (newline_positions - Simd::splat(3)).into();
            _mm512_maskz_permutexvar_epi8(first_chunk_mask, indices, a).into()
        };
        maybe_highest_digit_per_record = unsafe {
            let a: __m512i = chunk.into();
            let indices: __m512i = (newline_positions - Simd::splat(4)).into();
            _mm512_maskz_permutexvar_epi8(first_chunk_mask, indices, a).into()
        };
        maybe_first_char_in_temperature_per_record = unsafe {
            let a: __m512i = chunk.into();
            let indices: __m512i = (newline_positions - Simd::splat(5)).into();
            _mm512_maskz_permutexvar_epi8(first_chunk_mask, indices, a).into()
        };
        chunk
    };
    // Now process the next 64 bytes of the buffer. The code is quite similar to the previous block,
    // but there are some small changes. We might look into moving these blocks into one or more dedicated functions in the future :)
    let (second_chunk, maybe_offset_last_newline_in_second_chunk) = {
        let chunk: Simd<u8, 64> = Simd::from_slice(&buffer[64..128]);
        let first_chunk: __m512i = first_chunk.into();
        let newline_mask = chunk.simd_eq(newline_splat).to_bitmask();
        let first_newline_in_chunk_position: u8 =
            unsafe { u8::try_from(newline_mask.trailing_zeros()).unwrap_unchecked() };
        newlines_in_second_chunk =
            unsafe { newline_mask.count_ones().try_into().unwrap_unchecked() };
        let maybe_last_newline_in_chunk_position: u8 = {
            unsafe {
                63u8.unchecked_sub(u8::try_from(newline_mask.leading_zeros()).unwrap_unchecked())
            }
        };

        let second_chunk_mask = mask_up_to(newlines_in_second_chunk);
        let newline_positions_in_second_chunk: Simd<u8, 64> = {
            let indices: Simd<u8, 64> = Simd::from_array(std::array::from_fn(|i| i as u8));
            unsafe {
                let mask: __mmask64 = newline_mask.into();
                let a: __m512i = indices.into();
                _mm512_maskz_compress_epi8(mask, a).into()
            }
        };

        let permutex_indices = |i| {
            Simd::simd_min(
                newline_positions_in_second_chunk - Simd::splat(i),
                Simd::splat(unsafe {
                    128u8
                        .unchecked_sub(i)
                        .unchecked_add(first_newline_in_chunk_position)
                }),
            )
        };
        let post_decimal_digit_per_record_in_chunk: Simd<u8, 64> = unsafe {
            let a: __m512i = chunk.into();
            let indices: __m512i = permutex_indices(1).into();
            _mm512_maskz_permutex2var_epi8(second_chunk_mask, a, indices, first_chunk).into()
        };
        // We found at most 10 newlines in the previous chunk hence this does not lead to data loss.
        post_decimal_digit_per_record = post_decimal_digit_per_record
            + post_decimal_digit_per_record_in_chunk.rotate_elements_right::<10>();

        let lowest_integral_digit_per_record_in_chunk: Simd<u8, 64> = unsafe {
            let a: __m512i = chunk.into();
            let indices: __m512i = permutex_indices(3).into();
            _mm512_maskz_permutex2var_epi8(second_chunk_mask, a, indices, first_chunk).into()
        };
        // We found at most 10 newlines in the previous chunk hence this does not lead to data loss.
        lowest_integral_digit_per_record = lowest_integral_digit_per_record
            + lowest_integral_digit_per_record_in_chunk.rotate_elements_right::<10>();

        let maybe_highest_digit_per_record_in_chunk: Simd<u8, 64> = unsafe {
            let a: __m512i = chunk.into();
            let indices: __m512i = permutex_indices(4).into();
            _mm512_maskz_permutex2var_epi8(second_chunk_mask, a, indices, first_chunk).into()
        };
        // We found at most 10 newlines in the previous chunk hence this does not lead to data loss.
        maybe_highest_digit_per_record = maybe_highest_digit_per_record
            + maybe_highest_digit_per_record_in_chunk.rotate_elements_right::<10>();

        let maybe_first_char_in_temperature_per_record_in_chunk: Simd<u8, 64> = unsafe {
            let a: __m512i = chunk.into();
            let indices: __m512i = permutex_indices(5).into();
            _mm512_maskz_permutex2var_epi8(second_chunk_mask, a, indices, first_chunk).into()
        };
        // We found at most 10 newlines in the previous chunk hence this does not lead to data loss.
        maybe_first_char_in_temperature_per_record = maybe_first_char_in_temperature_per_record
            + maybe_first_char_in_temperature_per_record_in_chunk.rotate_elements_right::<10>();

        // We found at most 10 newlines in the previous chunk hence this does not lead to data loss.
        let newline_positions_offset =
            Mask::from_bitmask(second_chunk_mask).select(Simd::splat(64), Simd::splat(0));
        let newline_positions_in_second_chunk =
            newline_positions_in_second_chunk + newline_positions_offset;
        newline_positions =
            newline_positions + (newline_positions_in_second_chunk.rotate_elements_right::<10>());

        (chunk, maybe_last_newline_in_chunk_position)
    };
    // We process the last 64 bytes of the 192 byte buffer. The code is very similar to the previous block and we might consider extracting the logic
    // into its own function.
    let (maybe_last_newline_position_in_third_chunk, is_newline_in_chunk_position) = {
        let chunk: Simd<u8, 64> = Simd::from_slice(&buffer[128..192]);
        let newline_mask = chunk.simd_eq(newline_splat).to_bitmask();
        newlines_in_third_chunk =
            unsafe { newline_mask.count_ones().try_into().unwrap_unchecked() };
        let first_newline_in_chunk_position =
            unsafe { u8::try_from(newline_mask.trailing_zeros()).unwrap_unchecked() };
        let maybe_last_newline_position_in_chunk = unsafe {
            63u8.unchecked_sub(u8::try_from(newline_mask.leading_zeros()).unwrap_unchecked())
        };
        let third_chunk_mask = mask_up_to(newlines_in_third_chunk);
        let newline_positions_in_third_chunk: Simd<u8, 64> = {
            let indices: Simd<u8, 64> = Simd::from_array(std::array::from_fn(|i| i as u8));
            unsafe {
                let mask: __mmask64 = newline_mask.into();
                let a: __m512i = indices.into();
                _mm512_maskz_compress_epi8(mask, a).into()
            }
        };

        let second_chunk: __m512i = second_chunk.into();
        let permutex_indices = |i| {
            Simd::simd_min(
                newline_positions_in_third_chunk - Simd::splat(i),
                Simd::splat(unsafe {
                    128u8
                        .unchecked_sub(i)
                        .unchecked_add(first_newline_in_chunk_position)
                }),
            )
        };
        let post_decimal_digit_per_record_in_chunk: Simd<u8, 64> = unsafe {
            let a: __m512i = chunk.into();
            let indices: __m512i = permutex_indices(1).into();
            _mm512_maskz_permutex2var_epi8(third_chunk_mask, a, indices, second_chunk).into()
        };
        // We found at most 21 newlines in the previous chunks hence this does not lead to data loss.
        post_decimal_digit_per_record = post_decimal_digit_per_record
            + post_decimal_digit_per_record_in_chunk.rotate_elements_right::<21>();

        let lowest_integral_digit_per_record_in_chunk: Simd<u8, 64> = unsafe {
            let a: __m512i = chunk.into();
            let indices: __m512i = permutex_indices(3).into();
            _mm512_maskz_permutex2var_epi8(third_chunk_mask, a, indices, second_chunk).into()
        };
        // We found at most 21 newlines in the previous two chunks hence this does not lead to data loss.
        lowest_integral_digit_per_record = lowest_integral_digit_per_record
            + lowest_integral_digit_per_record_in_chunk.rotate_elements_right::<21>();

        let maybe_highest_digit_per_record_in_chunk: Simd<u8, 64> = unsafe {
            let a: __m512i = chunk.into();
            let indices: __m512i = permutex_indices(4).into();
            _mm512_maskz_permutex2var_epi8(third_chunk_mask, a, indices, second_chunk).into()
        };
        // We found at most 21 newlines in the previous chunks hence this does not lead to data loss.
        maybe_highest_digit_per_record = maybe_highest_digit_per_record
            + maybe_highest_digit_per_record_in_chunk.rotate_elements_right::<21>();

        let maybe_first_char_in_temperature_per_record_in_chunk: Simd<u8, 64> = unsafe {
            let a: __m512i = chunk.into();
            let indices: __m512i = permutex_indices(5).into();
            _mm512_maskz_permutex2var_epi8(third_chunk_mask, a, indices, second_chunk).into()
        };
        // We found at most 21 newlines in the previous two chunks hence this does not lead to data loss.
        maybe_first_char_in_temperature_per_record = maybe_first_char_in_temperature_per_record
            + maybe_first_char_in_temperature_per_record_in_chunk.rotate_elements_right::<21>();

        // We found at most 21 newlines in the previous chunks hence this does not lead to data loss.
        let newline_positions_offset =
            Mask::from_bitmask(third_chunk_mask).select(Simd::splat(128), Simd::splat(0));
        let newline_positions_in_third_chunk =
            newline_positions_in_third_chunk + newline_positions_offset;
        newline_positions =
            newline_positions + (newline_positions_in_third_chunk.rotate_elements_right::<21>());
        (
            maybe_last_newline_position_in_chunk,
            newlines_in_third_chunk != 0,
        )
    };
    // We are almost done, we just need to make sure our collected data is stored contiguously
    let mask = {
        let first_chunk_mask = mask_up_to(newlines_in_first_chunk);
        let second_chunk_mask_stark = 10;
        let second_chunk_mask_end = 10 + newlines_in_second_chunk;
        let second_chunk_mask =
            mask_up_to(second_chunk_mask_end) & (!mask_up_to(second_chunk_mask_stark));
        let third_chunk_mask_start = 21;
        let third_chunk_mask_end = 21 + newlines_in_third_chunk;
        let third_chunk_mask =
            mask_up_to(third_chunk_mask_end) & (!mask_up_to(third_chunk_mask_start));
        first_chunk_mask | second_chunk_mask | third_chunk_mask
    };
    let compress = move |input: Simd<u8, 64>| -> Simd<u8, 32> {
        let mask: __mmask64 = mask.into();
        let mask_compress: Simd<u8, 64> = unsafe {
            let a: __m512i = input.into();
            _mm512_maskz_compress_epi8(mask, a).into()
        };
        mask_compress.resize(0)
    };

    let post_decimal_digit_per_record = compress(post_decimal_digit_per_record);
    let lowest_integral_digit_per_record = compress(lowest_integral_digit_per_record);
    let maybe_highest_digit_per_record = compress(maybe_highest_digit_per_record);
    let maybe_first_char_in_temperature_per_record =
        compress(maybe_first_char_in_temperature_per_record);
    let newline_positions = compress(newline_positions);
    let number_of_records_found = unsafe {
        newlines_in_first_chunk
            .unchecked_add(newlines_in_second_chunk)
            .unchecked_add(newlines_in_third_chunk)
    };
    // Compute the last_newline_position. Note that since rows have a maximum of 107 bytes we must have found a newline
    // within the two last chunks.
    //
    // Branchless conditional negation taken from https://graphics.stanford.edu/~seander/bithacks.html#ConditionalNegate
    let is_newline_in_last_chunk = is_newline_in_chunk_position;
    let maybe_offset_last_newline_in_second_chunk =
        unsafe { i8::try_from(maybe_offset_last_newline_in_second_chunk).unwrap_unchecked() };
    let maybe_offset_last_newline_in_third_chunk_plus_64 = unsafe {
        i8::try_from(maybe_last_newline_position_in_third_chunk)
            .unwrap_unchecked()
            .unchecked_add(64)
    };

    let conditional_negation = |i: i8, condition: bool| -> i8 {
        unsafe { (i ^ -i8::from(condition)).unchecked_add(i8::from(condition)) }
    };
    let last_newline_position = unsafe {
        let a: u8 = 64i8
            .unchecked_add(
                maybe_offset_last_newline_in_second_chunk.unchecked_add(conditional_negation(
                    maybe_offset_last_newline_in_second_chunk,
                    is_newline_in_last_chunk,
                )) >> 1,
            )
            .try_into()
            .unwrap_unchecked();
        let b: u8 = {
            ((maybe_offset_last_newline_in_third_chunk_plus_64 as i16).unchecked_add(
                conditional_negation(
                    maybe_offset_last_newline_in_third_chunk_plus_64,
                    !is_newline_in_last_chunk,
                ) as i16,
            ) >> 1)
                .try_into()
                .unwrap_unchecked()
        };
        a.unchecked_add(b)
    };

    debug_assert_eq!(
        last_newline_position,
        newline_positions[(number_of_records_found - 1) as usize]
    );
    (
        buffer,
        BufferExtracts {
            number_of_records_found,
            post_decimal_digit_per_record,
            lowest_integral_digit_per_record,
            maybe_highest_digit_per_record,
            maybe_first_char_in_temperature_per_record,
            newline_positions,
            last_newline_position,
        },
    )
}
#[derive(Debug)]
struct BufferExtracts {
    /// The number of records within the given buffer. This number is guaranteed
    /// to be <= 32. All entries in the arrays in this struct are only valid up to (but
    /// not including) this index.
    number_of_records_found: u8,
    /// The digit after the `.` in the temperature per record in the buffer.
    post_decimal_digit_per_record: Simd<u8, 32>,
    /// The digit directly preceding the `.` in the temperature per record in the buffer.
    lowest_integral_digit_per_record: Simd<u8, 32>,
    /// The byte two steps to the left of the `.` in the temperature per record in the buffer.
    /// This may be a digit, a minus symbol, or a semicolon.
    maybe_highest_digit_per_record: Simd<u8, 32>,
    /// The first byte where a temperature value can start from per record in the buffer.
    maybe_first_char_in_temperature_per_record: Simd<u8, 32>,
    /// The positions of newlines within the given buffer
    newline_positions: Simd<u8, 32>,
    /// The position of the last newline in the buffe
    /// The position of the last newline in the buffer
    last_newline_position: u8,
}
