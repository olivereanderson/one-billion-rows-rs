use std::{
    collections::HashMap,
    hash::{BuildHasher, Hash},
    io::Write,
};

/// A temperature value multiplied by 10.
#[repr(transparent)]
pub(crate) struct UpscaledTempValue(pub(crate) i16);

/// The temperature stats of a weather station.
// NOTE: `min_times_ten` and `max_times_ten` could be placed in
// a UpscaledTempValue, but we save some typing if we don't.
#[derive(Copy, Clone)]
pub struct StationStats {
    min_times_ten: i16,
    max_times_ten: i16,
    sum_times_ten: i64,
    count: u32,
}

impl StationStats {
    #[inline(always)]
    /// Update the station's stats with a newly discovered value.
    pub(crate) fn update(&mut self, UpscaledTempValue(value_times_ten): UpscaledTempValue) {
        // SAFETY: Since the method is private and only called within `process` we know that the sum
        // cannot possibly overflow i64::MAX because i64::MAX > 999*10**9.
        self.sum_times_ten = unsafe { self.sum_times_ten.unchecked_add(value_times_ten as i64) };
        // Update maximum if necessary
        let current_max = self.max_times_ten;
        self.max_times_ten = std::cmp::max(value_times_ten, current_max);
        // Update minimum if necessary
        let current_min = self.min_times_ten;
        self.min_times_ten = std::cmp::min(value_times_ten, current_min);
        // SAFETY: Since the method is private and only called within `process` we know that the count
        // can never exceed 10**9 < u32::MAX
        self.count = unsafe { self.count.unchecked_add(1) };
    }

    pub(crate) fn join(&mut self, other: Self) {
        let Self {
            min_times_ten,
            max_times_ten,
            sum_times_ten,
            count,
        } = other;
        self.sum_times_ten = unsafe { self.sum_times_ten.unchecked_add(sum_times_ten) };
        self.count = unsafe { self.count.unchecked_add(count) };
        let current_max = self.max_times_ten;
        self.max_times_ten = std::cmp::max(current_max, max_times_ten);
        let current_min = self.min_times_ten;
        self.min_times_ten = std::cmp::min(current_min, min_times_ten);
    }

    #[inline(always)]
    pub(crate) fn finalize(self) -> FinalizedStationStats {
        let div_by_ten_repr = |x: i16| -> RoundedTempValue {
            let is_negative = x < 0;
            let integral_part = x / 10;
            let fractional_part = x - 10 * integral_part;
            let integral_part = (integral_part as i8).unsigned_abs();
            let fractional_part = (fractional_part as i8).unsigned_abs();
            let fractional_part =
                fractional_part + (u8::from(is_negative) * RoundedTempValue::IS_NEGATIVE_BITFLAG);
            RoundedTempValue {
                integral_part,
                fractional_part,
            }
        };
        let min = div_by_ten_repr(self.min_times_ten);
        let max = div_by_ten_repr(self.max_times_ten);
        let avg = {
            let x = self.sum_times_ten;
            let is_positive = x >= 0;
            let x = x.unsigned_abs();
            let y = self.count as u64 * 10;
            let i = x / y;
            let r_1 = (x - (y * i)) * 10;
            let d_1 = r_1 / y;
            let r_2 = (r_1 - d_1 * y) * 10;
            let round_up_positive = (r_2 >= (5 * y)) & is_positive;
            let round_up_negative = (r_2 > (5 * y)) & (!is_positive);
            let i = i as u8;
            let d_1 = d_1 as u8;
            let d_1 = d_1 + u8::from(round_up_positive) + u8::from(round_up_negative);
            let d_1_rounded_up_lt_10 = d_1 < 10;
            let integral_part = i + (d_1 / 10);
            let fractional_part = u8::from(d_1_rounded_up_lt_10) * d_1;
            let is_positive = is_positive | ((integral_part == 0) & (fractional_part == 0));
            let fractional_part =
                fractional_part + (u8::from(!is_positive) * RoundedTempValue::IS_NEGATIVE_BITFLAG);
            RoundedTempValue {
                integral_part,
                fractional_part,
            }
        };
        FinalizedStationStats { min, max, avg }
    }
}

/// Upsert the station temperature into the hashmap. This uses the entry api.
/// If passing an owned key is expensive you may prefer to extract a potential
/// value yourself and update it with `[StationStats::update]`.
#[inline(always)]
pub(crate) fn update_stationstats_map<K: Hash + Eq, S: BuildHasher>(
    map: &mut HashMap<K, StationStats, S>,
    key: K,
    station_temperature: UpscaledTempValue,
) {
    let dummy = StationStats {
        min_times_ten: i16::MAX,
        max_times_ten: i16::MIN,
        sum_times_ten: 0,
        count: 0,
    };
    map.entry(key).or_insert(dummy).update(station_temperature);
}

/// Represents a temperature value with one
/// digit (rounded towards positive infinity).
#[derive(Debug)]
pub(crate) struct RoundedTempValue {
    /// The absolute value of the integral part of the temperature value.
    integral_part: u8,
    /// The fractional part of the temperature value which is between 0 and 9
    /// (but may take on a greater value if a bitflag is set: see below).
    ///
    /// # Important
    /// We use the 4'th bit (2^4) as a niche to indicate whether the rounded
    /// temeprature value is negative.
    fractional_part: u8,
}
impl RoundedTempValue {
    /// If this bit is set in [`Self::fractional_part`] then the entire value of [`Self`]
    /// should be interpreted as a negative number.
    const IS_NEGATIVE_BITFLAG: u8 = 1 << 4;
}
impl std::fmt::Display for RoundedTempValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut buffer = [b'-', 0, 0, 0, 0];
        let is_positive = (self.fractional_part & RoundedTempValue::IS_NEGATIVE_BITFLAG) == 0;
        let fractional_part = self.fractional_part & (!RoundedTempValue::IS_NEGATIVE_BITFLAG);
        let integral_part = self.integral_part;
        let abs_val_ge_10 = integral_part >= 10;
        let bytes_to_write = 3 + u8::from(!is_positive) + u8::from(abs_val_ge_10);
        write!(&mut buffer[1..], "{}.{}", integral_part, fractional_part).unwrap();
        let rounded_temp_val_str: &str = unsafe {
            std::str::from_utf8_unchecked(
                &buffer[(is_positive as usize)..][..(bytes_to_write as usize)],
            )
        };
        f.write_str(rounded_temp_val_str)
    }
}

pub(crate) struct FinalizedStationStats {
    /// The minimum value.
    pub(crate) min: RoundedTempValue,
    /// The maximum.
    pub(crate) max: RoundedTempValue,
    /// The average value.
    pub(crate) avg: RoundedTempValue,
}
