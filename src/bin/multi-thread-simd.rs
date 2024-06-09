use memmap::MmapOptions;
use one_billion_rows_rs::OneBillionRowsChallengeRows;
use std::{io::Write, time::Instant};
fn main() {
    let begin = Instant::now();
    let file_name = std::env::args()
        .nth(1)
        .unwrap_or("./measurements.txt".to_string());
    // let input = std::fs::read(&file_name).unwrap();
    let file = std::fs::File::open(file_name).unwrap();
    let mmap = unsafe { MmapOptions::new().map(&file).unwrap() };
    let input = &mmap[..];
    // SAFETY: This is only safe if we trust input to be of the correct format. In a more realistic
    // setting we would use checksums, or sacrifice some performance in the parser to prevent undefined
    // behaviour.
    let input = unsafe { OneBillionRowsChallengeRows::new(input) };
    let num_threads = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let mut writer = std::io::stdout().lock();
    one_billion_rows_rs::simd::solve_challenge_with_threads(input, &mut writer, num_threads);
    let _ = writer.flush();
    let _ = write!(&mut writer, "Took {:?}", begin.elapsed());
    let _ = writer.flush();
}
