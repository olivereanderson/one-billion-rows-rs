use one_billion_rows_rs::RowsFile;
use std::{io::Write, time::Instant};
fn main() {
    let begin = Instant::now();
    let file_name = std::env::args()
        .nth(1)
        .unwrap_or("./measurements.txt".to_string());
    // let input = std::fs::read(&file_name).unwrap();
    let input = unsafe { RowsFile::new(file_name) };
    // SAFETY: This is only safe if we trust input to be of the correct format. In a more realistic
    // setting we would use checksums, or sacrifice some performance in the parser to prevent undefined
    // behaviour.
    let num_threads = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let mut writer = std::io::stdout().lock();
    one_billion_rows_rs::scalar::solve_challenge_with_threads(input, &mut writer, num_threads);
    let _ = writer.flush();
    let _ = write!(&mut writer, "Took {:?}", begin.elapsed());
    let _ = writer.flush();
}
