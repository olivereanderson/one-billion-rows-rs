These are my solutions to the One billion rows challenge in Rust.

## Warning

Make sure that the measurement files you hand over to the binaries satisfy the
required format. I ocassionally opt-in to the unsafe superset of Rust based on
the assumption that the expected format is respected.

## TODO
- Clean up the code. It is currently not very readable at all and has loads of uncommented unsafe blocks.
- Make the performance more consistent with the case where the file does not already exist in cache. This can
  be done by reading the file in chunks in a main thread that then dispatches the chunks to worker threads.

## How to run

There are a few versions with different characteristics.

## Scalar style code

- `cargo run --release --bin single-thread-scalar [-- <path to measurements.txt>]`
- `cargo run --release --bin multi-thread-scalar [-- <path to measurements.txt>]`

## SIMD versions

- `cargo run --release --bin single-thread-simd [-- <path to measurements.txt>]`
- `cargo run --release --bin multi-thread-simd [-- <path to measurements.txt>]`
