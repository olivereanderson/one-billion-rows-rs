These are my solutions to the One billion rows challenge in Rust.

## Warning

Make sure that the measurement files you hand over to the binaries satisfy the
required format. I ocassionally opt-in to the unsafe superset of Rust based on
the assumption that the expected format is respected.

## How to run

There are a few versions with different characteristics.

## Scalar style code

- `cargo run --release --bin single-thread-scalar [-- <path to measurements.txt>]`
- `cargo run --release --bin multi-thread-scalar [-- <path to measurements.txt>]`

## SIMD versions

- `cargo run --release --bin single-thread-simd [-- <path to measurements.txt>]`
- `cargo run --release --bin multi-thread-simd [-- <path to measurements.txt>]`
