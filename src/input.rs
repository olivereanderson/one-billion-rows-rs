use std::{fs::File, path::Path};

use mmap_rs::{Mmap, MmapFlags, MmapOptions, PageSize};

pub struct RowsFile(pub(crate) File);

pub(crate) struct RowsFileMmap(Mmap);

impl RowsFileMmap {
    pub(crate) fn rows(&self) -> OneBillionRowsChallengeRows<'_> {
        debug_assert_eq!(self.0[self.0.len() - 1], b'\n');
        OneBillionRowsChallengeRows(&self.0)
    }
}

impl RowsFile {
    /// This is the only constructor.
    ///
    /// # SAFETY
    ///
    /// By calling this method you acknowledge that you  have checked that this
    /// corresponds to at most one billion rows extracted from the one billion rows
    /// challenge input where input starts with the first byte of a station name
    /// and ends with a newline character `\n`. It is also crucial that the file
    /// is not mutated for the duration of this `RowsFile`.
    pub unsafe fn new<P: AsRef<Path>>(filepath: P) -> Self {
        Self(std::fs::File::open(filepath).unwrap())
    }

    pub(crate) fn mmap(&self) -> RowsFileMmap {
        let size = self.0.metadata().unwrap().len();
        let map_options = MmapOptions::new(size as usize).unwrap();
        let mmap = unsafe {
            map_options
                .with_file(&self.0, 0)
                // .with_page_size(PageSize::_2M)
                .with_flags(MmapFlags::TRANSPARENT_HUGE_PAGES)
                .map()
        };
        //dbg!(MmapOptions::page_sizes());
        //dbg!(&mmap);

        RowsFileMmap(mmap.unwrap())
    }
}

pub struct OneBillionRowsChallengeRows<'input>(pub(crate) &'input [u8]);
impl<'input> OneBillionRowsChallengeRows<'input> {
    /// Returns an iterator of up to `num_chunks` items. These are non-overlapping
    /// subsets of rows that collectively cover `self`.
    ///
    /// NOTE: make sure not to call this with zero chunks as it will panic in that case.
    // We do not bother using NonZeroUsize here as it is an internal method only called in
    // a couple of places
    // We do not bother using NonZeroUsize here as it is an internal method only called in
    // a couple of places.
    pub(crate) fn chunks(
        &self,
        num_chunks: usize,
    ) -> impl Iterator<Item = OneBillionRowsChallengeRows<'input>> {
        let length = self.0.len();
        let chunk_size = length / num_chunks;
        let chunk_size = usize::from(chunk_size == 0) * length + chunk_size;
        struct ChunkIter<'a> {
            input: &'a [u8],
            chunk_size: usize,
        }
        impl<'a> Iterator for ChunkIter<'a> {
            type Item = OneBillionRowsChallengeRows<'a>;
            fn next(&mut self) -> Option<Self::Item> {
                let length_of_rest = self.input.len();
                if length_of_rest == 0 {
                    return None;
                }
                let end = std::cmp::min(self.chunk_size, length_of_rest);
                let take_up_to = end
                    + self.input[end..]
                        .iter()
                        .position(|byte| *byte == b'\n')
                        .map(|idx| idx + 1)
                        .unwrap_or(0);
                let out = &self.input[..take_up_to];
                let next_from = take_up_to;
                self.input = &self.input[next_from..];
                Some(OneBillionRowsChallengeRows(out))
            }
        }

        ChunkIter::<'input> {
            input: self.0,
            chunk_size,
        }
    }
}
