use super::terms_store::{read_length_prefixed_string, write_length_prefixed_string};
use anyhow::{Context, Result, anyhow, ensure};
use bytemuck::TransparentWrapper;
use dsi_bitstream::prelude::*;
use dsi_progress_logger::{ProgressLog, no_logging};
use itertools::Itertools;
use mmap_rs::{MmapFlags, MmapOptions};
use rayon::prelude::*;
use rdst::{RadixKey, RadixSort};
use rustc_hash::{FxBuildHasher, FxHashSet};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Cursor, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use sux::bits::BitFieldVec;
use tempfile::TempDir;
use value_traits::slices::SliceByValue;
use webgraph::utils::{ArcMmapHelper, MmapHelper};

/// Sorts and deduplicates strings and spills to disk to save memory
///
/// Calls to [`Self::push_boxed_bytes`] add the given bytestring to an in-memory buffer
/// (a `HashSet`) to perform deduplication as soon as possible.
///
/// Once that `HashSet` reaches the configured threshold (in number of bytes of data,
/// not of the `HashSet` itself, which usually has ~1× overhead), that buffer is sorted
/// and written to disk to a single file.
///
/// Once the number of written files reaches the other configured threshold, all files
/// are read, merged together in a single new file, and then deleted.
/// Higher thresholds reduce the number of writes (as merges form a k-ary tree
/// with k=that threshold), but require more RAM (k * Rust's DEFAULT_BUF_SIZE * number of threads)
pub(super) struct ExternalDeduplicatingStringSorter {
    tempdirs: Vec<TempDir>,
    sorted_files: Vec<File>,
    max_buffer_size: usize,
    max_num_files: usize,
    buffer_size: usize,
    buffer: FxHashSet<Box<[u8]>>,
    written_size: usize,
    num_unique_items_upperbound: usize, // only counting those in files
}

impl ExternalDeduplicatingStringSorter {
    pub fn new(max_buffer_size: usize, max_num_files: usize) -> Self {
        Self {
            tempdirs: Vec::new(), // Create it only if needed
            sorted_files: Vec::new(),
            buffer: FxHashSet::with_hasher(FxBuildHasher),
            buffer_size: 0,
            max_buffer_size,
            max_num_files,
            written_size: 0,
            num_unique_items_upperbound: 0,
        }
    }

    /// only counting those in files
    pub fn num_unique_items_upperbound(&self) -> usize {
        self.num_unique_items_upperbound
    }

    pub fn push_boxed_bytes(&mut self, bytes: Box<[u8]>) -> Result<()> {
        let bytes_len = size_of::<usize>() + bytes.len();

        if self.buffer_size + bytes.len() > self.max_buffer_size {
            self.flush_buffer()?;
        }
        ensure!(
            bytes_len < self.max_buffer_size,
            "String does not fit in {} bytes buffer",
            self.max_buffer_size
        );

        if self.buffer.insert(bytes) {
            self.buffer_size += bytes_len;
        }
        Ok(())
    }

    fn flush_buffer(&mut self) -> Result<()> {
        // Sort buffer
        let mut sort_buffer: Vec<_> = self.buffer.drain().collect();
        sort_buffer.par_sort_unstable();
        sort_buffer.dedup();

        // Flush buffer to file
        self.write_sorted_items(sort_buffer.into_iter().map(Ok))
            .context("Could not write sorted buffer to disk")?;

        // Reset buffer
        self.written_size += self.buffer_size;
        self.buffer_size = 0;

        if self.sorted_files.len() > self.max_num_files {
            self.compact_files().context("Could not compact files")?;
        }

        Ok(())
    }

    fn write_sorted_items(
        &mut self,
        sorted_items: impl Iterator<Item = Result<Box<[u8]>>>,
    ) -> Result<()> {
        if self.tempdirs.is_empty() {
            self.tempdirs
                .push(TempDir::new().context("Could not create temporary directory")?);
        }
        let path = self
            .tempdirs
            .first()
            .unwrap()
            .path()
            .join(format!("{}", self.sorted_files.len()));
        let mut file = File::create_new(&path)
            .with_context(|| format!("Could not create {}", path.display()))?;

        {
            let mut writer = BufWriter::new(&mut file);
            for string in sorted_items {
                let string = string.context("Could not read input item")?;

                write_length_prefixed_string(&mut writer, &string, &path)?;

                self.num_unique_items_upperbound += 1;
            }
        }
        file.flush()
            .with_context(|| format!("Could not flush {}", path.display()))?;

        // Make file readable
        file.rewind()
            .with_context(|| format!("Could not rewind {}", path.display()))?;
        self.sorted_files.push(file);
        Ok(())
    }

    /// Merges every file of this sorter currently on disk into a single one
    pub fn compact_files(&mut self) -> Result<()> {
        // FIXME: the current implementation causes 1 large file and 9 small files that get merged
        // into the larger one. This is incredibly inefficient because we keep compacting the same
        // items over and over (it's quadratic). We should do some sort of merge tree instead.
        let mut sorted_files = Vec::new();
        std::mem::swap(&mut sorted_files, &mut self.sorted_files);
        self.num_unique_items_upperbound = 0; // self.write_sorted_items() will re-count them after dedup

        self.tempdirs.clear();

        let merged_items =
            Self::iter_written_boxed_bytes(sorted_files).context("Could not read merged items")?;
        self.write_sorted_items(merged_items)
            .context("Could not write merged items")?;

        Ok(())
    }

    fn drain_written_boxed_bytes(&mut self) -> Result<impl Iterator<Item = Result<Box<[u8]>>>> {
        let mut sorted_files = Vec::new();
        std::mem::swap(&mut sorted_files, &mut self.sorted_files);
        Self::iter_written_boxed_bytes(sorted_files)
    }

    fn iter_written_boxed_bytes(
        sorted_files: Vec<File>,
    ) -> Result<impl Iterator<Item = Result<Box<[u8]>>>> {
        #[expect(clippy::match_same_arms)]
        Ok(sorted_files
            .into_iter()
            .map(|file| {
                // Don't use BufReader here, even though it's tempting to simplify this code.
                // Some strings from Wikidata are very long (megabytes) and keep the BufReader's
                // internal buffer capacity pretty large, wasting memory for the entire duration
                // of the file read.
                // This ends up using num_threads * max_num_files * ~10MB of RAM, which can be
                // significant.
                //
                // Using 'file' directly instead of mmapping works too, but wastes a significant
                // amount of time in syscall overhead because of the small reads. (It doubles the
                // *overall* time of the merging phase on NVMe.)
                let file_len =
                    usize::try_from(file.metadata().context("Could not stat SST file")?.len())
                        .context("file size overflowed usize")?;
                let mut reader = Cursor::new(unsafe {
                    MmapOptions::new(file_len)
                        .context("Could not initialize mmap")?
                        .with_flags(MmapFlags::SEQUENTIAL)
                        .with_file(&file, 0)
                        .map()
                        .context("Could not mmap SST")?
                });
                let get_position =
                    |file: &mut Cursor<_>| Some(file.stream_position().map_err(Into::into));
                Ok(
                    std::iter::repeat(()).map_while(move |()| -> Option<Result<_>> {
                        read_length_prefixed_string(&mut reader, get_position).transpose()
                    }),
                )
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .kmerge_by(|left, right| match (left, right) {
                (Ok(left), Ok(right)) => left < right,
                (Err(_), Ok(_)) => true,  // return error first
                (Ok(_), Err(_)) => false, // return error first
                (_, _) => true,           // doesn't matter
            })
            .dedup_by(|left, right| match (left, right) {
                (Ok(left), Ok(right)) => left == right,
                (Err(_), Ok(_)) => true,  // return error first
                (Ok(_), Err(_)) => false, // return error first
                (_, _) => true,           // doesn't matter
            }))
    }

    /// Consome this external sorter and returns sorted items
    pub fn drain_boxed_bytes(&mut self) -> Result<impl Iterator<Item = Result<Box<[u8]>>>> {
        let mut buffer = HashSet::default();
        std::mem::swap(&mut buffer, &mut self.buffer);
        let mut buffer: Vec<_> = buffer.into_iter().collect();
        buffer.par_sort_unstable();
        Ok(self.drain_written_boxed_bytes()?
            // TODO: merge at the same time as the others
            .merge_by(buffer.into_iter().map(Ok),|left, right| match (left, right) {
                (Ok(left), Ok(right)) => left < right,
                (_, _) => true, // doesn't matter, we are going to error anyway
            })
            .dedup_by(|left, right| match (left, right) {
                (Ok(left), Ok(right)) => left == right,
                (_, _) => true, // doesn't matter, we are going to error anyway
            }))
    }

    pub fn merge(mut self, mut other: Self) -> Result<Self> {
        if self.sorted_files.len() + other.sorted_files.len() > self.max_num_files {
            let (l, r) = rayon::join(
                || self.compact_files().context("Could not compact files"),
                || other.compact_files().context("Could not compact files"),
            );
            l?;
            r?;
        }

        self.tempdirs.extend(other.tempdirs);
        self.sorted_files.extend(other.sorted_files);
        self.written_size += other.written_size;
        self.num_unique_items_upperbound += other.num_unique_items_upperbound;

        // merge smallest buffer into largest
        if self.buffer.len() < other.buffer.len() {
            (self.buffer, other.buffer) = (other.buffer, self.buffer);
            (self.buffer_size, other.buffer_size) = (other.buffer_size, self.buffer_size);
        }
        for bytes in other.buffer {
            self.push_boxed_bytes(bytes)?;
        }

        Ok(self)
    }
}

/// Sorts arrays and spills to disk to save memory
///
/// May error with ENOMEM if sysctl setting `vm.max_map_count` is too low.
pub(super) struct ExternalArraySorter<const N: usize> {
    tempdirs: Vec<TempDir>,
    num_files_in_first_tempdir: usize,
    sorted_files: Vec<Vec<PathBuf>>, // one vec for each partition
    max_buffer_len: usize,
    current_buffer_len: usize,
    buffers: Vec<BitFieldVec<usize>>,
    num_partitions: usize,
    max_value: usize,
    max_num_files_per_partition: usize,
}

impl<const N: usize> ExternalArraySorter<N> {
    /// At the end you need to process each partition in parallel, then it's good
    /// to have at least as many partitions as threads. Multiply that by
    /// a small number if partitions are uneven, so a single thread doesn't end up alone
    /// at the end writing a huge partition while other threads are already done.
    ///
    /// More partitions also save time on merging.
    ///
    /// On the other hand, more partitions cause sorted quads to be split into
    /// more (and smaller) files and as many mmaps.
    pub fn new(max_value: usize, max_buffer_size: usize, num_partitions: usize) -> Result<Self> {
        let bit_width = usize::try_from(usize::BITS - max_value.leading_zeros())
            .context("Weird pointer size")?;

        Ok(Self {
            tempdirs: Vec::new(), // Create it only if needed
            num_files_in_first_tempdir: 0,
            sorted_files: (0..num_partitions).map(|_| Vec::new()).collect(),
            buffers: (0..num_partitions)
                .map(|_| BitFieldVec::new(bit_width, 0))
                .collect(),
            max_buffer_len: max_buffer_size / bit_width,
            current_buffer_len: 0,
            num_partitions,
            max_value,
            max_num_files_per_partition: 128,
        })
    }

    #[inline]
    fn get_partition(&self, item: [usize; N]) -> Result<usize> {
        ensure!(
            item[0] <= self.max_value,
            "Got item {item:?}, but max value is {}",
            self.max_value
        );
        let num_values_per_partition = (self.max_value + 1).div_ceil(self.num_partitions);
        Ok(item[0] / num_values_per_partition)
    }

    pub fn push(&mut self, item: [usize; N]) -> Result<()> {
        let partition_id = self.get_partition(item)?;
        self.push_to_partition(item, partition_id)
    }

    fn push_to_partition(&mut self, item: [usize; N], partition_id: usize) -> Result<()> {
        let buffer = &self.buffers[partition_id];
        if self.current_buffer_len >= self.max_buffer_len {
            // if we need to flush a buffer and this one is not too small
            // (at least half the average), flush it
            if buffer.len() * self.num_partitions * 2 > self.max_buffer_len {
                self.flush_buffer(partition_id)?;
            }
        }
        let buffer = &mut self.buffers[partition_id];
        buffer.extend(item);
        self.current_buffer_len += N;
        Ok(())
    }

    fn flush_buffer(&mut self, partition_id: usize) -> Result<()> {
        #[derive(TransparentWrapper, Clone, Copy)]
        #[repr(transparent)]
        struct Quad<const N: usize>([usize; N]);

        impl<const N: usize> RadixKey for Quad<N> {
            const LEVELS: usize = N * usize::LEVELS;

            #[inline]
            fn get_level(&self, level: usize) -> u8 {
                self.0[N - level / usize::LEVELS - 1].get_level(level % usize::LEVELS)
            }
        }

        let buffer = &mut self.buffers[partition_id];
        if buffer.is_empty() {
            return Ok(());
        }

        if self.sorted_files[partition_id].len() >= self.max_num_files_per_partition {
            self.compact_partition(partition_id)
                .context("Could not compact partition")?;
        }
        let buffer = &mut self.buffers[partition_id];

        // sort items in the buffer using a regular Vec (because BitFieldVec does not implement
        // sorting, especially not radix sorting)
        let num_quads = buffer
            .len()
            .checked_div(N)
            .ok_or_else(|| anyhow!("buffer size is not a multiple of {N}"))?;
        let mut quads: Vec<[usize; N]> = Vec::with_capacity(num_quads);
        let mut buffer_iter = buffer.iter();
        for _ in 0..num_quads {
            let mut quad = [0; N];
            for elem in &mut quad {
                *elem = buffer_iter
                    .next()
                    .ok_or_else(|| anyhow!("buffer_iter is shorter than expected"))?;
            }
            quads.push(quad);
        }

        Quad::<N>::wrap_slice_mut(&mut quads).radix_sort_unstable();

        debug_assert!(quads.is_sorted(), "Quads were not sorted after radix sort");
        self.current_buffer_len -= buffer.len();
        buffer.clear();

        self.write_sorted_quads(partition_id, quads.into_iter().map(Ok))?;

        Ok(())
    }

    fn write_sorted_quads(
        &mut self,
        partition_id: usize,
        quads: impl IntoIterator<Item = Result<[usize; N]>>,
    ) -> Result<usize> {
        // write sorted quads to disk
        if self.tempdirs.is_empty() {
            self.tempdirs
                .push(TempDir::new().context("Could not create temporary directory")?);
        }
        let path = self
            .tempdirs
            .first()
            .unwrap()
            .path()
            .join(format!("{}", self.num_files_in_first_tempdir));
        let mut num_quads = 0;
        SortedArraysFile::create(
            &path,
            quads.into_iter().inspect(|_| num_quads += 1),
            no_logging!(),
        )
        .with_context(|| format!("Could not write quads to {}", path.display()))?;
        self.num_files_in_first_tempdir += 1;

        self.sorted_files[partition_id].push(path);

        Ok(num_quads)
    }

    fn compact_partition(&mut self, partition_id: usize) -> Result<()> {
        // FIXME: the current implementation causes 1 large file and 9 small files that get merged
        // into the larger one. This is incredibly inefficient because we keep compacting the same
        // items over and over (it's quadratic). We should do some sort of merge tree instead.
        let mut sorted_files = Vec::new();
        std::mem::swap(&mut sorted_files, &mut self.sorted_files[partition_id]);

        let merged_quads = Self::iter_written_quads(&sorted_files)?;
        let num_quads = self
            .write_sorted_quads(partition_id, merged_quads)
            .context("Could not write compacted items")?;
        log::debug!("Compacted {num_quads} quads in partition {partition_id}");

        for path in sorted_files {
            std::fs::remove_file(&path)
                .with_context(|| format!("Could not remove {}", path.display()))?;
        }

        Ok(())
    }

    pub fn merge(mut self, mut other: Self) -> Result<Self> {
        ensure!(
            self.max_value == other.max_value,
            "Tried to merge ExternalArraySorter with different max_value ({} with {})",
            self.max_value,
            other.max_value
        );

        self.tempdirs.extend(other.tempdirs);
        for (self_partition, other_partition) in self
            .sorted_files
            .iter_mut()
            .zip(other.sorted_files.iter_mut())
        {
            if self_partition.len() < other_partition.len() {
                std::mem::swap(self_partition, other_partition);
            }
            self_partition.append(other_partition);
        }

        // TODO: flush all partitions if they are already above a certain size,
        // so we can just move the buffers instead of copying them.
        for (partition_id, partition) in other.buffers.into_iter().enumerate() {
            let num_quads = partition
                .len()
                .checked_div(N)
                .ok_or_else(|| anyhow!("buffer size is not a multiple of {N}"))?;
            for quad_id in 0..num_quads {
                let mut quad = [0; N];
                for (i, elem) in quad.iter_mut().enumerate() {
                    *elem = partition.index_value(quad_id * N + i);
                }
                self.push_to_partition(quad, partition_id)
                    .context("Could not push merged item")?;
            }
        }

        Ok(self)
    }

    pub fn flush_buffers(&mut self) -> Result<()> {
        for partition_id in 0..self.buffers.len() {
            self.flush_buffer(partition_id)
                .context("Could not flush buffer before reading")?;
        }
        Ok(())
    }

    pub fn iter_partitions(&mut self) -> Result<Vec<impl Iterator<Item = Result<[usize; N]>>>> {
        self.flush_buffers()?;
        self.sorted_files
            .iter()
            .map(|partition| Self::iter_written_quads(partition))
            .collect()
    }

    fn iter_written_quads(files: &[PathBuf]) -> Result<impl Iterator<Item = Result<[usize; N]>>> {
        #[expect(clippy::match_same_arms)]
        Ok(files
            .iter()
            .map(|path| SortedArraysFile::mmap(path)?.owned_iter())
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .kmerge_by(|left, right| match (left, right) {
                (Ok(left), Ok(right)) => left < right,
                (Err(_), Ok(_)) => true,  // return error first
                (Ok(_), Err(_)) => false, // return error first
                (_, _) => true,           // doesn't matter
            }))
    }
}

pub struct SortedArraysFile<const N: usize> {
    data: Arc<MmapHelper<u64>>,
}

impl<const N: usize> SortedArraysFile<N> {
    pub fn create(
        path: impl AsRef<Path>,
        items: impl Iterator<Item = Result<[usize; N]>>,
        pl: &mut impl ProgressLog,
    ) -> Result<Self> {
        let path = path.as_ref();
        let file = File::create_new(path)
            .with_context(|| format!("Could not create {}", path.display()))?;

        let mut writer =
            BufBitWriter::<LE, _>::new(WordAdapter::<usize, _>::new(BufWriter::new(file)));
        // TODO: configurable.
        // lower frame size: inversely higher space usage
        // higher frame size: linearly higher random access time
        let min_frame_size = 100;
        let max_frame_size = 10000;

        let mut previous_item = [0; N];
        let mut actual_previous_item = [0; N];
        let mut current_frame_size = 0;
        for item in items {
            let item = item?;
            ensure!(
                item >= previous_item,
                "{item:?} after {previous_item:?} (actually: {actual_previous_item:?})"
            );
            ensure!(
                item >= actual_previous_item,
                "{item:?} after {previous_item:?} (actually: {actual_previous_item:?})"
            );

            ensure!(item != [0; N], "invalid quad: {item:?}");

            if (current_frame_size > min_frame_size && item[0] != previous_item[0])
                || current_frame_size >= max_frame_size
            {
                // new frame
                previous_item = [0; N];
                current_frame_size = 0;
                writer
                    .write_bits(1, 1)
                    .context("Could not write frame bit")?;
            } else {
                writer
                    .write_bits(0, 1)
                    .context("Could not write non-frame bit")?;
            }

            // quads are sorted lexicographically, so the first term of a quad is guaranteed to be
            // >= the first term of the previous quad
            let mut must_zigzag = false;
            for (&previous_cell, &cell) in previous_item.iter().zip(item.iter()) {
                if must_zigzag {
                    let diff = i64::try_from(cell).context("current term overflows i64")?
                        - i64::try_from(previous_cell).context("previous term overflows i64")?;
                    let zigzag = diff.to_nat();
                    writer
                        .write_gamma(zigzag)
                        .context("Could not write gamma")?;
                } else {
                    let diff = u64::try_from(cell)
                        .context("current term overflows u64")?
                        .checked_sub(
                            u64::try_from(previous_cell).context("previous term overflows u64")?,
                        )
                        .context(
                            "write_sorted_array_file got non-sorted quads after the initial check",
                        )?;
                    writer.write_gamma(diff).context("Could not write gamma")?;

                    if diff > 0 {
                        // this term is a strict increase, so terms after it in the quad are
                        // not guaranteed to be >= the corresponding term in the previous quad,
                        // so we must zigzag-encode them all for the rest of this term.
                        must_zigzag = true;
                    }
                }
            }
            previous_item = item;
            actual_previous_item = item;

            current_frame_size += 1;

            pl.light_update();
        }

        // mark end of file
        writer
            .write_bits(1, 1)
            .context("Could not write last frame bit")?;
        for _ in 0..N {
            writer
                .write_gamma(0)
                .context("Could not write final gammas")?;
        }

        writer
            .into_inner() // BufBitWriter -> WordAdapter
            .with_context(|| format!("Could not flush {}", path.display()))?
            .into_inner() // WordAdapter -> BufWriter
            .into_inner() // BufWriter -> File
            .with_context(|| format!("Could not flush {}", path.display()))?
            .flush()
            .with_context(|| format!("Could not flush {}", path.display()))?;

        Self::mmap(path)
    }

    pub fn mmap(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let data = Arc::new(
            MmapHelper::mmap(path, MmapFlags::SEQUENTIAL)
                .with_context(|| format!("Could not mmap array file {}", path.display()))?,
        );

        Ok(Self { data })
    }

    pub fn file_len(&self) -> usize {
        #![expect(clippy::expect_used)] // can't happen, or mmap() would have failed in the constructor
        self.data
            .len()
            .checked_mul(size_of::<u64>())
            .expect("File length (in bytes) overflowed usize")
    }

    /// Same as [`Self::iter`] but instead of quads, yields:
    /// * `(Some(bit_position), quad)` on the first quad of a frame,
    /// * and `(None, quad)` on quads inside a frame
    pub fn iter_with_positions(
        &self,
        from_bit_position: usize,
    ) -> Result<impl Iterator<Item = Result<(Option<u64>, [usize; N])>> + '_> {
        Self::_iter_with_positions(&*self.data, from_bit_position)
    }

    /// Same as [`Self::iter_with_positions`] but increments an internal [`Arc`] to return `'static`
    pub fn owned_iter_with_positions(
        &self,
        from_bit_position: usize,
    ) -> Result<impl Iterator<Item = Result<(Option<u64>, [usize; N])>> + 'static + use<N>> {
        Self::_iter_with_positions(ArcMmapHelper(Arc::clone(&self.data)), from_bit_position)
    }

    fn _iter_with_positions<'a>(
        data: impl AsRef<[u64]> + 'a,
        from_bit_position: usize,
    ) -> Result<impl Iterator<Item = Result<(Option<u64>, [usize; N])>> + 'a> {
        let mut reader = BufBitReader::<LE, _>::new(MemWordReader::<u64, _>::new(data));

        let mut first_quad = from_bit_position == 0;
        let mut actual_previous_item = [0_usize; N];
        let mut previous_item = [0_usize; N];

        reader
            .set_bit_pos(u64::try_from(from_bit_position).context("bit position overflowed u64")?)
            .with_context(|| format!("Could not seek to bit position {from_bit_position}"))?;
        if from_bit_position != 0 {
            ensure!(
                reader.read_bits(1).context("Could not read frame bit")? == 1,
                "_iter_with_positions started from {from_bit_position} which is not the start of a frame."
            )
        }
        reader
            .set_bit_pos(u64::try_from(from_bit_position).context("bit position overflowed u64")?)
            .with_context(|| format!("Could not seek to bit position {from_bit_position}"))?;

        Ok(std::iter::repeat(()).map_while(move |()| {
            (|| {
                let mut item = [0_usize; N];

                let new_frame = reader.read_bits(1).context("Could not read frame bit")? == 1;
                let bit_pos = if new_frame {
                    previous_item = [0; N];
                    // -1 because we want to start at the bit we just read
                    Some(reader.bit_pos().context("Could not get bit position")? - 1)
                } else {
                    None
                };

                // quads are sorted lexicographically, so the first term of a quad is guaranteed to be
                // >= the first term of the previous quad
                let mut must_zigzag = false;
                for (&previous_cell, cell) in previous_item.iter().zip(item.iter_mut()) {
                    if must_zigzag {
                        let zigzag = reader.read_gamma().context("Could not read gamma")?;
                        let diff = zigzag.to_int();

                        *cell = u64::try_from(previous_cell)
                            .context("previous value overflows u64")?
                            .checked_add_signed(diff)
                            .context("new value (old + relative delta) overflows u64")?
                            .try_into()
                            .context("value overflows usize")?;
                    } else {
                        let diff = reader.read_gamma().context("Could not read gamma")?;
                        *cell = u64::try_from(previous_cell)
                            .context("previous value overflows u64")?
                            .checked_add(diff)
                            .context("new value (old + absolute delta) overflows u64")?
                            .try_into()
                            .context("value overflows usize")?;

                        if diff > 0 {
                            // this term is a strict increase, so terms after it in the quad are
                            // not guaranteed to be >= the corresponding term in the previous quad,
                            // so we must zigzag-encode them all for the rest of this term.
                            must_zigzag = true;
                        }
                    }
                }

                if new_frame && item == [0; N] {
                    // zeroed item marks the end of the file
                    return Ok(None);
                }
                ensure!(
                    item >= previous_item,
                    "{item:?} {actual_previous_item:?} {previous_item:?}"
                );
                ensure!(
                    item >= actual_previous_item,
                    "{item:?} {actual_previous_item:?} {previous_item:?}"
                );
                previous_item = item;
                actual_previous_item = item;

                if first_quad {
                    // the very first quad
                    first_quad = false;
                    Ok(Some((Some(0), item)))
                } else {
                    Ok(Some((bit_pos, item)))
                }
            })()
            .transpose()
        }))
    }

    /// Same as [`Self::iter`] but starts from a specific frame
    pub fn iter_from_position(
        &self,
        from_bit_position: usize,
    ) -> Result<impl Iterator<Item = Result<[usize; N]>> + '_> {
        Ok(self.iter_with_positions(from_bit_position)?.map(|item| {
            let (_bit_pos, quad) = item?;
            Ok(quad)
        }))
    }

    /// Same as [`Self::iter_from_position`] but increments an internal [`Arc`] to return `'static`
    pub fn owned_iter_from_position(
        &self,
        from_bit_position: usize,
    ) -> Result<impl Iterator<Item = Result<[usize; N]>> + 'static + use<N>> {
        Ok(self
            .owned_iter_with_positions(from_bit_position)?
            .map(|item| {
                let (_bit_pos, quad) = item?;
                Ok(quad)
            }))
    }

    #[expect(clippy::iter_not_returning_iterator)]
    /// Returns every quad in the given file
    pub fn iter(&self) -> Result<impl Iterator<Item = Result<[usize; N]>> + '_> {
        self.iter_from_position(0)
    }

    /// Same as [`Self::iter`] but increments an internal [`Arc`] to return `'static`
    pub fn owned_iter(
        &self,
    ) -> Result<impl Iterator<Item = Result<[usize; N]>> + 'static + use<N>> {
        self.owned_iter_from_position(0)
    }
}

#[cfg(test)]
#[test]
fn test_read_write_sorted_array() -> Result<()> {
    let tempdir = tempfile::tempdir().context("Could nto create temp dir")?;
    let quads: Vec<_> = vec![[1, 2, 2, 2], [1, 2, 3, 4]];
    let array_file = SortedArraysFile::create(
        tempdir.path().join("test.bitstream"),
        quads.iter().copied().map(Ok),
        no_logging!(),
    )?;
    let actual: Vec<_> = array_file.iter()?.map(Result::unwrap).collect();
    ensure!(actual == quads, "Expected {quads:?}, got {actual:?}");
    Ok(())
}
