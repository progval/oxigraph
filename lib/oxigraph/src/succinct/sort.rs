use super::terms_store::{read_length_prefixed_string, write_length_prefixed_string};
use anyhow::{Context, Result, anyhow, ensure};
use dsi_bitstream::prelude::*;
use dsi_progress_logger::{ProgressLog, no_logging};
use itertools::Itertools;
use rayon::prelude::*;
use rdst::RadixSort;
use rustc_hash::FxHashSet;
use std::fs::File;
use std::io::{BufWriter, Cursor, Seek, Write};
use sux::bits::BitFieldVec;
use sux::traits::bit_field_slice::BitFieldSliceCore;
use tempfile::TempDir;

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
    pub(super) num_unique_items_upperbound: usize, // only counting those in files
}

impl ExternalDeduplicatingStringSorter {
    pub fn new(max_buffer_size: usize, max_num_files: usize) -> Result<Self> {
        Ok(Self {
            tempdirs: Vec::new(), // Create it only if needed
            sorted_files: Vec::new(),
            buffer: FxHashSet::with_hasher(Default::default()),
            buffer_size: 0,
            max_buffer_size,
            max_num_files,
            written_size: 0,
            num_unique_items_upperbound: 0,
        })
    }

    pub fn push_str(&mut self, s: String) -> Result<()> {
        self.push_boxed_bytes(s.into_bytes().into())
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
                    mmap_rs::MmapOptions::new(file_len)
                        .context("Could not initialize mmap")?
                        .with_flags(mmap_rs::MmapFlags::SEQUENTIAL)
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
                (_, _) => true, // doesn't matter, we are going to error anyway
            })
            .dedup_by(|left, right| match (left, right) {
                (Ok(left), Ok(right)) => left == right,
                (_, _) => true, // doesn't matter, we are going to error anyway
            }))
    }

    /// Consome this external sorter and returns sorted items
    pub fn drain_boxed_bytes(&mut self) -> Result<impl Iterator<Item = Result<Box<[u8]>>>> {
        let mut buffer = Default::default();
        std::mem::swap(&mut buffer, &mut self.buffer);
        let mut buffer: Vec<_> = buffer.into_iter().map(Box::from).collect();
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
        self.sorted_files.extend(other.sorted_files.into_iter());
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
pub(super) struct ExternalArraySorter<const N: usize> {
    tempdirs: Vec<TempDir>,
    sorted_files: Vec<Vec<mmap_rs::Mmap>>, // one vec for each partition
    max_buffer_size: usize,
    buffers: Vec<BitFieldVec<usize>>,
    num_partitions: usize,
    max_value: usize,
}

impl<const N: usize> ExternalArraySorter<N> {
    /// At the end you need to process each partition in parallel, then it's good
    /// to have at least as many partitions as threads. Multiply that by
    /// about 10 if partitions are uneven, so a single thread doesn't end up alone
    /// at the end writing a huge partition while other threads are already done.
    ///
    /// More partitions also save time on merging.
    pub fn new(max_value: usize, max_buffer_size: usize, num_partitions: usize) -> Result<Self> {
        let bit_width = usize::try_from(usize::BITS - max_value.leading_zeros())
            .context("Weird pointer size")?;


        Ok(Self {
            tempdirs: Vec::new(), // Create it only if needed
            sorted_files: (0..num_partitions).map(|_| Vec::new()).collect(),
            buffers: (0..num_partitions)
                .map(|_| {
                    BitFieldVec::with_capacity(
                        bit_width,
                        max_buffer_size / bit_width / num_partitions,
                    )
                })
                .collect(),
            max_buffer_size,
            num_partitions,
            max_value,
        })
    }

    #[inline(always)]
    fn item_width_bytes() -> usize {
        N * usize::try_from(usize::BITS).expect("weird pointer size") / 8
    }

    #[inline(always)]
    fn get_partition(&self, item: [usize; N]) -> Result<usize> {
        ensure!(
            item[0] <= self.max_value,
            "Got item {item:?}, but max value is {:?}",
            self.max_value
        );
        Ok((item[0] * self.num_partitions) / (self.max_value + 1))
    }

    pub fn push(&mut self, item: [usize; N]) -> Result<()> {
        let partition_id = self.get_partition(item)?;
        self.push_to_partition(item, partition_id)
    }

    fn push_to_partition(&mut self, item: [usize; N], partition_id: usize) -> Result<()> {
        let buffer = &self.buffers[partition_id];
        if buffer.len() + Self::item_width_bytes() >= self.max_buffer_size {
            self.flush_buffer(partition_id)?;
        }
        let buffer = &mut self.buffers[partition_id];
        buffer.extend(item);
        Ok(())
    }

    fn flush_buffer(&mut self, partition_id: usize) -> Result<()> {
        let buffer = &mut self.buffers[partition_id];

        // sort items in the buffer using a regular Vec (because BitFieldVec does not implement
        // sorting, especially not radix sorting)
        let chunks = buffer.iter().chunks(4);
        let mut sorted_vec: Vec<[usize; N]> = chunks
            .into_iter()
            .map(|chunk| {
                chunk.collect_array().ok_or_else(|| {
            anyhow!(
                "ExternalDeduplicatingStringSorter<{}> buffer length is not a multiple of 4",
                N
            )
        })
            })
            .collect::<Result<_>>()?;
        if N != 4 {
            todo!("Add support for N != 4");
        }
        // FIXME: replace '4' with 'N'
        bytemuck::cast_slice_mut::<_, [u8; (4 * (usize::BITS / 8)) as usize]>(&mut sorted_vec)
            .radix_sort_unstable();
        buffer.clear();

        // write sorted quads to disk
        let path = self
            .tempdirs
            .first()
            .unwrap()
            .path()
            .join(format!("{}", self.sorted_files[partition_id].len()));
        let file = File::create_new(&path)
            .with_context(|| format!("Could not create {}", path.display()))?;
        let mut writer = BufBitWriter::new(WordAdapter::<usize, _>::new(file));
        write_sorted_array_file(&mut writer, sorted_vec.into_iter().map(Ok), no_logging!())
            .with_context(|| format!("Could not write quads to {}", path.display()))?;
        let mut file = writer
            .into_inner()
            .with_context(|| format!("Could not flush {}", path.display()))?
            .into_inner();

        file.rewind()
            .with_context(|| format!("Could not rewind {}", path.display()))?;

        let file_len = usize::try_from(
            file.metadata()
                .context("Could not stat sorted array file")?
                .len(),
        )
        .context("file size overflowed usize")?;
        let data = unsafe {
            mmap_rs::MmapOptions::new(file_len)
                .context("Could not initialize mmap")?
                .with_flags(mmap_rs::MmapFlags::SEQUENTIAL)
                .with_file(&file, 0)
                .map()
                .context("Could not mmap sorted array file")?
        };

        self.sorted_files[partition_id].push(data);
        buffer.clear();

        Ok(())
    }

    pub fn merge(mut self, other: Self) -> Result<Self> {
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
            .zip(other.sorted_files.into_iter())
        {
            self_partition.extend(other_partition);
        }

        // TODO: flush all partitions if they are already above a certain size,
        // so we can just move the buffers instead of copying them.
        for (partition_id, partition) in other.buffers.into_iter().enumerate() {
            let chunks = partition.iter().chunks(N);
            for item in chunks.into_iter() {
                self.push_to_partition(
                    item.collect_array()
                        .ok_or_else(|| anyhow!("buffer size is not a multiple of {N}"))?,
                    partition_id,
                )
                .context("Could not push merged item")?;
            }
        }

        Ok(self)
    }

    pub fn iter_partitions(&mut self) -> Result<Vec<impl Iterator<Item = Result<[usize; N]>>>> {
        for partition_id in 0..self.buffers.len() {
            self.flush_buffer(partition_id)
                .context("Could not flush buffer before reading")?;
        }
        Ok(self
            .sorted_files
            .iter()
            .map(|partition| {
                partition
                    .iter()
                    .map(|file| read_sorted_array_file(BufBitReader::new(MemWordReader::new(file))))
                    .kmerge_by(|left, right| match (left, right) {
                        (Ok(left), Ok(right)) => left < right,
                        (_, _) => true, // doesn't matter, we are going to error anyway
                    })
            })
            .collect())
    }
}

pub fn write_sorted_array_file<const N: usize>(
    writer: &mut (impl BitWrite<LE> + GammaWrite<LE>),
    items: impl Iterator<Item = Result<[usize; N]>>,
    pl: &mut impl ProgressLog,
) -> Result<()> {
    let mut previous_item = [0; N];
    for item in items {
        let item = item?;
        // guaranteed to be increasing, no need to zigzag
        writer
            .write_gamma(
                u64::try_from(
                    item[0]
                        .checked_sub(previous_item[0])
                        .context("write_sorted_array_file got non-sorted quads")?,
                )
                .context("value overflows u64")?,
            )
            .context("Could not write gamma")?;

        for (&previous_cell, &cell) in previous_item[1..].iter().zip(item[1..].iter()) {
            // TODO: we only need to zigzag if item[0] increased. otherwise we know it's positive
            // because of lexicographic order
            let zigzag = (i64::try_from(cell).context("value overflows i64")?
                - i64::try_from(previous_cell).context("value overflows i64")?)
            .to_nat();
            writer
                .write_gamma(zigzag)
                .context("Could not write gamma")?;
        }
        previous_item = item;
        pl.light_update();
    }

    Ok(())
}

pub fn read_sorted_array_file<'a, const N: usize>(
    mut reader: impl BitRead<LE> + GammaRead<LE> + 'a,
) -> impl Iterator<Item = Result<[usize; N]>> + 'a {
    let mut previous_item = [0usize; N];
    std::iter::repeat(()).map(move |()| {
        let mut item = [0usize; N];
        // guaranteed to be increasing, no need to zigzag
        item[0] = reader
            .read_gamma()
            .context("Could not read gamma")?
            .checked_add(previous_item[0] as u64)
            .context("value overflows u64")?
            .try_into()
            .context("value overflows usize")?;

        for (&previous_cell, cell) in previous_item[1..].iter().zip(item[1..].iter_mut()) {
            // TODO: we only need to zigzag if item[0] increased. otherwise we know it's positive
            // because of lexicographic order
            let zigzag = reader.read_gamma().context("Could not read gamma")?;
            let diff = zigzag.to_int();

            *cell = u64::try_from(previous_cell)
                .context("value overflows u64")?
                .checked_add_signed(diff)
                .context("value overflows u64")?
                .try_into()
                .context("value overflows usize")?;
        }

        previous_item = item;

        Ok(item)
    })
}
