use anyhow::{Context, Result, ensure};
use itertools::Itertools;
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::fs::File;
use std::io::{BufWriter, Cursor, Seek, Write};
use tempfile::TempDir;
use super::terms_store::{read_length_prefixed_string, write_length_prefixed_string};

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
