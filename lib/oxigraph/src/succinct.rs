use crate::model::{GraphName, NamedOrBlankNode, Quad, Term, Triple};
use anyhow::{Context, Result, anyhow, ensure};
use bytemuck::TransparentWrapper;
use dsi_progress_logger::{ProgressLog, concurrent_progress_logger, progress_logger};
use epserde::deser::{
    Deserialize as EpDeserialize, DeserializeInner as EpDeserializeInner, MemCase,
};
use epserde::ser::Serialize as EpSerialize;
use itertools::Itertools;
use lender::{Lender, Lending};
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use serde::{Deserialize, Serialize};
use std::borrow::Borrow;
use std::fs::File;
use std::hash::Hasher;
use std::io::{BufRead, BufReader, BufWriter, Cursor, Read, Seek, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use sux::bits::bit_field_vec::BitFieldVec;
use sux::func::{VBuilder, VFunc};
use sux::traits::BitFieldSliceCore;
use sux::traits::bit_field_slice::BitFieldSlice;
use sux::utils::{FromIntoIterator, RewindableIoLender};
use tempfile::TempDir;
use webgraph::prelude::MmapHelper;
use zstd::stream::read::Decoder;

/// workaround while https://github.com/vigna/sux-rs/pull/78 is not merged
#[derive(Debug, TransparentWrapper)]
#[repr(transparent)]
pub struct RawTerm(pub [u8]);

impl epserde::traits::type_info::TypeHash for RawTerm {
    fn type_hash(hasher: &mut impl Hasher) {
        <&[u8]>::type_hash(hasher)
    }

    fn type_hash_val(&self, hasher: &mut impl Hasher) {
        <&[u8]>::type_hash_val(&&self.0, hasher)
    }
}
impl sux::utils::ToSig<[u64; 2]> for RawTerm {
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 2] {
        <&[u8]>::to_sig(&key.borrow().0, seed)
    }
}

/// workaround while https://github.com/vigna/sux-rs/pull/78 is not merged
#[derive(Debug)]
#[repr(transparent)]
pub struct BoxedRawTerm(pub Box<[u8]>);

impl Borrow<RawTerm> for BoxedRawTerm {
    fn borrow(&self) -> &RawTerm {
        RawTerm::wrap_ref(self.0.as_ref())
    }
}

impl sux::utils::ToSig<[u64; 2]> for BoxedRawTerm {
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 2] {
        <&[u8]>::to_sig(key.borrow().0.as_ref(), seed)
    }
}

pub struct TermMphf<D: BitFieldSlice<usize> = MmapHelper<usize>> {
    vfunc: MemCase<VFunc<RawTerm, usize, D>>,
    marker: PhantomData<D>,
}

impl<D: BitFieldSlice<usize>> TermMphf<D> {
    pub fn serialize(&self, path: impl AsRef<Path>) -> Result<()>
    where
        VFunc<RawTerm, usize, D>: EpSerialize,
    {
        let path = path.as_ref();
        std::fs::create_dir(&path)
            .with_context(|| format!("Could not create {}", path.display()))?;

        let vfunc_path = path.join("mphf.vfunc");
        let mut file = File::create(&vfunc_path)
            .with_context(|| format!("Could not create {}", vfunc_path.display()))?;
        self.vfunc
            .serialize(&mut file)
            .with_context(|| format!("Could write VFunc to {}", vfunc_path.display()))?;
        Ok(())
    }
}

impl<D: BitFieldSlice<usize> + EpDeserializeInner> TermMphf<D>
where
    VFunc<RawTerm, usize, D>: EpDeserialize,
{
    pub fn deserialize(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<TermMphf<<D as EpDeserializeInner>::DeserType<'static>>>
    where
        <D as EpDeserializeInner>::DeserType<'static>: AsRef<[usize]>,
        for<'a> VFunc<RawTerm, usize, D>: EpDeserializeInner<
            DeserType<'a> = VFunc<RawTerm, usize, <D as EpDeserializeInner>::DeserType<'a>>,
        >,
    {
        let path = path.as_ref();
        let vfunc_path = path.join("mphf.vfunc");

        let flags = epserde::deser::mem_case::Flags::RANDOM_ACCESS;
        // SAFETY: this is unsafe because we can't guarantee the file won't be modified while we
        // access it, but there is nothing we can do about this.
        let vfunc = unsafe { <VFunc<RawTerm, usize, D>>::mmap(&vfunc_path, flags) }
            .with_context(|| format!("Could write VFunc to {}", vfunc_path.display()))?;
        Ok(TermMphf {
            vfunc,
            marker: PhantomData,
        })
    }
}

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
struct ExternalSorter {
    tempdirs: Vec<TempDir>,
    sorted_files: Vec<File>,
    max_buffer_size: usize,
    max_num_files: usize,
    buffer_size: usize,
    buffer: FxHashSet<Box<[u8]>>,
    written_size: usize,
    num_unique_items_upperbound: usize, // only counting those in files
}

impl ExternalSorter {
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

fn write_length_prefixed_string(writer: &mut impl Write, string: &[u8], path: &Path) -> Result<()> {
    // write string's length
    writer
        .write_all(
            &u32::try_from(string.len())
                .with_context(|| format!("String is longer than {} bytes", u32::MAX))?
                .to_ne_bytes(),
        )
        .with_context(|| format!("Could not write to {}", path.display()))?;

    // write string
    writer
        .write_all(string)
        .with_context(|| format!("Could not write to {}", path.display()))?;

    Ok(())
}
fn read_length_prefixed_string<R: Read>(
    file: &mut R,
    get_position: impl FnOnce(&mut R) -> Option<Result<u64>>,
) -> Result<Option<Box<[u8]>>> {
    let mut length_bytes = [0; _];
    if let Err(e) = file.read_exact(&mut length_bytes) {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            return Ok(None);
        }
        Err(e).context("Could not read next string's length")?;
    }
    let length = usize::try_from(u32::from_ne_bytes(length_bytes))
        .with_context(|| format!("String is longer than {} bytes", usize::MAX))?;
    let mut string = vec![0; length];
    file.read_exact(&mut string).with_context(|| {
        if let Some(position) = get_position(file) {
            format!(
                "Could not read next string of length {length} from offset {}",
                position
                    .map(|pos| pos.to_string())
                    .unwrap_or_else(|e| format!("<error: {e}>")),
            )
        } else {
            format!("Could not read next string of length {length}",)
        }
    })?;
    Ok(Some(string.into()))
}

fn deduplicate_terms(quads: impl ParallelIterator<Item = Result<Quad>>) -> Result<ExternalSorter> {
    fn push_term(sorter: &mut ExternalSorter, term: Term) -> Result<()> {
        match term {
            Term::NamedNode(n) => sorter.push_str(n.as_str().to_owned()),
            Term::BlankNode(n) => sorter.push_str(n.as_str().to_owned()),
            Term::Literal(l) => sorter.push_str(l.to_string()), // XXX is that injective?
            #[cfg(feature = "rdf-12")]
            Term::Triple(t) => {
                let Triple {
                    subject,
                    predicate,
                    object,
                } = *t;
                sorter.push_str(match subject {
                    NamedOrBlankNode::NamedNode(n) => n.as_str().to_owned(),
                    NamedOrBlankNode::BlankNode(n) => n.as_str().to_owned(),
                })?;
                sorter.push_str(predicate.as_str().to_owned())?;
                push_term(sorter, object)
            }
        }
    }

    let unique_sorted_terms = quads
        .fold(
            || {
                // 100MiB in-memory buffer per thread
                ExternalSorter::new(100 * 1024 * 1024, 10)
                    .context("Could not create sorter ExternalSorter")
            },
            |thread_sorter, quad| -> Result<_> {
                let mut thread_sorter: ExternalSorter = thread_sorter?;
                let Quad {
                    subject,
                    predicate,
                    object,
                    graph_name,
                } = quad?;
                thread_sorter
                    .push_str(match subject {
                        NamedOrBlankNode::NamedNode(n) => n.as_str().to_owned(),
                        NamedOrBlankNode::BlankNode(n) => n.as_str().to_owned(),
                    })
                    .context("Could not push subject")?;
                thread_sorter
                    .push_str(predicate.as_str().to_owned())
                    .context("Could not push predicate")?;
                thread_sorter
                    .push_str(match graph_name {
                        GraphName::NamedNode(n) => n.as_str().to_owned(),
                        GraphName::BlankNode(n) => n.as_str().to_owned(),
                        GraphName::DefaultGraph => "".to_owned(), // XXX I guess?
                    })
                    .context("Could not push graph name")?;
                push_term(&mut thread_sorter, object).context("Could not push term")?;

                Ok(thread_sorter)
            },
        )
        .reduce(
            || {
                ExternalSorter::new(100 * 1024 * 1024, 10)
                    .context("Could not create reducer ExternalSorter")
            },
            |left, right| {
                left?
                    .merge(right?)
                    .context("Could not merge ExternalSorter")
            },
        )?;

    Ok(unique_sorted_terms)
}

#[derive(Serialize, Deserialize, Clone)]
struct TermsStoreConfiguration {
    terms_per_frame: usize,
    frames_per_file: usize,
    num_terms: usize,
}

pub fn write_unique_terms(
    quads: impl ParallelIterator<Item = Result<Quad>>,
    dir: &Path,
    approx_num_quads: Option<usize>,
) -> Result<()> {
    let mut config = TermsStoreConfiguration {
        terms_per_frame: 16,
        frames_per_file: 1024 * 1024,
        num_terms: 0,
    };
    let compression_level = 1; // we are going to read it very often and in small chunks

    let mut pl = concurrent_progress_logger!(
        item_name = "quad",
        display_memory = true,
        local_speed = true,
        expected_updates = approx_num_quads,
    );
    pl.start("Reading quads and deduplicating terms...");
    let mut sorter = deduplicate_terms(quads.map_with(pl.clone(), |pl, quad| {
        pl.light_update();
        quad
    }))?;
    pl.done();
    drop(pl);

    std::fs::create_dir(dir).with_context(|| format!("Could not create {}", dir.display()))?;
    let mut pl = progress_logger!(
        item_name = "term",
        display_memory = true,
        local_speed = true,
        expected_updates = Some(sorter.num_unique_items_upperbound),
    );
    pl.start("Writing terms...");
    let term_chunk_iterator = sorter
        .drain_boxed_bytes()
        .context("Could not read final iterator of terms")?
        .chunks(config.terms_per_frame * config.frames_per_file);
    for (file_id, big_chunk) in term_chunk_iterator.into_iter().enumerate() {
        let file_path = dir.join(format!("{file_id:0>10}.zst"));
        let mut file = File::create(&file_path)
            .with_context(|| format!("Could not create {}", file_path.display()))?;
        let term_chunk_iterator = big_chunk.chunks(config.terms_per_frame);
        for (frame_id, small_chunk) in term_chunk_iterator.into_iter().enumerate() {
            let mut uncompressed_frame = Cursor::new(Vec::new());
            for term in small_chunk {
                write_length_prefixed_string(&mut uncompressed_frame, &term?, &file_path)?;
                pl.light_update();
                config.num_terms += 1;
            }
            let uncompressed_frame = &uncompressed_frame.into_inner();
            let mut compressed_frame =
                Vec::with_capacity(zstd::zstd_safe::compress_bound(uncompressed_frame.len()));
            zstd::zstd_safe::compress(&mut compressed_frame, uncompressed_frame, compression_level)
                .map_err(|errno| {
                    anyhow!(
                        "Could not compress frame: {}",
                        zstd::zstd_safe::get_error_name(errno)
                    )
                })?;
            file.write_all(&compressed_frame).with_context(|| {
                format!(
                    "Could not write frame {frame_id} of {}",
                    file_path.display(),
                )
            })?;
        }
    }
    pl.done();

    let config_path = dir.join("config.json");
    let config_file = File::create(&config_path)
        .with_context(|| format!("Could not create {}", config_path.display()))?;
    serde_json::to_writer_pretty(config_file, &config)
        .context("Could not write terms store config")?;

    Ok(())
}

struct TermsFile<D> {
    first_term_id: usize,
    num_terms: usize,
    path: PathBuf,
    compressed_frames: D,
}

fn list_terms_files(
    dir: &Path,
) -> Result<(TermsStoreConfiguration, Vec<TermsFile<impl AsRef<[u8]>>>)> {
    let config_path = dir.join("config.json");
    let config_file = File::open(&config_path)
        .with_context(|| format!("Could not open {}", config_path.display()))?;
    let config: TermsStoreConfiguration = serde_json::from_reader(config_file)
        .with_context(|| format!("Could not read config from {}", config_path.display()))?;

    let mut entries = std::fs::read_dir(dir)
        .with_context(|| format!("Could not list {}", dir.display()))?
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("Could not stat {} entry", dir.display()))?
        .into_iter()
        .filter(|entry| entry.path().extension() == Some("zst".as_ref()))
        .collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.path());

    let terms_per_file = config.terms_per_frame * config.frames_per_file;
    let num_files = config.num_terms.div_ceil(terms_per_file);
    ensure!(
        num_files == entries.len(),
        "Inconsistent number of .zst files in {}: expected {num_files}, got {}",
        dir.display(),
        entries.len()
    );

    Ok((
        config.clone(),
        entries
            .into_par_iter()
            .enumerate()
            .map(|(file_id, entry)| -> Result<_> {
                let num_terms = if file_id == num_files - 1 {
                    let num_terms = config.num_terms % terms_per_file;
                    if num_terms == 0 {
                        terms_per_file
                    } else {
                        num_terms
                    }
                } else {
                    terms_per_file
                };

                let terms_file_path = entry.path();
                let terms_file = File::open(&terms_file_path)
                    .with_context(|| format!("Could not open {}", terms_file_path.display()))?;
                let terms_file_len = usize::try_from(
                    entry
                        .metadata()
                        .with_context(|| format!("Could not stat {}", terms_file_path.display()))?
                        .len(),
                )
                .context("File is larger than usize")?;

                let compressed_frames = unsafe {
                    mmap_rs::MmapOptions::new(terms_file_len)
                        .context("Could not initialize mmap")?
                        .with_file(&terms_file, 0)
                        .map()
                        .with_context(|| format!("Could not mmap {}", terms_file_path.display()))?
                };

                Ok(TermsFile {
                    first_term_id: file_id * terms_per_file,
                    num_terms,
                    path: terms_file_path,
                    compressed_frames,
                })
            })
            .collect::<Result<_>>()?,
    ))
}

pub fn index_terms(dir: &Path) -> Result<()> {
    let (config, terms_files) = list_terms_files(dir)?;
    let mut pl = concurrent_progress_logger!(
        item_name = "frame",
        display_memory = true,
        local_speed = true,
        expected_updates = Some(config.num_terms / config.terms_per_frame),
    );
    pl.start("Indexing terms...");

    terms_files
        .into_par_iter()
        .try_for_each_with(pl.clone(), |pl, terms_file| {
            let TermsFile {
                first_term_id: _,
                num_terms,
                path,
                compressed_frames,
            } = terms_file;
            let num_frames = num_terms.div_ceil(config.terms_per_frame);
            let compressed_frames = compressed_frames.as_ref();

            let mut efb =
                sux::dict::elias_fano::EliasFanoBuilder::new(num_terms, compressed_frames.len());
            let mut offset = 0;
            for frame_id in 0..num_frames {
                ensure!(
                    !compressed_frames[offset..].is_empty(),
                    "Expected {num_frames} in {}, but there are only {frame_id}",
                    path.display()
                );
                efb.push(offset);
                offset += zstd::zstd_safe::find_frame_compressed_size(&compressed_frames[offset..])
                    .map_err(|errno| {
                        anyhow!(
                            "Could not get compressed size of frame {frame_id} of {}: {}",
                            path.display(),
                            zstd::zstd_safe::get_error_name(errno)
                        )
                    })?;
                pl.light_update();
            }
            ensure!(
                compressed_frames[offset..].is_empty(),
                "Expected {num_frames} in {}, but there are more ({} unread bytes)",
                path.display(),
                compressed_frames[offset..].len()
            );

            let ef = efb.build_with_seq();

            let index_file_path = path.with_extension("ef");
            let mut index_file = File::create(&index_file_path)
                .with_context(|| format!("Could not create {}", index_file_path.display()))?;
            ef.serialize(&mut index_file).with_context(|| {
                format!(
                    "Could not write Elias-Fano index to {}",
                    index_file_path.display()
                )
            })?;

            Ok(())
        })?;
    pl.done();

    Ok(())
}

pub fn build_terms_mph(dir: &Path) -> Result<TermMphf<BitFieldVec<usize>>> {
    let (config, terms_files) = list_terms_files(dir)?;

    let terms_lender = RewindableIoFlattenLender::new(
        terms_files
            .iter()
            .map(|terms_file| {
                let TermsFile {
                    first_term_id: _,
                    num_terms,
                    path,
                    compressed_frames,
                } = terms_file;

                ZstdLengthPrefixedStringLender::new(Cursor::new(compressed_frames))
                    .map_err(DecodeError)
            })
            .collect::<Result<_, DecodeError>>()?,
    );

    let mut pl = progress_logger!(
        item_name = "term",
        display_memory = true,
        local_speed = true,
        expected_updates = Some(config.num_terms),
    );
    pl.start("Building MPHF...");

    let builder = VBuilder::<_, BitFieldVec<usize>>::default().expected_num_keys(config.num_terms)
        .check_dups(true)
        .offline(true) // Save memory by spilling to disk
        .low_mem(true) // Save memory by using slightly more CPU;
        ;
    let vfunc = MemCase::encase(
        builder
            .try_build_func::<RawTerm, BoxedRawTerm>(
                terms_lender,
                FromIntoIterator::from(0..config.num_terms),
                &mut pl,
            )
            .context("Could not build VFunc")?,
    );
    pl.done();

    Ok(TermMphf {
        vfunc,
        marker: PhantomData,
    })
}

/// Reads a zstd-compressed file using [`read_length_prefixed_string`] on each item of each frame
struct ZstdLengthPrefixedStringLender<R: BufRead> {
    decoder: Decoder<'static, R>,
    string: Option<BoxedRawTerm>,
}

impl<R: BufRead> ZstdLengthPrefixedStringLender<R> {
    pub fn new(read: R) -> Result<Self> {
        Ok(ZstdLengthPrefixedStringLender {
            decoder: Decoder::with_buffer(read)?,
            string: None,
        })
    }
}

impl<'lend, R: BufRead> Lending<'lend> for ZstdLengthPrefixedStringLender<R> {
    type Lend = Result<&'lend BoxedRawTerm, DecodeError>;
}

impl<R: BufRead> Lender for ZstdLengthPrefixedStringLender<R> {
    fn next(&mut self) -> Option<<Self as Lending<'_>>::Lend> {
        match read_length_prefixed_string(&mut self.decoder, |_| None) {
            Ok(Some(string)) => {
                if let Some(previous_string) = &self.string {
                    if string <= previous_string.0 {
                        return Some(Err(DecodeError(anyhow!(
                            "Unsorted strings: {} ({:?}) after {} ({:?})",
                            String::from_utf8_lossy(&string),
                            &string,
                            String::from_utf8_lossy(&previous_string.0),
                            &previous_string.0,
                        ))));
                    }
                }
                self.string = Some(BoxedRawTerm(string));
                Some(Ok(self.string.as_ref().unwrap()))
            }
            Ok(None) => None,
            Err(e) => Some(Err(e.into())),
        }
    }
}

impl<R: BufRead + Seek> RewindableIoLender<BoxedRawTerm> for ZstdLengthPrefixedStringLender<R> {
    type Error = DecodeError;

    fn rewind(mut self) -> Result<Self, Self::Error> {
        let mut read = self.decoder.finish();
        read.rewind().context("Could not rewind")?;
        self.decoder =
            Decoder::with_buffer(read).context("Could not create new decoder to rewind")?;
        Ok(self)
    }
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct DecodeError(#[from] anyhow::Error);

/// Equivalent to [`Lender::flatten`] but implements [`RewindableIoLender`]
struct RewindableIoFlattenLender<L> {
    lenders: Vec<L>,
    current_index: usize,
}

impl<L> RewindableIoFlattenLender<L> {
    pub fn new(lenders: Vec<L>) -> Self {
        Self {
            lenders,
            current_index: 0,
        }
    }
}

impl<'lend, L: Lending<'lend>> Lending<'lend> for RewindableIoFlattenLender<L> {
    type Lend = L::Lend;
}

impl<L: Lender> Lender for RewindableIoFlattenLender<L> {
    fn next(&mut self) -> Option<<Self as Lending<'_>>::Lend> {
        // This is equivalent to:
        //
        //  while let Some(current_lender) = self.lenders.get_mut(self.current_index) {
        //      if let Some(item) = current_lender.next() {
        //          return Some(item);
        //      }
        //      // exhausted the current lender, go to the next one
        //      self.current_index += 1
        //  }
        //  // exhausted all lenders
        //  None
        //
        //  but the borrow-checker forces us to write it this way because it doesn't understand we
        //  only borrow one lender at a time.
        self.lenders[self.current_index..]
            .iter_mut()
            .flat_map(|current_lender| {
                if let Some(item) = current_lender.next() {
                    return Some(item);
                }

                // exhausted the current lender, go to the next one
                self.current_index += 1;

                None
            })
            .next()
    }
}

impl<T, L: RewindableIoLender<T>> RewindableIoLender<T> for RewindableIoFlattenLender<L> {
    type Error = <L as RewindableIoLender<T>>::Error;

    fn rewind(mut self) -> Result<Self, Self::Error> {
        let mut new_lenders = Vec::with_capacity(self.lenders.len());
        for lender in self.lenders.drain(0..=self.current_index) {
            new_lenders.push(lender.rewind()?);
        }
        new_lenders.extend(self.lenders.drain(..));
        std::mem::swap(&mut new_lenders, &mut self.lenders);
        self.current_index = 0;
        Ok(self)
    }
}
