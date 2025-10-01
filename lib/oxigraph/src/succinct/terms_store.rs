use super::sort::ExternalDeduplicatingStringSorter;
use crate::model::{GraphName, Quad, Term};
use anyhow::{Context, Result, anyhow, bail, ensure};
use dsi_progress_logger::{ProgressLog, concurrent_progress_logger, progress_logger};
use epserde::deser::mem_case::{Flags, MemCase};
use epserde::deser::{Deserialize as EpDeserialize, DeserializeInner as EpDeserializeInner};
use epserde::ser::Serialize as EpSerialize;
use itertools::Itertools;
use mmap_rs::Mmap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use sux::dict::elias_fano::{EfSeqDict, EliasFanoBuilder};
use sux::traits::{IndexedDict, IndexedSeq};

pub(super) fn write_length_prefixed_string(
    writer: &mut impl Write,
    string: &[u8],
    path: &Path,
) -> Result<()> {
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
pub(super) fn read_length_prefixed_string<R: Read>(
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

pub fn serialize_term(term: &Term) -> Result<Vec<u8>> {
    serde_json::to_vec(term).with_context(|| format!("Could not serialize term {term:?}"))
}

pub fn deserialize_term(bytes: &[u8]) -> Result<Term> {
    serde_json::from_slice(bytes).with_context(|| {
        format!(
            "Could not deserialize term '{}'",
            String::from_utf8_lossy(bytes)
        )
    })
}

pub fn serialize_graph_name(graph_name: &GraphName) -> Result<Vec<u8>> {
    if graph_name == &GraphName::DefaultGraph {
        // The empty string is very handy in datasets that have most of their quads
        // in the default graph, because it comes first in the sorted list of terms,
        // which means that the DefaultGraph gets hashed to id 0.
        // And because sorted quad files have to write the graph at the beginning
        // of each frame, the shorter the graph id is, the better.
        // And because we use gamma coding
        // (https://docs.rs/dsi-bitstream/latest/dsi_bitstream/codes/index.html),
        // 0 is encoded as a single bit whereas any other value takes at least four bits.
        return Ok(Vec::new());
    } else {
        serde_json::to_vec(graph_name)
            .with_context(|| format!("Could not serialize graph name {graph_name:?}"))
    }
}

pub fn deserialize_graph_name(bytes: &[u8]) -> Result<GraphName> {
    if bytes.is_empty() {
        Ok(GraphName::DefaultGraph)
    } else {
        serde_json::from_slice(bytes).with_context(|| {
            format!(
                "Could not deserialize graph name '{}'",
                String::from_utf8_lossy(bytes)
            )
        })
    }
}

fn deduplicate_terms(
    quads: impl ParallelIterator<Item = Result<Quad>>,
) -> Result<ExternalDeduplicatingStringSorter> {
    fn push_term(sorter: &mut ExternalDeduplicatingStringSorter, term: Term) -> Result<()> {
        sorter.push_boxed_bytes(serialize_term(&term)?.into_boxed_slice())
    }

    let unique_sorted_terms = quads
        .fold(
            || {
                // 100MiB in-memory buffer per thread
                ExternalDeduplicatingStringSorter::new(100 * 1024 * 1024, 16)
                    .context("Could not create sorter ExternalDeduplicatingStringSorter")
            },
            |thread_sorter, quad| -> Result<_> {
                let mut thread_sorter: ExternalDeduplicatingStringSorter = thread_sorter?;
                let Quad {
                    subject,
                    predicate,
                    object,
                    graph_name,
                } = quad?;
                push_term(&mut thread_sorter, subject.into()).context("Could not push subject")?;
                push_term(&mut thread_sorter, predicate.into())
                    .context("Could not push predicate")?;
                push_term(&mut thread_sorter, object).context("Could not push term")?;

                // TODO: deduplicate and store graph names separately
                thread_sorter
                    .push_boxed_bytes(serialize_graph_name(&graph_name)?.into_boxed_slice())
                    .context("Could not push graph name")?;

                Ok(thread_sorter)
            },
        )
        .reduce(
            || {
                ExternalDeduplicatingStringSorter::new(100 * 1024 * 1024, 10)
                    .context("Could not create reducer ExternalDeduplicatingStringSorter")
            },
            |left, right| {
                left?
                    .merge(right?)
                    .context("Could not merge ExternalDeduplicatingStringSorter")
            },
        )?;

    Ok(unique_sorted_terms)
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TermStoreConfiguration {
    pub terms_per_frame: usize,
    pub frames_per_file: usize,
    pub num_terms: usize,
}

pub fn write_unique_terms(
    quads: impl ParallelIterator<Item = Result<Quad>>,
    dir: &Path,
    approx_num_quads: Option<usize>,
) -> Result<()> {
    let mut config = TermStoreConfiguration {
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

pub(super) struct TermsFile<D> {
    pub first_term_id: usize,
    pub num_terms: usize,
    pub path: PathBuf,
    pub compressed_frames: D,
}

pub(super) fn list_terms_files(
    dir: &Path,
) -> Result<(TermStoreConfiguration, Vec<TermsFile<Mmap>>)> {
    let config_path = dir.join("config.json");
    let config_file = File::open(&config_path)
        .with_context(|| format!("Could not open {}", config_path.display()))?;
    let config: TermStoreConfiguration = serde_json::from_reader(config_file)
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
        item_name = "term",
        display_memory = true,
        local_speed = true,
        expected_updates = Some(config.num_terms),
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

            let mut efb = EliasFanoBuilder::new(num_frames, compressed_frames.len());
            let mut offset = 0;
            for frame_id in 0..num_frames {
                ensure!(
                    !compressed_frames[offset..].is_empty(),
                    "Expected {num_frames} in {}, but there are only {frame_id}",
                    path.display()
                );
                let frame_compressed_size =
                    zstd::zstd_safe::find_frame_compressed_size(&compressed_frames[offset..])
                        .map_err(|errno| {
                            anyhow!(
                                "Could not get compressed size of frame {frame_id} of {}: {}",
                                path.display(),
                                zstd::zstd_safe::get_error_name(errno)
                            )
                        })?;
                efb.push(offset);

                offset += frame_compressed_size;
            }
            ensure!(
                compressed_frames[offset..].is_empty(),
                "Expected {num_frames} in {}, but there are more ({} unread bytes)",
                path.display(),
                compressed_frames[offset..].len()
            );

            let ef = efb.build_with_seq_and_dict();

            let index_file_path = path.with_extension("frames.ef");
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

pub struct TermStore {
    config: TermStoreConfiguration,
    path: PathBuf,
    partitions: Vec<TermsPartition>,
}

impl TermStore {
    pub fn mmap(path: PathBuf) -> Result<Self> {
        let (config, files) = list_terms_files(&path).with_context(|| {
            format!(
                "Could not open term store configuration from {}",
                path.display()
            )
        })?;

        let partitions = files
            .into_iter()
            .enumerate()
            .map(
                |(partition_id, TermsFile {
                     first_term_id,
                     num_terms,
                     path,
                     compressed_frames,
                 })| {
                    let num_frames = num_terms.div_ceil(config.terms_per_frame);
                    let frames_index_path = path.with_extension("frames.ef");
                    let frames_index = EfSeqDict::mmap(&frames_index_path, Flags::RANDOM_ACCESS)
                        .with_context(|| {
                            format!(
                                "Could not epdeserialize frames index from {}",
                                frames_index_path.display()
                            )
                        })?;
                    ensure!(frames_index.len() == num_frames, "frames_index ({}) of partition {partition_id} does not match expected number of frames ({num_frames})", frames_index.len());

                    Ok(TermsPartition {
                        path,
                        first_term_id,
                        frames_index,
                        compressed_frames,
                    })
                },
            )
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            config,
            path,
            partitions,
        })
    }

    pub fn len(&self) -> usize {
        self.config.num_terms
    }

    pub fn get(&self, id: usize) -> Result<Option<Box<[u8]>>> {
        if id >= self.len() {
            return Ok(None);
        }

        // TODO: add a small cache or something, so we don't have to decompress popular
        // frames every time

        // Compute which partition the term is in
        let num_terms_per_partition = self.config.terms_per_frame * self.config.frames_per_file;
        let partition_id = id / num_terms_per_partition;
        let first_term_in_partition = num_terms_per_partition * partition_id;
        let partition = &self.partitions[partition_id];

        // Compute which frame the term is in
        ensure!(
            first_term_in_partition == partition.first_term_id,
            "Unexpected first_term_id in partition {partition_id} of store {}",
            self.path.display()
        );
        let frame_id = (id - first_term_in_partition) / self.config.terms_per_frame;
        ensure!(
            frame_id < partition.frames_index.len(),
            "Inconsistent partition lengths in terms store {}",
            self.path.display()
        );
        let frame_position = partition.frames_index.get(frame_id);

        // Compute the offset of the term within the frame
        let offset_in_frame = id % self.config.terms_per_frame;

        // Decompress the frame
        let mut decompressed_frame = Vec::with_capacity(
            usize::try_from(zstd::zstd_safe::BLOCKSIZE_MAX)
                .context("decompressed zstd frame size overflows usize")?,
        );
        zstd::zstd_safe::decompress(
            &mut decompressed_frame,
            &partition.compressed_frames[frame_position..],
        )
        .map_err(|errno| {
            anyhow!(
                "Could not decompressed frame at offset {frame_position} of {}: {}",
                partition.path.display(),
                zstd::zstd_safe::get_error_name(errno)
            )
        })?;
        let mut frame_reader = Cursor::new(decompressed_frame);

        // Skip all terms before the one we are looking for, then read the right one
        let mut last_term = None;
        for _ in 0..=offset_in_frame {
            last_term = Some(
                read_length_prefixed_string(&mut frame_reader, |_| None)
                    .with_context(|| {
                        format!(
                            "Could not read string frame at offset {frame_position} of {}",
                            partition.path.display()
                        )
                    })?
                    .with_context(|| {
                        format!(
                            "Frame at offset {frame_position} of {} has fewer terms than expected",
                            partition.path.display()
                        )
                    })?,
            );
        }

        Ok(Some(last_term.expect("Loop didn't run").into()))
    }
}

struct TermsPartition {
    path: PathBuf,
    first_term_id: usize,
    compressed_frames: Mmap,
    frames_index: MemCase<<EfSeqDict as EpDeserializeInner>::DeserType<'static>>,
}
