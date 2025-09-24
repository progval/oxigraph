use super::sort::ExternalDeduplicatingStringSorter;
use crate::model::{GraphName, NamedOrBlankNode, Quad, Term, Triple};
use anyhow::{Context, Result, anyhow, ensure};
use dsi_progress_logger::{ProgressLog, concurrent_progress_logger, progress_logger};
use epserde::ser::Serialize as EpSerialize;
use itertools::Itertools;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};

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

fn deduplicate_terms(
    quads: impl ParallelIterator<Item = Result<Quad>>,
) -> Result<ExternalDeduplicatingStringSorter> {
    fn push_term(sorter: &mut ExternalDeduplicatingStringSorter, term: Term) -> Result<()> {
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
pub struct TermsStoreConfiguration {
    pub terms_per_frame: usize,
    pub frames_per_file: usize,
    pub num_terms: usize,
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

pub(super) struct TermsFile<D> {
    pub first_term_id: usize,
    pub num_terms: usize,
    pub path: PathBuf,
    pub compressed_frames: D,
}

pub(super) fn list_terms_files(
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
                sux::dict::elias_fano::EliasFanoBuilder::new(num_terms, compressed_frames.len()); // .context("Could not initialize EliasFanoBuilder")?;
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
