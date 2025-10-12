use super::sort::ExternalDeduplicatingStringSorter;
use crate::model::{GraphName, Quad, Term};
use anyhow::{Context, Result, anyhow, bail, ensure};
use dsi_progress_logger::{ProgressLog, concurrent_progress_logger, progress_logger};
use epserde::deser::mem_case::{Flags, MemCase};
use epserde::deser::{Deserialize as EpDeserialize, DeserializeInner as EpDeserializeInner};
use epserde::ser::Serialize as EpSerialize;
use itertools::Itertools;
use lender::{Lender, Lending};
use mmap_rs::Mmap;
use oxrdf::vocab::{rdf, xsd};
use oxrdf::{BlankNode, Literal, NamedNode, NamedNodeRef};
use quick_cache::sync::Cache;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Cursor, Read, Seek, Write};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use sux::dict::elias_fano::{EfSeq, EfSeqDict, EliasFanoBuilder};
use sux::traits::IndexedSeq;
use sux::utils::RewindableIoLender;

// Increasing either these values doesn't give noticeably better compression on wikidata-20240320-truthy-BETA.
pub const DEFAULT_ZSTD_TRAINING_SAMPLES: NonZeroUsize = NonZeroUsize::new(10_000).unwrap();
pub const DEFAULT_ZSTD_DICTIONARY_SIZE: NonZeroUsize = NonZeroUsize::new(1_000_000).unwrap();

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

const TERM_TYPE_NAMED_NODE: u8 = 1;
const TERM_TYPE_BLANK_NODE: u8 = 2;
const TERM_TYPE_LITERAL_SIMPLE: u8 = 4;
const TERM_TYPE_LITERAL_LANGUAGE: u8 = 5;
const TERM_TYPE_LITERAL_TYPED: u8 = 6;
const TERM_TYPE_LITERAL_TYPED_IN_DICT: u8 = 7;

pub fn serialize_term(term: &Term, dictionary: Option<&TermDictionary>) -> Result<Vec<u8>> {
    match term {
        Term::NamedNode(nn) => Ok([TERM_TYPE_NAMED_NODE]
            .into_iter()
            .chain(nn.as_str().as_bytes().into_iter().copied())
            .collect()),
        Term::BlankNode(bn) => Ok([TERM_TYPE_BLANK_NODE]
            .into_iter()
            .chain(bn.as_str().as_bytes().into_iter().copied())
            .collect()),
        Term::Literal(lit) => match (lit.language(), lit.datatype()) {
            (None, xsd::STRING) => Ok([TERM_TYPE_LITERAL_SIMPLE]
                .into_iter()
                .chain(lit.value().as_bytes().into_iter().copied())
                .collect()),
            (Some(lang), rdf::LANG_STRING) => Ok([TERM_TYPE_LITERAL_LANGUAGE]
                .into_iter()
                .chain(lang.as_bytes().into_iter().copied())
                .chain(lit.value().as_bytes().into_iter().copied())
                .chain(
                    u16::try_from(lang.as_bytes().len())
                        .context("Language is 2^16 bytes or longer")?
                        .to_be_bytes()
                        .into_iter(),
                )
                .collect()),
            (None, datatype) => match dictionary
                .and_then(|dict| dict.get_datatype_id(datatype).transpose())
                .transpose()?
            {
                Some(datatype_id) => Ok([TERM_TYPE_LITERAL_TYPED_IN_DICT]
                    .into_iter()
                    .chain(lit.value().as_bytes().into_iter().copied())
                    .chain(datatype_id.to_be_bytes().into_iter())
                    .collect()),
                None => Ok([TERM_TYPE_LITERAL_TYPED]
                    .into_iter()
                    .chain(lit.value().as_bytes().into_iter().copied())
                    .chain(datatype.as_str().as_bytes().into_iter().copied())
                    .chain(datatype.as_str().as_bytes().len().to_be_bytes().into_iter())
                    .collect()),
            },
            (Some(lang), datatype) => bail!(
                "{term:?} has both a language ({lang:?}) and a non-langString type ({datatype:?})"
            ),
        },
        #[cfg(feature = "rdf-12")]
        Term::Triple(_) => todo!("Term::Triple"),
    }
}

pub fn deserialize_term(bytes: &[u8], dictionary: Option<&TermDictionary>) -> Result<Term> {
    let &tag = bytes
        .get(0)
        .context("Empty byte string is not a valid term")?;
    Ok(match tag {
        TERM_TYPE_NAMED_NODE => Term::NamedNode(NamedNode::new_unchecked(
            str::from_utf8(&bytes[1..]).context("Non-UTF8 NamedNode in store")?,
        )),
        TERM_TYPE_BLANK_NODE => Term::BlankNode(BlankNode::new_unchecked(
            str::from_utf8(&bytes[1..]).context("Non-UTF8 BlankNode in store")?,
        )),
        TERM_TYPE_LITERAL_SIMPLE => Term::Literal(Literal::new_simple_literal(
            str::from_utf8(&bytes[1..]).context("Non-UTF8 Literal value in store")?,
        )),
        TERM_TYPE_LITERAL_LANGUAGE => {
            let lang_length_offset = bytes
                .len()
                .checked_sub(size_of::<u16>())
                .context("Language tag literal in store is smaller than size_of<u16>()+1")?;
            let lang_length = usize::from(u16::from_be_bytes(
                bytes[lang_length_offset..].try_into().unwrap(),
            ));

            Term::Literal(Literal::new_language_tagged_literal_unchecked(
                str::from_utf8(&bytes[1 + lang_length..lang_length_offset])
                    .context("Non-UTF8 Literal value in store")?,
                str::from_utf8(&bytes[1..1 + lang_length])
                    .context("Non-UTF8 Literal language in store")?,
            ))
        }
        TERM_TYPE_LITERAL_TYPED_IN_DICT => {
            let type_id_size = size_of::<u32>();
            let type_id_offset = bytes.len().checked_sub(type_id_size).context(
                "Datatype-in-dict tagged literal in store is smaller than size_of<u32>()+1",
            )?;
            let type_id = u32::from_be_bytes(bytes[type_id_offset..].try_into().unwrap());
            let type_ = dictionary
                .context("Terms store was compressed with a dictionary, but no dictionary was given to decompress")?
                .get_datatype(type_id)
                .with_context(|| format!("Unknown datatype id: {type_id}"))?;
            Term::Literal(Literal::new_typed_literal(
                str::from_utf8(&bytes[1..type_id_offset])
                    .context("Non-UTF8 Literal value in store")?,
                type_,
            ))
        }
        TERM_TYPE_LITERAL_TYPED => {
            let type_length_offset = bytes
                .len()
                .checked_sub(size_of::<u32>())
                .context("Language tag literal in store is smaller than size_of<u32>()+1")?;
            let type_length = u32::from_be_bytes(bytes[type_length_offset..].try_into().unwrap());
            let type_length = usize::try_from(type_length).context("Datatype overflowed usize")?;
            let type_offset = type_length_offset
                .checked_sub(type_length)
                .context("Inconsistent type_length")?;

            Term::Literal(Literal::new_typed_literal(
                str::from_utf8(&bytes[type_offset..type_length_offset])
                    .context("Non-UTF8 Literal value in store")?,
                NamedNode::new_unchecked(
                    str::from_utf8(&bytes[1..type_offset])
                        .context("Non-UTF8 Literal type in store")?,
                ),
            ))
        }
        _ => bail!("Unknown term tag in store: 0x{tag:x}"),
    })
}

pub fn serialize_graph_name(graph_name: &GraphName) -> Result<Vec<u8>> {
    match graph_name {
        GraphName::DefaultGraph => {
            // The empty string is very handy in datasets that have most of their quads
            // in the default graph, because it comes first in the sorted list of terms,
            // which means that the DefaultGraph gets hashed to id 0.
            // And because sorted quad files have to write the graph at the beginning
            // of each frame, the shorter the graph id is, the better.
            // And because we use gamma coding
            // (https://docs.rs/dsi-bitstream/latest/dsi_bitstream/codes/index.html),
            // 0 is encoded as a single bit whereas any other value takes at least four bits.
            Ok(Vec::new())
        }
        GraphName::NamedNode(nn) => Ok([TERM_TYPE_NAMED_NODE]
            .into_iter()
            .chain(nn.as_str().as_bytes().into_iter().copied())
            .collect()),
        GraphName::BlankNode(bn) => Ok([TERM_TYPE_BLANK_NODE]
            .into_iter()
            .chain(bn.as_str().as_bytes().into_iter().copied())
            .collect()),
    }
}

pub fn deserialize_graph_name(bytes: &[u8]) -> Result<GraphName> {
    match bytes.get(0).copied() {
        None => Ok(GraphName::DefaultGraph),
        Some(TERM_TYPE_NAMED_NODE) => Ok(GraphName::NamedNode(NamedNode::new_unchecked(
            str::from_utf8(&bytes[1..]).context("Non-UTF8 NamedNode in store")?,
        ))),
        Some(TERM_TYPE_BLANK_NODE) => Ok(GraphName::BlankNode(BlankNode::new_unchecked(
            str::from_utf8(&bytes[1..]).context("Non-UTF8 NamedNode in store")?,
        ))),
        Some(tag) => bail!("Unknown graph_name tag in store: 0x{tag:x}"),
    }
}

pub struct TermDictionary {
    name: String,
    datatypes: Mmap,
    datatypes_offsets: MemCase<<EfSeq as EpDeserializeInner>::DeserType<'static>>,
}

impl TermDictionary {
    pub fn mmap(path: &Path) -> Result<Self> {
        let datatypes_path = path.join("datatypes.strings");
        let datatypes_offsets_path = path.join("datatypes.ef");

        let datatypes_file = File::open(&datatypes_path)
            .with_context(|| format!("Could not open {}", datatypes_path.display()))?;
        let datatypes_file_len = usize::try_from(
            datatypes_file
                .metadata()
                .with_context(|| format!("Could not stat {}", datatypes_path.display()))?
                .len(),
        )
        .context("File is larger than usize")?;
        Ok(Self {
            name: path
                .file_name()
                .context("TermDictionary path cannot be root")?
                .to_str()
                .context("TermDictionary file name is not valid UTF-8")?
                .to_string(),
            datatypes: unsafe {
                mmap_rs::MmapOptions::new(datatypes_file_len)
                    .context("Could not initialize mmap")?
                    .with_file(&datatypes_file, 0)
                    .map()
                    .with_context(|| format!("Could not mmap {}", datatypes_path.display()))?
            },
            datatypes_offsets: EfSeq::mmap(&datatypes_offsets_path, Flags::RANDOM_ACCESS)
                .with_context(|| {
                    format!(
                        "Could not epdeserialize frames index from {}",
                        datatypes_offsets_path.display()
                    )
                })?,
        })
    }

    /// Reads all files in the store and writes a new dictionary that can be used to compress it
    /// more efficiently, and returns its file name.
    ///
    /// Call [`recompress_with_dictionary`] to use that dictionary.
    pub fn train(dir: &Path) -> Result<Self> {
        let (config, terms_files) = list_terms_files(dir)?;

        let dictionary = config.get_dictionary(dir)?;

        let mut pl = concurrent_progress_logger!(
            item_name = "term",
            display_memory = true,
            local_speed = true,
            expected_updates = Some(config.num_terms),
        );
        pl.start("Reading terms...");

        let mut unique_sorted_datatypes = terms_files
            .into_par_iter()
            .map_with(pl.clone(), |pl, terms_file| {
                let TermsFile {
                    first_term_id: _,
                   num_terms: _,
                    path: _,
                    compressed_frames,
                } = terms_file;
                let compressed_frames = compressed_frames.as_ref();

                let max_buffer_size = 100_000_000;
                let max_num_files = 10;

                let mut namednode_sorter  = ExternalDeduplicatingStringSorter::new(
                    max_buffer_size, max_num_files).context("Could not initialize sorter")?;

                let mut push_namednode = |nn: NamedNodeRef<'_>| namednode_sorter.push_boxed_bytes(nn.as_str().as_bytes().to_vec().into());
                push_namednode(xsd::STRING)?;

                FrameLender::new(compressed_frames, config.terms_per_frame)?
                    .try_for_each(|term| {
                    let term = term?;
                        if term.is_empty() {
                            // FIXME: that's GraphName::DefaultGraph because we currently store
                            // graph names in the terms store, but we shouldn't.
                            pl.light_update();
                            return Ok(());
                        }
                        let term = deserialize_term(term, dictionary.as_ref()).context("Could not deserialize term from store")?;
                        match &term {
                            Term::NamedNode(_) => (),
                            Term::BlankNode(_) => (),
                            Term::Literal(lit) => match (lit.language(), lit.datatype()) {
                                (None, xsd::STRING) => (),
                                (Some(_lang), rdf::LANG_STRING) =>(),
                                (None, datatype) => push_namednode(datatype)?,
                                (Some(lang), datatype) => bail!(
                                    "{term:?} has both a language ({lang:?}) and a non-langString type ({datatype:?})"
                                ),
                            },
                            #[cfg(feature = "rdf-12")]
                            Term::Triple(_) => todo!("Term::Triple"),
                        }
                        pl.light_update();
                        Ok(())
                    })?;

                Ok(namednode_sorter)
            })
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
        pl.done();

        let dict_dir = tempfile::tempdir_in(dir).with_context(|| {
            format!("Could not create temporary directory in {}", dir.display())
        })?;

        // Write datatypes
        let mut pl = progress_logger!(
            item_name = "datatype",
            display_memory = true,
            local_speed = true,
        );
        pl.start("Writing datatypes...");
        let datatypes_path = dict_dir.path().join("datatypes.strings");
        let file = File::create_new(&datatypes_path)
            .with_context(|| format!("Could not create {}", datatypes_path.display()))?;
        let mut writer = BufWriter::new(file);
        let mut num_datatypes = 0usize;
        let mut file_len = 0usize;
        for datatype in unique_sorted_datatypes
            .drain_boxed_bytes()
            .context("Could not read deduplicated datatypes")?
        {
            let datatype = datatype?;
            write_length_prefixed_string(&mut writer, &datatype, &datatypes_path)?;

            num_datatypes = num_datatypes
                .checked_add(1)
                .context("Number of data types overflows usize")?;
            file_len = file_len
                .checked_add(size_of::<u32>() + datatype.len())
                .context("datatypes dictionary size overflowed usize")?;
            pl.light_update();
        }
        let mut file = writer
            .into_inner()
            .with_context(|| format!("Could not flush to {}", datatypes_path.display()))?;
        file.flush()
            .with_context(|| format!("Could not flush to {}", datatypes_path.display()))?;
        pl.done();

        // Index the data types
        let mut pl = progress_logger!(
            item_name = "datatype",
            display_memory = true,
            local_speed = true,
            expected_updates = Some(num_datatypes),
        );
        pl.start("Indexing datatypes...");
        file.rewind()
            .with_context(|| format!("Could not rewind {}", datatypes_path.display()))?;
        let mut reader = BufReader::new(file);
        let mut efb = EliasFanoBuilder::new(num_datatypes + 1, file_len);
        efb.push(0);
        let mut offset = 0usize;
        for i in 0..num_datatypes {
            let datatype = read_length_prefixed_string(&mut reader, |reader| {
                Some(reader.stream_position().map_err(Into::into))
            })
            .with_context(|| format!("Could not read {i}th datatype"))?
            .context("Unexpected end of file")?;

            // write to the index
            let len = datatype.len();
            offset = offset + size_of::<u32>() + len; // can't overflow, we already checked the file len
            efb.push(offset);

            pl.light_update()
        }

        let ef = efb.build_with_seq();
        let datatypes_index_path = dict_dir.path().join("datatypes.ef");
        let mut index_file = File::create(&datatypes_index_path)
            .with_context(|| format!("Could not create {}", datatypes_index_path.display()))?;
        ef.serialize(&mut index_file).with_context(|| {
            format!(
                "Could not write Elias-Fano index to {}",
                datatypes_index_path.display()
            )
        })?;
        pl.done();

        let tmp_dict_path = dict_dir.keep();
        let dict_path = dir.join("dict");
        std::fs::rename(&tmp_dict_path, &dict_path).context("Could not rename dict")?;

        let dict = Self::mmap(&dict_path)
            .with_context(|| format!("Could not mmap dictionary from {}", dict_path.display()))?;
        Ok(dict)
    }

    fn get_datatype_id(&self, datatype: NamedNodeRef<'_>) -> Result<Option<u32>> {
        // TODO: use VFunc instead of bisection
        let mut left = 0;
        let mut right = u32::try_from(self.datatypes_offsets.len() - 1)
            .context("Number of datatypes overflowed u32")?;
        while left < right {
            let pivot_id = (left + right) / 2;
            assert!(
                pivot_id > left || pivot_id < right,
                "{left} | {pivot_id} | {right}"
            );
            let pivot = self.get_datatype(pivot_id)?;
            if pivot == datatype {
                return Ok(Some(pivot_id));
            } else if pivot < datatype {
                if left == pivot_id {
                    break;
                }
                assert!(left < pivot_id);
                left = pivot_id;
            } else {
                if right == pivot_id {
                    break;
                }
                assert!(right > pivot_id);
                right = pivot_id;
            }
        }
        if self.get_datatype(left)? == datatype {
            Ok(Some(left))
        } else {
            Ok(None)
        }
    }
    fn get_datatype(&self, id: u32) -> Result<NamedNode> {
        let id = usize::try_from(id).with_context(|| format!("Invalid datatype id: {id}"))?;
        ensure!(
            id < self.datatypes_offsets.len(),
            "Invalid datatype id: {id}"
        );

        let offset = self.datatypes_offsets.get(id);
        let bytes = read_length_prefixed_string(&mut Cursor::new(&self.datatypes), |_| {
            Some(u64::try_from(offset).context("Offset overflowed u64"))
        })
        .context("Could not get datatype")?
        .context("Unexpected end of file")?;
        Ok(NamedNode::new_unchecked(
            str::from_utf8(&bytes).context("Non-UTF8 NamedNode in store")?,
        ))
    }
}

pub fn recompress_with_dictionary(dir: &Path, dictionary: &TermDictionary) -> Result<()> {
    let (mut config, terms_files) = list_terms_files(dir)?;

    let old_dictionary = config.get_dictionary(dir)?;

    let mut pl = concurrent_progress_logger!(
        item_name = "term",
        display_memory = true,
        local_speed = true,
        expected_updates = Some(config.num_terms),
    );
    pl.start("Recompressing terms...");

    let recompressed_files = terms_files
        .par_iter()
        .map_with(pl.clone(), |pl, terms_file| {
            let TermsFile {
                first_term_id: _,
                num_terms: _,
                path,
                compressed_frames,
            } = terms_file;
            let compressed_frames = compressed_frames.as_ref();

            let recompressed_file = tempfile::NamedTempFile::with_prefix_in(
                path.file_name().expect("File has no name"),
                path.parent().expect("File has no parent directory"),
            )
            .context("Could not create temporary file")?;
            let mut writer = BufWriter::new(recompressed_file);

            let mut buf = Vec::with_capacity(config.terms_per_frame.into());
            FrameLender::new(compressed_frames, config.terms_per_frame)?.try_for_each(
                |term| -> Result<_> {
                    let term = term?;
                    if term.is_empty() {
                        // FIXME: that's GraphName::DefaultGraph because we currently store
                        // graph names in the terms store, but we shouldn't.
                        buf.push(term.into());
                    } else {
                        let term = deserialize_term(term, old_dictionary.as_ref())
                            .context("Could not deserialize term from store")?;
                        buf.push(serialize_term(&term, Some(dictionary))?.into());
                    }
                    if buf.len() == usize::from(config.terms_per_frame) {
                        write_frame(&mut writer, buf.drain(..).map(Ok))?;
                    }
                    pl.light_update();
                    Ok(())
                },
            )?;
            if !buf.is_empty() {
                write_frame(&mut writer, buf.drain(..).map(Ok))?;
            }

            let mut recompressed_file = writer
                .into_inner()
                .with_context(|| format!("Could not flush to {}", path.display()))?;
            recompressed_file
                .flush()
                .with_context(|| format!("Could not flush to {}", path.display()))?;

            Ok(recompressed_file)
        })
        .collect::<Result<Vec<_>>>()?;

    pl.done();

    log::info!("Invalidating index files...");
    for original_file in &terms_files {
        let index_file_path = original_file.path.with_extension("frames.ef");
        if std::fs::exists(&index_file_path)
            .with_context(|| format!("Could not check if {} exists", index_file_path.display()))?
        {
            std::fs::remove_file(&index_file_path).with_context(|| {
                format!("Could not remove index file {}", index_file_path.display())
            })?;
        }
    }

    log::info!("Committing recompressed files...");
    config.dictionary_filename = Some(dictionary.name.clone());
    let config_path = dir.join("config.json");
    let config_file = File::create(&config_path)
        .with_context(|| format!("Could not create {}", config_path.display()))?;
    serde_json::to_writer_pretty(config_file, &config)
        .context("Could not write terms store config")?;

    for (original_file, recompressed_file) in
        terms_files.into_iter().zip(recompressed_files.into_iter())
    {
        recompressed_file
            .persist(original_file.path)
            .context("Could not persist recompressed file.")?;
    }

    Ok(())
}

fn deduplicate_terms(
    quads: impl ParallelIterator<Item = Result<Quad>>,
) -> Result<ExternalDeduplicatingStringSorter> {
    fn push_term(sorter: &mut ExternalDeduplicatingStringSorter, term: Term) -> Result<()> {
        sorter.push_boxed_bytes(serialize_term(&term, None)?.into_boxed_slice())
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
    pub terms_per_frame: NonZeroUsize,
    pub frames_per_file: usize,
    pub num_terms: usize,
    pub dictionary_filename: Option<String>,
}

impl TermStoreConfiguration {
    pub fn get_dictionary(&self, path: impl AsRef<Path>) -> Result<Option<TermDictionary>> {
        self.dictionary_filename
            .as_ref()
            .map(|dict_name| TermDictionary::mmap(&path.as_ref().join(dict_name)))
            .transpose()
            .context("Could not mmap dictionary")
    }
}

pub fn write_unique_terms(
    quads: impl ParallelIterator<Item = Result<Quad>>,
    dir: &Path,
    approx_num_quads: Option<usize>,
) -> Result<()> {
    let mut config = TermStoreConfiguration {
        terms_per_frame: NonZeroUsize::new(16).unwrap(),
        frames_per_file: 1024 * 1024,
        num_terms: 0,
        dictionary_filename: None,
    };

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
        .chunks(usize::from(config.terms_per_frame) * config.frames_per_file);
    for (file_id, big_chunk) in term_chunk_iterator.into_iter().enumerate() {
        let strings_path = dir.join(format!("{file_id:0>10}.strings"));
        let strings_file = File::create(&strings_path)
            .with_context(|| format!("Could not create {}", strings_path.display()))?;

        let mut writer = BufWriter::new(strings_file);
        let term_chunk_iterator = big_chunk.chunks(config.terms_per_frame.into());
        for (frame_id, small_chunk) in term_chunk_iterator.into_iter().enumerate() {
            let chunk_len = write_frame(&mut writer, small_chunk).with_context(|| {
                format!(
                    "Could not write frame {frame_id} to {}",
                    strings_path.display()
                )
            })?;
            pl.update_with_count(chunk_len);
            config.num_terms += chunk_len;
        }
        writer
            .into_inner()
            .with_context(|| format!("Could not flush to {}", strings_path.display()))?
            .flush()
            .with_context(|| format!("Could not flush to {}", strings_path.display()))?;
    }
    pl.done();

    let config_path = dir.join("config.json");
    let config_file = File::create(&config_path)
        .with_context(|| format!("Could not create {}", config_path.display()))?;
    serde_json::to_writer_pretty(config_file, &config)
        .context("Could not write terms store config")?;

    Ok(())
}

pub fn write_frame(
    writer: &mut impl Write,
    terms: impl Iterator<Item = Result<Box<[u8]>>>,
) -> Result<usize> {
    let mut previous_string = Box::<[u8]>::default();
    let mut num_strings = 0;
    for string in terms {
        let string = string?;

        // compute number of bytes shared between this string and the previous one
        let longest_common_prefix = previous_string
            .iter()
            .zip(string.iter())
            .take_while(|(c1, c2)| c1 == c2)
            .count();

        // compute number of bytes to trim at the end of the previous one, to use as prefix
        // of this one
        let mut buf = [0; 8];
        let num_discard_bytes: u32 = previous_string
            .len()
            .checked_sub(longest_common_prefix)
            .expect("Incorrect longest_common_prefix")
            .try_into()
            .context("String is 2^32 bytes or longer")?;

        // write how many bytes to discard from the previous string
        let varint_bytes = tiny_varint::encode(num_discard_bytes, &mut buf)
            .map_err(|_| anyhow!("varint of u32 does not fit in 8 bytes"))?;
        writer
            .write_all(&buf[0..varint_bytes])
            .context("Could not write")?;

        // write how many bytes are in the suffix of this string
        let num_new_bytes: u32 = string
            .len()
            .checked_sub(longest_common_prefix)
            .expect("Incorrect longest_common_prefix")
            .try_into()
            .context("String is 2^32 bytes or longer")?;
        let varint_bytes = tiny_varint::encode(num_new_bytes, &mut buf)
            .map_err(|_| anyhow!("varint of u32 does not fit in 8 bytes"))?;
        writer
            .write_all(&buf[0..varint_bytes])
            .context("Could not write")?;

        // write the new string minus the shared prefix
        writer
            .write_all(&string[longest_common_prefix..])
            .context("Could not write")?;

        previous_string = string;
        num_strings += 1;
    }

    Ok(num_strings)
}

pub fn get_from_frame(
    frame: &[u8],
    terms_per_frame: NonZeroUsize,
    position: usize,
) -> Result<Box<[u8]>> {
    Ok(FrameLender::new(frame, terms_per_frame)?
        .skip(position)
        .next()
        .with_context(|| format!("Frame has fewer than {} strings", position + 1))?
        .context("Could not decode frame")?
        .into())
}

pub struct FrameLender<'frame> {
    // TODO: rename
    all_data: &'frame [u8],
    data: &'frame [u8],
    previous_string: Vec<u8>,
    errored: Option<String>,
    max_strings_per_frame: NonZeroUsize,
    remaining_strings_in_frame: usize,
}

impl<'frame> FrameLender<'frame> {
    pub fn new(frame: &'frame [u8], max_strings_per_frame: NonZeroUsize) -> Result<Self> {
        Ok(FrameLender {
            all_data: frame,
            data: frame,
            previous_string: Vec::new(),
            errored: None,
            max_strings_per_frame,
            remaining_strings_in_frame: max_strings_per_frame.into(),
        })
    }
}

impl<'frame, 'lend> Lending<'lend> for FrameLender<'frame> {
    type Lend = Result<&'lend [u8], anyhow::Error>;
}

impl<'frame> Lender for FrameLender<'frame> {
    fn next(&mut self) -> Option<<Self as Lending<'_>>::Lend> {
        if self.data.is_empty() {
            return None;
        }
        if let Some(e) = &self.errored {
            return Some(Err(anyhow!("Previous iteration failed: {e}")));
        }

        if self.remaining_strings_in_frame == 0 {
            // new frame
            self.previous_string.clear();
            self.remaining_strings_in_frame = self.max_strings_per_frame.into();
        }
        self.remaining_strings_in_frame -= 1;

        match (|| {
            let (num_discard_bytes, varint_size) = tiny_varint::decode::<u32>(self.data)
                .map_err(|e| anyhow!("Could not decode num_discard_bytes: {e:?}"))?;
            let num_discard_bytes: usize = num_discard_bytes
                .try_into()
                .context("String size overflows usize")?;
            self.data = &self.data[varint_size..];

            let (num_new_bytes, varint_size) = tiny_varint::decode::<u32>(self.data)
                .map_err(|e| anyhow!("Could not decode num_new_bytes: {e:?}"))?;
            let num_new_bytes: usize = num_new_bytes
                .try_into()
                .context("String size overflows usize")?;
            self.data = &self.data[varint_size..];

            self.previous_string.resize(
                self.previous_string
                    .len()
                    .checked_sub(num_discard_bytes)
                    .context(
                        "num_discard_bytes is greater than the length of the previous string",
                    )?,
                0,
            );
            self.previous_string.extend(&self.data[..num_new_bytes]);

            ensure!(
                self.data.len() >= num_new_bytes,
                "Fewer remaining bytes than num_new_bytes"
            );
            self.data = &self.data[num_new_bytes..];

            Ok(())
        })() {
            Ok(()) => Some(Ok(&self.previous_string[..])),
            Err(e) => {
                self.errored = Some(format!("{e}")); // anyhow::Error does not impl Clone
                Some(Err(e))
            }
        }
    }
}
impl<'frame> RewindableIoLender<[u8]> for FrameLender<'frame> {
    type Error = anyhow::Error;

    fn rewind(mut self) -> Result<Self, Self::Error> {
        self.data = self.all_data;
        self.errored = None;
        self.previous_string.clear();
        self.remaining_strings_in_frame = self.max_strings_per_frame.into();
        Ok(self)
    }
}

pub fn get_frame_size(frame: &[u8], max_num_strings: usize) -> Result<usize> {
    let mut data = frame;

    for _ in 0..max_num_strings {
        if data.len() == 0 {
            // end of file
            break;
        }

        let (_num_discard_bytes, varint_size) = tiny_varint::decode::<u32>(data)
            .map_err(|e| anyhow!("Could not decode num_discard_bytes: {e:?}"))?;
        data = &data[varint_size..];

        let (num_new_bytes, varint_size) = tiny_varint::decode::<u32>(data)
            .map_err(|e| anyhow!("Could not decode num_new_bytes: {e:?}"))?;
        let num_new_bytes: usize = num_new_bytes
            .try_into()
            .context("String size overflows usize")?;
        data = &data[varint_size..];

        ensure!(
            data.len() >= num_new_bytes,
            "Fewer remaining bytes than num_new_bytes"
        );
        data = &data[num_new_bytes..];
    }

    Ok(frame
        .len()
        .checked_sub(data.len())
        .expect("data is longer than frame, but it should be a subslice of it"))
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
        .filter(|entry| entry.path().extension() == Some("strings".as_ref()))
        .collect::<Vec<_>>();
    entries.sort_by_key(|entry| entry.path());

    let terms_per_file = usize::from(config.terms_per_frame) * config.frames_per_file;
    let num_files = config.num_terms.div_ceil(terms_per_file);
    ensure!(
        num_files == entries.len(),
        "Inconsistent number of .strings files in {}: expected {num_files}, got {}",
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
        expected_updates = Some(
            terms_files
                .iter()
                .map(|tf| tf.num_terms.div_ceil(config.terms_per_frame.into()))
                .sum()
        ),
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
            let num_frames = num_terms.div_ceil(config.terms_per_frame.into());
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
                    get_frame_size(&compressed_frames[offset..], config.terms_per_frame.into())
                        .with_context(|| {
                            format!(
                                "Could not get compressed size of frame {frame_id} of {}",
                                path.display(),
                            )
                        })?;
                efb.push(offset);
                pl.light_update();

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
                    let num_frames = num_terms.div_ceil(config.terms_per_frame.into());
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
            path,
            partitions,
            config,
        })
    }

    pub fn len(&self) -> usize {
        self.config.num_terms
    }

    fn get_frame(&self, partition_id: usize, frame_id: usize) -> Result<&[u8]> {
        let partition = &self.partitions[partition_id];
        let frame_position = partition.frames_index.get(frame_id);

        Ok(&partition.compressed_frames[frame_position..])
    }

    pub fn get(&self, id: usize) -> Result<Option<Box<[u8]>>> {
        if id >= self.len() {
            return Ok(None);
        }

        // TODO: add a small cache or something, so we don't have to decompress popular
        // frames every time

        // Compute which partition the term is in
        let num_terms_per_partition =
            usize::from(self.config.terms_per_frame) * self.config.frames_per_file;
        let partition_id = id / num_terms_per_partition;
        let first_term_in_partition = num_terms_per_partition * partition_id;
        let partition = &self.partitions[partition_id];

        // Compute which frame the term is in
        ensure!(
            first_term_in_partition == partition.first_term_id,
            "Unexpected first_term_id in partition {partition_id} of store {}",
            self.path.display()
        );
        let frame_id = (id - first_term_in_partition) / usize::from(self.config.terms_per_frame);
        ensure!(
            frame_id < partition.frames_index.len(),
            "Inconsistent partition lengths in terms store {}",
            self.path.display()
        );

        // Compute the offset of the term within the frame
        let offset_in_frame = id % usize::from(self.config.terms_per_frame);

        Ok(Some(
            get_from_frame(
                self.get_frame(partition_id, frame_id)?,
                self.config.terms_per_frame,
                offset_in_frame,
            )
            .context("Frame is shorter than expected")?,
        ))
    }
}

struct TermsPartition {
    path: PathBuf,
    first_term_id: usize,
    compressed_frames: Mmap,
    frames_index: MemCase<<EfSeqDict as EpDeserializeInner>::DeserType<'static>>,
}
