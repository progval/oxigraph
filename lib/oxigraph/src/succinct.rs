use crate::io::RdfParseError;
use crate::model::{GraphName, NamedOrBlankNode, Quad, Term, Triple};
use crate::storage::numeric_encoder::EncodedTerm;
use anyhow::{Context, Result, anyhow, ensure};
use dashmap::DashSet;
use itertools::Itertools;
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use rustc_hash::FxHashSet;
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Seek, Write};
use std::sync::atomic::{AtomicU64, Ordering};
use tempfile::TempDir;

pub struct TermMphf {}

/// Sorts and deduplicates strings and spills to disk to save memory
struct ExternalSorter {
    tempdirs: Vec<TempDir>,
    sorted_files: Vec<File>,
    max_buffer_size: usize,
    buffer_size: usize,
    buffer: FxHashSet<Box<[u8]>>,
    written_size: usize,
}

impl ExternalSorter {
    pub fn new(max_buffer_size: usize) -> Result<Self> {
        Ok(Self {
            tempdirs: Vec::new(), // Create it only if needed
            sorted_files: Vec::new(),
            buffer: FxHashSet::with_hasher(Default::default()),
            buffer_size: 0,
            max_buffer_size,
            written_size: 0,
        })
    }

    pub fn push_str(&mut self, s: String) -> Result<()> {
        self.push_boxed_bytes(s.into_bytes().into())
    }

    pub fn push_boxed_bytes(&mut self, bytes: Box<[u8]>) -> Result<()> {
        let bytes_len = bytes.len() + 1;
        if self.buffer_size + bytes.len() > self.max_buffer_size {
            self.flush_buffer()?;
        }
        ensure!(
            bytes_len < self.max_buffer_size,
            "String does not fit in {} bytes buffer",
            self.max_buffer_size
        );

        ensure!(!bytes.contains(&b'\0'), "String contains null character");
        if self.buffer.insert(bytes) {
            self.buffer_size += bytes_len;
        }
        Ok(())
    }

    fn flush_buffer(&mut self) -> Result<()> {
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

        // Sort buffer
        let mut sort_buffer: Vec<_> = self.buffer.drain().collect();
        sort_buffer.par_sort_unstable();
        sort_buffer.dedup();

        // Flush buffer to file
        {
            let mut writer = BufWriter::new(&mut file);
            let mut first = true;
            for string in sort_buffer {
                if !first {
                    // String separator
                    writer
                        .write_all(b"\0")
                        .with_context(|| format!("Could not write to {}", path.display()))?;
                    first = false;
                }
                writer
                    .write_all(&string)
                    .with_context(|| format!("Could not write to {}", path.display()))?;
            }
        }

        // Make file readable
        file.rewind()
            .with_context(|| format!("Could not rewind {}", path.display()))?;
        self.sorted_files.push(file);

        // Reset buffer
        self.written_size += self.buffer_size;
        self.buffer_size = 0;

        Ok(())
    }

    /// Consome this external sorter and returns sorted items
    pub fn iter_vec_bytes(self) -> impl Iterator<Item = Result<Vec<u8>, std::io::Error>> {
        let mut buffer: Vec<_> = self.buffer.into_iter().collect();
        buffer.par_sort_unstable();
        self.sorted_files
            .into_iter()
            .map(|file| BufReader::new(file).split(b'\0'))
            .kmerge_by(|left, right| match (left, right) {
                (Ok(left), Ok(right)) => left < right,
                (_, _) => true, // doesn't matter, we are going to error anyway
            })
            // TODO: merge at the same time as the others
            .merge_by(buffer.into_iter().dedup().map(Vec::from).map(Ok),|left, right| match (left, right) {
                (Ok(left), Ok(right)) => left < right,
                (_, _) => true, // doesn't matter, we are going to error anyway
            })
    }

    /// See [`Self::iter_vec_bytes`]
    pub fn iter_boxed_bytes(self) -> impl Iterator<Item = Result<Box<[u8]>, std::io::Error>> {
        self.iter_vec_bytes()
            .map(|item| Ok(item?.into_boxed_slice()))
    }

    pub fn merge(mut self, mut other: Self) -> Result<Self> {
        self.tempdirs.extend(other.tempdirs);
        self.sorted_files.extend(other.sorted_files.into_iter());
        self.written_size += other.written_size;

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

pub fn build_term_mphf(quads: impl ParallelIterator<Item = Result<Quad>>) -> Result<TermMphf> {
    let num_quads = AtomicU64::new(0);

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

    let unique_sorted_quads = quads
        .fold(
            || ExternalSorter::new(100 * 1024 * 1024), // 100MiB in-memory buffer per thread
            |thread_sorter, quad| -> Result<_> {
                let mut thread_sorter: ExternalSorter = thread_sorter?;
                let Quad {
                    subject,
                    predicate,
                    object,
                    graph_name,
                } = quad?;
                thread_sorter.push_str(match subject {
                    NamedOrBlankNode::NamedNode(n) => n.as_str().to_owned(),
                    NamedOrBlankNode::BlankNode(n) => n.as_str().to_owned(),
                })?;
                thread_sorter.push_str(predicate.as_str().to_owned())?;
                thread_sorter.push_str(match graph_name {
                    GraphName::NamedNode(n) => n.as_str().to_owned(),
                    GraphName::BlankNode(n) => n.as_str().to_owned(),
                    GraphName::DefaultGraph => "".to_owned(), // XXX I guess?
                })?;
                push_term(&mut thread_sorter, object)?;

                let current_num_quads = num_quads.fetch_add(1, Ordering::Relaxed) + 1;
                if current_num_quads % 100_000_000 == 0 {
                    eprintln!("Loaded {}M quads", current_num_quads / 1_000_000,);
                }
                Ok(thread_sorter)
            },
        )
        .reduce(
            || ExternalSorter::new(100 * 1024 * 1024), // 100MiB in-memory buffer per thread
            |left, right| left?.merge(right?),
        )?;

    println!(
        "{} unique terms",
        unique_sorted_quads.iter_boxed_bytes().count()
    );

    todo!("build_term_mphf");
}
