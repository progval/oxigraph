use super::quads_store::{QuadOrder, QuadStoreConfiguration};
use super::{quads_store, secondary_indexes, terms_mphf, terms_store};
use crate::io::{RdfFormat, RdfParseError, RdfParser};
use crate::model::{NamedNode, Quad};
use anyhow::{Context, Result, bail, ensure};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader, Cursor, Read};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

pub struct DatabaseBuilder {
    parse_quad_args: Option<ParseQuadsArgs>,
    location: PathBuf,
    approx_num_quads: Option<usize>,
    approx_quads_per_file: Option<usize>,
    rebuild: bool,
    rdf_format_from_path: fn(&Path) -> Result<RdfFormat>,
    quad_orders: Vec<QuadOrder>,
}

#[derive(Clone, Default)]
pub struct ParseQuadsArgs {
    pub file: Vec<PathBuf>,
    pub format: Option<RdfFormat>,
    pub base: Option<String>,
    pub lenient: bool,
    pub parallel_parser: bool,
    pub graph: Option<String>,
}

fn default_rdf_format_from_path(_: &Path) -> Result<RdfFormat> {
    bail!("Guessing RDF format from path is not enabled")
}

impl DatabaseBuilder {
    pub fn new(location: PathBuf) -> Self {
        Self {
            location,
            parse_quad_args: None,
            approx_num_quads: None,
            approx_quads_per_file: None,
            rebuild: false,
            rdf_format_from_path: default_rdf_format_from_path,
            quad_orders: vec![QuadOrder::Spog, QuadOrder::Opsg],
        }
    }

    pub fn with_quad_orders(self, quad_orders: Vec<QuadOrder>) -> Self {
        Self {
            quad_orders,
            ..self
        }
    }

    pub fn with_approx_quads_per_file(self, approx_quads_per_file: Option<usize>) -> Self {
        Self {
            approx_quads_per_file,
            ..self
        }
    }

    pub fn with_approx_num_quads(self, approx_quads_per_file: Option<usize>) -> Self {
        Self {
            approx_quads_per_file,
            ..self
        }
    }

    /// Whether to regenerate files that already exist
    pub fn with_rebuild(self, rebuild: bool) -> Self {
        Self { rebuild, ..self }
    }

    pub fn with_parse_quad_args(self, parse_quad_args: Option<ParseQuadsArgs>) -> Self {
        Self {
            parse_quad_args,
            ..self
        }
    }
    pub fn with_rdf_format_from_path(
        self,
        rdf_format_from_path: fn(&Path) -> Result<RdfFormat>,
    ) -> Self {
        Self {
            rdf_format_from_path,
            ..self
        }
    }

    pub fn terms_path(&self) -> PathBuf {
        self.location.join("terms")
    }

    fn compute_approx_num_quads(&self) -> Option<usize> {
        if let Some(approx_num_quads) = self.approx_num_quads {
            return Some(approx_num_quads);
        }
        if let Some(approx_quads_per_file) = self.approx_quads_per_file {
            if let Some(parse_quad_args) = &self.parse_quad_args {
                return Some(approx_quads_per_file * parse_quad_args.file.len());
            }
        }
        None
    }

    pub fn build_all(&self) -> Result<()> {
        ensure!(
            self.parse_quad_args.is_some(),
            "Cannot build from scratch without providing parse_quad_args"
        );

        log::info!("Extracting terms...");
        self.extract_terms().context("Could not extract terms")?;

        log::info!("Training zstd dictionary on terms...");
        self.recompress_terms(
            terms_store::DEFAULT_ZSTD_TRAINING_SAMPLES,
            terms_store::DEFAULT_ZSTD_DICTIONARY_SIZE,
        )
        .context("Could not extract terms")?;

        log::info!("Indexing terms...");
        self.index_terms().context("Could not index terms")?;

        log::info!("Building term MPHF...");
        self.build_terms_mphf()
            .context("Could not build terms MPHF")?;

        self.compress_all_quads_stores()?;

        for &quad_order in &self.quad_orders {
            log::info!("Building first-term index on {quad_order} quad store");
            self.index_quad_store(quad_order)
                .with_context(|| format!("Could not index {quad_order} quad store"))?;
        }

        // we only need one
        if let Some(quad_order) = self.quad_orders.first() {
            self.index_by_second_term(*quad_order).with_context(|| {
                format!("Could not index by second term using {quad_order} quad store")
            })?;
        }

        Ok(())
    }

    pub fn extract_terms(&self) -> Result<()> {
        if !self.rebuild && self.terms_path().join("config.json").exists() {
            log::info!("Skipping terms extraction, already done.");
            return Ok(());
        }
        let parse_quad_args = self
            .parse_quad_args
            .as_ref()
            .context("parse_quad_args not set")?;
        if !self.location.exists() {
            std::fs::create_dir(&self.location)
                .with_context(|| format!("Could not create {}", self.location.display()))?;
        }
        if parse_quad_args.parallel_parser {
            // parse in parallel, process in parallel
            terms_store::write_unique_terms(
                get_parallel_iterator_from_parallel_parsers(
                    parse_quad_args,
                    self.rdf_format_from_path,
                )?,
                &self.terms_path(),
                self.compute_approx_num_quads(),
            )
            .context("Could not deduplicate or write terms")
        } else {
            // parse sequentially, process in parallel
            terms_store::write_unique_terms(
                get_parallel_iterator_from_sequential_parsers(
                    parse_quad_args,
                    self.rdf_format_from_path,
                )?,
                &self.terms_path(),
                self.compute_approx_num_quads(),
            )
            .context("Could not deduplicate or write terms")
        }
    }

    pub fn recompress_terms(
        &self,
        samples: NonZeroUsize,
        max_dictionary_size: NonZeroUsize,
    ) -> Result<()> {
        let dictionary_name =
            terms_store::train_zstd_dictionary(&self.terms_path(), samples, max_dictionary_size)?;
        terms_store::recompress_with_dictionary(&self.terms_path(), dictionary_name)?;
        Ok(())
    }

    pub fn index_terms(&self) -> Result<()> {
        terms_store::index_terms(&self.terms_path()).context("Could not index terms")
    }

    pub fn build_terms_mphf(&self) -> Result<()> {
        let mphf_path = self.location.join("terms_mphf");
        if !self.rebuild && mphf_path.exists() {
            log::info!("Skipping MPHF construction, already done.");
            return Ok(());
        }
        let mphf = terms_mphf::build_terms_mphf(&self.terms_path())
            .context("Could not build terms MPHF")?;
        mphf.serialize(&mphf_path)
            .with_context(|| format!("Could not write terms MPHF to {}", mphf_path.display()))
    }

    pub fn compress_all_quads_stores(&self) -> Result<()> {
        let mut already_compressed_order = None;
        for &order in &self.quad_orders {
            if let Some(already_compressed_order) = already_compressed_order {
                // Prefer reading from an already compressed quad store than from
                // input files, it's faster because we don't have to parse them.
                log::info!(
                    "Creating {order} quads store (from {already_compressed_order} quads)..."
                );
                self.recompress_quad_store(already_compressed_order, order)
                    .with_context(|| format!("Could not compress {order} quads"))?;
            } else {
                log::info!("Creating {order} quad store...");
                self.compress_quad_store(order)
                    .with_context(|| format!("Could not compress {order} quads"))?;
                already_compressed_order = Some(order);
            }
        }

        Ok(())
    }

    pub fn compress_quad_store(&self, order: QuadOrder) -> Result<()> {
        let parse_quad_args = self
            .parse_quad_args
            .as_ref()
            .context("parse_quad_args not set")?;
        let quads_path = self.location.join(format!("quads-{order}"));
        let mphf_path = self.location.join("terms_mphf");

        if !self.rebuild && quads_path.join("config.json").exists() {
            log::info!("Skipping quads extraction, already done.");
            return Ok(());
        }

        let terms_mphf = terms_mphf::TermMphf::load(&mphf_path)
            .with_context(|| format!("Could not mmap terms MPHF from {}", mphf_path.display()))?;
        if !self.location.exists() {
            std::fs::create_dir(&self.location)
                .with_context(|| format!("Could not create {}", self.location.display()))?;
        }

        if parse_quad_args.parallel_parser {
            // parse in parallel, process in parallel
            quads_store::compress_parsed_quads(
                get_parallel_iterator_from_parallel_parsers(
                    parse_quad_args,
                    self.rdf_format_from_path,
                )?,
                &quads_path,
                &terms_mphf,
                self.compute_approx_num_quads(),
                order,
            )
            .context("Could not compress quads")
        } else {
            // parse sequentially, process in parallel
            quads_store::compress_parsed_quads(
                get_parallel_iterator_from_sequential_parsers(
                    parse_quad_args,
                    self.rdf_format_from_path,
                )?,
                &quads_path,
                &terms_mphf,
                self.compute_approx_num_quads(),
                order,
            )
            .context("Could not compress quads")
        }
    }

    pub fn recompress_quad_store(&self, from_order: QuadOrder, to_order: QuadOrder) -> Result<()> {
        let quads_from_path = self.location.join(format!("quads-{from_order}"));
        let quads_to_path = self.location.join(format!("quads-{to_order}"));

        if !self.rebuild && quads_to_path.join("config.json").exists() {
            log::info!("Skipping {to_order} quads extraction, already done.");
            return Ok(());
        }

        let config_path = quads_from_path.join("config.json");
        let config_file = File::open(&config_path)
            .with_context(|| format!("Could not open {}", config_path.display()))?;
        let config: QuadStoreConfiguration = serde_json::from_reader(config_file)
            .with_context(|| format!("Could not read config from {}", config_path.display()))?;
        let quads = quads_store::par_iter_quads(&quads_from_path, from_order)
            .context("Could not start reading quads")?
            .map(|quad| quad.map(from_order.reverse_mapper()));

        quads_store::compress_quads(
            quads,
            &quads_to_path,
            config.num_terms,
            Some(config.num_quads),
            to_order,
        )
        .context("Could not compress quads")
    }

    pub fn index_quad_store(&self, order: QuadOrder) -> Result<()> {
        log::info!("Indexing {order} quad store frames");
        self.index_quad_store_frames(order)?;

        log::info!("Indexing {order} quad store by first term");
        self.index_quad_store_by_first_term(order)?;

        log::info!("Indexing {order} quad store by first two terms");
        self.index_quad_store_by_first_two_terms(order)?;

        Ok(())
    }

    pub fn index_quad_store_frames(&self, order: QuadOrder) -> Result<()> {
        let quads_path = self.location.join(format!("quads-{order}"));
        quads_store::index_frames(&quads_path).context("Could not index quad store frames")
    }

    pub fn index_quad_store_by_first_term(&self, order: QuadOrder) -> Result<()> {
        let quads_path = self.location.join(format!("quads-{order}"));
        quads_store::index_quads_by_first_term(&quads_path)
            .context("Could not index quad store by first term")
    }

    pub fn index_quad_store_by_first_two_terms(&self, order: QuadOrder) -> Result<()> {
        let quads_path = self.location.join(format!("quads-{order}"));
        quads_store::index_quads_by_first_two_terms(&quads_path)
            .context("Could not index quad store by first two terms")
    }

    pub fn index_by_second_term(&self, order: QuadOrder) -> Result<()> {
        let quads_path = self.location.join(format!("quads-{order}"));
        let index_path = self.location.join(format!("secondary-{order}"));
        secondary_indexes::build_secondary_index(&quads_path, &index_path)
            .context("Could not index by second term")
    }
}

fn get_parallel_iterator_from_sequential_parsers(
    args: &ParseQuadsArgs,
    rdf_format_from_path: fn(&Path) -> Result<RdfFormat>,
) -> Result<impl ParallelIterator<Item = Result<Quad>>> {
    let graph = if let Some(iri) = &args.graph {
        Some(
            NamedNode::new(iri)
                .with_context(|| format!("The target graph name {iri} is invalid"))?,
        )
    } else {
        None
    };

    Ok(args
        .file
        .iter()
        .map(move |file| {
            Ok((
                file.display().to_string(),
                get_quads(
                    file.display().to_string(),
                    deko::read::AnyDecoder::new(
                        File::open(file)
                            .with_context(|| format!("Could not open {}", file.display()))?,
                    ),
                    args.format.map_or_else(
                        || {
                            rdf_format_from_path(&file.with_extension("")).with_context(|| {
                                format!("Could not guess type of file {}", file.display())
                            })
                        },
                        Ok,
                    )?,
                    args.base.as_deref(),
                    graph.clone(),
                    args.lenient,
                )?,
            ))
        })
        .collect::<Result<Vec<_>>>()?
        .into_par_iter()
        .flat_map_iter(move |(file_name, file)| {
            file.map(move |quad| match quad {
                Err(e) if !args.lenient => {
                    eprintln!("Parsing error in {file_name}: {e}");
                    None
                }
                quad => Some(quad.with_context(|| format!("Could not parse {file_name}"))),
            })
        })
        .flatten())
}

fn get_parallel_iterator_from_parallel_parsers(
    args: &ParseQuadsArgs,
    rdf_format_from_path: fn(&Path) -> Result<RdfFormat>,
) -> Result<impl ParallelIterator<Item = Result<Quad>>> {
    let graph = if let Some(iri) = &args.graph {
        Some(
            NamedNode::new(iri)
                .with_context(|| format!("The target graph name {iri} is invalid"))?,
        )
    } else {
        None
    };
    Ok(args
        .file
        .iter()
        .map(move |file| {
            Ok((
                file.display().to_string(),
                get_parallel_quads(
                    file.display().to_string(),
                    deko::read::AnyDecoder::new(
                        File::open(file)
                            .with_context(|| format!("Could not open {}", file.display()))?,
                    ),
                    args.format.map_or_else(
                        || {
                            rdf_format_from_path(&file.with_extension("")).with_context(|| {
                                format!("Could not guess type of file {}", file.display())
                            })
                        },
                        Ok,
                    )?,
                    args.base.as_deref(),
                    graph.clone(),
                    args.lenient,
                )?,
            ))
        })
        .collect::<Result<Vec<_>>>()?
        .into_par_iter()
        .flat_map(move |(file_name, file)| {
            file.map(move |quad| match quad {
                Err(e) if !args.lenient => {
                    eprintln!("Parsing error in {file_name}: {e}");
                    None
                }
                quad => Some(quad.with_context(|| format!("Could not parse {file_name}"))),
            })
        })
        .flatten())
}

fn get_quads<R: Read + Send + 'static>(
    _reader_name: String,
    reader: R,
    format: RdfFormat,
    base_iri: Option<&str>,
    to_graph_name: Option<NamedNode>,
    lenient: bool,
) -> Result<impl Iterator<Item = Result<Quad, RdfParseError>> + 'static> {
    let mut parser = RdfParser::from_format(format);
    if let Some(to_graph_name) = to_graph_name {
        parser = parser.with_default_graph(to_graph_name);
    }
    if let Some(base_iri) = base_iri {
        parser = parser
            .with_base_iri(base_iri)
            .with_context(|| format!("Invalid base IRI {base_iri}"))?;
    }
    if lenient {
        parser = parser.lenient();
    }
    Ok(parser.for_reader(reader))
}

fn get_parallel_quads<R: Read + Send + 'static>(
    reader_name: String,
    reader: R,
    format: RdfFormat,
    base_iri: Option<&str>,
    to_graph_name: Option<NamedNode>,
    lenient: bool,
) -> Result<impl ParallelIterator<Item = Result<Quad, RdfParseError>> + 'static> {
    // whether we can parallelize parsing by splitting on newlines
    ensure!(
        format == RdfFormat::NQuads || format == RdfFormat::NTriples,
        "get_parallel_quads only supports NTriples and NQuads, not {format}"
    );

    let mut parser = RdfParser::from_format(format);
    if let Some(to_graph_name) = to_graph_name {
        parser = parser.with_default_graph(to_graph_name);
    }
    if let Some(base_iri) = base_iri {
        parser = parser
            .with_base_iri(base_iri)
            .with_context(|| format!("Invalid base IRI {base_iri}"))?;
    }
    if lenient {
        parser = parser.lenient();
    }

    let mut reader = BufReader::new(reader);
    let buf_size = 10 * 1024 * 1024; // read blocks of at most 10MiB at a time
    // let buf_size = 10 * 1024; // XXX debugging
    let mut buf = Vec::with_capacity(buf_size);
    Ok(std::iter::repeat(())
        .map_while(move |()| -> Option<Result<_>> {
            if !buf.is_empty() {
                assert_eq!(buf[0], b'<', "buf={:?}", String::from_utf8_lossy(&buf));
            }

            let num_bytes_in_buf = buf.len();
            if num_bytes_in_buf >= buf_size {
                // That's a big line. Build a chunk with only that line in it.
                if let Err(e) = reader.read_until(b'\n', &mut buf) {
                    return Some(Err(e).map_err(Into::into));
                }
                let mut chunk = Vec::new();
                std::mem::swap(&mut chunk, &mut buf);
                println!("big line: {}", String::from_utf8_lossy(&chunk));
                return Some(Ok(chunk));
            }
            assert!(!buf.contains(&b'\0'), "{:?}", String::from_utf8_lossy(&buf));
            buf.resize(buf_size, 0);
            match reader.read(&mut buf[num_bytes_in_buf..]) {
                Ok(0) =>
                // reached end of file
                {
                    (num_bytes_in_buf > 0).then(|| {
                        // one last chunk
                        buf.shrink_to(num_bytes_in_buf);
                        let mut chunk = Vec::with_capacity(buf_size);
                        std::mem::swap(&mut chunk, &mut buf);
                        eprintln!("last chunk: {:?}", String::from_utf8_lossy(&chunk));
                        Ok(chunk)
                    })
                }
                Ok(num_bytes_read) => {
                    buf.resize(num_bytes_in_buf + num_bytes_read, 0);
                    assert!(
                        !buf.contains(&b'\0'),
                        "{} {} {:?} {:?}",
                        num_bytes_in_buf,
                        num_bytes_read,
                        buf.iter().enumerate().find(|(_i, c)| **c == b'\0'),
                        String::from_utf8_lossy(&buf)
                    );
                    // look for last line break in the buffer
                    let Some((last_linebreak, _)) =
                        buf.iter().enumerate().rfind(|(_i, c)| **c == b'\n')
                    else {
                        // line is larger than the buffer. we'll deal with it next iteration
                        return Some(Ok(Vec::new()));
                    };
                    let mut chunk = Vec::new();
                    std::mem::swap(&mut chunk, &mut buf);
                    assert!(
                        !chunk.contains(&b'\0'),
                        "{:?}",
                        String::from_utf8_lossy(&chunk)
                    );
                    assert!(!buf.contains(&b'\0'), "{:?}", String::from_utf8_lossy(&buf));
                    buf.extend(chunk.drain(last_linebreak + 1..)); // move the start of the next line
                    assert!(
                        !chunk.contains(&b'\0'),
                        "{:?}",
                        String::from_utf8_lossy(&chunk)
                    );
                    assert!(!buf.contains(&b'\0'), "{:?}", String::from_utf8_lossy(&buf));
                    // println!("normal chunk: {}", String::from_utf8_lossy(&chunk));
                    // println!("drained: {}", String::from_utf8_lossy(&buf));
                    assert_eq!(chunk[0], b'<', "{}", String::from_utf8_lossy(&chunk));
                    assert_eq!(
                        chunk[chunk.len() - 1],
                        b'\n',
                        "{}",
                        String::from_utf8_lossy(&chunk)
                    );
                    assert_eq!(
                        chunk[chunk.len() - 2],
                        b'.',
                        "{}",
                        String::from_utf8_lossy(&chunk)
                    );
                    if !buf.is_empty() {
                        assert_eq!(
                            buf[0],
                            b'<',
                            "chunk={:?} buf={:?}",
                            String::from_utf8_lossy(&chunk),
                            String::from_utf8_lossy(&buf)
                        );
                    }
                    Some(Ok(chunk))
                }
                Err(e) => {
                    Some(Err(e).with_context(|| format!("Could not read from {reader_name}")))
                }
            }
        })
        .par_bridge()
        .flat_map_iter(move |chunk| match chunk {
            Ok(chunk) => parser.clone().for_reader(Cursor::new(chunk)),
            Err(e) => todo!("err: {e}"),
        }))
}
