use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand, ValueHint};
use oxigraph::io::{RdfFormat, RdfParseError, RdfParser};
use oxigraph::model::{NamedNode, Quad};
use oxigraph::succinct;
use oxigraph_cli::utils::{rdf_format_from_name, rdf_format_from_path};
use rayon::prelude::*;
use std::io::{BufRead, BufReader, Cursor, Read};
use std::path::PathBuf;

#[derive(Args, Clone)]
pub struct ParseQuadsArgs {
    /// File(s) to load
    ///
    /// If multiple files are provided, they are loaded in parallel.
    #[arg(num_args = 1.., value_hint = ValueHint::FilePath)]
    file: Vec<PathBuf>,
    /// The format of the file(s) to load
    ///
    /// It can be an extension like "nt" or a MIME type like "application/n-triples".
    ///
    /// By default, the format is guessed from the loaded file extension.
    #[arg(long)]
    format: Option<String>,
    /// Base IRI of the file(s) to load
    #[arg(long, value_hint = ValueHint::Url)]
    base: Option<String>,
    /// Attempt to keep loading even if the data file is invalid
    ///
    /// This disables most of the validation on RDF content.
    #[arg(long)]
    lenient: bool,
    /// Run multiple parsers in parallel on each file.
    ///
    /// Only available for NTriples and NQuads.
    #[arg(long)]
    parallel_parser: bool,
    /// Name of the graph to load the data to
    ///
    /// By default, the default graph is used.
    ///
    /// Only available when loading a graph file (N-Triples, Turtle...) and not a dataset file (N-Quads, TriG...).
    #[arg(long, value_hint = ValueHint::Url)]
    graph: Option<String>,
}

#[derive(Parser)]
#[command(about, version, name = "oxigraph-succinct-load")]
/// Oxigraph loader into the "succinct" storage format
pub struct GlobalArgs {
    /// Directory in which Oxigraph data are persisted
    #[arg(short, long, value_hint = ValueHint::DirPath)]
    location: PathBuf,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Clone)]
pub enum Commands {
    /// Step 1: read quads from the input and builds a terms/ directory with unique terms
    ExtractTerms {
        #[command(flatten)]
        parse_args: ParseQuadsArgs,
        #[arg(long)]
        /// Provides an estimated time of completion
        approx_quads_per_file: Option<usize>,
    },
    /// Step 2a: read the terms/ directory and makes each term accessible in O(1) given its position,
    /// allowing a O(1) map from ids to terms
    IndexTerms {},
    /// Step 2b: build a O(1) map from terms to ids
    BuildTermsMphf {},
    /// Step 3: read all quads again, and write them in a succinct format
    ///
    /// May error with ENOMEM if sysctl setting `vm.max_map_count` is too low.
    CompressQuads {
        #[command(flatten)]
        parse_args: ParseQuadsArgs,
        #[arg(long)]
        /// Provides an estimated time of completion
        approx_quads_per_file: Option<usize>,
        #[arg(long)]
        order: succinct::quads_store::QuadOrder,
    },
}

pub fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = GlobalArgs::parse();

    #[expect(clippy::shadow_same)]
    let args = &args;
    let terms_path = args.location.join("terms");
    match &args.command {
        Commands::ExtractTerms {
            parse_args,
            approx_quads_per_file,
        } => {
            let approx_num_quads = approx_quads_per_file
                .map(|approx_quads_per_file| approx_quads_per_file * parse_args.file.len());
            if !args.location.exists() {
                std::fs::create_dir(&args.location)
                    .with_context(|| format!("Could not create {}", args.location.display()))?;
            }
            if parse_args.parallel_parser {
                // parse in parallel, process in parallel
                succinct::terms_store::write_unique_terms(
                    get_parallel_iterator_from_parallel_parsers(&parse_args)?,
                    &terms_path,
                    approx_num_quads,
                )
                .context("Could not deduplicate or write terms")?
            } else {
                // parse sequentially, process in parallel
                succinct::terms_store::write_unique_terms(
                    get_parallel_iterator_from_sequential_parsers(&parse_args)?,
                    &terms_path,
                    approx_num_quads,
                )
                .context("Could not deduplicate or write terms")?
            }
        }
        Commands::IndexTerms {} => {
            succinct::terms_store::index_terms(&terms_path).context("Could not index terms")?;
        }
        Commands::BuildTermsMphf {} => {
            let mphf = succinct::terms_mphf::build_terms_mph(&terms_path)
                .context("Could not build terms MPHF")?;
            let mphf_path = args.location.join("terms_mphf");
            mphf.serialize(&mphf_path).with_context(|| {
                format!("Could not write terms MPHF to {}", mphf_path.display())
            })?;
        }
        Commands::CompressQuads {
            parse_args,
            approx_quads_per_file,
            order,
        } => {
            let quads_path = args.location.join(format!("quads-{order}"));
            let mphf_path = args.location.join("terms_mphf");
            let terms_mphf =
                succinct::terms_mphf::TermMphf::load(&mphf_path).with_context(|| {
                    format!("Could not mmap terms MPHF from {}", mphf_path.display())
                })?;
            let approx_num_quads = approx_quads_per_file
                .map(|approx_quads_per_file| approx_quads_per_file * parse_args.file.len());
            if !args.location.exists() {
                std::fs::create_dir(&args.location)
                    .with_context(|| format!("Could not create {}", args.location.display()))?;
            }

            if parse_args.parallel_parser {
                // parse in parallel, process in parallel
                succinct::quads_store::compress_quads(
                    get_parallel_iterator_from_parallel_parsers(&parse_args)?,
                    &quads_path,
                    &terms_mphf,
                    approx_num_quads,
                    *order,
                )
                .context("Could not compress quads")?
            } else {
                // parse sequentially, process in parallel
                succinct::quads_store::compress_quads(
                    get_parallel_iterator_from_sequential_parsers(&parse_args)?,
                    &quads_path,
                    &terms_mphf,
                    approx_num_quads,
                    *order,
                )
                .context("Could not compress quads")?
            }
        }
    }

    Ok(())
}

fn get_parallel_iterator_from_sequential_parsers(
    args: &ParseQuadsArgs,
) -> Result<impl ParallelIterator<Item = Result<Quad>>> {
    let format = if let Some(format) = &args.format {
        Some(rdf_format_from_name(format)?)
    } else {
        None
    };
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
                        std::fs::File::open(file)
                            .with_context(|| format!("Could not open {}", file.display()))?,
                    ),
                    format.map_or_else(
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
) -> Result<impl ParallelIterator<Item = Result<Quad>>> {
    let format = if let Some(format) = &args.format {
        Some(rdf_format_from_name(format)?)
    } else {
        None
    };
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
                        std::fs::File::open(file)
                            .with_context(|| format!("Could not open {}", file.display()))?,
                    ),
                    format.map_or_else(
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
