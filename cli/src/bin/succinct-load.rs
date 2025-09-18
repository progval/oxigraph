use anyhow::{Context, Result, anyhow, ensure};
use clap::{Parser, ValueHint};
use oxigraph::io::{RdfFormat, RdfParseError, RdfParser};
use oxigraph::model::{NamedNode, Quad};
use oxigraph::succinct;
use oxigraph_cli::utils::{rdf_format_from_name, rdf_format_from_path};
use rayon::prelude::*;
use std::io::{BufRead, BufReader, Cursor, Read};
use std::path::PathBuf;
use std::process::{Command, Stdio};

#[derive(Parser)]
#[command(about, version, name = "oxigraph-succinct-load")]
/// Oxigraph loader into the "succinct" storage format
pub struct Args {
    /// Directory in which Oxigraph data are persisted
    #[arg(short, long, value_hint = ValueHint::DirPath)]
    location: PathBuf,
    /// File(s) to load
    ///
    /// If multiple files are provided, they are loaded in parallel.
    #[arg(short, long, num_args = 0.., value_hint = ValueHint::FilePath)]
    file: Vec<PathBuf>,
    /// Shell commands that stream a file to load on their stdout.
    ///
    /// If multiple command are provided, they are loaded in parallel.
    #[arg(short, long, num_args = 0.., requires = "format")]
    command: Vec<String>,
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

pub fn main() -> Result<()> {
    let args = Args::parse();

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

    ensure!(
        args.file.is_empty() || args.command.is_empty(),
        "--file or --command must be provided"
    );
    ensure!(
        args.command.is_empty() || format.is_some(),
        "--command requires --format"
    );

    #[expect(clippy::shadow_same)]
    let args = &args;
    #[expect(clippy::shadow_same)]
    let graph = &graph;

    let file_quad_factories: Vec<_> = args
        .file
        .iter()
        .map(|file| {
            move || {
                get_quads(
                    file.display().to_string(),
                    std::fs::File::open(file)
                        .with_context(|| format!("Could not open {}", file.display()))?,
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
                )
            }
        })
        .collect();

    let parallel_file_quad_factories: Vec<_> = args
        .file
        .iter()
        .map(|file| {
            move || {
                get_parallel_quads(
                    file.display().to_string(),
                    std::fs::File::open(file)
                        .with_context(|| format!("Could not open {}", file.display()))?,
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
                )
            }
        })
        .collect();

    let command_quad_factories: Vec<_> = args
        .command
        .iter()
        .map(|command| {
            move || {
                get_quads(
                    command.clone(),
                    Command::new("sh")
                        .arg("-c")
                        .arg(command)
                        .stdout(Stdio::piped())
                        .spawn()
                        .with_context(|| format!("Could not spawn {command}"))?
                        .stdout
                        .take()
                        .unwrap(),
                    format.unwrap(),
                    args.base.as_deref(),
                    graph.clone(),
                    args.lenient,
                )
            }
        })
        .collect();

    let parallel_command_quad_factories: Vec<_> = args
        .command
        .iter()
        .map(|command| {
            move || {
                get_parallel_quads(
                    command.clone(),
                    Command::new("sh")
                        .arg("-c")
                        .arg(command)
                        .stdout(Stdio::piped())
                        .spawn()
                        .with_context(|| format!("Could not spawn {command}"))?
                        .stdout
                        .take()
                        .unwrap(),
                    format.unwrap(),
                    args.base.as_deref(),
                    graph.clone(),
                    args.lenient,
                )
            }
        })
        .collect();

    /*
    let get_quad_iterator = || -> Result<_> {
        let file_quad_iterators = file_quad_factories
            .iter()
            .map(|factory| (factory)())
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten();
        let command_quad_iterators = command_quad_factories
            .iter()
            .map(|factory| (factory)())
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten();
        Ok(file_quad_iterators
            .chain(command_quad_iterators)
            .flat_map(if args.lenient {
                ignore_quad_error
            } else {
                some_quad
            }))
    };
    */

    let get_quad_parallel_iterator = || -> Result<_> {
        let file_quad_iterators = file_quad_factories
            .iter()
            .map(|factory| (factory)())
            .collect::<Result<Vec<_>>>()?
            .into_par_iter()
            .flatten_iter();
        let command_quad_iterators = command_quad_factories
            .iter()
            .map(|factory| (factory)())
            .collect::<Result<Vec<_>>>()?
            .into_par_iter()
            .flatten_iter();
        Ok(file_quad_iterators
            .chain(command_quad_iterators)
            .flat_map(if args.lenient {
                ignore_quad_error
            } else {
                some_quad
            }))
    };

    let get_parallel_quad_parallel_iterator = || -> Result<_> {
        let file_quad_iterators = parallel_file_quad_factories
            .iter()
            .map(|factory| (factory)())
            .collect::<Result<Vec<_>>>()?
            .into_par_iter()
            .flatten();
        let command_quad_parallel_iterators = parallel_command_quad_factories
            .iter()
            .map(|factory| (factory)())
            .collect::<Result<Vec<_>>>()?
            .into_par_iter()
            .flatten();
        Ok(file_quad_iterators
            .chain(command_quad_parallel_iterators)
            .flat_map(if args.lenient {
                ignore_quad_error
            } else {
                some_quad
            }))
    };

    let mphf = if args.parallel_parser {
        let parallel_parallel_iterators = (get_parallel_quad_parallel_iterator)()?;
        // parse in parallel, process in parallel
        eprintln!("parse in parallel");
        succinct::build_term_mphf(parallel_parallel_iterators).context("Could not build MPH")?
    } else {
        // parse sequentially, process in parallel
        eprintln!("parse sequentially");
        let parallel_iterators = (get_quad_parallel_iterator)()?;
        succinct::build_term_mphf(parallel_iterators).context("Could not build MPH")?
    };

    Ok(())
}

#[expect(clippy::unnecessary_wraps)]
fn some_quad(quad: Result<Quad, RdfParseError>) -> Option<Result<Quad, RdfParseError>> {
    Some(quad)
}
fn ignore_quad_error(quad: Result<Quad, RdfParseError>) -> Option<Result<Quad, RdfParseError>> {
    if let Err(e) = quad {
        eprintln!("Parsing error: {e}");
        return None;
    }
    Some(quad)
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
    Ok(parser.rename_blank_nodes().for_reader(reader))
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
    let parser = parser.rename_blank_nodes();

    let mut reader = BufReader::new(reader);
    let buf_size = 10 * 1024 * 1024; // read blocks of at most 10MiB at a time
    //let buf_size = 10 * 1024; // XXX debugging
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
                    //println!("normal chunk: {}", String::from_utf8_lossy(&chunk));
                    //println!("drained: {}", String::from_utf8_lossy(&buf));
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
