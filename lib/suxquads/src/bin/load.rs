use anyhow::{Context, Result, bail, ensure};
use clap::{Parser, ValueHint};
use oxrdfio::{RdfFormat, RdfParseError, RdfParser};
use oxrdf::{NamedNode, Quad};
use std::ffi::OsStr;
use std::io::Read;
use std::path::{Path, PathBuf};
use rayon::prelude::*;
use std::process::{Command, Stdio};

#[derive(Parser)]
#[command(about, version, name = "oxigraph")]
/// Oxigraph command line toolkit and SPARQL HTTP server
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
        Ok(file_quad_iterators.chain(command_quad_iterators))
    };

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
        Ok(file_quad_iterators.chain(command_quad_iterators))
    };

    Ok(())
}

fn get_quads(
    reader: impl Read + 'static,
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
fn format_from_path<T>(path: &Path, from_extension: impl FnOnce(&str) -> Result<T>) -> Result<T> {
    if let Some(ext) = path.extension().and_then(OsStr::to_str) {
        from_extension(ext).map_err(|e| {
            e.context(format!(
                "Not able to guess the file format from file name extension '{ext}'"
            ))
        })
    } else {
        bail!(
            "The path {} has no extension to guess a file format from",
            path.display()
        )
    }
}

fn rdf_format_from_path(path: &Path) -> Result<RdfFormat> {
    format_from_path(path, |ext| {
        RdfFormat::from_extension(ext)
            .with_context(|| format!("The file extension '{ext}' is unknown"))
    })
}

fn rdf_format_from_name(name: &str) -> Result<RdfFormat> {
    if let Some(t) = RdfFormat::from_extension(name) {
        return Ok(t);
    }
    if let Some(t) = RdfFormat::from_media_type(name) {
        return Ok(t);
    }
    bail!("The file format '{name}' is unknown")
}
