use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand, ValueHint};
use oxigraph::succinct;
use oxigraph::succinct::quads_store::QuadStoreConfiguration;
use oxigraph::succinct::terms_store::TermStoreConfiguration;
use oxigraph_cli::utils::{rdf_format_from_name, rdf_format_from_path};
use std::fs::File;
use std::path::PathBuf;

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

impl TryFrom<ParseQuadsArgs> for succinct::database_builder::ParseQuadsArgs {
    type Error = anyhow::Error;

    fn try_from(value: ParseQuadsArgs) -> Result<Self, Self::Error> {
        let ParseQuadsArgs {
            file,
            format,
            base,
            lenient,
            parallel_parser,
            graph,
        } = value;
        Ok(succinct::database_builder::ParseQuadsArgs {
            file,
            format: if let Some(format) = format {
                Some(rdf_format_from_name(&format)?)
            } else {
                None
            },
            base,
            lenient,
            parallel_parser,
            graph,
        })
    }
}

#[derive(Subcommand, Clone)]
pub enum Commands {
    /// Run all the other commands at once for the default set of quad orders
    BuildAll {
        #[command(flatten)]
        parse_args: ParseQuadsArgs,
        #[arg(long)]
        /// Provides an estimated time of completion
        approx_quads_per_file: Option<usize>,
    },
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
    /// Step 4: Read compressed quads in one order, and write them to an other order
    ///
    /// This is more efficient than 'compress-quads', but requires that 'compress-quad'
    /// already ran once.
    ///
    /// May error with ENOMEM if sysctl setting `vm.max_map_count` is too low.
    RecompressQuads {
        #[arg(long)]
        from_order: succinct::quads_store::QuadOrder,
        #[arg(long)]
        to_order: succinct::quads_store::QuadOrder,
    },
    /// Step 5: read all quads (from a quad store) and compress them as a symmetrized BGraph
    SymmetricBv {
        #[arg(long)]
        order: succinct::quads_store::QuadOrder,
    },
    // Step 5:
    //  * cargo install webgraph-cli
    //  * webgraph build ef /srv/oxigraph-data/wikidata.succinct/symmetric_bvgraph/graph
    //  * webgraph build dcf /srv/oxigraph-data/wikidata.succinct/symmetric_bvgraph/graph
    /// Step 7 run LLP to cluster similar nodes together
    Llp {
        #[arg(long, default_values_t = vec!["-0".to_string(), "-1".to_string(), "-2".to_string(), "-3".to_string(), "-4".to_string(), "-5".to_string(), "-6".to_string(), "-7".to_string(), "-8".to_string(), "-9".to_string(), "-10".to_string()])]
        gammas: Vec<String>,
    },
    /// Step ??: Build an index from the first term of each quad to its position in a compressed
    /// quad list
    IndexQuadsByFirstTerm {
        #[arg(long)]
        order: succinct::quads_store::QuadOrder,
    },
    /// Step ??: Build an index to the start of each frame
    IndexQuadsFrames {
        #[arg(long)]
        order: succinct::quads_store::QuadOrder,
    },
    /// Step ??+1: Build an index from the first two terms of each quad to their position in a compressed
    /// quad list
    IndexQuadsByFirstTwoTerms {
        #[arg(long)]
        order: succinct::quads_store::QuadOrder,
    },
}

pub fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = GlobalArgs::parse();

    let db_builder = succinct::database_builder::DatabaseBuilder::new(args.location.clone())
        .with_rdf_format_from_path(rdf_format_from_path);
    match args.command {
        Commands::BuildAll {
            parse_args,
            approx_quads_per_file,
        } => {
            db_builder
                .with_parse_quad_args(Some(parse_args.try_into()?))
                .with_approx_num_quads(approx_quads_per_file)
                .build_all()?;
        }
        Commands::ExtractTerms {
            parse_args,
            approx_quads_per_file,
        } => {
            db_builder
                .with_parse_quad_args(Some(parse_args.try_into()?))
                .with_approx_num_quads(approx_quads_per_file)
                .extract_terms()?;
        }
        Commands::IndexTerms {} => {
            db_builder.index_terms()?;
        }
        Commands::BuildTermsMphf {} => db_builder.build_terms_mphf()?,
        Commands::CompressQuads {
            parse_args,
            approx_quads_per_file,
            order,
        } => {
            db_builder
                .with_parse_quad_args(Some(parse_args.try_into()?))
                .with_approx_quads_per_file(approx_quads_per_file)
                .compress_quad_store(order)?;
        }
        Commands::RecompressQuads {
            from_order,
            to_order,
        } => {
            db_builder.recompress_quad_store(from_order, to_order)?;
        }
        Commands::SymmetricBv { order } => {
            let config_path = db_builder.terms_path().join("config.json");
            let config_file = File::open(&config_path)
                .with_context(|| format!("Could not open {}", config_path.display()))?;
            let terms_store_config: TermStoreConfiguration = serde_json::from_reader(config_file)
                .with_context(|| {
                format!("Could not read config from {}", config_path.display())
            })?;

            let quads_path = args.location.join(format!("quads-{order}"));
            let config_path = quads_path.join("config.json");
            let config_file = File::open(&config_path)
                .with_context(|| format!("Could not open {}", config_path.display()))?;
            let quads_store_config: QuadStoreConfiguration = serde_json::from_reader(config_file)
                .with_context(|| {
                format!("Could not read config from {}", config_path.display())
            })?;

            let quads = succinct::quads_store::par_iter_quads(&quads_path, order)
                .context("Could not start reading quads")?;
            let symmetric_graph_path = args.location.join("symmetric_bvgraph");
            succinct::webgraph::symmetric_bv(
                quads,
                &symmetric_graph_path,
                terms_store_config.num_terms,
                quads_store_config.num_quads,
            )
            .with_context(|| {
                format!(
                    "Could not BvComp quads from {} to {}",
                    quads_path.display(),
                    symmetric_graph_path.display()
                )
            })?;
        }
        Commands::Llp { gammas } => {
            let symmetric_graph_path = args.location.join("symmetric_bvgraph");
            let permutation_path = args.location.join("llp.perm");
            succinct::webgraph::llp(&symmetric_graph_path, &permutation_path, &gammas)
                .context("Could not run LLP")?;
        }
        Commands::IndexQuadsFrames { order } => {
            db_builder.index_quad_store_frames(order)?;
        }
        Commands::IndexQuadsByFirstTerm { order } => {
            db_builder.index_quad_store_by_first_term(order)?;
        }
        Commands::IndexQuadsByFirstTwoTerms { order } => {
            db_builder.index_quad_store_by_first_two_terms(order)?;
        }
    }

    Ok(())
}
