use anyhow::{Context, Result, bail};
use dsi_bitstream::prelude::BigEndian;
use epserde::deser::Deserialize;
use epserde::prelude::Flags;
use itertools::Itertools;
use lender::Lender;
use rayon::prelude::*;
use std::num::NonZeroUsize;
use std::path::Path;
use webgraph::graphs::arc_list_graph::ArcListGraph;
use webgraph::graphs::bvgraph::{BvComp, CompFlags};
use webgraph::prelude::*;
use webgraph::traits::SequentialLabeling;
use webgraph::utils::par_sort_pairs::ParSortPairs;
use webgraph_algo::preds::MinGain;
use webgraph_algo::{combine_labels, labels_to_ranks};

/// Given an iterator of quads and a path, writes all pairwise combinations of the quad
/// to a bvgraph at the given path
///
/// The pairwise combinations do not include the graph name as it it would bloat the BVGraph
/// and the list of successors of the default graph would take forever to process
pub fn symmetric_bv(
    quads: impl ParallelIterator<Item = Result<[usize; 4]>>,
    path: impl AsRef<Path>,
    num_terms: usize,
    num_quads: usize,
) -> Result<()> {
    let path = path.as_ref();

    let num_partitions = NonZeroUsize::new(256).unwrap();
    let num_terms_per_partition = num_terms.div_ceil(num_partitions.into());

    let pairs = quads
        .flat_map_iter(|quad| {
            let [s, p, o, _g] = quad.expect("Could not read quad");
            [(s, p), (s, o), (p, s), (p, o), (o, s), (o, p)]
        })
        .inspect(|(src, dst)| {
            assert!(*src < num_terms);
            assert!(*dst < num_terms);
        });

    let pair_sorter = ParSortPairs::new(num_terms)
        .context("Could not initialize ParSortPairs")?
        .batch_size(NonZeroUsize::new(10_000_000).unwrap()) // default uses too much RAM
        .expected_num_pairs(num_quads * 6) // mild overapprox (because there are duplicates)
        .num_partitions(num_partitions);

    let sorted_pairs = pair_sorter
        .par_sort_pairs(pairs)
        .context("Could not initialize ParSortPairs::par_sort_pairs")?;

    let bvcomp_tmp_dir = tempfile::tempdir().unwrap();

    // let mut g = webgraph::graphs::vec_graph::VecGraph::new();
    // for i in 0..num_terms {
    // g.add_node(i);
    // }
    // for partition in sorted_pairs {
    // for (src, dst) in partition.into_iter().dedup() {
    // println!("{src} {dst}");
    // g.add_arc(src, dst);
    // }
    // }
    // use epserde::ser::Serialize;
    // unsafe { g.serialize(&mut std::io::BufWriter::new( std::fs::File::create("/tmp/negative_node_id_offset.vecgraph.epserde")?))? };

    std::fs::create_dir_all(path)
        .with_context(|| format!("Could not create {}", path.display()))?;

    // TODO: Switch to LittleEndian once webgraph publishes a release that includes this fix:
    // https://github.com/vigna/webgraph-rs/pull/141
    BvComp::parallel_iter::<BigEndian, _>(
        &path.join("graph"),
        sorted_pairs
            .into_iter()
            .enumerate()
            .map(|(partition_id, partition)| {
                let partition = partition.into_iter().dedup();
                #[cfg(debug_assertions)]
                {
                    let mut prev_arc = None;
                    for arc in partition.clone() {
                        if let Some(prev_arc) = prev_arc {
                            assert!(
                                arc > prev_arc,
                                "{arc:?} after {prev_arc:?} in partition {partition_id}"
                            );
                        }
                        let (src, dst) = arc;
                        assert!(
                            src < num_terms,
                            "src={src} is greater then num_terms={num_terms}"
                        );
                        assert!(
                            dst < num_terms,
                            "dst={dst} is greater then num_terms={num_terms}"
                        );
                        assert!(src >= partition_id * num_terms_per_partition);
                        assert!(src < (partition_id + 1) * num_terms_per_partition);
                        // assert_ne!(src, dst);
                        prev_arc = Some(arc);
                    }
                }
                ArcListGraph::new(num_terms, partition)
                    .iter_from(partition_id * num_terms_per_partition)
                    .take(num_terms_per_partition)
            }),
        num_terms,
        CompFlags {
            // BvComp stores as many successor lists as the value of `compression_window`.
            // As we have some very long successor lists (eg.
            // http://www.wikidata.org/prop/direct/P31) this can use tens of gigabytes
            // of RAM as compression_window defaults to 7
            compression_window: 1,
            ..Default::default()
        },
        &rayon::ThreadPoolBuilder::default()
            .num_threads(1)
            .build()
            .unwrap(),
        bvcomp_tmp_dir.path(),
    )
    .context("Could not run BvComp")?;

    Ok(())
}

/// Equivalent to the 'webgraph run llp' CLI but uses file-backed mmap instead of
/// loading in THP memory, to save memory.
pub fn llp(graph_path: &Path, permutation_path: &Path, gammas: &[String]) -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let work_dir = temp_dir.path();
    log::info!("Using workdir: {}", work_dir.display());

    let graph_path = graph_path.join("graph");

    // Load the graph in THP memory
    let graph = BvGraph::with_basename(&graph_path)
        .mode::<Mmap>()
        .flags(MemoryFlags::TRANSPARENT_HUGE_PAGES | MemoryFlags::RANDOM_ACCESS)
        .endianness::<BigEndian>()
        .load()
        .with_context(|| format!("Could not mmap graph from {}", graph_path.display()))?;

    // Load degree cumulative function in THP memory
    let dcf_path = graph_path.with_extension(DEG_CUMUL_EXTENSION);
    let deg_cumul = DCF::mmap(
        &dcf_path,
        Flags::TRANSPARENT_HUGE_PAGES | Flags::RANDOM_ACCESS,
    )
    .with_context(|| {
        format!(
            "Could not mmap degree cumulative function from {}",
            dcf_path.display()
        )
    })?;

    // parse the gamma format
    let mut gammas = gammas
        .into_iter()
        .map(|gamma| {
            let t: Vec<_> = gamma.split('-').collect();
            if t.len() != 2 {
                bail!("Invalid gamma: {}", gamma);
            }

            Ok(if t[0].is_empty() {
                1.0
            } else {
                t[0].parse::<usize>()? as f64
            } * (0.5_f64).powf(t[1].parse::<usize>()? as f64))
        })
        .collect::<Result<Vec<_>>>()?;

    gammas.sort_by(|a, b| a.total_cmp(b));

    let predicate = MinGain::try_from(MinGain::DEFAULT_THRESHOLD)?;
    let granularity = Granularity::default();

    // compute the LLP
    webgraph_algo::llp::layered_label_propagation_labels_only(
        graph,
        &*deg_cumul,
        gammas,
        Some(rayon::current_num_threads().max(1)),
        None, // chunk_size
        granularity,
        0, // seed
        predicate,
        work_dir,
    )
    .context("Could not compute the LLP")?;

    log::info!("Combining labels...");
    let labels = combine_labels(work_dir)?;

    log::info!("Combined labels...");
    let rank_perm = labels_to_ranks(&labels);

    log::info!("Saving permutation...");
    webgraph_cli::run::llp::store_perm(
        &rank_perm,
        permutation_path,
        false, // epserde
    )?;

    Ok(())
}
