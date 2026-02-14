use super::quads_store::get_quad_partitions;
use super::webgraph::DynamicBvGraph;
use crate::succinct::sort::SortedArraysFile;
use anyhow::{Context, Result};
use dsi_progress_logger::{ProgressLog, concurrent_progress_logger, progress_logger};
use epserde::deser::mem_case::{Flags, MemCase};
use epserde::deser::{Deserialize as EpDeserialize, DeserInner as EpDeserInner};
use epserde::ser::Serialize as EpSerialize;
use itertools::Itertools;
use rayon::prelude::*;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::Ordering;
use sux::bits::{AtomicBitVec, BitVec};
use sux::dict::elias_fano::{EfSeqDict, EliasFanoBuilder};
use sux::traits::IndexedDict;
use webgraph::graphs::bvgraph::{BvGraph, MemoryFlags};
use webgraph::traits::RandomAccessGraph;

pub struct Contraction<
    T = EfSeqDict as EpDeserInner>::DeserType<
        'static,
    >,
>(MemCase<T>)
where for<'a> T: IndexedDict<Input = usize, Output<'a> = usize>;

impl Contraction<EfSeqDict> {
    pub fn mmap(
        path: &Path,
    ) -> Result<Contraction<<EfSeqDict as EpDeserInner>::DeserType<'static>>> {
        Ok(Contraction(
            EfSeqDict::mmap(&path, Flags::RANDOM_ACCESS).with_context(|| {
                format!(
                    "Could not epdeserialize Contraction index from {}",
                    path.display()
                )
            })?,
        ))
    }
}

impl<T: IndexedDict<Input = usize, Output = usize>> Contraction<T> {
    pub fn get(&self, term_id: usize) -> Option<usize> {
        self.0.index_of(term_id)
    }
}

/// Given a quad store, builds a BvGraph that maps second terms to first terms
pub fn build_secondary_index(quads_dir: &Path, index_dir: &Path) -> Result<()> {
    let (config, partitions) = get_quad_partitions(quads_dir)?;

    // List all terms in the second position
    let mut pl = concurrent_progress_logger!(
        item_name = "quad",
        display_memory = true,
        local_speed = true,
        expected_updates = Some(config.num_quads),
    );
    pl.start("Listing second terms...");
    let present = AtomicBitVec::new(config.num_terms);
    partitions
        .par_iter()
        .try_for_each_with(pl.clone(), |pl, path| -> Result<_> {
            let terms = SortedArraysFile::<4>::mmap(&path)
            .with_context(|| format!("Could not mmap array file {}", path.display()))?
            .owned_iter()
            .with_context(|| format!("Could not read array file {}", path.display()))?
            .map(|quad| -> Result<_>{
                pl.light_update();
                let [_, b, _, _] = quad?;
                Ok(b)
            })
            // deduplicate early, to reduce the number of atomic operations
            .dedup_by(|left, right| match (left, right) {
                (Ok(left), Ok(right)) => left == right,
                (Err(_), Ok(_)) => true,  // return error first
                (Ok(_), Err(_)) => false, // return error first
                (_, _) => true,           // doesn't matter
            });
            for term in terms {
                present.set(term?, true, Ordering::Relaxed);
            }
            Ok(())
        })?;
    pl.done();
    let present = BitVec::from(present);

    // Build a map from term ids to their (smaller) "id as predicate", so the resulting BvGraph does
    // not need to store empty adjacency lists
    //
    // We could easily use EliasFanoConcurrentBuilder here but it's unsafe and does not bring much
    // performance improvement.
    let num_present = present.par_count_ones();
    let mut pl = progress_logger!(
        item_name = "second term",
        display_memory = true,
        local_speed = true,
        expected_updates = Some(num_present),
    );
    pl.start("Listing second terms...");
    let mut efb = EliasFanoBuilder::new(num_present, config.num_terms);
    for term in present.iter_ones() {
        efb.push(term);
        pl.light_update()
    }
    pl.done();

    log::info!("Building Contraction...");
    let termid_to_predid = Contraction(MemCase::encase(efb.build_with_seq_and_dict()));

    let mut pl = concurrent_progress_logger!(
        item_name = "quad",
        display_memory = true,
        local_speed = true,
        expected_updates = Some(config.num_quads),
    );
    pl.start("Building BVGraph...");
    let pairs = partitions
        .into_par_iter()
        .map_with(pl.clone(), |pl, path| -> Result<_> {
            let mut pl = pl.clone();
            Ok(SortedArraysFile::<4>::mmap(&path)
            .with_context(|| format!("Could not mmap array file {}", path.display()))?
            .owned_iter()
            .with_context(|| format!("Could not read array file {}", path.display()))?
            .map(move |quad| -> Result<_>{
                pl.light_update();
                let [a, b, _, _] = quad?;
                Ok((a, b))
            })
            // deduplicate early, to reduce the number of Elias-Fano requests and
            // the number of pairs we produce
            .dedup_by(|left, right| match (left, right) {
                (Ok(left), Ok(right)) => left == right,
                (Err(_), Ok(_)) => true,  // return error first
                (Ok(_), Err(_)) => false, // return error first
                (_, _) => true,           // doesn't matter
            })
            .map(|pair| -> Result<_>{
                let (a, b) = pair?;
                Ok((termid_to_predid.get(b).context("Second term missing from contraction")?, a))
            }))
        })
        .collect::<Result<Vec<_>>>()?
        .into_par_iter()
        .flatten_iter();

    super::webgraph::bv(pairs, index_dir, num_present, None)?;

    log::info!("Serializing Contraction...");
    let termid_to_predid_path = index_dir.join("termid_to_predid.ef");
    let mut termid_to_predid_file = File::create(&termid_to_predid_path)
        .with_context(|| format!("Could not create {}", termid_to_predid_path.display()))?;
    termid_to_predid
        .0
        .serialize(&mut termid_to_predid_file)
        .with_context(|| {
            format!(
                "Could not write Contraction to {}",
                termid_to_predid_path.display()
            )
        })?;

    Ok(())
}

/// Given a term 'b', returns all terms 'a' such that there is a quad matching `(a, b, _, _)`
/// in the quad store
pub struct SecondaryIndex {
    /// Maps term_id to internal_id
    contraction: Contraction,
    /// Maps internal_id to iterators of term_id
    graph: DynamicBvGraph,
}

impl SecondaryIndex {
    pub fn mmap(path: &Path) -> Result<Self> {
        let contraction_path = path.join("termid_to_predid.ef");
        let contraction = Contraction::mmap(&contraction_path)?;

        let bvgraph_basepath = path.join("graph");
        let graph = BvGraph::with_basename(&bvgraph_basepath)
            .flags(MemoryFlags::RANDOM_ACCESS)
            .graph_mode::<webgraph::graphs::bvgraph::Mmap>() // 500MiB
            .offsets_mode::<webgraph::graphs::bvgraph::LoadMem>() // 30kiB
            .load()?;

        Ok(Self { contraction, graph })
    }

    pub fn get(&self, term: usize) -> Option<impl Iterator<Item = usize> + use<>> {
        // FIXME: don't materialize the adjacency list here. Unfortunately, I can't find a way to make it
        // 'static without materializing

        // We can't even use BitFieldVec for this because it also can't return 'static iterators
        // let max_term_id = self.contraction.0.get(self.contraction.0.len() - 1);
        // let bit_width = usize::try_from(usize::BITS - max_term_id.leading_zeros()).expect("Bit width overflowed usize");
        // let mut successors = BitFieldVec::new(bit_width, 0);
        // successors.extend(self.graph.successors(self.contraction.get(term)?));
        // Some(successors.into_iter())

        let successors: Vec<_> = self.graph.successors(self.contraction.get(term)?).collect();
        Some(successors.into_iter())
    }
}
