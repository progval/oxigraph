use super::sort::{ExternalArraySorter, write_sorted_array_file};
use super::terms_mphf::TermMphf;
use crate::model::Quad;
use anyhow::{Context, Result, anyhow};
use dsi_bitstream::prelude::*;
use dsi_progress_logger::{ProgressLog, concurrent_progress_logger};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use sux::traits::bit_field_slice::BitFieldSlice;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum)]
pub enum QuadOrder {
    Spog,
    Opsg,
}

impl QuadOrder {
    pub fn predicate(self) -> fn([usize; 4]) -> [usize; 4] {
        fn order_spog(quad: [usize; 4]) -> [usize; 4] {
            quad
        }
        fn order_opsg(quad: [usize; 4]) -> [usize; 4] {
            let [s, p, o, g] = quad;
            [o, p, s, g]
        }
        use QuadOrder::*;
        match self {
            Spog => order_spog,
            Opsg => order_opsg,
        }
    }
}

impl std::fmt::Display for QuadOrder {
    #[inline(always)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use QuadOrder::*;
        let s = match self {
            Spog => "spog",
            Opsg => "opsg",
        };
        write!(f, "{}", s)
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct QuadStoreConfiguration {
    num_partitions: usize,
    num_quads: usize,
}

pub fn compress_quads(
    quads: impl ParallelIterator<Item = Result<Quad>>,
    dst_dir: &Path,
    mphf: &TermMphf<impl BitFieldSlice<usize> + Sync + Send>,
    approx_num_quads: Option<usize>,
    order: QuadOrder,
) -> Result<()> {
    let mut config = QuadStoreConfiguration {
        num_partitions: (4 * usize::from(
        std::thread::available_parallelism().context("Could not count CPU threads")?,
    ))
    .min(16)
    .max(256) // avoid too many files
    .next_power_of_two(),
        num_quads: 0,
    };

    let max_value = mphf.len();

    let sorter_pool = thread_local::ThreadLocal::new();

    let order_quad = order.predicate();

    // 500MiB in-memory buffer per thread
    let max_buffer_size = 500 * 1024 * 1024;
    let mut pl = concurrent_progress_logger!(
        item_name = "quad",
        display_memory = true,
        local_speed = true,
        expected_updates = approx_num_quads,
    );
    pl.start("Reading and sorting quads...");
    let num_quads = AtomicUsize::new(0);
    let sorter_pool_ref = &sorter_pool;
    quads.try_for_each_init(
        || -> Result<_> {
            Ok((
                pl.clone(),
                sorter_pool_ref
                    .get_or(|| {
                        std::cell::RefCell::new(ExternalArraySorter::<4>::new(
                            max_value,
                            max_buffer_size,
                            config.num_partitions,
                        ))
                    })
                    .borrow_mut(),
            ))
        },
        |acc, quad| -> Result<_> {
            let (pl, sorter) = acc.as_mut().unwrap(); // XXX debug
            let mut sorter = sorter
                .as_mut()
                .map_err(|e| anyhow!("Could not create sorter ExternalArraySorter: {e:#?}"))?;
            // let (mut pl, mut sorter, num_quads) = acc?;
            let quad = quad?;
            let Quad {
                subject,
                predicate,
                object,
                graph_name,
            } = &quad;
            let compressed_quad = order_quad([
                mphf.hash_namedorblanknode(subject)
                    .with_context(|| format!("Unknown subject: {subject:?}"))
                    .unwrap(),
                mphf.hash_namednode(predicate)
                    .with_context(|| format!("Unknown predicate: {predicate:?}"))
                    .unwrap(),
                mphf.hash_term(object)
                    .with_context(|| format!("Unknown object: {object:?}"))
                    .unwrap(),
                mphf.hash_graphname(graph_name)
                    .with_context(|| format!("Unknown graph name: {graph_name:?}"))
                    .unwrap(),
            ]);
            assert!(
                compressed_quad.iter().all(|term_id| *term_id < max_value),
                "Got quad {compressed_quad:?} (from {quad:?}), but max value is {}",
                max_value,
            );
            sorter
                .push(compressed_quad)
                .context("Could not push quad to sorter")?;
            pl.light_update();
            num_quads.fetch_add(1, Ordering::Relaxed);
            Ok(())
        },
    )?;

    let mut sorted_quads =
        sorter_pool
            .into_iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|sorter| {
                let mut sorter = sorter.into_inner().expect("could not get sorter"); // XXX debug
                sorter.flush_buffers().expect("could not flush"); // XXX debug
                Ok(sorter)
            })
            .reduce(
                || {
                    Ok(ExternalArraySorter::<4>::new(
                        max_value,
                        max_buffer_size,
                        config.num_partitions,
                    )
                    .context("Could not create sorter ExternalArraySorter")?)
                },
                |left: Result<_>, right| {
                    Ok(
                        left.unwrap()
                            .merge(right.unwrap())
                            .expect("Could not merge ExternalDeduplicatingStringSorter"),
                        //.context("Could not merge ExternalDeduplicatingStringSorter")?,
                    )
                },
            )?;
    pl.done();

    config.num_quads = num_quads.into_inner();
    std::fs::create_dir(dst_dir)
        .with_context(|| format!("Could not create {}", dst_dir.display()))?;
    let mut pl = concurrent_progress_logger!(
        item_name = "quad",
        display_memory = true,
        local_speed = true,
        expected_updates = Some(config.num_quads),
    );
    pl.start("Merging and writing quads...");
    sorted_quads
        .iter_partitions()
        .context("Could not read partitions")?
        .into_par_iter()
        .enumerate()
        .try_for_each_with(pl.clone(), |pl, (partition_id, partition)| -> Result<_> {
            let path = dst_dir.join(format!("{partition_id}.quads.bitstream"));
            let file = File::create(&path)
                .with_context(|| format!("Could not create {}", path.display()))?;
            let mut writer = BufBitWriter::new(WordAdapter::<usize, _>::new(BufWriter::new(file)));
            write_sorted_array_file(&mut writer, partition.into_iter(), pl)
                .with_context(|| format!("Could not write quads to {}", path.display()))?;
            writer
                .into_inner() // BufBitWriter -> WordAdapter
                .with_context(|| format!("Could not flush {}", path.display()))?
                .into_inner() // WordAdapter -> BufWriter
                .into_inner() // BufWriter -> File
                .with_context(|| format!("Could not flush {}", path.display()))?
                .flush()
                .with_context(|| format!("Could not flush {}", path.display()))?;
            Ok(())
        })?;
    pl.done();

    let config_path = dst_dir.join("config.json");
    let config_file = File::create(&config_path)
        .with_context(|| format!("Could not create {}", config_path.display()))?;
    serde_json::to_writer_pretty(config_file, &config)
        .context("Could not write terms store config")?;

    Ok(())
}
