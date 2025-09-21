use super::sort::{ExternalArraySorter, write_sorted_array_file};
use super::terms_mphf::TermMphf;
use crate::model::Quad;
use anyhow::{Context, Result};
use dsi_bitstream::prelude::*;
use dsi_progress_logger::{ProgressLog, concurrent_progress_logger};
use rayon::prelude::*;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use sux::traits::bit_field_slice::BitFieldSlice;

pub fn compress_quads(
    quads: impl ParallelIterator<Item = Result<Quad>>,
    dst_dir: &Path,
    mphf: &TermMphf<impl BitFieldSlice<usize> + Sync + Send>,
    approx_num_quads: Option<usize>,
) -> Result<()> {
    let num_partitions = (4 * usize::from(
        std::thread::available_parallelism().context("Could not count CPU threads")?,
    ))
    .min(16)
    .max(256) // avoid too many files
    .next_power_of_two();

    let max_value = mphf.len();

    // 100MiB in-memory buffer per thread
    let max_buffer_size = 100 * 1024 * 1024;
    let mut pl = concurrent_progress_logger!(
        item_name = "quad",
        display_memory = true,
        local_speed = true,
        expected_updates = approx_num_quads,
    );
    pl.start("Reading and sorting quads...");
    let (mut sorted_quads, num_quads) = quads
        .fold(
            || {
                Ok((
                    pl.clone(),
                    ExternalArraySorter::<4>::new(max_value, max_buffer_size, num_partitions)
                        .context("Could not create sorter ExternalArraySorter")?,
                    0,
                ))
            },
            |acc: Result<(_, _, _)>, quad| {
                let (mut pl, mut sorter, num_quads) = acc.unwrap(); // XXX debug
                // let (mut pl, mut sorter, num_quads) = acc?;
                let quad = quad?;
                let Quad {
                    subject,
                    predicate,
                    object,
                    graph_name,
                } = &quad;
                let compressed_quad = [
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
                ];
                assert!(
                    compressed_quad.iter().all(|term_id| *term_id < max_value),
                    "Got quad {compressed_quad:?} (from {quad:?}), but max value is {}",
                    max_value,
                );
                sorter
                    .push(compressed_quad)
                    .context("Could not push quad to sorter")?;
                pl.light_update();
                Ok((pl, sorter, num_quads + 1))
            },
        )
        .map(|item| {
            let (_pl, mut sorter, num_quads) = item?;
            sorter.flush_buffers()?;
            Ok((sorter, num_quads))
        })
        .reduce(
            || {
                Ok((
                    ExternalArraySorter::<4>::new(max_value, max_buffer_size, num_partitions)
                        .context("Could not create sorter ExternalArraySorter")?,
                    0,
                ))
            },
            |left: Result<(_, _)>, right| {
                let (left_sorter, left_num_quads) = left.unwrap(); //XXX debug
                let (right_sorter, right_num_quads) = right.unwrap(); // XXX debug
                // let (left_sorter, left_num_quads) = left?;
                // let (right_sorter, right_num_quads) = right?;

                Ok((
                    left_sorter
                        .merge(right_sorter)
                        .expect("Could not merge ExternalDeduplicatingStringSorter"),
                    //.context("Could not merge ExternalDeduplicatingStringSorter")?,
                    left_num_quads + right_num_quads,
                ))
            },
        )?;
    pl.done();

    std::fs::create_dir(dst_dir)
        .with_context(|| format!("Could not create {}", dst_dir.display()))?;
    let mut pl = concurrent_progress_logger!(
        item_name = "quad",
        display_memory = true,
        local_speed = true,
        expected_updates = Some(num_quads),
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
            let mut writer = BufBitWriter::new(WordAdapter::<usize, _>::new(file));
            write_sorted_array_file(&mut writer, partition.into_iter(), pl)
                .with_context(|| format!("Could not write quads to {}", path.display()))?;
            let mut file = writer
                .into_inner()
                .with_context(|| format!("Could not flush {}", path.display()))?
                .into_inner();
            file.flush()
                .with_context(|| format!("Could not flush {}", path.display()))?;
            Ok(())
        })?;
    pl.done();

    Ok(())
}
