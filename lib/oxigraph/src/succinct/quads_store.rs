use super::sort::{
    ExternalArraySorter, read_sorted_array_file, read_sorted_array_file_internal,
    write_sorted_array_file,
};
use super::terms_mphf::TermMphf;
use crate::model::Quad;
use anyhow::{Context, Result, anyhow, ensure};
use dsi_bitstream::prelude::*;
use dsi_progress_logger::{ProgressLog, concurrent_progress_logger};
use epserde::deser::Deserialize as EpDeserialize;
use epserde::prelude::Flags;
use epserde::ser::Serialize as EpSerialize;
use lender::IteratorExt;
use mmap_rs::MmapFlags;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use sux::bits::BitFieldVec;
use sux::dict::elias_fano::EfSeq;
use sux::traits::bit_field_slice::BitFieldSlice;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, clap::ValueEnum)]
pub enum QuadOrder {
    Spog,
    Opsg,
}

impl QuadOrder {
    // returns a function that maps [s, p, o, g] to this order
    pub fn mapper(self) -> fn([usize; 4]) -> [usize; 4] {
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

    // returns a function that maps from this order to [s, p, o, g]
    pub fn reverse_mapper(self) -> fn([usize; 4]) -> [usize; 4] {
        fn order_spog(quad: [usize; 4]) -> [usize; 4] {
            quad
        }
        fn order_opsg(quad: [usize; 4]) -> [usize; 4] {
            let [o, p, s, g] = quad;
            [s, p, o, g]
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
    pub num_partitions: usize,
    pub num_quads: usize,
    pub num_terms: usize,
}

pub fn compress_quads(
    quads: impl ParallelIterator<Item = Result<Quad>>,
    dst_dir: &Path,
    mphf: &TermMphf<impl BitFieldSlice<usize> + Sync + Send>,
    approx_num_quads: Option<usize>,
    order: QuadOrder,
) -> Result<()> {
    let mut config = QuadsStoreConfiguration {
        num_partitions: (4 * usize::from(
        std::thread::available_parallelism().context("Could not count CPU threads")?,
    ))
    .min(16)
    .max(256) // avoid too many files
    .next_power_of_two(),
        num_quads: 0,
        num_terms,
    };

    let max_value = num_terms - 1;

    let sorter_pool = thread_local::ThreadLocal::new();

    let order_quad = order.mapper();
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
            let sorter = sorter
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
                compressed_quad.iter().all(|term_id| *term_id <= max_value),
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
                let mut sorter = sorter.into_inner().context("Could not get sorter")?;
                sorter.flush_buffers().context("Could not flush")?;
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
                    Ok(left?
                        .merge(right?)
                        .context("Could not merge ExternalDeduplicatingStringSorter")?)
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

fn get_quad_partitions(dir: &Path) -> Result<(QuadStoreConfiguration, Vec<PathBuf>)> {
    let config_path = dir.join("config.json");
    let config_file = File::open(&config_path)
        .with_context(|| format!("Could not open {}", config_path.display()))?;
    let config: QuadStoreConfiguration = serde_json::from_reader(config_file)
        .with_context(|| format!("Could not read config from {}", config_path.display()))?;

    Ok((
        config.clone(),
        (0..config.num_partitions)
            .map(move |partition_id| dir.join(format!("{partition_id}.quads.bitstream")))
            .collect(),
    ))
}

pub fn par_iter_quads(
    dir: &Path,
    order: QuadOrder,
) -> Result<impl ParallelIterator<Item = Result<[usize; 4]>>> {
    let (_config, partitions) = get_quad_partitions(dir)?;
    Ok(partitions
        .into_par_iter()
        .map(|path| {
            let de_order_quad = order.mapper();
            let data = MmapHelper::mmap(&path, MmapFlags::SEQUENTIAL)
                .with_context(|| format!("Could not mmap array file {}", path.display()))?;
            Ok(
                read_sorted_array_file(BufBitReader::new(MemWordReader::<u64, _>::new(data)), 0)
                    .with_context(|| format!("Could not read array file {}", path.display()))?
                    .map(move |quad| Ok(de_order_quad(quad?))),
            )
        })
        .collect::<Result<Vec<_>>>()?
        .into_par_iter()
        .flatten_iter())
}

pub fn index_frames(dir: &Path) -> Result<()> {
    let (config, partitions) = get_quad_partitions(dir)?;
    let mut pl = concurrent_progress_logger!(
        item_name = "quad",
        display_memory = true,
        local_speed = true,
        expected_updates = Some(config.num_quads * 2), // two passes
    );
    pl.start("Indexing quads");

    partitions
        .into_par_iter()
        .try_for_each_with(pl.clone(), |pl, path| {
            let data = MmapHelper::mmap(&path, MmapFlags::SEQUENTIAL)
                .with_context(|| format!("Could not mmap array file {}", path.display()))?;

            let file_len = std::fs::metadata(&path)
                .with_context(|| format!("Could not stat array file {}", path.display()))?
                .len();

            let mut num_frames = 0;
            read_sorted_array_file_internal::<4>(
                BufBitReader::new(MemWordReader::<u64, _>::new(&data)),
                0,
            )
            .with_context(|| format!("Could not read array file {}", path.display()))?
            .try_for_each(|item| -> Result<_> {
                let (bit_pos, _quad) = item?;
                pl.light_update();

                if bit_pos.is_some() {
                    num_frames += 1;
                }
                Ok(())
            })
            .with_context(|| format!("Could not read frame offsets from {}", path.display()))?;

            let file_len_bits = usize::try_from(file_len * 8)
                .with_context(|| format!("Size (in bits) of {} overflows usize", path.display()))?;
            let mut efb = EliasFanoBuilder::new(num_frames, file_len_bits);

            read_sorted_array_file_internal::<4>(
                BufBitReader::new(MemWordReader::<u64, _>::new(data)),
                0,
            )
            .with_context(|| format!("Could not read array file {}", path.display()))?
            .try_for_each(|item| -> Result<_> {
                let (bit_pos, _quad) = item?;
                pl.light_update();

                let Some(bit_pos) = bit_pos else {
                    // not a new frame
                    return Ok(());
                };

                // Shouldn't fail, we checked the file size before
                let bit_pos = usize::try_from(bit_pos).context("bit pos overflowed usize")?;

                ensure!(
                    bit_pos < file_len_bits,
                    "bit_pos={bit_pos} is past the end of {} ({file_len_bits})",
                    path.display()
                );
                efb.push(bit_pos);

                Ok(())
            })
            .with_context(|| format!("Could not read frame offsets from {}", path.display()))?;
            let ef = efb.build_with_seq();

            let index_file_path = path.with_extension("frames.ef");
            let mut index_file = File::create(&index_file_path)
                .with_context(|| format!("Could not create {}", index_file_path.display()))?;
            ef.serialize(&mut index_file).with_context(|| {
                format!(
                    "Could not write Elias-Fano index to {}",
                    index_file_path.display()
                )
            })?;

            Ok(())
        })
}

pub fn index_quads_by_first_term(dir: &Path) -> Result<()> {
    let (config, partitions) = get_quad_partitions(dir)?;
    let mut pl = concurrent_progress_logger!(
        item_name = "quad",
        display_memory = true,
        local_speed = true,
        expected_updates = Some(config.num_quads),
    );
    pl.start("Indexing quads");

    let num_terms_per_partition = config.num_terms.div_ceil(config.num_partitions);
    partitions
        .into_par_iter()
        .enumerate()
        .try_for_each_with(pl.clone(), |pl, (partition_id, path)| {
            let data = webgraph::utils::MmapHelper::mmap(&path, MmapFlags::SEQUENTIAL)
                .with_context(|| format!("Could not mmap array file {}", path.display()))?;
            let num_terms_in_partition = if partition_id == config.num_partitions - 1 {
                config.num_terms.checked_sub(num_terms_per_partition * (config.num_partitions - 1)).unwrap()
            } else {
                num_terms_per_partition
            };
            let file_len = std::fs::metadata(&path)
                .with_context(|| format!("Could not stat array file {}", path.display()))?
                .len();

            let first_first_term_in_partition = num_terms_per_partition * partition_id;
            let mut previous_relative_first_term = 0;
            let mut previous_bit_pos = 0;

            let file_len_bits = usize::try_from(file_len * 8).with_context(|| {
                    format!("Size (in bits) of {} overflows usize", path.display())
                })?;
            let mut efb = sux::dict::elias_fano::EliasFanoBuilder::new(
                num_terms_in_partition,
                file_len_bits,
            );
            read_sorted_array_file_internal::<4>(BufBitReader::new(MemWordReader::<u64, _>::new(
                data,
            )), 0)
            .with_context(|| format!("Could not read array file {}", path.display()))?
            .try_for_each(|item| -> Result<_> {
                let (bit_pos, quad) = item?;
                pl.light_update();

                let Some(bit_pos) = bit_pos else {
                    // not a new frame
                    return Ok(());
                };

                // Shouldn't fail, we checked the file size before
                let bit_pos = usize::try_from(bit_pos).context("bit pos overflowed usize")?;

                // use the term relative to the start of the partition so we don't
                // have to pad the beginning of the Elias-Fano sequence with billions
                // of 0s.
                let relative_first_term = quad[0]
                    .checked_sub(first_first_term_in_partition)
                    .with_context(|| format!("quad {quad:?} in wrong partition {partition_id} (first term should be >= {first_first_term_in_partition})"))?;
                ensure!(
                    relative_first_term < num_terms_in_partition,
                    "quad {quad:?} is in wrong partition {partition_id} (first term should be >= {} and < {}, num_terms_in_partition={num_terms_in_partition})",
                    first_first_term_in_partition, first_first_term_in_partition + num_terms_in_partition,
                );
                if previous_relative_first_term != relative_first_term {
                    // don't push to the Elias-Fano sequence if we already pushed
                    // the position of the first frame containing the first_term
                    ensure!(relative_first_term > previous_relative_first_term, "{} after {}", first_first_term_in_partition + relative_first_term, first_first_term_in_partition + previous_relative_first_term);

                    // fill the blanks for terms with no quad
                    for _ in (previous_relative_first_term + 1)..relative_first_term {
                        efb.push(previous_bit_pos);
                    }

                    ensure!(bit_pos < file_len_bits, "bit_pos={bit_pos} is past the end of {} ({file_len_bits})", path.display());
                    efb.push(bit_pos);

                    previous_relative_first_term = relative_first_term;
                }

                // used as placeholder for terms that don't have any quad.
                previous_bit_pos = bit_pos;

                Ok(())
            })
            .with_context(|| format!("Could not read frame offsets from {}", path.display()))?;
            let ef = efb.build_with_seq();

            let index_file_path = path.with_extension("1term.ef");
            let mut index_file = File::create(&index_file_path)
                .with_context(|| format!("Could not create {}", index_file_path.display()))?;
            ef.serialize(&mut index_file).with_context(|| {
                format!(
                    "Could not write Elias-Fano index to {}",
                    index_file_path.display()
                )
            })?;

            Ok(())
        })
}

pub fn index_quads_by_first_two_terms(dir: &Path) -> Result<()> {
    let (config, partitions) = get_quad_partitions(dir)?;
    let mut pl = concurrent_progress_logger!(
        item_name = "partitions",
        display_memory = true,
        local_speed = true,
        expected_updates = Some(config.num_partitions),
    );
    pl.start("Indexing quads");

    partitions
        .into_par_iter()
        .try_for_each_with(pl.clone(), |pl, path| {
            let data = MmapHelper::mmap(&path, MmapFlags::SEQUENTIAL)
                .with_context(|| format!("Could not mmap array file {}", path.display()))?;

            let data = &data;
            let get_iter = || {
                read_sorted_array_file_internal::<4>(BufBitReader::new(
                    MemWordReader::<u64, _>::new(data),
                ), 0)
                .with_context(|| format!("Could not read array file {}", path.display()),
                )
            };

            let keys = sux::utils::lenders::FromResultLenderFactory::new(|| -> Result<_, _> {
                let mut previous_pair = None;
                Ok(get_iter()?
                    .map(
                        move |item: Result<(Option<u64>, [usize; 4])>| -> Result<_> {
                            let (_position, quad) = item?;
                            let pair = TermsPair(quad[0], quad[1]);
                            if Some(pair) == previous_pair {
                                return Ok(None);
                            }
                            previous_pair = Some(pair);
                            Ok(Some(pair))
                        },
                    )
                    .flat_map(Result::transpose)
                    .into_lender())
            })?;
            let values = sux::utils::lenders::FromResultLenderFactory::new(|| -> Result<_> {
                let mut frame_id: Option<usize> = None;
                let mut first_frame_id_of_current_first_term = 0;
                let mut previous_pair = None;
                Ok(get_iter()?
                    .map(
                        move |item: Result<(Option<u64>, [usize; 4])>| -> Result<_> {
                            let (position, quad) = item?;
                            if let Some(position) = position {
                                if frame_id.is_none() {
                                    ensure!(
                                        position == 0,
                                        "Got position={position:?} for first frame"
                                    );
                                }
                                frame_id = Some(frame_id.map(|id| id + 1).unwrap_or(0));
                            }
                            let frame_id = frame_id.context("Got quad before first frame")?;

                            let pair = TermsPair(quad[0], quad[1]);
                            if Some(pair) == previous_pair {
                                return Ok(None);
                            } else if let Some(previous_pair) = previous_pair {
                                if previous_pair.0 != pair.0 {
                                    first_frame_id_of_current_first_term = frame_id;
                                }
                            }
                            previous_pair = Some(pair);
                            // store the offset from the first frame of (x, _, _, _) to the first
                            // frame of (x, y, _, _).
                            // This is a smaller value than storing the first frame of (x, y_, _, )
                            // as an absolute number, so it takes less space on disk.
                            // And we can use the Elias-Fano sequence built by
                            // index_quads_by_first_term to get the first frame of (x, _, _, _)
                            Ok(Some(frame_id.checked_sub(first_frame_id_of_current_first_term).expect("(x, y, _, _) is before the first occurence of (x, _, _, _)")))
                        },
                    )
                    .flat_map(Result::transpose)
                    .into_lender())
            })?;

            let builder = sux::func::VBuilder::<_, BitFieldVec<usize>>::default()
                .offline(true) // Save memory by spilling to disk
                .low_mem(true) // Save memory by using slightly more CPU;
                ;
            let vfunc = builder.try_build_func(keys, values, dsi_progress_logger::no_logging!())?;
            let vfunc_path = path.with_extension("2terms.vfunc");
            let mut file = File::create(&vfunc_path)
                .with_context(|| format!("Could not create {}", vfunc_path.display()))?;
            vfunc.serialize(&mut file).with_context(|| {
                format!("Could not write terms MPHF to {}", vfunc_path.display())
            })?;

            pl.update();

            Ok(())
        })
}

#[derive(Clone, Copy, Debug, epserde::Epserde, PartialEq, Eq)]
pub struct TermsPair(usize, usize);

impl<S> sux::utils::ToSig<S> for TermsPair
where
    [usize]: sux::utils::ToSig<S>,
{
    fn to_sig(key: impl std::borrow::Borrow<Self>, seed: u64) -> S {
        let key = key.borrow();
        <[usize]>::to_sig([key.0, key.1], seed)
    }
}

// #[derive(Clone)]
// struct FnIntoIter<'a, I: IntoIterator, F: FnMut() -> I>(&'a F);
//
// impl<'a, I: IntoIterator, F: FnMut() -> I> IntoIterator for FnIntoIter<'a, I, F> {
// type IntoIter = I::IntoIter;
// type Item = I::Item;
//
// fn into_iter(self) -> Self::IntoIter {
// (self.0)().into_iter()
// }
// }
