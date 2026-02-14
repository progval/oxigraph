use super::sort::{ExternalArraySorter, SortedArraysFile};
use super::terms_mphf::{TermHasher, TermMphf};
use crate::model::Quad;
use crate::succinct::queryable_dataset::SuccinctDatasetError;
use anyhow::{Context, Result, anyhow, ensure};
use dsi_progress_logger::{ProgressLog, concurrent_progress_logger};
use epserde::deser::mem_case::{Flags, MemCase};
use epserde::deser::{DeserInner as EpDeserInner, Deserialize as EpDeserialize};
use epserde::ser::Serialize as EpSerialize;
use fallible_iterator::FallibleIterator;
use lender::{FallibleLender, IteratorExt};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use sux::bits::BitFieldVec;
use sux::dict::elias_fano::{EfSeq, EfSeqDict, EliasFanoBuilder};
use sux::func::{VBuilder, VFunc};
use sux::traits::bit_field_slice::BitFieldSlice;
use sux::traits::{IndexedDict, IndexedSeq};

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

pub fn compress_parsed_quads<D: BitFieldSlice<usize> + Sync + Send>(
    quads: impl ParallelIterator<Item = Result<Quad>>,
    dst_dir: &Path,
    mphf: &TermMphf<D>,
    approx_num_quads: Option<usize>,
    order: QuadOrder,
) -> Result<()>
where
    for<'a> D: EpDeserInner<DeserType<'a>: BitFieldSlice<usize>>,
{
    let compressed_quads = quads.map(|quad| {
        let quad = quad.context("Could not read quad")?;
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
        ensure!(
            compressed_quad.iter().all(|term_id| *term_id < mphf.len()),
            "Got quad {compressed_quad:?} (from {quad:?}), but there are fewer known terms ({})",
            mphf.len(),
        );
        Ok(compressed_quad)
    });
    compress_quads(
        compressed_quads,
        dst_dir,
        mphf.len(),
        approx_num_quads,
        order,
    )
}

pub fn compress_quads(
    quads: impl ParallelIterator<Item = Result<[usize; 4]>>,
    dst_dir: &Path,
    num_terms: usize,
    approx_num_quads: Option<usize>,
    order: QuadOrder,
) -> Result<()> {
    let mut config = QuadStoreConfiguration {
        num_partitions: (4 * usize::from(
        std::thread::available_parallelism().context("Could not count CPU threads")?,
    ))
    .max(16)
    .min(256) // avoid too many files
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
            let (pl, sorter) = acc.as_mut().expect("Could not get sorter"); // FIXME don't panic
            let sorter = sorter
                .as_mut()
                .map_err(|e| anyhow!("Could not create sorter ExternalArraySorter: {e:#?}"))?;
            let quad = order_quad(quad?);
            sorter.push(quad).context("Could not push quad to sorter")?;
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
            SortedArraysFile::create(&path, partition.into_iter(), pl)
                .with_context(|| format!("Could not write quads to {}", path.display()))?;
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

pub(super) fn get_quad_partitions(dir: &Path) -> Result<(QuadStoreConfiguration, Vec<PathBuf>)> {
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
            Ok(SortedArraysFile::mmap(&path)
                .with_context(|| format!("Could not mmap array file {}", path.display()))?
                .owned_iter()
                .with_context(|| format!("Could not read array file {}", path.display()))?
                .map(move |quad| Ok(de_order_quad(quad?))))
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
            let arrays_file = SortedArraysFile::<4>::mmap(&path)
                .with_context(|| format!("Could not mmap array file {}", path.display()))?;

            let mut num_frames: usize = 0;
            arrays_file
                .iter_with_positions(0)
                .with_context(|| format!("Could not read array file {}", path.display()))?
                .try_for_each(|item| -> Result<_> {
                    let (bit_pos, _quad) = item?;
                    pl.light_update();

                    if bit_pos.is_some() {
                        num_frames = num_frames
                            .checked_add(1)
                            .context("number of frames overflowed usize")?;
                    }
                    Ok(())
                })
                .with_context(|| format!("Could not read frame offsets from {}", path.display()))?;

            let file_len_bits = usize::try_from(arrays_file.file_len())
                .with_context(|| format!("Size (in bytes) of {} overflows usize", path.display()))?
                .checked_mul(8)
                .with_context(|| format!("Size (in bits) of {} overflows usize", path.display()))?;
            let mut efb = EliasFanoBuilder::new(num_frames, file_len_bits);

            arrays_file
                .iter_with_positions(0)
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
            let ef = efb.build_with_seq_and_dict();

            let index_file_path = path.with_extension("frames.ef");
            let mut index_file = File::create(&index_file_path)
                .with_context(|| format!("Could not create {}", index_file_path.display()))?;
            // SAFETY: this may leak padding bytes, but we only read data that is to be shared
            // alongside the index.
            unsafe { ef.serialize(&mut index_file) }.with_context(|| {
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
            let num_terms_in_partition = if partition_id == config.num_partitions - 1 {
                // saturating_sub is needed when there are fewer terms than partitions
                // (ie. in tiny databases)
                config.num_terms.saturating_sub(num_terms_per_partition * (config.num_partitions - 1))
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
            let mut efb = EliasFanoBuilder::new(
                num_terms_in_partition,
                file_len_bits,
            );
            efb.push(0); // first term is always in the first frame if present

            SortedArraysFile::<4>::mmap(&path)
            .with_context(|| format!("Could not mmap array file {}", path.display()))?
            .iter_with_positions(0)
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
            ensure!(ef.len() == num_terms_in_partition, "Expected {num_terms_in_partition} terms in partition {partition_id}, wrote {} in Elias-Fano index", ef.len());

            let index_file_path = path.with_extension("1term.ef");
            let mut index_file = File::create(&index_file_path)
                .with_context(|| format!("Could not create {}", index_file_path.display()))?;

            // SAFETY: this may leak padding bytes, but we only read data that is to be shared
            // alongside the index.
            unsafe { ef.serialize(&mut index_file) }.with_context(|| {
                format!(
                    "Could not write Elias-Fano index to {}",
                    index_file_path.display()
                )
            })?;

            Ok(())
        })?;

    pl.done();
    Ok(())
}

pub fn index_quads_by_first_two_terms(dir: &Path) -> Result<()> {
    let (config, partitions) = get_quad_partitions(dir)?;
    let mut pl = concurrent_progress_logger!(
        item_name = "partition",
        display_memory = true,
        local_speed = true,
        expected_updates = Some(config.num_partitions),
    );
    pl.start("Indexing quads");

    partitions
        .into_par_iter()
        .try_for_each_with(pl.clone(), |pl, path| {
            let array_file = SortedArraysFile::mmap(&path)
                .with_context(|| format!("Could not mmap array file {}", path.display()))?;

            let get_iter = || {
                array_file.iter_with_positions(0)
                .with_context(|| format!("Could not read array file {}", path.display()))
                .map(|iter| iter.map(|item| item.map_err(SuccinctDatasetError::from)))
            };

            let keys = sux::utils::lenders::FromIntoFallibleLenderFactory::new(|| -> Result<_, SuccinctDatasetError> {
                let mut previous_pair = None;
                Ok(super::from_iter_ref::from_iter_ref(fallible_iterator::convert::<_, SuccinctDatasetError, _>(get_iter()?)
                    .filter_map(
                        move |item: (Option<u64>, [usize; 4])| -> Result<_, SuccinctDatasetError> {
                            let (_position, quad) = item;
                            let pair = TermsPair(quad[0], quad[1]);
                            if Some(pair) == previous_pair {
                                return Ok(None);
                            }
                            previous_pair = Some(pair);
                            Ok(Some(pair))
                        },
                    )
                    ))
            })?;
            let values = sux::utils::lenders::FromIntoFallibleLenderFactory::new(|| -> Result<_, SuccinctDatasetError> {
                let mut frame_id: Option<usize> = None;
                let mut first_frame_id_of_current_first_term = 0;
                let mut previous_pair = None;
                Ok(super::from_iter_ref::from_iter_ref(fallible_iterator::convert(get_iter()?)
                    .filter_map(
                        move |item: (Option<u64>, [usize; 4])| -> Result<_, SuccinctDatasetError> {
                            let (position, quad) = item;
                            if let Some(position) = position {
                                if frame_id.is_none() {
                                    if position != 0 {
                                        Err(anyhow!("Got position={position:?} for first frame"))?;
                                    }
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
                    )))
            })?;

            let builder = VBuilder::<_, BitFieldVec<usize>>::default()
                .offline(true) // Save memory by spilling to disk
                .low_mem(true) // Save memory by using slightly more CPU;
                ;
            let vfunc = builder.try_build_func(keys, values, dsi_progress_logger::no_logging!())?;
            let vfunc_path = path.with_extension("2terms.vfunc");
            let mut file = File::create(&vfunc_path)
                .with_context(|| format!("Could not create {}", vfunc_path.display()))?;
            // SAFETY: this may leak padding bytes, but we only read data that is to be shared
            // alongside the vfunc.
            unsafe { vfunc.serialize(&mut file) }.with_context(|| {
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

pub struct QuadStore {
    config: QuadStoreConfiguration,
    path: PathBuf,
    partitions: Vec<QuadPartition>,
}

impl QuadStore {
    pub fn mmap(path: PathBuf) -> Result<Self> {
        let (config, partition_paths) = get_quad_partitions(&path).with_context(|| {
            format!(
                "Could not read quad store configuration from {}",
                path.display()
            )
        })?;

        let num_terms_per_partition = config.num_terms.div_ceil(config.num_partitions);
        let partitions = partition_paths
            .into_par_iter()
            .enumerate()
            .map(|(partition_id, partition_path)| {
                let first_first_term_in_partition = num_terms_per_partition * partition_id;
                QuadPartition::mmap(first_first_term_in_partition, partition_path)
            })
            .collect::<Result<_>>()
            .with_context(|| {
                format!(
                    "Could not open partitions of quad store at {}",
                    path.display()
                )
            })?;

        Ok(Self {
            config,
            path,
            partitions,
        })
    }

    fn get_partition(&self, term: usize) -> Result<&QuadPartition> {
        ensure!(
            term < self.config.num_terms,
            "Invalid term: {term} (only {} terms in store {})",
            self.config.num_terms,
            self.path.display()
        );
        let num_terms_per_partition = self.config.num_terms.div_ceil(self.config.num_partitions);
        Ok(&self.partitions[term / num_terms_per_partition])
    }

    pub fn iter_all_quads(
        &self,
    ) -> Result<Option<impl Iterator<Item = Result<[usize; 4]>> + use<>>> {
        let parts = self
            .partitions
            .iter()
            .map(|partition| partition.iter_all_quads())
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(parts.into_iter().flatten()))
    }

    pub fn iter_quads_by_first_term(
        &self,
        term: usize,
    ) -> Result<Option<impl Iterator<Item = Result<[usize; 4]>> + use<>>> {
        self.get_partition(term)?.iter_quads_by_first_term(term)
    }

    pub fn iter_quads_by_first_two_terms(
        &self,
        term1: usize,
        term2: usize,
    ) -> Result<Option<impl Iterator<Item = Result<[usize; 4]>> + use<>>> {
        self.get_partition(term1)?
            .iter_quads_by_first_two_terms(term1, term2)
    }

    /// Same as [`iter_quads_by_first_two_terms`](Self::iter_quads_by_first_two_terms) but does not
    /// perform early checks for false positives
    ///
    /// This is useful when the caller knows it is very likely to get results
    pub fn iter_quads_by_first_two_terms_without_pruning(
        &self,
        term1: usize,
        term2: usize,
    ) -> Result<Option<impl Iterator<Item = Result<[usize; 4]>> + use<>>> {
        self.get_partition(term1)?
            .iter_quads_by_first_two_terms_without_pruning(term1, term2)
    }
}

struct QuadPartition {
    _path: PathBuf,
    first_first_term: usize,
    quads: SortedArraysFile<4>,
    /// frame_id -> bit_position
    frame_index: MemCase<EfSeqDict>,
    /// term -> bit_position
    first_term_index: MemCase<EfSeq>,
    /// TermsPair -> frame_id (may have false positives)
    first_two_terms_index: MemCase<VFunc<TermsPair, usize, BitFieldVec<usize>>>,
}

impl QuadPartition {
    pub fn mmap(first_first_term: usize, path: PathBuf) -> Result<Self> {
        let quads = SortedArraysFile::mmap(&path)
            .with_context(|| format!("Could not mmap array file {}", path.display()))?;

        let frame_index_path = path.with_extension("frames.ef");
        let frame_index = unsafe { EfSeqDict::mmap(&frame_index_path, Flags::RANDOM_ACCESS) }
            .with_context(|| {
                format!(
                    "Could not epdeserialize frame index from {}",
                    frame_index_path.display()
                )
            })?;

        let first_term_index_path = path.with_extension("1term.ef");
        let first_term_index = unsafe { EfSeq::mmap(&first_term_index_path, Flags::RANDOM_ACCESS) }
            .with_context(|| {
                format!(
                    "Could not epdeserialize first-term index from {}",
                    first_term_index_path.display()
                )
            })?;

        let first_two_terms_index_path = path.with_extension("2terms.vfunc");
        let first_two_terms_index = unsafe {
            VFunc::<TermsPair, usize, BitFieldVec<usize>>::mmap(
                &first_two_terms_index_path,
                Flags::RANDOM_ACCESS,
            )
        }
        .with_context(|| {
            format!(
                "Could not epdeserialize first-two-terms index from {}",
                first_two_terms_index_path.display()
            )
        })?;
        let res = Self {
            _path: path,
            first_first_term,
            quads,
            frame_index,
            first_term_index,
            first_two_terms_index,
        };
        if let Some(first_quad) = res.iter_all_quads().context("Could not iter quads")?.next() {
            let first_quad = first_quad?;
            ensure!(
                first_quad[0] >= first_first_term,
                "Expected partition to start at {first_first_term} but first quad is {first_quad:?}"
            );
        }
        Ok(res)
    }

    pub fn iter_all_quads(&self) -> Result<impl Iterator<Item = Result<[usize; 4]>> + use<>> {
        Ok(self.quads.owned_iter()?)
    }

    pub fn iter_quads_by_first_term(
        &self,
        term: usize,
    ) -> Result<Option<impl Iterator<Item = Result<[usize; 4]>> + use<>>> {
        // get the positition of the first frame that contains a quad with the term.
        // If the first term is not in any quad, then this is the frame of a quad it
        // would come right after
        let from_bit_position = self
            .first_term_index
            .uncase()
            .get(term - self.first_first_term);

        let mut took_error = false;
        Ok(Some(
            self.quads
                .owned_iter_from_position(from_bit_position)?
                .skip_while(move |quad| {
                    if let Ok(quad) = quad {
                        // skip quads until we find one that matches
                        quad[0] < term
                    } else {
                        // in case of error, let it through ASAP
                        false
                    }
                })
                .take_while(move |quad| {
                    if let Ok(quad) = quad {
                        // take quads until we find one that doesn't matches
                        quad[0] == term
                    } else {
                        // in case of errors, let the first one through then stop,
                        // so that the caller 1. knows about it  2. doesn't materialize
                        // a potentially infinite iterator of errors
                        if took_error {
                            false
                        } else {
                            took_error = true;
                            true
                        }
                    }
                }),
        ))
    }

    pub fn iter_quads_by_first_two_terms(
        &self,
        term1: usize,
        term2: usize,
    ) -> Result<Option<impl Iterator<Item = Result<[usize; 4]>> + use<>>> {
        self._iter_quads_by_first_two_terms::<true>(term1, term2)
    }

    /// Same as [`iter_quads_by_first_two_terms`](Self::iter_quads_by_first_two_terms) but does not
    /// perform early checks for false positives
    ///
    /// This is useful when the caller knows it is very likely to get results
    pub fn iter_quads_by_first_two_terms_without_pruning(
        &self,
        term1: usize,
        term2: usize,
    ) -> Result<Option<impl Iterator<Item = Result<[usize; 4]>> + use<>>> {
        self._iter_quads_by_first_two_terms::<false>(term1, term2)
    }

    #[inline(always)]
    fn _iter_quads_by_first_two_terms<const EARLY_PRUNING: bool>(
        &self,
        term1: usize,
        term2: usize,
    ) -> Result<Option<impl Iterator<Item = Result<[usize; 4]>> + use<EARLY_PRUNING>>> {
        let first_term_index = self.first_term_index.uncase();
        let frame_index = self.frame_index.uncase();

        let relative_term1 = term1
            .checked_sub(self.first_first_term)
            .context("term1 is before the start of the partition")?;
        ensure!(
            relative_term1 < first_term_index.len(),
            "term1 is after the end of the partition"
        );

        // get the offset of the id of the first frame matching (term1, term2, _, _) from the first
        // frame matching (term1, _, _, _) if any, or nonsense if no quad matches (term1, term2, _, _)
        let relative_frame_id = self
            .first_two_terms_index
            .uncase()
            .get(TermsPair(term1, term2));

        // get the id of the first frame matching (term1, _, _, _)
        let position_of_first_frame_with_first_term = first_term_index.get(relative_term1);
        let id_of_first_frame_with_first_term = frame_index.index_of(position_of_first_frame_with_first_term)
            .with_context(|| format!("First frame matching ({term1}, _, _, _) has position {position_of_first_frame_with_first_term}, but the frame index does not know any frame at that position"))?;

        // add them together
        let Some(maybe_first_frame_id) =
            relative_frame_id.checked_add(id_of_first_frame_with_first_term)
        else {
            return Ok(None);
        };

        if maybe_first_frame_id > frame_index.len() {
            // frame does not exist, so the `first_two_terms_index` returned a false positive
            return Ok(None);
        }
        let maybe_from_bit_position = frame_index.get(maybe_first_frame_id);

        if EARLY_PRUNING {
            // Quick check based only on the first term.
            // It is redundant with the next checks (based on the first two terms) but
            // is faster because it does a read in the frame index (small EF) right after the read we
            // just did (so it's most likely already cached) instead of a random read in the quad file
            if relative_term1 + 1 < first_term_index.len() {
                if first_term_index.get(relative_term1 + 1) < maybe_from_bit_position {
                    // the first match of `(relative_term1+1, _, _, _)` has to be after (or equal to)
                    // the first match of `(relative_term1, term2, _, _)`'.
                    // If it is not, it means `first_two_terms_index` returned a false positive
                    return Ok(None);
                }
            }

            if maybe_first_frame_id > 0 {
                // maybe_first_frame_id is not the id of the first frame.
                // Let's get the first quad of the previous frame
                let previous_frame_bit_position = frame_index.get(maybe_first_frame_id - 1);
                let first_quad_in_previous_frame = self
                    .quads
                    .iter_from_position(previous_frame_bit_position)?
                    .next()
                    .with_context(|| {
                        format!("Got no quad when reading from frame {maybe_first_frame_id}-1")
                    })?
                    .with_context(|| format!("Could not peek frame {maybe_first_frame_id}-1"))?;
                if (
                    first_quad_in_previous_frame[0],
                    first_quad_in_previous_frame[1],
                ) > (term1, term2)
                {
                    // maybe_first_frame_id was allegedly the id of the first frame
                    // containing (term1, term2, _, _).
                    // However, we find that maybe_first_frame_id-1 contains a quad
                    // that comes after (term1, term2, _, _).
                    // This means that maybe_first_frame_id was a false positive.
                    return Ok(None);
                }
            }

            if maybe_first_frame_id + 1 < frame_index.len() {
                // maybe_first_frame_id is not the id of the last frame.
                // Let's get the first quad of the next frame
                let next_frame_bit_position = frame_index.get(maybe_first_frame_id + 1);
                let first_quad_in_next_frame = self
                    .quads
                    .iter_from_position(next_frame_bit_position)?
                    .next()
                    .with_context(|| {
                        format!("Got no quad when reading from frame {maybe_first_frame_id}+1")
                    })?
                    .with_context(|| format!("Could not peek frame {maybe_first_frame_id}+1"))?;
                if (first_quad_in_next_frame[0], first_quad_in_next_frame[1]) < (term1, term2) {
                    // maybe_first_frame_id was allegedly the id of the first frame
                    // containing (term1, term2, _, _).
                    // However, we find that maybe_first_frame_id+1 contains a quad
                    // that comes before (term1, term2, _, _).
                    // This means that maybe_first_frame_id was a false positive.
                    return Ok(None);
                }
            }
        }

        // At this point, we're still not sure `maybe_from_bit_position` is not a false
        // positive.
        // However, thanks to the previous checks we won't read more than one frame
        // if it is a false positive.

        let mut took_error = false;
        Ok(Some(
            self.quads
                .owned_iter_from_position(maybe_from_bit_position)?
                .skip_while(move |quad| {
                    if let Ok(quad) = quad {
                        // skip quads until we find one that matches
                        (quad[0], quad[1]) < (term1, term2)
                    } else {
                        // in case of error, let it through ASAP
                        false
                    }
                })
                .take_while(move |quad| {
                    if let Ok(quad) = quad {
                        // take quads until we find one that doesn't matches
                        (quad[0], quad[1]) == (term1, term2)
                    } else {
                        // in case of errors, let the first one through then stop,
                        // so that the caller 1. knows about it  2. doesn't materialize
                        // a potentially infinite iterator of errors
                        if took_error {
                            false
                        } else {
                            took_error = true;
                            true
                        }
                    }
                }),
        ))
    }
}
