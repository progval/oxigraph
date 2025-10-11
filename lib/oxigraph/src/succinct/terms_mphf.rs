use super::terms_store::{FrameLender, TermsFile, list_terms_files};
use crate::model::{GraphName, NamedNode, NamedOrBlankNode, Term};
use crate::succinct::terms_store::{serialize_graph_name, serialize_term};
use anyhow::{Context, Result, ensure};
use bytemuck::TransparentWrapper;
use dsi_progress_logger::{ProgressLog, progress_logger};
use epserde::deser::{
    Deserialize as EpDeserialize, DeserializeInner as EpDeserializeInner, MemCase,
};
use epserde::ser::Serialize as EpSerialize;
use lender::{Lender, Lending};
use std::borrow::Borrow;
use std::fs::File;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::path::Path;
use sux::bits::bit_field_vec::BitFieldVec;
use sux::func::{VBuilder, VFunc};
use sux::traits::bit_field_slice::BitFieldSlice;
use sux::utils::{FromIntoIterator, RewindableIoLender};

/// workaround while https://github.com/vigna/sux-rs/pull/78 is not merged
#[derive(Debug, TransparentWrapper)]
#[repr(transparent)]
pub struct RawTerm(pub [u8]);

impl epserde::traits::type_info::TypeHash for RawTerm {
    fn type_hash(hasher: &mut impl Hasher) {
        <&[u8]>::type_hash(hasher)
    }

    fn type_hash_val(&self, hasher: &mut impl Hasher) {
        <&[u8]>::type_hash_val(&&self.0, hasher)
    }
}
impl sux::utils::ToSig<[u64; 2]> for RawTerm {
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 2] {
        <&[u8]>::to_sig(&key.borrow().0, seed)
    }
}

/// workaround while https://github.com/vigna/sux-rs/pull/78 is not merged
#[derive(Debug)]
#[repr(transparent)]
pub struct BoxedRawTerm(pub Box<[u8]>);

impl Borrow<RawTerm> for BoxedRawTerm {
    fn borrow(&self) -> &RawTerm {
        RawTerm::wrap_ref(self.0.as_ref())
    }
}

impl sux::utils::ToSig<[u64; 2]> for BoxedRawTerm {
    fn to_sig(key: impl Borrow<Self>, seed: u64) -> [u64; 2] {
        <&[u8]>::to_sig(key.borrow().0.as_ref(), seed)
    }
}

// Turns a term into an integer
pub trait TermHasher {
    /// Returns the number of known terms
    fn len(&self) -> usize;

    fn hash_bytes(&self, s: impl AsRef<[u8]>) -> Result<usize>;

    fn hash_namedorblanknode(&self, term: &NamedOrBlankNode) -> Result<usize> {
        self.hash_bytes(serialize_term(&term.clone().into())?)
    }
    fn hash_namednode(&self, term: &NamedNode) -> Result<usize> {
        self.hash_bytes(serialize_term(&term.clone().into())?)
    }

    fn hash_term(&self, term: &Term) -> Result<usize> {
        self.hash_bytes(serialize_term(term)?)
    }

    fn hash_graphname(&self, graph_name: &GraphName) -> Result<usize> {
        self.hash_bytes(serialize_graph_name(graph_name)?)
    }
}

pub struct TermMphf<D: BitFieldSlice<usize> = BitFieldVec<usize>> {
    vfunc: MemCase<VFunc<RawTerm, usize, D>>,
    marker: PhantomData<D>,
}

pub type DefaultDeserializedTermMphf =
    TermMphf<<BitFieldVec<usize> as EpDeserializeInner>::DeserType<'static>>;

impl<D: BitFieldSlice<usize>> TermHasher for TermMphf<D> {
    fn len(&self) -> usize {
        self.vfunc.len()
    }

    fn hash_bytes(&self, s: impl AsRef<[u8]>) -> Result<usize> {
        // TODO check in the list of terms store that it is not a collision
        let hash = self.vfunc.get(RawTerm::wrap_ref(s.as_ref()));
        ensure!(
            hash < self.vfunc.len(),
            "hash={hash} for vfunc of length={}",
            self.vfunc.len()
        );
        Ok(hash)
    }
}

impl<D: BitFieldSlice<usize>> TermMphf<D> {
    pub fn serialize(&self, path: impl AsRef<Path>) -> Result<()>
    where
        VFunc<RawTerm, usize, D>: EpSerialize,
    {
        let path = path.as_ref();
        std::fs::create_dir(&path)
            .with_context(|| format!("Could not create {}", path.display()))?;

        let vfunc_path = path.join("mphf.vfunc");
        let mut file = File::create(&vfunc_path)
            .with_context(|| format!("Could not create {}", vfunc_path.display()))?;
        self.vfunc
            .serialize(&mut file)
            .with_context(|| format!("Could write VFunc to {}", vfunc_path.display()))?;
        Ok(())
    }
}

impl TermMphf<BitFieldVec<usize>> {
    /// Alternative to [`Self::load`] that does not force this structure to be in memory
    ///
    /// This saves memory, but is not worth it unless it is accessed infrequently.
    /// If you used `mmap` and a program that should maximize CPU usage does not,
    /// this is probably why. This can be seen as page faults in the `VFunc::get_by_sig`
    /// function when profiling eg. with `cargo flamegraph`.
    pub fn mmap(
        path: impl AsRef<Path>,
    ) -> Result<TermMphf<<BitFieldVec<usize> as EpDeserializeInner>::DeserType<'static>>> {
        let path = path.as_ref();
        let vfunc_path = path.join("mphf.vfunc");

        let flags = epserde::deser::mem_case::Flags::RANDOM_ACCESS;
        let vfunc = <VFunc<RawTerm, usize, BitFieldVec<usize>>>::mmap(&vfunc_path, flags)
            .with_context(|| format!("Could mmap VFunc from {}", vfunc_path.display()))?;
        Ok(TermMphf {
            vfunc,
            marker: PhantomData,
        })
    }

    pub fn load(
        path: impl AsRef<Path>,
    ) -> Result<TermMphf<<BitFieldVec<usize> as EpDeserializeInner>::DeserType<'static>>> {
        let path = path.as_ref();
        let vfunc_path = path.join("mphf.vfunc");

        let vfunc = <VFunc<RawTerm, usize, BitFieldVec<usize>>>::load_mem(&vfunc_path)
            .with_context(|| format!("Could load VFunc from {}", vfunc_path.display()))?;
        Ok(TermMphf {
            vfunc,
            marker: PhantomData,
        })
    }
}

pub fn build_terms_mphf(dir: &Path) -> Result<TermMphf<BitFieldVec<usize>>> {
    let (config, terms_files) = list_terms_files(dir)?;

    let terms_lender = RewindableIoFlattenLender::new(
        terms_files
            .iter()
            .map(|terms_file| {
                let TermsFile {
                    first_term_id: _,
                    num_terms: _,
                    path,
                    compressed_frames,
                } = terms_file;

                sux::utils::FromResultLenderFactory::new(|| {
                    Ok(FrameLender::new(compressed_frames, config.terms_per_frame)
                        .with_context(|| format!("Could not decompress {}", path.display()))
                        .map_err(DecodeError)?
                        .map(
                            lender::hrc_mut!(for<'all> |term: Result<&'all [u8]>| -> Result<
                                    BoxedRawTerm,
                                    DecodeError,
                                > {
                                    Ok(BoxedRawTerm(term.map_err(DecodeError)?.into()))
                                }),
                        ))
                })
            })
            .collect::<Result<_, DecodeError>>()?,
    );

    let mut pl = progress_logger!(
        item_name = "term",
        display_memory = true,
        local_speed = true,
        expected_updates = Some(config.num_terms),
    );
    pl.start("Building MPHF...");

    let builder = VBuilder::<_, BitFieldVec<usize>>::default().expected_num_keys(config.num_terms)
        .check_dups(true)
        .offline(true) // Save memory by spilling to disk
        .low_mem(true) // Save memory by using slightly more CPU;
        ;
    let vfunc = MemCase::encase(
        builder
            .try_build_func::<RawTerm, BoxedRawTerm>(
                terms_lender,
                FromIntoIterator::from(0..config.num_terms),
                &mut pl,
            )
            .context("Could not build VFunc")?,
    );
    pl.done();

    ensure!(
        vfunc.len() == config.num_terms,
        "vfunc.len()={}, expected {}",
        vfunc.len(),
        config.num_terms
    );

    Ok(TermMphf {
        vfunc,
        marker: PhantomData,
    })
}

// Reads a compressed terms file using [`super::terms_store::FrameLender`] on each frame
// struct FileLender<'a> {
// data: &'a [u8],
// lender: FrameLender<'a>,
// terms_per_frame: usize,
// lender_position: usize,
// }
//
// impl<'a> FileLender<'a> {
// pub fn new(data: &'a [u8], terms_per_frame: usize) -> Result<Self> {
// Ok(FileLender {
// data,
// lender: FrameLender::new(data),
// terms_per_frame,
// lender_position: 0,
// })
// }
// }
//
// impl<'a, 'lend> Lending<'lend> for FileLender<'a> {
// type Lend = Result<&'lend BoxedRawTerm, DecodeError>;
// }
//
// impl<'a> Lender for FileLender<'a> {
// fn next(&mut self) -> Option<<Self as Lending<'_>>::Lend> {
// if self.lender_position == self.terms_per_frame - 1 {
// if self.data.is_empty() {
// return None;
// }
// self.lender = FrameLender::new(self.data);
//
// TODO: don't unnecessarily decode the frame twice here
// let frame_length = super::terms_store::get_frame_size(self.data);
// self.data = self.data[frame_length..];
// self.lender_position = 0;
// }
//
// match self.lender.next() {
// Some(res) => {
// self.lender_position += 1;
// Some(res)
// },
// None => None,
// }
// }
// }
//
// impl<R: BufRead + Seek> RewindableIoLender<BoxedRawTerm> for ZstdLengthPrefixedStringLender<R> {
// type Error = DecodeError;
//
// fn rewind(mut self) -> Result<Self, Self::Error> {
// let mut read = self.decoder.finish();
// read.rewind().context("Could not rewind")?;
// self.string = None;
// self.decoder =
// Decoder::with_buffer(read).context("Could not create new decoder to rewind")?;
// Ok(self)
// }
// }

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
struct DecodeError(#[from] anyhow::Error);

/// Equivalent to [`Lender::flatten`] but implements [`RewindableIoLender`]
struct RewindableIoFlattenLender<L> {
    lenders: Vec<L>,
    current_index: usize,
}

impl<L> RewindableIoFlattenLender<L> {
    pub fn new(lenders: Vec<L>) -> Self {
        Self {
            lenders,
            current_index: 0,
        }
    }
}

impl<'lend, L: Lending<'lend>> Lending<'lend> for RewindableIoFlattenLender<L> {
    type Lend = L::Lend;
}

impl<L: Lender> Lender for RewindableIoFlattenLender<L> {
    fn next(&mut self) -> Option<<Self as Lending<'_>>::Lend> {
        // This is equivalent to:
        //
        //  while let Some(current_lender) = self.lenders.get_mut(self.current_index) {
        //      if let Some(item) = current_lender.next() {
        //          return Some(item);
        //      }
        //      // exhausted the current lender, go to the next one
        //      self.current_index += 1
        //  }
        //  // exhausted all lenders
        //  None
        //
        //  but the borrow-checker forces us to write it this way because it doesn't understand we
        //  only borrow one lender at a time.
        self.lenders[self.current_index..]
            .iter_mut()
            .flat_map(|current_lender| {
                if let Some(item) = current_lender.next() {
                    return Some(item);
                }

                // exhausted the current lender, go to the next one
                self.current_index += 1;

                None
            })
            .next()
    }
}

impl<T, L: RewindableIoLender<T>> RewindableIoLender<T> for RewindableIoFlattenLender<L> {
    type Error = <L as RewindableIoLender<T>>::Error;

    fn rewind(mut self) -> Result<Self, Self::Error> {
        let mut new_lenders = Vec::with_capacity(self.lenders.len());
        for lender in self
            .lenders
            .drain(0..=self.current_index.min(self.lenders.len() - 1))
        {
            new_lenders.push(lender.rewind()?);
        }
        new_lenders.extend(self.lenders.drain(..));
        std::mem::swap(&mut new_lenders, &mut self.lenders);
        self.current_index = 0;
        Ok(self)
    }
}
