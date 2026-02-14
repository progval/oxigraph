use super::terms_store::{FrameLender, TermsFile, list_terms_files};
use crate::model::{GraphName, NamedNode, NamedOrBlankNode, Term};
use crate::succinct::terms_store::{deserialize_term, serialize_graph_name, serialize_term};
use anyhow::{Context, Result, ensure};
use bytemuck::TransparentWrapper;
use dsi_progress_logger::{ProgressLog, progress_logger};
use epserde::deser::{DeserInner as EpDeserInner, Deserialize as EpDeserialize, MemCase};
use epserde::ser::Serialize as EpSerialize;
use lender::{FallibleLender, FallibleLending};
use std::borrow::Borrow;
use std::fs::File;
use std::hash::Hasher;
use std::marker::PhantomData;
use std::path::Path;
use sux::bits::bit_field_vec::BitFieldVec;
use sux::func::{VBuilder, VFunc};
use sux::traits::bit_field_slice::BitFieldSlice;
use sux::utils::FallibleRewindableLender;

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

    /// Returns true if there are no known terms
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn hash_bytes(&self, s: impl AsRef<[u8]>) -> Result<usize>;

    fn hash_namedorblanknode(&self, term: &NamedOrBlankNode) -> Result<usize> {
        self.hash_bytes(serialize_term(&term.clone().into(), None)?)
    }
    fn hash_namednode(&self, term: &NamedNode) -> Result<usize> {
        self.hash_bytes(serialize_term(&term.clone().into(), None)?)
    }

    fn hash_term(&self, term: &Term) -> Result<usize> {
        self.hash_bytes(serialize_term(term, None)?)
    }

    fn hash_graphname(&self, graph_name: &GraphName) -> Result<usize> {
        self.hash_bytes(serialize_graph_name(graph_name))
    }
}

pub struct TermMphf<
    D: BitFieldSlice<usize> = BitFieldVec<usize>,
    V: EpDeserInner = VFunc<RawTerm, usize, D>,
> where
    for<'a> D: EpDeserInner<DeserType<'a>: BitFieldSlice<usize>>,
{
    vfunc: MemCase<V>,
    marker: PhantomData<D>,
}

pub type DefaultDeserializedTermMphf = TermMphf<BitFieldVec<usize>>;

impl<D: BitFieldSlice<usize>> TermHasher for TermMphf<D>
where
    for<'a> D: EpDeserInner<DeserType<'a>: BitFieldSlice<usize>>,
{
    fn len(&self) -> usize {
        self.vfunc.uncase().len()
    }

    fn hash_bytes(&self, s: impl AsRef<[u8]>) -> Result<usize> {
        let vfunc = self.vfunc.uncase();

        // TODO check in the list of terms store that it is not a collision
        let hash = vfunc.get(RawTerm::wrap_ref(s.as_ref()));
        ensure!(
            hash < vfunc.len(),
            "hash={hash} for vfunc of length={}",
            vfunc.len()
        );
        Ok(hash)
    }
}

impl<D: BitFieldSlice<usize>> TermMphf<D, epserde::deser::Owned<VFunc<RawTerm, usize, D>>>
where
    for<'a> D: EpDeserInner<DeserType<'a>: BitFieldSlice<usize>>,
    VFunc<RawTerm, usize, D>: EpSerialize,
{
    pub fn serialize(&self, path: impl AsRef<Path>) -> Result<()>
    where
        VFunc<RawTerm, usize, D>: EpSerialize,
    {
        let path = path.as_ref();
        std::fs::create_dir_all(path)
            .with_context(|| format!("Could not create {}", path.display()))?;

        let vfunc_path = path.join("mphf.vfunc");
        let mut file = File::create(&vfunc_path)
            .with_context(|| format!("Could not create {}", vfunc_path.display()))?;
        // SAFETY: this may leak padding bytes, but we only read data that is to be shared
        // alongside the vfunc.
        unsafe { self.vfunc.uncase().serialize(&mut file) }
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
    pub fn mmap(path: impl AsRef<Path>) -> Result<TermMphf<BitFieldVec<usize>>> {
        let path = path.as_ref();
        let vfunc_path = path.join("mphf.vfunc");

        let flags = epserde::deser::mem_case::Flags::RANDOM_ACCESS;
        let vfunc =
            unsafe { <VFunc<RawTerm, usize, BitFieldVec<usize>>>::mmap(&vfunc_path, flags) }
                .with_context(|| format!("Could mmap VFunc from {}", vfunc_path.display()))?;
        Ok(TermMphf {
            vfunc,
            marker: PhantomData,
        })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<TermMphf<BitFieldVec<usize>>> {
        let path = path.as_ref();
        let vfunc_path = path.join("mphf.vfunc");

        let vfunc = unsafe { <VFunc<RawTerm, usize, BitFieldVec<usize>>>::load_mem(&vfunc_path) }
            .with_context(|| format!("Could load VFunc from {}", vfunc_path.display()))?;
        Ok(TermMphf {
            vfunc,
            marker: PhantomData,
        })
    }
}

pub fn build_terms_mphf(dir: &Path, dest: &Path) -> Result<()> {
    let (config, terms_files) = list_terms_files(dir)?;
    let dictionary = config.get_dictionary(dir)?;

    let terms_lender = FallibleRewindableFlattenLender::new(
        terms_files
            .iter()
            .map(|terms_file| {
                let TermsFile {
                    compressed_frames, ..
                } = terms_file;

                sux::utils::FromIntoFallibleLenderFactory::new(|| {
                    Ok(super::from_iter_ref::from_iter_ref(
                        FrameLender::new(compressed_frames, config.terms_per_frame)
                        .map_err(DecodeError) // on Result<Item>
                        .map(
                            lender::hrc_mut!(for<'all> |term: &'all [u8]| -> Result<
                                    BoxedRawTerm,
                                    DecodeError,
                                > {
                                    let term = if term.is_empty() {
                                        // FIXME: that's GraphName::DefaultGraph because we currently store
                                        // graph names in the terms store, but we shouldn't.
                                        term.to_vec()
                                    } else {
                                        match &dictionary {
                                            // reencode without the dictionary, to avoid dictionary
                                            // lookups when hashing terms
                                            // FIXME: this seems to be buggy somehow, and makes the
                                            // quads extraction crash, claiming it can't hash
                                            // some literals with datatype
                                            Some(dictionary) => serialize_term(
                                                &deserialize_term(term, Some(dictionary))
                                                    .context("Could not deserialize term")?,
                                                None,
                                            )
                                            .context("Could not serialize term")?,
                                            None => term.to_vec(),
                                        }
                                    };
                                    Ok(BoxedRawTerm(term.into()))
                                }),
                        )
                        .iter(),
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
    let Ok(counting_lender) = sux::utils::lenders::FromIntoFallibleLenderFactory::new(
        || -> Result<_, std::convert::Infallible> {
            Ok(super::from_iter_ref::from_iter_ref(
                fallible_iterator::convert(
                    (0..config.num_terms).map(Ok::<_, std::convert::Infallible>),
                ),
            ))
        },
    );
    let vfunc = builder
        .try_build_func::<RawTerm, BoxedRawTerm>(terms_lender, counting_lender, &mut pl)
        .context("Could not build VFunc")?;
    pl.done();

    ensure!(
        vfunc.len() == config.num_terms,
        "vfunc.len()={}, expected {}",
        vfunc.len(),
        config.num_terms
    );

    let terms_mphf = TermMphf {
        vfunc: MemCase::<epserde::deser::Owned<VFunc<RawTerm, usize, BitFieldVec<usize>>>>::encase(
            vfunc,
        ),
        marker: PhantomData,
    };
    terms_mphf
        .serialize(&dest)
        .with_context(|| format!("Could not write terms MPHF to {}", dest.display()))
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
// impl<R: BufRead + Seek> FallibleRewindableLender<BoxedRawTerm> for ZstdLengthPrefixedStringLender<R> {
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

/// Equivalent to [`Lender::flatten`] but implements [`FallibleRewindableLender`]
struct FallibleRewindableFlattenLender<L> {
    lenders: Vec<L>,
    current_index: usize,
}

impl<L> FallibleRewindableFlattenLender<L> {
    pub fn new(lenders: Vec<L>) -> Self {
        Self {
            lenders,
            current_index: 0,
        }
    }
}

impl<'lend, L: FallibleLending<'lend>> FallibleLending<'lend>
    for FallibleRewindableFlattenLender<L>
{
    type Lend = L::Lend;
}

impl<L: FallibleLender> FallibleLender for FallibleRewindableFlattenLender<L> {
    type Error = L::Error;

    fn next(&mut self) -> Result<Option<<Self as FallibleLending<'_>>::Lend>, Self::Error> {
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
        for current_lender in &mut self.lenders[self.current_index..] {
            if let Some(item) = current_lender.next()? {
                return Ok(Some(item));
            }

            // exhausted the current lender, go to the next one
            self.current_index += 1;
        }

        Ok(None)
    }
}

impl<L: FallibleRewindableLender> FallibleRewindableLender for FallibleRewindableFlattenLender<L> {
    type RewindError = <L as FallibleRewindableLender>::RewindError;

    fn rewind(mut self) -> Result<Self, Self::RewindError> {
        let mut new_lenders = Vec::with_capacity(self.lenders.len());
        for lender in self
            .lenders
            .drain(0..=self.current_index.min(self.lenders.len() - 1))
        {
            new_lenders.push(lender.rewind()?);
        }
        new_lenders.append(&mut self.lenders);
        std::mem::swap(&mut new_lenders, &mut self.lenders);
        self.current_index = 0;
        Ok(self)
    }
}
