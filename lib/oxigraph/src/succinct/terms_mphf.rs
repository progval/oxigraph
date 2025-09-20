use anyhow::{Context, Result, anyhow};
use std::path::Path;
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
use std::io::{BufRead, Cursor, Seek};
use std::marker::PhantomData;
use sux::bits::bit_field_vec::BitFieldVec;
use sux::func::{VBuilder, VFunc};
use sux::traits::bit_field_slice::BitFieldSlice;
use sux::utils::{FromIntoIterator, RewindableIoLender};
use webgraph::prelude::MmapHelper;
use zstd::stream::read::Decoder;
use super::terms_store::{list_terms_files, TermsFile, read_length_prefixed_string};

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

pub struct TermMphf<D: BitFieldSlice<usize> = MmapHelper<usize>> {
    vfunc: MemCase<VFunc<RawTerm, usize, D>>,
    marker: PhantomData<D>,
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

impl<D: BitFieldSlice<usize> + EpDeserializeInner> TermMphf<D>
where
    VFunc<RawTerm, usize, D>: EpDeserialize,
{
    pub fn deserialize(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<TermMphf<<D as EpDeserializeInner>::DeserType<'static>>>
    where
        <D as EpDeserializeInner>::DeserType<'static>: AsRef<[usize]>,
        for<'a> VFunc<RawTerm, usize, D>: EpDeserializeInner<
            DeserType<'a> = VFunc<RawTerm, usize, <D as EpDeserializeInner>::DeserType<'a>>,
        >,
    {
        let path = path.as_ref();
        let vfunc_path = path.join("mphf.vfunc");

        let flags = epserde::deser::mem_case::Flags::RANDOM_ACCESS;
        // SAFETY: this is unsafe because we can't guarantee the file won't be modified while we
        // access it, but there is nothing we can do about this.
        let vfunc = unsafe { <VFunc<RawTerm, usize, D>>::mmap(&vfunc_path, flags) }
            .with_context(|| format!("Could write VFunc to {}", vfunc_path.display()))?;
        Ok(TermMphf {
            vfunc,
            marker: PhantomData,
        })
    }
}



pub fn build_terms_mph(dir: &Path) -> Result<TermMphf<BitFieldVec<usize>>> {
    let (config, terms_files) = list_terms_files(dir)?;

    let terms_lender = RewindableIoFlattenLender::new(
        terms_files
            .iter()
            .map(|terms_file| {
                let TermsFile {
                    first_term_id: _,
                    num_terms,
                    path,
                    compressed_frames,
                } = terms_file;

                ZstdLengthPrefixedStringLender::new(Cursor::new(compressed_frames))
                    .map_err(DecodeError)
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

    Ok(TermMphf {
        vfunc,
        marker: PhantomData,
    })
}

/// Reads a zstd-compressed file using [`read_length_prefixed_string`] on each item of each frame
struct ZstdLengthPrefixedStringLender<R: BufRead> {
    decoder: Decoder<'static, R>,
    string: Option<BoxedRawTerm>,
}

impl<R: BufRead> ZstdLengthPrefixedStringLender<R> {
    pub fn new(read: R) -> Result<Self> {
        Ok(ZstdLengthPrefixedStringLender {
            decoder: Decoder::with_buffer(read)?,
            string: None,
        })
    }
}

impl<'lend, R: BufRead> Lending<'lend> for ZstdLengthPrefixedStringLender<R> {
    type Lend = Result<&'lend BoxedRawTerm, DecodeError>;
}

impl<R: BufRead> Lender for ZstdLengthPrefixedStringLender<R> {
    fn next(&mut self) -> Option<<Self as Lending<'_>>::Lend> {
        match read_length_prefixed_string(&mut self.decoder, |_| None) {
            Ok(Some(string)) => {
                if let Some(previous_string) = &self.string {
                    if string <= previous_string.0 {
                        return Some(Err(DecodeError(anyhow!(
                            "Unsorted strings: {} ({:?}) after {} ({:?})",
                            String::from_utf8_lossy(&string),
                            &string,
                            String::from_utf8_lossy(&previous_string.0),
                            &previous_string.0,
                        ))));
                    }
                }
                self.string = Some(BoxedRawTerm(string));
                Some(Ok(self.string.as_ref().unwrap()))
            }
            Ok(None) => None,
            Err(e) => Some(Err(e.into())),
        }
    }
}

impl<R: BufRead + Seek> RewindableIoLender<BoxedRawTerm> for ZstdLengthPrefixedStringLender<R> {
    type Error = DecodeError;

    fn rewind(mut self) -> Result<Self, Self::Error> {
        let mut read = self.decoder.finish();
        read.rewind().context("Could not rewind")?;
        self.decoder =
            Decoder::with_buffer(read).context("Could not create new decoder to rewind")?;
        Ok(self)
    }
}

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
        for lender in self.lenders.drain(0..=self.current_index) {
            new_lenders.push(lender.rewind()?);
        }
        new_lenders.extend(self.lenders.drain(..));
        std::mem::swap(&mut new_lenders, &mut self.lenders);
        self.current_index = 0;
        Ok(self)
    }
}


