use super::quads_store::{QuadOrder, QuadStore};
use super::sort::{BBReader, BBWriter, Compressor, DeltaCompressor};
use anyhow::{Context, Result};
use dsi_bitstream::prelude::*;
use dsi_progress_logger::{ProgressLog, concurrent_progress_logger};
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

fn inc(counter: &AtomicUsize, value: usize) {
    counter.fetch_add(value, Ordering::Relaxed);
}

#[expect(clippy::struct_field_names)]
#[derive(Debug)]
struct AnalyzingCompressorCounters<const N: usize> {
    true_newframe_bits: AtomicUsize,
    false_newframe_bits: AtomicUsize,
    i64_bits: [AtomicUsize; N],
    u64_bits: [AtomicUsize; N],
}

impl<const N: usize> Default for AnalyzingCompressorCounters<N> {
    fn default() -> Self {
        AnalyzingCompressorCounters {
            true_newframe_bits: AtomicUsize::default(),
            false_newframe_bits: AtomicUsize::default(),
            i64_bits: (0..N)
                .map(|_| AtomicUsize::default())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            u64_bits: (0..N)
                .map(|_| AtomicUsize::default())
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        }
    }
}

#[derive(Clone, Debug)]
struct AnalyzingCompressor<const N: usize, C: Compressor> {
    write_counters: Arc<AnalyzingCompressorCounters<N>>,
    read_counters: Arc<[(Codes, AnalyzingCompressorCounters<N>)]>,
    underlying_compressor: C,
}

impl<const N: usize, C: Compressor> Compressor for AnalyzingCompressor<N, C> {
    fn write_newframe_bit(&mut self, writer: &mut BBWriter, is_new_frame: bool) -> Result<usize> {
        if is_new_frame {
            inc(&self.write_counters.true_newframe_bits, 1)
        } else {
            inc(&self.write_counters.false_newframe_bits, 1)
        }
        self.underlying_compressor
            .write_newframe_bit(writer, is_new_frame)
    }
    fn write_i64(&mut self, writer: &mut BBWriter, column: usize, value: i64) -> Result<usize> {
        let num_bits = self
            .underlying_compressor
            .write_i64(writer, column, value)?;
        inc(&self.write_counters.i64_bits[column], num_bits);
        Ok(num_bits)
    }
    fn write_u64(&mut self, writer: &mut BBWriter, column: usize, value: u64) -> Result<usize> {
        let num_bits = self
            .underlying_compressor
            .write_u64(writer, column, value)?;
        inc(&self.write_counters.u64_bits[column], num_bits);
        Ok(num_bits)
    }

    fn read_newframe_bit(&mut self, reader: &mut BBReader<impl AsRef<[u64]>>) -> Result<bool> {
        let is_new_frame = self.underlying_compressor.read_newframe_bit(reader)?;
        for (_, counters) in self.read_counters.iter() {
            if is_new_frame {
                inc(&counters.true_newframe_bits, 1)
            } else {
                inc(&counters.false_newframe_bits, 1)
            }
        }
        Ok(is_new_frame)
    }
    fn read_i64(&mut self, reader: &mut BBReader<impl AsRef<[u64]>>, column: usize) -> Result<i64> {
        let value = self.underlying_compressor.read_i64(reader, column)?;
        let zigzag = value.to_nat();

        // dummy writer, we just need something that implements the write methods
        // let mut buf = [0_usize; 2];
        // let mut writer = BufBitWriter::<LE, _>::new(MemWordWriterSlice::new(&mut buf));
        let mut writer =
            BufBitWriter::<LE, _>::new(MemWordWriterVec::new(Vec::<usize>::with_capacity(8)));

        for (code, counters) in self.read_counters.iter() {
            let num_bits = if *code == Codes::Unary {
                // too slow to actually do it
                (zigzag + 1).try_into().context("zigzag overflowed u64")?
            } else {
                code.write(&mut writer, zigzag)
                    .context("Could not write to dummy writer")?
            };
            inc(&counters.i64_bits[column], num_bits);
        }

        Ok(value)
    }
    fn read_u64(&mut self, reader: &mut BBReader<impl AsRef<[u64]>>, column: usize) -> Result<u64> {
        let value = self.underlying_compressor.read_u64(reader, column)?;

        // dummy writer, we just need something that implements the write methods
        // let mut buf = [0_usize; 2];
        // let mut writer = BufBitWriter::<LE, _>::new(MemWordWriterSlice::new(&mut buf));
        let mut writer =
            BufBitWriter::<LE, _>::new(MemWordWriterVec::new(Vec::<usize>::with_capacity(8)));

        for (code, counters) in self.read_counters.iter() {
            let num_bits = if *code == Codes::Unary {
                // too slow to actually do it
                (value + 1).try_into().context("value overflowed u64")?
            } else {
                code.write(&mut writer, value)
                    .context("Could not write to dummy writer")?
            };
            inc(&counters.u64_bits[column], num_bits);
        }

        Ok(value)
    }
}

#[expect(clippy::print_stdout, clippy::use_debug)]
pub fn analyze_quad_store_codes(path: &Path) -> Result<()> {
    let num_quads = QuadStore::mmap(path.to_path_buf())
        .context("Could not open quad store")?
        .num_quads();

    let codes = vec![
        Codes::Unary,
        Codes::Gamma,
        Codes::Delta,
        Codes::Omega,
        Codes::VByteBe,
        Codes::VByteLe,
        Codes::Zeta { k: 2 },
        Codes::Zeta { k: 3 },
        Codes::Zeta { k: 4 },
        Codes::Zeta { k: 5 },
        Codes::Zeta { k: 6 },
        Codes::Zeta { k: 7 },
        Codes::Pi { k: 1 },
        Codes::Pi { k: 2 },
        Codes::Pi { k: 3 },
        Codes::Pi { k: 4 },
    ];

    let compressor = AnalyzingCompressor {
        write_counters: Arc::new(AnalyzingCompressorCounters::<4>::default()),
        read_counters: codes
            .into_iter()
            .map(|code| (code, AnalyzingCompressorCounters::default()))
            .collect(),
        underlying_compressor: DeltaCompressor,
    };

    let order = QuadOrder::Spog; // doesn't matter

    let mut pl = concurrent_progress_logger!(
        item_name = "quad",
        display_memory = true,
        local_speed = true,
        expected_updates = Some(num_quads),
    );
    pl.start("Analyzing quad codes...");
    super::quads_store::par_iter_quads_with_compressor(path, order, &compressor)?
        .for_each_with(pl.clone(), |pl, _quad| pl.light_update());
    pl.done();

    println!(
        "Analysis on all {num_quads} quads:\n{:#?}",
        compressor.read_counters
    );

    Ok(())
}
