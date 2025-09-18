use crate::io::RdfParseError;
use crate::model::Quad;
use crate::storage::numeric_encoder::EncodedTerm;
use anyhow::Result;
use dashmap::DashSet;
use rayon::prelude::*;
use rustc_hash::FxBuildHasher;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct TermMphf {}

pub fn build_term_mphf(
    quads: impl ParallelIterator<Item = Result<Quad, RdfParseError>>,
) -> Result<TermMphf> {
    let terms = DashSet::<EncodedTerm, _>::with_hasher(FxBuildHasher::default());

    let num_quads = AtomicU64::new(0);

    quads.try_for_each(|quad| -> Result<_> {
        let Quad {
            subject,
            predicate,
            object,
            graph_name,
        } = quad?;
        for term in [
            subject.into(),
            predicate.into(),
            object.into(),
            graph_name.into(),
        ] {
            terms.insert(term);
        }

        let current_num_quads = num_quads.fetch_add(1, Ordering::Relaxed) + 1;
        if current_num_quads % 100_000_000 == 0 {
            eprintln!(
                "Loaded {}M quads, {}M unique terms",
                current_num_quads / 1_000_000,
                terms.len() / 1_000_000
            );
        }
        Ok(())
    })?;

    todo!("build_term_mphf");
}
