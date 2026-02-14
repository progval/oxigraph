use super::queryable_dataset::SuccinctDatasetView;
use crate::model::*;
use anyhow::{Context, Result, bail};
use spareval::{InternalQuad, QueryableDataset};
use std::path::Path;

/// Reimplementation of [`oxigraph::store::Store`] based on the Succinct backend instead of RocksDB
/// or Memory storage
#[derive(Clone)]
pub struct SuccinctStore<D = SuccinctDatasetView>
where
    for<'a> D: QueryableDataset<'a>,
{
    dataset: D,
}

impl SuccinctStore {
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Self> {
        Ok(SuccinctStore {
            dataset: SuccinctDatasetView::new(path.as_ref(), true)?,
        })
    }
}

impl<D: Clone> SuccinctStore<D>
where
    for<'a> D: QueryableDataset<'a>,
{
    pub fn queryable_dataset(self) -> D {
        self.dataset
    }

    pub fn quads_for_pattern(
        &self,
        subject: Option<NamedOrBlankNodeRef<'_>>,
        predicate: Option<NamedNodeRef<'_>>,
        object: Option<TermRef<'_>>,
        graph_name: Option<GraphNameRef<'_>>,
    ) -> Result<impl Iterator<Item = Result<Quad>> + use<D>> {
        let subject = subject
            .map(|s| self.dataset.internalize_term(s.into()))
            .transpose()
            .context("Could not internalize subject")?;
        let predicate = predicate
            .map(|p| self.dataset.internalize_term(p.into()))
            .transpose()
            .context("Could not internalize predicate")?;
        let object = object
            .map(|o| self.dataset.internalize_term(o.into()))
            .transpose()
            .context("Could not internalize object")?;
        let graph_name = graph_name
            .map(|g| {
                match g {
                    GraphNameRef::DefaultGraph => None,
                    GraphNameRef::NamedNode(g) => Some(self.dataset.internalize_term(g.into())),
                    GraphNameRef::BlankNode(g) => Some(self.dataset.internalize_term(g.into())),
                }
                .transpose()
            })
            .transpose()
            .context("Could not internalize graph_name")?;
        let dataset = self.dataset.clone();
        Ok(dataset
            .internal_quads_for_pattern(
                subject.as_ref(),
                predicate.as_ref(),
                object.as_ref(),
                graph_name.as_ref().map(|g| g.as_ref()),
            )
            .map(move |quad| {
                let InternalQuad {
                    subject,
                    predicate,
                    object,
                    graph_name,
                } = quad?;
                Ok(Quad {
                    subject: dataset
                        .externalize_term(subject)?
                        .try_into()
                        .context("Unexpected subject type")?,
                    predicate: dataset
                        .externalize_term(predicate)?
                        .try_into()
                        .context("Unexpected predicate type")?,
                    object: dataset.externalize_term(object)?,
                    graph_name: match graph_name {
                        Some(graph_name) => match dataset.externalize_term(graph_name)? {
                            Term::BlankNode(g) => g.into(),
                            Term::NamedNode(g) => g.into(),
                            g => bail!("Unexpected graph_name type: {g:?}"),
                        },
                        None => GraphName::DefaultGraph,
                    },
                })
            }))
    }

    pub fn contains_named_graph(&self, _graph_name: &NamedNode) -> Result<bool> {
        todo!("contains_graph_name");
    }
}
