use super::quads_store::{QuadOrder, QuadStore};
use super::terms_mphf::{DefaultDeserializedTermMphf, TermHasher, TermMphf};
use super::terms_store::TermStore;
use anyhow::{Context, Result, anyhow, ensure};
use oxrdf::{GraphName, Term};
use spareval::{InternalQuad, QueryableDataset};
use std::path::Path;
use std::sync::{Arc, RwLock};

#[derive(Debug, thiserror::Error)]
#[error("{0:#}")]
pub struct SuccinctDatasetError(#[from] anyhow::Error);

#[derive(Clone)]
pub struct SuccinctDatasetView(pub Arc<SuccinctDatasetViewInner>);

pub struct SuccinctDatasetViewInner {
    terms_mphf: DefaultDeserializedTermMphf,
    terms: TermStore,
    spog_quads: QuadStore,
    opsg_quads: QuadStore,

    // if the default graph exists in the dataset, this is its id.
    default_graph_name: Option<usize>,

    extras: RwLock<Vec<Term>>,
    _dataset: InternalizedDatasetSpec,
}

impl SuccinctDatasetView {
    pub fn new(path: &Path, mmap_mphf: bool) -> Result<Self> {
        let spog_quads_path = path.join(format!("quads-spog"));
        let spog_quads =
            QuadStore::mmap(spog_quads_path.clone()).context("Could not mmap spog quads")?;

        let opsg_quads_path = path.join(format!("quads-opsg"));
        let opsg_quads =
            QuadStore::mmap(opsg_quads_path.clone()).context("Could not mmap opsg quads")?;

        let terms_path = path.join(format!("terms"));
        let terms = TermStore::mmap(terms_path.clone()).context("Could not mmap terms store")?;

        let mphf_path = path.join("terms_mphf");
        let terms_mphf = if mmap_mphf {
            TermMphf::mmap(&mphf_path).with_context(|| {
                format!("Could not mmap terms MPHF from {}", mphf_path.display())
            })?
        } else {
            TermMphf::load(&mphf_path).with_context(|| {
                format!("Could not load terms MPHF from {}", mphf_path.display())
            })?
        };
        let view = Self(Arc::new(SuccinctDatasetViewInner {
            terms_mphf,
            terms,
            spog_quads,
            opsg_quads,
            default_graph_name: None,
            extras: RwLock::default(),
            _dataset: InternalizedDatasetSpec {
                _default: None,
                _named: None,
            },
        }));

        let default_graph_name = view.internalize_graph_name(&GraphName::DefaultGraph)?;

        Ok(Self(Arc::new(SuccinctDatasetViewInner {
            default_graph_name,
            ..Arc::into_inner(view.0).context("Arc leaked before SuccinctDatasetView creation")?
        })))
    }
}

fn map_iterator<
    'a,
    T,
    U: 'a,
    IterT: Iterator<Item = Result<T>> + 'a,
    IterU: Iterator<Item = Result<U>> + 'a,
>(
    iter: Result<Option<IterT>>,
    mut f: impl FnMut(IterT) -> IterU + 'a,
) -> Box<dyn Iterator<Item = Result<U, SuccinctDatasetError>> + 'a> {
    match iter {
        Ok(Some(iter)) => Box::new(f(iter).map(|item| item.map_err(SuccinctDatasetError))),
        Ok(None) => Box::new(std::iter::empty()),
        Err(e) => Box::new(std::iter::once(Err(SuccinctDatasetError(e)))),
    }
}

impl<'a> QueryableDataset<'a> for SuccinctDatasetView {
    type InternalTerm = usize;

    type Error = SuccinctDatasetError;

    /// Fetches quads according to a pattern
    ///
    /// For `graph_name`, `Some(None)` encodes the default graph and `Some(Some(_))` a named graph
    fn internal_quads_for_pattern(
        &self,
        subject: Option<&Self::InternalTerm>,
        predicate: Option<&Self::InternalTerm>,
        object: Option<&Self::InternalTerm>,
        graph_name: Option<Option<&Self::InternalTerm>>,
    ) -> impl Iterator<Item = Result<InternalQuad<Self::InternalTerm>, Self::Error>> + use<'a> {
        let graph_name: Option<usize> = graph_name.map(|gn| match gn {
            None => {
                let Ok(gn) = self.0.terms_mphf.hash_graphname(&GraphName::DefaultGraph) else {
                    todo!("Support for datasets where no term is in the default graph");
                };
                if gn != 0 {
                    // hash collision
                    todo!("Support for datasets where no term is in the default graph");
                }
                gn
            }
            Some(&gn) => gn,
        });
        let subject: Option<usize> = subject.copied();
        let predicate: Option<usize> = predicate.copied();
        let object: Option<usize> = object.copied();
        let notself = self.clone();
        match (subject, predicate, object, graph_name) {
            (Some(subject), Some(predicate), _, _) => map_iterator::<'a, _, _, _, _>(
                self.0
                    .spog_quads
                    .iter_quads_by_first_two_terms(subject, predicate),
                move |iter| {
                    let notself = notself.clone();
                    iter.filter_map(move |quad| -> Option<Result<_>> {
                        ({
                            || -> Result<Option<_>> {
                                let quad = QuadOrder::Spog.mapper()(quad?);
                                ensure!(quad[0] == subject, "subject did not match");
                                ensure!(quad[1] == predicate, "predicate did not match");
                                if let Some(object) = object {
                                    if quad[2] != object {
                                        return Ok(None);
                                    }
                                }
                                if let Some(graph_name) = graph_name {
                                    if quad[3] != graph_name {
                                        return Ok(None);
                                    }
                                }
                                Ok(Some(notself.to_internal_quad(quad)))
                            }
                        })()
                        .transpose()
                    })
                },
            ),
            (_, Some(predicate), Some(object), _) => map_iterator(
                self.0
                    .opsg_quads
                    .iter_quads_by_first_two_terms(object, predicate),
                move |iter| {
                    let notself = notself.clone();
                    iter.filter_map(move |quad| -> Option<Result<_>> {
                        ({
                            || -> Result<Option<_>> {
                                let quad = QuadOrder::Opsg.mapper()(quad?);
                                if let Some(subject) = subject {
                                    if quad[0] != subject {
                                        return Ok(None);
                                    }
                                }
                                ensure!(quad[1] == predicate, "predicate did not match");
                                ensure!(quad[2] == object, "subject did not match");
                                if let Some(graph_name) = graph_name {
                                    if quad[3] != graph_name {
                                        return Ok(None);
                                    }
                                }
                                Ok(Some(inner.to_internal_quad(quad)))
                            }
                        })()
                        .transpose()
                    })
                },
            ),
            _ => todo!("internal_quads_for_pattern pattern"),
        }
    }

    /// Builds an internal term from the [`Term`] struct
    fn internalize_term(&self, term: Term) -> Result<Self::InternalTerm, Self::Error> {
        if let Ok(id) = self.0.terms_mphf.hash_term(&term) {
            if let Some(expected_term_str) = self.0.terms.get(id)? {
                let term_str_matches = match &term {
                    // TODO: dedup with TermHasher
                    Term::NamedNode(n) => expected_term_str == n.as_str(),
                    Term::BlankNode(n) => expected_term_str == n.as_str(),
                    Term::Literal(l) => expected_term_str == l.to_string(),
                    #[cfg(feature = "rdf-12")]
                    Term::Triple(_) => todo!("Term::Triple"),
                };

                if term_str_matches {
                    // not a hash collision
                    return Ok(id);
                }
            }
        }
        let mut extras = self.0.extras.write().expect("Poisoned RwLock");
        match extras.iter().position(|item| *item == term) {
            Some(id) => Ok(id),
            None => {
                extras.push(term);
                Ok(self.0.terms.len() + extras.len() - 1)
            }
        }
    }

    /// Builds a [`Term`] from an internal term
    fn externalize_term(&self, term: Self::InternalTerm) -> Result<Term, Self::Error> {
        if let Some(s) = self.0.terms.get(term)? {
            Ok(s.parse()
                .with_context(|| format!("Could not parse stored term {s:?}"))?)
        } else {
            Err(anyhow!("Unknown term: {term}"))?
        }
    }
}

impl SuccinctDatasetView {
    fn to_internal_quad(&self, quad: [usize; 4]) -> InternalQuad<usize> {
        let [subject, predicate, object, graph_name] = quad;
        InternalQuad {
            subject,
            predicate,
            object,
            graph_name: if Some(graph_name) == self.0.default_graph_name {
                None
            } else {
                Some(graph_name)
            },
        }
    }

    fn internalize_graph_name(&self, graph_name: &GraphName) -> Result<Option<usize>> {
        if let Ok(id) = self.0.terms_mphf.hash_graphname(graph_name) {
            if let Some(expected_graph_name_bytes) = self.0.terms.get(id)? {
                let graph_name_str_matches = match graph_name {
                    // TODO: dedup with TermHasher
                    GraphName::NamedNode(n) => expected_graph_name_str == n.as_str(),
                    GraphName::BlankNode(n) => expected_graph_name_str == n.as_str(),
                    GraphName::DefaultGraph => expected_graph_name_str == "".to_owned(),
                };
                if graph_name_str_matches {
                    // not a hash collision
                    return Ok(Some(id));
                }
            }
        }
        Ok(None)
    }
}

pub struct InternalizedDatasetSpec {
    _default: Option<Vec<usize>>,
    _named: Option<Vec<usize>>,
}
