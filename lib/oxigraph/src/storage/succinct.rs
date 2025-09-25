pub use crate::storage::error::StorageError;
use crate::storage::numeric_encoder::{EncodedQuad, EncodedTerm, StrHash, StrLookup};
use std::path::Path;

#[derive(Clone)]
pub struct SuccinctStorage {}
impl SuccinctStorage {
    pub fn open(_path: &Path) -> Result<Self, StorageError> {
        todo!("SuccinctStorage::open")
    }

    pub fn snapshot(&self) -> SuccinctStorageReader<'static> {
        todo!("SuccinctStorage::snapshot")
    }
}

pub struct SuccinctStorageReader<'a> {
    _storage: &'a SuccinctStorage,
}

impl<'a> SuccinctStorageReader<'a> {
    pub fn len(&self) -> Result<usize, StorageError> {
        todo!("SuccinctStorageReader::len")
    }

    pub fn is_empty(&self) -> Result<bool, StorageError> {
        todo!("SuccinctStorageReader::is_empty")
    }

    pub fn contains(&self, _quad: &EncodedQuad) -> Result<bool, StorageError> {
        todo!("SuccinctStorageReader::contains")
    }

    pub fn quads_for_pattern(
        &self,
        _subject: Option<&EncodedTerm>,
        _predicate: Option<&EncodedTerm>,
        _object: Option<&EncodedTerm>,
        _graph_name: Option<&EncodedTerm>,
    ) -> SuccinctQuadIterator<'a> {
        todo!("SuccinctStorageReader::quads_for_pattern")
    }

    pub fn named_graphs(&self) -> SuccinctDecodingGraphIterator<'a> {
        todo!("SuccinctStorageReader::named_graphs")
    }
    pub fn contains_named_graph(&self, _graph_name: &EncodedTerm) -> Result<bool, StorageError> {
        todo!("SuccinctStorageReader::contains_named_graph")
    }

    pub fn contains_str(&self, _key: &StrHash) -> Result<bool, StorageError> {
        todo!("SuccinctStorageReader::contains_str")
    }

    pub fn validate(&self) -> Result<(), StorageError> {
        todo!("SuccinctStorageReader::validate")
    }
}

impl StrLookup for SuccinctStorageReader<'_> {
    fn get_str(&self, _key: &StrHash) -> Result<Option<String>, StorageError> {
        todo!("SuccinctStorageReader::get_str")
    }
}

pub struct SuccinctQuadIterator<'a> {
    _storage: &'a SuccinctStorage,
}

impl<'a> SuccinctQuadIterator<'a> {}

impl Iterator for SuccinctQuadIterator<'_> {
    type Item = Result<EncodedQuad, StorageError>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!("SuccinctQuadIterator::next")
    }
}

pub struct SuccinctDecodingGraphIterator<'a> {
    _storage: &'a SuccinctStorage,
}

impl Iterator for SuccinctDecodingGraphIterator<'_> {
    type Item = Result<EncodedTerm, StorageError>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!("SuccinctDecodingGraphIterator::next")
    }
}
