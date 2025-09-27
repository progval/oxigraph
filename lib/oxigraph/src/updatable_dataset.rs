use crate::model::{GraphNameRef, NamedOrBlankNodeRef, QuadRef};
use crate::storage::numeric_encoder::{EncodedQuad, EncodedTerm, StrHash};
use oxrdf::Quad;
use std::error::Error;
use std::path::Path;

pub trait UpdatableDataset<'a> {
    type Error: Error;

    type Reader<'reader>: Reader<'reader, Error = Self::Error>
    where
        Self: 'reader;
    type WriteOnlyTransaction<'transaction>: WriteOnlyTransaction<'transaction, Error = Self::Error>
    where
        Self: 'transaction;
    type ReadWriteTransaction<'transaction>: ReadWriteTransaction<'transaction, Error = Self::Error>
    where
        Self: 'transaction;
    type BulkLoader<'loader>: BulkLoader<'loader, Error = Self::Error>
    where
        Self: 'loader;

    fn snapshot(&self) -> Self::Reader<'static>;
    fn start_transaction(&self) -> Result<Self::WriteOnlyTransaction<'_>, Self::Error>;
    fn start_readable_transaction(&self) -> Result<Self::ReadWriteTransaction<'_>, Self::Error>;
    #[cfg_attr(
        any(target_family = "wasm", not(feature = "rocksdb")),
        expect(dead_code)
    )]
    fn flush(&self) -> Result<(), Self::Error>;
    #[cfg_attr(
        any(target_family = "wasm", not(feature = "rocksdb")),
        expect(dead_code)
    )]
    fn compact(&self) -> Result<(), Self::Error>;
    #[cfg_attr(
        any(target_family = "wasm", not(feature = "rocksdb")),
        expect(dead_code)
    )]
    fn backup(&self, target_directory: &Path) -> Result<(), Self::Error>;
    fn bulk_loader(&self) -> Result<Self::BulkLoader<'_>, Self::Error>;
}

pub trait Reader<'a> {
    type Error: Error;
    type TermIterator<'iter>: Iterator<Item = Result<EncodedTerm, Self::Error>> + 'iter
    where
        Self: 'iter;
    type QuadIterator<'iter>: Iterator<Item = Result<EncodedQuad, Self::Error>> + 'iter
    where
        Self: 'iter;

    fn len(&self) -> Result<usize, Self::Error>;
    fn is_empty(&self) -> Result<bool, Self::Error>;
    fn contains(&self, quad: &EncodedQuad) -> Result<bool, Self::Error>;
    fn quads_for_pattern(
        &self,
        subject: Option<&EncodedTerm>,
        predicate: Option<&EncodedTerm>,
        object: Option<&EncodedTerm>,
        graph_name: Option<&EncodedTerm>,
    ) -> Self::QuadIterator<'a>;
    fn named_graphs(&self) -> Self::TermIterator<'a>;
    fn contains_named_graph(&self, graph_name: &EncodedTerm) -> Result<bool, Self::Error>;
    fn contains_str(&self, key: &StrHash) -> Result<bool, Self::Error>;
    /// Validate that all the storage invariants held in the data
    fn validate(&self) -> Result<(), Self::Error>;
}

pub trait WriteOnlyTransaction<'a> {
    type Error: Error;

    /// Adds a quad to this store.
    ///
    /// Returns `true` if the quad was not already in the store.
    ///
    /// Usage example:
    /// ```
    /// use oxigraph::model::*;
    /// use oxigraph::store::Store;
    ///
    /// let ex = NamedNodeRef::new_unchecked("http://example.com");
    /// let quad = QuadRef::new(ex, ex, ex, GraphNameRef::DefaultGraph);
    ///
    /// let store = Store::new()?;
    /// let mut transaction = store.start_transaction()?;
    /// transaction.insert(quad);
    /// transaction.commit()?;
    /// assert!(store.contains(quad)?);
    /// # Result::<_,oxigraph::store::StorageError>::Ok(())
    /// ```
    fn insert(&mut self, quad: QuadRef<'_>);

    /// Inserts a graph into this store.
    ///
    /// Returns `true` if the graph was not already in the store.
    ///
    /// Usage example:
    /// ```
    /// use oxigraph::model::NamedNodeRef;
    /// use oxigraph::store::Store;
    ///
    /// let ex = NamedNodeRef::new_unchecked("http://example.com");
    /// let store = Store::new()?;
    /// let mut transaction = store.start_transaction()?;
    /// transaction.insert_named_graph(ex);
    /// transaction.commit()?;
    /// assert_eq!(
    ///     store.named_graphs().collect::<Result<Vec<_>, _>>()?,
    ///     vec![ex.into_owned().into()]
    /// );
    /// # Result::<_,oxigraph::store::StorageError>::Ok(())
    /// ```
    fn insert_named_graph(&mut self, graph_name: NamedOrBlankNodeRef<'_>);

    /// Removes a quad from this store.
    ///
    /// Returns `true` if the quad was in the store and has been removed.
    ///
    /// Usage example:
    /// ```
    /// use oxigraph::model::*;
    /// use oxigraph::store::Store;
    ///
    /// let ex = NamedNodeRef::new_unchecked("http://example.com");
    /// let quad = QuadRef::new(ex, ex, ex, GraphNameRef::DefaultGraph);
    /// let store = Store::new()?;
    /// let mut transaction = store.start_transaction()?;
    /// transaction.insert(quad);
    /// transaction.remove(quad);
    /// transaction.commit()?;
    /// assert!(!store.contains(quad)?);
    /// # Result::<_,oxigraph::store::StorageError>::Ok(())
    /// ```
    fn remove(&mut self, quad: QuadRef<'_>);

    fn clear_default_graph(&mut self) -> Result<(), Self::Error> {
        self.clear_graph(GraphNameRef::DefaultGraph)
    }

    /// Clears a graph from this store.
    ///
    /// Usage example:
    /// ```
    /// use oxigraph::model::{NamedNodeRef, QuadRef};
    /// use oxigraph::store::Store;
    ///
    /// let ex = NamedNodeRef::new_unchecked("http://example.com");
    /// let quad = QuadRef::new(ex, ex, ex, ex);
    /// let store = Store::new()?;
    /// let mut transaction = store.start_transaction()?;
    /// transaction.insert(quad);
    /// transaction.clear_graph(ex)?;
    /// transaction.commit()?;
    /// assert!(store.is_empty()?);
    /// assert_eq!(1, store.named_graphs().count());
    /// # Result::<_,oxigraph::store::StorageError>::Ok(())
    /// ```
    fn clear_graph(&mut self, graph_name: GraphNameRef<'_>) -> Result<(), Self::Error>;

    fn clear_all_graphs(&mut self) -> Result<(), Self::Error>;

    fn clear_all_named_graphs(&mut self) -> Result<(), Self::Error>;

    fn remove_all_named_graphs(&mut self) -> Result<(), Self::Error>;

    /// Clears the store.
    ///
    /// Usage example:
    /// ```
    /// use oxigraph::model::*;
    /// use oxigraph::store::Store;
    ///
    /// let ex = NamedNodeRef::new_unchecked("http://example.com");
    /// let store = Store::new()?;
    /// let mut transaction = store.start_transaction()?;
    /// transaction.insert(QuadRef::new(ex, ex, ex, ex));
    /// transaction.clear()?;
    /// transaction.commit()?;
    /// assert!(store.is_empty()?);
    /// # Result::<_,oxigraph::store::StorageError>::Ok(())
    /// ```
    fn clear(&mut self) -> Result<(), Self::Error>;

    /// Commits the transaction, i.e., apply its modifications to the underlying store.
    ///
    /// Usage example:
    /// ```
    /// use oxigraph::model::*;
    /// use oxigraph::store::Store;
    ///
    /// let ex = NamedNodeRef::new_unchecked("http://example.com");
    /// let store = Store::new()?;
    /// let mut transaction = store.start_transaction()?;
    /// transaction.insert(QuadRef::new(ex, ex, ex, ex));
    /// transaction.commit()?;
    /// assert!(store.contains(QuadRef::new(ex, ex, ex, ex))?);
    /// # Result::<_,oxigraph::store::StorageError>::Ok(())
    /// ```
    fn commit(self) -> Result<(), Self::Error>;
}

pub trait ReadWriteTransaction<'a>: WriteOnlyTransaction<'a> {
    type Reader<'reader>: Reader<'reader, Error = Self::Error>
    where
        Self: 'reader;

    fn reader(&self) -> Self::Reader<'_>;
    fn remove_named_graph(
        &mut self,
        graph_name: NamedOrBlankNodeRef<'_>,
    ) -> Result<(), Self::Error>;
}

pub trait BulkLoader<'a> {
    type Error: Error;

    /// Adds a `callback` evaluated from time to time with the number of loaded triples.
    fn on_progress(self, callback: impl Fn(u64) + Send + Sync + 'static) -> Self;

    /// Allow the bulk loader to save also data to the database during the bulk loading instead of only when [`commit`](Self::commit) is called.
    ///
    /// When used with the RocksDB storage, it allows the storage to compact the data while the loading continues.
    fn without_atomicity(self) -> Self;

    /// Adds a set of quads using the bulk loader.
    ///
    /// See [the struct](Self) documentation for more details.
    fn load_batch(&mut self, quads: Vec<Quad>, max_num_threads: usize) -> Result<(), Self::Error>;

    /// Saves all the quads loaded using the bulk loader into the dataset.
    fn commit(self) -> Result<(), Self::Error>;
}
