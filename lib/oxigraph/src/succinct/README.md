# Succinct backend

This backend is based on the [`sux` crate](https://docs.rs/sux) and its data structures.
While `sux` underpins [WebGraph's Rust implementation](https://docs.rs/webgraph),
it does not significantly use WebGraph as WebGraph is not designed to work with triples or quads.

`sux` provides two primitives we use to build indexes:

* [Elias-Fano sequences](https://docs.rs/sux/latest/sux/dict/elias_fano/), which are very small structures monotonically mapping a range of integers to a set of integers.
  Queries run in `O(range_size)` but are actually really fast in practice.
* [VFunc static function](https://docs.rs/sux/latest/sux/func/), which is a data structure that maps arbitrary keys to integers in `O(key_size)`.
  It achieves compactness by being probabilistic: it gives perfect results when queried with one of the keys it was built with, but undefined results for other keys.

# Basics

This backend currently does not support writes, and requires a long compression process to create its database
(`wikidata-20240320-truthy-BETA` takes 2h10min on a Ryzen 5650G with 32GB of RAM and a NVMe disk).

It may be updated in the future to support writes by writing to a WAL and then rebuilding the database.

To allow database construction to be parallel, terms and quads are sharded into partitions
(using a term id or the quad's first term's id as key).

# Terms

Terms are sorted alphabetically and zstd-compressed.
Zstd files are built so that each frame contains exactly the same number of terms (`terms_per_frame`, 16 by default).
Sorting them means that similar terms are in the same zstd frame, so they compress well.

TODO: use zstd dictionaires for even better compression

Each term is then associated to an id, which is its position in the zstd-compressed files.

We map each frame's id to its position in the zstd-compressed files using an Elias-Fano sequence.
This allows getting a term's frame given its id, by dividing the id by `terms_per_frame`.

We map each term to its position using a VFunc static function/MPH.

Using these two constructs, we can map term strings to their id and vice versa very quickly.

This stores `wikidata-20240320-truthy-BETA` terms in:

* 20GiB for the compressed terms
* 137MiB for the Elias-Fano indexes
* 7.5GiB for the VFunc

# Quads

We write two quad stores: one of subject-predicate-object-graph and one of object-predicate-subject-graph.
Other orders would work, but ids of the first term of each quad (subject and object, respectively) should be somewhat evenly distributed for good performance.

Each quad store is made of a list of quads and a bunch of indexes.


## List of quads

This list is written in a custom binary format using [`dsi-bitstream`](https://docs.rs/dsi-bitstream):

* Split quads into frames using whatever heuristic (by default: split every 100 quads if they have different first terms, or every 10000 quads if they have the first term)
* For each frame of quads:
    * bit 0 if this is the first frame of the file, bit 1 otherwise
    * the first quad of the frame, with each of its term written using [gamma-coding](https://docs.rs/dsi-bitstream/latest/dsi_bitstream/codes/)
    * For each other quad of the frame:
        * bit 0 (indicating this is not a new frame)
        * `let mut zigzag = false`
        * for the i-th term of the quad:
            * if `!zigzag`: the gamma-coded increase between the previous quad's i-th term and this quad's i-th term
                * (that increase is guaranteed to be >= due to quads being lexicographically sorted)
                * if the increase > 0, set `zigzag = true` (ie. the `i+1`th term of this quad is not guaranteed to be >= to the `i+1`th term of the previous quad )
            * if `zigzag`: the gamma-coded [zigzag-encoded](https://docs.rs/dsi-bitstream/latest/dsi_bitstream/codes/trait.ToInt.html) difference between the previous quad's i-th term and this quad's i-th term
* end with bit 1, then `0` gamma-encoded four times (ie. as if we had the quad `(0, 0, 0, 0)`)

This allows a compact, mmappable, seekable, and fast to decompress, representation of the quads.

This stores `wikidata-20240320-truthy-BETA` quads in

* 37GiB for spog quads
* 29GiB for opsg quads

## Indexes

There are two indexes:

* An Elias-Fano sequence mapping each term to the offset of the first frame that contains quads with that term as first term
* A VFunc static function mapping each pair of terms to the offset of the first frame that contains quads with these term as first two terms

As quads are sorted, this allows getting the beginning of the list of quads starting with the given two terms.
Getting the rest of the list is done by iterating from that frame (discarding the few quads lexicographically smaller than the ones we are looking for), and iterating until we find a quad lexicographically greater than the ones we are looking for.

This indexes `wikidata-20240320-truthy-BETA` quads in

* 620MiB for single-term indexes of spog quads
* 280MiB for two-terms indexes of spog quads
* 1.9GiB for single-term indexes of opsg quads
* 1.6GiB for two-terms indexes of opsg quads
