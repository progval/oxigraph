# Succinct backend

This backend is based on the [`sux` crate](https://docs.rs/sux) and its data structures.
While `sux` underpins [WebGraph's Rust implementation](https://docs.rs/webgraph),
this backend does not significantly use WebGraph as WebGraph is not designed to work with triples or quads.

`sux` provides two primitives we use to build indexes:

* [Elias-Fano sequences](https://docs.rs/sux/latest/sux/dict/elias_fano/), which are very small structures monotonically mapping a range of integers to a set of integers.
  Queries run in `O(range_size)` but are actually really fast in practice (~100ns/query when in RAM).
* [VFunc static function](https://docs.rs/sux/latest/sux/func/), which is a data structure that maps arbitrary keys to integers in `O(key_size)`.
  It achieves compactness by being probabilistic: it gives perfect results when queried with one of the keys it was built with, but undefined results for other keys.

# Basics

This backend currently does not support writes, and requires a long compression process to create its database
(`wikidata-20240320-truthy-BETA` takes 2h10min on a Ryzen 5650G with 32GB of RAM and a NVMe disk).

It may be updated in the future to support writes by writing to a WAL and then rebuilding the database.

To allow database construction to be parallel, terms and quads are sharded into partitions
(using a term id or the quad's first term's id as key).

# Terms

**Terms are sorted alphabetically** and zstd-compressed.
Zstd files are built so that each frame contains exactly the same number of terms (`terms_per_frame`, 16 by default).
Sorting them means that similar terms are in the same zstd frame, so they compress well.

Each term is then associated to an id, which is its position in the zstd-compressed files.

We map each frame's id to its position in the zstd-compressed files using an Elias-Fano sequence.
This allows getting a term's frame given its id, by dividing the id by `terms_per_frame`.

We map each term to its position using a VFunc static function/MPH.

Using these two constructs, we can map term strings to their id and vice versa very quickly.

This stores `wikidata-20240320-truthy-BETA` terms in:

* 1MiB for the zstd dictionary
* 17GiB for the compressed terms (or 20GiB without a zstd dictionary)
* 137MiB for the Elias-Fano indexes
* 7.5GiB for the VFunc

# Quads

We write two quad stores: one of subject-predicate-object-graph and one of object-predicate-subject-graph.
Other orders would work, but ids of the first term of each quad (subject and object, respectively) should be somewhat evenly distributed for good performance.

Each quad store is made of a **sorted list of quads** and a bunch of indexes.


## List of quads

This list is sorted and written in a custom binary format using [`dsi-bitstream`](https://docs.rs/dsi-bitstream):

* Split quads into frames using whatever heuristic (by default: split every 100 quads if they have different first terms, or every 10000 quads if they have the same first term)
* For each frame of quads:
    * bit 0 if this is the first frame of the file, bit 1 otherwise
    * the first quad of the frame, with each of its term written using [gamma-coding](https://docs.rs/dsi-bitstream/latest/dsi_bitstream/codes/)
    * For each other quad of the frame:
        * bit 0 (indicating this is not a new frame)
        * `let mut zigzag = false`
        * for the i-th term of the quad:
            * if `!zigzag`: the gamma-coded increase between the previous quad's i-th term and this quad's i-th term
                * (that increase is guaranteed to be >= 0 due to quads being lexicographically sorted)
                * if the increase > 0, set `zigzag = true` (ie. the `i+1`th term of this quad is not guaranteed to be >= to the `i+1`th term of the previous quad )
            * if `zigzag`: the gamma-coded [zigzag-encoded](https://docs.rs/dsi-bitstream/latest/dsi_bitstream/codes/trait.ToInt.html) difference between the previous quad's i-th term and this quad's i-th term
* end with bit 1, then `0` gamma-encoded four times (ie. as if we had the quad `(0, 0, 0, 0)`)

This allows a compact, mmappable, seekable, and fast to decompress, representation of the quads.

This stores `wikidata-20240320-truthy-BETA` quads in

* 37GiB for spog quads
* 29GiB for opsg quads

## First-term index

Each quad store has an Elias-Fano sequence mapping each term to the offset of the first frame that contains quads with that term as first term

As quads are sorted, this allows getting the beginning of the list of quads starting with the given term.
Getting the rest of the list is done by iterating from that frame (discarding the few quads lexicographically smaller than the ones we are looking for), and iterating until we find a quad lexicographically greater than the ones we are looking for.

This indexes `wikidata-20240320-truthy-BETA` quads in

* 620MiB for spog quads
* 1.9GiB for opsg quads

## First-two-terms index

This index maps pairs of terms `(term1, term2)` to the position in the quad lists where we can find quads that match `(term1, term2, _, _)`.

For this index, we cannot use an Elias-Fano index, because it is not a monotone function from integers to integers (Technically it **is**: it maps u128 integers to u64 integers as we could concatenate the first two. But there would lead to a huge number of holes (for pairs of terms that don't exist) that need to be filled up with values, wasting space.)

We could naively do this with a static function, mapping each pair of terms to the offset of the first frame that contains quads with these terms as first two terms.
This has two issues:

1. It would result in an unnecessarily big static function, as it would need to store all these offsets. Each offset takes take 28 bits on `wikidata-20240320-truthy-BETA` (because partitions weigh 200 to 250MiB).
2. Because static functions are probabilistic, querying this one with a pair of quads not in the store would result in reading nonsense by starting at an arbitrary position instead of a frame start.

Instead, we use two indexes:

* A frame index: an Elias-Fano sequence mapping storing the starting position of each frame, so we can map a frame id to its position and vice versa (note: that alone is enough to address the issue with false positive above)
* A VFunc static function mapping each pair of terms to the offset of the first frame matching `(term1, term2, _, _)` from the first frame matching `(term1, _, _, _)`. These offsets are significantly smaller than 28 bits, both because they are relative instead of absolute, and because they count frames instead of counting bits in the quads list.

We can then resolve this in a few steps:

1. get the position of the first frame matching `(term1, _, _, _)` using the index from the previous section
2. get the id of that frame from its position using the frame index
3. query the static function to get the id offset
4. add both ids together to get the id of the first frame matching `(term1, term2, _, _)`
5. get the position of that frame from its id using the frame index

As quads are sorted, this allows getting the beginning of the list of quads starting with the given two terms.
Getting the rest of the list is done by iterating from that frame (discarding the few quads lexicographically smaller than the ones we are looking for), and iterating until we find a quad lexicographically greater than the ones we are looking for.

TODO: in case of false positive that gives a tiny offset, we may end up reading a significant amount of quads before we realize there is none matching the one we want. We could use a VFilter to screen out false positives.
(Large offsets are not an issue because we just read in the sorted list of quads past where they are supposed to be, and immediately realize the quads are lexicographically too large.)

This indexes `wikidata-20240320-truthy-BETA` quads in

* 74MiB for the Elias-Fano of frame offsets of spog quads
* 280MiB for the vfunc of spog quads
* 74MiB for the Elias-Fano of frame offsets of opsg quads
* 1.6GiB for the vfunc of opsg quads

## Secondary index

Finally, we need an index to query quads matching `(_, predicate, _, _)`.

We could do this with an extra quad store (ordered by psog or posg) and a first-term index, but this can be achieved with a lightweight index mapping each `predicate` to the list of `subject` such that there are quads matching `(subject, predicate, _, _)`.

For this, we extract the list of term ids used as a predicate in any quad, and build an Elias-Fano structure containing them all. We call it a contraction (because it contracts a range of integers to a smaller one).
As a small fraction of terms are every used as predicate, this allows mapping predicates' term ids to an internal id that is a much smaller integer, and avoid wasting space in the next structure.

Next, we build a [BvGraph](https://docs.rs/webgraph/latest/webgraph/graphs/bvgraph/), which is a data structure that maps from integers to sets of integers, using as keys the internal ids we just computed, and as values subject ids.

This indexes `wikidata-20240320-truthy-BETA` in:

* 30kiB for the contraction
* 503MiB for BvGraph itself, 30kiB for its own internal Elias-Fano
