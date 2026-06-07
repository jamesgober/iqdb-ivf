//! # iqdb-ivf
//!
//! Inverted-file (IVF) approximate nearest-neighbour index for the iQDB
//! vector database. [`IvfIndex`] partitions the vector space with
//! deterministic k-means and answers queries by exhaustively scanning only
//! the [`IvfConfig::n_probes`] clusters whose centroids are nearest to the
//! query. It is the complement to a graph index: more memory-efficient at
//! very large scale and with more predictable latency, because every query
//! probes the same fixed number of clusters.
//!
//! Two variants share one surface. **IVF-Flat** (the default) stores
//! vectors verbatim and scores probed clusters exactly. **IVF-PQ**
//! (`use_pq = true`) compresses each vector to a Product-Quantization code
//! and scores by asymmetric distance (ADC), optionally exact-reranking a
//! shortlist against the retained vectors. Both honour metadata filters and
//! the same `smaller-is-nearer` ordering contract as every other iQDB index.
//!
//! ## Tiered API
//!
//! - **Tier 1 — the lazy path.** [`IvfConfig::default`] plus
//!   [`IvfIndex::new`](iqdb_index::Index::new), [`IvfIndex::train`], and
//!   the [`iqdb_index::IndexCore`] insert/search calls cover the whole
//!   common case with no generics to name.
//! - **Tier 2 — the configured path.** The builder-style
//!   [`IvfConfig::with_n_clusters`] / [`IvfConfig::with_n_probes`] /
//!   [`IvfConfig::with_use_pq`] (and friends) tune partitioning, recall,
//!   and compression, while [`IvfIndex::set_n_probes`],
//!   [`IvfIndex::set_pq_refine_factor`], [`IvfIndex::suggest_n_probes`],
//!   [`IvfIndex::retrain`], and [`IvfIndex::cluster_stats`] tune and
//!   inspect a live index.
//! - **Tier 3 — the trait seam.** [`IvfIndex`] implements
//!   [`iqdb_index::Index`] and [`iqdb_index::IndexCore`], so it is
//!   interchangeable with any other backend behind those traits.
//!
//! ## Design
//!
//! - Storage is `n_clusters` inverted lists, each owning the
//!   `(VectorId, Arc<[f32]>, Option<Metadata>, seq)` tuples for the
//!   vectors assigned to its centroid. The payload is wrapped in
//!   [`Arc<[f32]>`](std::sync::Arc) so the engine shares one allocation
//!   between this index and its record store — the same zero-copy
//!   contract `iqdb-flat` and `iqdb-hnsw` use.
//! - **Training is mandatory.** [`IvfIndex::new`](iqdb_index::Index::new)
//!   returns an untrained index; the four entry points that depend on
//!   centroids ([`insert`](iqdb_index::IndexCore::insert),
//!   [`insert_batch`](iqdb_index::IndexCore::insert_batch),
//!   [`search`](iqdb_index::IndexCore::search), and
//!   [`search_batch`](iqdb_index::IndexCore::search_batch))
//!   short-circuit with [`iqdb_types::IqdbError::InvalidConfig`] until the
//!   caller invokes [`IvfIndex::train`]. This UX cliff is intentional and
//!   documented loudly.
//! - K-means runs through a seeded SplitMix64 PRNG, accumulating centroid
//!   sums in `f64` and reducing in fixed order, so identical `seed` plus
//!   identical sample produce byte-identical centroids on every platform.
//! - All distance math is delegated to [`iqdb_distance::compute_batch`];
//!   IVF never reimplements a metric. For
//!   [`iqdb_types::DistanceMetric::DotProduct`] the raw inner product is
//!   negated at the boundary so [`iqdb_types::Hit::distance`] is always
//!   *smaller-is-nearer*, identical across all five metrics.
//! - Top-`k` selection uses a bounded max-heap keyed on `(distance, seq)`
//!   via [`f32::total_cmp`] — `O(n log k)`, NaN-safe, and deterministic:
//!   ties break on insertion order (lower sequence number wins).
//! - Metadata filtering goes through [`iqdb_filter::FilterEvaluator`];
//!   pathological filters surface as
//!   [`iqdb_types::IqdbError::InvalidFilter`] at query time, before any
//!   cluster data is touched.
//!
//! ## Example
//!
//! ```
//! use std::sync::Arc;
//!
//! use iqdb_index::{Index, IndexCore};
//! use iqdb_ivf::{IvfConfig, IvfIndex};
//! use iqdb_types::{DistanceMetric, SearchParams, VectorId};
//!
//! # fn main() -> iqdb_types::Result<()> {
//! let cfg = IvfConfig::default()
//!     .with_n_clusters(2)
//!     .with_n_probes(2)
//!     .with_training_sample_size(16)
//!     .with_seed(7);
//! let mut idx = IvfIndex::new(2, DistanceMetric::Euclidean, cfg)?;
//!
//! // Train on a representative sample before any insert / search.
//! let sample: Vec<Vec<f32>> = vec![
//!     vec![0.0, 0.0], vec![0.1, -0.1], vec![-0.1, 0.1],
//!     vec![10.0, 10.0], vec![10.1, 9.9], vec![9.9, 10.1],
//! ];
//! let sample_refs: Vec<&[f32]> = sample.iter().map(|v| v.as_slice()).collect();
//! idx.train(&sample_refs)?;
//!
//! idx.insert(VectorId::from(1u64), Arc::<[f32]>::from(&[0.0, 0.0][..]), None)?;
//! idx.insert(VectorId::from(2u64), Arc::<[f32]>::from(&[10.0, 10.0][..]), None)?;
//!
//! let hits = idx.search(&[0.0, 0.0], &SearchParams::new(1, DistanceMetric::Euclidean))?;
//! assert_eq!(hits.len(), 1);
//! assert_eq!(hits[0].id, VectorId::U64(1));
//! # Ok(())
//! # }
//! ```

#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(warnings)]
#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]
#![deny(unused_must_use)]
#![deny(unused_results)]
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![deny(clippy::print_stdout)]
#![deny(clippy::print_stderr)]
#![deny(clippy::dbg_macro)]
#![deny(clippy::unreachable)]
#![deny(clippy::undocumented_unsafe_blocks)]
#![forbid(unsafe_code)]

mod assign;
mod config;
mod index;
mod pq_variant;
mod rng;
mod search;
mod stats;
mod topk;
mod train;

pub use crate::config::IvfConfig;
pub use crate::index::IvfIndex;
pub use crate::stats::IvfClusterStats;

// Re-export the `Hit` type that searches return so callers can drive
// `IvfIndex` without a second `use` line for the result type.
pub use iqdb_types::Hit;

/// The version of this crate, taken from `Cargo.toml` at compile time.
///
/// # Examples
///
/// ```
/// let version = iqdb_ivf::VERSION;
/// assert_eq!(version.split('.').count(), 3);
/// ```
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
