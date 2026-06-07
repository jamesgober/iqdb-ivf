# iqdb-ivf &mdash; API Reference

> Complete reference for every public item in `iqdb-ivf` v1.0.0, with
> descriptions, parameters, errors, and runnable examples.

`iqdb-ivf` is the inverted-file (IVF) approximate-nearest-neighbour index
for the iQDB vector database. It partitions the vector space with
deterministic k-means and answers each query by scanning only the few
clusters nearest the query. Two variants share one surface: **IVF-Flat**
(exact intra-cluster scan) and **IVF-PQ** (Product-Quantization codes with
optional exact refine).

---

## Table of Contents

- **[Installation](#installation)**
- **[Tiered API](#tiered-api)**
- **[Quick Start](#quick-start)**
- **[Public APIs](#public-apis)**
  - [`IvfConfig`](#ivfconfig)
  - [`IvfIndex` — construction](#ivfindex--construction)
  - [`IvfIndex` — training and lifecycle](#ivfindex--training-and-lifecycle)
  - [`IvfIndex` — insertion and deletion](#ivfindex--insertion-and-deletion)
  - [`IvfIndex` — search](#ivfindex--search)
  - [`IvfIndex` — probe and refine tuning](#ivfindex--probe-and-refine-tuning)
  - [`IvfIndex` — introspection](#ivfindex--introspection)
  - [`IvfClusterStats`](#ivfclusterstats)
  - [`Hit`](#hit)
  - [`VERSION`](#version)
- **[Traits implemented](#traits-implemented)**
- **[Errors](#errors)**
- **[Determinism](#determinism)**
- **[Performance notes](#performance-notes)**
- **[Feature flags](#feature-flags)**
- **[Notes](#notes)**

---

## Installation

```toml
[dependencies]
iqdb-ivf = "1.0"
```

`iqdb-ivf` re-exports the result types it returns ([`Hit`](#hit)) but takes
its core vocabulary — `VectorId`, `Metadata`, `SearchParams`,
`DistanceMetric`, `Filter`, `IqdbError` — from `iqdb-types`, and the
`Index` / `IndexCore` traits from `iqdb-index`. A typical consumer depends
on all three:

```toml
[dependencies]
iqdb-ivf   = "1.0"
iqdb-index = "1.0"
iqdb-types = "1.0"
```

MSRV is Rust **1.87** (edition 2024). The crate is `std`-only and has no
optional features.

---

## Tiered API

- **Tier 1 — the lazy path.** [`IvfConfig::default`](#ivfconfig) plus
  [`IvfIndex::new`](#ivfindex--construction),
  [`IvfIndex::train`](#ivfindex--training-and-lifecycle), and the
  `IndexCore` [`insert`](#ivfindex--insertion-and-deletion) /
  [`search`](#ivfindex--search) calls cover the whole common case.
- **Tier 2 — the configured path.** The builder-style `with_*` methods on
  [`IvfConfig`](#ivfconfig) tune partitioning, recall, and compression;
  [`set_n_probes`](#ivfindex--probe-and-refine-tuning),
  [`set_pq_refine_factor`](#ivfindex--probe-and-refine-tuning),
  [`suggest_n_probes`](#ivfindex--probe-and-refine-tuning),
  [`retrain`](#ivfindex--training-and-lifecycle), and
  [`cluster_stats`](#ivfindex--introspection) tune and inspect a live
  index.
- **Tier 3 — the trait seam.** [`IvfIndex`](#traits-implemented)
  implements `iqdb_index::Index` and `iqdb_index::IndexCore`, so it is
  interchangeable with any other backend behind those traits.

---

## Quick Start

```rust
use std::sync::Arc;

use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, SearchParams, VectorId};

# fn main() -> iqdb_types::Result<()> {
// 1. Configure and construct (untrained).
let cfg = IvfConfig::default()
    .with_n_clusters(2)
    .with_n_probes(2)
    .with_training_sample_size(64)
    .with_seed(7);
let mut idx = IvfIndex::new(2, DistanceMetric::Euclidean, cfg)?;

// 2. Train on a representative sample (mandatory before insert/search).
let sample: Vec<Vec<f32>> = vec![
    vec![0.0, 0.0], vec![0.2, -0.1], vec![-0.1, 0.1],
    vec![10.0, 10.0], vec![10.2, 9.8], vec![9.9, 10.1],
];
let sample_refs: Vec<&[f32]> = sample.iter().map(|v| v.as_slice()).collect();
idx.train(&sample_refs)?;

// 3. Insert and search.
idx.insert(VectorId::from(1u64), Arc::<[f32]>::from(&[0.0, 0.0][..]), None)?;
idx.insert(VectorId::from(2u64), Arc::<[f32]>::from(&[10.0, 10.0][..]), None)?;

let hits = idx.search(&[0.0, 0.0], &SearchParams::new(1, DistanceMetric::Euclidean))?;
assert_eq!(hits[0].id, VectorId::U64(1));
# Ok(())
# }
```

---

## Public APIs

### `IvfConfig`

Typed configuration for [`IvfIndex`](#ivfindex--construction). A plain
`Copy` struct with documented defaults and builder-style `with_*` overrides.

```rust
pub struct IvfConfig {
    pub n_clusters: usize,
    pub n_probes: usize,
    pub training_sample_size: usize,
    pub use_pq: bool,
    pub pq_subvectors: Option<usize>,
    pub pq_refine_factor: u32,
    pub seed: u64,
}
```

**Fields**

| Field | Default | Meaning |
|-------|---------|---------|
| `n_clusters` | `256` | Number of k-means partitions (inverted lists). Spec heuristic: `sqrt(N)` for moderate corpora, `4·sqrt(N)` for very large. Must be ≥ 1. |
| `n_probes` | `8` | Clusters scanned per query. Higher → higher recall, higher latency. Must be ≥ 1 and ≤ `n_clusters`. |
| `training_sample_size` | `65_536` | Cap on the k-means training sample; larger samples are deterministically subsampled down. Must be ≥ 1. |
| `use_pq` | `false` | Enable IVF-PQ (Product Quantization within clusters). |
| `pq_subvectors` | `None` | Subvector count `M` for IVF-PQ. Required `Some(m)` with `m ≥ 1` and `m \| dim` when `use_pq`. |
| `pq_refine_factor` | `4` | IVF-PQ refine factor. `0` returns the pure ADC top-`k`; `N ≥ 1` shortlists `N·k` by ADC and exact-reranks. Ignored when `use_pq` is `false`. |
| `seed` | `0xDEAD_BEEF_CAFE_F00D` | Seed for the SplitMix64 PRNG (k-means++ init and subsampling). |

**Constructors and builders**

- `IvfConfig::default() -> Self` — the documented operating point above.
- `with_n_clusters(self, usize) -> Self`
- `with_n_probes(self, usize) -> Self`
- `with_training_sample_size(self, usize) -> Self`
- `with_use_pq(self, bool) -> Self`
- `with_pq_subvectors(self, Option<usize>) -> Self`
- `with_pq_refine_factor(self, u32) -> Self`
- `with_seed(self, u64) -> Self`

Each `with_*` consumes and returns `self`, so they chain.

**`validate(&self) -> Result<()>`** — called automatically by
[`IvfIndex::new`](#ivfindex--construction). Returns
`IqdbError::InvalidConfig { reason }` naming the failing check
(`n_clusters == 0`, `n_probes == 0`, `n_probes > n_clusters`,
`training_sample_size == 0`, or `use_pq` without a valid
`pq_subvectors`). The `m | dim` divisibility check and the metric guard
run later, in [`IvfIndex::new`](#ivfindex--construction), where `dim` and
`metric` are known.

**Example — default vs tuned**

```rust
use iqdb_ivf::IvfConfig;

let cfg = IvfConfig::default();
assert_eq!(cfg.n_clusters, 256);
assert_eq!(cfg.n_probes, 8);

let tuned = IvfConfig::default()
    .with_n_clusters(1024)
    .with_n_probes(16)
    .with_seed(42);
assert_eq!(tuned.n_clusters, 1024);
assert!(tuned.validate().is_ok());
```

**Example — IVF-PQ config**

```rust
use iqdb_ivf::IvfConfig;

// 8 subvectors; valid once paired with a dim divisible by 8.
let pq = IvfConfig::default()
    .with_use_pq(true)
    .with_pq_subvectors(Some(8))
    .with_pq_refine_factor(4);
assert!(pq.validate().is_ok());
```

---

### `IvfIndex` — construction

```rust
// Via the `Index` trait (idiomatic):
fn new(dim: usize, metric: DistanceMetric, config: IvfConfig) -> Result<Self>

// Inherent equivalent (no trait import needed):
pub fn new_unconfigured(dim: usize, metric: DistanceMetric, cfg: IvfConfig) -> Result<Self>
```

Builds an empty, **untrained** index. Both entry points are identical;
`Index::new` is the trait method, `new_unconfigured` is the inherent
method for when the concrete type is already in scope.

**Parameters**

- `dim` — vector dimensionality. Must be > 0.
- `metric` — the `DistanceMetric` fixed for the index's lifetime. For
  IVF-PQ only `Euclidean`, `DotProduct`, and `Manhattan` are supported.
- `config` / `cfg` — an [`IvfConfig`](#ivfconfig).

**Errors**

- `IqdbError::InvalidConfig` — `dim == 0`, the config fails
  [`validate`](#ivfconfig), or (IVF-PQ) `pq_subvectors` does not divide
  `dim`.
- `IqdbError::InvalidMetric` — `use_pq` with `Cosine` or `Hamming`.

**Example**

```rust
use iqdb_index::Index;
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::DistanceMetric;

# fn main() -> iqdb_types::Result<()> {
let idx = IvfIndex::new(128, DistanceMetric::Cosine, IvfConfig::default())?;
assert_eq!(idx.dim(), 128);
assert!(!idx.is_trained());
# Ok(())
# }
```

---

### `IvfIndex` — training and lifecycle

**`train(&mut self, sample: &[&[f32]]) -> Result<()>`**

Runs k-means on `sample` to produce the centroids (and, when `use_pq`, the
PQ codebooks). Mandatory before insert/search succeed. After a successful
train the sample is no longer referenced — only the derived centroids are
retained.

- **Parameters:** `sample` — at least `n_clusters` vectors, each of length
  `dim`. For IVF-PQ the sample must contain at least 256 vectors (the PQ
  codebook size `K`).
- **Errors:** `IqdbError::InvalidConfig` (empty sample,
  `sample.len() < n_clusters`, or already trained — use
  [`retrain`](#ivfindex--training-and-lifecycle) instead),
  `IqdbError::DimensionMismatch` (a sample vector of the wrong length), or
  a propagated PQ-training error.

**`retrain(&mut self) -> Result<()>`**

Rebuilds the centroids (and PQ codebooks) over the *currently-stored*
vectors and reassigns every entry — without losing data. Use it after a
delete-heavy phase or a distribution shift, when
[`cluster_stats`](#ivfindex--introspection) shows imbalance.

- **Determinism:** the working set is the live snapshot sorted by insertion
  sequence, deterministically subsampled to `training_sample_size`; same
  seed + same history → byte-identical result.
- **Complexity:** the training stages stay `O(training_sample_size)`;
  reassign/re-encode touches every entry.
- **Errors:** `IqdbError::InvalidConfig` if not yet trained. An empty
  trained index is a no-op returning `Ok(())`.

**`is_trained(&self) -> bool`** — `true` once `train` has produced
centroids.

**Example — train, then retrain after deletes**

```rust
use std::sync::Arc;
use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, VectorId};

# fn main() -> iqdb_types::Result<()> {
let cfg = IvfConfig::default().with_n_clusters(4).with_n_probes(4)
    .with_training_sample_size(256).with_seed(1);
let mut idx = IvfIndex::new(3, DistanceMetric::Euclidean, cfg)?;

let sample: Vec<Vec<f32>> = (0..64)
    .map(|i| vec![(i % 4) as f32, (i % 3) as f32, (i % 2) as f32])
    .collect();
let refs: Vec<&[f32]> = sample.iter().map(|v| v.as_slice()).collect();
idx.train(&refs)?;
assert!(idx.is_trained());

idx.insert(VectorId::from(1u64), Arc::<[f32]>::from(&[0.0, 0.0, 0.0][..]), None)?;
idx.delete(&VectorId::from(1u64))?;
idx.retrain()?; // no-op here (empty), but always valid once trained
# Ok(())
# }
```

---

### `IvfIndex` — insertion and deletion

These come from the `IndexCore` trait.

**`insert(&mut self, id: VectorId, vector: Arc<[f32]>, metadata: Option<Metadata>) -> Result<()>`**

Assigns `vector` to its nearest cluster (and, in IVF-PQ mode, encodes its
PQ code up front). The payload `Arc` is stored verbatim — zero-copy.

- **Errors:** `IqdbError::InvalidConfig` (not trained),
  `IqdbError::DimensionMismatch`, `IqdbError::Duplicate` (id already
  present).

**`insert_batch(&mut self, items: Vec<(VectorId, Arc<[f32]>, Option<Metadata>)>) -> Result<()>`**

Default trait method: inserts each item in turn.

**`delete(&mut self, id: &VectorId) -> Result<()>`**

Cluster-local `swap_remove`. `O(cluster_size)`.

- **Errors:** `IqdbError::NotFound` when the id is absent.

**Example**

```rust
use std::sync::Arc;
use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, VectorId};

# fn main() -> iqdb_types::Result<()> {
let cfg = IvfConfig::default().with_n_clusters(2).with_n_probes(2)
    .with_training_sample_size(16).with_seed(5);
let mut idx = IvfIndex::new(2, DistanceMetric::Euclidean, cfg)?;
let sample = [[0.0f32, 0.0], [9.0, 9.0], [0.1, 0.0], [9.1, 9.0]];
let refs: Vec<&[f32]> = sample.iter().map(|v| v.as_slice()).collect();
idx.train(&refs)?;

idx.insert(VectorId::from(1u64), Arc::<[f32]>::from(&[0.0, 0.0][..]), None)?;
assert_eq!(idx.len(), 1);

// Re-inserting the same id is rejected.
assert!(idx.insert(VectorId::from(1u64), Arc::<[f32]>::from(&[0.0, 0.0][..]), None).is_err());

idx.delete(&VectorId::from(1u64))?;
assert!(idx.is_empty());
# Ok(())
# }
```

---

### `IvfIndex` — search

**`search(&self, query: &[f32], params: &SearchParams) -> Result<Vec<Hit>>`**

Two passes: rank centroids exactly, then scan the `n_probes` nearest
clusters. Returns up to `params.k` hits, sorted nearest-first, with ties
broken by insertion order. Distances are always *smaller-is-nearer* —
`DotProduct` is negated at the boundary.

- **Parameters:** `query` of length `dim`; `params` carrying `k`, `metric`
  (must match the index's metric), and an optional `filter`.
- **Filter:** when `params.filter` is `Some`, the predicate is validated
  once (a pathological filter surfaces as `IqdbError::InvalidFilter`) and
  applied per entry *before* distance is computed. A selective filter may
  return fewer than `k` hits within the probed clusters — IVF does not
  widen the probe set to compensate.
- **Errors:** `IqdbError::DimensionMismatch`, `IqdbError::InvalidMetric`
  (metric mismatch), `IqdbError::InvalidFilter`,
  `IqdbError::InvalidConfig` (not trained). `k == 0` or an empty index
  returns `Ok(vec![])`.

**`search_batch(&self, queries: &[&[f32]], params: &SearchParams) -> Result<Vec<Vec<Hit>>>`**

Default trait method: one `search` per query.

**Example — unfiltered and filtered**

```rust
use std::sync::Arc;
use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, Filter, Metadata, SearchParams, Value, VectorId};

# fn main() -> iqdb_types::Result<()> {
let cfg = IvfConfig::default().with_n_clusters(2).with_n_probes(2)
    .with_training_sample_size(16).with_seed(7);
let mut idx = IvfIndex::new(2, DistanceMetric::Euclidean, cfg)?;
let sample = [[0.0f32, 0.0], [9.0, 9.0], [1.0, 1.0], [8.0, 8.0]];
let refs: Vec<&[f32]> = sample.iter().map(|v| v.as_slice()).collect();
idx.train(&refs)?;

let hot: Metadata = [("tier".to_string(), Value::String("hot".into()))].into_iter().collect();
idx.insert(VectorId::from(1u64), Arc::<[f32]>::from(&[0.0, 0.0][..]), None)?;
idx.insert(VectorId::from(2u64), Arc::<[f32]>::from(&[1.0, 1.0][..]), Some(hot))?;

// Nearest overall.
let near = idx.search(&[0.0, 0.0], &SearchParams::new(1, DistanceMetric::Euclidean))?;
assert_eq!(near[0].id, VectorId::U64(1));

// Nearest with tier=hot.
let params = SearchParams {
    filter: Some(Filter::eq("tier", Value::String("hot".into()))),
    ..SearchParams::new(1, DistanceMetric::Euclidean)
};
let filtered = idx.search(&[0.0, 0.0], &params)?;
assert_eq!(filtered[0].id, VectorId::U64(2));
# Ok(())
# }
```

---

### `IvfIndex` — probe and refine tuning

**`n_probes(&self) -> usize`** — the current probe count.

**`set_n_probes(&mut self, n: usize) -> Result<()>`** — override the probe
count at runtime. Errors with `IqdbError::InvalidConfig` when `n == 0` or
`n > n_clusters`. Does not require training.

**`suggest_n_probes(&self, coverage: f32) -> Result<usize>`** — recommend
the smallest probe count whose largest clusters cover at least `coverage`
of the live vectors. Pure and side-effect-free; compose with
`set_n_probes`. Monotone non-decreasing in `coverage`, clamped into
`[1, n_clusters]`.

- **Errors:** `IqdbError::InvalidConfig` when `coverage` is non-finite or
  outside `[0.0, 1.0]`.
- **Edges:** untrained/empty → `1`; `coverage == 0.0` → `1`;
  `coverage == 1.0` → `n_clusters`.

**`pq_refine_factor(&self) -> u32`** / **`set_pq_refine_factor(&mut self, factor: u32)`**
— read/override the IVF-PQ refine factor at runtime. `0` returns the pure
ADC top-`k`; `N ≥ 1` shortlists `N·k` by ADC and exact-reranks. A no-op for
IVF-Flat (the value is stored but unused).

**Example — tune probes from the live distribution**

```rust
use std::sync::Arc;
use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, VectorId};

# fn main() -> iqdb_types::Result<()> {
let cfg = IvfConfig::default().with_n_clusters(8).with_n_probes(1)
    .with_training_sample_size(512).with_seed(2);
let mut idx = IvfIndex::new(4, DistanceMetric::Euclidean, cfg)?;
let sample: Vec<Vec<f32>> = (0..200)
    .map(|i| vec![(i % 8) as f32, (i % 5) as f32, (i % 3) as f32, (i % 2) as f32])
    .collect();
let refs: Vec<&[f32]> = sample.iter().map(|v| v.as_slice()).collect();
idx.train(&refs)?;
for (i, v) in sample.iter().enumerate() {
    idx.insert(VectorId::from(i as u64), Arc::<[f32]>::from(v.as_slice()), None)?;
}

let probes = idx.suggest_n_probes(0.90)?;
idx.set_n_probes(probes)?;
assert_eq!(idx.n_probes(), probes);
# Ok(())
# }
```

---

### `IvfIndex` — introspection

- **`dim(&self) -> usize`** — the configured dimensionality.
- **`metric(&self) -> DistanceMetric`** — the configured metric.
- **`len(&self) -> usize`** — searchable (live) vector count, `O(1)`.
- **`is_empty(&self) -> bool`** — `true` when `len() == 0`.
- **`cluster_stats(&self) -> IvfClusterStats`** — snapshot of inverted-list
  occupancy; see [`IvfClusterStats`](#ivfclusterstats).
- **`stats(&self) -> IndexStats`** (`IndexCore`) — `n_vectors`, an
  approximate `memory_bytes`, `index_type = "ivf"`, and no `disk_bytes`.
- **`flush(&mut self) -> Result<()>`** (`IndexCore`) — `Ok(())`; IVF is
  purely in-memory.

---

### `IvfClusterStats`

Diagnostic snapshot returned by
[`cluster_stats`](#ivfindex--introspection).

```rust
pub struct IvfClusterStats {
    pub n_clusters: usize,
    pub cluster_sizes: Vec<usize>, // empty before training, else len == n_clusters
    pub avg_size: f32,
    pub size_variance: f32,        // higher → more imbalance
}
```

A long tail of empty clusters next to a few packed ones (high
`size_variance`) is the signal to call
[`retrain`](#ivfindex--training-and-lifecycle).

```rust
use iqdb_ivf::IvfClusterStats;

let stats = IvfClusterStats {
    n_clusters: 4,
    cluster_sizes: vec![3, 3, 4, 2],
    avg_size: 3.0,
    size_variance: 0.5,
};
assert_eq!(stats.cluster_sizes.iter().sum::<usize>(), 12);
```

---

### `Hit`

Re-exported from `iqdb-types` for convenience. A search result:

```rust
pub struct Hit {
    pub id: VectorId,
    pub distance: f32,          // smaller is nearer, across all metrics
    pub metadata: Option<Metadata>,
}
```

---

### `VERSION`

```rust
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
```

The crate version string, e.g. `"1.0.0"`.

```rust
assert_eq!(iqdb_ivf::VERSION.split('.').count(), 3);
```

---

## Traits implemented

`IvfIndex` implements both index traits from `iqdb-index`:

- **`IndexCore`** — `insert`, `insert_batch`, `delete`, `search`,
  `search_batch`, `len`, `is_empty`, `dim`, `metric`, `flush`, `stats`.
- **`Index`** — `type Config = IvfConfig;` and
  `new(dim, metric, config)`.

This means `IvfIndex` is a drop-in replacement for any other index behind
those traits (for example `iqdb_flat::FlatIndex` or
`iqdb_hnsw::HnswIndex`).

---

## Errors

Every fallible method returns `iqdb_types::Result<T>` =
`Result<T, IqdbError>`. `IqdbError` is `#[non_exhaustive]`; the variants
this crate produces are:

| Variant | When |
|---------|------|
| `InvalidConfig { reason }` | bad `IvfConfig`, `dim == 0`, untrained use, sequence-counter overflow, internal invariant violation. `reason` is a `&'static str` naming the cause. |
| `InvalidMetric` | `params.metric` ≠ the index metric, or IVF-PQ with `Cosine`/`Hamming`. |
| `InvalidFilter` | a search filter exceeds the evaluator's depth or `In`-cardinality caps. |
| `DimensionMismatch { expected, found }` | a vector or query of the wrong length. |
| `Duplicate` | `insert` of an id already present. |
| `NotFound` | `delete` of an absent id. |

`IqdbError` implements `error_forge::ForgeError`, so callers get
`kind()` / `caption()` / `is_retryable()` for structured logging.

---

## Determinism

Given the same `IvfConfig::seed`, the same training sample, and the same
`n_clusters` / `training_sample_size`, `train` produces byte-identical
centroids on every supported platform. This holds because every PRNG draw
comes from the in-tree SplitMix64 generator, all reductions run in a fixed
sequential order, and centroid sums accumulate in `f64` before a single
downcast to `f32`. In IVF-PQ mode the same seed flows into the codebook
trainer, so the codes are reproducible too. `retrain` is deterministic in
the seed plus the insert/delete history.

---

## Performance notes

- **Two-pass search.** Pass 1 ranks all `n_clusters` centroids exactly;
  Pass 2 scans only the `n_probes` nearest clusters. Query cost is
  predictable and roughly `O(n_clusters + n_probes · n/n_clusters)`.
- **`n_probes` is the dial.** It trades recall for latency without ever
  widening behind your back — latency stays predictable.
- **Filter-first.** A metadata filter is evaluated before distance, so a
  selective predicate skips distance work proportionally.
- **IVF-PQ.** Compresses each vector to `M` bytes and scores by ADC from a
  per-query lookup table built once and reused across probes; the optional
  exact refine reranks an `N·k` shortlist against the retained vectors.
- **Zero-copy payloads.** Inserted `Arc<[f32]>` values are stored verbatim;
  the index shares one allocation with the caller's record store.
- All distance math is delegated to `iqdb_distance::compute_batch`, which
  uses SIMD kernels where the target allows.

Benchmarks live in [`benches/ivf_bench.rs`](../benches/ivf_bench.rs)
(`cargo bench`).

---

## Feature flags

None. `iqdb-ivf` is `std`-only and ships its full surface unconditionally.

---

## Notes

- **Plain-PQ.** IVF-PQ trains codebooks on the raw working set (not on
  per-cluster residuals), so the ADC table is cluster-independent and built
  once per query. This is the frozen 1.0 contract.
- **Assignment uses squared L2.** k-means is mathematically tied to L2
  centroids, so partitioning always assigns in squared-L2 space regardless
  of the configured search metric; the configured metric still governs
  ranking at query time.
- **Persistence and caching are out of scope** for this crate — they are
  separate iQDB crates.

---

<sub>Copyright &copy; 2026 <strong>James Gober</strong>.</sub>
