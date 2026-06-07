# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

### Changed

### Fixed

### Security

---

## [1.0.0] - 2026-06-07

First stable release. The inverted-file index is feature-complete and its
public API is frozen under the SemVer 1.x guarantee — no breaking changes
until 2.0. Built on the stable `iqdb-types`, `iqdb-distance`, `iqdb-index`,
`iqdb-filter`, and `iqdb-quantize` 1.0 crates.

### Added

- **`IvfIndex`** — the inverted-file approximate-nearest-neighbour index,
  implementing `iqdb_index::Index` and `iqdb_index::IndexCore`. Storage is
  one inverted list per centroid, each owning `(VectorId, Arc<[f32]>,
  Option<Metadata>, seq)` tuples; payloads are stored as zero-copy
  `Arc<[f32]>`.
- **Deterministic k-means training** (`IvfIndex::train`) — k-means++ seeding
  and Lloyd's iterations driven by an in-tree SplitMix64 PRNG, with `f64`
  centroid accumulation and fixed-order reductions, so identical seed +
  identical sample produce byte-identical centroids on every platform.
  Training is mandatory before insert/search; the dependent entry points
  short-circuit with `IqdbError::InvalidConfig` until trained.
- **IVF-Flat search** — a two-pass query (exact centroid ranking, then an
  exact intra-cluster scan over the `n_probes` nearest clusters) with a
  bounded-heap `(distance, seq)` top-`k`. One *smaller-is-nearer* ordering
  across all five metrics; `DotProduct` is negated at the boundary.
- **IVF-PQ search** — set `use_pq = true` with `pq_subvectors = Some(m)`
  (`m | dim`) to compress each vector to an `m`-byte Product-Quantization
  code (`K = 256`) trained on the same working set as the centroids
  (plain-PQ). Intra-cluster scoring uses an asymmetric-distance (ADC) lookup
  table built once per query via `iqdb_quantize::PqAdcTables`. Supported
  metrics: `Euclidean`, `DotProduct`, `Manhattan`. Centroid ranking stays
  exact.
- **IVF-PQ refine** (`IvfConfig::pq_refine_factor`, default `4`) — shortlist
  `factor × k` candidates by ADC and exact-rerank them against the retained
  `Arc<[f32]>` vectors. Set `0` to return the pure ADC top-`k`. Tunable at
  runtime via `IvfIndex::set_pq_refine_factor` / `pq_refine_factor`.
- **Metadata-filtered search** — a `SearchParams::filter` is validated once
  through `iqdb_filter::FilterEvaluator::new` (pathological filters surface
  as `IqdbError::InvalidFilter`) and evaluated per entry *before* distance,
  across both IVF-Flat and IVF-PQ. The unfiltered hot path carries no
  per-entry filter branch. For IVF-PQ the filter runs before the ADC lookup,
  and the refine shortlist inherits filter survivorship.
- **`IvfIndex::retrain`** — rebuild centroids (and, in IVF-PQ mode,
  codebooks) over the currently-stored vectors and reassign/re-encode every
  entry, without losing data. Deterministic in the seed plus the
  insert/delete history; the training stages stay
  `O(training_sample_size)`. No-op on an empty trained index.
- **Probe tuning** — `IvfIndex::n_probes` / `set_n_probes`, and
  `suggest_n_probes(coverage)`, a pure, monotone, cluster-size-driven
  recommendation clamped to `[1, n_clusters]` that composes with
  `set_n_probes`. IVF never widens the probe set at query time, so latency
  stays predictable.
- **`IvfConfig`** — a `Copy` config with documented defaults and
  builder-style `with_*` overrides (`with_n_clusters`, `with_n_probes`,
  `with_training_sample_size`, `with_use_pq`, `with_pq_subvectors`,
  `with_pq_refine_factor`, `with_seed`) plus `validate`.
- **`IvfClusterStats`** (`IvfIndex::cluster_stats`) — a diagnostic snapshot
  of inverted-list occupancy (`cluster_sizes`, `avg_size`, `size_variance`)
  to detect the cluster imbalance that motivates `retrain`.
- **`IndexCore` surface** — `insert`, `insert_batch`, `delete` (cluster-local
  `swap_remove`), `search`, `search_batch`, `len`, `is_empty`, `dim`,
  `metric`, `flush`, and `stats` (reporting `index_type = "ivf"`).
- **Tracing instrumentation** at the insert/delete/search boundary, with
  structured `error.kind` / `error.reason` fields via
  `error_forge::ForgeError`. The per-entry scan carries no tracing call.
- **`VERSION`** constant.
- Documentation: a complete `docs/API.md`, a worked `README.md`, six
  runnable examples (`quick_start`, `ivf_pq`, `filtered_search`, `retrain`,
  `probe_tuning`, `recall_oracle`), and `criterion` benchmarks in
  `benches/ivf_bench.rs`.
- Tests: per-module unit tests, property-based invariants (`proptest`)
  including full-probe equivalence to the exact `iqdb-flat` oracle, and a
  recall-at-scale validation suite for IVF-Flat and IVF-PQ.

### Notes

- **Assignment uses squared L2.** k-means is tied to L2 centroids, so
  partitioning always assigns in squared-L2 space regardless of the
  configured search metric; the configured metric still governs ranking at
  query time.
- **Plain-PQ, not residual-PQ.** Codebooks are trained on the raw working
  set so the ADC table is cluster-independent and built once per query. This
  is the frozen 1.0 contract.
- **Persistence and caching are out of scope** for this crate; they are
  separate iQDB crates.

---

## [0.1.0] - 2026-05-30

Initial scaffold and repository bootstrap. No domain logic yet &mdash; this release establishes the structure, tooling, and quality gates the implementation will be built on.

### Added

- `Cargo.toml` with crate metadata, Rust 2024 edition, MSRV 1.87.
- Dual `Apache-2.0 OR MIT` license files.
- `README.md`, `CHANGELOG.md`, and a documentation skeleton.
- `REPS.md` compliance baseline.
- `.github/workflows/ci.yml` CI matrix; `deny.toml`, `clippy.toml`, `rustfmt.toml`.
- `dev/DIRECTIVES.md` and `dev/ROADMAP.md` (committed engineering standards + plan).

[Unreleased]: https://github.com/jamesgober/iqdb-ivf/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/jamesgober/iqdb-ivf/compare/v0.1.0...v1.0.0
[0.1.0]: https://github.com/jamesgober/iqdb-ivf/releases/tag/v0.1.0
