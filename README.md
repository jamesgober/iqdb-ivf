<h1 align="center">
    <img width="99" alt="Rust logo" src="https://raw.githubusercontent.com/jamesgober/rust-collection/72baabd71f00e14aa9184efcb16fa3deddda3a0a/assets/rust-logo.svg">
    <br>
    <b>iqdb-ivf</b>
    <br>
    <sub><sup>iQDB IVF INDEX</sup></sub>
</h1>

<div align="center">
    <a href="https://crates.io/crates/iqdb-ivf"><img alt="Crates.io" src="https://img.shields.io/crates/v/iqdb-ivf"></a>
    <a href="https://crates.io/crates/iqdb-ivf"><img alt="Downloads" src="https://img.shields.io/crates/d/iqdb-ivf?color=%230099ff"></a>
    <a href="https://docs.rs/iqdb-ivf"><img alt="docs.rs" src="https://img.shields.io/docsrs/iqdb-ivf"></a>
    <a href="https://github.com/jamesgober/iqdb-ivf/actions"><img alt="CI" src="https://github.com/jamesgober/iqdb-ivf/actions/workflows/ci.yml/badge.svg"></a>
    <a href="https://github.com/rust-lang/rfcs/blob/master/text/2495-min-rust-version.md"><img alt="MSRV" src="https://img.shields.io/badge/MSRV-1.87%2B-blue"></a>
</div>

<br>

<div align="left">
    <p>
        <strong>iqdb-ivf</strong> partitions the vector space into clusters with deterministic k-means and searches only the few clusters nearest each query. It is the complement to a graph index: more memory-efficient at very large scale and with more predictable latency, because every query probes the same fixed number of clusters.
    </p>
    <p>
        Two variants share one surface. <strong>IVF-Flat</strong> stores vectors verbatim and scans probed clusters exactly. <strong>IVF-PQ</strong> compresses each vector to a Product-Quantization code, scores by asymmetric distance, and optionally exact-reranks a shortlist &mdash; trading a little accuracy for a large drop in memory and scan cost.
    </p>
    <br>
    <hr>
    <p>
        <strong>MSRV is 1.87+</strong> (Rust 2024 edition). Clustered search, IVF-Flat and IVF-PQ, predictable latency at scale.
    </p>
    <blockquote>
        <strong>Status: stable (1.0).</strong> The full index &mdash; k-means training, IVF-Flat and IVF-PQ search, metadata-filtered search, retrain, and probe tuning &mdash; is committed under the SemVer 1.x guarantee: no breaking changes until 2.0. See <a href="./CHANGELOG.md"><code>CHANGELOG.md</code></a>.
    </blockquote>
</div>

<hr>
<br>

<h2>What it does</h2>

- **Inverted-file index** &mdash; deterministic k-means partitions the space; each query scans only the `n_probes` nearest clusters
- **IVF-Flat and IVF-PQ** &mdash; store vectors as-is for exact intra-cluster scoring, or compress them with Product Quantization for huge-scale memory savings
- **Predictable latency** &mdash; a fixed probe count means deterministic query cost; IVF never widens the probe set behind your back
- **Deterministic** &mdash; same seed + same data → byte-identical centroids on every platform; one *smaller-is-nearer* ordering across all five metrics with a stable insertion-order tiebreaker
- **Trainable + retrainable** &mdash; rebuild clusters as the distribution drifts, without losing data
- **Tunable recall** &mdash; `n_probes` is the recall/latency dial, with a cluster-size-driven `suggest_n_probes` to pick it

<br>

## Installation

```toml
[dependencies]
iqdb-ivf   = "1.0"
iqdb-index = "1.0"   # the Index / IndexCore traits
iqdb-types = "1.0"   # VectorId, SearchParams, DistanceMetric, Filter, ...
```

`iqdb-ivf` is `std`-only with no optional features.

<br>

## Quick Start

IVF must learn its partitions before it can index anything, so the lifecycle
is **configure → train → insert → search**:

```rust
use std::sync::Arc;

use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, SearchParams, VectorId};

fn main() -> iqdb_types::Result<()> {
    let cfg = IvfConfig::default()
        .with_n_clusters(2)
        .with_n_probes(2)
        .with_training_sample_size(64)
        .with_seed(7);
    let mut idx = IvfIndex::new(2, DistanceMetric::Euclidean, cfg)?;

    // Train on a representative sample of the data distribution.
    let sample: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0], vec![0.2, -0.1], vec![-0.1, 0.1],
        vec![10.0, 10.0], vec![10.2, 9.8], vec![9.9, 10.1],
    ];
    let sample_refs: Vec<&[f32]> = sample.iter().map(|v| v.as_slice()).collect();
    idx.train(&sample_refs)?;

    idx.insert(VectorId::from(1u64), Arc::<[f32]>::from(&[0.0, 0.0][..]), None)?;
    idx.insert(VectorId::from(2u64), Arc::<[f32]>::from(&[10.0, 10.0][..]), None)?;

    let hits = idx.search(&[0.0, 0.0], &SearchParams::new(1, DistanceMetric::Euclidean))?;
    assert_eq!(hits[0].id, VectorId::U64(1));
    Ok(())
}
```

The complete surface — every method, parameter, error, and more examples — is
in [`docs/API.md`](./docs/API.md).

<br>

## IVF-Flat vs IVF-PQ

IVF-Flat keeps the raw vectors and scores probed clusters exactly. IVF-PQ
compresses each vector to an `M`-byte Product-Quantization code and scores by
asymmetric distance (ADC); with a non-zero refine factor it shortlists by ADC
and exact-reranks against the retained vectors, recovering most of the accuracy
at a fraction of the scan cost and memory. Switch variants with one config flag
(Product Quantization needs at least 256 training vectors):

```rust
use iqdb_ivf::IvfConfig;

// IVF-PQ: 8 subvectors over a 128-D space, exact-rerank a 4×k shortlist.
let cfg = IvfConfig::default()
    .with_n_clusters(256)
    .with_n_probes(16)
    .with_use_pq(true)
    .with_pq_subvectors(Some(8))
    .with_pq_refine_factor(4);
assert!(cfg.validate().is_ok());
```

Supported IVF-PQ metrics are `Euclidean`, `DotProduct`, and `Manhattan`.

<br>

## Filtered search

A search can carry a `Filter`. IVF evaluates the predicate per entry *before*
computing distance, so a selective filter skips distance work proportionally.
A selective filter may return fewer than `k` hits within the probed clusters —
IVF will not widen the probe set to compensate, which is what keeps latency
predictable.

```rust
# use iqdb_ivf::IvfIndex;
# use iqdb_index::IndexCore;
use iqdb_types::{DistanceMetric, Filter, SearchParams, Value};

# fn demo(idx: &IvfIndex) -> iqdb_types::Result<()> {
let params = SearchParams {
    filter: Some(Filter::eq("tier", Value::String("hot".into()))),
    ..SearchParams::new(10, DistanceMetric::Euclidean)
};
let hits = idx.search(&[0.0, 0.0], &params)?;
# Ok(())
# }
```

<br>

## Tuning recall vs latency

`n_probes` is the dial: more probes raise recall and latency.
`suggest_n_probes(coverage)` reads the live cluster-size distribution and
recommends the smallest probe count covering the requested fraction of
vectors; apply it with `set_n_probes`:

```rust
# use iqdb_ivf::IvfIndex;
# fn demo(idx: &mut IvfIndex) -> iqdb_types::Result<()> {
let probes = idx.suggest_n_probes(0.90)?; // cover ~90% of vectors
idx.set_n_probes(probes)?;
# Ok(())
# }
```

When cluster balance drifts after many deletes or a distribution shift,
`cluster_stats()` reports the imbalance and `retrain()` rebuilds the centroids
(and PQ codebooks) over the currently-stored vectors without losing data.

<br>

## Tiered API

- **Tier 1 — the lazy path.** `IvfConfig::default()` + `IvfIndex::new` +
  `train` + the `IndexCore` insert/search calls.
- **Tier 2 — the configured path.** The `with_*` builders on `IvfConfig`, plus
  `set_n_probes`, `suggest_n_probes`, `set_pq_refine_factor`, `retrain`, and
  `cluster_stats` to tune and inspect a live index.
- **Tier 3 — the trait seam.** `IvfIndex` implements `iqdb_index::Index` and
  `iqdb_index::IndexCore`, so it is interchangeable with `iqdb-flat`,
  `iqdb-hnsw`, or any other backend behind those traits.

<br>

## Performance

- **Two-pass search** &mdash; Pass 1 ranks all centroids exactly; Pass 2 scans only the `n_probes` nearest clusters. Query cost is predictable and proportional to `n_clusters + n_probes · n/n_clusters`.
- **Distance** is delegated to [`iqdb-distance`](https://crates.io/crates/iqdb-distance), which dispatches to SIMD kernels (AVX2/NEON) where the target allows; IVF never reimplements a metric.
- **Top-`k`** uses a bounded max-heap of size `k` keyed by `(distance, sequence)` — `O(n log k)`, NaN-safe via `f32::total_cmp`.
- **Filter-first** &mdash; metadata filters run before distance, so selective predicates cut the distance workload.
- **IVF-PQ** builds the ADC lookup table once per query and reuses it across every probed cluster; the optional refine reranks an `N·k` shortlist exactly.
- **Zero-copy payloads** &mdash; inserted `Arc<[f32]>` values are stored verbatim, sharing one allocation with the caller's record store.

Benchmarks live in [`benches/ivf_bench.rs`](./benches/ivf_bench.rs) (`cargo bench`).

<br>

## Examples

Runnable end-to-end programs in [`examples/`](./examples):

| Example | Shows |
|---------|-------|
| [`quick_start`](./examples/quick_start.rs) | the configure → train → insert → search lifecycle |
| [`ivf_pq`](./examples/ivf_pq.rs) | IVF-PQ build, search, and runtime refine tuning |
| [`filtered_search`](./examples/filtered_search.rs) | metadata-filtered search |
| [`retrain`](./examples/retrain.rs) | rebuilding cluster balance after heavy deletes |
| [`probe_tuning`](./examples/probe_tuning.rs) | `suggest_n_probes` + `set_n_probes` + `cluster_stats` |
| [`recall_oracle`](./examples/recall_oracle.rs) | measuring recall@k against the exact `iqdb-flat` oracle |

```sh
cargo run --example quick_start
```

<br>

## Determinism

Given the same `IvfConfig::seed`, the same training sample, and the same
`n_clusters` / `training_sample_size`, training produces byte-identical
centroids on every supported platform — every PRNG draw comes from an in-tree
SplitMix64 generator, all reductions run in a fixed order, and centroid sums
accumulate in `f64` before a single downcast to `f32`. In IVF-PQ mode the same
seed flows into the codebook trainer, so search results are reproducible too.

<br>

## Status

`v1.0.0` is **stable**: k-means training, IVF-Flat and IVF-PQ search,
metadata-filtered search, `retrain`, probe tuning, and the full
`Index` / `IndexCore` trait implementation are committed under the SemVer 1.x
guarantee — no breaking changes until 2.0. The surface is covered by unit,
property-based, differential (against the exact `iqdb-flat` oracle), and
recall-at-scale tests, plus a runnable <a href="./examples"><code>examples/</code></a>
suite, and is recorded in the <a href="./dev/ROADMAP.md"><code>ROADMAP</code></a>.
Only additive, non-breaking changes are made within 1.x.

<hr>
<br>

## Where It Fits

`iqdb-ivf` is a Phase-3 index for large data. It builds on:

- `iqdb-types` &mdash; core types
- `iqdb-distance` &mdash; centroid + candidate distances
- `iqdb-index` &mdash; implements the `Index` / `IndexCore` traits
- `iqdb-filter` &mdash; metadata pre-filter evaluation
- `iqdb-quantize` &mdash; Product Quantization for IVF-PQ

<br>

## Standards

Built to the iQDB Rust standard. See <a href="./REPS.md"><code>REPS.md</code></a> (Rust Efficiency &amp; Performance Standards) and <a href="./dev/DIRECTIVES.md"><code>dev/DIRECTIVES.md</code></a> for the engineering law and the definition of done. Before a PR: `cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, and `cargo test --all-features` must be clean.

<br>

<div id="license">
    <h2>License</h2>
    <p>Licensed under either of</p>
    <ul>
        <li><b>Apache License, Version 2.0</b> &mdash; <a href="./LICENSE-APACHE">LICENSE-APACHE</a></li>
        <li><b>MIT License</b> &mdash; <a href="./LICENSE-MIT">LICENSE-MIT</a></li>
    </ul>
    <p>at your option.</p>
</div>

<div align="center">
  <h2></h2>
  <sup>COPYRIGHT <small>&copy;</small> 2026 <strong>JAMES GOBER.</strong></sup>
</div>
