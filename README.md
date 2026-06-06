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
        <strong>iqdb-ivf</strong> partitions the vector space into clusters and searches only the most relevant ones. It is the complement to HNSW: more memory-efficient at very large scale and with more predictable latency.
    </p>
    <p>
        It ships IVF-Flat first; IVF-PQ (quantized within clusters) layers on `iqdb-quantize`.
    </p>
    <br>
    <hr>
    <p>
        <strong>MSRV is 1.87+</strong> (Rust 2024 edition). Clustered search. IVF-Flat and IVF-PQ. Predictable latency at scale.
    </p>
    <blockquote>
        <strong>Status: pre-1.0, in active development.</strong> The public API is being designed across the 0.x series and frozen at <code>1.0.0</code>. See <a href="./CHANGELOG.md"><code>CHANGELOG.md</code></a>.
    </blockquote>
</div>

<hr>
<br>

<h2>What it does</h2>

- **Inverted file index** &mdash; k-means partitions the space; search only the most relevant clusters
- **IVF-Flat and IVF-PQ** &mdash; store vectors as-is, or quantized within clusters via iqdb-quantize
- **Predictable latency** &mdash; probe a fixed number of clusters; deterministic query cost
- **Huge-scale profile** &mdash; more memory-efficient than HNSW at 100M+ vectors
- **Trainable + retrainable** &mdash; rebuild clusters as the distribution drifts without losing data


<br>

## Installation

```toml
[dependencies]
iqdb-ivf = "0.1"
```

<br>

## Status

This is the <code>v0.1.0</code> scaffold: structure, tooling, and quality gates are in place; the implementation lands across the 0.x series per the <a href="./dev/ROADMAP.md"><code>ROADMAP</code></a> and <a href="./docs/API.md"><code>docs/API.md</code></a>.

<hr>
<br>

## Where It Fits

`iqdb-ivf` is a Phase-3 index for large data. It builds on:

- `iqdb-types` &mdash; core types
- `iqdb-distance` &mdash; centroid + candidate distances
- `iqdb-index` &mdash; implements the trait
- `iqdb-filter` &mdash; filtered cluster search
- `iqdb-quantize` &mdash; optional, for IVF-PQ

It is unblocked once index/distance/filter exist; quantize is optional.

<br>

## Contributing

See <a href="./dev/DIRECTIVES.md"><code>dev/DIRECTIVES.md</code></a> for engineering standards and the definition of done. Before a PR: `cargo fmt --all`, `cargo clippy --all-targets --all-features -- -D warnings`, and `cargo test --all-features` must be clean.

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
