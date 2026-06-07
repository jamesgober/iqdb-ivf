//! Criterion benches for `IvfIndex` hot paths.
//!
//! Covers the costs that dominate IVF: the two-pass IVF-Flat search at a
//! realistic probe count, the same search probing every cluster, the
//! IVF-PQ ADC search with exact refine, the metadata-filtered scan, and
//! k-means training. Data is deterministic (clustered, seeded from the
//! row index) so a second run reproduces the baseline.

use std::hint::black_box;
use std::sync::Arc;

use criterion::{Criterion, criterion_group, criterion_main};
use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, Filter, Metadata, SearchParams, Value, VectorId};

const METRIC: DistanceMetric = DistanceMetric::Euclidean;

fn arc(v: &[f32]) -> Arc<[f32]> {
    Arc::from(v)
}

/// Deterministic clustered row over `n_centres` latent centres.
fn row(i: usize, dim: usize, n_centres: u64) -> Vec<f32> {
    let c = (i as u64) % n_centres;
    (0..dim)
        .map(|j| {
            let base = (c
                .wrapping_mul(2_654_435_761)
                .wrapping_add(j as u64 * 40_503)
                % 1_000) as f32
                / 1_000.0
                * 20.0;
            let jitter = (((i * 131 + j * 71) % 100) as f32 - 50.0) * 0.01;
            base + jitter
        })
        .collect()
}

fn query(dim: usize) -> Vec<f32> {
    (0..dim).map(|j| (j as f32).cos() * 5.0 + 5.0).collect()
}

fn build(
    n: usize,
    dim: usize,
    n_clusters: usize,
    n_probes: usize,
    use_pq: bool,
    with_meta: bool,
) -> IvfIndex {
    let mut cfg = IvfConfig::default()
        .with_n_clusters(n_clusters)
        .with_n_probes(n_probes)
        .with_training_sample_size(8192)
        .with_seed(0x51F7);
    if use_pq {
        cfg = cfg
            .with_use_pq(true)
            .with_pq_subvectors(Some(dim / 8))
            .with_pq_refine_factor(4);
    }
    let mut idx = IvfIndex::new(dim, METRIC, cfg).expect("valid config");
    let rows: Vec<Vec<f32>> = (0..n).map(|i| row(i, dim, n_clusters as u64)).collect();
    let refs: Vec<&[f32]> = rows.iter().map(|v| v.as_slice()).collect();
    idx.train(&refs).expect("train succeeds");
    for (i, v) in rows.iter().enumerate() {
        let meta = if with_meta {
            let tier = if i % 5 == 0 { "hot" } else { "cold" };
            let m: Metadata = [("tier".to_string(), Value::String(tier.into()))]
                .into_iter()
                .collect();
            Some(m)
        } else {
            None
        };
        idx.insert(VectorId::from(i as u64), arc(v), meta)
            .expect("fresh id");
    }
    idx
}

fn bench_search_flat(c: &mut Criterion) {
    let (n, dim, n_clusters) = (50_000_usize, 128_usize, 256_usize);
    let query = query(dim);
    let params = SearchParams::new(10, METRIC);
    for n_probes in [8_usize, 32, 256] {
        let idx = build(n, dim, n_clusters, n_probes, false, false);
        let name = format!("ivf/search_flat/n{n}/d{dim}/c{n_clusters}/p{n_probes}/k10");
        let _ = c.bench_function(&name, |b| {
            b.iter(|| idx.search(black_box(&query), black_box(&params)));
        });
    }
}

fn bench_search_pq(c: &mut Criterion) {
    let (n, dim, n_clusters, n_probes) = (50_000_usize, 128_usize, 256_usize, 16_usize);
    let idx = build(n, dim, n_clusters, n_probes, true, false);
    let query = query(dim);
    let params = SearchParams::new(10, METRIC);
    let name = format!("ivf/search_pq/n{n}/d{dim}/c{n_clusters}/p{n_probes}/refine4/k10");
    let _ = c.bench_function(&name, |b| {
        b.iter(|| idx.search(black_box(&query), black_box(&params)));
    });
}

fn bench_search_filtered(c: &mut Criterion) {
    let (n, dim, n_clusters, n_probes) = (50_000_usize, 128_usize, 256_usize, 32_usize);
    let idx = build(n, dim, n_clusters, n_probes, false, true);
    let query = query(dim);
    let params = SearchParams {
        filter: Some(Filter::eq("tier", Value::String("hot".into()))),
        ..SearchParams::new(10, METRIC)
    };
    let name = format!("ivf/search_filtered/n{n}/d{dim}/c{n_clusters}/p{n_probes}/sel20/k10");
    let _ = c.bench_function(&name, |b| {
        b.iter(|| idx.search(black_box(&query), black_box(&params)));
    });
}

fn bench_train(c: &mut Criterion) {
    let (n, dim, n_clusters) = (10_000_usize, 64_usize, 64_usize);
    let rows: Vec<Vec<f32>> = (0..n).map(|i| row(i, dim, n_clusters as u64)).collect();
    let refs: Vec<&[f32]> = rows.iter().map(|v| v.as_slice()).collect();
    let cfg = IvfConfig::default()
        .with_n_clusters(n_clusters)
        .with_n_probes(8)
        .with_training_sample_size(8192)
        .with_seed(0x51F7);
    let name = format!("ivf/train/n{n}/d{dim}/c{n_clusters}");
    let _ = c.bench_function(&name, |b| {
        b.iter(|| {
            let mut idx = IvfIndex::new(dim, METRIC, cfg).expect("valid config");
            idx.train(black_box(&refs)).expect("train succeeds");
            idx
        });
    });
}

criterion_group!(
    benches,
    bench_search_flat,
    bench_search_pq,
    bench_search_filtered,
    bench_train
);
criterion_main!(benches);
