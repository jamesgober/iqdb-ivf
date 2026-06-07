//! K-means determinism contract.
//!
//! The same `seed` + same training sample MUST produce byte-identical
//! `cluster_stats` and identical post-training queries. Different
//! seeds on the same data MAY diverge (and on this synthetic dataset
//! actually do).

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, SearchParams, VectorId};

fn deterministic_sample(n: usize) -> Vec<Vec<f32>> {
    // Sequential, non-random data — no platform-specific RNG involved.
    (0..n)
        .map(|i| {
            let f = i as f32;
            vec![f * 0.013, (f * 0.027).sin(), (f * 0.005).cos()]
        })
        .collect()
}

/// Build, train, and populate the index with the same dataset so
/// `cluster_stats` reflects the trained centroid positions through
/// post-insert occupancy. Training alone leaves inverted lists
/// empty, so the cluster-size signal only surfaces after inserts.
fn build_and_populate(cfg: IvfConfig, dim: usize, data: &[Vec<f32>]) -> IvfIndex {
    let mut idx = IvfIndex::new(dim, DistanceMetric::Euclidean, cfg).unwrap();
    let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
    idx.train(&refs).unwrap();
    for (i, v) in data.iter().enumerate() {
        idx.insert(
            VectorId::from(i as u64),
            Arc::<[f32]>::from(v.as_slice()),
            None,
        )
        .unwrap();
    }
    idx
}

#[test]
fn same_seed_same_sample_yields_identical_cluster_stats() {
    let data = deterministic_sample(200);
    let cfg = IvfConfig::default()
        .with_n_clusters(8)
        .with_n_probes(2)
        .with_training_sample_size(200)
        .with_seed(0xC0FF_EEEE);

    let a = build_and_populate(cfg, 3, &data);
    let b = build_and_populate(cfg, 3, &data);

    let stats_a = a.cluster_stats();
    let stats_b = b.cluster_stats();
    assert_eq!(
        stats_a, stats_b,
        "deterministic train must yield equal stats"
    );
}

#[test]
fn same_seed_same_sample_yields_identical_search_results() {
    let data = deterministic_sample(200);
    let cfg = IvfConfig::default()
        .with_n_clusters(8)
        .with_n_probes(3)
        .with_training_sample_size(200)
        .with_seed(0xC0FF_EEEE);

    let a = build_and_populate(cfg, 3, &data);
    let b = build_and_populate(cfg, 3, &data);

    let params = SearchParams::new(10, DistanceMetric::Euclidean);
    for q in [&data[0], &data[100], &data[199]] {
        let hits_a = a.search(q, &params).unwrap();
        let hits_b = b.search(q, &params).unwrap();
        let ids_a: Vec<_> = hits_a.iter().map(|h| h.id.clone()).collect();
        let ids_b: Vec<_> = hits_b.iter().map(|h| h.id.clone()).collect();
        assert_eq!(
            ids_a, ids_b,
            "identical seeds + data + query → identical hits"
        );
    }
}

#[test]
fn different_seeds_can_diverge() {
    let data = deterministic_sample(200);
    let cfg_a = IvfConfig::default()
        .with_n_clusters(8)
        .with_n_probes(2)
        .with_training_sample_size(200)
        .with_seed(1);
    let cfg_b = IvfConfig::default()
        .with_n_clusters(8)
        .with_n_probes(2)
        .with_training_sample_size(200)
        .with_seed(2);

    let a = build_and_populate(cfg_a, 3, &data);
    let b = build_and_populate(cfg_b, 3, &data);

    // Not a hard requirement (some datasets converge to a unique
    // optimum regardless of seed), but on this sample two distinct
    // seeds reliably produce measurably different cluster size
    // distributions.
    let sa = a.cluster_stats();
    let sb = b.cluster_stats();
    assert_ne!(
        sa.cluster_sizes, sb.cluster_sizes,
        "seeds 1 and 2 should produce different partitions on this dataset"
    );
}

#[test]
fn subsampling_path_is_deterministic() {
    // training_sample_size < sample.len() exercises the partial
    // Fisher–Yates path; that path is the most platform-sensitive
    // step in train.rs.
    let data = deterministic_sample(500);
    let cfg = IvfConfig::default()
        .with_n_clusters(16)
        .with_n_probes(2)
        .with_training_sample_size(100)
        .with_seed(99);

    let a = build_and_populate(cfg, 3, &data);
    let b = build_and_populate(cfg, 3, &data);

    assert_eq!(a.cluster_stats(), b.cluster_stats());
}
