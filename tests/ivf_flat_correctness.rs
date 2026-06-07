//! End-to-end correctness against [`iqdb_flat::FlatIndex`], the exact
//! search oracle. With `n_probes == n_clusters` IVF must visit every
//! cluster and therefore agree with flat on the full top-`k` set.
//! With `n_probes < n_clusters` the recall@k should still be
//! materially non-trivial on well-clustered data.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use iqdb_flat::{FlatConfig, FlatIndex};
use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, SearchParams, VectorId};

/// Build a 2-D dataset of 4 well-separated clusters.
fn dataset() -> Vec<Vec<f32>> {
    let centres = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0)];
    let mut out = Vec::with_capacity(centres.len() * 50);
    for (cx, cy) in centres {
        for i in 0..50 {
            // Spread points around each centre in a small jitter; no
            // RNG needed — sequential math is enough for separation.
            let f = i as f32;
            let dx = (f * 0.11).sin() * 0.5;
            let dy = (f * 0.17).cos() * 0.5;
            out.push(vec![cx + dx, cy + dy]);
        }
    }
    out
}

fn populate_flat(data: &[Vec<f32>], metric: DistanceMetric) -> FlatIndex {
    let mut idx = FlatIndex::new(2, metric, FlatConfig).unwrap();
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

fn populate_ivf(data: &[Vec<f32>], n_probes: usize, metric: DistanceMetric) -> IvfIndex {
    let cfg = IvfConfig::default()
        .with_n_clusters(4)
        .with_n_probes(n_probes)
        .with_training_sample_size(1024)
        .with_seed(7);
    let mut idx = IvfIndex::new(2, metric, cfg).unwrap();
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
fn full_probe_matches_flat_top_k() {
    let data = dataset();
    let metric = DistanceMetric::Euclidean;
    let flat = populate_flat(&data, metric);
    let ivf = populate_ivf(&data, 4, metric); // n_probes == n_clusters

    let params = SearchParams::new(10, metric);
    for q in [&data[0], &data[60], &data[120], &data[180]] {
        let hits_flat = flat.search(q, &params).unwrap();
        let hits_ivf = ivf.search(q, &params).unwrap();
        let ids_flat: Vec<_> = hits_flat.iter().map(|h| h.id.clone()).collect();
        let ids_ivf: Vec<_> = hits_ivf.iter().map(|h| h.id.clone()).collect();
        assert_eq!(ids_flat, ids_ivf, "full-probe IVF must match flat exactly");
    }
}

#[test]
fn small_probe_recall_is_non_trivial() {
    let data = dataset();
    let metric = DistanceMetric::Euclidean;
    let flat = populate_flat(&data, metric);
    let ivf = populate_ivf(&data, 1, metric); // probe only the nearest cluster

    let params = SearchParams::new(10, metric);
    let mut hits = 0;
    let mut total = 0;
    for q in [&data[10], &data[60], &data[110], &data[170]] {
        let flat_ids: std::collections::HashSet<_> = flat
            .search(q, &params)
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        let ivf_ids: std::collections::HashSet<_> = ivf
            .search(q, &params)
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        hits += flat_ids.intersection(&ivf_ids).count();
        total += flat_ids.len();
    }
    let recall = (hits as f32) / (total as f32);
    assert!(
        recall >= 0.5,
        "recall@10 = {recall} should be >= 0.5 on this well-clustered data"
    );
}

#[test]
fn empty_search_returns_empty() {
    let data = dataset();
    let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
    let cfg = IvfConfig::default()
        .with_n_clusters(4)
        .with_n_probes(4)
        .with_training_sample_size(64)
        .with_seed(1);
    let mut idx = IvfIndex::new(2, DistanceMetric::Euclidean, cfg).unwrap();
    idx.train(&refs).unwrap();
    // No inserts.
    let hits = idx
        .search(
            &[0.0, 0.0],
            &SearchParams::new(5, DistanceMetric::Euclidean),
        )
        .unwrap();
    assert!(hits.is_empty());
}
