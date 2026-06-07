//! `IvfIndex::suggest_n_probes` — probe-count recommendation contract.
//!
//! Pins the documented properties of the pure suggestion function:
//!
//! - monotone in `coverage` (higher coverage → at least as many
//!   probes);
//! - the suggested probes cover at least the requested fraction of
//!   the live vector count;
//! - the documented edges (`0.0 → 1`, `1.0 → n_clusters`, single
//!   cluster → `1`, untrained → `1`);
//! - out-of-range and NaN `coverage` surface as `InvalidConfig`;
//! - composing with `set_n_probes` produces a searchable index.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, IqdbError, SearchParams, VectorId};

const DIM: usize = 4;

/// Build a populated `IvfIndex` with deliberately *skewed* cluster
/// sizes so the cumulative-coverage walk is non-trivial. We place
/// most vectors near one centre and a long tail near three others;
/// the largest cluster ends up holding the bulk of the live count.
fn skewed_index() -> IvfIndex {
    let cfg = IvfConfig::default()
        .with_n_clusters(4)
        .with_n_probes(4)
        .with_training_sample_size(256)
        .with_seed(0xBAD_F00D);
    let mut idx = IvfIndex::new(DIM, DistanceMetric::Euclidean, cfg).unwrap();

    // Centre 0 gets 64 vectors; centres 1-3 get 4 each. The
    // training sample contains representatives from every centre
    // so k-means produces 4 distinct partitions.
    let centres: Vec<Vec<f32>> = (0..4)
        .map(|c| (0..DIM).map(|i| c as f32 * 5.0 + i as f32 * 0.1).collect())
        .collect();

    let mut data: Vec<Vec<f32>> = Vec::new();
    for i in 0..64 {
        let jitter = (i as f32) * 0.001;
        let v: Vec<f32> = centres[0].iter().map(|c| c + jitter).collect();
        data.push(v);
    }
    for centre in centres.iter().skip(1) {
        for i in 0..4 {
            let jitter = (i as f32) * 0.001;
            let v: Vec<f32> = centre.iter().map(|x| x + jitter).collect();
            data.push(v);
        }
    }
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

fn single_cluster_index() -> IvfIndex {
    let cfg = IvfConfig::default()
        .with_n_clusters(1)
        .with_n_probes(1)
        .with_training_sample_size(8)
        .with_seed(42);
    let mut idx = IvfIndex::new(DIM, DistanceMetric::Euclidean, cfg).unwrap();
    let data: Vec<Vec<f32>> = (0..8)
        .map(|i| (0..DIM).map(|j| (i + j) as f32 * 0.1).collect())
        .collect();
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
fn suggest_is_monotonic_in_coverage() {
    let idx = skewed_index();
    let mut last: usize = 0;
    for step in 0..=10 {
        let coverage = step as f32 * 0.1;
        let n = idx.suggest_n_probes(coverage).unwrap();
        assert!(
            n >= last,
            "suggest is non-monotonic at coverage = {coverage}: {n} < {last}",
        );
        last = n;
    }
}

#[test]
fn suggest_covers_requested_fraction() {
    let idx = skewed_index();
    let stats = idx.cluster_stats();
    let live: usize = stats.cluster_sizes.iter().sum();
    let mut sorted = stats.cluster_sizes.clone();
    sorted.sort_by(|a, b| b.cmp(a));

    for &coverage in &[0.25_f32, 0.5, 0.75, 0.9] {
        let n = idx.suggest_n_probes(coverage).unwrap();
        let cumsum: usize = sorted.iter().take(n).sum();
        let target = ((live as f64) * (coverage as f64)).ceil() as usize;
        assert!(
            cumsum >= target,
            "coverage {coverage}: suggested {n} probes cover {cumsum} < target {target}",
        );
    }
}

#[test]
fn suggest_edge_zero_returns_one() {
    let idx = skewed_index();
    assert_eq!(idx.suggest_n_probes(0.0).unwrap(), 1);
}

#[test]
fn suggest_edge_one_returns_n_clusters() {
    let idx = skewed_index();
    assert_eq!(
        idx.suggest_n_probes(1.0).unwrap(),
        idx.cluster_stats().n_clusters
    );
}

#[test]
fn suggest_on_single_cluster_index_always_returns_one() {
    let idx = single_cluster_index();
    for &c in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
        assert_eq!(idx.suggest_n_probes(c).unwrap(), 1, "coverage = {c}");
    }
}

#[test]
fn suggest_rejects_out_of_range() {
    let idx = skewed_index();
    for &bad in &[-0.1_f32, 1.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let err = idx.suggest_n_probes(bad).unwrap_err();
        assert!(
            matches!(err, IqdbError::InvalidConfig { reason } if reason.contains("coverage")),
            "coverage {bad} should be InvalidConfig, got {err:?}",
        );
    }
}

#[test]
fn suggest_then_set_n_probes_composes() {
    let mut idx = skewed_index();
    let n = idx.suggest_n_probes(0.8).unwrap();
    assert!(n >= 1 && n <= idx.cluster_stats().n_clusters);
    idx.set_n_probes(n).unwrap();
    let params = SearchParams::new(4, DistanceMetric::Euclidean);
    let query: Vec<f32> = (0..DIM).map(|i| i as f32 * 0.1).collect();
    let hits = idx.search(&query, &params).unwrap();
    assert_eq!(hits.len(), 4);
}

#[test]
fn suggest_on_untrained_returns_one() {
    let cfg = IvfConfig::default()
        .with_n_clusters(4)
        .with_n_probes(2)
        .with_training_sample_size(16)
        .with_seed(9);
    let idx = IvfIndex::new(DIM, DistanceMetric::Euclidean, cfg).unwrap();
    assert!(!idx.is_trained());
    for &c in &[0.0_f32, 0.5, 1.0] {
        assert_eq!(idx.suggest_n_probes(c).unwrap(), 1, "coverage = {c}");
    }
}
