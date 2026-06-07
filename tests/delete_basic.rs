//! Delete removes the id from its cluster's inverted list, decrements
//! `len()`, and disqualifies the id from subsequent searches.
//!
//! Cargo discovers this file as an integration-test binary. Delete is
//! a cluster-local `swap_remove`; rebalancing the inverted lists after
//! a delete-heavy phase is the job of `IvfIndex::retrain`, not of
//! `delete` itself.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, IqdbError, SearchParams, VectorId};

fn populate(n: usize) -> IvfIndex {
    let data: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            let f = i as f32;
            vec![f * 0.01, ((f + 1.0) * 0.013).sin()]
        })
        .collect();
    let cfg = IvfConfig::default()
        .with_n_clusters(4)
        .with_n_probes(4)
        .with_training_sample_size(256)
        .with_seed(11);
    let mut idx = IvfIndex::new(2, DistanceMetric::Euclidean, cfg).unwrap();
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
fn deleted_ids_are_absent_from_search() {
    let mut idx = populate(100);
    for i in [3u64, 27, 42, 75] {
        idx.delete(&VectorId::from(i)).unwrap();
    }
    assert_eq!(idx.len(), 96);

    let hits = idx
        .search(
            &[0.0, 0.0],
            &SearchParams::new(100, DistanceMetric::Euclidean),
        )
        .unwrap();
    let hit_ids: std::collections::HashSet<VectorId> = hits.iter().map(|h| h.id.clone()).collect();
    for i in [3u64, 27, 42, 75] {
        assert!(
            !hit_ids.contains(&VectorId::from(i)),
            "deleted id {i} should not appear in search results"
        );
    }
}

#[test]
fn double_delete_returns_not_found() {
    let mut idx = populate(20);
    idx.delete(&VectorId::from(5u64)).unwrap();
    let err = idx.delete(&VectorId::from(5u64)).unwrap_err();
    assert!(matches!(err, IqdbError::NotFound));
}

#[test]
fn delete_before_train_returns_not_found() {
    let cfg = IvfConfig::default()
        .with_n_clusters(2)
        .with_n_probes(2)
        .with_training_sample_size(16)
        .with_seed(1);
    let mut idx = IvfIndex::new(2, DistanceMetric::Euclidean, cfg).unwrap();
    let err = idx.delete(&VectorId::from(99u64)).unwrap_err();
    assert!(matches!(err, IqdbError::NotFound));
}
