//! `n_clusters > sample.len()` is the brief's "graceful, no panic"
//! path. Confirms `train` returns `InvalidConfig`, the index stays
//! untrained, and a subsequent attempt with a sufficient sample
//! succeeds.
//!
//! Cargo integration-test binary; named in the approved plan
//! §"Test plan". User brief: *"Handle n_clusters > sample size
//! gracefully (no panic)."*

#![allow(clippy::unwrap_used)]

use iqdb_index::Index;
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, IqdbError};

#[test]
fn small_sample_rejected_and_index_stays_untrained() {
    let cfg = IvfConfig::default()
        .with_n_clusters(10)
        .with_n_probes(4)
        .with_training_sample_size(64)
        .with_seed(1);
    let mut idx = IvfIndex::new(2, DistanceMetric::Euclidean, cfg).unwrap();
    assert!(!idx.is_trained());

    let too_few: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32, (i * 2) as f32]).collect();
    let refs: Vec<&[f32]> = too_few.iter().map(|v| v.as_slice()).collect();

    let err = idx.train(&refs).unwrap_err();
    match err {
        IqdbError::InvalidConfig { reason } => {
            assert!(reason.contains("sample.len() >= n_clusters"));
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }

    assert!(
        !idx.is_trained(),
        "failed train must leave the index untrained"
    );

    // A sufficient sample on the *same* index then trains cleanly.
    let plenty: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32, (i * 2) as f32]).collect();
    let plenty_refs: Vec<&[f32]> = plenty.iter().map(|v| v.as_slice()).collect();
    idx.train(&plenty_refs).unwrap();
    assert!(idx.is_trained());
}

#[test]
fn pq_training_sample_smaller_than_pq_k_returns_clear_error() {
    // IVF-PQ training trains a `ProductQuantizer` with K = 256 on
    // the same working set as the coarse k-means. A sample of 100
    // satisfies the coarse k-means (n_clusters = 4) but is below
    // PQ's `training_set.len() >= n_centroids` requirement (K=256),
    // so train must surface the quantizer's `InvalidConfig` cleanly.
    let cfg = IvfConfig::default()
        .with_n_clusters(4)
        .with_n_probes(4)
        .with_training_sample_size(256)
        .with_use_pq(true)
        .with_pq_subvectors(Some(2))
        .with_seed(1);
    let mut idx = IvfIndex::new(4, DistanceMetric::Euclidean, cfg).unwrap();
    let small: Vec<Vec<f32>> = (0..100).map(|i| vec![i as f32, 0.0, 0.0, 0.0]).collect();
    let refs: Vec<&[f32]> = small.iter().map(|v| v.as_slice()).collect();

    let err = idx.train(&refs).unwrap_err();
    match err {
        IqdbError::InvalidConfig { reason } => {
            assert!(
                reason.contains("n_centroids"),
                "expected PQ-sample-size error mentioning n_centroids; got: {reason}"
            );
        }
        other => panic!("expected InvalidConfig from PQ trainer, got {other:?}"),
    }
    assert!(
        !idx.is_trained(),
        "failed PQ train must leave the index untrained"
    );
}
