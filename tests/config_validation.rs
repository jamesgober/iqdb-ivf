//! Validation surface on `IvfIndex::new`: covers zero `dim`, invalid
//! `IvfConfig` fields, and the live IVF-PQ guard rails
//! (`use_pq=true` requires `pq_subvectors=Some(m)` with `m | dim`;
//! Cosine and Hamming are rejected with `InvalidMetric`).

#![allow(clippy::unwrap_used)]

use iqdb_index::Index;
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, IqdbError};

fn new_with(cfg: IvfConfig) -> Result<IvfIndex, IqdbError> {
    IvfIndex::new(2, DistanceMetric::Euclidean, cfg)
}

fn new_with_dim_metric(
    dim: usize,
    metric: DistanceMetric,
    cfg: IvfConfig,
) -> Result<IvfIndex, IqdbError> {
    IvfIndex::new(dim, metric, cfg)
}

#[test]
fn zero_dim_rejected() {
    let err = IvfIndex::new(0, DistanceMetric::Euclidean, IvfConfig::default()).unwrap_err();
    match err {
        IqdbError::InvalidConfig { reason } => {
            assert!(reason.contains("dim"));
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

#[test]
fn zero_n_clusters_rejected() {
    let err = new_with(IvfConfig::default().with_n_clusters(0)).unwrap_err();
    assert!(matches!(
        err,
        IqdbError::InvalidConfig { reason } if reason.contains("n_clusters")
    ));
}

#[test]
fn zero_n_probes_rejected() {
    let err = new_with(IvfConfig::default().with_n_probes(0)).unwrap_err();
    assert!(matches!(
        err,
        IqdbError::InvalidConfig { reason } if reason.contains("n_probes")
    ));
}

#[test]
fn n_probes_exceeding_n_clusters_rejected() {
    let err = new_with(IvfConfig::default().with_n_clusters(4).with_n_probes(8)).unwrap_err();
    assert!(matches!(
        err,
        IqdbError::InvalidConfig { reason } if reason.contains("n_probes")
    ));
}

#[test]
fn use_pq_true_without_pq_subvectors_rejected() {
    let err = new_with(IvfConfig::default().with_use_pq(true)).unwrap_err();
    match err {
        IqdbError::InvalidConfig { reason } => {
            assert!(reason.contains("pq_subvectors"));
            assert!(reason.contains("Some"));
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

#[test]
fn use_pq_true_with_zero_pq_subvectors_rejected() {
    let err = new_with(
        IvfConfig::default()
            .with_use_pq(true)
            .with_pq_subvectors(Some(0)),
    )
    .unwrap_err();
    assert!(matches!(
        err,
        IqdbError::InvalidConfig { reason } if reason.contains("pq_subvectors")
    ));
}

#[test]
fn use_pq_true_with_indivisible_pq_subvectors_rejected() {
    // dim = 6, m = 4: 6 % 4 != 0 → reject at IvfIndex::new_unconfigured
    // (where dim is known).
    let err = new_with_dim_metric(
        6,
        DistanceMetric::Euclidean,
        IvfConfig::default()
            .with_use_pq(true)
            .with_pq_subvectors(Some(4)),
    )
    .unwrap_err();
    match err {
        IqdbError::InvalidConfig { reason } => {
            assert!(reason.contains("pq_subvectors"));
            assert!(reason.contains("divide"));
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}

#[test]
fn use_pq_true_with_cosine_rejected() {
    let err = new_with_dim_metric(
        4,
        DistanceMetric::Cosine,
        IvfConfig::default()
            .with_use_pq(true)
            .with_pq_subvectors(Some(2)),
    )
    .unwrap_err();
    assert!(matches!(err, IqdbError::InvalidMetric));
}

#[test]
fn use_pq_true_with_hamming_rejected() {
    let err = new_with_dim_metric(
        4,
        DistanceMetric::Hamming,
        IvfConfig::default()
            .with_use_pq(true)
            .with_pq_subvectors(Some(2)),
    )
    .unwrap_err();
    assert!(matches!(err, IqdbError::InvalidMetric));
}

#[test]
fn use_pq_true_with_valid_config_accepted() {
    // dim = 4, m = 2, Euclidean → all guards pass.
    assert!(
        new_with_dim_metric(
            4,
            DistanceMetric::Euclidean,
            IvfConfig::default()
                .with_use_pq(true)
                .with_pq_subvectors(Some(2)),
        )
        .is_ok()
    );
}

#[test]
fn pq_refine_factor_zero_is_valid() {
    let cfg = IvfConfig::default()
        .with_use_pq(true)
        .with_pq_subvectors(Some(2))
        .with_pq_refine_factor(0);
    assert!(new_with_dim_metric(4, DistanceMetric::Euclidean, cfg).is_ok());
}

#[test]
fn set_pq_refine_factor_updates_value() {
    let cfg = IvfConfig::default()
        .with_use_pq(true)
        .with_pq_subvectors(Some(2))
        .with_pq_refine_factor(4);
    let mut idx = new_with_dim_metric(4, DistanceMetric::Euclidean, cfg).unwrap();
    assert_eq!(idx.pq_refine_factor(), 4);
    idx.set_pq_refine_factor(0);
    assert_eq!(idx.pq_refine_factor(), 0);
    idx.set_pq_refine_factor(8);
    assert_eq!(idx.pq_refine_factor(), 8);
}

#[test]
fn zero_training_sample_size_rejected() {
    let err = new_with(IvfConfig::default().with_training_sample_size(0)).unwrap_err();
    assert!(matches!(
        err,
        IqdbError::InvalidConfig { reason } if reason.contains("training_sample_size")
    ));
}

#[test]
fn valid_default_config_accepted() {
    assert!(new_with(IvfConfig::default()).is_ok());
}

#[test]
fn set_n_probes_rejects_zero() {
    let cfg = IvfConfig::default().with_n_clusters(8).with_n_probes(2);
    let mut idx = new_with(cfg).unwrap();
    let err = idx.set_n_probes(0).unwrap_err();
    assert!(matches!(
        err,
        IqdbError::InvalidConfig { reason } if reason.contains("n >= 1")
    ));
}

#[test]
fn set_n_probes_rejects_above_n_clusters() {
    let cfg = IvfConfig::default().with_n_clusters(4).with_n_probes(2);
    let mut idx = new_with(cfg).unwrap();
    let err = idx.set_n_probes(5).unwrap_err();
    assert!(matches!(
        err,
        IqdbError::InvalidConfig { reason } if reason.contains("n_clusters")
    ));
}

#[test]
fn set_n_probes_updates_value_on_success() {
    let cfg = IvfConfig::default().with_n_clusters(4).with_n_probes(2);
    let mut idx = new_with(cfg).unwrap();
    idx.set_n_probes(3).unwrap();
    assert_eq!(idx.n_probes(), 3);
}
