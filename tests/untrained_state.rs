//! Verifies that the four trained-only entry points short-circuit with
//! `IqdbError::InvalidConfig` before `train` is called, and that they
//! succeed after a valid `train`.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, IqdbError, SearchParams, VectorId};

fn make_index() -> IvfIndex {
    let cfg = IvfConfig::default()
        .with_n_clusters(2)
        .with_n_probes(2)
        .with_training_sample_size(16)
        .with_seed(1);
    IvfIndex::new(2, DistanceMetric::Euclidean, cfg).unwrap()
}

fn make_sample() -> Vec<Vec<f32>> {
    vec![
        vec![0.0, 0.0],
        vec![0.1, -0.1],
        vec![-0.1, 0.1],
        vec![10.0, 10.0],
        vec![10.1, 9.9],
        vec![9.9, 10.1],
    ]
}

fn assert_untrained(err: IqdbError) {
    match err {
        IqdbError::InvalidConfig { reason } => {
            assert!(
                reason.contains("trained"),
                "reason does not mention training: {reason}"
            );
        }
        other => panic!("expected InvalidConfig (untrained), got {other:?}"),
    }
}

#[test]
fn insert_before_train_fails() {
    let mut idx = make_index();
    let err = idx
        .insert(
            VectorId::from(1u64),
            Arc::<[f32]>::from(&[0.0, 0.0][..]),
            None,
        )
        .unwrap_err();
    assert_untrained(err);
}

#[test]
fn insert_batch_before_train_fails() {
    let mut idx = make_index();
    let items = vec![(
        VectorId::from(1u64),
        Arc::<[f32]>::from(&[0.0, 0.0][..]),
        None,
    )];
    let err = idx.insert_batch(items).unwrap_err();
    assert_untrained(err);
}

#[test]
fn search_before_train_fails() {
    let idx = make_index();
    let err = idx
        .search(
            &[0.0, 0.0],
            &SearchParams::new(1, DistanceMetric::Euclidean),
        )
        .unwrap_err();
    assert_untrained(err);
}

#[test]
fn search_batch_before_train_fails() {
    let idx = make_index();
    let q: &[f32] = &[0.0, 0.0];
    let err = idx
        .search_batch(&[q], &SearchParams::new(1, DistanceMetric::Euclidean))
        .unwrap_err();
    assert_untrained(err);
}

#[test]
fn after_train_inserts_and_searches_succeed() {
    let mut idx = make_index();
    let sample = make_sample();
    let sample_refs: Vec<&[f32]> = sample.iter().map(|v| v.as_slice()).collect();

    idx.train(&sample_refs).unwrap();
    assert!(idx.is_trained());

    idx.insert(
        VectorId::from(1u64),
        Arc::<[f32]>::from(&[0.0, 0.0][..]),
        None,
    )
    .unwrap();
    idx.insert(
        VectorId::from(2u64),
        Arc::<[f32]>::from(&[10.0, 10.0][..]),
        None,
    )
    .unwrap();

    let hits = idx
        .search(
            &[0.0, 0.0],
            &SearchParams::new(1, DistanceMetric::Euclidean),
        )
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].id, VectorId::U64(1));
}

#[test]
fn double_train_is_rejected() {
    let mut idx = make_index();
    let sample = make_sample();
    let sample_refs: Vec<&[f32]> = sample.iter().map(|v| v.as_slice()).collect();
    idx.train(&sample_refs).unwrap();
    let err = idx.train(&sample_refs).unwrap_err();
    match err {
        IqdbError::InvalidConfig { reason } => {
            assert!(reason.contains("already trained"));
        }
        other => panic!("expected InvalidConfig, got {other:?}"),
    }
}
