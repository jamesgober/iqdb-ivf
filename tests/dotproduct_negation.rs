//! DotProduct returns the *largest* dot product first because we
//! negate at the search boundary so smaller-is-nearer holds. Verifies
//! IVF agrees with FlatIndex on the top hit under DotProduct.
//!
//! Cargo discovers this file as an integration-test binary. The
//! approved plan names it under §"Test plan".

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use iqdb_flat::{FlatConfig, FlatIndex};
use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, SearchParams, VectorId};

#[test]
fn ivf_top1_matches_flat_under_dotproduct() {
    let data: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![0.7, 0.7],
        vec![-1.0, 0.0],
        vec![0.5, 0.5],
        vec![2.0, 2.0], // largest magnitude — biggest dot with [1,1]
        vec![1.5, 1.5],
        vec![0.9, 0.9],
    ];
    let metric = DistanceMetric::DotProduct;

    let mut flat = FlatIndex::new(2, metric, FlatConfig).unwrap();
    for (i, v) in data.iter().enumerate() {
        flat.insert(
            VectorId::from(i as u64),
            Arc::<[f32]>::from(v.as_slice()),
            None,
        )
        .unwrap();
    }

    let cfg = IvfConfig::default()
        .with_n_clusters(2)
        .with_n_probes(2)
        .with_training_sample_size(64)
        .with_seed(3);
    let mut ivf = IvfIndex::new(2, metric, cfg).unwrap();
    let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
    ivf.train(&refs).unwrap();
    for (i, v) in data.iter().enumerate() {
        ivf.insert(
            VectorId::from(i as u64),
            Arc::<[f32]>::from(v.as_slice()),
            None,
        )
        .unwrap();
    }

    let query = [1.0_f32, 1.0];
    let flat_top = flat.search(&query, &SearchParams::new(1, metric)).unwrap();
    let ivf_top = ivf.search(&query, &SearchParams::new(1, metric)).unwrap();
    assert_eq!(flat_top.len(), 1);
    assert_eq!(ivf_top.len(), 1);
    assert_eq!(flat_top[0].id, ivf_top[0].id);
    // The Hit.distance is the *negated* dot product. Index 5 is
    // [2.0, 2.0] → raw dot 4.0, distance -4.0.
    assert_eq!(flat_top[0].id, VectorId::U64(5));
    assert!(
        (flat_top[0].distance - ivf_top[0].distance).abs() < 1e-5,
        "distances should agree: flat={} ivf={}",
        flat_top[0].distance,
        ivf_top[0].distance,
    );
    assert!(
        flat_top[0].distance < 0.0,
        "negation should produce a negative distance"
    );
}
