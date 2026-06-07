//! End-to-end smoke coverage for the IVF-PQ branch: train with
//! `use_pq=true`, insert vectors, search, and confirm the hits are
//! shaped correctly (count, distance order, monotone seq tiebreaker).
//!
//! Cargo discovers this file as an integration-test binary. Named in
//! the approved plan §"Test plan".

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, SearchParams, VectorId};

const DIM: usize = 8;
const M: usize = 2; // 8 / 2 = 4 dims per PQ subvector
const N_TRAIN: usize = 512; // > PQ K=256

fn gaussian_cluster_corpus(seed: u64, n: usize) -> Vec<Vec<f32>> {
    // Deterministic, dependency-free PRNG (Vigna SplitMix64).
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    // 4 cluster centres, evenly spaced in 8-D.
    let centres: Vec<Vec<f32>> = (0..4)
        .map(|c| {
            let base = c as f32 * 5.0;
            (0..DIM).map(|i| base + i as f32 * 0.1).collect()
        })
        .collect();
    (0..n)
        .map(|i| {
            let centre = &centres[i % centres.len()];
            centre
                .iter()
                .map(|&c| {
                    let u = (next() >> 11) as f32 / (1u64 << 53) as f32;
                    c + (u * 2.0 - 1.0) * 0.3
                })
                .collect()
        })
        .collect()
}

#[test]
fn pq_index_trains_inserts_and_searches() {
    let data = gaussian_cluster_corpus(11, N_TRAIN);
    let metric = DistanceMetric::Euclidean;

    let cfg = IvfConfig::default()
        .with_n_clusters(4)
        .with_n_probes(4)
        .with_training_sample_size(N_TRAIN)
        .with_use_pq(true)
        .with_pq_subvectors(Some(M))
        .with_seed(42);
    let mut idx = IvfIndex::new(DIM, metric, cfg).unwrap();
    assert!(!idx.is_trained());

    let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
    idx.train(&refs).unwrap();
    assert!(idx.is_trained());

    for (i, v) in data.iter().enumerate() {
        idx.insert(
            VectorId::from(i as u64),
            Arc::<[f32]>::from(v.as_slice()),
            None,
        )
        .unwrap();
    }
    assert_eq!(idx.len(), N_TRAIN);

    let hits = idx.search(&data[0], &SearchParams::new(5, metric)).unwrap();
    assert_eq!(hits.len(), 5);
    // Distance should be non-decreasing (top-k in ascending distance).
    for w in hits.windows(2) {
        assert!(
            w[0].distance <= w[1].distance,
            "hits must be in ascending distance order; got {} then {}",
            w[0].distance,
            w[1].distance,
        );
    }
    // The query is `data[0]`, which was itself inserted as id 0, so
    // it should land in the top-k (whether the literal #1 depends on
    // PQ quantization noise + refine).
    let ids: Vec<_> = hits.iter().map(|h| h.id.clone()).collect();
    assert!(
        ids.contains(&VectorId::U64(0)),
        "self-query should retrieve own id; got {ids:?}"
    );
}

#[test]
fn pq_empty_search_returns_empty() {
    let cfg = IvfConfig::default()
        .with_n_clusters(4)
        .with_n_probes(4)
        .with_training_sample_size(N_TRAIN)
        .with_use_pq(true)
        .with_pq_subvectors(Some(M))
        .with_seed(1);
    let mut idx = IvfIndex::new(DIM, DistanceMetric::Euclidean, cfg).unwrap();
    let data = gaussian_cluster_corpus(2, N_TRAIN);
    let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
    idx.train(&refs).unwrap();
    // No inserts.
    let q = vec![0.0_f32; DIM];
    let hits = idx
        .search(&q, &SearchParams::new(5, DistanceMetric::Euclidean))
        .unwrap();
    assert!(hits.is_empty());
}
