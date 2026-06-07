//! Recall validation at scale.
//!
//! A 2 000-vector, 16-dimensional corpus drawn from 16 well-separated
//! latent clusters, validated against the exact [`iqdb_flat::FlatIndex`]
//! oracle. This is the test that backs the crate's recall claims:
//!
//! - **Exhaustive IVF-Flat is exact.** With `n_probes == n_clusters` the
//!   IVF-Flat top-`k` is byte-identical to the brute-force oracle.
//! - **Tuned IVF-Flat keeps high recall.** Probing a small fraction of
//!   the clusters still recovers the overwhelming majority of true
//!   neighbours on clustered data.
//! - **IVF-PQ with refine stays competitive.** The lossy ADC scan, when
//!   exact-reranked, recovers most of the true neighbour set at a
//!   fraction of the memory.

#![allow(clippy::unwrap_used)]

use std::collections::HashSet;
use std::sync::Arc;

use iqdb_flat::{FlatConfig, FlatIndex};
use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, SearchParams, VectorId};

const DIM: usize = 16;
const N_TRUE_CLUSTERS: u64 = 16;
const POINTS_PER_CLUSTER: usize = 125;
const N: usize = (N_TRUE_CLUSTERS as usize) * POINTS_PER_CLUSTER; // 2_000
const N_CLUSTERS: usize = 32;
const METRIC: DistanceMetric = DistanceMetric::Euclidean;

/// Deterministic centre coordinate for latent cluster `c`, dimension `j`.
fn centre_coord(c: u64, j: usize) -> f32 {
    let h = c
        .wrapping_mul(2_654_435_761)
        .wrapping_add((j as u64).wrapping_mul(40_503));
    ((h % 1_000) as f32) / 1_000.0 * 20.0
}

/// Row `i`: assigned to latent cluster `i % N_TRUE_CLUSTERS`, jittered.
fn row(i: usize) -> Vec<f32> {
    let c = (i as u64) % N_TRUE_CLUSTERS;
    (0..DIM)
        .map(|j| {
            let jitter = ((((i * 131 + j * 71) % 100) as f32) - 50.0) * 0.006; // ±0.3
            centre_coord(c, j) + jitter
        })
        .collect()
}

/// A held-out query near latent cluster `c` (different jitter seed).
fn query_near(c: u64) -> Vec<f32> {
    (0..DIM)
        .map(|j| {
            let jitter = ((((c as usize * 909 + j * 13) % 100) as f32) - 50.0) * 0.004;
            centre_coord(c, j) + jitter
        })
        .collect()
}

fn corpus() -> Vec<Vec<f32>> {
    (0..N).map(row).collect()
}

fn build_flat(data: &[Vec<f32>]) -> FlatIndex {
    let mut idx = FlatIndex::new(DIM, METRIC, FlatConfig).unwrap();
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

fn build_ivf(data: &[Vec<f32>], n_probes: usize, use_pq: bool) -> IvfIndex {
    let mut cfg = IvfConfig::default()
        .with_n_clusters(N_CLUSTERS)
        .with_n_probes(n_probes)
        .with_training_sample_size(4096)
        .with_seed(0xABCD);
    if use_pq {
        // 8 subvectors over a 16-D space → 2 dims per subvector; refine
        // exact-reranks a 4×k shortlist.
        cfg = cfg
            .with_use_pq(true)
            .with_pq_subvectors(Some(8))
            .with_pq_refine_factor(4);
    }
    let mut idx = IvfIndex::new(DIM, METRIC, cfg).unwrap();
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

/// Mean recall@k of `ivf` against the exact `flat` oracle over one query
/// per latent cluster.
fn mean_recall(flat: &FlatIndex, ivf: &IvfIndex, k: usize) -> f32 {
    let params = SearchParams::new(k, METRIC);
    let mut hit = 0usize;
    let mut total = 0usize;
    for c in 0..N_TRUE_CLUSTERS {
        let q = query_near(c);
        let truth: HashSet<VectorId> = flat
            .search(&q, &params)
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        let got: HashSet<VectorId> = ivf
            .search(&q, &params)
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        hit += truth.intersection(&got).count();
        total += truth.len();
    }
    (hit as f32) / (total as f32)
}

#[test]
fn exhaustive_ivf_flat_is_exact_at_scale() {
    let data = corpus();
    let flat = build_flat(&data);
    let ivf = build_ivf(&data, N_CLUSTERS, false); // probe every cluster
    let params = SearchParams::new(10, METRIC);
    for c in 0..N_TRUE_CLUSTERS {
        let q = query_near(c);
        let want: Vec<_> = flat
            .search(&q, &params)
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        let got: Vec<_> = ivf
            .search(&q, &params)
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(
            want, got,
            "full-probe IVF-Flat must equal the oracle (cluster {c})"
        );
    }
}

#[test]
fn tuned_ivf_flat_keeps_high_recall_at_scale() {
    let data = corpus();
    let flat = build_flat(&data);
    // Probe a quarter of the clusters.
    let ivf = build_ivf(&data, N_CLUSTERS / 4, false);
    let recall = mean_recall(&flat, &ivf, 10);
    assert!(
        recall >= 0.90,
        "recall@10 = {recall} with {} of {N_CLUSTERS} probes should be >= 0.90",
        N_CLUSTERS / 4
    );
}

#[test]
fn ivf_pq_with_refine_stays_competitive_at_scale() {
    let data = corpus();
    let flat = build_flat(&data);
    let ivf = build_ivf(&data, N_CLUSTERS, true); // full probe, PQ + refine
    let recall = mean_recall(&flat, &ivf, 10);
    assert!(
        recall >= 0.70,
        "IVF-PQ recall@10 = {recall} with refine should be >= 0.70"
    );
}
