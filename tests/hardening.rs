//! Retrain failure safety, config size bounds, NaN ranking, and Hamming
//! probe routing.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashSet;
use std::sync::Arc;

use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, IqdbError, SearchParams, VectorId};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

fn corpus(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut r = Rng(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| r.unit()).collect())
        .collect()
}

fn build(metric: DistanceMetric, cfg: IvfConfig, data: &[Vec<f32>]) -> IvfIndex {
    let refs: Vec<&[f32]> = data.iter().map(|v| v.as_slice()).collect();
    let mut ivf = IvfIndex::new(data[0].len(), metric, cfg).unwrap();
    ivf.train(&refs).unwrap();
    for (i, v) in data.iter().enumerate() {
        ivf.insert(VectorId::from(i as u64), Arc::from(v.as_slice()), None)
            .unwrap();
    }
    ivf
}

#[test]
fn failed_retrain_keeps_every_vector() {
    let data = corpus(200, 4, 3);
    let cfg = IvfConfig::default().with_n_clusters(16).with_n_probes(16);
    let mut ivf = build(DistanceMetric::Euclidean, cfg, &data);
    for i in 0..190u64 {
        ivf.delete(&VectorId::from(i)).unwrap();
    }
    assert_eq!(ivf.len(), 10);

    // Ten live vectors cannot train sixteen clusters.
    let err = ivf.retrain().unwrap_err();
    assert!(matches!(err, IqdbError::InvalidConfig { .. }), "{err:?}");

    assert_eq!(ivf.len(), 10);
    let hits = ivf
        .search(&data[195], &SearchParams::new(5, DistanceMetric::Euclidean))
        .unwrap();
    assert_eq!(hits.len(), 5);
    assert_eq!(hits[0].id, VectorId::from(195u64));
    // Still mutable and consistent with the old model.
    ivf.delete(&VectorId::from(199u64)).unwrap();
    assert!(matches!(
        ivf.insert(VectorId::from(195u64), Arc::from(data[0].as_slice()), None),
        Err(IqdbError::Duplicate)
    ));
    assert_eq!(ivf.len(), 9);
}

#[test]
fn failed_pq_retrain_keeps_every_vector_and_code() {
    let data = corpus(400, 8, 5);
    let cfg = IvfConfig::default()
        .with_n_clusters(4)
        .with_n_probes(4)
        .with_use_pq(true)
        .with_pq_subvectors(Some(2))
        .with_pq_refine_factor(0);
    let mut ivf = build(DistanceMetric::Euclidean, cfg, &data);
    let p = SearchParams::new(5, DistanceMetric::Euclidean);
    for i in 0..300u64 {
        ivf.delete(&VectorId::from(i)).unwrap();
    }
    let before: Vec<VectorId> = ivf
        .search(&data[350], &p)
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();

    // A hundred vectors are enough for k-means but not for a 256-entry
    // PQ codebook, so the second training stage fails.
    assert!(ivf.retrain().is_err());

    assert_eq!(ivf.len(), 100);
    let after: Vec<VectorId> = ivf
        .search(&data[350], &p)
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect();
    assert_eq!(before, after);
}

#[test]
fn successful_retrain_still_keeps_every_vector() {
    let data = corpus(500, 4, 9);
    let cfg = IvfConfig::default().with_n_clusters(8).with_n_probes(8);
    let mut ivf = build(DistanceMetric::Euclidean, cfg, &data);
    for i in 0..250u64 {
        ivf.delete(&VectorId::from(i)).unwrap();
    }
    ivf.retrain().unwrap();
    assert_eq!(ivf.len(), 250);
    assert_eq!(ivf.cluster_stats().cluster_sizes.iter().sum::<usize>(), 250);
    let hits = ivf
        .search(&data[400], &SearchParams::new(1, DistanceMetric::Euclidean))
        .unwrap();
    assert_eq!(hits[0].id, VectorId::from(400u64));
}

#[test]
fn oversized_n_clusters_is_rejected_before_allocating() {
    let cfg = IvfConfig::default()
        .with_n_clusters(1 << 40)
        .with_n_probes(1);
    assert!(matches!(
        IvfIndex::new(4, DistanceMetric::Euclidean, cfg),
        Err(IqdbError::InvalidConfig { .. })
    ));
    assert!(cfg.validate().is_err());
    let at_cap = IvfConfig::default()
        .with_n_clusters(1 << 20)
        .with_n_probes(1);
    assert!(at_cap.validate().is_ok());
}

#[test]
fn huge_k_is_bounded_by_the_probed_entries() {
    let data = corpus(100, 4, 15);
    let cfg = IvfConfig::default().with_n_clusters(4).with_n_probes(4);
    let ivf = build(DistanceMetric::Euclidean, cfg, &data);
    let hits = ivf
        .search(
            &data[0],
            &SearchParams::new(usize::MAX, DistanceMetric::Euclidean),
        )
        .unwrap();
    assert_eq!(hits.len(), 100);
}

#[test]
fn overflowing_cosine_candidate_does_not_outrank_an_exact_match() {
    let cfg = IvfConfig::default().with_n_clusters(1).with_n_probes(1);
    let sample: Vec<&[f32]> = vec![&[1.0, 1.0], &[1.0, -1.0]];
    let mut ivf = IvfIndex::new(2, DistanceMetric::Cosine, cfg).unwrap();
    ivf.train(&sample).unwrap();
    ivf.insert(VectorId::from(1u64), Arc::from(&[1.0f32, 1.0][..]), None)
        .unwrap();
    ivf.insert(
        VectorId::from(2u64),
        Arc::from(
            &[
                std::hint::black_box(1e20f32),
                std::hint::black_box(-1e20f32),
            ][..],
        ),
        None,
    )
    .unwrap();
    let q = [std::hint::black_box(1e20f32), std::hint::black_box(1e20f32)];
    let hits = ivf
        .search(&q, &SearchParams::new(2, DistanceMetric::Cosine))
        .unwrap();
    assert_eq!(hits[0].id, VectorId::from(1u64), "{hits:?}");
}

fn exact_hamming(data: &[Vec<f32>], q: &[f32], k: usize) -> HashSet<VectorId> {
    let mut scored: Vec<(f32, u64)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let d = v
                .iter()
                .zip(q)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count() as f32;
            (d, i as u64)
        })
        .collect();
    scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    scored
        .iter()
        .take(k)
        .map(|&(_, i)| VectorId::from(i))
        .collect()
}

#[test]
fn hamming_probes_follow_the_query() {
    // Binary codes. Under Hamming every centroid (a mean) used to score
    // about `dim`, so the same clusters were probed for every query.
    let dim = 64;
    let mut r = Rng(11);
    let data: Vec<Vec<f32>> = (0..4000)
        .map(|_| (0..dim).map(|_| (r.next() & 1) as f32).collect())
        .collect();
    let cfg = IvfConfig::default().with_n_clusters(32).with_n_probes(4);
    let ivf = build(DistanceMetric::Hamming, cfg, &data);
    let p = SearchParams::new(10, DistanceMetric::Hamming);
    let mut found = 0usize;
    let queries: Vec<&Vec<f32>> = data.iter().step_by(97).take(40).collect();
    for q in &queries {
        let got: HashSet<VectorId> = ivf
            .search(q, &p)
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        found += got.intersection(&exact_hamming(&data, q, 10)).count();
    }
    let recall = found as f32 / (10 * queries.len()) as f32;
    // 4 of 32 clusters on random codes: the same as Euclidean routing on
    // this data (0.475), against 0.112 before.
    assert!(recall > 0.4, "hamming recall@10 = {recall}");
}
