//! Property-based invariants for `IvfIndex`.
//!
//! The load-bearing property: an IVF-Flat index that probes *every*
//! cluster must return exactly what the brute-force
//! [`iqdb_flat::FlatIndex`] oracle returns — same ids, same order —
//! because it scores the same candidate set with the same
//! `(distance, seq)` tiebreaker. The remaining properties pin the
//! ordering, cardinality, filter-subset, and probe-suggestion contracts
//! that must hold for any probe count.

#![allow(clippy::unwrap_used)]

use std::collections::HashSet;
use std::sync::Arc;

use iqdb_flat::{FlatConfig, FlatIndex};
use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, Filter, Metadata, SearchParams, Value, VectorId};
use proptest::prelude::*;

fn arc(v: &[f32]) -> Arc<[f32]> {
    Arc::from(v)
}

/// Deterministic clustered row: `seed` selects one of `n_centres`
/// well-separated centres and jitters around it, so k-means has real
/// structure to recover.
fn clustered_row(seed: u64, dim: usize, n_centres: u64) -> Vec<f32> {
    let centre = seed % n_centres;
    (0..dim)
        .map(|j| {
            let base = (centre as f32) * 10.0;
            let jitter =
                (((seed.wrapping_mul(131).wrapping_add(j as u64 * 71)) % 100) as f32) * 0.01;
            base + jitter
        })
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    /// Full-probe IVF-Flat reproduces the exact flat oracle: same ids in
    /// the same order, because it scores every vector with the identical
    /// `(distance, seq)` tiebreaker.
    #[test]
    fn full_probe_matches_flat_oracle(
        n in 8usize..60,
        dim in 1usize..12,
        n_clusters in 1usize..8,
        k in 0usize..30,
    ) {
        prop_assume!(n_clusters <= n);
        let metric = DistanceMetric::Euclidean;
        let rows: Vec<Vec<f32>> =
            (0..n).map(|i| clustered_row(i as u64, dim, n_clusters as u64)).collect();

        let mut flat = FlatIndex::new(dim, metric, FlatConfig).unwrap();
        for (i, r) in rows.iter().enumerate() {
            flat.insert(VectorId::from(i as u64), arc(r), None).unwrap();
        }

        let cfg = IvfConfig::default()
            .with_n_clusters(n_clusters)
            .with_n_probes(n_clusters) // probe everything
            .with_training_sample_size(4096)
            .with_seed(42);
        let mut ivf = IvfIndex::new(dim, metric, cfg).unwrap();
        let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
        ivf.train(&refs).unwrap();
        for (i, r) in rows.iter().enumerate() {
            ivf.insert(VectorId::from(i as u64), arc(r), None).unwrap();
        }

        let query = clustered_row(99_991, dim, n_clusters as u64);
        let params = SearchParams::new(k, metric);
        let flat_ids: Vec<_> =
            flat.search(&query, &params).unwrap().into_iter().map(|h| h.id).collect();
        let ivf_ids: Vec<_> =
            ivf.search(&query, &params).unwrap().into_iter().map(|h| h.id).collect();
        prop_assert_eq!(flat_ids, ivf_ids);
    }

    /// Top-`k` is sorted best-first and never longer than `k` or the live
    /// count, for any probe count.
    #[test]
    fn topk_sorted_and_bounded(
        n in 1usize..60,
        dim in 1usize..12,
        n_clusters in 1usize..8,
        k in 0usize..40,
        n_probes_raw in 1usize..8,
    ) {
        prop_assume!(n_clusters <= n);
        let n_probes = n_probes_raw.min(n_clusters);
        let metric = DistanceMetric::Euclidean;
        let rows: Vec<Vec<f32>> =
            (0..n).map(|i| clustered_row(i as u64 + 1, dim, n_clusters as u64)).collect();

        let cfg = IvfConfig::default()
            .with_n_clusters(n_clusters)
            .with_n_probes(n_probes)
            .with_training_sample_size(4096)
            .with_seed(11);
        let mut ivf = IvfIndex::new(dim, metric, cfg).unwrap();
        let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
        ivf.train(&refs).unwrap();
        for (i, r) in rows.iter().enumerate() {
            ivf.insert(VectorId::from(i as u64), arc(r), None).unwrap();
        }
        let query = clustered_row(5, dim, n_clusters as u64);
        let hits = ivf.search(&query, &SearchParams::new(k, metric)).unwrap();

        prop_assert!(hits.len() <= k);
        prop_assert!(hits.len() <= n);
        for w in hits.windows(2) {
            prop_assert!(w[0].distance.total_cmp(&w[1].distance) != std::cmp::Ordering::Greater);
        }
    }

    /// Every filtered hit satisfies the predicate and is drawn from the
    /// unfiltered candidate set.
    #[test]
    fn filtered_results_respect_predicate(
        n in 4usize..50,
        dim in 1usize..8,
        n_clusters in 1usize..6,
        k in 1usize..30,
    ) {
        prop_assume!(n_clusters <= n);
        let metric = DistanceMetric::Euclidean;
        let rows: Vec<Vec<f32>> =
            (0..n).map(|i| clustered_row(i as u64 + 2, dim, n_clusters as u64)).collect();

        let cfg = IvfConfig::default()
            .with_n_clusters(n_clusters)
            .with_n_probes(n_clusters) // exhaustive: filter is the only difference
            .with_training_sample_size(4096)
            .with_seed(3);
        let mut ivf = IvfIndex::new(dim, metric, cfg).unwrap();
        let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
        ivf.train(&refs).unwrap();
        for (i, r) in rows.iter().enumerate() {
            let flag = i % 2 == 0;
            let meta: Metadata =
                [("flag".to_string(), Value::Bool(flag))].into_iter().collect();
            ivf.insert(VectorId::from(i as u64), arc(r), Some(meta)).unwrap();
        }
        let query = clustered_row(8, dim, n_clusters as u64);

        let unfiltered_ids: HashSet<VectorId> = ivf
            .search(&query, &SearchParams::new(n, metric))
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        let filtered_params = SearchParams {
            filter: Some(Filter::eq("flag", Value::Bool(true))),
            ..SearchParams::new(k, metric)
        };
        let filtered = ivf.search(&query, &filtered_params).unwrap();

        for h in &filtered {
            prop_assert!(unfiltered_ids.contains(&h.id), "filtered hit not in candidate set");
            let meta = h.metadata.as_ref().expect("inserted with metadata");
            prop_assert_eq!(meta.get("flag"), Some(&Value::Bool(true)));
        }
    }

    /// `suggest_n_probes` is monotone non-decreasing in coverage and
    /// always clamped into `[1, n_clusters]`.
    #[test]
    fn suggest_n_probes_monotone_and_clamped(
        n in 4usize..60,
        n_clusters in 2usize..10,
        c_a in 0.0f32..1.0,
        c_b in 0.0f32..1.0,
    ) {
        prop_assume!(n_clusters <= n);
        let metric = DistanceMetric::Euclidean;
        let dim = 4;
        let rows: Vec<Vec<f32>> =
            (0..n).map(|i| clustered_row(i as u64 + 4, dim, n_clusters as u64)).collect();
        let cfg = IvfConfig::default()
            .with_n_clusters(n_clusters)
            .with_n_probes(1)
            .with_training_sample_size(4096)
            .with_seed(9);
        let mut ivf = IvfIndex::new(dim, metric, cfg).unwrap();
        let refs: Vec<&[f32]> = rows.iter().map(|r| r.as_slice()).collect();
        ivf.train(&refs).unwrap();
        for (i, r) in rows.iter().enumerate() {
            ivf.insert(VectorId::from(i as u64), arc(r), None).unwrap();
        }

        let (lo, hi) = if c_a <= c_b { (c_a, c_b) } else { (c_b, c_a) };
        let p_lo = ivf.suggest_n_probes(lo).unwrap();
        let p_hi = ivf.suggest_n_probes(hi).unwrap();
        prop_assert!((1..=n_clusters).contains(&p_lo));
        prop_assert!((1..=n_clusters).contains(&p_hi));
        prop_assert!(p_lo <= p_hi, "monotone: cov {} -> {}, {} -> {}", lo, p_lo, hi, p_hi);
    }
}

/// Training is mandatory: the centroid-dependent entry points fail before
/// `train`. A single deterministic assertion, not a property.
#[test]
fn untrained_entry_points_error() {
    let cfg = IvfConfig::default().with_n_clusters(2).with_n_probes(1);
    let mut ivf = IvfIndex::new(3, DistanceMetric::Euclidean, cfg).unwrap();
    assert!(
        ivf.insert(VectorId::from(1u64), arc(&[0.0, 0.0, 0.0]), None)
            .is_err()
    );
    assert!(
        ivf.search(
            &[0.0, 0.0, 0.0],
            &SearchParams::new(1, DistanceMetric::Euclidean)
        )
        .is_err()
    );
}
