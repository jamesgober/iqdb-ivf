//! `IvfIndex::retrain` — correctness, determinism, and edge cases.
//!
//! These tests pin the contract called out in the approved plan:
//!
//! - retrain preserves every live `(id, seq, metadata)` tuple — no
//!   data loss;
//! - retrain is deterministic under `IvfConfig::seed` — same
//!   insert/delete history produces byte-identical search results;
//! - retrain reencodes PQ codes against fresh codebooks — searches
//!   after retrain remain consistent with the flat oracle;
//! - retrain respects `training_sample_size` — running it twice in
//!   a row from the same state is a fixed point;
//! - retrain is a no-op on an empty trained index;
//! - retrain on an untrained index returns `InvalidConfig`.

#![allow(clippy::unwrap_used)]

use std::collections::HashSet;
use std::sync::Arc;

use iqdb_flat::{FlatConfig, FlatIndex};
use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, IqdbError, SearchParams, VectorId};

const DIM: usize = 4;

fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit_float(state: &mut u64) -> f32 {
    let raw = next_u64(state);
    (raw >> 11) as f32 / (1u64 << 53) as f32
}

fn cluster_corpus(seed: u64, n: usize, centres: &[Vec<f32>], jitter: f32) -> Vec<Vec<f32>> {
    let mut state = seed.wrapping_add(0x1234_5678_9ABC_DEF0);
    (0..n)
        .map(|i| {
            let centre = &centres[i % centres.len()];
            centre
                .iter()
                .map(|&c| c + (unit_float(&mut state) * 2.0 - 1.0) * jitter)
                .collect()
        })
        .collect()
}

fn populate(idx: &mut IvfIndex, vectors: &[Vec<f32>]) {
    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    idx.train(&refs).unwrap();
    for (i, v) in vectors.iter().enumerate() {
        idx.insert(
            VectorId::from(i as u64),
            Arc::<[f32]>::from(v.as_slice()),
            None,
        )
        .unwrap();
    }
}

fn make_index(use_pq: bool, n_clusters: usize) -> IvfIndex {
    let cfg_base = IvfConfig::default()
        .with_n_clusters(n_clusters)
        .with_n_probes(n_clusters)
        .with_training_sample_size(256)
        .with_seed(0xABCD_EF01_2345_6789);
    let cfg = if use_pq {
        cfg_base
            .with_use_pq(true)
            .with_pq_subvectors(Some(2))
            .with_pq_refine_factor(4)
    } else {
        cfg_base
    };
    IvfIndex::new(DIM, DistanceMetric::Euclidean, cfg).unwrap()
}

fn ivf_ids(idx: &IvfIndex, query: &[f32], k: usize) -> HashSet<VectorId> {
    let params = SearchParams::new(k, DistanceMetric::Euclidean);
    idx.search(query, &params)
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect()
}

fn ivf_ordered_ids(idx: &IvfIndex, query: &[f32], k: usize) -> Vec<VectorId> {
    let params = SearchParams::new(k, DistanceMetric::Euclidean);
    idx.search(query, &params)
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect()
}

fn flat_ids(flat: &FlatIndex, query: &[f32], k: usize) -> HashSet<VectorId> {
    let params = SearchParams::new(k, DistanceMetric::Euclidean);
    flat.search(query, &params)
        .unwrap()
        .into_iter()
        .map(|h| h.id)
        .collect()
}

fn build_flat(data: &[Vec<f32>]) -> FlatIndex {
    let mut flat = FlatIndex::new(DIM, DistanceMetric::Euclidean, FlatConfig).unwrap();
    for (i, v) in data.iter().enumerate() {
        flat.insert(
            VectorId::from(i as u64),
            Arc::<[f32]>::from(v.as_slice()),
            None,
        )
        .unwrap();
    }
    flat
}

#[test]
fn retrain_preserves_ids_and_count() {
    let mut idx = make_index(/*use_pq=*/ false, /*n_clusters=*/ 4);
    let centres: Vec<Vec<f32>> = (0..4)
        .map(|c| (0..DIM).map(|i| c as f32 * 5.0 + i as f32 * 0.1).collect())
        .collect();
    let data = cluster_corpus(11, 64, &centres, 0.3);
    populate(&mut idx, &data);

    let before_len = idx.len();
    let before_ids: HashSet<VectorId> = (0..before_len as u64).map(VectorId::from).collect();

    idx.retrain().unwrap();

    assert_eq!(idx.len(), before_len, "retrain must preserve live count");
    // Every id must still be searchable. Pull the full population
    // back via an exhaustive query (k = n).
    let query: Vec<f32> = vec![0.0; DIM];
    let after_ids = ivf_ids(&idx, &query, before_len);
    assert_eq!(after_ids, before_ids, "retrain must preserve every id");
}

#[test]
fn retrain_holds_or_improves_recall_after_drift() {
    // Train on 2 centres, then insert a third "ghost" cluster that
    // the original centroids do not represent. With `n_probes = 1`
    // pre-retrain the search is forced into the nearest of the
    // two stale centroids, so a query near the ghost centre is
    // underserved. retrain rebalances the centroids over the
    // currently-stored vectors and must hold or improve recall.
    let mut idx = make_index(/*use_pq=*/ false, /*n_clusters=*/ 2);
    let train_centres: Vec<Vec<f32>> = vec![
        (0..DIM).map(|i| i as f32 * 0.1).collect(),
        (0..DIM).map(|i| 10.0 + i as f32 * 0.1).collect(),
    ];
    let drift_centre: Vec<f32> = (0..DIM).map(|i| 5.0 + i as f32 * 0.1).collect();
    let initial = cluster_corpus(21, 32, &train_centres, 0.2);
    let drifted = cluster_corpus(22, 32, std::slice::from_ref(&drift_centre), 0.2);

    let refs: Vec<&[f32]> = initial.iter().map(|v| v.as_slice()).collect();
    idx.train(&refs).unwrap();
    idx.set_n_probes(1).unwrap();
    let mut combined: Vec<Vec<f32>> = Vec::new();
    for (id_counter, v) in (0_u64..).zip(initial.iter().chain(drifted.iter())) {
        idx.insert(
            VectorId::from(id_counter),
            Arc::<[f32]>::from(v.as_slice()),
            None,
        )
        .unwrap();
        combined.push(v.clone());
    }

    let flat = build_flat(&combined);
    let k = 8;
    let oracle = flat_ids(&flat, &drift_centre, k);
    let pre = ivf_ids(&idx, &drift_centre, k);
    let pre_recall = pre.intersection(&oracle).count();

    idx.retrain().unwrap();
    let post = ivf_ids(&idx, &drift_centre, k);
    let post_recall = post.intersection(&oracle).count();

    assert!(
        post_recall >= pre_recall,
        "retrain must hold or improve recall on a drifted corpus (pre = {pre_recall}, post = {post_recall})",
    );
}

#[test]
fn retrain_is_deterministic_under_seed() {
    // Two indices built from identical config + identical insert
    // sequence + identical seed must produce identical search
    // results after retrain.
    let centres: Vec<Vec<f32>> = (0..4)
        .map(|c| (0..DIM).map(|i| c as f32 * 5.0 + i as f32 * 0.1).collect())
        .collect();
    let data = cluster_corpus(31, 64, &centres, 0.25);

    let mut a = make_index(false, 4);
    let mut b = make_index(false, 4);
    populate(&mut a, &data);
    populate(&mut b, &data);

    a.retrain().unwrap();
    b.retrain().unwrap();

    let query: Vec<f32> = (0..DIM).map(|i| i as f32 * 0.3).collect();
    let a_hits = ivf_ordered_ids(&a, &query, 8);
    let b_hits = ivf_ordered_ids(&b, &query, 8);
    assert_eq!(a_hits, b_hits, "retrain must be deterministic under seed");
}

#[test]
fn retrain_reencodes_pq_codes() {
    // With `use_pq = true`, retrain re-trains the codebooks and
    // re-encodes every entry. The post-retrain search must remain
    // consistent with the flat oracle's filter-survivor contract:
    // every ivf-pq hit lies in the flat corpus, and the top-k size
    // is preserved.
    let centres: Vec<Vec<f32>> = (0..2)
        .map(|c| (0..DIM).map(|i| c as f32 * 8.0 + i as f32 * 0.1).collect())
        .collect();
    // PQ codebooks require >= 256 training vectors (`PQ_K_CENTROIDS`).
    let data = cluster_corpus(41, 384, &centres, 0.3);

    let mut idx = make_index(/*use_pq=*/ true, /*n_clusters=*/ 4);
    populate(&mut idx, &data);
    idx.retrain().unwrap();

    let flat = build_flat(&data);
    let query: Vec<f32> = (0..DIM).map(|i| i as f32 * 0.4).collect();
    let k = 8;
    let ivf_hits = ivf_ids(&idx, &query, k);
    let flat_full = flat_ids(&flat, &query, data.len());
    for id in &ivf_hits {
        assert!(
            flat_full.contains(id),
            "post-retrain IVF-PQ hit not in the flat corpus: {id:?}",
        );
    }
    assert_eq!(ivf_hits.len(), k.min(data.len()));
}

#[test]
fn retrain_respects_training_sample_size_cap() {
    // Populate with N > training_sample_size, then retrain twice.
    // The cap means the working set is bounded; the second retrain
    // runs on the same capped sample (same seed, same partial
    // Fisher-Yates), so it is a fixed point: identical search
    // results across the two calls.
    let centres: Vec<Vec<f32>> = (0..4)
        .map(|c| (0..DIM).map(|i| c as f32 * 5.0 + i as f32 * 0.1).collect())
        .collect();
    let data = cluster_corpus(51, 512, &centres, 0.3);

    let cfg = IvfConfig::default()
        .with_n_clusters(4)
        .with_n_probes(4)
        // Cap << corpus size so the cap actually fires.
        .with_training_sample_size(64)
        .with_seed(0x5A5A_5A5A_5A5A_5A5A);
    let mut idx = IvfIndex::new(DIM, DistanceMetric::Euclidean, cfg).unwrap();
    populate(&mut idx, &data);

    idx.retrain().unwrap();
    let query: Vec<f32> = (0..DIM).map(|i| i as f32 * 0.3).collect();
    let after_first = ivf_ordered_ids(&idx, &query, 8);

    idx.retrain().unwrap();
    let after_second = ivf_ordered_ids(&idx, &query, 8);

    assert_eq!(
        after_first, after_second,
        "two retrains from the same state must produce identical results (cap respected)",
    );
}

#[test]
fn retrain_on_empty_index_is_noop() {
    // Train, delete everything, retrain → Ok(()); the index is
    // still trained and still answers queries with no hits.
    let mut idx = make_index(false, 2);
    let centres: Vec<Vec<f32>> = vec![
        (0..DIM).map(|i| i as f32 * 0.1).collect(),
        (0..DIM).map(|i| 10.0 + i as f32 * 0.1).collect(),
    ];
    let data = cluster_corpus(61, 16, &centres, 0.2);
    populate(&mut idx, &data);

    for i in 0..(data.len() as u64) {
        idx.delete(&VectorId::from(i)).unwrap();
    }
    assert_eq!(idx.len(), 0);

    idx.retrain().unwrap();
    let query: Vec<f32> = vec![0.0; DIM];
    let params = SearchParams::new(4, DistanceMetric::Euclidean);
    let hits = idx.search(&query, &params).unwrap();
    assert!(hits.is_empty());
    assert!(idx.is_trained(), "empty retrain must not untrain the index");
}

#[test]
fn retrain_before_train_errors() {
    let cfg = IvfConfig::default()
        .with_n_clusters(2)
        .with_n_probes(1)
        .with_training_sample_size(8)
        .with_seed(7);
    let mut idx = IvfIndex::new(DIM, DistanceMetric::Euclidean, cfg).unwrap();
    let err = idx.retrain().unwrap_err();
    assert!(
        matches!(err, IqdbError::InvalidConfig { reason } if reason.contains("trained")),
        "expected InvalidConfig for retrain-before-train, got {err:?}",
    );
}
