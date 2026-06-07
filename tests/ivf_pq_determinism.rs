//! Same `IvfConfig` seed + same training sample + same inserts →
//! byte-identical IVF-PQ search results. The PQ seed flows from
//! `IvfConfig::seed` via `pq_variant::train_pq` so the codebooks
//! and codes are reproducible end-to-end.
//!
//! Cargo discovers this file as an integration-test binary. Named
//! in the approved plan §"Test plan".

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, SearchParams, VectorId};

const DIM: usize = 8;
const M: usize = 2;
const N_TRAIN: usize = 512;
const SEED_A: u64 = 0xA11CE;
const SEED_B: u64 = 0xB0B;

fn corpus(seed: u64, n: usize) -> Vec<Vec<f32>> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
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

fn build(seed: u64, data: &[Vec<f32>], refine_factor: u32) -> IvfIndex {
    let cfg = IvfConfig::default()
        .with_n_clusters(4)
        .with_n_probes(4)
        .with_training_sample_size(N_TRAIN)
        .with_use_pq(true)
        .with_pq_subvectors(Some(M))
        .with_pq_refine_factor(refine_factor)
        .with_seed(seed);
    let mut idx = IvfIndex::new(DIM, DistanceMetric::Euclidean, cfg).unwrap();
    let refs: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
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
fn pq_same_seed_same_data_yields_identical_search_results() {
    let data = corpus(31, N_TRAIN);
    // Refine ON (the operating point default): same seed must still
    // produce byte-identical results end-to-end.
    let a = build(SEED_A, &data, 4);
    let b = build(SEED_A, &data, 4);

    let params = SearchParams::new(10, DistanceMetric::Euclidean);
    for i in (0..N_TRAIN).step_by(37) {
        let q = &data[i];
        let hits_a = a.search(q, &params).unwrap();
        let hits_b = b.search(q, &params).unwrap();
        assert_eq!(hits_a.len(), hits_b.len(), "len mismatch at query {i}");
        for (h_a, h_b) in hits_a.iter().zip(hits_b.iter()) {
            assert_eq!(h_a.id, h_b.id, "id mismatch at query {i}");
            assert_eq!(
                h_a.distance.to_bits(),
                h_b.distance.to_bits(),
                "distance mismatch at query {i}: {} vs {}",
                h_a.distance,
                h_b.distance,
            );
        }
    }
}

#[test]
fn pq_different_seeds_can_diverge_in_adc_only_mode() {
    // Negative-control sanity check: different seeds train
    // different codebooks → at least some queries disagree in the
    // ADC ordering. Run with `pq_refine_factor = 0` so the test
    // observes the raw codebook output; refine ON exact-reranks
    // the shortlist and tends to erase codebook noise, which is
    // the whole point of refine.
    let data = corpus(37, N_TRAIN);
    let a = build(SEED_A, &data, 0);
    let b = build(SEED_B, &data, 0);

    let params = SearchParams::new(10, DistanceMetric::Euclidean);
    let mut any_difference = false;
    for i in (0..N_TRAIN).step_by(11) {
        let q = &data[i];
        let hits_a = a.search(q, &params).unwrap();
        let hits_b = b.search(q, &params).unwrap();
        let ids_a: Vec<_> = hits_a.iter().map(|h| h.id.clone()).collect();
        let ids_b: Vec<_> = hits_b.iter().map(|h| h.id.clone()).collect();
        if ids_a != ids_b {
            any_difference = true;
            break;
        }
    }
    assert!(
        any_difference,
        "different seeds should be able to produce different ADC-only top-k orderings on a non-trivial corpus"
    );
}
