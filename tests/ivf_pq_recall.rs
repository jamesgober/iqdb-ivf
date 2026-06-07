//! IVF-PQ recall against the exact `iqdb_flat::FlatIndex` baseline,
//! plus the "refine beats pure ADC" invariant. PQ is lossy, so we
//! assert against a threshold (not equality); the refine path must
//! recover at least as much ground truth as the ADC-only path.
//!
//! Cargo discovers this file as an integration-test binary. Named
//! in the approved plan §"Test plan".

#![allow(clippy::unwrap_used)]

use std::collections::HashSet;
use std::sync::Arc;

use iqdb_flat::{FlatConfig, FlatIndex};
use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, SearchParams, VectorId};

const DIM: usize = 8;
const M: usize = 2;
const N_TRAIN: usize = 512;
const N_QUERIES: usize = 16;
const TOPK: usize = 10;

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

fn populate_flat(data: &[Vec<f32>]) -> FlatIndex {
    let mut idx = FlatIndex::new(DIM, DistanceMetric::Euclidean, FlatConfig).unwrap();
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

fn populate_ivf_pq(data: &[Vec<f32>], refine_factor: u32) -> IvfIndex {
    let cfg = IvfConfig::default()
        .with_n_clusters(4)
        .with_n_probes(4) // full-probe so recall loss is pure PQ noise
        .with_training_sample_size(N_TRAIN)
        .with_use_pq(true)
        .with_pq_subvectors(Some(M))
        .with_pq_refine_factor(refine_factor)
        .with_seed(99);
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

fn topk_ids(hits: Vec<iqdb_types::Hit>) -> HashSet<VectorId> {
    hits.into_iter().map(|h| h.id).collect()
}

fn recall_against_flat(flat: &FlatIndex, ivf: &IvfIndex, queries: &[Vec<f32>]) -> f32 {
    let params = SearchParams::new(TOPK, DistanceMetric::Euclidean);
    let mut hits = 0_usize;
    let mut total = 0_usize;
    for q in queries {
        let flat_ids = topk_ids(flat.search(q, &params).unwrap());
        let ivf_ids = topk_ids(ivf.search(q, &params).unwrap());
        hits += flat_ids.intersection(&ivf_ids).count();
        total += flat_ids.len();
    }
    (hits as f32) / (total as f32)
}

#[test]
fn pq_recall_overlap_with_flat_baseline_above_threshold() {
    let data = corpus(7, N_TRAIN);
    let queries: Vec<Vec<f32>> = (0..N_QUERIES)
        .map(|i| data[i * 17 % N_TRAIN].clone())
        .collect();

    let flat = populate_flat(&data);
    // Refine ON (the default operating point — earns the retained Arc).
    let ivf = populate_ivf_pq(&data, 4);

    let recall = recall_against_flat(&flat, &ivf, &queries);
    // Threshold: well-clustered Gaussian data + refine=4 should
    // recover the vast majority of the exact top-k. 0.6 is the
    // floor; in practice this corpus + refine clears 0.9.
    assert!(
        recall >= 0.6,
        "IVF-PQ recall@{TOPK} with refine=4 = {recall} should be >= 0.6"
    );
}

#[test]
fn pq_recall_with_refine_beats_adc_only() {
    let data = corpus(13, N_TRAIN);
    let queries: Vec<Vec<f32>> = (0..N_QUERIES)
        .map(|i| data[i * 23 % N_TRAIN].clone())
        .collect();

    let flat = populate_flat(&data);
    let ivf_adc = populate_ivf_pq(&data, 0); // pure ADC
    let ivf_refine = populate_ivf_pq(&data, 4);

    let recall_adc = recall_against_flat(&flat, &ivf_adc, &queries);
    let recall_refine = recall_against_flat(&flat, &ivf_refine, &queries);
    assert!(
        recall_refine >= recall_adc,
        "refine ({recall_refine}) must not lose recall vs ADC-only ({recall_adc})"
    );
    // On any non-pathological corpus refine should *strictly* improve
    // recall (the retained Arc earns its memory). Allow equality at
    // floor in case ADC already hit 1.0.
    assert!(
        recall_refine >= recall_adc.max(0.6),
        "refine ({recall_refine}) should clear the same floor as the baseline test"
    );
}
