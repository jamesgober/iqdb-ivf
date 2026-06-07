//! Verifies the IVF-PQ ADC pass applies the same DotProduct
//! negation as IVF-Flat. Both `pq.distance(query, code,
//! DotProduct)` and `iqdb_distance::compute(DotProduct, q, v)`
//! return the raw inner product (larger = more similar); the
//! IVF-PQ scan negates after `PqAdcTables::distance` exactly as
//! IVF-Flat negates after `compute_batch`. Result: IVF-PQ under
//! DotProduct must return the same top hit as the exact baseline,
//! and `Hit::distance` is negative for any positively-correlated
//! query / vector pair.
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

fn pos_dotproduct_corpus(seed: u64, n: usize) -> Vec<Vec<f32>> {
    // Strictly-positive vectors so the dot product with a positive
    // query is positive — exercises the "raw dot > 0 → distance < 0"
    // case after negation.
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    (0..n)
        .map(|_| {
            (0..DIM)
                .map(|_| {
                    let u = (next() >> 11) as f32 / (1u64 << 53) as f32;
                    1.0 + u * 4.0 // [1, 5]
                })
                .collect()
        })
        .collect()
}

fn populate_flat(data: &[Vec<f32>], metric: DistanceMetric) -> FlatIndex {
    let mut idx = FlatIndex::new(DIM, metric, FlatConfig).unwrap();
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

fn populate_ivf_pq(data: &[Vec<f32>], metric: DistanceMetric) -> IvfIndex {
    let cfg = IvfConfig::default()
        .with_n_clusters(4)
        .with_n_probes(4)
        .with_training_sample_size(N_TRAIN)
        .with_use_pq(true)
        .with_pq_subvectors(Some(M))
        .with_pq_refine_factor(4) // refine ON to mimic the operating point
        .with_seed(17);
    let mut idx = IvfIndex::new(DIM, metric, cfg).unwrap();
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
fn pq_top_hit_under_dotproduct_matches_flat() {
    let metric = DistanceMetric::DotProduct;
    let data = pos_dotproduct_corpus(3, N_TRAIN);
    let flat = populate_flat(&data, metric);
    let ivf = populate_ivf_pq(&data, metric);

    // A query with all-positive coordinates — every stored vector is
    // also strictly positive, so the raw dot product is always > 0
    // and the negated distance is always < 0.
    let query = vec![1.0_f32; DIM];
    let flat_hit = flat.search(&query, &SearchParams::new(1, metric)).unwrap();
    let ivf_hit = ivf.search(&query, &SearchParams::new(1, metric)).unwrap();
    assert_eq!(flat_hit.len(), 1);
    assert_eq!(ivf_hit.len(), 1);
    assert_eq!(
        flat_hit[0].id, ivf_hit[0].id,
        "IVF-PQ top hit must match flat under DotProduct (with refine ON)"
    );
    // Negation produces a negative distance (raw dot > 0 → -dot < 0).
    assert!(
        flat_hit[0].distance < 0.0,
        "negated DotProduct should produce a negative distance; got {}",
        flat_hit[0].distance,
    );
    assert!(
        ivf_hit[0].distance < 0.0,
        "negated DotProduct should produce a negative distance; got {}",
        ivf_hit[0].distance,
    );
}

#[test]
fn pq_topk_overlap_under_dotproduct_above_threshold() {
    let metric = DistanceMetric::DotProduct;
    let data = pos_dotproduct_corpus(19, N_TRAIN);
    let flat = populate_flat(&data, metric);
    let ivf = populate_ivf_pq(&data, metric);

    let params = SearchParams::new(10, metric);
    let mut hits = 0_usize;
    let mut total = 0_usize;
    for i in (0..N_TRAIN).step_by(32) {
        let q = &data[i];
        let flat_ids: HashSet<_> = flat
            .search(q, &params)
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        let ivf_ids: HashSet<_> = ivf
            .search(q, &params)
            .unwrap()
            .into_iter()
            .map(|h| h.id)
            .collect();
        hits += flat_ids.intersection(&ivf_ids).count();
        total += flat_ids.len();
    }
    let recall = (hits as f32) / (total as f32);
    assert!(
        recall >= 0.6,
        "IVF-PQ DotProduct recall@10 = {recall} should be >= 0.6 with refine=4"
    );
}
