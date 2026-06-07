//! Filtered IVF search correctness.
//!
//! Pins the filtered-search contract across both variants:
//!
//! - IVF-Flat with `n_probes == n_clusters` matches the exact
//!   `iqdb_flat::FlatIndex` oracle on the same corpus + filter.
//! - IVF-PQ with `n_probes == n_clusters` matches the oracle on the
//!   filter-survivor set (ranking can diverge under PQ lossiness).
//! - The `filter = None` hot path returns the same results as the
//!   unfiltered scan on a representative population.
//! - With `pq_refine_factor > 0`, no returned `Hit` violates the
//!   filter — refine cannot leak excluded entries through the
//!   shortlist.
//! - Pathological filters surface as `IqdbError::InvalidFilter`
//!   from `FilterEvaluator::new`.

#![allow(clippy::unwrap_used)]

use std::collections::HashSet;
use std::sync::Arc;

use iqdb_filter::MAX_FILTER_DEPTH;
use iqdb_flat::{FlatConfig, FlatIndex};
use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, Filter, IqdbError, Metadata, SearchParams, Value, VectorId};

const DIM: usize = 4;
const N_CLUSTERS: usize = 4;
// >= `iqdb_quantize::ProductQuantizer`'s `K = 256` codebook size so
// PQ training has enough representatives. The Flat oracle and the
// IVF-Flat path are insensitive to corpus size.
const N_VECTORS: usize = 384;
const TOPK: usize = 8;

/// Deterministic SplitMix64-style PRNG for the corpus generator.
/// Identical seed → identical corpus → tests are reproducible.
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

/// Generate `N_VECTORS` vectors clumped around 4 well-separated
/// centres, with two metadata fields: `colour` (one of
/// "red"/"green"/"blue") and `year` (an `Int`). Returns vectors +
/// metadata; the two oracles see the same data.
fn corpus(seed: u64) -> Vec<(Vec<f32>, Metadata)> {
    let mut state = seed.wrapping_add(0xDEAD_BEEF_CAFE_F00D);
    let colours = ["red", "green", "blue"];
    let centres: Vec<Vec<f32>> = (0..N_CLUSTERS)
        .map(|c| {
            let base = c as f32 * 5.0;
            (0..DIM).map(|i| base + i as f32 * 0.1).collect()
        })
        .collect();
    (0..N_VECTORS)
        .map(|i| {
            let centre = &centres[i % centres.len()];
            let vector: Vec<f32> = centre
                .iter()
                .map(|&c| c + (unit_float(&mut state) * 2.0 - 1.0) * 0.4)
                .collect();
            let meta: Metadata = [
                (
                    "colour".to_string(),
                    Value::String(colours[i % colours.len()].to_string()),
                ),
                ("year".to_string(), Value::Int(2000 + (i % 30) as i64)),
            ]
            .into_iter()
            .collect();
            (vector, meta)
        })
        .collect()
}

fn populate_flat(data: &[(Vec<f32>, Metadata)]) -> FlatIndex {
    let mut idx = FlatIndex::new(DIM, DistanceMetric::Euclidean, FlatConfig).unwrap();
    for (i, (v, meta)) in data.iter().enumerate() {
        idx.insert(
            VectorId::from(i as u64),
            Arc::<[f32]>::from(v.as_slice()),
            Some(meta.clone()),
        )
        .unwrap();
    }
    idx
}

fn populate_ivf(data: &[(Vec<f32>, Metadata)], use_pq: bool) -> IvfIndex {
    let cfg_base = IvfConfig::default()
        .with_n_clusters(N_CLUSTERS)
        .with_n_probes(N_CLUSTERS) // exhaustive so the comparison can be set-equal.
        .with_training_sample_size(N_VECTORS)
        .with_seed(0xC0FF_EE42);
    let cfg = if use_pq {
        cfg_base
            .with_use_pq(true)
            .with_pq_subvectors(Some(2))
            .with_pq_refine_factor(4)
    } else {
        cfg_base
    };
    let mut idx = IvfIndex::new(DIM, DistanceMetric::Euclidean, cfg).unwrap();
    let refs: Vec<&[f32]> = data.iter().map(|(v, _)| v.as_slice()).collect();
    idx.train(&refs).unwrap();
    for (i, (v, meta)) in data.iter().enumerate() {
        idx.insert(
            VectorId::from(i as u64),
            Arc::<[f32]>::from(v.as_slice()),
            Some(meta.clone()),
        )
        .unwrap();
    }
    idx
}

fn id_set(hits: &[iqdb_types::Hit]) -> HashSet<VectorId> {
    hits.iter().map(|h| h.id.clone()).collect()
}

#[test]
fn filtered_search_matches_flat_oracle_ivf_flat() {
    // Population, filter, and query are all deterministic so the
    // assertion is stable.
    let data = corpus(1);
    let flat = populate_flat(&data);
    let ivf = populate_ivf(&data, /*use_pq=*/ false);

    let filter = Filter::and(vec![
        Filter::eq("colour", Value::String("blue".to_string())),
        Filter::gt("year", Value::Int(2010)),
    ]);
    let mut params = SearchParams::new(TOPK, DistanceMetric::Euclidean);
    params.filter = Some(filter);

    let query: Vec<f32> = (0..DIM).map(|i| i as f32 * 0.1).collect();
    let flat_hits = flat.search(&query, &params).unwrap();
    let ivf_hits = ivf.search(&query, &params).unwrap();

    // n_probes == n_clusters → IVF-Flat is exhaustive. Same survivors
    // + same metric + same tiebreaker (seq) → same id sequence.
    let flat_ids: Vec<VectorId> = flat_hits.iter().map(|h| h.id.clone()).collect();
    let ivf_ids: Vec<VectorId> = ivf_hits.iter().map(|h| h.id.clone()).collect();
    assert_eq!(
        ivf_ids, flat_ids,
        "exhaustive IVF-Flat must match the FlatIndex oracle byte-for-byte"
    );
}

#[test]
fn filtered_search_matches_flat_oracle_ivf_pq() {
    // PQ is lossy — ranking can differ from the exact oracle — but
    // when IVF-PQ probes every cluster the *survivor set* must
    // match: the filter rejects the same entries regardless of
    // distance estimate.
    let data = corpus(2);
    let flat = populate_flat(&data);
    let ivf = populate_ivf(&data, /*use_pq=*/ true);

    let filter = Filter::is_in(
        "colour",
        vec![
            Value::String("red".to_string()),
            Value::String("green".to_string()),
        ],
    );
    let mut params = SearchParams::new(TOPK, DistanceMetric::Euclidean);
    params.filter = Some(filter);

    let query: Vec<f32> = (0..DIM).map(|i| (i + 5) as f32 * 0.1).collect();
    let ivf_hits = ivf.search(&query, &params).unwrap();

    // Compute the full filter-survivor set on the flat oracle by
    // searching with k = N_VECTORS — every IVF-PQ hit must live
    // inside that set.
    let mut wide = params.clone();
    wide.k = N_VECTORS;
    let flat_survivors = id_set(&flat.search(&query, &wide).unwrap());
    let ivf_ids = id_set(&ivf_hits);
    for id in &ivf_ids {
        assert!(
            flat_survivors.contains(id),
            "IVF-PQ returned a Hit that should have been filtered out: {id:?}",
        );
    }
}

#[test]
fn filter_none_path_unchanged() {
    // The filter-None unfiltered path: same corpus, same query,
    // expected order matches the exact flat oracle when IVF-Flat is
    // exhaustive.
    let data = corpus(3);
    let ivf = populate_ivf(&data, /*use_pq=*/ false);

    let params = SearchParams::new(TOPK, DistanceMetric::Euclidean);
    let query: Vec<f32> = (0..DIM).map(|i| i as f32 * 0.1).collect();
    let hits = ivf.search(&query, &params).unwrap();
    assert_eq!(hits.len(), TOPK);
    let flat = populate_flat(&data);
    let flat_hits = flat.search(&query, &params).unwrap();
    let ivf_ids: Vec<VectorId> = hits.iter().map(|h| h.id.clone()).collect();
    let flat_ids: Vec<VectorId> = flat_hits.iter().map(|h| h.id.clone()).collect();
    assert_eq!(ivf_ids, flat_ids);
}

#[test]
fn pq_refine_respects_filter() {
    // With `pq_refine_factor > 0` (default 4), the refine shortlist
    // is drawn from `cand_addrs`, which only contains filter
    // survivors. The contract: no returned Hit violates the filter.
    let data = corpus(4);
    let ivf = populate_ivf(&data, /*use_pq=*/ true);

    let filter = Filter::eq("colour", Value::String("red".to_string()));
    let mut params = SearchParams::new(TOPK, DistanceMetric::Euclidean);
    params.filter = Some(filter);

    let query: Vec<f32> = (0..DIM).map(|i| i as f32 * 0.05).collect();
    let hits = ivf.search(&query, &params).unwrap();
    for hit in &hits {
        let meta = hit.metadata.as_ref().expect("populated with metadata");
        let colour = meta.get("colour").expect("colour field present");
        assert_eq!(
            colour,
            &Value::String("red".to_string()),
            "PQ refine returned a Hit that should have been filtered out: {:?}",
            hit,
        );
    }
}

#[test]
fn pathological_filter_surfaces_invalid_filter() {
    // Build a filter deeper than the evaluator's depth cap. The
    // error must come from `FilterEvaluator::new` (the validation
    // gate), before any cluster data is touched.
    let data = corpus(5);
    let ivf = populate_ivf(&data, /*use_pq=*/ false);

    let mut deep = Filter::eq("colour", Value::String("blue".to_string()));
    for _ in 0..=MAX_FILTER_DEPTH {
        deep = Filter::not(deep);
    }
    let mut params = SearchParams::new(TOPK, DistanceMetric::Euclidean);
    params.filter = Some(deep);

    let query: Vec<f32> = (0..DIM).map(|i| i as f32 * 0.1).collect();
    let err = ivf.search(&query, &params).unwrap_err();
    assert!(
        matches!(err, IqdbError::InvalidFilter),
        "expected InvalidFilter from FilterEvaluator::new, got {err:?}",
    );
}
