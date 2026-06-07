//! Metadata-filtered IVF search.
//!
//! A search can carry a `Filter` in its `SearchParams`. IVF evaluates the
//! predicate per entry *before* computing distance, so a selective filter
//! does proportionally less distance work. The filter is validated once on
//! entry — a pathological filter surfaces as `IqdbError::InvalidFilter`
//! before any cluster data is touched.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example filtered_search
//! ```

use std::sync::Arc;

use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, Filter, Metadata, Result, SearchParams, Value, VectorId};

fn arc(v: &[f32]) -> Arc<[f32]> {
    Arc::from(v)
}

fn meta(tier: &str) -> Metadata {
    [("tier".to_string(), Value::String(tier.to_string()))]
        .into_iter()
        .collect()
}

fn main() -> Result<()> {
    let metric = DistanceMetric::Euclidean;
    let cfg = IvfConfig::default()
        .with_n_clusters(2)
        .with_n_probes(2)
        .with_training_sample_size(64)
        .with_seed(7);
    let mut idx = IvfIndex::new(2, metric, cfg)?;

    let sample: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0],
        vec![0.5, 0.5],
        vec![1.0, 1.0],
        vec![9.0, 9.0],
        vec![10.0, 10.0],
        vec![11.0, 11.0],
    ];
    let sample_refs: Vec<&[f32]> = sample.iter().map(|v| v.as_slice()).collect();
    idx.train(&sample_refs)?;

    // The two closest vectors to the origin are "cold"; the "hot" match is
    // slightly farther. The filter changes which one wins.
    idx.insert(VectorId::from(1u64), arc(&[0.0, 0.0]), Some(meta("cold")))?;
    idx.insert(VectorId::from(2u64), arc(&[0.5, 0.5]), Some(meta("cold")))?;
    idx.insert(VectorId::from(3u64), arc(&[1.0, 1.0]), Some(meta("hot")))?;

    let params = SearchParams::new(1, metric);
    let unfiltered = idx.search(&[0.0, 0.0], &params)?;
    println!("unfiltered nearest: id={}", unfiltered[0].id);
    assert_eq!(unfiltered[0].id, VectorId::U64(1));

    let filtered_params = SearchParams {
        filter: Some(Filter::eq("tier", Value::String("hot".to_string()))),
        ..SearchParams::new(1, metric)
    };
    let filtered = idx.search(&[0.0, 0.0], &filtered_params)?;
    println!("nearest with tier=hot: id={}", filtered[0].id);
    assert_eq!(filtered[0].id, VectorId::U64(3));

    // Every returned hit carries its metadata.
    let returned_tier = filtered[0]
        .metadata
        .as_ref()
        .and_then(|m| m.get("tier"))
        .cloned();
    assert_eq!(returned_tier, Some(Value::String("hot".to_string())));

    Ok(())
}
