//! Tuning the recall/latency dial: `n_probes`.
//!
//! `n_probes` is the number of clusters each query scans. More probes →
//! higher recall, higher latency. `suggest_n_probes(coverage)` reads the
//! live cluster-size distribution and recommends the smallest probe count
//! that covers the requested fraction of vectors; you apply it with
//! `set_n_probes`. The probe count stays explicit, so latency stays
//! predictable — IVF never widens the probe set behind your back.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example probe_tuning
//! ```

use std::sync::Arc;

use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, Result, SearchParams, VectorId};

const DIM: usize = 8;

fn arc(v: &[f32]) -> Arc<[f32]> {
    Arc::from(v)
}

fn row(i: usize) -> Vec<f32> {
    let c = (i % 10) as f32;
    (0..DIM)
        .map(|j| c * 6.0 + (((i * 19 + j * 7) % 100) as f32 - 50.0) * 0.02)
        .collect()
}

fn main() -> Result<()> {
    let metric = DistanceMetric::Euclidean;
    let cfg = IvfConfig::default()
        .with_n_clusters(16)
        .with_n_probes(1)
        .with_training_sample_size(4096)
        .with_seed(2024);
    let mut idx = IvfIndex::new(DIM, metric, cfg)?;

    let rows: Vec<Vec<f32>> = (0..1000).map(row).collect();
    let refs: Vec<&[f32]> = rows.iter().map(|v| v.as_slice()).collect();
    idx.train(&refs)?;
    for (i, v) in rows.iter().enumerate() {
        idx.insert(VectorId::from(i as u64), arc(v), None)?;
    }

    let stats = idx.cluster_stats();
    println!(
        "clusters={} avg_size={:.1} variance={:.1}",
        stats.n_clusters, stats.avg_size, stats.size_variance
    );

    // Recommendation rises with the requested coverage.
    for coverage in [0.0_f32, 0.5, 0.8, 0.95, 1.0] {
        let suggested = idx.suggest_n_probes(coverage)?;
        println!("coverage {coverage:.2} -> suggest {suggested} probes");
    }

    // Apply the 90%-coverage recommendation and query with it.
    let probes = idx.suggest_n_probes(0.90)?;
    idx.set_n_probes(probes)?;
    println!("applied n_probes = {} ({})", idx.n_probes(), probes);

    let hits = idx.search(&row(7), &SearchParams::new(10, metric))?;
    println!("top-10 with {probes} probes: {} hits", hits.len());
    assert!(!hits.is_empty());

    Ok(())
}
