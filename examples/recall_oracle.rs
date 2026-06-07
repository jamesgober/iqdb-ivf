//! Measuring IVF recall against the exact `iqdb_flat::FlatIndex` oracle.
//!
//! `FlatIndex` scans every vector on every query, so its results are the
//! ground truth. This example builds the same corpus in both indexes and
//! reports recall@k for IVF-Flat at a few probe counts — the empirical
//! version of the recall/latency tradeoff `n_probes` controls.
//!
//! `iqdb-flat` is a dev-dependency, so this lives as an example rather than
//! in the shipped crate.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example recall_oracle
//! ```

use std::collections::HashSet;
use std::sync::Arc;

use iqdb_flat::{FlatConfig, FlatIndex};
use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, SearchParams, VectorId};

const DIM: usize = 16;
const N: usize = 2000;
const N_CLUSTERS: usize = 32;
const METRIC: DistanceMetric = DistanceMetric::Euclidean;

fn arc(v: &[f32]) -> Arc<[f32]> {
    Arc::from(v)
}

fn row(i: usize) -> Vec<f32> {
    let c = (i % 20) as u64;
    (0..DIM)
        .map(|j| {
            let base = (c
                .wrapping_mul(2_654_435_761)
                .wrapping_add(j as u64 * 40_503)
                % 1_000) as f32
                / 1_000.0
                * 20.0;
            base + (((i * 131 + j * 71) % 100) as f32 - 50.0) * 0.006
        })
        .collect()
}

fn main() {
    let rows: Vec<Vec<f32>> = (0..N).map(row).collect();

    // Exact oracle.
    let mut flat = FlatIndex::new(DIM, METRIC, FlatConfig).expect("valid dim");
    for (i, v) in rows.iter().enumerate() {
        flat.insert(VectorId::from(i as u64), arc(v), None)
            .expect("fresh id");
    }

    let queries: Vec<Vec<f32>> = (0..20).map(row).collect();
    let k = 10;
    let params = SearchParams::new(k, METRIC);

    println!("recall@{k} vs exact flat oracle (n={N}, clusters={N_CLUSTERS}):");
    for n_probes in [1usize, 2, 4, 8, N_CLUSTERS] {
        let cfg = IvfConfig::default()
            .with_n_clusters(N_CLUSTERS)
            .with_n_probes(n_probes)
            .with_training_sample_size(4096)
            .with_seed(0xABCD);
        let mut ivf = IvfIndex::new(DIM, METRIC, cfg).expect("valid config");
        let refs: Vec<&[f32]> = rows.iter().map(|v| v.as_slice()).collect();
        ivf.train(&refs).expect("train succeeds");
        for (i, v) in rows.iter().enumerate() {
            ivf.insert(VectorId::from(i as u64), arc(v), None)
                .expect("fresh id");
        }

        let mut hit = 0usize;
        let mut total = 0usize;
        for q in &queries {
            let truth: HashSet<VectorId> = flat
                .search(q, &params)
                .expect("search")
                .into_iter()
                .map(|h| h.id)
                .collect();
            let got: HashSet<VectorId> = ivf
                .search(q, &params)
                .expect("search")
                .into_iter()
                .map(|h| h.id)
                .collect();
            hit += truth.intersection(&got).count();
            total += truth.len();
        }
        let recall = (hit as f32) / (total as f32);
        println!("  n_probes={n_probes:>3}  recall={recall:.3}");
    }
}
