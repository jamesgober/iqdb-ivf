//! Rebuilding cluster balance with `retrain`.
//!
//! Deletes are cluster-local `swap_remove`s, so after a delete-heavy phase
//! (or a shift in the inserted distribution) the inverted lists can drift
//! out of balance. `retrain` rebuilds the centroids — and, in IVF-PQ mode,
//! the codebooks — over the currently-stored vectors and reassigns every
//! entry, without losing any data.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example retrain
//! ```

use std::sync::Arc;

use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, Result, SearchParams, VectorId};

const DIM: usize = 4;

fn arc(v: &[f32]) -> Arc<[f32]> {
    Arc::from(v)
}

/// Row in one of three latent clusters.
fn row(i: usize) -> Vec<f32> {
    let c = (i % 3) as f32;
    (0..DIM)
        .map(|j| c * 8.0 + (((i * 13 + j * 29) % 100) as f32 - 50.0) * 0.01)
        .collect()
}

fn main() -> Result<()> {
    let metric = DistanceMetric::Euclidean;
    let cfg = IvfConfig::default()
        .with_n_clusters(6)
        .with_n_probes(6)
        .with_training_sample_size(4096)
        .with_seed(99);
    let mut idx = IvfIndex::new(DIM, metric, cfg)?;

    let rows: Vec<Vec<f32>> = (0..300).map(row).collect();
    let refs: Vec<&[f32]> = rows.iter().map(|v| v.as_slice()).collect();
    idx.train(&refs)?;
    for (i, v) in rows.iter().enumerate() {
        idx.insert(VectorId::from(i as u64), arc(v), None)?;
    }

    let before = idx.cluster_stats();
    println!(
        "before deletes: sizes={:?} variance={:.2}",
        before.cluster_sizes, before.size_variance
    );

    // Delete every vector from latent cluster 0 — a heavy, skewed delete.
    for i in (0..300).filter(|i| i % 3 == 0) {
        idx.delete(&VectorId::from(i as u64))?;
    }
    let after_delete = idx.cluster_stats();
    println!(
        "after deletes:  sizes={:?} variance={:.2}",
        after_delete.cluster_sizes, after_delete.size_variance
    );

    // Rebuild centroids over what remains; no data is lost.
    let live_before = idx.len();
    idx.retrain()?;
    assert_eq!(
        idx.len(),
        live_before,
        "retrain preserves every live vector"
    );

    let after_retrain = idx.cluster_stats();
    println!(
        "after retrain:  sizes={:?} variance={:.2}",
        after_retrain.cluster_sizes, after_retrain.size_variance
    );

    // The index still answers queries correctly after the rebuild.
    let hits = idx.search(&row(1), &SearchParams::new(3, metric))?;
    println!("post-retrain top-3: {} hits", hits.len());
    assert_eq!(hits.len(), 3);

    Ok(())
}
