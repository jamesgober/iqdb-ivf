//! IVF-PQ: Product-Quantization inside each inverted list.
//!
//! Setting `use_pq = true` compresses every stored vector to an `M`-byte
//! Product-Quantization code and scores probed clusters by asymmetric
//! distance (ADC). With a non-zero refine factor the search shortlists by
//! ADC and exact-reranks against the retained vectors, recovering most of
//! the accuracy of IVF-Flat at a fraction of the scan cost.
//!
//! Product quantization trains a 256-entry codebook per subvector, so the
//! training sample must contain at least 256 vectors.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example ivf_pq
//! ```

use std::sync::Arc;

use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, Result, SearchParams, VectorId};

const DIM: usize = 8;
const N: usize = 600;

fn arc(v: &[f32]) -> Arc<[f32]> {
    Arc::from(v)
}

/// Deterministic clustered row over 6 latent centres.
fn row(i: usize) -> Vec<f32> {
    let c = (i % 6) as f32;
    (0..DIM)
        .map(|j| c * 5.0 + (((i * 17 + j * 31) % 100) as f32 - 50.0) * 0.02)
        .collect()
}

fn main() -> Result<()> {
    let metric = DistanceMetric::Euclidean;
    let cfg = IvfConfig::default()
        .with_n_clusters(8)
        .with_n_probes(8) // probe all clusters for this demo
        .with_training_sample_size(4096)
        .with_use_pq(true)
        .with_pq_subvectors(Some(4)) // 4 subvectors over 8 dims → 2 dims each
        .with_pq_refine_factor(4)
        .with_seed(0x1234);

    let mut idx = IvfIndex::new(DIM, metric, cfg)?;

    let rows: Vec<Vec<f32>> = (0..N).map(row).collect();
    let refs: Vec<&[f32]> = rows.iter().map(|v| v.as_slice()).collect();
    idx.train(&refs)?;
    for (i, v) in rows.iter().enumerate() {
        idx.insert(VectorId::from(i as u64), arc(v), None)?;
    }

    // Query near latent cluster 3.
    let query = row(3);
    let hits = idx.search(&query, &SearchParams::new(5, metric))?;

    println!("IVF-PQ top-5 near cluster 3:");
    for (rank, hit) in hits.iter().enumerate() {
        println!("  #{rank}: id={} distance={:.4}", hit.id, hit.distance);
    }
    assert_eq!(hits.len(), 5);

    // Refine is tunable at runtime without a rebuild. Setting it to 0
    // returns the pure ADC top-k (faster, slightly less accurate).
    idx.set_pq_refine_factor(0);
    let adc_only = idx.search(&query, &SearchParams::new(5, metric))?;
    println!("pure-ADC top-5 (refine disabled): {} hits", adc_only.len());
    assert_eq!(adc_only.len(), 5);

    let stats = idx.stats();
    println!(
        "index_type={} n_vectors={}",
        stats.index_type, stats.n_vectors
    );

    Ok(())
}
