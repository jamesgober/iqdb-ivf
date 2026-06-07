//! The shortest end-to-end use of `iqdb-ivf`: configure, train, insert,
//! search, inspect. This is the Tier-1 path.
//!
//! Run with:
//!
//! ```sh
//! cargo run --example quick_start
//! ```

use std::sync::Arc;

use iqdb_index::{Index, IndexCore};
use iqdb_ivf::{IvfConfig, IvfIndex};
use iqdb_types::{DistanceMetric, Result, SearchParams, VectorId};

fn arc(v: &[f32]) -> Arc<[f32]> {
    Arc::from(v)
}

fn main() -> Result<()> {
    // A 2-D Euclidean IVF index with two clusters. Probing both clusters
    // makes this small example exact; on a real corpus you would probe a
    // small fraction of many clusters.
    let cfg = IvfConfig::default()
        .with_n_clusters(2)
        .with_n_probes(2)
        .with_training_sample_size(64)
        .with_seed(7);
    let mut idx = IvfIndex::new(2, DistanceMetric::Euclidean, cfg)?;

    // IVF must learn its partitions before it can index anything. Train on
    // a representative sample of the data distribution.
    let sample: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0],
        vec![0.2, -0.1],
        vec![-0.1, 0.1],
        vec![10.0, 10.0],
        vec![10.2, 9.8],
        vec![9.9, 10.1],
    ];
    let sample_refs: Vec<&[f32]> = sample.iter().map(|v| v.as_slice()).collect();
    idx.train(&sample_refs)?;

    // Now insert the corpus. Each vector is assigned to its nearest cluster.
    idx.insert(VectorId::from(1u64), arc(&[0.0, 0.0]), None)?;
    idx.insert(VectorId::from(2u64), arc(&[10.0, 10.0]), None)?;
    idx.insert(VectorId::from(3u64), arc(&[0.1, 0.1]), None)?;

    let hits = idx.search(
        &[0.0, 0.0],
        &SearchParams::new(2, DistanceMetric::Euclidean),
    )?;

    println!("nearest 2 to (0, 0):");
    for (rank, hit) in hits.iter().enumerate() {
        println!("  #{rank}: id={} distance={:.3}", hit.id, hit.distance);
    }

    // Nearest is the exact match at the origin; second is its neighbour.
    assert_eq!(hits[0].id, VectorId::U64(1));
    assert_eq!(hits[1].id, VectorId::U64(3));

    let stats = idx.stats();
    println!(
        "index_type={} n_vectors={} ~memory_bytes={}",
        stats.index_type, stats.n_vectors, stats.memory_bytes
    );
    assert_eq!(stats.index_type, "ivf");
    assert_eq!(idx.len(), 3);

    Ok(())
}
