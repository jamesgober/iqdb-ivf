//! [`IvfClusterStats`] — diagnostic snapshot of cluster occupancy.
//!
//! Exposed through [`crate::IvfIndex::cluster_stats`] for operators
//! debugging cluster imbalance. A healthy IVF index has roughly
//! uniform `cluster_sizes`; a long tail of empty or near-empty
//! clusters next to a few packed ones indicates centroid drift and is
//! the signal that motivates the `retrain()` workflow.

/// Diagnostic snapshot of inverted-list occupancy.
///
/// Returned by [`crate::IvfIndex::cluster_stats`] after the index has
/// been trained and (typically) populated. Before training the
/// `cluster_sizes` slice is empty and `avg_size` / `size_variance`
/// are `0.0`.
///
/// # Examples
///
/// ```
/// use iqdb_ivf::IvfClusterStats;
///
/// // Synthetic example — IvfIndex::cluster_stats returns the real one.
/// let stats = IvfClusterStats {
///     n_clusters: 4,
///     cluster_sizes: vec![3, 3, 4, 2],
///     avg_size: 3.0,
///     size_variance: 0.5,
/// };
/// assert_eq!(stats.cluster_sizes.iter().sum::<usize>(), 12);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct IvfClusterStats {
    /// Number of partitions the index was configured with.
    ///
    /// Echoed from [`crate::IvfConfig::n_clusters`] so a caller does
    /// not need to also carry the config to interpret the snapshot.
    pub n_clusters: usize,

    /// Live vector count per cluster, indexed by cluster id.
    ///
    /// Always either empty (before training) or of length
    /// [`Self::n_clusters`]. A cluster size of `0` is meaningful and
    /// is what motivates `retrain()` work in a follow-up.
    pub cluster_sizes: Vec<usize>,

    /// Arithmetic mean of [`Self::cluster_sizes`] as `f32`.
    ///
    /// `0.0` when [`Self::cluster_sizes`] is empty.
    pub avg_size: f32,

    /// Population variance of [`Self::cluster_sizes`] as `f32`.
    ///
    /// `0.0` when [`Self::cluster_sizes`] is empty. Higher values
    /// signal cluster imbalance.
    pub size_variance: f32,
}

impl IvfClusterStats {
    /// Build an `IvfClusterStats` from the cluster-size vector.
    ///
    /// Computes `avg_size` and `size_variance` in a single sequential
    /// pass over `cluster_sizes`, in fixed iteration order, so the
    /// snapshot is deterministic and free of any cross-platform
    /// float-reduction drift.
    #[must_use]
    pub(crate) fn from_sizes(n_clusters: usize, cluster_sizes: Vec<usize>) -> Self {
        let len = cluster_sizes.len();
        if len == 0 {
            return Self {
                n_clusters,
                cluster_sizes,
                avg_size: 0.0,
                size_variance: 0.0,
            };
        }
        // f64 reductions for cross-platform-stable arithmetic; downcast
        // at the end. Matches the train.rs reduction discipline.
        let mut sum: f64 = 0.0;
        for &s in &cluster_sizes {
            sum += s as f64;
        }
        let avg = sum / (len as f64);
        let mut var_sum: f64 = 0.0;
        for &s in &cluster_sizes {
            let d = (s as f64) - avg;
            var_sum += d * d;
        }
        let variance = var_sum / (len as f64);
        Self {
            n_clusters,
            cluster_sizes,
            avg_size: avg as f32,
            size_variance: variance as f32,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn empty_sizes_yields_zero_avg_and_variance() {
        let stats = IvfClusterStats::from_sizes(4, Vec::new());
        assert_eq!(stats.n_clusters, 4);
        assert!(stats.cluster_sizes.is_empty());
        assert_eq!(stats.avg_size, 0.0);
        assert_eq!(stats.size_variance, 0.0);
    }

    #[test]
    fn uniform_sizes_have_zero_variance() {
        let stats = IvfClusterStats::from_sizes(4, vec![5, 5, 5, 5]);
        assert_eq!(stats.avg_size, 5.0);
        assert_eq!(stats.size_variance, 0.0);
    }

    #[test]
    fn variance_matches_hand_computation() {
        // sizes: [1, 3, 5, 7]; mean = 4; deviations = [-3, -1, 1, 3];
        // variance = (9 + 1 + 1 + 9) / 4 = 5.
        let stats = IvfClusterStats::from_sizes(4, vec![1, 3, 5, 7]);
        assert!((stats.avg_size - 4.0).abs() < 1e-6);
        assert!((stats.size_variance - 5.0).abs() < 1e-6);
    }
}
