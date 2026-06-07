//! Nearest-centroid assignment for IVF.
//!
//! K-means is mathematically tied to Euclidean (L2) centroids: the
//! centroid that minimizes the sum of within-cluster distances is the
//! arithmetic mean only under L2. This module therefore always assigns
//! in **squared L2** space regardless of the index's configured search
//! metric — the same posture FAISS takes for its IVF families. The
//! configured `DistanceMetric` still governs ranking at query time
//! through [`crate::search`]; it just does not govern partitioning.
//!
//! Squared L2 (no `sqrt`) keeps assignment fast and lets every
//! comparison stay in `f32` without losing ordering: for non-negative
//! `a, b`, `a < b` iff `a*a < b*b`. The kernel below is the obvious
//! sequential one — k-means assignment is the inner loop of training,
//! and using a fixed iteration order is what lets the determinism
//! contract hold.

/// Compute the squared L2 distance between `a` and `b`.
///
/// Caller MUST ensure `a.len() == b.len()`; this is enforced upstream
/// by the training step and [`crate::IvfIndex`] before any call site
/// reaches this kernel. Running in fixed component order makes the
/// reduction reproducible across platforms.
#[must_use]
pub(crate) fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "squared_l2 requires same-dim slices");
    let mut sum: f32 = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// Return the index of the centroid nearest to `vector` under squared
/// L2.
///
/// Ties are broken by **lower index wins** — never `<=`, always `<`
/// — so the assignment is fully deterministic in centroid order.
/// Caller MUST ensure `centroids` is non-empty and every centroid has
/// the same dimension as `vector`; both are invariants the index
/// maintains.
#[must_use]
pub(crate) fn assign_to_cluster(centroids: &[Vec<f32>], vector: &[f32]) -> usize {
    debug_assert!(
        !centroids.is_empty(),
        "assign_to_cluster needs at least one centroid"
    );
    let mut best_idx: usize = 0;
    let mut best_dist = squared_l2(&centroids[0], vector);
    for (i, c) in centroids.iter().enumerate().skip(1) {
        let d = squared_l2(c, vector);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn squared_l2_of_equal_vectors_is_zero() {
        let v = vec![1.0_f32, 2.0, 3.0];
        assert_eq!(squared_l2(&v, &v), 0.0);
    }

    #[test]
    fn squared_l2_matches_hand_computation() {
        // (1-4)^2 + (2-6)^2 + (3-3)^2 = 9 + 16 + 0 = 25.
        let a = [1.0_f32, 2.0, 3.0];
        let b = [4.0_f32, 6.0, 3.0];
        assert!((squared_l2(&a, &b) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn assign_picks_the_obviously_nearest_centroid() {
        let centroids = vec![vec![0.0_f32, 0.0], vec![10.0, 10.0], vec![100.0, 100.0]];
        let v = [9.5_f32, 10.5];
        assert_eq!(assign_to_cluster(&centroids, &v), 1);
    }

    #[test]
    fn assign_breaks_ties_in_favour_of_lower_index() {
        // Both centroids equidistant from `v` at the origin. Cluster 0
        // must win.
        let centroids = vec![vec![1.0_f32, 0.0], vec![-1.0, 0.0]];
        let v = [0.0_f32, 0.0];
        assert_eq!(assign_to_cluster(&centroids, &v), 0);
    }

    #[test]
    fn assign_single_centroid_returns_zero() {
        let centroids = vec![vec![5.0_f32, -5.0]];
        let v = [0.0_f32, 0.0];
        assert_eq!(assign_to_cluster(&centroids, &v), 0);
    }
}
