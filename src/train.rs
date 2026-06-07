//! K-means training for IVF.
//!
//! Hand-rolled k-means with k-means++ seeding and Lloyd's iterations,
//! driven by the seeded [`crate::rng::SplitMix64`] PRNG. The
//! determinism contract is the load-bearing reason every loop here is
//! sequential and every reduction runs in fixed order:
//!
//! > **Determinism.** Given the same [`crate::IvfConfig::seed`], the
//! > same `sample` slice (same pointers, same dimension, same byte
//! > content), and the same configured `n_clusters` and
//! > `training_sample_size`, [`train_kmeans`] returns byte-identical
//! > `Vec<Vec<f32>>` centroids on every supported platform. This
//! > holds because (1) all PRNG draws come from the in-tree
//! > [`SplitMix64`](crate::rng::SplitMix64); (2) all reductions run
//! > in a fixed sequential order; (3) centroid sums accumulate in
//! > `f64` and downcast to `f32` once at the end of each iteration,
//! > sidestepping the `f32` catastrophic-cancellation cliff for large
//! > clusters.
//!
//! ## Algorithm
//!
//! 1. **Pre-checks.** Reject empty samples, vectors of wrong
//!    dimension, and `sample.len() < n_clusters` (the brief's
//!    "graceful, no panic" path). All errors are
//!    [`iqdb_types::IqdbError`].
//! 2. **Subsample.** If `sample.len() > training_sample_size`, pick
//!    `training_sample_size` distinct indices via a partial
//!    Fisher–Yates shuffle keyed off the PRNG. Deterministic in
//!    `seed`.
//! 3. **k-means++ seeding.** Pick the first centroid uniformly,
//!    then for each remaining centroid pick a point with probability
//!    proportional to its squared distance from the nearest already-
//!    chosen centroid (the canonical Arthur–Vassilvitskii procedure).
//! 4. **Lloyd's iterations.** Up to [`MAX_ITERS`] passes; each pass
//!    assigns every (sub)sampled point to its nearest centroid, then
//!    recomputes every centroid as the mean of its assigned points.
//!    Convergence triggers when the maximum relative centroid shift
//!    drops below [`REL_TOL`].
//! 5. **Empty-cluster recovery.** When a Lloyd's pass leaves a
//!    cluster with zero assignments, the centroid is moved to the
//!    sample point that is furthest from any current centroid — a
//!    deterministic recovery that preserves the determinism contract.

use iqdb_types::{IqdbError, Result};

use crate::assign::{assign_to_cluster, squared_l2};
use crate::rng::SplitMix64;

/// Maximum number of Lloyd's iterations.
///
/// `25` is enough for the centroids to stabilize on every dataset
/// we benchmark; data-driven max-iters tuning is a possible future
/// knob. Holding the value here (rather than on [`crate::IvfConfig`])
/// keeps the public surface minimal — the cap is an implementation
/// detail of the algorithm, not a property of the index.
pub(crate) const MAX_ITERS: usize = 25;

/// Convergence threshold on the *relative* centroid shift.
///
/// At the end of each Lloyd's iteration we compute
/// `max over centroids of ||old - new||_2 / max(||old||_2, 1.0)`
/// and stop once it drops below this value. Using a relative shift
/// (rather than absolute) keeps the threshold meaningful across
/// datasets whose vectors are scaled very differently.
pub(crate) const REL_TOL: f32 = 1e-4;

/// Train k-means centroids on `sample`.
///
/// Returns `n_clusters` centroids, each a `Vec<f32>` of length `dim`,
/// or an [`IqdbError`] if the pre-checks reject the inputs. See the
/// module-level docs for the determinism contract.
pub(crate) fn train_kmeans(
    dim: usize,
    n_clusters: usize,
    seed: u64,
    sample: &[&[f32]],
    training_sample_size: usize,
) -> Result<Vec<Vec<f32>>> {
    // -- Pre-checks ---------------------------------------------------
    if sample.is_empty() {
        return Err(IqdbError::InvalidConfig {
            reason: "IvfIndex::train requires a non-empty sample",
        });
    }
    for v in sample {
        if v.len() != dim {
            return Err(IqdbError::DimensionMismatch {
                expected: dim,
                found: v.len(),
            });
        }
    }
    if sample.len() < n_clusters {
        return Err(IqdbError::InvalidConfig {
            reason: "IvfIndex::train requires sample.len() >= n_clusters",
        });
    }

    let mut rng = SplitMix64::new(seed);

    // -- Subsample if needed -----------------------------------------
    // Build a Vec of slice references into `sample` of length
    // `training_sample_size` (or `sample.len()` if that's smaller).
    // `retrain()` calls `subsample_refs` directly with its own
    // freshly-seeded RNG to pre-cap the working set; that is why the
    // helper is `pub(crate)` rather than inlined here.
    let target_len = training_sample_size.min(sample.len());
    let working_set: Vec<&[f32]> = subsample_refs(sample, target_len, &mut rng);

    // -- k-means++ seeding -------------------------------------------
    let centroids = kmeans_plus_plus(dim, n_clusters, &working_set, &mut rng);

    // -- Lloyd's iterations ------------------------------------------
    let final_centroids = lloyd(centroids, &working_set, dim);
    Ok(final_centroids)
}

/// Deterministic subsample of `sample` down to `target_len` distinct
/// elements via partial Fisher–Yates over the supplied [`SplitMix64`]
/// instance.
///
/// When `target_len >= sample.len()`, returns a verbatim
/// `sample.to_vec()` and **does not draw from `rng`** — same shape
/// the inlined loop in [`train_kmeans`] always had, so `train()`'s
/// k-means++ pass starts from the same RNG state in the
/// no-subsample case.
///
/// When `target_len < sample.len()`, advances `rng` by exactly
/// `target_len` calls to `next_below` (one per swap), in fixed order
/// — identical to the original inlined version before this helper
/// was extracted.
///
/// [`crate::index::IvfIndex::retrain`] uses this directly with a
/// fresh `SplitMix64::new(seed)` to pre-cap its working set to
/// [`crate::IvfConfig::training_sample_size`]; the helper is the
/// load-bearing primitive that keeps retrain's k-means and PQ
/// training both bounded by the same configured cap as the original
/// `train()`.
pub(crate) fn subsample_refs<'a>(
    sample: &[&'a [f32]],
    target_len: usize,
    rng: &mut SplitMix64,
) -> Vec<&'a [f32]> {
    let n = sample.len();
    // Belt-and-suspenders clamp; callers already pre-min, but keeping
    // the invariant inside the helper means future call sites cannot
    // walk off the end of the swap loop.
    let target = target_len.min(n);
    if target == n {
        return sample.to_vec();
    }
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..target {
        let remaining = (n - i) as u64;
        let pick = i + (rng.next_below(remaining) as usize);
        indices.swap(i, pick);
    }
    indices[..target].iter().map(|&i| sample[i]).collect()
}

/// Pick `n_clusters` initial centroids via the Arthur–Vassilvitskii
/// k-means++ procedure.
///
/// Caller has already verified `n_clusters >= 1`, `working_set` is
/// non-empty and dimension-matched, and `working_set.len() >=
/// n_clusters`.
fn kmeans_plus_plus(
    dim: usize,
    n_clusters: usize,
    working_set: &[&[f32]],
    rng: &mut SplitMix64,
) -> Vec<Vec<f32>> {
    let n = working_set.len();
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(n_clusters);

    // 1. First centroid: uniformly random.
    let first_idx = rng.next_below(n as u64) as usize;
    centroids.push(working_set[first_idx].to_vec());

    // `min_sq[i]` = squared L2 distance from `working_set[i]` to its
    // nearest already-chosen centroid. Initialized from centroid 0.
    let mut min_sq: Vec<f32> = working_set
        .iter()
        .map(|v| squared_l2(working_set[first_idx], v))
        .collect();

    // 2. Remaining centroids: weighted by `min_sq`.
    for _ in 1..n_clusters {
        // Sum the weights into `f64` for stable accumulation; this
        // matters when `n` is large and the dynamic range of `min_sq`
        // values is wide.
        let mut total: f64 = 0.0;
        for &w in &min_sq {
            total += w as f64;
        }
        let next_idx = if total <= 0.0 {
            // All sampled points already coincide with a centroid;
            // the corpus has < `n_clusters` distinct points. Fall
            // back to a uniform pick over the working set so we still
            // produce `n_clusters` distinct entries (subject to the
            // duplicate count in the data — true duplicates remain
            // duplicates).
            rng.next_below(n as u64) as usize
        } else {
            // Inverse-CDF draw against the `min_sq` distribution.
            let target = rng.next_open_unit() * total;
            let mut running: f64 = 0.0;
            let mut chosen: usize = n - 1; // fallback: last index
            for (i, &w) in min_sq.iter().enumerate() {
                running += w as f64;
                if running >= target {
                    chosen = i;
                    break;
                }
            }
            chosen
        };
        centroids.push(working_set[next_idx].to_vec());

        // Refresh `min_sq` against the newly chosen centroid only —
        // saves the O(n_clusters) outer recompute per step.
        let new_centroid = &centroids[centroids.len() - 1];
        for (i, v) in working_set.iter().enumerate() {
            let d = squared_l2(new_centroid, v);
            if d < min_sq[i] {
                min_sq[i] = d;
            }
        }
    }

    debug_assert_eq!(centroids.len(), n_clusters);
    debug_assert!(centroids.iter().all(|c| c.len() == dim));
    let _ = dim; // dim is encoded in centroid length above; kept for clarity.
    centroids
}

/// Run Lloyd's iterations on `centroids` until convergence or
/// [`MAX_ITERS`].
fn lloyd(mut centroids: Vec<Vec<f32>>, working_set: &[&[f32]], dim: usize) -> Vec<Vec<f32>> {
    let n_clusters = centroids.len();
    let n = working_set.len();

    // Reusable buffers, allocated once outside the iteration loop.
    let mut sums: Vec<Vec<f64>> = vec![vec![0.0_f64; dim]; n_clusters];
    let mut counts: Vec<usize> = vec![0_usize; n_clusters];
    let mut assignments: Vec<usize> = vec![0_usize; n];

    for _iter in 0..MAX_ITERS {
        // Reset accumulators.
        for s in sums.iter_mut() {
            for v in s.iter_mut() {
                *v = 0.0;
            }
        }
        for c in counts.iter_mut() {
            *c = 0;
        }

        // -- Assignment pass ---------------------------------------
        for (i, v) in working_set.iter().enumerate() {
            let c = assign_to_cluster(&centroids, v);
            assignments[i] = c;
            let s = &mut sums[c];
            for (k, &x) in v.iter().enumerate() {
                s[k] += x as f64;
            }
            counts[c] += 1;
        }

        // -- Update pass + max-shift tracking ----------------------
        let mut max_rel_shift: f32 = 0.0;
        for c in 0..n_clusters {
            let count = counts[c];
            if count == 0 {
                // Empty cluster — deterministic recovery: find the
                // sample point with the largest min-distance to any
                // current centroid (the "most isolated" point in the
                // current model), scanning in fixed order so ties go
                // to the lowest sample index.
                let mut best_idx: usize = 0;
                let mut best_dist: f32 = -1.0;
                for (i, v) in working_set.iter().enumerate() {
                    let mut nearest: f32 = squared_l2(&centroids[0], v);
                    for cc in centroids.iter().skip(1) {
                        let d = squared_l2(cc, v);
                        if d < nearest {
                            nearest = d;
                        }
                    }
                    if nearest > best_dist {
                        best_dist = nearest;
                        best_idx = i;
                    }
                }
                let new_centroid = working_set[best_idx].to_vec();
                let shift = relative_shift(&centroids[c], &new_centroid);
                if shift > max_rel_shift {
                    max_rel_shift = shift;
                }
                centroids[c] = new_centroid;
                continue;
            }

            // Mean = sum / count, downcast to f32.
            let inv = 1.0_f64 / (count as f64);
            let new_centroid: Vec<f32> = sums[c].iter().map(|&s| (s * inv) as f32).collect();
            let shift = relative_shift(&centroids[c], &new_centroid);
            if shift > max_rel_shift {
                max_rel_shift = shift;
            }
            centroids[c] = new_centroid;
        }

        if max_rel_shift < REL_TOL {
            break;
        }
    }

    let _ = assignments; // silence the unused-results lint on the buffer.
    centroids
}

/// `||old - new||_2 / max(||old||_2, 1.0)`.
///
/// Computed sequentially in fixed component order. Used as the
/// convergence criterion in [`lloyd`].
fn relative_shift(old: &[f32], new: &[f32]) -> f32 {
    debug_assert_eq!(old.len(), new.len());
    let mut diff_sq: f32 = 0.0;
    let mut old_norm_sq: f32 = 0.0;
    for i in 0..old.len() {
        let d = old[i] - new[i];
        diff_sq += d * d;
        old_norm_sq += old[i] * old[i];
    }
    let diff = diff_sq.sqrt();
    let denom = old_norm_sq.sqrt().max(1.0);
    diff / denom
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    fn refs(slice: &[Vec<f32>]) -> Vec<&[f32]> {
        slice.iter().map(|v| v.as_slice()).collect()
    }

    #[test]
    fn rejects_empty_sample() {
        let err = train_kmeans(2, 3, 0, &[], 100).unwrap_err();
        match err {
            IqdbError::InvalidConfig { reason } => {
                assert!(reason.contains("non-empty sample"));
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn rejects_sample_smaller_than_clusters() {
        let data = vec![vec![0.0_f32, 0.0], vec![1.0, 1.0]];
        let sample = refs(&data);
        let err = train_kmeans(2, 5, 0, &sample, 100).unwrap_err();
        match err {
            IqdbError::InvalidConfig { reason } => {
                assert!(reason.contains("sample.len() >= n_clusters"));
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn rejects_dim_mismatch() {
        let bad = vec![1.0_f32, 2.0, 3.0]; // dim 3 sample but dim=2
        let sample = vec![bad.as_slice()];
        let err = train_kmeans(2, 1, 0, &sample, 100).unwrap_err();
        match err {
            IqdbError::DimensionMismatch { expected, found } => {
                assert_eq!(expected, 2);
                assert_eq!(found, 3);
            }
            other => panic!("expected DimensionMismatch, got {other:?}"),
        }
    }

    #[test]
    fn converges_on_two_obvious_clusters() {
        // Two tight clusters at (0,0) and (10,10).
        let data: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![0.1, -0.1],
            vec![-0.05, 0.05],
            vec![10.0, 10.0],
            vec![10.1, 9.9],
            vec![9.95, 10.05],
        ];
        let sample = refs(&data);
        let centroids = train_kmeans(2, 2, 1, &sample, 100).unwrap();
        assert_eq!(centroids.len(), 2);
        let mut near_origin = 0;
        let mut near_ten = 0;
        for c in &centroids {
            if c[0].abs() < 1.0 && c[1].abs() < 1.0 {
                near_origin += 1;
            }
            if (c[0] - 10.0).abs() < 1.0 && (c[1] - 10.0).abs() < 1.0 {
                near_ten += 1;
            }
        }
        assert_eq!(near_origin, 1);
        assert_eq!(near_ten, 1);
    }

    #[test]
    fn same_seed_produces_identical_centroids() {
        let data: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![(i as f32) * 0.1, ((i * 3) as f32) * 0.07])
            .collect();
        let sample = refs(&data);
        let a = train_kmeans(2, 4, 1234, &sample, 100).unwrap();
        let b = train_kmeans(2, 4, 1234, &sample, 100).unwrap();
        assert_eq!(a, b, "same seed + same data → identical centroids");
    }

    #[test]
    fn different_seeds_can_diverge() {
        let data: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![(i as f32) * 0.1, ((i * 3) as f32) * 0.07])
            .collect();
        let sample = refs(&data);
        let a = train_kmeans(2, 4, 1, &sample, 100).unwrap();
        let b = train_kmeans(2, 4, 2, &sample, 100).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn subsampling_is_deterministic() {
        let data: Vec<Vec<f32>> = (0..100)
            .map(|i| vec![(i as f32) * 0.01, ((i * 7) as f32) * 0.013])
            .collect();
        let sample = refs(&data);
        let a = train_kmeans(2, 5, 99, &sample, 25).unwrap();
        let b = train_kmeans(2, 5, 99, &sample, 25).unwrap();
        assert_eq!(a, b);
    }
}
