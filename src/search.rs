//! IVF search.
//!
//! Two passes over the index. Pass 1 (centroid ranking) is shared
//! across both IVF-Flat and IVF-PQ; Pass 2 (intra-cluster scan)
//! branches on [`crate::IvfConfig::use_pq`] **and** on whether the
//! query supplied a metadata filter:
//!
//! 1. **Centroid ranking.** Score the query against every centroid
//!    via [`iqdb_distance::compute_batch`] and pick the
//!    [`crate::IvfConfig::n_probes`] indices with the smallest
//!    distance under the configured metric. Centroid distances are
//!    always exact — IVF-PQ does **not** quantize centroids. The
//!    filter is intentionally not applied at the centroid pass:
//!    filters live on entries, not on cluster centres.
//! 2. **Intra-cluster scan.**
//!    - **IVF-Flat** (default): exact `compute_batch` over every
//!      stored vector in each probed list, collect into a flat
//!      candidate buffer, top-`k` via
//!      [`crate::topk::select_topk_indices`] with the
//!      `(distance, seq)` tiebreaker.
//!    - **IVF-PQ** (`use_pq = true`): delegate to
//!      [`crate::pq_variant`], which builds an ADC lookup
//!      table once per query (plain-PQ → cluster-independent),
//!      scores every entry via
//!      [`iqdb_quantize::PqAdcTables::distance`], and optionally
//!      exact-reranks a `pq_refine_factor * k` shortlist against
//!      the retained `Arc<[f32]>` vectors. The DotProduct sign
//!      convention is identical to the Flat path.
//!
//! For [`DistanceMetric::DotProduct`] the raw inner product is
//! "larger is more similar"; both branches negate at both passes so
//! the "smaller is nearer" invariant on [`iqdb_types::Hit::distance`]
//! holds for IVF the same way it does for flat and hnsw.
//!
//! ## Filter contract
//!
//! When `params.filter = Some(filter)`, the filter is validated once
//! through [`FilterEvaluator::new`] — pathological filters (depth or
//! `In` cardinality caps) surface as
//! [`IqdbError::InvalidFilter`] there. The validated evaluator is
//! then threaded into a dedicated `*_filtered` scan function on both
//! the IVF-Flat and IVF-PQ branches; the filter-`None` hot path is
//! reached through a separate `*_unfiltered` function so the inner
//! per-entry loop carries no per-entry filter branch at all.
//!
//! `params.metric != index.metric` returns [`IqdbError::InvalidMetric`];
//! the metric is fixed at construction time (matches flat / hnsw).
//! An empty index (no inserts after `train`) and `k == 0` return
//! `Ok(Vec::new())`.
//!
//! A selective filter may legitimately return fewer than `k` hits
//! within the probed clusters. IVF deliberately does **not** widen the
//! probe set to satisfy a filter — that would break the "you always
//! probe N, latency is predictable" property that motivates IVF over a
//! graph index. Callers that want higher coverage can grow `n_probes`
//! directly via [`crate::IvfIndex::set_n_probes`] or consult
//! [`crate::IvfIndex::suggest_n_probes`] for a cluster-size-driven
//! recommendation.

use iqdb_distance::compute_batch;
use iqdb_filter::FilterEvaluator;
use iqdb_types::{DistanceMetric, Hit, IqdbError, Result, SearchParams};

use crate::index::{InvertedListEntry, IvfIndex};
use crate::pq_variant;
use crate::topk::select_topk_indices;

/// Run IVF search — dispatches to IVF-Flat or IVF-PQ based on
/// [`crate::IvfConfig::use_pq`] and to the filtered or unfiltered
/// scan based on [`SearchParams::filter`].
///
/// See the module-level docs for the contract. The function is
/// the body of `IvfIndex::search_inner`; it is split out here to
/// keep `index.rs` focused on the trait wiring.
pub(crate) fn ivf_search(
    index: &IvfIndex,
    query: &[f32],
    params: &SearchParams,
) -> Result<Vec<Hit>> {
    if query.len() != index.dim() {
        return Err(IqdbError::DimensionMismatch {
            expected: index.dim(),
            found: query.len(),
        });
    }
    if params.metric != index.metric() {
        return Err(IqdbError::InvalidMetric);
    }
    if params.k == 0 || index.is_empty() {
        return Ok(Vec::new());
    }

    // Build the evaluator once. The validation walk in
    // `FilterEvaluator::new` enforces depth and `In` caps and
    // surfaces a pathological filter as `IqdbError::InvalidFilter`
    // before we touch any cluster data.
    let evaluator: Option<FilterEvaluator> = match &params.filter {
        None => None,
        Some(f) => Some(FilterEvaluator::new(f.clone())?),
    };

    let centroids = index.centroids_slice();
    if centroids.is_empty() {
        // Defensive: validate() + train() invariants make this
        // unreachable, but returning empty is safer than panicking.
        return Ok(Vec::new());
    }

    // -- Pass 1: rank centroids --------------------------------------
    // Exact distance on centroids regardless of use_pq; IVF-PQ does
    // not quantize centroids. Every centroid Vec is dim-length;
    // that is an invariant train.rs upholds and IvfIndex preserves.
    let centroid_refs: Vec<&[f32]> = centroids.iter().map(|c| c.as_slice()).collect();
    let mut centroid_dists = vec![0.0_f32; centroid_refs.len()];
    compute_batch(index.metric(), query, &centroid_refs, &mut centroid_dists)?;
    if matches!(index.metric(), DistanceMetric::DotProduct) {
        for d in centroid_dists.iter_mut() {
            *d = -*d;
        }
    }

    // Use centroid id as the tiebreaker seq so ties on centroid
    // distance go to the lower cluster id (deterministic).
    let centroid_seqs: Vec<u64> = (0..centroid_refs.len() as u64).collect();
    let n_probes = index.n_probes().min(centroid_refs.len());
    let probed: Vec<usize> = select_topk_indices(&centroid_dists, &centroid_seqs, n_probes);

    // -- Pass 2: intra-cluster scan (branch on use_pq × filter) ------
    // The four-function split (unfiltered vs filtered for each of
    // Flat and PQ) mirrors `iqdb-flat`'s no-alloc hot path: the
    // unfiltered inner loop carries no per-entry
    // `Option<&FilterEvaluator>` branch.
    match (index.cfg().use_pq, evaluator.as_ref()) {
        (false, None) => scan_flat_unfiltered(index, query, &probed, params.k),
        (false, Some(eval)) => scan_flat_filtered(index, eval, query, &probed, params.k),
        (true, None) => pq_variant::scan_pq_unfiltered(index, query, &probed, params.k),
        (true, Some(eval)) => pq_variant::scan_pq_filtered(index, eval, query, &probed, params.k),
    }
}

/// IVF-Flat intra-cluster scan + top-`k` over **every** entry in
/// each probed cluster.
///
/// Scores every entry via [`iqdb_distance::compute_batch`] over the
/// retained `Arc<[f32]>` vectors, with the DotProduct sign-flip
/// applied in-place, then runs the standard `(distance, seq)` top-`k`
/// selection. The filter-`None` hot path lives in its own function so
/// the unfiltered allocation profile carries no filter overhead.
fn scan_flat_unfiltered(
    index: &IvfIndex,
    query: &[f32],
    probed: &[usize],
    k: usize,
) -> Result<Vec<Hit>> {
    let mut candidate_distances: Vec<f32> = Vec::new();
    let mut candidate_seqs: Vec<u64> = Vec::new();
    // Each candidate is identified by (cluster_id, position_in_cluster);
    // mapping back to (id, metadata) happens at hit-construction time.
    let mut candidate_addrs: Vec<(usize, usize)> = Vec::new();

    for &c in probed {
        let list: &[InvertedListEntry] = index.inverted_list(c);
        if list.is_empty() {
            continue;
        }
        let vec_refs: Vec<&[f32]> = list.iter().map(|e| e.vector_slice()).collect();
        let mut dists = vec![0.0_f32; vec_refs.len()];
        compute_batch(index.metric(), query, &vec_refs, &mut dists)?;
        if matches!(index.metric(), DistanceMetric::DotProduct) {
            for d in dists.iter_mut() {
                *d = -*d;
            }
        }
        for (pos, entry) in list.iter().enumerate() {
            candidate_distances.push(dists[pos]);
            candidate_seqs.push(entry.seq());
            candidate_addrs.push((c, pos));
        }
    }

    if candidate_distances.is_empty() {
        return Ok(Vec::new());
    }

    let chosen = select_topk_indices(&candidate_distances, &candidate_seqs, k);
    let mut hits = Vec::with_capacity(chosen.len());
    for cand_idx in chosen {
        let (c, pos) = candidate_addrs[cand_idx];
        let entry = &index.inverted_list(c)[pos];
        hits.push(Hit {
            id: entry.id().clone(),
            distance: candidate_distances[cand_idx],
            metadata: entry.metadata().cloned(),
        });
    }
    Ok(hits)
}

/// IVF-Flat intra-cluster scan + top-`k` restricted to entries that
/// satisfy `evaluator`.
///
/// Survivors are collected before `compute_batch` runs so we never
/// score filter-excluded entries (matches `iqdb-flat`'s filtered
/// path). The candidate buffers shrink to the filter survivor
/// count; the final `(distance, seq)` top-`k` selection is the same
/// tiebreaker as the unfiltered path.
fn scan_flat_filtered(
    index: &IvfIndex,
    evaluator: &FilterEvaluator,
    query: &[f32],
    probed: &[usize],
    k: usize,
) -> Result<Vec<Hit>> {
    let mut candidate_distances: Vec<f32> = Vec::new();
    let mut candidate_seqs: Vec<u64> = Vec::new();
    let mut candidate_addrs: Vec<(usize, usize)> = Vec::new();

    for &c in probed {
        let list: &[InvertedListEntry] = index.inverted_list(c);
        if list.is_empty() {
            continue;
        }
        // Walk the cluster once, collecting only survivor positions
        // and slice references. The distance call below operates on
        // just the survivors, so a selective filter cuts the
        // `compute_batch` workload proportionally.
        let mut survivor_positions: Vec<usize> = Vec::new();
        let mut survivor_refs: Vec<&[f32]> = Vec::new();
        for (pos, entry) in list.iter().enumerate() {
            if evaluator.evaluate(entry.metadata()) {
                survivor_positions.push(pos);
                survivor_refs.push(entry.vector_slice());
            }
        }
        if survivor_refs.is_empty() {
            continue;
        }
        let mut dists = vec![0.0_f32; survivor_refs.len()];
        compute_batch(index.metric(), query, &survivor_refs, &mut dists)?;
        if matches!(index.metric(), DistanceMetric::DotProduct) {
            for d in dists.iter_mut() {
                *d = -*d;
            }
        }
        for (i, &pos) in survivor_positions.iter().enumerate() {
            candidate_distances.push(dists[i]);
            candidate_seqs.push(list[pos].seq());
            candidate_addrs.push((c, pos));
        }
    }

    if candidate_distances.is_empty() {
        return Ok(Vec::new());
    }

    let chosen = select_topk_indices(&candidate_distances, &candidate_seqs, k);
    let mut hits = Vec::with_capacity(chosen.len());
    for cand_idx in chosen {
        let (c, pos) = candidate_addrs[cand_idx];
        let entry = &index.inverted_list(c)[pos];
        hits.push(Hit {
            id: entry.id().clone(),
            distance: candidate_distances[cand_idx],
            metadata: entry.metadata().cloned(),
        });
    }
    Ok(hits)
}
