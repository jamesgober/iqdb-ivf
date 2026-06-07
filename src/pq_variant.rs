//! IVF-PQ — Product-Quantization branch of [`crate::IvfIndex`].
//!
//! Plain-PQ: the codebooks are trained on the same working set as
//! the coarse k-means centroids (not on residuals), so the per-query
//! ADC table is cluster-independent and can be built **once** per
//! query and reused across every probed cluster. The `M × K` table
//! comes from
//! [`iqdb_quantize::ProductQuantizer::build_query_tables`]; per-code
//! scoring runs through
//! [`iqdb_quantize::PqAdcTables::distance`].
//!
//! ## Sign convention
//!
//! [`iqdb_quantize::PqAdcTables::distance`] returns
//! [`iqdb_distance::compute_batch`] semantics: the raw inner product
//! for [`iqdb_types::DistanceMetric::DotProduct`] (larger = more
//! similar). The intra-cluster scan negates DotProduct in-place
//! immediately after scoring — **exactly mirroring** the IVF-Flat
//! path in [`crate::search`] — so the "smaller is nearer" invariant
//! on [`iqdb_types::Hit::distance`] holds for both variants.
//!
//! ## Refine policy
//!
//! Driven by [`crate::IvfConfig::pq_refine_factor`]:
//!
//! - `0` — pure ADC. Top-`k` of the ADC candidate buffer is returned
//!   as-is; the retained `Arc<[f32]>` vectors are not consulted.
//! - `N >= 1` — shortlist `N * k` by ADC, then exact-rerank the
//!   shortlist using the retained `Arc<[f32]>` vectors via the same
//!   distance path IVF-Flat uses (and the same DotProduct
//!   negation), returning top-`k` of the reranked shortlist. The
//!   default `N = 4` matches the FAISS production default.

use iqdb_distance::compute_batch;
use iqdb_filter::FilterEvaluator;
use iqdb_quantize::{ProductQuantizer, Quantizer};
use iqdb_types::{DistanceMetric, Hit, IqdbError, Result};

use crate::config::IvfConfig;
use crate::index::IvfIndex;
use crate::topk::select_topk_indices;

/// `K`, the number of centroids per PQ subvector codebook.
///
/// Fixed at `256` so each PQ code byte is one `u8`, matching the spec
/// and the `iqdb-quantize` ceiling.
const PQ_K_CENTROIDS: usize = 256;

/// Train the IVF-PQ codebooks on `sample`.
///
/// Called from [`IvfIndex::train`] after the coarse k-means
/// succeeds, only when `cfg.use_pq` is true. The codebooks are
/// trained on the same working set as the centroids (plain-PQ), so
/// the ADC lookup table is cluster-independent at query time.
///
/// The PQ seed is `cfg.seed` — the same seed that drives the
/// coarse k-means — so the IVF-PQ build is deterministic end-to-end
/// from a single config seed.
///
/// # Errors
///
/// Propagates [`iqdb_quantize::ProductQuantizer::train`] errors as-is.
/// In particular, a training sample smaller than [`PQ_K_CENTROIDS`]
/// surfaces as `IqdbError::InvalidConfig` with a `reason` from the
/// quantizer.
pub(crate) fn train_pq(cfg: &IvfConfig, sample: &[&[f32]]) -> Result<ProductQuantizer> {
    // `cfg.validate()` + `IvfIndex::new_unconfigured` guarantee
    // `Some(m >= 1)` and `m | dim`, so the construction below can't
    // fail on shape grounds. The defensive `ok_or` keeps this honest
    // under deny(clippy::unwrap_used) without crashing if invariants
    // ever change upstream.
    let m = cfg.pq_subvectors.ok_or(IqdbError::InvalidConfig {
        reason: "IvfConfig.pq_subvectors must be Some(_) when use_pq = true",
    })?;
    let mut pq = ProductQuantizer::with_config(m, PQ_K_CENTROIDS, cfg.seed);
    pq.train(sample)?;
    Ok(pq)
}

/// Run the IVF-PQ intra-cluster scan + optional refine over the
/// `probed` clusters and return the top-`k` hits, with no metadata
/// filter applied.
///
/// `query.len() == index.dim()` and the metric have already been
/// validated by the caller in [`crate::search::ivf_search`]; this
/// function is responsible for the ADC pass, the optional exact
/// refine, the `(distance, seq)` top-k tiebreaker, and constructing
/// the final `Hit`s.
///
/// Scoring every entry in every probed cluster via
/// [`iqdb_quantize::PqAdcTables::distance`] is the load-bearing reason
/// IVF-PQ has competitive recall; this is the unfiltered hot path and
/// carries no per-entry filter branch.
pub(crate) fn scan_pq_unfiltered(
    index: &IvfIndex,
    query: &[f32],
    probed: &[usize],
    k: usize,
) -> Result<Vec<Hit>> {
    let pq = index.pq().ok_or(IqdbError::InvalidConfig {
        reason: "IvfIndex is in IVF-PQ mode but the quantizer was not trained",
    })?;
    let metric = index.metric();
    let tables = pq.build_query_tables(query, metric)?;

    let mut cand_distances: Vec<f32> = Vec::new();
    let mut cand_seqs: Vec<u64> = Vec::new();
    let mut cand_addrs: Vec<(usize, usize)> = Vec::new();

    for &c in probed {
        let list = index.inverted_list(c);
        for (pos, entry) in list.iter().enumerate() {
            let code = entry.pq_code_or_err()?;
            let mut d = tables.distance(code)?;
            // Mirror IVF-Flat's DotProduct negation byte-for-byte
            // (see `crate::search::scan_flat_unfiltered`): the raw
            // inner product is "larger = more similar" and we want
            // "smaller = nearer" on `Hit::distance`.
            if matches!(metric, DistanceMetric::DotProduct) {
                d = -d;
            }
            cand_distances.push(d);
            cand_seqs.push(entry.seq());
            cand_addrs.push((c, pos));
        }
    }

    if cand_distances.is_empty() {
        return Ok(Vec::new());
    }

    finish_scan_pq(index, query, k, cand_distances, cand_seqs, cand_addrs)
}

/// Run the IVF-PQ intra-cluster scan + optional refine, restricted
/// to entries that satisfy `evaluator`.
///
/// "Filter before ADC": the evaluator runs *before*
/// [`iqdb_quantize::PqAdcTables::distance`], so filter-excluded
/// entries never consume the per-code subvector table lookups.
/// Because every candidate in the buffer is already a filter
/// survivor, the refine pass — which works off `cand_addrs` indices
/// — automatically inherits the filter: the shortlist drawn from
/// `cand_addrs` cannot contain a filter-excluded entry, so
/// `refine_shortlist` operates on already-filtered candidates
/// without needing an extra check.
pub(crate) fn scan_pq_filtered(
    index: &IvfIndex,
    evaluator: &FilterEvaluator,
    query: &[f32],
    probed: &[usize],
    k: usize,
) -> Result<Vec<Hit>> {
    let pq = index.pq().ok_or(IqdbError::InvalidConfig {
        reason: "IvfIndex is in IVF-PQ mode but the quantizer was not trained",
    })?;
    let metric = index.metric();
    let tables = pq.build_query_tables(query, metric)?;

    let mut cand_distances: Vec<f32> = Vec::new();
    let mut cand_seqs: Vec<u64> = Vec::new();
    let mut cand_addrs: Vec<(usize, usize)> = Vec::new();

    for &c in probed {
        let list = index.inverted_list(c);
        for (pos, entry) in list.iter().enumerate() {
            if !evaluator.evaluate(entry.metadata()) {
                continue;
            }
            let code = entry.pq_code_or_err()?;
            let mut d = tables.distance(code)?;
            if matches!(metric, DistanceMetric::DotProduct) {
                d = -d;
            }
            cand_distances.push(d);
            cand_seqs.push(entry.seq());
            cand_addrs.push((c, pos));
        }
    }

    if cand_distances.is_empty() {
        return Ok(Vec::new());
    }

    finish_scan_pq(index, query, k, cand_distances, cand_seqs, cand_addrs)
}

/// Shared top-`k` + optional refine tail for both
/// [`scan_pq_unfiltered`] and [`scan_pq_filtered`].
///
/// Owns the `cand_*` buffers so the two scan functions can hand
/// them off without a second allocation; the refine shortlist is
/// drawn from these buffers, so filter survivorship is preserved
/// transparently by the caller's filter discipline.
fn finish_scan_pq(
    index: &IvfIndex,
    query: &[f32],
    k: usize,
    cand_distances: Vec<f32>,
    cand_seqs: Vec<u64>,
    cand_addrs: Vec<(usize, usize)>,
) -> Result<Vec<Hit>> {
    let refine_factor = index.cfg().pq_refine_factor;
    if refine_factor == 0 {
        // Pure ADC: skip the refine pass entirely.
        let chosen = select_topk_indices(&cand_distances, &cand_seqs, k);
        return build_hits_from_adc(index, &chosen, &cand_distances, &cand_addrs);
    }

    // Refine: shortlist N*k candidates by ADC, then exact-rerank
    // those using the retained `Arc<[f32]>` vectors.
    let shortlist_k = (refine_factor as usize)
        .saturating_mul(k)
        .min(cand_distances.len());
    let shortlist = select_topk_indices(&cand_distances, &cand_seqs, shortlist_k);
    refine_shortlist(index, query, &shortlist, &cand_addrs, k)
}

fn build_hits_from_adc(
    index: &IvfIndex,
    chosen: &[usize],
    cand_distances: &[f32],
    cand_addrs: &[(usize, usize)],
) -> Result<Vec<Hit>> {
    let mut hits = Vec::with_capacity(chosen.len());
    for &cand_idx in chosen {
        let (c, pos) = cand_addrs[cand_idx];
        let entry = &index.inverted_list(c)[pos];
        hits.push(Hit {
            id: entry.id().clone(),
            distance: cand_distances[cand_idx],
            metadata: entry.metadata().cloned(),
        });
    }
    Ok(hits)
}

fn refine_shortlist(
    index: &IvfIndex,
    query: &[f32],
    shortlist: &[usize],
    cand_addrs: &[(usize, usize)],
    k: usize,
) -> Result<Vec<Hit>> {
    if shortlist.is_empty() {
        return Ok(Vec::new());
    }
    let metric = index.metric();
    let mut refs: Vec<&[f32]> = Vec::with_capacity(shortlist.len());
    let mut seqs: Vec<u64> = Vec::with_capacity(shortlist.len());
    let mut addrs: Vec<(usize, usize)> = Vec::with_capacity(shortlist.len());
    for &cand_idx in shortlist {
        let (c, pos) = cand_addrs[cand_idx];
        let entry = &index.inverted_list(c)[pos];
        refs.push(entry.vector_slice());
        seqs.push(entry.seq());
        addrs.push((c, pos));
    }
    let mut exact = vec![0.0_f32; refs.len()];
    compute_batch(metric, query, &refs, &mut exact)?;
    if matches!(metric, DistanceMetric::DotProduct) {
        for d in exact.iter_mut() {
            *d = -*d;
        }
    }
    let chosen = select_topk_indices(&exact, &seqs, k);
    let mut hits = Vec::with_capacity(chosen.len());
    for &idx in &chosen {
        let (c, pos) = addrs[idx];
        let entry = &index.inverted_list(c)[pos];
        hits.push(Hit {
            id: entry.id().clone(),
            distance: exact[idx],
            metadata: entry.metadata().cloned(),
        });
    }
    Ok(hits)
}
