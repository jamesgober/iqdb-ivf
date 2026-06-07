//! [`IvfIndex`] — the inverted-file ANN index.
//!
//! Holds `n_clusters` k-means centroids plus a parallel `Vec` of
//! inverted lists. Each inverted list owns the `(VectorId,
//! Arc<[f32]>, Option<Metadata>, seq)` tuples for the vectors that
//! assigned to its centroid; cluster sizes are bounded by
//! `O(n / n_clusters)` so the per-list scan stays cheap.
//!
//! ## Untrained state
//!
//! [`IvfConfig`] alone is not enough to answer queries — IVF needs
//! centroids, which means it needs training data. The four entry
//! points that depend on centroids ([`IndexCore::insert`],
//! [`IndexCore::insert_batch`], [`IndexCore::search`], and
//! [`IndexCore::search_batch`]) short-circuit with
//! [`IqdbError::InvalidConfig`] when the index is not yet trained.
//! [`IndexCore::delete`] is allowed before training but will always
//! return [`IqdbError::NotFound`] because the index is empty until
//! training enables inserts.
//!
//! ## Delete semantics
//!
//! `delete(id)` looks up the cluster from the `id_to_cluster` map and
//! `swap_remove`s the matching entry from that cluster's inverted
//! list (linear scan, cluster-local). The `seq` stamp on each entry
//! remains the topk tiebreaker, so `swap_remove` is safe — the
//! invariant that drives correctness does not depend on the position.
//! Re-clustering after a delete is not performed inline; callers
//! that want to rebalance after a delete-heavy phase call
//! [`IvfIndex::retrain`], which rebuilds centroids and (when
//! `use_pq`) PQ codebooks from the currently-stored vectors without
//! losing any data.

use std::collections::HashMap;
use std::mem::size_of;
use std::sync::Arc;

use error_forge::ForgeError;
use iqdb_index::{Index, IndexCore, IndexStats};
use iqdb_quantize::{PqCode, ProductQuantizer, Quantizer};
use iqdb_types::{DistanceMetric, Hit, IqdbError, Metadata, Result, SearchParams, VectorId};

use crate::assign::assign_to_cluster;
use crate::config::IvfConfig;
use crate::pq_variant;
use crate::rng::SplitMix64;
use crate::search;
use crate::stats::IvfClusterStats;
use crate::train::{subsample_refs, train_kmeans};

/// One entry in an inverted list.
///
/// Stored by-value inside the per-cluster `Vec`; the payload itself
/// is shared via [`Arc<[f32]>`] so the engine can hand the same
/// allocation to the index and its authoritative record store
/// without copying (audit M1, same contract as flat / hnsw). When the
/// index is in IVF-PQ mode (`cfg.use_pq = true`), each entry also
/// carries a [`PqCode`] — the ADC pass scores against the code, and
/// the retained `Arc<[f32]>` is the source of truth for the
/// exact-rerank refine pass.
#[derive(Debug)]
pub(crate) struct InvertedListEntry {
    id: VectorId,
    vector: Arc<[f32]>,
    pq_code: Option<PqCode>,
    metadata: Option<Metadata>,
    seq: u64,
}

impl InvertedListEntry {
    pub(crate) fn id(&self) -> &VectorId {
        &self.id
    }

    pub(crate) fn vector_slice(&self) -> &[f32] {
        &self.vector
    }

    pub(crate) fn metadata(&self) -> Option<&Metadata> {
        self.metadata.as_ref()
    }

    pub(crate) fn seq(&self) -> u64 {
        self.seq
    }

    /// The PQ code for this entry; errors with `InvalidConfig` if
    /// the index is in IVF-PQ mode but the entry has no code (an
    /// internal invariant violation — the IVF-PQ search path is the
    /// only caller and the invariant guarantees `Some(_)` post-train).
    ///
    /// The crate's deny lints forbid `unwrap`/`expect`, so this turns
    /// the invariant violation into a structured `InvalidConfig`.
    pub(crate) fn pq_code_or_err(&self) -> Result<&PqCode> {
        self.pq_code.as_ref().ok_or(IqdbError::InvalidConfig {
            reason: "IvfIndex entry is missing its PqCode in IVF-PQ mode (index invariant violated)",
        })
    }
}

/// IVF-Flat approximate nearest-neighbour index.
///
/// See the crate-level docs for the design notes and the
/// [`iqdb_index::IndexCore`] / [`iqdb_index::Index`] contracts this
/// type satisfies.
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
///
/// use iqdb_index::{Index, IndexCore};
/// use iqdb_ivf::{IvfConfig, IvfIndex};
/// use iqdb_types::{DistanceMetric, SearchParams, VectorId};
///
/// # fn main() -> iqdb_types::Result<()> {
/// let cfg = IvfConfig::default()
///     .with_n_clusters(2)
///     .with_n_probes(2)
///     .with_training_sample_size(16)
///     .with_seed(42);
/// let mut idx = IvfIndex::new(2, DistanceMetric::Euclidean, cfg)?;
///
/// // Train on a small dimensional sample first.
/// let sample: Vec<Vec<f32>> = vec![
///     vec![0.0, 0.0], vec![0.1, -0.1], vec![-0.1, 0.1],
///     vec![10.0, 10.0], vec![10.1, 9.9], vec![9.9, 10.1],
/// ];
/// let sample_refs: Vec<&[f32]> = sample.iter().map(|v| v.as_slice()).collect();
/// idx.train(&sample_refs)?;
///
/// idx.insert(VectorId::from(1u64), Arc::<[f32]>::from(&[0.0, 0.0][..]), None)?;
/// idx.insert(VectorId::from(2u64), Arc::<[f32]>::from(&[10.0, 10.0][..]), None)?;
///
/// let hits = idx.search(&[0.0, 0.0], &SearchParams::new(1, DistanceMetric::Euclidean))?;
/// assert_eq!(hits[0].id, VectorId::U64(1));
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct IvfIndex {
    dim: usize,
    metric: DistanceMetric,
    cfg: IvfConfig,
    trained: bool,

    // Row-major `Vec<Vec<f32>>`. Length is `cfg.n_clusters` after
    // training and `0` before. Owned (not Arc) because centroids are
    // computed inside the index and never shared with callers.
    centroids: Vec<Vec<f32>>,

    // One inverted list per centroid; `inverted_lists.len() ==
    // cfg.n_clusters` at all times (empty lists are pre-allocated at
    // construction so search-pass indexing is unconditional).
    inverted_lists: Vec<Vec<InvertedListEntry>>,

    // O(1) "which cluster does this id live in?" for delete + duplicate
    // checks. Always equal to `live_count` in cardinality.
    id_to_cluster: HashMap<VectorId, usize>,

    // Monotonic insertion stamp used as the topk tiebreaker.
    next_seq: u64,

    // Live (non-deleted) vector count. Tracked separately so `len()`
    // is O(1).
    live_count: usize,

    // Product quantizer for the IVF-PQ variant. `Some` iff
    // `cfg.use_pq` is `true` AND the index has been trained. The
    // codebooks are trained on the same working set as the coarse
    // k-means centroids (plain-PQ); per-entry `PqCode`s on
    // `InvertedListEntry` are produced through this quantizer.
    pq: Option<ProductQuantizer>,
}

impl IvfIndex {
    /// Build an empty index, mirroring [`Index::new`].
    ///
    /// Returns [`IqdbError::InvalidConfig`] when `dim == 0` or when
    /// [`IvfConfig::validate`] rejects the config. Calling this
    /// directly is the convenient path when the concrete type is
    /// known.
    pub fn new_unconfigured(dim: usize, metric: DistanceMetric, cfg: IvfConfig) -> Result<Self> {
        if dim == 0 {
            return Err(IqdbError::InvalidConfig {
                reason: "IvfIndex dim must be greater than zero",
            });
        }
        cfg.validate()?;
        // IVF-PQ guards that need both `dim` and `metric`.
        // `cfg.validate()` already enforces `pq_subvectors = Some(m >= 1)`
        // when `use_pq = true`; here we check the dim/metric pairing.
        if cfg.use_pq {
            match metric {
                DistanceMetric::Euclidean
                | DistanceMetric::DotProduct
                | DistanceMetric::Manhattan => {}
                // Cosine and Hamming are unsupported by IVF-PQ; the wildcard
                // also rejects any metric a future `iqdb-types` may add, since
                // `DistanceMetric` is `#[non_exhaustive]`.
                _ => return Err(IqdbError::InvalidMetric),
            }
            // Safe: `validate()` guaranteed `Some(m >= 1)` above.
            let m = cfg.pq_subvectors.unwrap_or(0);
            if m == 0 || !dim.is_multiple_of(m) {
                return Err(IqdbError::InvalidConfig {
                    reason: "IvfConfig.pq_subvectors must divide IvfIndex dim",
                });
            }
        }
        // Pre-allocate empty inverted lists so probe-time indexing is
        // unconditional even before training. We treat the centroid
        // Vec separately because its actual contents land at train()
        // time.
        let mut inverted_lists: Vec<Vec<InvertedListEntry>> = Vec::with_capacity(cfg.n_clusters);
        for _ in 0..cfg.n_clusters {
            inverted_lists.push(Vec::new());
        }
        Ok(Self {
            dim,
            metric,
            cfg,
            trained: false,
            centroids: Vec::new(),
            inverted_lists,
            id_to_cluster: HashMap::new(),
            next_seq: 0,
            live_count: 0,
            pq: None,
        })
    }

    /// The dimensionality the index was built for.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The distance metric the index was built for.
    #[must_use]
    pub fn metric(&self) -> DistanceMetric {
        self.metric
    }

    /// Number of searchable vectors in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.live_count
    }

    /// True when the index holds no searchable vectors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    /// True once [`Self::train`] has produced centroids.
    #[must_use]
    pub fn is_trained(&self) -> bool {
        self.trained
    }

    /// Current probe count.
    #[must_use]
    pub fn n_probes(&self) -> usize {
        self.cfg.n_probes
    }

    /// Override `n_probes` at runtime.
    ///
    /// Returns [`IqdbError::InvalidConfig`] when `n == 0` or `n >
    /// n_clusters`. Does not require the index to be trained — the
    /// validation here is identical to [`IvfConfig::validate`]'s
    /// rules.
    pub fn set_n_probes(&mut self, n: usize) -> Result<()> {
        if n == 0 {
            return Err(IqdbError::InvalidConfig {
                reason: "IvfIndex::set_n_probes requires n >= 1",
            });
        }
        if n > self.cfg.n_clusters {
            return Err(IqdbError::InvalidConfig {
                reason: "IvfIndex::set_n_probes requires n <= n_clusters",
            });
        }
        self.cfg.n_probes = n;
        Ok(())
    }

    /// Current IVF-PQ refine factor (see [`IvfConfig::pq_refine_factor`]).
    #[must_use]
    pub fn pq_refine_factor(&self) -> u32 {
        self.cfg.pq_refine_factor
    }

    /// Override `pq_refine_factor` at runtime.
    ///
    /// `0` disables refine (return ADC top-`k`); `N >= 1` shortlists
    /// `N × k` candidates by ADC and exact-reranks. No-op for
    /// IVF-Flat (`use_pq = false`) — the setter still stores the
    /// value but search ignores it. Any `u32` is legal; this method
    /// mirrors [`Self::set_n_probes`].
    pub fn set_pq_refine_factor(&mut self, factor: u32) {
        self.cfg.pq_refine_factor = factor;
    }

    /// Diagnostic snapshot of cluster occupancy. See [`IvfClusterStats`].
    #[must_use]
    pub fn cluster_stats(&self) -> IvfClusterStats {
        let sizes: Vec<usize> = if self.trained {
            self.inverted_lists.iter().map(|l| l.len()).collect()
        } else {
            Vec::new()
        };
        IvfClusterStats::from_sizes(self.cfg.n_clusters, sizes)
    }

    /// Train k-means on `sample`.
    ///
    /// Must be called before [`IndexCore::insert`] /
    /// [`IndexCore::search`] will succeed. After a successful train,
    /// `sample` is no longer referenced — the index only retains the
    /// derived centroids. See the crate-level docs for the
    /// determinism contract.
    ///
    /// Calling `train` on an already-trained index returns
    /// [`IqdbError::InvalidConfig`]; rebuilding centroids over the
    /// currently-stored vectors is what [`Self::retrain`] is for.
    pub fn train(&mut self, sample: &[&[f32]]) -> Result<()> {
        if self.trained {
            return Err(IqdbError::InvalidConfig {
                reason: "IvfIndex is already trained; use retrain() to rebuild centroids",
            });
        }
        let centroids = train_kmeans(
            self.dim,
            self.cfg.n_clusters,
            self.cfg.seed,
            sample,
            self.cfg.training_sample_size,
        )?;
        debug_assert_eq!(centroids.len(), self.cfg.n_clusters);
        self.centroids = centroids;
        // Second stage: train the IVF-PQ codebooks on the same working
        // set (plain-PQ; codebooks are cluster-independent so the ADC
        // table can be built once per query and reused across probes).
        // [`Self::retrain`] is the entry point that re-quantizes every
        // existing entry through a freshly-trained PQ after a rebuild.
        if self.cfg.use_pq {
            let pq = pq_variant::train_pq(&self.cfg, sample)?;
            self.pq = Some(pq);
        }
        self.trained = true;
        Ok(())
    }

    /// Rebuild centroids (and, when `use_pq`, PQ codebooks) over the
    /// currently-stored vectors, then reassign and re-encode every
    /// entry against the new model — without losing any data.
    ///
    /// This is the entry point for the workflow the spec calls
    /// out: cluster balance can drift after many deletes or after a
    /// distribution shift in inserted vectors, and a single
    /// `retrain()` call recovers a balanced inverted-list layout
    /// without forcing the caller to reconstruct the corpus.
    ///
    /// # Determinism
    ///
    /// The working set passed into k-means is the snapshot of stored
    /// vectors sorted ascending by their internal `seq` stamp
    /// (insertion order) and then deterministically subsampled down
    /// to [`IvfConfig::training_sample_size`] via the same
    /// SplitMix64 / partial Fisher–Yates procedure
    /// [`Self::train`] uses. Same seed + same insert/delete history
    /// → byte-identical centroids → byte-identical PQ codebooks →
    /// byte-identical re-encoded `PqCode`s.
    ///
    /// The seq-sort matters because [`IndexCore::delete`] uses
    /// `swap_remove`, which makes the *in-memory* layout of the
    /// inverted lists order-sensitive to the delete pattern. Sorting
    /// by `seq` removes that sensitivity — the retrain result
    /// depends only on the set of currently-live `(id, vector,
    /// metadata, seq)` tuples, not on how they came to live where
    /// they live.
    ///
    /// # Cap and complexity
    ///
    /// The k-means and PQ training both run on the **same** capped
    /// sample (size `min(live_count, training_sample_size)`), so
    /// retrain stays O(`training_sample_size`) for the training
    /// stages — never O(n) on a large index. Reassign + re-encode
    /// touches every entry, so the overall cost is
    /// O(`training_sample_size` · `n_clusters` + n · m) on a
    /// `dim`-dimensional, `m`-subvector PQ index.
    ///
    /// # Errors
    ///
    /// - [`IqdbError::InvalidConfig`] when the index has not yet
    ///   been trained (call [`Self::train`] first).
    /// - Propagates errors from the internal k-means trainer, the
    ///   PQ trainer, and
    ///   [`iqdb_quantize::Quantizer::quantize`] when the snapshot
    ///   fails the same shape checks `train` would.
    ///
    /// On an empty trained index (`len() == 0`), `retrain` is a
    /// no-op and returns `Ok(())`; the centroids stay as they were
    /// trained.
    pub fn retrain(&mut self) -> Result<()> {
        if !self.trained {
            return Err(IqdbError::InvalidConfig {
                reason: "IvfIndex must be trained before retrain()",
            });
        }
        if self.live_count == 0 {
            // Nothing to retrain on. Centroids stay; the caller
            // probably emptied the index and wants the next train()
            // to happen via insert/search semantics, not here.
            return Ok(());
        }

        // -- 1. Snapshot all entries; reset bookkeeping ---------------
        let mut snapshot: Vec<InvertedListEntry> = Vec::with_capacity(self.live_count);
        for list in self.inverted_lists.iter_mut() {
            let taken = std::mem::take(list);
            for entry in taken {
                snapshot.push(entry);
            }
        }
        // Restore empty per-cluster Vecs so search-pass indexing
        // stays unconditional during the reassign loop below.
        debug_assert!(self.inverted_lists.iter().all(|l| l.is_empty()));
        self.id_to_cluster.clear();
        self.live_count = 0;

        // -- 2. Order by seq for delete-pattern determinism ----------
        snapshot.sort_by_key(|e| e.seq);

        // -- 3. Build the capped working-set refs --------------------
        let all_refs: Vec<&[f32]> = snapshot.iter().map(|e| e.vector_slice()).collect();
        let target_len = self.cfg.training_sample_size.min(all_refs.len());
        let mut rng = SplitMix64::new(self.cfg.seed);
        let capped: Vec<&[f32]> = subsample_refs(&all_refs, target_len, &mut rng);

        // -- 4. Train k-means on the capped sample -------------------
        // Pass `training_sample_size = capped.len()` so the
        // `train_kmeans` internal subsample is a no-op — k-means and
        // the PQ retrain below must see *the same* working set
        // (plain-PQ contract).
        let centroids = train_kmeans(
            self.dim,
            self.cfg.n_clusters,
            self.cfg.seed,
            &capped,
            capped.len(),
        )?;
        debug_assert_eq!(centroids.len(), self.cfg.n_clusters);
        self.centroids = centroids;

        // -- 5. Retrain the PQ codebooks on the same capped sample ---
        if self.cfg.use_pq {
            let pq = pq_variant::train_pq(&self.cfg, &capped)?;
            self.pq = Some(pq);
        }

        // -- 6. Reassign + re-encode every entry ---------------------
        // The snapshot is iterated in seq-sorted order; the original
        // `seq` stamp on each entry is preserved so the top-k
        // tiebreaker invariants stay stable across the retrain.
        for mut entry in snapshot {
            let cluster = assign_to_cluster(&self.centroids, &entry.vector);
            if let Some(pq) = self.pq.as_ref() {
                // Re-encode against the freshly-trained codebooks.
                // The old `PqCode` is dropped as part of this
                // assignment.
                entry.pq_code = Some(pq.quantize(&entry.vector)?);
            }
            let id = entry.id.clone();
            let _prev = self.id_to_cluster.insert(id, cluster);
            self.inverted_lists[cluster].push(entry);
            self.live_count += 1;
        }

        Ok(())
    }

    /// Recommend an `n_probes` value that covers at least `coverage`
    /// of the live vector count when the largest clusters are
    /// probed first.
    ///
    /// Pure and side-effect-free: the caller composes with
    /// [`Self::set_n_probes`] to apply the suggestion:
    ///
    /// ```no_run
    /// # use iqdb_ivf::IvfIndex;
    /// # fn demo(idx: &mut IvfIndex) -> iqdb_types::Result<()> {
    /// let n = idx.suggest_n_probes(0.80)?;
    /// idx.set_n_probes(n)?;
    /// # Ok(()) }
    /// ```
    ///
    /// Returning a number — never widening probes at query time —
    /// preserves the "you always probe N, latency is predictable"
    /// property that makes IVF attractive next to graph indexes.
    ///
    /// # Algorithm
    ///
    /// Collect the live cluster sizes, sort them descending, walk
    /// the cumulative sum, and return the 1-based index of the
    /// first prefix whose total reaches
    /// `ceil(live_count * coverage)`. The result is clamped into
    /// `[1, n_clusters]`.
    ///
    /// Properties enforced by the test plan: monotone in
    /// `coverage` (higher coverage → at least as many probes),
    /// and the suggested probes cover at least the requested
    /// fraction.
    ///
    /// # Errors
    ///
    /// Returns [`IqdbError::InvalidConfig`] when `coverage` is not
    /// finite or falls outside `[0.0, 1.0]`. NaN is rejected by the
    /// finiteness check.
    ///
    /// # Edge cases
    ///
    /// - `!is_trained()` or `is_empty()` → returns `Ok(1)`.
    /// - `n_clusters == 1` → returns `Ok(1)`.
    /// - `coverage == 0.0` → returns `Ok(1)`.
    /// - `coverage == 1.0` → returns `Ok(n_clusters)`.
    pub fn suggest_n_probes(&self, coverage: f32) -> Result<usize> {
        if !coverage.is_finite() || !(0.0..=1.0).contains(&coverage) {
            return Err(IqdbError::InvalidConfig {
                reason: "IvfIndex::suggest_n_probes requires coverage in [0.0, 1.0]",
            });
        }
        if !self.trained || self.live_count == 0 || self.cfg.n_clusters == 1 {
            return Ok(1);
        }
        if coverage == 0.0 {
            return Ok(1);
        }
        if coverage == 1.0 {
            return Ok(self.cfg.n_clusters);
        }
        // Coverage walk over the descending cluster-size prefix.
        let mut sizes: Vec<usize> = self.inverted_lists.iter().map(|l| l.len()).collect();
        sizes.sort_by(|a, b| b.cmp(a));
        // `ceil(live_count * coverage)` in f64 to keep the conversion
        // honest at large `live_count`; `live_count == 0` was
        // already short-circuited above.
        let target_f = (self.live_count as f64) * (coverage as f64);
        let target = target_f.ceil() as usize;
        let mut cumsum: usize = 0;
        for (i, &s) in sizes.iter().enumerate() {
            cumsum = cumsum.saturating_add(s);
            if cumsum >= target {
                let n = i + 1;
                return Ok(n.clamp(1, self.cfg.n_clusters));
            }
        }
        // If we walked every cluster without hitting `target`, the
        // index is small enough that every probe is needed. Fall
        // back to `n_clusters`.
        Ok(self.cfg.n_clusters)
    }

    // -- Crate-internal accessors used by search.rs + pq_variant.rs --

    pub(crate) fn centroids_slice(&self) -> &[Vec<f32>] {
        &self.centroids
    }

    pub(crate) fn inverted_list(&self, cluster: usize) -> &[InvertedListEntry] {
        &self.inverted_lists[cluster]
    }

    pub(crate) fn cfg(&self) -> &IvfConfig {
        &self.cfg
    }

    pub(crate) fn pq(&self) -> Option<&ProductQuantizer> {
        self.pq.as_ref()
    }

    fn check_dim(&self, vector_len: usize) -> Result<()> {
        if vector_len != self.dim {
            return Err(IqdbError::DimensionMismatch {
                expected: self.dim,
                found: vector_len,
            });
        }
        Ok(())
    }

    /// Common entry-gate for the four trained-only methods.
    fn require_trained(&self) -> Result<()> {
        if !self.trained {
            return Err(IqdbError::InvalidConfig {
                reason: "IvfIndex must be trained before use",
            });
        }
        Ok(())
    }

    fn approximate_memory_bytes(&self) -> usize {
        let arc_header_bytes = 2 * size_of::<usize>();
        let centroid_bytes: usize = self
            .centroids
            .iter()
            .map(|c| c.capacity() * size_of::<f32>())
            .sum::<usize>()
            + self.centroids.capacity() * size_of::<Vec<f32>>();
        let mut list_bytes: usize = 0;
        for list in &self.inverted_lists {
            list_bytes += list.capacity() * size_of::<InvertedListEntry>();
            for entry in list {
                list_bytes += entry.vector.len() * size_of::<f32>() + arc_header_bytes;
            }
        }
        let id_to_cluster_bytes =
            self.id_to_cluster.capacity() * (size_of::<VectorId>() + size_of::<usize>());
        centroid_bytes + list_bytes + id_to_cluster_bytes
    }

    fn insert_inner(
        &mut self,
        id: VectorId,
        vector: Arc<[f32]>,
        metadata: Option<Metadata>,
    ) -> Result<()> {
        self.require_trained()?;
        self.check_dim(vector.len())?;
        if self.id_to_cluster.contains_key(&id) {
            return Err(IqdbError::Duplicate);
        }
        let seq = self.next_seq;
        self.next_seq = self
            .next_seq
            .checked_add(1)
            .ok_or(IqdbError::InvalidConfig {
                reason: "IvfIndex insertion sequence counter overflowed u64",
            })?;
        let cluster = assign_to_cluster(&self.centroids, &vector);
        // When in IVF-PQ mode, encode the vector through the trained
        // quantizer up-front and store the code alongside the Arc.
        // The retained Arc is the source of truth for the refine pass;
        // the code is what the ADC scan consumes.
        let pq_code = match self.pq.as_ref() {
            Some(pq) => Some(pq.quantize(&vector)?),
            None => None,
        };
        let _prev = self.id_to_cluster.insert(id.clone(), cluster);
        self.inverted_lists[cluster].push(InvertedListEntry {
            id,
            vector,
            pq_code,
            metadata,
            seq,
        });
        self.live_count += 1;
        Ok(())
    }

    fn delete_inner(&mut self, id: &VectorId) -> Result<()> {
        let cluster = self.id_to_cluster.remove(id).ok_or(IqdbError::NotFound)?;
        let list = &mut self.inverted_lists[cluster];
        // Linear scan inside the single cluster; O(cluster_size).
        let pos = list
            .iter()
            .position(|e| &e.id == id)
            .ok_or(IqdbError::NotFound)?;
        let _entry = list.swap_remove(pos);
        self.live_count -= 1;
        Ok(())
    }

    fn search_inner(&self, query: &[f32], params: &SearchParams) -> Result<Vec<Hit>> {
        self.require_trained()?;
        search::ivf_search(self, query, params)
    }
}

impl IndexCore for IvfIndex {
    #[tracing::instrument(
        level = "debug",
        skip_all,
        fields(vector_id = %id, n = self.live_count, dim = self.dim),
    )]
    fn insert(
        &mut self,
        id: VectorId,
        vector: Arc<[f32]>,
        metadata: Option<Metadata>,
    ) -> Result<()> {
        match self.insert_inner(id, vector, metadata) {
            Ok(()) => Ok(()),
            Err(err) => {
                tracing::error!(
                    error.kind = err.kind(),
                    error.reason = err.caption(),
                    "ivf insert failed",
                );
                Err(err)
            }
        }
    }

    #[tracing::instrument(
        level = "debug",
        skip_all,
        fields(vector_id = %id, n = self.live_count),
    )]
    fn delete(&mut self, id: &VectorId) -> Result<()> {
        match self.delete_inner(id) {
            Ok(()) => Ok(()),
            Err(err) => {
                tracing::error!(
                    error.kind = err.kind(),
                    error.reason = err.caption(),
                    "ivf delete failed",
                );
                Err(err)
            }
        }
    }

    #[tracing::instrument(
        level = "debug",
        skip_all,
        fields(
            k = params.k,
            dim = self.dim,
            n = self.live_count,
            filter = params.filter.is_some(),
            metric = ?params.metric,
        ),
    )]
    fn search(&self, query: &[f32], params: &SearchParams) -> Result<Vec<Hit>> {
        match self.search_inner(query, params) {
            Ok(hits) => Ok(hits),
            Err(err) => {
                tracing::error!(
                    error.kind = err.kind(),
                    error.reason = err.caption(),
                    "ivf search failed",
                );
                Err(err)
            }
        }
    }

    fn len(&self) -> usize {
        IvfIndex::len(self)
    }

    fn is_empty(&self) -> bool {
        IvfIndex::is_empty(self)
    }

    fn dim(&self) -> usize {
        IvfIndex::dim(self)
    }

    fn metric(&self) -> DistanceMetric {
        IvfIndex::metric(self)
    }

    fn flush(&mut self) -> Result<()> {
        // IVF is purely in-memory; persistence is a separate crate.
        // Nothing to flush.
        Ok(())
    }

    fn stats(&self) -> IndexStats {
        IndexStats {
            n_vectors: self.live_count,
            memory_bytes: self.approximate_memory_bytes(),
            disk_bytes: None,
            index_type: "ivf",
            extra: None,
        }
    }
}

impl Index for IvfIndex {
    type Config = IvfConfig;

    fn new(dim: usize, metric: DistanceMetric, config: Self::Config) -> Result<Self> {
        Self::new_unconfigured(dim, metric, config)
    }
}
