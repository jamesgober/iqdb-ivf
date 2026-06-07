//! [`IvfConfig`] — the typed configuration consumed by
//! [`iqdb_index::Index::new`] for [`crate::IvfIndex`].
//!
//! Mirrors the seed-carrying shape of `iqdb_hnsw::HnswConfig` so the
//! determinism contract surfaces on the public type: identical
//! `seed` + identical training sample → identical centroids. Use
//! [`IvfConfig::default`] for the operating point or the builder-style
//! `with_*` methods to override a single field.

use iqdb_types::{IqdbError, Result};

/// Default number of inverted-list partitions produced by k-means.
///
/// The spec heuristic is `sqrt(N)` (or `4 * sqrt(N)` for larger
/// corpora). `256` is the operating point for `N ≈ 65_536`, which is
/// the inflection point above which IVF starts to beat FlatIndex on
/// real datasets. The default is conservative; tuning to a corpus
/// happens at construction.
const DEFAULT_N_CLUSTERS: usize = 256;

/// Default number of clusters probed at query time.
///
/// `8` of `256` is a strong recall/latency baseline for the FAISS-style
/// `n_probes ≈ sqrt(n_clusters)` heuristic. Probes are the latency knob
/// queries reach for through [`crate::IvfIndex::set_n_probes`].
const DEFAULT_N_PROBES: usize = 8;

/// Default cap on training-sample size.
///
/// Above this the trainer subsamples deterministically via the seeded
/// PRNG; smaller samples run faster and produce essentially identical
/// centroids on real distributions. `65_536` is the canonical FAISS
/// default and is large enough to over-sample the corpus when
/// `n_clusters = 256`.
const DEFAULT_TRAINING_SAMPLE_SIZE: usize = 65_536;

/// Default seed for the k-means++ PRNG.
///
/// Same constant style as `HnswConfig`'s default so a project that pins
/// one seed across the index family gets reproducible builds with
/// minimal ceremony.
const DEFAULT_SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;

/// Default IVF-PQ refine factor.
///
/// `4` is the standard FAISS production default. With `pq_refine_factor
/// = 4`, the IVF-PQ search shortlists `4 × k` candidates by ADC and
/// then exact-reranks them using the retained `Arc<[f32]>` vectors
/// before returning top-`k`. Set to `0` to disable refine and return
/// the pure ADC top-`k`. Ignored when [`IvfConfig::use_pq`] is `false`.
const DEFAULT_PQ_REFINE_FACTOR: u32 = 4;

/// Configuration for [`crate::IvfIndex`] construction (see
/// [`iqdb_index::Index::new`]).
///
/// All fields have documented defaults; see the field-level docs and
/// the crate `README.md` for the tradeoffs each one controls.
///
/// # Examples
///
/// ```
/// use iqdb_ivf::IvfConfig;
///
/// let cfg = IvfConfig::default();
/// assert_eq!(cfg.n_clusters, 256);
/// assert_eq!(cfg.n_probes, 8);
///
/// let tuned = IvfConfig::default()
///     .with_n_clusters(64)
///     .with_n_probes(4)
///     .with_seed(42);
/// assert_eq!(tuned.n_clusters, 64);
/// assert_eq!(tuned.n_probes, 4);
/// assert_eq!(tuned.seed, 42);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IvfConfig {
    /// Number of k-means partitions (inverted lists) the trainer
    /// produces.
    ///
    /// Spec heuristic: `sqrt(N)` for moderate corpora, `4 * sqrt(N)`
    /// for very large ones. Must be at least `1`. Default `256`.
    pub n_clusters: usize,

    /// Number of clusters searched at query time.
    ///
    /// Larger values raise recall at higher per-query cost. Must be
    /// at least `1` and no greater than [`n_clusters`](Self::n_clusters).
    /// Default `8`.
    pub n_probes: usize,

    /// Cap on the training sample passed to k-means.
    ///
    /// When the caller supplies more vectors than this, the trainer
    /// subsamples down to this many via the seeded PRNG. Must be at
    /// least `1`. Default `65_536`.
    pub training_sample_size: usize,

    /// Enable Product Quantization within each inverted list.
    ///
    /// When `true`, [`Self::pq_subvectors`] must be `Some(m)` with
    /// `m >= 1` and `m | dim` at index-construction time. The IVF-PQ
    /// branch trains a [`iqdb_quantize::ProductQuantizer`] over the
    /// same working set used for the coarse k-means (plain-PQ), stores
    /// a per-entry [`iqdb_quantize::PqCode`] alongside the retained
    /// `Arc<[f32]>` vector, and scores intra-cluster candidates via
    /// ADC. Supported metrics: `Euclidean`, `DotProduct`, `Manhattan`
    /// — `Cosine` and `Hamming` are rejected at construction with
    /// [`IqdbError::InvalidMetric`]. Defaults to `false` (IVF-Flat).
    pub use_pq: bool,

    /// Subvector count `M` for IVF-PQ.
    ///
    /// Required to be `Some(m)` with `m >= 1` and `m | dim` whenever
    /// [`use_pq`](Self::use_pq) is `true`. Ignored when `use_pq` is
    /// `false`. Each subvector compresses to one byte (`K = 256`),
    /// so smaller `m` compresses harder at the cost of more
    /// reconstruction error per code.
    pub pq_subvectors: Option<usize>,

    /// IVF-PQ refine factor.
    ///
    /// `0` disables refine: the search returns the pure ADC top-`k`.
    /// `N >= 1` enables refine: the search shortlists `N × k`
    /// candidates by ADC, then exact-reranks the shortlist using the
    /// retained `Arc<[f32]>` vectors (same distance path as IVF-Flat,
    /// same DotProduct sign convention) before returning top-`k`.
    /// Default `4`. Ignored when [`use_pq`](Self::use_pq) is `false`.
    /// Tunable at runtime via [`crate::IvfIndex::set_pq_refine_factor`].
    pub pq_refine_factor: u32,

    /// Seed for the internal SplitMix64 PRNG used by k-means++
    /// initialization and by deterministic subsampling of the
    /// training set.
    ///
    /// Identical `seed` + identical training sample → byte-identical
    /// centroids on every platform. When [`use_pq`](Self::use_pq) is
    /// `true`, the same seed flows into the PQ codebook trainer so
    /// the per-subvector codebooks are also reproducible.
    pub seed: u64,
}

impl IvfConfig {
    /// Override `n_clusters`.
    #[must_use]
    pub fn with_n_clusters(mut self, n_clusters: usize) -> Self {
        self.n_clusters = n_clusters;
        self
    }

    /// Override `n_probes`.
    #[must_use]
    pub fn with_n_probes(mut self, n_probes: usize) -> Self {
        self.n_probes = n_probes;
        self
    }

    /// Override `training_sample_size`.
    #[must_use]
    pub fn with_training_sample_size(mut self, training_sample_size: usize) -> Self {
        self.training_sample_size = training_sample_size;
        self
    }

    /// Override `use_pq`.
    ///
    /// When `true`, [`Self::pq_subvectors`] must also be set; the
    /// metric/dim divisibility checks happen at
    /// [`IvfIndex::new_unconfigured`](iqdb_index::Index::new) time
    /// when both `dim` and `metric` are known.
    #[must_use]
    pub fn with_use_pq(mut self, use_pq: bool) -> Self {
        self.use_pq = use_pq;
        self
    }

    /// Override `pq_subvectors`.
    ///
    /// Required to be `Some(m)` with `m >= 1` and `m | dim` whenever
    /// [`use_pq`](Self::use_pq) is `true`; otherwise ignored.
    #[must_use]
    pub fn with_pq_subvectors(mut self, pq_subvectors: Option<usize>) -> Self {
        self.pq_subvectors = pq_subvectors;
        self
    }

    /// Override `pq_refine_factor`.
    ///
    /// `0` disables refine; `N >= 1` shortlists `N × k` candidates by
    /// ADC and exact-reranks. Ignored when [`use_pq`](Self::use_pq) is
    /// `false`.
    #[must_use]
    pub fn with_pq_refine_factor(mut self, pq_refine_factor: u32) -> Self {
        self.pq_refine_factor = pq_refine_factor;
        self
    }

    /// Override the PRNG seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Validate the configuration.
    ///
    /// Called by [`IvfIndex::new`](iqdb_index::Index::new) before the
    /// index is built.
    /// The error variant is always [`IqdbError::InvalidConfig`] with a
    /// short `&'static str` `reason` naming exactly which check failed,
    /// so a caller can branch on the message or thread it into a log.
    pub fn validate(&self) -> Result<()> {
        if self.n_clusters == 0 {
            return Err(IqdbError::InvalidConfig {
                reason: "IvfConfig.n_clusters must be greater than zero",
            });
        }
        if self.n_probes == 0 {
            return Err(IqdbError::InvalidConfig {
                reason: "IvfConfig.n_probes must be greater than zero",
            });
        }
        if self.n_probes > self.n_clusters {
            return Err(IqdbError::InvalidConfig {
                reason: "IvfConfig.n_probes must be <= n_clusters",
            });
        }
        if self.training_sample_size == 0 {
            return Err(IqdbError::InvalidConfig {
                reason: "IvfConfig.training_sample_size must be greater than zero",
            });
        }
        if self.use_pq {
            match self.pq_subvectors {
                Some(m) if m >= 1 => {}
                Some(_) => {
                    return Err(IqdbError::InvalidConfig {
                        reason: "IvfConfig.pq_subvectors must be >= 1 when use_pq = true",
                    });
                }
                None => {
                    return Err(IqdbError::InvalidConfig {
                        reason: "IvfConfig.use_pq = true requires pq_subvectors = Some(_)",
                    });
                }
            }
            // The `m | dim` divisibility check and the metric guard
            // (Cosine/Hamming → InvalidMetric) happen at
            // `IvfIndex::new_unconfigured` time, where both `dim` and
            // `metric` are known. `pq_refine_factor` is always legal
            // (a `u32` can't be negative; `0` = no refine).
        }
        Ok(())
    }
}

impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            n_clusters: DEFAULT_N_CLUSTERS,
            n_probes: DEFAULT_N_PROBES,
            training_sample_size: DEFAULT_TRAINING_SAMPLE_SIZE,
            use_pq: false,
            pq_subvectors: None,
            pq_refine_factor: DEFAULT_PQ_REFINE_FACTOR,
            seed: DEFAULT_SEED,
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn default_values_are_the_documented_operating_point() {
        let cfg = IvfConfig::default();
        assert_eq!(cfg.n_clusters, 256);
        assert_eq!(cfg.n_probes, 8);
        assert_eq!(cfg.training_sample_size, 65_536);
        assert!(!cfg.use_pq);
        assert_eq!(cfg.pq_subvectors, None);
        assert_eq!(cfg.pq_refine_factor, 4);
        assert_eq!(cfg.seed, 0xDEAD_BEEF_CAFE_F00D);
    }

    #[test]
    fn with_helpers_compose() {
        let cfg = IvfConfig::default()
            .with_n_clusters(16)
            .with_n_probes(4)
            .with_training_sample_size(1_024)
            .with_seed(42);
        assert_eq!(cfg.n_clusters, 16);
        assert_eq!(cfg.n_probes, 4);
        assert_eq!(cfg.training_sample_size, 1_024);
        assert_eq!(cfg.seed, 42);
    }

    #[test]
    fn validate_accepts_defaults() {
        assert!(IvfConfig::default().validate().is_ok());
    }

    #[test]
    fn validate_rejects_zero_n_clusters() {
        let err = IvfConfig::default()
            .with_n_clusters(0)
            .validate()
            .unwrap_err();
        match err {
            IqdbError::InvalidConfig { reason } => {
                assert!(reason.contains("n_clusters"));
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_zero_n_probes() {
        let err = IvfConfig::default()
            .with_n_probes(0)
            .validate()
            .unwrap_err();
        match err {
            IqdbError::InvalidConfig { reason } => {
                assert!(reason.contains("n_probes"));
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_n_probes_exceeding_n_clusters() {
        let err = IvfConfig::default()
            .with_n_clusters(4)
            .with_n_probes(8)
            .validate()
            .unwrap_err();
        match err {
            IqdbError::InvalidConfig { reason } => {
                assert!(reason.contains("n_probes"));
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_zero_training_sample_size() {
        let err = IvfConfig::default()
            .with_training_sample_size(0)
            .validate()
            .unwrap_err();
        match err {
            IqdbError::InvalidConfig { reason } => {
                assert!(reason.contains("training_sample_size"));
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_use_pq_true_without_pq_subvectors() {
        let err = IvfConfig::default()
            .with_use_pq(true)
            .validate()
            .unwrap_err();
        match err {
            IqdbError::InvalidConfig { reason } => {
                assert!(reason.contains("pq_subvectors"));
                assert!(reason.contains("Some"));
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn validate_rejects_use_pq_true_with_zero_pq_subvectors() {
        let err = IvfConfig::default()
            .with_use_pq(true)
            .with_pq_subvectors(Some(0))
            .validate()
            .unwrap_err();
        match err {
            IqdbError::InvalidConfig { reason } => {
                assert!(reason.contains("pq_subvectors"));
                assert!(reason.contains(">= 1"));
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn validate_accepts_use_pq_true_with_valid_pq_subvectors() {
        // The `m | dim` check moves to `IvfIndex::new_unconfigured`,
        // so config-level validate accepts any `Some(m >= 1)`.
        let cfg = IvfConfig::default()
            .with_use_pq(true)
            .with_pq_subvectors(Some(8));
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_accepts_pq_refine_factor_zero() {
        let cfg = IvfConfig::default()
            .with_use_pq(true)
            .with_pq_subvectors(Some(8))
            .with_pq_refine_factor(0);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn with_pq_refine_factor_sets_field() {
        let cfg = IvfConfig::default().with_pq_refine_factor(16);
        assert_eq!(cfg.pq_refine_factor, 16);
    }
}
