# iqdb-ivf -- Roadmap

> Path from scaffold to a stable 1.0. Hard parts were front-loaded; each phase
> had hard exit criteria. **Status: 1.0.0 shipped.** With every dependency
> (`iqdb-types`, `iqdb-distance`, `iqdb-index`, `iqdb-filter`,
> `iqdb-quantize`) stable at 1.0, the implementation was completed against the
> published crates and the whole feature set landed in a single stable release;
> the 0.2–0.5 phases below are recorded as completed work folded into 1.0.0.
>
> **Anti-deferral rule:** no listed hard task moves to a later phase unless this
> file records the move and the reason.

---

## v0.1.0 -- Scaffold (DONE)

Compiles, CI green, structure correct, no domain logic.

- [x] Manifest, README, CHANGELOG, REPS, license, CI, lints in place.
- [x] API surface sketched in `docs/API.md`.

---

## v0.2.0 -- k-means training + assignment + IVF-Flat search (DONE — folded into 1.0.0)

- [x] Deterministic k-means++ training over a representative sample.
- [x] Nearest-centroid assignment and the two-pass IVF-Flat search.
- [x] Every public item has rustdoc + a runnable example.
- [x] Core invariants property-tested (full-probe equivalence to the exact
      `iqdb-flat` oracle).

---

## v0.3.0 -- probe tuning + retrain + filtered cluster search (DONE — folded into 1.0.0)

- [x] `n_probes` / `set_n_probes` and `suggest_n_probes`.
- [x] `retrain` over the currently-stored vectors.
- [x] Metadata-filtered search via `iqdb-filter`.
- [x] New surface tested and benchmarked where it is a hot path.

---

## v0.4.0 -- IVF-PQ via iqdb-quantize + feature freeze (DONE — folded into 1.0.0)

- [x] IVF-PQ (`use_pq`) with `PqAdcTables` ADC scoring and exact refine.
- [x] No `todo!` / `unimplemented!`. Feature freeze declared.

---

## v0.5.0 -- recall validation at scale + API freeze (DONE — folded into 1.0.0)

- [x] Recall validated at scale for IVF-Flat and IVF-PQ against the exact
      `iqdb-flat` oracle (`tests/recall_at_scale.rs`).
- [x] Public API frozen (recorded below). `cargo audit` + `cargo deny` clean.

---

## v1.0.0 -- Stable (DONE)

- [x] Definition of Done (DIRECTIVES section 7) satisfied.
- [x] Public API frozen until 2.0.
- [x] Release note written (`docs/release/v1.0.0.md`).

### Frozen 1.0 public API

- `IvfIndex`: `new` (via `Index`), `new_unconfigured`, `train`, `retrain`,
  `dim`, `metric`, `len`, `is_empty`, `is_trained`, `n_probes`,
  `set_n_probes`, `suggest_n_probes`, `pq_refine_factor`,
  `set_pq_refine_factor`, `cluster_stats`, and the full `IndexCore`
  implementation (`insert`, `insert_batch`, `delete`, `search`,
  `search_batch`, `flush`, `stats`).
- `IvfConfig` (fields + `with_*` builders + `validate` + `Default`).
- `IvfClusterStats` (fields).
- `VERSION`; `Hit` re-export.

---

## Out of scope for 1.0

- Persistence/caching -- separate crates.
- Residual-PQ (per-cluster codebooks) -- additive 1.x candidate, not a break.
- Distributed clustering -- reserved phase.
