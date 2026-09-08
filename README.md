# Replication data and code

**The folding bias of training noise in paired surrogate comparisons**  
Wooyeol Jeong · Monghwan Seo · Sungchul Lee — Department of Mathematics, Yonsei University

This archive holds the processed data behind every table in the paper, together with the
analysis scripts that produced it. It is released for **inspection and verification**, not as a
turnkey pipeline — see [Reproducing results](#reproducing-results) for exactly what runs here
and what does not.

Licensed under the MIT License (see `LICENSE`). Cite via `CITATION.cff`.

## Layout

```
data/   processed datasets — every published table cell is read from these files
  unified/       main 387-configuration suite and derived analyses
  b9_crossfit/   fully cross-fitted re-scoring (incl. archived per-seed discrepancies)
  b10_lowsnr/    targeted low-signal panel (+ cache/ density fields it is rebuilt from)
  b11_nongauss/  non-Gaussian control suite
  b12_efficiency/ plug-in vs nonparametric estimator comparison
  b13_supproxy/  vanilla-basis truncation sweep (regenerated — see note below)
  experiments/   boundary experiments (architecture, width, Burgers, dropout, protocol)
  kou/           per-window Kou calibrations for the BTC-informed stress mappings
  legacy/        one file from an earlier Heston-only lineage — see the warning below
code/   analysis scripts (transparency; see Reproducing results)
  builders/      the seven table builders
meta/   seed_lists.csv · v2_splits.npz · suite_manifest.csv
```

> **`data/legacy/` — earlier lineage.** `correlation_by_protocol.csv` comes from an earlier,
> Heston-only experiment series whose conventions differ from the main suite. It is included
> because Table `tab:cross-proto` is computed from it, and for no other purpose. Do not mix its
> numbers with the main suite.

## Table → file

All 31 tables in the paper. 27 are data-backed; the rest are described below the table.

| Table | Released file |
|---|---|
| `tab:efficiency` | `data/b12_efficiency/tab_efficiency.csv` |
| `tab:cross-proto` | `data/legacy/correlation_by_protocol.csv` |
| `tab:validation-summary` | `data/unified/b5_thm35_all.csv` |
| `tab:js-decomp` | `data/unified/bias_removal_results.csv` |
| `tab:gstar-deviation` | `data/unified/bias_reduction_gstar.csv + data/unified/bias_reduction_gstar_decomposed.csv` |
| `tab:kernel-structure` | `data/unified/b4_table4_kernel_structure.csv` |
| `tab:multi-event-summary` | `data/unified/multi_event_thm35.csv` |
| `tab:lowsnr-summary` | `data/b10_lowsnr/b10_lowsnr_all.csv + data/b10_lowsnr/b10_lowsnr_summary.csv` |
| `tab:estimator-classes` | `data/unified/estimator_compare.csv` |
| `tab:k99-quantile` | `data/unified/multidim_hedging_per_sample.csv` |
| `tab:sup-proxy-sensitivity` | `data/b13_supproxy/b13_supproxy_cells.csv` |
| `tab:multi-event-distance` | `data/unified/btc_events_mahalanobis_distance.csv  <- ADD A COPY RULE: source is output/extrapolation/btc_events_mahalanobis_distance.csv, which is outside every path the assembler currently walks` |
| `tab:per-exp-validation` | `data/unified/per_experiment_validation.csv` |
| `tab:per-exp-gaussianity` | `data/unified/per_config_gaussianity_v2.csv + data/unified/per_experiment_gaussianity.csv` |
| `tab:sup-boundary-stress` | `data/unified/seed_sweep_breakdown.csv` |
| `tab:sup-estimator-cells` | `data/unified/estimator_compare.csv` |
| `tab:sup-mix-cells` | `data/unified/mix_estimator.csv` |
| `tab:sup-mm1` | `data/unified/mm1_queue.csv` |
| `tab:e1-arch` | `data/experiments/e1_vs_deeponet.csv + data/experiments/e1_kou_vs_deeponet.csv` |
| `tab:e2-width` | `data/experiments/e2_width_summary.csv` |
| `tab:e4-burgers` | `data/experiments/e4_burgers_diagnostics.csv + data/experiments/e4_deployment_rule.csv` |
| `tab:e5-dropout` | `data/experiments/e5_dropout_diagnostics_n20.csv` |
| `tab:e5i-protocol` | `data/experiments/e5i_protocol_comparison.csv` |
| `tab:e6-mlp` | `not released — its two source files live in output/mlp_replication/, a directory outside every prefix the assembler walks` |
| `tab:b7-portfolio` | `data/unified/b7_portfolio_v2.csv` |
| `tab:crossfit` | `data/b9_crossfit/b9_crossfit_all.csv + data/b9_crossfit/b9_crossfit_summary.csv` |
| `tab:nongauss` | `data/b11_nongauss/b11_cells.csv + data/b11_nongauss/b11_gates.json` |

**Not backed by a single released file (4):**

- `tab:building-blocks` — hand-authored — no measured numbers.
- `tab:provenance` — hand-authored — no measured numbers.
- `tab:boundary-summary` — narrative synthesis — each row summarises a different experiment.
- `tab:param-pairs` — hand-authored — no measured numbers.

`tab:boundary-summary` is a synthesis table; its rows draw on
`data/unified/{seed_sweep_breakdown,lowsnr_sweep,dist_ladder,extrapolation_oob,kou_tail_noise,
estimator_compare,mix_estimator,mm1_queue,mm1_Tladder}.csv` and
`data/experiments/{e1_*,e2_width_summary,e4_*,e5_dropout_diagnostics_n20}.csv`.

### Two caveats on specific tables

- **`tab:sup-proxy-sensitivity`** — the original sweep script was not preserved. The released
  cells are in `data/b13_supproxy/b13_supproxy_cells.csv`; the regeneration inputs (8 npz, 24 MB)
  are available on request. `code/builders/build_b13.py` is included for inspection and **requires
  the pair npz** to run. Of the table's 40 printed cells, 38 reproduce exactly; the two Kou
  median-‖α‖ entries differ by one unit at a rounding boundary (the recomputed value is 1012.5177,
  printed as 1012). `data/b13_supproxy/` also holds the gate checks (condition number and held-out
  recovery both match the stored `verify_aproj.csv` / `verify_loss.csv` values exactly).
- **`tab:estimator-classes`** — the printed Lilliefors column is not in
  `data/unified/estimator_compare.csv`, which carries the KS rejection counts only. That column
  is not reproducible from the released files.

## Reproducing results

With `data/` and `code/` alone, a downloader can check every published table and figure cell against the released files — the numbers in the paper are read straight out of the shipped `b9_crossfit_*.csv`, `b10_lowsnr_*.csv`/`verdict.json`, `b11_*.csv`/`b11_gates.json`, `tab_efficiency.csv`, and `b13_supproxy_cells.csv`/`b13_gates.json`, which are included in full. One of the seven builders re-runs end to end from the release tree: `build_b12.py` has no file inputs at all (closed-form mpmath evaluation at dps=40 plus a fixed-seed Monte Carlo, seed 20260902). `build_b13.py` is likewise deterministic and contains no RNG, but reads the eight `{heston,kou}_pair{A..D}.npz` arrays, which are not redistributed here and are available on request. The other five — `build_b9.py`, `build_b10.py`, `build_b10_i2.py`, `build_b11.py`, `build_b11_tree.py` — will not import here: each does `import b5_thm35_387` at module scope, and `code/b5_thm35_387.py` imports `torch` (line 37) and `TNO.common.tno_model` (line 47) unconditionally, so they raise `ModuleNotFoundError` before their cache-hit branch is ever reached. We want to be precise that this is an import-time dependency rather than a computational one: on the cached path none of the five loads a checkpoint, and four of them use nothing from that module except the closed-form `folded_normal_mean` (lines 82–87, numpy and `scipy.special.erf` only). The trained checkpoints (5.69 GB of `.pt`), the `TNO` package (`TNO/common/tno_model.py`), and the `TPDF` package (`TPDF/core/semi_closed_joint.py`, required by `code/heston_teacher.py` line 14 for the Lewis-formula ground truth) are not redistributed here and are available from the corresponding author on request. The scripts are shipped so that the exact procedure behind every published number can be inspected line by line; they are not a turnkey pipeline, and the five NN-suite builders are transparency artifacts rather than runnable ones.

### Run order

| # | Script | Runs from this archive? |
|---|---|---|
| 1 | `code/builders/build_b12.py` | **yes** |
| 2 | `code/builders/build_b13.py` | **yes** |
| 3 | `code/builders/build_b9.py` | no |
| 4 | `code/builders/build_b10.py` | no |
| 5 | `code/builders/build_b10_i2.py` | no |
| 6 | `code/builders/build_b11.py` | no |
| 7 | `code/builders/build_b11_tree.py` | no |

Run everything from the repository root. If you have the full materials, the dependent builders
must run in the order shown: `build_b10.py` → `build_b10_i2.py` → `build_b11.py` →
`build_b11_tree.py`. **`build_b10_i2.py` cannot be run on its own** — it asserts that
`b10_lowsnr_all.csv` still has its 72 pre-append rows, and the shipped file has 87, so
`build_b10.py` must rewrite it first. `build_b9.py`, `build_b12.py` and `build_b13.py` are
independent of that chain. The builders append to `build.log`, so that file is never
byte-comparable across runs.

## `meta/`

| File | Contents |
|---|---|
| `seed_lists.csv` | 25 checkpoint ensembles covering all 3,399 trained models: directory, role, filename pattern, `zero_pad` width, seed range. Seed numbering is contiguous in every ensemble but one. **`zero_pad` matters** — seed 60 is `seed_060.pt` in one ensemble and `seed_0060.pt` in another. |
| `v2_splits.npz` | The V2 resampling permutations, materialised so they no longer depend on a NumPy RNG stream: 387 arrays of shape (20, n_seeds) for the cross-fit suite plus one (20, 200) array for the low-signal panel. Replaying these reproduces every published `V2_median` exactly. |
| `suite_manifest.csv` | One row per (configuration, payoff) across the 387-configuration suite: identifiers, calibration/evaluation split sizes, and the model-parameter pair λ₁/λ₂ as JSON. |

λ values are derived for **333 of 387 rows** — every Heston block. The remaining 54 (the two Kou
stages and the BTC-informed blocks) carry `lambda_status = 미확인`; each row's `lambda_source`
column names where those parameters actually live.

## What is not here

- **Trained checkpoints** (3,399 files, 5.69 GB). Not redistributed; available from the
  corresponding author on reasonable request. `meta/seed_lists.csv` documents what exists.
- **The `TNO` and `TPDF` packages**, which `code/b5_thm35_387.py` and `code/heston_teacher.py`
  import. Available on request.
- **Bitcoin/USDT 1-minute spot prices**, obtained from the Binance public API and not
  redistributed here. Scripts that consume them carry a `<BTC_SPOT_1M_PARQUET>` placeholder.
- **The eight per-pair `.npz` arrays** (`{heston,kou}_pair{A..D}`, 24 MB) holding the noise
  kernels, d-bar and g* that `build_b13.py` reads. Available on request.

Absolute paths in the scripts have been replaced by the placeholders `<PROJECT_ROOT>`,
`<BTC_SPOT_1M_PARQUET>` and `<BTC_DATA_ROOT>`. Some comments are in Korean; they are the
authors' working notes and were left as written.
