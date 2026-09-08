# Replication data and code

**The folding bias of training noise in paired surrogate comparisons**  
Wooyeol Jeong · Monghwan Seo · Sungchul Lee — Department of Mathematics, Yonsei University

This archive holds the processed data behind every table in the paper, together with the
analysis scripts that produced it. It is released for **inspection and verification**, not as a
turnkey pipeline — see [Reproducing results](#reproducing-results) for exactly what runs here
and what does not.

Licensed under the MIT License (see `LICENSE`). Cite via `CITATION.cff`.

## Layout

The folders follow the structure of the manuscript.

```
code/
  core/         shared model, pricing and projection utilities
  suite/        the 387-configuration validation suite (Section 5.2)
  tables/       one builder per table of the paper and of Online Resource 1
  boundary/     boundary experiments (Section 5.4, Online Resource 1 Section S4)
  robustness/   architecture, width, Burgers, dropout and MLP panels (Section S5)
  btc/          Bitcoin-informed calibration and stress diagnostic (Section S2)
data/
  suite/              ratios, decompositions and kernel structure of the 387 configurations
  cross_protocol/     protocol-pair correlations behind Table 2
  crossfit/           cross-fitted re-scoring, including the archived per-seed discrepancies
  low_signal_panel/   targeted low-signal panel (+ cache/ density fields it is rebuilt from)
  non_gaussian/       Wasserstein bound and Edgeworth control suite
  efficiency/         plug-in against the nonparametric magnitude average
  vanilla_proxy/      vanilla-basis projection of the optimal direction
  btc/                window calibrations, event distances and stress ratios
  boundary/           per-cell results of the boundary experiments
  robustness/         per-cell results of the replication panels
meta/           seed_lists.csv · v2_splits.npz · suite_manifest.csv
```

108 files, 5.9 MB: 50 CSV, 43 Python scripts, and the `meta/` index files.

> **`data/cross_protocol/`.** `correlation_by_protocol.csv` was computed in an earlier
> Heston-only experiment series with its own conventions; it is used for Table 2 only.

## Table → file

All 27 tables in the paper (10 in the main text, 17 in Online Resource 1). 23 are backed by a
released file; the remaining 4 are described below the table.

| Table | Location | Released file |
|---|---|---|
| `tab:cross-proto` | main text | `data/cross_protocol/correlation_by_protocol.csv` |
| `tab:js-decomp` | main text | `data/suite/plugin_reconstruction.csv` |
| `tab:gstar-deviation` | main text | `data/suite/gstar_deviation.csv + data/suite/gstar_deviation_decomposed.csv` |
| `tab:lowsnr-summary` | main text | `data/low_signal_panel/panel_cells.csv + data/low_signal_panel/panel_summary.csv` |
| `tab:efficiency` | main text | `data/efficiency/efficiency_table.csv` |
| `tab:crossfit` | main text | `data/crossfit/crossfit_cells.csv + data/crossfit/crossfit_summary.csv` |
| `tab:nongauss` | main text | `data/non_gaussian/cells.csv + data/non_gaussian/gates.json` |
| `tab:multi-event-distance` | Online Resource 1 | `data/btc/event_distances.csv` |
| `tab:multi-event-summary` | Online Resource 1 | `data/btc/stress_ratios.csv` |
| `tab:per-exp-validation` | Online Resource 1 | `data/suite/ratios_by_experiment.csv` |
| `tab:per-exp-gaussianity` | Online Resource 1 | `data/suite/gaussianity_cells.csv + data/suite/gaussianity_by_experiment.csv` |
| `tab:kernel-structure` | Online Resource 1 | `data/suite/kernel_structure.csv` |
| `tab:sup-boundary-stress` | Online Resource 1 | `data/boundary/ensemble_size.csv` |
| `tab:sup-estimator-classes` | Online Resource 1 | `data/boundary/estimator_classes.csv` |
| `tab:sup-estimator-cells` | Online Resource 1 | `data/boundary/estimator_classes.csv` |
| `tab:sup-mix-cells` | Online Resource 1 | `data/boundary/cross_class_pools.csv` |
| `tab:sup-mm1` | Online Resource 1 | `data/boundary/mm1_queue.csv + data/boundary/mm1_run_length.csv` |
| `tab:sup-fno` | Online Resource 1 | `data/robustness/fno_heston.csv + data/robustness/fno_kou.csv` |
| `tab:sup-width` | Online Resource 1 | `data/robustness/width_summary.csv + data/robustness/width_cells.csv` |
| `tab:sup-burgers` | Online Resource 1 | `data/robustness/burgers_cells.csv + data/boundary/deployment_flag.csv` |
| `tab:sup-dropout` | Online Resource 1 | `data/robustness/dropout_cells.csv` |
| `tab:sup-protocol-pairs` | Online Resource 1 | `data/robustness/protocol_pairs.csv` |
| `tab:sup-mlp` | Online Resource 1 | `data/robustness/mlp_summary.csv + data/robustness/mlp_cells.csv` |

Scripts keep the authors' pipeline paths; the corresponding released file for each table is
listed above.

**Not backed by a single released file (4):**

- `tab:building-blocks` — hand-authored — no measured numbers.
- `tab:provenance` — hand-authored — no measured numbers.
- `tab:boundary-summary` — narrative synthesis — each row summarises a different experiment.
- `tab:param-pairs` — hand-authored — no measured numbers.

`tab:boundary-summary` is a synthesis table; its rows draw on
`data/boundary/{ensemble_size,signal_strength,distance_ladder,extrapolation,tail_noise,
estimator_classes,cross_class_pools,mm1_queue,mm1_run_length}.csv` and
`data/robustness/{fno_heston,fno_kou,width_summary,burgers_cells,dropout_cells}.csv`.

### Caveats

- **Vanilla-proxy sensitivity sweep** — this table was removed from the submitted manuscript.
  The cells (`data/vanilla_proxy/sensitivity_cells.csv`), gate checks (`gates.json`) and
  `code/tables/build_vanilla_proxy.py` are kept for completeness; the script requires the eight
  pair npz arrays, which are not redistributed and are available on request.
- **Estimator classes** — the Lilliefors column printed in the manuscript is not in
  `data/boundary/estimator_classes.csv`, which carries KS and AD rejection counts. That column is
  not reproducible from the released files.

## Reproducing results

With `data/` and `code/` alone, a downloader can check every published table cell against the
released files — the numbers in the paper are read straight out of the shipped
`data/crossfit/*.csv`, `data/low_signal_panel/*.csv`/`verdict.json`,
`data/non_gaussian/*.csv`/`gates.json`, `data/efficiency/efficiency_table.csv`, and
`data/vanilla_proxy/sensitivity_cells.csv`/`gates.json` (retained; not printed in the
manuscript), which are included in full. Two of the seven builders also re-run end to end from
this tree: `code/tables/build_efficiency.py` has no file inputs at all (closed-form mpmath
evaluation at dps=40 plus a fixed-seed Monte Carlo, seed 20260902), and
`code/tables/build_vanilla_proxy.py` is deterministic but reads the eight pair npz arrays listed
below. The other five import `code/suite/score_suite.py`, which loads `torch` and the `TNO`
package at module scope, so they need the full materials; on the cached path none of them reads a
checkpoint. The trained checkpoints, the `TNO` package (`TNO/common/tno_model.py`) and the `TPDF`
package (`TPDF/core/semi_closed_joint.py`, used by `code/core/heston_reference.py` for the
Lewis-formula ground truth) are not redistributed here and are available from the corresponding
author on request.

### Run order

| # | Script | Runs from this archive? |
|---|---|---|
| 1 | `code/tables/build_efficiency.py` | **yes** |
| 2 | `code/tables/build_vanilla_proxy.py` | **yes** |
| 3 | `code/tables/build_crossfit.py` | no |
| 4 | `code/tables/build_low_signal_panel.py` | no |
| 5 | `code/tables/build_low_signal_spreads.py` | no |
| 6 | `code/tables/build_non_gaussian.py` | no |
| 7 | `code/tables/build_non_gaussian_tree.py` | no |

Run everything from the repository root. If you have the full materials, the dependent builders
must run in the order shown: `build_low_signal_panel.py` → `build_low_signal_spreads.py` →
`build_non_gaussian.py` → `build_non_gaussian_tree.py`. **`build_low_signal_spreads.py` cannot be
run on its own** — it asserts that `panel_cells.csv` still has its 72 pre-append rows, and the
shipped file has 87, so `build_low_signal_panel.py` must rewrite it first. `build_crossfit.py`,
`build_efficiency.py` and `build_vanilla_proxy.py` are independent of that chain. The builders
append to `build.log`, so that file is never byte-comparable across runs.

## `meta/`

| File | Contents |
|---|---|
| `seed_lists.csv` | 25 checkpoint ensembles covering all 3,399 trained models: directory, role, filename pattern, `zero_pad` width, seed range. Seed numbering is contiguous in every ensemble but one. **`zero_pad` matters** — seed 60 is `seed_060.pt` in one ensemble and `seed_0060.pt` in another. |
| `v2_splits.npz` | The resampling permutations of the cross-fitted re-scoring, materialised so they no longer depend on a NumPy RNG stream. Replaying these reproduces every published median exactly. The file holds 389 keys: 388 split arrays (301 of shape (20, 200) and 87 of shape (20, 30)) plus the `b9_keys` index array naming the configuration each split belongs to. |
| `suite_manifest.csv` | One row per (configuration, payoff) across the 387-configuration suite: identifiers, calibration/evaluation split sizes, and the model-parameter pair λ₁/λ₂ as JSON. |

λ values are derived for **333 of 387 rows** — every Heston block. The remaining 54 (the two Kou
stages and the BTC-informed blocks) carry `lambda_status = not_extracted`; each row's
`lambda_source` column names where those parameters actually live.

## What is not here

- **Trained checkpoints** (3,399 files, 5.69 GB). Not redistributed; available from the
  corresponding author on reasonable request. `meta/seed_lists.csv` documents what exists.
- **The `TNO` and `TPDF` packages**, which `code/suite/score_suite.py` and
  `code/core/heston_reference.py` import. Available on request.
- **Bitcoin/USDT 1-minute spot prices**, obtained from the Binance public API and not
  redistributed here. Scripts that consume them carry a `<BTC_SPOT_1M_PARQUET>` placeholder.
- **The eight per-pair `.npz` arrays** (`{heston,kou}_pair{A..D}`, 24 MB) holding the noise
  kernels, d-bar and g* that `code/tables/build_vanilla_proxy.py` reads. Available on request.

Absolute paths in the scripts have been replaced by the placeholders `<PROJECT_ROOT>`,
`<BTC_SPOT_1M_PARQUET>` and `<BTC_DATA_ROOT>`.
