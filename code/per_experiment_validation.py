"""
per_experiment_validation.py — Stat 1 of supplementary table
Aggregate b5_thm35_all.csv 387 configs into per-experiment breakdown.

Pure post-processing. No new measurement / simulation / inference.

Output: NeuMoR/output/unified/per_experiment_validation.csv (9 rows × 7 cols)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
SRC_CSV = ROOT / "NeuMoR/output/unified/b5_thm35_all.csv"
OUT_CSV = ROOT / "NeuMoR/output/unified/per_experiment_validation.csv"

# exp_label → list of block values
EXP_LABEL_MAP = {
    "Exp 01 (Heston in-sample)":              ["Exp01"],
    "Exp 02 (Heston OOS cross-validation)":   ["Exp02"],
    "Exp 03 (Heston cross-protocol)":         ["Exp03"],
    "Exp 05 (Heston multi-axis perturbations)": ["Exp05_axis", "Exp05_diag"],
    "Exp 06 (Heston edge regimes)":           ["Exp06"],
    "Exp 09 (Heston BTC-calibrated clipped)": ["Exp09"],
    "Kou Stage A":                             ["KouAB_stageA"],
    "Kou Stage B":                             ["KouAB_stageB"],
    "BTC Kou in-box":                          ["KouBTC"],
}

ACCEPT_LO, ACCEPT_HI = 0.7, 1.3


def main():
    df = pd.read_csv(SRC_CSV)
    n_total = len(df)

    rows = []
    n_used = 0
    for label, blocks in EXP_LABEL_MAP.items():
        sub = df[df.block.isin(blocks)]
        n = len(sub)
        if n == 0:
            rows.append(dict(exp_label=label, n_configs=0,
                             ratio_min=float("nan"), ratio_max=float("nan"),
                             ratio_median=float("nan"),
                             n_pass=0, pass_pct=float("nan")))
            continue
        n_pass = int(((sub.ratio >= ACCEPT_LO) & (sub.ratio <= ACCEPT_HI)).sum())
        rows.append(dict(
            exp_label=label,
            n_configs=n,
            ratio_min=round(float(sub.ratio.min()), 4),
            ratio_max=round(float(sub.ratio.max()), 4),
            ratio_median=round(float(sub.ratio.median()), 4),
            n_pass=n_pass,
            pass_pct=round(n_pass / n * 100.0, 2),
        ))
        n_used += n

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    # ── stdout ───────────────────────────────────────────────────────────────
    print(f"=== Stat 1: per-experiment validation breakdown ===")
    print(f"Source: b5_thm35_all.csv (n_total = {n_total})")
    print(f"Configs covered by exp_label map: {n_used}/{n_total}")
    print(f"Acceptance band: ratio ∈ [{ACCEPT_LO}, {ACCEPT_HI}]\n")
    print(out.to_string(index=False))

    # Aggregate footer
    n_pass_total = int(((df.ratio >= ACCEPT_LO) & (df.ratio <= ACCEPT_HI)).sum())
    print(f"\nTotal: n={n_total}, ratio range [{df.ratio.min():.4f}, {df.ratio.max():.4f}], "
          f"median {df.ratio.median():.4f}, pass {n_pass_total}/{n_total} "
          f"({n_pass_total / n_total * 100:.2f}%)")
    print(f"\nOutput: {OUT_CSV}")



if __name__ == "__main__":
    main()
