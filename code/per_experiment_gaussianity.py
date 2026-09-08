"""
per_experiment_gaussianity.py — Stat 2 step 2
Load _residuals_387.npz, run KS / Anderson-Darling / Shapiro-Wilk
per (block, config_id, payoff), aggregate per 9 exp_label.

Outputs:
    per_config_gaussianity.csv      (387 rows × 9 cols)
    per_experiment_gaussianity.csv  (9 rows × 11 cols)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kstest, anderson, shapiro

ROOT      = Path(__file__).resolve().parents[2]
RES_NPZ   = ROOT / "NeuMoR/output/day5/_residuals_387.npz"
OUT_DIR   = ROOT / "NeuMoR/output/unified"
PCFG_CSV  = OUT_DIR / "per_config_gaussianity.csv"
PEXP_CSV  = OUT_DIR / "per_experiment_gaussianity.csv"

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


def ad_p_approx(stat: float, n: int) -> float:
    """Stephens (1986) AD p-value approximation, normal-distribution case.
    Uses modified statistic A*^2 = A^2 * (1 + 0.75/n + 2.25/n²)."""
    if n < 8:
        # piecewise approximation breaks down for very small n; return raw scaling
        n = max(n, 3)
    A2 = stat * (1.0 + 0.75 / n + 2.25 / (n * n))
    if A2 < 0.200:
        p = 1.0 - np.exp(-13.436 + 101.14 * A2 - 223.73 * A2 * A2)
    elif A2 < 0.340:
        p = 1.0 - np.exp(-8.318 + 42.796 * A2 - 59.938 * A2 * A2)
    elif A2 < 0.600:
        p = np.exp(0.9177 - 4.279 * A2 - 1.38 * A2 * A2)
    elif A2 < 13.0:
        p = np.exp(1.2937 - 5.709 * A2 + 0.0186 * A2 * A2)
    else:
        p = 0.0
    return float(np.clip(p, 0.0, 1.0))


def main():
    print(f"=== per-experiment Gaussianity tests ===")
    print(f"Loading residuals: {RES_NPZ.relative_to(ROOT)}")
    data = np.load(RES_NPZ)
    keys = list(data.keys())
    print(f"  {len(keys)} entries (expected 387)")

    rows = []
    for key in keys:
        arr = np.asarray(data[key], dtype=np.float64)
        n   = len(arr)
        block, config_id, payoff = key.split("__")
        mean = float(arr.mean())
        std  = float(arr.std(ddof=1)) if n > 1 else 0.0
        if std < 1e-30 or n < 3:
            rows.append(dict(block=block, config_id=config_id, payoff=payoff,
                             n_seeds=n,
                             ks_stat=np.nan, ks_p=np.nan,
                             ad_stat=np.nan, ad_p_approx=np.nan,
                             sw_stat=np.nan, sw_p=np.nan))
            continue
        z = (arr - mean) / std
        try:
            ks_stat, ks_p = kstest(z, "norm")
            ks_stat, ks_p = float(ks_stat), float(ks_p)
        except Exception:
            ks_stat, ks_p = np.nan, np.nan
        try:
            adr     = anderson(arr, dist="norm")
            ad_stat = float(adr.statistic)
            ad_p    = ad_p_approx(ad_stat, n)
        except Exception:
            ad_stat, ad_p = np.nan, np.nan
        try:
            sw_stat, sw_p = shapiro(arr)
            sw_stat, sw_p = float(sw_stat), float(sw_p)
        except Exception:
            sw_stat, sw_p = np.nan, np.nan
        rows.append(dict(block=block, config_id=config_id, payoff=payoff,
                         n_seeds=n,
                         ks_stat=ks_stat, ks_p=ks_p,
                         ad_stat=ad_stat, ad_p_approx=ad_p,
                         sw_stat=sw_stat, sw_p=sw_p))

    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PCFG_CSV, index=False)
    print(f"\nper_config_gaussianity.csv: {len(df)} rows × 9 cols → {PCFG_CSV.relative_to(ROOT)}")

    # Aggregate
    agg_rows = []
    n_used = 0
    for label, blocks in EXP_LABEL_MAP.items():
        sub = df[df.block.isin(blocks)]
        n   = len(sub)
        n_used += n
        if n == 0:
            agg_rows.append(dict(
                exp_label=label, n_configs=0,
                ks_p_median=np.nan, ks_p_min=np.nan, ks_reject_pct=np.nan,
                ad_p_median=np.nan, ad_p_min=np.nan, ad_reject_pct=np.nan,
                sw_p_median=np.nan, sw_p_min=np.nan, sw_reject_pct=np.nan,
            ))
            continue
        agg_rows.append(dict(
            exp_label=label,
            n_configs=n,
            ks_p_median=round(float(sub.ks_p.median()), 4),
            ks_p_min=round(float(sub.ks_p.min()), 4),
            ks_reject_pct=round(float((sub.ks_p < 0.05).mean() * 100), 2),
            ad_p_median=round(float(sub.ad_p_approx.median()), 4),
            ad_p_min=round(float(sub.ad_p_approx.min()), 4),
            ad_reject_pct=round(float((sub.ad_p_approx < 0.05).mean() * 100), 2),
            sw_p_median=round(float(sub.sw_p.median()), 4),
            sw_p_min=round(float(sub.sw_p.min()), 4),
            sw_reject_pct=round(float((sub.sw_p < 0.05).mean() * 100), 2),
        ))
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(PEXP_CSV, index=False)
    print(f"per_experiment_gaussianity.csv: {len(agg)} rows × 11 cols → {PEXP_CSV.relative_to(ROOT)}")
    print(f"\nCoverage: {n_used}/{len(df)} configs mapped to exp_label\n")

    print("=== per_config_gaussianity.csv (head) ===")
    print(df.head(6).to_string(index=False))
    print()
    print("=== per_experiment_gaussianity.csv ===")
    print(agg.to_string(index=False))

    # Total reject summary
    print()
    print(f"Total reject @ 5% (across all 387):")
    print(f"  KS : {int((df.ks_p < 0.05).sum())}/{(df.ks_p.notna()).sum()} configs")
    print(f"  AD : {int((df.ad_p_approx < 0.05).sum())}/{(df.ad_p_approx.notna()).sum()} configs")
    print(f"  SW : {int((df.sw_p < 0.05).sum())}/{(df.sw_p.notna()).sum()} configs")



if __name__ == "__main__":
    main()
