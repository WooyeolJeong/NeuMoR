from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erfc

ROOT = Path(__file__).resolve().parents[2]
SRC  = ROOT / "NeuMoR/output/unified/b5_thm35_all.csv"
OUT  = ROOT / "NeuMoR/output/unified/bias_decomp_387_configs.csv"

def jensen_J(sigma: np.ndarray, mu: np.ndarray) -> np.ndarray:
    sigma = np.asarray(sigma, dtype=np.float64)
    mu    = np.asarray(mu,    dtype=np.float64)
    out   = np.zeros_like(sigma)
    valid = sigma > 0
    s = np.where(valid, mu / np.where(valid, sigma, 1.0), 0.0)
    abs_s = np.abs(s)
    term = np.sqrt(2.0/np.pi) * np.exp(-0.5*s*s) - abs_s * erfc(abs_s/np.sqrt(2.0))
    out[valid] = sigma[valid] * term[valid]
    return out

def map_model(exp: str) -> str:
    if exp.startswith("kou"):
        return "kou"
    return "heston"

def main() -> None:
    df = pd.read_csv(SRC)
    n_total = len(df)
    assert n_total == 387, f"expected 387 configs, got {n_total}"

    sigma = df["sigma_delta"].to_numpy(dtype=np.float64)
    mu    = df["mu_delta"].to_numpy(dtype=np.float64)
    th    = df["theory_mr"].to_numpy(dtype=np.float64)
    emp   = df["empirical_mr"].to_numpy(dtype=np.float64)

    s        = np.where(sigma > 0, mu / np.where(sigma > 0, sigma, 1.0), np.nan)
    J        = jensen_J(sigma, mu)
    abs_s    = np.abs(s)
    J_frac   = np.where(th != 0, J / th, np.nan)

    J_via_identity = th - np.abs(mu)
    sanity_abs_err = np.abs(J - J_via_identity)

    sanity_rel_err = np.where(
        np.abs(J_via_identity) > 1e-15,
        sanity_abs_err / np.maximum(np.abs(J_via_identity), 1e-30),
        sanity_abs_err,
    )

    dev_pct = np.where(th != 0, np.abs(emp / th - 1.0) * 100.0, np.nan)

    out = pd.DataFrame({
        "block":         df["block"].values,
        "config_id":     df["config_id"].values,
        "payoff":        df["payoff"].values,
        "exp":           df["exp"].values,
        "pair":          df["pair"].values,
        "model":         df["exp"].map(map_model).values,
        "proto_pair":    df["proto_pair"].values,
        "scenario":      df["scenario"].values,
        "sigma_Delta":   sigma,
        "mu_Delta":      mu,
        "Delta":         np.nan,
        "b_Delta":       np.nan,
        "s":             s,
        "J":             J,
        "S":             np.nan,
        "J_plus_S":      np.nan,
        "J_over_JplusS": np.nan,
        "J_fraction":    J_frac,
        "theory_mr":     th,
        "empirical_mr":  emp,
        "deviation_pct": dev_pct,
        "sanity_abs_err": sanity_abs_err,
        "sanity_rel_err": sanity_rel_err,
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("=" * 72)
    print(f"saved: {OUT}")
    print(f"rows : {len(out)}   cols: {out.shape[1]}")
    print("=" * 72)
    print(f"Total configs: {n_total}\n")

    print("=== Identity sanity:  J  ?=  theory_mr − |mu_delta|  ===")
    print(f"  abs err : median {np.median(sanity_abs_err):.3e}  "
          f"max {sanity_abs_err.max():.3e}")
    print(f"  rel err : median {np.median(sanity_rel_err):.3e}  "
          f"max {sanity_rel_err.max():.3e}")
    print(f"  (folded-normal identity; rel err ≲ 1e-10 = machine precision OK)")

    finite_s = abs_s[np.isfinite(abs_s)]
    print(f"\n=== |s| distribution (n={len(finite_s)}) ===")
    print(f"  range  : [{finite_s.min():.3f}, {finite_s.max():.3f}]   "
          f"median {np.median(finite_s):.3f}")
    n_low   = int(np.sum(finite_s < 1.0))
    n_mid   = int(np.sum((finite_s >= 1.0) & (finite_s < 3.0)))
    n_high  = int(np.sum(finite_s >= 3.0))
    print(f"  |s| <  1   (J-dominant)  : {n_low:3d} configs")
    print(f"  1 ≤ |s| <  3 (transition): {n_mid:3d} configs")
    print(f"  |s| ≥  3   (S-dominant)  : {n_high:3d} configs")

    finite_jf = J_frac[np.isfinite(J_frac)]
    print(f"\n=== J_fraction = J / theory_mr  (n={len(finite_jf)}) ===")
    print(f"  range  : [{finite_jf.min():.3e}, {finite_jf.max():.3e}]   "
          f"median {np.median(finite_jf):.3e}")
    print(f"  J_fraction > 0.01  (J-active candidate): "
          f"{int(np.sum(finite_jf > 0.01)):3d} configs")
    print(f"  J_fraction > 0.1                       : "
          f"{int(np.sum(finite_jf > 0.1)):3d} configs")
    print(f"  J_fraction > 0.5                       : "
          f"{int(np.sum(finite_jf > 0.5)):3d} configs")

    print(f"\n=== Per-payoff median |s| ===")
    for p in ["ATM", "OTM_K110", "OTM_K125"]:
        sub = out[out["payoff"] == p]
        med = float(sub["s"].abs().median())
        max_jf = float(sub["J_fraction"].max())
        print(f"  {p:9s} (n={len(sub):3d})  median |s| = {med:6.3f}   "
              f"max J_fraction = {max_jf:.3e}")

    print(f"\n=== Per-block median |s| and J_fraction max ===")
    for blk in sorted(out["block"].unique()):
        sub = out[out["block"] == blk]
        med_s   = float(sub["s"].abs().median())
        max_jf  = float(sub["J_fraction"].max())
        n_act01 = int((sub["J_fraction"] > 0.01).sum())
        print(f"  {blk:14s} (n={len(sub):3d})  median |s| = {med_s:6.3f}   "
              f"max J_fraction = {max_jf:.3e}   "
              f"J-active>{0.01:.2f}: {n_act01}")

    print(f"\n=== Top-10 configs by J_fraction (J-active candidates) ===")
    top10 = out.nlargest(10, "J_fraction")[
        ["block","config_id","payoff","s","J","theory_mr","J_fraction"]
    ]
    for _, r in top10.iterrows():
        print(f"  {r['block']:14s} {r['config_id']:22s} {r['payoff']:9s}   "
              f"|s|={abs(r['s']):6.3f}   J_frac={r['J_fraction']:.3e}")

    finite_dev = dev_pct[np.isfinite(dev_pct)]
    print(f"\n=== deviation_pct  =  |empirical_mr/theory_mr − 1| × 100  ===")
    print(f"  range : [{finite_dev.min():.3f}, {finite_dev.max():.3f}]   "
          f"median {np.median(finite_dev):.3f}%")
    print(f"  ≤  1%: {int(np.sum(finite_dev <= 1)):3d} configs")
    print(f"  ≤  4%: {int(np.sum(finite_dev <= 4)):3d} configs")
    print(f"  ≤ 10%: {int(np.sum(finite_dev <= 10)):3d} configs")
    print(f"  > 10%: {int(np.sum(finite_dev >  10)):3d} configs")

if __name__ == "__main__":
    main()
