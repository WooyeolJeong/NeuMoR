"""
issue4_k99_quantile.py — Issue 4 (a): k_99 quantile breakdown
==============================================================
multidim_hedging_per_sample.csv 1152 rows →
per-(model, layer, payoff_category) k_for_VR99 quantile 분포.

Outputs: NeuMoR/output/issue4_quantile/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[2]
UNIFIED = ROOT / "NeuMoR/output/unified"
OUTDIR  = ROOT / "NeuMoR/output/issue4_quantile"
OUTDIR.mkdir(parents=True, exist_ok=True)

K_GRID = [1, 2, 3, 5, 10, 20, 50, 256, 300]

df = pd.read_csv(UNIFIED / "multidim_hedging_per_sample.csv")
print(f"Loaded: {df.shape}  columns: {df.columns.tolist()}")


# ── Phase 2: payoff category ──────────────────────────────────────────────────

def categorise(pid):
    if pid == "call_k=+0.00":
        return "ATM"
    elif pid.startswith("call_k=+"):
        return "OTM_call"
    elif pid.startswith("put_k=-"):
        return "OTM_put"
    elif "portfolio" in pid:
        return "portfolio"
    else:
        return "other"

df["payoff_cat"] = df["payoff_id"].map(categorise)

print("\nPayoff category counts (per unique payoff_id):")
print(df.groupby(["payoff_cat","payoff_id"]).size().to_string())


# ── Phase 1: per-sample k_99, k_95, VR_at_k10 ────────────────────────────────

print(f"\nPhase 1: per-sample k_99/k_95/VR_at_k10 (Option A — smallest k in grid)")

GROUP_COLS = ["model","pair","layer","payoff_id","payoff_cat"]

rows = []
for keys, sub in df.groupby(GROUP_COLS):
    sub_s = sub.sort_values("k")
    vr    = dict(zip(sub_s["k"], sub_s["VR_k"]))

    # k_99: smallest k where VR_k >= 0.99
    above99 = [k for k in K_GRID if vr.get(k, 0) >= 0.99]
    k99 = int(min(above99)) if above99 else np.nan
    censored99 = np.isnan(k99)

    # k_95
    above95 = [k for k in K_GRID if vr.get(k, 0) >= 0.95]
    k95 = int(min(above95)) if above95 else np.nan
    censored95 = np.isnan(k95)

    # VR at k=10
    vr_k10 = float(vr.get(10, np.nan))

    row = dict(zip(GROUP_COLS, keys))
    row.update(k_99=k99, censored99=censored99,
               k_95=k95, censored95=censored95,
               VR_at_k10=round(vr_k10, 6))
    rows.append(row)

df_samp = pd.DataFrame(rows)
df_samp.to_csv(OUTDIR / "k99_per_sample.csv", index=False)

n_groups = len(df_samp)
print(f"  Groups: {n_groups} (target 144)")
for mn in ["heston","kou"]:
    sub = df_samp[df_samp["model"]==mn]
    print(f"  {mn}: censored99={sub['censored99'].sum()}  censored95={sub['censored95'].sum()}")

# Sanity: NaN ↔ censored=True 1:1
assert (df_samp["k_99"].isna() == df_samp["censored99"]).all(), "k_99 NaN / censored mismatch"
print("  Sanity NaN↔censored: PASS")


# ── Phase 3: Table A — overall (model, layer) ─────────────────────────────────

def agg_k99(sub, col="k_99", cens="censored99"):
    vals   = sub[col].dropna()
    n_cens = int(sub[cens].sum())
    n_tot  = len(sub)
    if len(vals) == 0:
        return dict(n=n_tot, median=np.nan, q25=np.nan, q75=np.nan,
                    max=np.nan, censored_frac=1.0)
    return dict(
        n=n_tot,
        median=round(float(np.median(vals)), 1),
        q25   =round(float(np.percentile(vals, 25)), 1),
        q75   =round(float(np.percentile(vals, 75)), 1),
        max   =round(float(vals.max()), 1),
        censored_frac=round(n_cens / n_tot, 4),
    )

tA_rows = []
for (mn, ly), sub in df_samp.groupby(["model","layer"]):
    r99 = agg_k99(sub, "k_99", "censored99")
    r95 = agg_k99(sub, "k_95", "censored95")
    vr10 = sub["VR_at_k10"].dropna()
    tA_rows.append(dict(
        model=mn, layer=ly,
        k99_n=r99["n"], k99_median=r99["median"], k99_q25=r99["q25"],
        k99_q75=r99["q75"], k99_max=r99["max"], k99_censored_frac=r99["censored_frac"],
        k95_median=r95["median"], k95_q25=r95["q25"], k95_q75=r95["q75"],
        VR_k10_median=round(float(vr10.median()), 4),
        VR_k10_q25   =round(float(vr10.quantile(0.25)), 4),
        VR_k10_q75   =round(float(vr10.quantile(0.75)), 4),
    ))

df_tA = pd.DataFrame(tA_rows)
df_tA.to_csv(OUTDIR / "k99_overall.csv", index=False)

print("\n=== Table A: Overall (model, layer) ===")
print(df_tA.to_string(index=False))

# Sanity cross-check vs saturation CSV
sat = pd.read_csv(UNIFIED / "multidim_hedging_saturation.csv")
print("\n  Cross-check vs saturation CSV k_for_VR99_median:")
for mn in ["heston","kou"]:
    sat_A = sat[(sat["model"]==mn) & (sat["layer"]=="A")]["k_for_VR99_median"].values[0]
    our_A = df_tA[(df_tA["model"]==mn) & (df_tA["layer"]=="A")]["k99_median"].values[0]
    match = "PASS" if sat_A == our_A else f"MISMATCH (sat={sat_A})"
    print(f"  {mn} layer A: saturation={sat_A}  ours={our_A}  → {match}")


# ── Phase 3: Table B — per payoff category ────────────────────────────────────

tB_rows = []
for (mn, ly, cat), sub in df_samp.groupby(["model","layer","payoff_cat"]):
    r99  = agg_k99(sub, "k_99", "censored99")
    r95  = agg_k99(sub, "k_95", "censored95")
    vr10 = sub["VR_at_k10"].dropna()
    tB_rows.append(dict(
        model=mn, layer=ly, payoff_cat=cat,
        k99_n=r99["n"], k99_median=r99["median"], k99_q25=r99["q25"],
        k99_q75=r99["q75"], k99_max=r99["max"], k99_censored_frac=r99["censored_frac"],
        k95_median=r95["median"],
        VR_k10_median=round(float(vr10.median()), 4) if len(vr10) else np.nan,
    ))

df_tB = pd.DataFrame(tB_rows)
df_tB.to_csv(OUTDIR / "k99_per_payoff.csv", index=False)

print("\n=== Table B: Per Payoff Category ===")
print(df_tB.to_string(index=False))

# call vs put summary
print("\n  call vs put k_99 median (layer A):")
for mn in ["heston","kou"]:
    for cat in ["ATM","OTM_call","OTM_put"]:
        sub = df_tB[(df_tB["model"]==mn) & (df_tB["layer"]=="A") & (df_tB["payoff_cat"]==cat)]
        if len(sub):
            print(f"    {mn} {cat}: k99_median={sub['k99_median'].values[0]}")


# ── Phase 4: Figure ───────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
CAT_ORDER = ["ATM", "OTM_call", "OTM_put", "portfolio"]
colors    = {"A": "steelblue", "B1": "darkorange"}

for ax, mn in zip(axes, ["heston","kou"]):
    sub_m   = df_samp[(df_samp["model"]==mn) & (df_samp["layer"]=="A")]
    cats    = [c for c in CAT_ORDER if c in sub_m["payoff_cat"].unique()]
    box_data = [sub_m[sub_m["payoff_cat"]==c]["k_99"].dropna().values for c in cats]
    cens_frac= [sub_m[sub_m["payoff_cat"]==c]["censored99"].mean() for c in cats]

    bp = ax.boxplot(box_data, labels=cats, patch_artist=True,
                    medianprops=dict(color="black", lw=2))
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue"); patch.set_alpha(0.6)

    # annotate censored fraction
    for i, (cf, bd) in enumerate(zip(cens_frac, box_data)):
        med = float(np.median(bd)) if len(bd) else 0
        ax.annotate(f"cens={cf:.0%}", xy=(i+1, med),
                    xytext=(i+1, med*1.3 if med > 0 else 2),
                    ha="center", fontsize=7, color="gray")

    ax.set_yscale("log")
    ax.set_xlabel("Payoff category", fontsize=10)
    ax.set_ylabel("k for VR ≥ 0.99 (log scale)", fontsize=10)
    ax.set_title(mn.capitalize(), fontsize=12)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("k_99 Distribution by Payoff Category (layer A)", fontsize=12)
fig.tight_layout()
fig_path = OUTDIR / "fig_k99_quantile_breakdown.png"
fig.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure: {fig_path}")



print(f"\n=== Output files ===")
print(f"  {OUTDIR}/k99_per_sample.csv   ({len(df_samp)} rows)")
print(f"  {OUTDIR}/k99_overall.csv      ({len(df_tA)} rows)")
print(f"  {OUTDIR}/k99_per_payoff.csv   ({len(df_tB)} rows)")
print(f"  {fig_path}")
