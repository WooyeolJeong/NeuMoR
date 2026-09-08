"""
hedging_step2.py — Multi-dimensional Hedging via K_eta Eigenvectors (Step 2)
=============================================================================

Theorem 4.7 verification: VR_k vs k curve, saturation at r_eff.
144 samples (Layer A 136 + Layer B-1 8).

Outputs (NeuMoR/output/unified/):
  multidim_hedging_per_sample.csv      (1152 rows: 144 × 8 k values)
  multidim_hedging_summary.csv         (~40 rows: model × layer × k)
  multidim_hedging_saturation.csv      (~6 rows: model × layer + all)
  NeuMoR/figures/fig_multidim_VR_curve.png
  NeuMoR/figures/fig_theorem47_verification.png

Run:
  python NeuMoR/code/hedging_step2.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
UNIFIED = ROOT / "NeuMoR/output/unified"
TMP     = ROOT / "NeuMoR/output/day5/_tmp"
FIGURES = ROOT / "NeuMoR/figures"
FIGURES.mkdir(parents=True, exist_ok=True)

PNL_CSV  = UNIFIED / "pnl_attribution.csv"
PORT_CSV = ROOT / "NeuMoR/output/exp12/portfolio.csv"

LOG_STRIKES_CALL = [0.0,  0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.60, 0.70]
LOG_STRIKES_PUT  = [-0.05,-0.10,-0.15,-0.20,-0.30,-0.50,-0.60,-0.70]
K_VALUES = [1, 2, 3, 5, 10, 20, 50, None]   # None = n_y (full rank)


def parse_payoff(payoff_id, y):
    if payoff_id.startswith("call_k="):
        k = float(payoff_id.replace("call_k=", ""))
        return np.maximum(np.exp(y) - np.exp(k), 0.0)
    elif payoff_id.startswith("put_k="):
        k = float(payoff_id.replace("put_k=", ""))
        return np.maximum(np.exp(k) - np.exp(y), 0.0)
    raise ValueError(f"Unknown payoff_id: {payoff_id}")


def build_portfolio(port_df, y):
    g = np.zeros(len(y))
    for _, row in port_df.iterrows():
        K = float(row["K"]); s = float(row["size"])
        if row["type"] == "call":
            g += s * np.maximum(np.exp(y) - K, 0.0)
        elif row["type"] == "put":
            g += s * np.maximum(K - np.exp(y), 0.0)
    return g


# ── Phase 0: Sanity check ─────────────────────────────────────────────────────

print("="*65)
print("PHASE 0: Sanity check")
print("="*65)

pnl = pd.read_csv(PNL_CSV)
pnl_ab = pnl[pnl["layer"].isin(["A","B1"])].copy().reset_index(drop=True)
assert len(pnl_ab) == 144, f"Expected 144 samples, got {len(pnl_ab)}"

port_45 = pd.read_csv(PORT_CSV)
port_45 = port_45[port_45["type"] != "digital"].reset_index(drop=True)

npz_cache = {}
for model in ["heston","kou"]:
    for pair in ["A","B","C","D"]:
        key = (model, pair)
        npz_path = UNIFIED / f"{model}_pair{pair}.npz"
        assert npz_path.exists(), f"Missing: {npz_path}"
        npz = np.load(npz_path, allow_pickle=True)
        npz_cache[key] = npz

        # Eigenvector orthonormality check
        ev = npz["eigvecs_eta"]                        # (n_y, n_y), columns descending
        gram = ev.T @ ev                               # should be I
        err  = np.abs(gram - np.eye(ev.shape[0])).max()
        assert err < 1e-8, f"{key}: eigvecs not orthonormal (max err {err:.2e})"

        # Eigvals descending
        evals = npz["eigvals_eta"]
        assert np.all(np.diff(evals) <= 1e-15), f"{key}: eigvals not descending"

print(f"  All 8 npz: eigvectors orthonormal, eigvals descending")

# Quick k=1 check on one sample
npz0  = npz_cache[("heston","A")]
y0    = npz0["y_grid"]; dy0 = float(y0[1]-y0[0])
ev0   = npz0["eigvecs_eta"]   # (300, 300)
l1_0  = np.load(TMP/"heston_pairA_l1_1000.npy").astype(np.float64)
l2_0  = np.load(TMP/"heston_pairA_l2_1000.npy").astype(np.float64)
g0    = parse_payoff("call_k=+0.00", y0)
dV_g  = (l1_0-l2_0) @ g0 * dy0
beta1 = float(g0 @ ev0[:,0] * dy0)
dV_v1 = (l1_0-l2_0) @ ev0[:,0] * dy0
dV_h  = dV_g - beta1 * dV_v1
VR_k1 = 1 - np.var(dV_h) / np.var(dV_g)
print(f"  k=1 sanity (heston A, ATM): VR={VR_k1:.4f}")
print("PHASE 0 PASS\n")


# ── Phase 1: Per-sample, per-k computation ────────────────────────────────────

print("="*65)
print("PHASE 1: Per-sample VR(k) computation (144 × 8 = 1152)")
print("="*65)

rows = []

for _, row in pnl_ab.iterrows():
    model     = str(row["model"])
    pair      = str(row["pair"])
    layer     = str(row["layer"])
    payoff_id = str(row["payoff_id"])

    npz   = npz_cache[(model, pair)]
    y     = npz["y_grid"]; dy = float(y[1]-y[0])
    evals = npz["eigvals_eta"]    # (n_y,) descending
    evecs = npz["eigvecs_eta"]    # (n_y, n_y) columns descending
    n_y   = len(y)
    N     = 1000 if model == "heston" else 200
    n_cal = int(npz["metadata"][0]["n_cal"])

    l1 = np.load(TMP/f"{model}_pair{pair}_l1_{N}.npy").astype(np.float64)
    l2 = np.load(TMP/f"{model}_pair{pair}_l2_{N}.npy").astype(np.float64)
    # Use CAL seeds only (same as K_eta_cal)
    l1c = l1[:n_cal]; l2c = l2[:n_cal]

    g = build_portfolio(port_45, y) if layer=="B1" else parse_payoff(payoff_id, y)

    # Per-seed ΔV(g)
    delta_pdf = l1c - l2c                             # (n_cal, n_y)
    dV_g_seed = delta_pdf @ g * dy                    # (n_cal,)
    var_g = float(np.var(dV_g_seed))

    # Per-seed ΔV(v_j) for all j at once
    dV_v_all = delta_pdf @ evecs * dy                 # (n_cal, n_y)

    # beta_j = g · v_j (discrete, NO dy)
    # beta_j^opt = Cov(ΔV(g),ΔV(v_j))/Var(ΔV(v_j)) = (g^T K v_j)/(λ_j) = g·v_j
    betas_all = g @ evecs                             # (n_y,)

    # ||g||²_{K_eta} = g^T K_eta g = sum_j lambda_j * (g·v_j)²
    g_Keta_norm2 = float(np.sum(evals * betas_all**2))

    for k_spec in K_VALUES:
        k = n_y if k_spec is None else k_spec
        k_label = n_y if k_spec is None else k_spec

        # Hedged residual
        hedge = dV_v_all[:, :k] @ betas_all[:k]      # (n_cal,)
        dV_hedged = dV_g_seed - hedge
        var_hedged = float(np.var(dV_hedged))
        VR_k = 1.0 - var_hedged / var_g if var_g > 1e-30 else float("nan")

        # Predicted VR (Theorem 4.7): sum_{j=1}^k lambda_j beta_j^2 / ||g||²_{K_eta}
        pred_VR = float(np.sum(evals[:k] * betas_all[:k]**2) / g_Keta_norm2) if g_Keta_norm2 > 1e-30 else float("nan")
        diff_pct = abs(VR_k - pred_VR) / max(abs(pred_VR), 1e-10) * 100 if (not np.isnan(VR_k) and not np.isnan(pred_VR)) else float("nan")

        rows.append(dict(
            model=model, pair=pair, layer=layer,
            payoff_id=payoff_id, k=k_label,
            VR_k=round(VR_k, 6) if not np.isnan(VR_k) else float("nan"),
            predicted_VR=round(pred_VR, 6) if not np.isnan(pred_VR) else float("nan"),
            diff_pct=round(diff_pct, 3) if not np.isnan(diff_pct) else float("nan"),
        ))

df_res = pd.DataFrame(rows)
print(f"  Computed {len(df_res)} (sample, k) rows")
print(f"  k=1 VR  range: [{df_res[df_res.k==1].VR_k.min():.4f}, {df_res[df_res.k==1].VR_k.max():.4f}]  med={df_res[df_res.k==1].VR_k.median():.4f}")
k_ny = df_res.k.max()
print(f"  k=n_y VR range: [{df_res[df_res.k==k_ny].VR_k.min():.6f}, {df_res[df_res.k==k_ny].VR_k.max():.6f}]")
print(f"  Theorem 4.7 diff_pct max: {df_res.diff_pct.max():.3f}%")
print()


# ── Phase 2: Aggregation ──────────────────────────────────────────────────────

print("="*65)
print("PHASE 2: Aggregation")
print("="*65)

def agg_vr(sub):
    vr = sub.VR_k.dropna().values
    pd_ = sub.predicted_VR.dropna().values
    dp  = sub.diff_pct.dropna().values
    if len(vr) == 0:
        return dict(n=0, VR_median=float("nan"), VR_q25=float("nan"), VR_q75=float("nan"),
                    n_VR_above_30=0, n_VR_above_50=0, n_VR_above_70=0, n_VR_above_90=0,
                    n_VR_above_99=0, predicted_VR_median=float("nan"), diff_pct_median=float("nan"))
    def pct(arr, thr): return int(np.sum(arr >= thr))
    return dict(
        n=len(sub),
        VR_median=round(float(np.median(vr)),4),
        VR_q25=round(float(np.percentile(vr,25)),4),
        VR_q75=round(float(np.percentile(vr,75)),4),
        n_VR_above_30=pct(vr,0.30), n_VR_above_50=pct(vr,0.50),
        n_VR_above_70=pct(vr,0.70), n_VR_above_90=pct(vr,0.90),
        n_VR_above_99=pct(vr,0.99),
        predicted_VR_median=round(float(np.median(pd_)),4) if len(pd_) else float("nan"),
        diff_pct_median=round(float(np.median(dp)),4) if len(dp) else float("nan"),
    )

sum_rows = []
K_UNIQUE = sorted(df_res.k.unique())
for model in ["heston","kou"]:
    for layer in ["A","B1"]:
        for k in K_UNIQUE:
            sub = df_res[(df_res.model==model)&(df_res.layer==layer)&(df_res.k==k)]
            if len(sub)==0: continue
            r = agg_vr(sub); r.update(model=model,layer=layer,k=k)
            sum_rows.append(r)
    for k in K_UNIQUE:
        sub = df_res[(df_res.model==model)&(df_res.k==k)]
        r = agg_vr(sub); r.update(model="all_"+model,layer="all",k=k)
        sum_rows.append(r)
for k in K_UNIQUE:
    sub = df_res[df_res.k==k]
    r = agg_vr(sub); r.update(model="all",layer="all",k=k)
    sum_rows.append(r)

df_sum = pd.DataFrame(sum_rows)
# Print summary
print(f"  {'model':10} {'layer':5} {'k':5} {'VR_med':8} [q25,q75]         ≥50% ≥70% ≥90% ≥99%  thm_diff%")
for model in ["all_heston","all_kou","all"]:
    for k in K_UNIQUE:
        r = df_sum[(df_sum.model==model)&(df_sum.k==k)]
        if len(r)==0: continue
        r = r.iloc[0]
        print(f"  {r.model:10} {r.layer:5} {r.k:5d} {r.VR_median:8.4f} [{r.VR_q25:.4f},{r.VR_q75:.4f}]  "
              f"{r.n_VR_above_50:4d} {r.n_VR_above_70:4d} {r.n_VR_above_90:4d} {r.n_VR_above_99:4d}  "
              f"{r.diff_pct_median:.3f}%")
print()


# ── Phase 3: Saturation analysis ──────────────────────────────────────────────

print("="*65)
print("PHASE 3: Saturation analysis")
print("="*65)

sat_rows = []
K_ARR = np.array(K_UNIQUE)

def find_k_for_vr(sample_rows, thr):
    """For each sample, find minimum k where VR >= thr."""
    k_needed = []
    for sid in sample_rows.payoff_id.unique() if 'payoff_id' in sample_rows.columns else [None]:
        if sid is not None:
            sub = sample_rows[sample_rows.payoff_id==sid]
        else:
            sub = sample_rows
        for _, sr in sub.iterrows():
            pass  # not ideal — do it per-sample differently
    return k_needed

for model in ["heston","kou","all"]:
    for layer in ["A","B1","all"] if model!="all" else ["all"]:
        if model == "all":
            sub = df_res
        elif layer == "all":
            sub = df_res[df_res.model==model]
        else:
            sub = df_res[(df_res.model==model)&(df_res.layer==layer)]
        if len(sub)==0: continue

        # For each sample, find k where VR crosses 0.95 and 0.99
        k95_list, k99_list = [], []
        # group by (pair, payoff_id)
        for (pair, pid), grp in sub.groupby(["pair","payoff_id"]):
            grp_s = grp.sort_values("k")
            for thr, lst in [(0.95, k95_list), (0.99, k99_list)]:
                found = grp_s[grp_s.VR_k >= thr]
                if len(found):
                    lst.append(int(found.iloc[0]["k"]))
                else:
                    lst.append(int(grp_s["k"].max()))

        # r_eff avg for this group
        if model == "all":
            reff_vals = []
            for m2 in ["heston","kou"]:
                for p2 in ["A","B","C","D"]:
                    npz2 = npz_cache[(m2,p2)]
                    reff_vals.append(float(npz2["reff_eta"]))
        elif layer == "all":
            reff_vals = [float(npz_cache[(model,p)]["reff_eta"]) for p in ["A","B","C","D"]]
        else:
            reff_vals = [float(npz_cache[(model,p)]["reff_eta"]) for p in ["A","B","C","D"]]
        r_eff_avg = float(np.mean(reff_vals))

        # VR at k = round(r_eff) for each sample
        k_reff = int(round(r_eff_avg))
        sub_reff = sub[sub.k==k_reff]
        vr_at_reff = float(sub_reff.VR_k.median()) if len(sub_reff) else float("nan")

        sat_rows.append(dict(
            model=model, layer=layer,
            n=len(sub)//len(K_UNIQUE),
            k_for_VR95_median=int(np.median(k95_list)) if k95_list else -1,
            k_for_VR99_median=int(np.median(k99_list)) if k99_list else -1,
            r_eff_avg=round(r_eff_avg,3),
            k_reff=k_reff,
            VR_at_k_eq_reff_median=round(vr_at_reff,4),
        ))
        print(f"  {model:10} {layer:5}: k@95%={int(np.median(k95_list)) if k95_list else -1:4d}  "
              f"k@99%={int(np.median(k99_list)) if k99_list else -1:4d}  "
              f"r_eff={r_eff_avg:.2f}  VR@r_eff={vr_at_reff:.4f}")

df_sat = pd.DataFrame(sat_rows)
print()


# ── Phase 4: Verdict ──────────────────────────────────────────────────────────

print("="*65)
print("PHASE 4: Verdict")
print("="*65)

VR_k2  = float(df_sum[(df_sum.model=="all") & (df_sum.k==2)]["VR_median"].iloc[0])
VR_k5  = float(df_sum[(df_sum.model=="all") & (df_sum.k==5)]["VR_median"].iloc[0])
k99_med = int(df_sat[df_sat.model=="all"]["k_for_VR99_median"].iloc[0])
r_eff_a = float(df_sat[df_sat.model=="all"]["r_eff_avg"].iloc[0])

if VR_k2 >= 0.80 and VR_k5 >= 0.95 and k99_med <= r_eff_a + 1:
    verdict = "STRONG"
elif VR_k2 >= 0.60 and VR_k5 >= 0.80:
    verdict = "MODERATE"
else:
    verdict = "WEAK"

print(f"  VR_k=2 median: {VR_k2:.4f}  (threshold: ≥0.80 → strong, ≥0.60 → moderate)")
print(f"  VR_k=5 median: {VR_k5:.4f}  (threshold: ≥0.95 → strong, ≥0.80 → moderate)")
print(f"  k for 99% VR (median): {k99_med}  r_eff avg: {r_eff_a:.2f}")
print(f"  Verdict: {verdict}")
print()


# ── Phase 5: Theorem 4.7 verification ────────────────────────────────────────

print("="*65)
print("PHASE 5: Theorem 4.7 verification")
print("="*65)

diff_all = df_res.diff_pct.dropna()
print(f"  median diff_pct (predicted vs measured): {diff_all.median():.4f}%")
print(f"  max    diff_pct: {diff_all.max():.4f}%")
print(f"  % samples with diff < 1%: {100*(diff_all < 1.0).mean():.1f}%")
print(f"  % samples with diff < 5%: {100*(diff_all < 5.0).mean():.1f}%")
print()


# ── Figures ───────────────────────────────────────────────────────────────────

print("="*65)
print("Generating figures")
print("="*65)

MODELS = ["heston","kou"]
LAYERS = ["A","B1"]
MODEL_COLORS = {"A":"#2196F3","B1":"#FF5722"}
LAYER_LABELS = {"A":"Layer A (single options)","B1":"Layer B-1 (portfolio)"}

# Figure 1: VR(k) curves
fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))
for ax, model in zip(axes1, MODELS):
    for layer in LAYERS:
        sub = df_res[(df_res.model==model)&(df_res.layer==layer)]
        if len(sub)==0: continue
        x_vals = []; med_vals=[]; q25_vals=[]; q75_vals=[]
        for k in K_UNIQUE:
            sk = sub[sub.k==k].VR_k.dropna()
            x_vals.append(k); med_vals.append(sk.median())
            q25_vals.append(sk.quantile(0.25)); q75_vals.append(sk.quantile(0.75))
        ax.plot(x_vals, med_vals, marker="o", color=MODEL_COLORS[layer],
                label=f"{LAYER_LABELS[layer]} (n={len(sub)//len(K_UNIQUE)})", lw=1.8)
        ax.fill_between(x_vals, q25_vals, q75_vals, alpha=0.2, color=MODEL_COLORS[layer])

    # r_eff line
    reff_m = np.mean([float(npz_cache[(model,p)]["reff_eta"]) for p in ["A","B","C","D"]])
    ax.axvline(reff_m, color="gray", lw=1.2, ls="--", label=f"r_eff={reff_m:.2f}")
    for hl, ls in [(0.5,":"),(0.7,"--"),(0.9,":"),(0.99,"--")]:
        ax.axhline(hl, color="black", lw=0.6, ls=ls, alpha=0.5)

    ax.set_xscale("log"); ax.set_xlabel("k (number of hedge dimensions)", fontsize=11)
    ax.set_ylabel("VR_k (Variance Reduction)", fontsize=11)
    ax.set_title(f"{model.capitalize()}: VR vs k", fontsize=12)
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(K_UNIQUE); ax.set_xticklabels([str(k) for k in K_UNIQUE], fontsize=8)

fig1.suptitle(r"Multi-dimensional $K_\eta$ hedging: VR($k$) curve (Theorem 4.7)", fontsize=12)
fig1.tight_layout()
p1 = FIGURES/"fig_multidim_VR_curve.png"
fig1.savefig(p1, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Fig → {p1.relative_to(ROOT)}")

# Figure 2: Theorem 4.7 verification scatter
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
LAYER_COLORS2 = {"A":"#2196F3","B1":"#FF5722"}
for ax, model in zip(axes2, MODELS):
    for layer in LAYERS:
        sub = df_res[(df_res.model==model)&(df_res.layer==layer)].dropna(subset=["VR_k","predicted_VR"])
        if len(sub)==0: continue
        ax.scatter(sub.predicted_VR, sub.VR_k, s=6, alpha=0.4,
                   color=LAYER_COLORS2[layer], label=LAYER_LABELS[layer])
    ax.plot([0,1],[0,1], color="black", lw=1.0, ls="--")
    ax.set_xlabel("Predicted VR (Theorem 4.7)", fontsize=10)
    ax.set_ylabel("Measured VR", fontsize=10)
    ax.set_title(f"{model.capitalize()}: Measured vs Predicted", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlim(-0.05,1.05); ax.set_ylim(-0.05,1.05)

fig2.suptitle("Theorem 4.7: Predicted vs Measured VR across all (sample, k)", fontsize=12)
fig2.tight_layout()
p2 = FIGURES/"fig_theorem47_verification.png"
fig2.savefig(p2, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Fig → {p2.relative_to(ROOT)}")


# ── Save CSVs ─────────────────────────────────────────────────────────────────

df_res.to_csv(UNIFIED/"multidim_hedging_per_sample.csv", index=False)
df_sum.to_csv(UNIFIED/"multidim_hedging_summary.csv", index=False)
df_sat.to_csv(UNIFIED/"multidim_hedging_saturation.csv", index=False)

print(f"\nCSV → multidim_hedging_per_sample.csv  ({len(df_res)} rows)")
print(f"CSV → multidim_hedging_summary.csv     ({len(df_sum)} rows)")
print(f"CSV → multidim_hedging_saturation.csv  ({len(df_sat)} rows)")


# ── Stdout summary ────────────────────────────────────────────────────────────

print()
print("="*65)
print("STDOUT SUMMARY")
print("="*65)
print()
print("Phase 2 VR_k medians (all samples):")
print(f"  {'k':>6}  {'VR_med':>8}  ≥50% ≥70% ≥90% ≥99%  thm_diff%")
for k in K_UNIQUE:
    r = df_sum[(df_sum.model=="all")&(df_sum.k==k)]
    if len(r)==0: continue
    r = r.iloc[0]
    print(f"  {r.k:6d}  {r.VR_median:8.4f}  {r.n_VR_above_50:4d} {r.n_VR_above_70:4d} "
          f"{r.n_VR_above_90:4d} {r.n_VR_above_99:4d}  {r.diff_pct_median:.3f}%")
print()
print("Phase 3 saturation:")
for _, r in df_sat[df_sat.layer!="all"].iterrows():
    print(f"  {r.model:10} {r.layer:5}: k@95%={r.k_for_VR95_median:4d}  "
          f"k@99%={r.k_for_VR99_median:4d}  r_eff={r.r_eff_avg:.2f}  "
          f"VR@k=r_eff={r.VR_at_k_eq_reff_median:.4f}")
all_r = df_sat[df_sat.model=="all"].iloc[0]
print(f"  {'all':10} {'all':5}: k@95%={all_r.k_for_VR95_median:4d}  "
      f"k@99%={all_r.k_for_VR99_median:4d}  r_eff={all_r.r_eff_avg:.2f}")
print()
print(f"Phase 4 verdict: {verdict}")
print(f"  VR_k=2={VR_k2:.4f}  VR_k=5={VR_k5:.4f}  k@99%={k99_med}  r_eff={r_eff_a:.2f}")
print()
print(f"Phase 5 Theorem 4.7: median diff={diff_all.median():.4f}%  max={diff_all.max():.4f}%")
print()
print("HEDGING STEP 2 COMPLETE.")
