"""
bias_removal_thm35.py — §3 GT-aware bias removal experiment
=============================================================

For each of 387 configs in b5_thm35_all.csv:
  Delta   = g^T d_ana * dy           (true signal, GT)
  b_Delta = mu_delta - Delta          (true differential bias)
  J_hat   = folded-normal Jensen term (NN-only computable)
  S_hat   = |mu_delta| - |Delta|      (systematic term, GT-aware)
  corrected = empirical_mr - J_hat - S_hat
  ratio_corrected = corrected / |Delta|

Compare against b5 raw ratio = empirical_mr / theory_mr.

Outputs:
  NeuMoR/output/unified/bias_removal_results.csv   (387 rows × 15 cols)
  NeuMoR/output/unified/bias_removal_summary.csv   (~10 rows × 11 cols)
  NeuMoR/figures/fig_bias_removal.png

Run:
  python NeuMoR/code/bias_removal_thm35.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import erfc, erf

ROOT    = Path(__file__).resolve().parents[2]
UNIFIED = ROOT / "NeuMoR/output/unified"
FIG_DIR = ROOT / "NeuMoR/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from NeuMoR.src.heston_teacher import lewis_pdf
from NeuMoR.src.kou.kou_pricing import kou_log_density

# ── Config ────────────────────────────────────────────────────────────────────

HESTON_BASE = {"v": 0.10, "kappa": 1.0, "omega": 0.2, "xi": 0.3, "rho": -0.5, "T": 1.0}
HESTON_LO   = {"v": 0.051, "kappa": 0.502, "omega": 0.100, "xi": 0.102, "rho": -0.800}
HESTON_HI   = {"v": 0.300, "kappa": 2.000, "omega": 0.399, "xi": 0.599, "rho":  0.598}

KOU_FIXED = np.array([0.30, 5.0, 0.5, 10.0, 10.0], dtype=np.float64)
KOU_LO    = np.array([0.10,  0.5, 0.30,  3.0,  3.0], dtype=np.float64)
KOU_HI    = np.array([0.80, 50.0, 0.70, 50.0, 50.0], dtype=np.float64)

# Standard Heston pair deltas
HESTON_PAIR_DELTA = {"A": 0.001, "B": 0.005, "C": 0.010, "D": 0.030, "E": 0.100}
HESTON_F_DELTA    = {"F1": 0.0001, "F2": 0.0002, "F3": 0.0003, "F4": 0.0005, "F5": 0.0007}
HESTON_EXP06      = {
    "G1": {"kappa": 0.005}, "G2": {"rho": 0.005}, "G3": {"omega": 0.005},
    "H1": {"kappa": 0.002}, "H2": {"kappa": 0.020},
    "H3": {"xi": 0.003},    "H4": {"rho": 0.002},  "H5": {"omega": 0.001},
}
DIAG_BASE_STEPS = {
    "vkappa":  {"v": 0.001, "kappa": 0.01},
    "vomega":  {"v": 0.001, "omega": 0.002},
    "vxi":     {"v": 0.001, "xi": 0.003},
    "kappaxi": {"kappa": 0.01, "xi": 0.003},
    "vrho":    {"v": 0.001, "rho": 0.005},
    "all5":    {"v": 0.001, "kappa": 0.01, "omega": 0.002, "xi": 0.003, "rho": 0.005},
}
KOU_BTC_PARAMS = {
    "Luna_pre":  [0.587, 28.7, 0.43,  9.3, 13.9],
    "Luna_post": [0.740, 20.7, 0.33, 12.3,  7.4],
    "FTX_pre":   [0.405, 27.2, 0.60, 15.3, 12.8],
    "FTX_post":  [0.553, 13.8, 0.50, 10.0,  6.6],
    "ETF_pre":   [0.438, 36.5, 0.57, 17.6, 19.5],
    "ETF_post":  [0.599, 27.0, 0.50, 11.6, 12.7],
}

# ── Payoffs ───────────────────────────────────────────────────────────────────

def make_payoff(name: str, y: np.ndarray) -> np.ndarray:
    if name == "ATM":
        return np.maximum(np.exp(y) - 1.0, 0.0)
    elif name == "OTM_K110":
        return np.maximum(np.exp(y) - 1.10, 0.0)
    elif name == "OTM_K125":
        return np.maximum(np.exp(y) - 1.25, 0.0)
    raise ValueError(f"Unknown payoff: {name}")

# ── d_ana computation (cached) ────────────────────────────────────────────────

_DANA_CACHE: dict = {}

def heston_dana(p1: dict, p2: dict, y: np.ndarray) -> np.ndarray:
    key = (tuple(sorted(p1.items())), tuple(sorted(p2.items())), id(y))
    if key not in _DANA_CACHE:
        d1 = lewis_pdf(p1, y)
        d2 = lewis_pdf(p2, y)
        _DANA_CACHE[key] = d1 - d2
    return _DANA_CACHE[key]

def kou_dana(p1_raw: np.ndarray, p2_raw: np.ndarray, y: np.ndarray) -> np.ndarray:
    key = (tuple(p1_raw.tolist()), tuple(p2_raw.tolist()), len(y))
    if key not in _DANA_CACHE:
        T = 1.0
        dy = float(y[1] - y[0])
        d1 = kou_log_density(y, T, tuple(p1_raw.tolist()))
        d2 = kou_log_density(y, T, tuple(p2_raw.tolist()))
        _DANA_CACHE[key] = d1 - d2
    return _DANA_CACHE[key]

# ── Heston p2 reconstruction ──────────────────────────────────────────────────

def reconstruct_heston_p2(row: pd.Series) -> dict:
    blk  = row["block"]
    pair = row.get("pair", np.nan)
    p2   = HESTON_BASE.copy()

    if blk in ("Exp01", "Exp03"):
        p2["v"] = HESTON_BASE["v"] + HESTON_PAIR_DELTA[pair]
    elif blk == "Exp02":
        if pair in HESTON_PAIR_DELTA:
            p2["v"] = HESTON_BASE["v"] + HESTON_PAIR_DELTA[pair]
        elif pair in HESTON_F_DELTA:
            p2["v"] = HESTON_BASE["v"] + HESTON_F_DELTA[pair]
        elif pair in ("G1","G2","G3"):
            k, dv = list(HESTON_EXP06[pair].items())[0]
            p2[k]  = HESTON_BASE[k] + dv
        else:
            raise ValueError(f"Exp02 unknown pair {pair}")
    elif blk == "Exp06":
        for k, dv in HESTON_EXP06[pair].items():
            p2[k] = HESTON_BASE[k] + dv
    elif blk == "Exp05_axis":
        axis = str(row["axis"])
        step = float(row["step"])
        p2[axis] = float(np.clip(HESTON_BASE[axis] + step, HESTON_LO[axis], HESTON_HI[axis]))
    elif blk == "Exp05_diag":
        diag  = str(row["diag"])
        scale = float(row["scale"])
        for ax, base_step in DIAG_BASE_STEPS[diag].items():
            p2[ax] = float(np.clip(HESTON_BASE[ax] + base_step * scale,
                                   HESTON_LO[ax], HESTON_HI[ax]))
    elif blk == "Exp09":
        # Load from exp09 calibrations
        cal = _exp09_params()
        win = str(row["window"])
        p1_btc = cal[win]
        v2 = p1_btc["v"] + 0.01
        if v2 > 0.300 or abs(v2 - p1_btc["v"]) < 1e-9:
            v2 = p1_btc["v"] - 0.01
        return p1_btc, {**p1_btc, "v": float(np.clip(v2, 0.051, 0.300))}
    else:
        raise ValueError(f"Unknown Heston block: {blk}")

    # Default p1 = BASE_H
    return HESTON_BASE.copy(), p2

_EXP09_PARAMS = None
def _exp09_params():
    global _EXP09_PARAMS
    if _EXP09_PARAMS is None:
        df = pd.read_csv(ROOT / "NeuMoR/output/exp09/calibrations.csv")
        _EXP09_PARAMS = {}
        for _, row in df.iterrows():
            w = str(row["window"])
            _EXP09_PARAMS[w] = {k: float(row[f"clipped_{k}"]) for k in
                                 ["v","kappa","omega","xi","rho"]}
            _EXP09_PARAMS[w]["T"] = 1.0
    return _EXP09_PARAMS

# ── Jensen term ───────────────────────────────────────────────────────────────

def j_hat(sigma: float, mu: float) -> float:
    if sigma < 1e-30:
        return 0.0
    s = mu / sigma
    return float(sigma * (np.sqrt(2.0/np.pi) * np.exp(-0.5*s**2)
                          - abs(s) * erfc(abs(s) / np.sqrt(2.0))))

# ── y_grid loading ────────────────────────────────────────────────────────────

_H_YGRID = None
_K_YGRID = None

def heston_y():
    global _H_YGRID
    if _H_YGRID is None:
        npz = np.load(UNIFIED / "heston_pairA.npz", allow_pickle=True)
        _H_YGRID = npz["y_grid"].astype(np.float64)
    return _H_YGRID

def kou_y():
    global _K_YGRID
    if _K_YGRID is None:
        npz = np.load(UNIFIED / "kou_pairA.npz", allow_pickle=True)
        _K_YGRID = npz["y_grid"].astype(np.float64)
    return _K_YGRID

# ══════════════════════════════════════════════════════════════════════════════
# Phase 0: Sanity check
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PHASE 0: Sanity check")
print("=" * 60)

b5 = pd.read_csv(UNIFIED / "b5_thm35_all.csv")
assert b5[["sigma_delta","mu_delta","empirical_mr","theory_mr"]].isna().sum().sum() == 0
print(f"  b5 shape: {b5.shape}, no NaN/Inf")

# Check identity: theory_mr ≈ J + |mu| across all rows
j_all   = b5.apply(lambda r: j_hat(r.sigma_delta, r.mu_delta), axis=1)
mu_abs  = b5.mu_delta.abs()
reconst = j_all + mu_abs
diff    = (reconst - b5.theory_mr).abs()
print(f"  theory_mr = J + |mu| max_err = {diff.max():.2e}  (expect < 1e-10)")
assert diff.max() < 1e-5, f"Identity violated: max err {diff.max():.2e}"
print(f"  Identity OK")

# Spot check 1 config: Exp01 pairA ATM
row0  = b5[(b5.block=="Exp01") & (b5.pair=="A") & (b5.payoff=="ATM")].iloc[0]
y_h   = heston_y()
dy_h  = float(y_h[1] - y_h[0])
g0    = make_payoff("ATM", y_h)
p1_h  = HESTON_BASE.copy()
p2_h  = {**HESTON_BASE, "v": HESTON_BASE["v"] + HESTON_PAIR_DELTA["A"]}
da0   = heston_dana(p1_h, p2_h, y_h)
# mu_delta from scratch
npz_a = np.load(UNIFIED / "heston_pairA.npz", allow_pickle=True)
d_hat_a = npz_a["d_hat"]
mu0_recomputed = float(g0 @ d_hat_a * dy_h)
mu0_csv = float(row0.mu_delta)
print(f"  Spot check mu_delta: csv={mu0_csv:.8f}  recomputed={mu0_recomputed:.8f}  "
      f"diff={abs(mu0_csv - mu0_recomputed):.2e}")
assert abs(mu0_csv - mu0_recomputed) < 1e-5, "mu_delta mismatch in spot check"
print("  Spot check OK")
print()

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: GT-aware bias removal
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PHASE 1: GT-aware bias removal (387 configs)")
print("=" * 60)

results = []

for idx, row in b5.iterrows():
    blk     = row["block"]
    payoff  = row["payoff"]
    sigma   = float(row["sigma_delta"])
    mu      = float(row["mu_delta"])
    emp_mr  = float(row["empirical_mr"])
    thy_mr  = float(row["theory_mr"])

    # Get y_grid and payoff vector
    is_kou = blk.startswith("Kou")
    y  = kou_y() if is_kou else heston_y()
    dy = float(y[1] - y[0])
    g  = make_payoff(payoff, y)

    # Get d_ana
    try:
        if is_kou:
            if blk in ("KouAB_stageA", "KouAB_stageB"):
                pair = str(row["pair"])
                decomp = np.load(ROOT / "NeuMoR/output/day4/kou/kernel"
                                 / f"decomp_pair{pair}.npz")
                d_ana = (decomp["tp1"] - decomp["tp2"]).astype(np.float64)
            else:  # KouBTC
                scn = str(row["scenario"])
                p1_raw = np.clip(np.array(KOU_BTC_PARAMS[scn], dtype=np.float64),
                                 KOU_LO, KOU_HI)
                p2_raw = p1_raw.copy()
                p2_raw[0] = float(np.clip(p2_raw[0] + 0.01, KOU_LO[0], KOU_HI[0]))
                d_ana = kou_dana(p1_raw, p2_raw, y)
        else:
            if blk == "Exp09":
                p1_e9, p2_e9 = reconstruct_heston_p2(row)
                d_ana = heston_dana(p1_e9, p2_e9, y)
            else:
                _, p2_h = reconstruct_heston_p2(row)
                d_ana = heston_dana(HESTON_BASE.copy(), p2_h, y)
    except Exception as e:
        print(f"  [WARN] {row.config_id} {payoff}: d_ana failed: {e}")
        d_ana = None

    if d_ana is None:
        continue

    # True signal and decomposition
    Delta   = float(g @ d_ana * dy)
    b_Delta = mu - Delta
    J       = j_hat(sigma, mu)
    S       = abs(mu) - abs(Delta)
    corrected = emp_mr - J - S

    Delta_abs = abs(Delta)
    ratio_corrected = corrected / Delta_abs if Delta_abs > 1e-20 else float("nan")
    ratio_b5 = float(row["ratio"])

    results.append(dict(
        model="kou" if is_kou else "heston",
        block=blk,
        exp=row.get("exp", blk),
        config_id=row["config_id"],
        payoff=payoff,
        sigma_delta=sigma,
        mu_delta=mu,
        Delta=Delta,
        b_Delta=b_Delta,
        J_hat=J,
        S_hat=S,
        empirical_mr=emp_mr,
        theory_mr=thy_mr,
        corrected=corrected,
        ratio_corrected=ratio_corrected,
        ratio_b5_raw=ratio_b5,
    ))

df_res = pd.DataFrame(results)
print(f"  Computed: {len(df_res)} / 387 configs")
n_nan = df_res.ratio_corrected.isna().sum()
print(f"  NaN ratio_corrected: {n_nan}")

# Basic distribution check
valid = df_res.ratio_corrected.dropna()
print(f"  ratio_corrected: min={valid.min():.4f}  max={valid.max():.4f}  "
      f"median={valid.median():.4f}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Aggregation + verdict
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PHASE 2: Aggregation")
print("=" * 60)

summary_rows = []
combos = [("heston", p) for p in ["ATM","OTM_K110","OTM_K125"]] + \
         [("kou",    p) for p in ["ATM","OTM_K110","OTM_K125"]] + \
         [("all",    p) for p in ["ATM","OTM_K110","OTM_K125"]] + \
         [("heston", "all"), ("kou", "all"), ("all", "all")]

for (model_cls, payoff_cls) in combos:
    sub = df_res.copy()
    if model_cls != "all":
        sub = sub[sub.model == model_cls]
    if payoff_cls != "all":
        sub = sub[sub.payoff == payoff_cls]
    sub = sub.dropna(subset=["ratio_corrected"])
    if len(sub) == 0:
        continue

    rc  = sub.ratio_corrected.values
    rb5 = sub.ratio_b5_raw.values

    improvement = np.where(
        np.abs(rb5 - 1) > 1e-10,
        np.abs(rc - 1) / np.abs(rb5 - 1),
        np.nan
    )
    imp_med = float(np.nanmedian(improvement))
    rc_med  = float(np.median(rc))

    summary_rows.append(dict(
        model_class=model_cls,
        payoff=payoff_cls,
        n=len(sub),
        ratio_corrected_med=round(rc_med, 4),
        ratio_corrected_q25=round(float(np.percentile(rc, 25)), 4),
        ratio_corrected_q75=round(float(np.percentile(rc, 75)), 4),
        ratio_corrected_min=round(float(rc.min()), 4),
        ratio_corrected_max=round(float(rc.max()), 4),
        ratio_b5_med=round(float(np.median(rb5)), 4),
        in_band_95_105=int(np.sum((rc >= 0.95) & (rc <= 1.05))),
        improvement_med=round(imp_med, 4),
    ))

    if model_cls == "all" and payoff_cls == "all":
        print(f"  ALL: n={len(sub)}  ratio_corr_med={rc_med:.4f}  "
              f"b5_med={float(np.median(rb5)):.4f}  improvement_med={imp_med:.4f}")
    elif payoff_cls == "all":
        print(f"  {model_cls:8s} (all payoffs): "
              f"ratio_corr_med={rc_med:.4f}  improvement_med={imp_med:.4f}")

df_sum = pd.DataFrame(summary_rows)

print()
print("Per-model per-payoff:")
print(f"  {'model':8} {'payoff':10} {'n':5} {'rc_med':8} {'b5_med':8} "
      f"{'imp_med':8} {'in_band':8}")
for _, r in df_sum[(df_sum.model_class!="all") & (df_sum.payoff!="all")].iterrows():
    print(f"  {r.model_class:8} {r.payoff:10} {r.n:5d} "
          f"{r.ratio_corrected_med:8.4f} {r.ratio_b5_med:8.4f} "
          f"{r.improvement_med:8.4f} {r.in_band_95_105:8d}")

# Verdict
all_row = df_sum[(df_sum.model_class=="all") & (df_sum.payoff=="all")].iloc[0]
rc_med_all  = float(all_row.ratio_corrected_med)
imp_med_all = float(all_row.improvement_med)

print()
print("VERDICT:")
if 0.95 <= rc_med_all <= 1.05 and imp_med_all < 0.5:
    verdict = "STRONG — bias removal works (median ratio ∈ [0.95,1.05], improvement < 0.5)"
elif 0.85 <= rc_med_all <= 1.15 and imp_med_all < 1.0:
    verdict = "PARTIAL — bias removal partially works (median ∈ [0.85,1.15], improvement < 1.0)"
else:
    verdict = "NOT WORKING — (median outside [0.85,1.15] or improvement ≥ 1.0)"
print(f"  ratio_corrected_med = {rc_med_all:.4f}")
print(f"  improvement_med     = {imp_med_all:.4f}")
print(f"  → {verdict}")

# ── Save CSVs ─────────────────────────────────────────────────────────────────

df_res.to_csv(UNIFIED / "bias_removal_results.csv", index=False)
df_sum.to_csv(UNIFIED / "bias_removal_summary.csv", index=False)
print()
print(f"CSV → NeuMoR/output/unified/bias_removal_results.csv  ({len(df_res)} rows)")
print(f"CSV → NeuMoR/output/unified/bias_removal_summary.csv  ({len(df_sum)} rows)")

# ── Phase 3: Figure ───────────────────────────────────────────────────────────

print()
print("=" * 60)
print("PHASE 3: Figure")
print("=" * 60)

PAYOFFS   = ["ATM", "OTM_K110", "OTM_K125"]
MODELS    = ["heston", "kou"]
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
fig.suptitle("Bias removal: b5 raw ratio vs corrected ratio\n"
             "(closer to diagonal = same; above diagonal = corrected closer to 1)",
             fontsize=11)

for ri, model in enumerate(MODELS):
    for ci, payoff in enumerate(PAYOFFS):
        ax  = axes[ri, ci]
        sub = df_res[(df_res.model==model) & (df_res.payoff==payoff)].dropna(
            subset=["ratio_corrected"])

        x = sub.ratio_b5_raw.values
        y_v = sub.ratio_corrected.values

        ax.scatter(x, y_v, s=15, alpha=0.5, color="steelblue")
        lims = [min(x.min(), y_v.min(), 0.7), max(x.max(), y_v.max(), 1.3)]
        ax.plot(lims, lims, "k--", lw=0.8, label="y=x (no change)")
        ax.axhline(1.0, color="crimson", lw=0.8, ls="--", alpha=0.7, label="ratio=1")
        ax.axvline(1.0, color="crimson", lw=0.8, ls=":",  alpha=0.4)

        sm = df_sum[(df_sum.model_class==model) & (df_sum.payoff==payoff)].iloc[0]
        ax.set_title(f"{model.capitalize()} {payoff}\n"
                     f"b5_med={sm.ratio_b5_med:.3f} → corr_med={sm.ratio_corrected_med:.3f}  "
                     f"imp={sm.improvement_med:.3f}", fontsize=8)
        ax.set_xlabel("b5 raw ratio", fontsize=8)
        ax.set_ylabel("ratio corrected", fontsize=8)
        ax.set_xlim(lims); ax.set_ylim(lims)
        if ri == 0 and ci == 2:
            ax.legend(fontsize=7)

plt.tight_layout()
p = FIG_DIR / "fig_bias_removal.png"
fig.savefig(p, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Fig → NeuMoR/figures/fig_bias_removal.png")
print()
print("BIAS REMOVAL EXPERIMENT COMPLETE.")
