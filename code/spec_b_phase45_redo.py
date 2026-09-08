"""
spec_b_phase45_redo.py — Phase 4 + Phase 5 redo with helper-based prediction
=============================================================================
Reuses 200 trained MLP seeds (heston_seeds/, kou_seeds/) from spec_b_mlp.py.
Fix: N_CAL_H_MLP=80, N_CAL_K_MLP=80, N_TEST_MLP=20 (was 950/150 — empty test split).
Helper unification: predict_split + compute_all_metrics used by both phases.
Sanity: dry-run on Exp01-A (ATM) before full execution; STOP on NaN.

CSV outputs (overwriting):
  mlp_thm35_50configs.csv  — 39 rows (Heston Exp01 5×3 + Heston b2 4×3 + Kou b2 4×3)
                             new cols: n_cal_mlp, n_test_mlp
  mlp_reff_snr_8pairs.csv  — 8 rows (Heston A-D, Kou A-D)
                             r_eff_deeponet from b4_table4_kernel_structure.csv (reff_eta)
                             gain_sq_deeponet from b3_asymptotic_gain.csv (gain_sq_asym)
                             new cols: n_cal_mlp, n_test_mlp

Figures (overwriting):
  fig_mlp_vs_deeponet_ratio.png
  fig_mlp_vs_deeponet_reff.png
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from NeuMoR.code.neumor_core import (
    build_k_eta, compute_d_hat, compute_g_star,
    snr_squared, compute_gain, effective_rank,
)

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()         else
    "cpu"
)
print(f"Device: {DEVICE}")

OUTDIR = ROOT / "NeuMoR/output/mlp_replication"

# ─── Constants ──────────────────────────────────────────────────────────────
N_SEEDS       = 100
N_CAL_H_MLP   = 80
N_CAL_K_MLP   = 80
N_TEST_MLP    = 20

KOU_LO = np.array([0.10,  0.5, 0.30,  3.0,  3.0], dtype=np.float32)
KOU_HI = np.array([0.80, 50.0, 0.70, 50.0, 50.0], dtype=np.float32)
H_PARAM_KEYS  = ["v", "kappa", "omega", "xi", "rho"]

HESTON_BASE  = {"v": 0.10, "kappa": 1.0, "omega": 0.2, "xi": 0.3, "rho": -0.5, "T": 1.0}
HESTON_DELTA = {"A": 0.001, "B": 0.005, "C": 0.010, "D": 0.030}

KOU_BASE_SIGMA  = 0.30
KOU_DELTA       = {"A": 0.001, "B": 0.005, "C": 0.010, "D": 0.030}
KOU_BASE_PARAMS = np.array([KOU_BASE_SIGMA, 10.0, 0.5, 15.0, 10.0], dtype=np.float64)


# ─── Architecture (same as spec_b_mlp.py) ───────────────────────────────────
class DensityMLP(nn.Module):
    def __init__(self, n_param=5, hidden=256, n_layers=5):
        super().__init__()
        d_in = n_param + 1
        layers = []
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d_in, hidden), nn.ReLU()]
            d_in = hidden
        layers += [nn.Linear(hidden, 1), nn.Softplus()]
        self.net = nn.Sequential(*layers)

    def forward(self, lam, y_grid):
        B, P = lam.shape
        n_y  = y_grid.shape[0]
        dy   = float((y_grid[1] - y_grid[0]).item())
        lam_exp = lam[:, None, :].expand(B, n_y, P)
        y_exp   = y_grid[None, :, None].expand(B, n_y, 1)
        x_flat  = torch.cat([lam_exp, y_exp], dim=-1).reshape(B * n_y, P + 1)
        out  = self.net(x_flat).reshape(B, n_y)
        mass = out.sum(dim=-1, keepdim=True) * dy
        return out / mass.clamp(min=1e-30)


# ─── Inference helpers ──────────────────────────────────────────────────────
def load_mlp_seeds(model_name, save_dir):
    models = []
    for seed in range(N_SEEDS):
        path = save_dir / f"seed_{seed:03d}.pt"
        ck   = torch.load(path, map_location=DEVICE, weights_only=False)
        mlp  = DensityMLP(n_param=5, hidden=256, n_layers=5).to(DEVICE)
        mlp.load_state_dict(ck["state_dict"])
        mlp.eval()
        models.append(mlp)
    return models


def infer_mlp(models, lam_arr, y_np):
    y_t   = torch.tensor(y_np, dtype=torch.float32).to(DEVICE)
    lam_t = torch.tensor(lam_arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    out   = np.zeros((len(models), len(y_np)), dtype=np.float32)
    with torch.no_grad():
        for i, m in enumerate(models):
            out[i] = m(lam_t, y_t).cpu().numpy().squeeze()
    return out.astype(np.float64)


def infer_mlp_kou_normalised(models, raw_arr, y_np):
    norm = (raw_arr.astype(np.float32) - KOU_LO) / (KOU_HI - KOU_LO)
    return infer_mlp(models, norm.astype(np.float64), y_np)


# ─── Statistics helpers ─────────────────────────────────────────────────────
def folded_normal_mean(mu, sigma):
    if sigma < 1e-30:
        return abs(mu)
    s = mu / sigma
    return float(sigma * np.sqrt(2.0 / np.pi) * np.exp(-0.5 * s**2)
                 + mu * erf(s / np.sqrt(2.0)))


def thm35_ratio(g, d_hat, K_test, l1_test, l2_test, dy):
    noise2  = float(g @ (K_test @ g)) * dy**2
    sigma   = float(np.sqrt(max(noise2, 0.0)))
    mu      = float(g @ d_hat) * dy
    theory  = folded_normal_mean(mu, sigma)
    emp_vals = np.abs((l1_test - l2_test) @ g * dy)
    empirical = float(emp_vals.mean()) if emp_vals.size > 0 else float("nan")
    ratio = empirical / theory if abs(theory) > 1e-20 else float("nan")
    return sigma, mu, theory, empirical, ratio


# ═════════════════════════════════════════════════════════════════════════════
# NEW HELPERS: predict_split + compute_all_metrics
# ═════════════════════════════════════════════════════════════════════════════

def predict_split(models, lam1_raw, lam2_raw, y_np, n_cal, kou_norm=False):
    """Inference for both lambdas + cal/test split. Single source of truth."""
    if kou_norm:
        pdfs1 = infer_mlp_kou_normalised(models, lam1_raw, y_np)
        pdfs2 = infer_mlp_kou_normalised(models, lam2_raw, y_np)
    else:
        pdfs1 = infer_mlp(models, lam1_raw, y_np)
        pdfs2 = infer_mlp(models, lam2_raw, y_np)
    n_total = pdfs1.shape[0]
    n_tst   = n_total - n_cal
    assert n_total == N_SEEDS, f"Expected {N_SEEDS} seeds, got {n_total}"
    assert n_tst > 0, f"n_test must be > 0 (got n_total={n_total}, n_cal={n_cal})"
    return dict(
        l1_cal=pdfs1[:n_cal], l1_tst=pdfs1[n_cal:],
        l2_cal=pdfs2[:n_cal], l2_tst=pdfs2[n_cal:],
        n_cal_used=n_cal, n_tst_used=n_tst,
    )


def compute_all_metrics(pred, g_payoff, dy, g_baseline=None):
    """
    Cal-side: K_eta, d_hat, r_eff, g*.
    Test-side: K_test, ratio (Thm 3.5), gain_sq.
    g_payoff: payoff for Thm 3.5. g_baseline: baseline for gain_sq (default = g_payoff).
    """
    if g_baseline is None:
        g_baseline = g_payoff
    K_cal = build_k_eta(pred["l1_cal"], pred["l2_cal"])
    d_hat = compute_d_hat(pred["l1_cal"], pred["l2_cal"])
    r_eff_val = effective_rank(K_cal)
    g_star_arr, _, _, _, _ = compute_g_star(K_cal, d_hat, dy)
    K_test = build_k_eta(pred["l1_tst"], pred["l2_tst"])
    sigma_d, mu_d, theory, empirical, ratio = thm35_ratio(
        g_payoff, d_hat, K_test, pred["l1_tst"], pred["l2_tst"], dy
    )
    gain = compute_gain(g_star_arr, g_baseline, d_hat, K_test, dy)
    return dict(
        sigma_d=sigma_d, mu_d=mu_d, theory=theory,
        empirical=empirical, ratio=ratio,
        r_eff=r_eff_val,
        snr2_star=gain["snr2_star"],
        snr2_baseline=gain["snr2_baseline"],
        gain_sq=gain["gain_sq"],
        n_cal_used=pred["n_cal_used"],
        n_tst_used=pred["n_tst_used"],
    )


# ═════════════════════════════════════════════════════════════════════════════
# Setup: load seeds and y_grids
# ═════════════════════════════════════════════════════════════════════════════

print("\n=== Loading data + seeds ===")
h_npz = np.load(ROOT / "TNO/data/Heston/tno_dataset_v1.npz", allow_pickle=True)
h_y   = h_npz["y_grid"].astype(np.float64)
h_dy  = float(h_y[1] - h_y[0])

k_npz = np.load(ROOT / "NeuMoR/output/kou/dataset/kou_train_10k.npz", allow_pickle=True)
k_y   = k_npz["y_grid"].astype(np.float64)
k_dy  = float(k_y[1] - k_y[0])

print(f"  Heston y_grid: n_y={len(h_y)}, dy={h_dy:.6f}")
print(f"  Kou    y_grid: n_y={len(k_y)}, dy={k_dy:.6f}")

print("  Loading Heston MLP (100 seeds)...")
h_mlp = load_mlp_seeds("heston", OUTDIR / "heston_seeds")
print("  Loading Kou MLP (100 seeds)...")
k_mlp = load_mlp_seeds("kou", OUTDIR / "kou_seeds")

PAYOFFS = {
    "ATM":      lambda y: np.maximum(np.exp(y) - 1.0, 0.0),
    "OTM_K110": lambda y: np.maximum(np.exp(y) - 1.10, 0.0),
    "OTM_K125": lambda y: np.maximum(np.exp(y) - 1.25, 0.0),
}

EXP01_PAIRS = {
    "Exp01-A": (HESTON_BASE, {**HESTON_BASE, "v": 0.101}),
    "Exp01-B": (HESTON_BASE, {**HESTON_BASE, "v": 0.105}),
    "Exp01-C": (HESTON_BASE, {**HESTON_BASE, "v": 0.110}),
    "Exp01-D": (HESTON_BASE, {**HESTON_BASE, "v": 0.130}),
    "Exp01-E": (HESTON_BASE, {**HESTON_BASE, "v": 0.200}),
}

# Reference DeepONet ratios from b5
b5 = pd.read_csv(ROOT / "NeuMoR/output/unified/b5_thm35_all.csv")


# ═════════════════════════════════════════════════════════════════════════════
# DRY-RUN: Exp01-A ATM only → must be finite before full execution
# ═════════════════════════════════════════════════════════════════════════════

print("\n=== DRY-RUN: Exp01-A ATM ===")
p1, p2 = EXP01_PAIRS["Exp01-A"]
lam1 = np.array([p1[k] for k in H_PARAM_KEYS], dtype=np.float64)
lam2 = np.array([p2[k] for k in H_PARAM_KEYS], dtype=np.float64)
pred_dry = predict_split(h_mlp, lam1, lam2, h_y, N_CAL_H_MLP)
m_dry = compute_all_metrics(pred_dry, PAYOFFS["ATM"](h_y), h_dy)
print(f"  n_cal={m_dry['n_cal_used']}, n_tst={m_dry['n_tst_used']}")
print(f"  sigma_delta = {m_dry['sigma_d']:.4e}")
print(f"  mu_delta    = {m_dry['mu_d']:.4e}")
print(f"  theory_mr   = {m_dry['theory']:.4e}")
print(f"  empirical_mr= {m_dry['empirical']:.4e}")
print(f"  ratio_mlp   = {m_dry['ratio']:.4f}")
print(f"  r_eff_mlp   = {m_dry['r_eff']:.4f}")
print(f"  gain_sq_mlp = {m_dry['gain_sq']:.4f}")
assert np.isfinite(m_dry["ratio"]),   f"DRY-RUN FAIL: ratio={m_dry['ratio']}"
assert np.isfinite(m_dry["gain_sq"]), f"DRY-RUN FAIL: gain_sq={m_dry['gain_sq']}"
assert np.isfinite(m_dry["r_eff"]),   f"DRY-RUN FAIL: r_eff={m_dry['r_eff']}"
print("  DRY-RUN OK — proceeding to full execution\n")


# ═════════════════════════════════════════════════════════════════════════════
# Phase 4: Thm 3.5 ratio (39 rows = 5+4+4 configs × 3 payoffs)
# ═════════════════════════════════════════════════════════════════════════════

print("=== Phase 4: Thm 3.5 ratio (helper-based) ===")
thm35_rows = []

# ── Heston Exp01 (5 × 3 = 15 rows) ─────────────────────────────────────────
for cid, (p1, p2) in EXP01_PAIRS.items():
    lam1 = np.array([p1[k] for k in H_PARAM_KEYS], dtype=np.float64)
    lam2 = np.array([p2[k] for k in H_PARAM_KEYS], dtype=np.float64)
    pred = predict_split(h_mlp, lam1, lam2, h_y, N_CAL_H_MLP)
    for pname, gfn in PAYOFFS.items():
        m = compute_all_metrics(pred, gfn(h_y), h_dy)
        ratio = m["ratio"]
        in_rng = bool(0.7 <= ratio <= 1.3) if np.isfinite(ratio) else False
        ref = b5[(b5["config_id"] == cid) & (b5["payoff"] == pname)]
        ratio_d = float(ref["ratio"].iloc[0]) if len(ref) > 0 else float("nan")
        in_rng_d = bool(ref["in_range"].iloc[0]) if len(ref) > 0 else False
        thm35_rows.append(dict(
            config_id=cid, model="heston", payoff=pname,
            sigma_delta_mlp=round(m["sigma_d"], 8),
            mu_delta_mlp=round(m["mu_d"], 8),
            theory_mr_mlp=round(m["theory"], 8),
            empirical_mr_mlp=round(m["empirical"], 8),
            ratio_mlp=round(ratio, 4),
            ratio_deeponet=round(ratio_d, 4) if np.isfinite(ratio_d) else float("nan"),
            in_range_mlp=in_rng,
            in_range_deeponet=in_rng_d,
            n_cal_mlp=m["n_cal_used"],
            n_test_mlp=m["n_tst_used"],
        ))
    print(f"  {cid}: ratio_mlp={[round(r['ratio_mlp'], 4) for r in thm35_rows[-3:]]}")

# ── Heston b2 pairs (4 × 3 = 12 rows; not in b5 → ratio_deeponet=NaN) ──────
for pair in ["A", "B", "C", "D"]:
    p1 = HESTON_BASE.copy()
    p2 = {**HESTON_BASE, "v": HESTON_BASE["v"] + HESTON_DELTA[pair]}
    lam1 = np.array([p1[k] for k in H_PARAM_KEYS], dtype=np.float64)
    lam2 = np.array([p2[k] for k in H_PARAM_KEYS], dtype=np.float64)
    pred = predict_split(h_mlp, lam1, lam2, h_y, N_CAL_H_MLP)
    cid = f"b2-heston-{pair}"
    for pname, gfn in PAYOFFS.items():
        m = compute_all_metrics(pred, gfn(h_y), h_dy)
        ratio = m["ratio"]
        in_rng = bool(0.7 <= ratio <= 1.3) if np.isfinite(ratio) else False
        thm35_rows.append(dict(
            config_id=cid, model="heston", payoff=pname,
            sigma_delta_mlp=round(m["sigma_d"], 8),
            mu_delta_mlp=round(m["mu_d"], 8),
            theory_mr_mlp=round(m["theory"], 8),
            empirical_mr_mlp=round(m["empirical"], 8),
            ratio_mlp=round(ratio, 4),
            ratio_deeponet=float("nan"),
            in_range_mlp=in_rng,
            in_range_deeponet=False,
            n_cal_mlp=m["n_cal_used"],
            n_test_mlp=m["n_tst_used"],
        ))
    print(f"  {cid}: ratio_mlp={[round(r['ratio_mlp'], 4) for r in thm35_rows[-3:]]}")

# ── Kou b2 pairs (4 × 3 = 12 rows; match KouB-X stageB in b5) ──────────────
for pair in ["A", "B", "C", "D"]:
    lam1 = KOU_BASE_PARAMS.copy()
    lam2 = KOU_BASE_PARAMS.copy()
    lam2[0] = KOU_BASE_SIGMA + KOU_DELTA[pair]
    pred = predict_split(k_mlp, lam1, lam2, k_y, N_CAL_K_MLP, kou_norm=True)
    cid = f"kou-{pair}"
    kou_b5_cid = f"KouB-{pair}"  # explicit match: KouAB_stageB
    for pname, gfn in PAYOFFS.items():
        m = compute_all_metrics(pred, gfn(k_y), k_dy)
        ratio = m["ratio"]
        in_rng = bool(0.7 <= ratio <= 1.3) if np.isfinite(ratio) else False
        ref = b5[(b5["config_id"] == kou_b5_cid) & (b5["payoff"] == pname)]
        ratio_d = float(ref["ratio"].iloc[0]) if len(ref) > 0 else float("nan")
        in_rng_d = bool(ref["in_range"].iloc[0]) if len(ref) > 0 else False
        thm35_rows.append(dict(
            config_id=cid, model="kou", payoff=pname,
            sigma_delta_mlp=round(m["sigma_d"], 8),
            mu_delta_mlp=round(m["mu_d"], 8),
            theory_mr_mlp=round(m["theory"], 8),
            empirical_mr_mlp=round(m["empirical"], 8),
            ratio_mlp=round(ratio, 4),
            ratio_deeponet=round(ratio_d, 4) if np.isfinite(ratio_d) else float("nan"),
            in_range_mlp=in_rng,
            in_range_deeponet=in_rng_d,
            n_cal_mlp=m["n_cal_used"],
            n_test_mlp=m["n_tst_used"],
        ))
    print(f"  {cid}: ratio_mlp={[round(r['ratio_mlp'], 4) for r in thm35_rows[-3:]]}")

df_thm35 = pd.DataFrame(thm35_rows)
df_thm35.to_csv(OUTDIR / "mlp_thm35_50configs.csv", index=False)

# Sanity: all ratio_mlp finite
n_finite_ratio = int(df_thm35["ratio_mlp"].notna().sum())
print(f"\n  Phase 4 CSV: {len(df_thm35)} rows, ratio_mlp finite {n_finite_ratio}/{len(df_thm35)}")
assert n_finite_ratio == len(df_thm35), "Some ratio_mlp NaN — STOP"


# ═════════════════════════════════════════════════════════════════════════════
# Phase 5: r_eff + gain_sq (8 pairs)
# ═════════════════════════════════════════════════════════════════════════════

print("\n=== Phase 5: r_eff + gain_sq (helper-based) ===")
b3 = pd.read_csv(ROOT / "NeuMoR/output/unified/b3_asymptotic_gain.csv")
b4 = pd.read_csv(ROOT / "NeuMoR/output/unified/b4_table4_kernel_structure.csv")

reff_rows = []

# ── Heston A-D ──────────────────────────────────────────────────────────────
for pair in ["A", "B", "C", "D"]:
    p1 = HESTON_BASE.copy()
    p2 = {**HESTON_BASE, "v": HESTON_BASE["v"] + HESTON_DELTA[pair]}
    lam1 = np.array([p1[k] for k in H_PARAM_KEYS], dtype=np.float64)
    lam2 = np.array([p2[k] for k in H_PARAM_KEYS], dtype=np.float64)
    pred = predict_split(h_mlp, lam1, lam2, h_y, N_CAL_H_MLP)
    g_atm = np.maximum(np.exp(h_y) - 1.0, 0.0)
    m = compute_all_metrics(pred, g_atm, h_dy)
    b4_row = b4[(b4["model"] == "heston") & (b4["pair"] == pair)]
    b3_row = b3[(b3["model"] == "heston") & (b3["pair"] == pair)]
    r_eff_d   = float(b4_row["reff_eta"].iloc[0]) if len(b4_row) > 0 else float("nan")
    gain_sq_d = float(b3_row["gain_sq_asym"].iloc[0]) if len(b3_row) > 0 else float("nan")
    reff_rows.append(dict(
        model="heston", pair=pair,
        r_eff_mlp=round(m["r_eff"], 4),
        r_eff_deeponet=round(r_eff_d, 4),
        gain_sq_mlp=round(m["gain_sq"], 4),
        gain_sq_deeponet=round(gain_sq_d, 4),
        n_cal_mlp=m["n_cal_used"],
        n_test_mlp=m["n_tst_used"],
    ))
    print(f"  heston-{pair}: r_eff_mlp={m['r_eff']:.4f}  r_eff_deep={r_eff_d:.4f}  "
          f"gain_sq_mlp={m['gain_sq']:.4f}  gain_sq_deep={gain_sq_d:.4f}")

# ── Kou A-D ─────────────────────────────────────────────────────────────────
for pair in ["A", "B", "C", "D"]:
    lam1 = KOU_BASE_PARAMS.copy()
    lam2 = KOU_BASE_PARAMS.copy()
    lam2[0] = KOU_BASE_SIGMA + KOU_DELTA[pair]
    pred = predict_split(k_mlp, lam1, lam2, k_y, N_CAL_K_MLP, kou_norm=True)
    g_atm = np.maximum(np.exp(k_y) - 1.0, 0.0)
    m = compute_all_metrics(pred, g_atm, k_dy)
    b4_row = b4[(b4["model"] == "kou") & (b4["pair"] == pair)]
    b3_row = b3[(b3["model"] == "kou") & (b3["pair"] == pair)]
    r_eff_d   = float(b4_row["reff_eta"].iloc[0]) if len(b4_row) > 0 else float("nan")
    gain_sq_d = float(b3_row["gain_sq_asym"].iloc[0]) if len(b3_row) > 0 else float("nan")
    reff_rows.append(dict(
        model="kou", pair=pair,
        r_eff_mlp=round(m["r_eff"], 4),
        r_eff_deeponet=round(r_eff_d, 4),
        gain_sq_mlp=round(m["gain_sq"], 4),
        gain_sq_deeponet=round(gain_sq_d, 4),
        n_cal_mlp=m["n_cal_used"],
        n_test_mlp=m["n_tst_used"],
    ))
    print(f"  kou-{pair}:    r_eff_mlp={m['r_eff']:.4f}  r_eff_deep={r_eff_d:.4f}  "
          f"gain_sq_mlp={m['gain_sq']:.4f}  gain_sq_deep={gain_sq_d:.4f}")

df_reff = pd.DataFrame(reff_rows)
df_reff.to_csv(OUTDIR / "mlp_reff_snr_8pairs.csv", index=False)

# Sanity: all 8 r_eff_deeponet and gain_sq_deeponet finite
n_finite_r = int(df_reff["r_eff_deeponet"].notna().sum())
n_finite_g = int(df_reff["gain_sq_deeponet"].notna().sum())
print(f"\n  Phase 5 CSV: {len(df_reff)} rows")
print(f"    r_eff_deeponet finite:   {n_finite_r}/{len(df_reff)}")
print(f"    gain_sq_deeponet finite: {n_finite_g}/{len(df_reff)}")
assert n_finite_r == len(df_reff), "Some r_eff_deeponet NaN — STOP"
assert n_finite_g == len(df_reff), "Some gain_sq_deeponet NaN — STOP"


# ═════════════════════════════════════════════════════════════════════════════
# Figures
# ═════════════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fig 1: ratio_mlp vs ratio_deeponet
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, mn in zip(axes, ["heston", "kou"]):
    sub = df_thm35[df_thm35["model"] == mn].dropna(subset=["ratio_mlp", "ratio_deeponet"])
    ax.scatter(sub["ratio_deeponet"], sub["ratio_mlp"], s=24, alpha=0.7, color="steelblue")
    lims = [0.5, 1.5]
    ax.fill_between(lims, [0.7, 0.7], [1.3, 1.3], alpha=0.10, color="green", label="[0.7,1.3]")
    ax.plot(lims, lims, "k--", lw=1, label="y=x")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("ratio (DeepONet)"); ax.set_ylabel("ratio (MLP)")
    ax.set_title(f"{mn.capitalize()} (n={len(sub)})")
    ax.legend(fontsize=8)
fig.suptitle("Thm 3.5 Ratio: MLP vs DeepONet", fontsize=12)
fig.tight_layout()
fig.savefig(OUTDIR / "fig_mlp_vs_deeponet_ratio.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Fig 2: gain_sq + r_eff comparison
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
colors = {"heston": "steelblue", "kou": "darkorange"}

ax = axes[0]
for mn in ["heston", "kou"]:
    sub = df_reff[df_reff["model"] == mn]
    ax.scatter(sub["gain_sq_deeponet"], sub["gain_sq_mlp"],
               s=70, color=colors[mn], label=mn, zorder=3)
all_g = pd.concat([df_reff["gain_sq_deeponet"], df_reff["gain_sq_mlp"]]).dropna()
lims = [all_g.min() * 0.9, all_g.max() * 1.1]
ax.plot(lims, lims, "k--", lw=1, label="y=x")
ax.set_xlabel("gain_sq (DeepONet)"); ax.set_ylabel("gain_sq (MLP)")
ax.set_title("SNR² Gain: MLP vs DeepONet")
ax.legend(fontsize=9)

ax = axes[1]
for mn in ["heston", "kou"]:
    sub = df_reff[df_reff["model"] == mn]
    ax.scatter(sub["r_eff_deeponet"], sub["r_eff_mlp"],
               s=70, color=colors[mn], label=mn, zorder=3)
all_r = pd.concat([df_reff["r_eff_deeponet"], df_reff["r_eff_mlp"]]).dropna()
lims_r = [all_r.min() * 0.9, all_r.max() * 1.1]
ax.plot(lims_r, lims_r, "k--", lw=1, label="y=x")
ax.set_xlabel("r_eff (DeepONet)"); ax.set_ylabel("r_eff (MLP)")
ax.set_title("Effective Rank: MLP vs DeepONet")
ax.legend(fontsize=9)

fig.suptitle("MLP vs DeepONet — 8 pairs (Heston A-D, Kou A-D)", fontsize=12)
fig.tight_layout()
fig.savefig(OUTDIR / "fig_mlp_vs_deeponet_reff.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"\n=== Figures saved ===")


# ═════════════════════════════════════════════════════════════════════════════
# Summary stdout
# ═════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}\nSUMMARY\n{'='*70}\n")

print("[mlp_thm35_50configs.csv ratio_mlp distribution]")
r_mlp = df_thm35["ratio_mlp"]
print(f"  median={r_mlp.median():.4f}  q25={r_mlp.quantile(0.25):.4f}  "
      f"q75={r_mlp.quantile(0.75):.4f}")
print(f"  min={r_mlp.min():.4f}  max={r_mlp.max():.4f}")
print(f"  in_range_mlp fraction: {df_thm35['in_range_mlp'].mean():.4f}  "
      f"({df_thm35['in_range_mlp'].sum()}/{len(df_thm35)})")

print("\n[mlp_thm35_50configs.csv ratio_deeponet distribution (cross-check)]")
r_dp = df_thm35["ratio_deeponet"].dropna()
print(f"  median={r_dp.median():.4f}  q25={r_dp.quantile(0.25):.4f}  "
      f"q75={r_dp.quantile(0.75):.4f}")
print(f"  min={r_dp.min():.4f}  max={r_dp.max():.4f}  n_finite={len(r_dp)}")
print(f"  in_range_deeponet fraction: {df_thm35['in_range_deeponet'].mean():.4f}  "
      f"({df_thm35['in_range_deeponet'].sum()}/{len(df_thm35)})")

print("\n[mlp_reff_snr_8pairs.csv — r_eff comparison]")
print(f"  {'model':<8}{'pair':<6}{'r_eff_mlp':<14}{'r_eff_deeponet':<16}")
for _, row in df_reff.iterrows():
    print(f"  {row['model']:<8}{row['pair']:<6}{row['r_eff_mlp']:<14.4f}"
          f"{row['r_eff_deeponet']:<16.4f}")

print("\n[mlp_reff_snr_8pairs.csv — gain_sq comparison]")
print(f"  {'model':<8}{'pair':<6}{'gain_sq_mlp':<14}{'gain_sq_deeponet':<16}")
for _, row in df_reff.iterrows():
    print(f"  {row['model']:<8}{row['pair']:<6}{row['gain_sq_mlp']:<14.4f}"
          f"{row['gain_sq_deeponet']:<16.4f}")

print(f"\n=== COMPLETE ===")
