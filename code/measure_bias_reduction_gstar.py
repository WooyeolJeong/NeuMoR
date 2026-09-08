"""
measure_bias_reduction_gstar.py — direct measurement of bias at g_ATM vs g*.
Direct measurement of bias quantities at g_ATM vs g* (K-projection optimal).
8 pairs × 2 payoffs. Reuses existing seed ensembles (no new simulation).
Output: NeuMoR/output/unified/bias_reduction_gstar.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import erf

ROOT    = Path(__file__).resolve().parents[2]
NPZ_DIR = ROOT / "NeuMoR/output/unified"
TMP_DIR = ROOT / "NeuMoR/output/day5/_tmp"
OUT     = NPZ_DIR / "bias_reduction_gstar.csv"

# Test split per b2 protocol
TEST_RANGE = {"heston": (950, 1000), "kou": (150, 200)}
N_FILE     = {"heston": 1000,         "kou": 200}

def folded_normal_mean(mu, sigma):
    """E|X| for X ~ N(mu, sigma²)."""
    if sigma <= 0:
        return abs(mu)
    s = mu / sigma
    return sigma * (np.sqrt(2/np.pi) * np.exp(-s**2 / 2) + s * erf(s / np.sqrt(2)))

def measure_payoff(g, y_grid, K_test, d_ana, d_hat, l1_tst, l2_tst):
    """Compute σ_Δ, Δ, b_Δ, theory_mr, empirical_mr, deviation, residual for payoff g."""
    dy = float(y_grid[1] - y_grid[0])
    noise2 = float(g @ (K_test @ g)) * dy**2
    sigma  = float(np.sqrt(max(noise2, 0.0)))
    delta  = float(g @ d_ana) * dy
    b_p    = d_hat - d_ana
    b_del  = float(g @ b_p) * dy
    mu_g   = delta + b_del                          # = g @ d_hat * dy
    theory = folded_normal_mean(mu_g, sigma)
    dpdf   = l1_tst - l2_tst                        # (N_tst, n_grid)
    emp    = float(np.abs(dpdf @ g * dy).mean())
    abs_d  = abs(delta) if abs(delta) > 1e-20 else float("nan")
    dev_pct = abs(emp/abs_d - 1.0) * 100 if not np.isnan(abs_d) else float("nan")
    res_pct = abs(emp - theory) / abs_d * 100 if not np.isnan(abs_d) else float("nan")
    return dict(sigma=sigma, delta=delta, b_delta=b_del, mu=mu_g, s=mu_g/sigma if sigma>0 else float('nan'),
                theory_mr=theory, empirical_mr=emp,
                deviation_pct=dev_pct, residual_pct=res_pct)

# ── Main loop ──────────────────────────────────────────────────────────────────
rows = []
print("=== g* bias reduction measurement (8 pairs × {ATM, g*}) ===\n")

for model in ["heston", "kou"]:
    lo, hi = TEST_RANGE[model]
    Nfile  = N_FILE[model]
    for pair in ["A","B","C","D"]:
        npz = np.load(NPZ_DIR / f"{model}_pair{pair}.npz", allow_pickle=True)
        y       = npz["y_grid"]
        K_tst   = npz["K_eta_tst"]
        d_ana   = npz["d_ana"]
        d_hat   = npz["d_hat"]
        g_atm   = npz["g_atm"]
        g_star  = npz["g_star_eta_dhat"]
        snr2_atm   = float(npz["snr2_atm_dhat"])
        snr2_gstar = float(npz["snr2_gstar_eta_dhat"])

        l1 = np.load(TMP_DIR / f"{model}_pair{pair}_l1_{Nfile}.npy")[lo:hi]
        l2 = np.load(TMP_DIR / f"{model}_pair{pair}_l2_{Nfile}.npy")[lo:hi]

        r_atm = measure_payoff(g_atm,  y, K_tst, d_ana, d_hat, l1, l2)
        r_g   = measure_payoff(g_star, y, K_tst, d_ana, d_hat, l1, l2)

        gain_sq = snr2_gstar / snr2_atm if snr2_atm > 0 else float("nan")
        pred_red = 1.0 / np.sqrt(gain_sq) if gain_sq > 0 else float("nan")
        # actual reduction in deviation magnitude: g* / ATM (smaller is better)
        actual_red = r_g["deviation_pct"] / r_atm["deviation_pct"] if r_atm["deviation_pct"] > 0 else float("nan")

        rows.append(dict(
            model=model, pair=pair,
            snr2_atm=round(snr2_atm, 6),
            snr2_gstar=round(snr2_gstar, 6),
            gain_sq=round(gain_sq, 4),
            sigma_atm=r_atm["sigma"], sigma_gstar=r_g["sigma"],
            delta_atm=r_atm["delta"], delta_gstar=r_g["delta"],
            b_delta_atm=r_atm["b_delta"], b_delta_gstar=r_g["b_delta"],
            s_atm=r_atm["s"], s_gstar=r_g["s"],
            mr_emp_atm=r_atm["empirical_mr"],
            mr_emp_gstar=r_g["empirical_mr"],
            mr_theory_atm=r_atm["theory_mr"],
            mr_theory_gstar=r_g["theory_mr"],
            deviation_atm_pct=round(r_atm["deviation_pct"], 4),
            deviation_gstar_pct=round(r_g["deviation_pct"], 4),
            residual_atm_pct=round(r_atm["residual_pct"], 4),
            residual_gstar_pct=round(r_g["residual_pct"], 4),
            predicted_reduction_from_sqrt_gain=round(pred_red, 4),
            actual_reduction=round(actual_red, 4),
        ))

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)

# ── Sanity stdout ──────────────────────────────────────────────────────────────
def stats(x):
    return f"median {np.median(x):.2f}% [{np.min(x):.2f}, {np.max(x):.2f}]"

for m in ["heston","kou"]:
    sub = df[df.model == m]
    atm_dev = sub["deviation_atm_pct"].values
    gst_dev = sub["deviation_gstar_pct"].values
    atm_res = sub["residual_atm_pct"].values
    gst_res = sub["residual_gstar_pct"].values
    gain    = sub["gain_sq"].values
    actred  = sub["actual_reduction"].values

    label = m.capitalize()
    print(f"{label}: ATM deviation {stats(atm_dev)}, g* deviation {stats(gst_dev)}")
    print(f"          actual reduction factor: median {np.median(actred):.3f} (lower = bigger improvement)")
    print(f"          predicted from 1/√gain:  median {np.median(1/np.sqrt(gain)):.3f}")
    print(f"          ATM residual {stats(atm_res)}, g* residual {stats(gst_res)}")
    print()

print(f"Output: {OUT}")
