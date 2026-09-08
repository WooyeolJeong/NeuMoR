"""
mlp_residual_decomposition.py — Issue 9b Q2: MLP residual NN error decomposition
==================================================================================
Apply Follow-up D decomposition (Jensen + Systematic + residual) to MLP 39 configs.
Compare to DeepONet residual (bias_removal_results.csv).

Decomposition (per row):
  s        = (Delta + b_Delta) / sigma_delta            [b_Delta is NN-specific]
  J        = sigma_delta * (sqrt(2/pi)*exp(-s^2/2) - |s|*erfc(|s|/sqrt(2)))
  S        = |Delta + b_Delta| - |Delta|
  M_pred   = J + S
  raw_acc  = |empirical_mr - |Delta|| / max(|Delta|, 1e-8)
  residual = (empirical_mr - (|Delta| + M_pred)) / max(|Delta|, 1e-8)
  expl_fr  = M_pred / max(empirical_mr - |Delta|, 1e-8)

For MLP:      sigma=sigma_delta_mlp, mu=mu_delta_mlp,  empirical=empirical_mr_mlp
              b_Delta_mlp = mu_delta_mlp - Delta (derived)
For DeepONet: sigma=sigma_delta (b5/bias_removal), mu=mu_delta, b_Delta from bias_removal
              empirical_mr from bias_removal

Row scope: 27 rows (Heston Exp01 15 + Kou stageB 12). b2-heston excluded (no Lewis).

Outputs in NeuMoR/output/mlp_replication/:
  mlp_decomposition_per_config.csv (27)
  mlp_decomposition_summary.csv (per (model, block))
  mlp_decomposition_vs_deeponet.csv (27)
  fig_mlp_vs_deeponet_residual.png
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import erfc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT   = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "NeuMoR/output/mlp_replication"

# ─── Load sources ────────────────────────────────────────────────────────────
mlp = pd.read_csv(OUTDIR / "mlp_thm35_50configs.csv")
br  = pd.read_csv(ROOT / "NeuMoR/output/unified/bias_removal_results.csv")

print(f"Loaded mlp_thm35: {len(mlp)} rows")
print(f"Loaded bias_removal: {len(br)} rows")

# ─── Phase 1: Join ───────────────────────────────────────────────────────────
# Exp01-A~E (Heston): direct config_id match
# kou-A~D: rename to KouB-A~D for stageB match
# b2-heston-A~D: drop (no Lewis Delta)

mlp_keep = mlp[mlp["config_id"].str.startswith("Exp01") | mlp["config_id"].str.startswith("kou-")].copy()
mlp_keep["bias_removal_cid"] = mlp_keep["config_id"].apply(
    lambda c: c.replace("kou-", "KouB-") if c.startswith("kou-") else c
)

print(f"\nPhase 1: Filter b2-heston, rename kou → KouB")
print(f"  mlp_keep rows: {len(mlp_keep)}")
print(f"  unique cids: {sorted(mlp_keep['config_id'].unique())}")

# Join
joined = mlp_keep.merge(
    br[["model", "block", "config_id", "payoff",
        "sigma_delta", "mu_delta", "Delta", "b_Delta",
        "J_hat", "S_hat", "empirical_mr", "theory_mr"]],
    left_on=["bias_removal_cid", "payoff"],
    right_on=["config_id", "payoff"],
    how="left",
    suffixes=("", "_dp"),
)

n_matched = int(joined["Delta"].notna().sum())
print(f"  joined rows: {len(joined)}  matched (Delta finite): {n_matched}")
assert n_matched == len(joined), f"Some rows unmatched: {len(joined) - n_matched}"
assert n_matched == 27, f"Expected 27 matched rows, got {n_matched}"

# ─── Decomposition formula ───────────────────────────────────────────────────
def decompose(sigma_delta, mu_delta, delta_lewis, empirical_mr):
    """
    Returns: dict(b_Delta, s, J, S, M_pred, raw_acc, residual, expl_fr).
    b_Delta = mu_delta - delta_lewis (NN-specific bias).
    """
    b_Delta  = mu_delta - delta_lewis
    abs_dl   = max(abs(delta_lewis), 1e-8)
    eff_mu   = delta_lewis + b_Delta  # = mu_delta
    s        = eff_mu / sigma_delta if sigma_delta > 1e-30 else 0.0
    J        = sigma_delta * (
        np.sqrt(2.0 / np.pi) * np.exp(-0.5 * s**2)
        - abs(s) * erfc(abs(s) / np.sqrt(2.0))
    )
    S        = abs(eff_mu) - abs(delta_lewis)
    M_pred   = J + S
    raw_acc  = abs(empirical_mr - abs(delta_lewis)) / abs_dl
    residual = (empirical_mr - (abs(delta_lewis) + M_pred)) / abs_dl
    denom_ef = (empirical_mr - abs(delta_lewis))
    expl_fr  = M_pred / denom_ef if abs(denom_ef) > 1e-12 else float("nan")
    return dict(b_Delta=b_Delta, s=s, J=J, S=S,
                M_pred=M_pred, raw_acc=raw_acc, residual=residual, expl_fr=expl_fr)

# ─── Sanity: verify decompose() matches bias_removal J_hat/S_hat ─────────────
print("\nSanity check: recompute DeepONet J/S, compare to bias_removal J_hat/S_hat")
max_diff_J = 0.0
max_diff_S = 0.0
for _, r in joined.iterrows():
    d = decompose(r["sigma_delta"], r["mu_delta"], r["Delta"], r["empirical_mr"])
    diff_J = abs(d["J"] - r["J_hat"])
    diff_S = abs(d["S"] - r["S_hat"])
    max_diff_J = max(max_diff_J, diff_J)
    max_diff_S = max(max_diff_S, diff_S)
print(f"  max |J_recomputed - J_hat| = {max_diff_J:.3e}")
print(f"  max |S_recomputed - S_hat| = {max_diff_S:.3e}")
assert max_diff_J < 1e-15, "J recomputation mismatch — formula error"
assert max_diff_S < 1e-15, "S recomputation mismatch — formula error"
print(f"  PASS — Phase 2 formula matches Follow-up D exactly")

# ─── Phase 2: MLP decomposition (27 rows) ────────────────────────────────────
# ─── Phase 3: DeepONet decomposition (same 27 rows, same formula) ────────────
per_rows = []
for _, r in joined.iterrows():
    delta_lewis = r["Delta"]

    m_mlp = decompose(r["sigma_delta_mlp"], r["mu_delta_mlp"],
                      delta_lewis, r["empirical_mr_mlp"])
    m_dp  = decompose(r["sigma_delta"],     r["mu_delta"],
                      delta_lewis, r["empirical_mr"])

    per_rows.append(dict(
        config_id=r["config_id"],
        model=r["model"],
        block=r["block"],
        payoff=r["payoff"],
        Delta=r["Delta"],
        # MLP side
        sigma_delta_mlp=r["sigma_delta_mlp"],
        mu_delta_mlp=r["mu_delta_mlp"],
        b_Delta_mlp=m_mlp["b_Delta"],
        empirical_mr_mlp=r["empirical_mr_mlp"],
        J_mlp=m_mlp["J"], S_mlp=m_mlp["S"], M_pred_mlp=m_mlp["M_pred"],
        raw_acc_err_mlp=m_mlp["raw_acc"],
        residual_NN_error_mlp=m_mlp["residual"],
        explained_fraction_mlp=m_mlp["expl_fr"],
        # DeepONet side
        sigma_delta_deeponet=r["sigma_delta"],
        mu_delta_deeponet=r["mu_delta"],
        b_Delta_deeponet=m_dp["b_Delta"],
        empirical_mr_deeponet=r["empirical_mr"],
        J_deeponet=m_dp["J"], S_deeponet=m_dp["S"], M_pred_deeponet=m_dp["M_pred"],
        raw_acc_err_deeponet=m_dp["raw_acc"],
        residual_NN_error_deeponet=m_dp["residual"],
        explained_fraction_deeponet=m_dp["expl_fr"],
    ))

df_per = pd.DataFrame(per_rows)
df_per.to_csv(OUTDIR / "mlp_decomposition_per_config.csv", index=False)
print(f"\nWritten: mlp_decomposition_per_config.csv ({len(df_per)} rows)")

# Sanity: 모두 finite
for col in ["raw_acc_err_mlp", "residual_NN_error_mlp",
            "raw_acc_err_deeponet", "residual_NN_error_deeponet"]:
    n_fin = int(df_per[col].notna().sum())
    assert n_fin == len(df_per), f"{col}: {len(df_per) - n_fin} NaN"
print("  All 27 rows finite (raw_acc, residual for MLP and DeepONet)")

# ─── Phase 4: vs_deeponet CSV (subset of per-config columns) ─────────────────
vs_cols = ["config_id", "model", "block", "payoff",
           "Delta", "b_Delta_mlp", "b_Delta_deeponet",
           "sigma_delta_mlp", "sigma_delta_deeponet",
           "empirical_mr_mlp", "empirical_mr_deeponet",
           "raw_acc_err_mlp", "raw_acc_err_deeponet",
           "residual_NN_error_mlp", "residual_NN_error_deeponet",
           "explained_fraction_mlp", "explained_fraction_deeponet"]
df_vs = df_per[vs_cols].copy()
df_vs.to_csv(OUTDIR / "mlp_decomposition_vs_deeponet.csv", index=False)
print(f"Written: mlp_decomposition_vs_deeponet.csv ({len(df_vs)} rows)")

# ─── Summary by (model, block) ───────────────────────────────────────────────
def agg_block(g):
    return pd.Series(dict(
        n_configs=len(g),
        raw_acc_err_mlp_median=g["raw_acc_err_mlp"].median(),
        raw_acc_err_mlp_q75=g["raw_acc_err_mlp"].quantile(0.75),
        residual_NN_error_mlp_median=g["residual_NN_error_mlp"].median(),
        residual_NN_error_mlp_q75=g["residual_NN_error_mlp"].quantile(0.75),
        residual_NN_error_mlp_abs_median=g["residual_NN_error_mlp"].abs().median(),
        raw_acc_err_deeponet_median=g["raw_acc_err_deeponet"].median(),
        raw_acc_err_deeponet_q75=g["raw_acc_err_deeponet"].quantile(0.75),
        residual_NN_error_deeponet_median=g["residual_NN_error_deeponet"].median(),
        residual_NN_error_deeponet_q75=g["residual_NN_error_deeponet"].quantile(0.75),
        residual_NN_error_deeponet_abs_median=g["residual_NN_error_deeponet"].abs().median(),
        explained_fraction_mlp_median=g["explained_fraction_mlp"].median(),
        explained_fraction_deeponet_median=g["explained_fraction_deeponet"].median(),
    ))

df_sum = (df_per.groupby(["model", "block"], as_index=False)
                .apply(agg_block, include_groups=False)
                .reset_index(drop=True))
# Add overall (model only)
df_sum_overall = (df_per.groupby(["model"], as_index=False)
                        .apply(agg_block, include_groups=False)
                        .reset_index(drop=True))
df_sum_overall.insert(1, "block", "OVERALL")

df_sum_full = pd.concat([df_sum, df_sum_overall], ignore_index=True)
df_sum_full.to_csv(OUTDIR / "mlp_decomposition_summary.csv", index=False)
print(f"Written: mlp_decomposition_summary.csv ({len(df_sum_full)} rows)")

# ─── Figure: residual scatter (MLP vs DeepONet, 2-panel by model) ────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
COL = {"heston": "steelblue", "kou": "darkorange"}
TITLE = {"heston": "Heston (Exp01, n=15)", "kou": "Kou (stageB, n=12)"}
for ax, mn in zip(axes, ["heston", "kou"]):
    sub = df_per[df_per["model"] == mn]
    # signed residuals as percentages
    x = sub["residual_NN_error_deeponet"] * 100
    y = sub["residual_NN_error_mlp"]      * 100
    ax.scatter(x, y, s=44, color=COL[mn], alpha=0.85, zorder=4,
               edgecolors="white", linewidths=0.4)
    # ±1-4% bands (signed)
    for lo, hi, lbl in [(-4, -1, "−4..−1%"), (1, 4, "1..4%")]:
        ax.axhspan(lo, hi, alpha=0.07, color="green")
        ax.axvspan(lo, hi, alpha=0.07, color="green")
    # y=x diagonal
    rng = [-7, 7]
    ax.plot(rng, rng, "k--", lw=0.8, alpha=0.5, label="y=x")
    ax.axhline(0, color="black", lw=0.4, ls=":", alpha=0.4)
    ax.axvline(0, color="black", lw=0.4, ls=":", alpha=0.4)
    ax.set_xlim(rng); ax.set_ylim(rng)
    ax.set_xlabel("DeepONet residual NN error  [%]", fontsize=9)
    ax.set_ylabel("MLP residual NN error  [%]",       fontsize=9)
    ax.set_title(TITLE[mn], fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.tick_params(labelsize=8)
fig.suptitle("Residual NN error: MLP vs DeepONet (signed, |Delta|-normalised)",
             fontsize=11)
fig.tight_layout()
fig.savefig(OUTDIR / "fig_mlp_vs_deeponet_residual.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Written: fig_mlp_vs_deeponet_residual.png")

# ─── stdout summary ──────────────────────────────────────────────────────────
print(f"\n{'='*72}\nSUMMARY\n{'='*72}\n")

print("Row counts:")
print(f"  mlp_thm35 total: 39  →  decomposition-eligible: 27 (Exp01 15 + Kou stageB 12)")
print(f"  excluded: 12 (b2-heston-A..D × 3 payoffs, no Lewis Delta)")

print("\nPer (model, block) summary:")
print(df_sum_full.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n--- Key one-liners ---")
for mn, label in [("heston", "Heston Exp01 (n=15)"), ("kou", "Kou stageB (n=12)")]:
    sub = df_per[df_per["model"] == mn]
    mlp_med  = sub["residual_NN_error_mlp"].median() * 100
    dp_med   = sub["residual_NN_error_deeponet"].median() * 100
    mlp_abs  = sub["residual_NN_error_mlp"].abs().median() * 100
    dp_abs   = sub["residual_NN_error_deeponet"].abs().median() * 100
    raw_mlp  = sub["raw_acc_err_mlp"].median() * 100
    raw_dp   = sub["raw_acc_err_deeponet"].median() * 100
    print(f"\n  {label}:")
    print(f"    raw_acc_err median:           MLP={raw_mlp:6.2f}%  DeepONet={raw_dp:6.2f}%")
    print(f"    residual_NN_error median:     MLP={mlp_med:+6.2f}%  DeepONet={dp_med:+6.2f}%  (signed)")
    print(f"    |residual_NN_error| median:   MLP={mlp_abs:6.2f}%  DeepONet={dp_abs:6.2f}%  (abs)")

print("\nQ2 check — '1-4% ballpark':")
for mn in ["heston", "kou"]:
    sub = df_per[df_per["model"] == mn]
    mlp_abs = sub["residual_NN_error_mlp"].abs().median() * 100
    in_band = bool(1.0 <= mlp_abs <= 4.0)
    print(f"  {mn}: |MLP residual| median = {mlp_abs:.2f}%  → in [1,4]%: {in_band}")

print("\n=== COMPLETE ===")
