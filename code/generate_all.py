"""
NeuMoR Paper — Figure & Table Generator
Generates all 15 artifacts for the paper.
Run from project root:
    python NeuMoR/output/paper_artifacts/generate_all.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = Path(__file__).parent
TABLES = OUT_ROOT / "tables"
FIGS   = OUT_ROOT / "figures"
EXP    = PROJECT_ROOT / "NeuMoR" / "output"
BTC_5M = PROJECT_ROOT / "TNO" / "data" / "BTC" / "BTC_nsde_5m_full.csv"

SMOOTH_PAYOFFS = {"ATM", "OTM_K110", "OTM_K125"}

# ── Matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 100,
})
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
PAIR_COLORS = {p: COLORS[i] for i, p in enumerate(["A", "B", "C", "D", "E"])}
AXIS_COLORS = {a: COLORS[i] for i, a in enumerate(["v", "kappa", "omega", "xi", "rho"])}
AXIS_LABELS = {"v": r"$v$ (variance)", "kappa": r"$\kappa$ (mean rev)",
               "omega": r"$\omega$ (long var)", "xi": r"$\xi$ (vol-vol)",
               "rho": r"$\rho_{SV}$ (leverage)"}

# ── LaTeX helpers ─────────────────────────────────────────────────────────────
def to_latex(df: pd.DataFrame, caption: str, label: str,
             float_fmt: str = "{:.4f}", index: bool = False) -> str:
    n_cols = len(df.columns)
    col_fmt = "l" + "r" * (n_cols - 1) if not index else "l" * (n_cols + 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
    ]
    header = " & ".join(str(c) for c in df.columns) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(float_fmt.format(v))
            else:
                cells.append(str(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def save_table(df: pd.DataFrame, name: str, caption: str, label: str,
               float_fmt: str = "{:.4f}", index: bool = False):
    csv_path = TABLES / f"{name}.csv"
    tex_path = TABLES / f"{name}.tex"
    df.to_csv(csv_path, index=index)
    tex = to_latex(df, caption, label, float_fmt=float_fmt, index=index)
    tex_path.write_text(tex)
    print(f"  Saved {csv_path.name} + {tex_path.name}")


def save_fig(fig: plt.Figure, name: str):
    png_path = FIGS / f"{name}.png"
    pdf_path = FIGS / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {png_path.name} + {pdf_path.name}")


FLAGS = []  # collect issues found


def flag(msg: str):
    print(f"  [FLAG] {msg}")
    FLAGS.append(msg)


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 4.1 — Per-axis α values (smooth payoffs only)
# ══════════════════════════════════════════════════════════════════════════════
def make_table_4_1():
    print("\n[Table 4.1] Per-axis α values")
    df = pd.read_csv(EXP / "exp05" / "alpha_per_axis.csv")
    df_smooth = df[df["payoff"].isin(SMOOTH_PAYOFFS)]

    rows = []
    for axis in ["v", "kappa", "omega", "xi", "rho"]:
        sub = df_smooth[df_smooth["axis"] == axis]
        alphas = sub["alpha"].values
        r2s = sub["r2"].values
        mean_a = np.mean(alphas)
        std_a  = np.std(alphas, ddof=1)
        se_a   = std_a / np.sqrt(len(alphas))
        # 95% CI using t(2) = 4.303
        ci_lo  = mean_a - 4.303 * se_a
        ci_hi  = mean_a + 4.303 * se_a
        mean_r2 = np.mean(r2s)
        rows.append({
            "Axis": AXIS_LABELS[axis].replace("$", ""),
            "α (mean)": round(mean_a, 3),
            "95% CI": f"[{ci_lo:.3f}, {ci_hi:.3f}]",
            "R² (mean)": round(mean_r2, 3),
            "Smooth payoffs": len(alphas),
        })

    out = pd.DataFrame(rows)

    # Flag: R² range check
    min_r2 = df_smooth["r2"].min()
    max_r2 = df_smooth["r2"].max()
    if min_r2 < 0.92:
        flag(f"Table 4.1: min R²={min_r2:.3f} < 0.92 (skeleton claimed 0.97+). "
             f"Range is [{min_r2:.3f}, {max_r2:.3f}].")

    save_table(out, "table_4_1",
               caption=r"Per-axis power-law exponents $\alpha_a$ for $(1-\rho) \sim C_a |\Delta\lambda_a|^{\alpha_a}$. "
                       r"Fit to smooth payoffs (ATM, OTM 110\%, OTM 125\%) across 5 parameter pairs. "
                       r"95\% CI uses $t_{(2)}$ distribution.",
               label="tab:alpha_per_axis",
               float_fmt="{:.3f}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4.1 — (1−ρ) vs Δλ per axis (log-log)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig_4_1():
    print("\n[Figure 4.1] (1-ρ) vs Δλ per axis")
    df_multi = pd.read_csv(EXP / "exp05" / "theorem1prime_multiaxis.csv")
    df_alpha = pd.read_csv(EXP / "exp05" / "anisotropic_parameters.csv")
    df_smooth = df_multi[df_multi["payoff"] == "ATM"]  # use ATM as representative

    axes_list = ["v", "kappa", "omega", "xi", "rho"]
    fig, axs = plt.subplots(2, 3, figsize=(12, 7))
    axs_flat = axs.flatten()

    for idx, axis in enumerate(axes_list):
        ax = axs_flat[idx]
        sub = df_smooth[df_smooth["axis"] == axis].copy()
        sub = sub[sub["rho"] < 0.9999]  # exclude degenerate

        deltas = sub["delta"].abs().values
        rho_vals = sub["rho"].values
        one_minus_rho = 1.0 - rho_vals

        # Get fit params
        fits = df_alpha[(df_alpha["axis"] == axis) & (df_alpha["payoff"] == "ATM")]
        alpha_fit = fits["alpha"].values[0]
        C_fit     = fits["C"].values[0]

        # Filter positive 1-rho
        mask = one_minus_rho > 0
        deltas = deltas[mask]
        one_minus_rho = one_minus_rho[mask]

        ax.loglog(deltas, one_minus_rho, "o", color=AXIS_COLORS[axis],
                  ms=6, label="data")

        # Fit line
        d_range = np.logspace(np.log10(deltas.min()), np.log10(deltas.max()), 50)
        ax.loglog(d_range, C_fit * d_range**alpha_fit, "-", color="k",
                  lw=1.5, label=rf"$\alpha={alpha_fit:.2f}$")

        ax.set_title(AXIS_LABELS[axis])
        ax.set_xlabel(r"$|\Delta\lambda_a|$")
        ax.set_ylabel(r"$1 - \rho$")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    # Hide the 6th panel
    axs_flat[-1].set_visible(False)
    fig.suptitle(r"Anisotropic Correlation Decay: $(1-\rho) \sim C_a |\Delta\lambda_a|^{\alpha_a}$",
                 y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig_4_1")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4.2 — Kernel eigenvalue spectrum
# ══════════════════════════════════════════════════════════════════════════════
def make_fig_4_2():
    print("\n[Figure 4.2] Kernel eigenvalue spectrum")
    df = pd.read_csv(EXP / "exp04" / "kernel_eigenvalues.csv")
    eigs = df["eigenvalue"].values

    # Effective rank: r_eff = (Σλ)² / Σλ²
    r_eff = (eigs.sum())**2 / (eigs**2).sum()
    # Participation ratio: same formula
    pr    = r_eff

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(df["rank"].values, eigs, "b-o", ms=3, lw=1.5,
                label=f"Eigenvalues (N=1000 seeds)")
    ax.axvline(r_eff, color="r", ls="--", lw=1.5,
               label=rf"$r_{{\rm eff}} = {r_eff:.1f}$")
    ax.set_xlabel("Eigenvalue rank $k$")
    ax.set_ylabel("Eigenvalue $\lambda_k$")
    ax.set_title("Noise Kernel $\\hat{K}$ Eigenvalue Spectrum (Pair A, ATM)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    save_fig(fig, "fig_4_2")

    print(f"    r_eff = {r_eff:.1f}, PR = {pr:.1f}, N_eigs = {len(eigs)}")
    if r_eff < 10:
        flag(f"Figure 4.2: effective rank r_eff={r_eff:.1f} is quite low — kernel highly structured.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4.3 — g* payoff shape vs ATM
# ══════════════════════════════════════════════════════════════════════════════
def make_fig_4_3():
    print("\n[Figure 4.3] g* payoff shape vs ATM")
    df = pd.read_csv(EXP / "exp10" / "payoffs.csv")

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for i, (pair_label, pair_title) in enumerate([("A", "Pair A ($\\Delta v=0.001$)"),
                                                    ("D", "Pair D ($\\Delta v=0.030$)")]):
        sub = df[df["pair"] == pair_label].copy()
        y   = sub["y"].values
        g_star = sub["g_A_tikh"].values

        # ATM call payoff: max(e^y - 1, 0)
        atm = np.maximum(np.exp(y) - 1.0, 0.0)

        # Normalize both for shape comparison
        g_star_norm = g_star / (np.abs(g_star).max() + 1e-15)
        atm_norm    = atm / (atm.max() + 1e-15)

        ax = axs[i]
        ax.plot(y, atm_norm, "-", color=COLORS[1], lw=2, label="ATM call (norm.)")
        ax.plot(y, g_star_norm, "-", color=COLORS[0], lw=2, label=r"$g^*$ (Tikhonov, norm.)")
        ax.set_xlabel("Log-return $y$")
        ax.set_ylabel("Payoff magnitude (normalized)")
        ax.set_title(pair_title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-3, 3)

    fig.suptitle(r"Optimal Discriminating Payoff $g^*$ vs ATM Benchmark")
    fig.tight_layout()
    save_fig(fig, "fig_4_3")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 5.1 — Theorem 3.1 validation summary
# ══════════════════════════════════════════════════════════════════════════════
def make_table_5_1():
    print("\n[Table 5.1] Theorem 3.1 validation summary")

    records = []

    # --- Exp01: in-sample ---
    df = pd.read_csv(EXP / "exp01" / "theorem_1_prime.csv")
    df_s = df[df["payoff"].isin(SMOOTH_PAYOFFS)]
    ratios = df_s["ratio_prime"].dropna().values
    records.append(dict(Experiment="Exp 01 (in-sample)",
                        Type="Pairs A-E × 3 smooth payoffs",
                        Configs=len(ratios),
                        Ratio_min=ratios.min(), Ratio_max=ratios.max(),
                        All_pass=all(0.7 <= r <= 1.3 for r in ratios)))

    # --- Exp02: OoS cross-val ---
    df = pd.read_csv(EXP / "exp02" / "cv_test.csv")
    df_s = df[df["payoff"].isin(SMOOTH_PAYOFFS)]
    ratios = df_s["ratio_cv"].dropna().values
    records.append(dict(Experiment="Exp 02 (OoS cross-val)",
                        Type="Train/test seed split",
                        Configs=len(ratios),
                        Ratio_min=ratios.min(), Ratio_max=ratios.max(),
                        All_pass=all(0.7 <= r <= 1.3 for r in ratios)))

    # --- Exp03: cross-protocol (AA protocol only = same conditions) ---
    df = pd.read_csv(EXP / "exp03" / "jensen_isolation_testA.csv")
    df_s = df[df["payoff"].isin(SMOOTH_PAYOFFS)]
    ratios = df_s["ratio"].dropna().values
    records.append(dict(Experiment="Exp 03 (cross-protocol)",
                        Type="6 protocol pairs × Pairs A-E",
                        Configs=len(ratios),
                        Ratio_min=ratios.min(), Ratio_max=ratios.max(),
                        All_pass=all(0.7 <= r <= 1.3 for r in ratios)))

    # --- Exp05 multi-axis ---
    df = pd.read_csv(EXP / "exp05" / "theorem1prime_multiaxis.csv")
    df_s = df[df["payoff"].isin(SMOOTH_PAYOFFS)]
    ratios = df_s["ratio"].dropna().values
    records.append(dict(Experiment="Exp 05 (multi-axis)",
                        Type="5 axes × 5 pairs × 3 smooth payoffs",
                        Configs=len(ratios),
                        Ratio_min=ratios.min(), Ratio_max=ratios.max(),
                        All_pass=all(0.7 <= r <= 1.3 for r in ratios)))

    # --- Exp05 diagonal ---
    df = pd.read_csv(EXP / "exp05" / "diagonal_sweeps.csv")
    df_s = df[df["payoff"].isin(SMOOTH_PAYOFFS)]
    ratios = df_s["ratio"].dropna().values
    records.append(dict(Experiment="Exp 05 (diagonal sweep)",
                        Type="Multi-axis diagonal paths",
                        Configs=len(ratios),
                        Ratio_min=ratios.min(), Ratio_max=ratios.max(),
                        All_pass=all(0.7 <= r <= 1.3 for r in ratios)))

    # --- Exp06 edge cases ---
    df = pd.read_csv(EXP / "exp06" / "edge_case_distributions.csv")
    df_s = df[df["payoff"].isin(SMOOTH_PAYOFFS)]
    ratios = df_s["ratio"].dropna().values
    records.append(dict(Experiment="Exp 06 (edge cases s→0)",
                        Type="Special s-value configurations",
                        Configs=len(ratios),
                        Ratio_min=ratios.min(), Ratio_max=ratios.max(),
                        All_pass=all(0.7 <= r <= 1.3 for r in ratios)))

    # --- Exp09 real BTC ---
    df = pd.read_csv(EXP / "exp09" / "theorem1prime_realmarket.csv")
    df_s = df[df["payoff"].isin(SMOOTH_PAYOFFS)]
    # Use ratio_lam1 as the main validation ratio
    ratios = df_s["ratio_lam1"].dropna().values
    records.append(dict(Experiment="Exp 09 (real BTC)",
                        Type="4 calibration windows × 3 smooth payoffs",
                        Configs=len(ratios),
                        Ratio_min=ratios.min(), Ratio_max=ratios.max(),
                        All_pass=all(0.7 <= r <= 1.3 for r in ratios)))

    out_df = pd.DataFrame(records)
    total_configs = out_df["Configs"].sum()
    total_pass    = out_df["All_pass"].all()
    print(f"    Total smooth configs: {total_configs}")
    print(f"    All pass [0.7, 1.3]: {total_pass}")

    # Check ratios
    all_ratios_combined = []
    for rec in records:
        all_ratios_combined.extend([rec["Ratio_min"], rec["Ratio_max"]])
    global_min = min(all_ratios_combined)
    global_max = max(all_ratios_combined)
    if global_min < 0.7 or global_max > 1.3:
        flag(f"Table 5.1: Some ratios outside [0.7, 1.3]. Range: [{global_min:.4f}, {global_max:.4f}]")

    # Format for display
    display_df = out_df.copy()
    display_df["Ratio range"] = display_df.apply(
        lambda r: f"[{r['Ratio_min']:.3f}, {r['Ratio_max']:.3f}]", axis=1)
    display_df["Pass [0.7,1.3]"] = display_df["All_pass"].map({True: "✓", False: "✗"})
    final = display_df[["Experiment", "Type", "Configs", "Ratio range", "Pass [0.7,1.3]"]]

    save_table(final, "table_5_1",
               caption=r"Theorem~3.1 validation across all smooth-payoff configurations. "
                       r"Ratio $= \mathbb{E}_\omega[|\hat\Delta|]_{\rm empirical} / \mathbb{E}_\omega[|\hat\Delta|]_{\rm theory}$. "
                       r"All ratios in $[0.7, 1.3]$ except where noted.",
               label="tab:thm31_validation")
    return out_df


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 5.2 — Cross-protocol correlation isolation (Theorem 4.1)
# ══════════════════════════════════════════════════════════════════════════════
def make_table_5_2():
    print("\n[Table 5.2] Cross-protocol correlation isolation")
    df = pd.read_csv(EXP / "exp03" / "correlation_by_protocol.csv")
    # ATM payoff only, pairs A and D (representative small & medium regime)
    sub = df[(df["payoff"] == "ATM") & (df["pair"].isin(["A", "D"]))].copy()

    # Compute sigma_delta and |s|
    sub["sigma_delta_fmt"] = sub["sigma_delta"].apply(lambda x: f"{x:.4e}")
    sub["|s|"] = sub["s"].abs().round(2)

    proto_labels = {"AA": "Same arch + loss", "AB1": "Same arch, diff loss",
                    "AB2": "Same arch, MSE only", "AB3": "Same arch + λ noise",
                    "B1B1": "Same (loss 1)", "B2B2": "Same (loss 2)"}
    sub["Protocol"] = sub["protocol_pair"].map(proto_labels)
    sub["Pair"] = sub["pair"]
    sub["ρ"] = sub["rho"].round(4)

    pivot = sub.pivot_table(index="Protocol", columns="Pair",
                            values="ρ", aggfunc="first").reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={"A": "ρ (Pair A)", "D": "ρ (Pair D)"})

    # Add protocol order
    order = ["Same arch + loss", "Same arch, diff loss",
             "Same arch, MSE only", "Same arch + λ noise",
             "Same (loss 1)", "Same (loss 2)"]
    pivot["_ord"] = pivot["Protocol"].map({v: i for i, v in enumerate(order)})
    pivot = pivot.sort_values("_ord").drop("_ord", axis=1)

    save_table(pivot, "table_5_2",
               caption=r"Empirical correlation $\rho$ between $\hat V_1$ and $\hat V_2$ under different training "
                       r"protocols (ATM payoff). High $\rho$ (same architecture/loss) reduces measurement bias "
                       r"per Theorem~4.1. Columns: Pair A ($\Delta v=0.001$), Pair D ($\Delta v=0.030$).",
               label="tab:protocol_correlation",
               float_fmt="{:.4f}")

    # Flag if any protocol gives unexpected very negative rho
    neg = df[df["rho"] < -0.5]
    if len(neg) > 0:
        flag(f"Table 5.2: {len(neg)} (protocol, pair, payoff) combos have ρ < -0.5 (max neg: {neg['rho'].min():.3f})")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5.1 — Theoretical vs empirical bias scatter
# ══════════════════════════════════════════════════════════════════════════════
def make_fig_5_1():
    print("\n[Figure 5.1] Theoretical vs empirical bias scatter")

    all_dfs = []

    def load_and_tag(path, tag, emp_col, theory_col, payoff_col=None,
                     payoff_filter=None):
        df = pd.read_csv(path)
        if payoff_filter is not None and payoff_col:
            df = df[df[payoff_col].isin(payoff_filter)]
        df = df.copy()
        df["_emp"]    = df[emp_col].abs()
        df["_theory"] = df[theory_col].abs()
        df["_tag"]    = tag
        keep = df[["_emp", "_theory", "_tag"]].dropna()
        keep = keep[(keep["_emp"] > 0) & (keep["_theory"] > 0)]
        return keep

    # exp01
    all_dfs.append(load_and_tag(EXP / "exp01" / "theorem_1_prime.csv",
        "Exp01 (pairs A-E)", "bias_empirical", "bias_theory_prime",
        "payoff", SMOOTH_PAYOFFS))

    # exp02
    all_dfs.append(load_and_tag(EXP / "exp02" / "cv_test.csv",
        "Exp02 (OoS CV)", "E_abs_test", "E_abs_pred",
        "payoff", SMOOTH_PAYOFFS))

    # exp05 multiaxis
    all_dfs.append(load_and_tag(EXP / "exp05" / "theorem1prime_multiaxis.csv",
        "Exp05 (multi-axis)", "bias_emp", "bias_pred",
        "payoff", SMOOTH_PAYOFFS))

    # exp05 diagonal
    all_dfs.append(load_and_tag(EXP / "exp05" / "diagonal_sweeps.csv",
        "Exp05 (diagonal)", "bias_emp", "bias_pred",
        "payoff", SMOOTH_PAYOFFS))

    # exp06
    all_dfs.append(load_and_tag(EXP / "exp06" / "edge_case_distributions.csv",
        "Exp06 (edge cases)", "bias_emp", "bias_pred",
        "payoff", SMOOTH_PAYOFFS))

    # exp09
    all_dfs.append(load_and_tag(EXP / "exp09" / "theorem1prime_realmarket.csv",
        "Exp09 (real BTC)", "mu1", "ratio_lam1",
        "payoff", SMOOTH_PAYOFFS))
    # exp09 ratio_lam1 is already ratio, not absolute bias; skip for scatter
    all_dfs = all_dfs[:-1]  # drop exp09 which has different format

    combined = pd.concat(all_dfs, ignore_index=True)
    tags = combined["_tag"].unique()

    fig, ax = plt.subplots(figsize=(7, 6))
    markers = ["o", "s", "^", "D", "v", "P"]
    for i, tag in enumerate(tags):
        sub = combined[combined["_tag"] == tag]
        ax.loglog(sub["_theory"], sub["_emp"], markers[i % len(markers)],
                  color=COLORS[i % len(COLORS)], ms=4, alpha=0.7,
                  label=f"{tag} (n={len(sub)})")

    # y=x line
    all_vals = combined[["_emp", "_theory"]].values.flatten()
    all_vals = all_vals[all_vals > 0]
    v_min, v_max = all_vals.min(), all_vals.max()
    ax.loglog([v_min, v_max], [v_min, v_max], "k-", lw=2, label=r"$y = x$")
    ax.fill_between([v_min, v_max], [0.7 * v_min, 0.7 * v_max],
                    [1.3 * v_min, 1.3 * v_max], alpha=0.1, color="gray",
                    label="±30% band")

    # Regression line
    log_t = np.log(combined["_theory"].values)
    log_e = np.log(combined["_emp"].values)
    slope, intercept, r_val, _, _ = stats.linregress(log_t, log_e)
    x_line = np.logspace(np.log10(v_min), np.log10(v_max), 50)
    ax.loglog(x_line, np.exp(intercept) * x_line**slope, "r--", lw=1.5,
              label=rf"Regression ($R^2={r_val**2:.3f}$, slope={slope:.3f})")

    ax.set_xlabel(r"Theoretical bias $|\mathbb{E}[|\hat\Delta|] - |\Delta||_{\rm theory}$")
    ax.set_ylabel(r"Empirical bias $|\mathbb{E}[|\hat\Delta|] - |\Delta||_{\rm emp}$")
    ax.set_title("Theorem 3.1 Validation: Theory vs Empirical Bias")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    save_fig(fig, "fig_5_1")

    total_n = len(combined)
    in_band = ((combined["_emp"] / combined["_theory"]).between(0.7, 1.3)).sum()
    print(f"    N={total_n}, in ±30% band: {in_band} ({100*in_band/total_n:.1f}%)")
    if in_band / total_n < 0.90:
        flag(f"Figure 5.1: only {100*in_band/total_n:.1f}% of points in ±30% band (expected >90%)")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5.2 — SNR convergence (g* vs ATM, N dependency)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig_5_2():
    print("\n[Figure 5.2] SNR convergence")
    df = pd.read_csv(EXP / "exp07_6" / "snr_convergence_extended.csv")
    fits = pd.read_csv(EXP / "exp07_6" / "convergence_fits_extended.csv")

    # Use tau_0.01 held_out
    sub = df[(df["scheme"] == "tau_0.01") & (df["eval_type"] == "held_out")].copy()

    fig, ax = plt.subplots(figsize=(8, 5))
    for pair in ["A", "B", "C", "D", "E"]:
        pair_sub = sub[sub["pair"] == pair].sort_values("N")
        N_vals  = pair_sub["N"].values
        snr_vals = pair_sub["SNR_ratio_vs_ATM"].values
        ax.plot(N_vals, snr_vals, "o-", color=PAIR_COLORS[pair], lw=1.5,
                ms=5, label=f"Pair {pair}")

        # Asymptotic line from fits
        fit_row = fits[(fits["pair"] == pair) & (fits["scheme"] == "tau_0.01")]
        if len(fit_row) > 0 and fit_row["best_model"].values[0] in ["H3", "H1"]:
            snr_inf = fit_row["SNR_inf_best"].values[0]
            ax.axhline(snr_inf, ls="--", color=PAIR_COLORS[pair], lw=1, alpha=0.5)

    ax.axhline(1.0, ls="-", color="k", lw=1, alpha=0.4, label="ATM baseline")
    ax.set_xscale("log")
    ax.set_xlabel("Seed ensemble size $N$")
    ax.set_ylabel(r"SNR ratio: $\hat{s}(g^*) / \hat{s}(\text{ATM})$")
    ax.set_title(r"SNR Convergence: $g^*_\tau$ vs ATM (held-out evaluation, $\tau=0.01\lambda_{\max}$)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Print asymptotic values
    for pair in ["A", "B", "C", "D", "E"]:
        fit_row = fits[(fits["pair"] == pair) & (fits["scheme"] == "tau_0.01")]
        if len(fit_row) > 0:
            snr_inf = fit_row["SNR_inf_best"].values[0]
            print(f"    Pair {pair} asymptotic SNR_∞ = {snr_inf:.3f}×")

    fig.tight_layout()
    save_fig(fig, "fig_5_2")

    # Check claims
    fit_main = fits[fits["scheme"] == "tau_0.01"]
    inf_vals = fit_main["SNR_inf_best"].dropna().values
    print(f"    SNR_∞ range: [{inf_vals.min():.3f}, {inf_vals.max():.3f}]")
    if inf_vals.min() < 1.5 or inf_vals.max() > 3.5:
        flag(f"Figure 5.2: SNR_∞ range [{inf_vals.min():.3f}, {inf_vals.max():.3f}] "
             f"deviates from skeleton claim [1.8, 2.9]")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 5.3 — Gaussian assumption tests (Assumption A1)
# ══════════════════════════════════════════════════════════════════════════════
def make_table_5_3():
    print("\n[Table 5.3] Gaussian assumption tests")
    df = pd.read_csv(EXP / "exp06" / "sub_gaussian_bounds.csv")
    # All 20 configs (5 pairs × 4 payoffs, including DIGITAL for completeness)
    ratios = df["ratio_sg_vs_gauss"].values
    print(f"    σ_sg/σ_gauss: mean={ratios.mean():.4f}, std={ratios.std():.4f}, "
          f"range=[{ratios.min():.4f}, {ratios.max():.4f}]")

    # Normality tests from exp02
    df_norm = pd.read_csv(EXP / "exp02" / "normality_tests.csv")
    n_tests = len(df_norm)
    n_reject_ks  = (df_norm["ks_p"] < 0.05).sum()
    n_reject_sw  = (df_norm["ad_reject5"].sum()) if "ad_reject5" in df_norm.columns else \
                   (df_norm["shapiro_p"] < 0.05).sum()
    mean_r2 = 0.9994  # from skeleton; not directly in this csv

    rows = [
        {"Test": "KS normality (exp02)", "Statistic": "KS $p$-value",
         "N tests": n_tests, "Result": f"{n_reject_ks}/{n_tests} rejections at 5\\%"},
        {"Test": "Sub-Gaussian ratio (exp06)", "Statistic": r"$\sigma_{sg}/\sigma_{gauss}$",
         "N tests": len(df), "Result": f"{ratios.mean():.4f} ± {ratios.std():.4f}"},
        {"Test": "AD test (exp02)", "Statistic": "AD statistic",
         "N tests": n_tests, "Result": f"{int(n_reject_sw)}/{n_tests} rejections at 5\\%"},
    ]
    # Edge cases
    df_edge = pd.read_csv(EXP / "exp06" / "edge_case_distributions.csv")
    df_edge_smooth = df_edge[df_edge["payoff"].isin(SMOOTH_PAYOFFS)]
    n_edge = len(df_edge_smooth)
    n_rej_edge = df_edge_smooth["ad_reject"].sum()
    rows.append({"Test": "AD test (exp06 edge)", "Statistic": "AD statistic",
                 "N tests": n_edge, "Result": f"{int(n_rej_edge)}/{n_edge} rejections at 5\\%"})

    out = pd.DataFrame(rows)
    save_table(out, "table_5_3",
               caption=r"Empirical validation of Assumption~A1 (Gaussian noise). "
                       r"AD = Anderson-Darling; sub-Gaussian ratio $<1.05$ confirms "
                       r"near-Gaussian tails.",
               label="tab:gaussian_tests")

    if n_reject_ks > 5:
        flag(f"Table 5.3: KS test rejects {n_reject_ks}/{n_tests} — Gaussianity weaker than expected")
    if ratios.max() > 1.05:
        flag(f"Table 5.3: σ_sg/σ_gauss max = {ratios.max():.4f} > 1.05 (Pair E effects?)")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 5.4 — BTC calibration parameters
# ══════════════════════════════════════════════════════════════════════════════
def make_table_5_4():
    print("\n[Table 5.4] BTC calibration parameters")
    df = pd.read_csv(EXP / "exp12" / "calibrations.csv")

    # Check raw vs clipped from exp09
    df09 = pd.read_csv(EXP / "exp09" / "calibrations.csv")
    print(f"    exp09 raw kappa range: [{df09['raw_kappa'].min():.0f}, {df09['raw_kappa'].max():.0f}]")
    print(f"    exp09 raw xi range:    [{df09['raw_xi'].min():.2f}, {df09['raw_xi'].max():.2f}]")

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "Scenario": row["scenario"],
            "Model": row["model"],
            "Description": row["desc"],
            "$v$": round(row["v"], 4),
            r"$\kappa$": row["kappa"],
            r"$\omega$": round(row["omega"], 4),
            r"$\xi$": row["xi"],
            r"$\rho_{SV}$": round(row["rho"], 3),
            "Clipped": "Yes" if row["clipped"] else "No",
            "Regime": row["regime"],
        })

    out = pd.DataFrame(rows)
    save_table(out, "table_5_4",
               caption=r"BTC scenario calibration parameters (Exp~09/12). "
                       r"$\kappa$ and $\xi$ clipped to training support boundary "
                       r"[$\kappa \leq 2.0$, $\xi \leq 0.6$]. "
                       r"All scenarios have $T=1$ year.",
               label="tab:btc_calibration",
               float_fmt="{:.4f}")

    if all(row["clipped"] for _, row in df.iterrows()):
        flag("Table 5.4: ALL BTC scenarios are clipped — all κ and ξ exceed training range. "
             "This is a significant out-of-distribution concern for all real-world results.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5.3 — BTC realized variance with calibration windows
# ══════════════════════════════════════════════════════════════════════════════
def make_fig_5_3():
    print("\n[Figure 5.3] BTC realized variance timeline")
    # Load 5m data
    df = pd.read_csv(BTC_5M)
    df["time"] = pd.to_datetime(df["time_kst"], utc=True)
    df = df.set_index("time").sort_index()

    # Filter 2020-01 to 2024-05
    df = df["2020-01-01":"2024-05-01"]

    # Daily realized variance: sum of squared 5m returns, annualized
    # 5m returns: 288 per day. Annualize × 252.
    daily_rv = df["log_return"].resample("1D").apply(
        lambda r: (r**2).sum() * 252 if len(r) > 10 else np.nan
    ).dropna()

    # 30-day rolling mean for smoothness
    rv_smooth = daily_rv.rolling(30, min_periods=10).mean()

    # Calibration windows from exp09
    windows_09 = {
        "Luna (W1)": ("2022-05-05", "2022-05-20"),
        "FTX (W2)":  ("2022-11-04", "2022-11-19"),
        "ETF app. (W3)": ("2024-01-08", "2024-01-23"),
    }
    # exp12 model pair windows (approximate from descriptions)
    scenario_pairs = {
        "Luna": ("2022-02-01", "2022-08-01", COLORS[0]),
        "FTX":  ("2022-09-01", "2023-01-01", COLORS[1]),
        "ETF":  ("2023-10-01", "2024-04-01", COLORS[2]),
    }
    key_events = {
        "Luna collapse": "2022-05-09",
        "FTX collapse":  "2022-11-11",
        "BTC ETF":       "2024-01-10",
    }

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily_rv.index, daily_rv.values, lw=0.5, color="lightgray", alpha=0.8)
    ax.plot(rv_smooth.index, rv_smooth.values, lw=1.5, color="steelblue",
            label="30-day RV (ann.)")

    # Shade scenario windows
    for name, (t0, t1, color) in scenario_pairs.items():
        ax.axvspan(pd.Timestamp(t0), pd.Timestamp(t1),
                   alpha=0.12, color=color, label=f"{name} window")

    # Key events
    for evt, date in key_events.items():
        ax.axvline(pd.Timestamp(date), ls="--", color="red", lw=1, alpha=0.6)
        ax.text(pd.Timestamp(date), ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 5,
                evt, rotation=90, fontsize=8, color="red", va="top", ha="right")

    ax.set_xlabel("Date")
    ax.set_ylabel("Daily Realized Variance (annualized)")
    ax.set_title("BTC Realized Variance 2020–2024 with Calibration Windows")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.2)
    ax.set_xlim(pd.Timestamp("2020-01-01"), pd.Timestamp("2024-05-01"))
    fig.tight_layout()
    save_fig(fig, "fig_5_3")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 6.1 — Portfolio composition
# ══════════════════════════════════════════════════════════════════════════════
def make_table_6_1():
    print("\n[Table 6.1] Portfolio composition")
    df = pd.read_csv(EXP / "exp12" / "portfolio.csv")

    summary = df.groupby(["type", "subtype"]).agg(
        Count=("pos_id", "count"),
        K_range=("K", lambda x: f"[{x.min():.2f}, {x.max():.2f}]"),
        T_labels=("T_label", lambda x: ", ".join(sorted(x.unique()))),
        Size_range=("size", lambda x: f"[{x.min()}, {x.max()}]"),
    ).reset_index()
    summary.columns = ["Type", "Subtype", "Count", "Strike range",
                       "Maturities", "Position sizes"]

    total_row = pd.DataFrame([{
        "Type": "Total", "Subtype": "", "Count": len(df),
        "Strike range": "", "Maturities": "",
        "Position sizes": f"[{df['size'].min()}, {df['size'].max()}]",
    }])
    out = pd.concat([summary, total_row], ignore_index=True)
    save_table(out, "table_6_1",
               caption=r"Portfolio composition for capital charge case study (Exp~12). "
                       r"Position sizes in number of contracts; notional $\approx \$50M$.",
               label="tab:portfolio_composition")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 6.2 — Capital charge comparison
# ══════════════════════════════════════════════════════════════════════════════
def make_table_6_2():
    print("\n[Table 6.2] Capital charge comparison")
    df = pd.read_csv(EXP / "exp12" / "capital_implications.csv")

    rows = []
    for scenario in ["Luna", "FTX", "ETF"]:
        sub = df[df["scenario"] == scenario]
        atm_row = sub[sub["strategy"] == "Naive ATM"].iloc[0]
        pos_row = sub[sub["strategy"] == "Position-specific"].iloc[0]
        rows.append({
            "Scenario": scenario,
            "Naive ATM ($M)": round(atm_row["capital_charge_M"], 3),
            "Position-specific ($M)": round(pos_row["capital_charge_M"], 3),
            "Saving (\\$M)": round(pos_row["saving_vs_ATM_M"], 3),
            "Saving (\\%)": round(pos_row["pct_saving"], 1),
            "ATM above threshold": "Yes" if atm_row["above_threshold"] else "No",
            "Pos-spec. above threshold": "Yes" if pos_row["above_threshold"] else "No",
        })

    out = pd.DataFrame(rows)
    save_table(out, "table_6_2",
               caption=r"Capital charge comparison across three BTC volatility scenarios. "
                       r"All values in USD millions (notional $\approx \$50M$). "
                       r"``Above threshold'' = model risk flagged for regulatory action.",
               label="tab:capital_charges",
               float_fmt="{:.3f}")

    savings = out["Saving (\\%)"].values
    print(f"    Capital saving range: {savings.min():.1f}% – {savings.max():.1f}%")
    if not (50 <= savings.min() <= 60 and 85 <= savings.max() <= 90):
        flag(f"Table 6.2: Capital saving range [{savings.min():.1f}%, {savings.max():.1f}%] "
             f"deviates from claimed [56%, 86%].")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 6.3 — Decision accuracy
# ══════════════════════════════════════════════════════════════════════════════
def make_table_6_3():
    print("\n[Table 6.3] Decision accuracy")
    df = pd.read_csv(EXP / "exp12" / "decision_analysis.csv")

    rows = []
    for _, row in df.iterrows():
        fp = row["fp_rate"] if pd.notna(row["fp_rate"]) else "—"
        fn = row["fn_rate"] if pd.notna(row["fn_rate"]) else "—"
        rows.append({
            "Scenario": row["scenario"],
            "Strategy": row["strategy"],
            "True state": row["true_decision"],
            "Agree rate": f"{row['agree_rate']:.0%}",
            "FP rate": f"{fp:.0%}" if fp != "—" else "—",
            "FN rate": f"{fn:.0%}" if fn != "—" else "—",
        })
    out = pd.DataFrame(rows)
    save_table(out, "table_6_3",
               caption=r"Decision accuracy: does the strategy correctly identify significant model risk? "
                       r"FP = false positive (alarm when no risk). "
                       r"ETF scenario is the key differentiator: Naive ATM gives 100\% FP rate.",
               label="tab:decision_accuracy")

    # Check ETF result
    etf_atm = df[(df["scenario"] == "ETF") & (df["strategy"] == "Naive ATM")]
    if len(etf_atm) > 0 and etf_atm["fp_rate"].values[0] != 1.0:
        flag("Table 6.3: ETF Naive ATM FP rate != 1.0 — double check.")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6.1 — Capital saving bar chart
# ══════════════════════════════════════════════════════════════════════════════
def make_fig_6_1():
    print("\n[Figure 6.1] Capital saving bar chart")
    df = pd.read_csv(EXP / "exp12" / "capital_implications.csv")

    scenarios = ["Luna", "FTX", "ETF"]
    strategies = ["Naive ATM", "Position-specific"]
    strat_labels = ["Naive ATM", "Pos-specific ($g^*$)"]

    fig, ax = plt.subplots(figsize=(8, 5))
    n_scen   = len(scenarios)
    n_strat  = len(strategies)
    width    = 0.35
    x        = np.arange(n_scen)

    for j, (strat, strat_lbl) in enumerate(zip(strategies, strat_labels)):
        vals = []
        for scen in scenarios:
            row = df[(df["scenario"] == scen) & (df["strategy"] == strat)]
            vals.append(row["capital_charge_M"].values[0])
        bars = ax.bar(x + j * width - width * (n_strat - 1) / 2,
                      vals, width, label=strat_lbl,
                      color=COLORS[j], alpha=0.85)

        # Annotate saving %
        if strat == "Position-specific":
            for xi, (scen, val) in zip(x, zip(scenarios, vals)):
                save_row = df[(df["scenario"] == scen) & (df["strategy"] == strat)]
                pct = save_row["pct_saving"].values[0]
                ax.text(xi + j * width - width * (n_strat - 1) / 2,
                        val + 0.02, f"−{pct:.0f}%",
                        ha="center", va="bottom", fontsize=9, color="darkgreen")

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=12)
    ax.set_ylabel("Capital Charge (\\$M)")
    ax.set_title("Regulatory Capital Charges: Naive ATM vs Position-Specific $g^*$")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    save_fig(fig, "fig_6_1")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("NeuMoR Paper Artifact Generator")
    print("=" * 70)

    # Tables
    make_table_4_1()
    make_table_5_1()
    make_table_5_2()
    make_table_5_3()
    make_table_5_4()
    make_table_6_1()
    make_table_6_2()
    make_table_6_3()

    # Figures
    make_fig_4_1()
    make_fig_4_2()
    make_fig_4_3()
    make_fig_5_1()
    make_fig_5_2()
    make_fig_5_3()
    make_fig_6_1()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nGenerated {len(list(TABLES.glob('*.csv')))} CSV tables")
    print(f"Generated {len(list(TABLES.glob('*.tex')))} LaTeX tables")
    print(f"Generated {len(list(FIGS.glob('*.png')))} PNG figures")
    print(f"Generated {len(list(FIGS.glob('*.pdf')))} PDF figures")

    if FLAGS:
        print(f"\n{'='*70}")
        print(f"FLAGS ({len(FLAGS)} issues found):")
        for i, f in enumerate(FLAGS, 1):
            print(f"  {i}. {f}")
    else:
        print("\nNo flags raised — all values consistent with claims.")

    # Write summary.md
    summary_lines = [
        "# NeuMoR Paper Artifacts — Generation Summary\n",
        f"Generated: 2026-04-21\n",
        "\n## Files Generated\n",
    ]
    for p in sorted(TABLES.glob("*.csv")):
        summary_lines.append(f"- tables/{p.name}")
    for p in sorted(FIGS.glob("*.png")):
        summary_lines.append(f"- figures/{p.name}")
    summary_lines.append("\n## Flags\n")
    if FLAGS:
        for f in FLAGS:
            summary_lines.append(f"- WARN  {f}")
    else:
        summary_lines.append("- None")
    (OUT_ROOT / "summary.md").write_text("\n".join(summary_lines) + "\n")
    print(f"\nSummary written to summary.md")
