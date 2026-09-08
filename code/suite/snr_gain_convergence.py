
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
FIG_DIR = ROOT / "NeuMoR/figures"
PDF_DIR = ROOT / "NeuMoR/pdfs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from NeuMoR.code.neumor_core import (
    build_k_eta, compute_g_star, snr_squared, atm_call, compute_d_hat
)

N_BOOTSTRAP = 1000
RNG_SEED    = 42

CONFIGS = {
    "heston": dict(
        n_total   = 1000,
        hold_lo   = 950,
        hold_hi   = 1000,
        n_cal_vals= [50, 100, 150, 200, 300, 500, 700, 950],
    ),
    "kou": dict(
        n_total   = 200,
        hold_lo   = 150,
        hold_hi   = 200,
        n_cal_vals= [30, 50, 75, 100, 125, 150],
    ),
}

rows       = []
asym_rows  = []
rng        = np.random.default_rng(RNG_SEED)

for model, cfg in CONFIGS.items():
    n_total    = cfg["n_total"]
    hold_lo    = cfg["hold_lo"]
    hold_hi    = cfg["hold_hi"]
    n_cal_vals = cfg["n_cal_vals"]
    n_test     = hold_hi - hold_lo

    print(f"\n{'='*60}")
    print(f"{model.upper()}  —  total={n_total}, test=seeds {hold_lo}-{hold_hi-1} ({n_test} seeds)")

    for pair in ["A", "B", "C", "D"]:

        l1_all = np.load(TMP / f"{model}_pair{pair}_l1_{n_total}.npy").astype(np.float64)
        l2_all = np.load(TMP / f"{model}_pair{pair}_l2_{n_total}.npy").astype(np.float64)

        l1_tst = l1_all[hold_lo:hold_hi]
        l2_tst = l2_all[hold_lo:hold_hi]
        K_tst  = build_k_eta(l1_tst, l2_tst)

        npz    = np.load(UNIFIED / f"{model}_pair{pair}.npz", allow_pickle=True)
        y_grid = npz["y_grid"]
        dy     = float(y_grid[1] - y_grid[0])
        g_atm  = atm_call(y_grid)
        snr2_atm = snr_squared(g_atm, npz["d_hat"], K_tst, dy)

        d_hat_full = npz["d_hat"]

        b2_gain = float(npz["gain_sq_eta_dhat"])

        print(f"\n  pair {pair}  (B.2 asymptotic gain={b2_gain:.4f})")

        for n_cal in n_cal_vals:
            l1_cal = l1_all[:n_cal]
            l2_cal = l2_all[:n_cal]

            K_cal = build_k_eta(l1_cal, l2_cal)
            d_hat_cal = compute_d_hat(l1_cal, l2_cal)
            g_star, *_ = compute_g_star(K_cal, d_hat_cal, dy)
            snr2_star  = snr_squared(g_star, d_hat_full, K_tst, dy)
            gain_pt    = snr2_star / max(snr2_atm, 1e-80)

            bs_gains = np.empty(N_BOOTSTRAP)
            for b in range(N_BOOTSTRAP):
                idx      = rng.choice(n_cal, size=n_cal, replace=True)
                K_bs     = build_k_eta(l1_all[idx], l2_all[idx])
                d_hat_bs = compute_d_hat(l1_all[idx], l2_all[idx])
                g_bs, *_ = compute_g_star(K_bs, d_hat_bs, dy)
                s2       = snr_squared(g_bs, d_hat_full, K_tst, dy)
                bs_gains[b] = s2 / max(snr2_atm, 1e-80)

            ci_lo = float(np.percentile(bs_gains, 2.5))
            ci_hi = float(np.percentile(bs_gains, 97.5))

            print(f"    N={n_cal:4d}: gain={gain_pt:.4f}  CI=[{ci_lo:.4f}, {ci_hi:.4f}]")

            rows.append(dict(
                model=model, pair=pair,
                n_cal=n_cal,
                gain_sq=round(gain_pt, 6),
                gain_lin=round(float(np.sqrt(gain_pt)), 6),
                ci_lo_sq=round(ci_lo, 6),
                ci_hi_sq=round(ci_hi, 6),
                ci_lo_lin=round(float(np.sqrt(max(ci_lo, 0))), 6),
                ci_hi_lin=round(float(np.sqrt(max(ci_hi, 0))), 6),
            ))

        n_max = n_cal_vals[-1]
        row_max = [r for r in rows if r["model"]==model and r["pair"]==pair and r["n_cal"]==n_max]
        gain_asym = row_max[0]["gain_sq"]
        reldiff   = abs(gain_asym - b2_gain) / max(abs(b2_gain), 1e-60)
        status    = "OK" if reldiff < 1e-4 else "WARN"
        print(f"    Asymptotic check: N={n_max} gain={gain_asym:.6f}  B.2={b2_gain:.6f}  "
              f"reldiff={reldiff:.2e}  [{status}]")
        asym_rows.append(dict(
            model=model, pair=pair, n_cal_max=n_max,
            gain_sq_asym=round(gain_asym, 6),
            gain_lin_asym=round(float(np.sqrt(gain_asym)), 6),
            gain_sq_b2=round(b2_gain, 6),
            reldiff=round(reldiff, 8),
            match=status,
        ))

df      = pd.DataFrame(rows)
df_asym = pd.DataFrame(asym_rows)
df.to_csv(UNIFIED / "b3_snr_convergence.csv", index=False)
df_asym.to_csv(UNIFIED / "b3_asymptotic_gain.csv", index=False)
print(f"\nCSV → NeuMoR/output/unified/b3_snr_convergence.csv")
print(f"CSV → NeuMoR/output/unified/b3_asymptotic_gain.csv")

print()
print("ASYMPTOTIC GAIN TABLE (N_max)")
print("-" * 55)
print(f"  {'Model':8} {'Pair':4} {'N_max':6} {'gain_sq':10} {'gain_lin':10} {'vs B.2':8}")
for _, r in df_asym.iterrows():
    print(f"  {r['model']:8} {r['pair']:4} {r['n_cal_max']:6d} "
          f"{r['gain_sq_asym']:10.4f} {r['gain_lin_asym']:10.4f} {r['match']:8}")

print()
print("SANITY CHECKS")
print("-" * 50)
for model in ["heston", "kou"]:
    for pair in ["A", "B", "C", "D"]:
        sub = df[(df.model==model) & (df.pair==pair)].sort_values("n_cal")
        gains = sub.gain_sq.values

        if model == "heston":
            kou_sub = df[(df.model=="kou") & (df.pair==pair)]

            shared_n = set(sub.n_cal.values) & set(kou_sub.n_cal.values)
            for n in sorted(shared_n):
                hg = float(sub[sub.n_cal==n].gain_sq)
                kg = float(kou_sub[kou_sub.n_cal==n].gain_sq)
                if hg <= kg:
                    print(f"  [WARN] Heston pair{pair} gain({hg:.3f}) <= Kou gain({kg:.3f}) at N={n}")

pair_colors  = {"A": "steelblue", "B": "darkorange", "C": "forestgreen", "D": "crimson"}
pair_markers = {"A": "o", "B": "s", "C": "^", "D": "D"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
fig.suptitle(r"Fig 3 (updated): SNR$^2$ gain vs $N_{\mathrm{cal}}$ — Unified formulation ($K_\eta$, $\hat{d}$)",
             fontsize=12)

for ax, model in zip(axes, ["heston", "kou"]):
    sub_m = df[df.model == model]
    asym_m = df_asym[df_asym.model == model]

    for pair in ["A", "B", "C", "D"]:
        sub = sub_m[sub_m.pair == pair].sort_values("n_cal")
        n_vals  = sub.n_cal.values
        g_vals  = sub.gain_sq.values
        ci_lo   = sub.ci_lo_sq.values
        ci_hi   = sub.ci_hi_sq.values
        asym_g  = float(asym_m[asym_m.pair == pair].gain_sq_asym)

        col = pair_colors[pair]
        mrk = pair_markers[pair]

        ax.plot(n_vals, g_vals, color=col, marker=mrk, ms=5, lw=1.5,
                label=f"Pair {pair}")
        ax.fill_between(n_vals, ci_lo, ci_hi, color=col, alpha=0.15)
        ax.axhline(asym_g, color=col, lw=0.8, ls="--", alpha=0.5)

    ax.set_xscale("log")
    ax.set_xlabel(r"$N_{\mathrm{cal}}$ (log scale)", fontsize=11)
    ax.set_ylabel(r"SNR$^2$ gain (vs ATM)", fontsize=10)
    ax.set_title(f"{model.capitalize()}", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)

    cfg = CONFIGS[model]
    n_max = cfg["n_cal_vals"][-1]
    asym_gains = asym_m.gain_sq_asym.values
    txt = (f"$N_{{\\mathrm{{max}}}}={n_max}$ asymptote:\n"
           f"[{asym_gains.min():.2f}, {asym_gains.max():.2f}]")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

plt.tight_layout()
for ext, ddir in [("png", FIG_DIR), ("pdf", PDF_DIR)]:
    p = ddir / f"b3_fig3_snr_convergence.{ext}"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Fig → {p.relative_to(ROOT)}")
plt.close(fig)

print("\nB.3 complete.")
