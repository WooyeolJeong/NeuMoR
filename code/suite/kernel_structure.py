
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

ROOT    = Path(__file__).resolve().parents[2]
UNIFIED = ROOT / "NeuMoR/output/unified"
FIG_DIR = ROOT / "NeuMoR/figures"
PDF_DIR = ROOT / "NeuMoR/pdfs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from NeuMoR.code.neumor_core import effective_rank

rows = []

for model in ["heston", "kou"]:
    for pair in ["A", "B", "C", "D"]:
        npz = np.load(UNIFIED / f"{model}_pair{pair}.npz", allow_pickle=True)

        eigvals_eta = npz["eigvals_eta"]
        eigvals_k11 = npz["eigvals_k11"]
        y_grid      = npz["y_grid"]
        dy          = float(y_grid[1] - y_grid[0])

        ev_eta  = eigvals_eta.clip(0)
        lam_max = float(ev_eta[0])
        tr_eta  = float(ev_eta.sum())
        tr_dy   = tr_eta * dy
        reff_eta = float(ev_eta.sum()**2 / (ev_eta**2).sum()) if (ev_eta**2).sum() > 0 else 0.0

        total = tr_eta
        top1  = float(ev_eta[:1].sum()  / total) if total > 0 else 0.0
        top3  = float(ev_eta[:3].sum()  / total) if total > 0 else 0.0
        top10 = float(ev_eta[:10].sum() / total) if total > 0 else 0.0

        ev_k11   = eigvals_k11.clip(0)
        reff_k11 = float(ev_k11.sum()**2 / (ev_k11**2).sum()) if (ev_k11**2).sum() > 0 else 0.0

        gain_sq_dhat  = float(npz["gain_sq_eta_dhat"])
        gain_sq_dana  = float(npz["gain_sq_eta_dana"])
        gain_lin_dhat = float(npz["gain_lin_eta_dhat"])
        gain_lin_dana = float(npz["gain_lin_eta_dana"])

        reff_from_file = float(npz["reff_eta"])
        assert abs(reff_eta - reff_from_file) < 1e-4, \
            f"{model} pair{pair}: r_eff mismatch {reff_eta:.4f} vs {reff_from_file:.4f}"

        rows.append(dict(
            model=model, pair=pair,
            lam_max_eta=lam_max,
            tr_eta_dy=tr_dy,
            reff_eta=reff_eta,
            top1_share=top1,
            top3_share=top3,
            top10_share=top10,
            reff_k11=reff_k11,
            gain_sq_dhat=gain_sq_dhat,
            gain_sq_dana=gain_sq_dana,
            gain_lin_dhat=gain_lin_dhat,
            gain_lin_dana=gain_lin_dana,
        ))

df = pd.DataFrame(rows)

print("SANITY CHECKS")
print("-" * 50)
for _, r in df.iterrows():
    monotone = r["top1_share"] <= r["top3_share"] <= r["top10_share"]
    reff_ok  = 1.0 <= r["reff_eta"]
    print(f"  {r['model']} pair{r['pair']}: "
          f"r_eff={r['reff_eta']:.3f}  top1={r['top1_share']:.3f}  "
          f"top3={r['top3_share']:.3f}  top10={r['top10_share']:.3f}  "
          f"monotone={'OK' if monotone else 'FAIL'}  "
          f"reff>=1={'OK' if reff_ok else 'FAIL'}")

print()
print("TABLE 4 — Kernel Structure (K_eta, unified formulation)")
print("=" * 90)
hdr = f"{'Model':8} {'Pair':4} {'lam_max':12} {'tr*dy':12} {'r_eff(Keta)':12} {'top-1':7} {'top-3':7} {'top-10':7} {'r_eff(K11)':10}"
print(hdr)
print("-" * 90)
for _, r in df.iterrows():
    print(f"  {r['model']:6} {r['pair']:4} "
          f"{r['lam_max_eta']:12.6e} "
          f"{r['tr_eta_dy']:12.6e} "
          f"{r['reff_eta']:12.3f} "
          f"{r['top1_share']:7.4f} "
          f"{r['top3_share']:7.4f} "
          f"{r['top10_share']:7.4f} "
          f"{r['reff_k11']:10.3f}")

print()
print("ISSUE 2 CHECK — r_eff(K_eta) vs gain_sq (within-model)")
print("=" * 60)

def corr_report(name, reff_vals, gain_vals):
    p_r, p_p = pearsonr(reff_vals, gain_vals)
    s_r, s_p = spearmanr(reff_vals, gain_vals)
    print(f"  {name}:")
    print(f"    Pearson  r = {p_r:+.4f}  (p={p_p:.3f})")
    print(f"    Spearman r = {s_r:+.4f}  (p={s_p:.3f})")

    paired = sorted(zip(reff_vals, gain_vals))
    print(f"    r_eff order: {[f'{r:.3f}' for r,g in paired]}")
    print(f"    gain order:  {[f'{g:.3f}' for r,g in paired]}")
    both_pos  = p_r > 0 and s_r > 0
    both_half = p_r > 0.5 and s_r > 0.5
    if both_half:
        verdict = "RESOLVED (both > 0.5, strong positive)"
    elif both_pos:
        verdict = "WEAKLY RESOLVED (both > 0, qualitative positive)"
    else:
        verdict = "NOT RESOLVED (negative correlation detected)"
    print(f"    → {verdict}")
    return p_r, s_r

he = df[df.model == "heston"]
ko = df[df.model == "kou"]
he_pr, he_sr = corr_report("Heston (K_eta)", he.reff_eta.values, he.gain_sq_dhat.values)
print()
ko_pr, ko_sr = corr_report("Kou    (K_eta)", ko.reff_eta.values, ko.gain_sq_dhat.values)

print()
print("  Legacy reference (K_11, for comparison):")
he_pr_k11, _ = pearsonr(he.reff_k11.values, he.gain_sq_dhat.values)
he_sr_k11, _ = spearmanr(he.reff_k11.values, he.gain_sq_dhat.values)
print(f"    Heston K_11: Pearson={he_pr_k11:+.4f}  Spearman={he_sr_k11:+.4f}")
print(f"    (K_11 r_eff is constant across pairs — correlation undefined in strict sense)")

print(f"    Heston r_eff(K_11) values: {he.reff_k11.values.round(3)}")

csv_path = UNIFIED / "b4_table4_kernel_structure.csv"
df.to_csv(csv_path, index=False)
print(f"\nCSV → {csv_path.relative_to(ROOT)}")

markers_h = ["o", "s", "^", "D"]
markers_k = ["o", "s", "^", "D"]
colors_h  = "steelblue"
colors_k  = "darkorange"

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
fig.suptitle(r"Fig 5: $r_{\mathrm{eff}}(K_\eta)$ vs Linear SNR gain — Unified formulation",
             fontsize=12)

he_sub = df[df.model == "heston"]
ko_sub = df[df.model == "kou"]

for i, (_, r) in enumerate(he_sub.iterrows()):
    ax.scatter(r.reff_eta, r.gain_lin_dhat,
               color=colors_h, marker=markers_h[i], s=90, zorder=5,
               label=f"Heston {r.pair}")
    ax.annotate(r.pair, (r.reff_eta, r.gain_lin_dhat),
                textcoords="offset points", xytext=(0, 8), fontsize=9, color=colors_h,
                ha="center")

for i, (_, r) in enumerate(ko_sub.iterrows()):
    ax.scatter(r.reff_eta, r.gain_lin_dhat,
               color=colors_k, marker=markers_k[i], s=90, zorder=5,
               label=f"Kou {r.pair}")
    ax.annotate(r.pair, (r.reff_eta, r.gain_lin_dhat),
                textcoords="offset points", xytext=(0, 8), fontsize=9, color=colors_k,
                ha="center")

ax.set_xlabel(r"$r_{\mathrm{eff}}(K_\eta)$", fontsize=11)
ax.set_ylabel("Linear SNR gain (g* vs ATM)", fontsize=10)
ax.grid(True, alpha=0.3)

he_reff = he_sub.reff_eta.values
ko_reff = ko_sub.reff_eta.values
txt = (f"Heston: $r_{{\\mathrm{{eff}}}}\\in[{he_reff.min():.2f},{he_reff.max():.2f}]$\n"
       f"Kou:    $r_{{\\mathrm{{eff}}}}\\in[{ko_reff.min():.2f},{ko_reff.max():.2f}]$")
ax.text(0.03, 0.97, txt, transform=ax.transAxes,
        fontsize=8, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

ax.legend(fontsize=8, loc="lower right")

plt.tight_layout()
for ext, ddir in [("png", FIG_DIR), ("pdf", PDF_DIR)]:
    p = ddir / f"b4_fig5_reff_vs_gain_v2.{ext}"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Fig → {p.relative_to(ROOT)}")
plt.close(fig)

caption_fig5 = (
    "Fig 5: Effective rank r_eff(K_eta) vs linear SNR gain (g* / ATM) for 8 (model, pair) "
    "combinations under the unified K_eta formulation. Blue = Heston, orange = Kou; pairs A–D "
    "denote increasing parameter perturbation magnitude. Linear SNR gain = sqrt(SNR^2 gain); "
    "SNR^2 gain = (linear gain)^2. "
    "Note: Under the legacy K_11 formulation, r_eff was constant across all pairs within each "
    "model (Heston: 15.15, Kou: 1.63), preventing within-class analysis; those points are "
    "excluded from this figure."
)
print(f"\nCAPTION (Fig 5):\n{caption_fig5}")

print()
print("SUMMARY — r_eff(K_eta) vs gain_sq (d_hat)")
print("-" * 55)
print(f"  {'Model':8} {'Pair':4} {'r_eff_eta':10} {'gain_sq':10} {'gain_lin':10}")
for _, r in df.iterrows():
    print(f"  {r['model']:8} {r['pair']:4} "
          f"{r['reff_eta']:<10.3f} {r['gain_sq_dhat']:<10.4f} {r['gain_lin_dhat']:.4f}")

print()
print("ISSUE 2 VERDICT:")
he_resolved = he_pr > 0.5 and he_sr > 0.5
ko_resolved = ko_pr > 0.5 and ko_sr > 0.5
if he_resolved and ko_resolved:
    print("  RESOLVED — both models show strong positive within-model r_eff vs gain correlation.")
    print("  K_11 negative correlation was a formulation artifact.")
    print("  Remark 4.11 can be re-stated with K_eta: within-model tension eliminated.")
elif he_pr > 0 and he_sr > 0 and ko_pr > 0 and ko_sr > 0:
    print("  WEAKLY RESOLVED — positive but below 0.5 threshold in at least one model.")
else:
    print("  NOT RESOLVED — inspect individual correlations above.")
