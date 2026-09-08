"""
decompose_bias_JS.py — J/S decomposition of bias_reduction_gstar.csv.
J/S decomposition of bias_reduction_gstar.csv results.
No new simulation — pure post-processing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import erfc

ROOT = Path(__file__).resolve().parents[2]
IN_CSV  = ROOT / "NeuMoR/output/unified/bias_reduction_gstar.csv"
OUT_CSV = ROOT / "NeuMoR/output/unified/bias_reduction_gstar_decomposed.csv"

df = pd.read_csv(IN_CSV)

def decompose(sigma, delta, b_delta, mr_theory):
    """Return J, S, J/|Δ|*100, S/|Δ|*100, sanity_match, |b_Δ/Δ|."""
    mu = delta + b_delta
    s  = mu / sigma if sigma > 0 else float("nan")
    J  = sigma * (np.sqrt(2/np.pi) * np.exp(-s**2 / 2)
                  - abs(s) * erfc(abs(s) / np.sqrt(2)))
    S  = abs(mu) - abs(delta)
    abs_d = abs(delta)
    J_pct = J / abs_d * 100 if abs_d > 0 else float("nan")
    S_pct = S / abs_d * 100 if abs_d > 0 else float("nan")
    # Sanity: |Δ| + J + S should equal mr_theory
    sanity = (abs_d + J + S) / mr_theory if mr_theory != 0 else float("nan")
    bdelta_ratio = abs(b_delta / delta) if abs_d > 0 else float("nan")
    return J, S, J_pct, S_pct, sanity, bdelta_ratio

rows = []
for _, r in df.iterrows():
    J_a, S_a, Ja_pct, Sa_pct, sn_a, bd_a = decompose(r.sigma_atm, r.delta_atm, r.b_delta_atm, r.mr_theory_atm)
    J_g, S_g, Jg_pct, Sg_pct, sn_g, bd_g = decompose(r.sigma_gstar, r.delta_gstar, r.b_delta_gstar, r.mr_theory_gstar)

    rows.append(dict(
        model=r.model, pair=r.pair,
        # ATM block
        sigma_atm=r.sigma_atm,
        J_atm=J_a, S_atm=S_a,
        J_atm_pct=round(Ja_pct, 3), S_atm_pct=round(Sa_pct, 3),
        JplusS_atm_pct=round(Ja_pct + Sa_pct, 3),
        bdelta_ratio_atm=round(bd_a, 4),
        sanity_atm=round(sn_a, 4),
        # g* block
        sigma_gstar=r.sigma_gstar,
        J_gstar=J_g, S_gstar=S_g,
        J_gstar_pct=round(Jg_pct, 3), S_gstar_pct=round(Sg_pct, 3),
        JplusS_gstar_pct=round(Jg_pct + Sg_pct, 3),
        bdelta_ratio_gstar=round(bd_g, 4),
        sanity_gstar=round(sn_g, 4),
        # changes
        J_reduction=round(J_g / J_a, 4) if J_a > 0 else float("nan"),
        S_change=round(S_g / S_a, 4) if abs(S_a) > 1e-30 else float("nan"),
    ))

out = pd.DataFrame(rows)
out.to_csv(OUT_CSV, index=False)

# ── Sanity check: J + S + |Δ| should match theory_mr ──────────────────────────
print(f"Sanity check (|Δ|+J+S)/mr_theory ratio:")
print(f"  ATM: median {out.sanity_atm.median():.6f}, min {out.sanity_atm.min():.6f}, max {out.sanity_atm.max():.6f}")
print(f"  g* : median {out.sanity_gstar.median():.6f}, min {out.sanity_gstar.min():.6f}, max {out.sanity_gstar.max():.6f}")
print(f"  (should be 1.0000 if formulas correct)\n")

# ── Summary stdout ────────────────────────────────────────────────────────────
for m in ["heston","kou"]:
    sub = out[out.model == m]
    Ja_med = sub.J_atm_pct.median()
    Sa_med = sub.S_atm_pct.median()
    Jg_med = sub.J_gstar_pct.median()
    Sg_med = sub.S_gstar_pct.median()
    Jred   = sub.J_reduction.median()
    Schg   = sub.S_change.median()
    label  = m.capitalize()
    print(f"{label}: ATM J={Ja_med:.2f}%, S={Sa_med:+.2f}% | g* J={Jg_med:.2f}%, S={Sg_med:+.2f}% | J reduction {Jred:.3f}x, S change {Schg:.3f}x")

print()
for m in ["heston","kou"]:
    sub = out[out.model == m]
    print(f"{m.capitalize()} |b_Δ/Δ| median: ATM {sub.bdelta_ratio_atm.median():.4f}, g* {sub.bdelta_ratio_gstar.median():.4f}")

print()
hes_bd = out[out.model=='heston'].bdelta_ratio_gstar.median()
kou_bd = out[out.model=='kou'].bdelta_ratio_gstar.median()
print(f"Kou g* increases S because |b_Δ/Δ| at g*: {kou_bd:.3f} vs Heston {hes_bd:.3f}")

print(f"\nOutput: {OUT_CSV}")
