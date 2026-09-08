import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import numpy as np
import csv
from mpmath import mp, erf as merf, exp as mexp, sqrt as msqrt, pi as mpi, mpf
from scipy.special import erf

mp.dps = 40
HERE = Path(__file__).resolve().parent

S_GRID = [mpf("0.00"), mpf("0.15"), mpf("0.35"), mpf("0.80"), mpf("1.50"), mpf("3.00")]
N_GRID = [10, 20, 50, 200]
R = 200_000
SEED = 20260902

def f_mp(s):
    return msqrt(2 / mpi) * mexp(-s * s / 2) + s * merf(s / msqrt(2))

def avar_np_mp(s):
    return 1 + s * s - f_mp(s) ** 2

def avar_pi_mp(s):
    return merf(s / msqrt(2)) ** 2 + mexp(-s * s) / mpi

def m_np(mu, sigma):
    z = mu / sigma
    return sigma * np.sqrt(2.0 / np.pi) * np.exp(-0.5 * z * z) + mu * erf(z / np.sqrt(2.0))

rng = np.random.default_rng(SEED)
rows = []
print(f"[B12] asymptotic: mpmath dps={mp.dps} | finite-N: R={R:,} seed={SEED}")
for s in S_GRID:
    ratio_mp = avar_pi_mp(s) / avar_np_mp(s)
    var_ratio = float(mp.nstr(ratio_mp, 15))
    rmse_ratio = float(mp.nstr(msqrt(ratio_mp), 15))
    sf = float(s)
    target = m_np(sf, 1.0)
    print(f"  s={sf:.2f}  asym var={var_ratio:.6f} rmse={rmse_ratio:.6f}")
    for N in N_GRID:
        x = rng.standard_normal((R, N)) + sf
        mr_np = np.abs(x).mean(axis=1)
        mu_h = x.mean(axis=1)
        sg_h = x.std(axis=1, ddof=0)
        mr_pi = m_np(mu_h, np.maximum(sg_h, 1e-300))
        e_np = (mr_np - target) ** 2
        e_pi = (mr_pi - target) ** 2
        r_hat = e_pi.mean() / e_np.mean()
        cov = np.cov(e_pi, e_np)[0, 1]
        se = r_hat * np.sqrt(
            (e_pi.var(ddof=1) / e_pi.mean() ** 2
             + e_np.var(ddof=1) / e_np.mean() ** 2
             - 2 * cov / (e_pi.mean() * e_np.mean())) / R
        )
        rows.append(dict(s=f"{sf:.2f}", N=N,
                         var_ratio_asym=var_ratio, rmse_ratio_asym=rmse_ratio,
                         mse_ratio_sim=float(r_hat), mc_se=float(se),
                         R=R, seed=SEED, target_m=float(target)))
        print(f"    N={N:<4d} sim={r_hat:.6f}  MC-SE={se:.6f}")

with open(HERE / "tab_efficiency.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print(f"[B12] wrote tab_efficiency.csv ({len(rows)} rows)")

print("\n[B12] table body (3dp):")
for s in S_GRID:
    sf = f"{float(s):.2f}"
    rs = [r for r in rows if r["s"] == sf]
    a = rs[0]
    cells = " & ".join(f"${r['mse_ratio_sim']:.3f}$" for r in rs)
    print(f"${sf}$ & ${a['var_ratio_asym']:.3f}$ & ${a['rmse_ratio_asym']:.3f}$ & {cells} \\\\")
