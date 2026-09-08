"""
b7_portfolio_v2.py — B.7 (v2): §6 방식 재현 + K_eta 전환
==========================================================

§6 (exp12.py) 와 동일한 portfolio-level net aggregation 사용:
    G_naive   = net_sz * g_ATM          (signed net_sz = Σ_j size_j)
    G_posspec = Σ_j size_j * g_j        (signed)
    sigma     = sqrt(G^T K_eta G * dy^2)  (unbiased: ×N/(N-1))
    mu        = G^T d_hat * dy
    Δ_true    = G^T d_lewis * dy
    bias      = folded_normal_mean(mu, sigma) - |Δ_true|
    capital   = (3*sigma + |bias|) * NOTIONAL

Step A: K_eta 로 §6 saving 재현 확인 (PASS 기준: reldiff < 5%)
        ※ K_11 은 §6 에서 sigma 결정 안 함 → K_eta 가 올바른 K.
Step B: (PASS 시) K_eta 결과 확정, legacy 와 비교 table.
Step C: ATM-only trivial test (saving = 0% 검증).

Outputs:
    NeuMoR/output/unified/b7_portfolio_v2.csv
    NeuMoR/figures/b7_fig4_portfolio_saving_v2.png + .pdf

Run:
    python NeuMoR/code/b7_portfolio_v2.py
"""

import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from NeuMoR.code.neumor_core import build_k_eta, compute_d_hat
from TNO.common.tno_model import DensityDeepONet_HestonLog
from NeuMoR.src.heston_teacher import lewis_pdf
from NeuMoR.src.kou.kou_pricing import kou_log_density

UNIFIED = ROOT / "NeuMoR/output/unified"
FIG_DIR = ROOT / "NeuMoR/figures"
PDF_DIR = ROOT / "NeuMoR/pdfs"
for d in [UNIFIED, FIG_DIR, PDF_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)

N_SEEDS    = 200
NOTIONAL   = 1_000_000.0
PASS_THRESH = 5.0       # reldiff < 5% → PASS

HESTON_RANGE = {
    "v": (0.05,0.30), "kappa": (0.5,2.0), "omega": (0.05,0.30),
    "xi": (0.1,0.5),  "rho": (-0.9,-0.1),
}
KOU_PARAM_KEYS = ["sigma","lam_J","p","eta1","eta2"]
KOU_LO = np.array([0.10, 0.5, 0.30, 3.0, 3.0], dtype=np.float32)
KOU_HI = np.array([0.80,50.0, 0.70,50.0,50.0], dtype=np.float32)

# Heston §6 baseline (exp12.py capital_implications.csv — Heston only)
LEGACY_HESTON = {
    "Luna": {"naive_M": 1.8297, "posspec_M": 0.3314, "saving_pct": 81.89,
             "sig_naive": 0.07646, "bias_naive": 1.6004,
             "sig_posspec": 0.06083, "bias_posspec": -0.14890},
    "FTX":  {"naive_M": 1.7056, "posspec_M": 0.2370, "saving_pct": 86.11,
             "sig_naive": 0.09156, "bias_naive": 1.4309,
             "sig_posspec": 0.05558, "bias_posspec":  0.07023},
    "ETF":  {"naive_M": 0.7639, "posspec_M": 0.3393, "saving_pct": 55.58,
             "sig_naive": 0.07441, "bias_naive": 0.5406,
             "sig_posspec": 0.05975, "bias_posspec": -0.16009},
}


# ── Folded-normal mean ────────────────────────────────────────────────────────

def folded_normal_mean(mu: float, sigma: float) -> float:
    """E[|X|] for X ~ N(mu, sigma^2). Theorem 3.1 formula."""
    if sigma < 1e-30:
        return abs(mu)
    s = mu / sigma
    return float(sigma * np.sqrt(2.0/np.pi) * np.exp(-0.5*s**2)
                 + mu * erf(s / np.sqrt(2.0)))


# ── Payoff functions ──────────────────────────────────────────────────────────

def get_payoff(pos_row, y: np.ndarray) -> np.ndarray:
    K = float(pos_row["K"])
    t = str(pos_row["type"])
    if t == "call":    return np.maximum(np.exp(y) - K, 0.0)
    elif t == "put":   return np.maximum(K - np.exp(y), 0.0)
    elif t == "digital": return (y >= np.log(K)).astype(np.float64)
    return np.maximum(np.exp(y) - 1.0, 0.0)

def atm_call(y): return np.maximum(np.exp(y) - 1.0, 0.0)


# ── Clipping ─────────────────────────────────────────────────────────────────

def clip_heston(p, label=""):
    out = {}
    for k, v in p.items():
        if k in HESTON_RANGE:
            lo, hi = HESTON_RANGE[k]
            cv = float(np.clip(v, lo, hi))
            if abs(cv-v) > 1e-9: print(f"  [CLIP] {label}.{k}: {v:.4f}→{cv:.4f}")
            out[k] = cv
        else: out[k] = v
    return out

def clip_kou(raw, label=""):
    c = np.clip(raw, KOU_LO, KOU_HI)
    for i, k in enumerate(KOU_PARAM_KEYS):
        if abs(c[i]-raw[i]) > 1e-9: print(f"  [CLIP] {label}.{k}: {raw[i]:.4f}→{c[i]:.4f}")
    return c


# ── Model loading ─────────────────────────────────────────────────────────────

def load_heston():
    SA = ROOT/"NeuMoR/save/exp01"; SB = ROOT/"NeuMoR/save/exp07_5"
    models, y, pk = [], None, None
    for s in range(N_SEEDS):
        p = (SA/f"seed_{s:02d}.pt") if s<50 else (SB/f"seed_{s:03d}.pt")
        ck = torch.load(p, map_location=DEVICE, weights_only=False)
        cfg = ck["config"]
        m = DensityDeepONet_HestonLog(lambda_dim=cfg["lambda_dim"], n_y=cfg["n_y"],
                                       rank=cfg["rank"], branch_hidden=cfg["branch_hidden"]).to(DEVICE)
        m.load_state_dict(ck["state_dict"]); m.eval(); models.append(m)
        if y is None: y=ck["y_grid"].astype(np.float64); pk=list(ck["param_keys"])
    return models, y, pk

def load_kou():
    SA = ROOT/"NeuMoR/save/kou/stageA"; SB = ROOT/"NeuMoR/save/kou/stageB"
    models = []
    for s in range(N_SEEDS):
        p = (SA/f"seed_{s:04d}.pt") if s<30 else (SB/f"seed_{s:04d}.pt")
        ck = torch.load(p, map_location=DEVICE, weights_only=False)
        m = DensityDeepONet_HestonLog(lambda_dim=5, n_y=256, rank=192, branch_hidden=512).to(DEVICE)
        m.load_state_dict(ck["state_dict"]); m.eval(); models.append(m)
    return models, np.linspace(-4.0, 4.0, 256, dtype=np.float64)


# ── Inference ────────────────────────────────────────────────────────────────

def infer_h(models, params, y, pk):
    dy = float(y[1]-y[0])
    lam = torch.tensor(np.array([params[k] for k in pk],dtype=np.float32)).unsqueeze(0).to(DEVICE)
    out = np.zeros((len(models), len(y)), dtype=np.float32)
    with torch.no_grad():
        for i, m in enumerate(models):
            r = m(lam, dy); out[i]=(r[0] if isinstance(r,(list,tuple)) else r).cpu().numpy().squeeze()
    return out.astype(np.float64)

def infer_k(models, raw, y):
    dy = float(y[1]-y[0])
    ln = torch.tensor(((raw-KOU_LO)/(KOU_HI-KOU_LO))[None],dtype=torch.float32)
    out = np.zeros((len(models), len(y)), dtype=np.float32)
    with torch.no_grad():
        for i, m in enumerate(models):
            r = m(ln.to(DEVICE), dy); out[i]=(r[0] if isinstance(r,(list,tuple)) else r).cpu().numpy().squeeze()
    return out.astype(np.float64)


# ── True P&L (fine-grid Lewis/Kou integration) ───────────────────────────────

def compute_delta_pi_true(pos_df, p1_fn, p2_fn, y, n_fine=2000):
    """Σ_j size_j * (V1_j - V2_j) on fine grid — matches exp12.py delta_pi_true."""
    y_fine  = np.linspace(float(y[0]), float(y[-1]), n_fine)
    dy_fine = float(y_fine[1] - y_fine[0])
    p1_fine = p1_fn(y_fine).astype(np.float64)
    p2_fine = p2_fn(y_fine).astype(np.float64)
    d_fine  = p1_fine - p2_fine
    total   = 0.0
    for _, pos in pos_df.iterrows():
        sz  = float(pos["size"])
        g   = get_payoff(pos, y_fine)
        total += sz * float(np.dot(g, d_fine)) * dy_fine
    return total


# ── §6 portfolio capital (net aggregation) ─────────────────────────────────

def portfolio_capital(pos_df, K, d_hat, y, unbiased_n=None, delta_pi_true=0.0):
    """
    §6 방식: signed sizes, portfolio-level G, K_eta for sigma.

    sigma     = sqrt(G^T K_ub G * dy^2)   (unbiased correction)
    mu        = G^T d_hat * dy             (NN ensemble mean estimate)
    bias      = folded_normal_mean(mu, sigma) - |delta_pi_true|
    capital   = (3*sigma + |bias|) * NOTIONAL

    delta_pi_true: Σ_j size_j*(V1_j - V2_j) on fine grid — SAME reference
                   for BOTH naive and posspec strategies (matches exp12.py).
    """
    dy    = float(y[1] - y[0])
    sizes = pos_df["size"].values.astype(float)    # signed
    net_sz = float(sizes.sum())
    g_atm  = atm_call(y)

    # Unbiased K correction (match §6 ddof=1 empirical std)
    n = unbiased_n if unbiased_n is not None else N_SEEDS
    K_ub = K * (n / (n - 1))

    # Portfolio payoffs
    G_naive   = net_sz * g_atm                                    # (n_y,)
    payoffs   = np.array([get_payoff(r, y) for _, r in pos_df.iterrows()])  # (50, n_y)
    G_posspec = (sizes[:, None] * payoffs).sum(axis=0)            # (n_y,)

    def cap_one(G):
        var   = float(G @ (K_ub @ G)) * dy**2
        sigma = float(np.sqrt(max(var, 0.0)))
        mu    = float(G @ d_hat) * dy
        EN    = folded_normal_mean(mu, sigma)
        bias  = EN - abs(delta_pi_true)
        return sigma, mu, bias, (3*sigma + abs(bias)) * NOTIONAL

    sig_n, mu_n, b_n, cap_n = cap_one(G_naive)
    sig_p, mu_p, b_p, cap_p = cap_one(G_posspec)

    saving = (cap_n - cap_p) / max(abs(cap_n), 1e-20) * 100
    if saving < 0:
        print(f"  WARNING: position-specific exceeds naive! saving={saving:.2f}%")

    return dict(
        naive_M=cap_n/1e6, posspec_M=cap_p/1e6, saving_pct=saving,
        sig_naive=sig_n, sig_posspec=sig_p, mu_naive=mu_n, mu_posspec=mu_p,
        delta_pi_true=delta_pi_true, bias_naive=b_n, bias_posspec=b_p,
        G_naive=G_naive, G_posspec=G_posspec,
    )


# ── ATM-only trivial test ─────────────────────────────────────────────────────

def atm_only_test(pos_df, K, d_hat, y, p1_fn, p2_fn):
    """모든 position 을 ATM call 로 교체 → saving 0% 검증."""
    fake_df = pos_df.copy()
    fake_df["type"] = "call"; fake_df["K"] = 1.0; fake_df["subtype"] = "ATM"
    delta_atm = compute_delta_pi_true(fake_df, p1_fn, p2_fn, y)
    res = portfolio_capital(fake_df, K, d_hat, y, delta_pi_true=delta_atm)
    diff = abs(res["saving_pct"])
    ok   = diff < 1e-6
    print(f"  ATM-only trivial test: saving={res['saving_pct']:.2e}%  "
          f"{'PASS' if ok else 'FAIL (bug!)'}")
    return ok


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("B.7 v2: §6 방식 재현 + K_eta 전환")
    print("="*70)
    t0 = time.time()

    pos_df  = pd.read_csv(ROOT/"NeuMoR/output/exp12/portfolio.csv")
    kou_btc = pd.read_csv(ROOT/"NeuMoR/output/kou/day3/btc_calibrations.csv")
    net_sz  = float(pos_df["size"].sum())
    print(f"\nPortfolio: 50 positions  net_size={net_sz:+.0f}  "
          f"sum|size|={pos_df['size'].abs().sum():.0f}")

    print("\nLoading Heston models (200 seeds)...")
    h_models, h_y, h_pk = load_heston()
    print("Loading Kou models (200 seeds)...")
    k_models, k_y = load_kou()

    kou_scen = {}
    for scen in ["Luna","FTX","ETF"]:
        pre  = kou_btc[kou_btc["scenario"]==f"{scen}_pre"].iloc[0]
        post = kou_btc[kou_btc["scenario"]==f"{scen}_post"].iloc[0]
        kou_scen[scen] = {
            "pre":  clip_kou(np.array([float(pre[k])  for k in KOU_PARAM_KEYS],dtype=np.float32), f"{scen}_pre"),
            "post": clip_kou(np.array([float(post[k]) for k in KOU_PARAM_KEYS],dtype=np.float32), f"{scen}_post"),
        }

    # ── Load Heston calibrated params (calibrations.csv — same as exp12.py) ──
    h_cal = pd.read_csv(ROOT/"NeuMoR/output/exp12/calibrations.csv")
    heston_scen = {}
    for scen in ["Luna","FTX","ETF"]:
        m1 = h_cal[(h_cal["scenario"]==scen) & (h_cal["model"]=="M1")].iloc[0]
        m2 = h_cal[(h_cal["scenario"]==scen) & (h_cal["model"]=="M2")].iloc[0]
        heston_scen[scen] = {
            "pre":  {k: float(m1[k]) for k in ["v","kappa","omega","xi","rho","T"]},
            "post": {k: float(m2[k]) for k in ["v","kappa","omega","xi","rho","T"]},
        }

    # ── Inference + fine-grid delta_pi_true ──────────────────────────────
    cache = {}   # (scen, model) → {l1, l2, y, d_hat, delta_pi_true, p1_fn, p2_fn}
    for scen in ["Luna","FTX","ETF"]:
        for mname in ["heston","kou"]:
            tag = f"{scen}-{mname}"
            if mname == "heston":
                y    = h_y
                pre  = heston_scen[scen]["pre"]
                post = heston_scen[scen]["post"]
                print(f"  [{tag}] params: pre v={pre['v']:.4f} kap={pre['kappa']:.2f} "
                      f"xi={pre['xi']:.2f}  post v={post['v']:.4f} kap={post['kappa']:.2f}")
                l1 = infer_h(h_models, pre,  y, h_pk)
                l2 = infer_h(h_models, post, y, h_pk)
                p1_fn = lambda yy, _pre=pre:  lewis_pdf(_pre, yy).astype(np.float64)
                p2_fn = lambda yy, _post=post: lewis_pdf(_post, yy).astype(np.float64)
            else:
                y  = k_y
                pc = kou_scen[scen]["pre"]; qc = kou_scen[scen]["post"]
                l1 = infer_k(k_models, pc, y)
                l2 = infer_k(k_models, qc, y)
                p1_fn = lambda yy, _pc=pc: kou_log_density(yy,1.0,tuple(_pc)).astype(np.float64)
                p2_fn = lambda yy, _qc=qc: kou_log_density(yy,1.0,tuple(_qc)).astype(np.float64)
            # True P&L on fine grid (posspec, Σ_j size_j*(V1_j-V2_j))
            # matches exp12.py delta_pi_true
            dpt = compute_delta_pi_true(pos_df, p1_fn, p2_fn, y)
            d_hat = compute_d_hat(l1, l2)
            cache[tag] = dict(l1=l1, l2=l2, y=y, d_hat=d_hat,
                              delta_pi_true=dpt, p1_fn=p1_fn, p2_fn=p2_fn)
            print(f"  [{tag}] done  ({len(l1)} seeds)  delta_pi_true={dpt:+.5f}")

    # ══════════════════════════════════════════════════════════════════════
    # STEP A: K_eta 로 §6 재현 검증
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("STEP A: K_eta — §6 saving 재현 검증")
    print(f"  (K_eta = Cov(pdf1-pdf2) = §6 empirical sigma 의 analytical equivalent)")
    print("="*70)

    step_a_rows = []
    heston_all_pass = True

    for scen in ["Luna","FTX","ETF"]:
        for mname in ["heston","kou"]:
            tag = f"{scen}-{mname}"
            dat = cache[tag]
            y   = dat["y"]
            # Use ALL seeds (200) — matching exp12.py empirical sigma
            K_eta = build_k_eta(dat["l1"], dat["l2"])
            res   = portfolio_capital(pos_df, K_eta, dat["d_hat"], y, N_SEEDS,
                                      delta_pi_true=dat["delta_pi_true"])

            if mname == "heston":
                leg     = LEGACY_HESTON[scen]
                legacy  = leg["saving_pct"]
                reldiff = abs(res["saving_pct"] - legacy) / max(abs(legacy), 1e-6) * 100
                passed  = reldiff < PASS_THRESH
                if not passed: heston_all_pass = False
                flag = "[PASS]" if passed else "[FAIL]"
                print(f"  {flag} {tag:15s}: K_eta={res['saving_pct']:.1f}%  "
                      f"§6={legacy:.1f}%  reldiff={reldiff:.1f}%  "
                      f"naive=${res['naive_M']:.3f}M (§6: ${leg['naive_M']:.3f}M)  "
                      f"posspec=${res['posspec_M']:.3f}M (§6: ${leg['posspec_M']:.3f}M)")
                # element-level comparison vs §6
                print(f"    delta_pi_true: K_eta={res['delta_pi_true']:+.5f}  "
                      f"(mu_naive={res['mu_naive']:+.5f}  mu_posspec={res['mu_posspec']:+.5f})")
                print(f"    sigma naive:   K_eta={res['sig_naive']:.5f}  §6={leg['sig_naive']:.5f}  "
                      f"diff={res['sig_naive']-leg['sig_naive']:+.5f}")
                print(f"    sigma posspec: K_eta={res['sig_posspec']:.5f}  §6={leg['sig_posspec']:.5f}  "
                      f"diff={res['sig_posspec']-leg['sig_posspec']:+.5f}")
                print(f"    bias naive:    K_eta={res['bias_naive']:+.5f}  §6={leg['bias_naive']:+.5f}  "
                      f"diff={res['bias_naive']-leg['bias_naive']:+.5f}")
                print(f"    bias posspec:  K_eta={res['bias_posspec']:+.5f}  §6={leg['bias_posspec']:+.5f}  "
                      f"diff={res['bias_posspec']-leg['bias_posspec']:+.5f}")
            else:
                # Kou: no §6 baseline (exp12.py Heston only)
                legacy  = float("nan")
                reldiff = float("nan")
                passed  = True   # N/A — not validated against §6
                flag = "[N/A]"
                print(f"  {flag} {tag:15s}: K_eta={res['saving_pct']:.1f}%  "
                      f"(no §6 Kou baseline)  "
                      f"naive=${res['naive_M']:.3f}M  posspec=${res['posspec_M']:.3f}M")

            step_a_rows.append(dict(
                tag=tag, scen=scen, mname=mname,
                keta_saving=res["saving_pct"], legacy=legacy,
                reldiff=reldiff, passed=passed,
                naive_M=res["naive_M"], posspec_M=res["posspec_M"],
                sig_naive=res["sig_naive"], sig_posspec=res["sig_posspec"],
                mu_naive=res["mu_naive"], mu_posspec=res["mu_posspec"],
                delta_pi_true=res["delta_pi_true"],
                bias_naive=res["bias_naive"], bias_posspec=res["bias_posspec"],
            ))

    if heston_all_pass:
        print("\n  Heston ALL PASS — K_eta 가 §6 saving 을 5% 오차 내 재현.")
    else:
        print("\n  Heston FAIL — check element-level diffs above.")

    # ══════════════════════════════════════════════════════════════════════
    # STEP C: ATM-only trivial test
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("STEP C: ATM-only trivial test (모든 position ATM → saving=0%)")
    print("="*70)
    dat_test = cache["Luna-heston"]
    K_test   = build_k_eta(dat_test["l1"], dat_test["l2"])
    atm_ok   = atm_only_test(pos_df, K_test, dat_test["d_hat"], dat_test["y"],
                             dat_test["p1_fn"], dat_test["p2_fn"])

    # ══════════════════════════════════════════════════════════════════════
    # STEP B: 확정 결과 + CSV + Figure
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("STEP B: K_eta 확정 결과")
    print("="*70)

    csv_rows = []
    for r in step_a_rows:
        delta_pp = (r["keta_saving"] - r["legacy"]) if not np.isnan(r["legacy"]) else float("nan")
        csv_rows.append(dict(
            scenario=r["scen"], model=r["mname"],
            naive_M=round(r["naive_M"],3), posspec_M=round(r["posspec_M"],3),
            saving_pct=round(r["keta_saving"],1),
            legacy_pct=r["legacy"],
            delta_pp=round(delta_pp, 1) if not np.isnan(delta_pp) else float("nan"),
            step_a_pass=r["passed"],
        ))

    df = pd.DataFrame(csv_rows)
    csv_path = UNIFIED / "b7_portfolio_v2.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV → {csv_path.relative_to(ROOT)}")

    # ── Summary table ──────────────────────────────────────────────────
    print()
    print(f"{'Scenario':8} | {'Model':8} | {'K_eta saving':14} | {'§6 baseline':12} | {'Δ%p':8} | {'A-pass':6}")
    print("-" * 70)
    for r in step_a_rows:
        leg_str  = f"{r['legacy']:12.1f}" if not np.isnan(r["legacy"]) else "         N/A"
        diff_str = f"{r['keta_saving']-r['legacy']:+8.1f}" if not np.isnan(r["legacy"]) else "     N/A"
        pass_str = "PASS" if r["passed"] else "FAIL"
        if r["mname"] == "kou": pass_str = "N/A"
        print(f"  {r['scen']:8} | {r['mname']:8} | {r['keta_saving']:14.1f} | "
              f"{leg_str} | {diff_str} | {pass_str:6}")

    # ── Figure ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    scenarios = ["Luna","FTX","ETF"]
    x = np.arange(len(scenarios)); w = 0.35
    colors = {"heston":"steelblue","kou":"darkorange"}

    for mi, mn in enumerate(["heston","kou"]):
        sub    = df[df.model==mn].set_index("scenario")
        saving = [sub.loc[s,"saving_pct"] for s in scenarios]
        legacy = [sub.loc[s,"legacy_pct"] for s in scenarios]
        off    = (mi - 0.5) * w
        ax.bar(x+off, saving, w, label=mn.capitalize(), color=colors[mn], alpha=0.85, zorder=3)
        for xi,(sv,lg) in enumerate(zip(saving,legacy)):
            if mn == "heston":   # only show §6 baseline for Heston
                ax.scatter(xi+off, lg, marker="_", s=120, color=colors[mn], zorder=5, linewidths=2)
            ax.annotate(f"{sv:.0f}%", (xi+off,sv), textcoords="offset points",
                        xytext=(0,6), ha="center", fontsize=8, color=colors[mn])

    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=11)
    ax.set_ylabel("Position-specific saving vs ATM-naive (%)", fontsize=10)
    ax.set_title("Fig 4 (v2): Portfolio capital saving — §6 aggregation + K_eta", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(df.saving_pct.max()+15, 100))
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.text(0.01, 0.97, "Horizontal marks = §6 Heston baseline (exp12.py)",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    plt.tight_layout()
    for ext, dd in [("png",FIG_DIR),("pdf",PDF_DIR)]:
        p = dd/f"b7_fig4_portfolio_saving_v2.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Fig → {p.relative_to(ROOT)}")
    plt.close(fig)

    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")
    print(f"\nKey insight: sigma 는 K_eta 가 결정 (§6 empirical std = G^T K_eta G * dy^2).")
    print(f"K_11 은 §6 에서 g_A 계산 에만 사용됨. σ와 무관.")

    cap = (
        "Fig 4 (v2): Portfolio capital saving (position-specific vs ATM-naive), "
        "§6 net aggregation + K_eta unified formulation. "
        "G_naive = net_sz*g_ATM, G_posspec = Σ_j size_j*g_j (signed). "
        "sigma = sqrt(G^T K_eta G * dy^2), bias = E[|ΔΠ̂|] - |Δ_true| (Theorem 3.1). "
        "Horizontal marks = legacy §6 result (K_11 g_A, same aggregation)."
    )
    print(f"\nCAPTION:\n{cap}")


if __name__ == "__main__":
    main()
