
import os, sys, json, csv
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEU  = HERE.parents[1]
ROOT = NEU.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(NEU / "code"))
sys.path.insert(0, str(NEU / "code" / "suite"))
sys.path.insert(0, str(NEU / "code" / "core"))
from NeuMoR.code.neumor_core import snr_squared          # noqa: E402
from vanilla_proxy import build_basis                     # noqa: E402

UNIFIED = NEU / "output" / "unified"
TOLS    = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4]
PAIRS   = ["A", "B", "C", "D"]
MODELS  = ["heston", "kou"]

PAPER = {
    "rank":  {"heston": [17, 16, 8, 4, 2],      "kou": [17, 17, 10, 6, 3]},
    "rmin":  {"heston": [91.5, 91.5, 83.8, 32.6, 41.1], "kou": [92.6, 92.6, 83.7, 81.3, 87.3]},
    "rmax":  {"heston": [96.5, 96.3, 93.3, 71.7, 66.2], "kou": [96.9, 96.9, 93.7, 90.2, 95.4]},
    "alpha": {"heston": [2830, 2808, 385, 68, 12],      "kou": [1012, 1012, 282, 135, 10]},
}

GATE_COND = {("heston","A"):12207237979.29882, ("heston","B"):11933564376.229378,
             ("heston","C"):11397672256.496006,("heston","D"):8629452388.239006,
             ("kou","A"):3488203517.38403,     ("kou","B"):3487327404.375528,
             ("kou","C"):3486098647.8967752,   ("kou","D"):3483001779.0983887}

GATE_RECA = {("heston","A"):92.873,("heston","B"):96.4936,("heston","C"):94.1477,("heston","D"):91.524,
             ("kou","A"):96.4527,("kou","B"):96.9129,("kou","C"):95.1416,("kou","D"):92.6179}

def median(v):
    s = sorted(v); n = len(s)
    return s[n//2] if n % 2 else 0.5*(s[n//2-1]+s[n//2])

rows, gates = [], []
print("[B13] sup-proxy tolerance sweep (reconstructed procedure)")
for model in MODELS:
    for pair in PAIRS:
        z = np.load(UNIFIED / f"{model}_pair{pair}.npz", allow_pickle=True)
        y = np.asarray(z["y_grid"], float); dy = float(y[1]-y[0])
        d = np.asarray(z["d_hat"], float)
        Kc = np.asarray(z["K_eta_cal"], float); Kt = np.asarray(z["K_eta_tst"], float)
        gs = np.asarray(z["g_star_eta_dhat"], float)
        lam_max = float(np.asarray(z["eigvals_eta"], float)[0]); tau = 0.01*lam_max
        B, labels = build_basis(y)
        A  = Kc + tau*np.eye(len(y))
        GA = B.T @ A @ B
        cA = B.T @ d
        cond = float(np.linalg.cond(GA))
        gates.append(dict(model=model, pair=pair, cond=cond, cond_ref=GATE_COND[(model,pair)],
                          cond_relerr=abs(cond-GATE_COND[(model,pair)])/GATE_COND[(model,pair)],
                          c_norm=float(np.linalg.norm(cA)), tau=tau, dy=dy, n_y=len(y),
                          basis=B.shape[1], lab0=labels[0], lab16=labels[-1]))

        s2_gs = snr_squared(gs, d, Kt, dy)
        sv = np.linalg.svd(GA, compute_uv=False)
        for t in TOLS:
            r = int((sv > t*sv[0]).sum())
            al = np.linalg.pinv(GA, rcond=t) @ cA
            h  = B @ al
            rec = 100.0*snr_squared(h, d, Kt, dy)/s2_gs
            rows.append(dict(model=model, pair=pair, tol=t, rank=r,
                             rec_oos=rec, alpha_free=float(np.linalg.norm(al)),
                             alpha_dyw=float(np.linalg.norm(al))/dy,
                             alpha_dym=float(np.linalg.norm(al))*dy))

print("\n=== Gate 1: cond(G_A) vs verify_aproj.cond_BtAB ===")
ok = 0
for g in gates:
    st = "OK" if g["cond_relerr"] < 1e-6 else "FAIL"
    ok += st == "OK"
    print("  %-7s %s  cond=%.6e ref=%.6e relerr=%.2e  %s" % (g["model"],g["pair"],g["cond"],g["cond_ref"],g["cond_relerr"],st))
print("  -> %d/8 passed" % ok)
print("\n=== Gate 2: ||c_A|| (dy convention check) ===")
for g in gates: print("  %-7s %s  ||c_A||=%.15g   (dy=%.6f)" % (g["model"],g["pair"],g["c_norm"],g["dy"]))
print("\n=== Gate 3: recA_oos (t=1e-12) vs verify_loss.csv ===")
for m in MODELS:
    for p in PAIRS:
        r0 = [x for x in rows if x["model"]==m and x["pair"]==p and x["tol"]==1e-12][0]
        ref = GATE_RECA[(m,p)]
        print("  %-7s %s  recomputed=%.4f  stored=%.4f  diff=%+.4f pp" % (m,p,r0["rec_oos"],ref,r0["rec_oos"]-ref))

print("\n=== Table regeneration (median over pairs A-D) ===")
out = {}
for m in MODELS:
    print("  [%s]  t        rank(paper)  recovery%%min-max(paper)     median||alpha|| free/dyw/dym (paper)" % m)
    for i,t in enumerate(TOLS):
        sub = [x for x in rows if x["model"]==m and x["tol"]==t]
        rk = int(median([x["rank"] for x in sub]))
        rc = [x["rec_oos"] for x in sub]
        af = median([x["alpha_free"] for x in sub]); aw = median([x["alpha_dyw"] for x in sub]); am = median([x["alpha_dym"] for x in sub])
        print("    %7.0e  %2d (%2d)   %6.1f-%6.1f (%.1f-%.1f)   %10.4g / %10.4g / %10.4g  (%d)" % (
            t, rk, PAPER["rank"][m][i], min(rc), max(rc), PAPER["rmin"][m][i], PAPER["rmax"][m][i],
            af, aw, am, PAPER["alpha"][m][i]))
        out[(m,t)] = (rk, min(rc), max(rc), af, aw, am)

with open(HERE/"b13_supproxy_cells.csv","w",newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
with open(HERE/"b13_gates.json","w") as fh:
    json.dump(dict(gates=gates, note="cond/c_norm/tau gates; recA_oos cross-check"), fh, indent=1)
print("\n[B13] wrote b13_supproxy_cells.csv (%d rows), b13_gates.json" % len(rows))
