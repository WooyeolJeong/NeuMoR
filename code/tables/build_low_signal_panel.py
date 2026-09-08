import sys, os, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NEUMOR = HERE.parents[1]
ROOT = NEUMOR.parent
CODE = NEUMOR / "code"
CACHE = HERE / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
LOG = HERE / "build.log"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "suite"))
sys.path.insert(0, str(CODE / "core"))
sys.path.insert(0, str(ROOT))

N_CAL, N_EVAL, N_SEEDS = 150, 50, 200
R_V2 = 20
RNG_SEED = 20260827
BAND = (0.7, 1.3)
TARGETS = [0.05, 0.15, 0.25, 0.40, 0.60, 0.80, 1.00]
BINS = [("B1", 0.0, 0.25), ("B2", 0.25, 0.50), ("B3", 0.50, 1.00)]
BIN_MIN_N = 15
Y0_WINDOW = 2.0
BFLY_WIDTHS = [0.25, 0.50, 1.00]

_logf = open(LOG, "a")
def log(*a):
    m = " ".join(str(x) for x in a)
    print(m, flush=True); _logf.write(m + "\n"); _logf.flush()

log("=" * 84)
log("[B10] X2-LS start")

from scipy.optimize import brentq
from scipy.stats import kurtosis
from statsmodels.stats.diagnostic import lilliefors

from NeuMoR.code.neumor_core import build_k_eta, compute_d_hat, effective_rank
import score_suite as b5
folded = b5.folded_normal_mean

BASE = {"v": 0.10, "kappa": 1.0, "omega": 0.2, "xi": 0.3, "rho": -0.5, "T": 1.0}
LAM = {
    "base":   dict(BASE),
    "P1_l2":  {**BASE, "v": 0.110},
    "P2_l2":  {**BASE, "kappa": 1.30},
    "P3_l2":  {**BASE, "v": 0.104, "kappa": 1.04, "omega": 0.208, "xi": 0.312, "rho": -0.48},
}
PAIRS = [
    ("P1", "base", "P1_l2", "Exp01-C",          "Exp01"),
    ("P2", "base", "P2_l2", "Exp05ax-kappa4",   "Exp05_axis"),
    ("P3", "base", "P3_l2", "Exp05dg-all5_s4",  "Exp05_diag"),
]

MANIFEST = CACHE / "manifest.json"
need = [k for k in LAM if not (CACHE / f"fields_{k}.npy").exists()]
if not need and MANIFEST.exists():
    log("[P1] cache complete -> SKIP inference")
    y_grid = np.load(CACHE / "y_grid.npy")
else:
    log(f"[P1] cache miss for {need} -> loading Protocol A ({N_SEEDS} seeds)")
    t0 = time.time()
    models, y_grid, pk = b5.load_heston_protocol("A", n_seeds=N_SEEDS)
    log(f"[P1] models loaded  n={len(models)}  n_y={len(y_grid)}  {time.time()-t0:.1f}s")
    t1 = time.time()
    for k, params in LAM.items():
        f = b5.infer_heston(models, params, y_grid, pk, cache_key=None)
        assert f.shape == (N_SEEDS, len(y_grid)), f.shape
        np.save(CACHE / f"fields_{k}.npy", f)
        log(f"[P1]   inferred {k:8s} -> fields_{k}.npy  {f.shape}")
    np.save(CACHE / "y_grid.npy", y_grid)
    MANIFEST.write_text(json.dumps(
        {"lambdas": LAM, "n_seeds": N_SEEDS, "n_y": int(len(y_grid)),
         "protocol": "A", "param_keys": list(pk)}, indent=2))
    log(f"[P1] inference done {time.time()-t1:.1f}s  (ONE pass, no further inference)")
    del models

FIELDS = {k: np.load(CACHE / f"fields_{k}.npy") for k in LAM}
dy = float(y_grid[1] - y_grid[0])
log(f"[P1] fields in memory; dy={dy:.6f}  n_y={len(y_grid)}")

_rng = np.random.default_rng(RNG_SEED)
V2_SPLITS = [_rng.permutation(N_SEEDS) for _ in range(R_V2)]

SE_EVAL = np.sqrt((1.0 - 2.0 / np.pi) / N_EVAL) / np.sqrt(2.0 / np.pi)
SE_SIG = 1.0 / np.sqrt(2.0 * N_CAL)
SE_PRED = float(np.sqrt(SE_EVAL ** 2 + SE_SIG ** 2))
log(f"[P1] analytic floor: se_eval={SE_EVAL:.6f} se_sigma={SE_SIG:.6f} se_pred={SE_PRED:.6f}")

def bin_of(s):
    for nm, lo, hi in BINS:
        if lo <= s < hi:
            return nm
    return "OUT"

class Pair:
    def __init__(self, tag, k1, k2, cid, blk):
        self.tag, self.cid, self.blk = tag, cid, blk
        F1, F2 = FIELDS[k1], FIELDS[k2]
        self.D = F1 - F2
        l1c, l2c = F1[:N_CAL], F2[:N_CAL]
        self.K = build_k_eta(l1c, l2c)
        self.d = compute_d_hat(l1c, l2c)
        self.reff = effective_rank(self.K)
        self.g_sig = self.d / np.linalg.norm(self.d)
        w, V = np.linalg.eigh(self.K)
        o = np.argsort(w)[::-1]
        self.w, self.V = w[o], V[:, o]
        self.v = {}
        for k in (1, 2):
            u = self.V[:, k - 1]
            u = u - self.g_sig * (self.g_sig @ u)
            self.v[k] = u / np.linalg.norm(u)

        d, y = self.d, y_grid
        m = np.abs(y) <= Y0_WINDOW
        idx = np.where(m)[0]
        cr, sl = [], []
        for i in idx[:-1]:
            if np.sign(d[i]) != np.sign(d[i + 1]) and d[i + 1] != d[i]:
                cr.append(float(y[i] - d[i] * (y[i + 1] - y[i]) / (d[i + 1] - d[i])))
                sl.append(float(abs((d[i + 1] - d[i]) / (y[i + 1] - y[i]))))
        self.n_cross = len(cr)
        if cr:
            o2 = np.argsort(sl)[::-1]
            self.y0 = cr[o2[0]]; self.y0_slope = sl[o2[0]]
            self.y0_2 = cr[o2[1]] if len(cr) > 1 else self.y0 + 0.5
            self.no_cross = False
        else:
            self.y0 = float(y[idx[np.argmin(np.abs(d[idx]))]]); self.y0_slope = float("nan")
            self.y0_2 = self.y0 + 0.5; self.no_cross = True
        self.s0 = self.s_of(0.0, 1)

    def delta_seeds(self, g):
        return self.D @ g * dy

    def stats(self, g, cal_idx=None, ev_idx=None):
        dlt = self.delta_seeds(g)
        c = dlt[:N_CAL] if cal_idx is None else dlt[cal_idx]
        e = dlt[N_CAL:N_SEEDS] if ev_idx is None else dlt[ev_idx]
        mu = float(c.mean()); sg = float(c.std(ddof=0))
        emp = float(np.abs(e).mean())
        th = folded(mu, sg)
        return mu, sg, emp, th, c

    def g_rot(self, th, k):
        return np.cos(th) * self.g_sig + np.sin(th) * self.v[k]

    def s_of(self, th, k):
        mu, sg, _, _, _ = self.stats(self.g_rot(th, k))
        return abs(mu) / sg if sg > 0 else float("inf")

    def theta_for(self, target, k):
        f = lambda t: self.s_of(t, k) - target
        return brentq(f, 0.0, np.pi / 2, xtol=1e-14, rtol=1e-15, maxiter=200)

P = {t: Pair(t, a, b, c, d) for t, a, b, c, d in PAIRS}
log("\n[P2] pair setup")
for t, _, _, _, _ in PAIRS:
    p = P[t]
    orth = max(abs(p.v[k] @ p.g_sig) for k in (1, 2))
    log(f"[P2] {t} {p.cid:18s} r_eff={p.reff:6.3f} s(theta=0)={p.s0:8.3f} "
        f"crossings={p.n_cross} y0={p.y0:+.4f} |d'|={p.y0_slope:.4e} "
        f"y0_2nd={p.y0_2:+.4f} no_cross={p.no_cross} max|<v,g>|={orth:.2e}")
    for k in (1, 2):
        r = float((p.v[k] @ p.K @ p.v[k]) / (p.g_sig @ p.K @ p.g_sig))
        log(f"[P2]    v_{k}: <v,Kv>/<g,Kg> = {r:.4e}")
        assert r > 1e-3, f"GATE R3a FAIL {t} k={k} ratio={r:.3e}"

eqmax = 0.0
for t in P:
    p = P[t]
    for g in (p.g_sig, p.v[1]):
        a = float(g @ p.K @ g) * dy ** 2
        b = float(p.delta_seeds(g)[:N_CAL].var(ddof=0))
        eqmax = max(eqmax, abs(a - b) / max(abs(b), 1e-300))
log(f"[P2] gate EQ  max rel|g@K@g dy^2 - var(Delta)| = {eqmax:.3e}")
assert eqmax < 1e-9, "GATE EQ FAIL"

def make_cell(p, family, knob, g, extra=None):
    mu, sg, emp, th, c = p.stats(g)
    ratio = emp / th if abs(th) > 1e-300 else float("nan")
    s_cal = abs(mu) / sg if sg > 0 else float("inf")
    v2 = []
    for perm in V2_SPLITS:
        ci, ei = perm[:N_CAL], perm[N_CAL:]
        m2, s2, e2, t2, _ = p.stats(g, ci, ei)
        v2.append(e2 / t2 if abs(t2) > 1e-300 else np.nan)
    v2 = np.asarray(v2, float)
    try:
        lp = float(lilliefors(c, dist="norm", pvalmethod="table")[1])
    except Exception:
        lp = float("nan")
    row = dict(
        cell_id=f"{p.tag}-{family}-{knob}", pair=p.tag, config_id=p.cid, block=p.blk,
        family=family, knob=knob, s_cal=s_cal, bin=bin_of(s_cal),
        ratio_V1=ratio, V2_median=float(np.nanmedian(v2)),
        V2_q10=float(np.nanquantile(v2, 0.10)), V2_q90=float(np.nanquantile(v2, 0.90)),
        err_folded=abs(th - emp) / emp, err_naive=abs(abs(mu) - emp) / emp,
        lillie_p=lp, ex_kurt=float(kurtosis(c, fisher=True, bias=False)),
        se_floor_eval=SE_EVAL, se_floor_sigma=SE_SIG, se_floor_pred=SE_PRED,
        mu_hat=mu, sigma_hat=sg, theory_mr=th, empirical_mr=emp,
        n_cal=N_CAL, n_eval=N_EVAL,
    )
    if extra:
        row.update(extra)
    return row

rows = []
log("\n[P2] R-group (rotation)")
for t in P:
    p = P[t]
    rows.append(make_cell(p, "R", "theta0", p.g_rot(0.0, 1),
                          {"theta": 0.0, "k": 0, "anchor": "theta0"}))
    for k in (1, 2):
        for tg in TARGETS:
            th = p.theta_for(tg, k)
            rows.append(make_cell(p, "R", f"th{tg:.2f}_k{k}", p.g_rot(th, k),
                                  {"theta": th, "k": k, "s_target": tg}))
log(f"[P2]   R base cells = {len(rows)}")

log("[P2] I-group (instruments)")
for t in P:
    p = P[t]
    flag = "_nocross" if p.no_cross else ""
    for w in BFLY_WIDTHS:
        g = np.maximum(0.0, 1.0 - np.abs(y_grid - p.y0) / w)
        rows.append(make_cell(p, "I", f"bfly_w{w:.2f}{flag}", g,
                              {"y0": p.y0, "width": w, "instrument": "butterfly"}))
    for j, K in enumerate([p.y0, p.y0_2], 1):
        g = (y_grid >= K).astype(float)
        rows.append(make_cell(p, "I", f"dig_K{K:+.3f}{flag}", g,
                              {"y0": p.y0, "strike": K, "instrument": "digital"}))
log(f"[P2]   total after I = {len(rows)}")

def counts():
    c = {b: 0 for b, _, _ in BINS}
    for r in rows:
        if r["bin"] in c:
            c[r["bin"]] += 1
    return c

log(f"[P2] bin counts (base) = {counts()}")
added = 0
for nm, lo, hi in BINS:
    tset = sorted({r["s_target"] for r in rows
                   if r.get("s_target") is not None and lo <= r["s_target"] < hi})
    guard = 0
    while counts()[nm] < BIN_MIN_N:
        guard += 1
        assert guard <= 12, f"GATE bin-fill FAIL {nm}: no convergence"
        edges = [lo] + tset + [hi]
        gaps = [(edges[i + 1] - edges[i], i) for i in range(len(edges) - 1)]
        _, gi = max(gaps)
        newt = 0.5 * (edges[gi] + edges[gi + 1])
        if newt <= lo or newt >= hi:
            break
        for t in P:
            for k in (1, 2):
                th = P[t].theta_for(newt, k)
                rows.append(make_cell(P[t], "R", f"th{newt:.4f}_k{k}_bisect",
                                      P[t].g_rot(th, k),
                                      {"theta": th, "k": k, "s_target": newt,
                                       "bisect": True}))
                added += 1
        tset = sorted(set(tset + [newt]))
        log(f"[P2]   bisect {nm}: new target |s|={newt:.4f} -> n={counts()[nm]}")
log(f"[P2] bin counts (filled) = {counts()}  (+{added} bisect cells)")
for nm, _, _ in BINS:
    assert counts()[nm] >= BIN_MIN_N, f"GATE bin FAIL {nm}={counts()[nm]}<{BIN_MIN_N}"

log("\n[P2] sanity")
anch = [r for r in rows if r.get("anchor") == "theta0"]
ok_i = True
for r in anch:
    good = 0.95 <= r["ratio_V1"] <= 1.05
    ok_i &= good
    log(f"[S-i] {r['pair']} theta=0  s_cal={r['s_cal']:8.3f} ratio_V1={r['ratio_V1']:.6f} "
        f"err_folded={r['err_folded']:.6f} err_naive={r['err_naive']:.6f} "
        f"|diff|={abs(r['err_folded']-r['err_naive']):.3e}  {'PASS' if good else 'FAIL'}")
log(f"[S-i] anchors {'PASS' if ok_i else 'FAIL'} ({sum(1 for r in anch if 0.95<=r['ratio_V1']<=1.05)}/{len(anch)})")

p0 = P["P1"]; g0 = p0.g_rot(p0.theta_for(0.15, 1), 1)
r1 = p0.stats(g0)[2] / p0.stats(g0)[3]
r2 = p0.stats(2 * g0)[2] / p0.stats(2 * g0)[3]
scale_err = abs(r1 - r2)
ok_ii = scale_err < 1e-10
log(f"[S-ii] scale invariance g->2g: ratio {r1:.15f} vs {r2:.15f} |diff|={scale_err:.3e} "
    f"{'PASS' if ok_ii else 'FAIL'}")
assert ok_i and ok_ii, "SANITY FAIL -> HOLD"

df = pd.DataFrame(rows)
COLS = ["cell_id", "pair", "config_id", "block", "family", "knob", "s_cal", "bin",
        "ratio_V1", "V2_median", "V2_q10", "V2_q90", "err_folded", "err_naive",
        "lillie_p", "ex_kurt", "se_floor_eval", "se_floor_sigma", "se_floor_pred",
        "theta", "k", "s_target", "bisect", "anchor", "y0", "width", "strike",
        "instrument", "mu_hat", "sigma_hat", "theory_mr", "empirical_mr",
        "n_cal", "n_eval"]
for c in COLS:
    if c not in df.columns:
        df[c] = np.nan
df = df[COLS].sort_values(["pair", "family", "s_cal"]).reset_index(drop=True)
df.to_csv(HERE / "b10_lowsnr_all.csv", index=False)
log(f"\n[OUT] b10_lowsnr_all.csv  rows={len(df)}")

def blk(sub):
    inb = ((sub.ratio_V1 >= BAND[0]) & (sub.ratio_V1 <= BAND[1])).mean()
    v2b = ((sub.V2_median >= BAND[0]) & (sub.V2_median <= BAND[1])).mean()
    return dict(n=len(sub),
                ratioV1_in_band=inb, V2med_in_band=v2b,
                V2med_min=sub.V2_median.min(), V2med_max=sub.V2_median.max(),
                V2med_median=sub.V2_median.median(),
                err_folded_median=sub.err_folded.median(),
                err_naive_median=sub.err_naive.median(),
                s_cal_min=sub.s_cal.min(), s_cal_max=sub.s_cal.max(),
                lillie_rej05=int((sub.lillie_p < 0.05).sum()),
                ex_kurt_min=sub.ex_kurt.min(), ex_kurt_max=sub.ex_kurt.max())

srows = []
for nm, _, _ in BINS + [("OUT", None, None)]:
    for fam in ["R", "I"]:
        sub = df[(df["bin"] == nm) & (df.family == fam)]
        if len(sub):
            srows.append(dict(bin=nm, family=fam, **blk(sub)))
    sub = df[df["bin"] == nm]
    if len(sub):
        srows.append(dict(bin=nm, family="ALL", **blk(sub)))
srows.append(dict(bin="__TOTAL__", family="ALL", **blk(df)))
sm = pd.DataFrame(srows)
sm.to_csv(HERE / "b10_lowsnr_summary.csv", index=False)
log(f"[OUT] b10_lowsnr_summary.csv  rows={len(sm)}")

unmet = []
cc = counts()
for nm, _, _ in BINS:
    if cc[nm] < BIN_MIN_N:
        unmet.append(f"bin {nm} n={cc[nm]}<{BIN_MIN_N}")
v2in = ((df.V2_median >= BAND[0]) & (df.V2_median <= BAND[1])).mean()
if v2in < 0.90:
    unmet.append(f"V2_median in band {v2in:.1%}<90%")
for nm, _, _ in BINS:
    m = df[df["bin"] == nm].err_folded.median()
    if not (m <= 0.10):
        unmet.append(f"bin {nm} err_folded median {m:.4f}>0.10")
verdict = "PASS" if not unmet else "PARTIAL"
log("\n" + "=" * 84)
log(f"[VERDICT] {verdict}" + ("" if not unmet else "  unmet: " + "; ".join(unmet)))
log("=" * 84)

print("\n===== b10_lowsnr_summary.csv =====")
print(sm.to_string(index=False))
json.dump({"verdict": verdict, "unmet": unmet, "n_cells": int(len(df)),
           "bins": {k: int(v) for k, v in cc.items()},
           "sanity_i": bool(ok_i), "sanity_ii": bool(ok_ii),
           "scale_err": scale_err, "eq_gate": eqmax,
           "y0": {t: {"y0": P[t].y0, "n_cross": P[t].n_cross,
                      "slope": P[t].y0_slope, "y0_2nd": P[t].y0_2,
                      "no_cross": P[t].no_cross, "s_theta0": P[t].s0} for t in P}},
          open(HERE / "verdict.json", "w"), indent=2)
log("[OUT] verdict.json")
