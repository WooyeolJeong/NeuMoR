import sys, os, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NEUMOR = HERE.parents[1]; ROOT = NEUMOR.parent; CODE = NEUMOR / "code"
CACHE = HERE / "cache"
ALL = HERE / "b10_lowsnr_all.csv"; SUM = HERE / "b10_lowsnr_summary.csv"
VJ = HERE / "verdict.json"; LOG = HERE / "build.log"
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / "suite"))
sys.path.insert(0, str(CODE / "core")); sys.path.insert(0, str(ROOT))

N_CAL, N_EVAL, N_SEEDS = 150, 50, 200
R_V2, RNG_SEED = 20, 20260827
BAND = (0.7, 1.3); BIN_MIN_N = 15
BINS = [("B1", 0.0, 0.25), ("B2", 0.25, 0.50), ("B3", 0.50, 1.00)]
K_WINDOW = 1.5
MASS_MIN = 0.02
WIDTH_MIN_CELLS = 3
I2_TARGETS = [0.05, 0.15, 0.30, 0.60, 1.00]
MATCH_TOL = 0.05

_logf = open(LOG, "a")
def log(*a):
    m = " ".join(str(x) for x in a)
    print(m, flush=True); _logf.write(m + "\n"); _logf.flush()

log("=" * 84); log("[B10-I2] digital-SPREAD addendum (cache-only)")

from scipy.stats import kurtosis
from statsmodels.stats.diagnostic import lilliefors
from NeuMoR.code.neumor_core import compute_d_hat
import score_suite as b5
folded = b5.folded_normal_mean

PAIRKEY = {"P1": ("base", "P1_l2", "Exp01-C", "Exp01"),
           "P2": ("base", "P2_l2", "Exp05ax-kappa4", "Exp05_axis"),
           "P3": ("base", "P3_l2", "Exp05dg-all5_s4", "Exp05_diag")}

y = np.load(CACHE / "y_grid.npy"); dy = float(y[1] - y[0]); n_y = len(y)
FLD = {k: np.load(CACHE / f"fields_{k}.npy") for k in ["base", "P1_l2", "P2_l2", "P3_l2"]}
log(f"[I2] cache loaded -> NO inference.  n_y={n_y} dy={dy:.6f}")

_rng = np.random.default_rng(RNG_SEED)
V2_SPLITS = [_rng.permutation(N_SEEDS) for _ in range(R_V2)]
SE_EVAL = np.sqrt((1.0 - 2.0 / np.pi) / N_EVAL) / np.sqrt(2.0 / np.pi)
SE_SIG = 1.0 / np.sqrt(2.0 * N_CAL)
SE_PRED = float(np.sqrt(SE_EVAL ** 2 + SE_SIG ** 2))

def bin_of(s):
    for nm, lo, hi in BINS:
        if lo <= s < hi:
            return nm
    return "OUT"

raw0 = ALL.read_text()
old = pd.read_csv(ALL)
COLS = list(old.columns)
assert len(old) == 72 and "I2" not in set(old.family), f"base state unexpected: {len(old)}"
assert raw0.endswith("\n"), "base csv must end with newline for text append"
n_lines0 = raw0.count("\n")
log(f"[I2] base csv: {len(old)} rows / {n_lines0} lines, families={sorted(set(old.family))}")

rows, meta = [], {}
for tag, (k1k, k2k, cid, blk) in PAIRKEY.items():
    F1, F2 = FLD[k1k], FLD[k2k]
    D = F1 - F2
    d = compute_d_hat(F1[:N_CAL], F2[:N_CAL])
    pbar = 0.5 * (F1[:N_CAL].mean(0) + F2[:N_CAL].mean(0))

    S = np.concatenate([np.zeros((N_SEEDS, 1)), np.cumsum(D, axis=1)], axis=1)
    Pc = np.concatenate([[0.0], np.cumsum(pbar)])
    Fcum = np.concatenate([[0.0], np.cumsum(d)])

    cand = [int(i) for i in np.where(np.abs(y) <= K_WINDOW)[0]]
    A, B = [], []
    for ai in range(len(cand)):
        for bi in range(ai + WIDTH_MIN_CELLS, len(cand)):
            a, b = cand[ai], cand[bi]
            if b - a >= WIDTH_MIN_CELLS:
                A.append(a); B.append(b)
    A = np.asarray(A); B = np.asarray(B)

    delta = (S[:, B + 1] - S[:, A + 1]) * dy
    mass = (Pc[B + 1] - Pc[A + 1]) * dy
    cal = delta[:N_CAL]; ev = delta[N_CAL:N_SEEDS]
    mu = cal.mean(0); sg = cal.std(0, ddof=0)
    ok = (mass >= MASS_MIN) & (sg > 0)
    s_all = np.where(ok, np.abs(mu) / np.where(sg > 0, sg, 1.0), np.inf)
    log(f"[I2] {tag} {cid}: combos={len(A)}  screened-in={int(ok.sum())} "
        f"(mass>={MASS_MIN}, width>={WIDTH_MIN_CELLS}*dy)  |s| range "
        f"[{s_all[ok].min():.4f}, {s_all[ok].max():.4f}]")

    chosen, unmet, ach = [], [], []
    for t in I2_TARGETS:
        candi = np.where(ok)[0]
        candi = np.array([c for c in candi if c not in chosen])
        j = candi[np.argmin(np.abs(s_all[candi] - t))]
        err = abs(s_all[j] - t)
        if err > MATCH_TOL:
            unmet.append({"target": t, "best_s": float(s_all[j]), "match_err": float(err)})
            log(f"[I2] {tag}   target |s|={t:.2f} -> UNMET (best {s_all[j]:.4f}, err {err:.4f} > {MATCH_TOL})")
            continue
        chosen.append(int(j)); ach.append((t, int(j), float(s_all[j]), float(err)))

    meta[tag] = {"n_combos": int(len(A)), "n_screened": int(ok.sum()),
                 "s_min": float(s_all[ok].min()), "unmet": unmet,
                 "achieved": [{"target": t, "K1": float(y[A[j]]), "K2": float(y[B[j]]),
                               "s_cal": s, "match_err": e} for t, j, s, e in ach]}

    for t, j, sj, ej in ach:
        a, b = int(A[j]), int(B[j])
        K1, K2 = float(y[a]), float(y[b])
        dl = delta[:, j]
        c, e = dl[:N_CAL], dl[N_CAL:N_SEEDS]
        m_, s_ = float(c.mean()), float(c.std(ddof=0))
        emp = float(np.abs(e).mean()); th = folded(m_, s_)

        FK1 = float(Fcum[-1] - Fcum[a + 1]) * dy
        FK2 = float(Fcum[-1] - Fcum[b + 1]) * dy
        rel = abs(m_ - (FK1 - FK2)) / max(abs(m_), 1e-300)
        assert rel < 1e-12, f"GATE SP FAIL {tag} ({K1},{K2}) rel={rel:.3e}"
        v2 = []
        for perm in V2_SPLITS:
            ci, ei = perm[:N_CAL], perm[N_CAL:]
            c2 = dl[ci]; m2, s2 = float(c2.mean()), float(c2.std(ddof=0))
            e2 = float(np.abs(dl[ei]).mean()); t2 = folded(m2, s2)
            v2.append(e2 / t2 if abs(t2) > 1e-300 else np.nan)
        v2 = np.asarray(v2, float)
        try:
            lp = float(lilliefors(c, dist="norm", pvalmethod="table")[1])
        except Exception:
            lp = float("nan")
        rows.append({
            "cell_id": f"{tag}-I2-spr_K{K1:+.3f}_{K2:+.3f}", "pair": tag,
            "config_id": cid, "block": blk, "family": "I2",
            "knob": f"spread_K1{K1:+.4f}_K2{K2:+.4f}_starg{t:.2f}",
            "s_cal": sj, "bin": bin_of(sj),
            "ratio_V1": emp / th if abs(th) > 1e-300 else np.nan,
            "V2_median": float(np.nanmedian(v2)),
            "V2_q10": float(np.nanquantile(v2, .10)),
            "V2_q90": float(np.nanquantile(v2, .90)),
            "err_folded": abs(th - emp) / emp, "err_naive": abs(abs(m_) - emp) / emp,
            "lillie_p": lp, "ex_kurt": float(kurtosis(c, fisher=True, bias=False)),
            "se_floor_eval": SE_EVAL, "se_floor_sigma": SE_SIG, "se_floor_pred": SE_PRED,
            "theta": np.nan, "k": np.nan, "s_target": t, "bisect": np.nan,
            "anchor": np.nan, "y0": np.nan, "width": K2 - K1, "strike": K1,
            "instrument": "digital_spread", "mu_hat": m_, "sigma_hat": s_,
            "theory_mr": th, "empirical_mr": emp, "n_cal": N_CAL, "n_eval": N_EVAL})
        log(f"[I2] {tag}   target |s|={t:.2f} -> K1={K1:+.4f} K2={K2:+.4f} "
            f"(w={K2-K1:.4f}, mass={mass[j]:.4f})  s_cal={sj:.4f} err={ej:.4f}  "
            f"ratio_V1={rows[-1]['ratio_V1']:.6f}  bin={bin_of(sj)}")

assert rows, "GATE FAIL: no I2 spread cells accepted"
log(f"[I2] gate SP (mu == F(K1)-F(K2)) PASS for all {len(rows)} accepted cells (rtol 1e-12)")

new = pd.DataFrame(rows)[COLS]
with open(ALL, "a") as f:
    f.write(new.to_csv(index=False, header=False))
raw1 = ALL.read_text()
assert raw1.startswith(raw0), "GATE FAIL: base text not preserved"
log(f"\n[OUT] b10_lowsnr_all.csv  +{len(new)} rows appended "
    f"(base {n_lines0} lines byte-identical prefix: PASS)")

df = pd.read_csv(ALL)

def blk(sub):
    return dict(n=len(sub),
                ratioV1_in_band=((sub.ratio_V1 >= BAND[0]) & (sub.ratio_V1 <= BAND[1])).mean(),
                V2med_in_band=((sub.V2_median >= BAND[0]) & (sub.V2_median <= BAND[1])).mean(),
                V2med_min=sub.V2_median.min(), V2med_max=sub.V2_median.max(),
                V2med_median=sub.V2_median.median(),
                err_folded_median=sub.err_folded.median(),
                err_naive_median=sub.err_naive.median(),
                s_cal_min=sub.s_cal.min(), s_cal_max=sub.s_cal.max(),
                lillie_rej05=int((sub.lillie_p < 0.05).sum()),
                ex_kurt_min=sub.ex_kurt.min(), ex_kurt_max=sub.ex_kurt.max())

srows = []
for nm in [b[0] for b in BINS] + ["OUT"]:
    for fam in ["R", "I", "I2"]:
        sub = df[(df["bin"] == nm) & (df.family == fam)]
        if len(sub):
            srows.append(dict(bin=nm, family=fam, **blk(sub)))
    sub = df[df["bin"] == nm]
    if len(sub):
        srows.append(dict(bin=nm, family="ALL", **blk(sub)))
srows.append(dict(bin="__TOTAL__", family="ALL", **blk(df)))
sm = pd.DataFrame(srows); sm.to_csv(SUM, index=False)
log(f"[OUT] b10_lowsnr_summary.csv regenerated  rows={len(sm)}")

cc = {nm: int((df["bin"] == nm).sum()) for nm, _, _ in BINS}
unmet_v = [f"bin {nm} n={cc[nm]}<{BIN_MIN_N}" for nm, _, _ in BINS if cc[nm] < BIN_MIN_N]
v2in = ((df.V2_median >= BAND[0]) & (df.V2_median <= BAND[1])).mean()
if v2in < 0.90:
    unmet_v.append(f"V2_median in band {v2in:.1%}<90%")
for nm, _, _ in BINS:
    m = df[df["bin"] == nm].err_folded.median()
    if not (m <= 0.10):
        unmet_v.append(f"bin {nm} err_folded median {m:.4f}>0.10")
verdict = "PASS" if not unmet_v else "PARTIAL"
log("\n" + "=" * 84)
log(f"[VERDICT incl. I2] {verdict}" + ("" if not unmet_v else "  unmet: " + "; ".join(unmet_v)))
log("=" * 84)

vj = json.loads(VJ.read_text())
assert vj.get("verdict") == "PASS"
vj["I2_spread"] = {"verdict_incl_I2": verdict, "unmet_incl_I2": unmet_v,
                   "n_cells_incl_I2": int(len(df)), "bins_incl_I2": cc,
                   "screens": {"K_window": K_WINDOW, "mass_min": MASS_MIN,
                               "width_min_cells": WIDTH_MIN_CELLS,
                               "match_tol": MATCH_TOL},
                   "per_pair": meta,
                   "single_digital_negative": {
                       "s_floor": {"P1": 0.1514, "P2": 0.6061, "P3": 0.7594},
                       "note": "grid quantization of mu(K)=F(K); P2 min at window edge "
                               "K=-1.99 is a normalization-degenerate near-total digital"}}
VJ.write_text(json.dumps(vj, indent=2))
log("[OUT] verdict.json  += I2_spread (existing verdict fields untouched)")

print("\n===== panel_summary.csv (rebuilt, including the digital spreads) =====")
print(sm.to_string(index=False))
