import sys, os, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NEUMOR = HERE.parents[1]
ROOT = NEUMOR.parent
CODE = NEUMOR / "code"
CACHE = HERE / "cache"
NPZ = HERE / "b9_perseed_deltas.npz"
IDX = HERE / "b9_perseed_index.csv"
LOG = HERE / "build.log"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "suite"))
sys.path.insert(0, str(CODE / "core"))
R_V2 = 20
RNG_SEED = 20260826
BAND = (0.7, 1.3)

_logf = open(LOG, "a")
def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True); _logf.write(msg + "\n"); _logf.flush()

log("=" * 80)
log(f"[B9] start {time.strftime('%H:%M:%S') if False else '(ts via args n/a)'}  NEUMOR={NEUMOR.name}")

import score_suite as b5
folded = b5.folded_normal_mean

if NPZ.exists() and IDX.exists():
    log("[P1] cache found -> skip inference, load per-seed deltas")
    z = np.load(NPZ); deltas = {k: z[k] for k in z.files}; z.close()
    idx = pd.read_csv(IDX)
else:
    log("[P1] no cache -> running b5.main() with measure_block capture + to_csv no-op")
    CAP = {}
    _orig_measure = b5.measure_block
    def patched_measure(l1_all, l2_all, y, payoff_fn, block, config_id,
                        n_cal, n_test_lo, n_test_hi, extra=None):
        dy = float(y[1] - y[0]); diff = (l1_all - l2_all)
        for pn in b5.SMOOTH_PAYOFFS:
            g = payoff_fn(pn, y)
            CAP[f"{block}__{config_id}__{pn}"] = dict(
                delta=(diff @ g * dy).astype(np.float64), n_cal=int(n_cal),
                tlo=int(n_test_lo), thi=int(n_test_hi), nseed=int(l1_all.shape[0]))
        return _orig_measure(l1_all, l2_all, y, payoff_fn, block, config_id,
                             n_cal, n_test_lo, n_test_hi, extra)
    b5.measure_block = patched_measure
    _orig_tocsv = pd.DataFrame.to_csv
    pd.DataFrame.to_csv = lambda self, *a, **k: None
    t0 = time.time()
    try:
        b5.main()
    except SystemExit:
        pass
    except Exception as e:
        log(f"[P1] main() ended with {type(e).__name__}: {str(e)[:120]} (capture may be complete)")
    pd.DataFrame.to_csv = _orig_tocsv
    log(f"[P1] inference done in {time.time()-t0:.0f}s; captured {len(CAP)} rows")
    deltas = {k: v["delta"] for k, v in CAP.items()}
    np.savez_compressed(NPZ, **deltas)
    idx = pd.DataFrame([dict(key=k, block=k.split("__")[0], config_id=k.split("__")[1],
        payoff=k.split("__")[2], n_cal=v["n_cal"], n_test_lo=v["tlo"],
        n_test_hi=v["thi"], n_seeds=v["nseed"]) for k, v in CAP.items()])
    idx.to_csv(IDX, index=False)
    log(f"[P1] saved cache {NPZ.name} ({NPZ.stat().st_size/1e3:.0f}KB) + index {IDX.name}")

log(f"[COV] captured rows: {len(deltas)} (expect 387)")
b5csv = pd.read_csv(NEUMOR / "output" / "unified" / "b5_thm35_all.csv")
b5csv["key"] = b5csv.block + "__" + b5csv.config_id + "__" + b5csv.payoff
b5map = dict(zip(b5csv.key, b5csv.ratio))
b5block = dict(zip(b5csv.key, b5csv.block))
missing = [k for k in b5csv.key if k not in deltas]
if len(deltas) != 387 or missing:
    log(f"[COV] COVERAGE FAIL: have {len(deltas)}, b5 has {len(b5csv)}; missing {len(missing)}")
    for k in missing[:20]: log("   missing:", k)
    from collections import Counter
    log("   missing by block:", dict(Counter(b5block.get(k, '?') for k in missing)))
    log("[COV] STOP — coverage incomplete."); sys.exit(2)
log("[COV] PASS — all 387 recoverable")

idx = idx.set_index("key")

def split_of(key):
    r = idx.loc[key]; d = deltas[key]
    return d[:int(r.n_cal)], d[int(r.n_test_lo):int(r.n_test_hi)], int(r.n_cal), int(r.n_test_hi - r.n_test_lo)

def ratio_from(mu_src, sig_src, emp_src):
    mu = float(mu_src.mean()); sig = float(sig_src.std(ddof=0)); emp = float(np.abs(emp_src).mean())
    th = folded(mu, sig)
    return (emp / th) if abs(th) > 1e-20 else float("nan")

log("[P2] legacy recompute (sigma from TEST) -> sanity vs b5 ratio (|diff|<=0.001)")
sanity_rows = []; max_diff = 0.0; n_fail = 0
for key in b5csv.key:
    cal, test, ncal, ntest = split_of(key)
    ratio_leg = ratio_from(cal, test, test)
    rb5 = float(b5map[key]); diff = abs(ratio_leg - rb5)
    max_diff = max(max_diff, diff); n_fail += (diff > 0.001)
    sanity_rows.append(dict(key=key, ratio_b5=rb5, ratio_legacy=ratio_leg, absdiff=diff))
log(f"[P2] sanity(i): max|ratio_legacy - ratio_b5| = {max_diff:.2e} ; fails(>0.001) = {n_fail}")
if n_fail > 0:
    sf = pd.DataFrame([r for r in sanity_rows if r["absdiff"] > 0.001]).sort_values("absdiff", ascending=False)
    sf.to_csv(HERE / "b9_sanity_fail.csv", index=False)
    log("[P2] SANITY (i) FAIL — offending rows (top 20):")
    for r in sf.head(20).to_dict("records"): log(f"   {r['key']}  b5={r['ratio_b5']:.4f} legacy={r['ratio_legacy']:.4f} d={r['absdiff']:.2e}")
    log("[P2] STOP — pipeline-equivalence gate failed; V1/V2 NOT written."); sys.exit(3)
log("[P2] sanity(i) PASS — full checkpoint->field->Delta recovery path validated")

rng = np.random.default_rng(RNG_SEED)
rows = []
for key in sorted(b5csv.key):
    r = idx.loc[key]; d = deltas[key]; ncal = int(r.n_cal); ntest = int(r.n_test_hi - r.n_test_lo)
    cal, test, _, _ = split_of(key)
    ratio_V1 = ratio_from(cal, cal, test)

    nseed = int(r.n_seeds); reps = np.zeros(R_V2)
    for j in range(R_V2):
        perm = rng.permutation(nseed); P = d[perm[:ncal]]; Q = d[perm[ncal:ncal + ntest]]
        r1 = ratio_from(P, P, Q)
        r2 = ratio_from(Q, Q, P)
        reps[j] = 0.5 * (r1 + r2)
    q10, med, q90 = np.percentile(reps, [10, 50, 90])
    rows.append(dict(config_id=r.config_id, block=r.block, payoff=r.payoff,
        ratio_b5=float(b5map[key]), ratio_V1=float(ratio_V1),
        V2_median=float(med), V2_q10=float(q10), V2_q90=float(q90),
        n_cal=ncal, n_test=ntest))
alldf = pd.DataFrame(rows)
alldf.to_csv(HERE / "b9_crossfit_all.csv", index=False)
log(f"[P2] wrote b9_crossfit_all.csv ({len(alldf)} rows)")

def band_in(x): return int(((x >= BAND[0]) & (x <= BAND[1])).sum())
srows = []
for blk, g in list(alldf.groupby("block")) + [("__TOTAL__", alldf)]:
    dV1 = g.ratio_V1; dM = g.V2_median; dd = (g.ratio_V1 - g.ratio_b5).abs()
    srows.append(dict(block=blk, n=len(g),
        ratioV1_min=float(dV1.min()), ratioV1_max=float(dV1.max()), ratioV1_median=float(dV1.median()),
        V2med_min=float(dM.min()), V2med_max=float(dM.max()), V2med_median=float(dM.median()),
        in_band_0713=band_in(dV1), dratio_med=float(dd.median()), dratio_max=float(dd.max())))
sumdf = pd.DataFrame(srows)
sumdf.to_csv(HERE / "b9_crossfit_summary.csv", index=False)
log(f"[P2] wrote b9_crossfit_summary.csv ({len(sumdf)} rows)")

oob = alldf[(alldf.ratio_V1 < BAND[0]) | (alldf.ratio_V1 > BAND[1])]
log(f"[SAN-ii] V1 out-of-[0.7,1.3] configs: {len(oob)}")
for r in oob.to_dict("records"):
    log(f"   OOB: {r['block']}/{r['config_id']}/{r['payoff']}  ratio_V1={r['ratio_V1']:.4f} (b5={r['ratio_b5']:.4f})")

log("\n===== b9_crossfit_summary.csv =====")
with pd.option_context("display.width", 200, "display.max_columns", 20):
    log(sumdf.round(4).to_string(index=False))
tot = sumdf[sumdf.block == "__TOTAL__"].iloc[0]
log(f"\n[DONE] V1 Total range [{tot.ratioV1_min:.4f}, {tot.ratioV1_max:.4f}] median {tot.ratioV1_median:.4f} | "
    f"in-band {int(tot.in_band_0713)}/387 | |Δratio| med {tot.dratio_med:.4f} max {tot.dratio_max:.4f} | "
    f"V2med range [{tot.V2med_min:.4f}, {tot.V2med_max:.4f}]")
log("[DONE] B9 complete.")
_logf.close()
