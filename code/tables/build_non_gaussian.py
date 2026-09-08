import sys, os, time, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import norm, t as student_t, laplace, gamma as gammadist
from scipy.special import gammaln

HERE = Path(__file__).resolve().parent
NEUMOR = HERE.parents[1]
sys.path.insert(0, str(NEUMOR / "code"))
sys.path.insert(0, str(NEUMOR / "code" / "suite"))
sys.path.insert(0, str(NEUMOR / "code" / "core"))
import score_suite as b5
folded = b5.folded_normal_mean

PHI = norm.pdf
def He3(z): return z**3 - 3*z
def He4(z): return z**4 - 6*z**2 + 3
S2PI = np.sqrt(2.0/np.pi)
QOPT = dict(epsabs=1e-12, epsrel=1e-10, limit=400)

def kink_quad(fn, mu):
    a, _ = integrate.quad(fn, -np.inf, -mu, **QOPT)
    b, _ = integrate.quad(fn, -mu, np.inf, **QOPT)
    return a + b

def E_true_quad(mu, pdf):
    return kink_quad(lambda z: abs(mu + z) * pdf(z), mu)

def C_k(mu, k):
    He = He3 if k == 3 else He4
    return kink_quad(lambda z: abs(mu + z) * He(z) * PHI(z), mu)

def W1_to_normal(cdf, atom_at=None):
    g = lambda x: abs(cdf(x) - norm.cdf(x))
    pts = sorted(set([0.0] + ([atom_at] if atom_at is not None else [])))
    lims = [-np.inf] + pts + [np.inf]; tot = 0.0
    for lo, hi in zip(lims[:-1], lims[1:]):
        v, _ = integrate.quad(g, lo, hi, **QOPT); tot += v
    return tot

def make_t(nu):
    sc = np.sqrt((nu - 2.0) / nu)
    return dict(name=f"t{nu}", g1=0.0, g2=6.0/(nu-4.0),
                pdf=lambda z: student_t.pdf(z/sc, nu)/sc,
                cdf=lambda x: student_t.cdf(x/sc, nu), atom=None,
                E0_closed=(np.exp(gammaln((nu+1)/2)-gammaln(nu/2))*2*np.sqrt(nu)/((nu-1)*np.sqrt(np.pi)))*sc)

def make_laplace():
    b = 1.0/np.sqrt(2.0)
    return dict(name="Laplace", g1=0.0, g2=3.0,
                pdf=lambda z: laplace.pdf(z, scale=b),
                cdf=lambda x: laplace.cdf(x, scale=b), atom=None, E0_closed=b)

def make_zi(p):
    sig = 1.0/np.sqrt(p)
    return dict(name=f"ZI{p}", g1=0.0, g2=3.0/p - 3.0, p=p, sig=sig, atom=0.0,
                pdf=lambda z: p*norm.pdf(z, scale=sig),
                cdf=lambda x: (1-p)*(x >= 0) + p*norm.cdf(x, scale=sig),
                E_true=lambda mu: (1-p)*abs(mu) + p*folded(mu, sig),
                E0_closed=(0.0*0 + (1-p)*0 + p*folded(0.0, sig)))

def make_gamma(k, sign):
    sk = np.sqrt(k)
    def pdf(z):
        g = sign*z*sk + k
        return np.where(g > 0, gammadist.pdf(np.maximum(g, 1e-300), k)*sk, 0.0)
    def cdf(x):
        if sign > 0: return gammadist.cdf(np.maximum(x*sk + k, 0.0), k)
        return 1.0 - gammadist.cdf(np.maximum(-x*sk + k, 0.0), k)
    return dict(name=f"gamma{k}{'p' if sign>0 else 'm'}", g1=sign*2.0/sk, g2=6.0/k,
                pdf=pdf, cdf=cdf, atom=None, E0_closed=None)

FAMILIES = [dict(name="Gaussian", g1=0.0, g2=0.0, pdf=norm.pdf, cdf=norm.cdf, atom=None, E0_closed=S2PI)]
FAMILIES += [make_t(nu) for nu in [4.5, 5, 6, 8, 12, 20, 50]]
FAMILIES += [make_laplace()]
FAMILIES += [make_zi(p) for p in [0.5, 0.2, 0.1, 0.05]]
FAMILIES += [make_gamma(k, sgn) for k in [2, 8] for sgn in [+1, -1]]

S_GRID = [0, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0, 1.5, 2, 3, 5]

print("="*80); print(f"[B11] {len(FAMILIES)} families x {len(S_GRID)} s = {len(FAMILIES)*len(S_GRID)} cells")
t0 = time.time()

W1 = {}
for fam in FAMILIES:
    W1[fam["name"]] = 0.0 if fam["name"] == "Gaussian" else W1_to_normal(fam["cdf"], fam.get("atom"))

rows = []
for fam in FAMILIES:
    w1 = W1[fam["name"]]
    for s in S_GRID:
        mu = float(s)
        if "E_true" in fam:
            E_true = fam["E_true"](mu)
        elif fam["name"] == "Gaussian":
            E_true = folded(mu, 1.0)
        else:
            E_true = E_true_quad(mu, fam["pdf"])
        m_f = folded(mu, 1.0)
        delta = E_true - m_f
        c3 = C_k(mu, 3); c4 = C_k(mu, 4)
        E_edge = m_f + fam["g1"]/6.0*c3 + fam["g2"]/24.0*c4
        edge_valid = bool(E_edge > 0)
        edge_capture = (1.0 - abs(E_edge - E_true)/abs(delta)) if abs(delta) > 1e-9 else np.nan
        rows.append(dict(family=fam["name"], gamma1=fam["g1"], gamma2=fam["g2"], s=s, mu=mu, sigma=1.0,
            E_true=E_true, m_folded=m_f, delta=delta, delta_rel=delta/E_true if E_true > 0 else np.nan,
            ratio_true=E_true/m_f if m_f > 0 else np.nan, W1=w1,
            bound_ok=bool(abs(delta) <= w1 + 1e-9), slack=(abs(delta)/w1 if w1 > 1e-15 else np.nan),
            C3=c3, C4=c4, E_edge=E_edge, edge_valid=edge_valid, edge_capture=edge_capture,
            ratio_edge=E_edge/m_f if m_f > 0 else np.nan))
cells = pd.DataFrame(rows)
cells.to_csv(HERE/"b11_cells.csv", index=False)
print(f"[B11] cells computed in {time.time()-t0:.0f}s -> b11_cells.csv ({len(cells)} rows)")

G = {}
gau = cells[cells.family == "Gaussian"]
G["G1"] = dict(passed=bool((gau.delta.abs() < 1e-10).all() and W1["Gaussian"] < 1e-10),
               max_abs_delta=float(gau.delta.abs().max()), W1=float(W1["Gaussian"]))
G["G2"] = dict(passed=bool(cells.bound_ok.all()), n_violate=int((~cells.bound_ok).sum()))
lap0 = cells[(cells.family == "Laplace") & (cells.s == 0)].iloc[0]
G3_ET_TGT = np.sqrt(0.5)
G3_EE_TGT = S2PI * (1 - 3/24)
G["G3"] = dict(passed=bool(abs(lap0.E_true - G3_ET_TGT) < 1e-9 and abs(lap0.E_edge - G3_EE_TGT) < 1e-6),
               E_true=float(lap0.E_true), E_edge=float(lap0.E_edge),
               E_true_target=float(G3_ET_TGT), E_edge_target=float(G3_EE_TGT))
g4 = []
for fam in FAMILIES:
    if fam["name"].startswith("t") and fam.get("E0_closed") is not None:
        row = cells[(cells.family == fam["name"]) & (cells.s == 0)].iloc[0]
        g4.append((fam["name"], float(row.E_true), fam["E0_closed"], abs(row.E_true - fam["E0_closed"])))
G["G4"] = dict(passed=bool(all(d < 1e-9 for _, _, _, d in g4)), max_diff=float(max(d for *_, d in g4)),
               detail={n: [q, c, d] for n, q, c, d in g4})
c4_0 = float(cells[(cells.family == "Laplace") & (cells.s == 0)].iloc[0].C4)
G["G5"] = dict(passed=bool(abs(c4_0 - (-S2PI)) < 1e-8), C4_0=c4_0, target=float(-S2PI))
mir = []
for k in [2, 8]:
    p0 = cells[(cells.family == f"gamma{k}p") & (cells.s == 0)].iloc[0].E_true
    m0 = cells[(cells.family == f"gamma{k}m") & (cells.s == 0)].iloc[0].E_true
    ps = cells[(cells.family == f"gamma{k}p") & (cells.s == 1.0)].iloc[0]
    ms = cells[(cells.family == f"gamma{k}m") & (cells.s == 1.0)].iloc[0]

    split = ps.delta - ms.delta
    edge_split = (ps.gamma1/6*ps.C3) - (ms.gamma1/6*ms.C3)
    mir.append((k, abs(p0 - m0), float(split), float(edge_split), bool(np.sign(split) == np.sign(edge_split))))
G["G6"] = dict(passed=bool(all(d < 1e-10 and sg for _, d, _, _, sg in mir)),
               detail={f"k{k}": dict(s0_diff=d, split_s1=sp, edge_split=es, sign_match=sg) for k, d, sp, es, sg in mir})

gate_all = all(G[g]["passed"] for g in G)
print("\n[GATES]")
for g in ["G1","G2","G3","G4","G5","G6"]:
    print(f"  {g}: {'PASS' if G[g]['passed'] else 'FAIL'}  {json.dumps({k:v for k,v in G[g].items() if k!='detail' and k!='passed'}, default=lambda o: round(o,8) if isinstance(o,float) else o)}")
if not gate_all:
    print("\n[STOP] gate failure — writing partial cells, NOT summary/overlay.")
    json.dump(G, open(HERE/"b11_gates.json","w"), indent=2, default=float)
    sys.exit(3)
print("[GATES] all PASS")

suff_rows = []
g2fams = ["t6", "Laplace", "ZI0.5", "gamma2p"]
for s in S_GRID:
    ds = [float(cells[(cells.family == f) & (cells.s == s)].iloc[0].delta) for f in g2fams]
    med = np.median(ds); spread = (max(ds) - min(ds))
    suff_rows.append(dict(s=s, **{f"delta_{f}": d for f, d in zip(g2fams, ds)},
        spread=spread, median_delta=med, spread_over_med=(spread/abs(med) if abs(med) > 1e-12 else np.nan)))
suff = pd.DataFrame(suff_rows)
suff.to_csv(HERE/"b11_kurt_sufficiency.csv", index=False)

def regime(s): return "s<=1" if s <= 1.0 else "s>1"
cells["s_regime"] = cells.s.map(regime)
srows = []
for (fam, reg), g in cells.groupby(["family", "s_regime"]):
    ec = g.edge_capture.dropna()
    srows.append(dict(family=fam, s_regime=reg, gamma2=float(g.gamma2.iloc[0]), n=len(g),
        bound_ok_all=bool(g.bound_ok.all()), slack_min=float(g.slack.min(skipna=True)),
        slack_max=float(g.slack.max(skipna=True)), slack_med=float(g.slack.median(skipna=True)),
        edge_capture_med=float(ec.median()) if len(ec) else np.nan,
        edge_valid_all=bool(g.edge_valid.all()), delta_med=float(g.delta.median())))
summ = pd.DataFrame(srows)
summ.to_csv(HERE/"b11_summary.csv", index=False)

inval = cells[(~cells.edge_valid) & (cells.s <= 1.0)]
edge_valid_boundary = float(inval.gamma2.min()) if len(inval) else None

def edge_pred_ratio(s, g1, g2):
    mu = float(s); m_f = folded(mu, 1.0)
    Ee = m_f + g1/6.0*C_k(mu, 3) + g2/24.0*C_k(mu, 4)
    return Ee/m_f if m_f > 0 else np.nan

ov = []

b5c = pd.read_csv(NEUMOR/"output/unified/b5_thm35_all.csv")
gv2 = pd.read_csv(NEUMOR/"output/unified/per_config_gaussianity_v2.csv")
key = ["block","config_id","payoff"]
mrg = b5c.merge(gv2[key+["kurt","skew"]], on=key, how="inner")

kurt_med = float(mrg["kurt"].median()); kurt_conv = "raw(ref3)" if abs(kurt_med-3) < abs(kurt_med) else "excess(ref0)"
for _, r in mrg.iterrows():
    s = abs(r["mu_delta"])/r["sigma_delta"] if r["sigma_delta"] > 0 else np.nan
    g2 = (r["kurt"] - 3.0) if kurt_conv.startswith("raw") else r["kurt"]
    if not np.isfinite(s): continue
    sk = float(r["skew"])
    pr = edge_pred_ratio(s, sk, g2)
    ov.append(dict(source="nn_suite", cell=f"{r['block']}/{r['config_id']}/{r['payoff']}", s=s, gamma1=sk,
        gamma2=g2, ratio_obs=float(r["ratio"]), ratio_pred_edge=pr, ratio_pred_null=1.0,
        caveat="measured moments have sampling error; kurt=%s" % kurt_conv))

b10 = pd.read_csv(NEUMOR/"output/b10_lowsnr/b10_lowsnr_all.csv")
for _, r in b10.iterrows():
    s = float(r.s_cal); g2 = float(r.ex_kurt)
    pr = edge_pred_ratio(s, 0.0, g2)
    ov.append(dict(source="b10_null", cell=str(r.cell_id), s=s, gamma1=0.0, gamma2=g2,
        ratio_obs=float(r.ratio_V1), ratio_pred_edge=pr, ratio_pred_null=1.0,
        caveat="gamma2~0 null anchor; skew unused"))
overlay = pd.DataFrame(ov)
overlay.to_csv(HERE/"b11_overlay.csv", index=False)

def ov_summ(df):
    d = df.dropna(subset=["ratio_obs","ratio_pred_edge"])
    med_abs = float((d.ratio_pred_edge - d.ratio_obs).abs().median())
    sign_ok = int((np.sign(d.ratio_pred_edge-1) == np.sign(d.ratio_obs-1)).sum())
    return len(d), med_abs, sign_ok
res = {src: ov_summ(overlay[overlay.source==src]) for src in overlay.source.unique()}

json.dump(dict(gates=G, edge_valid_boundary=edge_valid_boundary, kurt_conv=kurt_conv,
    overlay=res), open(HERE/"b11_gates.json","w"), indent=2, default=float)

print("\n===== b11_summary.csv (gamma2<=3 excerpt) =====")
with pd.option_context("display.width",200,"display.max_columns",20):
    print(summ[summ.gamma2 <= 3].round(4).to_string(index=False))
print(f"\n[STEP6] kurtosis-sufficiency spread/|median delta| by s (gamma2=3 fams {g2fams}):")
print(suff[["s","spread","median_delta","spread_over_med"]].round(5).to_string(index=False))
print(f"\n[EDGE] edge_valid boundary (smallest gamma2 with E_edge<=0 at s<=1): {edge_valid_boundary}")
ec_lowg = cells[(cells.gamma2 <= 3) & (cells.s <= 1.0)].edge_capture.dropna()
print(f"[EDGE] edge_capture median (gamma2<=3, s<=1): {float(ec_lowg.median()):.4f}  (n={len(ec_lowg)})")
print(f"\n[7a OVERLAY] kurt_conv={kurt_conv} (median kurt {kurt_med:.3f})")
for src,(n,ma,so) in res.items(): print(f"  {src}: n={n}  median|pred_edge-obs|={ma:.4f}  sign-agree(vs 1) {so}/{n}")
print(f"\nelapsed {time.time()-t0:.0f}s | wrote b11_cells.csv({len(cells)}), b11_summary.csv({len(summ)}), b11_overlay.csv({len(overlay)}), b11_kurt_sufficiency.csv")
