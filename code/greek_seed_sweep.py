"""Seed subsample sweep: when does folded-normal (Thm 3.1) break as N shrinks?
Method: fix per-seed Delta_hat (200) from cross_class NN ensemble; at each N draw B=100
without-replacement random subsets (N=200 = full set, single); recompute mu=sample-mean,
sigma=sample-std(ddof=1) -> folded MR_theory; MR_emp=mean|Delta|; ratio, KS/AD reject, skew, kurt.
Read-only on paper. sigma_Delta^2=<g,K_eta g>dy^2 == sample var of Delta_hat (identity), so
sample std suffices. RNG seed fixed. No retraining.
"""
import numpy as np, os, csv
from scipy.special import erf
from scipy import stats
np.random.seed(12345)

ROOT = "<PROJECT_ROOT>"
OUTCSV = os.path.join(ROOT, "NeuMoR/output/unified/seed_sweep_breakdown.csv")

def folded_mean(mu, sig):
    if sig <= 0: return abs(mu)
    s = mu/sig
    return sig*np.sqrt(2/np.pi)*np.exp(-s*s/2) + mu*erf(s/np.sqrt(2))

def y_from_pair(m):
    return np.load(os.path.join(ROOT, f"NeuMoR/output/unified/{m}_pairA.npz"), allow_pickle=True)["y_grid"].astype(float)

def find_cfg(L, t, atol=1e-4): return [i for i,r in enumerate(L) if np.allclose(r, t, atol=atol)]

SPECS = {"heston": dict(base=[1.0,0.2,0.3,-0.5], v1=0.100, pairs={"A":0.101,"B":0.105,"C":0.110,"D":0.130}),
         "kou":    dict(base=None, v1=0.300, pairs={"A":0.301,"B":0.305,"C":0.310,"D":0.330})}

def payoffs(y):
    lk = np.log(1.10)
    return [("price_ATMcall", np.maximum(np.exp(y)-1,0)),
            ("delta_ATM",     np.exp(y)*(y>0)),
            ("digital_OTM110",(y>lk).astype(float))]

# precompute per-seed Delta_hat
DH = {}   # (model,pair,payoff) -> (200,)
for model, spec in SPECS.items():
    E = np.load(os.path.join(ROOT, f"NeuMoR/output/cross_class/ensemble_{model}_NN.npz"))
    M, L = E["matrix"], E["lambdas"]; y = y_from_pair(model); dy = float(y[1]-y[0])
    base = spec["base"]
    if base is None:
        cand=[tuple(np.round(L[i,1:],5)) for i in range(len(L)) if round(float(L[i,0]),5) in [round(v,5) for v in spec["pairs"].values()]]
        base=list(max(set(cand),key=cand.count))
    i1 = find_cfg(L, [spec["v1"]]+list(base)); p1 = M[:, i1[0], :].astype(float)
    for pair, v2 in spec["pairs"].items():
        i2 = find_cfg(L, [v2]+list(base)); p2 = M[:, i2[0], :].astype(float)
        for pn, g in payoffs(y):
            DH[(model,pair,pn)] = (p1-p2) @ g * dy

NS = [5,10,20,50,100,200]; B = 100
rows = []; pooled = {}
for (model,pair,pn), dall in DH.items():
    S = len(dall)
    for N in NS:
        draws = [dall] if N>=S else [dall[np.random.choice(S,N,replace=False)] for _ in range(B)]
        R=[];KS=[];AD=[];SK=[];KU=[]
        for d in draws:
            m=d.mean(); sd=d.std(ddof=1); MRth=folded_mean(m,sd); MRemp=float(np.mean(np.abs(d)))
            R.append(MRemp/MRth if MRth!=0 else np.nan)
            z=(d-m)/sd
            KS.append(1 if stats.kstest(z,'norm').pvalue<0.05 else 0)
            ad=stats.anderson(d,'norm'); AD.append(1 if ad.statistic>ad.critical_values[2] else 0)
            SK.append(float(stats.skew(d))); KU.append(float(stats.kurtosis(d,fisher=False)))
        rows.append(dict(model=model,pair=pair,payoff=pn,N=N,B=len(draws),
            ratio_med=float(np.median(R)),ratio_q25=float(np.percentile(R,25)),ratio_q75=float(np.percentile(R,75)),
            KS_rejrate=float(np.mean(KS)),AD_rejrate=float(np.mean(AD)),
            skew_med=float(np.median(SK)),kurt_med=float(np.median(KU))))
        p=pooled.setdefault((pn,N),dict(R=[],KS=[],AD=[],SK=[],KU=[]))
        p["R"]+=R;p["KS"]+=KS;p["AD"]+=AD;p["SK"]+=SK;p["KU"]+=KU

keys=list(rows[0].keys())
with open(OUTCSV,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();[w.writerow(r) for r in rows]

print("=== POOLED over 8 pairs (Heston A-D + Kou A-D), B=100 subsets each (N=200: full, single) ===")
for pn,_ in payoffs(np.array([0.0,1.0])):
    print(f"\n--- {pn} ---")
    print(f"{'N':>4s}{'ratio_med':>10s}{'[q25':>9s}{'q75]':>9s}{'KS_rej%':>9s}{'AD_rej%':>9s}{'skew_med':>10s}{'kurt_med':>10s}")
    for N in NS:
        p=pooled[(pn,N)]
        print(f"{N:>4d}{np.median(p['R']):10.3f}{np.percentile(p['R'],25):9.3f}{np.percentile(p['R'],75):9.3f}"
              f"{100*np.mean(p['KS']):9.1f}{100*np.mean(p['AD']):9.1f}{np.median(p['SK']):10.3f}{np.median(p['KU']):10.3f}")
print(f"\nCSV -> {OUTCSV}  ({len(rows)} rows)   [Gaussian ref: skew=0, kurt=3]")
