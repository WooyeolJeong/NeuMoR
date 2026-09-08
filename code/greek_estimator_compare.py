"""Estimator comparison: does folded-normal (Thm 3.1, Gaussian assumption) break for
non-Gaussian tree estimators (RF/GB) vs NN? Uses cross_class ensembles (each estimator's
own seed ensemble, same pair lambdas). Read-only on paper. No retraining.
"""
import numpy as np, os, csv
from scipy.special import erf
from scipy import stats

ROOT = "<PROJECT_ROOT>"
OUTCSV = os.path.join(ROOT, "NeuMoR/output/unified/estimator_compare.csv")

def build_k_eta(p1,p2):
    e1=p1-p1.mean(0,keepdims=True); e2=p2-p2.mean(0,keepdims=True)
    eta=e1-e2; K=(eta.T@eta)/len(eta); return (K+K.T)/2
def folded_mean(mu,sig):
    if sig<=0: return abs(mu)
    s=mu/sig; return sig*np.sqrt(2/np.pi)*np.exp(-s*s/2)+mu*erf(s/np.sqrt(2))
def y_from_pair(m):
    return np.load(os.path.join(ROOT,f"NeuMoR/output/unified/{m}_pairA.npz"),allow_pickle=True)["y_grid"].astype(float)
def find_cfg(L,t,atol=1e-4): return [i for i,r in enumerate(L) if np.allclose(r,t,atol=atol)]
def payoffs(y):
    lk=np.log(1.10)
    return [("price_ATMcall",np.maximum(np.exp(y)-1,0)),("delta_ATM",np.exp(y)*(y>0)),("digital_OTM110",(y>lk).astype(float))]

SPECS={"heston":dict(base=[1.0,0.2,0.3,-0.5],v1=0.100,pairs={"A":0.101,"B":0.105,"C":0.110,"D":0.130}),
       "kou":dict(base=None,v1=0.300,pairs={"A":0.301,"B":0.305,"C":0.310,"D":0.330})}
ESTS=["NN","RF","GB","MC"]

rows=[]
for est in ESTS:
    for model,spec in SPECS.items():
        f=os.path.join(ROOT,f"NeuMoR/output/cross_class/ensemble_{model}_{est}.npz")
        E=np.load(f); M,L=E["matrix"],E["lambdas"]; y=y_from_pair(model); dy=float(y[1]-y[0])
        base=spec["base"]
        if base is None:
            cand=[tuple(np.round(L[i,1:],5)) for i in range(len(L)) if round(float(L[i,0]),5) in [round(v,5) for v in spec["pairs"].values()]]
            base=list(max(set(cand),key=cand.count))
        i1=find_cfg(L,[spec["v1"]]+list(base)); p1=M[:,i1[0],:].astype(float)
        for pair,v2 in spec["pairs"].items():
            i2=find_cfg(L,[v2]+list(base)); p2=M[:,i2[0],:].astype(float)
            d_hat=p1.mean(0)-p2.mean(0); K=build_k_eta(p1,p2)
            for pn,g in payoffs(y):
                mu=float(g@d_hat*dy); sig=float(np.sqrt(max(g@(K@g)*dy**2,0))); s=mu/sig if sig>0 else np.inf
                MRth=folded_mean(mu,sig); dseed=(p1-p2)@g*dy; MRemp=float(np.mean(np.abs(dseed)))
                ratio=MRemp/MRth if MRth!=0 else np.nan
                z=(dseed-dseed.mean())/dseed.std(ddof=1)
                ksp=float(stats.kstest(z,'norm').pvalue)
                ad=stats.anderson(dseed,'norm'); adrej=int(ad.statistic>ad.critical_values[2])
                rows.append(dict(est=est,model=model,pair=pair,payoff=pn,n_seed=len(p1),s=s,
                    ratio=ratio,KS_p=ksp,KS_rej=int(ksp<0.05),AD_rej=adrej,
                    skew=float(stats.skew(dseed)),kurt=float(stats.kurtosis(dseed,fisher=False))))

keys=list(rows[0].keys())
with open(OUTCSV,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();[w.writerow(r) for r in rows]

import statistics as st
print("=========== ESTIMATOR x PAYOFF (pooled 8 pairs; Gaussian ref skew=0 kurt=3) ===========")
print(f"{'est':4s}{'payoff':16s}{'ratio_med':>10s}{'ratio_rng':>18s}{'KSrej':>7s}{'ADrej':>7s}{'skew_med':>10s}{'kurt_med':>10s}{'kurt_max':>10s}")
for est in ESTS:
    for pn,_ in payoffs(np.array([0.,1.])):
        rr=[r for r in rows if r['est']==est and r['payoff']==pn]
        rt=[r['ratio'] for r in rr]; sk=[r['skew'] for r in rr]; ku=[r['kurt'] for r in rr]
        print(f"{est:4s}{pn:16s}{st.median(rt):10.3f}  [{min(rt):6.3f},{max(rt):6.3f}]{sum(r['KS_rej'] for r in rr):4d}/{len(rr):<2d}"
              f"{sum(r['AD_rej'] for r in rr):4d}/{len(rr):<2d}{st.median(sk):10.3f}{st.median(ku):10.3f}{max(ku):10.2f}")
    print()
print("=========== per-pair detail: RF/GB digital_OTM110 (worst-case?) ===========")
print(f"{'est':4s}{'model':7s}{'pair':5s}{'s':>7s}{'ratio':>8s}{'KSp':>7s}{'ADrej':>6s}{'skew':>9s}{'kurt':>9s}")
for r in [r for r in rows if r['est'] in ('RF','GB') and r['payoff']=='digital_OTM110']:
    print(f"{r['est']:4s}{r['model']:7s}{r['pair']:5s}{r['s']:7.2f}{r['ratio']:8.3f}{r['KS_p']:7.2f}{r['AD_rej']:6d}{r['skew']:9.3f}{r['kurt']:9.2f}")
print(f"\nCSV -> {OUTCSV}  ({len(rows)} rows)")
