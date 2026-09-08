import numpy as np, os, csv
from scipy.special import erf
from scipy import stats

ROOT = "<PROJECT_ROOT>"
OUTCSV = os.path.join(ROOT, "NeuMoR/output/unified/greek_pilot2.csv")

def build_k_eta(p1, p2):
    e1 = p1 - p1.mean(0, keepdims=True); e2 = p2 - p2.mean(0, keepdims=True)
    eta = e1 - e2; K = (eta.T @ eta) / len(eta); return (K + K.T) / 2

def folded_mean(mu, sig):
    if sig <= 0: return abs(mu)
    s = mu / sig
    return sig*np.sqrt(2/np.pi)*np.exp(-s*s/2) + mu*erf(s/np.sqrt(2))

def payoffs(y, dy):
    ny = len(y); lk = np.log(1.10); lB = np.log(1.30)
    def spike(y0):
        g = np.zeros(ny); g[int(np.argmin(np.abs(y-y0)))] = 1.0/dy; return g
    return [
        ("price_ATMcall",  np.maximum(np.exp(y)-1.0, 0.0)),
        ("delta_ATM",      np.exp(y)*(y > 0.0)),
        ("gamma_ATM",      spike(0.0)),
        ("gamma_OTM110",   spike(lk)),
        ("digital_ATM",    (y > 0.0).astype(float)),
        ("digital_OTM110", (y > lk).astype(float)),
        ("barrier_uo130",  (np.exp(y)-1.0)*((y > 0.0) & (y < lB))),
    ]

def y_from_pair(model):
    d = np.load(os.path.join(ROOT, f"NeuMoR/output/unified/{model}_pairA.npz"), allow_pickle=True)
    return d["y_grid"].astype(float)

def find_cfg(lambdas, target, atol=1e-4):
    return [i for i,r in enumerate(lambdas) if np.allclose(r, target, atol=atol)]

SPECS = {
  "heston": dict(base_tail=[1.0,0.2,0.3,-0.5], p1val=0.100, pairs={"A":0.101,"B":0.105,"C":0.110,"D":0.130}),
  "kou":    dict(base_tail=None, p1val=0.300, pairs={"A":0.301,"B":0.305,"C":0.310,"D":0.330}),
}

rows = []
for model, spec in SPECS.items():
    E = np.load(os.path.join(ROOT, f"NeuMoR/output/cross_class/ensemble_{model}_NN.npz"))
    M, LAM = E["matrix"], E["lambdas"]
    y = y_from_pair(model); dy = float(y[1]-y[0])
    base_tail = spec["base_tail"]
    if base_tail is None:
        cand = [tuple(np.round(LAM[i,1:],5)) for i in range(len(LAM)) if round(float(LAM[i,0]),5) in [round(v,5) for v in spec["pairs"].values()]]
        base_tail = list(max(set(cand), key=cand.count)); print(f"[{model}] auto base_tail={base_tail}")
    i1 = find_cfg(LAM, [spec["p1val"]]+list(base_tail))
    if not i1: print(f"!! {model} lambda1 missing"); continue
    if len(i1) > 1: print(f"[{model}] lambda1 matched {i1}, using {i1[0]}")
    p1 = M[:, i1[0], :].astype(float)
    for pair, v2 in spec["pairs"].items():
        i2 = find_cfg(LAM, [v2]+list(base_tail))
        if not i2: print(f"!! {model} {pair} lambda2 missing"); continue
        p2 = M[:, i2[0], :].astype(float)
        d_hat = p1.mean(0) - p2.mean(0); K = build_k_eta(p1, p2)
        for pname, g in payoffs(y, dy):
            Kg = K @ g
            mu = float(g @ d_hat * dy); s2 = float(g @ Kg * dy**2); sig = np.sqrt(max(s2,0.0))
            s = mu/sig if sig > 0 else np.inf
            MRth = folded_mean(mu, sig)
            dseed = (p1 - p2) @ g * dy
            MRemp = float(np.mean(np.abs(dseed))); ratio = MRemp/MRth if MRth != 0 else np.nan
            z = (dseed - dseed.mean())/dseed.std(ddof=1)
            ksp = float(stats.kstest(z, 'norm').pvalue)
            ad = stats.anderson(dseed, 'norm'); adrej = int(ad.statistic > ad.critical_values[2])
            var_i = g*Kg*dy**2; sig_i = g*d_hat*dy
            bands = [(0,1),(1,2),(2,3),(3,y.max()+1e-9)]
            def frac(a, tot): return [float(a[(y>lo)&(y<=hi)].sum())/tot if tot != 0 else np.nan for lo,hi in bands]
            vf = frac(var_i, var_i.sum())
            rows.append(dict(model=model, pair=pair, payoff=pname, s=s, MR_theory=MRth, MR_emp=MRemp, ratio=ratio,
                KS_p=ksp, KS_rej=int(ksp<0.05), AD_rej=adrej,
                sig2_le1=vf[0], sig2_1to2=vf[1], sig2_2to3=vf[2], sig2_gt3=vf[3], n_seed=len(p1)))

keys = list(rows[0].keys())
with open(OUTCSV,'w',newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow(r) for r in rows]

PAYORDER = ["price_ATMcall","delta_ATM","gamma_ATM","gamma_OTM110","digital_ATM","digital_OTM110","barrier_uo130"]
print("\n============ RATIO / GAUSSIANITY ============")
print(f"{'model':7s}{'pair':5s}{'payoff':16s}{'s':>8s}{'MR_th':>11s}{'MR_emp':>11s}{'ratio':>8s}{'KSp':>7s}{'KSrej':>6s}{'ADrej':>6s}")
for r in sorted(rows, key=lambda r:(r['model'],r['pair'],PAYORDER.index(r['payoff']))):
    print(f"{r['model']:7s}{r['pair']:5s}{r['payoff']:16s}{r['s']:8.2f}{r['MR_theory']:11.3e}{r['MR_emp']:11.3e}{r['ratio']:8.3f}{r['KS_p']:7.2f}{r['KS_rej']:6d}{r['AD_rej']:6d}")

print("\n============ SUMMARY per payoff (over 8 pairs) ============")
import statistics as st
print(f"{'payoff':16s}{'ratio_med':>10s}{'ratio_min':>10s}{'ratio_max':>10s}{'KS_rej':>8s}{'AD_rej':>8s}")
for pay in PAYORDER:
    rr=[r for r in rows if r['payoff']==pay]; rt=[r['ratio'] for r in rr]
    print(f"{pay:16s}{st.median(rt):10.3f}{min(rt):10.3f}{max(rt):10.3f}{sum(r['KS_rej'] for r in rr):5d}/{len(rr):<2d}{sum(r['AD_rej'] for r in rr):5d}/{len(rr):<2d}")

print("\n============ TAIL: sigma^2 contribution % (y-band), pair A only ============")
print(f"{'model':7s}{'payoff':16s}{'y<=1':>9s}{'1-2':>9s}{'2-3':>9s}{'y>3':>9s}")
for r in sorted([r for r in rows if r['pair']=='A'], key=lambda r:(r['model'],PAYORDER.index(r['payoff']))):
    print(f"{r['model']:7s}{r['payoff']:16s}{r['sig2_le1']*100:8.1f}%{r['sig2_1to2']*100:8.1f}%{r['sig2_2to3']*100:8.1f}%{r['sig2_gt3']*100:8.1f}%")
print(f"\nCSV -> {OUTCSV}  ({len(rows)} rows)")
