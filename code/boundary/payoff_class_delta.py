import numpy as np, os, csv
from scipy.special import erf
from scipy import stats

ROOT = "<PROJECT_ROOT>"
OUTCSV = os.path.join(ROOT, "NeuMoR/output/unified/greek_delta_pilot.csv")

def build_k_eta(p1, p2):
    e1 = p1 - p1.mean(0, keepdims=True); e2 = p2 - p2.mean(0, keepdims=True)
    eta = e1 - e2; K = (eta.T @ eta) / len(eta); return (K + K.T) / 2

def folded_mean(mu, sig):
    if sig <= 0: return abs(mu)
    s = mu / sig
    return sig * np.sqrt(2/np.pi) * np.exp(-s*s/2) + mu * erf(s/np.sqrt(2))

def payoffs(y):
    return [("price_ATMcall", np.maximum(np.exp(y)-1.0, 0.0)),
            ("delta_ATM",      np.exp(y)*(y > 0.0)),
            ("delta_OTM1.10",  np.exp(y)*(y > np.log(1.10)))]

def y_from_pair(model):
    d = np.load(os.path.join(ROOT, f"NeuMoR/output/unified/{model}_pairA.npz"), allow_pickle=True)
    return d["y_grid"].astype(float)

def find_cfg(lambdas, target, atol=1e-4):
    hits = [i for i,r in enumerate(lambdas) if np.allclose(r, target, atol=atol)]
    return hits

SPECS = {
  "heston": dict(base_tail=[1.0,0.2,0.3,-0.5], p1val=0.100,
                 pairs={"A":0.101,"B":0.105,"C":0.110,"D":0.130}),
  "kou":    dict(base_tail=None, p1val=0.300,
                 pairs={"A":0.301,"B":0.305,"C":0.310,"D":0.330}),
}

rows = []
for model, spec in SPECS.items():
    E = np.load(os.path.join(ROOT, f"NeuMoR/output/cross_class/ensemble_{model}_NN.npz"))
    M, LAM = E["matrix"], E["lambdas"]
    y = y_from_pair(model); dy = float(y[1]-y[0]); ny = len(y)
    assert M.shape[2] == ny, (M.shape, ny)

    if spec["base_tail"] is None:
        tails = [LAM[find_cfg(LAM,[v]+list(LAM[find_cfg(LAM[:,0:1]==v if False else None,None)] ) )] for v in []]
    base_tail = spec["base_tail"]
    if base_tail is None:

        cand = [tuple(np.round(LAM[i,1:],5)) for i in range(len(LAM)) if round(float(LAM[i,0]),5) in [round(v,5) for v in spec["pairs"].values()]]
        base_tail = list(max(set(cand), key=cand.count))
        print(f"[{model}] auto base_tail = {base_tail}")
    lam1 = [spec["p1val"]] + list(base_tail)
    i1 = find_cfg(LAM, lam1)
    print(f"[{model}] lambda1={lam1} -> cfg idx {i1}")
    if not i1:
        print(f"  !! lambda1 MISSING, skip {model}"); continue
    if len(i1) > 1:
        dmax = float(np.abs(M[:, i1[0], :] - M[:, i1[1], :]).max())
        print(f"  [{model}] lambda1 matched {len(i1)} cfgs (same param point); density maxdiff={dmax:.2e}; using cfg {i1[0]}")
    p1 = M[:, i1[0], :].astype(float)
    for pair, v2 in spec["pairs"].items():
        lam2 = [v2] + list(base_tail); i2 = find_cfg(LAM, lam2)
        if not i2:
            print(f"  !! {model} {pair} lambda2={lam2} MISSING — skip"); continue
        if len(i2) > 1: print(f"  [{model} {pair}] lambda2 matched {i2}, using {i2[0]}")
        p2 = M[:, i2[0], :].astype(float)
        S = len(p1)
        d_hat = p1.mean(0) - p2.mean(0)
        K = build_k_eta(p1, p2)
        Kdotg_cache = {}
        for pname, g in payoffs(y):
            Kg = K @ g
            mu  = float(g @ d_hat * dy)
            s2  = float(g @ Kg * dy**2); sig = np.sqrt(max(s2, 0.0))
            s   = mu / sig if sig > 0 else np.inf
            MRth = folded_mean(mu, sig)
            dhat_seed = (p1 - p2) @ g * dy
            MRemp = float(np.mean(np.abs(dhat_seed)))
            ratio = MRemp / MRth if MRth != 0 else np.nan
            z = (dhat_seed - dhat_seed.mean()) / dhat_seed.std(ddof=1)
            ksp = float(stats.kstest(z, 'norm').pvalue)
            ad = stats.anderson(dhat_seed, 'norm'); ad_stat = float(ad.statistic); ad_crit5 = float(ad.critical_values[2])

            thr = 0.0 if pname != "delta_OTM1.10" else np.log(1.10)
            sig_i = g * d_hat * dy
            var_i = g * Kg * dy**2
            bands = [(thr,1),(1,2),(2,3),(3,y.max()+1e-9)]
            def frac(arr, tot):
                out=[]
                for lo,hi in bands:
                    m=(y>lo)&(y<=hi); out.append(float(arr[m].sum())/tot if tot!=0 else np.nan)
                return out
            sig_frac = frac(sig_i, sig_i.sum()); var_frac = frac(var_i, var_i.sum())
            rows.append(dict(model=model, pair=pair, dperturb=round(v2-spec["p1val"],4), payoff=pname,
                mu=mu, sigma=sig, s=s, MR_theory=MRth, MR_emp=MRemp, ratio=ratio,
                KS_p=ksp, KS_rej=int(ksp<0.05), AD_stat=ad_stat, AD_crit5=ad_crit5, AD_rej=int(ad_stat>ad_crit5),
                sigVAR_near_le1=var_frac[0], sigVAR_1to2=var_frac[1], sigVAR_2to3=var_frac[2], sigVAR_gt3=var_frac[3],
                SIG_near_le1=sig_frac[0], SIG_1to2=sig_frac[1], SIG_2to3=sig_frac[2], SIG_gt3=sig_frac[3], n_seed=S))

keys = list(rows[0].keys())
with open(OUTCSV,'w',newline='') as f:
    w=csv.DictWriter(f, fieldnames=keys); w.writeheader()
    for r in rows: w.writerow(r)

print("\n================ RATIO / GAUSSIANITY (emp/theory, KS/AD reject@5%) ================")
print(f"{'model':7s}{'pair':5s}{'payoff':14s}{'s':>8s}{'MR_th':>11s}{'MR_emp':>11s}{'ratio':>8s}{'KSp':>7s}{'KSrej':>6s}{'ADrej':>6s}")
for r in rows:
    print(f"{r['model']:7s}{r['pair']:5s}{r['payoff']:14s}{r['s']:8.2f}{r['MR_theory']:11.3e}{r['MR_emp']:11.3e}{r['ratio']:8.3f}{r['KS_p']:7.2f}{r['KS_rej']:6d}{r['AD_rej']:6d}")

print("\n================ TAIL DECOMPOSITION — sigma^2 contribution % by y-band ================")
print(f"{'model':7s}{'pair':5s}{'payoff':14s}{'y<=1':>9s}{'1-2':>9s}{'2-3':>9s}{'y>3':>9s}   (signal% near<=1)")
for r in rows:
    print(f"{r['model']:7s}{r['pair']:5s}{r['payoff']:14s}{r['sigVAR_near_le1']*100:8.1f}%{r['sigVAR_1to2']*100:8.1f}%{r['sigVAR_2to3']*100:8.1f}%{r['sigVAR_gt3']*100:8.1f}%   sig<=1:{r['SIG_near_le1']*100:5.0f}%")

print("\n================ SUMMARY: price(ATM call) vs delta ================")
import statistics as st
for pay in ["price_ATMcall","delta_ATM","delta_OTM1.10"]:
    rr=[r for r in rows if r['payoff']==pay]
    ratios=[r['ratio'] for r in rr]; ksrej=sum(r['KS_rej'] for r in rr); adrej=sum(r['AD_rej'] for r in rr)
    print(f"  {pay:14s}: ratio median={st.median(ratios):.3f} range=[{min(ratios):.3f},{max(ratios):.3f}]  KS_rej={ksrej}/{len(rr)}  AD_rej={adrej}/{len(rr)}")
print(f"\nCSV -> {OUTCSV}  ({len(rows)} rows)")
