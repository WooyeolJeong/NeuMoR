"""Low-SNR (|s|->0) boundary for folded-normal (Thm 3.1), NN estimator only.
cal/eval split (front=cal, back=eval, no shuffle): Heston 140/60, Kou 119/51.
Digital strike scan -> zero-signal k* chosen on cal d_hat_cal; ratio/gaussianity on EVAL (in-sample also).
J-dominance J/(J+|S|) where d_ana (analytical density diff, unified pair npz) available.
Bootstrap B=100 (resample, recompute mu/sig/theory/emp). Read-only on paper. No retrain.
"""
import numpy as np, os, csv
from scipy.special import erf, erfc
from scipy import stats
np.random.seed(7)

ROOT = "<PROJECT_ROOT>"
OUT = os.path.join(ROOT, "NeuMoR/output/unified/lowsnr_sweep.csv")

def folded_mean(mu, sig):
    if sig <= 0: return abs(mu)
    s = mu/sig; return sig*np.sqrt(2/np.pi)*np.exp(-s*s/2) + mu*erf(s/np.sqrt(2))
def J_term(mu, sig):
    if sig <= 0: return 0.0
    s = mu/sig; return sig*(np.sqrt(2/np.pi)*np.exp(-s*s/2) - abs(s)*erfc(abs(s)/np.sqrt(2)))
def find_cfg(L,t,atol=1e-4): return [i for i,r in enumerate(L) if np.allclose(r,t,atol=atol)]

def stats_of(arr, g_signal_mu):
    """arr = per-seed Delta_hat. returns dict of theory/emp/gaussian."""
    mu = float(arr.mean()); sig = float(arr.std(ddof=1))
    if sig < 1e-12: return None
    s = mu/sig; MRth = folded_mean(mu,sig); MRemp = float(np.mean(np.abs(arr)))
    z = (arr-mu)/sig
    ksp = float(stats.kstest(z,'norm').pvalue)
    ad = stats.anderson(arr,'norm'); adrej = int(ad.statistic>ad.critical_values[2])
    return dict(mu=mu,sig=sig,s=s,MRth=MRth,MRemp=MRemp,ratio=MRemp/MRth,
                KS_p=ksp,KS_rej=int(ksp<0.05),AD_rej=adrej,J=J_term(mu,sig))

def boot_ratio(arr,B=100):
    R=[]
    n=len(arr)
    for _ in range(B):
        b=arr[np.random.randint(0,n,n)]; m=b.mean(); sd=b.std(ddof=1)
        if sd<1e-12: continue
        R.append(np.mean(np.abs(b))/folded_mean(m,sd))
    return (float(np.percentile(R,25)),float(np.percentile(R,75))) if R else (np.nan,np.nan)

SPECS={"heston":dict(base=[1.0,0.2,0.3,-0.5],v1=0.100,ncal=140,
                     pairs={"A":0.101,"B":0.105,"C":0.110,"D":0.130},
                     ladder=[0.1001,0.1002,0.1003,0.1005,0.1007]),
       "kou":dict(base=None,v1=0.300,ncal=119,
                  pairs={"A":0.301,"B":0.305,"C":0.310,"D":0.330},ladder=[])}

rows=[]; kstar_report=[]
for model,spec in SPECS.items():
    E=np.load(os.path.join(ROOT,f"NeuMoR/output/cross_class/ensemble_{model}_NN.npz"))
    M,L=E["matrix"],E["lambdas"]
    yg=np.load(os.path.join(ROOT,f"NeuMoR/output/unified/{model}_pairA.npz"),allow_pickle=True)["y_grid"].astype(float)
    dy=float(yg[1]-yg[0]); ncal=spec["ncal"]
    base=spec["base"]
    if base is None:
        cand=[tuple(np.round(L[i,1:],5)) for i in range(len(L)) if round(float(L[i,0]),5) in [round(v,5) for v in spec["pairs"].values()]]
        base=list(max(set(cand),key=cand.count))
    p1=M[:,find_cfg(L,[spec["v1"]]+list(base))[0],:].astype(float)
    g_price=np.maximum(np.exp(yg)-1,0); g_delta=np.exp(yg)*(yg>0); g_digOTM=(yg>np.log(1.10)).astype(float)

    def emit(pair,kind,pname,g,p2,d_ana):
        Dall=(p1-p2)@g*dy
        Dana = float(g@d_ana*dy) if d_ana is not None else None
        for tag,arr in [("eval",Dall[ncal:]),("insample",Dall)]:
            st=stats_of(arr,None)
            if st is None:
                rows.append(dict(model=model,pair=pair,kind=kind,payoff=pname,seedset=tag,degenerate=1)); continue
            q25,q75=boot_ratio(arr)
            if d_ana is not None:
                S=abs(st["mu"])-abs(Dana); Jf=st["J"]/(st["J"]+abs(S)) if (st["J"]+abs(S))>0 else np.nan
            else: S=None; Jf=None
            rows.append(dict(model=model,pair=pair,kind=kind,payoff=pname,seedset=tag,degenerate=0,
                s=st["s"],mu=st["mu"],sigma=st["sig"],MR_theory=st["MRth"],MR_emp=st["MRemp"],ratio=st["ratio"],
                boot_q25=q25,boot_q75=q75,KS_p=st["KS_p"],KS_rej=st["KS_rej"],AD_rej=st["AD_rej"],
                Delta_true=(Dana if Dana is not None else np.nan),S=(S if S is not None else np.nan),
                J=st["J"],J_frac=(Jf if Jf is not None else np.nan)))

    for pair,v2 in spec["pairs"].items():
        p2=M[:,find_cfg(L,[v2]+list(base))[0],:].astype(float)
        d_ana=np.load(os.path.join(ROOT,f"NeuMoR/output/unified/{model}_pair{pair}.npz"),allow_pickle=True)["d_ana"].astype(float)
        for pn,g in [("price_ATMcall",g_price),("delta_ATM",g_delta),("digital_OTM110",g_digOTM)]:
            emit(pair,"std",pn,g,p2,d_ana)
        if pair in ("A","D"):
            # k* from cal d_hat_cal, then coarse+fine strike scan
            dhat_cal=(p1[:ncal]-p2[:ncal]).mean(0)
            def mu_cal(k): return float(((yg>k).astype(float))@dhat_cal*dy)
            coarse=np.linspace(-3,3,41); mc=np.array([mu_cal(k) for k in coarse])
            sgn=np.where(np.diff(np.sign(mc)))[0]
            kstar=float(coarse[sgn[0]] + (coarse[sgn[0]+1]-coarse[sgn[0]])*abs(mc[sgn[0]])/(abs(mc[sgn[0]])+abs(mc[sgn[0]+1]))) if len(sgn) else float(coarse[np.argmin(np.abs(mc))])
            fine=np.linspace(kstar-0.4,kstar+0.4,25)
            ks_all=sorted(set(np.round(np.concatenate([coarse,fine]),4)))
            for k in ks_all:
                emit(pair,"scan",f"digit_k={k:+.3f}",(yg>k).astype(float),p2,d_ana)
            # realized |s| at k* (eval)
            Dks=((p1-p2)@((yg>kstar).astype(float))*dy)[ncal:]
            kstar_report.append((model,pair,round(kstar,3),round(abs(Dks.mean()/Dks.std(ddof=1)),3) if Dks.std(ddof=1)>1e-12 else np.nan))
    for v2 in spec["ladder"]:
        hit=find_cfg(L,[v2]+list(base))
        if hit: emit(f"ladder_dv={v2-spec['v1']:.4f}","ladder","delta_ATM",g_delta,M[:,hit[0],:].astype(float),None)

# CSV
keys=sorted(set().union(*[r.keys() for r in rows]))
with open(OUT,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();[w.writerow(r) for r in rows]

# ---- report ----
BINS=[("|s|<0.1",0,0.1),("0.1-0.5",0.1,0.5),("0.5-1",0.5,1),("1-3",1,3),(">=3",3,1e9)]
def summarize(seedset):
    print(f"\n================ |s| BINS  (seedset={seedset}) ================")
    print(f"{'model':7s}{'bin':10s}{'n':>4s}{'ratio_med':>10s}{'[q25':>8s}{'q75]':>8s}{'bootIQR':>9s}{'KS%':>6s}{'AD%':>6s}{'Jfrac_med':>10s}")
    for model in SPECS:
        for bn,lo,hi in BINS:
            rr=[r for r in rows if r['model']==model and r.get('degenerate')==0 and r['seedset']==seedset and lo<=abs(r['s'])<hi]
            if not rr: continue
            rt=[r['ratio'] for r in rr]; iqr=[ (r['boot_q75']-r['boot_q25']) for r in rr if not np.isnan(r['boot_q75'])]
            jf=[r['J_frac'] for r in rr if not np.isnan(r['J_frac'])]
            print(f"{model:7s}{bn:10s}{len(rr):4d}{np.median(rt):10.3f}{np.percentile(rt,25):8.3f}{np.percentile(rt,75):8.3f}"
                  f"{(np.median(iqr) if iqr else np.nan):9.3f}{100*np.mean([r['KS_rej'] for r in rr]):6.0f}{100*np.mean([r['AD_rej'] for r in rr]):6.0f}"
                  f"{(np.median(jf) if jf else np.nan):10.3f}")
summarize("eval"); summarize("insample")
print("\n=== 행사가 스캔 k* 와 실현 |s|(eval) ===")
for m,p,ks,ss in kstar_report: print(f"  {m} pair{p}: k*={ks:+.3f}  |s|_eval(at k*)={ss}")
ndeg=sum(1 for r in rows if r.get('degenerate')==1)
print(f"\n(degenerate σ<1e-12 제외: {ndeg}행)  CSV -> {OUT}  ({len(rows)}행)")
