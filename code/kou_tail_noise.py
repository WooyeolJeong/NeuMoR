"""Tail-noise x tail-payoff (last boundary). High-jump Kou (in-box) pushes noise to tail;
deep-tail digital ladder tests folded-normal/Gaussianity there. Ref cell = cache low-jump Kou.
NN only, stageB 170 seeds (no stageA mix). No retrain (inference only). Read-only on paper.
"""
import numpy as np, os, sys, time, glob, csv
from scipy.special import erf
from scipy import stats
np.random.seed(3); t0=time.time()
ROOT="<PROJECT_ROOT>"; sys.path.insert(0,ROOT)
import torch
from TNO.common.tno_model import DensityDeepONet_HestonLog
DEVICE="cpu"; SB=ROOT+"/NeuMoR/save/kou/stageB"; OUT=ROOT+"/NeuMoR/output/unified/kou_tail_noise.csv"
def fold(mu,sig):
    if sig<=0:return abs(mu)
    s=mu/sig;return sig*np.sqrt(2/np.pi)*np.exp(-s*s/2)+mu*erf(s/np.sqrt(2))
def boot(a,B=100):
    R=[];n=len(a)
    for _ in range(B):
        b=a[np.random.randint(0,n,n)];m=b.mean();sd=b.std(ddof=1)
        if sd>1e-14:R.append(np.mean(np.abs(b))/fold(m,sd))
    return (np.percentile(R,25),np.percentile(R,75)) if R else (np.nan,np.nan)
def gstats(D):
    mu=D.mean();sd=D.std(ddof=1)
    deg=(abs(mu)<1e-12 and sd<1e-12)
    if sd<1e-14: return dict(mu=mu,sig=sd,degenerate=int(deg),s=np.nan,ratio=np.nan,KS_p=np.nan,KS_rej=0,AD_rej=0,skew=np.nan,kurt=np.nan,q25=np.nan,q75=np.nan)
    s=mu/sd;z=(D-mu)/sd;ad=stats.anderson(D,'norm');q25,q75=boot(D)
    return dict(mu=mu,sig=sd,degenerate=int(deg),s=s,ratio=np.mean(np.abs(D))/fold(mu,sd),
        KS_p=float(stats.kstest(z,'norm').pvalue),KS_rej=int(stats.kstest(z,'norm').pvalue<0.05),
        AD_rej=int(ad.statistic>ad.critical_values[2]),skew=float(stats.skew(D)),kurt=float(stats.kurtosis(D,fisher=False)),q25=q25,q75=q75)

def load(n):
    fs=sorted(glob.glob(SB+"/*.pt"))[:n];models=[];yg=lo=hi=None
    for f in fs:
        ck=torch.load(f,map_location=DEVICE,weights_only=False)
        m=DensityDeepONet_HestonLog(lambda_dim=5,n_y=ck['n_y'],rank=ck['rank'],branch_hidden=ck['branch_hidden']).to(DEVICE)
        m.load_state_dict(ck['state_dict']);m.eval();models.append(m)
        if yg is None: yr=ck['y_range'];yg=np.linspace(yr[0],yr[1],ck['n_y']);lo=np.asarray(ck['param_lo'],float);hi=np.asarray(ck['param_hi'],float)
    return models,yg,lo,hi
def infer(models,praw,yg,lo,hi):
    dy=float(yg[1]-yg[0]);ln=torch.tensor(((np.array(praw,float)-lo)/(hi-lo))[None],dtype=torch.float32)
    out=np.zeros((len(models),len(yg)))
    with torch.no_grad():
        for i,m in enumerate(models): r=m(ln.to(DEVICE),dy);out[i]=(r[0] if isinstance(r,(list,tuple)) else r).cpu().numpy().squeeze()
    return out

models,yg,lo,hi=load(170);dy=float(yg[1]-yg[0]);print(f"[load 170 stageB] {time.time()-t0:.0f}s")
pT0=infer(models,[0.30,45,0.5,3.5,10.0],yg,lo,hi)
pT1=infer(models,[0.31,45,0.5,3.5,10.0],yg,lo,hi)   # sigma axis
pT2=infer(models,[0.30,45,0.5,4.0,10.0],yg,lo,hi)   # eta1 (tail-thickness) axis
print(f"[infer T0/T1/T2] {time.time()-t0:.0f}s")
# cache ref (low-jump)
E=np.load(ROOT+"/NeuMoR/output/cross_class/ensemble_kou_NN.npz");L=E["lambdas"];M=E["matrix"]
def cfg(v):
    h=[i for i in range(len(L)) if np.allclose(L[i],[v,5.0,0.5,10.0,10.0],atol=1e-4)];return h[0]
pA1=M[:,cfg(0.300),:].astype(float);pA2=M[:,cfg(0.301),:].astype(float)

print("\n=== density sanity (170 seeds) ===")
for lab,P in [("T0 base(λJ45,η1 3.5)",pT0),("T1(σ0.31)",pT1),("T2(η1 4.0)",pT2)]:
    mass=P.sum(1)*dy;print(f"  {lab:22s} Σp·dy=[{mass.min():.4f},{mass.max():.4f}] min_p={P.min():.2e} neg={np.mean(P<0):.3f} NaN={int(np.isnan(P).sum())}")

PAYOFFS=[("price_ATMcall",np.maximum(np.exp(yg)-1,0)),("delta_ATM",np.exp(yg)*(yg>0))]+[(f"digital_k{k}",(yg>k).astype(float)) for k in [1.0,1.5,2.0,2.5,3.0]]
CELLS=[("T1_tailnoise_σ",pT0,pT1),("T2_tailnoise_η",pT0,pT2),("refA_nearnoise",pA1,pA2)]
rows=[]
print("\n=== Part2/3: cell × payoff  (μ, σ_Δ, s, ratio[q25,q75], KS/AD, skew, kurt) ===")
print(f"{'cell':16s}{'payoff':14s}{'mu':>11s}{'sigma':>11s}{'s':>7s}{'ratio':>7s}{'q25':>7s}{'q75':>7s}{'KSr':>4s}{'ADr':>4s}{'skew':>7s}{'kurt':>7s}")
for cell,p1,p2 in CELLS:
    for pn,g in PAYOFFS:
        D=(p1-p2)@g*dy; st=gstats(D); st.update(cell=cell,payoff=pn); rows.append(st)
        deg="DEG" if st['degenerate'] else ""
        print(f"{cell:16s}{pn:14s}{st['mu']:11.3e}{st['sig']:11.3e}{st['s'] if not np.isnan(st['s']) else -9:7.2f}"
              f"{st['ratio'] if not np.isnan(st['ratio']) else -9:7.3f}{st['q25'] if not np.isnan(st['q25']) else -9:7.3f}{st['q75'] if not np.isnan(st['q75']) else -9:7.3f}"
              f"{st['KS_rej']:4d}{st['AD_rej']:4d}{st['skew'] if not np.isnan(st['skew']) else -9:7.3f}{st['kurt'] if not np.isnan(st['kurt']) else -9:7.2f} {deg}")

# Part 4a: noise-variance |y|-band fractions (does noise live in tail?)
print("\n=== Part4a: 노이즈분산 Var(η(y)) |y|-band 분율 (테일 이동?) ===")
print(f"{'cell':16s}{'|y|<=1':>9s}{'1-2':>9s}{'2-3':>9s}{'>3':>9s}")
band4a=[]
for cell,p1,p2 in CELLS:
    eta=(p1-p2)-(p1-p2).mean(0); v=eta.var(0)   # per-point noise variance
    tot=v.sum()
    fr=[v[(np.abs(yg)<=1)].sum()/tot, v[(np.abs(yg)>1)&(np.abs(yg)<=2)].sum()/tot, v[(np.abs(yg)>2)&(np.abs(yg)<=3)].sum()/tot, v[np.abs(yg)>3].sum()/tot]
    band4a.append((cell,fr));print(f"{cell:16s}{fr[0]*100:8.1f}%{fr[1]*100:8.1f}%{fr[2]*100:8.1f}%{fr[3]*100:8.1f}%")

# Part 4b: pointwise eta at tail grid points — per-seed skew/kurt/KS
print("\n=== Part4b: 점별 η(ω,y) 진단 (테일 격자점, per-seed skew/kurt/KS) ===")
print(f"{'cell':16s}{'y':>6s}{'skew':>8s}{'kurt':>8s}{'KS_p':>7s}{'KSrej':>6s}")
ptrows=[]
for cell,p1,p2 in CELLS:
    eta=(p1-p2)-(p1-p2).mean(0)
    for yv in [1.5,2.0,2.5,3.0]:
        idx=int(np.argmin(np.abs(yg-yv)));col=eta[:,idx]
        if col.std(ddof=1)<1e-14: print(f"{cell:16s}{yg[idx]:6.2f}   (degenerate σ~0)");continue
        z=(col-col.mean())/col.std(ddof=1);ksp=float(stats.kstest(z,'norm').pvalue)
        sk=float(stats.skew(col));ku=float(stats.kurtosis(col,fisher=False))
        print(f"{cell:16s}{yg[idx]:6.2f}{sk:8.3f}{ku:8.2f}{ksp:7.2f}{int(ksp<0.05):6d}")
        ptrows.append(dict(cell=cell,y=round(yg[idx],2),skew=sk,kurt=ku,KS_p=ksp,KS_rej=int(ksp<0.05)))

with open(OUT,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=sorted(set().union(*[r.keys() for r in rows])));w.writeheader();[w.writerow(r) for r in rows]
print(f"\nTOTAL {time.time()-t0:.0f}s  CSV -> {OUT} ({len(rows)}행)")
