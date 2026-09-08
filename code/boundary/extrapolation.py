import numpy as np, os, sys, time, csv
from scipy.special import erf
from scipy import stats
t0=time.time(); ROOT="<PROJECT_ROOT>"; sys.path.insert(0,ROOT)
import torch
from TNO.common.tno_model import DensityDeepONet_HestonLog
from NeuMoR.src.heston_teacher import lewis_pdf
DEVICE="cpu"; N=50
SAVE01=os.path.join(ROOT,"NeuMoR/save/exp01"); SAVE075=os.path.join(ROOT,"NeuMoR/save/exp07_5")
OUT=os.path.join(ROOT,"NeuMoR/output/unified/extrapolation_oob.csv")
def folded_mean(mu,sig):
    if sig<=0: return abs(mu)
    s=mu/sig; return sig*np.sqrt(2/np.pi)*np.exp(-s*s/2)+mu*erf(s/np.sqrt(2))
def load_models(n):
    models=[];yg=None;pk=None
    for seed in range(n):
        path=os.path.join(SAVE01,f"seed_{seed:02d}.pt") if seed<50 else os.path.join(SAVE075,f"seed_{seed:03d}.pt")
        ck=torch.load(path,map_location=DEVICE,weights_only=False);cfg=ck["config"]
        m=DensityDeepONet_HestonLog(lambda_dim=cfg["lambda_dim"],n_y=cfg["n_y"],rank=cfg["rank"],branch_hidden=cfg["branch_hidden"]).to(DEVICE)
        m.load_state_dict(ck["state_dict"]);m.eval();models.append(m)
        if yg is None: yg=np.asarray(ck["y_grid"],float);pk=list(ck["param_keys"])
    return models,yg,pk
def infer(models,params,yg,pk):
    dy=float(yg[1]-yg[0]);lt=torch.tensor(np.array([params[k] for k in pk],dtype=np.float32)).unsqueeze(0).to(DEVICE)
    out=np.zeros((len(models),len(yg)))
    with torch.no_grad():
        for i,m in enumerate(models):
            r=m(lt,dy);out[i]=(r[0] if isinstance(r,(list,tuple)) else r).cpu().numpy().squeeze()
    return out

models,yg,pk=load_models(N); dy=float(yg[1]-yg[0]); print(f"load {N}: {time.time()-t0:.0f}s")
base={"v":0.10,"kappa":1.0,"omega":0.2,"xi":0.3,"rho":-0.5}
CFG={"base_xi0.3":base,"oob_xi0.7":dict(base,xi=0.7),"oob_xi1.2":dict(base,xi=1.2)}
P={k:infer(models,pr,yg,pk) for k,pr in CFG.items()}
LEW={k:lewis_pdf(dict(pr,T=1.0),yg) for k,pr in CFG.items()}
print(f"infer+lewis done: {time.time()-t0:.0f}s")
print("=== density sanity (50 seeds) ===")
for k,Pk in P.items():
    mass=Pk.sum(1)*dy; print(f"  {k:12s}: Σp·dy=[{mass.min():.4f},{mass.max():.4f}] min_p={Pk.min():.2e} neg_frac={np.mean(Pk<0):.3f} NaN={int(np.isnan(Pk).sum())}")
    lm=float(np.trapz(LEW[k],yg)); print(f"                 Lewis: Σ={lm:.4f} min={LEW[k].min():.2e}")
gs={"price_ATMcall":np.maximum(np.exp(yg)-1,0),"delta_ATM":np.exp(yg)*(yg>0),"digital_OTM110":(yg>np.log(1.10)).astype(float)}
rows=[]
print("\n=== folded-normal on in-box(base) vs OOB  (50 seeds) ===")
print(f"{'oob':10s}{'payoff':16s}{'s':>8s}{'ratio':>8s}{'KSp':>6s}{'KSr':>4s}{'ADr':>4s}{'skew':>7s}{'kurt':>7s}{'brel_OOB':>9s}{'Δtrue':>10s}")
p1=P["base_xi0.3"]; d_lew_base=LEW["base_xi0.3"]
for ok in ["oob_xi0.7","oob_xi1.2"]:
    p2=P[ok]; d_lew=d_lew_base-LEW[ok]
    for pn,g in gs.items():
        D=(p1-p2)@g*dy; mu=D.mean();sd=D.std(ddof=1);s=mu/sd if sd>1e-12 else np.inf
        MRth=folded_mean(mu,sd);MRemp=np.mean(np.abs(D));ratio=MRemp/MRth
        z=(D-mu)/sd;ksp=float(stats.kstest(z,'norm').pvalue);ad=stats.anderson(D,'norm')
        adrej=int(ad.statistic>ad.critical_values[2]);ksrej=int(ksp<0.05)
        Dtrue=float(g@d_lew*dy); brel=abs(mu-Dtrue)/max(abs(Dtrue),1e-12)
        rows.append(dict(oob=ok,payoff=pn,s=s,ratio=ratio,KS_p=ksp,KS_rej=ksrej,AD_rej=adrej,
            skew=float(stats.skew(D)),kurt=float(stats.kurtosis(D,fisher=False)),brel_OOB=brel,Delta_true=Dtrue,n_seed=N))
        print(f"{ok:10s}{pn:16s}{s:8.2f}{ratio:8.3f}{ksp:6.2f}{ksrej:4d}{adrej:4d}{float(stats.skew(D)):7.3f}{float(stats.kurtosis(D,fisher=False)):7.2f}{brel:9.3f}{Dtrue:10.3e}")
with open(OUT,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();[w.writerow(r) for r in rows]
print(f"\nTOTAL: {time.time()-t0:.0f}s  CSV -> {OUT} ({len(rows)} rows)")
