import sys, os, itertools
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
from pathlib import Path
import numpy as np, pandas as pd
from scipy import integrate
from scipy.stats import kurtosis, skew, norm
HERE = Path(__file__).resolve().parent; NEUMOR = HERE.parents[1]; CROSS = NEUMOR/"output/cross_class"
sys.path.insert(0, str(NEUMOR/"code")); sys.path.insert(0, str(NEUMOR/"code"/"suite"))
sys.path.insert(0, str(NEUMOR/"code"/"core")); import score_suite as b5
folded = b5.folded_normal_mean
YWIN = 1.5; CALL_KS = [0.9, 1.0, 1.1]; PAYS = [f"call_{k}" for k in CALL_KS] + ["digital_1.0"]
def He3(z): return z**3-3*z
def He4(z): return z**4-6*z**2+3
def C_k(mu,k):
    He=He3 if k==3 else He4
    a,_=integrate.quad(lambda z:abs(mu+z)*He(z)*norm.pdf(z),-np.inf,-mu,limit=200)
    b,_=integrate.quad(lambda z:abs(mu+z)*He(z)*norm.pdf(z),-mu,np.inf,limit=200)
    return a+b
def edge_pred_ratio(s,g1,g2):
    mf=folded(float(s),1.0); return (mf+g1/6*C_k(s,3)+g2/24*C_k(s,4))/mf if mf>0 else np.nan
def near_pairs(lam):
    L=np.asarray(lam,float); Ls=L.std(0); Ls=np.where(Ls==0,1,Ls); Ln=(L-L.mean(0))/Ls
    prs=list(itertools.combinations(range(len(L)),2))
    d=np.array([np.sqrt(((Ln[a]-Ln[b])**2).sum()) for a,b in prs])
    return [prs[i] for i in range(len(prs)) if d[i]<=np.percentile(d,10)]
def payoff_mat(y):
    S=np.exp(y); return np.array([np.maximum(S-K,0.0) for K in CALL_KS]+[(S>=1.0).astype(float)])
def trapw(y):
    dy=y[1]-y[0]; w=np.full(len(y),dy); w[0]=w[-1]=dy/2; return w

rows=[]
SRC=[("heston_RF",np.linspace(-5,5,300)),("heston_GB",np.linspace(-5,5,300)),
     ("kou_RF",np.linspace(-4,4,256)),("kou_GB",np.linspace(-4,4,256))]
for name,y in SRC:
    p=CROSS/f"ensemble_{name}.npz"
    if not p.exists(): print(f"  skip {name}: missing"); continue
    z=np.load(p,allow_pickle=True); M=z["matrix"]; L=z["lambdas"]; z.close()
    ymask=np.abs(y)<=YWIN; gps=np.where(ymask)[0]
    w=trapw(y)[gps]; P=payoff_mat(y)[:,gps]
    Vg=np.einsum('scg,pg->scp', M[:,:,gps], P*w[None,:])
    near=near_pairs(L)
    for (a,bb) in near:
        for pi,pl in enumerate(PAYS):
            d=Vg[:,a,pi]-Vg[:,bb,pi]
            mu=float(d.mean()); sg=float(d.std(ddof=0))
            if not (sg>0 and np.isfinite(mu)): continue
            s=abs(mu)/sg
            if s>5 or s<1e-3: continue
            g1=float(skew(d)); g2=float(kurtosis(d,fisher=True))
            emp=float(np.abs(d).mean()); th=folded(mu,sg)
            if th<=0: continue
            rows.append(dict(source="tree_percell", cell=f"{name}/{a}-{bb}/{pl}", s=s, gamma1=g1, gamma2=g2,
                ratio_obs=emp/th, ratio_pred_edge=edge_pred_ratio(s,g1,g2), ratio_pred_null=1.0,
                caveat=f"cross_class near-pair set (not b5-387); {name}; measured moments have sampling error"))
tp=pd.DataFrame(rows)
print(f"[7b] tree_percell cells: {len(tp)} (from {len(SRC)} sources)")

agg=[dict(source="tree_aggregate", cell="RF_block_median", s=np.nan, gamma1=np.nan, gamma2=8.0,
        ratio_obs=0.864, ratio_pred_edge=np.nan, ratio_pred_null=1.0,
        caveat="tree_aggregate: RF block median (not per-cell); RF kurt~8, obs ratio~0.864"),
     dict(source="tree_aggregate", cell="GB_block_median", s=np.nan, gamma1=np.nan, gamma2=15.0,
        ratio_obs=0.58, ratio_pred_edge=np.nan, ratio_pred_null=1.0,
        caveat="tree_aggregate: GB block median (not per-cell); GB kurt~15, obs ratio~0.58")]

ovp=HERE/"b11_overlay.csv"; base=pd.read_csv(ovp)
full=pd.concat([base, tp, pd.DataFrame(agg)], ignore_index=True)
full.to_csv(ovp, index=False)
print(f"[7b] appended -> b11_overlay.csv now {len(full)} rows (7a {len(base)} + tree_percell {len(tp)} + tree_aggregate 2)")

def summ(df):
    d=df.dropna(subset=["ratio_obs","ratio_pred_edge"])
    if not len(d): return (0,np.nan,0)
    return len(d), float((d.ratio_pred_edge-d.ratio_obs).abs().median()), int((np.sign(d.ratio_pred_edge-1)==np.sign(d.ratio_obs-1)).sum())
for src in ["tree_percell"]:
    n,ma,so=summ(full[full.source==src])
    print(f"  {src}: n={n} median|pred_edge-obs|={ma:.4f} sign-agree(vs 1) {so}/{n}")

if len(tp):
    tp2=tp.dropna(subset=["ratio_pred_edge"])
    for lo,hi in [(0,3),(3,8),(8,20),(20,1e9)]:
        b=tp2[(tp2.gamma2>=lo)&(tp2.gamma2<hi)]
        if len(b):
            print(f"  g2[{lo},{hi}): n={len(b)} obs_ratio med {b.ratio_obs.median():.3f} pred_edge med {b.ratio_pred_edge.median():.3f} edge_valid {int((b.ratio_pred_edge>0).sum())}/{len(b)}")
print(f"  s range {tp.s.min():.3f}~{tp.s.max():.3f} | gamma2 range {tp.gamma2.min():.2f}~{tp.gamma2.max():.2f}")
