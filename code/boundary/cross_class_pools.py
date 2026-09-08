import numpy as np, sys, time, csv
from scipy.special import erf
from scipy import stats
np.random.seed(4); t0=time.time()
ROOT="<PROJECT_ROOT>"; UNI=ROOT+"/NeuMoR/output/unified"; CC=ROOT+"/NeuMoR/output/cross_class"
OUT=UNI+"/mix_estimator.csv"
def FN(mu,sig):
    if sig<=0:return abs(mu)
    s=mu/sig;return sig*np.sqrt(2/np.pi)*np.exp(-s*s/2)+mu*erf(s/np.sqrt(2))
def mix_moments(comp):
    def raw(mu,s):return np.array([mu,mu**2+s**2,mu**3+3*mu*s**2,mu**4+6*mu**2*s**2+3*s**4])
    E=sum(w*raw(mu,s) for mu,s,w in comp);var=E[1]-E[0]**2
    if var<=0:return np.nan,np.nan
    m3=E[2]-3*E[0]*E[1]+2*E[0]**3;m4=E[3]-4*E[0]*E[2]+6*E[0]**2*E[1]-3*E[0]**4
    return m3/var**1.5,m4/var**2
def bootci(fn,arrs,B=200):
    o=[]
    for _ in range(B):
        rs=[a[np.random.randint(0,len(a),len(a))] for a in arrs];v=fn(*rs)
        if v==v:o.append(v)
    return (np.percentile(o,25),np.percentile(o,75)) if o else (np.nan,np.nan)

CFG={"heston":dict(base=[0.10,1,0.2,0.3,-0.5],part={"A":0.101,"B":0.105,"C":0.110,"D":0.130},ncal_nn=140,ncal_mc=140,neval_nn=(140,150)),
     "kou":   dict(base=[0.30,5,0.5,10,10],  part={"A":0.301,"B":0.305,"C":0.310,"D":0.330},ncal_nn=119,ncal_mc=140,neval_nn=(119,129))}
DATA={}
for cls in ["heston","kou"]:
    nn=np.load(CC+f"/ensemble_{cls}_NN.npz");mc=np.load(CC+f"/ensemble_{cls}_MC.npz")
    NNm=nn['matrix'].astype(float);MCm=mc['matrix'].astype(float);L=nn['lambdas']
    yg=np.load(UNI+f"/{cls}_pairA.npz",allow_pickle=True)["y_grid"].astype(float);dy=float(yg[1]-yg[0])
    def idx(vec):
        for i in range(len(L)):
            if np.allclose(L[i],vec,atol=1e-4):return i
    c=CFG[cls];bi=idx(c["base"])
    DATA[cls]=dict(NN=NNm,MC=MCm,yg=yg,dy=dy,bi=bi,pidx={pr:idx([v]+c["base"][1:]) for pr,v in c["part"].items()},
                   dana={pr:np.load(UNI+f"/{cls}_pair{pr}.npz",allow_pickle=True)["d_ana"].astype(float) for pr in "ABCD"})
def ddens(cls,est,pr):
    D=DATA[cls];M=D["NN"] if est=="NN" else D["MC"];return M[:,D["bi"],:]-M[:,D["pidx"][pr],:]
def payoffs(yg):
    return [("price",np.maximum(np.exp(yg)-1,0)),("delta",np.exp(yg)*(yg>0)),("digital",(yg>np.log(1.10)).astype(float))]
def Dvec(cls,est,pr,g,sl):
    return ddens(cls,est,pr)[sl]@g*DATA[cls]["dy"]

print("=== Part1a: estimator bias (mu - d_true)/sigma [cal] ===")
print(f"{'cls':7s}{'pair':5s}{'payoff':8s}{'Δ_true':>11s}{'bias_MC/σ':>11s}{'bias_NN/σ':>11s}{'sMC':>7s}{'sNN':>7s}")
rows=[]
for cls in ["heston","kou"]:
    D=DATA[cls];c=CFG[cls]
    for pr in "ABCD":
        for pn,g in payoffs(D["yg"]):
            dtrue=float(D["dana"][pr]@g*D["dy"])
            DM=Dvec(cls,"MC",pr,g,slice(0,c["ncal_mc"]));DN=Dvec(cls,"NN",pr,g,slice(0,c["ncal_nn"]))
            mM,sM=DM.mean(),DM.std(ddof=1);mN,sN=DN.mean(),DN.std(ddof=1)
            bM=(mM-dtrue)/sM if sM>1e-18 else np.nan;bN=(mN-dtrue)/sN if sN>1e-18 else np.nan
            rows.append(dict(part="1",cls=cls,pair=pr,payoff=pn,d_true=dtrue,bias_MC=bM,bias_NN=bN,sMC=mM/sM if sM>1e-18 else 0,sNN=mN/sN if sN>1e-18 else 0))
            if pn!="price" or pr in "AD":
                print(f"{cls:7s}{pr:5s}{pn:8s}{dtrue:11.2e}{bM:11.3f}{bN:11.3f}{mM/sM if sM>1e-18 else 0:7.2f}{mN/sN if sN>1e-18 else 0:7.2f}")

print("\n=== Part1b: digital sign change k*_true/k*_NN/k*_MC (0.002 grid, cal) ===")
KG=np.arange(-1.5,1.5001,0.002)
def zeros(dvec,yg,dy):
    sig=np.array([(dvec*(yg>k)).sum()*dy for k in KG]);zs=[]
    for i in range(len(KG)-1):
        if sig[i]*sig[i+1]<0:
            t=sig[i]/(sig[i]-sig[i+1]);zs.append(KG[i]+t*(KG[i+1]-KG[i]))
    return zs
def near0(zs):return min(zs,key=abs) if zs else float('nan')
print(f"{'cls':7s}{'pair':5s}{'k*true':>9s}{'k*NN':>9s}{'k*MC':>9s}{'|NN-MC|':>9s}{'|NN-tr|':>9s}")
ZMAP={}
for cls in ["heston","kou"]:
    D=DATA[cls];c=CFG[cls]
    for pr in "ABCD":
        kt=near0(zeros(D["dana"][pr],D["yg"],D["dy"]))
        kn=near0(zeros(ddens(cls,"NN",pr)[:c["ncal_nn"]].mean(0),D["yg"],D["dy"]))
        km=near0(zeros(ddens(cls,"MC",pr)[:c["ncal_mc"]].mean(0),D["yg"],D["dy"]))
        ZMAP[(cls,pr)]=(kt,kn,km);rows.append(dict(part="1z",cls=cls,pair=pr,k_true=kt,k_NN=kn,k_MC=km,gap_nnmc=abs(kn-km),gap_nntr=abs(kn-kt)))
        print(f"{cls:7s}{pr:5s}{kt:9.4f}{kn:9.4f}{km:9.4f}{abs(kn-km):9.4f}{abs(kn-kt):9.4f}")
rank=sorted([(abs(kn-km),cls,pr) for (cls,pr),(kt,kn,km) in ZMAP.items()],reverse=True)
print("  |k*_NN-k*_MC| ranking:",", ".join(f"{cls}{pr}={g:.3f}" for g,cls,pr in rank[:6]))

print("\n=== Part2A: straddle window candidates (|k*_NN-k*_MC|>=0.004, cal sign change) ===")
def sstat(cls,est,pr,k,sl):
    g=(DATA[cls]["yg"]>k).astype(float);D=Dvec(cls,est,pr,g,sl);sd=D.std(ddof=1)
    return D.mean(),sd,(D.mean()/sd if sd>1e-18 else 0.0)
CAND=[]
for (cls,pr),(kt,kn,km) in ZMAP.items():
    if abs(kn-km)<0.004 or np.isnan(kn) or np.isnan(km):
        print(f"  {cls}{pr}: gap={abs(kn-km):.4f} <0.004 -> no window"); continue
    c=CFG[cls];lo,hi=min(kn,km),max(kn,km)
    win=[k for k in KG if lo-1e-9<=k<=hi+1e-9]
    cells=[]
    for k in win:
        mN,sN,zN=sstat(cls,"NN",pr,k,slice(0,c["ncal_nn"]));mM,sM,zM=sstat(cls,"MC",pr,k,slice(0,c["ncal_mc"]))
        if np.sign(mN)!=np.sign(mM) and abs(zN)>1e-6 and abs(zM)>1e-6:
            cells.append(dict(k=k,zN=zN,zM=zM,minabs=min(abs(zN),abs(zM))))
    cells.sort(key=lambda x:-x["minabs"]);pick=cells[:4]
    print(f"  {cls}{pr}: window[{lo:.3f},{hi:.3f}] sign-change cells {len(cells)}; picks: "+(", ".join(f"k={c2['k']:.3f}(sNN={c2['zN']:.2f},sMC={c2['zM']:.2f})" for c2 in pick) or "none"))
    for c2 in pick: CAND.append((cls,pr,c2["k"]))

with open(UNI+"/bias_removal_results.csv") as f: brows=list(csv.DictReader(f))
CASEC=[r for r in brows if float(r["Delta"])!=0 and np.sign(float(r["Delta"])+float(r["b_Delta"]))!=np.sign(float(r["Delta"]))]
print(f"\n=== Part2B: Case C = {len(CASEC)} cells / {len(brows)} rows ===")
for r in CASEC[:20]: print(f"  {r['model']} {r['block']} {r['payoff']} cfg={r['config_id']} Δ={float(r['Delta']):.2e} b={float(r['b_Delta']):.2e}")

def pool_metrics(D1,D2):
    Dp=np.concatenate([D1,D2]);m1,s1=D1.mean(),D1.std(ddof=1);m2,s2=D2.mean(),D2.std(ddof=1)
    mp,sp=Dp.mean(),Dp.std(ddof=1);naive=FN(mp,sp);strat=0.5*FN(m1,s1)+0.5*FN(m2,s2);ma=np.mean(np.abs(Dp))
    return dict(pg=strat/naive if naive>0 else np.nan,rn=ma/naive if naive>0 else np.nan,rs=ma/strat if strat>0 else np.nan,
               m1=m1,s1=s1,m2=m2,s2=s2,mp=mp,sp=sp,ma=ma,z1=m1/s1 if s1>1e-18 else 0,z2=m2/s2 if s2>1e-18 else 0,zp=mp/sp if sp>1e-18 else 0)
def ksad(D):
    if D.std(ddof=1)<1e-18:return -1,-1
    z=(D-D.mean())/D.std(ddof=1);return int(stats.kstest(z,'norm').pvalue<0.05),int(stats.anderson(D,'norm').statistic>stats.anderson(D,'norm').critical_values[2])
print("\n=== Part3A: pooled tests (headline eval n=20, full n=60 alongside) ===")
print(f"{'cell':14s}{'sc':>3s}{'sNNe':>6s}{'sMCe':>6s}{'spe':>6s}{'pred_gap':>9s}{'r_naive':>9s}{'r_strat':>8s}{'idOK':>5s}{'KSe':>4s}{'ADe':>4s}{'KSf':>4s}{'ADf':>4s}{'ku_o':>6s}{'ku_p':>6s}")
p3=[]
for cls,pr,k in CAND:
    c=CFG[cls];g=(DATA[cls]["yg"]>k).astype(float)
    e0,e1=c["neval_nn"]
    D1e=Dvec(cls,"NN",pr,g,slice(e0,e0+10));D2e=Dvec(cls,"MC",pr,g,slice(140,150))
    D1f=Dvec(cls,"NN",pr,g,slice(0,30));D2f=Dvec(cls,"MC",pr,g,slice(0,30))
    me=pool_metrics(D1e,D2e);mf=pool_metrics(D1f,D2f)
    Dpe=np.concatenate([D1e,D2e]);Dpf=np.concatenate([D1f,D2f])
    strad_e=(np.sign(me["m1"])!=np.sign(me["m2"])) and abs(me["z1"])>1e-6 and abs(me["z2"])>1e-6
    degen=(abs(me["mp"])<1e-12 and me["sp"]<1e-12)
    KSe,ADe=ksad(Dpe);KSf,ADf=ksad(Dpf)
    sk_p,ku_p=mix_moments([(me["m1"],me["s1"],0.5),(me["m2"],me["s2"],0.5)])
    sk_o=float(stats.skew(Dpe));ku_o=float(stats.kurtosis(Dpe,fisher=False))
    idok=int(abs(me["rn"]-me["pg"]*me["rs"])<1e-9) if me["rn"]==me["rn"] else -1
    pg_ci=bootci(lambda a,b:pool_metrics(a,b)["pg"],[D1e,D2e]);rn_ci=bootci(lambda a,b:pool_metrics(a,b)["rn"],[D1e,D2e]);rs_ci=bootci(lambda a,b:pool_metrics(a,b)["rs"],[D1e,D2e])
    nm=f"{cls[:3]}{pr}k{k:.2f}"
    p3.append(dict(part="3A",cls=cls,pair=pr,k=k,strad_eval=int(strad_e),degen=int(degen),sNN_e=me["z1"],sMC_e=me["z2"],spool_e=me["zp"],
        pred_gap=me["pg"],pg_lo=pg_ci[0],pg_hi=pg_ci[1],r_naive=me["rn"],rn_lo=rn_ci[0],rn_hi=rn_ci[1],r_strat=me["rs"],rs_lo=rs_ci[0],rs_hi=rs_ci[1],
        id_ok=idok,KS_eval=KSe,AD_eval=ADe,KS_full=KSf,AD_full=ADf,kurt_obs=ku_o,kurt_pred=ku_p,skew_obs=sk_o,skew_pred=sk_p,
        pred_gap_full=mf["pg"],r_naive_full=mf["rn"],r_strat_full=mf["rs"],sNN_f=mf["z1"],sMC_f=mf["z2"]))
    print(f"{nm:14s}{int(strad_e):>3d}{me['z1']:6.2f}{me['z2']:6.2f}{me['zp']:6.2f}{me['pg']:9.3f}{me['rn']:9.3f}{me['rs']:8.3f}{idok:>5d}{KSe:>4d}{ADe:>4d}{KSf:>4d}{ADf:>4d}{ku_o:6.2f}{ku_p:6.2f}")

allrows=rows+p3
with open(OUT,'w',newline='') as f:
    keys=sorted(set().union(*[r.keys() for r in allrows]));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();[w.writerow(r) for r in allrows]

LIM=np.sqrt(np.pi/2)
print(f"\n=== Part4: summary (sqrt(pi/2)={LIM:.4f}) ===")
if p3:
    top=max(p3,key=lambda r:r["pred_gap"] if r["pred_gap"]==r["pred_gap"] else -9)
    print(f"  max pred_gap(eval): {top['cls']}{top['pair']} k={top['k']:.3f} pred_gap={top['pred_gap']:.3f}[{top['pg_lo']:.2f},{top['pg_hi']:.2f}] r_naive={top['r_naive']:.3f}[{top['rn_lo']:.2f},{top['rn_hi']:.2f}] r_strat={top['r_strat']:.3f} (attainment {top['pred_gap']/LIM*100:.0f}% of {LIM:.3f})")
    print(f"    full: pred_gap_full={top['pred_gap_full']:.3f} r_naive_full={top['r_naive_full']:.3f} sNN_f={top['sNN_f']:.2f} sMC_f={top['sMC_f']:.2f}")
    rz=[r for r in p3 if r["strad_eval"]==1];print(f"  eval straddle realized: {len(rz)}/{len(p3)} cells")
    print(f"  stratified r_strat range: [{min(r['r_strat'] for r in p3):.3f},{max(r['r_strat'] for r in p3):.3f}]")
else: print("  no candidates - Part3 skipped")
print(f"  Case C: {len(CASEC)} cells")
print(f"\nTOTAL {time.time()-t0:.0f}s CSV -> {OUT} ({len(allrows)} rows)")
