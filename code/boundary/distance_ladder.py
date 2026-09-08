import numpy as np, os, sys, csv
from scipy.special import erf, erfc
from scipy import stats
np.random.seed(11)
ROOT="<PROJECT_ROOT>"; sys.path.insert(0,ROOT)
OUT=os.path.join(ROOT,"NeuMoR/output/unified/dist_ladder.csv")

def folded_mean(mu,sig):
    if sig<=0: return abs(mu)
    s=mu/sig; return sig*np.sqrt(2/np.pi)*np.exp(-s*s/2)+mu*erf(s/np.sqrt(2))
def J_term(mu,sig):
    if sig<=0: return 0.0
    s=mu/sig; return sig*(np.sqrt(2/np.pi)*np.exp(-s*s/2)-abs(s)*erfc(abs(s)/np.sqrt(2)))
def find_cfg(L,t,atol=1e-4): return [i for i,r in enumerate(L) if np.allclose(r,t,atol=atol)]
def boot(arr,B=100):
    R=[];n=len(arr)
    for _ in range(B):
        b=arr[np.random.randint(0,n,n)];m=b.mean();sd=b.std(ddof=1)
        if sd>1e-12:R.append(np.mean(np.abs(b))/folded_mean(m,sd))
    return (np.percentile(R,25),np.percentile(R,75)) if R else (np.nan,np.nan)

E=np.load(os.path.join(ROOT,"NeuMoR/output/cross_class/ensemble_heston_NN.npz"))
M,L=E["matrix"],E["lambdas"]
yg=np.load(os.path.join(ROOT,"NeuMoR/output/unified/heston_pairA.npz"),allow_pickle=True)["y_grid"].astype(float)
dy=float(yg[1]-yg[0]); base=[1.0,0.2,0.3,-0.5]; ncal=140
p1=M[:,find_cfg(L,[0.100]+base)[0],:].astype(float)
PAIRS={"A":0.101,"B":0.105,"C":0.110,"D":0.130,"E":0.200}
g_price=np.maximum(np.exp(yg)-1,0);g_delta=np.exp(yg)*(yg>0);g_dig=(yg>np.log(1.10)).astype(float)

dana_E=None; Tused=None
try:
    from NeuMoR.src.heston_teacher import lewis_pdf
    dana_A=np.load(os.path.join(ROOT,"NeuMoR/output/unified/heston_pairA.npz"),allow_pickle=True)["d_ana"].astype(float)
    best=None
    for T in [1.0, 30/365, 30/252, 0.5, 2.0]:
        pr={"v":0.100,"kappa":1.0,"omega":0.2,"xi":0.3,"rho":-0.5,"T":T}
        p_lo=lewis_pdf(pr,yg); pr2=dict(pr,v=0.101); p_hi=lewis_pdf(pr2,yg)
        d_test=p_lo-p_hi
        err=np.max(np.abs(d_test-dana_A))/max(np.max(np.abs(dana_A)),1e-12)
        if best is None or err<best[1]: best=(T,err,d_test)
    Tused,valerr,_=best
    print(f"[Lewis] pair A d_ana check: best T={Tused:.4f}, rel maxerr={valerr:.3f}")
    if valerr<0.05:
        pr={"v":0.100,"kappa":1.0,"omega":0.2,"xi":0.3,"rho":-0.5,"T":Tused}
        dana_E=lewis_pdf(pr,yg)-lewis_pdf(dict(pr,v=0.200),yg)
        print(f"[Lewis] pair E d_ana built OK (T={Tused:.4f})")
    else:
        print(f"[Lewis] check failed (err>{0.05}) -> pair E S/J skipped")
except Exception as ex:
    print(f"[Lewis] unavailable: {ex} -> pair E S/J skipped")

rows=[]
def emit(pair,kind,pname,g,p2,d_ana):
    D=(p1-p2)@g*dy
    mu=D.mean();sd=D.std(ddof=1)
    if sd<1e-12: rows.append(dict(pair=pair,payoff=pname,kind=kind,degenerate=1));return
    s=mu/sd;MRth=folded_mean(mu,sd);MRemp=np.mean(np.abs(D));q25,q75=boot(D)
    z=(D-mu)/sd;ksp=stats.kstest(z,'norm').pvalue;ad=stats.anderson(D,'norm')
    if d_ana is not None:
        Dtrue=float(g@d_ana*dy);S=abs(mu)-abs(Dtrue);Jf=J_term(mu,sd)/(J_term(mu,sd)+abs(S)) if (J_term(mu,sd)+abs(S))>0 else np.nan
        babs=abs(mu-Dtrue)
        brel=babs/max(abs(Dtrue),1e-12)
    else: S=np.nan;Jf=np.nan;brel=np.nan
    rows.append(dict(pair=pair,dv=round(PAIRS[pair]-0.1,4) if pair in PAIRS else np.nan,payoff=pname,kind=kind,degenerate=0,
        s=s,ratio=MRemp/MRth,boot_q25=q25,boot_q75=q75,KS_p=float(ksp),KS_rej=int(ksp<0.05),
        AD_rej=int(ad.statistic>ad.critical_values[2]),skew=float(stats.skew(D)),kurt=float(stats.kurtosis(D,fisher=False)),
        Jfrac=Jf,brel=brel,n=len(D)))

for pair,v2 in PAIRS.items():
    p2=M[:,find_cfg(L,[v2]+base)[0],:].astype(float)
    if pair in ("A","B","C","D"): d_ana=np.load(os.path.join(ROOT,f"NeuMoR/output/unified/heston_pair{pair}.npz"),allow_pickle=True)["d_ana"].astype(float)
    else: d_ana=dana_E
    for pn,g in [("price_ATMcall",g_price),("delta_ATM",g_delta),("digital_OTM110",g_dig)]:
        emit(pair,"std",pn,g,p2,d_ana)

p2E=M[:,find_cfg(L,[0.200]+base)[0],:].astype(float)
dhat_cal=(p1[:ncal]-p2E[:ncal]).mean(0)
coarse=np.linspace(-2,2,41);mc=np.array([((yg>k).astype(float))@dhat_cal*dy for k in coarse])
sgn=np.where(np.diff(np.sign(mc)))[0]
kstar=float(coarse[sgn[0]]) if len(sgn) else float(coarse[np.argmin(np.abs(mc))])
picks=[]
for k in np.linspace(-1.5,1.5,80):
    Dev=((p1-p2E)@((yg>k).astype(float))*dy)[ncal:]
    if Dev.std(ddof=1)>1e-12 and 0.5<=abs(Dev.mean()/Dev.std(ddof=1))<3: picks.append((k,abs(Dev.mean()/Dev.std(ddof=1))))
picks=sorted(picks,key=lambda x:x[1])[:3]
print(f"[pair E scan] k*={kstar:+.3f}; picked |s|_eval configs: {[(round(k,3),round(s,2)) for k,s in picks]}")
for k,_ in picks:
    D=((p1-p2E)@((yg>k).astype(float))*dy)
    emit("E",f"scan_k={k:+.2f}","digital_scan",(yg>k).astype(float),p2E,dana_E)

keys=sorted(set().union(*[r.keys() for r in rows]))
with open(OUT,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();[w.writerow(r) for r in rows]

print("\n===== Part1 distance ladder (all 200 in-sample) =====")
print(f"{'pair':5s}{'dv':>7s}{'payoff':16s}{'s':>8s}{'ratio':>7s}{'[q25':>7s}{'q75]':>7s}{'KSr':>4s}{'ADr':>4s}{'skew':>8s}{'kurt':>7s}{'Jfrac':>7s}{'brel':>7s}")
for r in rows:
    if r.get('degenerate'): print(f"{r['pair']:5s} {r['payoff']} DEGENERATE"); continue
    print(f"{r['pair']:5s}{(r['dv'] if not np.isnan(r['dv']) else 0):7.3f}{r['payoff']:16s}{r['s']:8.2f}{r['ratio']:7.3f}{r['boot_q25']:7.3f}{r['boot_q75']:7.3f}{r['KS_rej']:4d}{r['AD_rej']:4d}{r['skew']:8.3f}{r['kurt']:7.2f}{(r['Jfrac'] if not np.isnan(r['Jfrac']) else -9):7.3f}{(r['brel'] if not np.isnan(r['brel']) else -9):7.3f}")
print(f"\nCSV -> {OUT} ({len(rows)} rows). (Jfrac/brel=-9 = no d_ana)")
