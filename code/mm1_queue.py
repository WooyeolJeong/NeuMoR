"""비금융 인스턴스: M/M/1 대기행렬 시나리오 쌍비교로 folded-normal(Thm3.1)+CRN 상관처방(Thm4.1)
도메인 무관성 실증. 순수 이산사건 시뮬 출력 분석. 금융 코드 import 없음. 계수측도(합, dy 없음).
lambda=도착률 a, m=1, rho=a<1. 참분포 pi_a(n)=(1-rho)rho^n. 추정기=1런 시간평균 점유분포.
V(a,g)=Σ g(n)pi_a(n). Δ=V(a1,g)-V(a2,g). within-seed CRN=공통난수. NN/밀도연산자 없음.
FN(mu,sig)=sig*sqrt(2/pi)e^{-s^2/2}+mu*erf(s/sqrt2), s=mu/sig.
Lindley 벡터화: D[i]=C[i]+cummax(A[j]-C[j-1]), C=cumsum(service). CRN=고객인덱스 사전생성 exp 블록.
"""
import numpy as np, time, csv, sys
from scipy.special import erf
from scipy import stats
t0=time.time()
UNI="<PROJECT_ROOT>/NeuMoR/output/unified"
OUT=UNI+"/mm1_queue.csv"; NMAX=150
def FN(mu,sig):
    if sig<=0:return abs(mu)
    s=mu/sig;return sig*np.sqrt(2/np.pi)*np.exp(-s*s/2)+mu*erf(s/np.sqrt(2))
n_arr=np.arange(NMAX+1)
G={"EN":n_arr.astype(float),"SLA10":(n_arr>=10).astype(float),"SLA20":(n_arr>=20).astype(float),"over5":np.maximum(n_arr-5,0).astype(float)}
KEYS=["EN","SLA10","SLA20","over5"]
def Vtrue(a,key):
    if key=="EN":return a/(1-a)
    if key=="SLA10":return a**10
    if key=="SLA20":return a**20
    if key=="over5":return a**6/(1-a)
def ensure(ia,sv,rng,a_tgt,T):
    while np.cumsum(ia)[-1] <= a_tgt*T*1.10:
        ia=np.concatenate([ia,rng.exponential(1.0,len(ia))]);sv=np.concatenate([sv,rng.exponential(1.0,len(sv))])
    return ia,sv
def simulate(ia,sv,a,T,burn):
    A=np.cumsum(ia)/a; m=A<T; A=A[m]; S=sv[:len(ia)][m]; K=len(A)
    if K==0: return np.eye(NMAX+1)[0]*0+ (np.arange(NMAX+1)==0), 0
    C=np.cumsum(S); Cprev=np.concatenate([[0.0],C[:-1]]); D=C+np.maximum.accumulate(A-Cprev)
    ev_t=np.concatenate([A,D]); ev_d=np.concatenate([np.ones(K),-np.ones(K)])
    o=np.argsort(ev_t,kind='stable'); ev_t=ev_t[o]; ev_d=ev_d[o]
    edges=np.concatenate([[0.0],ev_t,[T]]); Nvals=np.concatenate([[0.0],np.cumsum(ev_d)])
    lo=np.maximum(edges[:-1],burn); hi=np.minimum(edges[1:],T); w=np.clip(hi-lo,0.0,None)
    Ni=np.clip(Nvals,0,NMAX).astype(int); occ=np.bincount(Ni,weights=w,minlength=NMAX+1)[:NMAX+1]
    tot=occ.sum(); pi=occ/tot if tot>0 else occ
    valid=w>1e-15; maxN=int(Nvals[valid].max()) if valid.any() else 0
    return pi,maxN
def Vvec(pi):return np.array([pi@G[k] for k in KEYS])

# ===== Part 0: smoke =====
print("=== Part0: smoke (단일 런 T=100000, burn 10000, a=0.70) ===")
rng=np.random.default_rng(0); ia=rng.exponential(1,20000); sv=rng.exponential(1,20000); ia,sv=ensure(ia,sv,rng,0.80,100000)
pi0,mx0=simulate(ia,sv,0.70,100000,10000)
pia=(1-0.70)*0.70**n_arr
print(f"  max|pi_hat-pi_true|={np.abs(pi0-pia).max():.2e}  Σpi_hat={pi0.sum():.6f}  maxN={mx0}")
print(f"  {'payoff':8s}{'V_hat':>12s}{'V_true':>12s}{'relerr':>10s}")
for k in KEYS:
    vh=pi0@G[k]; vt=Vtrue(0.70,k); print(f"  {k:8s}{vh:12.6f}{vt:12.6f}{abs(vh-vt)/abs(vt):10.2e}")
if np.abs(pi0-pia).max()>1e-2:
    print("  [게이트 실패] 시뮬레이터 정합 불량 — 중단"); sys.exit(1)
print(f"  smoke OK {time.time()-t0:.0f}s")

# ===== Part 1: 본실행 =====
NS=200; A_BASE=0.70; T=10000; BURN=1000; PAIRS=[0.002,0.005,0.01,0.02,0.05]
def run_all(pairs):
    Vb=np.zeros((NS,4)); Vc=np.zeros((NS,len(pairs),4)); Vi=np.zeros((NS,len(pairs),4)); mxs=[]
    for w in range(NS):
        rng=np.random.default_rng(w); ia=rng.exponential(1,20000); sv=rng.exponential(1,20000); ia,sv=ensure(ia,sv,rng,0.80,T)
        pib,mb=simulate(ia,sv,A_BASE,T,BURN); Vb[w]=Vvec(pib); mxs.append(mb)
        for pj,da in enumerate(pairs):
            a2=A_BASE+da
            pic,mc=simulate(ia,sv,a2,T,BURN); Vc[w,pj]=Vvec(pic); mxs.append(mc)
            r2=np.random.default_rng(10000*w+pj); ia2=r2.exponential(1,20000); sv2=r2.exponential(1,20000); ia2,sv2=ensure(ia2,sv2,r2,a2+0.05,T)
            pii,mi=simulate(ia2,sv2,a2,T,BURN); Vi[w,pj]=Vvec(pii); mxs.append(mi)
    return Vb,Vc,Vi,mxs
Vb,Vc,Vi,mxs=run_all(PAIRS)
print(f"\n=== Part1: 200시드×11런 완료 {time.time()-t0:.0f}s  maxN관측=[{min(mxs)},{max(mxs)}]"+(" WARN >120" if max(mxs)>120 else "")+" ===")

# ===== Part 2: Thm 3.1 =====
def bootci(D,B=200):
    o=[]
    for _ in range(B):
        r=D[np.random.randint(0,len(D),len(D))]; mu=r.mean(); sd=r.std(ddof=1)
        if sd>1e-18: o.append(np.mean(np.abs(r))/FN(mu,sd))
    return (np.percentile(o,25),np.percentile(o,75)) if o else (np.nan,np.nan)
def cell(Db,Dp,a2,key):
    D=Db-Dp; mu=D.mean(); sd=D.std(ddof=1); dtrue=Vtrue(A_BASE,key)-Vtrue(a2,key)
    degen=(abs(mu)<1e-12 and sd<1e-12)
    s=mu/sd if sd>1e-18 else 0.0; fn=FN(mu,sd); mre=np.mean(np.abs(D))
    ratio=mre/fn if fn>0 else np.nan; bd=mu-dtrue; J=fn-abs(mu); S=abs(mu)-abs(dtrue)
    ident=abs(dtrue)+J+S-fn
    if sd>1e-18:
        z=(D-mu)/sd; ks=int(stats.kstest(z,'norm').pvalue<0.05); ad=stats.anderson(D,'norm'); adr=int(ad.statistic>ad.critical_values[2])
    else: ks=adr=-1
    sk=float(stats.skew(D)) if sd>1e-18 else np.nan; ku=float(stats.kurtosis(D,fisher=False)) if sd>1e-18 else np.nan
    lo,hi=bootci(D) if not degen and sd>1e-18 else (np.nan,np.nan)
    return dict(a2=a2,key=key,mu=mu,sigma=sd,s=s,d_true=dtrue,b_delta=bd,J=J,S=S,ident_resid=ident,
        MR_emp=mre,FN=fn,ratio=ratio,ratio_lo=lo,ratio_hi=hi,KS_rej=ks,AD_rej=adr,skew=sk,kurt=ku,degen=int(degen),
        JJS=J/(J+abs(S)) if (J+abs(S))>0 else np.nan)
rows=[]
print("\n=== Part2: Thm3.1 (pair×mode×payoff) ===")
print(f"{'da':>6s}{'mode':>5s}{'pay':>6s}{'s':>8s}{'ratio':>7s}{'[CI]':>13s}{'KS':>3s}{'AD':>3s}{'skew':>7s}{'kurt':>7s}{'b_Δ':>10s}{'J/(J+|S|)':>10s}{'dg':>3s}")
for pj,da in enumerate(PAIRS):
    a2=A_BASE+da
    for mode,Vp in [("CRN",Vc),("IND",Vi)]:
        for ki,key in enumerate(KEYS):
            c=cell(Vb[:,ki],Vp[:,pj,ki],a2,key); c["mode"]=mode; c["part"]=2; rows.append(c)
            print(f"{da:6.3f}{mode:>5s}{key:>6s}{c['s']:8.2f}{c['ratio']:7.3f}[{c['ratio_lo']:.2f},{c['ratio_hi']:.2f}]{c['KS_rej']:3d}{c['AD_rej']:3d}{c['skew']:7.2f}{c['kurt']:7.2f}{c['b_delta']:10.2e}{c['JJS']:10.3f}{c['degen']:3d}")
# s-스펙트럼 가드
alls=[abs(r['s']) for r in rows if r['part']==2 and r['degen']==0]
print(f"\n  s-스펙트럼: |s| 범위 [{min(alls):.2f},{max(alls):.2f}]  (>10 전셀? {all(x>10 for x in alls)})")
if all(x>10 for x in alls):
    print("  전셀 |s|>10 → da=0.001 파트너 추가 (folding 활성 확보)")
    Vb1,Vc1,Vi1,_=run_all([0.001])
    for mode,Vp in [("CRN",Vc1),("IND",Vi1)]:
        for ki,key in enumerate(KEYS):
            c=cell(Vb1[:,ki],Vp[:,0,ki],0.701,key); c["mode"]=mode; c["part"]=2; rows.append(c)
            print(f"{0.001:6.3f}{mode:>5s}{key:>6s}{c['s']:8.2f}{c['ratio']:7.3f}[{c['ratio_lo']:.2f},{c['ratio_hi']:.2f}]{c['KS_rej']:3d}{c['AD_rej']:3d}{c['skew']:7.2f}{c['kurt']:7.2f}{c['b_delta']:10.2e}{c['JJS']:10.3f}{c['degen']:3d}")

# ===== Part 3: Thm 4.1 =====
print("\n=== Part3: Thm4.1 (CRN vs 독립) ===")
print(f"{'da':>6s}{'pay':>6s}{'ρ_CRN':>8s}{'ρ_IND':>8s}{'σΔ_IND/σΔ_CRN':>14s}{'MR_IND/MR_CRN':>14s}{'ratioCRN':>9s}{'ratioIND':>9s}")
for pj,da in enumerate(PAIRS):
    for ki,key in enumerate(KEYS):
        b=Vb[:,ki]; pc=Vc[:,pj,ki]; pi_=Vi[:,pj,ki]
        rc=np.corrcoef(b,pc)[0,1]; ri=np.corrcoef(b,pi_)[0,1]
        Dc=b-pc; Di=b-pi_; sc=Dc.std(ddof=1); si=Di.std(ddof=1)
        mrc=np.mean(np.abs(Dc)); mri=np.mean(np.abs(Di))
        ratc=mrc/FN(Dc.mean(),sc) if sc>1e-18 else np.nan; rati=mri/FN(Di.mean(),si) if si>1e-18 else np.nan
        rows.append(dict(part=3,a2=A_BASE+da,key=key,rho_CRN=rc,rho_IND=ri,sigrat=si/sc if sc>1e-18 else np.nan,mrrat=mri/mrc if mrc>1e-18 else np.nan,ratioCRN=ratc,ratioIND=rati))
        print(f"{da:6.3f}{key:>6s}{rc:8.3f}{ri:8.3f}{si/sc if sc>1e-18 else np.nan:14.2f}{mri/mrc if mrc>1e-18 else np.nan:14.2f}{ratc:9.3f}{rati:9.3f}")

# ===== Part 4: 초기화 편향 =====
print("\n=== Part4: 초기화 편향 (pair D da=0.05 CRN, burn 1000 vs 0) ===")
def crn_pairD(burn):
    Vb=np.zeros((NS,4)); Vp=np.zeros((NS,4))
    for w in range(NS):
        rng=np.random.default_rng(w); ia=rng.exponential(1,20000); sv=rng.exponential(1,20000); ia,sv=ensure(ia,sv,rng,0.85,T)
        pib,_=simulate(ia,sv,A_BASE,T,burn); Vb[w]=Vvec(pib)
        pip,_=simulate(ia,sv,0.75,T,burn); Vp[w]=Vvec(pip)
    return Vb,Vp
print(f"{'burn':>6s}{'pay':>6s}{'b_Δ':>11s}{'S':>11s}{'J':>11s}{'ratio':>8s}{'KS':>3s}{'AD':>3s}")
for burn in [1000,0]:
    Vb4,Vp4=crn_pairD(burn)
    for ki,key in enumerate(KEYS):
        c=cell(Vb4[:,ki],Vp4[:,ki],0.75,key); c["part"]=4; c["burn"]=burn; c["mode"]="CRN"; rows.append(c)
        print(f"{burn:6d}{key:>6s}{c['b_delta']:11.2e}{c['S']:11.2e}{c['J']:11.2e}{c['ratio']:8.3f}{c['KS_rej']:3d}{c['AD_rej']:3d}")

with open(OUT,'w',newline='') as f:
    keys=sorted(set().union(*[r.keys() for r in rows]));w_=csv.DictWriter(f,fieldnames=keys);w_.writeheader();[w_.writerow(r) for r in rows]
print(f"\nTOTAL {time.time()-t0:.0f}s  CSV -> {OUT} ({len(rows)}행)")
