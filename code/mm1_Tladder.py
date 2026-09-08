"""mm1_Tladder.py — M/M/1 SLA-20 rare-event run-length (T) ladder.
예측: T↑ → 레벨20 excursion↑ → zero-atom↓ → CLT 가우시안화(kurt→3) + |s|↑(σ_Δ~1/√T) → ratio→1.
시뮬레이션 코어는 mm1_queue.py와 동일(sim/ensure/Lindley 벡터화), T·블록만 파라미터화. burn=1000 고정.
T=10k 4셀 재계산은 atom(미저장) 추출용 — {s,ratio,kurt} 기준 CSV 일치 게이트.
FN(mu,sig)=sig√(2/π)e^{-s²/2}+mu·erf(s/√2), 계수측도. base a=0.70, m=1.
"""
import numpy as np, time, csv
from scipy.special import erf
from scipy import stats
t0=time.time()
UNI="<PROJECT_ROOT>/NeuMoR/output/unified"
OUT=UNI+"/mm1_Tladder.csv"; T1CSV=UNI+"/mm1_queue.csv"; NMAX=150
def FN(mu,sig):
    if sig<=0:return abs(mu)
    s=mu/sig;return sig*np.sqrt(2/np.pi)*np.exp(-s*s/2)+mu*erf(s/np.sqrt(2))
n_arr=np.arange(NMAX+1)
G={"EN":n_arr.astype(float),"SLA10":(n_arr>=10).astype(float),"SLA20":(n_arr>=20).astype(float),"over5":np.maximum(n_arr-5,0).astype(float)}
KEYS=["SLA20","SLA10","EN","over5"]  # SLA20 헤드라인
def Vtrue(a,k):return {"EN":a/(1-a),"SLA10":a**10,"SLA20":a**20,"over5":a**6/(1-a)}[k]
def ensure(ia,sv,rng,at,T):
    while np.cumsum(ia)[-1]<=at*T*1.10: ia=np.concatenate([ia,rng.exponential(1.0,len(ia))]);sv=np.concatenate([sv,rng.exponential(1.0,len(sv))])
    return ia,sv
def sim(ia,sv,a,T,burn):
    A=np.cumsum(ia)/a;m=A<T;A=A[m];S=sv[:len(ia)][m];K=len(A)
    C=np.cumsum(S);Cp=np.concatenate([[0.0],C[:-1]]);D=C+np.maximum.accumulate(A-Cp)
    et=np.concatenate([A,D]);ed=np.concatenate([np.ones(K),-np.ones(K)]);o=np.argsort(et,kind='stable');et=et[o];ed=ed[o]
    edg=np.concatenate([[0.0],et,[T]]);Nv=np.concatenate([[0.0],np.cumsum(ed)])
    lo=np.maximum(edg[:-1],burn);hi=np.minimum(edg[1:],T);w=np.clip(hi-lo,0.0,None)
    occ=np.bincount(np.clip(Nv,0,NMAX).astype(int),weights=w,minlength=NMAX+1)[:NMAX+1]
    valid=w>1e-15;mx=int(Nv[valid].max()) if valid.any() else 0
    return occ/occ.sum(),mx
def Vvec(pi):return np.array([pi@G[k] for k in KEYS])
DAS=[0.002,0.05]; PIDX={0.002:0,0.05:4}  # 시드 규칙은 mm1_queue.py와 동일
def run_T(T,seeds=200):
    block=20000 if T==10000 else int(0.9*T)+20000  # T=10k는 기준 실행 정확 재현(단일 rng ia-then-sv 분할 동일)
    Vb=np.zeros((seeds,4));Vc={da:np.zeros((seeds,4)) for da in DAS};Vi={da:np.zeros((seeds,4)) for da in DAS};bmax=[]
    for w in range(seeds):
        rng=np.random.default_rng(w);ia=rng.exponential(1,block);sv=rng.exponential(1,block);ia,sv=ensure(ia,sv,rng,0.85,T)
        pib,mb=sim(ia,sv,0.70,T,1000);Vb[w]=Vvec(pib);bmax.append(mb)
        for da in DAS:
            a2=0.70+da
            pic,_=sim(ia,sv,a2,T,1000);Vc[da][w]=Vvec(pic)
            r2=np.random.default_rng(10000*w+PIDX[da]);ia2=r2.exponential(1,block);sv2=r2.exponential(1,block);ia2,sv2=ensure(ia2,sv2,r2,0.85,T)
            pii,_=sim(ia2,sv2,a2,T,1000);Vi[da][w]=Vvec(pii)
    return Vb,Vc,Vi,np.array(bmax)
def bootci(D,B=200):
    o=[]
    for _ in range(B):
        r=D[np.random.randint(0,len(D),len(D))];mu=r.mean();sd=r.std(ddof=1)
        if sd>1e-18:o.append(np.mean(np.abs(r))/FN(mu,sd))
    return (np.percentile(o,25),np.percentile(o,75)) if o else (np.nan,np.nan)
def cell(Db,Dp,a2,key,T,mode):
    D=Db-Dp;mu=D.mean();sd=D.std(ddof=1);dtrue=Vtrue(0.70,key)-Vtrue(a2,key)
    atom=float(np.mean(np.abs(D)<=1e-15*max(1.0,abs(mu))))
    degen=(abs(mu)<1e-12 and sd<1e-12);s=mu/sd if sd>1e-18 else 0.0;fn=FN(mu,sd);mre=np.mean(np.abs(D))
    ratio=mre/fn if fn>0 else np.nan
    if sd>1e-18:
        z=(D-mu)/sd;ks=int(stats.kstest(z,'norm').pvalue<0.05);ad=stats.anderson(D,'norm');adr=int(ad.statistic>ad.critical_values[2]);sk=float(stats.skew(D));ku=float(stats.kurtosis(D,fisher=False))
    else: ks=adr=-1;sk=ku=np.nan
    lo,hi=bootci(D) if (not degen and sd>1e-18) else (np.nan,np.nan)
    return dict(part=2,T=T,da=round(a2-0.70,3),mode=mode,payoff=key,mu=mu,sigma=sd,s=s,ratio=ratio,ratio_lo=lo,ratio_hi=hi,
        atom=atom,KS_rej=ks,AD_rej=adr,skew=sk,kurt=ku,degen=int(degen),d_true=dtrue,b_delta=mu-dtrue,MR_emp=mre,FN=fn)

# ===== Part 1: T 사다리 =====
TS=[10000,30000,100000]; rows=[]; DATA={}
np.random.seed(2)
for T in TS:
    if T>=100000 and (time.time()-t0)>0:  # 45분 가드: T=100k 직전 잔여시간 점검
        proj=(time.time()-t0)  # 이미 경과
        if proj>1800: seeds=100; print(f"[가드] 경과 {proj:.0f}s — T=100k 100시드 축소")
        else: seeds=200
    else: seeds=200
    Vb,Vc,Vi,bmax=run_T(T,seeds); DATA[T]=(Vb,Vc,Vi,bmax,seeds)
    fr20=float(np.mean(bmax>=20))
    print(f"[T={T} seeds={seeds}] {time.time()-t0:.0f}s  base maxN min/med/max={bmax.min()}/{int(np.median(bmax))}/{bmax.max()} frac(maxN≥20)={fr20:.2f}"+(">120" if bmax.max()>120 else ""))
    for da in DAS:
        for mode,Vp in [("CRN",Vc[da]),("IND",Vi[da])]:
            for ki,key in enumerate(KEYS):
                rows.append(cell(Vb[:,ki],Vp[:,ki],0.70+da,key,T,mode))

# ===== 게이트: T=10k 재계산 vs 기준 CSV =====
with open(T1CSV) as f: t1=list(csv.DictReader(f))
def t1cell(da,mode,key):
    for r in t1:
        if r.get('part')=='2' and r.get('key')==key and r.get('mode')==mode and abs(float(r['a2'])-(0.70+da))<1e-9:
            return float(r['ratio']),float(r['s']),float(r['kurt'])
    return None
print("\n=== 게이트: T=10k 재계산 vs 기준 CSV (ratio/s/kurt) ===")
maxdiff=0
for da in DAS:
    for mode in ["CRN","IND"]:
        for key in KEYS:
            rr=[r for r in rows if r['T']==10000 and r['da']==da and r['mode']==mode and r['payoff']==key][0]
            t=t1cell(da,mode,key)
            if t:
                d=max(abs(rr['ratio']-t[0]),abs(rr['s']-t[1]),abs(rr['kurt']-t[2]));maxdiff=max(maxdiff,d)
print(f"  최대 |재계산-CSV| (ratio/s/kurt, 16셀) = {maxdiff:.2e}  ({'일치' if maxdiff<1e-6 else '불일치'})")

# ===== Part 3: 수리 곡선 (SLA20 across T) + 대조군 =====
print("\n=== Part3: 수리 곡선 SLA20 (T별 atom·kurt·|s|·ratio) ===")
print(f"{'da':>6s}{'mode':>5s}{'T':>8s}{'atom':>7s}{'kurt':>8s}{'|s|':>7s}{'ratio':>7s}{'[CI]':>13s}{'KS':>3s}{'AD':>3s}")
for da in DAS:
    for mode in ["CRN","IND"]:
        for T in TS:
            r=[x for x in rows if x['T']==T and x['da']==da and x['mode']==mode and x['payoff']=="SLA20"][0]
            print(f"{da:6.3f}{mode:>5s}{T:8d}{r['atom']:7.2f}{r['kurt']:8.2f}{abs(r['s']):7.2f}{r['ratio']:7.3f}[{r['ratio_lo']:.2f},{r['ratio_hi']:.2f}]{r['KS_rej']:3d}{r['AD_rej']:3d}")
print("\n  대조군 (SLA10/EN/over5) ratio 범위 (전 T·da·mode):")
for key in ["SLA10","EN","over5"]:
    rr=[x['ratio'] for x in rows if x['payoff']==key]
    kk=[x['kurt'] for x in rows if x['payoff']==key]
    print(f"    {key:6s}: ratio [{min(rr):.3f},{max(rr):.3f}]  kurt [{min(kk):.2f},{max(kk):.2f}]  atom [{min(x['atom'] for x in rows if x['payoff']==key):.3f},{max(x['atom'] for x in rows if x['payoff']==key):.3f}]")

# ===== Part 4: 배포 진단 규칙 (기술적 요약) =====
print("\n=== Part4: 진단 규칙 (ground truth 불요 — Δ̂ 앙상블만) ===")
# 신규 48셀 (atom·kurt·|s| 실측) + 기준 40셀 (kurt·|s| CSV, atom: non-SLA20≈0 / SLA20 미측정은 kurt·|s|분기)
newc=[(r['atom'],r['kurt'],abs(r['s']),r['ratio'],r['payoff'],'new') for r in rows if r['degen']==0]
t1c=[]
for r in t1:
    if r.get('part')=='2' and r.get('key') in KEYS and int(r['degen'])==0:
        atom_imp=0.0 if r['key']!="SLA20" else np.nan  # non-SLA20 atom≈0 (검증), SLA20 미측정
        t1c.append((atom_imp,float(r['kurt']),abs(float(r['s'])),float(r['ratio']),r['key'],'t1'))
allc=newc+t1c
# 규칙: ratio<0.9 를 (kurt>K AND |s|<S) 로 분리 (atom은 상관지표로 병기)
def rule(atom,kurt,sabs,K=5.0,S=1.5):
    a_ok=(atom==atom and atom>0.10)
    return a_ok or (kurt>K and sabs<S)
bad=[c for c in allc if c[3]<0.9]; good=[c for c in allc if c[3]>=0.9]
tp=sum(1 for c in bad if rule(*c[:3])); fp=sum(1 for c in good if rule(*c[:3]))
print(f"  규칙: atom>10% OR (kurt>5 AND |s|<1.5)  [전부 Δ̂ 앙상블서 계산가능, GT 불요]")
print(f"  ratio<0.9 셀 {len(bad)}개 중 {tp}개 포착(TP), ratio≥0.9 셀 {len(good)}개 중 {fp}개 오포착(FP)")
print(f"  ratio<0.9 셀 목록: "+", ".join(f"{c[4]}(r={c[3]:.2f},k={c[1]:.1f},|s|={c[2]:.2f},atom={c[0] if c[0]==c[0] else 'NA'})" for c in sorted(bad,key=lambda x:x[3])[:12]))
print(f"  [주의] 기술적 요약 — 일반 법칙 아님. atom: 신규셀 실측, 기준셀 non-SLA20 imputed 0, SLA20 미측정은 kurt·|s|분기로 포착.")

with open(OUT,'w',newline='') as f:
    keys=sorted(set().union(*[r.keys() for r in rows]));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();[w.writerow(r) for r in rows]
print(f"\nTOTAL {time.time()-t0:.0f}s  CSV -> {OUT} ({len(rows)}행)")
