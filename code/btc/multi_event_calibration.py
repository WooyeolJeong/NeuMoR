
import sys
import time
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from scipy.special import erf

ROOT    = Path(__file__).resolve().parents[2]
UNIFIED = ROOT / "NeuMoR/output/unified"
UNIFIED.mkdir(parents=True, exist_ok=True)
FIGURES = ROOT / "NeuMoR/figures"
FIGURES.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
from TNO.common.tno_model import DensityDeepONet_HestonLog
from NeuMoR.src.heston_teacher import lewis_pdf
from NeuMoR.src.kou.kou_pricing import kou_log_density

DATA_PATH       = Path("<BTC_SPOT_1M_PARQUET>")
EXP09_CSV       = ROOT / "NeuMoR/output/exp09/calibrations.csv"
FEASIBILITY_CSV = ROOT / "NeuMoR/output/kou/btc_feasibility.csv"
B6_THM35_CSV    = ROOT / "NeuMoR/output/unified/b6_btc_thm35.csv"

DEVICE = ("mps" if torch.backends.mps.is_available() else
          "cuda" if torch.cuda.is_available() else "cpu")
N_SEEDS = 200
N_CAL   = 150
N_TEST  = 50

H_BOUNDS = {"v":(0.051,0.300),"kappa":(0.502,2.000),"omega":(0.100,0.399),
             "xi":(0.102,0.599),"rho":(-0.800,0.598)}
KOU_LO = np.array([0.10, 0.5,  0.30, 3.0,  3.0],  dtype=np.float32)
KOU_HI = np.array([0.80, 50.0, 0.70, 50.0, 50.0], dtype=np.float32)
KOU_BOX = {"sigma":(0.10,0.80),"lam_J":(0.5,50.0),"p":(0.30,0.70),
           "eta_1":(3.0,50.0),"eta_2":(3.0,50.0)}

C_SWEEP = [2.0, 2.5, 3.0]
K_DAILY = 22

WINDOWS = [
    ("Luna",    "pre",  "2022-02-01","2022-04-30"),
    ("Luna",    "post", "2022-06-01","2022-08-31"),
    ("FTX",     "pre",  "2022-09-01","2022-11-08"),
    ("FTX",     "post", "2022-11-16","2023-01-31"),
    ("ETF",     "pre",  "2023-10-01","2023-12-31"),
    ("ETF",     "post", "2024-02-01","2024-04-30"),
    ("COVID",   "pre",  "2020-01-01","2020-02-29"),
    ("COVID",   "post", "2020-04-01","2020-05-31"),
    ("China",   "pre",  "2021-03-01","2021-04-30"),
    ("China",   "post", "2021-06-01","2021-07-31"),
    ("3AC",     "pre",  "2022-05-16","2022-07-10"),
    ("3AC",     "post", "2022-07-16","2022-09-15"),
    ("SVB",     "pre",  "2023-01-01","2023-02-28"),
    ("SVB",     "post", "2023-04-01","2023-05-31"),
    ("Halving", "pre",  "2024-02-01","2024-03-31"),
    ("Halving", "post", "2024-05-01","2024-06-30"),
]
EXISTING = {"Luna","FTX","ETF"}
FEASIB_MAP = {("Luna","pre"):"Luna_pre",("Luna","post"):"Luna_post",
              ("FTX","pre"):"FTX_pre",("FTX","post"):"FTX_post",
              ("ETF","pre"):"ETF_pre",("ETF","post"):"ETF_post"}

PAYOFFS_DEF = {
    "ATM_call":     lambda y: np.maximum(np.exp(y) - 1.0,    0.0),
    "OTM_put_m030": lambda y: np.maximum(np.exp(-0.30) - np.exp(y), 0.0),
    "OTM_put_m050": lambda y: np.maximum(np.exp(-0.50) - np.exp(y), 0.0),
}

t_start = time.time()

print("="*70); print("PHASE 0: Sanity check"); print("="*70)

raw_df  = pd.read_parquet(DATA_PATH)
ts      = pd.to_datetime(raw_df["timestamp"])
price1m = pd.Series(raw_df["close"].values, index=ts, dtype=np.float64)
price5m = price1m.resample("5min").last().dropna()
log5m   = np.log(price5m).diff().dropna()
price_daily_full = price5m.resample("1D").last().dropna()
log_daily_full   = np.log(price_daily_full).diff().dropna()

print(f"  span: {ts.iloc[0].date()} ~ {ts.iloc[-1].date()}")
print(f"  5-min bars: {len(log5m):,}  |  daily bars: {len(log_daily_full):,}\n")

for event, window, d0, d1 in WINDOWS:
    t0 = pd.Timestamp(d0, tz="UTC"); t1 = pd.Timestamp(d1+" 23:59:59", tz="UTC")
    sub = log5m[t0:t1]
    exp = (pd.Timestamp(d1)-pd.Timestamp(d0)).days * 24 * 12
    pct = len(sub)/exp if exp else 0
    if pct < 0.5:
        print(f"  FAIL {event} {window}: {len(sub):,}/{exp:,} ({100*pct:.0f}%)"); sys.exit(1)
    gaps = int((sub.index.to_series().diff().dropna() > pd.Timedelta("1h")).sum())
    print(f"  {event:8} {window:4}: {len(sub):6,} bars ({100*pct:.0f}%)  gaps>1h:{gaps}")
print("\nPHASE 0 PASS\n")

print("="*70); print("PHASE 1: Heston MoM calibration"); print("="*70)

def clip_heston(raw):
    cl, wc = {}, {}
    for k,v in raw.items():
        lo,hi = H_BOUNDS[k]; c = float(np.clip(v,lo,hi))
        cl[k]=c; wc[k]=(abs(c-v)>1e-8)
    return cl, wc

heston_cal = {}
for event, window, d0, d1 in WINDOWS:
    t0=pd.Timestamp(d0,tz="UTC"); t1=pd.Timestamp(d1+" 23:59:59",tz="UTC")
    sub = log5m[t0:t1]
    rv  = (sub**2).resample("1D").sum()*252
    rv  = rv[rv>0]; arr=rv.values; n=len(arr)
    if n < 5:
        print(f"  {event:8} {window:4}: SKIP n_rv={n}"); heston_cal[(event,window)]=None; continue
    v_r   = float(arr.mean())
    om_r  = float(np.median(arr))
    rv_lag= arr[:-1]-arr[:-1].mean(); rv_lead=arr[1:]-arr[:-1].mean()
    ar1   = float(np.clip(np.dot(rv_lag,rv_lead)/max(np.dot(rv_lag,rv_lag),1e-20),0.01,0.999))
    kap_r = float(-np.log(ar1)*252)
    xi_r  = float(np.std(np.diff(arr),ddof=1)*np.sqrt(252)) if n>=4 else 0.3
    r5    = sub.values
    try:    rho_r = float(np.corrcoef(np.sign(r5[:-1]), r5[1:]**2)[0,1])
    except: rho_r = -0.5
    raw = {"v":v_r,"kappa":kap_r,"omega":om_r,"xi":xi_r,"rho":rho_r}
    cl, wc = clip_heston(raw)
    in_b = all(H_BOUNDS[k][0]<=cl[k]<=H_BOUNDS[k][1] for k in H_BOUNDS)
    out_p= [k for k in H_BOUNDS if not(H_BOUNDS[k][0]<=cl[k]<=H_BOUNDS[k][1])]
    heston_cal[(event,window)] = {"n_5m":len(sub),"n_rv":n,"raw":raw,"cl":cl,"wc":wc,
                                   "in_box":in_b,"out_p":out_p,"d0":d0,"d1":d1}
    cs = ",".join(k for k,v in wc.items() if v) or "none"
    print(f"  {event:8} {window:4}: v={cl['v']:.3f} κ={cl['kappa']:.3f} "
          f"ω={cl['omega']:.3f} ξ={cl['xi']:.3f} ρ={cl['rho']:.3f}  clip=[{cs}]")
print()

print("="*70); print("PHASE 2: Kou σ_diff — 5-min BNS bipower"); print("="*70)

sigma_diff_map = {}
for event, window, d0, d1 in WINDOWS:
    t0=pd.Timestamp(d0,tz="UTC"); t1=pd.Timestamp(d1+" 23:59:59",tz="UTC")
    sub=log5m[t0:t1]; n=len(sub); T=n*(5/60/24/365)
    a=np.abs(sub.values)
    BV=(np.pi/2)*(1/(n-1))*np.sum(a[1:]*a[:-1])
    sd=float(np.sqrt(max(BV*(n/T),1e-20)))
    sigma_diff_map[(event,window)]=sd
    print(f"  {event:8} {window:4}: σ_diff={sd:.4f}")
print()

print("="*70); print("PHASE 3: Kou jump detection (daily, η pooled)"); print("="*70)

def rolling_bipower_d(vals, K):
    n=len(vals); a=np.abs(vals); out=np.full(n,np.nan)
    for i in range(K,n):
        w=a[i-K:i]; bv=(np.pi/2)*np.mean(w[1:]*w[:-1]); out[i]=np.sqrt(max(bv,1e-20))
    return out

vals_full    = log_daily_full.values
sigma_l_full = rolling_bipower_d(vals_full, K_DAILY)
valid_full   = ~np.isnan(sigma_l_full)
L_full       = np.full(len(vals_full), np.nan)
L_full[valid_full] = vals_full[valid_full] / sigma_l_full[valid_full]

global_params = {}
for c in C_SWEEP:
    jmask = np.abs(L_full) > c
    det   = vals_full[jmask]
    n_j   = len(det)
    p_g   = float(np.sum(det>0)/n_j) if n_j>0 else float("nan")
    up    = det[det>0]; dn = -det[det<0]
    e1    = float(1/np.mean(up))  if len(up)>=3  else float("nan")
    e2    = float(1/np.mean(dn)) if len(dn)>=3  else float("nan")
    global_params[c] = {"p":p_g,"eta_1":e1,"eta_2":e2,"n_jumps_full":n_j}
    print(f"  c={c:.1f} full-history: n={n_j}  p={p_g:.3f}  η1={e1:.2f}  η2={e2:.2f}")

lam_J_map = {}
for event, window, d0, d1 in WINDOWS:
    t0=pd.Timestamp(d0,tz="UTC"); t1=pd.Timestamp(d1+" 23:59:59",tz="UTC")
    sub_d = log_daily_full[t0:t1]; n_d=len(sub_d)
    T_w = n_d/252.0

    idx_full = log_daily_full.index
    idx_w    = sub_d.index
    pos_in_full = np.array([idx_full.get_loc(i) for i in idx_w], dtype=int)
    sl_w = sigma_l_full[pos_in_full]
    L_w  = np.full(n_d, np.nan)
    v_w  = ~np.isnan(sl_w)
    L_w[v_w] = sub_d.values[v_w] / sl_w[v_w]
    for c in C_SWEEP:
        jm = np.abs(L_w) > c
        lam_J_map[(event,window,c)] = float(np.sum(jm)/T_w) if T_w>0 else float("nan")
    r25 = lam_J_map[(event,window,2.5)]
    print(f"  {event:8} {window:4}: λJ@2.5={r25:.1f}")
print()

print("="*70); print("PHASE 4: Kou c selection"); print("="*70)

feasib_df  = pd.read_csv(FEASIBILITY_CSV)
feasib     = {r["scenario"]:{"sigma_diff":float(r["sigma_diff"]),"lam_J":float(r["lam_J"]),
              "p":float(r["p_up"]),"eta_1":float(r["eta1"]),"eta_2":float(r["eta2"])}
              for _,r in feasib_df.iterrows()}

val_rows, sum_rows = [], []
for c in C_SWEEP:
    gp = global_params[c]; diffs=[]
    for (ev,win), sk in FEASIB_MAP.items():
        old = feasib[sk]
        new = {"sigma_diff": sigma_diff_map[(ev,win)],
               "lam_J":      lam_J_map[(ev,win,c)],
               "p":          gp["p"], "eta_1":gp["eta_1"], "eta_2":gp["eta_2"]}
        row = {"event":ev,"window":win,"c":c}
        for param in ["sigma_diff","lam_J","p","eta_1","eta_2"]:
            nv,ov = new[param],old[param]
            dp = abs(nv-ov)/abs(ov) if (not np.isnan(nv) and abs(ov)>1e-20) else float("nan")
            row[f"{param}_old"]=round(ov,4); row[f"{param}_new"]=round(nv,4)
            row[f"{param}_diff_pct"]=round(dp*100,1) if not np.isnan(dp) else float("nan")
            if not np.isnan(dp): diffs.append(dp)
        val_rows.append(row)
    med=float(np.nanmedian(diffs))*100; mx=float(np.nanmax(diffs))*100
    pass_=(med<=20.0 and mx<50.0)
    sum_rows.append({"c":c,"median_diff_pct":round(med,1),"max_diff_pct":round(mx,1),"pass":pass_})
    print(f"  c={c:.1f}: median={med:.1f}%  max={mx:.1f}%  [{'PASS' if pass_ else 'FAIL'}]")

passing = [r for r in sum_rows if r["pass"]]
best_c  = min(passing, key=lambda r:r["median_diff_pct"])["c"] if passing else 2.5
action  = "PASS-adopt" if passing else "FAIL-default-2.5"
for r in sum_rows: r["best_c"]=(r["c"]==best_c)
print(f"\n  Best c={best_c}  ({action})\n")

gp_best = global_params[best_c]

print("="*70); print("PHASE 5: Stage 1 output"); print("="*70)

h_rows = []
for event,window,d0,d1 in WINDOWS:
    h = heston_cal.get((event,window))
    if h is None: continue
    in_b,out_p = h["in_box"],h["out_p"]
    h_rows.append(dict(event=event,window=window,date_start=d0,date_end=d1,
        n_5m_bars=h["n_5m"],n_rv_days=h["n_rv"],
        **{f"raw_{k}":round(h["raw"][k],5) for k in H_BOUNDS},
        **{f"clipped_{k}":round(h["cl"][k],5) for k in H_BOUNDS},
        **{f"was_clipped_{k}":h["wc"][k] for k in H_BOUNDS},
        in_training_box=in_b,
        out_of_range_params=",".join(out_p) if out_p else ""))
df_h = pd.DataFrame(h_rows)

def kou_in_box(sigma_d, lam_j, p_, e1, e2):
    params = {"sigma":sigma_d,"lam_J":lam_j,"p":p_,"eta_1":e1,"eta_2":e2}
    out = [k for k,(lo,hi) in KOU_BOX.items()
           if np.isnan(params[k]) or not(lo<=params[k]<=hi)]
    return len(out)==0, out

k_rows = []
for event,window,d0,d1 in WINDOWS:
    sd  = sigma_diff_map[(event,window)]
    lj  = lam_J_map[(event,window,best_c)]
    p_  = gp_best["p"]; e1=gp_best["eta_1"]; e2=gp_best["eta_2"]
    nj  = int(np.sum(np.abs(log_daily_full[pd.Timestamp(d0,tz="UTC"):
                                           pd.Timestamp(d1+" 23:59:59",tz="UTC")].values /
              sigma_l_full[np.where(valid_full)[0][:len(log_daily_full[pd.Timestamp(d0,tz="UTC"):
                                                                       pd.Timestamp(d1+" 23:59:59",tz="UTC")])]]
              if False else []) > best_c))
    in_b, out_p = kou_in_box(sd,lj,p_,e1,e2)
    k_rows.append(dict(event=event,window=window,date_start=d0,date_end=d1,
        n_5m_bars=len(log5m[pd.Timestamp(d0,tz="UTC"):pd.Timestamp(d1+" 23:59:59",tz="UTC")]),
        sigma_diff=round(sd,4),lam_J=round(lj,2),
        p=round(p_,3),eta_1=round(e1,2),eta_2=round(e2,2),
        best_c=best_c,pooled_source="BTC full history 2017-08~2026-03",
        in_training_box=in_b,out_of_range_params=",".join(out_p) if out_p else ""))
df_k = pd.DataFrame(k_rows)

df_pool = pd.DataFrame([{"best_c":best_c,"n_jumps_full":gp_best["n_jumps_full"],
    "p_global":round(gp_best["p"],4),"eta_1_global":round(gp_best["eta_1"],4),
    "eta_2_global":round(gp_best["eta_2"],4)}])

df_kval  = pd.DataFrame(val_rows)
df_ksum  = pd.DataFrame(sum_rows)

exp09 = pd.read_csv(EXP09_CSV)
EXP09_MAP = {}
for _,row in exp09.iterrows():
    wn=row["window"]
    if "luna" in wn:   key=("Luna","post")
    elif "ftx" in wn:  key=("FTX","post")
    elif "etf" in wn:  key=("ETF","post")
    else: continue
    EXP09_MAP[key]={k:float(row[f"clipped_{k}"]) for k in ["v","kappa","omega","xi","rho"]}

hv_rows=[]; all_hdiff=[]
for key,old_p in EXP09_MAP.items():
    h=heston_cal.get(key)
    if h is None: continue
    new_p=h["cl"]; row={"event":key[0],"window":key[1]}; dw=[]
    for param in ["v","kappa","omega","xi","rho"]:
        ov,nv=old_p[param],new_p[param]
        dp=abs(nv-ov)/abs(ov) if abs(ov)>1e-10 else float("nan")
        row[f"{param}_old"]=round(ov,5); row[f"{param}_new"]=round(nv,5)
        row[f"{param}_diff_pct"]=round(dp*100,1) if not np.isnan(dp) else float("nan")
        if not np.isnan(dp): dw.append(dp); all_hdiff.append(dp)
    row["median_diff_pct"]=round(float(np.nanmedian(dw))*100,1)
    hv_rows.append(row)
    print(f"  {key[0]:8} {key[1]:4}: median_diff={row['median_diff_pct']}%")

h_overall = float(np.nanmedian(all_hdiff))*100
h_pass    = h_overall<=30.0
df_hval   = pd.DataFrame(hv_rows)
print(f"\n  Overall Heston diff: {h_overall:.1f}%  [{'PASS' if h_pass else 'FAIL'}]\n")

print("="*70); print("PHASE 6: Training box check"); print("="*70)

h_in=sum(1 for r in h_rows if r["in_training_box"])
k_in=sum(1 for r in k_rows if r["in_training_box"])
print(f"  Heston in-range: {h_in}/16  |  Kou in-range: {k_in}/16")
for r in k_rows:
    if r["out_of_range_params"]:
        print(f"    K {r['event']:8} {r['window']:4}: OUT [{r['out_of_range_params']}]")
print()

print("="*70); print("PHASE 8: NN Inference (200 seeds × 16 windows × 2 models)"); print("="*70)

print("  Loading Heston 200 seeds...", flush=True)
SA=ROOT/"NeuMoR/save/exp01"; SB=ROOT/"NeuMoR/save/exp07_5"
h_models,h_y,h_pk=[],None,None
for s in range(N_SEEDS):
    p=(SA/f"seed_{s:02d}.pt") if s<50 else (SB/f"seed_{s:03d}.pt")
    if not p.exists(): print(f"  MISSING {p}"); sys.exit(1)
    ck=torch.load(p,map_location=DEVICE,weights_only=False)
    cfg=ck["config"]
    m=DensityDeepONet_HestonLog(lambda_dim=cfg["lambda_dim"],n_y=cfg["n_y"],
                                 rank=cfg["rank"],branch_hidden=cfg["branch_hidden"]).to(DEVICE)
    m.load_state_dict(ck["state_dict"]); m.eval(); h_models.append(m)
    if h_y is None: h_y=ck["y_grid"].astype(np.float64); h_pk=list(ck["param_keys"])
print(f"  Heston y: [{h_y[0]:.1f},{h_y[-1]:.1f}]  n_y={len(h_y)}")

print("  Loading Kou 200 seeds...", flush=True)
SA2=ROOT/"NeuMoR/save/kou/stageA"; SB2=ROOT/"NeuMoR/save/kou/stageB"
k_models=[]
for s in range(N_SEEDS):
    p=(SA2/f"seed_{s:04d}.pt") if s<30 else (SB2/f"seed_{s:04d}.pt")
    if not p.exists(): print(f"  MISSING {p}"); sys.exit(1)
    ck=torch.load(p,map_location=DEVICE,weights_only=False)
    m=DensityDeepONet_HestonLog(lambda_dim=5,n_y=256,rank=192,branch_hidden=512).to(DEVICE)
    m.load_state_dict(ck["state_dict"]); m.eval(); k_models.append(m)
k_y=np.linspace(-4.0,4.0,256,dtype=np.float64)
print(f"  Kou   y: [{k_y[0]:.1f},{k_y[-1]:.1f}]  n_y={len(k_y)}")

def infer_h(models, params, y, pk):
    dy=float(y[1]-y[0])
    lam=torch.tensor(np.array([params[k] for k in pk],dtype=np.float32)).unsqueeze(0).to(DEVICE)
    out=np.zeros((len(models),len(y)),dtype=np.float32)
    with torch.no_grad():
        for i,m in enumerate(models):
            r=m(lam,dy); out[i]=(r[0] if isinstance(r,(list,tuple)) else r).cpu().numpy().squeeze()
    return out.astype(np.float64)

def infer_k(models, raw, y):
    dy=float(y[1]-y[0])
    ln=torch.tensor(((raw-KOU_LO)/(KOU_HI-KOU_LO))[None],dtype=torch.float32)
    out=np.zeros((len(models),len(y)),dtype=np.float32)
    with torch.no_grad():
        for i,m in enumerate(models):
            r=m(ln.to(DEVICE),dy); out[i]=(r[0] if isinstance(r,(list,tuple)) else r).cpu().numpy().squeeze()
    return out.astype(np.float64)

pdfs = {}
for event,window,d0,d1 in WINDOWS:

    h = heston_cal.get((event,window))
    if h:
        params = {**h["cl"],"T":1.0}
        pdfs[(event,window,"heston")] = infer_h(h_models, params, h_y, h_pk)

    sd=sigma_diff_map[(event,window)]; lj=lam_J_map[(event,window,best_c)]
    raw=np.array([sd,lj,gp_best["p"],gp_best["eta_1"],gp_best["eta_2"]],dtype=np.float32)

    raw=np.clip(raw,KOU_LO,KOU_HI)
    pdfs[(event,window,"kou")] = infer_k(k_models, raw, k_y)
    print(f"  {event:8} {window:4}: H={pdfs.get((event,window,'heston') ,np.array([])).shape}  "
          f"K={pdfs[(event,window,'kou')].shape}", flush=True)
print()

print("="*70); print("PHASE 9: Thm 3.5 ratio"); print("="*70)

def folded_normal(mu, sigma):
    if sigma < 1e-30: return abs(mu)
    s=mu/sigma
    return float(sigma*math.sqrt(2/math.pi)*math.exp(-0.5*s**2) + mu*math.erf(s/math.sqrt(2)))

EVENTS = ["Luna","FTX","ETF","COVID","China","3AC","SVB","Halving"]
thm_rows=[]

for event in EVENTS:
    for mname in ["heston","kou"]:
        key_pre  = (event,"pre", mname)
        key_post = (event,"post",mname)
        if key_pre not in pdfs or key_post not in pdfs: continue

        l1 = pdfs[key_pre];  l2 = pdfs[key_post]
        y  = h_y if mname=="heston" else k_y
        dy = float(y[1]-y[0])

        l1c,l1t = l1[:N_CAL], l1[N_CAL:]
        l2c,l2t = l2[:N_CAL], l2[N_CAL:]

        eta_c = (l1c-l1c.mean(0)) - (l2c-l2c.mean(0))
        K_cal = (eta_c.T @ eta_c) / N_CAL
        d_hat = l1c.mean(0) - l2c.mean(0)

        if mname=="heston":
            h=heston_cal[(event,"pre")];  pre_p={**h["cl"],"T":1.0}
            h2=heston_cal[(event,"post")]; post_p={**h2["cl"],"T":1.0}
            d_lewis = lewis_pdf(pre_p,y) - lewis_pdf(post_p,y)
            was_cl  = ",".join(k for k,v in h["wc"].items() if v) or "none"
        else:
            sd_pre  = sigma_diff_map[(event,"pre")];   lj_pre  = lam_J_map[(event,"pre",best_c)]
            sd_post = sigma_diff_map[(event,"post")];  lj_post = lam_J_map[(event,"post",best_c)]
            pre_t  = tuple(np.clip([sd_pre, lj_pre,  gp_best["p"],gp_best["eta_1"],gp_best["eta_2"]],KOU_LO,KOU_HI).tolist())
            post_t = tuple(np.clip([sd_post,lj_post, gp_best["p"],gp_best["eta_1"],gp_best["eta_2"]],KOU_LO,KOU_HI).tolist())
            d_lewis = kou_log_density(y,1.0,pre_t) - kou_log_density(y,1.0,post_t)
            was_cl  = "n/a"

        in_tb = (heston_cal[(event,"pre")]["in_box"] and heston_cal[(event,"post")]["in_box"]
                 if mname=="heston" else
                 k_rows[next(i for i,r in enumerate(k_rows) if r["event"]==event and r["window"]=="pre")]["in_training_box"] and
                 k_rows[next(i for i,r in enumerate(k_rows) if r["event"]==event and r["window"]=="post")]["in_training_box"])

        eta_t = (l1t-l1t.mean(0)) - (l2t-l2t.mean(0))
        K_tst = (eta_t.T @ eta_t) / N_TEST

        for pname, pf in PAYOFFS_DEF.items():
            g = pf(y)
            noise2   = float(g @ (K_tst @ g)) * dy**2
            sigma_d  = float(math.sqrt(max(noise2,0.0)))
            mu_d     = float(g @ d_hat) * dy
            theory   = folded_normal(mu_d, sigma_d)
            dpdf_t   = l1t - l2t
            emp      = float(np.abs(dpdf_t @ g * dy).mean())
            ratio    = emp/theory if abs(theory)>1e-20 else float("nan")
            in_range = (0.7<=ratio<=1.3) if not np.isnan(ratio) else False
            flag     = " [WARN]" if (np.isnan(ratio) or abs(ratio)>10) else ""
            print(f"  {event:8} {mname:7} {pname:15}: ratio={ratio:.4f}  "
                  f"emp={emp:.5f}  theory={theory:.5f}{flag}")
            thm_rows.append(dict(event=event,window_pair="pre→post",model=mname,
                payoff=pname,sigma_delta=round(sigma_d,6),mu_delta=round(mu_d,6),
                theory_mr=round(theory,6),empirical_mr=round(emp,6),
                ratio=round(ratio,4) if not np.isnan(ratio) else float("nan"),
                in_range=in_range,in_training_box=in_tb,was_clipped=was_cl))
    print()

df_thm = pd.DataFrame(thm_rows)
n_inrange = int(df_thm["in_range"].sum())
print(f"  In-range (0.7-1.3): {n_inrange}/{len(df_thm)}\n")

if B6_THM35_CSV.exists():
    b6_old = pd.read_csv(B6_THM35_CSV)
    print("  Consistency check vs b6_btc_thm35.csv (existing 3 events):")
    for _,old_r in b6_old.iterrows():
        sc,mo,pn = old_r["scenario"],old_r["model"],old_r["payoff"]
        new_r = df_thm[(df_thm.event==sc)&(df_thm.model==mo)&(df_thm.payoff==pn)]
        if new_r.empty: continue
        r_old=float(old_r["ratio"]); r_new=float(new_r.iloc[0]["ratio"])
        diff=abs(r_new-r_old)/max(abs(r_old),1e-6)
        flag=" [>10%]" if diff>0.10 else ""
        print(f"    {sc:6} {mo:7} {pn:15}: old={r_old:.4f}  new={r_new:.4f}  "
              f"reldiff={100*diff:.1f}%{flag}")
    print()

print("="*70); print("PHASE 10: Figure"); print("="*70)

MARKERS = {"ATM_call":"o","OTM_put_m030":"s","OTM_put_m050":"^"}
COLORS  = {"ATM_call":"#2196F3","OTM_put_m030":"#FF5722","OTM_put_m050":"#4CAF50"}
x_labels= EVENTS; x_pos=range(len(EVENTS))

fig,axes=plt.subplots(1,2,figsize=(14,5))
for ax,mname in zip(axes,["heston","kou"]):
    sub=df_thm[df_thm.model==mname]
    for pname,(mk,col) in {p:(m,c) for p,m,c in zip(MARKERS,MARKERS.values(),COLORS.values())}.items():
        mk=MARKERS[pname]; col=COLORS[pname]
        ys=[]; xs=[]
        for xi,ev in enumerate(EVENTS):
            row=sub[(sub.event==ev)&(sub.payoff==pname)]
            if not row.empty and not np.isnan(row.iloc[0]["ratio"]):
                xs.append(xi); ys.append(float(row.iloc[0]["ratio"]))
        ax.scatter(xs,ys,marker=mk,color=col,s=60,zorder=3,label=pname)
    ax.axhline(1.0,color="black",lw=1.2,zorder=2)
    ax.axhline(0.7,color="gray",lw=0.8,ls="--",zorder=2)
    ax.axhline(1.3,color="gray",lw=0.8,ls="--",zorder=2)
    ax.set_xticks(list(x_pos)); ax.set_xticklabels(x_labels,rotation=30,ha="right",fontsize=9)
    ax.set_ylabel("Thm 3.5 ratio (emp/theory)"); ax.set_title(f"{mname.capitalize()}")
    ax.set_ylim(0.4,2.0); ax.legend(fontsize=8); ax.grid(axis="y",alpha=0.3)
axes[0].set_xlabel("Event"); axes[1].set_xlabel("Event")
fig.suptitle("Multi-Event BTC §5.4 Extension — Thm 3.5 Ratio",fontsize=12)
fig.tight_layout()
fig_path=FIGURES/"fig_multi_event_thm35_ratio.png"
fig.savefig(fig_path,dpi=150,bbox_inches="tight")
plt.close()
print(f"  Fig → {fig_path.relative_to(ROOT)}")

df_h.to_csv(UNIFIED/"multi_event_heston_calibrations.csv",index=False)
df_k.to_csv(UNIFIED/"multi_event_kou_calibrations.csv",index=False)
df_pool.to_csv(UNIFIED/"multi_event_kou_pooled_params.csv",index=False)
df_kval.to_csv(UNIFIED/"multi_event_kou_validation.csv",index=False)
df_ksum.to_csv(UNIFIED/"multi_event_kou_validation_summary.csv",index=False)
df_hval.to_csv(UNIFIED/"multi_event_heston_validation.csv",index=False)
df_thm.to_csv(UNIFIED/"multi_event_thm35.csv",index=False)

print(f"\nCSV → multi_event_heston_calibrations.csv  ({len(df_h)} rows)")
print(f"CSV → multi_event_kou_calibrations.csv     ({len(df_k)} rows)")
print(f"CSV → multi_event_kou_pooled_params.csv    ({len(df_pool)} rows)")
print(f"CSV → multi_event_kou_validation.csv        ({len(df_kval)} rows)")
print(f"CSV → multi_event_kou_validation_summary.csv ({len(df_ksum)} rows)")
print(f"CSV → multi_event_heston_validation.csv    ({len(df_hval)} rows)")
print(f"CSV → multi_event_thm35.csv                ({len(df_thm)} rows)")

elapsed = time.time()-t_start
print()
print("="*70); print("STDOUT SUMMARY"); print("="*70)
print(f"Elapsed: {elapsed:.1f}s")
print(f"\nPHASE 4 Kou c selection: best_c={best_c}  action={action}")
for r in sum_rows:
    print(f"  c={r['c']:.1f}: median={r['median_diff_pct']}%  max={r['max_diff_pct']}%  "
          f"{'PASS' if r['pass'] else 'FAIL'}  {'<- BEST' if r['best_c'] else ''}")
print(f"\nPHASE 3 pooled η (c={best_c}): p={gp_best['p']:.3f}  "
      f"η1={gp_best['eta_1']:.2f}  η2={gp_best['eta_2']:.2f}  "
      f"n_jumps={gp_best['n_jumps_full']}")
print(f"\nPHASE 6 in-range: Heston {h_in}/16  Kou {k_in}/16")
print(f"PHASE 7 Heston vs exp09: median={h_overall:.1f}%  [{'PASS' if h_pass else 'FAIL'}]")
print(f"\nPHASE 9 Thm 3.5: {n_inrange}/{len(df_thm)} in-range (0.7-1.3)")
print("\nPer-event ratios:")
for event in EVENTS:
    for mname in ["heston","kou"]:
        sub=df_thm[(df_thm.event==event)&(df_thm.model==mname)]
        if sub.empty: continue
        ratios=" ".join(f"{r['payoff'].replace('OTM_put_','')[:7]}={r['ratio']:.3f}"
                        for _,r in sub.iterrows())
        print(f"  {event:8} {mname:7}: {ratios}")
print()
print("STAGE 1+2 COMPLETE.")
