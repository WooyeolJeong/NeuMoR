
import sys
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.special import erf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from NeuMoR.code.neumor_core import (
    build_k_eta, compute_d_hat, split_cal_test, effective_rank,
)
from TNO.common.tno_model import DensityDeepONet_HestonLog

UNIFIED = ROOT / "NeuMoR/output/unified"
UNIFIED.mkdir(parents=True, exist_ok=True)

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available()          else
    "cpu"
)

SMOOTH_PAYOFFS = ["ATM", "OTM_K110", "OTM_K125"]

def make_heston_payoff(name: str, y: np.ndarray) -> np.ndarray:
    if name == "ATM":
        return np.maximum(np.exp(y) - 1.0, 0.0)
    elif name == "OTM_K110":
        return np.maximum(np.exp(y) - 1.10, 0.0)
    elif name == "OTM_K125":
        return np.maximum(np.exp(y) - 1.25, 0.0)
    raise ValueError(name)

def make_kou_payoff(name: str, y: np.ndarray) -> np.ndarray:
    if name == "ATM":
        return np.maximum(np.exp(y) - 1.0, 0.0)
    elif name == "OTM_K110":
        return np.maximum(np.exp(y) - 1.10, 0.0)
    elif name == "OTM_K125":
        return np.maximum(np.exp(y) - 1.25, 0.0)
    raise ValueError(name)

def folded_normal_mean(mu: float, sigma: float) -> float:
    if sigma < 1e-30:
        return abs(mu)
    s = mu / sigma
    return float(sigma * np.sqrt(2.0/np.pi) * np.exp(-0.5*s**2)
                 + mu * erf(s / np.sqrt(2.0)))

def thm35_ratio(g, d_hat, K_test, l1_test, l2_test, dy):
    noise2 = float(g @ (K_test @ g)) * dy**2
    sigma  = float(np.sqrt(max(noise2, 0.0)))
    mu     = float(g @ d_hat) * dy
    theory = folded_normal_mean(mu, sigma)
    dpdf_test = l1_test - l2_test
    emp_vals  = np.abs(dpdf_test @ g * dy)
    empirical = float(emp_vals.mean())
    ratio = empirical / theory if abs(theory) > 1e-20 else float("nan")
    return ratio, empirical, theory, sigma, mu

def load_heston_protocol(proto: str, n_seeds: int = None):
    SAVE_A  = ROOT / "NeuMoR/save/exp01"
    SAVE_A2 = ROOT / "NeuMoR/save/exp07_5"
    SAVE_B  = ROOT / "NeuMoR/save/exp03"

    PROTO_SPECS = {
        "A":  (None,                      192, 512, 200),
        "B1": (SAVE_B/"protocol_B1",       96, 256,  30),
        "B2": (SAVE_B/"protocol_B2",      192, 512,  30),
        "B3": (SAVE_B/"protocol_B3",      192, 512,  30),
    }
    save_dir, rank, bh, default_n = PROTO_SPECS[proto]
    if n_seeds is None:
        n_seeds = default_n

    models = []
    y, pk = None, None
    for s in range(n_seeds):
        if proto == "A":
            p = (SAVE_A / f"seed_{s:02d}.pt") if s < 50 else (SAVE_A2 / f"seed_{s:03d}.pt")
        else:
            p = save_dir / f"seed_{s:02d}.pt"
        ck = torch.load(p, map_location=DEVICE, weights_only=False)
        cfg = ck["config"]
        m = DensityDeepONet_HestonLog(
            lambda_dim=cfg["lambda_dim"], n_y=cfg["n_y"],
            rank=cfg["rank"], branch_hidden=cfg["branch_hidden"]
        ).to(DEVICE)
        m.load_state_dict(ck["state_dict"]); m.eval(); models.append(m)
        if y is None:
            y  = ck["y_grid"].astype(np.float64)
            pk = list(ck["param_keys"])
    return models, y, pk

def load_kou_models(n_seeds: int = 200):
    SA = ROOT / "NeuMoR/save/kou/stageA"
    SB = ROOT / "NeuMoR/save/kou/stageB"
    models = []
    for s in range(n_seeds):
        p = (SA / f"seed_{s:04d}.pt") if s < 30 else (SB / f"seed_{s:04d}.pt")
        ck = torch.load(p, map_location=DEVICE, weights_only=False)
        m = DensityDeepONet_HestonLog(
            lambda_dim=5, n_y=256, rank=192, branch_hidden=512
        ).to(DEVICE)
        m.load_state_dict(ck["state_dict"]); m.eval(); models.append(m)
    return models, np.linspace(-4.0, 4.0, 256, dtype=np.float64)

_INFER_CACHE: dict = {}

def infer_heston(models, params: dict, y: np.ndarray, pk: list, cache_key: str = None) -> np.ndarray:
    if cache_key and cache_key in _INFER_CACHE:
        return _INFER_CACHE[cache_key]
    dy = float(y[1] - y[0])
    lam = torch.tensor(
        np.array([params[k] for k in pk], dtype=np.float32)
    ).unsqueeze(0).to(DEVICE)
    out = np.zeros((len(models), len(y)), dtype=np.float32)
    with torch.no_grad():
        for i, m in enumerate(models):
            r = m(lam, dy)
            out[i] = (r[0] if isinstance(r, (list, tuple)) else r).cpu().numpy().squeeze()
    result = out.astype(np.float64)
    if cache_key:
        _INFER_CACHE[cache_key] = result
    return result

KOU_LO = np.array([0.10,  0.5, 0.30,  3.0,  3.0], dtype=np.float32)
KOU_HI = np.array([0.80, 50.0, 0.70, 50.0, 50.0], dtype=np.float32)

def infer_kou(models, raw: np.ndarray, y: np.ndarray, cache_key: str = None) -> np.ndarray:
    if cache_key and cache_key in _INFER_CACHE:
        return _INFER_CACHE[cache_key]
    dy = float(y[1] - y[0])
    ln = torch.tensor(
        ((raw - KOU_LO) / (KOU_HI - KOU_LO))[None], dtype=torch.float32
    )
    out = np.zeros((len(models), len(y)), dtype=np.float32)
    with torch.no_grad():
        for i, m in enumerate(models):
            r = m(ln.to(DEVICE), dy)
            out[i] = (r[0] if isinstance(r, (list, tuple)) else r).cpu().numpy().squeeze()
    result = out.astype(np.float64)
    if cache_key:
        _INFER_CACHE[cache_key] = result
    return result

def measure_block(
    l1_all, l2_all, y, payoff_fn,
    block: str, config_id: str,
    n_cal: int, n_test_lo: int, n_test_hi: int,
    extra: dict = None,
):
    dy = float(y[1] - y[0])
    l1_cal = l1_all[:n_cal]
    l2_cal = l2_all[:n_cal]
    l1_tst = l1_all[n_test_lo:n_test_hi]
    l2_tst = l2_all[n_test_lo:n_test_hi]

    K_cal  = build_k_eta(l1_cal, l2_cal)
    K_test = build_k_eta(l1_tst, l2_tst)
    d_hat  = compute_d_hat(l1_cal, l2_cal)
    reff   = effective_rank(K_cal)

    rows = []
    for pname in SMOOTH_PAYOFFS:
        g = payoff_fn(pname, y)
        ratio, emp, theory, sigma, mu = thm35_ratio(g, d_hat, K_test, l1_tst, l2_tst, dy)
        in_range = (0.7 <= ratio <= 1.3) if not np.isnan(ratio) else False
        row = dict(
            block=block, config_id=config_id, payoff=pname,
            n_cal=n_cal, n_test=(n_test_hi - n_test_lo),
            r_eff=round(reff, 3),
            sigma_delta=round(sigma, 6),
            mu_delta=round(mu, 8),
            theory_mr=round(theory, 8),
            empirical_mr=round(emp, 8),
            ratio=round(ratio, 4) if not np.isnan(ratio) else float("nan"),
            in_range=in_range,
        )
        if extra:
            row.update(extra)
        rows.append(row)
    return rows

def main():
    t0 = time.time()
    print("=" * 70)
    print("B.5: Thm 3.5 Validation — 387 configs, K_eta unified formulation")
    print("=" * 70)

    all_rows = []

    BASE_H = {"v": 0.10, "kappa": 1.0, "omega": 0.2, "xi": 0.3, "rho": -0.5, "T": 1.0}

    print("\nLoading Protocol A (200 seeds)...")
    hA_models, h_y, h_pk = load_heston_protocol("A", n_seeds=200)
    h_dy = float(h_y[1] - h_y[0])
    N_H_CAL, N_H_TEST_LO, N_H_TEST_HI = 150, 150, 200

    def infer_A(params, ckey):
        return infer_heston(hA_models, params, h_y, h_pk, cache_key=f"A_{ckey}")

    print("\n[Exp01] Pairs A-E × 3 smooth payoffs = 15")
    EXP01_PAIRS = {
        "A": (BASE_H, {**BASE_H, "v": 0.101}),
        "B": (BASE_H, {**BASE_H, "v": 0.105}),
        "C": (BASE_H, {**BASE_H, "v": 0.110}),
        "D": (BASE_H, {**BASE_H, "v": 0.130}),
        "E": (BASE_H, {**BASE_H, "v": 0.200}),
    }
    for pid, (p1, p2) in EXP01_PAIRS.items():
        l1 = infer_A(p1, f"exp01_p{pid}_v{p1['v']}")
        l2 = infer_A(p2, f"exp01_p{pid}_v{p2['v']}")
        rows = measure_block(l1, l2, h_y, make_heston_payoff,
                             "Exp01", f"Exp01-{pid}",
                             N_H_CAL, N_H_TEST_LO, N_H_TEST_HI,
                             {"exp": "exp01", "pair": pid})
        all_rows.extend(rows)
    print(f"  → {len([r for r in all_rows if r['block']=='Exp01'])} rows")

    print("\n[Exp02] 13 pairs × 3 smooth payoffs = 39")
    EXP02_PAIRS = {
        **EXP01_PAIRS,
        "F1": (BASE_H, {**BASE_H, "v": 0.1001}),
        "F2": (BASE_H, {**BASE_H, "v": 0.1002}),
        "F3": (BASE_H, {**BASE_H, "v": 0.1003}),
        "F4": (BASE_H, {**BASE_H, "v": 0.1005}),
        "F5": (BASE_H, {**BASE_H, "v": 0.1007}),
        "G1": (BASE_H, {**BASE_H, "kappa": 1.005}),
        "G2": (BASE_H, {**BASE_H, "rho": -0.495}),
        "G3": (BASE_H, {**BASE_H, "omega": 0.205}),
    }
    for pid, (p1, p2) in EXP02_PAIRS.items():
        l1 = infer_A(p1, f"exp02_p{pid}_1")
        l2 = infer_A(p2, f"exp02_p{pid}_2")
        rows = measure_block(l1, l2, h_y, make_heston_payoff,
                             "Exp02", f"Exp02-{pid}",
                             N_H_CAL, N_H_TEST_LO, N_H_TEST_HI,
                             {"exp": "exp02", "pair": pid})
        all_rows.extend(rows)
    print(f"  → {len([r for r in all_rows if r['block']=='Exp02'])} rows")

    print("\nLoading Protocol B1/B2/B3 (30 seeds each)...")
    B_models = {}
    B_y_dict = {}
    B_pk_dict = {}
    for bname in ["B1", "B2", "B3"]:
        B_models[bname], B_y_dict[bname], B_pk_dict[bname] = load_heston_protocol(bname, n_seeds=30)

    EXP03_PAIRS_AE = {
        "A": (BASE_H, {**BASE_H, "v": 0.101}),
        "B": (BASE_H, {**BASE_H, "v": 0.105}),
        "C": (BASE_H, {**BASE_H, "v": 0.110}),
        "D": (BASE_H, {**BASE_H, "v": 0.130}),
        "E": (BASE_H, {**BASE_H, "v": 0.200}),
    }

    PROTO_PAIRS = {
        "AA":   ("A",  "A",  200, 150, 150, 200),
        "AB1":  ("A",  "B1",  30,  24,  24,  30),
        "AB2":  ("A",  "B2",  30,  24,  24,  30),
        "AB3":  ("A",  "B3",  30,  24,  24,  30),
        "B1B1": ("B1", "B1",  30,  24,  24,  30),
        "B2B2": ("B2", "B2",  30,  24,  24,  30),
    }

    print("\n[Exp03] 6 protocol pairs × 5 pairs × 3 payoffs = 90")
    for pp_name, (p1_proto, p2_proto, n_seeds, n_cal, tlo, thi) in PROTO_PAIRS.items():
        for pid, (params1, params2) in EXP03_PAIRS_AE.items():

            if p1_proto == "A":
                l1_all = infer_A(params1, f"A_{params1['v']}")[:n_seeds]
            else:
                l1_all = infer_heston(B_models[p1_proto], params1, B_y_dict[p1_proto],
                                      B_pk_dict[p1_proto],
                                      cache_key=f"{p1_proto}_{params1}")
            if p2_proto == "A":
                l2_all = infer_A(params2, f"A_{params2['v']}")[:n_seeds]
            else:
                l2_all = infer_heston(B_models[p2_proto], params2, B_y_dict[p2_proto],
                                      B_pk_dict[p2_proto],
                                      cache_key=f"{p2_proto}_{params2}")
            y_use = B_y_dict[p2_proto] if p2_proto != "A" else h_y
            rows = measure_block(l1_all, l2_all, y_use, make_heston_payoff,
                                 "Exp03", f"Exp03-{pp_name}-{pid}",
                                 n_cal, tlo, thi,
                                 {"exp": "exp03", "proto_pair": pp_name, "pair": pid})
            all_rows.extend(rows)
    print(f"  → {len([r for r in all_rows if r['block']=='Exp03'])} rows")

    print("\n[Exp05-axis] 5 axes × 5 pairs × 3 payoffs = 75")
    AXIS_PAIRS_DEF = {
        "v":     [0.001, 0.005, 0.010, 0.030, 0.100],
        "kappa": [0.01,  0.05,  0.10,  0.30,  1.00],
        "omega": [0.002, 0.010, 0.020, 0.060, 0.200],
        "xi":    [0.003, 0.015, 0.030, 0.090, 0.300],
        "rho":   [0.005, 0.025, 0.050, 0.150, 0.500],
    }
    axis_pair_ids = ["Av", "Bv", "Cv", "Dv", "Ev"]
    for axis, steps in AXIS_PAIRS_DEF.items():
        for i, step in enumerate(steps):
            pair_id = f"{axis}{i+1}"
            p1 = BASE_H.copy()
            p2 = BASE_H.copy(); p2[axis] = p2[axis] + step

            HESTON_LO = {"v": 0.051, "kappa": 0.502, "omega": 0.100, "xi": 0.102, "rho": -0.800}
            HESTON_HI = {"v": 0.300, "kappa": 2.000, "omega": 0.399, "xi": 0.599, "rho":  0.598}
            for k in ["v", "kappa", "omega", "xi", "rho"]:
                p2[k] = float(np.clip(p2[k], HESTON_LO[k], HESTON_HI[k]))
            l1 = infer_A(p1, f"A_base")
            l2 = infer_A(p2, f"A_{axis}{step}")
            rows = measure_block(l1, l2, h_y, make_heston_payoff,
                                 "Exp05_axis", f"Exp05ax-{pair_id}",
                                 N_H_CAL, N_H_TEST_LO, N_H_TEST_HI,
                                 {"exp": "exp05_axis", "axis": axis, "step": step})
            all_rows.extend(rows)
    print(f"  → {len([r for r in all_rows if r['block']=='Exp05_axis'])} rows")

    print("\n[Exp05-diag] 6 diagonals × 5 scales × 3 payoffs = 90")
    DIAG_BASE_STEPS = {
        "vkappa":  {"v": 0.001, "kappa": 0.01},
        "vomega":  {"v": 0.001, "omega": 0.002},
        "vxi":     {"v": 0.001, "xi": 0.003},
        "kappaxi": {"kappa": 0.01, "xi": 0.003},
        "vrho":    {"v": 0.001, "rho": 0.005},
        "all5":    {"v": 0.001, "kappa": 0.01, "omega": 0.002, "xi": 0.003, "rho": 0.005},
    }
    DIAG_SCALES = [1, 2, 4, 8, 16]
    HESTON_LO = {"v": 0.051, "kappa": 0.502, "omega": 0.100, "xi": 0.102, "rho": -0.800}
    HESTON_HI = {"v": 0.300, "kappa": 2.000, "omega": 0.399, "xi": 0.599, "rho":  0.598}
    for dname, base_steps in DIAG_BASE_STEPS.items():
        for scale in DIAG_SCALES:
            p1 = BASE_H.copy()
            p2 = BASE_H.copy()
            for ax, step in base_steps.items():
                p2[ax] = float(np.clip(p2[ax] + step * scale, HESTON_LO[ax], HESTON_HI[ax]))
            pair_id = f"{dname}_s{scale}"
            l1 = infer_A(p1, "A_base")
            l2 = infer_A(p2, f"A_diag_{pair_id}")
            rows = measure_block(l1, l2, h_y, make_heston_payoff,
                                 "Exp05_diag", f"Exp05dg-{pair_id}",
                                 N_H_CAL, N_H_TEST_LO, N_H_TEST_HI,
                                 {"exp": "exp05_diag", "diag": dname, "scale": scale})
            all_rows.extend(rows)
    print(f"  → {len([r for r in all_rows if r['block']=='Exp05_diag'])} rows")

    print("\n[Exp06] 8 edge pairs × 3 smooth payoffs = 24")
    EXP06_PAIRS = {
        "G1": (BASE_H, {**BASE_H, "kappa": 1.005}),
        "G2": (BASE_H, {**BASE_H, "rho": -0.495}),
        "G3": (BASE_H, {**BASE_H, "omega": 0.205}),
        "H1": (BASE_H, {**BASE_H, "kappa": 1.002}),
        "H2": (BASE_H, {**BASE_H, "kappa": 1.020}),
        "H3": (BASE_H, {**BASE_H, "xi": 0.303}),
        "H4": (BASE_H, {**BASE_H, "rho": -0.498}),
        "H5": (BASE_H, {**BASE_H, "omega": 0.201}),
    }
    for pid, (p1, p2) in EXP06_PAIRS.items():
        l1 = infer_A(p1, "A_base")
        l2 = infer_A(p2, f"A_edge_{pid}")
        rows = measure_block(l1, l2, h_y, make_heston_payoff,
                             "Exp06", f"Exp06-{pid}",
                             N_H_CAL, N_H_TEST_LO, N_H_TEST_HI,
                             {"exp": "exp06", "pair": pid})
        all_rows.extend(rows)
    print(f"  → {len([r for r in all_rows if r['block']=='Exp06'])} rows")

    print("\n[Exp09] 4 BTC windows × 3 smooth payoffs = 12")
    h_exp09 = pd.read_csv(ROOT / "NeuMoR/output/exp09/calibrations.csv")
    EXP09_PARAMS = {}
    for _, row in h_exp09.iterrows():
        wname = str(row["window"])
        EXP09_PARAMS[wname] = {
            "v":     float(row["clipped_v"]),
            "kappa": float(row["clipped_kappa"]),
            "omega": float(row["clipped_omega"]),
            "xi":    float(row["clipped_xi"]),
            "rho":   float(row["clipped_rho"]),
            "T":     1.0,
        }
    for wname, p1 in EXP09_PARAMS.items():
        v2 = p1["v"] + 0.01
        if v2 > 0.300 or abs(v2 - p1["v"]) < 1e-9:
            v2 = p1["v"] - 0.01
        p2 = {**p1, "v": float(np.clip(v2, 0.051, 0.300))}
        l1 = infer_A(p1, f"exp09_{wname}_p1")
        l2 = infer_A(p2, f"exp09_{wname}_p2")
        rows = measure_block(l1, l2, h_y, make_heston_payoff,
                             "Exp09", f"Exp09-{wname}",
                             N_H_CAL, N_H_TEST_LO, N_H_TEST_HI,
                             {"exp": "exp09", "window": wname, "clipped": True})
        all_rows.extend(rows)
    print(f"  → {len([r for r in all_rows if r['block']=='Exp09'])} rows")

    KOU_FIXED = np.array([0.30, 5.0, 0.5, 10.0, 10.0], dtype=np.float32)
    KOU_SIGMA_BASE = 0.30
    KOU_PAIRS_DEF = {
        "A": 0.001, "B": 0.005, "C": 0.010, "D": 0.030,
    }

    print("\nLoading Kou models (N=30 for stageA regime)...")
    k30_models, k30_y = load_kou_models(n_seeds=30)
    KOU_N30_CAL, KOU_N30_TEST_LO, KOU_N30_TEST_HI = 24, 24, 30

    print("\n[KouAB_stageA] Kou pairs A-D × 3 payoffs, N=30 seeds = 12")
    for pid, dsig in KOU_PAIRS_DEF.items():
        p1 = KOU_FIXED.copy()
        p2 = KOU_FIXED.copy(); p2[0] = KOU_SIGMA_BASE + dsig
        l1 = infer_kou(k30_models, p1, k30_y, cache_key=f"kou30_base")
        l2 = infer_kou(k30_models, p2, k30_y, cache_key=f"kou30_{dsig}")
        rows = measure_block(l1, l2, k30_y, make_kou_payoff,
                             "KouAB_stageA", f"KouA-{pid}",
                             KOU_N30_CAL, KOU_N30_TEST_LO, KOU_N30_TEST_HI,
                             {"exp": "kou_stageA", "pair": pid, "n_seeds": 30})
        all_rows.extend(rows)
    print(f"  → {len([r for r in all_rows if r['block']=='KouAB_stageA'])} rows")

    print("\nLoading Kou models (N=200 for stageB regime)...")
    k200_models, k200_y = load_kou_models(n_seeds=200)
    KOU_N200_CAL, KOU_N200_TEST_LO, KOU_N200_TEST_HI = 150, 150, 200

    print("\n[KouAB_stageB] Kou pairs A-D × 3 payoffs, N=200 seeds = 12")
    for pid, dsig in KOU_PAIRS_DEF.items():
        p1 = KOU_FIXED.copy()
        p2 = KOU_FIXED.copy(); p2[0] = KOU_SIGMA_BASE + dsig
        l1 = infer_kou(k200_models, p1, k200_y, cache_key=f"kou200_base")
        l2 = infer_kou(k200_models, p2, k200_y, cache_key=f"kou200_{dsig}")
        rows = measure_block(l1, l2, k200_y, make_kou_payoff,
                             "KouAB_stageB", f"KouB-{pid}",
                             KOU_N200_CAL, KOU_N200_TEST_LO, KOU_N200_TEST_HI,
                             {"exp": "kou_stageB", "pair": pid, "n_seeds": 200})
        all_rows.extend(rows)
    print(f"  → {len([r for r in all_rows if r['block']=='KouAB_stageB'])} rows")

    print("\n[KouBTC] 6 BTC scenarios × 3 payoffs, N=200 seeds = 18")
    KOU_BTC_PARAMS = {
        "Luna_pre":  {"sigma": 0.587, "lam_J": 28.7, "p": 0.43, "eta1":  9.3, "eta2": 13.9},
        "Luna_post": {"sigma": 0.740, "lam_J": 20.7, "p": 0.33, "eta1": 12.3, "eta2":  7.4},
        "FTX_pre":   {"sigma": 0.405, "lam_J": 27.2, "p": 0.60, "eta1": 15.3, "eta2": 12.8},
        "FTX_post":  {"sigma": 0.553, "lam_J": 13.8, "p": 0.50, "eta1": 10.0, "eta2":  6.6},
        "ETF_pre":   {"sigma": 0.438, "lam_J": 36.5, "p": 0.57, "eta1": 17.6, "eta2": 19.5},
        "ETF_post":  {"sigma": 0.599, "lam_J": 27.0, "p": 0.50, "eta1": 11.6, "eta2": 12.7},
    }
    DELTA_SIGMA_BTC = 0.01
    for scn, pd_dict in KOU_BTC_PARAMS.items():
        p1 = np.clip(
            np.array([pd_dict["sigma"], pd_dict["lam_J"], pd_dict["p"],
                      pd_dict["eta1"], pd_dict["eta2"]], dtype=np.float32),
            KOU_LO, KOU_HI)
        p2 = p1.copy()
        p2[0] = float(np.clip(p2[0] + DELTA_SIGMA_BTC, KOU_LO[0], KOU_HI[0]))
        l1 = infer_kou(k200_models, p1, k200_y, cache_key=f"koubtc_{scn}_p1")
        l2 = infer_kou(k200_models, p2, k200_y, cache_key=f"koubtc_{scn}_p2")
        rows = measure_block(l1, l2, k200_y, make_kou_payoff,
                             "KouBTC", f"KouBTC-{scn}",
                             KOU_N200_CAL, KOU_N200_TEST_LO, KOU_N200_TEST_HI,
                             {"exp": "kou_btc", "scenario": scn})
        all_rows.extend(rows)
    print(f"  → {len([r for r in all_rows if r['block']=='KouBTC'])} rows")

    df = pd.DataFrame(all_rows)
    csv_all = UNIFIED / "b5_thm35_all.csv"
    df.to_csv(csv_all, index=False)

    summary_rows = []
    blocks_order = ["Exp01","Exp02","Exp03","Exp05_axis","Exp05_diag",
                    "Exp06","Exp09","KouAB_stageA","KouAB_stageB","KouBTC"]
    for blk in blocks_order:
        sub = df[df["block"] == blk]
        if len(sub) == 0:
            continue
        ratios = sub["ratio"].dropna().values
        n_pass = int(sub["in_range"].sum())
        summary_rows.append(dict(
            block=blk,
            n_configs=len(sub),
            n_pass=n_pass,
            pass_pct=round(100.0 * n_pass / len(sub), 1),
            ratio_min=round(float(ratios.min()), 4) if len(ratios) > 0 else float("nan"),
            ratio_max=round(float(ratios.max()), 4) if len(ratios) > 0 else float("nan"),
            ratio_mean=round(float(ratios.mean()), 4) if len(ratios) > 0 else float("nan"),
        ))

    df_sum = pd.DataFrame(summary_rows)
    csv_sum = UNIFIED / "b5_thm35_summary.csv"
    df_sum.to_csv(csv_sum, index=False)

    total_configs  = len(df)
    total_pass     = int(df["in_range"].sum())
    all_ratios     = df["ratio"].dropna().values

    print()
    print("=" * 80)
    print(f"B.5 SUMMARY — Thm 3.5 ratio [0.7, 1.3] validation")
    print("=" * 80)
    print(f"{'Block':18} | {'N':5} | {'Pass':5} | {'Pass%':7} | {'r_min':7} | {'r_max':7} | {'r_mean':7}")
    print("-" * 80)
    for r in summary_rows:
        print(f"  {r['block']:16} | {r['n_configs']:5} | {r['n_pass']:5} | "
              f"{r['pass_pct']:6.1f}% | {r['ratio_min']:7.4f} | {r['ratio_max']:7.4f} | {r['ratio_mean']:7.4f}")
    print("-" * 80)
    print(f"  {'TOTAL':16} | {total_configs:5} | {total_pass:5} | "
          f"{100.0*total_pass/total_configs:6.1f}% | "
          f"{float(all_ratios.min()):7.4f} | {float(all_ratios.max()):7.4f} | "
          f"{float(all_ratios.mean()):7.4f}")

    print()
    print("=" * 80)
    print("CROSS-CHECK: B.5 BTC configs vs B.6 (b6_btc_thm35.csv)")
    print("=" * 80)
    print("  Mapping attempt:")
    print("    B.5 Exp09:  λ1=clipped window params, λ2=λ1+(v+=0.01)  [exp09/calibrations.csv]")
    print("    B.6 Heston: λ1=M1 pre-shock, λ2=M2 post-shock          [exp12/calibrations.csv]")
    print("    B.5 KouBTC: λ1=single BTC calib, λ2=λ1+(σ+=0.01)")
    print("    B.6 Kou:    λ1=pre, λ2=post (event-level pair)")
    print("  → Mapping FAILED: different (λ1,λ2) definitions. Direct ratio comparison not valid.")
    print("  → Fallback: PASS/FAIL consistency check (both must pass [0.7,1.3]).")
    print()
    b6_path = UNIFIED / "b6_btc_thm35.csv"
    if b6_path.exists():
        df_b6 = pd.read_csv(b6_path)
        b5_btc = df[df["block"].isin(["Exp09","KouBTC"])]
        b5_btc_atm = b5_btc[b5_btc["payoff"] == "ATM"]
        b5_btc_pass = int(b5_btc["in_range"].sum())
        b5_btc_n    = len(b5_btc)
        b6_atm = df_b6[df_b6["payoff"] == "ATM_call"]
        b6_pass = int(df_b6["in_range"].sum())
        b6_n    = len(df_b6)

        print(f"  {'Subset':25} | {'N':5} | {'Pass':5} | {'ratio_min':10} | {'ratio_max':10}")
        print(f"  " + "-" * 60)
        r5 = b5_btc["ratio"].dropna()
        r6 = df_b6["ratio"].dropna()
        print(f"  {'B.5 BTC (Exp09+KouBTC)':25} | {b5_btc_n:5} | {b5_btc_pass:5} | "
              f"{float(r5.min()):10.4f} | {float(r5.max()):10.4f}")
        print(f"  {'B.6 BTC':25} | {b6_n:5} | {b6_pass:5} | "
              f"{float(r6.min()):10.4f} | {float(r6.max()):10.4f}")
        print()
        if b5_btc_pass == b5_btc_n and b6_pass == b6_n:
            print("  SANITY: PASS — both B.5 and B.6 BTC subsets fully pass [0.7,1.3].")
            print("          K_eta computation, split, NN inference self-consistent across experiments.")
        else:
            failed_b5 = b5_btc[~b5_btc["in_range"]]
            failed_b6 = df_b6[~df_b6["in_range"]]
            print("  SANITY: FAIL — investigate.")
            if len(failed_b5) > 0:
                print(f"    B.5 failures: {failed_b5[['block','config_id','payoff','ratio']].to_string(index=False)}")
            if len(failed_b6) > 0:
                print(f"    B.6 failures: {failed_b6[['scenario','model','payoff','ratio']].to_string(index=False)}")
    else:
        print("  B.6 file not found. Run b6_btc_real_data.py first.")

    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")
    print(f"CSV → {csv_all.relative_to(ROOT)}")
    print(f"CSV → {csv_sum.relative_to(ROOT)}")

    print()
    ratio_ok = float(all_ratios.min()) >= 0.7 and float(all_ratios.max()) <= 1.3
    all_pass = total_pass == total_configs
    print("VERDICT:", end="  ")
    if all_pass:
        print(f"ALL {total_configs} PASS  ratio in [{float(all_ratios.min()):.4f}, {float(all_ratios.max()):.4f}]")
    elif ratio_ok:
        print(f"{total_pass}/{total_configs} PASS  ratio in [0.7,1.3] but {total_configs-total_pass} flags")
    else:
        print(f"FAIL — {total_configs-total_pass} configs outside [0.7,1.3]. Check b5_thm35_all.csv")

if __name__ == "__main__":
    main()
