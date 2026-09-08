"""
E2 main: branch_hidden {64,128,256,512,1024} x seeds 0-199, protocol (A) verbatim.
3 phases (each resumable): train -> infer -> diagnose.
Protocol (A) = exp07_6 step_train verbatim (loss MSE+0.3KL+0.7KL_tail alpha=6|y|^2,
Adam lr=1e-3, 300 epochs, batch 64, data tno_dataset_v1.npz, rank=192, n_y=300).
Ratio = boundary-suite (b5_thm35_387.thm35_ratio verbatim): mu=g.d_hat(cal)*dy,
sigma=sqrt(g.K_test.g*dy^2) with K_test=build_k_eta(test), empirical=mean|dpdf_test.g*dy|.
Cooling: sleep 60s every 100 models. Writes ONLY under experiments/e2_width/.
Existing code/npz/ckpt read-only.
"""
import sys, time, csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.special import erf
from scipy.stats import kurtosis, skew, anderson
from statsmodels.stats.diagnostic import lilliefors

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from TNO.common.tno_model import DensityDeepONet_HestonLog
from NeuMoR.code.neumor_core import build_k_eta, compute_d_hat  # reuse

DATA_PATH = ROOT / "TNO" / "data" / "Heston" / "tno_dataset_v1.npz"
YGRID_NPZ = ROOT / "NeuMoR" / "output" / "day4" / "heston" / "kernel" / "decomp_pairA.npz"
E2   = ROOT / "NeuMoR" / "experiments" / "e2_width"
CKPT = E2 / "ckpt"; PDFS = E2 / "pdfs"
for d in (E2, CKPT, PDFS): d.mkdir(parents=True, exist_ok=True)

DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")

WIDTHS = [64, 128, 256, 512, 1024]
SEEDS  = list(range(200))
RANK, EPOCHS, BATCH = 192, 300, 64
COOL_EVERY, COOL_SEC = 100, 60
N_CAL, TEST_LO, TEST_HI = 150, 150, 200
TIME_CAP_S = 6 * 3600   # hard stop; resume via skip-existing ckpt
_T_GLOBAL = time.time()

BASE = {"v": 0.10, "kappa": 1.0, "omega": 0.2, "xi": 0.3, "rho": -0.5}
PAIR_V2 = {"A": 0.101, "B": 0.105, "C": 0.11, "D": 0.13}


# ---- protocol (A) verbatim (exp07_6 L56-86) ---------------------------------
class HestonDataset(Dataset):
    def __init__(self):
        data = np.load(DATA_PATH)
        self.lambdas    = data["lambdas"].astype(np.float32)
        self.pdfs       = data["pdfs"].astype(np.float32)
        self.y_grid     = data["y_grid"].astype(np.float32)
        self.param_keys = [str(k) for k in data["param_keys"]]
    def __len__(self): return len(self.lambdas)
    def __getitem__(self, i): return self.lambdas[i], self.pdfs[i]

def make_tail_weights(y_grid, alpha=6.0, p=2.0):
    y = torch.tensor(y_grid, dtype=torch.float32)
    w = 1.0 + alpha * (torch.abs(y) / (torch.abs(y).max() + 1e-12)) ** p
    return w.unsqueeze(0)

def loss_fn(pred, log_pred, target, tw):
    eps = 1e-12; t = target + eps; p_ = pred + eps
    mse = ((p_ - t) ** 2).mean()
    kl  = (t * (torch.log(t) - torch.log(p_))).mean()
    klt = (t * (torch.log(t) - torch.log(p_)) * tw).mean()
    return mse + 0.3 * kl + 0.7 * klt

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available(): torch.mps.manual_seed(seed)
    np.random.seed(seed)
    import random; random.seed(seed)
    torch.use_deterministic_algorithms(False)

# ---- ratio verbatim (b5_thm35_387 L82-100) ----------------------------------
def folded_normal_mean(mu, sigma):
    if sigma < 1e-30: return abs(mu)
    s = mu / sigma
    return float(sigma * np.sqrt(2.0/np.pi) * np.exp(-0.5*s**2) + mu * erf(s/np.sqrt(2.0)))

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
# -----------------------------------------------------------------------------


def payoffs(y):
    ey = np.exp(y)
    return {
        "ATM_call":      np.maximum(ey - 1.00, 0.0),
        "OTM_call_1.10": np.maximum(ey - 1.10, 0.0),
        "OTM_call_1.25": np.maximum(ey - 1.25, 0.0),
        "OTM_put_0.90":  np.maximum(0.90 - ey, 0.0),
    }

def lam_vec(v_val, param_keys):
    d = dict(BASE); d["v"] = v_val
    return np.array([d[k] for k in param_keys], dtype=np.float32)


def _write_timing(timing):
    with open(E2 / "e2_width_timing.csv", "w", newline="") as f:
        wt = csv.writer(f); wt.writerow(["width", "n_trained_this_run", "n_skipped",
                                         "train_wall_s_this_run", "sec_per_model"])
        for w in WIDTHS:
            t = timing[w]; spm = (t["sec"]/t["trained"]) if t["trained"] else float("nan")
            wt.writerow([w, t["trained"], t["skipped"], round(t["sec"], 2), round(spm, 3)])


def phase_train(ds, y_grid):
    dy = float(y_grid[1] - y_grid[0]); n_y = len(y_grid)
    lambda_dim = ds.lambdas.shape[1]; N_ds = len(ds)
    tail_w = make_tail_weights(y_grid).to(DEVICE)
    timing = {w: {"trained": 0, "skipped": 0, "sec": 0.0} for w in WIDTHS}
    counter = 0
    for w in WIDTHS:
        (CKPT / f"w{w}").mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            path = CKPT / f"w{w}" / f"seed_{seed:04d}.pt"
            if path.exists():
                timing[w]["skipped"] += 1; continue
            if time.time() - _T_GLOBAL > TIME_CAP_S:
                print(f"[e2] TIME CAP {TIME_CAP_S}s hit at w={w} seed={seed} "
                      f"(trained so far this run: {counter}). Stopping; resume via re-run.",
                      flush=True)
                _write_timing(timing); sys.exit(2)
            set_seed(seed)
            loader = DataLoader(ds, batch_size=BATCH, shuffle=True)
            model = DensityDeepONet_HestonLog(lambda_dim=lambda_dim, n_y=n_y,
                                              rank=RANK, branch_hidden=w).to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            t0 = time.time(); final_loss = float("nan"); nan_flag = False
            for epoch in range(1, EPOCHS + 1):
                model.train(); tot = 0.0
                for lam, tgt in loader:
                    lam, tgt = lam.to(DEVICE), tgt.to(DEVICE)
                    pred, logp = model(lam, dy)
                    loss = loss_fn(pred, logp, tgt, tail_w)
                    if not torch.isfinite(loss): nan_flag = True; break
                    opt.zero_grad(); loss.backward(); opt.step()
                    tot += loss.item() * lam.size(0)
                if nan_flag: break
                final_loss = tot / N_ds
            el = time.time() - t0
            torch.save({"seed": seed, "config": {"lambda_dim": lambda_dim, "n_y": n_y,
                        "rank": RANK, "branch_hidden": w}, "y_grid": np.asarray(y_grid),
                        "param_keys": ds.param_keys, "state_dict": model.state_dict(),
                        "final_loss": final_loss, "nan": nan_flag}, path)
            timing[w]["trained"] += 1; timing[w]["sec"] += el
            counter += 1
            if counter % 20 == 0:
                print(f"  [train] w={w} seed={seed} loss={final_loss:.3e} {el:.1f}s "
                      f"(w-cum {timing[w]['sec']:.0f}s trained {timing[w]['trained']})", flush=True)
            if counter % COOL_EVERY == 0:
                print(f"  -- cooling {COOL_SEC}s after {counter} trained --", flush=True)
                time.sleep(COOL_SEC)
    _write_timing(timing)
    print("[phase_train] done", {w: timing[w] for w in WIDTHS}, flush=True)


def _load_model(path, lambda_dim, n_y):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    m = DensityDeepONet_HestonLog(lambda_dim=lambda_dim, n_y=n_y,
                                  rank=ck["config"]["rank"],
                                  branch_hidden=ck["config"]["branch_hidden"]).to(DEVICE)
    m.load_state_dict(ck["state_dict"]); m.eval(); return m

def phase_infer(ds, y_grid):
    dy = float(y_grid[1] - y_grid[0]); n_y = len(y_grid); lambda_dim = ds.lambdas.shape[1]
    pk = ds.param_keys
    for w in WIDTHS:
        # build per-seed pdf at v=0.10 (l1) once; v=pair2 per pair
        need = any(not (PDFS / f"w{w}_pair{X}_l{s}.npy").exists()
                   for X in "ABCD" for s in (1, 2))
        if not need:
            print(f"  [infer] w={w} all npy exist, skip", flush=True); continue
        # cache all-seed pdf at each unique v
        vmap = {"0.10": 0.10, **{f"{v:.3f}": v for v in PAIR_V2.values()}}
        pdf_cache = {}  # v_key -> (200, n_y)
        for vk, vv in vmap.items():
            lam = torch.tensor(lam_vec(vv, pk)).unsqueeze(0).to(DEVICE)
            arr = np.zeros((len(SEEDS), n_y), dtype=np.float64)
            for i, seed in enumerate(SEEDS):
                m = _load_model(CKPT / f"w{w}" / f"seed_{seed:04d}.pt", lambda_dim, n_y)
                with torch.no_grad():
                    out = m(lam, dy); pdf = (out[0] if isinstance(out,(tuple,list)) else out)
                arr[i] = pdf.cpu().numpy().squeeze()
            pdf_cache[vk] = arr
        for X in "ABCD":
            np.save(PDFS / f"w{w}_pair{X}_l1.npy", pdf_cache["0.10"])
            np.save(PDFS / f"w{w}_pair{X}_l2.npy", pdf_cache[f"{PAIR_V2[X]:.3f}"])
        print(f"  [infer] w={w} pdfs saved (v_keys={list(vmap)})", flush=True)


def phase_diagnose(y_grid):
    dy = float(y_grid[1] - y_grid[0])
    pf = payoffs(np.asarray(y_grid, dtype=np.float64))
    rows = []
    for w in WIDTHS:
        for X in "ABCD":
            l1 = np.load(PDFS / f"w{w}_pair{X}_l1.npy").astype(np.float64)  # (200,n_y)
            l2 = np.load(PDFS / f"w{w}_pair{X}_l2.npy").astype(np.float64)
            l1c, l1t = l1[:N_CAL], l1[TEST_LO:TEST_HI]
            l2c, l2t = l2[:N_CAL], l2[TEST_LO:TEST_HI]
            K_test = build_k_eta(l1t, l2t)
            d_hat  = compute_d_hat(l1c, l2c)
            dpdf_all = l1 - l2  # (200,n_y)
            for pn, g in pf.items():
                dhat_all = dpdf_all @ g * dy  # (200,) per-seed Delta_hat(g)
                kt = float(kurtosis(dhat_all, fisher=True))
                sk = float(skew(dhat_all))
                try:
                    _, lp = lilliefors(dhat_all, dist="norm"); lil_rej = bool(lp < 0.05)
                except Exception:
                    lil_rej = None
                ad = anderson(dhat_all, dist="norm")
                ad_rej = bool(ad.statistic > ad.critical_values[2])  # 5% level
                ratio, emp, theory, sigma, mu = thm35_ratio(g, d_hat, K_test, l1t, l2t, dy)
                rows.append(dict(width=w, pair=X, payoff=pn,
                                 kurt=round(kt, 4), skew=round(sk, 4),
                                 lilliefors_rej=lil_rej, ad_rej=ad_rej,
                                 ratio=round(ratio, 4) if not np.isnan(ratio) else float("nan"),
                                 sigma_delta=round(sigma, 8), mu_delta=round(mu, 8)))
    cols = ["width","pair","payoff","kurt","skew","lilliefors_rej","ad_rej",
            "ratio","sigma_delta","mu_delta"]
    with open(E2 / "e2_width_diagnostics.csv", "w", newline="") as f:
        wt = csv.DictWriter(f, fieldnames=cols); wt.writeheader(); wt.writerows(rows)
    # summary per width
    srows = []
    for w in WIDTHS:
        sub = [r for r in rows if r["width"] == w]
        ku = np.array([r["kurt"] for r in sub], float)
        ra = np.array([r["ratio"] for r in sub], float); ra = ra[~np.isnan(ra)]
        lil = [r["lilliefors_rej"] for r in sub if r["lilliefors_rej"] is not None]
        adr = [r["ad_rej"] for r in sub]
        srows.append(dict(width=w, n_cells=len(sub),
            kurt_median=round(float(np.median(ku)),4), kurt_min=round(float(ku.min()),4),
            kurt_max=round(float(ku.max()),4),
            rej_rate_lilliefors=round(float(np.mean(lil)),4) if lil else float("nan"),
            rej_rate_ad=round(float(np.mean(adr)),4),
            ratio_min=round(float(ra.min()),4) if len(ra) else float("nan"),
            ratio_max=round(float(ra.max()),4) if len(ra) else float("nan"),
            ratio_median=round(float(np.median(ra)),4) if len(ra) else float("nan")))
    scols = ["width","n_cells","kurt_median","kurt_min","kurt_max",
             "rej_rate_lilliefors","rej_rate_ad","ratio_min","ratio_max","ratio_median"]
    with open(E2 / "e2_width_summary.csv", "w", newline="") as f:
        wt = csv.DictWriter(f, fieldnames=scols); wt.writeheader(); wt.writerows(srows)
    print("[phase_diagnose] done", len(rows), "cells", flush=True)


def main():
    t0 = time.time()
    ds = HestonDataset()
    y_grid = np.load(YGRID_NPZ)["y_grid"].astype(np.float64)  # decomp_pairA grid == dataset
    assert len(y_grid) == 300, len(y_grid)
    print(f"[e2] device={DEVICE} widths={WIDTHS} seeds=0-199 n_y={len(y_grid)} "
          f"N_samples={len(ds)} param_keys={ds.param_keys}", flush=True)
    phase_train(ds, y_grid)
    phase_infer(ds, y_grid)
    phase_diagnose(y_grid)
    print(f"[e2] ALL DONE total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
