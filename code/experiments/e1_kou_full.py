"""
E1-Kou Phase 2a: FNO1d seeds 0-29 (N=30, Kou protocol verbatim) + 16-cell diagnostics
+ DeepONet compare (N=30 subset & N=200 full). ratio = b5_thm35_387.thm35_ratio verbatim
(N=30 split cal 0-23/test 24-29; N=200 split cal 0-149/test 150-199). 4 payoffs (task).
Reuses e2_width_run (thm35_ratio, build_k_eta, compute_d_hat, payoffs) + fno_model.
Kou pair lambda: KOU_FIXED + sigma-perturb, normalized (raw-LO)/(HI-LO) [b5 verbatim].
PAR4-MPS. Writes ONLY under experiments/e1_fno_kou/. Existing code/ckpt/npz read-only.
Worker: python e1_kou_full.py worker <sid> <ns>   Driver: python e1_kou_full.py
"""
import sys, time, csv, shutil, random, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import kurtosis, skew, anderson
from statsmodels.stats.diagnostic import lilliefors

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "NeuMoR" / "experiments" / "e1_fno"))
sys.path.insert(0, str(ROOT / "NeuMoR" / "experiments" / "e2_width"))
from fno_model import FNO1d
import e2_width_run as e2   # thm35_ratio, build_k_eta, compute_d_hat, payoffs

DATA = ROOT / "NeuMoR" / "output" / "kou" / "dataset" / "kou_train_10k.npz"
YGRID_NPZ = ROOT / "NeuMoR" / "output" / "day4" / "kou" / "kernel" / "decomp_pairA.npz"
DEEPO_NPY = ROOT / "NeuMoR" / "output" / "day5" / "_tmp"
PILOT = HERE / "ckpt" / "pilot"; CKPTF = HERE / "ckpt" / "full"; PDFS = HERE / "pdfs"
for d in (CKPTF, PDFS): d.mkdir(parents=True, exist_ok=True)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_Y, N_EPOCHS, BATCH, LR = 256, 500, 128, 1e-3     # Kou protocol verbatim
MODES, WIDTH, LAYERS = 32, 64, 4
SEEDS = list(range(30)); N_WORKERS = 4
# b5 verbatim
KOU_FIXED = np.array([0.30, 5.0, 0.5, 10.0, 10.0], dtype=np.float32)
KOU_SIGMA_BASE = 0.30
KOU_PAIRS_DEF = {"A": 0.001, "B": 0.005, "C": 0.010, "D": 0.030}
KOU_LO = np.array([0.10, 0.5, 0.30, 3.0, 3.0], dtype=np.float32)
KOU_HI = np.array([0.80, 50.0, 0.70, 50.0, 50.0], dtype=np.float32)
N30 = (24, 24, 30); N200 = (150, 150, 200)


class KouDataset(Dataset):
    def __init__(self, lambdas, pdfs):
        self.lambdas = torch.tensor(lambdas, dtype=torch.float32)
        self.pdfs = torch.tensor(pdfs, dtype=torch.float32)
    def __len__(self): return len(self.lambdas)
    def __getitem__(self, i): return self.lambdas[i], self.pdfs[i]

def make_tail_weights(y, a=6.0, p=2.0):
    yt = torch.tensor(y, dtype=torch.float32)
    return (1.0 + a * (torch.abs(yt) / (torch.abs(yt).max() + 1e-12)) ** p).unsqueeze(0)

def loss_fn(pred, log_pred, target, tw):
    eps = 1e-12; t = target + eps; p = pred + eps
    mse = ((p - t) ** 2).mean(); kl_pw = t * (torch.log(t) - torch.log(p))
    return mse + 0.3 * kl_pw.mean() + 0.7 * (kl_pw * tw).mean()


def train_one(ds, y_grid, tail_w, seed):
    dy = float(y_grid[1] - y_grid[0]); N = len(ds)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True)
    model = FNO1d(y_grid=np.asarray(y_grid, np.float32), lambda_dim=5, n_y=N_Y,
                  modes=MODES, width=WIDTH, layers=LAYERS).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
    final_loss = float("nan"); nan = False
    for _ in range(N_EPOCHS):
        model.train(); tot = 0.0
        for lam, tgt in loader:
            lam, tgt = lam.to(DEVICE), tgt.to(DEVICE)
            pred, logp = model(lam, dy); loss = loss_fn(pred, logp, tgt, tail_w)
            if not torch.isfinite(loss): nan = True; break
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item() * lam.size(0)
        if nan: break
        sched.step(); final_loss = tot / N
    torch.save({"seed": seed, "arch": "FNO1d_kou",
                "config": {"modes": MODES, "width": WIDTH, "layers": LAYERS, "n_y": N_Y, "lambda_dim": 5},
                "y_grid": np.asarray(y_grid), "state_dict": model.state_dict(),
                "final_loss": final_loss, "nan": nan}, CKPTF / f"seed_{seed:04d}.pt")
    return nan


def _kou_ds_ygrid():
    d = np.load(DATA)
    return (KouDataset(d["lambdas"].astype(np.float32), d["pdfs"].astype(np.float32)),
            np.load(YGRID_NPZ)["y_grid"].astype(np.float64))

def worker(sid, ns):
    ds, y_grid = _kou_ds_ygrid(); tail_w = make_tail_weights(y_grid).to(DEVICE)
    done = 0
    for idx, seed in enumerate(SEEDS):
        if idx % ns != sid: continue
        if (CKPTF / f"seed_{seed:04d}.pt").exists(): continue
        train_one(ds, y_grid, tail_w, seed); done += 1
        if done % 5 == 0: time.sleep(60)
    print(f"[worker {sid}] trained {done}", flush=True)


def _norm(raw): return ((raw - KOU_LO) / (KOU_HI - KOU_LO)).astype(np.float32)

def phase_infer(y_grid):
    dy = float(y_grid[1] - y_grid[0])
    if all((PDFS / f"fno_kou_pair{X}_l{s}.npy").exists() for X in "ABCD" for s in (1, 2)):
        print("[infer] npy exist, skip", flush=True); return
    # base pdf (lambda1) once
    def eval_all(lam_norm):
        arr = np.zeros((len(SEEDS), N_Y), dtype=np.float64)
        for i, seed in enumerate(SEEDS):
            ck = torch.load(CKPTF / f"seed_{seed:04d}.pt", map_location=DEVICE, weights_only=False)
            m = FNO1d(y_grid=ck["y_grid"], lambda_dim=5, n_y=N_Y, modes=MODES, width=WIDTH, layers=LAYERS).to(DEVICE)
            m.load_state_dict(ck["state_dict"]); m.eval()
            with torch.no_grad():
                o = m(torch.tensor(lam_norm[None]).to(DEVICE), dy)
                arr[i] = (o[0] if isinstance(o, (tuple, list)) else o).cpu().numpy().squeeze()
        return arr
    base = eval_all(_norm(KOU_FIXED))
    for X, dsig in KOU_PAIRS_DEF.items():
        p2 = KOU_FIXED.copy(); p2[0] = KOU_SIGMA_BASE + dsig
        np.save(PDFS / f"fno_kou_pair{X}_l1.npy", base)
        np.save(PDFS / f"fno_kou_pair{X}_l2.npy", eval_all(_norm(p2)))
    print("[infer] saved 8 npy", flush=True)


def diagnose(get_l1l2, n_seeds, split, y_grid):
    dy = float(y_grid[1] - y_grid[0]); pf = e2.payoffs(np.asarray(y_grid, np.float64))
    n_cal, tlo, thi = split; rows = []
    for X in "ABCD":
        l1, l2 = get_l1l2(X)
        l1c, l1t = l1[:n_cal], l1[tlo:thi]; l2c, l2t = l2[:n_cal], l2[tlo:thi]
        K_test = e2.build_k_eta(l1t, l2t); d_hat = e2.compute_d_hat(l1c, l2c)
        dpdf = l1 - l2
        for pn, g in pf.items():
            dh = dpdf @ g * dy      # all n_seeds
            kt = float(kurtosis(dh, fisher=True)); sk = float(skew(dh))
            try:
                _, lp = lilliefors(dh, dist="norm"); lil = bool(lp < 0.05)
            except Exception:
                lil = None
            ad = anderson(dh, dist="norm"); adr = bool(ad.statistic > ad.critical_values[2])
            ratio, emp, th, sig, mu = e2.thm35_ratio(g, d_hat, K_test, l1t, l2t, dy)
            rows.append(dict(pair=X, payoff=pn, n_seeds=n_seeds,
                             kurt=round(kt, 4), skew=round(sk, 4),
                             lilliefors_rej=lil, ad_rej=adr,
                             ratio=round(ratio, 4) if not np.isnan(ratio) else float("nan"),
                             sigma_delta=round(sig, 8), mu_delta=round(mu, 8)))
    return rows


def _have(): return sum(1 for s in SEEDS if (CKPTF / f"seed_{s:04d}.pt").exists())

def _agg(rows, arch, n_seeds, put_excl=False):
    sub = rows
    if put_excl: sub = [r for r in rows if r["payoff"] != "OTM_put_0.90"]
    ra = np.array([r["ratio"] for r in sub], float); ku = np.array([r["kurt"] for r in sub], float)
    lil = [bool(r["lilliefors_rej"]) for r in sub if r["lilliefors_rej"] is not None]
    adr = [bool(r["ad_rej"]) for r in sub]
    tag = "12cell(call-only)" if put_excl else "16cell"
    print(f"  {arch} N={n_seeds} [{tag}]: n={len(sub)} ratio[{ra.min():.4f},{ra.max():.4f}] "
          f"med={np.median(ra):.4f} kurt_med={np.median(ku):.4f} "
          f"rej_lil={np.mean(lil):.3f} rej_ad={np.mean(adr):.3f}", flush=True)


def main():
    t0 = time.time()
    for s in (0, 1, 2):
        src = PILOT / f"fno_seed_{s:04d}.pt"; dst = CKPTF / f"seed_{s:04d}.pt"
        if src.exists() and not dst.exists(): shutil.copy(src, dst); print(f"[copy] pilot {s}", flush=True)
    print(f"[e1kouF] device={DEVICE} workers={N_WORKERS} existing={_have()}/30", flush=True)
    procs = [subprocess.Popen([sys.executable, __file__, "worker", str(sid), str(N_WORKERS)])
             for sid in range(N_WORKERS)]
    for p in procs: p.wait()
    print(f"[e1kouF] train wall {time.time()-t0:.0f}s ckpts={_have()}/30", flush=True)
    assert _have() == 30, f"incomplete {_have()}/30"

    ds, y_grid = _kou_ds_ygrid()
    phase_infer(y_grid)

    # FNO N=30 diagnostics
    fno_rows = diagnose(lambda X: (np.load(PDFS / f"fno_kou_pair{X}_l1.npy"),
                                   np.load(PDFS / f"fno_kou_pair{X}_l2.npy")), 30, N30, y_grid)
    cols = ["pair", "payoff", "n_seeds", "kurt", "skew", "lilliefors_rej", "ad_rej", "ratio", "sigma_delta", "mu_delta"]
    with open(HERE / "e1_kou_diagnostics.csv", "w", newline="") as f:
        wt = csv.DictWriter(f, fieldnames=cols); wt.writeheader(); wt.writerows(fno_rows)
    print(f"[diagnose] FNO {len(fno_rows)} cells", flush=True)

    # DeepONet N=30 subset & N=200 full
    def deepo(X, N):
        l1 = np.load(DEEPO_NPY / f"kou_pair{X}_l1_200.npy").astype(np.float64)[:N]
        l2 = np.load(DEEPO_NPY / f"kou_pair{X}_l2_200.npy").astype(np.float64)[:N]
        return l1, l2
    d30 = diagnose(lambda X: deepo(X, 30), 30, N30, y_grid)
    d200 = diagnose(lambda X: deepo(X, 200), 200, N200, y_grid)

    ccols = ["arch", "n_seeds", "pair", "payoff", "kurt", "rej_lil", "rej_ad", "ratio", "sigma_delta"]
    out = []
    for arch, rs in [("FNO1d_kou", fno_rows), ("DeepONet_kou", d30), ("DeepONet_kou", d200)]:
        for r in rs:
            out.append(dict(arch=arch, n_seeds=r["n_seeds"], pair=r["pair"], payoff=r["payoff"],
                            kurt=r["kurt"], rej_lil=r["lilliefors_rej"], rej_ad=r["ad_rej"],
                            ratio=r["ratio"], sigma_delta=r["sigma_delta"]))
    with open(HERE / "e1_kou_vs_deeponet.csv", "w", newline="") as f:
        wt = csv.DictWriter(f, fieldnames=ccols); wt.writeheader(); wt.writerows(out)

    print("\n[aggregate] arch x n_seeds (16cell + 12cell call-only)", flush=True)
    _agg(fno_rows, "FNO1d_kou", 30); _agg(fno_rows, "FNO1d_kou", 30, True)
    _agg(d30, "DeepONet_kou", 30); _agg(d30, "DeepONet_kou", 30, True)
    _agg(d200, "DeepONet_kou", 200); _agg(d200, "DeepONet_kou", 200, True)
    print(f"[e1kouF] DONE total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(int(sys.argv[2]), int(sys.argv[3]))
    else:
        main()
