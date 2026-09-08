"""
E1 Phase 2: FNO1d 200 seeds (PAR4-MPS parallel) + E2-identical diagnostics + DeepONet compare.
Reuses e2_width_run (thm35_ratio, build_k_eta, compute_d_hat, payoffs, lam_vec, protocol A).
FNO1d modes32/width64/layers4 (pilot, unchanged). pilot seeds 0-2 copied to full/.
Worker: python e1_full.py worker <sid> <ns>   Driver: python e1_full.py
Writes ONLY under experiments/e1_fno/. Existing code/npz/ckpt read-only.
"""
import sys, time, csv, shutil, subprocess
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import kurtosis, skew, anderson
from statsmodels.stats.diagnostic import lilliefors

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "NeuMoR" / "experiments" / "e2_width"))
from fno_model import FNO1d
import e2_width_run as e2   # reuse

CKPTF = HERE / "ckpt" / "full"; PDFS = HERE / "pdfs"
PILOT = HERE / "ckpt" / "pilot"
for d in (CKPTF, PDFS): d.mkdir(parents=True, exist_ok=True)
N_WORKERS = 4
MODES, WIDTH, LAYERS = 32, 64, 4
SEEDS = list(range(200))
E2_DIAG = ROOT / "NeuMoR" / "experiments" / "e2_width" / "e2_width_diagnostics.csv"


def train_one(ds, y_grid, tail_w, seed):
    dy = float(y_grid[1] - y_grid[0]); n_y = len(y_grid); N = len(ds)
    e2.set_seed(seed)
    loader = DataLoader(ds, batch_size=e2.BATCH, shuffle=True)
    model = FNO1d(y_grid=ds.y_grid, lambda_dim=5, n_y=n_y,
                  modes=MODES, width=WIDTH, layers=LAYERS).to(e2.DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    final_loss = float("nan"); nan = False
    for _ in range(e2.EPOCHS):
        model.train(); tot = 0.0
        for lam, tgt in loader:
            lam, tgt = lam.to(e2.DEVICE), tgt.to(e2.DEVICE)
            pred, logp = model(lam, dy)
            loss = e2.loss_fn(pred, logp, tgt, tail_w)
            if not torch.isfinite(loss): nan = True; break
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * lam.size(0)
        if nan: break
        final_loss = tot / N
    torch.save({"seed": seed, "arch": "FNO1d",
                "config": {"modes": MODES, "width": WIDTH, "layers": LAYERS,
                           "n_y": n_y, "lambda_dim": 5},
                "y_grid": ds.y_grid, "param_keys": ds.param_keys,
                "state_dict": model.state_dict(), "final_loss": final_loss, "nan": nan},
               CKPTF / f"seed_{seed:04d}.pt")
    return final_loss, nan


def worker(sid, ns):
    ds = e2.HestonDataset()
    y_grid = np.load(e2.YGRID_NPZ)["y_grid"].astype(np.float64)
    tail_w = e2.make_tail_weights(ds.y_grid).to(e2.DEVICE)
    done = 0
    for idx, seed in enumerate(SEEDS):
        if idx % ns != sid: continue
        if (CKPTF / f"seed_{seed:04d}.pt").exists(): continue
        _, nan = train_one(ds, y_grid, tail_w, seed); done += 1
        if done % max(100 // ns, 1) == 0: time.sleep(60)   # cooling ~100 models total
    print(f"[worker {sid}] trained {done}", flush=True)


def _load_fno(path, n_y):
    ck = torch.load(path, map_location=e2.DEVICE, weights_only=False)
    c = ck["config"]
    m = FNO1d(y_grid=ck["y_grid"], lambda_dim=c["lambda_dim"], n_y=c["n_y"],
              modes=c["modes"], width=c["width"], layers=c["layers"]).to(e2.DEVICE)
    m.load_state_dict(ck["state_dict"]); m.eval(); return m


def phase_infer(ds, y_grid):
    dy = float(y_grid[1] - y_grid[0]); n_y = len(y_grid); pk = ds.param_keys
    if all((PDFS / f"fno_pair{X}_l{s}.npy").exists() for X in "ABCD" for s in (1, 2)):
        print("[infer] all npy exist, skip", flush=True); return
    vmap = {"0.10": 0.10, **{f"{v:.3f}": v for v in e2.PAIR_V2.values()}}
    cache = {}
    for vk, vv in vmap.items():
        lam = torch.tensor(e2.lam_vec(vv, pk)).unsqueeze(0).to(e2.DEVICE)
        arr = np.zeros((len(SEEDS), n_y), dtype=np.float64)
        for i, seed in enumerate(SEEDS):
            m = _load_fno(CKPTF / f"seed_{seed:04d}.pt", n_y)
            with torch.no_grad():
                out = m(lam, dy); pdf = out[0] if isinstance(out, (tuple, list)) else out
            arr[i] = pdf.cpu().numpy().squeeze()
        cache[vk] = arr
    for X in "ABCD":
        np.save(PDFS / f"fno_pair{X}_l1.npy", cache["0.10"])
        np.save(PDFS / f"fno_pair{X}_l2.npy", cache[f"{e2.PAIR_V2[X]:.3f}"])
    print(f"[infer] saved (v={list(vmap)})", flush=True)


def phase_diagnose(y_grid):
    dy = float(y_grid[1] - y_grid[0]); pf = e2.payoffs(np.asarray(y_grid, dtype=np.float64))
    rows = []
    for X in "ABCD":
        l1 = np.load(PDFS / f"fno_pair{X}_l1.npy").astype(np.float64)
        l2 = np.load(PDFS / f"fno_pair{X}_l2.npy").astype(np.float64)
        l1c, l1t = l1[:e2.N_CAL], l1[e2.TEST_LO:e2.TEST_HI]
        l2c, l2t = l2[:e2.N_CAL], l2[e2.TEST_LO:e2.TEST_HI]
        K_test = e2.build_k_eta(l1t, l2t); d_hat = e2.compute_d_hat(l1c, l2c)
        dpdf = l1 - l2
        for pn, g in pf.items():
            dh = dpdf @ g * dy
            kt = float(kurtosis(dh, fisher=True)); sk = float(skew(dh))
            try:
                _, lp = lilliefors(dh, dist="norm"); lil = bool(lp < 0.05)
            except Exception:
                lil = None
            ad = anderson(dh, dist="norm"); adr = bool(ad.statistic > ad.critical_values[2])
            ratio, emp, theory, sigma, mu = e2.thm35_ratio(g, d_hat, K_test, l1t, l2t, dy)
            rows.append(dict(pair=X, payoff=pn, kurt=round(kt, 4), skew=round(sk, 4),
                             lilliefors_rej=lil, ad_rej=adr,
                             ratio=round(ratio, 4) if not np.isnan(ratio) else float("nan"),
                             sigma_delta=round(sigma, 8), mu_delta=round(mu, 8)))
    cols = ["pair", "payoff", "kurt", "skew", "lilliefors_rej", "ad_rej",
            "ratio", "sigma_delta", "mu_delta"]
    with open(HERE / "e1_fno_diagnostics.csv", "w", newline="") as f:
        wt = csv.DictWriter(f, fieldnames=cols); wt.writeheader(); wt.writerows(rows)
    print(f"[diagnose] {len(rows)} FNO cells", flush=True)
    return rows


def comparison(fno_rows):
    # DeepONet w512 (16 cells) from e2_width_diagnostics.csv
    deepo = []
    with open(E2_DIAG) as f:
        for r in csv.DictReader(f):
            if int(r["width"]) == 512:
                deepo.append(r)
    out = []
    for r in fno_rows:
        out.append(dict(arch="FNO1d", pair=r["pair"], payoff=r["payoff"], kurt=r["kurt"],
                        rej_lil=r["lilliefors_rej"], rej_ad=r["ad_rej"],
                        ratio=r["ratio"], sigma_delta=r["sigma_delta"]))
    for r in deepo:
        out.append(dict(arch="DeepONet_w512", pair=r["pair"], payoff=r["payoff"],
                        kurt=float(r["kurt"]), rej_lil=(r["lilliefors_rej"] == "True"),
                        rej_ad=(r["ad_rej"] == "True"), ratio=float(r["ratio"]),
                        sigma_delta=float(r["sigma_delta"])))
    cols = ["arch", "pair", "payoff", "kurt", "rej_lil", "rej_ad", "ratio", "sigma_delta"]
    with open(HERE / "e1_vs_deeponet.csv", "w", newline="") as f:
        wt = csv.DictWriter(f, fieldnames=cols); wt.writeheader(); wt.writerows(out)

    print("\n[aggregate] arch | n | ratio_min | ratio_max | ratio_med | kurt_med | rej_lil | rej_ad",
          flush=True)
    for arch in ("FNO1d", "DeepONet_w512"):
        sub = [r for r in out if r["arch"] == arch]
        ra = np.array([r["ratio"] for r in sub], float)
        ku = np.array([r["kurt"] for r in sub], float)
        lil = [bool(r["rej_lil"]) for r in sub if r["rej_lil"] is not None]
        adr = [bool(r["rej_ad"]) for r in sub]
        print(f"  {arch}: n={len(sub)} ratio[{ra.min():.4f},{ra.max():.4f}] med={np.median(ra):.4f} "
              f"kurt_med={np.median(ku):.4f} rej_lil={np.mean(lil):.3f} rej_ad={np.mean(adr):.3f}",
              flush=True)


def _have():
    return sum(1 for s in SEEDS if (CKPTF / f"seed_{s:04d}.pt").exists())


def main():
    t0 = time.time()
    # copy pilot seeds 0-2 -> full (reuse; same-config training artifact)
    for s in (0, 1, 2):
        src = PILOT / f"fno_seed_{s:04d}.pt"; dst = CKPTF / f"seed_{s:04d}.pt"
        if src.exists() and not dst.exists():
            shutil.copy(src, dst); print(f"[copy] pilot seed {s} -> full", flush=True)
    print(f"[e1full] device={e2.DEVICE} workers={N_WORKERS} existing={_have()}/200", flush=True)

    procs = [subprocess.Popen([sys.executable, __file__, "worker", str(sid), str(N_WORKERS)])
             for sid in range(N_WORKERS)]
    for p in procs: p.wait()
    print(f"[e1full] train wall {time.time()-t0:.0f}s ckpts={_have()}/200", flush=True)
    assert _have() == 200, f"incomplete {_have()}/200"

    ds = e2.HestonDataset()
    y_grid = np.load(e2.YGRID_NPZ)["y_grid"].astype(np.float64)
    phase_infer(ds, y_grid)
    fno_rows = phase_diagnose(y_grid)
    comparison(fno_rows)
    print(f"[e1full] ALL DONE total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(int(sys.argv[2]), int(sys.argv[3]))
    else:
        main()
