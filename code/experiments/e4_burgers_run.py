"""
E4: non-financial domain transfer — parametric 1D viscous Burgers, DeepONet ensemble N=50.
Stages (auto, background): data-gen -> solver gate -> batch-benchmark -> train -> paired QoI diagnostics.
Solver = pseudo-spectral IFRK4 (Trefethen) + 2/3 dealiasing; diffusion exact via integrating factor.
ratio convention = BOUNDARY-SUITE (mu,sigma BOTH from cal seeds 0-39); NOT 387-suite (sigma from test).
Writes only under experiments/e4_burgers/. Gate fail -> GATE_FAIL marker + exit (no scheme/dt/res change).
Subcommand: `python e4_burgers_run.py worker <mps|cpu> <seed_csv>`  |  driver: `python e4_burgers_run.py`
"""
import sys, os, time, csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA_NPZ = HERE / "e4_burgers_data.npz"
CKPT = HERE / "ckpt"; CKPT.mkdir(parents=True, exist_ok=True)
LOG = HERE / "e4_burgers.log"

NX = 256; T = 1.0; CFL = 0.4
NU_R = (0.005, 0.05); A_R = (0.5, 1.5); B_R = (-0.5, 0.5)
NSAMP = 1000
LSTAR = (0.02, 1.0, 0.0)
DNU = [0.0005, 0.001, 0.002, 0.005]          # pairs A,B,C,D
SEEDS = list(range(50)); N_CAL = 40          # cal 0-39, test 40-49
EPOCHS = 300; BATCH = 64; RANK = 192; BH = 512
LO = np.array([NU_R[0], A_R[0], B_R[0]]); HI = np.array([NU_R[1], A_R[1], B_R[1]])


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


# ---------------- solver: pseudo-spectral IFRK4 + 2/3 dealias ----------------
def solve_burgers(nu, a, b, Nx=NX, Tf=T, cfl=CFL):
    x = np.arange(Nx) / Nx
    u = a * np.sin(2 * np.pi * x) + b * np.cos(4 * np.pi * x)
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=1.0 / Nx)
    k2 = k * k
    dealias = (np.abs(k) < (2.0 / 3.0) * np.max(np.abs(k)))
    dx = 1.0 / Nx
    umax = np.max(np.abs(u)) + 1e-12
    nsteps = int(np.ceil(Tf / (cfl * dx / umax))); dt = Tf / nsteps
    L = -nu * k2
    E = np.exp(dt * L / 2.0); E2 = E * E
    ik = 1j * k
    def Nl(uh):
        ur = np.real(np.fft.ifft(uh))
        return -0.5 * ik * (dealias * np.fft.fft(ur * ur))
    uh = np.fft.fft(u)
    e0 = np.sum(u * u) * dx; eprev = e0; emono = True
    chk = max(1, nsteps // 20)
    for n in range(nsteps):
        aa = dt * Nl(uh)
        bb = dt * Nl(E * (uh + aa / 2))
        cc = dt * Nl(E * uh + bb / 2)
        dd = dt * Nl(E2 * uh + E * cc)
        uh = E2 * uh + (E2 * aa + 2 * E * (bb + cc) + dd) / 6.0
        if (n + 1) % chk == 0:
            e = np.sum(np.real(np.fft.ifft(uh)) ** 2) * dx
            if e > eprev + 1e-8 * e0:
                emono = False
            eprev = e
    return x, np.real(np.fft.ifft(uh)), emono, nsteps


def rel_l2_256_vs_512(nu, a, b):
    _, u2, m2, _ = solve_burgers(nu, a, b, Nx=256)
    _, u5, m5, _ = solve_burgers(nu, a, b, Nx=512)
    u5c = u5[::2]                                   # subsample 512->256 (aligned grids)
    return float(np.sqrt(np.sum((u2 - u5c) ** 2)) / (np.sqrt(np.sum(u5c ** 2)) + 1e-30)), m2, m5


# ---------------- data generation (P-core parallel) ----------------
def _solve_one(args):
    nu, a, b = args
    _, u, _, _ = solve_burgers(nu, a, b, Nx=NX)
    return u


def gen_data():
    from scipy.stats import qmc
    U = qmc.LatinHypercube(d=3, seed=0).random(NSAMP)
    lam = LO + U * (HI - LO)
    import multiprocessing as mp
    P = max(1, mp.cpu_count() - 1)
    with mp.Pool(P) as pool:
        fields = np.array(pool.map(_solve_one, [tuple(r) for r in lam]))
    np.savez(DATA_NPZ, lambdas=lam, fields=fields, x=np.arange(NX) / NX)
    log(f"[stage1] data: {NSAMP} solves P={P} fields{fields.shape} -> {DATA_NPZ.name}")
    return P


# ---------------- gate: 6 lambda (5 random + worst corner) ----------------
def run_gate():
    rng = np.random.default_rng(123)
    lams = [(rng.uniform(*NU_R), rng.uniform(*A_R), rng.uniform(*B_R)) for _ in range(5)]
    lams.append((0.005, 1.5, -0.5))              # forced worst corner
    ok = True; res = []
    for i, (nu, a, b) in enumerate(lams):
        rl, m2, m5 = rel_l2_256_vs_512(nu, a, b)
        tag = "WORST" if i == 5 else f"rand{i}"
        p = (rl < 1e-3) and m2 and m5
        ok = ok and p
        res.append((tag, nu, a, b, rl, m2, m5, p))
        log(f"[gate] {tag}: nu={nu:.4f} a={a:.3f} b={b:+.3f} relL2={rl:.3e} emono256={m2} emono512={m5} {'PASS' if p else 'FAIL'}")
    return ok, res


# ---------------- model ----------------
def build_model():
    import torch, torch.nn as nn
    class DON(nn.Module):
        def __init__(self):
            super().__init__()
            self.branch = nn.Sequential(nn.Linear(3, BH), nn.ReLU(),
                                        nn.Linear(BH, BH), nn.ReLU(), nn.Linear(BH, RANK))
            self.trunk = nn.Parameter(torch.randn(NX, RANK) * 0.1)
        def forward(self, lam):
            return self.branch(lam) @ self.trunk.T          # linear output (no softmax)
    return DON()


def worker(device, seeds):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    if device == "cpu":
        torch.set_num_threads(1)
    d = np.load(DATA_NPZ)
    Xn = ((d["lambdas"] - LO) / (HI - LO)).astype(np.float32)
    Y = d["fields"].astype(np.float32)
    ds = TensorDataset(torch.tensor(Xn), torch.tensor(Y))
    for seed in seeds:
        p = CKPT / f"seed_{seed:04d}.pt"
        if p.exists():
            continue
        torch.manual_seed(seed); np.random.seed(seed)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=True)
        m = build_model().to(device)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(EPOCHS):
            m.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = ((m(xb) - yb) ** 2).mean()
                if not torch.isfinite(loss):
                    print(f"[DIVERGE] seed={seed}", flush=True); sys.exit(3)
                opt.zero_grad(); loss.backward(); opt.step()
        torch.save({"seed": seed, "state_dict": m.state_dict()}, p)


def _spawn(device, seedlists):
    import subprocess
    procs = [subprocess.Popen([sys.executable, __file__, "worker", device, ",".join(map(str, sl))])
             for sl in seedlists if sl]
    for p in procs:
        p.wait()


def train_all():
    import torch
    have_mps = torch.backends.mps.is_available()
    # --- benchmark: PAR4-MPS (seeds 0-3) vs PAR8-CPU (seeds 4-11) ---
    if have_mps:
        t = time.time(); _spawn("mps", [[0], [1], [2], [3]]); w_mps = time.time() - t; sps_mps = w_mps / 4
    else:
        w_mps = float("nan"); sps_mps = float("inf")
    t = time.time(); _spawn("cpu", [[4 + j] for j in range(8)]); w_cpu = time.time() - t; sps_cpu = w_cpu / 8
    log(f"[bench] PAR4-MPS sec/seed={sps_mps:.2f}(wall {w_mps:.1f}s) | PAR8-CPU sec/seed={sps_cpu:.2f}(wall {w_cpu:.1f}s)")
    dev, par = ("mps", 4) if sps_mps <= sps_cpu else ("cpu", 8)
    log(f"[bench] winner={dev} PAR{par}")
    # --- train remaining (skip-existing) with cooling every batch of `par` ---
    remaining = [s for s in SEEDS if not (CKPT / f"seed_{s:04d}.pt").exists()]
    for i in range(0, len(remaining), par):
        _spawn(dev, [[s] for s in remaining[i:i + par]])
        if i + par < len(remaining):
            time.sleep(5)                                    # cooling
    return dev, par, sps_mps, sps_cpu


# ---------------- QoI + diagnostics ----------------
def qois(u, dx, w, c):
    return dict(L1=float(np.sum(u) * dx), L2=float(np.sum(w * u) * dx),
                M=float(np.max(u)), E=float(np.mean(u > c)))


def folded_mean(mu, sigma):
    from scipy.special import erf
    if sigma < 1e-30:
        return abs(mu)
    s = mu / sigma
    return float(sigma * np.sqrt(2 / np.pi) * np.exp(-0.5 * s * s) + mu * erf(s / np.sqrt(2)))


def diagnose(c, dx, w):
    import torch
    from scipy.stats import kurtosis, skew, anderson
    from statsmodels.stats.diagnostic import lilliefors
    models = []
    for s in SEEDS:
        m = build_model()
        m.load_state_dict(torch.load(CKPT / f"seed_{s:04d}.pt", map_location="cpu", weights_only=False)["state_dict"])
        m.eval(); models.append(m)
    def predict(m, lam3):
        ln = (np.array(lam3) - LO) / (HI - LO)
        with torch.no_grad():
            return m(torch.tensor(ln, dtype=torch.float32).unsqueeze(0)).numpy().squeeze()
    rows = []
    for pn, dnu in zip("ABCD", DNU):
        lam1 = LSTAR; lam2 = (LSTAR[0] + dnu, LSTAR[1], LSTAR[2])
        q1 = [qois(predict(m, lam1), dx, w, c) for m in models]
        q2 = [qois(predict(m, lam2), dx, w, c) for m in models]
        for qn in ["L1", "L2", "M", "E"]:
            D = np.array([q1[s][qn] - q2[s][qn] for s in range(50)])
            kt = float(kurtosis(D, fisher=True)); sk = float(skew(D))
            try:
                _, lp = lilliefors(D, dist="norm"); lil = bool(lp < 0.05)
            except Exception:
                lil = None
            ad = anderson(D, dist="norm"); adr = bool(ad.statistic > ad.critical_values[2])
            cal, tst = D[:N_CAL], D[N_CAL:]
            mu = float(cal.mean()); sig = float(cal.std(ddof=1))     # boundary-suite: mu,sigma from cal
            theory = folded_mean(mu, sig); emp = float(np.mean(np.abs(tst)))
            ratio = emp / theory if abs(theory) > 1e-30 else float("nan")
            sabs = abs(mu) / sig if sig > 1e-30 else float("inf")
            rows.append(dict(pair=pn, qoi=qn, kurt=round(kt, 4), skew=round(sk, 4),
                             rej_lil=lil, rej_ad=adr,
                             ratio=round(ratio, 4) if not np.isnan(ratio) else float("nan"),
                             sigma_delta=round(sig, 8), mu_delta=round(mu, 8), abs_s=round(sabs, 4)))
    with open(HERE / "e4_burgers_diagnostics.csv", "w", newline="") as f:
        f.write("# ratio convention = boundary-suite (mu,sigma BOTH from cal seeds 0-39); NOT 387-suite (sigma from test)\n")
        wt = csv.DictWriter(f, fieldnames=["pair", "qoi", "kurt", "skew", "rej_lil", "rej_ad",
                                           "ratio", "sigma_delta", "mu_delta", "abs_s"])
        wt.writeheader(); wt.writerows(rows)
    return rows


def main():
    if LOG.exists():
        LOG.unlink()
    t_all = time.time(); log("[e4] START")

    t = time.time(); gen_data(); log(f"[stage1] wall={time.time()-t:.0f}s")

    t = time.time(); ok, res = run_gate(); log(f"[stage2] gate wall={time.time()-t:.0f}s ok={ok}")
    if not ok:
        (HERE / "GATE_FAIL").write_text("gate failed\n")
        log("[e4] GATE_FAIL — stopping (no scheme/dt/resolution change).")
        sys.exit(2)

    dx = 1.0 / NX; x = np.arange(NX) / NX
    w = np.exp(-0.5 * ((x - 0.5) / 0.1) ** 2)
    _, ubase, _, _ = solve_burgers(*LSTAR, Nx=NX)
    c = float(np.quantile(ubase, 0.70)); exc = float(np.mean(ubase > c))
    log(f"[c] c={c:.6f} base_exceedance={exc:.3f} (target 0.20-0.40)")

    t = time.time(); dev, par, sm, sc = train_all(); log(f"[stage3] train wall={time.time()-t:.0f}s dev={dev} PAR{par}")
    have = sum(1 for s in SEEDS if (CKPT / f"seed_{s:04d}.pt").exists())
    assert have == 50, f"incomplete {have}/50"

    t = time.time(); rows = diagnose(c, dx, w)
    log(f"[stage4] diagnose wall={time.time()-t:.0f}s -> e4_burgers_diagnostics.csv ({len(rows)} rows)")

    def summ(name, sub):
        ra = [r["ratio"] for r in sub if not (isinstance(r["ratio"], float) and np.isnan(r["ratio"]))]
        ku = [r["kurt"] for r in sub]
        rl = np.mean([1 if r["rej_lil"] else 0 for r in sub if r["rej_lil"] is not None])
        ra_ = np.mean([1 if r["rej_ad"] else 0 for r in sub])
        log(f"  {name}: n={len(sub)} ratio[{min(ra):.4f},{max(ra):.4f}] kurt[{min(ku):.3f},{max(ku):.3f}] "
            f"rej_lil={rl:.3f} rej_ad={ra_:.3f}")
    log("[aggregate] linear vs nonsmooth QoI:")
    summ("linear(L1,L2)", [r for r in rows if r["qoi"] in ("L1", "L2")])
    summ("nonsmooth(M,E)", [r for r in rows if r["qoi"] in ("M", "E")])
    log(f"[e4] DONE total={time.time()-t_all:.0f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(sys.argv[2], [int(s) for s in sys.argv[3].split(",")])
    else:
        main()
