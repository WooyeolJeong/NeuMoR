import sys, csv
from pathlib import Path
import numpy as np
import torch
from scipy.stats import kurtosis, skew, anderson
from statsmodels.stats.diagnostic import lilliefors

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "NeuMoR" / "experiments" / "e2_width"))
import e2_width_run as e2
import e5_dropout_run as e5r

SEEDS20 = list(range(20))

def deploy_rule(atom, kurt_ff, sabs, K=5.0, S=1.5):

    a_ok = (atom == atom and atom > 0.10)
    return a_ok or (kurt_ff > K and sabs < S)

def atom_frac(D, mu):

    return float(np.mean(np.abs(D) <= 1e-15 * max(1.0, abs(mu))))

def main():
    ds = e2.HestonDataset()
    y_grid = np.load(e2.YGRID_NPZ)["y_grid"].astype(np.float64)
    dy = float(y_grid[1] - y_grid[0]); n_y = len(y_grid)
    lambda_dim = ds.lambdas.shape[1]; pk = ds.param_keys
    tail_w = e2.make_tail_weights(y_grid).to(e2.DEVICE)
    pf = e2.payoffs(np.asarray(y_grid, dtype=np.float64))
    lam1 = torch.tensor(e2.lam_vec(0.10, pk)).unsqueeze(0)

    walls = {}
    for s in SEEDS20:
        p = e5r.CKPT / f"seed_{s:04d}.pt"
        if p.exists():
            walls[s] = None; continue
        fl, w = e5r.train_seed(ds, y_grid, tail_w, s); walls[s] = w
        print(f"[train] seed {s} loss={fl:.3e} wall={w:.1f}s", flush=True)
    have = sum(1 for s in SEEDS20 if (e5r.CKPT / f"seed_{s:04d}.pt").exists())
    print(f"[train] trained_this_run={sum(1 for s in SEEDS20 if walls.get(s))}/17new  ckpts={have}/20", flush=True)

    rows = []
    for s in SEEDS20:
        model = e5r.load_drop(e5r.CKPT / f"seed_{s:04d}.pt", lambda_dim, n_y)
        for X in "ABCD":
            lam2 = torch.tensor(e2.lam_vec(e2.PAIR_V2[X], pk)).unsqueeze(0)
            for coupled in (True, False):
                cname = "coupled" if coupled else "independent"
                f1 = e5r.PDFS / f"e5_seed{s}_pair{X}_{cname}_l1.npy"
                f2 = e5r.PDFS / f"e5_seed{s}_pair{X}_{cname}_l2.npy"
                if f1.exists() and f2.exists():
                    l1 = np.load(f1); l2 = np.load(f2)
                else:
                    l1, l2 = e5r.mc_sample(model, lam1, lam2, dy, n_y, coupled)
                    np.save(f1, l1); np.save(f2, l2)
                l1c, l1t = l1[:e2.N_CAL], l1[e2.TEST_LO:e2.TEST_HI]
                l2c, l2t = l2[:e2.N_CAL], l2[e2.TEST_LO:e2.TEST_HI]
                K_test = e2.build_k_eta(l1t, l2t); d_hat = e2.compute_d_hat(l1c, l2c)
                dpdf = l1 - l2
                for pn, g in pf.items():
                    dh = dpdf @ g * dy
                    kt = float(kurtosis(dh, fisher=True))
                    kff = float(kurtosis(dh, fisher=False))
                    sk = float(skew(dh))
                    try:
                        _, lp = lilliefors(dh, dist="norm"); lil = bool(lp < 0.05)
                    except Exception:
                        lil = None
                    ad = anderson(dh, dist="norm"); adr = bool(ad.statistic > ad.critical_values[2])
                    ratio, emp, theory, sigma, mu = e2.thm35_ratio(g, d_hat, K_test, l1t, l2t, dy)
                    sabs = abs(mu) / sigma if sigma > 1e-30 else float("inf")
                    at = atom_frac(dh, float(dh.mean()))
                    rows.append(dict(seed=s, coupling=cname, pair=X, payoff=pn,
                        kurt=round(kt, 4), kurt_nonfisher=round(kff, 4), skew=round(sk, 4),
                        lilliefors_rej=lil, ad_rej=adr,
                        ratio=round(ratio, 4) if not np.isnan(ratio) else float("nan"),
                        sigma_delta=round(sigma, 8), mu_delta=round(mu, 8),
                        abs_s=round(sabs, 4), atom=round(at, 4)))
    cols = ["seed", "coupling", "pair", "payoff", "kurt", "kurt_nonfisher", "skew",
            "lilliefors_rej", "ad_rej", "ratio", "sigma_delta", "mu_delta", "abs_s", "atom"]
    with open(HERE / "e5_dropout_diagnostics_n20.csv", "w", newline="") as f:
        wt = csv.DictWriter(f, fieldnames=cols); wt.writeheader(); wt.writerows(rows)
    print(f"[diagnose] {len(rows)} cells -> e5_dropout_diagnostics_n20.csv", flush=True)

    prev = []
    for s in SEEDS20:
        for X in "ABCD":
            sub = [r for r in rows if r["coupling"] == "coupled" and r["seed"] == s and r["pair"] == X]
            nlil = sum(r["lilliefors_rej"] is True for r in sub)
            nad = sum(r["ad_rej"] is True for r in sub)
            prev.append(dict(seed=s, pair=X, n_payoff=len(sub), rej_lil_n=nlil, rej_ad_n=nad,
                             kurt_max=round(max(r["kurt"] for r in sub), 4), any_lil=int(nlil > 0)))
    with open(HERE / "e5_seed_prevalence.csv", "w", newline="") as f:
        wt = csv.DictWriter(f, fieldnames=["seed", "pair", "n_payoff", "rej_lil_n", "rej_ad_n", "kurt_max", "any_lil"])
        wt.writeheader(); wt.writerows(prev)
    print("\n[prevalence] coupled #seeds(of 20) with >=1 Lilliefors-rej, per pair:", flush=True)
    for X in "ABCD":
        nf = sum(1 for s in SEEDS20 if any(p["seed"] == s and p["pair"] == X and p["any_lil"] for p in prev))
        km = max(p["kurt_max"] for p in prev if p["pair"] == X)
        print(f"  pair{X}: {nf}/20 seeds flagged  kurt_max_over_seeds={km:.2f}", flush=True)
    small = sum(1 for s in SEEDS20 if any(p["seed"] == s and p["pair"] in "ABC" and p["any_lil"] for p in prev))
    print(f"  small-gap(A/B/C) any-rej: {small}/20 seeds", flush=True)

    print("\n[deploy-rule] mm1_Tladder.py L54(atom)+L124-126(rule): "
          "alarm=(atom>0.10) OR (kurt_nonfisher>5 AND |s|<1.5); |s|=|mu_delta|/sigma_delta", flush=True)
    cc = [r for r in rows if r["coupling"] == "coupled"]
    for bname, lo, hi in [("[0.7,1.3]", 0.7, 1.3), ("[0.9,1.1]", 0.9, 1.1)]:
        tab = {"alarm_in": 0, "alarm_out": 0, "noalarm_in": 0, "noalarm_out": 0}
        for r in cc:
            alarm = deploy_rule(r["atom"], r["kurt_nonfisher"], r["abs_s"])
            inband = (not (isinstance(r["ratio"], float) and np.isnan(r["ratio"]))) and (lo <= r["ratio"] <= hi)
            tab[("alarm" if alarm else "noalarm") + ("_in" if inband else "_out")] += 1
        print(f"  band {bname}: n={len(cc)} | "
              f"alarm&in={tab['alarm_in']} alarm&out={tab['alarm_out']} "
              f"noalarm&in={tab['noalarm_in']} noalarm&out={tab['noalarm_out']}", flush=True)
    ss = [r["abs_s"] for r in cc]; ats = [r["atom"] for r in cc]; kff = [r["kurt_nonfisher"] for r in cc]
    nalarm = sum(deploy_rule(r["atom"], r["kurt_nonfisher"], r["abs_s"]) for r in cc)
    print(f"  coupled(n={len(cc)}): alarms={nalarm} | |s|[{min(ss):.2f},{max(ss):.2f}] "
          f"atom[{min(ats):.3f},{max(ats):.3f}] kurt_nf[{min(kff):.2f},{max(kff):.2f}]", flush=True)

    def agg(name, sub):
        ra = np.array([r["ratio"] for r in sub if not (isinstance(r["ratio"], float) and np.isnan(r["ratio"]))], float)
        ku = np.array([r["kurt"] for r in sub], float)
        lil = [bool(r["lilliefors_rej"]) for r in sub if r["lilliefors_rej"] is not None]
        adr = [bool(r["ad_rej"]) for r in sub]
        return (f"  {name}: n={len(sub)} ratio[{ra.min():.4f},{ra.max():.4f}] med={np.median(ra):.4f} "
                f"kurt_med={np.median(ku):.4f} rej_lil={np.mean(lil):.3f} rej_ad={np.mean(adr):.3f}")
    print("\n[aggregate 20-seed]", flush=True)
    print(agg("coupled", [r for r in rows if r["coupling"] == "coupled"]), flush=True)
    print(agg("independent", [r for r in rows if r["coupling"] == "independent"]), flush=True)
    nt = [s for s in SEEDS20 if walls.get(s)]
    if nt:
        print(f"\n[e5-ext] wall/seed newly-trained (mini): " +
              " ".join(f"s{s}={walls[s]:.1f}s" for s in nt), flush=True)
    print("[e5-ext] DONE", flush=True)

if __name__ == "__main__":
    main()
