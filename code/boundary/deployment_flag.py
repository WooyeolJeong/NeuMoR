import sys, csv
from pathlib import Path
import numpy as np
import torch
from scipy.stats import kurtosis

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[2]))
import e4_burgers_run as e

_, ubase, _, _ = e.solve_burgers(*e.LSTAR, Nx=e.NX)
c = float(np.quantile(ubase, 0.70))
dx = 1.0 / e.NX; x = np.arange(e.NX) / e.NX
w = np.exp(-0.5 * ((x - 0.5) / 0.1) ** 2)

models = []
for s in e.SEEDS:
    m = e.build_model()
    m.load_state_dict(torch.load(e.CKPT / f"seed_{s:04d}.pt", map_location="cpu", weights_only=False)["state_dict"])
    m.eval(); models.append(m)
def predict(m, lam3):
    ln = (np.array(lam3) - e.LO) / (e.HI - e.LO)
    with torch.no_grad():
        return m(torch.tensor(ln, dtype=torch.float32).unsqueeze(0)).numpy().squeeze()

diag = {}
with open(HERE / "e4_burgers_diagnostics.csv") as f:
    rows = [ln for ln in f if not ln.startswith("#")]
for r in csv.DictReader(rows):
    diag[(r["pair"], r["qoi"])] = (r["rej_lil"] == "True", float(r["abs_s"]))

def atom_frac(D):
    mu = D.mean()
    return float(np.mean(np.abs(D) <= 1e-15 * max(1.0, abs(mu))))
def deploy_rule(atom, kurt_nf, sabs, K=5.0, S=1.5):
    return (atom == atom and atom > 0.10) or (kurt_nf > K and sabs < S)

out = []; e_unique = {}
for pn, dnu in zip("ABCD", e.DNU):
    lam1 = e.LSTAR; lam2 = (e.LSTAR[0] + dnu, e.LSTAR[1], e.LSTAR[2])
    q1 = [e.qois(predict(m, lam1), dx, w, c) for m in models]
    q2 = [e.qois(predict(m, lam2), dx, w, c) for m in models]
    for qn in ["L1", "L2", "M", "E"]:
        D = np.array([q1[s][qn] - q2[s][qn] for s in range(len(models))])
        atom = atom_frac(D)
        kurt_nf = float(kurtosis(D, fisher=False))
        rej_lil, abs_s = diag[(pn, qn)]
        alarm = deploy_rule(atom, kurt_nf, abs_s)
        out.append(dict(pair=pn, qoi=qn, atom=round(atom, 4), kurt_nf=round(kurt_nf, 4),
                        abs_s=round(abs_s, 4), alarm=alarm, rej_lil=rej_lil))
        if qn == "E":
            e_unique[pn] = int(np.unique(np.round(D, 12)).size)

with open(HERE / "e4_deployment_rule.csv", "w", newline="") as f:
    wt = csv.DictWriter(f, fieldnames=["pair", "qoi", "atom", "kurt_nf", "abs_s", "alarm", "rej_lil"])
    wt.writeheader(); wt.writerows(out)

print("[e4-rule] c={:.6f}  e4_deployment_rule.csv (16 rows)".format(c))
print(open(HERE / "e4_deployment_rule.csv").read())

tab = {"alarm_rej": 0, "alarm_norej": 0, "noalarm_rej": 0, "noalarm_norej": 0}
for r in out:
    tab[("alarm" if r["alarm"] else "noalarm") + ("_rej" if r["rej_lil"] else "_norej")] += 1
print("[cross-table] alarm x rej_lil (n=16)")
print(f"                 rej_lil=True   rej_lil=False")
print(f"  alarm=True        {tab['alarm_rej']:>2}             {tab['alarm_norej']:>2}")
print(f"  alarm=False       {tab['noalarm_rej']:>2}             {tab['noalarm_norej']:>2}")
print("[E-cell Delta_hat unique-value counts]:", {k: e_unique[k] for k in "ABCD"})

print("[alarm cells]:", [f"{r['pair']}-{r['qoi']}" for r in out if r["alarm"]])
print("[rej_lil cells]:", [f"{r['pair']}-{r['qoi']}" for r in out if r["rej_lil"]])
