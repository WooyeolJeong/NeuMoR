"""
NeuMoR Experiment 03: Direct Jensen Isolation.
Loads Protocol A (Exp 01) + Protocol B (Exp 03) models.
Computes cross-protocol ρ, Tests A/B for Theorem 2, α robustness.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy.special import erf, erfc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from NeuMoR.src.nmrb_model import DensityDeepONet_HestonLog
from NeuMoR.src.heston_teacher import lewis_price, price_from_pdf

SAVE_A    = PROJECT_ROOT / "NeuMoR" / "save"   / "exp01"
SAVE_B    = PROJECT_ROOT / "NeuMoR" / "save"   / "exp03"
OUTPUT    = PROJECT_ROOT / "NeuMoR" / "output" / "exp03"
OUTPUT.mkdir(parents=True, exist_ok=True)

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

N_SEEDS      = 30        # use first 30 of Protocol A; all 30 of B
PAYOFF_NAMES = ["ATM", "OTM_K110", "OTM_K125", "DIGITAL"]
PAIR_NAMES   = ["A", "B", "C", "D", "E"]
DV_MAP       = {"A": 0.001, "B": 0.005, "C": 0.010, "D": 0.030, "E": 0.100}
BASE         = {"v": 0.10, "kappa": 1.0, "omega": 0.2, "xi": 0.3, "rho": -0.5, "T": 1.0}
PAIRS        = {
    "A": (0.10, 0.101), "B": (0.10, 0.105), "C": (0.10, 0.11),
    "D": (0.10, 0.13),  "E": (0.10, 0.20),
}

# Protocol definitions: (save_subdir, rank, branch_hidden)
PROTOCOL_SPECS = {
    "A":  (SAVE_A,                     192, 512),
    "B1": (SAVE_B / "protocol_B1",      96, 256),
    "B2": (SAVE_B / "protocol_B2",     192, 512),
    "B3": (SAVE_B / "protocol_B3",     192, 512),
}

# Protocol pairs: (proto for λ_1, proto for λ_2)
PROTOCOL_PAIRS = {
    "AA":  ("A",  "A"),
    "AB1": ("A",  "B1"),
    "AB2": ("A",  "B2"),
    "AB3": ("A",  "B3"),
    "B1B1": ("B1", "B1"),
    "B2B2": ("B2", "B2"),
}


# ── Folded-normal helpers ─────────────────────────────────────────────────

def E_abs(mu, sigma):
    if sigma < 1e-15:
        return abs(mu)
    s = mu / sigma
    return sigma * (np.sqrt(2 / np.pi) * np.exp(-0.5 * s * s)
                    + s * erf(s / np.sqrt(2)))


def jensen_component(mu, sigma):
    if sigma < 1e-15:
        return 0.0
    s = abs(mu) / sigma
    return sigma * (np.sqrt(2 / np.pi) * np.exp(-0.5 * s * s)
                    - s * erfc(s / np.sqrt(2)))


# ── Model loading ─────────────────────────────────────────────────────────

def load_model(seed: int, proto: str):
    save_dir, rank, branch_hidden = PROTOCOL_SPECS[proto]
    path = save_dir / f"seed_{seed:02d}.pt"
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    cfg  = ckpt["config"]
    model = DensityDeepONet_HestonLog(
        lambda_dim=cfg["lambda_dim"], n_y=cfg["n_y"],
        rank=cfg["rank"], branch_hidden=cfg["branch_hidden"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["y_grid"], ckpt["param_keys"]


def model_price(model, params: dict, payoff_name: str,
                y_grid: np.ndarray, param_keys: list) -> float:
    lam   = np.array([params[k] for k in param_keys], dtype=np.float32)
    lam_t = torch.tensor(lam).unsqueeze(0).to(DEVICE)
    dy    = float(y_grid[1] - y_grid[0])
    with torch.no_grad():
        out = model(lam_t, dy)
        pdf = out[0] if isinstance(out, (tuple, list)) else out
    f = pdf.cpu().numpy().squeeze(0)
    return price_from_pdf(y_grid, f, payoff_name, T=params.get("T", 1.0))


# ── Lewis ground truth ────────────────────────────────────────────────────

def compute_lewis() -> dict:
    print("[exp03] computing Lewis ground truth ...")
    true = {}
    for pname, (v1, v2) in PAIRS.items():
        p1, p2 = {**BASE, "v": v1}, {**BASE, "v": v2}
        for pn in PAYOFF_NAMES:
            true[(pname, pn)] = (lewis_price(p1, pn), lewis_price(p2, pn))
    return true


# ── Cross-protocol price computation ─────────────────────────────────────

def compute_prices(true_prices: dict) -> pd.DataFrame:
    print(f"[exp03] computing cross-protocol prices  (n_seeds={N_SEEDS}) ...")
    t0 = time.time()

    # Pre-load all models: {(proto, seed): (model, y_grid, param_keys)}
    # We need protocols A, B1, B2, B3, seeds 0..29
    protos_needed = {"A", "B1", "B2", "B3"}
    models = {}
    for proto in protos_needed:
        for seed in range(N_SEEDS):
            models[(proto, seed)] = load_model(seed, proto)
    print(f"  models loaded  ({time.time()-t0:.1f}s)")

    rows = []
    for pp_name, (proto1, proto2) in PROTOCOL_PAIRS.items():
        for seed_idx in range(N_SEEDS):
            m1, y1, pk1 = models[(proto1, seed_idx)]
            m2, y2, pk2 = models[(proto2, seed_idx)]
            for pname, (v1, v2) in PAIRS.items():
                p1, p2 = {**BASE, "v": v1}, {**BASE, "v": v2}
                V1t, V2t = true_prices[(pname, PAYOFF_NAMES[0])]  # placeholder
                for pn in PAYOFF_NAMES:
                    V1h = model_price(m1, p1, pn, y1, pk1)
                    V2h = model_price(m2, p2, pn, y2, pk2)
                    V1t, V2t = true_prices[(pname, pn)]
                    rows.append({
                        "protocol_pair": pp_name,
                        "proto1": proto1, "proto2": proto2,
                        "pair":        pname,
                        "payoff":      pn,
                        "seed_idx":    seed_idx,
                        "V1_hat":      V1h,
                        "V2_hat":      V2h,
                        "delta_hat":   V1h - V2h,
                        "V1_true":     V1t,
                        "V2_true":     V2t,
                        "delta_true":  V1t - V2t,
                    })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT / "prices_crossprotocol.csv", index=False)
    print(f"  {len(df)} rows saved  ({time.time()-t0:.1f}s)")
    return df


# ── Per-(protocol_pair, pair, payoff) stats ───────────────────────────────

def group_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pp, pair, payoff), g in df.groupby(["protocol_pair", "pair", "payoff"]):
        V1h = g["V1_hat"].values
        V2h = g["V2_hat"].values
        dh  = g["delta_hat"].values
        V1t = g["V1_true"].iloc[0]
        V2t = g["V2_true"].iloc[0]
        dt  = V1t - V2t

        b1  = V1h.mean() - V1t
        b2  = V2h.mean() - V2t
        bd  = b1 - b2
        mu  = dt + bd

        s1  = V1h.std(ddof=1)
        s2  = V2h.std(ddof=1)
        rho = np.corrcoef(V1h, V2h)[0, 1] if (s1 > 1e-15 and s2 > 1e-15) else 0.0
        sig = dh.std(ddof=1)
        s   = mu / sig if sig > 1e-15 else 0.0

        bias_emp  = np.abs(dh).mean() - abs(dt)
        bias_pred = E_abs(mu, sig) - abs(dt)
        ratio     = bias_emp / bias_pred if abs(bias_pred) > 1e-15 else np.nan

        rows.append({
            "protocol_pair": pp,
            "pair": pair, "payoff": payoff,
            "dv": DV_MAP[pair],
            "delta_true": dt,
            "b1": b1, "b2": b2, "b_delta": bd, "mu": mu,
            "sigma_1": s1, "sigma_2": s2, "rho": rho,
            "sigma_delta": sig, "s": s,
            "bias_emp": bias_emp, "bias_pred": bias_pred, "ratio": ratio,
            "jensen": jensen_component(mu, sig),
            "systematic": abs(mu) - abs(dt),
        })
    return pd.DataFrame(rows)


# ── Part 3: correlation by protocol ──────────────────────────────────────

def part3_correlation(stats: pd.DataFrame):
    # Reorder for readability: protocol_pair as outer index
    corr = stats[["protocol_pair", "pair", "payoff", "rho",
                  "sigma_1", "sigma_2", "sigma_delta", "mu", "s"]].copy()
    corr = corr.sort_values(["pair", "payoff", "protocol_pair"])
    corr.to_csv(OUTPUT / "correlation_by_protocol.csv", index=False)

    print("\n── Part 3: Correlation by protocol ─────────────────────────────")
    pd.set_option("display.float_format", "{:.4e}".format)
    pd.set_option("display.width", 200)
    print(corr.to_string(index=False))

    print("\nρ summary: AA vs cross-protocol (ATM, averaged over pairs):")
    for pp in PROTOCOL_PAIRS:
        sub = stats[(stats["protocol_pair"] == pp) & (stats["payoff"] == "ATM")]
        print(f"  {pp:6s}  mean_ρ(ATM)={sub['rho'].mean():.4f}  "
              f"range=[{sub['rho'].min():.4f}, {sub['rho'].max():.4f}]")


# ── Part 4 Test A: cross-protocol Theorem 1' ──────────────────────────────

def part4_test_a(stats: pd.DataFrame) -> pd.DataFrame:
    cols = ["protocol_pair", "pair", "payoff", "rho", "mu", "sigma_delta",
            "s", "bias_emp", "bias_pred", "ratio"]
    ta = stats[cols].copy()
    ta.to_csv(OUTPUT / "jensen_isolation_testA.csv", index=False)

    print("\n── Part 4 Test A: cross-protocol Theorem 1' ratio ───────────────")
    r = ta["ratio"].dropna()
    in_range = ((r >= 0.7) & (r <= 1.3)).sum()
    print(f"  ratio mean={r.mean():.4f}  median={r.median():.4f}  "
          f"IQR=[{r.quantile(.25):.4f},{r.quantile(.75):.4f}]")
    print(f"  within [0.7, 1.3]: {in_range}/{len(r)} = {100*in_range/len(r):.0f}%")

    print("\n  Per protocol_pair:")
    for pp in PROTOCOL_PAIRS:
        sub = ta[ta["protocol_pair"] == pp]["ratio"].dropna()
        if len(sub):
            ok = ((sub >= 0.7) & (sub <= 1.3)).sum()
            print(f"    {pp:6s}  mean={sub.mean():.4f}  "
                  f"[0.7,1.3]: {ok}/{len(sub)}")
    return ta


# ── Part 4 Test B: Theorem 2 direct isolation ─────────────────────────────

def part4_test_b(stats: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pair, payoff) in [(p, pn) for p in PAIR_NAMES for pn in PAYOFF_NAMES]:
        sub = stats[(stats["pair"] == pair) & (stats["payoff"] == payoff)]
        if len(sub) == 0:
            continue

        aa  = sub[sub["protocol_pair"] == "AA"]
        ab1 = sub[sub["protocol_pair"] == "AB1"]
        ab2 = sub[sub["protocol_pair"] == "AB2"]
        ab3 = sub[sub["protocol_pair"] == "AB3"]

        if len(aa) == 0 or len(ab1) == 0:
            continue

        aa_row  = aa.iloc[0]
        ab1_row = ab1.iloc[0]

        dt = float(aa_row["delta_true"])

        # Use average σ_1, σ_2 across AA and AB1 as "common" σ
        sigma1_common = 0.5 * (float(aa_row["sigma_1"]) + float(ab1_row["sigma_1"]))
        sigma2_common = 0.5 * (float(aa_row["sigma_2"]) + float(ab1_row["sigma_2"]))
        mu_common     = 0.5 * (float(aa_row["mu"])      + float(ab1_row["mu"]))

        def theory_at_rho(rho_val):
            sig = np.sqrt(max(sigma1_common**2 + sigma2_common**2
                              - 2*rho_val*sigma1_common*sigma2_common, 0.0))
            return E_abs(mu_common, sig) - abs(dt)

        rho_aa  = float(aa_row["rho"])
        rho_ab1 = float(ab1_row["rho"])

        bias_emp_aa  = float(aa_row["bias_emp"])
        bias_emp_ab1 = float(ab1_row["bias_emp"])
        bias_th_aa   = theory_at_rho(rho_aa)
        bias_th_ab1  = theory_at_rho(rho_ab1)

        delta_emp  = bias_emp_aa  - bias_emp_ab1
        delta_th   = bias_th_aa  - bias_th_ab1
        ratio_test = delta_emp / delta_th if abs(delta_th) > 1e-15 else np.nan

        rows.append({
            "pair": pair, "payoff": payoff, "delta_true": dt,
            "rho_AA": rho_aa, "rho_AB1": rho_ab1,
            "delta_rho": rho_aa - rho_ab1,
            "bias_emp_AA": bias_emp_aa, "bias_emp_AB1": bias_emp_ab1,
            "bias_th_AA":  bias_th_aa,  "bias_th_AB1":  bias_th_ab1,
            "delta_bias_emp": delta_emp, "delta_bias_th": delta_th,
            "ratio_testB": ratio_test,
            "sigma1_common": sigma1_common, "sigma2_common": sigma2_common,
            "mu_common": mu_common,
        })

    tb = pd.DataFrame(rows)
    tb.to_csv(OUTPUT / "jensen_isolation_testB.csv", index=False)

    print("\n── Part 4 Test B: Theorem 2 direct isolation (AA vs AB1) ───────")
    print("  Δρ = ρ_AA - ρ_AB1  |  Δbias_emp vs Δbias_theory(ρ)")
    print(tb[["pair","payoff","delta_rho",
              "delta_bias_emp","delta_bias_th","ratio_testB"]].to_string(index=False))

    r = tb["ratio_testB"].dropna()
    if len(r):
        in_range = ((r >= 0.5) & (r <= 2.0)).sum()
        print(f"\n  ratio_testB: mean={r.mean():.4f}  median={r.median():.4f}  "
              f"within [0.5, 2.0]: {in_range}/{len(r)}")
    return tb


# ── Part 4 Plot: bias vs ρ across protocol pairs ──────────────────────────

def part4_plot(stats: pd.DataFrame):
    pay_colors = {pn: plt.cm.Set1(i) for i, pn in enumerate(PAYOFF_NAMES)}

    # One subplot per pair (5 pairs). For each: plot bias_emp vs ρ + theory curve.
    fig, axes = plt.subplots(1, len(PAIR_NAMES), figsize=(4*len(PAIR_NAMES), 4),
                             sharey=False)
    rho_fine = np.linspace(0.0, 1.0, 200)

    for ax, pname in zip(axes, PAIR_NAMES):
        sub_p = stats[stats["pair"] == pname]

        for pn in PAYOFF_NAMES:
            sub = sub_p[sub_p["payoff"] == pn].sort_values("rho")
            if len(sub) == 0:
                continue
            c = pay_colors[pn]
            ax.scatter(sub["rho"], sub["bias_emp"],
                       color=c, s=70, zorder=5, label=pn if pname == PAIR_NAMES[0] else "_")
            # Label each point with protocol pair
            for _, row in sub.iterrows():
                ax.annotate(row["protocol_pair"],
                            (row["rho"], row["bias_emp"]),
                            fontsize=5, ha="left", va="bottom")

        # Theory curve: use AA σ_1, σ_2, μ; vary ρ → bias_theory(ρ)
        aa_row = sub_p[(sub_p["protocol_pair"] == "AA") & (sub_p["payoff"] == "ATM")]
        if len(aa_row):
            r = aa_row.iloc[0]
            s1, s2, mu = float(r["sigma_1"]), float(r["sigma_2"]), float(r["mu"])
            dt = float(r["delta_true"])
            curve = []
            for rho in rho_fine:
                sig = np.sqrt(max(s1**2 + s2**2 - 2*rho*s1*s2, 0.0))
                curve.append(E_abs(mu, sig) - abs(dt))
            ax.plot(rho_fine, curve, "k--", lw=1.5, alpha=0.6,
                    label="Theory Th.2" if pname == PAIR_NAMES[0] else "_")

        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("ρ")
        ax.set_ylabel("bias_emp")
        ax.set_title(f"Pair {pname} (Δv={DV_MAP[pname]})")
        if pname == PAIR_NAMES[0]:
            ax.legend(fontsize=6, ncol=2)

    fig.suptitle("Theorem 2 isolation: bias vs ρ across protocol pairs", fontsize=11)
    fig.tight_layout()
    path = OUTPUT / "bias_vs_rho_isolation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {path.name}")


# ── Part 5: α robustness ──────────────────────────────────────────────────

def part5_alpha_robustness(stats: pd.DataFrame) -> pd.DataFrame:
    # Within-protocol pairs: AA (Protocol A×A), B1B1, B2B2
    within_map = {
        "A":  "AA",
        "B1": "B1B1",
        "B2": "B2B2",
    }
    rows = []
    for proto_name, pp_name in within_map.items():
        sub_pp = stats[stats["protocol_pair"] == pp_name]
        if len(sub_pp) == 0:
            continue
        for pn in PAYOFF_NAMES:
            sub = sub_pp[sub_pp["payoff"] == pn].sort_values("dv")
            dvs  = sub["dv"].values
            omr  = (1 - sub["rho"].values)
            mask = (dvs > 0) & (omr > 0)
            if mask.sum() >= 2:
                slope, ic = np.polyfit(np.log(dvs[mask]), np.log(omr[mask]), 1)
            else:
                slope = np.nan
            rows.append({
                "protocol": proto_name,
                "protocol_pair": pp_name,
                "payoff": pn,
                "alpha": slope,
            })

    alpha_df = pd.DataFrame(rows)
    # Pivot for readability
    pivot = alpha_df.pivot_table(
        index="protocol", columns="payoff", values="alpha"
    ).reindex(["A", "B1", "B2"])
    pivot["mean_alpha"] = pivot[PAYOFF_NAMES].mean(axis=1)
    alpha_df.to_csv(OUTPUT / "alpha_robustness.csv", index=False)

    print("\n── Part 5: α robustness across protocols ────────────────────────")
    pd.set_option("display.float_format", "{:.3f}".format)
    print(pivot.to_string())
    return alpha_df


# ── Part 6: Summary ───────────────────────────────────────────────────────

def part6_summary(stats, ta, tb, alpha_df, true_prices):
    print("\n" + "="*80)
    print("EXPERIMENT 03 SUMMARY (Q1–Q6)")
    print("="*80)

    # Q1: Protocol B training losses
    print("\nQ1. Protocol B training losses (from checkpoints):")
    for proto in ["B1", "B2", "B3"]:
        save_dir = SAVE_B / f"protocol_{proto}"
        losses = []
        for s in range(N_SEEDS):
            p = save_dir / f"seed_{s:02d}.pt"
            if p.exists():
                ck = torch.load(p, map_location="cpu", weights_only=False)
                losses.append(ck.get("final_loss", np.nan))
        if losses:
            losses = np.array(losses)
            print(f"  {proto}: n={len(losses)}  mean={losses.mean():.4e}  "
                  f"std={losses.std():.4e}  min={losses.min():.4e}  max={losses.max():.4e}")

    # Compare to Protocol A (first 30 seeds)
    a_losses = []
    for s in range(N_SEEDS):
        p = SAVE_A / f"seed_{s:02d}.pt"
        if p.exists():
            ck = torch.load(p, map_location="cpu", weights_only=False)
            a_losses.append(ck.get("final_loss", np.nan))
    if a_losses:
        a_losses = np.array(a_losses)
        print(f"  A:  n={len(a_losses)}  mean={a_losses.mean():.4e}  "
              f"std={a_losses.std():.4e}")

    # Q2: ρ drop
    print("\nQ2. ρ measurement (ATM, mean over pairs A–E):")
    for pp in PROTOCOL_PAIRS:
        sub = stats[(stats["protocol_pair"] == pp) & (stats["payoff"] == "ATM")]
        if len(sub):
            print(f"  {pp:6s}  mean_ρ={sub['rho'].mean():.4f}  "
                  f"min={sub['rho'].min():.4f}  max={sub['rho'].max():.4f}")

    cross = [pp for pp in ["AB1", "AB2", "AB3"]]
    best_drop = None
    best_pp   = None
    aa_mean   = stats[(stats["protocol_pair"] == "AA") & (stats["payoff"] == "ATM")]["rho"].mean()
    for pp in cross:
        sub = stats[(stats["protocol_pair"] == pp) & (stats["payoff"] == "ATM")]
        if len(sub):
            drop = aa_mean - sub["rho"].mean()
            if best_drop is None or drop > best_drop:
                best_drop = drop
                best_pp   = pp
    print(f"  Most decorrelated: {best_pp} (Δρ = {best_drop:.4f} vs AA)")

    # Q3: Test A
    r = ta["ratio"].dropna()
    in_range = ((r >= 0.7) & (r <= 1.3)).sum()
    print(f"\nQ3. Test A (Theorem 1' cross-protocol): "
          f"ratio mean={r.mean():.4f}  within [0.7,1.3]: {in_range}/{len(r)}")

    # Q4: Test B
    r = tb["ratio_testB"].dropna()
    delta_rho_mean = tb["delta_rho"].mean()
    in_range_b = ((r >= 0.5) & (r <= 2.0)).sum() if len(r) else 0
    print(f"\nQ4. Test B (Theorem 2 direct): "
          f"mean Δρ(AA-AB1)={delta_rho_mean:.4f}  "
          f"ratio mean={r.mean():.4f}  within [0.5,2.0]: {in_range_b}/{len(r)}")
    if abs(delta_rho_mean) < 0.05:
        print("  WARNING: Δρ < 0.05 — insufficient ρ spread for strong test.")
        verdict_t2 = "WEAK (ρ spread too small)"
    elif in_range_b / max(len(r), 1) >= 0.7:
        verdict_t2 = "CONFIRMED"
    else:
        verdict_t2 = "PARTIAL"
    print(f"  Theorem 2 direct verdict: {verdict_t2}")

    # Q5: α robustness
    pivot = alpha_df.pivot_table(index="protocol", columns="payoff", values="alpha")
    pivot["mean_alpha"] = pivot[PAYOFF_NAMES].mean(axis=1)
    print(f"\nQ5. α robustness:")
    for proto in pivot.index:
        row = pivot.loc[proto]
        a_ref = 1.525
        diff  = abs(float(row["mean_alpha"]) - a_ref) / a_ref * 100
        print(f"  {proto}:  mean_α={row['mean_alpha']:.3f}  "
              f"({diff:.1f}% from A's 1.525)")
    alphas_all = pivot["mean_alpha"].values
    spread = np.nanmax(alphas_all) - np.nanmin(alphas_all)
    universality = "universal" if spread < 0.3 else ("protocol-specific" if spread > 0.8 else "partially robust")
    print(f"  α spread across protocols: {spread:.3f}  → {universality}")

    # Q6: Overall
    print(f"\nQ6. Overall:")
    print(f"  Theorem 1' cross-protocol: {'VALIDATED' if in_range/len(r) >= 0.8 else 'PARTIAL'}")
    print(f"  Theorem 2 direct Jensen isolation: {verdict_t2}")
    print(f"  Proposition 2' universality: {universality}")
    strong = (verdict_t2 == "CONFIRMED" and universality == "universal")
    mod    = (in_range/len(r) >= 0.8)
    submission = ("READY" if strong else "STRONG ENOUGH (add Exp 03 as §5 supplement)"
                  if mod else "NEEDS REVISION")
    print(f"  Paper submission readiness: {submission}")
    print("="*80)


# ── Main ──────────────────────────────────────────────────────────────────

def run():
    t0 = time.time()
    true_prices = compute_lewis()
    df    = compute_prices(true_prices)
    stats = group_stats(df)

    print(f"\n── prices_crossprotocol.csv rows by protocol_pair:")
    print(df.groupby("protocol_pair").size().to_string())

    part3_correlation(stats)
    ta = part4_test_a(stats)
    tb = part4_test_b(stats)
    part4_plot(stats)
    alpha_df = part5_alpha_robustness(stats)
    part6_summary(stats, ta, tb, alpha_df, true_prices)

    print(f"\n[exp03] total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    run()
