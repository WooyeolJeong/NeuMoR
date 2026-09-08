"""
Lewis formula ground truth for Heston marginal density and option prices.
"""

import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from TPDF.core.semi_closed_joint import lewis_marginal_x


# ── Marginal density ────────────────────────────────────────────────────

def lewis_pdf(params: dict, y_grid: np.ndarray) -> np.ndarray:
    """
    Heston log-price marginal PDF on y_grid via Lewis CF inversion.
    params must include: v, kappa, omega, xi, rho, T
    Returns mass-normalized pdf (N_y,).
    """
    f = lewis_marginal_x(y_grid, params=params)
    f = np.clip(f, 0.0, None)
    mass = np.trapz(f, x=y_grid)
    if mass > 0:
        f /= mass
    return f


# ── Payoff definitions (log-price space, S0=1) ──────────────────────────

PAYOFFS = {
    "ATM":      lambda y: np.maximum(np.exp(y) - 1.00, 0.0),
    "OTM_K110": lambda y: np.maximum(np.exp(y) - 1.10, 0.0),
    "OTM_K125": lambda y: np.maximum(np.exp(y) - 1.25, 0.0),
    "DIGITAL":  lambda y: (np.exp(y) > 1.10).astype(float),
}


# ── Price from density ───────────────────────────────────────────────────

def price_from_pdf(y_grid: np.ndarray, f: np.ndarray, payoff_name: str,
                   r: float = 0.0, T: float = 1.0) -> float:
    """
    Numerically integrate payoff * f over y_grid.
    For DIGITAL, integrates f only above the barrier to avoid discontinuity error.
    """
    disc = math.exp(-r * T)
    if payoff_name == "DIGITAL":
        # P(S_T > 1.10) = ∫_{log(1.10)}^{∞} f(y) dy  — no discontinuity in integrand
        barrier = math.log(1.10)
        mask = y_grid >= barrier
        if mask.sum() < 2:
            return 0.0
        return disc * float(np.trapz(f[mask], x=y_grid[mask]))
    g = PAYOFFS[payoff_name](y_grid)
    return disc * float(np.trapz(g * f, x=y_grid))


# ── Ground truth for a single (params, payoff) ──────────────────────────

def lewis_price(params: dict, payoff_name: str,
                y_min: float = -6.0, y_max: float = 6.0,
                n_y: int = 2000) -> float:
    """
    Lewis ground truth call / digital price.
    Uses fine grid independently of model grid.
    """
    y_grid = np.linspace(y_min, y_max, n_y)
    f = lewis_pdf(params, y_grid)
    return price_from_pdf(y_grid, f, payoff_name, T=params.get("T", 1.0))


# ── Sanity check: quadrature precision ──────────────────────────────────

def check_precision(params: dict, payoff_name: str, grids=(200, 400, 800, 1600)):
    """
    Print price at increasing quadrature density to verify convergence.
    """
    print(f"Precision check — {payoff_name}  "
          f"v={params['v']}, xi={params['xi']}, rho={params['rho']}")
    prices = []
    for n in grids:
        p = lewis_price(params, payoff_name, n_y=n)
        prices.append(p)
        print(f"  n_y={n:5d}  price={p:.8f}")
    diff = abs(prices[-1] - prices[-2])
    print(f"  |p(1600) - p(800)| = {diff:.2e}")
    return prices[-1]


if __name__ == "__main__":
    base = {"v": 0.10, "kappa": 1.0, "omega": 0.2, "xi": 0.3, "rho": -0.5, "T": 1.0}
    for pn in PAYOFFS:
        check_precision(base, pn)
        print()
