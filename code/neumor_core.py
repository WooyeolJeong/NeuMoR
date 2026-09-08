"""
neumor_core.py — Paper 1 Unified Formulation (Phase A → Phase B)
=================================================================

All Phase B experiments (B.2-B.8) import from this module.
Implements the unified formulation confirmed 2026-04-24.
See: NeuMoR/paper/paper1_unified_formulation.md

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UNIFIED FORMULATION SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

K (noise covariance):
    K_eta = Cov(η, η),  η^(s) = ε₁^(s) - ε₂^(s)
    ε^(s) = p̂^(s) - mean_s(p̂^(s))   ← sample mean centering

SNR²:
    SNR²(g) = (<g, d> dy)² / (<g, K_eta g> dy²)
    τ EXCLUDED from SNR evaluation (only used in g* derivation)

g*:
    g* = (K_eta + τI)⁻¹ d,  τ = 0.01 * λ_max(K_eta)
    Computed via eigendecomposition for numerical stability.

d:
    d_hat     = mean(p̂_λ1) - mean(p̂_λ2)          [NN ensemble mean]
    d_ana     = tp1 - tp2 from decomp_pair*.npz     [analytical/fixed MC]
    Main results use d_hat; d_ana used for sanity check.

Split:
    K_cal: seeds 0..N_cal-1,  K_test: seeds 950..999 (Heston) / 150..199 (Kou)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
from pathlib import Path
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[2]
DECOMP_ROOT = ROOT / "NeuMoR" / "output" / "day4"


# ── Noise covariance ──────────────────────────────────────────────────────────

def build_k_eta(pdfs_l1: np.ndarray, pdfs_l2: np.ndarray) -> np.ndarray:
    """K_eta = Cov(η, η),  η = ε₁ - ε₂.

    Formula:
        ε₁ = pdfs_l1 - mean(pdfs_l1, axis=0)   [sample mean centering]
        ε₂ = pdfs_l2 - mean(pdfs_l2, axis=0)
        η  = ε₁ - ε₂
        K_eta = (η^T η) / N                     [biased estimator]

    Args:
        pdfs_l1: (N, n_y) NN density outputs at λ₁, same seeds as pdfs_l2.
        pdfs_l2: (N, n_y) NN density outputs at λ₂.

    Returns:
        (n_y, n_y) symmetric PSD matrix. Symmetrized as (K + K^T)/2.
    """
    eps1 = pdfs_l1 - pdfs_l1.mean(axis=0, keepdims=True)
    eps2 = pdfs_l2 - pdfs_l2.mean(axis=0, keepdims=True)
    eta  = eps1 - eps2
    K    = (eta.T @ eta) / len(eta)
    return ((K + K.T) / 2).astype(np.float64)


def build_k_11(pdfs: np.ndarray) -> np.ndarray:
    """K_11 = Var(ε₁) — single-parameter variance.

    LEGACY: Kept for comparison with pre-unification (Phase 0) results.
    New experiments should use build_k_eta.

    Formula:
        ε = pdfs - mean(pdfs, axis=0)
        K_11 = (ε^T ε) / N

    Args:
        pdfs: (N, n_y) NN density outputs at one parameter point.

    Returns:
        (n_y, n_y) symmetric PSD matrix.
    """
    eps = pdfs - pdfs.mean(axis=0, keepdims=True)
    K   = (eps.T @ eps) / len(eps)
    return ((K + K.T) / 2).astype(np.float64)


# ── SNR ───────────────────────────────────────────────────────────────────────

def snr_squared(g: np.ndarray, d: np.ndarray, K: np.ndarray, dy: float) -> float:
    """Theoretical SNR² (no τ regularization in denominator).

    Formula:
        SNR²(g) = (<g, d> · dy)² / (<g, K g> · dy²)

    Note: τ is intentionally excluded. It is only a numerical device for
    computing g* and must not penalize payoffs with large L2 norm.

    Args:
        g:  (n_y,) payoff vector.
        d:  (n_y,) signal (density difference).
        K:  (n_y, n_y) noise covariance (K_eta recommended).
        dy: grid spacing scalar.

    Returns:
        float ≥ 0.
    """
    signal2 = float((g @ d * dy) ** 2)
    noise2  = float(g @ (K @ g) * dy**2)
    return signal2 / (max(abs(noise2), 1e-80))


def snr_linear(g: np.ndarray, d: np.ndarray, K: np.ndarray, dy: float) -> float:
    """Linear SNR = sqrt(SNR²).

    See snr_squared for formula and args.
    """
    return float(np.sqrt(snr_squared(g, d, K, dy)))


# ── g* ────────────────────────────────────────────────────────────────────────

def compute_g_star(
    K: np.ndarray,
    d: np.ndarray,
    dy: float,
    tau_factor: float = 0.01,
):
    """Optimal payoff g* via eigendecomposition.

    Formula:
        g* = Σᵢ <φᵢ, d> / (λᵢ + τ) · φᵢ
        τ  = tau_factor · λ_max(K)

    Eigendecomposition used instead of np.linalg.solve for numerical
    stability when K is near rank-deficient (e.g., Kou K_eta with r_eff≈1.5).

    Args:
        K:           (n_y, n_y) noise covariance.
        d:           (n_y,) signal.
        dy:          grid spacing.
        tau_factor:  τ = tau_factor · λ_max. Default 0.01 per §4.4.

    Returns:
        g_star:  (n_y,) optimal payoff.
        tau:     float, regularization value used.
        lam_max: float, largest eigenvalue.
        eigvals: (n_y,) eigenvalues descending, clipped to ≥ 0.
        eigvecs: (n_y, n_y) eigenvectors (columns), matching eigvals order.
    """
    ev, evec = eigh(K)
    ev   = np.real(ev[::-1]).clip(0)
    evec = np.real(evec[:, ::-1])
    lam_max = float(ev[0])
    tau     = tau_factor * lam_max
    coeff   = (evec.T @ d * dy) / (ev + tau)
    g_star  = evec @ coeff
    return g_star.astype(np.float64), tau, lam_max, ev, evec


# ── d ─────────────────────────────────────────────────────────────────────────

def compute_d_hat(pdfs_l1: np.ndarray, pdfs_l2: np.ndarray) -> np.ndarray:
    """NN ensemble mean difference: d̂ = mean(p̂_λ1) - mean(p̂_λ2).

    Args:
        pdfs_l1: (N, n_y) NN outputs at λ₁.
        pdfs_l2: (N, n_y) NN outputs at λ₂.

    Returns:
        (n_y,) array.
    """
    return (pdfs_l1.mean(axis=0) - pdfs_l2.mean(axis=0)).astype(np.float64)


def compute_d_analytical(
    model_name: str,
    pair: str,
    decomp_dir: Path = None,
) -> np.ndarray:
    """True density difference from stored decomp file.

    Loads tp1 and tp2 from decomp_pair{pair}.npz. These are precomputed
    true densities (Heston: Lewis formula; Kou: fixed 50k MC snapshot).
    Loading from file ensures reproducibility — no re-computation.

    Args:
        model_name: "heston", "kou", "gbm", "bates", or "rough_heston".
        pair:       "A", "B", "C", or "D".
        decomp_dir: Path to directory containing decomp_pair*.npz files.
                    Default: NeuMoR/output/day4/{model_name}/kernel/.

    Returns:
        (n_y,) array: tp1 - tp2.
    """
    if decomp_dir is None:
        decomp_dir = DECOMP_ROOT / model_name / "kernel"
    data = np.load(decomp_dir / f"decomp_pair{pair}.npz")
    return (data["tp1"].astype(np.float64) - data["tp2"].astype(np.float64))


def load_y_grid(model_name: str, pair: str = "A", decomp_dir: Path = None) -> np.ndarray:
    """Load y_grid from decomp file.

    Returns:
        (n_y,) array.
    """
    if decomp_dir is None:
        decomp_dir = DECOMP_ROOT / model_name / "kernel"
    return np.load(decomp_dir / f"decomp_pair{pair}.npz")["y_grid"].astype(np.float64)


# ── Split ─────────────────────────────────────────────────────────────────────

def split_cal_test(
    pdfs_l1: np.ndarray,
    pdfs_l2: np.ndarray,
    n_cal: int,
    test_range: tuple = (950, 1000),
):
    """Split seed arrays into calibration and test subsets.

    Cal:  seeds 0..n_cal-1
    Test: seeds test_range[0]..test_range[1]-1

    For Heston: test_range=(950, 1000).
    For Kou:    test_range=(150, 200).

    Args:
        pdfs_l1:    (N_total, n_y) full seed array at λ₁.
        pdfs_l2:    (N_total, n_y) full seed array at λ₂.
        n_cal:      number of calibration seeds.
        test_range: (lo, hi) slice for test seeds.

    Returns:
        l1_cal, l1_tst, l2_cal, l2_tst — each (N_cal or N_test, n_y).
    """
    lo, hi = test_range
    return (
        pdfs_l1[:n_cal],
        pdfs_l1[lo:hi],
        pdfs_l2[:n_cal],
        pdfs_l2[lo:hi],
    )


# ── Kernel statistics ─────────────────────────────────────────────────────────

def effective_rank(K: np.ndarray) -> float:
    """Effective rank r_eff = (Σλᵢ)² / Σλᵢ².

    Measures participation ratio. r_eff=1 means rank-1 (all energy in one mode).
    r_eff=n_y means fully isotropic.

    Args:
        K: (n_y, n_y) PSD matrix.

    Returns:
        float ≥ 1.
    """
    ev = np.linalg.eigvalsh(K).clip(0)
    s1, s2 = ev.sum(), (ev**2).sum()
    return float(s1**2 / s2) if s2 > 0 else 0.0


# ── Gain ─────────────────────────────────────────────────────────────────────

def compute_gain(
    g_star: np.ndarray,
    g_baseline: np.ndarray,
    d: np.ndarray,
    K_test: np.ndarray,
    dy: float,
) -> dict:
    """SNR² ratio and linear SNR ratio of g_star vs g_baseline.

    Args:
        g_star:     (n_y,) optimal payoff.
        g_baseline: (n_y,) baseline payoff (e.g., ATM call).
        d:          (n_y,) signal.
        K_test:     (n_y, n_y) held-out noise covariance.
        dy:         grid spacing.

    Returns:
        dict with:
            snr2_star:    SNR²(g_star)
            snr2_baseline: SNR²(g_baseline)
            gain_sq:      snr2_star / snr2_baseline
            gain_linear:  sqrt(gain_sq)
    """
    s2_star = snr_squared(g_star,     d, K_test, dy)
    s2_base = snr_squared(g_baseline, d, K_test, dy)
    gain_sq = s2_star / max(s2_base, 1e-80)
    return {
        "snr2_star":     s2_star,
        "snr2_baseline": s2_base,
        "gain_sq":       float(gain_sq),
        "gain_linear":   float(np.sqrt(gain_sq)),
    }


# ── ATM payoff helper ────────────────────────────────────────────────────────

def atm_call(y_grid: np.ndarray) -> np.ndarray:
    """ATM call payoff: max(exp(y) - 1, 0)."""
    return np.maximum(np.exp(y_grid) - 1.0, 0.0)
