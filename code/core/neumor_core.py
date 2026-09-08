
import numpy as np
from pathlib import Path
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[2]
DECOMP_ROOT = ROOT / "NeuMoR" / "output" / "day4"

def build_k_eta(pdfs_l1: np.ndarray, pdfs_l2: np.ndarray) -> np.ndarray:
    eps1 = pdfs_l1 - pdfs_l1.mean(axis=0, keepdims=True)
    eps2 = pdfs_l2 - pdfs_l2.mean(axis=0, keepdims=True)
    eta  = eps1 - eps2
    K    = (eta.T @ eta) / len(eta)
    return ((K + K.T) / 2).astype(np.float64)

def build_k_11(pdfs: np.ndarray) -> np.ndarray:
    eps = pdfs - pdfs.mean(axis=0, keepdims=True)
    K   = (eps.T @ eps) / len(eps)
    return ((K + K.T) / 2).astype(np.float64)

def snr_squared(g: np.ndarray, d: np.ndarray, K: np.ndarray, dy: float) -> float:
    signal2 = float((g @ d * dy) ** 2)
    noise2  = float(g @ (K @ g) * dy**2)
    return signal2 / (max(abs(noise2), 1e-80))

def snr_linear(g: np.ndarray, d: np.ndarray, K: np.ndarray, dy: float) -> float:
    return float(np.sqrt(snr_squared(g, d, K, dy)))

def compute_g_star(
    K: np.ndarray,
    d: np.ndarray,
    dy: float,
    tau_factor: float = 0.01,
):
    ev, evec = eigh(K)
    ev   = np.real(ev[::-1]).clip(0)
    evec = np.real(evec[:, ::-1])
    lam_max = float(ev[0])
    tau     = tau_factor * lam_max
    coeff   = (evec.T @ d * dy) / (ev + tau)
    g_star  = evec @ coeff
    return g_star.astype(np.float64), tau, lam_max, ev, evec

def compute_d_hat(pdfs_l1: np.ndarray, pdfs_l2: np.ndarray) -> np.ndarray:
    return (pdfs_l1.mean(axis=0) - pdfs_l2.mean(axis=0)).astype(np.float64)

def compute_d_analytical(
    model_name: str,
    pair: str,
    decomp_dir: Path = None,
) -> np.ndarray:
    if decomp_dir is None:
        decomp_dir = DECOMP_ROOT / model_name / "kernel"
    data = np.load(decomp_dir / f"decomp_pair{pair}.npz")
    return (data["tp1"].astype(np.float64) - data["tp2"].astype(np.float64))

def load_y_grid(model_name: str, pair: str = "A", decomp_dir: Path = None) -> np.ndarray:
    if decomp_dir is None:
        decomp_dir = DECOMP_ROOT / model_name / "kernel"
    return np.load(decomp_dir / f"decomp_pair{pair}.npz")["y_grid"].astype(np.float64)

def split_cal_test(
    pdfs_l1: np.ndarray,
    pdfs_l2: np.ndarray,
    n_cal: int,
    test_range: tuple = (950, 1000),
):
    lo, hi = test_range
    return (
        pdfs_l1[:n_cal],
        pdfs_l1[lo:hi],
        pdfs_l2[:n_cal],
        pdfs_l2[lo:hi],
    )

def effective_rank(K: np.ndarray) -> float:
    ev = np.linalg.eigvalsh(K).clip(0)
    s1, s2 = ev.sum(), (ev**2).sum()
    return float(s1**2 / s2) if s2 > 0 else 0.0

def compute_gain(
    g_star: np.ndarray,
    g_baseline: np.ndarray,
    d: np.ndarray,
    K_test: np.ndarray,
    dy: float,
) -> dict:
    s2_star = snr_squared(g_star,     d, K_test, dy)
    s2_base = snr_squared(g_baseline, d, K_test, dy)
    gain_sq = s2_star / max(s2_base, 1e-80)
    return {
        "snr2_star":     s2_star,
        "snr2_baseline": s2_base,
        "gain_sq":       float(gain_sq),
        "gain_linear":   float(np.sqrt(gain_sq)),
    }

def atm_call(y_grid: np.ndarray) -> np.ndarray:
    return np.maximum(np.exp(y_grid) - 1.0, 0.0)
