
import numpy as np
from scipy.integrate import quad
from scipy.fft import fft
from typing import Tuple

def kou_psi(u: complex, params: tuple, r: float = 0.0, q: float = 0.0) -> complex:
    sigma, lam_J, p, eta1, eta2 = params
    kappa_J = p * eta1 / (eta1 - 1) + (1 - p) * eta2 / (eta2 + 1) - 1
    drift = r - q - 0.5 * sigma**2 - lam_J * kappa_J
    jump  = lam_J * (p * eta1 / (eta1 - 1j * u) +
                     (1 - p) * eta2 / (eta2 + 1j * u) - 1)
    return 1j * u * drift - 0.5 * sigma**2 * u**2 + jump

def kou_char_fn(u: complex, t: float, params: tuple,
                r: float = 0.0, q: float = 0.0) -> complex:
    return np.exp(t * kou_psi(u, params, r, q))

def kou_call_price(K: float, T: float, S0: float, params: tuple,
                   r: float = 0.0, q: float = 0.0,
                   alpha: float = 1.5,
                   upper_limit: float = 500.0, limit: int = 500) -> float:
    k = np.log(K / S0)

    def integrand(omega: float) -> float:
        phi = kou_char_fn(omega - (alpha + 1) * 1j, T, params, r, q)
        denom = (alpha + 1j * omega) * (alpha + 1 + 1j * omega)
        f = np.exp(-r * T) * phi / denom
        return np.real(np.exp(-1j * omega * k) * f)

    integral, _ = quad(integrand, 0, upper_limit,
                       limit=limit, epsabs=1e-10, epsrel=1e-8)

    call = S0 * np.exp(-alpha * k) / np.pi * integral
    return max(call, 0.0)

def kou_put_price(K: float, T: float, S0: float, params: tuple,
                  r: float = 0.0, q: float = 0.0) -> float:
    call = kou_call_price(K, T, S0, params, r, q)
    return call - S0 * np.exp(-q * T) + K * np.exp(-r * T)

def kou_log_density(y_grid: np.ndarray, T: float, params: tuple,
                    r: float = 0.0, q: float = 0.0,
                    N_fft: int = 4096) -> np.ndarray:
    dy = y_grid[1] - y_grid[0]

    def _density_at(y: float) -> float:
        def integrand_real(u: float) -> float:
            phi = kou_char_fn(u, T, params, r, q)
            return np.real(np.exp(-1j * u * y) * phi)
        val, _ = quad(integrand_real, -np.inf, np.inf,
                      limit=300, epsabs=1e-10, epsrel=1e-8,
                      points=[0.0])
        return val / (2 * np.pi)

    N = len(y_grid)

    du = 2 * np.pi / (N * dy)
    u_grid = np.arange(N) * du - (N // 2) * du

    phi_vals = np.array([kou_char_fn(u, T, params, r, q) for u in u_grid])

    pdf = np.zeros(N)
    for n, y in enumerate(y_grid):
        integrand_vals = np.real(np.exp(-1j * u_grid * y) * phi_vals)
        pdf[n] = np.trapz(integrand_vals, u_grid) / (2 * np.pi)

    pdf = np.maximum(pdf, 0.0)
    mass = np.trapz(pdf, y_grid)
    if mass > 0:
        pdf /= mass
    return pdf

def kou_log_density_fft(y_grid: np.ndarray, T: float, params: tuple,
                         r: float = 0.0, q: float = 0.0,
                         N_fft: int = 8192) -> np.ndarray:
    from scipy.interpolate import interp1d

    alpha = 1.5
    eta   = 0.25
    lam   = 2 * np.pi / (N_fft * eta)

    u     = np.arange(N_fft) * eta
    k_vec = -N_fft * lam / 2 + np.arange(N_fft) * lam

    phi_mod = (np.exp(-r * T) *
               np.array([kou_char_fn(u[j] - (alpha + 1) * 1j, T, params, r, q)
                          for j in range(N_fft)]) /
               (alpha**2 + alpha - u**2 + 1j * (2 * alpha + 1) * u))

    x = np.exp(1j * u * N_fft * lam / 2) * phi_mod * eta
    x[0] *= 0.5
    call_fft = np.real(fft(x)) * np.exp(-alpha * k_vec) / np.pi

    return kou_log_density(y_grid, T, params, r, q)

def validate_kou_2002():
    S0    = 100.0
    r     = 0.05
    q     = 0.0
    T     = 1.0
    params = (0.16, 1.0, 0.4, 10.0, 5.0)

    K_atm = 100.0
    call_atm = kou_call_price(K_atm, T, S0, params, r, q)

    call_otm = kou_call_price(110.0, T, S0, params, r, q)

    call_dotm = kou_call_price(120.0, T, S0, params, r, q)

    put_atm = kou_put_price(K_atm, T, S0, params, r, q)

    from scipy.stats import norm as _norm
    d1 = (np.log(S0 / K_atm) + (r + 0.5 * 0.16**2) * T) / (0.16 * np.sqrt(T))
    d2 = d1 - 0.16 * np.sqrt(T)
    bs_call = S0 * _norm.cdf(d1) - K_atm * np.exp(-r * T) * _norm.cdf(d2)

    results = {
        "call_ATM_K100":  call_atm,
        "call_OTM_K110":  call_otm,
        "call_DOTM_K120": call_dotm,
        "put_ATM_K100":   put_atm,
        "BS_call_ATM":    bs_call,
        "params": params,
    }
    return results

if __name__ == "__main__":
    print("=== Kou (2002) Characteristic Function Validation ===\n")

    res = validate_kou_2002()
    print(f"Kou call  ATM  (K=100): {res['call_ATM_K100']:.4f}  (expected ~10.03)")
    print(f"Kou call  OTM  (K=110): {res['call_OTM_K110']:.4f}  (expected ~6.5)")
    print(f"Kou call  DOTM (K=120): {res['call_DOTM_K120']:.4f}  (expected ~4.0)")
    print(f"Kou put   ATM  (K=100): {res['put_ATM_K100']:.4f}  (expected ~5.28)")
    print(f"BS  call  ATM  (K=100): {res['BS_call_ATM']:.4f}  (BS benchmark)")

    S0, K, r, T = 100, 100, 0.05, 1.0
    pcp = res['call_ATM_K100'] - res['put_ATM_K100'] - S0 + K * np.exp(-r * T)
    print(f"\nPut-call parity residual: {pcp:.2e}  (should be ~0)")
