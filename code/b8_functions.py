"""
b8_functions.py — Shared utility functions for the vanilla-basis proxy.

Side-effect free: importing this module runs no I/O.

"""

import numpy as np

LOG_STRIKES_CALL = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.60, 0.70]
LOG_STRIKES_PUT  = [-0.05, -0.10, -0.15, -0.20, -0.30, -0.50, -0.60, -0.70]


def build_basis(y_grid: np.ndarray):
    cols, labels = [], []
    for k in LOG_STRIKES_CALL:
        payoff = np.maximum(np.exp(y_grid) - np.exp(k), 0.0)
        cols.append(payoff)
        labels.append(f"call_k={k:+.2f}")
    for k in LOG_STRIKES_PUT:
        payoff = np.maximum(np.exp(k) - np.exp(y_grid), 0.0)
        cols.append(payoff)
        labels.append(f"put_k={k:+.2f}")
    return np.column_stack(cols), labels


def proj_l2(G, g_star, dy):
    A = G.T @ G * dy
    b = G.T @ g_star * dy
    w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return G @ w, w


def proj_k(G, K, g_star, dy):
    KG = K @ G
    A  = G.T @ KG * dy**2
    b  = G.T @ (K @ g_star) * dy**2
    w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return G @ w, w


def proj_snr(G, K, d, dy):
    A = G.T @ (K @ G)
    b = G.T @ d
    w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return G @ w, w


def cos_l2(a, b, dy):
    dot   = float(a @ b * dy)
    na    = float(np.sqrt(a @ a * dy))
    nb    = float(np.sqrt(b @ b * dy))
    denom = na * nb
    return dot / denom if denom > 1e-60 else 0.0


def cos_K_norm(a, K, b, dy):
    dot   = float(a @ (K @ b) * dy)
    na    = float(np.sqrt(a @ (K @ a) * dy))
    nb    = float(np.sqrt(b @ (K @ b) * dy))
    denom = na * nb
    return dot / denom if denom > 1e-60 else 0.0


def residual_l2(g_star, g_proj, dy):
    diff = g_star - g_proj
    return float(np.sqrt(diff @ diff * dy))


def residual_K(g_star, g_proj, K, dy):
    diff = g_star - g_proj
    return float(np.sqrt(diff @ (K @ diff) * dy))
