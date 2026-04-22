"""
cov_utils.py — covariance estimators and minimum-variance portfolio solvers.

Estimators implemented:
  1. Sample covariance (baseline)
  2. Ledoit-Wolf analytical shrinkage (Oracle Approximating Shrinkage)
  3. Double-decay EWMA (separate vol and correlation decay)
  4. EWMA + Ledoit-Wolf shrinkage on the correlation matrix

Solvers:
  - Unconstrained minimum variance (closed-form with regularisation fallback)
  - Constrained minimum variance with per-weight cap (SLSQP)
"""

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Covariance estimators
# ---------------------------------------------------------------------------

def ledoit_wolf(R):
    """
    Analytical Ledoit-Wolf shrinkage toward a scaled identity target.
    Uses the Oracle Approximating Shrinkage (OAS) closed-form formula
    which is well-conditioned even when T < n.

    Reference: Chen, Wiesel, Eldar, Hero (2010) — Shrinkage Algorithms for
    MMSE Covariance Estimation. IEEE Trans. Signal Process.
    """
    T, n = R.shape
    S = np.cov(R, rowvar=False)          # sample cov
    trace_S  = np.trace(S)
    trace_S2 = np.trace(S @ S)
    mu = trace_S / n                     # scaled-identity target level

    # OAS shrinkage intensity
    rho_num = ((1 - 2 / n) * trace_S2 + trace_S ** 2)
    rho_den = (T + 1 - 2 / n) * (trace_S2 - trace_S ** 2 / n)
    rho = min(1.0, rho_num / rho_den) if rho_den > 0 else 1.0

    return (1 - rho) * S + rho * mu * np.eye(n)


def ewma_cov(R, lambda_corr, lambda_vol):
    """
    Double-decay EWMA covariance.
    Separate smoothing for volatility (lambda_vol) and correlation (lambda_corr).
    """
    T, n = R.shape
    R = R - R.mean(axis=0)

    # exponentially weighted volatilities
    vol = np.zeros((T, n))
    vol[0] = R.std(axis=0)
    for t in range(1, T):
        vol[t] = np.sqrt(lambda_vol * vol[t - 1] ** 2
                         + (1 - lambda_vol) * R[t - 1] ** 2)

    # standardise to correlation innovations
    Z = R / np.where(vol > 0, vol, 1e-10)

    # EWMA correlation matrix
    S = np.zeros((n, n))
    for t in range(T):
        z = Z[t].reshape(-1, 1)
        S = lambda_corr * S + (1 - lambda_corr) * (z @ z.T)

    D = np.diag(vol[-1])
    return D @ S @ D


def ewma_ledoit_wolf(R, lambda_corr, lambda_vol):
    """
    EWMA covariance with Ledoit-Wolf shrinkage applied to the correlation matrix.
    Combines the adaptiveness of EWMA with the stability of shrinkage.
    """
    T, n = R.shape
    R_dm = R - R.mean(axis=0)

    # EWMA vols
    vol = np.zeros((T, n))
    vol[0] = R_dm.std(axis=0)
    for t in range(1, T):
        vol[t] = np.sqrt(lambda_vol * vol[t - 1] ** 2
                         + (1 - lambda_vol) * R_dm[t - 1] ** 2)

    Z = R_dm / np.where(vol > 0, vol, 1e-10)

    # EWMA correlation
    S_corr = np.zeros((n, n))
    for t in range(T):
        z = Z[t].reshape(-1, 1)
        S_corr = lambda_corr * S_corr + (1 - lambda_corr) * (z @ z.T)

    # shrink the correlation matrix toward identity
    trace_S  = np.trace(S_corr)
    trace_S2 = np.trace(S_corr @ S_corr)
    mu = trace_S / n
    rho_num = ((1 - 2 / n) * trace_S2 + trace_S ** 2)
    rho_den = (T + 1 - 2 / n) * (trace_S2 - trace_S ** 2 / n)
    rho = min(0.5, rho_num / rho_den) if rho_den > 0 else 0.3  # cap at 0.5

    S_corr_shrunk = (1 - rho) * S_corr + rho * mu * np.eye(n)

    D = np.diag(vol[-1])
    return D @ S_corr_shrunk @ D


# ---------------------------------------------------------------------------
# Portfolio solvers
# ---------------------------------------------------------------------------

def _regularise(Sigma, floor=1e-8):
    """Add a tiny diagonal to guarantee positive definiteness."""
    eigs = np.linalg.eigvalsh(Sigma)
    shift = max(0.0, -eigs.min() + floor)
    return Sigma + shift * np.eye(len(Sigma))


def min_var_weights(Sigma):
    """
    Unconstrained long-only minimum variance via closed form.
    Falls back to constrained QP if solution has negative weights.
    """
    Sigma = _regularise(Sigma)
    n = len(Sigma)
    try:
        ones = np.ones(n)
        inv_ones = np.linalg.solve(Sigma, ones)
        w = inv_ones / inv_ones.sum()
        if np.all(w >= -1e-6):
            return np.clip(w, 0, None)
    except np.linalg.LinAlgError:
        pass
    return min_var_constrained(Sigma, max_w=1.0)


def min_var_constrained(Sigma, max_w=0.05):
    """
    Long-only minimum variance with per-weight cap solved via SLSQP.
    Warm-started from an inverse-volatility guess for faster convergence.
    """
    Sigma = _regularise(Sigma)
    n = len(Sigma)

    # inverse-volatility warm start
    inv_vol = 1.0 / np.sqrt(np.diag(Sigma))
    w0 = inv_vol / inv_vol.sum()
    w0 = np.clip(w0, 0, max_w)
    w0 /= w0.sum()

    result = minimize(
        fun=lambda w: w @ Sigma @ w,
        jac=lambda w: 2 * Sigma @ w,
        x0=w0,
        method="SLSQP",
        bounds=[(0, max_w)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    return result.x