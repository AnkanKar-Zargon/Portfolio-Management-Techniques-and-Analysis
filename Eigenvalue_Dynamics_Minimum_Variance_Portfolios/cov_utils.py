"""
cov_utils.py — covariance estimators and minimum-variance portfolio solvers.
"""

import numpy as np
from scipy.optimize import minimize


def ewma_cov(R, lambda_corr, lambda_vol):
    """
    Double-decay EWMA covariance matrix.
    Separate decay for volatilities (lambda_vol) and correlations (lambda_corr).

    R           : (T x n) numpy array of returns
    lambda_corr : smoothing factor for correlation, e.g. 0.5^(1/252)
    lambda_vol  : smoothing factor for volatility,   e.g. 0.5^(1/126)
    """
    T, n = R.shape

    # demean
    R = R - R.mean(axis=0)

    # EWMA volatilities
    vol = np.zeros((T, n))
    vol[0] = R.std(axis=0)
    for t in range(1, T):
        vol[t] = np.sqrt(lambda_vol * vol[t - 1] ** 2
                         + (1 - lambda_vol) * R[t - 1] ** 2)

    # standardise to get correlation innovations
    Z = R / np.where(vol > 0, vol, 1e-12)

    # EWMA correlation matrix
    S = np.zeros((n, n))
    for t in range(T):
        z = Z[t].reshape(-1, 1)
        S = lambda_corr * S + (1 - lambda_corr) * (z @ z.T)

    # scale back to covariance
    D = np.diag(vol[-1])
    return D @ S @ D


def min_var_weights(Sigma):
    """
    Unconstrained long-only minimum variance weights via closed form:
      w = Sigma^{-1} 1 / (1' Sigma^{-1} 1)
    Falls back to QP if the matrix is (near-)singular.
    """
    n = Sigma.shape[0]
    try:
        ones = np.ones(n)
        inv_ones = np.linalg.solve(Sigma, ones)
        w = inv_ones / inv_ones.sum()
        if np.all(w >= -1e-8):   # accept if effectively non-negative
            return np.clip(w, 0, None)
    except np.linalg.LinAlgError:
        pass
    # fallback: unconstrained QP (no short-sale here, use constrained)
    return min_var_constrained(Sigma, max_w=1.0)


def min_var_constrained(Sigma, max_w=0.05):
    """
    Long-only minimum variance with individual weight cap max_w.
    Solved with scipy SLSQP.
    """
    n = Sigma.shape[0]
    w0 = np.ones(n) / n

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bounds = [(0, max_w)] * n

    result = minimize(
        fun=lambda w: w @ Sigma @ w,
        x0=w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return result.x
