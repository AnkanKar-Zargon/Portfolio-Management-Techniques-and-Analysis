"""
cov_utils.py — covariance estimators and minimum-variance portfolio solvers.

Estimators:
  1. Sample covariance (baseline)
  2. Ledoit-Wolf OAS shrinkage toward scaled identity
  3. Ledoit-Wolf shrinkage toward single-factor (market) target
     — proven best-in-class by Ledoit & Wolf (2003, JEF)
  4. Double-decay EWMA with properly normalised correlation matrix
  5. EWMA vols + LW-shrunk correlation (best adaptive estimator)

Solvers:
  - Unconstrained GMV (closed-form + regularisation fallback)
  - Constrained GMV with per-weight cap (SLSQP, analytic Jacobian)
  - Volatility-targeted wrapper (scales weights to hit target ann. vol)
"""

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _regularise(Sigma, floor=1e-8):
    """Shift spectrum so smallest eigenvalue >= floor."""
    eig_min = np.linalg.eigvalsh(Sigma).min()
    shift   = max(0.0, floor - eig_min)
    return Sigma + shift * np.eye(len(Sigma))


def _cov_to_corr(Sigma):
    """Convert covariance matrix to correlation matrix."""
    std  = np.sqrt(np.diag(Sigma))
    denom = np.outer(std, std)
    return Sigma / np.where(denom > 0, denom, 1.0), std


# ---------------------------------------------------------------------------
# Covariance estimators
# ---------------------------------------------------------------------------

def sample_cov(R):
    """Plain sample covariance."""
    return np.cov(R, rowvar=False)


def ledoit_wolf_identity(R):
    """
    Ledoit-Wolf OAS shrinkage toward scaled identity.
    Good general-purpose regulariser.
    Ref: Chen, Wiesel, Eldar, Hero (2010) IEEE Trans. Signal Process.
    """
    T, n = R.shape
    S        = np.cov(R, rowvar=False)
    trace_S  = np.trace(S)
    trace_S2 = np.trace(S @ S)
    mu       = trace_S / n

    rho_num = (1 - 2 / n) * trace_S2 + trace_S ** 2
    rho_den = (T + 1 - 2 / n) * (trace_S2 - trace_S ** 2 / n)
    rho     = min(1.0, rho_num / rho_den) if rho_den > 0 else 1.0

    return (1 - rho) * S + rho * mu * np.eye(n)


def ledoit_wolf_market(R):
    """
    Ledoit-Wolf shrinkage toward a single-factor (market) target.
    The target is the covariance implied by a one-factor model where
    the factor is the equal-weighted portfolio.  This is the best
    out-of-sample estimator for industry portfolios per:
      Ledoit & Wolf (2003) 'Improved estimation of the covariance
      matrix of stock returns with an application to portfolio selection'
      Journal of Empirical Finance.

    Analytical shrinkage intensity from:
      Ledoit & Wolf (2004) 'A well-conditioned estimator for
      large-dimensional covariance matrices' JMVA.
    """
    T, n   = R.shape
    S      = np.cov(R, rowvar=False)
    mkt    = R.mean(axis=1)            # equal-weighted market return

    # single-factor target Sigma_F = beta beta' * var(mkt) + diag(res_var)
    var_mkt = mkt.var(ddof=1)
    beta    = np.array([np.cov(R[:, i], mkt)[0, 1] / var_mkt
                        for i in range(n)])
    F       = var_mkt * np.outer(beta, beta)
    np.fill_diagonal(F, np.diag(S))   # preserve diagonal (total variances)

    # Ledoit-Wolf analytical shrinkage intensity (Oracle formula)
    # pi-hat: sum of asymptotic variances of sqrt(T)*(S_ij - Sigma_ij)
    Xd  = R - R.mean(axis=0)
    pi_ = 0.0
    for i in range(n):
        for j in range(n):
            pi_ += np.mean((Xd[:, i] * Xd[:, j] - S[i, j]) ** 2)
    pi_ /= T

    # rho-hat: sum of asymptotic covariances (simplified)
    rho_  = np.sum((S - F) ** 2) / n   # simplified scaling
    alpha = max(0.0, min(1.0, pi_ / (rho_ * T) if rho_ > 0 else 0.0))

    return (1 - alpha) * S + alpha * F


def ewma_cov(R, lambda_corr=0.94, lambda_vol=0.97):
    """
    Double-decay EWMA covariance.
    Correlation and volatility decay independently.
    Correlation matrix is properly normalised to have unit diagonal.
    """
    T, n  = R.shape
    R_dm  = R - R.mean(axis=0)

    # --- exponentially weighted variances (RiskMetrics style) ---
    var = np.full(n, R_dm.var(axis=0))
    S   = np.zeros((n, n))         # running outer product for correlation

    for t in range(T):
        z   = R_dm[t]
        var = lambda_vol * var + (1 - lambda_vol) * z ** 2
        z_std = z / np.sqrt(np.where(var > 0, var, 1e-10))
        S   = lambda_corr * S + (1 - lambda_corr) * np.outer(z_std, z_std)

    # normalise S to be a proper correlation matrix
    d_inv = 1.0 / np.sqrt(np.where(np.diag(S) > 0, np.diag(S), 1.0))
    C     = S * np.outer(d_inv, d_inv)
    np.fill_diagonal(C, 1.0)

    D = np.diag(np.sqrt(var))
    return D @ C @ D


def ewma_ledoit_wolf(R, lambda_corr=0.94, lambda_vol=0.97):
    """
    EWMA vols + Ledoit-Wolf shrinkage on the EWMA correlation matrix.
    Best adaptive estimator: fast vol response + stable correlation.
    """
    T, n  = R.shape
    R_dm  = R - R.mean(axis=0)

    var = np.full(n, R_dm.var(axis=0))
    S   = np.zeros((n, n))

    for t in range(T):
        z     = R_dm[t]
        var   = lambda_vol * var + (1 - lambda_vol) * z ** 2
        z_std = z / np.sqrt(np.where(var > 0, var, 1e-10))
        S     = lambda_corr * S + (1 - lambda_corr) * np.outer(z_std, z_std)

    # normalise to correlation
    d_inv = 1.0 / np.sqrt(np.where(np.diag(S) > 0, np.diag(S), 1.0))
    C     = S * np.outer(d_inv, d_inv)
    np.fill_diagonal(C, 1.0)

    # OAS shrinkage on the correlation matrix
    trace_C  = np.trace(C)          # = n since diagonal is 1
    trace_C2 = np.trace(C @ C)
    mu       = trace_C / n          # = 1
    rho_num  = (1 - 2 / n) * trace_C2 + trace_C ** 2
    rho_den  = (T + 1 - 2 / n) * (trace_C2 - trace_C ** 2 / n)
    rho      = min(0.6, rho_num / rho_den) if rho_den > 0 else 0.3

    C_shrunk = (1 - rho) * C + rho * np.eye(n)  # shrink toward identity corr

    D = np.diag(np.sqrt(var))
    return D @ C_shrunk @ D


# ---------------------------------------------------------------------------
# Portfolio solvers
# ---------------------------------------------------------------------------

def min_var_weights(Sigma):
    """
    Unconstrained GMV via closed form w = Sigma^{-1}1 / (1'Sigma^{-1}1).
    Falls back to SLSQP if closed form is not long-only.
    """
    Sigma = _regularise(Sigma)
    ones  = np.ones(len(Sigma))
    try:
        v = np.linalg.solve(Sigma, ones)
        w = v / v.sum()
        if np.all(w >= -1e-6):
            return np.clip(w, 0, None)
    except np.linalg.LinAlgError:
        pass
    return min_var_constrained(Sigma, max_w=1.0)


def min_var_constrained(Sigma, max_w=0.05):
    """
    Long-only GMV with per-weight cap. Solved with SLSQP.
    Warm-started from inverse-volatility weights.
    """
    Sigma = _regularise(Sigma)
    n     = len(Sigma)

    inv_vol = 1.0 / np.sqrt(np.maximum(np.diag(Sigma), 1e-10))
    w0      = np.clip(inv_vol / inv_vol.sum(), 0, max_w)
    w0     /= w0.sum()

    res = minimize(
        fun=lambda w: float(w @ Sigma @ w),
        jac=lambda w: 2 * Sigma @ w,
        x0=w0,
        method="SLSQP",
        bounds=[(0, max_w)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"ftol": 1e-12, "maxiter": 3000},
    )
    return res.x


def vol_target(w, Sigma, target_ann_vol=0.10):
    """
    Scale weight vector so the portfolio hits target_ann_vol (annualised).
    Daily cov assumed, so ann factor = sqrt(252).
    Caps leverage at 1x (no borrowing).
    """
    port_vol_ann = np.sqrt(float(w @ Sigma @ w)) * np.sqrt(252)
    scale        = min(1.0, target_ann_vol / port_vol_ann) if port_vol_ann > 0 else 1.0
    return w * scale   # residual goes to cash (return = 0)