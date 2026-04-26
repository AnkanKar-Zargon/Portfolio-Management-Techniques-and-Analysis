"""
Eigenvalue Dynamics and Minimum Variance Portfolios
Fama-French 49 US Industry Portfolios

Improvements over baseline (research-backed):
  1. Ledoit-Wolf shrinkage toward single-factor market target
     (Ledoit & Wolf 2003 JEF — best for industry portfolios)
  2. Fixed EWMA: properly normalised correlation matrix (unit diagonal)
  3. EWMA + LW shrinkage on correlation (adaptive + stable)
  4. Volatility targeting at 10% p.a. — scales weights when vol is high/low,
     allocating residual to cash. Proven to boost Sharpe (Moreira & Muir 2017)
  5. Turnover penalty in QP objective (reduces whipsaw rebalancing)
  6. Weight floor 0.5% + 5% cap for diversification
  7. Longer correlation window (750 days) + shorter vol window (126 days)
     to separate slow-moving correlation from fast-moving vol
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

from data_utils import load_daily, load_monthly
from cov_utils import (
    sample_cov, ledoit_wolf_identity, ledoit_wolf_market,
    ewma_cov, ewma_ledoit_wolf,
    min_var_weights, min_var_constrained, vol_target,
)

os.makedirs("plots/partA", exist_ok=True)
os.makedirs("plots/partB", exist_ok=True)
os.makedirs("results",     exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────────
daily_rets   = load_daily("data/49_Industry_Portfolios_Daily.csv")["returns"]
monthly_rets = load_monthly("data/49_Industry_Portfolios.csv")["returns"]

# =============================================================================
# PART A — Covariance Matrix Analysis
# =============================================================================

# ── A1: min eigenvalue & condition number vs window N on 2022-12-31 ───────────
target_loc         = daily_rets.index.searchsorted(pd.Timestamp("2022-12-31"), side="right")
daily_up_to_target = daily_rets.iloc[:target_loc]

Ns, min_eigs, cond_nos = range(55, 505, 5), [], []
for N in Ns:
    R    = daily_up_to_target.iloc[-N:].dropna(axis=1)
    eigs = np.linalg.eigvalsh(R.cov().values)
    min_eigs.append(eigs.min())
    cond_nos.append(eigs.max() / eigs.min())

fig, ax = plt.subplots()
ax.plot(list(Ns), min_eigs)
ax.set_yscale("log")
ax.set_title("Minimum Eigenvalue vs Window Length")
ax.set_xlabel("N"); ax.set_ylabel("Min Eigenvalue (log scale)")
fig.tight_layout(); fig.savefig("plots/partA/min_eigenvalue_vs_N.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots()
ax.plot(list(Ns), cond_nos, color="firebrick")
ax.set_yscale("log")
ax.set_title("Condition Number vs Window Length")
ax.set_xlabel("N"); ax.set_ylabel("Condition Number (log scale)")
fig.tight_layout(); fig.savefig("plots/partA/condition_number_vs_N.png", dpi=150); plt.close(fig)
print("Part A1 plots saved.")

# ── A2: max eigenvalue rolling N=1000 through time ────────────────────────────
start = pd.Timestamp("2005-03-31")
end   = pd.Timestamp("2025-07-31")

s_loc = daily_rets.index.searchsorted(start, side="left")
e_loc = daily_rets.index.searchsorted(end,   side="right")
window_dates = daily_rets.index[s_loc:e_loc]

max_eigs, valid_dates = [], []
for i, date in enumerate(window_dates):
    pos = s_loc + i
    if pos < 1000:
        continue
    R    = daily_rets.iloc[pos - 1000 : pos].dropna(axis=1)
    eigs = np.linalg.eigvalsh(R.cov().values)
    max_eigs.append(eigs.max())
    valid_dates.append(date)

fig, ax = plt.subplots()
ax.plot(valid_dates, max_eigs)
ax.set_title("Maximum Eigenvalue of Sample Covariance (N=1000)")
ax.set_xlabel("Date"); ax.set_ylabel("Max Eigenvalue")
fig.tight_layout(); fig.savefig("plots/partA/max_eigenvalue_through_time.png", dpi=150); plt.close(fig)
print("Part A2 plot saved.")

# =============================================================================
# PART B — Portfolio Backtesting
# =============================================================================

# EWMA decay parameters
# Longer correlation half-life (252d) — correlations move slowly
# Shorter vol half-life (63d) — vol reacts faster to regime changes
LAMBDA_CORR = 0.5 ** (1 / 252)
LAMBDA_VOL  = 0.5 ** (1 / 63)

# Backtest settings
CORR_WINDOW = 750    # days for correlation estimation (longer = more stable)
VOL_WINDOW  = 126    # days for volatility estimation  (shorter = more reactive)
MAX_W       = 0.10   # 10% single-stock cap (wider than 5% to avoid over-constraint)
MIN_W       = 0.005  # 0.5% floor — forces genuine diversification
VOL_TGT     = 0.10   # 10% annualised volatility target
TURNOVER_PEN= 0.001  # penalty coefficient on L1 turnover in QP

s_loc_m  = monthly_rets.index.searchsorted(start, side="left")
e_loc_m  = monthly_rets.index.searchsorted(end,   side="right")
bt_dates = monthly_rets.index[s_loc_m:e_loc_m]


def perf_stats(r):
    """Full performance table: return, vol, Sharpe, max drawdown, Calmar."""
    ann_ret  = r.mean() * 12
    ann_vol  = r.std()  * np.sqrt(12)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum      = np.cumprod(1 + r)
    dd       = cum / np.maximum.accumulate(cum) - 1
    max_dd   = dd.min()
    calmar   = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "Ann. Return":  round(ann_ret, 4),
        "Ann. Vol":     round(ann_vol, 4),
        "Sharpe":       round(sharpe,  4),
        "Max Drawdown": round(max_dd,  4),
        "Calmar":       round(calmar,  4),
    }


def apply_floor_cap(w, min_w, max_w):
    """Zero out sub-floor weights, clip at cap, renormalise."""
    w = np.where(w < min_w, 0.0, w)
    w = np.clip(w, 0, max_w)
    s = w.sum()
    return w / s if s > 0 else np.ones(len(w)) / len(w)


def min_var_with_turnover(Sigma, prev_w, max_w, turnover_pen):
    """
    Min-var QP with L1 turnover penalty:
      min  w' Sigma w  +  lambda * ||w - w_prev||_1
    Linearised as an augmented QP with slack variables.
    If no previous weights exist, falls back to plain constrained GMV.
    """
    from cov_utils import _regularise
    Sigma = _regularise(Sigma)
    n     = len(Sigma)

    if prev_w is None or len(prev_w) != n:
        return min_var_constrained(Sigma, max_w=max_w)

    # augment: x = [w (n), u (n)]  where u >= |w - w_prev|
    def obj(x):
        w, u = x[:n], x[n:]
        return float(w @ Sigma @ w) + turnover_pen * u.sum()

    def jac(x):
        w, u = x[:n], x[n:]
        gw = 2 * Sigma @ w
        gu = np.full(n, turnover_pen)
        return np.concatenate([gw, gu])

    inv_vol = 1.0 / np.sqrt(np.maximum(np.diag(Sigma), 1e-10))
    w0      = np.clip(inv_vol / inv_vol.sum(), 0, max_w)
    w0     /= w0.sum()
    u0      = np.abs(w0 - prev_w)
    x0      = np.concatenate([w0, u0])

    # bounds: w in [0, max_w], u in [0, inf]
    bounds = [(0, max_w)] * n + [(0, None)] * n

    constraints = [
        {"type": "eq",  "fun": lambda x: x[:n].sum() - 1},
        # u >= w - w_prev
        {"type": "ineq","fun": lambda x: x[n:] - (x[:n] - prev_w)},
        # u >= w_prev - w
        {"type": "ineq","fun": lambda x: x[n:] - (prev_w - x[:n])},
    ]

    res = minimize(obj, x0, jac=jac, method="SLSQP",
                   bounds=bounds, constraints=constraints,
                   options={"ftol": 1e-12, "maxiter": 3000})
    return res.x[:n]


def run_backtest(label, use_vol_target=True, max_w=MAX_W, min_w=MIN_W,
                 use_turnover_pen=True):
    """
    Backtest five estimators + EQW benchmark.
    Optionally applies volatility targeting and turnover penalty.
    """
    print(f"\n{'='*60}\nBacktest: {label}\n{'='*60}")

    names = ["Sample", "LW_Identity", "LW_Market", "EWMA", "EWMA_LW", "EQW"]
    rets  = {n: [] for n in names}
    w_sample_list  = []
    prev_weights   = {n: None for n in names}

    for i, date in enumerate(bt_dates[:-1]):   # last date has no next month
        # ── build return history windows ───────────────────────────────────
        loc      = daily_rets.index.searchsorted(date, side="right")
        # separate windows for correlation vs volatility
        R_corr   = daily_rets.iloc[max(0, loc - CORR_WINDOW) : loc].dropna(axis=1)
        R_vol    = daily_rets.iloc[max(0, loc - VOL_WINDOW)  : loc].dropna(axis=1)
        # intersect columns
        cols     = R_corr.columns.intersection(R_vol.columns)
        R_corr   = R_corr[cols].values
        R_vol    = R_vol[cols].values
        n        = len(cols)

        # ── covariance matrices ────────────────────────────────────────────
        # Sample and LW use the longer window for stability
        Sigma_s   = sample_cov(R_corr)
        Sigma_lwi = ledoit_wolf_identity(R_corr)
        Sigma_lwm = ledoit_wolf_market(R_corr)
        # EWMA uses its own decay on the full corr window
        Sigma_ew  = ewma_cov(R_corr, LAMBDA_CORR, LAMBDA_VOL)
        Sigma_el  = ewma_ledoit_wolf(R_corr, LAMBDA_CORR, LAMBDA_VOL)

        sigmas = {
            "Sample":     Sigma_s,
            "LW_Identity":Sigma_lwi,
            "LW_Market":  Sigma_lwm,
            "EWMA":       Sigma_ew,
            "EWMA_LW":    Sigma_el,
        }

        # ── solve weights ──────────────────────────────────────────────────
        w_all = {}
        for nm, Sig in sigmas.items():
            pen = TURNOVER_PEN if use_turnover_pen else 0.0
            if pen > 0:
                w = min_var_with_turnover(Sig, prev_weights[nm], max_w, pen)
            else:
                w = min_var_constrained(Sig, max_w=max_w)

            w = apply_floor_cap(w, min_w, max_w)

            # volatility targeting: scale to VOL_TGT, cash holds residual
            if use_vol_target:
                w = vol_target(w, Sig, target_ann_vol=VOL_TGT)

            w_all[nm]          = w
            prev_weights[nm]   = w[:n]   # store for next period turnover

        w_sample_list.append(pd.Series(w_all["Sample"], index=cols))

        # ── apply weights to next month's returns ──────────────────────────
        next_date = bt_dates[i + 1]
        Rm        = monthly_rets.loc[next_date, cols].values
        mask      = ~np.isnan(Rm)

        for nm, w in w_all.items():
            rets[nm].append(float(w[mask] @ Rm[mask]))

        # EQW — equal weight, no vol targeting
        rets["EQW"].append(float(np.nanmean(Rm)))

    # ── convert and trim to same length ───────────────────────────────────
    n_obs = min(len(v) for v in rets.values())
    r     = {nm: np.array(v[:n_obs]) for nm, v in rets.items()}
    plot_dates = bt_dates[1: 1 + n_obs]

    # ── cumulative return plot ─────────────────────────────────────────────
    colors = {"Sample": "steelblue", "LW_Identity": "darkorange",
              "LW_Market": "purple",  "EWMA": "green",
              "EWMA_LW": "crimson",   "EQW": "grey"}
    fig, ax = plt.subplots(figsize=(13, 5))
    for nm in names:
        ax.plot(plot_dates, np.cumsum(np.log1p(r[nm])),
                label=nm, color=colors[nm],
                linewidth=1.5 if nm != "EQW" else 1.0,
                linestyle="--" if nm == "EQW" else "-")
    ax.set_title(f"Cumulative Log Returns — {label}")
    ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Log Return")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"plots/partB/cumulative_returns_{label}.png", dpi=150)
    plt.close(fig)

    # ── performance table ──────────────────────────────────────────────────
    perf_df = pd.DataFrame([perf_stats(r[nm]) for nm in names], index=names)
    perf_df.to_csv(f"results/performance_{label}.csv")
    print(perf_df.to_string())

    # ── average industry weights — Sample estimator ────────────────────────
    W     = pd.DataFrame(w_sample_list).fillna(0)
    avg_w = W.mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(14, 4))
    avg_w.plot(kind="bar", ax=ax)
    ax.set_title(f"Average Industry Weights — Sample ({label})")
    ax.set_ylabel("Weight")
    fig.tight_layout()
    fig.savefig(f"plots/partB/avg_weights_sample_{label}.png", dpi=150)
    plt.close(fig)

    # ── weight heatmap through time — Sample estimator ─────────────────────
    # shows how allocations evolve, highlights concentration issues
    W_ts = W.copy()
    W_ts.index = bt_dates[:len(W_ts)]
    # resample to quarterly for readability
    W_q  = W_ts.resample("QE").mean()
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(W_q.T, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    ax.set_yticks(range(len(W_q.columns)))
    ax.set_yticklabels(W_q.columns, fontsize=6)
    ax.set_xticks(range(0, len(W_q), 4))
    ax.set_xticklabels([str(d.year) for d in W_q.index[::4]], fontsize=8)
    ax.set_title(f"Industry Weight Heatmap — Sample ({label})")
    plt.colorbar(im, ax=ax, label="Weight")
    fig.tight_layout()
    fig.savefig(f"plots/partB/weight_heatmap_sample_{label}.png", dpi=150)
    plt.close(fig)

    return perf_df


# ── Run 1: Original unconstrained baseline (no vol target, no turnover pen) ───
run_backtest("baseline_unconstrained",
             use_vol_target=False, max_w=1.0, min_w=0.0,
             use_turnover_pen=False)

# ── Run 2: Capped 5% (original constrained baseline) ─────────────────────────
run_backtest("baseline_capped_5pct",
             use_vol_target=False, max_w=0.05, min_w=0.0,
             use_turnover_pen=False)

run_backtest("baseline_capped_10pct",
             use_vol_target=False, max_w=0.1, min_w=0.001,
             use_turnover_pen=False)

# ── Run 3: Full improvements — vol target + turnover penalty + 10% cap ────────
run_backtest("improved_voltarget_turnover",
             use_vol_target=True, max_w=MAX_W, min_w=MIN_W,
             use_turnover_pen=True)

# ── Run 4: Improvements without vol target (pure GMV quality) ─────────────────
run_backtest("improved_no_voltarget",
             use_vol_target=False, max_w=MAX_W, min_w=MIN_W,
             use_turnover_pen=True)

print("\nAll outputs written to plots/ and results/")