"""
Eigenvalue Dynamics and Minimum Variance Portfolios
Fama-French 49 US Industry Portfolios
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from data_utils import load_daily, load_monthly
from cov_utils import ewma_cov, min_var_weights, min_var_constrained

# ── Output dirs ──────────────────────────────────────────────────────────────
import os
os.makedirs("plots/partA", exist_ok=True)
os.makedirs("plots/partB", exist_ok=True)
os.makedirs("results",     exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
daily   = load_daily("data/49_Industry_Portfolios_Daily.csv")
monthly = load_monthly("data/49_Industry_Portfolios.csv")

daily_rets   = daily["returns"]    # DataFrame, index=date, cols=industries
monthly_rets = monthly["returns"]

# =============================================================================
# PART A — Covariance Matrix Analysis
# =============================================================================

# ── A1: min eigenvalue & condition number vs window length (2022-12-31) ──────
# use searchsorted so a non-trading day (weekend/holiday) still works
target_loc = daily_rets.index.searchsorted(pd.Timestamp("2022-12-31"), side="right")
daily_up_to_target = daily_rets.iloc[:target_loc]

Ns       = range(55, 505, 5)
min_eigs = []
cond_nos = []

for N in Ns:
    R = daily_up_to_target.iloc[-N:].dropna(axis=1)
    Sigma = R.cov().values
    eigs  = np.linalg.eigvalsh(Sigma)
    min_eigs.append(eigs.min())
    cond_nos.append(eigs.max() / eigs.min())

fig, ax = plt.subplots()
ax.plot(list(Ns), min_eigs)
ax.set_yscale("log")
ax.set_title("Minimum Eigenvalue vs Window Length")
ax.set_xlabel("N")
ax.set_ylabel("Min Eigenvalue (log scale)")
fig.tight_layout()
fig.savefig("plots/partA/min_eigenvalue_vs_N.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots()
ax.plot(list(Ns), cond_nos, color="firebrick")
ax.set_yscale("log")
ax.set_title("Condition Number vs Window Length")
ax.set_xlabel("N")
ax.set_ylabel("Condition Number (log scale)")
fig.tight_layout()
fig.savefig("plots/partA/condition_number_vs_N.png", dpi=150)
plt.close(fig)

print("Part A1 plots saved.")

# ── A2: max eigenvalue through time (N=1000 rolling, 2005-03-31 to 2025-07-31) ─
start = pd.Timestamp("2005-03-31")
end   = pd.Timestamp("2025-07-31")
# use iloc with searchsorted to avoid KeyError on non-trading-day boundaries
s_loc = daily_rets.index.searchsorted(start, side='left')
e_loc = daily_rets.index.searchsorted(end,   side='right')
window_dates = daily_rets.index[s_loc:e_loc]

max_eigs = []
valid_dates = []

for date in window_dates:
    loc = daily_rets.index.get_loc(date)
    if loc < 1000:
        continue
    R = daily_rets.iloc[loc - 1000 : loc].dropna(axis=1)
    Sigma = R.cov().values
    eigs  = np.linalg.eigvalsh(Sigma)
    max_eigs.append(eigs.max())
    valid_dates.append(date)

fig, ax = plt.subplots()
ax.plot(valid_dates, max_eigs)
ax.set_title("Maximum Eigenvalue of Sample Covariance (N=1000)")
ax.set_xlabel("Date")
ax.set_ylabel("Max Eigenvalue")
fig.tight_layout()
fig.savefig("plots/partA/max_eigenvalue_through_time.png", dpi=150)
plt.close(fig)

print("Part A2 plot saved.")

# =============================================================================
# PART B — Portfolio Backtesting
# =============================================================================

lambda_corr = 0.5 ** (1 / 252)   # correlation half-life 252 days
lambda_vol  = 0.5 ** (1 / 126)   # volatility half-life 126 days

# ── helper: run one backtest pass ─────────────────────────────────────────────
def run_backtest(weight_fn_sample, weight_fn_ewma, label):
    # same searchsorted pattern for monthly index
    s_loc_m = monthly_rets.index.searchsorted(start, side='left')
    e_loc_m = monthly_rets.index.searchsorted(end,   side='right')
    bt_dates = monthly_rets.index[s_loc_m:e_loc_m]

    w_sample_list = []
    r_sample, r_ewma, r_eqw = [], [], []

    for i, date in enumerate(bt_dates):
        # last 500 daily obs up to this month-end
        loc = daily_rets.index.searchsorted(date, side="right")
        R_hist = daily_rets.iloc[max(0, loc - 500) : loc].dropna(axis=1)

        Sigma_s = R_hist.cov().values
        Sigma_e = ewma_cov(R_hist.values, lambda_corr, lambda_vol)

        cols = R_hist.columns

        w_s = weight_fn_sample(Sigma_s, len(cols))
        w_e = weight_fn_ewma(Sigma_e,   len(cols))

        w_sample_list.append(pd.Series(w_s, index=cols))

        # apply weights to next month's returns
        if i + 1 < len(bt_dates):
            next_date = bt_dates[i + 1]
            Rm = monthly_rets.loc[next_date, cols].values
            mask = ~np.isnan(Rm)
            r_sample.append(float(w_s[mask] @ Rm[mask]))
            r_ewma.append(  float(w_e[mask] @ Rm[mask]))
            r_eqw.append(   float(np.nanmean(Rm)))

    r_sample = np.array(r_sample)
    r_ewma   = np.array(r_ewma)
    r_eqw    = np.array(r_eqw)
    plot_dates = bt_dates[1:]

    # cumulative log returns
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, r, name in zip(axes,
                           [r_sample, r_ewma, r_eqw],
                           ["Sample.Port", "EWMA.Port", "EQW.Port"]):
        ax.plot(plot_dates, np.cumsum(np.log1p(r)))
        ax.set_title(name)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Log Return")
    fig.suptitle(f"Cumulative Log Returns — {label}")
    fig.tight_layout()
    fig.savefig(f"plots/partB/cumulative_returns_{label}.png", dpi=150)
    plt.close(fig)

    # performance table
    def perf(r):
        return {
            "Ann. Return": round(r.mean() * 12, 4),
            "Ann. Vol":    round(r.std() * np.sqrt(12), 4),
            "Sharpe":      round(r.mean() / r.std() * np.sqrt(12), 4),
        }

    perf_df = pd.DataFrame(
        [perf(r_sample), perf(r_ewma), perf(r_eqw)],
        index=["Sample.Port", "EWMA.Port", "EQW.Port"]
    )
    perf_df.to_csv(f"results/performance_{label}.csv")
    print(f"\nPerformance ({label}):")
    print(perf_df.to_string())

    # average industry weights for Sample.Port
    W = pd.DataFrame(w_sample_list).fillna(0)
    avg_w = W.mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(14, 4))
    avg_w.plot(kind="bar", ax=ax)
    ax.set_title(f"Average Industry Weights — Sample.Port ({label})")
    ax.set_ylabel("Weight")
    fig.tight_layout()
    fig.savefig(f"plots/partB/avg_weights_sample_{label}.png", dpi=150)
    plt.close(fig)

    return perf_df


# ── unconstrained run ─────────────────────────────────────────────────────────
def w_unconstrained(Sigma, n):
    return min_var_weights(Sigma)

run_backtest(w_unconstrained, w_unconstrained, "unconstrained")

# ── constrained run (max weight 5%) ───────────────────────────────────────────
def w_constrained(Sigma, n):
    return min_var_constrained(Sigma, max_w=0.05)

run_backtest(w_constrained, w_constrained, "constrained_5pct")

print("\nAll outputs written to plots/ and results/")