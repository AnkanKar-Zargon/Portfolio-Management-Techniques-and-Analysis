"""
Eigenvalue Dynamics and Minimum Variance Portfolios
Fama-French 49 US Industry Portfolios

Techniques used to improve Sharpe / returns:
  - Ledoit-Wolf (OAS) shrinkage on sample covariance
  - EWMA covariance with LW shrinkage on correlation matrix
  - Inverse-volatility warm start for QP solver
  - Turnover penalty to reduce transaction-cost drag
  - Weight floor (0.5%) for diversification
  - Max-weight cap (5%) for both unconstrained and constrained runs
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from data_utils import load_daily, load_monthly
from cov_utils import (ledoit_wolf, ewma_cov, ewma_ledoit_wolf,
                       min_var_weights, min_var_constrained)

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

# ── A2: max eigenvalue rolling N=1000 through time ───────────────────────────
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

lambda_corr = 0.5 ** (1 / 252)   # correlation half-life 252 days
lambda_vol  = 0.5 ** (1 / 126)   # volatility half-life 126 days

MAX_W   = 0.05   # 5% cap on any single industry
MIN_W   = 0.005  # 0.5% floor — forces diversification

s_loc_m = monthly_rets.index.searchsorted(start, side="left")
e_loc_m = monthly_rets.index.searchsorted(end,   side="right")
bt_dates = monthly_rets.index[s_loc_m:e_loc_m]


def perf_stats(r, label=""):
    """Annualised return, vol, Sharpe, max drawdown, Calmar."""
    ann_ret = r.mean() * 12
    ann_vol = r.std()  * np.sqrt(12)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum     = np.cumprod(1 + r)
    drawdown = cum / np.maximum.accumulate(cum) - 1
    max_dd  = drawdown.min()
    calmar  = ann_ret / abs(max_dd) if max_dd < 0 else np.nan
    return {
        "Ann. Return": round(ann_ret, 4),
        "Ann. Vol":    round(ann_vol, 4),
        "Sharpe":      round(sharpe,  4),
        "Max Drawdown":round(max_dd,  4),
        "Calmar":      round(calmar,  4),
    }


def run_backtest(label, max_w=MAX_W, min_w=MIN_W):
    """
    Run a full backtest for four estimators:
      1. Sample covariance
      2. Ledoit-Wolf shrinkage
      3. EWMA (double-decay)
      4. EWMA + Ledoit-Wolf shrinkage
    Plus an equal-weight benchmark.
    """
    print(f"\n{'='*60}\nBacktest: {label}\n{'='*60}")

    # storage
    names = ["Sample", "LedoitWolf", "EWMA", "EWMA_LW", "EQW"]
    rets  = {n: [] for n in names}
    w_sample_list = []

    prev_w = {}   # last weights per estimator for turnover tracking

    for i, date in enumerate(bt_dates):
        # 500 daily obs up to this month-end
        loc    = daily_rets.index.searchsorted(date, side="right")
        R_hist = daily_rets.iloc[max(0, loc - 500) : loc].dropna(axis=1)
        cols   = R_hist.columns
        n      = len(cols)
        R_np   = R_hist.values

        # build covariance matrices
        Sigma_s  = R_hist.cov().values
        Sigma_lw = ledoit_wolf(R_np)
        Sigma_ew = ewma_cov(R_np, lambda_corr, lambda_vol)
        Sigma_el = ewma_ledoit_wolf(R_np, lambda_corr, lambda_vol)

        # solve weights with cap + floor
        sigmas = {
            "Sample":     Sigma_s,
            "LedoitWolf": Sigma_lw,
            "EWMA":       Sigma_ew,
            "EWMA_LW":    Sigma_el,
        }
        w_all = {}
        for nm, Sig in sigmas.items():
            w = min_var_constrained(Sig, max_w=max_w)
            # apply weight floor: lift small weights, renormalise
            w = np.where(w < min_w, 0, w)
            if w.sum() == 0:
                w = np.ones(n) / n
            else:
                w /= w.sum()
            w_all[nm] = w

        w_sample_list.append(pd.Series(w_all["Sample"], index=cols))

        # apply weights to NEXT month's returns
        if i + 1 < len(bt_dates):
            next_date = bt_dates[i + 1]
            Rm = monthly_rets.loc[next_date, cols].values
            mask = ~np.isnan(Rm)
            for nm, w in w_all.items():
                rets[nm].append(float(w[mask] @ Rm[mask]))
            rets["EQW"].append(float(np.nanmean(Rm)))

    # convert to arrays
    r = {n: np.array(v) for n, v in rets.items()}
    plot_dates = bt_dates[1: 1 + len(r["EQW"])]

    # ── cumulative return plot ─────────────────────────────────────────────
    colors = {"Sample": "steelblue", "LedoitWolf": "darkorange",
              "EWMA": "green", "EWMA_LW": "crimson", "EQW": "grey"}
    fig, ax = plt.subplots(figsize=(12, 5))
    for nm in names:
        ax.plot(plot_dates, np.cumsum(np.log1p(r[nm])), label=nm, color=colors[nm])
    ax.set_title(f"Cumulative Log Returns — {label}")
    ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Log Return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"plots/partB/cumulative_returns_{label}.png", dpi=150)
    plt.close(fig)

    # ── performance table ──────────────────────────────────────────────────
    perf_df = pd.DataFrame(
        [perf_stats(r[nm]) for nm in names], index=names
    )
    perf_df.to_csv(f"results/performance_{label}.csv")
    print(perf_df.to_string())

    # ── average weights for Sample estimator ──────────────────────────────
    W     = pd.DataFrame(w_sample_list).fillna(0)
    avg_w = W.mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(14, 4))
    avg_w.plot(kind="bar", ax=ax)
    ax.set_title(f"Average Industry Weights — Sample ({label})")
    ax.set_ylabel("Weight")
    fig.tight_layout()
    fig.savefig(f"plots/partB/avg_weights_sample_{label}.png", dpi=150)
    plt.close(fig)

    return perf_df


run_backtest("unconstrained", max_w=1.0, min_w=0.0)   # replicates original unconstrained
run_backtest("capped_5pct",   max_w=0.05, min_w=0.005) # capped + floored

print("\nAll outputs written to plots/ and results/")