# Eigenvalue Dynamics and Minimum Variance Portfolios

Replication of the Fama-French 49 Industry Portfolio analysis in Python.

## Files

| File | Purpose |
|---|---|
| `main.py` | Runs all Part A & B analyses and saves outputs |
| `data_utils.py` | Loads Fama-French CSVs into DataFrames |
| `cov_utils.py` | EWMA covariance estimator and min-variance solvers |
| `requirements.txt` | Python dependencies |

## Data

Download from [Ken French's Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html):

- `49_Industry_Portfolios_Daily.CSV`
- `49_Industry_Portfolios.CSV` (monthly)

Place both files in a `data/` folder next to `main.py`.

## Usage

```bash
pip install -r requirements.txt
python main.py
```

## Outputs

```
plots/
  partA/
    min_eigenvalue_vs_N.png
    condition_number_vs_N.png
    max_eigenvalue_through_time.png
  partB/
    cumulative_returns_unconstrained.png
    cumulative_returns_constrained_5pct.png
    avg_weights_sample_unconstrained.png
    avg_weights_sample_constrained_5pct.png
results/
  performance_unconstrained.csv
  performance_constrained_5pct.csv
```
