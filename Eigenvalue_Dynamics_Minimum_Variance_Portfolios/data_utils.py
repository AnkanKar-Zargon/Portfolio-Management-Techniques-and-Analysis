"""
data_utils.py — load Fama-French 49 Industry Portfolio CSVs.

Ken French CSVs contain multiple sub-tables (e.g. equal-weighted and
value-weighted) stacked in the same file, separated by text headers and
annual summary rows. We isolate only the first table (average value-weighted
returns) and drop any duplicate dates that would break datetime slicing.
"""

import pandas as pd
import numpy as np


def _parse_ff_csv(filepath, date_format):
    df = pd.read_csv(filepath, header=0, index_col=0, low_memory=False)

    idx_str = df.index.astype(str).str.strip()

    # expected digit count derived from format string
    expected_len = (date_format.count("%Y") * 4
                    + date_format.count("%m") * 2
                    + date_format.count("%d") * 2)

    # keep only purely-numeric rows of the exact date length
    numeric_mask = pd.to_numeric(idx_str, errors="coerce").notna()
    length_mask  = idx_str.str.len() == expected_len
    df = df[numeric_mask & length_mask].copy()

    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format=date_format)
    df = df.apply(pd.to_numeric, errors="coerce")
    df[df <= -99] = np.nan
    df = df / 100.0
    df.columns = df.columns.str.strip()

    # Ken French files stack multiple sub-tables; keep only the first occurrence
    # of each date (value-weighted returns appear first)
    df = df[~df.index.duplicated(keep="first")]
    df.sort_index(inplace=True)

    return df


def load_daily(filepath):
    rets = _parse_ff_csv(filepath, "%Y%m%d")
    return {"returns": rets}


def load_monthly(filepath):
    rets = _parse_ff_csv(filepath, "%Y%m")
    rets.index = rets.index + pd.offsets.MonthEnd(0)
    return {"returns": rets}