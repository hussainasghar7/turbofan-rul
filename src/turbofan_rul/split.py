"""Train/validation splitting by engine.

Rows from the same engine are highly correlated (cycle 101 looks almost like cycle 100).
A random row-wise split puts neighbouring cycles of one engine in both sets, so the
model is graded on engines it has effectively already seen, and validation error looks
far better than test error. Splitting by unit avoids this leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_by_unit(
    df: pd.DataFrame, val_fraction: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    units = np.sort(df["unit"].unique())
    rng = np.random.default_rng(seed)
    n_val = max(1, round(len(units) * val_fraction))
    val_units = set(rng.choice(units, size=n_val, replace=False).tolist())
    is_val = df["unit"].isin(val_units)
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)
