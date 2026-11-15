"""Feature engineering for RUL regression.

Every feature at cycle t may only use data from cycles <= t of the same engine.
In deployment you never see the future, so a feature that peeks ahead gives
great offline numbers and a useless model.
"""

from __future__ import annotations

import pandas as pd

from turbofan_rul.data import SENSOR_COLS


def select_informative_sensors(train: pd.DataFrame, min_std: float = 1e-3) -> list[str]:
    """Keep sensors that actually vary in the TRAINING data.

    In FD001 several sensors are (near-)constant over the whole fleet. They cannot
    carry information about degradation, and some models (e.g. a standardised linear
    model) misbehave on zero-variance inputs. The decision is made on training data
    only, so the test set never influences which columns we use.
    """
    stds = train[SENSOR_COLS].std()
    return [c for c in SENSOR_COLS if stds[c] > min_std]


def add_rolling_features(df: pd.DataFrame, cols: list[str], window: int = 10) -> pd.DataFrame:
    """Add a trailing rolling mean and std per engine for each column in `cols`.

    Raw sensor values are noisy; the trend over the last few cycles is a much
    steadier signal of wear. Rolling is done per unit so one engine's values never
    leak into another's, and it is trailing (cycles t-window+1..t), so no look-ahead.
    """
    out = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    grouped = out.groupby("unit")[cols]
    rolled_mean = grouped.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    rolled_std = grouped.rolling(window, min_periods=1).std().reset_index(level=0, drop=True)
    rolled_std = rolled_std.fillna(0.0)  # std of a single point is undefined; treat as 0
    out = out.join(rolled_mean.add_suffix(f"_mean{window}"))
    out = out.join(rolled_std.add_suffix(f"_std{window}"))
    return out


def feature_columns(sensors: list[str], window: int) -> list[str]:
    """Names of the model inputs, in a fixed order."""
    return (
        ["cycle"]
        + list(sensors)
        + [f"{c}_mean{window}" for c in sensors]
        + [f"{c}_std{window}" for c in sensors]
    )


def last_cycle_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per engine: its most recent cycle.

    This is exactly what the C-MAPSS test task asks: given each engine's history up
    to now, predict how many cycles it has left.
    """
    return df.sort_values(["unit", "cycle"]).groupby("unit").tail(1).reset_index(drop=True)
