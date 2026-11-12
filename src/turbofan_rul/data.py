"""Loading the C-MAPSS text files and building RUL labels.

File format (all FD001..FD004 subsets):
    train_FDxxx.txt  full run-to-failure trajectories, one row per engine cycle
    test_FDxxx.txt   trajectories cut off at some point before failure
    RUL_FDxxx.txt    true remaining life at the LAST test cycle of each engine

Each row has 26 space-separated numbers:
    unit, cycle, 3 operational settings, 21 sensor readings
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SUBSETS = ("FD001", "FD002", "FD003", "FD004")

ID_COLS = ["unit", "cycle"]
SETTING_COLS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ID_COLS + SETTING_COLS + SENSOR_COLS

DEFAULT_RUL_CAP = 125


@dataclass
class CMAPSSData:
    """One C-MAPSS subset, as loaded from disk."""

    subset: str
    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: pd.Series  # true RUL at the last cycle of each test unit, indexed by unit


def read_cmapss_file(path: str | Path) -> pd.DataFrame:
    """Read one train/test file into a DataFrame with named columns.

    The raw files have trailing spaces on every line, which a naive reader turns into
    two empty columns. Splitting on runs of whitespace avoids that.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{path}: expected {len(COLUMNS)} columns, found {df.shape[1]}")
    df.columns = COLUMNS
    df[ID_COLS] = df[ID_COLS].astype(int)
    return df


def read_rul_file(path: str | Path) -> pd.Series:
    """Read RUL_FDxxx.txt: one integer per test unit, in unit order (unit 1 first)."""
    values = pd.read_csv(path, sep=r"\s+", header=None, engine="python").iloc[:, 0]
    return pd.Series(values.astype(int).to_numpy(), index=range(1, len(values) + 1), name="rul")


def load_subset(data_dir: str | Path, subset: str = "FD001") -> CMAPSSData:
    """Load train, test and test-RUL files for one subset."""
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; choose one of {SUBSETS}")
    data_dir = Path(data_dir)
    files = {name: data_dir / f"{name}_{subset}.txt" for name in ("train", "test", "RUL")}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing C-MAPSS files: " + ", ".join(missing) + ". See data/README.md."
        )
    test = read_cmapss_file(files["test"])
    test_rul = read_rul_file(files["RUL"])
    if test["unit"].nunique() != len(test_rul):
        raise ValueError("Number of test units does not match number of RUL values")
    return CMAPSSData(subset, read_cmapss_file(files["train"]), test, test_rul)


def add_train_rul(df: pd.DataFrame, cap: int | None = DEFAULT_RUL_CAP) -> pd.DataFrame:
    """Add an 'rul' column to run-to-failure data.

    In training data every engine runs until it fails, so at each cycle
    RUL = (last cycle of that engine) - (current cycle).

    With `cap`, RUL is clipped from above ("piecewise-linear" target): early in life an
    engine is healthy and the sensors carry no information about how far away failure
    is, so asking the model to tell RUL=300 from RUL=200 only adds noise.
    """
    out = df.copy()
    last_cycle = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = last_cycle - out["cycle"]
    if cap is not None:
        out["rul"] = out["rul"].clip(upper=cap)
    return out
