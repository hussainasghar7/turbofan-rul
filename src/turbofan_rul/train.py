"""Train and evaluate the baseline models on one C-MAPSS subset.

Usage:
    python -m turbofan_rul.train --data-dir data/raw --subset FD001

Protocol:
  1. Hold out 20% of TRAINING engines (by unit) as validation; fit on the rest.
     All model comparisons and tuning decisions use validation only.
  2. Refit each model on all training engines.
  3. Score once on the official test set (last cycle of each test engine).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from turbofan_rul.data import DEFAULT_RUL_CAP, add_train_rul, load_subset
from turbofan_rul.features import (
    add_rolling_features,
    feature_columns,
    last_cycle_rows,
    select_informative_sensors,
)
from turbofan_rul.metrics import evaluate
from turbofan_rul.models import get_models
from turbofan_rul.split import split_by_unit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--subset", default="FD001")
    p.add_argument("--window", type=int, default=10, help="rolling window in cycles")
    p.add_argument("--rul-cap", type=int, default=DEFAULT_RUL_CAP)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("reports"))
    p.add_argument("--model-dir", type=Path, default=Path("models"))
    return p.parse_args(argv)


def predict_clipped(model, X: pd.DataFrame, cap: int) -> np.ndarray:
    """RUL can't be negative, and the model was trained on targets <= cap."""
    return np.clip(model.predict(X), 0, cap)


def run(args: argparse.Namespace) -> dict:
    data = load_subset(args.data_dir, args.subset)

    train = add_train_rul(data.train, cap=args.rul_cap)
    sensors = select_informative_sensors(train)
    feats = feature_columns(sensors, args.window)

    train = add_rolling_features(train, sensors, args.window)
    test_last = last_cycle_rows(add_rolling_features(data.test, sensors, args.window))
    y_test = data.test_rul.loc[test_last["unit"]].to_numpy()

    fit_df, val_df = split_by_unit(train, args.val_fraction, args.seed)

    results: dict[str, dict] = {}
    fitted = {}
    for name, model in get_models(args.seed).items():
        model.fit(fit_df[feats], fit_df["rul"])
        val_metrics = evaluate(val_df["rul"], predict_clipped(model, val_df[feats], args.rul_cap))

        final = get_models(args.seed)[name]
        final.fit(train[feats], train["rul"])
        y_pred = predict_clipped(final, test_last[feats], args.rul_cap)
        test_metrics = evaluate(y_test, y_pred)
        test_capped = evaluate(np.minimum(y_test, args.rul_cap), y_pred)

        results[name] = {
            "val_rmse": val_metrics["rmse"],
            "test_rmse": test_metrics["rmse"],
            "test_rmse_truth_capped": test_capped["rmse"],
            "test_nasa_score": test_metrics["nasa_score"],
        }
        fitted[name] = final

    best = min(results, key=lambda n: results[n]["val_rmse"])  # chosen on validation only
    summary = {
        "subset": args.subset,
        "config": {
            "window": args.window,
            "rul_cap": args.rul_cap,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
        },
        "sensors_used": sensors,
        "n_train_units": int(train["unit"].nunique()),
        "n_test_units": int(len(test_last)),
        "best_model_by_val": best,
        "results": results,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.subset}_results.json").write_text(json.dumps(summary, indent=2))
    (args.out_dir / f"{args.subset}_results.md").write_text(to_markdown(summary))

    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": fitted[best], "features": feats, "sensors": sensors, "config": summary["config"]},
        args.model_dir / f"{args.subset}_{best}.joblib",
    )
    return summary


def to_markdown(summary: dict) -> str:
    lines = [
        f"### {summary['subset']} results",
        "",
        "| Model | Val RMSE | Test RMSE | Test RMSE (truth capped) | Test NASA score |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in summary["results"].items():
        lines.append(
            f"| {name} | {r['val_rmse']:.2f} | {r['test_rmse']:.2f} | "
            f"{r['test_rmse_truth_capped']:.2f} | {r['test_nasa_score']:.0f} |"
        )
    lines += ["", f"Best model by validation RMSE: **{summary['best_model_by_val']}**", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(to_markdown(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
