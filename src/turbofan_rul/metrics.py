"""Evaluation metrics used in the C-MAPSS literature."""

from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    """Root mean squared error, in cycles."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nasa_score(y_true, y_pred) -> float:
    """Asymmetric scoring function from the PHM08 challenge (Saxena et al., 2008).

    d = predicted - true.
      d < 0 (early prediction): penalty exp(-d/13) - 1
      d >= 0 (late prediction): penalty exp(d/10) - 1
    Predicting failure too LATE is punished harder, because in maintenance a late
    call means an unplanned breakdown, while an early call only wastes some life.
    Lower is better. The score grows exponentially, so a few bad engines dominate it.
    """
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    penalties = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(penalties))


def evaluate(y_true, y_pred) -> dict[str, float]:
    return {"rmse": rmse(y_true, y_pred), "nasa_score": nasa_score(y_true, y_pred)}
