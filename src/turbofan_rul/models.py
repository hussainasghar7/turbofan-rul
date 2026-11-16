"""Baseline models, from trivial to strong.

Always start with a model that cannot be clever (predict the mean). Every later model
must beat it clearly; if it doesn't, something is wrong with the data or the features.
"""

from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def get_models(seed: int = 42) -> dict[str, object]:
    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "xgboost": XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
        ),
    }
