import math

import pytest

from turbofan_rul.metrics import nasa_score, rmse
from turbofan_rul.split import split_by_unit


def test_rmse_known_value():
    assert rmse([0, 0, 0, 0], [1, -1, 1, -1]) == pytest.approx(1.0)


def test_nasa_score_is_zero_for_perfect_predictions():
    assert nasa_score([10, 50, 100], [10, 50, 100]) == 0.0


def test_nasa_score_punishes_late_more_than_early():
    late = nasa_score([50], [70])  # predicted 20 cycles too late
    early = nasa_score([50], [30])  # predicted 20 cycles too early
    assert late == pytest.approx(math.exp(2.0) - 1)
    assert early == pytest.approx(math.exp(20 / 13) - 1)
    assert late > early


def test_split_by_unit_has_no_overlap(synthetic_data):
    fit, val = split_by_unit(synthetic_data.train, val_fraction=0.2, seed=1)
    assert set(fit["unit"]).isdisjoint(set(val["unit"]))
    assert set(fit["unit"]) | set(val["unit"]) == set(synthetic_data.train["unit"])
    assert len(fit) + len(val) == len(synthetic_data.train)


def test_split_is_reproducible(synthetic_data):
    _, a = split_by_unit(synthetic_data.train, seed=7)
    _, b = split_by_unit(synthetic_data.train, seed=7)
    assert set(a["unit"]) == set(b["unit"])
