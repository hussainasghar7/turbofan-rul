import pytest

from turbofan_rul.data import COLUMNS, add_train_rul, load_subset


def test_load_has_expected_columns(synthetic_data):
    assert list(synthetic_data.train.columns) == COLUMNS
    assert list(synthetic_data.test.columns) == COLUMNS


def test_one_rul_value_per_test_unit(synthetic_data):
    assert synthetic_data.test["unit"].nunique() == len(synthetic_data.test_rul)


def test_missing_files_give_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        load_subset(tmp_path, "FD001")


def test_unknown_subset_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_subset(tmp_path, "FD009")


def test_train_rul_is_zero_at_failure_and_decreasing(synthetic_data):
    df = add_train_rul(synthetic_data.train, cap=None)
    last = df.sort_values("cycle").groupby("unit").tail(1)
    assert (last["rul"] == 0).all()
    for _, g in df.groupby("unit"):
        assert g.sort_values("cycle")["rul"].diff().dropna().eq(-1).all()


def test_rul_cap_is_applied(synthetic_data):
    df = add_train_rul(synthetic_data.train, cap=125)
    assert df["rul"].max() == 125
    assert df["rul"].min() == 0
