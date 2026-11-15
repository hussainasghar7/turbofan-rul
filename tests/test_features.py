from turbofan_rul.features import (
    add_rolling_features,
    feature_columns,
    last_cycle_rows,
    select_informative_sensors,
)


def test_constant_sensors_are_dropped(synthetic_data):
    sensors = select_informative_sensors(synthetic_data.train)
    for constant in ["s1", "s5", "s10", "s16", "s18", "s19"]:
        assert constant not in sensors
    assert "s2" in sensors


def test_rolling_features_do_not_look_ahead(synthetic_data):
    df = synthetic_data.train
    before = add_rolling_features(df, ["s2"], window=5)

    # Change the LAST cycle of unit 1; no earlier feature of unit 1 may change.
    modified = df.copy()
    last_idx = modified[modified["unit"] == 1]["cycle"].idxmax()
    modified.loc[last_idx, "s2"] += 1000
    after = add_rolling_features(modified, ["s2"], window=5)

    unit1_before = before[before["unit"] == 1].iloc[:-1]
    unit1_after = after[after["unit"] == 1].iloc[:-1]
    assert unit1_before["s2_mean5"].equals(unit1_after["s2_mean5"])


def test_rolling_features_do_not_mix_units(synthetic_data):
    out = add_rolling_features(synthetic_data.train, ["s2"], window=5)
    first_rows = out.groupby("unit").head(1)
    # On an engine's first cycle the rolling mean is just its own value, std is 0.
    assert (first_rows["s2_mean5"] == first_rows["s2"]).all()
    assert (first_rows["s2_std5"] == 0).all()


def test_last_cycle_rows_one_per_unit(synthetic_data):
    last = last_cycle_rows(synthetic_data.test)
    assert len(last) == synthetic_data.test["unit"].nunique()
    max_cycles = synthetic_data.test.groupby("unit")["cycle"].max()
    assert (last.set_index("unit")["cycle"] == max_cycles).all()


def test_feature_columns_exist_after_rolling(synthetic_data):
    sensors = select_informative_sensors(synthetic_data.train)
    out = add_rolling_features(synthetic_data.train, sensors, window=10)
    assert set(feature_columns(sensors, 10)) <= set(out.columns)
