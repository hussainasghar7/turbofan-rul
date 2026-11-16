import json

from turbofan_rul.train import main


def test_end_to_end_on_synthetic_data(synthetic_dir, tmp_path):
    out, models = tmp_path / "reports", tmp_path / "models"
    code = main(
        [
            "--data-dir",
            str(synthetic_dir),
            "--subset",
            "FD001",
            "--out-dir",
            str(out),
            "--model-dir",
            str(models),
        ]
    )
    assert code == 0

    summary = json.loads((out / "FD001_results.json").read_text())
    r = summary["results"]
    assert set(r) == {"mean_baseline", "ridge", "xgboost"}
    # The synthetic data has a clear degradation signal, so a real model must win.
    assert r["xgboost"]["val_rmse"] < r["mean_baseline"]["val_rmse"]
    assert list(models.glob("FD001_*.joblib"))
