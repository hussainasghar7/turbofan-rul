# Turbofan RUL: Predictive Maintenance on NASA C-MAPSS

[![CI](https://github.com/<your-username>/turbofan-rul/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/turbofan-rul/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Predicting the **remaining useful life (RUL)** of turbofan engines from multivariate sensor
time series, so maintenance can be scheduled before failure instead of after it.

The project goes from a trivial baseline to deep sequence models and a deployable API,
with a leakage-safe evaluation protocol throughout.

## Problem

Each engine starts healthy, develops a fault, and degrades until it fails. For training
engines we see the whole run to failure. For test engines the record stops at some
unknown point, and the task is to predict **how many cycles remain**.

- **Input:** 21 sensor channels + 3 operating settings per engine cycle
- **Output:** RUL in cycles, evaluated at the last observed cycle of each test engine
- **Metrics:** RMSE and the asymmetric NASA/PHM08 score, which punishes late predictions
  (missed failures) harder than early ones

## Results (FD001)

> _Run `python -m turbofan_rul.train --subset FD001` on the real data and paste
> `reports/FD001_results.md` here. Never publish numbers from the synthetic data._

| Model | Val RMSE | Test RMSE | Test NASA score |
|---|---:|---:|---:|
| Mean baseline | – | – | – |
| Ridge regression | – | – | – |
| XGBoost | – | – | – |

## Approach

1. **Labels:** RUL = last cycle − current cycle, capped at 125 cycles (piecewise-linear target).
2. **Features:** informative sensors (near-constant ones dropped using training data only),
   plus trailing rolling mean and std per engine. No feature uses future cycles.
3. **Validation:** 20% of *engines* held out, never random rows, to avoid leakage between
   neighbouring cycles of the same engine.
4. **Models:** mean baseline → Ridge → XGBoost; best chosen on validation, test scored once.

Each decision is explained in [docs/design-notes.md](docs/design-notes.md).

## Quickstart

```bash
git clone https://github.com/<your-username>/turbofan-rul.git
cd turbofan-rul
python -m venv .venv && .venv\Scripts\activate      # Windows (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dev]"

# Put the C-MAPSS files in data/raw/ (see data/README.md), then:
python -m turbofan_rul.train --data-dir data/raw --subset FD001

# No data yet? Run everything on synthetic files:
python -m turbofan_rul.synthetic --out data/synthetic
python -m turbofan_rul.train --data-dir data/synthetic

pytest   # tests
```

## Project structure

```
├── src/turbofan_rul/
│   ├── data.py        # loading C-MAPSS files, RUL labels
│   ├── features.py    # sensor selection, rolling features (no look-ahead)
│   ├── split.py       # train/validation split by engine
│   ├── metrics.py     # RMSE, NASA score
│   ├── models.py      # baseline models
│   ├── train.py       # training + evaluation CLI
│   └── synthetic.py   # fake data in C-MAPSS format for tests/CI
├── tests/             # pytest suite (runs without the real dataset)
├── docs/design-notes.md
├── data/README.md     # how to obtain the dataset
└── .github/workflows/ci.yml
```

## Roadmap

- [x] Data loading, RUL labels, leakage-safe split
- [x] Classical baselines (mean, Ridge, XGBoost) with RMSE + NASA score
- [x] Tests and CI on synthetic data
- [ ] EDA notebook: sensor trends, correlation with RUL
- [ ] Deep sequence models (LSTM, 1D-CNN) in PyTorch, sliding windows
- [ ] Experiment tracking with MLflow
- [ ] Operating-condition normalisation for FD002 / FD004
- [ ] Prediction intervals (quantile regression or conformal prediction)
- [ ] FastAPI inference service + Docker image
- [ ] Interactive demo (Streamlit or Hugging Face Space)

## Data and citation

NASA C-MAPSS Turbofan Engine Degradation Simulation (Saxena & Goebel, 2008), NASA Ames
Prognostics Center of Excellence. See [data/README.md](data/README.md).

## License

MIT, see [LICENSE](LICENSE).
