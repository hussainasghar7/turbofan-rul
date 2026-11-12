# Data

This project uses the **NASA C-MAPSS Turbofan Engine Degradation Simulation** dataset
(Saxena & Goebel, 2008, NASA Ames Prognostics Data Repository). The data files are
**not** committed to this repository.

## Getting the files

1. Download the dataset ("Turbofan Engine Degradation Simulation", usually distributed as
   `CMAPSSData.zip`) from the NASA Prognostics Center of Excellence data repository.
   If the NASA page is unavailable, widely used mirrors exist on Kaggle and Hugging Face;
   check that the files are unmodified (26 columns, files named as below).
2. Unzip and copy these files into `data/raw/`:

```
data/raw/
├── train_FD001.txt   test_FD001.txt   RUL_FD001.txt
├── train_FD002.txt   test_FD002.txt   RUL_FD002.txt
├── train_FD003.txt   test_FD003.txt   RUL_FD003.txt
└── train_FD004.txt   test_FD004.txt   RUL_FD004.txt
```

## The four subsets

| Subset | Train engines | Test engines | Operating conditions | Fault modes |
|---|---:|---:|---:|---|
| FD001 | 100 | 100 | 1 | HPC degradation |
| FD002 | 260 | 259 | 6 | HPC degradation |
| FD003 | 100 | 100 | 1 | HPC + fan degradation |
| FD004 | 248 | 249 | 6 | HPC + fan degradation |

Start with FD001. FD002/FD004 need per-operating-condition normalisation (see roadmap).

## Citation

A. Saxena, K. Goebel, D. Simon, N. Eklund. "Damage Propagation Modeling for Aircraft
Engine Run-to-Failure Simulation." *International Conference on Prognostics and Health
Management (PHM08)*, Denver, CO, 2008.

## No data yet? Use synthetic files

`python -m turbofan_rul.synthetic --out data/synthetic` writes fake files in the same
format, so you can run the pipeline end to end. Never report numbers from synthetic data.
