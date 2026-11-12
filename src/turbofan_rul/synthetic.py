"""Generate small fake data in the exact C-MAPSS file format.

Used by the tests and by CI, so the full pipeline can run without downloading the
real dataset. The numbers are NOT realistic and must never be reported as results.

Usage:
    python -m turbofan_rul.synthetic --out data/synthetic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

N_SENSORS = 21
CONSTANT_SENSORS = {1, 5, 10, 16, 18, 19}  # mimic FD001, where these barely move


def _engine_trajectory(unit: int, life: int, rng: np.random.Generator, gains: np.ndarray):
    """Rows for one engine from cycle 1 to `life` (the failure cycle)."""
    cycles = np.arange(1, life + 1)
    wear = (cycles / life) ** 2  # slow at first, accelerating towards failure
    rows = np.empty((life, 2 + 3 + N_SENSORS))
    rows[:, 0] = unit
    rows[:, 1] = cycles
    rows[:, 2:5] = rng.normal(0, 0.002, size=(life, 3))
    for j in range(N_SENSORS):
        if (j + 1) in CONSTANT_SENSORS:
            rows[:, 5 + j] = 100.0 + j
        else:
            rows[:, 5 + j] = 500.0 + 10 * j + gains[j] * wear + rng.normal(0, 0.5, size=life)
    return rows


def generate(
    out_dir: str | Path, subset: str = "FD001", n_train: int = 30, n_test: int = 15, seed: int = 0
) -> Path:
    rng = np.random.default_rng(seed)
    gains = rng.choice([-1, 1], size=N_SENSORS) * rng.uniform(5, 20, size=N_SENSORS)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = [
        _engine_trajectory(u, int(rng.integers(150, 300)), rng, gains)
        for u in range(1, n_train + 1)
    ]

    test, rul = [], []
    for u in range(1, n_test + 1):
        life = int(rng.integers(150, 300))
        cut = int(rng.integers(30, life - 5))  # observe only the first `cut` cycles
        test.append(_engine_trajectory(u, life, rng, gains)[:cut])
        rul.append(life - cut)

    fmt = ["%d", "%d"] + ["%.4f"] * (3 + N_SENSORS)
    # Trailing space on each line, like the real files.
    np.savetxt(out_dir / f"train_{subset}.txt", np.vstack(train), fmt=fmt, newline=" \n")
    np.savetxt(out_dir / f"test_{subset}.txt", np.vstack(test), fmt=fmt, newline=" \n")
    np.savetxt(out_dir / f"RUL_{subset}.txt", np.array(rul), fmt="%d")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("data/synthetic"))
    p.add_argument("--subset", default="FD001")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    out = generate(args.out, args.subset, seed=args.seed)
    print(f"Wrote synthetic C-MAPSS-format data to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
