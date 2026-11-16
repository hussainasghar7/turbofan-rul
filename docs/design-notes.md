# Design notes

Why the pipeline is built the way it is. Each section is a question an interviewer is
likely to ask about this project.

## 1. What exactly is RUL, and where do the labels come from?

Remaining useful life = how many more cycles (flights) the engine will run before it fails.

In the **training** files every engine runs until failure, so the last row of each
engine is the failure point. For any earlier row:

```
RUL = (last cycle of this engine) − (current cycle)
```

Example: an engine fails at cycle 192. At cycle 150 its RUL is 42; at cycle 192 it is 0.

In the **test** files the record is cut off early. We only get the true RUL for the
*last* observed cycle of each engine, in `RUL_FDxxx.txt`. So the model is scored once
per test engine.

## 2. Why cap RUL at 125?

A brand-new engine and an engine 100 cycles into its life look the same to the sensors:
nothing has started to wear yet. If the label says 300 for one and 200 for the other,
the model is asked to learn a difference that is not in the data. That only adds noise.

Capping (`RUL = min(RUL, 125)`) says: "while healthy, just say *plenty of life left*."
The target becomes flat at first, then drops linearly to 0. This is the standard
piecewise-linear target in the C-MAPSS literature; 120–130 are common cap values.

**Honest-reporting note:** the true test RUL is *not* capped by NASA. We report RMSE
against the real values, and separately against capped truth, because papers differ
here and the numbers are not comparable otherwise.

## 3. Why split by engine and not by row?

Cycle 100 and cycle 101 of the same engine are nearly identical. With a random row split,
one lands in training and the other in validation, so the model is graded on data it has
practically seen. Validation error looks great; test error doesn't. This is **data leakage**.

Holding out whole engines mirrors the real situation: the model will be used on engines it
has never seen.

## 4. Why rolling mean and std?

Single sensor readings are noisy. Wear shows up as a slow *drift*. The mean over the last
10 cycles smooths the noise and exposes the drift; the std captures growing instability.

The window is **trailing** (cycles t−9..t) and computed **per engine**, so no feature uses
the future or another engine's data. `tests/test_features.py` checks both properties.

## 5. Why drop some sensors?

In FD001 several sensors are constant across the whole fleet. A constant cannot tell you
anything about wear. The selection uses training data only, so the test set never
influences any modelling decision.

## 6. Why RMSE *and* the NASA score?

RMSE treats "20 cycles too early" and "20 cycles too late" the same. In maintenance they
are not the same:

- **Too early:** you replace a part that had life left. Costly, but safe.
- **Too late:** the engine fails before the planned maintenance. Unplanned downtime or worse.

The NASA score uses exponential penalties, steeper for late predictions:
early by 20 cycles costs e^(20/13) − 1 ≈ 3.7; late by 20 cycles costs e^(20/10) − 1 ≈ 6.4.

## 7. Why start with a mean baseline?

It is the score you get with zero intelligence. If a sophisticated model barely beats it,
the problem is in the data or features, not the model. Every result table should include
the baseline.

## 8. Why choose the best model on validation, not test?

If you look at test scores to choose between models, the test set becomes part of training,
and the reported number is optimistically biased. The test set is touched once, at the end.

## Known limitations / next steps

- FD002 and FD004 have six operating conditions; sensor values jump with the condition,
  not with wear. They need per-condition normalisation before these features make sense.
- Validation is computed on all cycles of held-out engines, while test uses only the last
  cycle. A test-like validation (truncate each held-out engine at a random point) would be
  closer to the real task.
- No uncertainty estimate yet: a maintenance planner needs "40 ± 10 cycles", not just "40".
